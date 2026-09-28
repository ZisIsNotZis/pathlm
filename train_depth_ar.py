"""深度自适应 AR 训练 runner（ticket 23，方向修订后第一个实验）。

用法：python train_depth_ar.py <run_name> --d 256 --layers 8 [--dense]
                              [--steps 3600] [--seed 0] [--out-root DIR]
训练 + 测量电池（depth 曲线 / TF 接受率 / workspace 探针 / 出口提交生成 /
门曲线）→ .scratch/23-depth-ar/evidence/<run_name>/。
"""

import argparse, json, math, os, sys, time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.depth_ar import (DepthARConfig, DepthARModel, sample_exit_depths,
                             depth_curve, tf_acceptance, workspace_probe,
                             generation_quality)
from pathlm.data import load_enwik8_full, batch


def train(model: DepthARModel, train_arr, tcfg: dict, log_path: str,
          gate_log_path: str, exit_mode: str = "uniform"):
    """随机退出深度训练：每步均匀采样 d_i ∈ {0..L}；记录逐深度 CE 与门均值。"""
    steps, bs = tcfg["steps"], tcfg["batch_size"]
    seq = model.cfg.seq_len
    L = model.cfg.n_layers
    n_ce = 1 if model.cfg.dense else L + 1
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    torch_rng = torch.Generator().manual_seed(tcfg["seed"])
    t0 = time.time()
    depth_ce_acc = [0.0] * n_ce
    gate_acc = [0.0] * L
    n_logged = 0
    for step in range(steps):
        model.train()
        x, _ = batch(train_arr, bs, seq, torch_rng)
        x = x.cuda()
        if exit_mode == "full":
            depths = torch.full((bs, seq), L, dtype=torch.long).cuda()
        else:
            depths = sample_exit_depths(bs, seq, L, torch_rng).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            loss, aux = model(x, exit_depths=depths)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm) or gnorm > 1e4:
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()
        for k in range(n_ce):
            depth_ce_acc[k] += float(aux["depth_ce"][k].detach())
        for k in range(len(aux["gate_mean"])):
            gate_acc[k] += float(aux["gate_mean"][k])
        n_logged += 1
        if step % 200 == 0 or step == steps - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": round(float(loss.detach()), 4),
                    "depth_ce": [round(float(c), 4) for c in aux["depth_ce"]],
                    "gate_mean": [round(float(g), 4) for g in aux["gate_mean"]],
                    "gnorm": round(float(gnorm), 2),
                    "lr": round(float(sched.get_last_lr()[0]), 6),
                    "min": round((time.time() - t0) / 60, 2)}) + "\n")
            with open(gate_log_path, "a") as f:
                f.write(json.dumps({"step": step,
                                    "gate_mean": [round(float(g), 4)
                                                  for g in aux["gate_mean"]]}) + "\n")
    return time.time() - t0, [c / max(n_logged, 1) for c in depth_ce_acc], \
        [g / max(n_logged, 1) for g in gate_acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--exit-mode", choices=("uniform", "full"), default="uniform",
                    help="uniform = 随机退出深度 dᵢ~U{0..L}（混合深度填充）；"
                         "full = 全部 dᵢ=L（无混合填充，仅保留全深度并行监督——"
                         "税归因 ablation）")
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-bias-init", type=float, default=-2.0)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--out-root", default=".scratch/23-depth-ar/evidence")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    cfg = DepthARConfig(d_model=args.d, n_layers=args.layers, n_heads=args.heads,
                        seq_len=args.seq_len, dense=args.dense,
                        gate_bias_init=args.gate_bias_init)
    model = DepthARModel(cfg, vocab_size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.run_name}: {n_params/1e6:.2f}M params, vocab {vocab_size}, "
          f"steps {args.steps} x bs{args.batch_size} x seq{args.seq_len} "
          f"= {args.steps*args.batch_size*args.seq_len/1e6:.1f}M tokens", flush=True)

    tcfg = {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed}
    wall, avg_depth_ce, avg_gate = train(model, train_arr, tcfg,
                                         os.path.join(run_dir, "train_log.jsonl"),
                                         os.path.join(run_dir, "gate_log.jsonl"),
                                         exit_mode=args.exit_mode)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    model.eval()
    results = {
        "run": args.run_name, "config": asdict(cfg), "train": tcfg,
        "exit_mode": args.exit_mode,
        "params": n_params, "vocab_size": vocab_size,
        "train_tokens": args.steps * args.batch_size * args.seq_len,
        "wall_minutes": round(wall / 60, 2),
        "avg_train_depth_ce": [round(c, 4) for c in avg_depth_ce],
        "avg_train_gate_mean": [round(g, 4) for g in avg_gate],
    }
    if not args.dense:
        results["depth_curve"] = depth_curve(model, eval_arr, device="cuda",
                                             n_batches=args.eval_batches)
        results["tf_acceptance"] = tf_acceptance(model, eval_arr, device="cuda",
                                                 n_batches=args.eval_batches)
        results["workspace"] = workspace_probe(model, eval_arr, device="cuda",
                                               n_batches=4, batch_size=8)
        results["generation"] = generation_quality(model, eval_arr, device="cuda",
                                                   prompt_len=64, n_new=128)
    else:
        curve = depth_curve(model, eval_arr, device="cuda", n_batches=args.eval_batches)
        results["bpc"] = curve[-1]["bpc"]
        results["next_token_acc"] = curve[-1]["acc"]
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "train")}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
