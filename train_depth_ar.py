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
                             generation_quality, readout_workspace_energy,
                             spec_cost_model)
from pathlm.data import load_enwik8_full, batch


def _linear_anneal(v0: float, v1: float | None, step: int, steps: int,
                   anneal_frac: float) -> float:
    """通用线性退火：前 (1−frac)·steps 步恒 v0，末 frac·steps 步线性到 v1
    （起点连续）；v1=None 恒 v0。"""
    if v1 is None:
        return v0
    start = (1.0 - anneal_frac) * steps
    if step <= start:
        return v0
    frac = min((step - start) / max(steps - start, 1.0), 1.0)
    return v0 + (v1 - v0) * frac


def aux_weight_at(step: int, steps: int, aux_weight: float,
                  anneal_to: float | None = None,
                  anneal_frac: float = 0.2) -> float:
    """浅层 aux 权重时间表（ticket 23b）：常数 aux_weight；anneal_to 给定时
    在最后 anneal_frac 比例的步数内从 aux_weight 线性退火到 anneal_to
    （W3 = 1.0 → 0.1，最后 20% 步；ticket 25 C 臂 --exit-anneal 用
    anneal_frac=0.25 → 0.3）。起点连续：step ≤ start 恒 aux_weight。"""
    return _linear_anneal(aux_weight, anneal_to, step, steps, anneal_frac)


def final_weight_at(step: int, steps: int, final_weight: float,
                    anneal_to: float | None = None,
                    anneal_frac: float = 0.25) -> float:
    """最终深度权重时间表（ticket 25 C 臂 --exit-anneal）：final 权重与浅层
    aux 同窗线性退火（2× → 1.0，纯 trunk CE 权重终点恒 ≥ 1）；
    anneal_to=None 恒 final_weight（ticket 24 语义）。"""
    return _linear_anneal(final_weight, anneal_to, step, steps, anneal_frac)


def train(model: DepthARModel, train_arr, tcfg: dict, log_path: str,
          gate_log_path: str, exit_mode: str = "uniform",
          aux_weight: float = 1.0, aux_anneal_to: float | None = None,
          final_weight: float = 1.0,
          shallow_stopgrad: bool = False, dense_aux: bool = False,
          exit_anneal_to: float | None = None,
          exit_anneal_frac: float = 0.25,
          final_anneal_to: float | None = None):
    """随机退出深度训练：每步均匀采样 d_i ∈ {0..L}；记录逐深度 CE 与门均值。
    浅层 CE 权重按 aux_weight_at(step) 调度（默认恒 1 = v1 等权）；最终深度
    权重 final_weight（ticket 24：默认 1.0 = v1，2.0 = 契约「最终深度 2×」）。
    ticket 23c 开关：shallow_stopgrad = H3（浅头纯读出）；
    dense_aux = H1（dense 臂全深度监督，需 model.cfg.dense）。
    ticket 25 C 臂 --exit-anneal：exit_anneal_to 给定时浅层权重在末
    exit_anneal_frac 步线性降到该值，final 权重同窗线性降到 final_anneal_to
    （2× → 1.0）；调度关断时行为与 ticket-24 逐位一致。
    梯度范数监控（ticket 24）：全程累计 gnorm max/mean 与跳步数，随结果落盘。"""
    steps, bs = tcfg["steps"], tcfg["batch_size"]
    seq = model.cfg.seq_len
    L = model.cfg.n_layers
    n_ce = 1 if (model.cfg.dense and not dense_aux) else L + 1
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    torch_rng = torch.Generator().manual_seed(tcfg["seed"])
    t0 = time.time()
    depth_ce_acc = [0.0] * n_ce
    gate_acc = [0.0] * L
    n_logged = 0
    gnorm_sum = 0.0
    gnorm_max = 0.0
    n_grad_skip = 0
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
            if exit_anneal_to is not None:
                w = aux_weight_at(step, steps, aux_weight, exit_anneal_to,
                                  exit_anneal_frac)
                fw = final_weight_at(step, steps, final_weight, final_anneal_to,
                                     exit_anneal_frac)
            else:
                w = aux_weight_at(step, steps, aux_weight, aux_anneal_to)
                fw = final_weight
            loss, aux = model(x, exit_depths=depths, shallow_weight=w,
                              final_weight=fw,
                              shallow_stopgrad=shallow_stopgrad,
                              dense_aux=dense_aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gnorm_f = float(gnorm)
        gnorm_sum += gnorm_f
        gnorm_max = max(gnorm_max, gnorm_f)
        if not torch.isfinite(gnorm) or gnorm > 1e4:
            n_grad_skip += 1
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
                    "aux_w": round(w, 4),
                    "final_w": round(fw, 4),
                    "lr": round(float(sched.get_last_lr()[0]), 6),
                    "min": round((time.time() - t0) / 60, 2)}) + "\n")
            with open(gate_log_path, "a") as f:
                f.write(json.dumps({"step": step,
                                    "gate_mean": [round(float(g), 4)
                                                  for g in aux["gate_mean"]]}) + "\n")
    return time.time() - t0, [c / max(n_logged, 1) for c in depth_ce_acc], \
        [g / max(n_logged, 1) for g in gate_acc], {
            "gnorm_max": round(gnorm_max, 2),
            "gnorm_mean": round(gnorm_sum / max(steps, 1), 3),
            "grad_skips": n_grad_skip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--mlp-mult", type=int, default=4,
                    help="MLP 隐层倍数（ticket 24：100M 规格用 6）")
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
    ap.add_argument("--aux-weight", type=float, default=1.0,
                    help="浅层（depth 0..L-1）CE 权重，最终深度恒 1；1.0 = "
                         "v1 等权（ticket 23b）")
    ap.add_argument("--aux-anneal-to", type=float, default=None,
                    help="给定则浅层权重在最后 20%% 步从 --aux-weight 线性退火"
                         "到该值（W3：--aux-weight 1.0 --aux-anneal-to 0.1）")
    ap.add_argument("--final-weight", type=float, default=1.0,
                    help="最终深度 CE 权重（ticket 24：1.0 = v1 等权；"
                         "2.0 = 契约「权重均匀 + 最终深度 2×」）")
    ap.add_argument("--shallow-stopgrad", action="store_true",
                    help="浅层（depth 0..L-1）读出前 detach 状态：浅层 CE 只训练 "
                         "tied U 头，梯度不穿 trunk（ticket 23c，H3）")
    ap.add_argument("--dense-aux", action="store_true",
                    help="dense 臂开启全深度并行监督（L+1 个 CE，梯度穿 trunk；"
                         "ticket 23c，H1；需配合 --dense）")
    ap.add_argument("--no-fill-kv", dest="fill_kv", action="store_false",
                    help="no-fill ragged 语义（ticket 23e）：早退位置在未跑层"
                         "的 KV 中缺席（原引擎 per-layer cache 语义）；缺省 = "
                         "v1 fill（W0 复现点）")
    ap.add_argument("--proj-fill", action="store_true",
                    help="fill 基础上加逐层线性 adapter（恒等初始化，学得 "
                         "remap；ticket 23e 可选臂，需 fill_kv）")
    ap.add_argument("--readout-dims", type=int, default=None,
                    help="结构切分（ticket 25 B 臂）：unembed/出口头只读前 r 个"
                         "通道（tied 表列切片 E[:, :r]），trunk/embedding 全维度；"
                         "缺省 = 全维度读出")
    ap.add_argument("--exit-anneal", action="store_true",
                    help="C 臂冷却退火（ticket 25）：浅层 aux 权重均匀到 "
                         "(1−frac) 处、末 frac 线性降到 --exit-anneal-to；"
                         "final 权重从 --final-weight 同窗线性降到 "
                         "--final-anneal-to（纯 trunk CE 权重终点 1.0）")
    ap.add_argument("--exit-anneal-to", type=float, default=0.3,
                    help="浅层 aux 权重退火终点（缺省 0.3）")
    ap.add_argument("--exit-anneal-frac", type=float, default=0.25,
                    help="退火窗口占总步数比例（缺省末 25%%）")
    ap.add_argument("--final-anneal-to", type=float, default=1.0,
                    help="final 权重退火终点（缺省 1.0 = 2× 降回等权）")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--out-root", default=".scratch/23-depth-ar/evidence")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    cfg = DepthARConfig(d_model=args.d, n_layers=args.layers, n_heads=args.heads,
                        seq_len=args.seq_len, mlp_mult=args.mlp_mult,
                        dense=args.dense,
                        gate_bias_init=args.gate_bias_init,
                        fill_kv=args.fill_kv, proj_fill=args.proj_fill,
                        readout_dims=args.readout_dims)
    model = DepthARModel(cfg, vocab_size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.run_name}: {n_params/1e6:.2f}M params, vocab {vocab_size}, "
          f"steps {args.steps} x bs{args.batch_size} x seq{args.seq_len} "
          f"= {args.steps*args.batch_size*args.seq_len/1e6:.1f}M tokens", flush=True)

    tcfg = {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed}
    wall, avg_depth_ce, avg_gate, train_stats = train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"),
        exit_mode=args.exit_mode,
        aux_weight=args.aux_weight,
        aux_anneal_to=args.aux_anneal_to,
        final_weight=args.final_weight,
        shallow_stopgrad=args.shallow_stopgrad,
        dense_aux=args.dense_aux,
        exit_anneal_to=(args.exit_anneal_to if args.exit_anneal else None),
        exit_anneal_frac=args.exit_anneal_frac,
        final_anneal_to=(args.final_anneal_to if args.exit_anneal else None))
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    model.eval()
    results = {
        "run": args.run_name, "config": asdict(cfg), "train": tcfg,
        "exit_mode": args.exit_mode,
        "fill_kv": args.fill_kv, "proj_fill": args.proj_fill,
        "aux_weight": args.aux_weight, "aux_anneal_to": args.aux_anneal_to,
        "final_weight": args.final_weight,
        "readout_dims": args.readout_dims,
        "exit_anneal": args.exit_anneal,
        "exit_anneal_to": args.exit_anneal_to if args.exit_anneal else None,
        "exit_anneal_frac": args.exit_anneal_frac if args.exit_anneal else None,
        "final_anneal_to": args.final_anneal_to if args.exit_anneal else None,
        "shallow_stopgrad": args.shallow_stopgrad, "dense_aux": args.dense_aux,
        "gnorm_monitor": train_stats,
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
        results["spec_cost_model"] = spec_cost_model(
            results["tf_acceptance"], cfg.n_layers,
            draft_depths=(4, 6))
        if cfg.readout_dims is not None and cfg.readout_dims < cfg.d_model:
            results["readout_workspace_energy"] = readout_workspace_energy(
                model, eval_arr, device="cuda", n_batches=4, batch_size=8)
        results["workspace"] = workspace_probe(model, eval_arr, device="cuda",
                                               n_batches=4, batch_size=8)
        results["generation"] = generation_quality(model, eval_arr, device="cuda",
                                                   prompt_len=64, n_new=128)
    else:
        curve = depth_curve(model, eval_arr, device="cuda", n_batches=args.eval_batches)
        # 23c H1（dense+aux）也需全深度曲线（depth-L 主数字 + depth-4 出口 bpc）
        results["depth_curve"] = curve
        results["bpc"] = curve[-1]["bpc"]
        results["next_token_acc"] = curve[-1]["acc"]
        # ticket 25 电池：dense 也报 tf（中间读出无监督，截断资产 proxy 口径）
        results["tf_acceptance"] = tf_acceptance(model, eval_arr, device="cuda",
                                                 n_batches=args.eval_batches)
        results["tf_note"] = ("dense 中间深度读出无直接监督：tf 系 depth-k "
                              "残差态 tied 读出与 depth-L 读出的 argmax 一致率 "
                              "proxy（截断资产口径，非训练出口资产）")
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "train")}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
