"""M1 runner: config-driven training at base scale + the full eval battery.

Usage: python train_m1.py <run_name> --config configs/B0.json [--steps N] [--seed N]
Writes checkpoint + results.json to .scratch/04-m1-runs/evidence/<run_name>/.
Every metric that is meaningless for a run's config is auto-skipped; the
results.json carries the full config, so each row is reproducible.
"""

import argparse, json, math, os, random, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch, needle_batch
from pathlm.eval import bpc, repair, depth_curve, needle_acc, decode_speed, locality_sweep
from pathlm.model import PathLM


def sample_rounds(pcap: PathConfig, rng: random.Random, n_layers: int) -> list:
    """Base path decides the retry count; each round gets its own fresh path.
    w_diversity > 0 additionally samples one parallel path per step so the
    diversity loss has a partner to decorrelate from (the partner is an extra
    estimator pass, not a retry round)."""
    base = sample_path(pcap, rng, n_layers)
    paths = [base] + [sample_path(pcap, rng, n_layers) for _ in range(base.n_retries)]
    if pcap.w_diversity > 0:
        paths.append(sample_path(pcap, rng, n_layers))
    return paths


def train(model: PathLM, train_arr, tcfg: dict, pcap: PathConfig, log_path: str):
    steps, bs = tcfg["steps"], tcfg["batch_size"]
    seq = model.mcfg.seq_len
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    rng = random.Random(tcfg["seed"])
    torch_rng = torch.Generator().manual_seed(tcfg["seed"])
    t0 = time.time()
    for step in range(steps):
        model.train()
        if pcap.p_needle > 0 and rng.random() < pcap.p_needle:
            x, _, _ = needle_batch(bs, seq, model.n_real_tokens, model.mask_token,
                                   torch_rng, data=train_arr, anchors=pcap.anchors,
                                   anchor_frac=0.2, n_needles=8)
        else:
            x, _ = batch(train_arr, bs, seq, torch_rng)
        x = x.cuda()
        paths = sample_rounds(pcap, rng, model.mcfg.n_layers)
        # bf16 autocast: halves activation memory (the GPU is shared with a
        # resident llama-server) and speeds up base-scale training.
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            loss, _ = model(x, paths, x)  # targets = the clean tokens themselves
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm) or gnorm > 1e4:
            # gradient explosion (observed with the negative-JS reward term):
            # skip BEFORE opt.step — a poisoned optimizer state is unrecoverable
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step(); sched.step()
        if step % 200 == 0 or step == steps - 1:
            # pi-lens-ignore: unchecked-throwing-call-python
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(float(loss), 4),
                                    # pi-lens-ignore: unchecked-throwing-call-python
                                    "lr": round(float(sched.get_last_lr()[0]), 6),
                                    "min": round((time.time() - t0) / 60, 1)}) + "\n")
    return time.time() - t0


def run_battery(model: PathLM, eval_arr, vocab_size: int, pcap: PathConfig) -> dict:
    res: dict = {"bpc": bpc(model, eval_arr, vocab_size),
           "decode_speed": decode_speed(model, pcap)}
    if pcap.corrupt_wrong > 0 or pcap.corrupt_mask > 0:
        res["repair"] = repair(model, eval_arr)
    if pcap.w_dense_exit > 0:
        res["depth_curve"] = depth_curve(model, eval_arr)
    if pcap.p_needle > 0:
        res["needle_acc"] = needle_acc(model, eval_arr)
    if pcap.shuffle_locality > 0:
        res["locality_sweep"] = locality_sweep(model, eval_arr)
    if pcap.p_retry > 0 or pcap.p_token_retry > 0:
        # retry curve: force 1..4 rounds at inference (no retraining) — the
        # recurrent-transformer question; also gives the trained-length point
        res["retry_curve"] = {k: repair(model, eval_arr,
                                        rounds=k)["rounds"] for k in (2, 3, 4)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override config steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--out-root", default=".scratch/04-m1-runs/evidence")
    args = ap.parse_args()

    # pi-lens-ignore: unchecked-throwing-call-python
    cfg = json.load(open(args.config))
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    run_dir = os.path.join(args.out_root, args.run_name)
    # pi-lens-ignore: unchecked-throwing-call-python
    os.makedirs(run_dir, exist_ok=True)
    mcfg = ModelConfig(**cfg["model"])
    pcap = PathConfig(**cfg["path"])
    if pcap.reentry_mix and pcap.p_retry == 0 and pcap.p_token_retry == 0:
        raise ValueError("reentry_mix without a loop (p_retry/p_token_retry) is a no-op element")
    torch.manual_seed(cfg["train"]["seed"])
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    model = PathLM(mcfg, pcap, vocab_size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.run_name}: {n_params/1e6:.2f}M params, vocab {vocab_size}, "
          f"steps {cfg['train']['steps']}", flush=True)

    log_path = os.path.join(run_dir, "train_log.jsonl")
    wall = train(model, train_arr, cfg["train"], pcap, log_path)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    results = run_battery(model, eval_arr, vocab_size, pcap)
    results["config"] = cfg
    results["params"] = n_params
    results["wall_minutes"] = round(wall / 60, 1)
    results["train_tokens"] = cfg["train"]["steps"] * cfg["train"]["batch_size"] * mcfg.seq_len
    # pi-lens-ignore: unchecked-throwing-call-python
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k not in ("config",)}, indent=2))


if __name__ == "__main__":
    main()
