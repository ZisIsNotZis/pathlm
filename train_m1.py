"""M1 runner: config-driven training at base scale + the full eval battery.

Usage: python train_m1.py <run_name> --config configs/B0.json [--steps N] [--seed N]
Writes checkpoint + results.json to .scratch/04-m1-runs/evidence/<run_name>/.
Every metric that is meaningless for a run's config is auto-skipped; the
results.json carries the full config, so each row is reproducible.
"""

import argparse, contextlib, json, math, os, random, sys, time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch, needle_batch
from pathlm.eval import bpc, repair, depth_curve, needle_acc, decode_speed, locality_sweep, eval_pc
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


def train(model: PathLM, train_arr, tcfg: dict, pcap: PathConfig, log_path: str,
          ckpt_dir: str | None = None, ckpt_every: int = 0):
    """ckpt_every > 0 also writes model_step{step:06d}.pt every ckpt_every steps
    (curve mode: probe_curve.py reads them for effectiveness-vs-steps). Derived
    weights are intermediates, so ckpt_dir should stay out of git (runs/, .tmp/);
    the committed evidence is the curve jsonl, not the weights."""
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
        is_needle = pcap.p_needle > 0 and rng.random() < pcap.p_needle
        if is_needle:
            x, _, _ = needle_batch(bs, seq, model.n_real_tokens, model.mask_token,
                                   torch_rng, data=train_arr, anchors=pcap.anchors,
                                   anchor_frac=0.2, n_needles=8)
        else:
            x, _ = batch(train_arr, bs, seq, torch_rng)
        x = x.cuda()
        paths = sample_rounds(pcap, rng, model.mcfg.n_layers)
        # bf16 autocast: halves activation memory (the GPU is shared with a
        # resident llama-server) and speeds up base-scale training.
        # task-level corruption exemption: needle steps train the COPY task on
        # clean input (corruption is a global suppression of exact copying)
        clean_needle = pcap.needle_corrupt_free and is_needle
        ctx = (eval_pc(model, corrupt_wrong=0.0, corrupt_mask=0.0)
               if clean_needle else contextlib.nullcontext())
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16,
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
        if ckpt_dir and ckpt_every > 0 and (step + 1) % ckpt_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"model_step{step + 1:06d}.pt"))
        if step % 200 == 0 or step == steps - 1:
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(float(loss.detach()), 4),
                                        "lr": round(float(sched.get_last_lr()[0]), 6),
                                        "min": round((time.time() - t0) / 60, 1)}) + "\n")
            except OSError as e:
                raise RuntimeError(f"cannot append train log {log_path}: {e}") from e
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
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also save an intermediate checkpoint every N steps (0 = final only)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="where intermediate checkpoints go (default: <run_dir>/ckpts; "
                         "keep it gitignored — use runs/ or .tmp/)")
    args = ap.parse_args()

    try:
        cfg = json.load(open(args.config))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot load config {args.config}: {e}") from e
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    run_dir = os.path.join(args.out_root, args.run_name)
    try:
        os.makedirs(run_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"cannot create run dir {run_dir}: {e}") from e
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

    ckpt_dir = None
    if args.ckpt_every > 0:
        ckpt_dir = args.ckpt_dir or os.path.join(run_dir, "ckpts")
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"cannot create ckpt dir {ckpt_dir}: {e}") from e
        # derived weights must not bloat git: refuse an obvious in-repo track path
        if ckpt_dir.startswith(run_dir) and args.ckpt_dir is None:
            print(f"WARNING: intermediate checkpoints land in the run dir "
                  f"({ckpt_dir}); pass --ckpt-dir under runs/ or .tmp/ to keep "
                  f"them out of git", flush=True)

    log_path = os.path.join(run_dir, "train_log.jsonl")
    wall = train(model, train_arr, cfg["train"], pcap, log_path,
                 ckpt_dir=ckpt_dir, ckpt_every=args.ckpt_every)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    results = run_battery(model, eval_arr, vocab_size, pcap)
    results["config"] = cfg
    # D2 (ticket 11): results.json used to record only the RAW config file, so a
    # knob left unset silently took a PathConfig default and the leak was
    # invisible in the evidence (this is how C1M/C2M ended up at n_mtp=2 while
    # the rest of the table pinned 1). Record the RESOLVED config and name the
    # keys the file did not set.
    results["resolved"] = {"model": asdict(mcfg), "path": asdict(pcap)}
    unspecified = sorted(set(asdict(pcap)) - set(cfg.get("path", {})))
    results["unspecified_path_keys"] = unspecified
    if unspecified:
        print(f"WARNING: {args.run_name} relies on PathConfig defaults for "
              f"{unspecified} — pin them in the config if this run is compared "
              f"against others", flush=True)
    results["params"] = n_params
    results["wall_minutes"] = round(wall / 60, 1)
    results["train_tokens"] = cfg["train"]["steps"] * cfg["train"]["batch_size"] * mcfg.seq_len
    try:
        rf = open(os.path.join(run_dir, "results.json"), "w")
    except OSError as e:
        raise RuntimeError(f"cannot write results to {run_dir}: {e}") from e
    with rf as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k not in ("config",)}, indent=2))


if __name__ == "__main__":
    main()
