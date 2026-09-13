"""Effectiveness-vs-training-step curves (ticket 11).

Model convergence and effectiveness convergence are different questions:
a config's loss plateauing says nothing about whether the DELTA between two
configs has stabilised. This probes the second one — evaluate every
intermediate checkpoint on a fixed eval stream, then align two curves by step.

    # build a curve from a checkpointed run (see train_m1.py --ckpt-every)
    python3 probe_curve.py --config configs/B0.json \
        --ckpt-dir .tmp/ckpts/B0long --out .scratch/NN/evidence/curve_B0.jsonl

    # is the tax stable, shrinking, or still growing?
    python3 probe_curve.py --compare curve_B0.jsonl curve_C1.jsonl --metric bpc

Eval is forward-only and cheap relative to a training step, so this can run
alongside training. Curves are the committed evidence; the weights are not.
"""

import argparse, glob, json, os, re, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.config import ModelConfig, PathConfig
from pathlm.data import load_enwik8_full
from pathlm.eval import bpc, repair
from pathlm.model import PathLM

STEP_RE = re.compile(r"model_step(\d+)\.pt$")


def steps_in(ckpt_dir: str) -> list[tuple[int, str]]:
    found = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_step*.pt")):
        m = STEP_RE.search(p)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def build_curve(config_path: str, ckpt_dir: str, out_path: str,
                n_batches: int, n_repair_batches: int):
    cfg = json.load(open(config_path))
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    pts = steps_in(ckpt_dir)
    if not pts:
        raise SystemExit(f"no model_step*.pt in {ckpt_dir}")

    # torch.manual_seed is what eval.repair seeds the corruption draws from;
    # reseed per point so every checkpoint sees the SAME corrupted batches.
    with open(out_path, "w") as f:
        for step, path in pts:
            torch.manual_seed(cfg["train"]["seed"])
            model = PathLM(mcfg, pcap, vocab_size).cuda()
            model.load_state_dict(torch.load(path, map_location="cuda"))
            model.eval()
            rec = {"step": step, "ckpt": os.path.basename(path)}
            rec["bpc"] = bpc(model, eval_arr, vocab_size, n_batches=n_batches)["bpc"]
            if pcap.corrupt_wrong > 0 or pcap.corrupt_mask > 0:
                rp = repair(model, eval_arr, n_batches=n_repair_batches)["rounds"]
                rec["repair_r0"] = round(rp["r0"]["repair_acc"], 4)
                rec["repair_last"] = round(list(rp.values())[-1]["repair_acc"], 4)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  step {step:6d}  bpc {rec['bpc']:.4f}"
                  + (f"  repair {rec['repair_r0']:.4f}->{rec['repair_last']:.4f}"
                     if "repair_r0" in rec else ""), flush=True)
            del model
            torch.cuda.empty_cache()


def compare(a_path: str, b_path: str, metric: str):
    def load(p):
        return {json.loads(l)["step"]: json.loads(l) for l in open(p)}
    A, B = load(a_path), load(b_path)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("no shared steps between the two curves")
    na, nb = os.path.basename(a_path), os.path.basename(b_path)
    print(f"{'step':>8} {na:>14} {nb:>14} {'delta':>10}  trend")
    prev = None
    for s in shared:
        d = B[s][metric] - A[s][metric]
        trend = "" if prev is None else ("shrinking" if abs(d) < abs(prev) else "growing")
        print(f"{s:>8} {A[s][metric]:>14.4f} {B[s][metric]:>14.4f} {d:>+10.4f}  {trend}")
        prev = d
    ds = [B[s][metric] - A[s][metric] for s in shared]
    print(f"\ndelta: first {ds[0]:+.4f} -> last {ds[-1]:+.4f}  "
          f"(|last|/|first| = {abs(ds[-1]) / max(abs(ds[0]), 1e-9):.2f})")
    print("if |last|/|first| -> ~0 the measured effect is a convergence-rate "
          "difference that vanishes with training, not an asymptote difference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--out")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--repair-batches", type=int, default=20)
    ap.add_argument("--compare", nargs=2, metavar=("A.jsonl", "B.jsonl"))
    ap.add_argument("--metric", default="bpc")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1], args.metric)
        return
    for req in ("config", "ckpt_dir", "out"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_', '-')} is required to build a curve")
    build_curve(args.config, args.ckpt_dir, args.out, args.batches, args.repair_batches)


if __name__ == "__main__":
    main()
