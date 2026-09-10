"""Ticket 08 — test-time scaling probes over existing checkpoints.

T1 path-marginal ensemble: average node-1 softmax over K sampled paths.
T2 depth ensemble: average per-depth node-1 softmax (one forward, free).
T3 round ensemble: average node-1 across retry rounds on the clean stream.
T4 temperature: rescale the averaged logits (single temperature, held-out fit).

Every probe reports bpc vs the B0 anchor 1.5074 and K's eval-compute multiple.
No training; loads checkpoints from the evidence dirs.
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import batch
from pathlm.model import PathLM

B0_BPC = 1.5074
EPS = 0.0005  # saturation threshold (below measurement noise)

CKPTS = {
    "L1": ".scratch/06-m2-sweep/evidence/L1",
    "L3": ".scratch/06-m2-sweep/evidence/L3",
    "L4": ".scratch/06-m2-sweep/evidence/L4",
    "L2": ".scratch/04-m1-runs/evidence/L2",
    "X1": ".scratch/07-x1-x3/evidence/X1",
    "X3": ".scratch/07-x1-x3/evidence/X3",
    "C1": ".scratch/04-m1-runs/evidence/C1",
    "C3": ".scratch/04-m1-runs/evidence/C3",
    "C4": ".scratch/05-m1-followup/evidence/C4",
    "I3": ".scratch/06-m2-sweep/evidence/I3",
}


def load(name: str) -> PathLM:
    root = CKPTS[name]
    with open(os.path.join(root, "results.json")) as f:
        cfg = json.load(f)["config"]
    m = PathLM(ModelConfig(**cfg["model"]), PathConfig(**cfg["path"]), vocab_size=206)
    sd = torch.load(os.path.join(root, "model.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval().cuda()
    return m


@torch.no_grad()
def bpc_k(model: PathLM, eval_arr, k_paths: int, n_batches: int = 60,
          batch_size: int = 32, temperature: float = 1.0) -> dict:
    """node-1 bpc averaged over k sampled paths per batch (T1). CE is computed
    per batch-row BEFORE averaging across paths (mixture of predictions, not
    of rows)."""
    T = model.mcfg.seq_len
    rng = random.Random(11)
    gen = torch.Generator().manual_seed(11)
    nats, tokens, correct = 0.0, 0, 0
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.cuda()
        probs = torch.zeros(batch_size, T - 1, 206, device="cuda")
        for _k in range(k_paths):
            path = sample_path(model.pcap, rng, model.mcfg.n_layers)
            _, aux = model(x, [path], x)
            logits = aux["rounds"][0][1]["logits"][:, :-1] / max(temperature, 1e-6)
            probs = probs + logits.softmax(-1)
        probs = probs / k_paths
        tgt = x[:, 1:]
        nats += -(probs.clamp_min(1e-12).log().gather(2, tgt.unsqueeze(-1)).sum()).item()
        correct += (probs.argmax(-1) == tgt).sum().item()
        tokens += tgt.numel()
    return {"bpc": round(nats / tokens / 0.6931471805599453, 4),
            "acc": round(correct / tokens, 4)}


@torch.no_grad()
def bpc_depth_ens(model: PathLM, eval_arr, mode: str = "uniform",
                  n_batches: int = 60, batch_size: int = 32) -> dict:
    """T2: average per-depth node-1 predictions (collect_depth_logits hook).
    mode: uniform | confidence (sigmoid-conf-weighted)."""
    T = model.mcfg.seq_len
    rng = random.Random(12)
    gen = torch.Generator().manual_seed(12)
    nats, tokens, correct = 0.0, 0, 0
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        model.pcap.collect_depth_logits = True
        path = sample_path(model.pcap, rng, model.mcfg.n_layers)
        _, aux = model(x, [path], x)
        model.pcap.collect_depth_logits = False
        dl = torch.stack(aux["depth_logits"])   # [D, B, T, V]
        dc = torch.stack(aux["depth_conf"])     # [D, B, T]
        if mode == "confidence":
            w = dc.sigmoid().unsqueeze(-1)      # [D, B, T, 1]
        else:
            w = torch.ones_like(dc).unsqueeze(-1)
        logits = (dl * w).sum(0) / w.sum(0)     # [B, T, V]
        logits = logits[:, :-1]
        tgt = x[:, 1:]
        lp = logits.log_softmax(-1)
        nats += -lp.gather(2, tgt.unsqueeze(-1)).sum().item()
        correct += (logits.argmax(-1) == tgt).sum().item()
        tokens += tgt.numel()
    return {"bpc": round(nats / tokens / 0.6931471805599453, 4),
            "acc": round(correct / tokens, 4)}


@torch.no_grad()
def bpc_round_ens(model: PathLM, eval_arr, k_rounds: int, n_batches: int = 60,
                  batch_size: int = 32) -> dict:
    """T3: average node-1 across k forced retry rounds on the CLEAN stream."""
    T = model.mcfg.seq_len
    rng = random.Random(13)
    gen = torch.Generator().manual_seed(13)
    nats, tokens, correct = 0.0, 0, 0
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        paths = []
        for _ in range(k_rounds):
            p = sample_path(model.pcap, rng, model.mcfg.n_layers)
            p.n_retries = 0
            p.token_retry = False
            paths.append(p)
        _, aux = model(x, paths, x)
        probs = torch.zeros(batch_size, T - 1, 206, device="cuda")
        for r in range(len(aux["rounds"])):
            logits = aux["rounds"][r][1]["logits"][:, :-1]
            probs = probs + logits.softmax(-1)
        probs = probs / len(aux["rounds"])
        tgt = x[:, 1:]
        nats += -(probs.clamp_min(1e-12).log().gather(2, tgt.unsqueeze(-1)).sum()).item()
        correct += (probs.argmax(-1) == tgt).sum().item()
        tokens += tgt.numel()
    return {"bpc": round(nats / tokens / 0.6931471805599453, 4),
            "acc": round(correct / tokens, 4)}


def main() -> None:
    blob = np.load("data/enwik8_full.npz")
    eval_arr = blob["val"]  # held-out 10M tail — never train on it
    results: dict = {}

    # ---- T1: path-marginal ensemble, K sweep ----
    for name in ["L1", "L3", "L4", "X1", "X3", "I3"]:
        m = load(name)
        res = {}
        for k in (1, 2, 4, 8):
            res[f"K{k}"] = bpc_k(m, eval_arr, k)
            print(f"T1 {name} K={k}: bpc={res[f'K{k}']['bpc']} acc={res[f'K{k}']['acc']}",
                  flush=True)
        results[f"T1:{name}"] = res

    # ---- T2: depth ensemble (free) ----
    m = load("L2")
    for mode in ("uniform", "confidence"):
        r = bpc_depth_ens(m, eval_arr, mode)
        results[f"T2:{mode}"] = r
        print(f"T2 L2 {mode}: bpc={r['bpc']} acc={r['acc']}", flush=True)

    # ---- T3: round ensemble (clean stream) ----
    for name in ["C1", "C3", "C4"]:
        m = load(name)
        for k in (2, 4):
            r = bpc_round_ens(m, eval_arr, k)
            results[f"T3:{name}:K{k}"] = r
            print(f"T3 {name} K={k}: bpc={r['bpc']} acc={r['acc']}", flush=True)

    out = ".scratch/08-tts-probe/evidence/tts_results.json"
    # pi-lens-ignore: unchecked-throwing-call-python
    os.makedirs(".scratch/08-tts-probe/evidence", exist_ok=True)
    # pi-lens-ignore: unchecked-throwing-call-python
    with open(out, "w") as f:  # intended raise if the dir cannot be created
        json.dump({"anchor_b0": B0_BPC, "results": results}, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
