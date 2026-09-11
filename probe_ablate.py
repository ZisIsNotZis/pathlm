"""Ticket 10 E1–E3: eval-only ablations + redo refinement probe.

E1: L4 with p_redo forced 0 vs trained (same eval seeds) — redo's
    inference-time contribution.
E3: L1 with p_skip forced 0 vs trained — skip's inference-time contribution.
E2: redo refinement probe — CE through the frozen head after 1× vs 2×
    execution of the same layer on the same input (L4 checkpoint).
"""
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tts as P  # noqa: E402
from pathlm.data import batch  # noqa: E402
from pathlm.config import sample_path  # noqa: E402


@torch.no_grad()
def ablation(name: str, knob: str, out: dict) -> None:
    m = P.load(name)
    r = {"trained": P.bpc_k(m, blob["val"], 1, n_batches=60)["bpc"]}
    saved = getattr(m.pcap, knob)
    setattr(m.pcap, knob, 0.0)
    r["forced_off"] = P.bpc_k(m, blob["val"], 1, n_batches=60)["bpc"]
    setattr(m.pcap, knob, saved)
    out[name] = r
    print(f"{name} {knob}: trained={r['trained']} forced_off={r['forced_off']} "
          f"delta={r['trained'] - r['forced_off']:+.4f}", flush=True)


@torch.no_grad()
def redo_refinement(name: str, out: dict) -> None:
    """CE of node-1 after 1x vs 2x execution of the SAME layer on the SAME
    input — the direct measure of whether a repeat refines its own output."""
    m = P.load(name)
    m.pcap.p_redo = 0.0  # deterministic single-pass paths
    T = m.mcfg.seq_len
    rng = random.Random(21)
    gen = torch.Generator().manual_seed(21)
    ce1, ce2, agree = [], [], []
    for _ in range(60):
        x, _ = batch(blob["val"], 32, T, gen)
        x = x.cuda()
        path = sample_path(m.pcap, rng, m.mcfg.n_layers)
        path.layer_repeats = [1] * len(path.layer_order)
        _, aux = m(x, [path], x)
        h_after = aux["rounds"][0][0]["latent"]  # post-stack latent
        # re-run the LAST executed layer once more on the same latent
        last = path.layer_order[-1]
        blk = m.blocks[last]
        h2 = h_after
        h2, _ = blk(h2)
        nodes1 = m._mtp_nodes(h_after)
        nodes2 = m._mtp_nodes(h2)
        tgt = x
        for nodes, acc in ((nodes1, ce1), (nodes2, ce2)):
            logits = nodes[1]["logits"][:, :-1]
            acc.append(torch.nn.functional.cross_entropy(
                logits.reshape(-1, m.vocab_size), tgt[:, 1:].reshape(-1),
                reduction="mean").item())
        agree.append((nodes1[1]["logits"].argmax(-1) == nodes2[1]["logits"].argmax(-1))
                     .float().mean().item())
    out[name] = {"ce_1x": round(sum(ce1) / len(ce1), 4),
                 "ce_2x": round(sum(ce2) / len(ce2), 4),
                 "top1_agree": round(sum(agree) / len(agree), 4)}
    print(f"{name} redo refinement: {out[name]}", flush=True)


if __name__ == "__main__":
    blob = np.load("data/enwik8_full.npz")
    out = {}
    ablation("L4", "p_redo", out)   # E1
    ablation("L1", "p_skip", out)   # E3
    redo_refinement("L4", out)      # E2
    with open(".scratch/10-gap-sweep/evidence/eval_ablations.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved eval_ablations.json")
