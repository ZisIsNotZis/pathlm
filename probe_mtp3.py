"""Ticket 13 exp 3: MTP node accuracy + draft/verifier accept cascade.

For a checkpoint with n_mtp >= 2 this measures, on held-out enwik8:

1. per-node t+k accuracy and CE for k = 0..n_mtp (node k at row i predicts
   the clean token x[i+k]);
2. the accept cascade a_k for k >= 2 — the draft-vs-verifier agreement
   `argmax(node_k[i]) == argmax(node_1[i+k-1])` (node 1 at the verification
   row i+k-1 also predicts t_{i+k}).  Two versions are reported:
     - a_k            : unconditional P(agree)  = the Medusa acceptance
                        probability (what actually advances the sequence);
     - a_k_given_right: P(agree | verifier correct), the diagnostic quoted in
                        probe_mtp.py (its node-2 value was 0.7729 on B2);
3. the implied tokens/forward = 1 + a2 + a2*a3 (task's cascade approximation).

Writes a per-batch jsonl (one record per batch + a final summary record) next
to the checkpoint as mtp3_probe.jsonl.
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tts as P  # noqa: E402
from pathlm.config import sample_path  # noqa: E402
from pathlm.data import batch  # noqa: E402


@torch.no_grad()
def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ".tmp/p3/B3"
    m = P.load_ckpt(ckpt)
    T = m.mcfg.seq_len
    K = m.pcap.n_mtp
    rng = random.Random(31)
    gen = torch.Generator().manual_seed(31)
    acc = {k: [] for k in range(K + 1)}
    ce = {k: [] for k in range(K + 1)}
    agree = {k: [] for k in range(2, K + 1)}
    agree_right = {k: [] for k in range(2, K + 1)}
    out_path = os.path.join(ckpt, "mtp3_probe.jsonl")
    with open(out_path, "w") as fout:
        for b in range(60):
            x, _ = batch(blob["val"], 32, T, gen)
            x = x.cuda()
            _, aux = m(x, [sample_path(m.pcap, rng, m.mcfg.n_layers)], x)
            nodes = aux["rounds"][0]
            rec: dict = {"batch": b, "acc": {}, "ce": {}, "a": {}, "a_given_right": {}}
            for k in range(K + 1):
                if k not in nodes:
                    continue
                n = T - k  # i = 0..T-1-k, so node k predicts x[i+k] <= x[T-1]
                logits = nodes[k]["logits"][:, :n]
                tgt = x[:, k:k + n]
                ce[k].append(F.cross_entropy(logits.reshape(-1, m.vocab_size),
                                             tgt.reshape(-1)).item())
                acc_val = (logits.argmax(-1) == tgt).float().mean().item()
                acc[k].append(acc_val)
                rec["acc"][f"t+{k}"] = round(acc_val, 4)
                rec["ce"][f"t+{k}"] = round(ce[k][-1], 4)
            l1 = nodes[1]["logits"]
            for k in range(2, K + 1):
                if k not in nodes:
                    continue
                # node k[i] predicts t_{i+k}; verifier node-1 at row i+k-1
                n = T - k  # i = 0..T-k-1 (keeps i+k-1 <= T-2, x[i+k] <= x[T-1])
                draft = nodes[k]["logits"][:, :n].argmax(-1)
                vlogits = l1[:, k - 1:k - 1 + n]
                ok = draft == vlogits.argmax(-1)
                a = ok.float().mean().item()
                agree[k].append(a)
                rec["a"][f"a{k}"] = round(a, 4)
                vright = vlogits.argmax(-1) == x[:, k:k + n]
                ar = ok[vright].float().mean().item() if vright.any() else float("nan")
                agree_right[k].append(ar)
                rec["a_given_right"][f"a{k}"] = round(ar, 4)
            fout.write(json.dumps(rec) + "\n")

        def mean(v):
            return round(sum(v) / len(v), 4) if v else None

        summary = {
            "ckpt": ckpt,
            "n_mtp": K,
            "n_batches": 60,
            "batch_size": 32,
            "seq_len": T,
            "acc": {f"t+{k}": mean(acc[k]) for k in range(K + 1)},
            "ce": {f"t+{k}": mean(ce[k]) for k in range(K + 1)},
            "a": {f"a{k}": mean(agree[k]) for k in range(2, K + 1)},
            "a_given_right": {f"a{k}": mean(agree_right[k]) for k in range(2, K + 1)},
        }
        a2 = summary["a"].get("a2")
        a3 = summary["a"].get("a3")
        summary["tokens_per_forward"] = (
            round(1 + a2 + (a2 * a3 if a3 is not None else 0.0), 4)
            if a2 is not None else None)
        a2r = summary["a_given_right"].get("a2")
        a3r = summary["a_given_right"].get("a3")
        summary["tokens_per_forward_given_right"] = (
            round(1 + a2r + (a2r * a3r if a3r is not None else 0.0), 4)
            if a2r is not None else None)
        fout.write(json.dumps({"summary": summary}) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    blob = np.load("data/enwik8_full.npz")
    main()
