"""Ticket 10 T2: MTP accept-rate + probability composition on a n_mtp=2 ckpt.

1. node accuracy: node1 (t+1), node2_direct (t+2), node2_chain (t+2).
2. Draft accept rate: node-2 as the draft for t_{i+2} — verified by the next
   forward's node-1 at row i+1 (predicting t_{i+2}): P(node2 argmax == node1
   at i+1 argmax) and P(accept | node1 correct).
3. Composition: P_comp(t_{i+2}) = sum_t P1(t | i+1) * P2chain(t | i) — the
   chain through node-1's next-step marginal; compare CE(comp) vs CE(node2)
   vs CE(node1 shifted) — does composing beat either head alone?
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
from pathlm.data import batch  # noqa: E402
from pathlm.config import sample_path  # noqa: E402


@torch.no_grad()
def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ".scratch/10-gap-sweep/evidence/B2"
    m = P.load_ckpt(ckpt)
    assert m.pcap.n_mtp >= 2, "need n_mtp=2"
    T = m.mcfg.seq_len
    rng = random.Random(31)
    gen = torch.Generator().manual_seed(31)
    acc1, acc2d, acc2c = [], [], []
    agree_draft, accept_given_right = [], []
    ce_n1_shift, ce_n2, ce_comp = [], [], []
    for _ in range(60):
        x, _ = batch(blob["val"], 32, T, gen)
        x = x.cuda()
        _, aux = m(x, [sample_path(m.pcap, rng, m.mcfg.n_layers)], x)
        n1, n2 = aux["rounds"][0][1], aux["rounds"][0][2]
        # targets: node1 at row i predicts x[i+1]; node2 at row i predicts x[i+2]
        t1, t2 = x[:, 1:], x[:, 2:]
        l1 = n1["logits"][:, :-2]      # rows 0..T-3 predicting t1[i]
        l2d = n2["logits"][:, :-2]     # node-2 direct: t2
        l2c = n2["chain2"]["logits"][:, :-2] if "chain2" in n2 else l2d
        a1 = l1.argmax(-1) == t1
        a2d = l2d.argmax(-1) == t2
        a2c = l2c.argmax(-1) == t2
        acc1.append(a1.float().mean().item())
        acc2d.append(a2d.float().mean().item())
        acc2c.append(a2c.float().mean().item())
        # draft accept: at row i, node-1 predicts t_{i+1}; node-2 at row i-1
        # drafted t_{i+1} too (i.e. l2d at row i-1). Verify: draft == node1's
        # prediction at the verification row.
        draft = l2d[:, :-1]            # drafted t_{i+1} for rows 1..T-3
        verify = l1[:, 1:]             # node-1's own t_{i+1} estimate
        draft_ok = draft.argmax(-1) == verify.argmax(-1)
        agree_draft.append(draft_ok.float().mean().item())
        # accept given node-1 was right at the verification row
        v_right = (verify.argmax(-1) == t1[:, 1:])
        if v_right.any():
            accept_given_right.append(draft_ok[v_right].float().mean().item())
        # composition: P_comp(t_{i+2}) = sum_t P1(t|i+1) * P2chain(t|i)
        p1 = l1.softmax(-1)            # [B, T-2, V] over t_{i+1}
        p2 = l2c.softmax(-1)           # [B, T-2, V] over t_{i+2}
        # chain: sum_t p1[t] * p2[t] — alignment p2[b, i, :] with p1[b, i, :]
        comp = torch.einsum("btv,btv->bv", p1, p2)
        comp = comp / comp.sum(-1, keepdim=True).clamp_min(1e-12)
        ce_n1_shift.append(F.cross_entropy(l2d.reshape(-1, m.vocab_size),
                                           t2.reshape(-1)).item())
        ce_n2.append(F.cross_entropy(l2c.reshape(-1, m.vocab_size),
                                     t2.reshape(-1)).item())
        ce_comp.append(F.cross_entropy(comp, t2).item())
    out = {
        "acc_node1_t1": round(sum(acc1) / len(acc1), 4),
        "acc_node2_direct_t2": round(sum(acc2d) / len(acc2d), 4),
        "acc_node2_chain_t2": round(sum(acc2c) / len(acc2c), 4),
        "draft_agree_with_node1": round(sum(agree_draft) / len(agree_draft), 4),
        "accept_given_node1_right": round(sum(accept_given_right) / len(accept_given_right), 4),
        "ce_node2_direct_t2": round(sum(ce_n1_shift) / len(ce_n1_shift), 4),
        "ce_node2_chain_t2": round(sum(ce_n2) / len(ce_n2), 4),
        "ce_composed_t2": round(sum(ce_comp) / len(ce_comp), 4),
    }
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(ckpt, "mtp_accept.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    blob = np.load("data/enwik8_full.npz")
    main()
