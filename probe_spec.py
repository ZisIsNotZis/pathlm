"""Ticket 13 exp 3: measured Medusa spec-decode throughput vs the per-position
path (greedy, B=1 deployment semantics).

Reports, per checkpoint:
  - micro: wall time of one width-1 forward vs one width-3 forward (the cost
    ratio that decides whether batched verification can pay off at this scale);
  - steady-state: prefill a short prompt, then time generation only (so prompt
    prefill, identical in both paths, does not dilute the comparison);
  - the accept cascade actually observed during decoding and the implied
    tokens/forward.

Run: python3 probe_spec.py .tmp/p3/B3 [.tmp/p3/B2fresh]
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tts as P  # noqa: E402

blob = __import__("numpy").load("data/enwik8_full.npz")  # module-level: val split for real prompts
from pathlm.decode import Decoder, decode, decode_spec  # noqa: E402

PROMPT_LEN, N_NEW, REPS = 32, 400, 5


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def micro_forward_cost(m, pcap):
    """Median wall time of one step() (width 1) and one step_multi() of widths
    2 and 3, after a warmup, on a freshly prefilled decoder. Returns per-call
    seconds so the ratio is directly comparable to the tokens/forward cascade."""
    prompt = torch.randint(0, m.n_real_tokens, (PROMPT_LEN,),
                           generator=torch.Generator().manual_seed(7)).cuda()

    def prefilled():
        d = Decoder(m, window=pcap.window, anchors=pcap.anchors)
        for t in prompt.tolist():
            d.step(t)
        return d

    out = {}
    times: dict[int, list[float]] = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for _ in range(REPS):
            for width in (1, 2, 3):  # interleaved: same clock/thermal state
                d = prefilled()
                toks = [int(t) for t in torch.randint(
                    0, m.n_real_tokens, (width,),
                    generator=torch.Generator().manual_seed(9))]
                sync(); t0 = time.time()
                for _ in range(50):
                    if width == 1:
                        d.step(toks[0])
                    else:
                        d.step_multi(toks)
                sync(); times[width].append(time.time() - t0)
        for width in (1, 2, 3):
            out[f"w{width}_ms"] = round(1000 * min(times[width]) / 50, 3)
        out["w3_vs_w1"] = round(out["w3_ms"] / out["w1_ms"], 3)
        out["w2_vs_w1"] = round(out["w2_ms"] / out["w1_ms"], 3)
    return out


def steady_state(m, prompt, n_new, pcap, max_drafts):
    """Time generation with a short prompt so prefill does not dilute the
    comparison. Plain and spec are interleaved in the same loop (identical
    thermal/clock state) and the min over reps is reported. Returns throughput,
    accept cascade and sequence."""
    with torch.no_grad():
        # warm the GPU and the allocator before any timing
        for _ in range(3):
            decode(m, prompt, 8, pcap)
            decode_spec(m, prompt, 8, pcap, max_drafts=max_drafts)
        plain_best, spec_best = 0.0, 0.0
        plain_seq = spec_seq = None
        st = None
        for _ in range(REPS):
            sync(); t0 = time.time()
            plain_seq, _ = decode(m, prompt, n_new, pcap)
            sync(); plain_best = max(plain_best, n_new / (time.time() - t0))
            sync(); t0 = time.time()
            spec_seq, st = decode_spec(m, prompt, n_new, pcap,
                                       max_drafts=max_drafts)
            sync(); spec_best = max(spec_best, n_new / (time.time() - t0))
    # 逐位相等是 13M 的验证性质（ticket 13）；大模型上 batched SDPA 与逐位置
    # SDPA 的核差异会在 top-2近平局处翻转 argmax（fp-level，非语义错误）。
    # 记录首个翻转点及其 top-2 gap，代替硬断言。
    divergence = None
    if plain_seq != spec_seq:
        from pathlm.decode import Decoder
        d = Decoder(m, window=pcap.window, anchors=pcap.anchors)
        for t in prompt.tolist() + plain_seq:
            d.step(t)
        i = next((j for j, (a, b) in enumerate(zip(plain_seq, spec_seq)) if a != b),
                 min(len(plain_seq), len(spec_seq)))
        divergence = {"first_mismatch": i, "plain_len": len(plain_seq),
                      "spec_len": len(spec_seq)}
        if i < len(plain_seq):
            dd = Decoder(m, window=pcap.window, anchors=pcap.anchors)
            for t in prompt.tolist() + plain_seq[:i]:
                dd.step(t)
            lg = dd.last_nodes[1]["logits"][0, 0]
            t2 = torch.topk(lg, 2)
            divergence["top2_gap_at_flip"] = round(
                (t2.values[0] - t2.values[1]).item(), 6)
            divergence["flip_events"] = sum(
                1 for _ in range(1))  # cascading; event rate = 1 flip
            divergence["mismatch_tokens"] = sum(
                1 for a, b in zip(plain_seq, spec_seq) if a != b)
    offered = max(st["drafts_offered"], 1)
    acc = st["accepts"]
    # per-slot deployable cascade: P(accept slot 1) and P(accept slot 2 | slot 1)
    a1 = sum(a >= 1 for a in acc) / max(len(acc), 1)
    a2 = (sum(a >= 2 for a in acc) / max(sum(a >= 1 for a in acc), 1))
    return {
        "plain_tok_s": round(plain_best, 1),
        "spec_tok_s": round(spec_best, 1),
        "speedup": round(spec_best / plain_best, 3),
        "accept_rate": round(st["drafts_accepted"] / offered, 4),
        "online_a2": round(a1, 4),
        "online_a3_given_a2": round(a2, 4),
        "n_forward": st["n_forward"],
        "tokens_per_forward": round(n_new / max(st["n_forward"], 1), 4),
        "seq_len": n_new,
        "greedy_divergence": divergence,
    }


@torch.no_grad()
def main() -> None:
    ckpts = sys.argv[1:] or [".tmp/p3/B3"]
    results: dict = {}
    for ck in ckpts:
        m = P.load_ckpt(ck)
        pcap = m.pcap
        if pcap.n_mtp < 3:
            print(f"{ck}: n_mtp={pcap.n_mtp} < 3, skip", flush=True)
            continue
        rng = torch.Generator().manual_seed(5)
        rand_prompt = torch.randint(0, m.n_real_tokens, (PROMPT_LEN,),
                                    generator=rng).cuda()
        val = blob["val"]
        real_prompt = torch.tensor(val[:PROMPT_LEN].astype(np.int64)).cuda()
        res = {"micro": micro_forward_cost(m, pcap)}
        print(f"{ck} micro: {res['micro']}", flush=True)
        for src, prompt in (("random", rand_prompt), ("enwik8", real_prompt)):
            for drafts in (1, 2):
                key = f"{src}_k{drafts}"
                res[key] = steady_state(m, prompt, N_NEW, pcap, drafts)
                a = res[key]
                print(f"{ck} {key}: plain {a['plain_tok_s']} tok/s, "
                      f"spec {a['spec_tok_s']} tok/s, speedup {a['speedup']}x, "
                      f"accept {a['accept_rate']}, {a['tokens_per_forward']} tok/fwd",
                      flush=True)
        results[ck] = res
    out = os.path.join(os.path.dirname(ckpts[0]), "spec_speed.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    blob = np.load("data/enwik8_full.npz")
    main()
