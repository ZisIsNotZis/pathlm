"""2520-spec-at-scale 复现入口：spec 宽度缩放的接受级联 + 三口径账。

- **native 对照必须走同一条解码路径**：`decode()` 内部逐 token 调
  `Decoder.step()`（不是批量 teacher-forcing）。spec 臂走 `decode_spec()`
  的 `step_multi()`。两臂共用同一 `Decoder` 类与 KV/驱逐语义。
- 墙钟：同进程**逐次交替** plain/spec，报每次重复的 tok/s 与极差（min–max）。
- 算力：`R = (1+γ)/(1+τ)`，γ=max_drafts、τ=tokens_per_forward−1，
  以 native 逐 token = 1.0 的 width-1 前向为单位（纯解析，与 GPU 负载无关）。
  `c₃`（实测 width-3/width-1 墙钟比）只作墙钟描述量，**不代入 R**。

协议与 `probe_spec.py` 的 `micro_forward_cost`/`steady_state` 一致（同
PROMPT_LEN/N_NEW/REPS/seed/交错方式），本脚本额外记录每次重复以便算极差。

用法：
    python3 run.py --ckpt .scratch/20-spec-scale/evidence/INT3_100M_s0 \
                   --out results.json [--reps 5] [--ledger-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import probe_tts as P                                    # noqa: E402
import probe_spec as PS                                 # noqa: E402
from pathlm.decode import Decoder, decode, decode_spec  # noqa: E402

PROMPT_LEN, N_NEW = 32, 400


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def steady_state_reps(m, prompt, n_new, pcap, max_drafts, reps):
    """同进程逐次交替 plain/spec，返回每次重复的 tok/s（用于极差）+ 接受级联。"""
    with torch.no_grad():
        for _ in range(3):  # warm
            decode(m, prompt, 8, pcap)
            decode_spec(m, prompt, 8, pcap, max_drafts=max_drafts)
        plain_reps, spec_reps = [], []
        plain_seq = spec_seq = None
        st = None
        for _ in range(reps):
            sync(); t0 = time.time()
            plain_seq, _ = decode(m, prompt, n_new, pcap)
            sync(); plain_reps.append(n_new / (time.time() - t0))
            sync(); t0 = time.time()
            spec_seq, st = decode_spec(m, prompt, n_new, pcap,
                                       max_drafts=max_drafts)
            sync(); spec_reps.append(n_new / (time.time() - t0))
    divergence = None
    if plain_seq != spec_seq:
        i = next((j for j, (a, b) in enumerate(zip(plain_seq, spec_seq)) if a != b),
                 min(len(plain_seq), len(spec_seq)))
        divergence = {"first_mismatch": i, "plain_len": len(plain_seq),
                      "spec_len": len(spec_seq),
                      "mismatch_tokens": sum(1 for a, b in zip(plain_seq, spec_seq)
                                             if a != b)}
        if i < len(plain_seq):
            dd = Decoder(m, window=pcap.window, anchors=pcap.anchors)
            for t in prompt.tolist() + plain_seq[:i]:
                dd.step(t)
            lg = dd.last_nodes[1]["logits"][0, 0]
            t2 = torch.topk(lg, 2)
            divergence["top2_gap_at_flip"] = round(
                (t2.values[0] - t2.values[1]).item(), 6)
    offered = max(st["drafts_offered"], 1)
    acc = st["accepts"]
    n1 = sum(a >= 1 for a in acc)
    a1 = n1 / max(len(acc), 1)
    a2 = (sum(a >= 2 for a in acc) / max(n1, 1))
    return {
        "plain_tok_s_reps": [round(x, 1) for x in plain_reps],
        "spec_tok_s_reps": [round(x, 1) for x in spec_reps],
        "plain_tok_s_best": round(max(plain_reps), 1),
        "spec_tok_s_best": round(max(spec_reps), 1),
        "plain_tok_s_minmax": [round(min(plain_reps), 1), round(max(plain_reps), 1)],
        "spec_tok_s_minmax": [round(min(spec_reps), 1), round(max(spec_reps), 1)],
        # 同进程逐次交替 ⇒ 每个 rep 内 plain 与 spec 相邻；报「按 rep 配对」的 speedup
        # 与其极差，绝对墙钟在外部占用下不可信
        "speedup_reps_paired": [round(s / p, 3) for s, p in zip(spec_reps, plain_reps)],
        "speedup_ratio_of_best": round(max(spec_reps) / max(plain_reps), 3),
        "accept_rate": round(st["drafts_accepted"] / offered, 4),
        "online_a2": round(a1, 4),
        "online_a3_given_a2": round(a2, 4),
        "n_forward": st["n_forward"],
        "n_rounds": len(acc),
        "tokens_per_forward": round(n_new / max(st["n_forward"], 1), 4),
        "seq_len": n_new,
        "greedy_divergence": divergence,
    }


def compute_ledger(gamma: int, tok_per_fwd: float, speedup: float,
                   c_wall: float | None) -> dict:
    """每 token 解析算力比 R=(1+γ)/(1+τ)；c_wall 只作对照，不进 R。"""
    tau = tok_per_fwd - 1.0
    R = (1 + gamma) / (1 + tau)
    out = {
        "gamma_drafts": gamma,
        "tau_accepted": round(tau, 4),
        "tokens_per_forward": round(tok_per_fwd, 4),
        "per_token_compute_ratio__analytic": round(R, 4),
        "native_reference": 1.0,
        "verdict": "loss" if R > 1.05 else ("gain" if R < 0.95 else "flat"),
        "wall_speedup": round(speedup, 4),
    }
    if c_wall is not None:
        # 旧口径：把墙钟 c_wall 当 width-(1+γ)/width-1 算力比 ⇒ c_wall/(tok/fwd)
        out["old_wallclock_ledger__c_wall_over_tokfwd"] = round(
            c_wall / (1 + tau), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".scratch/20-spec-scale/evidence/INT3_100M_s0")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    blob = np.load(os.path.join(ROOT, "data/enwik8_full.npz"))
    m = P.load_ckpt(os.path.join(ROOT, args.ckpt) if not os.path.isabs(args.ckpt)
                    else args.ckpt)
    pcap = m.pcap
    if pcap.n_mtp < 3:
        raise SystemExit(f"n_mtp={pcap.n_mtp} < 3：本脚本协议针对 width-3 模型")
    rng = torch.Generator().manual_seed(5)
    rand_prompt = torch.randint(0, m.n_real_tokens, (PROMPT_LEN,),
                                generator=rng).cuda()
    real_prompt = torch.tensor(blob["val"][:PROMPT_LEN].astype(np.int64)).cuda()

    res: dict = {"checkpoint": args.ckpt, "n_mtp": pcap.n_mtp, "reps": args.reps}
    # 先测步时（micro，同 probe_spec 协议）再定预算
    res["micro"] = PS.micro_forward_cost(m, pcap)
    print("micro:", res["micro"], flush=True)
    c3 = res["micro"]["w3_vs_w1"]
    for src, prompt in (("random", rand_prompt), ("enwik8", real_prompt)):
        for drafts in (1, 2):
            key = f"{src}_k{drafts}"
            r = steady_state_reps(m, prompt, N_NEW, pcap, drafts, args.reps)
            r["speedup_best"] = r["speedup_ratio_of_best"]
            r["ledger"] = compute_ledger(drafts, r["tokens_per_forward"],
                                         r["speedup_best"], c3 if drafts == 2 else None)
            res[key] = r
            print(f"{key}: R={r['ledger']['per_token_compute_ratio__analytic']} "
                  f"tok/fwd={r['tokens_per_forward']} wall={r['speedup_best']}x "
                  f"speedup_reps_paired={r['speedup_reps_paired']}", flush=True)
            # 增量落盘：任一后续步骤报错也不丢已完成臂
            with open(os.path.join(HERE, "results.partial.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)

    out = os.path.abspath(args.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
