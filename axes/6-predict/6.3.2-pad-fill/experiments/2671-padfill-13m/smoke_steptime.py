#!/usr/bin/env python3
"""2671-padfill-13m · 步时冒烟（S1；契约 §3 唯一允许的调整）。

只量**步时**（batch 32 / T=512 / 交付结构），据「5 臂 × 2 seed = 10 run 的训练总时 ≤ 25 min」
反解 matched-step `S`（取整到 500 的倍数、≥1500）。墙钟**只**出现在本脚本输出与
`S1_REPORT.md` 的过程记录里，绝不进 `results.json` / `analysis.md` 的判据。

    python3 smoke_steptime.py --steps 120 --warmup 10
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import padfill as P  # noqa: E402
from gpu_lock import gpu_lock  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120, help="计时步数（契约要求 >=100）")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--arm", default="fill_bidir_r25")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--out", default=None, help="结果 json 路径（默认只打印）")
    a = ap.parse_args()
    if a.steps < 100:
        raise SystemExit("契约要求步时冒烟 >= 100 step")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    train_arr, _val, vocab = P.load_data()
    print(f"[smoke] device={device} arm={a.arm} vocab={vocab} "
          f"batch={P.BATCH_SIZE} T={P.SEQ_LEN} steps={a.steps} warmup={a.warmup}", flush=True)

    ctx = contextlib.nullcontext() if a.no_lock else gpu_lock(name="wp6.3.2-smoke")
    with ctx:
        torch.manual_seed(0)
        model = P.PadFillLM(attn_mask=P.ARMS[a.arm]["attn_mask"]).to(device)
        opt = P.build_optimizer(model)
        dg = torch.Generator().manual_seed(P.DATA_SEED)
        mg = torch.Generator().manual_seed(P.MASK_SEED)
        print(f"[smoke] params={P.count_params(model):,} lock_held={getattr(ctx, 'held', True)}",
              flush=True)
        P.train_steps(model, opt, a.arm, a.warmup, train_arr, device, dg, mg)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        P.train_steps(model, opt, a.arm, a.steps, train_arr, device, dg, mg,
                      step_offset=a.warmup)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    steptime = elapsed / a.steps
    S = P.steps_from_steptime(steptime)
    proj = P.INF_RUNS * S * steptime
    rec = {
        "device": device, "arm": a.arm, "timed_steps": a.steps, "warmup_steps": a.warmup,
        "elapsed_s": elapsed, "step_time_s": steptime, "params": P.count_params(model),
        "S_selected": S, "inf_runs": P.INF_RUNS, "inf_total_sec_budget": P.INF_TOTAL_SEC,
        "inf_total_sec_projected": proj, "step_quantum": P.STEP_QUANTUM,
        "step_floor": P.STEP_MIN,
    }
    print(f"[smoke] elapsed={elapsed:.2f}s over {a.steps} steps "
          f"-> step_time={steptime * 1000:.1f} ms/step", flush=True)
    print(f"[smoke] S = {S}  (10 run 训练预计 {proj:.0f}s ≤ {P.INF_TOTAL_SEC:.0f}s)", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print(f"[smoke] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
