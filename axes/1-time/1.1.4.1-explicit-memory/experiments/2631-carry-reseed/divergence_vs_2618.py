#!/usr/bin/env python3
"""2631-carry-reseed · 与 2618（ticket 35 的单次观测）的**同配置**对照：同 seed=0 重训。

用途：① 复核「2618 的读数是不是训练侧可复现的」；② 若逐位相同，说明本次的引擎修订
（ENG1/ENG2 之后）在**稠密 + 窗口 16** 这条路径上没有改变数值行为；若不同，则**如实报差**，
并把它记为「可能是代码修订或训练非确定性」——不得当作「同一次训练」。

口径：只比**训练日志**（每 200 步的 loss 与答案行 CE），不比评测读数（评测齐备由 `run.py`
自己出）。CPU-only，不取 GPU 锁。

用法：python3 divergence_vs_2618.py       # 输出 divergence_2618.json
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
THIS = "axes/1-time/1.1.4.1-explicit-memory/experiments/2631-carry-reseed"
PREV = "axes/1-time/1.1.4.1-explicit-memory/experiments/2618-carry-channel-13m"
PAIRS = [("none_s0", "U0"), ("plain_s0", "U1"), ("delta_s0", "U2"),
         ("plain_s0", "U1rep"), ("delta_s0", "U2"), ("none_s0", "U0")]


def load(rel: str) -> dict:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, rel)) if l.strip()]
    return {r["step"]: r for r in rows}


def main() -> int:
    out = {"baseline": "2618（ticket 35，commit cc64112 附近；单次观测）",
           "note": ("同 seed=0、同配方、同 token 预算；仅比训练日志的 loss 与答案行 CE。"
                    "两条日志的代码修订不同（ENG1/ENG2 之后），所以差异不能单独归因。"),
           "pairs": []}
    for mine, theirs in [("none_s0", "U0"), ("plain_s0", "U1"), ("delta_s0", "U2")]:
        a = load(f"{THIS}/train/{mine}/train_log.jsonl")
        b = load(f"{PREV}/train/{theirs}/train_log.jsonl")
        common = sorted(set(a) & set(b))
        dl = [abs(a[s]["loss"] - b[s]["loss"]) for s in common]
        da = [abs(a[s]["synth_answer_ce"] - b[s]["synth_answer_ce"]) for s in common
              if a[s].get("synth_answer_ce") is not None
              and b[s].get("synth_answer_ce") is not None]
        row = {"mine": mine, "baseline_arm": theirs, "n_common_steps": len(common),
               "max_abs_loss_diff": round(max(dl), 8) if dl else None,
               "max_abs_answer_ce_diff": round(max(da), 8) if da else None,
               "last_step": common[-1] if common else None,
               "mine_last": {k: a[common[-1]].get(k) for k in
                             ("loss", "synth_answer_ce")} if common else None,
               "baseline_last": {k: b[common[-1]].get(k) for k in
                                 ("loss", "synth_answer_ce")} if common else None,
               "bitwise_reproduced": bool(dl and max(dl) == 0.0 and (not da or max(da) == 0.0))}
        out["pairs"].append(row)
        print(f"{mine} vs {theirs}: 同 step {row['n_common_steps']}；"
              f"max|Δloss|={row['max_abs_loss_diff']} "
              f"max|ΔanswerCE|={row['max_abs_answer_ce_diff']} "
              f"逐位复现={row['bitwise_reproduced']}")
    out["all_bitwise"] = all(p["bitwise_reproduced"] for p in out["pairs"])
    out["reading"] = ("同 seed 重训与 2618 **逐位相同** ⇒ 训练在该配方上是确定性的"
                      "（差异来源只剩跨 seed 与代码修订）；否则按实测差如实报，"
                      "不称「同一次训练」。")
    with open(os.path.join(HERE, "divergence_2618.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
