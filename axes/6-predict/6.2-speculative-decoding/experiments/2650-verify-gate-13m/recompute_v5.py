"""离线重算 `results.json::checkpoints[*].criteria.V5_cost_ledger`（CPU，不取 GPU 锁）。

**为什么存在**：`verify_gate.v5_ledger` 首次落盘时有一个实现 bug（`tau_of` 的键是 int `K`，
阈值分支却用字符串 `"1"/"2"` 去查 ⇒ `k4_to_k8_threshold` 恒为 `None`）。该量**完全**由已落盘的
逐臂 τ/α₁ 派生、**不含任何重新测量**，因此按「同一口径、可复现」的原则离线重算并回写，
而不是重跑一轮（重跑只会引入 GPU 噪声、不改变结果）。

口径未改：仍用 `verify_gate.v5_ledger`（已修 bug 的版本）与 2640 `oracle_accept` 指针。
重算后 `criteria.V5_cost_ledger` 与 `recomputed_at` 一并回写；其余字段不动。

跑法：python3 recompute_v5.py            # 就地回写 results.json
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import verify_gate as V  # noqa: E402

RESULTS = os.path.join(HERE, "results.json")


def _arm_sum_from_stored(ck: dict) -> dict:
    """从落盘臂数据重建 `v5_ledger` 需要的字段（只 τ / α₁）。"""
    return {a: {"tau": v.get("tau"), "alpha1": v.get("alpha1")}
            for a, v in ck.get("arms", {}).items()}


def main() -> int:
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    with open(V.RESULTS_2640, encoding="utf-8") as f:
        ref = json.load(f)
    for name, ck in res["checkpoints"].items():
        arm_sum = _arm_sum_from_stored(ck)
        ck["criteria"]["V5_cost_ledger"] = V.v5_ledger(arm_sum, ref, name, V.GAMMA)
    res["v5_recomputed_note"] = ("criteria.V5_cost_ledger 由 recompute_v5.py 从已落盘 τ/α₁ "
                                 "离线重算（修 int/str 键 bug；不含重新测量）")
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for name, ck in res["checkpoints"].items():
        thr = ck["criteria"]["V5_cost_ledger"]["k4_to_k8_threshold"]
        print(f"{name}: k4_to_k8_threshold={thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
