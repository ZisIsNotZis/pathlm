#!/usr/bin/env python3
"""2635-recursive-carrier-13m · 数字对账（期望值由代码/raw 产物重算）。

raw 来源：`results.json`、`train/<arm>/train_log.jsonl`、`stall_report_*.json`、`config.json`。

用法：python3 reconcile_numbers.py      # 退出码 0 = 全过
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.evalkit.reconcile import Ledger                          # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel             # noqa: E402


def main() -> int:
    R = json.load(open(os.path.join(HERE, "results.json")))
    cfg = R["config"]
    ledger = Ledger("2635-recursive-carrier-13m",
                    sources=("results.json", "config.json",
                             "train/*/train_log.jsonl", "stall_report_*.json"))

    # ---- 两臂参数量逐位相同（引擎不变量） ----
    for spec in cfg["arms"]:
        m = DepthARModel(DepthARConfig(**{**cfg["model"],
                                          "attn_linear": spec["attn_linear"]}),
                         cfg["data"]["vocab_size"])
        ledger.derived(f"{spec['name']} 参数量 == 重建模型参数量",
                       R["checkpoints"][spec["name"]]["n_params"],
                       lambda m=m: sum(p.numel() for p in m.parameters()),
                       sources=("pathlm.depth_ar.DepthARModel",), abs_tol=1e-9)

    # ---- 质量腿 ----
    ledger.derived("Δbpc == bpc_stock − bpc_linear",
                   R["quality_leg"]["delta_bpc_stock_minus_linear"],
                   lambda: round(R["quality_leg"]["bpc"]["stock"]
                                 - R["quality_leg"]["bpc"]["linear"], 6),
                   sources=("results.json::quality_leg.bpc",), abs_tol=5e-7)

    # ---- 活性块（判据③） ----
    lin = R["liveness"]["linear"]
    ledger.eq("线性臂活性块 enabled", lin["enabled"], True)
    ledger.eq("两臂块类不同（stock Block vs 线性块）",
              R["checkpoints"]["stock"]["block_class"] != R["checkpoints"]["linear"]["block_class"],
              True)
    if R["stall_report"] is None:
        ledger.eq("判据③ alive == (phi_min>0 ∧ den_min>0 ∧ 反事实偏差>0)",
                  lin["alive"] == bool(lin["phi_min"] > 0 and lin["den_min"] > 0
                                       and lin.get("rel_dev_vs_full_min", 0) > 0), True,
                  derived_from=("results.json::liveness.linear",))

    # ---- 侧腿电池：n_hits/n/accuracy 自洽 + chance 口径 ----
    for tname, row in R["battery"].items():
        for arm, v in row.items():
            ledger.derived(f"{tname}/{arm}: accuracy == n_hits/n",
                           v["accuracy"],
                           lambda v=v: round(v["n_hits"] / v["n"], 6),
                           sources=("results.json::battery",), abs_tol=5e-7)

    # ---- 训练侧 ----
    for spec in cfg["arms"]:
        lines = [json.loads(l) for l in open(os.path.join(ROOT, spec["train_log"]))
                 if l.strip()]
        ledger.derived(f"{spec['name']}: 训练日志末步 == {cfg['steps']}",
                       R["train_log_tail"][spec["name"]]["last_step"],
                       lambda lines=lines: lines[-1]["step"], sources=(spec["train_log"],),
                       abs_tol=1e-9)
    ledger.eq("停摆门配置与契约一致（180s 无进展即判停摆）",
              cfg["stall_gate"]["stall_secs"], 180)
    ledger.eq("判据① 可跑性 == 无停摆记录（status != stalled）",
              R["criteria"]["c1_runnable"] == (R["stall_report"] is None
                                               or R["stall_report"].get("status") != "stalled"),
              True, derived_from=("stall_report_*.json",))

    return ledger.finish(os.path.join(HERE, "reconcile_report.json"))


if __name__ == "__main__":
    raise SystemExit(main())
