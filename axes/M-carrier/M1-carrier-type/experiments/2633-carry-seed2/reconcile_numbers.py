#!/usr/bin/env python3
"""2633-carry-seed2 · 数字对账（期望值一律由 raw 产物**重算**）。

raw 来源：`results.json`、`cluster_hits.json`（逐序列原始命中数）、`config.json`、
`train/<arm>/train_log.jsonl`、`equality_report.json`（SCHEMA §9 复用证据）。

用法：python3 reconcile_numbers.py      # 退出码 0 = 全过
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RES = os.path.abspath(os.path.join(HERE, "..", "2631-carry-reseed"))
for _p in (ROOT, HERE, RES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.evalkit.reconcile import Ledger                          # noqa: E402
import reseed_arms as ra                                            # noqa: E402
import seed3_stats as s3                                            # noqa: E402


def main() -> int:
    R = json.load(open(os.path.join(HERE, "results.json")))
    raw = json.load(open(os.path.join(HERE, "cluster_hits.json")))
    eq = json.load(open(os.path.join(HERE, "equality_report.json")))
    cfg = R["config"]
    ledger = Ledger("2633-carry-seed2",
                    sources=("results.json", "cluster_hits.json", "config.json",
                             "equality_report.json", "train/*/train_log.jsonl"))

    # ---- §9 复用证据 ----
    ledger.eq("采复用前：实现层逐文件 sha256 与 2631 运行时记录相同",
              eq["implementation_identical"], True)
    ledger.eq("采复用前：配置层共享字段无差异", eq["differing_fields"] == [], True)
    ledger.eq("采复用前：训练配方无（除 seed/out-root 之外的）差异",
              eq["train_recipe_differing_flags"] == [], True)
    ledger.eq("复用结论 verdict == reuse_ok", eq["verdict"] == "reuse_ok", True)

    # ---- 逐臂准确率 = Σ命中 / Σ提问（从 cluster_hits.json 重算） ----
    for leg in ("main", "corrupt", "aux_ood", "sens_K4"):
        for name, cells in raw[leg].items():
            got = R[f"{'main_leg' if leg == 'main' else leg + '_leg'}"][name]["accuracy"]
            ledger.derived(f"{leg}/{name}: accuracy == Σhits/Σn",
                           got,
                           lambda cells=cells: round(
                               sum(cells["cluster_hits"]) / sum(cells["cluster_n"]), 6),
                           sources=(f"cluster_hits.json::{leg}::{name}",), abs_tol=5e-7)

    # ---- 逐种子效应 ----
    for key, (base_rule, tgt_rule) in (("delta_vs_none", ("none", "delta")),
                                       ("delta_vs_plain", ("plain", "delta")),
                                       ("plain_vs_none", ("none", "plain"))):
        for row in R["effects"][key]["per_seed"]:
            sd = row["seed"]
            b = raw["main"][s3.arm_name(base_rule, sd)]
            t = raw["main"][s3.arm_name(tgt_rule, sd)]
            ledger.derived(f"{key} seed{sd}: Δpp == 100·(acc_t − acc_b) 由原始计数重算",
                           row["effect_pp"],
                           lambda b=b, t=t: round(100 * (sum(t["cluster_hits"]) / sum(t["cluster_n"])
                                                         - sum(b["cluster_hits"]) / sum(b["cluster_n"])), 3),
                           sources=("cluster_hits.json::main",), abs_tol=1e-3)
        ledger.derived(f"{key}: 极差 == max−min（逐种子效应）",
                       R["effects"][key]["range_pp"],
                       lambda key=key: round(max(x["effect_pp"] for x in R["effects"][key]["per_seed"])
                                             - min(x["effect_pp"] for x in R["effects"][key]["per_seed"]), 3),
                       sources=("results.json::effects",), abs_tol=1e-3)

    # ---- 判据布尔值与数字自洽（防手工改判） ----
    for key, thr in (("delta_vs_none", ra.DELTA_VS_NONE_MIN_PP),
                     ("delta_vs_plain", ra.DELTA_VS_PLAIN_MIN_PP)):
        ledger.eq(f"判据（{key}）== 逐种子 (Δ≥{thr}pp ∧ 配对 CI 下界>0)",
                  R["effects"][key]["stability"]["all_seeds_pass"] == all(
                      x["effect_pp"] >= thr and x["paired_boot_ci_pp"][0] > 0
                      for x in R["effects"][key]["per_seed"]), True,
                  derived_from=("results.json::effects",))
    ledger.eq("判据① 前提 == 三个 none 臂都不显著高于 chance",
              R["criteria"]["c1_premise_none_near_chance"] == (not any(
                  v["significantly_above_chance"] for v in R["premise_check"].values())),
              True, derived_from=("results.json::premise_check",))
    ledger.derived("判据④ 毁源对照：阈值 == 4×chance",
                   R["corrupt_control"]["bound"],
                   lambda: 4 * R["premise_check"]["0"]["chance"],
                   sources=("results.json::premise_check",), abs_tol=1e-9)

    # ---- 跨种子散布与判据⑥ ----
    for rule in ra.RULES:
        vals = [100 * R["main_leg"][s3.arm_name(rule, sd)]["accuracy"] for sd in s3.SEEDS3]
        ledger.derived(f"{rule}: 跨种子极差 == max−min",
                       R["arm_accuracy_spread_across_seeds"][rule]["range_pp"],
                       lambda vals=vals: round(max(vals) - min(vals), 3),
                       sources=("results.json::main_leg",), abs_tol=1e-3)
    ledger.derived("判据⑥：样本量估计 == ⌈2(z+z)²σ²/Δ²⌉（用 plain 的样本 sd）",
                   R["sample_size_analysis"]["seeds_needed_using_plain_sd"],
                   lambda: s3.seeds_needed_for_mean_effect(
                       R["arm_accuracy_spread_across_seeds"]["plain"]["sd_pp"],
                       ra.DELTA_VS_PLAIN_MIN_PP),
                   sources=("results.json::arm_accuracy_spread_across_seeds",),
                   abs_tol=1e-9)
    ledger.derived("判据⑥：塌回地板比例 == #{acc<4×chance}/3",
                   R["collapse_to_floor"]["plain"]["rate"],
                   lambda: round(sum(1 for sd in s3.SEEDS3
                                     if 100 * R["main_leg"][s3.arm_name("plain", sd)]["accuracy"]
                                     < 4 * 100 * R["premise_check"]["0"]["chance"]) / 3, 6),
                   sources=("results.json::main_leg",), abs_tol=5e-7)

    # ---- 训练侧：三臂新 checkpoint 的末步与活性 ----
    for name in cfg["new_in_2633"]:
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        lines = [json.loads(l) for l in open(os.path.join(ROOT, spec["train_log"]))
                 if l.strip()]
        ledger.derived(f"{name}: 训练日志末步 == 12000", R["train_log_tail"][name]["last_step"],
                       lambda lines=lines: lines[-1]["step"],
                       sources=(spec["train_log"],), abs_tol=1e-9)
    ledger.eq("复用臂 source 标注为 2631",
              [R["train_meta"][n]["origin"] for n in cfg["reused_from_2631"]]
              == ["2631（复用）"] * 6, True)

    return ledger.finish(os.path.join(HERE, "reconcile_report.json"))


if __name__ == "__main__":
    raise SystemExit(main())
