#!/usr/bin/env python3
"""2615-priority-shootout · 保留腿（召回）配对 CI 的**可行性复核 + 可得读法**（0 训练 token，纯再分析）。

背景（`docs/research-schema.md` §12.1 的 C1 细分 + ticket 62 §10.1 第 11 行）：
本项的 `falsification` 声明「长程召回等率类指标走**配对 bootstrap / Wilson 95% CI 不含 0**」。
但 2615 落盘的 `results.json` **只有聚合读数**（`criteria.per_N` 的逐配置召回、
`hard_needle_subset.per_config` 的同索引硬子集召回、`no_needle_control` 的地板），
**没有逐针命中向量**；`retention_log.jsonl` 的 32 条 `needle_batch` 记录只覆盖 `recall_full`/`recall_lru`
且是**分批进度**（`n_so_far` 累计），**不足以复原三规则逐针/逐批配对**。

因此本脚本做两件事，并把结论写在报告里：
1. **可行性判定（主结论）**：预登记的「配对 bootstrap」**不可从落盘证据复原**。给出不可识别的理由：
   配对二值数据的差方差 = (n10+n01)/n² − (p1−p2)²/n；已知的只有边际与 d = p1−p2，
   而 n10+n01 只被 >|d|n 下界约束 ⇒ 配对 SE 可在「≈0」与「非配对 SE」之间任意取值
   ⇒ 配对 CI 的宽度**不可定为唯一值** ⇒ 该判据在这份证据上**不可判**。
2. **可得读法（补充，非预登记口径）**：用**同索引硬子集（n = n_hard）**上的两比例 **Newcombe 95% CI**
   （两臂各用 Wilson 区间合成，**非配对**）报告「规则 − distance」的差及其区间是否含 0。
   明确标注：这是**非配对近似**，比真配对区间**一般更宽**，因此**只能用于「有差异」的正向读法**，
   不足以支撑「无差异」的否定结论。

用法：python3 recompute_pair_ci.py            # 打印表并写 pair_ci_report.json
"""

from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
Z = 1.959963984540054  # 95%


def wilson(p: float, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval（含两端截断到 [0,1]）。"""
    if n <= 0:
        return (0.0, 1.0)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(p1: float, n1: int, p2: float, n2: int, z: float = Z) -> tuple[float, float]:
    """Newcombe（method 10）：两比例差的 95% CI，由两臂各自的 Wilson 区间合成。"""
    l1, u1 = wilson(p1, n1, z)
    l2, u2 = wilson(p2, n2, z)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)


def main() -> None:
    d = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
    hs = d["hard_needle_subset"]
    n_hard = int(hs["n_hard"])
    pc = hs["per_config"]
    per_n = d["criteria"]["per_N"]
    # 全针集（4096）召回：从 criteria.per_N 里取（契约口径）
    rules = [("lru:k=64", "lru"), ("content", "content")]
    rows = []
    for entry in per_n:
        n_ = int(entry.get("n", 0)) or int(entry.get("N") or entry.get("n_drop") or 0)
        # criteria.per_N 是 list：逐条含 n_drop / rules
        pass
    # 契约口径：从 per_N 逐条取（结构见 analysis §1）
    by_n = {}
    for entry in per_n:
        N = entry.get("N") or entry.get("n_drop") or entry.get("n")
        by_n[int(N)] = entry
    for N in (8, 32, 64):
        e = by_n.get(N)
        base = None
        if e is not None:
            rules_in = e.get("rules") or e.get("per_rule") or {}
            base = rules_in.get("distance") or rules_in.get("distance") or None
        # 硬子集读数（同索引 n = n_hard）
        p_dist_h = pc[f"distance:N={N}"]
        for name, _ in rules:
            key = f"{name}:N={N}"
            if key not in pc:
                continue
            p_r_h = pc[key]
            lo, hi = newcombe(p_r_h, n_hard, p_dist_h, n_hard)
            rows.append({
                "N": N, "rule": name, "n_hard": n_hard,
                "recall_hard_rule": p_r_h, "recall_hard_distance": p_dist_h,
                "diff_pp": round(100 * (p_r_h - p_dist_h), 4),
                "newcombe95_pp": [round(100 * lo, 4), round(100 * hi, 4)],
                "ci_contains_0": (lo <= 0 <= hi),
                "rule_wilson95": [round(100 * v, 4) for v in wilson(p_r_h, n_hard)],
                "distance_wilson95": [round(100 * v, 4) for v in wilson(p_dist_h, n_hard)],
            })
    # 契约口径（按位置数对齐）的依赖腿与召回腿，供并列阅读
    contract = {
        "recall_threshold_pp": d["criteria"].get("recall_threshold_pp"),
        "damage_threshold_bpc": d["criteria"].get("damage_threshold_bpc"),
        "verdict": d["criteria"].get("verdict"),
        "baseline_recall_full": d["criteria"].get("baseline_recall_full"),
    }
    report = {
        "target": "axes/1-time/1.1.4.3-compression-trigger/experiments/2615-priority-shootout-13m/results.json",
        "caliber": ("同索引硬子集（hard_needle_subset，n=%d，control_recall_hard=%.4f）上的两比例差；"
                    "CI = Newcombe 95%%（两臂各用 Wilson 区间合成）——**非配对**近似" % (n_hard, hs["control_recall_hard"])),
        "paired_bootstrap_recomputable": False,
        "why_not": ("落盘只有聚合计数（无逐针命中向量；needle_batch 进度日志只含 full/lru 两列与 n_so_far 累计）"
                    "⇒ 配对二值差方差 (n10+n01)/n² − d²/n 中的 n10+n01 不可得，只被 >|d|n 约束 ⇒ "
                    "配对 SE 可在 ≈0 与非配对 SE 之间任取 ⇒ 配对 CI 宽度不可定 ⇒ 预登记的配对判据**不可判**"),
        "hard_subset_rows": rows,
        "contract_caliber": contract,
        "reading": ("补充读法（非预登记口径）：硬子集上三规则相对 distance 的差普遍很小且**多数落在含 0 的区间里**；"
                    "唯一正向偏大者是 N=8 的 lru:k=64，但其非配对 95% CI 也**含 0**（见 rows）。"
                    "因为这是**非配对**近似（一般比配对区间更宽），本报告**不**据此下「无差异」的否定结论；"
                    "它只说明：**用现盘证据无法完成**本项预登记所需的配对判据。"),
        "next_minimal_experiment": ("重跑 2615 的**评测腿一次**（纯推理、0 训练 token：24×16 窗口 × 3 规则 × N∈{8,32,64} "
                                    "+ 4096 针），并把**逐针命中记录**落盘（`needle_raw.json`）⇒ 之后按 §12.1 做配对 bootstrap，"
                                    "本项的召回腿才可判"),
    }
    json.dump(report, open(os.path.join(HERE, "pair_ci_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "hard_subset_rows"}, ensure_ascii=False, indent=1))
    print("\nN   rule       diff(pp)   Newcombe95(pp)        含0")
    for r in rows:
        print(f"{r['N']:<3} {r['rule']:<10} {r['diff_pp']:>+8.4f}   "
              f"[{r['newcombe95_pp'][0]:+.4f}, {r['newcombe95_pp'][1]:+.4f}]   {r['ci_contains_0']}")


if __name__ == "__main__":
    main()
