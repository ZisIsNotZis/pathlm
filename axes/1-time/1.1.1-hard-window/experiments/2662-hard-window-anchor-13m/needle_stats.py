#!/usr/bin/env python3
"""2634-carry-realtext-13m · 离散答案探针（needle v3）的统计层（契约判据④）。

两件事（都来自复核 P1-2/P1-4 指出的缺陷）：

1. **同索引硬子集**：旧实现让**每个臂用自己那一臂的 control 失败集**，于是四臂的索引集
   互不相同（「delta 臂高 3pp」这种比较因此不可比）。这里把索引集固定成**同一组**：
   - `intersection` = 四个臂的 control 都答不出的针（**最保守**：任何「无记忆」运行都答不出
     ⇒ 命中只可能来自记忆/搬运）；
   - `union` = 至少一个臂的 control 答不出的针（敏感性读数）。
   四臂在同一索引集上算 implanted 召回 ⇒ 两比例差可直接比较。
2. **方向正确的「高于地板」判定**：`tools.evalkit.carry_channel.two_proportion` 的定义是
   `diff = p_b − p_a`；判据要的是「植入 − 地板 > 0」，因此必须**把 control 传在 a、planted
   传在 b**。旧实现传反（`planted` 在前）⇒ 判的其实是「地板显著高于植入」，
   该判据**永远不可能通过**（不可达判据）。本模块的 `beyond_above_floor` 把参数顺序固化，
   并有单测用「应通过」的构造输入验证它**真能通过**。
"""

from __future__ import annotations

import random

from tools.evalkit.carry_channel import two_proportion, wilson_ci


def _percentile(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def paired_bootstrap_diff_pp(planted: list[int], control: list[int],
                             n_boot: int = 2000, seed: int = 2662,
                             alpha: float = 0.05) -> dict:
    """**按针配对**的 bootstrap CI（百分点）：逐针差 `hit_planted − hit_control` ∈ {−1,0,1}。

    判据②要的是「植入高于地板」，配对成立的前提 = 同一根针的植入/控制是同一位置的两份
    输入（`run.py` 已断言 `meta` 逐位相同）。重采样**针索引**（有放回）⇒ CI 只含针抽样
    噪声。与 `beyond_above_floor` 的 Newcombe 两比例差并列报（两种口径同号才可信）。
    """
    if len(planted) != len(control):
        raise ValueError("planted 与 control 的针数不一致（不配对）")
    n = len(planted)
    if n < 2:
        raise ValueError("配对 bootstrap 至少需要 2 根针")
    d = [p - c for p, c in zip(planted, control)]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        s = sum(d[rng.randrange(n)] for _ in range(n))
        draws.append(100.0 * s / n)
    draws.sort()
    return {"n": n,
            "mean_diff_pp": round(100.0 * sum(d) / n, 3),
            "diff_ci_95_pp": [round(_percentile(draws, alpha / 2), 3),
                              round(_percentile(draws, 1 - alpha / 2), 3)],
            "passes_lower_gt_0": bool(_percentile(draws, alpha / 2) > 0),
            "n_boot": n_boot,
            "note": "按针配对 bootstrap，方向 = 植入 − 地板（百分点）"}


def control_miss_indices(records: list[dict]) -> set[int]:
    """一臂 control（`plant=False`）答不出的针索引集合。"""
    return {i for i, r in enumerate(records) if r["hit"] == 0}


def hard_index_sets(control_by_arm: dict[str, list[dict]]) -> dict:
    """由各臂的 control 失败集派生**同一组**索引：交集（保守）/ 并集（敏感性）。

    前提：各臂的 needle 记录**逐索引对齐**（同批同针位，`run.py` 已断言 meta 逐位相同）。
    """
    if not control_by_arm:
        raise ValueError("hard_index_sets 收到空字典")
    lens = {len(v) for v in control_by_arm.values()}
    if len(lens) != 1:
        raise ValueError(f"各臂 needle 记录长度不一致：{lens}（索引集不可比）")
    miss = {name: control_miss_indices(recs) for name, recs in control_by_arm.items()}
    inter = set.intersection(*miss.values())
    union = set.union(*miss.values())
    return {
        "per_arm_control_miss": {k: len(v) for k, v in miss.items()},
        "n_indexed": next(iter(lens)),
        "intersection": sorted(inter),
        "union": sorted(union),
        "n_intersection": len(inter),
        "n_union": len(union),
        "definition": ("intersection = 四臂 control 全部答不出（任何无记忆运行都答不出）；"
                       "union = 至少一臂 control 答不出（敏感性）"),
    }


def recall_on_indices(records: list[dict], idx: list[int]) -> dict:
    """在**给定索引集**上算植入召回（hits/n + Wilson CI）。"""
    if not idx:
        return {"n": 0, "hits": 0, "recall": None, "wilson_ci": None}
    hits = sum(records[i]["hit"] for i in idx)
    n = len(idx)
    return {"n": n, "hits": int(hits), "recall": round(hits / n, 6),
            "wilson_ci": wilson_ci(int(hits), n)}


def beyond_above_floor(planted_hits: int, planted_n: int,
                       control_hits: int, control_n: int) -> dict:
    """判据④的操作化：**植入 − 地板** 的两比例差 95% CI 下界 > 0 ⇒ 该臂该桶「高于地板」。

    方向固化：`two_proportion(control, planted)`（该函数 diff = p_b − p_a）⇒
    返回的 `diff_ci_95` 是「植入 − 地板」的 CI。`passes` 即判据④的逐臂判定。
    """
    if planted_n <= 0 or control_n <= 0:
        raise ValueError("两比例差要求 n > 0")
    tp = two_proportion(control_hits, control_n, planted_hits, planted_n)
    lo = float(tp["diff_ci_95"][0])
    return {"planted_recall": round(planted_hits / planted_n, 6),
            "control_recall": round(control_hits / control_n, 6),
            "delta_pp": round(100 * (planted_hits / planted_n - control_hits / control_n), 3),
            "diff_ci_95": tp["diff_ci_95"],
            "diff_ci_95_pp": [round(100 * v, 3) for v in tp["diff_ci_95"]],
            "z": tp["z"], "p_value": tp["p_value"], "ci_overlap": tp["ci_overlap"],
            "passes": bool(lo > 0),
            "direction_note": ("正 = 植入高于地板（判据④要的方向）；"
                               "参数顺序 (control, planted) 已在函数内固化")}


def judge(planted_by_arm: dict[str, dict], control_by_arm: dict[str, dict],
          intervention_arms: tuple[str, ...]) -> dict:
    """逐臂判定 + 预登记聚合。

    预登记口径（README §4 判据④）：**每个干预臂**（delta 臂）的 `beyond` 桶召回都要
    高于其配对地板；只要有一个干预臂不过 ⇒ 该腿记「无区分力（如实报）」，不判机制。
    """
    per_arm = {name: beyond_above_floor(planted_by_arm[name]["hits"], planted_by_arm[name]["n"],
                                        control_by_arm[name]["hits"], control_by_arm[name]["n"])
               for name in planted_by_arm}
    return {
        "per_arm": per_arm,
        "intervention_arms": list(intervention_arms),
        "intervention_arms_all_pass": bool(
            all(per_arm[a]["passes"] for a in intervention_arms)),
        "any_arm_passes": bool(any(v["passes"] for v in per_arm.values())),
        "all_arms_pass": bool(all(v["passes"] for v in per_arm.values())),
        "rule": ("预登记口径 = 逐个**干预臂**（delta）都要通过；"
                 "`any_arm_passes`/`all_arms_pass` 只作旁证，不作判据"),
    }
