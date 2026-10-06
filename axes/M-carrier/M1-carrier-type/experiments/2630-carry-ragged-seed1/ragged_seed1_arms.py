"""2630-carry-ragged-seed1 · seed=1 两臂命名、跨 seed 统计与稳健性判据（纯逻辑）。

契约：同目录 `README.md`（§4 判据线逐字抄录 W2）。

本线是 2629（`projfill` / `projfill_carry`，seed=0）的**同配置 seed=1 复现**：
配方逐字段相同（随机退出 + proj-fill + 可选 delta 搬运通道），**唯一变量仍是
`carry_rule` 的开关**；本线新训 seed=1 的两臂，与 2629 的 seed=0 合并为 **n=2**。

**不复制单因素语义**：两臂的配方（`model_config_kwargs` / `train_flags` /
`expected_block_class` / `native_visibility`）直接复用 2629 的语义层
`ragged_projfill_arms.py`（同仓 sibling 目录；`MODEL.md` 登记其 sha256）。
本模块只加三件事：

1. **seed=1 的臂名映射**（`projfill_s1` / `projfill_carry_s1`），把 seed 带进名字，
   避免与 2629 的 seed=0 臂混淆；
2. **跨 seed 统计**（极差 / 均值 / 样本 sd / 符号一致性）；
3. **两条强断言的判据**（W2 判据②③），用词严格受证据类型约束：
   - `ood_effect_robust`：越界腿（单查询，距离 241–255）的 carry 效应**跨 seed 保持**
     ⇒ 证据类型 = **统计检验**（每 seed 的 `two_proportion` 95% 差值 CI 下界 > 0）
     ＋ **同号**（每 seed 效应 > 0）。**只此一腿**，不外推到召回。
   - `indist_redundancy_stable`：分布内召回（随机退出，多查询）的 carry 效应**跨 seed
     都不超过方差带**（`Δ < 0.014` 不可排序，`docs/methodology.md`）⇒ 「冗余」结论稳定。

**禁外推**（W2 判据④、SCHEMA §17.11）：本模块只描述本次两点（seed=0/1）的统计，
不推断其它 seed、其它尺度、其它退出模式。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
#: 2629（seed=0）目录 = 单因素语义层的唯一来源 + 被合并的 seed=0 结果。
SEED0_DIR = os.path.join(_ROOT, "axes", "M-carrier", "M1-carrier-type",
                         "experiments", "2629-carry-ragged-projfill")
SEED0_RESULTS = os.path.join(SEED0_DIR, "results.json")
if SEED0_DIR not in sys.path:
    sys.path.insert(0, SEED0_DIR)

from ragged_projfill_arms import (BASELINE_ARM, TARGET_ARM,  # noqa: E402
                                  ARM_NAMES as SEED0_ARM_NAMES)

#: 本实验的训练 seed（唯一变量 vs 2629 的 seed=0 复现）。
SEED = 1

#: 本实验两臂（seed=1）的臂名 —— 命名带上 seed，与 2629 的 seed=0 臂区分。
ARM_NAMES_S1 = ("projfill_s1", "projfill_carry_s1")
#: seed=0（2629）臂名 → 本实验臂名。
BASE_TO_S1 = {BASELINE_ARM: "projfill_s1", TARGET_ARM: "projfill_carry_s1"}
#: 反向映射。
S1_TO_BASE = {v: k for k, v in BASE_TO_S1.items()}

#: 本仓方差带（`docs/methodology.md`）；不足 1.4pp 的差不可排序。
VARIANCE_BAND_PP = 1.4

#: W2 判据②/③ 的强断言证据类型（写进报告，用词不得超出它）。
OOD_ROBUST_EVIDENCE = ("statistical_test: per-seed two_proportion 95% diff CI lower bound > 0 "
                       "+ same-sign-positive across seeds")
INDIST_STABLE_EVIDENCE = ("magnitude: |carry effect| <= variance band 1.4pp in every seed "
                          "(SCHEMA §12 / docs/methodology.md: Δ<0.014 不可排序)")


def seed0_to_seed1(name: str) -> str:
    """2629（seed=0）臂名 → 本实验（seed=1）臂名。"""
    return BASE_TO_S1[name]


def seed1_to_seed0(name: str) -> str:
    """本实验（seed=1）臂名 → 2629（seed=0）臂名。"""
    return S1_TO_BASE[name]


def train_seed() -> int:
    """本线两条臂共用的训练 seed（模型初始化与数据流都由它决定）。"""
    return SEED


def pp_range(values) -> float:
    """极差（pp）：`max−min`；**输入已经是 pp**（缺省口径 = `effect_pp`）。不足两点返回 0。"""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    return round(max(vals) - min(vals), 3)


def pp_mean(values) -> float:
    """算术均值（pp；输入已经是 pp）。"""
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 3)


def pp_sd(values) -> float:
    """**样本**标准差（n−1；n<2 时 0），单位 pp；输入已经是 pp。"""
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return round(var ** 0.5, 3)


def sign_consistent(values) -> bool:
    """全部同号且非零（严格 >0 或严格 <0）——用于「效应方向跨 seed 保持」。"""
    vals = [float(v) for v in values]
    if not vals:
        return False
    return all(v > 0 for v in vals) or all(v < 0 for v in vals)


def ood_effect_robust(per_seed: dict) -> dict:
    """W2 判据②：越界腿 carry 效应**跨 seed 保持** ⇒ 「机制确实在搬」记为稳健。

    `per_seed = {seed: {"effect_pp": float, "diff_ci_lo": float, "diff_ci_hi": float}}`
    （`effect_pp` = 目标臂 − 基准臂，单位 pp；`diff_ci_*` = `two_proportion` 95% 差值 CI）。

    判据（预登记，**只此一腿**）：
    1. 至少两个 seed；
    2. 每个 seed 的效应 **同号且为正**（通道确实在搬，不是反向）；
    3. 每个 seed 的**差值 CI 下界 > 0**（统计检验。

    证据类型 = `OOD_ROBUST_EVIDENCE`；**不外推到召回**（召回是另一条腿、另一个判据）。
    """
    seeds = sorted(per_seed)
    effects = [per_seed[s]["effect_pp"] for s in seeds]
    ci_los = [per_seed[s]["diff_ci_lo"] for s in seeds]
    positive = all(e > 0 for e in effects)
    significant = all(lo > 0 for lo in ci_los)
    robust = len(seeds) >= 2 and positive and significant
    return {
        "n_seeds": len(seeds), "seeds": seeds,
        "effects_pp": [round(float(e), 4) for e in effects],
        "same_sign_positive": bool(positive),
        "ci_lower_positive_each": bool(significant),
        "range_pp": pp_range(effects),
        "mean_pp": pp_mean(effects),
        "robust": bool(robust),
        "evidence_type": OOD_ROBUST_EVIDENCE,
        "no_extrapolation": "仅越界腿；不外推到分布内召回（§17.18 / W2 判据②）",
    }


def indist_redundancy_stable(per_seed: dict,
                             band_pp: float = VARIANCE_BAND_PP) -> dict:
    """W2 判据③：分布内召回 carry 效应**跨 seed 都不超过方差带** ⇒ 「冗余」结论稳定。

    与②分开：这里不比「是否有增益」，而问「seed=0 的冗余结论（|Δ| ≤ 带）是否跨 seed 稳定」。
    证据类型 = `INDIST_STABLE_EVIDENCE`（量级；不做等价比或非劣检验）。
    """
    seeds = sorted(per_seed)
    effects = [per_seed[s]["effect_pp"] for s in seeds]
    within = all(abs(float(e)) <= band_pp for e in effects)
    nonsig = all(per_seed[s]["diff_ci_lo"] <= 0 <= per_seed[s]["diff_ci_hi"]
                 for s in seeds)
    stable = len(seeds) >= 2 and within
    return {
        "n_seeds": len(seeds), "seeds": seeds,
        "effects_pp": [round(float(e), 4) for e in effects],
        "band_pp": band_pp,
        "all_within_band": bool(within),
        "all_ci_contain_zero": bool(nonsig),
        "range_pp": pp_range(effects),
        "mean_pp": pp_mean(effects),
        "stable": bool(stable),
        "evidence_type": INDIST_STABLE_EVIDENCE,
        "no_extrapolation": "仅两点（seed=0/1）/ 13M / 6k 快照；§17.11 禁外推",
    }
