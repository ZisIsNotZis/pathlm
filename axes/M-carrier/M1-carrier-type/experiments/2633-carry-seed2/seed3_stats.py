"""2633-carry-seed2 · 语义层：三 seed 合并判据 + 样本量估计（判据⑥）。

**复用 2631 的统计层**（`../2631-carry-reseed/reseed_arms.py`：聚类 bootstrap、配对
bootstrap、极差/均值/样本 sd、`effect_stable`、`corrupt_control_reading`）——同一函数、
同一口径，不重写。本模块只加「第三种子」这一层：

1. `SEEDS3` = (0, 1, 2)，与 2631 的 `effect_stable` 直接配合（它按 `seed` 字段逐种子判）；
2. `seeds_needed_for_mean_effect`：**种子层面**的样本量估计（判据⑥，任务下发文本要求
   「若不可判，给出需要多少样本才能判」）：把「跨种子效应」看成均值估计，
   用观测到的样本 sd 估「要让均值效应以 80% 功效显著于 min_pp，需要多少种子」；
3. `collapse_wilson`：plain 臂「塌回地板」的比例估计（Wilson 95% CI）——
   与「均值检验」并列报，因为它才是 plain 双稳态的正确统计量；
4. `mean_ci_t`：n 小样本下用 t 分布给种子层均值的 CI（只用 df=1/2 的临界值，够本用）。

**只读导入 2631 的模块**（不写任何共享文件）。
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "..", "2631-carry-reseed")
if _RES not in sys.path:
    sys.path.insert(0, _RES)

import reseed_arms as ra                                            # noqa: E402
from tools.evalkit.carry_channel import wilson_ci                    # noqa: E402

#: 本实验新增的第三种子；与 2631 的 (0, 1) 合并成三 seed 判据
SEEDS3: tuple[int, ...] = (0, 1, 2)

#: t 分布双侧 97.5% 分位（df=1,2）——三 seed 判决只用得到这两个
_T975 = {1: 12.706, 2: 4.303}


def arm_name(rule: str, seed: int) -> str:
    """规则 + 种子 → 臂名（三 seed 版）。

    seed ∈ {0,1} 时**委派** 2631 的 `reseed_arms.arm_name`（同一命名口径，不另立一套）；
    seed=2 是本实验新增档（2631 的 `SEEDS` 只到 1，它的函数会拒绝）⇒ 在本地按同一
    命名规则构造 `f"{rule}_s{seed}"`，并用单测钉住两者在重叠档上逐字相同。
    """
    if rule not in ra.RULES:
        raise ValueError(f"未知写入规则 {rule!r}，可选 {ra.RULES}")
    if seed not in SEEDS3:
        raise ValueError(f"未知种子 {seed!r}，可选 {SEEDS3}")
    if seed in ra.SEEDS:
        return ra.arm_name(rule, seed)
    return f"{rule}_s{seed}"


def seeds_needed_for_mean_effect(sd_pp: float, min_pp: float, *,
                                 alpha: float = 0.05, power: float = 0.8) -> int:
    """**种子层面**均值检验的样本量：n = ⌈2(z_{1−α/2}+z_{1−β})² σ² / Δ²⌉（双侧 t 近似）。

    用途（判据⑥）：若三 seed 仍判不出「delta − plain ≥ min_pp」，回答「还要多少种子」。
    **限定**：这是正态近似 + 用 2 个自由度的样本 sd 当 σ 的估计（自身极不确定）；
    若 `sd_pp` 本身来自 n≤3 的样本，返回值只作**规划量级**，不得当日志外结论（README §4a⑥）。
    """
    if sd_pp < 0 or min_pp <= 0:
        raise ValueError(f"sd_pp 必须 ≥0 且 min_pp >0，收到 {sd_pp} / {min_pp}")
    z_a = 1.959963984540054 if alpha == 0.05 else _norm_ppf(1 - alpha / 2)
    z_b = 0.8416212335729143 if power == 0.8 else _norm_ppf(power)
    n = 2.0 * (z_a + z_b) ** 2 * sd_pp ** 2 / min_pp ** 2
    return int(math.ceil(n))


def _norm_ppf(p: float) -> float:
    """标准正态分位（Acklam 有理逼近；|误差| < 1.2e-9，够本用）。"""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 必须在 (0,1)，收到 {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def mean_ci_t(values: list[float]) -> dict:
    """种子层均值的 t 双侧 95% CI（df = n−1；本模块只带 df∈{1,2} 的临界值）。

    `n ≥ 3` 时返回 `None`（不做外推、不近似）——本实验正好 n=3，`cautious` 标注为
    「df=2 的 t 临界值 4.303；n 小 ⇒ CI 很宽，只作描述」。
    """
    n = len(values)
    if n < 2:
        raise ValueError("mean_ci_t 至少需要 2 个观测")
    t = _T975.get(n - 1)
    m = sum(values) / n
    sd = (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5
    half = None if t is None else t * sd / math.sqrt(n)
    return {"n": n, "mean_pp": round(m, 3), "sd_pp": round(sd, 3),
            "t": t, "ci95_pp": None if half is None else [round(m - half, 3),
                                                          round(m + half, 3)],
            "note": ("df=2 ⇒ t=4.303；n 小、CI 宽，只作描述量，不作判据线"
                     if n == 3 else "本模块只带 df∈{1,2} 的临界值")}


def collapse_wilson(hits: list[bool]) -> dict:
    """「塌回地板」比例 k/n 的 Wilson 95% CI（plain 双稳态的正确统计量）。"""
    n = len(hits)
    if n == 0:
        raise ValueError("collapse_wilson 收到空列表")
    k = sum(1 for h in hits if h)
    return {"n": n, "k": int(k), "rate": round(k / n, 6),
            "wilson_ci": wilson_ci(int(k), int(n)),
            "note": "k/n 很小或 n 很小时 Wilson CI 会很宽（如实报，不用点估计下结论）"}


def judge_seed3(effects: dict) -> dict:
    """判据①②③的三 seed 版判定（调用 2631 的 `effect_stable`，不重写判据语义）。"""
    none_rules = ("delta_vs_none", "delta_vs_plain")
    out = {}
    for key in none_rules:
        row = effects[key]
        stable = ra.effect_stable(row["per_seed"], row["stability"]["min_pp_threshold"])
        out[key] = {"stable": stable["stable"],
                    "all_seeds_pass": stable["all_seeds_pass"],
                    "per_seed": stable["per_seed"],
                    "n_seeds": stable["n_seeds"],
                    "range_pp": stable["range_pp"],
                    "sign_consistent": stable["sign_consistent"]}
    return out
