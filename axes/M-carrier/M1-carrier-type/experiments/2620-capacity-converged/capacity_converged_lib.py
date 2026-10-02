"""2620 · 容量轴「收敛后」判据的**唯一实现处**（预登记判据 ①②③）。

契约：同目录 `README.md` §4（判据线在读数前写下，不得事后调整）。
本模块只做**纯函数计算**，不碰 GPU、不改任何共享文件（SCHEMA §22）。
可失败单测在同目录 `test_capacity_converged.py`（不放进 `tests/`，以保持
`pytest tests/` 的 294 基线不变）；`mutate.py` 对本模块打运行期猴子补丁做变异验证。

术语就地解释（首次出现）：
- `synth_answer_ce`：**答案行交叉熵**——只有「被问的 key 对应的答案位置」参与的 CE，
  单位 nats；**它是本任务里唯一决定准确率的量**（总 CE 被已学会的 filler 主导）。
- `末窗`：训练日志的最后 8,000 步（`step ∈ [16000, 23999]`）。
- `per 1k 步斜率`：对末窗的 `(step, 值)` 做普通最小二乘后乘 1000。
- `chance`：瞎猜命中率 = 1/64 = 0.015625（任务词表 64 符号、同行 key 互不相同）。
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- 预登记常数
CHANCE = 1.0 / 64.0
LAST_WINDOW = (16000, 23999)          # 24k 步训练的末窗（README §4①）
CONVERGENCE_SLOPE_MAX = 0.01          # |slope| < 0.01 nats / 1k 步
CONVERGENCE_REL_DROP_MAX = 0.01       # 或末窗相对下降 < 1%
MAIN_LEVEL_K = 8                      # 主腿（= 训练同档）
DELTA_PP_THRESHOLD = 3.0              # 主判据：大状态 ≥ 小状态 + 3pp
CORRUPT_INVALIDATE_MULT = 4.0         # 毁源作废线：CI 下界 > 4×chance


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var


def linear_window(lines: list[dict], key: str, lo: int, hi: int) -> dict:
    """末窗线性 OLS：`值` 对 `step` 的斜率（×1000）+ 相对下降。

    返回 `n_points=0` 是**可检测的失败**（读数不足），由调用方决定是否判门禁。
    """
    rows = [(r["step"], r[key]) for r in lines
            if lo <= r["step"] <= hi and r.get(key) is not None]
    rows.sort()
    xs = [float(s) for s, _ in rows]
    ys = [float(v) for _, v in rows]
    if len(rows) < 2:
        return {"key": key, "window": [lo, hi], "n_points": len(rows),
                "slope_per_1k": None, "first": None, "last": None,
                "rel_drop": None}
    slope = _ols_slope(xs, ys) * 1000.0
    first, last = ys[0], ys[-1]
    return {"key": key, "window": [lo, hi], "n_points": len(rows),
            "slope_per_1k": round(slope, 6),
            "first": round(first, 4), "last": round(last, 4),
            "rel_drop": round((first - last) / abs(first), 6) if first else None}


def loglog_window(lines: list[dict], key: str, lo: int, hi: int) -> dict:
    """U1 同方法斜率：`log(值)` 对 `log(step)` 的 OLS（无量纲）。

    与 U1 的 [200, 12000] 窗口同方法可比（U1 报 −0.065 / −0.062 / −0.056 / −0.051）。
    """
    rows = [(r["step"], r[key]) for r in lines
            if lo <= r["step"] <= hi and r.get(key) is not None and r[key] > 0]
    if len(rows) < 3:
        return {"key": key, "window": [lo, hi], "n_points": len(rows),
                "slope": None}
    xs = [math.log(s) for s, _ in rows]
    ys = [math.log(v) for _, v in rows]
    return {"key": key, "window": [lo, hi], "n_points": len(rows),
            "slope": round(_ols_slope(xs, ys), 6)}


def convergence_gate(ansce_linear: dict) -> dict:
    """判据 ①（门禁，编排方裁决口径）：**答案行 CE** 的末窗线性斜率。

    通过条件（或）：`|slope_per_1k| < 0.01` **或** `rel_drop < 1%`。
    **必须用答案行 CE**；总 CE 在本任务上恒真、无区分度（README §4①），
    本函数不提供「换成总 CE」的开关——那正是变异 M5 要抓红的行为。
    """
    s, rel, n = (ansce_linear.get("slope_per_1k"), ansce_linear.get("rel_drop"),
                 ansce_linear.get("n_points"))
    if s is None or rel is None or not n:
        return {"converged": False, "reason": "末窗读数不足（n_points < 2）",
                "metric": ansce_linear.get("key"), "slope_per_1k": s,
                "rel_drop": rel, "passes_slope": False, "passes_rel_drop": False,
                "thresholds": {"abs_slope_per_1k": CONVERGENCE_SLOPE_MAX,
                               "rel_drop": CONVERGENCE_REL_DROP_MAX}}
    passes_slope = abs(s) < CONVERGENCE_SLOPE_MAX
    passes_rel = rel < CONVERGENCE_REL_DROP_MAX
    return {"converged": bool(passes_slope or passes_rel),
            "metric": ansce_linear["key"],
            "slope_per_1k": s, "rel_drop": rel,
            "passes_slope": passes_slope, "passes_rel_drop": passes_rel,
            "thresholds": {"abs_slope_per_1k": CONVERGENCE_SLOPE_MAX,
                           "rel_drop": CONVERGENCE_REL_DROP_MAX},
            "reason": ("slope 与相对下降至少一条达标" if (passes_slope or passes_rel)
                       else "slope 与相对下降都未达标 ⇒ 未收敛")}


def capacity_verdict(acc_small: float, n_small: int, acc_big: float, n_big: int,
                     ci_small: list, ci_big: list,
                     threshold_pp: float = DELTA_PP_THRESHOLD) -> dict:
    """判据 ②（主判据）：大状态 ≥ 小状态 + 3pp **且** 两条 Wilson CI 不重叠。

    返回的 `delta_pp` 定义 = `100·(acc_big − acc_small)`（正 = 大状态更好，全篇同号）。
    """
    delta_pp = 100.0 * (acc_big - acc_small)
    ci_overlap = not (ci_big[0] > ci_small[1] or ci_small[0] > ci_big[1])
    return {"acc_small": acc_small, "acc_big": acc_big,
            "n_small": n_small, "n_big": n_big,
            "ci_small": list(ci_small), "ci_big": list(ci_big),
            "delta_pp": round(delta_pp, 3),
            "threshold_pp": threshold_pp,
            "meets_delta": bool(delta_pp >= threshold_pp),
            "ci_overlap": bool(ci_overlap),
            "capacity_axis_effective": bool(delta_pp >= threshold_pp
                                            and not ci_overlap)}


def corrupt_invalidates(acc: float, ci: list) -> bool:
    """判据 ③.1：毁源读数的 Wilson CI 下界 > 4×chance ⇒ 该格作废。"""
    return bool(ci[0] > CORRUPT_INVALIDATE_MULT * CHANCE)


def training_divergence(lines_a: list[dict], lines_b: list[dict],
                        keys=("loss", "synth_answer_ce")) -> dict:
    """判据 ③.2：两跑（同 seed 重跑 vs 原跑）在**同步骤**上的差。

    只比较两边都有的 step；报每步的差与全轨迹最大绝对差。
    """
    by_step = {r["step"]: r for r in lines_b if r.get("step") is not None}
    per_key = {}
    for key in keys:
        rows = []
        for r in lines_a:
            o = by_step.get(r["step"])
            if o is None or r.get(key) is None or o.get(key) is None:
                continue
            rows.append({"step": r["step"], "a": r[key], "b": o[key],
                         "abs_diff": round(abs(r[key] - o[key]), 6)})
        rows.sort(key=lambda d: d["step"])
        per_key[key] = {
            "n_steps_compared": len(rows),
            "max_abs_diff": max((d["abs_diff"] for d in rows), default=None),
            "final_abs_diff": rows[-1]["abs_diff"] if rows else None,
            "steps": rows,
        }
    return {"note": "同 seed 重跑的训练侧散度（判据 ③.2）；只比两边都有的 step",
            "per_key": per_key}


def reconciliation_checks(results: dict) -> dict:
    """§12 对账的纯函数部分：判据旗标必须与它引用的数字自洽。"""
    checks = {}
    conv = results.get("convergence", {})
    for arm, blk in conv.get("by_arm", {}).items():
        g = blk.get("gate", {})
        s, rel = g.get("slope_per_1k"), g.get("rel_drop")
        if s is None or rel is None:
            checks[f"convergence_{arm}"] = False
            continue
        expect = bool(abs(s) < CONVERGENCE_SLOPE_MAX
                      or rel < CONVERGENCE_REL_DROP_MAX)
        checks[f"convergence_{arm}"] = (expect == bool(g.get("converged")))
    for cell, blk in results.get("criteria", {}).get("c2_main_by_K", {}).items():
        v = blk.get("verdict", {})
        expect = bool(v.get("delta_pp", 0) >= DELTA_PP_THRESHOLD
                      and not v.get("ci_overlap", True))
        checks[f"main_{cell}"] = (expect == bool(v.get("capacity_axis_effective")))
    for cell, blk in results.get("criteria", {}).get("c3_1_corrupt", {}).items():
        ci, flag = blk.get("accuracy_ci"), blk.get("invalidated")
        if ci is None or flag is None:
            checks[f"corrupt_{cell}"] = False
            continue
        checks[f"corrupt_{cell}"] = (bool(ci[0] > CORRUPT_INVALIDATE_MULT * CHANCE)
                                     == bool(flag))
    return {"all_consistent": all(checks.values()) if checks else False,
            "per_check": checks,
            "note": "判据旗标与所引用的 slope/CI 数字自洽（§12 对账的算术部分）"}
