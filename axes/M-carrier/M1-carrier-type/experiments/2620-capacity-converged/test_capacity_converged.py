"""2620 · 本目录判据实现的可失败单测（**不放进 `tests/`**，保持 294 基线不变）。

跑法（在仓库根）：`python3 -m pytest axes/M-carrier/M1-carrier-type/experiments/
2620-capacity-converged/test_capacity_converged.py -q`

这些测试是 `mutate.py` 的靶子：每个语义接缝被故意破坏时必须抓红。
"""

import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capacity_converged_lib as L  # noqa: E402


def mk_lines(key, f, lo=0, hi=24000, step=200):
    return [{"step": s, key: f(s)} for s in range(lo, hi, step)]


# ---------------------------------------------------------------- 斜率
def test_linear_window_slope_sign_and_scale():
    """已知斜率能被复算：y = 0.002·step ⇒ 斜率 = 2.0 per 1k 步。"""
    lines = mk_lines("synth_answer_ce", lambda s: 3.0 - 0.002 * s)
    r = L.linear_window(lines, "synth_answer_ce", 16000, 23999)
    assert r["slope_per_1k"] == pytest.approx(-2.0, abs=1e-6)
    assert r["first"] > r["last"]          # 仍在下降
    assert r["rel_drop"] > 0


def test_linear_window_flat_series_is_zero_slope():
    lines = mk_lines("synth_answer_ce", lambda s: 1.234)
    r = L.linear_window(lines, "synth_answer_ce", 16000, 23999)
    assert r["slope_per_1k"] == pytest.approx(0.0, abs=1e-9)
    assert r["rel_drop"] == pytest.approx(0.0, abs=1e-9)


def test_linear_window_insufficient_points_is_detectable():
    lines = mk_lines("synth_answer_ce", lambda s: 1.0, lo=16000, hi=16400)
    r = L.linear_window(lines, "synth_answer_ce", 16000, 23999)
    assert r["n_points"] == 2
    lines1 = [{"step": 16000, "synth_answer_ce": 1.0}]
    assert L.linear_window(lines1, "synth_answer_ce", 16000, 23999)["n_points"] == 1
    assert L.linear_window(lines1, "synth_answer_ce", 16000, 23999)["slope_per_1k"] is None


def test_loglog_window_matches_power_law():
    """y = 3·step^-0.5 ⇒ log-log 斜率 = −0.5。"""
    lines = mk_lines("loss", lambda s: 3.0 * max(s, 1) ** -0.5, lo=200, hi=24000)
    r = L.loglog_window(lines, "loss", 200, 24000)
    assert r["slope"] == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------- 判据 ①
def test_convergence_gate_rejects_still_declining_ansce():
    """U1 末窗的答案行 CE 斜率量级（−0.03/1k、相对下降 7–8%）必须判「未收敛」。"""
    lines = mk_lines("synth_answer_ce", lambda s: 1.6 * math.exp(-0.03 * (s - 16000) / 1000))
    r = L.linear_window(lines, "synth_answer_ce", 16000, 23999)
    g = L.convergence_gate(r)
    assert r["slope_per_1k"] < -0.01
    assert r["rel_drop"] > 0.01
    assert g["converged"] is False


def test_convergence_gate_accepts_plateau():
    lines = mk_lines("synth_answer_ce", lambda s: 1.50 + 1e-8 * s)
    g = L.convergence_gate(L.linear_window(lines, "synth_answer_ce", 16000, 23999))
    assert g["converged"] is True
    assert g["passes_slope"] is True


def test_convergence_gate_accepts_small_relative_drop_even_if_slope_big():
    """相对下降 <1% 单条达标也可 ⇒ 契约写的是「或」。

    （用大基数序列构造：|slope| = 0.02/1k ≥ 阈值，但相对下降 0.16% < 1%（100 − 2e-5·step）。）"""
    lines = mk_lines("synth_answer_ce", lambda s: 100.0 - 0.00002 * (s - 16000))
    r = L.linear_window(lines, "synth_answer_ce", 16000, 23999)
    g = L.convergence_gate(r)
    assert abs(r["slope_per_1k"]) >= 0.01
    assert abs(r["rel_drop"]) < 0.01
    assert g["converged"] is True and g["passes_rel_drop"] is True


def test_convergence_gate_rejects_rising_ansce():
    """末窗**上升**（答案行 CE 变差）绝不能判成「收敛」——哪怕幅度看起来小。"""
    rising_big = mk_lines("synth_answer_ce", lambda s: 1.0 + 0.001 * (s - 16000))
    g = L.convergence_gate(L.linear_window(rising_big, "synth_answer_ce", 16000, 23999))
    assert g["rel_drop"] < 0 and abs(g["rel_drop"]) > 0.01
    assert g["converged"] is False


def test_convergence_gate_insufficient_points_is_not_converged():
    g = L.convergence_gate({"key": "synth_answer_ce", "n_points": 0,
                            "slope_per_1k": None, "rel_drop": None})
    assert g["converged"] is False


def test_total_loss_gate_would_be_vacuous_documentation():
    """文档性断言：总 CE 的斜率门槛在本任务上**恒真**（README §4① 的实测证据）。

    U1 末窗 [6000,10000] 总 CE 斜率实测 −0.00036 ~ −0.00055 /1k ⇒ 早已达标。
    这里把该数字钉住，防止有人误用总 CE 当门禁（变异 M5 抓的就是这件事）。
    """
    u1_total_ce_slopes = [-0.00036, -0.00046, -0.00055, -0.00050]
    assert all(abs(s) < L.CONVERGENCE_SLOPE_MAX for s in u1_total_ce_slopes)
    u1_ansce_slopes = [-0.03245, -0.02984, -0.02650, -0.02756]
    assert all(abs(s) >= L.CONVERGENCE_SLOPE_MAX for s in u1_ansce_slopes)


# ---------------------------------------------------------------- 判据 ②
def test_capacity_verdict_requires_both_delta_and_ci_separation():
    v = L.capacity_verdict(acc_small=0.30, n_small=20480, acc_big=0.35, n_big=20480,
                           ci_small=[0.2937, 0.3064], ci_big=[0.3436, 0.3564])
    assert v["delta_pp"] == pytest.approx(5.0, abs=1e-6)
    assert v["meets_delta"] and not v["ci_overlap"]
    assert v["capacity_axis_effective"] is True


def test_capacity_verdict_rejects_small_delta_even_with_separated_ci():
    v = L.capacity_verdict(0.30, 20480, 0.32, 20480,
                           [0.2937, 0.3064], [0.3136, 0.3264])
    assert v["delta_pp"] == pytest.approx(2.0, abs=1e-6)
    assert not v["meets_delta"]
    assert v["capacity_axis_effective"] is False


def test_capacity_verdict_rejects_overlapping_ci_even_with_big_delta():
    v = L.capacity_verdict(0.30, 256, 0.40, 256,
                           [0.2457, 0.3598], [0.3419, 0.4610])
    assert v["meets_delta"] is True and v["ci_overlap"] is True
    assert v["capacity_axis_effective"] is False


def test_corrupt_invalidate_line_is_4x_chance():
    assert L.CHANCE == pytest.approx(0.015625)
    assert L.corrupt_invalidates(0.05, [0.0630, 0.10]) is True    # 下界 > 0.0625
    assert L.corrupt_invalidates(0.05, [0.0620, 0.10]) is False


# ---------------------------------------------------------------- 判据 ③.2
def test_training_divergence_matched_steps_only():
    a = [{"step": 0, "loss": 1.0, "synth_answer_ce": 3.0},
         {"step": 200, "loss": 0.5, "synth_answer_ce": 2.0}]
    b = [{"step": 0, "loss": 1.0, "synth_answer_ce": 3.0},
         {"step": 200, "loss": 0.6, "synth_answer_ce": 2.0},
         {"step": 400, "loss": 0.4, "synth_answer_ce": 1.0}]
    d = L.training_divergence(a, b)
    assert d["per_key"]["loss"]["n_steps_compared"] == 2
    assert d["per_key"]["loss"]["max_abs_diff"] == pytest.approx(0.1, abs=1e-9)
    assert d["per_key"]["synth_answer_ce"]["final_abs_diff"] == 0.0


def test_reconciliation_flags_are_self_consistent():
    results = {
        "convergence": {"by_arm": {"S16": {"gate": {"slope_per_1k": -0.002,
                                                     "rel_drop": 0.003,
                                                     "converged": True}},
                                   "S128": {"gate": {"slope_per_1k": -0.03,
                                                      "rel_drop": 0.05,
                                                      "converged": False}}}},
        "criteria": {"c2_main_by_K": {"K8": {"verdict": {
                         "delta_pp": 5.0, "ci_overlap": False,
                         "capacity_axis_effective": True}}},
                     "c3_1_corrupt": {"S16_K8": {"accuracy_ci": [0.01, 0.02],
                                                 "invalidated": False}}},
    }
    r = L.reconciliation_checks(results)
    assert r["all_consistent"] is True
