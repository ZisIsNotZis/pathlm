#!/usr/bin/env python3
"""2688 的自包含单测（本实验目录内，不改共享 `tests/`）：钉住判据② 的三分类、判据③ 的杠杆判定
与 `2688` 新增的那对**预测函数**（契约 §1 的门槛由它们给出，预测与实测必须走同一条规则）。

负对照直接取 `2686` 的落盘 n=3 读数：它们在 n=3 下判 `undecidable`，在本 n（5 / 4）下允许翻转；
判据③ 的 n=4 负对照取本轮落盘的四个价（区间含 0 ⇒ 仍是 `undecidable`，不得被读成「锚点无效」）。

跑法：`python3 -m pytest axes/1-time/1.1.1-hard-window/experiments/2688-window-kcurve-seed34-13m/test_2688_seed34.py -q`
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_run():
    """按路径加载本目录的 `run.py`（不用 `import run`：同仓有多个 `run.py`，会撞名）。"""
    spec = importlib.util.spec_from_file_location("run_2688", os.path.join(HERE, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()
BAND = R.FROZEN_SIGMA_CELL

#: 本轮的四个杠杆价（`results.json::criteria.c3_anchor_lever_k64_n4.per_seed` 的落盘值）。
LEVER_N4 = [-0.007652, -0.008982, -0.003166, 0.000478]
#: 本轮的五个 k=128 税（`results.json::criteria.c2_k128_tax_n5` 的落盘值）。
K128_N5 = [0.012607, 0.009808, 0.002762, 0.005849, 0.010376]


def test_frozen_band_and_cross_cell_band_are_the_2662_values():
    """带冻结在 2662 的 0.000983（本实验不得重算/回填），跨格带 0.014 只作参考。"""
    assert BAND == 0.000983
    assert R.CROSS_CELL_BAND == 0.014


def test_contract_arm_order_matches_the_preregistration():
    """臂清单与顺序是契约（`config.json` 逐项核对），改了它判据面就变了。"""
    assert R.CONTRACT_ARMS == ["full",
                              "win64_s0", "win64_s1", "win64_s2", "win64_s3",
                              "win64a_s0", "win64a_s1", "win64a_s2", "win64a_s3",
                              "win128_s0", "win128_s1", "win128_s2", "win128_s3", "win128_s4"]
    assert R.CONTRACT_NEW_ARMS == ["win64_s3", "win64a_s3", "win128_s3", "win128_s4"]
    assert R.CONTRACT_SEEDS_K128 == [0, 1, 2, 3, 4]
    assert R.CONTRACT_SEEDS_LEVER == [0, 1, 2, 3]


def test_negative_control_k128_at_n3_is_undecidable():
    """负对照：`2686` 的 n=3 落盘读数必须判 `undecidable`（区间跨带）。"""
    v = R.classify_tax(R.K128_N3["mean"], R.K128_N3["sd"], 3)
    assert v["verdict"] == "undecidable"
    assert v["interval_95"][0] < BAND < v["interval_95"][1]


def test_k128_at_n5_clears_the_band_and_stays_below_the_cross_cell_band():
    """本轮的判词：n=5 判 `above_band`，且整条区间落在 0.014 跨格带以内（两带并列读）。"""
    csm = R.cross_seed_mean(K128_N5)
    v = R.classify_tax(csm["mean"], csm["sd"], csm["n_seeds"])
    assert csm["n_seeds"] == 5 and csm["same_sign"] is True
    assert v["verdict"] == "above_band"
    assert v["interval_95"][0] > BAND
    assert v["interval_95"][1] < R.CROSS_CELL_BAND


def test_lever_at_n4_is_still_undecidable_and_says_so():
    """本轮的判词：n=4 的杠杆区间含 0 ⇒ `undecidable` + `n_req`，且不得读成「锚点无效」。"""
    csm = R.cross_seed_mean(LEVER_N4)
    v = R.classify_lever(csm)
    assert csm["n_seeds"] == 4 and csm["same_sign"] is False
    assert v["verdict"] == "undecidable" and v["direction"] is None
    assert isinstance(v["n_req"], int) and v["n_req"] > 4
    assert "不得读成" in v["note"]


def test_lever_decidable_reports_direction():
    """区间整体 < 0 ⇒ 可判且方向为负（锚点使质量更好）；整体 > 0 ⇒ 正向。"""
    neg = R.classify_lever(R.cross_seed_mean([-0.0076, -0.0089, -0.0082]))
    assert neg["verdict"] == "decidable" and neg["direction"] == "price_negative"
    pos = R.classify_lever(R.cross_seed_mean([0.0076, 0.0089, 0.0082]))
    assert pos["verdict"] == "decidable" and pos["direction"] == "price_positive"


def test_forecasts_use_the_same_rule_and_report_their_thresholds():
    """契约 §1 的两条预测必须与实测走同一条判定规则，并给出可反解的 `sd` 门槛。"""
    fc_tax = R.forecast_tax(R.K128_N3, 5)
    assert fc_tax["predicted_verdict"] == "above_band"
    assert fc_tax["sd_upper_bound_for_prediction"] == 0.005968
    fc_lever = R.forecast_lever(R.LEVER_N3, 4)
    assert fc_lever["predicted_verdict"] == "decidable"
    assert fc_lever["predicted_direction"] == "price_negative"
    assert fc_lever["sd_upper_bound_for_prediction"] == 0.004148
    # 预测与实测同源：把实测的 (mean, sd, n) 喂进同一条规则，判词必须与落盘判定一致
    csm = R.cross_seed_mean(LEVER_N4)
    assert R.lever_from_mean_sd(csm["mean"], csm["sd"], 4)["verdict"] == "undecidable"


def test_expected_visible_pairs_matches_the_hardcoded_points():
    """掩码可见对数的闭式与三个硬编码点互为独立校验（判据④）。"""
    for (w, a), want in R.EXPECTED_VISIBLE.items():
        assert R.expected_visible_pairs(256, w, a) == want
