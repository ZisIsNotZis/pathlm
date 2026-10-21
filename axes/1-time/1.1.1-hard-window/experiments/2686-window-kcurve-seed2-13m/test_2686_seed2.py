#!/usr/bin/env python3
"""2686 的自包含单测（本实验目录内，不改共享 `tests/`）：钉住判据② 的三分类与判据③ 的杠杆判定。

两处判定都必须走过**失败路径**：`undecidable` 不是「没结论」，是「本 n 下不可判 + 给出所需样本量」。
负对照直接取 `2682` 的落盘 n=2 读数——它们在旧 n 下必须判 `undecidable`，在本 n=3 下才允许翻转。

跑法：`python3 -m pytest axes/1-time/1.1.1-hard-window/experiments/2686-window-kcurve-seed2-13m/test_2686_seed2.py -q`
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
    spec = importlib.util.spec_from_file_location("run_2686", os.path.join(HERE, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()
BAND = R.FROZEN_SIGMA_CELL

#: `2682/results.json` 的落盘 n=2 读数（负对照的输入，不重算）。
K128_N2 = {"values": [0.012607, 0.009808], "mean": 0.011208, "sd": 0.001979}
LEVER_N2 = {"values": [-0.007652, -0.008982], "mean": -0.008317, "sd": 0.00094}


def test_frozen_band_and_cross_cell_band_are_the_2662_values():
    """带冻结在 2662 的 0.000983（本实验不得重算/回填），跨格带 0.014 只作参考。"""
    assert BAND == 0.000983
    assert R.CROSS_CELL_BAND == 0.014


def test_contract_arm_order_matches_the_preregistration():
    """臂清单与顺序是契约（`config.json` 逐项核对），改了它判据面就变了。"""
    assert R.CONTRACT_ARMS == ["full", "win64_s0", "win64_s1", "win64_s2",
                              "win64a_s0", "win64a_s1", "win64a_s2",
                              "win128_s0", "win128_s1", "win128_s2"]
    assert R.CONTRACT_NEW_ARMS == ["win64_s2", "win64a_s2", "win128_s2"]
    assert R.CONTRACT_SEEDS == [0, 1, 2]


def test_negative_control_k128_at_n2_is_undecidable():
    """负对照：k=128 在 n=2 的落盘读数必须判 `undecidable`，且给出所需样本量 > 2。"""
    v = R.classify_tax(K128_N2["mean"], K128_N2["sd"], 2)
    assert v["verdict"] == "undecidable"
    assert v["interval_95"][0] < BAND < v["interval_95"][1]
    assert isinstance(v["n_req"], int) and v["n_req"] >= 1     # 与 2682 落盘 n_req=1 一致


def test_negative_control_lever_at_n2_is_undecidable():
    """负对照：k=64 的锚点杠杆在 n=2 的落盘读数区间含 0 ⇒ `undecidable`。"""
    csm = R.cross_seed_mean(LEVER_N2["values"])
    assert R.classify_lever(csm)["verdict"] == "undecidable"
    assert csm["t_interval_95"][0] < 0 < csm["t_interval_95"][1]


def test_same_values_at_n3_clear_the_band():
    """同一均值/散布下 n=3 的半宽收窄 ⇒ k=128 判 `above_band`（本轮的预测面）。"""
    v = R.classify_tax(K128_N2["mean"], K128_N2["sd"], 3)
    assert v["verdict"] == "above_band"
    assert v["interval_95"][0] > BAND


def test_lever_decidable_reports_direction():
    """区间整体 < 0 ⇒ 可判且方向为负（锚点使质量更好）；整体 > 0 ⇒ 正向。"""
    neg = R.classify_lever(R.cross_seed_mean([-0.0076, -0.0089, -0.0082]))
    assert neg["verdict"] == "decidable" and neg["direction"] == "price_negative"
    pos = R.classify_lever(R.cross_seed_mean([0.0076, 0.0089, 0.0082]))
    assert pos["verdict"] == "decidable" and pos["direction"] == "price_positive"


def test_undecidable_lever_is_never_read_as_no_effect():
    """区间含 0 时必须带 `n_req` 与那句「不得读成锚点无效」，不得只给一个布尔。"""
    v = R.classify_lever(R.cross_seed_mean([0.001, 0.02]))
    assert v["verdict"] == "undecidable"
    assert v["direction"] is None
    assert isinstance(v["n_req"], int) and v["n_req"] > 2
    assert "不得读成" in v["note"]
    mean_zero = R.classify_lever(R.cross_seed_mean([-0.02, 0.02]))
    assert mean_zero["verdict"] == "undecidable" and mean_zero["n_req"] is None
    assert "样本量不救" in mean_zero["n_req_note"]


def test_expected_visible_pairs_matches_the_hardcoded_points():
    """掩码可见对数的闭式与三个硬编码点互为独立校验（判据④）。"""
    for (w, a), want in R.EXPECTED_VISIBLE.items():
        assert R.expected_visible_pairs(256, w, a) == want


def test_k192_is_declared_not_measured():
    """k=192 的「不取得」是登记事实，不是读数：不得被当成 measured。"""
    assert R.K192_N2["measured_here"] is False
    assert R.K192_N2["verdict"] == "undecidable"
    assert R.K192_N2["n_seeds"] == 2
