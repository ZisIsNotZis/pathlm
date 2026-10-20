#!/usr/bin/env python3
"""2682 的自包含单测（本实验目录内，不改共享 `tests/`）：钉住判据② 的三分类。

判据② 的三分类（`README.md` §5a）是本实验的主判，它的**失败路径**必须被夹具走过——
`undecidable` 不是「没结论」，是「本 n 下不可判 + 给出所需样本量」，两者写错就没法复核。

跑法：`python3 -m pytest axes/1-time/1.1.1-hard-window/experiments/2682-window-kcurve-13m/test_2682_kcurve.py -q`
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
    spec = importlib.util.spec_from_file_location("run_2682", os.path.join(HERE, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()
BAND = R.FROZEN_SIGMA_CELL


def test_frozen_band_is_the_2662_value():
    """方差带冻结在 2662 的 0.000983（本实验不得重算/回填）。"""
    assert BAND == 0.000983
    assert R.CROSS_CELL_BAND == 0.014


def test_above_band_when_interval_clears_the_band():
    """税远大于带、种子又紧 ⇒ `above_band`（窗口在这个宽度上确实有代价）。"""
    v = R.classify_tax(mean=0.09, sd=0.003, n=2)
    assert v["verdict"] == "above_band"
    assert v["interval_95"][0] > BAND


def test_below_band_when_interval_sits_under_the_band():
    """税确定低于带 ⇒ `below_band`（窗口在这个宽度上不亏）。"""
    v = R.classify_tax(mean=0.0005, sd=0.00005, n=4)
    assert v["verdict"] == "below_band"
    assert v["interval_95"][1] < BAND


def test_negative_control_interval_straddles_the_band():
    """负对照：区间跨带 ⇒ `undecidable`，且**必须**给出所需样本量。"""
    v = R.classify_tax(mean=0.01, sd=0.02, n=2)
    assert v["verdict"] == "undecidable"
    assert v["interval_95"][0] < BAND < v["interval_95"][1]
    assert isinstance(v["n_req"], int) and v["n_req"] > 2


def test_negative_control_n_below_two_is_undecidable():
    """n<2 一律不可判（无跨 seed 区间），不得当成 below/above。"""
    v = R.classify_tax(mean=0.001, sd=None, n=1)
    assert v["verdict"] == "undecidable"
    assert v["n_req"] == 2                    # 先补到 n=2
    assert 0.001 > BAND                       # 且这组数据的均值确实越带（不是因均值小才不可判）


def test_expected_visible_pairs_matches_the_two_hardcoded_points():
    """掩码可见对数的闭式与两个硬编码点互为独立校验（判据⑦）。"""
    for (w, a), want in R.EXPECTED_VISIBLE.items():
        assert R.expected_visible_pairs(256, w, a) == want


def test_demo_prediction_only_judged_on_full_arm():
    """判据⑤ 只在释放版为全因果的臂上判（README §5c）。"""
    assert R.CONTRACT_DEMO_JUDGED_ARMS == ["full"]
