#!/usr/bin/env python3
"""2681 主判的**负对照单测**（复核 P2-6）：钉住 `decidable` 的失败路径。

`run.py` 的 n=4 主判是一个三项合取：`Δ_pp` 均值 > 0 **且** ≥3/4 同向 **且** 两个跨 seed
区间的下界都 > 0。此前这条失败路径**从未被任何夹具走过**（`tests/` 里对 `cross_seed_mean`
与 `two_level_bootstrap` 的引用为零）⇒「可判」这个判词只有一个方向被验证过。

本文件做四件事：

1. 正对照：一组应当判 `decidable=True` 的数（每 seed 都严格为正、方差为 0）；
2. 负对照 A：**同向但均值 < 0**（三项里第 1 项就挡掉）；
3. 负对照 B：**区间含 0**（均值 > 0、4/4 同向，但 t 区间的下界 < 0）；
4. 负对照 C：**方向不齐**（均值 > 0 但只有 2/4 严格为正）；
5. 源码钉：`run.py` 仍用同一个合取式（改了门就必须同步改这里，否则测试红）。

跑法：`python3 -m pytest axes/1-time/1.1.1-hard-window/experiments/2681-anchor-recall-seeds23-13m/test_2681_negative_controls.py -q`
"""

import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_run():
    """按路径加载本目录的 `run.py`（不用 `import run`：同仓有多个 `run.py`，会撞名）。"""
    spec = importlib.util.spec_from_file_location("run_2681", os.path.join(HERE, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()


def _decidable(deltas_pp, per_seed_diffs):
    """`run.py::main()` 里 n=4 主判的那个合取式，逐项照抄（源码钉由最后一个测试保证）。"""
    csm = R.cross_seed_mean(deltas_pp)
    two_level = R.two_level_bootstrap(per_seed_diffs, n_boot=200)
    t_lo = None if csm.get("t_interval_95") is None else csm["t_interval_95"][0]
    return bool(csm["mean"] > 0 and csm["n_positive"] >= 3
                and t_lo is not None and t_lo > 0
                and two_level.get("passes_lower_gt_0") is True), csm, two_level


def _diffs(n_per_seed, delta_hits):
    """造 n_seeds 组逐针配对差（每 seed 都有 `delta_hits` 根「植入命中而地板没中」的针）。"""
    return [[1] * delta_hits + [0] * (n_per_seed - delta_hits) for _ in range(4)]


def test_positive_control_is_decidable():
    """正对照：4/4 严格为正、方差 0 ⇒ 两个区间下界都 > 0 ⇒ 判可判。"""
    ok, csm, two = _decidable([2.0, 2.0, 2.0, 2.0], _diffs(1000, 20))
    assert csm["n_positive"] == 4 and csm["mean"] == 2.0
    assert csm["t_interval_95"] == [2.0, 2.0]
    assert ok is True


def test_negative_control_mean_below_zero():
    """负对照 A：**同向但均值 < 0** ⇒ 第 1 项（均值 > 0）挡掉。"""
    ok, csm, _ = _decidable([-1.0, -1.0, -1.0, -1.0], _diffs(1000, 0))
    assert csm["n_positive"] == 0 and csm["mean"] < 0
    assert ok is False


def test_negative_control_interval_contains_zero():
    """负对照 B：**某区间含 0** ⇒ 第 3 项（两个下界都 > 0）挡掉。

    构造：4/4 同向、均值 > 0，但 sd 大到 t 区间下界 < 0（df=3、t=3.182）。
    """
    deltas = [0.5, 0.5, 4.0, 4.0]           # 全正，均值 2.25
    ok, csm, _ = _decidable(deltas, _diffs(1000, 10))
    assert csm["n_positive"] == 4 and csm["mean"] > 0
    assert csm["t_interval_95"][0] < 0      # 区间跨 0
    assert ok is False


def test_negative_control_direction_not_aligned():
    """负对照 C：**方向不齐**（2/4 为正）⇒ 第 2 项（≥3/4）挡掉。"""
    ok, csm, _ = _decidable([3.0, 3.0, -0.1, -0.1], _diffs(1000, 10))
    assert csm["n_positive"] == 2 and csm["mean"] > 0
    assert ok is False


def test_negative_control_n_below_two_has_no_interval():
    """n<2 一律不可判：`cross_seed_mean` 必须给 `t_interval_95 = None`。"""
    csm = R.cross_seed_mean([5.0])
    assert csm["t_interval_95"] is None and csm["n_seeds"] == 1


def test_gate_expression_matches_run_py_source():
    """源码钉：`run.py` 的主判仍是这个合取式；改了门就必须在这里一起改。"""
    src = re.sub(r"\s+", " ", open(os.path.join(HERE, "run.py"), encoding="utf-8").read())
    want = ('decidable = bool(csm["mean"] > 0 and csm["n_positive"] >= 3 '
            'and t_lo is not None and t_lo > 0 '
            'and two_level.get("passes_lower_gt_0") is True)')
    assert want in src, "run.py 的 decidable 合取式变了：同步更新本测试的 _decidable()"
