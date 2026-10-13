#!/usr/bin/env python3
"""2638 的**自包含单测**（本实验目录内，避免改共享 `tests/` 目录——见边界纪律）。

跑法：`python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py -q`

钉住三件事（复核 N8 / B1）：
1. `frontier()` 的支配关系定义（两轴皆越小越好）；
2. `interp_window_at()` 的线性插值；
3. **符号约定**：`carry_minus_interp_window = bpc_interp − bpc_carry`
   ⇒ **< 0 = carry 被同预算纯窗口支配**（与 `SCHEMA.md` §17.3「Δ = 对照 − 实验，正 = 改善」一致）。
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run as R  # noqa: E402


def _row(name, total_bytes, bpc, kind="window", seed=0, k_eff=None):
    return {"name": name, "kind": kind, "seed": seed, "bpc": bpc,
            "total_bytes": total_bytes,
            "k_eff": total_bytes / 12288 if k_eff is None else k_eff}


def test_frontier_domination_smaller_both_axes():
    rows = [_row("a", 100, 1.0), _row("b", 200, 0.9), _row("c", 150, 1.1)]
    fr = R.frontier(rows)
    # b 内存更多、质量更好 ⇒ 不支配任何点；a（内存最少）不被支配
    assert fr["dominated_by"]["a"] == []
    assert fr["dominated_by"]["b"] == []
    assert fr["dominated_by"]["c"] == ["a"]          # a 内存更少且 bpc 更低
    assert fr["lower_envelope"] == ["a", "b"]        # 单调下降包络


def test_frontier_equal_bytes_needs_strictly_better_bpc():
    rows = [_row("x", 100, 1.0), _row("y", 100, 1.0)]
    fr = R.frontier(rows)
    assert fr["dominated_by"]["x"] == [] and fr["dominated_by"]["y"] == []
    rows2 = [_row("x", 100, 1.0), _row("y", 100, 0.99)]
    fr2 = R.frontier(rows2)
    assert fr2["dominated_by"]["x"] == ["y"] and fr2["dominated_by"]["y"] == []


def test_interp_window_linear():
    rows = [_row("win16", 16 * 12288, 1.6), _row("win24", 24 * 12288, 1.4)]
    got = R.interp_window_at(rows, 20.0)
    assert got is not None
    assert got["between"] == ["win16", "win24"]
    assert got["bpc_interp"] == pytest.approx(1.5, abs=1e-9)
    assert R.interp_window_at(rows, 8.0) is None      # 不外推


def test_sign_convention_carry_minus_interp():
    """符号钉子：rel = bpc_interp − bpc_carry；rel < 0 ⇒ carry 被同预算纯窗口支配。"""
    bpc_carry = 1.624979
    bpc_interp_dominated = 1.622720      # 纯窗口插值点更好
    bpc_interp_better = 1.630000         # 纯窗口插值点更差
    assert (bpc_interp_dominated - bpc_carry) < 0
    assert (bpc_interp_better - bpc_carry) > 0


def test_results_json_domination_matches_documented_sign_rule():
    """把文档约定的判定线（rel < 0 ⇒ 被支配）与落盘 `results.json` 的支配关系对起来。

    这是**实质**断言（不是子串检查）：① 校验 `carry_minus_interp_window` = `bpc_interp − bpc_carry`；
    ② 校验「rel < 0 ⇔ 被支配」与 `frontier.dominated_by` **逐臂一致**——若有人把判定线方向写反，
    本条会红。
    """
    import json
    import os
    d = json.load(open(os.path.join(HERE, "results.json")))
    rows = {r["name"]: r for r in d["rows"]}
    dom = d["frontier"]["dominated_by"]
    checked = 0
    for name, loc in d["location"].items():
        rel = loc["carry_minus_interp_window"]
        if rel is None:
            continue
        interp = loc["window_curve_at_k_eff"]["bpc_interp"]
        assert rel == pytest.approx(interp - rows[name]["bpc"], abs=1e-5), name
        dominated = len(dom.get(name, [])) > 0
        assert (rel < 0) == dominated, (name, rel, dominated)
        checked += 1
    assert checked >= 2


def test_keep_mask_visible_pairs_formula():
    import torch
    from pathlm.depth_ar import window_anchor_keep_mask
    T = 32
    for K in (1, 16, 17, 24, 32):
        km = window_anchor_keep_mask(T, K, 0)
        s = int(km.sum().item())
        exp = sum(min(i + 1, K) for i in range(T))
        assert s == exp, (K, s, exp)
        assert s < T * (T + 1) // 2 or K >= T
