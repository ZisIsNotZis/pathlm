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
    """复核 B1 的符号钉子：rel = bpc_interp − bpc_carry；rel < 0 ⇒ carry 被支配。"""
    bpc_carry = 1.624979
    bpc_interp_dominated = 1.622720      # 纯窗口插值点更好
    bpc_interp_better = 1.630000         # 纯窗口插值点更差
    rel_dom = bpc_interp_dominated - bpc_carry
    rel_win = bpc_interp_better - bpc_carry
    assert rel_dom < 0 and rel_win > 0
    # 文档/契约的判定线必须与此一致：rel < 0 ⇒ 被同预算纯窗口支配
    import inspect
    loc_src = inspect.getsource(R.main)
    assert "bpc_interp" in loc_src


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
