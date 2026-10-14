"""2671-padfill-13m · S2 驱动 `run.py` 的自有逻辑单测（CPU、秒级）。

变异靶子：`run.py` 的 `exceeds_reference_band`（参考带边界）与 `wall_clock_fields`（L5 禁墙钟自检）。
另覆盖 `delta_bpc_block`（主判 Δ 的符号与算术）、`cost_axis`（契约 §7 代价轴）、`mean_over_seeds`。

    python3 -m pytest test_run_s2.py -q
"""

import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run  # noqa: E402


def test_reference_band_boundary_is_strict():
    """参考带 0.014：严格大于才算越带，边界值不算（+-）。"""
    assert run.REFERENCE_BAND == 0.014
    assert run.exceeds_reference_band(0.014) is False
    assert run.exceeds_reference_band(-0.014) is False
    assert run.exceeds_reference_band(0.0140001) is True
    assert run.exceeds_reference_band(-0.02) is True


def test_delta_bpc_block_sign_and_arithmetic():
    """Δbpc = bpc(fill) − bpc(ar_at_M) = (s_fill − s_ar)/(n·ln2)；fill 更差 ⇒ 正。"""
    n = 128
    s_ar = 10.0 * n * math.log(2)          # bpc(ar) = 10.0
    s_fill = 10.5 * n * math.log(2)        # bpc(fill) = 10.5
    assert run.delta_bpc_block(s_fill, s_ar, n) == pytest.approx(0.5, abs=1e-9)
    assert run.delta_bpc_block(s_ar, s_fill, n) == pytest.approx(-0.5, abs=1e-9)


def test_ci_excludes_zero():
    assert run.ci_excludes_zero(0.01, 0.02) is True
    assert run.ci_excludes_zero(-0.02, -0.01) is True
    assert run.ci_excludes_zero(-0.01, 0.02) is False


def test_cost_axis_ar_reference():
    """AR：tokens-per-forward=1、每 block 顺序前向=T−1、每 token 等效 FLOPs=L。"""
    c = run.cost_axis("ar")
    assert c["tokens_per_forward"] == 1.0
    assert c["sequential_forwards_per_block"] == float(512 - 1)
    assert c["equivalent_flops_per_token"] == float(12)


def test_cost_axis_fill_matches_contract():
    """填空：tokens-per-forward≈r·T、顺序前向=1、每 token 等效 FLOPs=L/r。"""
    c = run.cost_axis("fill_bidir_r25")
    assert c["tokens_per_forward"] == pytest.approx(0.25 * 512)
    assert c["sequential_forwards_per_block"] == 1.0
    assert c["equivalent_flops_per_token"] == pytest.approx(12 / 0.25)


def test_wall_clock_fields_flags_wall_semantics():
    """L5：干净读数 0 命中；含墙钟语义的 key 必须被抓出（run.py 自检不得被削弱）。"""
    assert run.wall_clock_fields({"steps": 1500, "arms": ["ar"], "params": 12.8}) == []
    assert run.wall_clock_fields({"wall_clock_s": 1.0}) == [".wall_clock_s"]
    assert run.wall_clock_fields({"nested": {"elapsed": 2}}) == [".nested.elapsed"]


def test_mean_over_seeds_aligns_blocks():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [3.0, 4.0, 5.0, 6.0]
    assert run.mean_over_seeds([a, b]) == [2.0, 3.0, 4.0, 5.0]


def test_ar_target_off_by_one_shape_alignment():
    """AR next-token 目标必须与 logits 逐位对齐（win[i+1] ↔ logits[i]），不得再切 [:, :-1]。"""
    import torch
    import torch.nn.functional as F
    B, T, V = 2, 4, 7
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    win = torch.randint(0, V, (B, T + 1))
    loss = run.ar_next_token_loss(logits, win)
    manual = F.cross_entropy(logits.reshape(-1, V), win[:, 1:].reshape(-1))
    assert loss.ndim == 0 and bool(torch.isfinite(loss))
    assert torch.allclose(loss, manual)
