"""ticket 40-Y2（长 T 第三点 T=1024）的可失败单测。

钉住本实验依赖的**非退化**事实与闭式量（SCHEMA §17.16：同族三点须同相对刻度、
须用单测排除退化点）。全部 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_1024.py -q
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathlm.depth_ar import (DepthARConfig, DepthARModel,      # noqa: E402
                             window_anchor_keep_mask, window_mask_liveness)
from tools.evalkit.longrange_tasks import (var_recall_batch,   # noqa: E402
                                           check_local_prior_resistance)

D, L, H, MLP = 256, 8, 8, 4
#: 三个同族点：窗口占 T 的同一相对刻度（50%）
POINTS = ((128, 256), (256, 512), (512, 1024))


def _causal(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def _closed_form_pairs(T, w):
    """keep[i,j] = (j<=i) ∧ ((i−j)<w) ⇒ 行 i 保留 min(i+1, w) 个。"""
    return sum(min(i + 1, w) for i in range(T))


# ------------------------------------------------------------------ 退化点 / 非退化

def test_window512_at_T1024_is_nondegenerate():
    """主臂：window512@T=1024 必须**非退化**，且保留 0.7498 的因果对。"""
    keep = window_anchor_keep_mask(1024, 512, 0)
    assert not torch.equal(keep, _causal(1024)), "window512@T=1024 不得退化为恒等"
    live = window_mask_liveness(1024, 512, 0)
    assert live["alive"] is True and 0.0 < live["visible_frac"] < 1.0
    assert live["n_visible_pairs"] == _closed_form_pairs(1024, 512)
    assert live["visible_frac"] == pytest.approx(393472 / 524800, abs=1e-6)


def test_window1024_at_T1024_is_identity_and_thus_degenerate():
    """禁止凑数的点：window1024@T=1024 是恒等算子（§17.16 的退化点样例）。"""
    keep = window_anchor_keep_mask(1024, 1024, 0)
    assert torch.equal(keep, _causal(1024))
    assert window_mask_liveness(1024, 1024, 0)["alive"] is False


# ------------------------------------------------------------------ 闭式 / 同刻度

@pytest.mark.parametrize("w,T", POINTS)
def test_window_keep_count_matches_closed_form(w, T):
    keep = window_anchor_keep_mask(T, w, 0)
    assert int(keep.sum().item()) == _closed_form_pairs(T, w)
    assert window_mask_liveness(T, w, 0)["n_visible_pairs"] == _closed_form_pairs(T, w)


@pytest.mark.parametrize("w,T", POINTS)
def test_window_mask_is_causal_and_keeps_diagonal(w, T):
    """窗口必须含因果、恒开对角（否则整行被掩 ⇒ softmax NaN）。"""
    keep = window_anchor_keep_mask(T, w, 0)
    assert torch.equal(keep & ~_causal(T), torch.zeros_like(keep)), "窗口不得放出未来键"
    assert bool(keep.diagonal().all()), "对角必须恒开"
    assert bool(keep.any(dim=1).all()), "不允许出现全掩码行"


def test_three_points_have_same_relative_scale():
    """§17.16：同族三点必须同相对刻度（窗口/T = 50%）。"""
    assert all(abs(w / T - 0.5) < 1e-12 for w, T in POINTS), POINTS


# ------------------------------------------------------------------ 位置表参数增量

def test_seq_len_1024_position_table_param_delta():
    """T=1024 相对 T=512 的参数增量 = +131,072（位置表）；相对 T=256 = +196,608。"""
    def n(T):
        return sum(p.numel() for p in DepthARModel(
            DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=T,
                          mlp_mult=MLP, dense=True), 206).parameters())
    n256, n512, n1024 = n(256), n(512), n(1024)
    assert n512 - n256 == 65536
    assert n1024 - n512 == 131072
    assert n1024 - n256 == 196608


# ------------------------------------------------------------------ 占比公式

def test_attn_share_formula_T1024_is_040():
    """注意力占比公式 512T/(512T+786432)：T=1024 → 40.00%。"""
    other = 3 * D * D + D * D + 2 * MLP * D * D
    assert other == 786432
    assert round(512 * 1024 / (512 * 1024 + other), 4) == 0.4000


# ------------------------------------------------------------------ 窗口物理含义

def test_window_makes_far_source_invisible_at_T1024():
    """窗口 512 的物理含义：距离 ≥512 的源键在提问行不可见，<512 可见。"""
    keep = window_anchor_keep_mask(1024, 512, 0)
    row = 1023                                   # 提问行 = T-1
    assert bool(keep[row, row - 511]), "距离 511 仍应可见"
    assert not bool(keep[row, row - 512]), "距离 512 应被窗口挡掉"
    assert not bool(keep[row, 0]), "最远位置应不可见"


def test_window_forward_actually_changes_logits():
    """窗口开关是**活的**：同权重下「掩码开/关」前向必须不同（小尺度快测）。"""
    cfg = DepthARConfig(d_model=64, n_layers=2, n_heads=4, seq_len=64,
                        mlp_mult=4, dense=True, train_window=32)
    m = DepthARModel(cfg, 206).eval()
    x = torch.randint(0, 206, (2, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = m(x)[1]["depth_logits"][-1]
        m.cfg.train_window = 0
        b = m(x)[1]["depth_logits"][-1]
    assert float((a - b).abs().max()) > 0, "掩码没有改变前向 ⇒ 死开关"


# ------------------------------------------------------------------ §19 合成腿

@pytest.mark.parametrize("d", (256, 300, 512, 768, 1000))
def test_variable_distance_task_legal_at_T1024(d):
    """var_recall 在 T=1024 上可取 distance>255（新配置）；chance=1/64 且局部先验抗性成立。"""
    task = var_recall_batch(8, 1024, d, sym_vocab=64, filler_token=64, cue_token=205,
                            generator=torch.Generator().manual_seed(0))
    assert task.T == 1024 and int(task.dists.flatten()[0]) == d
    assert abs(task.chance() - 1 / 64) < 1e-12
    assert check_local_prior_resistance(task, k=8)["passed"], \
        "距离档必须不能被「只看最近 8 个 token」的局部先验补全"
