"""ticket 39-X3（长 T 首探 T=512）的可失败单测。

钉住本实验依赖的**非退化**事实与闭式量（SCHEMA §17.16：同族两点须同相对刻度、
须用单测排除退化点）。全部 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_512.py -q
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
                                           local_prior_accuracy,
                                           check_local_prior_resistance)

D, L, H, MLP = 256, 8, 8, 4
#: 两个同族点：窗口占 T 的同一相对刻度（50%）
POINTS = ((128, 256), (256, 512))


def _causal(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def _closed_form_pairs(T, w):
    """keep[i,j] = (j<=i) ∧ ((i−j)<w) ⇒ 行 i 保留 min(i+1, w) 个。"""
    return sum(min(i + 1, w) for i in range(T))


# ------------------------------------------------------------------ 退化点

def test_window256_at_T256_is_identity_and_thus_degenerate():
    """契约指定的「窗口 256@T=256」是恒等算子 ⇒ Δ≡0，不是真观测点（§17.16）。"""
    keep = window_anchor_keep_mask(256, 256, 0)
    assert torch.equal(keep, _causal(256)), "window256@T=256 应与精确因果掩码逐位相等"
    live = window_mask_liveness(256, 256, 0)
    assert live["alive"] is False, "恒等算子必须被活性量判为退化（alive=False）"
    assert live["visible_frac"] == 1.0


def test_window128_at_T256_is_nondegenerate():
    """补的真第二点：window128@T=256 必须**非退化**。"""
    keep = window_anchor_keep_mask(256, 128, 0)
    assert not torch.equal(keep, _causal(256)), "window128@T=256 不得退化为恒等"
    live = window_mask_liveness(256, 128, 0)
    assert live["alive"] is True and 0.0 < live["visible_frac"] < 1.0


def test_window256_at_T512_is_nondegenerate():
    """主臂：window256@T=512 必须非退化且保留 0.7495 的因果对。"""
    keep = window_anchor_keep_mask(512, 256, 0)
    assert not torch.equal(keep, _causal(512))
    live = window_mask_liveness(512, 256, 0)
    assert live["alive"] is True
    assert live["visible_frac"] == pytest.approx(98432 / 131328, abs=1e-6)


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


def test_two_points_have_same_relative_scale():
    """§17.16：同族两点必须同相对刻度（窗口/T = 50%）。"""
    assert all(abs(w / T - 0.5) < 1e-12 for w, T in POINTS), POINTS


def test_seq_len_512_position_table_param_delta():
    """T=512 相对 T=256 的参数增量 = +65,536（位置表），≈ +0.94%（更正 K10 的 +2.8%）。"""
    m256 = DepthARModel(DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=256,
                                      mlp_mult=MLP, dense=True), 206)
    m512 = DepthARModel(DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=512,
                                      mlp_mult=MLP, dense=True), 206)
    n256 = sum(p.numel() for p in m256.parameters())
    n512 = sum(p.numel() for p in m512.parameters())
    assert n512 - n256 == 65536
    assert round((n512 - n256) / n256 * 100, 2) == 0.94


def test_attn_share_formula_T512_is_025():
    """注意力占比公式 512T/(512T+786432)：T=256 → 14.29%，T=512 → 25.00%（修正值）。"""
    other = 3 * D * D + D * D + 2 * MLP * D * D
    assert other == 786432
    assert round(512 * 256 / (512 * 256 + other), 4) == 0.1429
    assert round(512 * 512 / (512 * 512 + other), 4) == 0.2500


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

@pytest.mark.parametrize("d", (255, 300, 480))
def test_variable_distance_task_legal_beyond_255(d):
    """var_recall 在 T=512 上可取 distance>255（本次新配置）；chance=1/64 且局部先验抗性成立。"""
    task = var_recall_batch(8, 512, d, sym_vocab=64, filler_token=64, cue_token=205,
                            generator=torch.Generator().manual_seed(0))
    assert task.T == 512 and int(task.dists.flatten()[0]) == d
    assert abs(task.chance() - 1 / 64) < 1e-12
    assert check_local_prior_resistance(task, k=8)["passed"], \
        "距离档必须不能被「只看最近 8 个 token」的局部先验补全"


def test_window_makes_far_source_invisible_at_T512():
    """窗口 256 的物理含义：距离 ≥256 的源键在提问行不可见（长程腿的机制前提）。"""
    keep = window_anchor_keep_mask(512, 256, 0)
    row = 511                                   # 提问行 = T-1
    assert bool(keep[row, row - 255]), "距离 255 仍应可见"
    assert not bool(keep[row, row - 256]), "距离 256 应被窗口挡掉"
    assert not bool(keep[row, 0]), "最远位置应不可见"
