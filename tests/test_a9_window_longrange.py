"""2623 · A9 合成长程判决（训练 window vs full）——语义层 `lr_eval.py` 的可失败单测。

Run: `python3 -m pytest tests/test_a9_window_longrange.py -q`

钉住本实验依赖的**非退化 / 同掩码 / 因果**事实（SCHEMA §17.16 / §22.8 / §19）：
1. `full` 臂口径 = 释放版（`keep_mask=None`）；`win16` = `window_anchor_keep_mask(T,16,0)`；
2. `window=16@T=256` **非退化**（`visible_frac < 1`），且**真的把关联回忆的源 key 挡在窗外**；
3. 训练侧掩码 == 评测侧掩码（`native_keep_mask` 与模型自带窗口前向逐位一致）；
4. `distance_source_inside_window` 的窗内外判据（`d < window`）；
5. `corrupt_span` **真改**源片段、且不动查询行/答案/距离（判据②对照臂的可用性）；
6. `aggregate_recall` 的聚合与 Wilson CI 口径。

全部 CPU，不取 GPU 锁。
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.join(ROOT, "axes", "A-mixing", "A9-old-info-retention",
                    "experiments", "2623-window-on-longrange-13m")
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,      # noqa: E402
                             window_anchor_keep_mask, window_mask_liveness)
from tools.evalkit.longrange_tasks import (assoc_recall_batch, var_recall_batch,  # noqa: E402
                                           corrupt_span)
from lr_eval import (native_keep_mask, arm_visibility,        # noqa: E402
                     distance_source_inside_window, aggregate_recall)

T = 256
SYM, FILL, CUE = 64, 64, 205


def _causal(n):
    return torch.tril(torch.ones(n, n, dtype=torch.bool))


def tiny_model(train_window=16, T=32, vocab=16):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=T, mlp_mult=4,
                        dense=True, train_window=train_window, train_anchors=0)
    m = DepthARModel(cfg, vocab)
    m.eval()
    return m


# ---------------------------------------------------------------- 1. 口径

def test_native_keep_mask_full_arm_is_release_path():
    """full 臂（train_window=0）⇒ None（释放版 is_causal），不是全 True 的掩码。"""
    assert native_keep_mask(T, 0, 0) is None


def test_native_keep_mask_win16_is_window_operator():
    keep = native_keep_mask(T, 16, 0)
    assert keep is not None and keep.shape == (T, T)
    assert torch.equal(keep, window_anchor_keep_mask(T, 16, 0))
    assert not torch.equal(keep, _causal(T)), "window16@T256 不得退化为恒等"


def test_anchors_only_without_window_rejected():
    with pytest.raises(ValueError):
        native_keep_mask(T, 0, 4)


def test_arm_visibility_reads_config():
    assert arm_visibility({"train_window": 16, "train_anchors": 0}) == {
        "train_window": 16, "train_anchors": 0}
    assert arm_visibility({}) == {"train_window": 0, "train_anchors": 0}


# ---------------------------------------------------------------- 2. 非退化 + 挡源

def test_window16_at_T256_is_nondegenerate():
    live = window_mask_liveness(T, 16, 0)
    assert live["alive"] is True and 0.0 < live["visible_frac"] < 1.0
    # 闭式：行 i 保留 min(i+1, 16) 个 ⇒ Σ min(i+1,16)
    expect = sum(min(i + 1, 16) for i in range(T))
    assert int(native_keep_mask(T, 16, 0).sum().item()) == expect


def test_window16_blocks_assoc_recall_source_key():
    """关联回忆的源 key 必须落在窗外：query 行(T-1) 看不到 2*qi 的 key。"""
    tb = assoc_recall_batch(4, T, 8, sym_vocab=SYM, filler_token=FILL, cue_token=CUE,
                            generator=torch.Generator().manual_seed(0),
                            query_index=0)
    keep = native_keep_mask(T, 16, 0)
    # qi=0 ⇒ 源 key 在位置 0，答案 value 在位置 1；query 行 = T-1
    assert int(tb.dists.min().item()) >= 16, "距离必须 ≥ 窗口，源才在窗外"
    assert bool(keep[T - 1, 0].item()) is False, "window16 必须挡住源 key 位置"
    assert bool(keep[T - 1, 1].item()) is False, "window16 必须挡住源 value 位置"


# ---------------------------------------------------------------- 3. 训练掩码 == 评测掩码

def test_training_mask_equals_eval_mask_bitwise():
    m = tiny_model(train_window=16, T=32)
    x = torch.randint(0, 16, (2, 32), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        native = m(x)[1]["depth_logits"][-1].float()
        from tools.evalkit.retention import masked_logits
        via_mask = masked_logits(m, x, native_keep_mask(32, 16, 0))
    assert torch.equal(native, via_mask), "训练自带掩码前向必须与评测掩码逐位一致"


# ---------------------------------------------------------------- 4. 距离腿窗内外

def test_distance_source_inside_window():
    assert distance_source_inside_window(T, 8, 16) is True
    assert distance_source_inside_window(T, 16, 16) is False
    assert distance_source_inside_window(T, 32, 16) is False
    # full 臂（window=0）恒可见
    assert distance_source_inside_window(T, 255, 0) is True
    # 与 var_recall 的物理位置一致：src_pos = (T-1) - d，query 行 = T-1
    tb = var_recall_batch(2, T, 32, sym_vocab=SYM, filler_token=FILL, cue_token=CUE,
                          generator=torch.Generator().manual_seed(1))
    keep = native_keep_mask(T, 16, 0)
    assert bool(keep[T - 1, tb.meta["src_pos"]].item()) is False


# ---------------------------------------------------------------- 5. 毁源对照臂

def test_corrupt_span_changes_source_only():
    tb = assoc_recall_batch(4, T, 8, sym_vocab=SYM, filler_token=FILL, cue_token=CUE,
                            generator=torch.Generator().manual_seed(2))
    tc = corrupt_span(tb, 0, 16, generator=torch.Generator().manual_seed(9))
    changed = (tb.tokens[:, 0:16] != tc.tokens[:, 0:16]).any().item()
    assert changed, "corrupt_span 必须真的改源片段"
    assert torch.equal(tb.tokens[:, 16:], tc.tokens[:, 16:]), "窗外不得被改"
    assert torch.equal(tb.rows, tc.rows) and torch.equal(tb.answers, tc.answers)
    assert torch.equal(tb.dists, tc.dists)


# ---------------------------------------------------------------- 6. 聚合口径

def test_aggregate_recall_wilson_and_totals():
    rows = [{"n": 100, "n_hits": 50}, {"n": 100, "n_hits": 70}]
    agg = aggregate_recall(rows, [241, 243], 0.015625)
    assert agg["n"] == 200 and agg["n_hits"] == 120
    assert agg["accuracy"] == pytest.approx(0.6)
    lo, hi = agg["accuracy_ci"]
    assert lo < 0.6 < hi and 0.0 <= lo and hi <= 1.0
    assert agg["query_key_distance"]["n"] == 2
