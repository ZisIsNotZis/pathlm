"""2621 · 搬运通道在 KV 压缩下的召回 —— 新语义层 `carry_quant.py` 的可失败单测。

Run: `python3 -m pytest tests/test_carry_quant.py -q`

每个语义开关一条：① `fp32` 释放版等值（仪器自检）；② `fp16`/`int8` 真的改了数值与
logits；③ 前向确实走 carry 通道（读出被计入）；④ `bytes_table` 的解析口径；⑤
非 carry / 非 `CarryChannelBlock` 模型被拒；⑥ `quant_accuracy` 与 release
`accuracy` 在 `fp32` 下同读数；⑦ keep_mask 语义（窗口掩码真的生效）。
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,
                             window_anchor_keep_mask)
from tools.evalkit.longrange_tasks import accuracy, assoc_recall_batch

from importlib import import_module

HERE = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type",
                    "experiments", "2621-carry-under-compression")
sys.path.insert(0, HERE)
cq = import_module("carry_quant")


def carry_model(seed=0, window=4, T=12, d=16, layers=3, heads=2):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=d, n_layers=layers, n_heads=heads, seq_len=T,
                        dense=True, train_window=window, train_anchors=0,
                        carry_rule="delta", carry_chunk=64)
    m = DepthARModel(cfg, vocab_size=10)
    # 让通道真的活着：把读出增益与写入权重从初始化的小值抬起来
    with torch.no_grad():
        for blk in m.blocks:
            blk.carry_scale.fill_(0.5)
        m.carry.w_k.normal_(std=0.3)
        m.carry.w_v.normal_(std=0.3)
        m.carry.w_q.normal_(std=0.3)
    m.eval()
    return m


def _tokens(B=4, T=12, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 10, (B, T), generator=g)


# ① fp32 释放版等值 ---------------------------------------------------------------

def test_carry_quant_fp32_bitwise_matches_released_forward():
    """`mode="fp32"` + 显式窗口掩码 == 模型自带窗口掩码的释放版前向（逐位）。"""
    m = carry_model()
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        release = m(x)[1]["depth_logits"][-1]
        fp32, tele = cq.carry_quant_logits(m, x, "fp32", keep)
    assert torch.equal(release, fp32), "fp32 档必须与释放版逐位相同（实现等值）"
    assert 0.0 <= tele["read_share"] < 1.0


# ② fp16 / int8 真的改了数值与 logits ---------------------------------------------

def test_carry_quant_int8_changes_values_and_logits():
    m = carry_model()
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    acc = None
    with torch.no_grad():
        fp32, _ = cq.carry_quant_logits(m, x, "fp32", keep)
        acc = cq._new_stats()
        int8, _ = cq.carry_quant_logits(m, x, "int8", keep, acc)
    stats = cq.summarize_stats(acc)
    assert stats["n_values"] > 0
    assert stats["changed_frac"] > 0.0, "int8 必须真的改变数值"
    assert stats["rel_err"] > 0.0
    assert not torch.equal(fp32, int8), "int8 必须改变 logits"


def test_carry_quant_fp16_changes_values_but_is_gentler_than_int8():
    """fp16 也改变数值（半个 ULP 级），但相对误差应比 int8 小。"""
    m = carry_model()
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    acc16 = cq._new_stats()
    acc8 = cq._new_stats()
    with torch.no_grad():
        cq.carry_quant_logits(m, x, "fp16", keep, acc16)
        cq.carry_quant_logits(m, x, "int8", keep, acc8)
    s16, s8 = cq.summarize_stats(acc16), cq.summarize_stats(acc8)
    assert s16["changed_frac"] > 0.0
    assert s8["rel_err"] > s16["rel_err"], "int8 的相对误差必须比 fp16 大"


# ③ 前向确实走 carry 通道 ---------------------------------------------------------

def test_carry_quant_forward_reads_the_carry_channel():
    """把读出增益置零 ⇒ logits 必须变（否则前向把通道整个丢了）。"""
    m = carry_model()
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        with_carry, tele = cq.carry_quant_logits(m, x, "int8", keep)
        for blk in m.blocks:
            blk.carry_scale.zero_()
        without_carry, tele0 = cq.carry_quant_logits(m, x, "int8", keep)
    assert tele["read_share"] > 0.0
    assert tele0["read_share"] == 0.0
    assert not torch.equal(with_carry, without_carry)


# ④ 解析字节口径 -----------------------------------------------------------------

def test_bytes_table_int8_is_half_of_fp16():
    m = carry_model()
    table = cq.bytes_table(m.cfg)
    per = table["per_mode"]
    assert per["fp16"]["per_position_bytes"] == 2.0 * per["fp32"]["values_per_position"]
    assert per["int8"]["per_position_bytes"] == per["fp16"]["per_position_bytes"] / 2
    assert per["int8"]["reduction_vs_fp16"] == 0.5
    assert table["baseline_mode"] == "fp16"


# ⑤ 只接受 carry 模型 --------------------------------------------------------------

def test_carry_quant_rejects_non_carry_model():
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12, dense=True)
    m = DepthARModel(cfg, vocab_size=10)
    m.eval()
    try:
        cq.carry_quant_logits(m, _tokens(), "int8", None)
        raise AssertionError("无通道模型必须被拒")
    except ValueError as e:
        assert "carry_rule" in str(e)


# ⑥ quant_accuracy(fp32) == release accuracy --------------------------------------

def test_quant_accuracy_matches_released_accuracy_at_fp32():
    m = carry_model()
    task = assoc_recall_batch(6, 12, 3, sym_vocab=10, filler_token=8,
                              cue_token=9,
                              generator=torch.Generator().manual_seed(5))
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        rel = accuracy(m, task, keep_mask=keep, device="cpu")
        got = cq.quant_accuracy(m, task, "fp32", keep_mask=keep, device="cpu")
    assert rel["n_hits"] == got["n_hits"]
    assert rel["accuracy"] == got["accuracy"]


# ⑦ keep_mask 语义：窗口掩码真的生效 ----------------------------------------------

def test_carry_quant_window_mask_changes_logits_vs_full_causal():
    m = carry_model()
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        windowed, _ = cq.carry_quant_logits(m, x, "fp32", keep)
        full, _ = cq.carry_quant_logits(m, x, "fp32", None)
    assert not torch.equal(windowed, full), "窗口掩码必须真的改变前向"
