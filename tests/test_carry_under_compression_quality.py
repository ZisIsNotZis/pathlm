"""2624 · 搬运×压缩质量腿（Z3）—— 新语义层 `quant_carry.py` + 三臂逻辑的可失败单测。

Run: `python3 -m pytest tests/test_carry_under_compression_quality.py -q`

靶子 = 本实验自己的语义逻辑（`quant_carry.py` + `compression_arms.py`）：两种块类
（`CarryChannelBlock` / `ComposableBlock`）的 `fp32` 档与释放版前向**逐位相等**、
`fp16`/`int8` 真的改了数值与 logits、非 carry / 非目标块类被拒、三行评测的定义
（权重来源 + KV 精度）、窗口掩码在 T=256 非退化（§17.16）、分量分解符号正确。
引擎三特性本身的活性/变异已由 `tests/test_eng1_composable.py` 钉住（本文件不重复）。

变异验证（每处抓红 + 逐字节还原）见同目录 `mutate.py`。
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,
                             window_anchor_keep_mask, window_mask_liveness)
from tools.evalkit.longrange_tasks import accuracy, assoc_recall_batch
from tools.evalkit.retention import masked_logits

from importlib import import_module

HERE = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory",
                    "experiments", "2624-carry-under-compression-quality")
sys.path.insert(0, HERE)
qc = import_module("quant_carry")
ca = import_module("compression_arms")


def carry_model(gqa=False, seed=0, window=4, T=12, d=16, layers=3, heads=2):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=d, n_layers=layers, n_heads=heads, seq_len=T,
                        dense=True, train_window=window, train_anchors=0,
                        carry_rule="delta", carry_chunk=64,
                        **({"n_kv_heads": heads // 2} if gqa else {}))
    m = DepthARModel(cfg, vocab_size=10)
    # 让通道真的活着：把读出增益与写入权重抬起来（否则读出差为 0，掩盖接线错误）
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


# ① 三行评测的定义 -------------------------------------------------------------

def test_trained_arms_differ_only_in_n_kv_heads():
    """两条被训练的臂只差 `n_kv_heads`；carry/window/seed 是常量。"""
    specs = {a["name"]: a for a in ca.TRAINED_ARMS}
    assert set(specs) == {"carry_mha", "carry_gqa"}
    f_mha = ca.model_config_kwargs(specs["carry_mha"])
    f_gqa = ca.model_config_kwargs(specs["carry_gqa"])
    assert f_mha.get("n_kv_heads") is None
    assert f_gqa["n_kv_heads"] == 2
    assert f_mha["carry_rule"] == f_gqa["carry_rule"] == "delta"
    assert f_mha["train_window"] == f_gqa["train_window"] == 16
    assert f_mha["train_anchors"] == f_gqa["train_anchors"] == 0
    only_gqa = {k for k in set(f_mha) | set(f_gqa) if f_mha.get(k) != f_gqa.get(k)}
    assert only_gqa == {"n_kv_heads"}


def test_expected_block_classes():
    assert ca.expected_block_class(ca.trained_arm("carry_mha")) == "CarryChannelBlock"
    assert ca.expected_block_class(ca.trained_arm("carry_gqa")) == "ComposableBlock"


def test_eval_rows_three_int8_shares_gqa_weights():
    rows = {a["name"]: a for a in ca.EVAL_ARMS}
    assert set(rows) == {"carry_mha_fp16", "carry_gqa_fp16", "carry_gqa_int8"}
    assert rows["carry_mha_fp16"]["weights"] == "carry_mha"
    assert rows["carry_mha_fp16"]["kv_mode"] == "fp16"
    assert rows["carry_gqa_fp16"]["weights"] == "carry_gqa"
    # 压缩行**复用** carry_gqa 的权重（不训练、不加参数），只换 KV 精度
    assert rows["carry_gqa_int8"]["weights"] == "carry_gqa"
    assert rows["carry_gqa_int8"]["kv_mode"] == "int8"
    assert ca.BASELINE_EVAL_ARM == "carry_mha_fp16"
    assert ca.COMPRESSED_EVAL_ARM == "carry_gqa_int8"


# ② fp32 释放版等值（仪器自检，两种块类各一条） --------------------------------

def test_fp32_bitwise_matches_released_composable():
    """GQA×carry 块：`mode="fp32"` + 窗口掩码 == 释放版前向（逐位）。"""
    m = carry_model(gqa=True)
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        rel = masked_logits(m, x, keep)
        q, _ = qc.quant_carry_logits(m, x, "fp32", keep)
    assert type(m.blocks[0]).__name__ == "ComposableBlock"
    assert torch.equal(rel, q)


def test_fp32_bitwise_matches_released_carrychannel():
    """MHA×carry 块：`mode="fp32"` + 窗口掩码 == 释放版前向（逐位）。"""
    m = carry_model(gqa=False)
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        rel = masked_logits(m, x, keep)
        q, _ = qc.quant_carry_logits(m, x, "fp32", keep)
    assert type(m.blocks[0]).__name__ == "CarryChannelBlock"
    assert torch.equal(rel, q)


def test_quant_accuracy_fp32_matches_released():
    """`quant_accuracy(fp32)` 与释放版 `accuracy` 在同一任务上同读数。"""
    m = carry_model(gqa=True, T=32, window=8, d=16, layers=2, heads=2)
    task = assoc_recall_batch(4, 32, 2, sym_vocab=8, filler_token=8, cue_token=9,
                              generator=torch.Generator().manual_seed(1))
    keep = window_anchor_keep_mask(32, 8, 0)
    with torch.no_grad():
        a = accuracy(m, task, keep_mask=keep)
        b = qc.quant_accuracy(m, task, "fp32", keep_mask=keep)
    assert a["n_hits"] == b["n_hits"] and a["n"] == b["n"]


# ③ 量化活性：真的改了数值与 logits --------------------------------------------

def test_int8_changes_values_and_logits():
    m = carry_model(gqa=True)
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    acc = qc._new_stats()
    with torch.no_grad():
        fp32, _ = qc.quant_carry_logits(m, x, "fp32", keep)
        fp16, _ = qc.quant_carry_logits(m, x, "fp16", keep)
        i8, _ = qc.quant_carry_logits(m, x, "int8", keep, acc)
    st = qc.quant_liveness_of(acc)
    # 量化确实改了数值（活性量非零）；且 int8 的 logits 与 fp32/fp16 不同
    assert st is not None and st["changed_frac"] > 0 and st["rel_err"] > 0
    assert not torch.equal(i8, fp32)
    assert not torch.equal(i8, fp16)
    # int8 引入的误差应不小于 fp16（更粗的精度）
    assert (i8 - fp32).abs().max() >= (fp16 - fp32).abs().max()


def test_carry_channel_read_is_wired_in_quant_forward():
    """量化前向确实走了 carry 通道：抬高读出增益后 read_share > 0。"""
    m = carry_model(gqa=True)
    x = _tokens()
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        _, tele = qc.quant_carry_logits(m, x, "int8", keep)
    assert tele["read_share"] > 0


# ④ 拒绝路径：非 carry / 非目标块类 / 未知档 -----------------------------------

def test_quant_carry_rejects_stock_and_bad_mode():
    torch.manual_seed(0)
    stock = DepthARModel(DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=12,
                                       dense=True), vocab_size=10).eval()
    x = _tokens()
    try:
        qc.quant_carry_logits(stock, x, "fp32")          # 无 carry ⇒ 拒
        raise AssertionError("应拒绝无 carry 的 stock 模型")
    except ValueError:
        pass
    m = carry_model(gqa=True)
    try:
        qc.quant_carry_logits(m, x, "int4")              # 未知档 ⇒ 拒
        raise AssertionError("应拒绝未知精度档 int4")
    except ValueError:
        pass


def test_quant_carry_rejects_gqa_only_block():
    """只有 GQA（无 carry）的 `GQABlock` 模型被拒（本模块只定义在带 carry 的块上）。"""
    torch.manual_seed(0)
    m = DepthARModel(DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=12,
                                   dense=True, n_kv_heads=1), vocab_size=10).eval()
    try:
        qc.quant_carry_logits(m, _tokens(), "fp32")
        raise AssertionError("应拒绝 GQABlock（无 carry）")
    except ValueError:
        pass


# ⑤ 窗口非退化（§17.16）与分量分解 ---------------------------------------------

def test_window_mask_nondegenerate_at_T256():
    """Z3 的 native 可见性（window16 @ T256）不是恒等算子（§17.16）。"""
    live = window_mask_liveness(256, 16, 0)
    assert live["alive"] is True
    assert 0.0 < live["visible_frac"] < 1.0


def test_compress_increment_signs():
    """分量分解：三项符号/数值 = 对应两行准确率差（pp）。"""
    acc = {"carry_mha_fp16": 0.10, "carry_gqa_fp16": 0.12, "carry_gqa_int8": 0.11}
    inc = ca.compress_increment(acc)
    assert inc["gqa_vs_mha_pp"] == 2.0
    assert inc["int8_vs_fp16_pp"] == -1.0
    assert inc["net_pp"] == 1.0
