"""KV 量化参照线的可失败单测（ticket 31 契约 K）。

Run: `python3 -m pytest tests/ -q`

钉住四件事：
1. **量化-反量化往返有上界**：int 档对称 per-channel，`|x̂ − x| ≤ 0.5·scale`
   （scale = 该通道 amax / qmax）；fp16 档相对上界 `|x̂ − x| ≤ 2⁻¹⁰·|x|`；
   fp32 档逐位恒等；
2. **per-channel 语义正确**：不同 `(head, channel)` 各自用自己的 scale
   （大通道不会吃掉小通道的分辨率）；
3. **实现等值**：`fp32` 档前向与释放版 dense 前向**逐位相等**（§9）；
4. **干预活性**：int8/int4 真的改变数值、改变 logits，且 int4 比 int8 差
   （bit 越少越糙）——不是「只改了个标签」。

破坏这些语义的变异见实验目录 `mutate.py`。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.depth_ar import DepthARConfig, DepthARModel   # noqa: E402
from tools.evalkit.kv_quant import (KV_MODES, cache_reduction,   # noqa: E402
                                    kv_cache_footprint, kv_quant_logits,
                                    quantize_kv, release_logits,
                                    summarize_stats)


def tiny_model(d=24, L=2, H=3, T=12, vocab=12):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=d, n_layers=L, n_heads=H, seq_len=T, mlp_mult=4,
                        dense=True)
    return DepthARModel(cfg, vocab_size=vocab)


def per_channel_scale(x, qmax):
    """独立算出参考 scale：[..., H, T, head_dim] 每 (head, channel) 一个，沿
    batch 与 token 取 amax。"""
    reduce = tuple(i for i in range(x.dim())
                   if i not in (x.dim() - 1, x.dim() - 3))
    return (x.abs().amax(dim=reduce, keepdim=True) / qmax).clamp_min(1e-12)


# --------------------------------------------------------------- 1. 往返误差上界


def test_fp32_mode_is_bitwise_identity():
    x = torch.randn(2, 3, 5, 4)
    dq, st = quantize_kv(x, "fp32")
    assert dq is x
    assert int(st["n_values"]) == 0 and st["n_changed"] == 0


@pytest.mark.parametrize("mode,qmax", [("int8", 127), ("int4", 7)])
def test_int_roundtrip_error_within_half_scale(mode, qmax):
    """对称 per-channel 取整：逐元素误差 ≤ 0.5·scale，且 scale = amax/qmax。"""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 6, 4) * 3
    dq, _ = quantize_kv(x, mode)
    scale = per_channel_scale(x, qmax)
    err = (dq - x).abs()
    assert torch.all(err <= 0.5 * scale + 1e-6), err.max().item()
    # 独立重算参考实现，逐位相等
    q = torch.round(x / scale).clamp(-qmax, qmax)
    assert torch.allclose(dq, q * scale, atol=1e-6)
    # amax 元素应被精确用到端点附近（未发生整体缩小 => qmax 正确）
    assert float((x / scale).abs().max()) <= qmax + 1e-6


def test_fp16_relative_bound_and_actually_changes_values():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 6, 4) * 3
    dq, st = quantize_kv(x, "fp16")
    err = (dq - x).abs()
    # fp16 有 10 bit 尾数 ⇒ 相对取整误差 ≤ 2^-11；用 2^-10 留安全余量。
    assert torch.all(err <= x.abs() * 2.0 ** -10 + 1e-7)
    # 活性：确实降了精度（否则「fp16 基线」是假标签）
    assert int((dq != x).sum()) > 0
    assert st["n_changed"] > 0 and st["max_abs_err"] > 0


# --------------------------------------------------------------- 2. per-channel 语义


def test_per_channel_scale_is_per_head_channel():
    """构造「大通道 + 小通道」：per-channel 下小通道仍按**自己**的 scale 量化；
    若误用全局 per-tensor scale，小通道会被压成 0，误差远超 0.5·自身 scale。"""
    torch.manual_seed(3)
    x = torch.empty(1, 2, 5, 4)
    x[..., 0] = 100.0 + torch.randn(1, 2, 5)        # 通道 0：幅度 ~100
    x[..., 1:] = 0.001 * torch.randn(1, 2, 5, 3)    # 通道 1..3：幅度 ~0.001
    dq, _ = quantize_kv(x, "int4")
    scale = per_channel_scale(x, 7)
    err = (dq - x).abs()
    assert torch.all(err <= 0.5 * scale + 1e-9)
    # 小通道的相对误差应有界（远小于全局 amax/qmax 会造成的量级）
    small = slice(1, 4)
    assert err[..., small].max() <= 0.5 * scale[..., small].max() + 1e-9
    assert scale[..., small].max() < 1e-3


def test_int4_error_exceeds_int8_error():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 6, 4) * 2
    d8, s8 = quantize_kv(x, "int8")
    d4, s4 = quantize_kv(x, "int4")
    s8, s4 = summarize_stats(s8), summarize_stats(s4)
    assert s4["max_abs_err"] > s8["max_abs_err"]
    assert s4["rel_err"] > s8["rel_err"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        quantize_kv(torch.randn(1, 2, 4, 4), "int3")


# --------------------------------------------------------------- 3. 实现等值（§9）


def test_fp32_forward_bitwise_equals_release_forward():
    """kv_quant_logits(fp32) ≡ 释放版 dense 前向（最终深度 logits）逐位相等。"""
    model = tiny_model()
    torch.manual_seed(5)
    x = torch.randint(0, 12, (2, 12))
    got, _ = kv_quant_logits(model, x, "fp32")
    ref = release_logits(model, x)
    assert torch.equal(got, ref), (got - ref).abs().max().item()


# --------------------------------------------------------------- 4. 干预活性


def test_quantization_is_live_changes_logits():
    model = tiny_model()
    torch.manual_seed(6)
    x = torch.randint(0, 12, (2, 12))
    fp32, _ = kv_quant_logits(model, x, "fp32")
    int4, st4 = kv_quant_logits(model, x, "int4")
    assert st4["n_values"] > 0 and st4["max_abs_err"] > 0
    assert not torch.equal(fp32, int4)
    assert float((fp32 - int4).abs().max()) > 0


def test_kv_quant_rejects_non_stock_block():
    """内容 top-k（TopKBlock）等自定义层会让单因素失效——必须拒绝。"""
    from pathlm.depth_ar import TopKBlock          # noqa: E402
    model = tiny_model()
    cfg = DepthARConfig(d_model=24, n_layers=2, n_heads=3, seq_len=12,
                        mlp_mult=4, dense=True, attn_sparse_k=4)
    torch.manual_seed(7)
    sparse = DepthARModel(cfg, vocab_size=12)
    assert isinstance(sparse.blocks[0], TopKBlock)
    with pytest.raises(ValueError):
        kv_quant_logits(sparse, torch.randint(0, 12, (1, 12)), "int8")


# --------------------------------------------------------------- 5. 内存口径


def test_kv_cache_footprint_reduction():
    cfg = DepthARConfig(d_model=256, n_layers=12, n_heads=8, seq_len=256,
                        mlp_mult=6, dense=True)
    base = kv_cache_footprint(cfg, "fp16", seq_lens=(256,))
    i8 = kv_cache_footprint(cfg, "int8", seq_lens=(256,))
    i4 = kv_cache_footprint(cfg, "int4", seq_lens=(256,))
    assert i8["per_position_bytes"] == base["per_position_bytes"] / 2
    assert i4["per_position_bytes"] == base["per_position_bytes"] / 4
    assert cache_reduction(base, i8)["reduction_frac"] == pytest.approx(0.5)
    assert cache_reduction(base, i4)["reduction_frac"] == pytest.approx(0.75)
    # 与手算一致：2(K,V)×12 层×8 头×32 维×2 字节 = 12288 B/位置
    assert base["per_position_bytes"] == 12288


def test_summarize_stats_fields_are_activity_quantities():
    acc = {"n_values": 8, "sum_abs_err": 1.6, "max_abs_err": 0.4,
           "sum_abs_x": 8.0, "n_changed": 4, "n_clipped": 0,
           "scale_min": 0.1, "scale_max": 0.2}
    s = summarize_stats(acc)
    assert s["mean_abs_err"] == pytest.approx(0.2)
    assert s["rel_err"] == pytest.approx(0.2)
    assert s["changed_frac"] == pytest.approx(0.5)
