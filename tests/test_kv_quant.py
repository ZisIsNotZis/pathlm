"""KV 量化参照线的可失败单测（ticket 31 契约 K）。

Run: `python3 -m pytest tests/ -q`

钉住五件事：
1. **量化-反量化往返有上界**：int 档对称 per-channel，`|x̂ − x| ≤ 0.5·scale`
   （scale = 该通道的因果 running max / qmax）；fp16 档相对上界 `|x̂ − x| ≤ 2⁻¹⁰·|x|`；
   fp32 档逐位恒等；
2. **per-channel 语义正确**：不同 `(head, channel)` 各自用自己的 scale
   （大通道不会吃掉小通道的分辨率）；
3. **前缀不变性（F2，修复 A2）**：改未来 token 不改过去位置的反量化值/过去 logits；
   改同批其他请求不改本请求结果（旧的「全序列 batch+token amax」实现三条全红）；
4. **实现等值**：`fp32` 档前向与释放版 dense 前向**逐位相等**（§9）；
5. **干预活性**：int8/int4 真的改变数值、改变 logits，且 int4 比 int8 差
   （bit 越少越糙）——不是「只改了个标签」。

破坏这些语义的变异见实验目录 `mutate.py`。
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
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
    batch 与 token 取 amax（**旧的非因果口径**，仅 `test_prefix_invariance_*` 用来
    展示旧实现在改未来 token / 同批邻居时会变，不再作为生产口径）。"""
    reduce = tuple(i for i in range(x.dim())
                   if i not in (x.dim() - 1, x.dim() - 3))
    return (x.abs().amax(dim=reduce, keepdim=True) / qmax).clamp_min(1e-12)


def causal_scale(x, qmax):
    """因果/流式参考 scale：位置 t 只用**同一请求 ≤ t** 的 token（running max），
    batch 轴不参与归约。这是 `quantize_kv` 的生产口径（修复 A2）。"""
    return (x.abs().cummax(dim=x.dim() - 2).values / qmax).clamp_min(1e-12)


# ------------------------------------------------- 0. 前缀不变性（F2 / 修复 A2）
#
# 旧实现沿 batch+token 取全序列 amax ⇒ 过去位置的反量化值随**未来 token**或**同批
# 其他请求**改变（`is_causal=True` 挡不住）。下面三条测试对旧实现应红、对因果实现应绿。


def test_prefix_invariance_future_token_does_not_change_past_values():
    """① 改未来 token 不改变过去位置的反量化值。

    直接复刻复核报告 A2 的反例：过去值小、未来值 1000；旧实现下过去的 scale 被
    未来 amax 改变（过去值被重新定量），因果实现下过去 scale 不含未来信息。"""
    torch.manual_seed(11)
    x = torch.ones(1, 2, 8, 4)                      # 过去位置幅度 0.5/1
    x[..., 3:, :] = 0.5
    dq0, _ = quantize_kv(x, "int4")
    x2 = x.clone()
    x2[..., -1, :] = 1000.0                         # 只改最后一个（未来）位置
    dq1, _ = quantize_kv(x2, "int4")
    assert torch.equal(dq0[..., :-1, :], dq1[..., :-1, :]), \
        "过去位置的反量化值被未来 token 改变了（因果性缺陷）"
    # 旧口径（全序列 amax）就会变——这就是被修复的那条通路
    s_old, s_old2 = per_channel_scale(x, 7), per_channel_scale(x2, 7)
    assert not torch.equal(s_old, s_old2)
    # 过去的反量化值 == 「只拿前缀序列算」的结果（逐位）
    dq_prefix, _ = quantize_kv(x[..., :-1, :], "int4")
    assert torch.equal(dq0[..., :-1, :], dq_prefix)


def test_prefix_invariance_other_requests_do_not_change_this_request():
    """② 改同批其他请求不改变本请求的结果（逐请求，batch 轴不参与归约）。"""
    torch.manual_seed(12)
    x = torch.randn(3, 2, 6, 4)
    dq0, _ = quantize_kv(x, "int4")
    x2 = x.clone()
    x2[1] = 1000.0                                  # 只改同批的第 2 个请求
    x2[2] = -500.0
    dq1, _ = quantize_kv(x2, "int4")
    assert torch.equal(dq0[0], dq1[0]), "本请求的反量化值被同批其他请求改变了（跨请求耦合）"
    assert torch.equal(dq0[1], dq1[1]) is False     # 被改的那个请求自己当然会变


def test_prefix_invariance_past_logits_unchanged_by_future_token_and_batch_peer():
    """①/② 的 logits 版：过去位置的 logits 不随未来 token、也不随同批邻居改变。"""
    model = tiny_model()
    torch.manual_seed(13)
    x = torch.randint(0, 12, (3, 12))
    base, _ = kv_quant_logits(model, x, "int4")
    future = x.clone()
    future[:, -1] = (future[:, -1] + 7) % 12        # 改未来（最后）token
    got_future, _ = kv_quant_logits(model, future, "int4")
    assert torch.equal(base[:, :-1, :], got_future[:, :-1, :]), \
        "过去位置的 logits 被未来 token 改变了"
    peer = x.clone()
    peer[1] = (peer[1] + 5) % 12                    # 改同批第 2 个请求
    peer[2] = (peer[2] + 3) % 12
    got_peer, _ = kv_quant_logits(model, peer, "int4")
    assert torch.equal(base[0], got_peer[0]), "本请求 logits 被同批其他请求改变了"


# --------------------------------------------------------------- 1. 往返误差上界


def test_fp32_mode_is_bitwise_identity():
    x = torch.randn(2, 3, 5, 4)
    dq, st = quantize_kv(x, "fp32")
    assert dq is x
    assert int(st["n_values"]) == 0 and st["n_changed"] == 0


@pytest.mark.parametrize("mode,qmax", [("int8", 127), ("int4", 7)])
def test_int_roundtrip_error_within_half_scale(mode, qmax):
    """对称 per-channel 取整：逐元素误差 ≤ 0.5·scale，且 scale 取因果 running max。"""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 6, 4) * 3
    dq, _ = quantize_kv(x, mode)
    scale = causal_scale(x, qmax)
    err = (dq - x).abs()
    assert torch.all(err <= 0.5 * scale + 1e-6), err.max().item()
    # 独立重算参考实现，逐位相等
    q = torch.round(x / scale).clamp(-qmax, qmax)
    assert torch.allclose(dq, q * scale, atol=1e-6)
    # amax 元素应被精确用到端点附近（未发生整体缩小 => qmax 正确）
    assert float((x / scale).abs().max()) <= qmax * (1 + 1e-5) + 1e-6
    # 因果性：位置 t 的 scale 只由 ≤ t 的历史决定（非递减 running max）
    assert torch.all(scale[..., 1:, :] >= scale[..., :-1, :] - 1e-12)
    assert torch.allclose(scale[..., 0, :],
                          x[..., 0, :].abs().clamp_min(1e-12) / qmax, atol=1e-9)


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
    scale = causal_scale(x, 7)
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


# ------------------------------------------------- 6. 组合入口同源（F1）

#: 窗口 / 锚点 / 头共享等「组合入口」所在的模块（2619 的语义 + 2620/2621 的组合臂）。
COMBO_MODULES = (
    "axes/5-quant/5.1-kv-quant/experiments/2620-kv-budget-gaps/gaps_arms.py",
    "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/rest_arms.py",
)


def test_combo_entries_route_through_the_same_causal_quantizer():
    """组合入口必须复用 `quantize_kv`（同一个因果实现），不得残留全序列 amax 的旧路径，
    也不得自己另起一套量化实现（F1：同文件的组合入口一并走同一实现）。"""
    for rel in COMBO_MODULES:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "quantize_kv" in src, f"{rel} 未复用 quantize_kv（可能自带旧量化路径）"
        assert "amax(" not in src, f"{rel} 自带 amax 归约（旧的非因果 scale 路径残留）"
        assert "cummax" not in src, f"{rel} 自己另起了一套量化实现（应复用 quantize_kv）"
