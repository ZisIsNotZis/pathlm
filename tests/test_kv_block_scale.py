"""2632 · KV 量化标尺方案（块 / KIVI / 逐值对照）的可失败单测。

契约：`axes/R-references/R1-quantization/experiments/2632-kv-block-scale/`
（`config.json` 判据线 + `README.md`）。
语义层：`tools/evalkit/kv_quant.py`（标尺方案）、
`.../2632-kv-block-scale/block_arms.py`（前向）、`.../scale_caliber.py`（字节账）。

钉住五件事：

1. **逐值标尺不回归**：`quantize_kv(x, mode)` 与 `scheme="per_value"` 逐位一致
   （既有 `2619/2620/2621/2631` 的价签不受影响）。
2. **块标尺（沿 head_dim）既摊薄又前缀确定**：每值标尺 = `2/G` B；改未来 token / 改同批
   邻居都不改过去位置的反量化值与前向 logits；`G=32`（= head_dim）时 int8 = **1.0625 B/值**。
3. **KIVI 式**：V per-token 前缀确定；K per-channel（沿 token 分组）**不**前缀确定
   （块内过去值会被同组后续位置改写）——如实钉住这条 trade-off。
4. **块标尺的块必须整除它所在的轴**：`G` 不整除 head_dim（如 32 维上取 64）直接报错。
5. **部署字节账**：`scale_caliber.deployment_bytes` 算出的值字节必须与该臂已记录的
   `bytes_per_position` 逐位一致；`mode` 与 `value_bytes` 标签不符直接报错。

破坏上述语义的变异见实验目录 `mutate.py`（`python3 mutate.py`）。
"""

import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "axes", "R-references", "R1-quantization",
                   "experiments", "2632-kv-block-scale")
sys.path.insert(0, EXP)

import block_arms as ba                                                # noqa: E402
import scale_caliber as sc                                             # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel               # noqa: E402
from tools.evalkit.kv_quant import (SCALE_SCHEMES, kv_quant_logits,   # noqa: E402
                                    quantize_kv, release_logits,
                                    scheme_scale_bytes_per_value)


def tiny_model(d=24, L=2, H=3, T=32, vocab=12):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=d, n_layers=L, n_heads=H, seq_len=T, mlp_mult=4, dense=True)
    return DepthARModel(cfg, vocab_size=vocab)


def _cfg():
    return {"model": {"n_layers": 12, "n_heads": 8, "d_model": 256, "seq_len": 256}}


def _point(arm="int8", nkv=8, kept_over_T=1.0, value_bytes=1.0):
    return {"arm": arm, "bytes_per_position": 2 * 12 * nkv * 32 * value_bytes * kept_over_T,
            "factor_n_kv_heads_eff": nkv, "factor_positions_kept_over_T": kept_over_T}


# ---------------------------------------------------------- 1. 逐值标尺不回归

def test_per_value_scheme_default_is_bitwise_unchanged():
    """不带 scheme 的调用 == scheme='per_value'（向后兼容，既有价签不受影响）。"""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 16, 8) * 2
    for mode in ("int8", "int4"):
        a, sa = quantize_kv(x, mode)
        b, sb = quantize_kv(x, mode, scheme="per_value")
        assert torch.equal(a, b) and sa == sb


def test_scalar_arg_positional_compat():
    """旧位置参数 (x, mode) 仍可用（2619/2620/2621 的调用点）。"""
    x = torch.randn(1, 2, 8, 4)
    dq, _ = quantize_kv(x, "int8")
    assert dq.shape == x.shape


# ---------------------------------------------------------- 2. 块标尺：摊薄 + 因果

def test_block_scale_bytes_per_value_is_two_over_group():
    assert scheme_scale_bytes_per_value("int8", "block", 32, group=32) == 0.0625
    assert scheme_scale_bytes_per_value("int8", "block", 32, group=16) == 0.125
    assert scheme_scale_bytes_per_value("int4", "block", 32, group=8) == 0.25
    # int8 净每值字节：1 + 2/G
    assert 1 + scheme_scale_bytes_per_value("int8", "block", 32, 32) == 1.0625
    # 逐值标尺的最贵价签
    assert scheme_scale_bytes_per_value("int8", "per_value", 32) == 2.0


def test_block_requires_group_dividing_head_dim():
    """head_dim = 32 时 G = 64 不整除 ⇒ 报错（沿 head_dim 的块做不到跨 token 又前缀确定）。"""
    x = torch.randn(1, 2, 8, 32)
    with pytest.raises(ValueError):
        quantize_kv(x, "int8", "block", group=64)
    with pytest.raises(ValueError):
        scheme_scale_bytes_per_value("int8", "block", 32, group=64)


@pytest.mark.parametrize("group", [8, 16, 32])
def test_block_error_bound_is_half_of_the_block_scale(group):
    """|x̂ − x| ≤ 0.5 · 块标尺（块标尺 = G 个相邻 head_dim 值的 amax / qmax）。"""
    torch.manual_seed(2)
    x = torch.randn(2, 3, 8, 32) * 3
    dq, _ = quantize_kv(x, "int4", "block", group=group)
    xr = x.reshape(2, 3, 8, 32 // group, group)
    scale = xr.abs().amax(dim=-1, keepdim=True) / 7
    err = (dq - x).reshape(2, 3, 8, 32 // group, group).abs()
    assert torch.all(err <= 0.5 * scale + 1e-6), err.max().item()


def test_make_logits_fn_honours_scheme_prefix_invariance():
    """`make_logits_fn` 真的把 scheme 接进前向：block 保持前缀不变，block_tok 被未来改写。"""
    model = tiny_model()
    toks = torch.randint(0, 12, (2, 32))
    spec = {"kind": "quant", "mode": "int8"}
    fut = toks.clone()
    fut[:, -1] = (fut[:, -1] + 7) % 12
    fn_block = ba.make_logits_fn(model, spec, 32, "cpu", "block", 8, 32)
    assert torch.equal(fn_block(toks)[:, :-1, :], fn_block(fut)[:, :-1, :]), \
        "block 前向应保持前缀不变"
    fn_tok = ba.make_logits_fn(model, spec, 32, "cpu", "block_tok", 8, 32)
    assert not torch.equal(fn_tok(toks)[:, :-1, :], fn_tok(fut)[:, :-1, :]), \
        "block_tok 前向应被未来 token 改写（scheme 必须真的接进前向）"


def test_block_prefix_invariance_future_token_and_peer_request():
    """改未来 token / 改同批邻居：过去位置的反量化值与前向 logits 都不变。"""
    torch.manual_seed(3)
    x = torch.randn(3, 2, 16, 8) * 2
    dq0, _ = quantize_kv(x, "int8", "block", group=8)
    xf = x.clone()
    xf[..., -1, :] = 1000.0
    dqf, _ = quantize_kv(xf, "int8", "block", group=8)
    assert torch.equal(dq0[..., :-1, :], dqf[..., :-1, :]), "未来 token 改了过去的反量化值"
    xp = x.clone()
    xp[1] = 1000.0
    dqp, _ = quantize_kv(xp, "int8", "block", group=8)
    assert torch.equal(dq0[0], dqp[0]), "同批邻居改了本请求的反量化值"

    model = tiny_model()
    toks = torch.randint(0, 12, (3, 32))
    base, _ = ba.masked_kv_quant_logits(model, toks, None, "int8", "block", 8, 32)
    fut = toks.clone()
    fut[:, -1] = (fut[:, -1] + 7) % 12
    got, _ = ba.masked_kv_quant_logits(model, fut, None, "int8", "block", 8, 32)
    assert torch.equal(base[:, :-1, :], got[:, :-1, :]), "block 前向被未来 token 改了"
    peer = toks.clone()
    peer[1] = (peer[1] + 5) % 12
    got2, _ = ba.masked_kv_quant_logits(model, peer, None, "int8", "block", 8, 32)
    assert torch.equal(base[0], got2[0]), "block 前向被同批邻居改了"


# ---------------------------------------------------------- 3. KIVI / block_tok 的 trade-off

def test_kivi_v_per_token_is_prefix_invariant():
    torch.manual_seed(4)
    x = torch.randn(2, 2, 32, 8)
    dq0, _ = quantize_kv(x, "int8", "kivi", kivi_group=16, is_key=False)
    xf = x.clone()
    xf[..., -1, :] = 1000.0
    dqf, _ = quantize_kv(xf, "int8", "kivi", kivi_group=16, is_key=False)
    assert torch.equal(dq0[..., :-1, :], dqf[..., :-1, :])


def test_kivi_k_per_channel_is_not_prefix_invariant():
    """K 沿 token 分组 ⇒ 改同组内未来位置会改写本组**过去位置**的读数（如实钉住）。"""
    torch.manual_seed(5)
    x = torch.randn(1, 2, 32, 8)
    dq0, _ = quantize_kv(x, "int8", "kivi", kivi_group=16, is_key=True)
    xf = x.clone()
    xf[..., 20, :] = 1000.0            # 落在第二个 16-token 块里
    dqf, _ = quantize_kv(xf, "int8", "kivi", kivi_group=16, is_key=True)
    assert torch.equal(dq0[..., :16, :], dqf[..., :16, :])           # 第一个块不受影响
    # 关键：同块内的**过去**位置（16..19，早于 20）也被改写（per-token 就不会）
    assert not torch.equal(dq0[..., 16:20, :], dqf[..., 16:20, :])
    assert not torch.equal(dq0[..., 20, :], dqf[..., 20, :])


def test_block_tok_is_not_prefix_invariant():
    torch.manual_seed(6)
    x = torch.randn(1, 2, 32, 8)
    dq0, _ = quantize_kv(x, "int8", "block_tok", group=16)
    xf = x.clone()
    xf[..., 20, :] = 1000.0
    dqf, _ = quantize_kv(xf, "int8", "block_tok", group=16)
    assert not torch.equal(dq0[..., 16:, :], dqf[..., 16:, :])


def test_kivi_needs_is_key():
    x = torch.randn(1, 2, 16, 8)
    with pytest.raises(ValueError):
        quantize_kv(x, "int8", "kivi")


# ---------------------------------------------------------- 4. 前向实现等值 + 活性

def test_scheme_per_value_forward_equals_kv_quant_logits():
    """block_arms 的 per_value 前向 == kv_quant.kv_quant_logits（同一件仪器）。"""
    model = tiny_model()
    toks = torch.randint(0, 12, (2, 32))
    got, _ = ba.masked_kv_quant_logits(model, toks, None, "int8", "per_value", 32, 32)
    ref, _ = kv_quant_logits(model, toks, "int8")
    assert torch.equal(got, ref)


def test_fp32_forward_bitwise_identity():
    model = tiny_model()
    toks = torch.randint(0, 12, (2, 32))
    got, _ = ba.masked_kv_quant_logits(model, toks, None, "fp32", "per_value", 32, 32)
    ref = release_logits(model, toks)
    assert torch.equal(got, ref)


def test_block_quantization_is_live_and_changes_logits():
    model = tiny_model()
    toks = torch.randint(0, 12, (2, 32))
    fp32, _ = ba.masked_kv_quant_logits(model, toks, None, "fp32", "block", 8, 32)
    int4, st = ba.masked_kv_quant_logits(model, toks, None, "int4", "block", 8, 32)
    assert st["n_values"] > 0 and st["max_abs_err"] > 0
    assert not torch.equal(fp32, int4)
    # int4 比 int8 糙
    _, s8 = ba.masked_kv_quant_logits(model, toks, None, "int8", "block", 8, 32)
    assert s8["sum_abs_err"] < st["sum_abs_err"]


# ---------------------------------------------------------- 5. 部署字节账

def test_deployment_bytes_reconcile_and_shrink():
    cfg, spec, point = _cfg(), {"mode": "int8", "value_bytes": 1.0}, _point()
    dep = sc.deployment_bytes(point, spec, cfg, "block", group=32)
    assert dep["value_bytes_per_position"] == 6144.0          # 2·12·8·32·1·1
    assert dep["scale_bytes_per_position"] == 6144.0 * 0.0625
    assert dep["bytes_per_position"] == pytest.approx(6144.0 * 1.0625)
    assert dep["bytes_per_position"] / 12288.0 == pytest.approx(0.53125)  # 省 46.9%
    # 逐值标尺（对照）反而更费：3 B/值
    a = sc.deployment_bytes(point, spec, cfg, "per_value")
    assert a["bytes_per_position"] == 6144.0 * 3.0
    assert a["bytes_per_position"] > dep["bytes_per_position"]


def test_deployment_bytes_reject_mode_value_bytes_mismatch():
    cfg = _cfg()
    with pytest.raises(ValueError):
        sc.deployment_bytes(_point(), {"mode": "int8", "value_bytes": 2.0}, cfg, "block")


def test_deployment_bytes_rejects_value_caliber_mismatch():
    """点的旧值字节与公式对不上（口径漂移）时必须报错，不得继续出数。"""
    cfg = _cfg()
    bad = _point()
    bad["bytes_per_position"] = 999.0            # 与公式不符
    with pytest.raises(ValueError):
        sc.deployment_bytes(bad, {"mode": "int8", "value_bytes": 1.0}, cfg, "block")


def test_all_schemes_have_bytes_defined():
    cfg, spec, point = _cfg(), {"mode": "int4", "value_bytes": 0.5}, _point(value_bytes=0.5)
    for scheme in SCALE_SCHEMES:
        dep = sc.deployment_bytes(point, spec, cfg, scheme, group=32)
        assert dep["bytes_per_position"] > dep["value_bytes_per_position"]


def test_quantizer_scale_count_equals_values_over_group():
    """P2-4：量化器**实际产出**的标尺个数 == `values / G`（与解析字节账对账）。

    block 沿 head_dim：标尺数 = `n_values / group`；kivi 的 V 侧 = `n_values / head_dim`。
    """
    torch.manual_seed(7)
    x = torch.randn(2, 3, 32, 32) * 2          # B=2,H=3,T=32,head_dim=32
    for group in (8, 16, 32):
        st = quantize_kv(x, "int8", "block", group=group)[1]
        assert st["n_scales"] == sc.analytic_scale_count(st["n_values"], group)
        assert st["n_scales"] * group == st["n_values"]
    # kivi 的 V 侧：整条 head_dim 共享一把 ⇒ 标尺数 = values/head_dim
    st_v = quantize_kv(x, "int8", "kivi", kivi_group=32, is_key=False)[1]
    assert st_v["n_scales"] * 32 == st_v["n_values"]
    # per_value（对照）：逐值一把 ⇒ 标尺数 == 值数
    st_pv = quantize_kv(x, "int8")[1]
    assert st_pv["n_scales"] == st_pv["n_values"]


def test_fp8_reference_is_cheaper_than_int8_block():
    """P2-5：fp8 解析为 1 B/值、无标尺 ⇒ 比 int8+块标尺（1.0625 B/值）更省。"""
    cfg = _cfg()
    fp8 = sc.fp8_reference(cfg)
    assert fp8["bytes_per_position"] == 6144.0            # 2·12·8·32·1.0
    assert fp8["ratio_vs_fp16"] == pytest.approx(0.5)
    assert fp8["quality_measured"] is False
    int8_block = sc.deployment_bytes(_point(), {"mode": "int8", "value_bytes": 1.0},
                                     cfg, "block", 32)
    assert int8_block["bytes_per_position"] > fp8["bytes_per_position"]


def test_staging_reports_worst_and_mean_calibers():
    """P2-2：staging 两套口径均给（持整块最坏 vs 平均半块），不得只给一个。"""
    cfg = _cfg()
    st = sc.staging_bytes_per_request(cfg, _point(), group=32)
    # 每位置 2·12·8·32 = 6144 个 fp16 值 × 2 B = 12288 B/位置
    assert st["worst_bytes_per_request"] == 32 * 12288
    assert st["mean_bytes_per_request"] == pytest.approx(15.5 * 12288)
    assert st["worst_bytes_per_request"] > st["mean_bytes_per_request"]


# ---------------------------------------------------------- 6. 落盘证据（若存在）

def test_recorded_result_prefix_invariance_holds_for_primary_scheme():
    """`results.json` 若存在：主臂（block G=32）的 27 点前缀不变性必须全 True。"""
    path = os.path.join(EXP, "results.json")
    if not os.path.exists(path):
        pytest.skip("results.json 尚未生成（运行 run.py 后生效）")
    import json
    rec = json.load(open(path))
    bad = [p["arm"] for p in rec["points"]
           if cfg_primary(rec["config"], p["arm"])
           and not (p["prefix_invariance"]["future_token_past_logits_equal"]
                    and p["prefix_invariance"]["peer_request_logits_equal"])]
    assert bad == [], f"主臂前缀不变性不成立：{bad}"


def cfg_primary(cfg: dict, arm: str) -> bool:
    return cfg["arms"][arm].get("mode") in ("int8", "int4")
