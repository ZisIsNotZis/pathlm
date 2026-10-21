"""2685 的单测（纯 CPU）：字节账口径 + `carry_quant.py` 的两条忠实性/单因素断言。

覆盖：
1. `config.json::analytic` 的解析字节账（G=32、位宽、状态值数）自己算得对；
2. `carry_scan_quant(state="none")` 与 `pathlm.depth_ar.carry_scan_chunked` 逐位相同
   ——「carry fp16 格走新代码」这条复现门在算子层的证据；
3. 状态量化真的改了数值（不是死干预），且往返误差落在 `0.5·scale` 的解析上界内；
4. `quant_carry_logits(none, none)` 与 `retention.masked_logits` 逐位相同（整模型层）；
5. 状态 int8 档改变 logits（活性量，SCHEMA §10）；
6. `num_state_values` 在逐层 / 单共享槽两种模式下都对（部署字节账的分母）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import carry_quant as cq                                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel, carry_scan_chunked,  # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.kv_quant import quantize_kv                     # noqa: E402
from tools.evalkit.retention import masked_logits                  # noqa: E402

EXP = Path(HERE)
CFG = json.loads((EXP / "config.json").read_text())
NONE = {"mode": "none"}


def _accum() -> dict:
    return {"n_values": 0, "n_scales": 0, "n_changed": 0, "n_clipped": 0,
            "sum_abs_err": 0.0, "sum_abs_x": 0.0, "max_abs_err": 0.0,
            "scale_min": None, "scale_max": None}


def _tiny_model(per_layer: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=8, n_layers=2, n_heads=2, seq_len=8, mlp_mult=1,
                        dense=True, topology="sequential", carry_rule="delta",
                        carry_per_layer=per_layer, carry_chunk=4)
    model = DepthARModel(cfg, vocab_size=32).eval()
    # 读出增益是单边零初始化（训练前 carry 通道不起作用）⇒ 单测里先把它置成非零，
    # 否则「状态量化改变 logits」这类活性断言在随机初始化模型上必然是假的。
    with torch.no_grad():
        for blk in model.blocks:
            blk.carry_scale.copy_(
                torch.linspace(-0.5, 0.5, blk.carry_scale.numel()))
    return model


def test_analytic_byte_ledger():
    an = CFG["analytic"]
    bpv = an["bytes_per_value"]
    assert an["group"] == 32
    assert bpv["fp16"] == 2.0 and bpv["int8"] == 1.0 + 2 / an["group"]
    assert bpv["int4"] == 0.5 + 2 / an["group"]
    vpp = an["kv_values_per_position_per_key"]
    assert vpp == CFG["model"]["n_layers"] * 2 * CFG["model"]["d_model"]
    assert vpp * bpv["fp16"] == 12288
    assert vpp * bpv["int8"] == 6528
    assert vpp * bpv["int4"] == 3456
    # 块标尺必须整除 head_dim（真配置里 G=32、head_dim = d_model / n_heads = 32）
    assert (CFG["model"]["d_model"] // CFG["model"]["n_heads"]) % an["group"] == 0
    # carry 的同档字节：窗口 16 的 KV + 逐层状态 98,304 值
    assert an["state_values"]["carrypl16_s0"] == (
        CFG["model"]["n_layers"] * (CFG["model"]["d_model"] // CFG["model"]["n_heads"]) ** 2
        * CFG["model"]["n_heads"])
    kv8, st8 = 16 * 6528, an["state_values"]["carrypl16_s0"] * bpv["int8"]
    assert kv8 + st8 == 208896
    assert an["state_values"]["carrypl16_s0"] * bpv["fp16"] == 196608
    kv4, st4 = 16 * 3456, an["state_values"]["carrypl16_s0"] * bpv["int4"]
    assert kv4 + st4 == 110592
    # 与纯窗 int8 的 208,896 B（k=32）天然同字节、与 its int4 的 110,592 B 亦同字节
    assert 32 * 6528 == 208896 and 32 * 3456 == 110592


def test_scan_without_state_quant_matches_release_scan():
    torch.manual_seed(1)
    B, H, T, dh = 2, 2, 8, 4
    q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))
    beta = torch.rand(B, H, T)
    want, _ = carry_scan_chunked(q, k, v, beta, rule="delta", chunk=4, decay=1.0)
    got = cq.carry_scan_quant(q, k, v, beta, "delta", 4, 1.0, NONE, _accum())
    assert torch.equal(got, want)


def test_state_quant_changes_values_and_respects_error_bound():
    torch.manual_seed(2)
    S = torch.randn(2, 2, 8, 8) * 3.0
    dq, _ = quantize_kv(S, "int8", "block", 8, 8, None)
    assert not torch.equal(dq, S)                     # 不是死干预
    # 块标尺 G=2（整除 head_dim=8）时检查解析上界：|x̂ − x| ≤ 0.5 · scale
    dq2, _ = quantize_kv(S, "int8", "block", 2, 2, None)
    amp = S.reshape(*S.shape[:-1], S.shape[-1] // 2, 2).abs().amax(-1, keepdim=True)
    bound = 0.5 * (amp / 127.0)
    err = (dq2 - S).abs().reshape(*S.shape[:-1], S.shape[-1] // 2, 2)
    assert (err <= bound + 1e-6).all()
    assert cq.roundtrip(S, {"mode": "int4", "scheme": "block", "group": 8}, None,
                        _accum(), "state").shape == S.shape


def test_quant_carry_logits_identity_matches_masked_logits():
    model = _tiny_model(per_layer=True)
    tokens = torch.randint(0, 32, (2, 8))
    km = window_anchor_keep_mask(8, 4, 0, device="cpu")
    want = masked_logits(model, tokens, km)
    got, acc = cq.quant_carry_logits(model, tokens, km, NONE, NONE)
    assert torch.equal(got, want), (got - want).abs().max().item()
    assert acc["n_values"] == 0                       # 全不量化 ⇒ 没有量化活性量


def test_state_int8_tier_is_active():
    model = _tiny_model(per_layer=True)
    tokens = torch.randint(0, 32, (2, 8))
    km = window_anchor_keep_mask(8, 4, 0, device="cpu")
    base, _ = cq.quant_carry_logits(model, tokens, km, NONE, NONE)
    st8 = {"mode": "int8", "scheme": "block", "group": 4}
    got, acc = cq.quant_carry_logits(model, tokens, km, NONE, st8)
    assert acc["n_values"] > 0 and acc["n_scales"] > 0
    assert not torch.equal(got, base)                  # 状态档真的改了 logits


def test_num_state_values_per_layer_and_shared():
    per_layer = _tiny_model(per_layer=True)
    shared = _tiny_model(per_layer=False)
    # 2 层 × 2 头 × 4 × 4；单共享槽 1 × 2 × 4 × 4
    assert cq.num_state_values(per_layer) == 2 * 2 * 4 * 4
    assert cq.num_state_values(shared) == 2 * 4 * 4
