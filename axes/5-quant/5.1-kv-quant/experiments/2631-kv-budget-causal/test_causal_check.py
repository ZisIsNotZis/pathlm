"""2631 · 因果口径 KV 预算图 —— `causal_check.py` 的可失败单测。

> 位置说明（SCHEMA §22 / 本波写入者纪律）：本波 `tests/**` 由 L1 线独占，故本单测**放在
> 本实验目录内**（与 `W4/2613`、`M1/2620` 的既例一致），用
> `python3 -m pytest axes/R-references/R1-quantization/experiments/2631-kv-budget-causal/test_causal_check.py -q`
> 运行；`tests/` 全量套件不含它。

钉住 `causal_check.py` 的四件事（每个语义开关都要有可失败单测）：

1. **因果生产量化器**：`prefix_invariance_check` 的两条前缀不变性必须都为 True；
2. **非因果量化器会被抓红**（先红锚）：把 `kv_quant.quantize_kv` 临时换成旧的
   「沿 batch+token 取全序列 amax」实现 ⇒ 前缀不变性检查必须报 False；
3. **新旧并列比较**：非量化臂逐位不变 / 量化臂改变要被正确分类，方向标签正确；
4. `quantized_arms_of` 只挑真正施加 int8/int4 的臂。

破坏上述语义的变异见 `mutate.py`（真抓红 + 逐字节还原）。
"""

import importlib.util
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import causal_check as cc                                              # noqa: E402
import tools.evalkit.kv_quant as kq                                    # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel                # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_model(d=24, L=2, H=4, T=32, vocab=16):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=d, n_layers=L, n_heads=H, seq_len=T, mlp_mult=4,
                        dense=True)
    return DepthARModel(cfg, vocab_size=vocab)


def _noncausal_quantize(x, mode):
    """旧的缺陷实现：scale 沿 batch + token 取全序列 amax（复核 A2 原缺陷）。"""
    if mode == "fp32":
        return x, kq._new_stats()
    acc = kq._new_stats()
    if mode == "fp16":
        dq = x.to(torch.float16).to(x.dtype)
    else:
        bits = kq._INT_BITS[mode]
        qmax = 2 ** (bits - 1) - 1
        reduce = tuple(i for i in range(x.dim()) if i not in (x.dim() - 1, x.dim() - 3))
        scale = (x.abs().amax(dim=reduce, keepdim=True) / qmax).clamp_min(1e-12)
        q = torch.round(x / scale).clamp(-qmax, qmax)
        dq = q * scale
    acc["n_values"] = x.numel()
    acc["sum_abs_err"] = float((dq.float() - x.float()).abs().sum())
    acc["max_abs_err"] = float((dq.float() - x.float()).abs().max())
    acc["sum_abs_x"] = float(x.abs().sum())
    acc["n_changed"] = int((dq != x).sum())
    return dq, acc


# --------------------------------------------------------- 1. 因果生产量化器：必须过


def test_prefix_invariance_holds_for_causal_production():
    model = tiny_model()
    torch.manual_seed(1)
    tokens = torch.randint(0, 16, (3, 32))
    for mode in ("int8", "int4"):
        inv = cc.prefix_invariance_check(model, tokens, mode)
        assert inv["future_token_past_logits_equal"] is True, mode
        assert inv["peer_request_logits_equal"] is True, mode


# --------------------------------------------------- 2. 非因果实现会被抓红（先红锚）


def test_prefix_invariance_check_flags_noncausal_implementation(monkeypatch):
    """把生产量化器换成旧的**非因果**实现 ⇒ 两条前缀不变性必须至少命中（peer 必然红）。"""
    model = tiny_model()
    torch.manual_seed(2)
    tokens = torch.randint(0, 16, (3, 32))
    # 因果实现先过
    assert cc.prefix_invariance_check(model, tokens, "int4")["future_token_past_logits_equal"]
    monkeypatch.setattr(kq, "quantize_kv", _noncausal_quantize)
    bad = cc.prefix_invariance_check(model, tokens, "int4")
    assert bad["peer_request_logits_equal"] is False, "同批邻居耦合未被抓到（A2 旁路）"
    assert (bad["future_token_past_logits_equal"] is False
            or bad["peer_request_logits_equal"] is False)


def test_prefix_invariance_of_logits_rejects_bad_shape():
    model = tiny_model()
    with pytest.raises(ValueError):
        cc.prefix_invariance_of_logits(lambda x: model(x)[1]["depth_logits"][-1],
                                       torch.zeros(1, 4, dtype=torch.long))


# ------------------------------------------------------- 3. 新旧口径并列比较


def _pt(arm, bpc, recall, bpp, baseline):
    return {"arm": arm, "bpc": bpc, "bytes_per_position": bpp,
            "recall_copy": {"accuracy": recall}, "family_baseline": baseline}


OLD = [_pt("fp16", 1.483569, 0.10, 12288, "fp16"),
       _pt("int8", 1.500000, 0.10, 6144, "fp16"),
       _pt("int4", 1.529269, 0.10, 3072, "fp16"),
       _pt("win32", 1.700000, 0.05, 1536, "fp16"),
       _pt("mha", 1.482026, 0.20, 12288, "mha"),
       _pt("gqa_int8", 1.499000, 0.19, 1536, "mha")]
NEW = [_pt("fp16", 1.483569, 0.10, 12288, "fp16"),          # 非量化：不变
       _pt("int8", 1.492000, 0.10, 6144, "fp16"),           # 量化：新 < 旧
       _pt("int4", 1.521000, 0.10, 3072, "fp16"),
       _pt("win32", 1.700000, 0.05, 1536, "fp16"),          # 非量化：不变
       _pt("mha", 1.482026, 0.20, 12288, "mha"),
       _pt("gqa_int8", 1.500000, 0.19, 1536, "mha")]        # 量化：新 > 旧
QUANT = ["int8", "int4", "gqa_int8"]


def test_compare_classifies_nonquant_unchanged_and_quant_changed():
    cmp = cc.compare_causal_vs_noncausal(OLD, NEW, QUANT)
    assert cmp["nonquant_bitwise_preserved"] is True
    assert set(cmp["quant_arms_changed"]) == {"int8", "int4", "gqa_int8"}
    assert set(cmp["quant_arms_unchanged"]) == set()
    assert cmp["quant_delta_bpc"]["int8"] == pytest.approx(-0.008)
    assert cmp["quant_delta_bpc"]["gqa_int8"] == pytest.approx(+0.001)


def test_compare_detects_nonquant_drift():
    """非量化臂若漂移 ⇒ 可控性证据必须为 False（防止把口径改动的锅算到量化头上）。"""
    drifted = [dict(p) for p in NEW]
    drifted[0] = _pt("fp16", 1.499999, 0.10, 12288, "fp16")   # 基线被动了
    cmp = cc.compare_causal_vs_noncausal(OLD, drifted, QUANT)
    assert cmp["nonquant_bitwise_preserved"] is False
    assert cmp["nonquant_max_abs_delta_bpc"] > 1e-6


def test_direction_label_signs():
    assert "改善" in cc.direction_label(-0.01)
    assert "变差" in cc.direction_label(+0.01)
    assert "不变" in cc.direction_label(0.0)


def test_int8_int4_quality_delta_reports_overstated():
    cmp = cc.compare_causal_vs_noncausal(OLD, NEW, QUANT)
    d = cc.int8_int4_quality_delta(cmp)
    assert d["int8"]["delta_bpc"] == pytest.approx(-0.008)
    assert d["int8"]["old_overstated_damage"] is True
    assert d["int8"]["old_understated_damage"] is False
    assert d["int4"]["old_overstated_damage"] is True


def test_int8_int4_quality_delta_missing_arm_is_none():
    only_int8 = [_pt("int8", 1.50, 0.1, 6144, "fp16")]
    new_only = [_pt("int8", 1.49, 0.1, 6144, "fp16")]
    cmp = cc.compare_causal_vs_noncausal(only_int8, new_only, ["int8"])
    assert cc.int8_int4_quality_delta(cmp)["int4"] is None


# ------------------------------------------------------- 4. quantized_arms_of


def test_quantized_arms_of_selects_only_int_arms():
    cfg = {"arms": {
        "fp16": {"kind": "quant", "mode": "fp16"},
        "int8": {"kind": "quant", "mode": "int8"},
        "int4": {"kind": "quant", "mode": "int4"},
        "win32": {"kind": "window", "d": 32},
        "gqa": {"kind": "head", "n_kv_heads": 2},
        "gqa_int8": {"kind": "quant_head", "mode": "int8"},
        "gqa_win64": {"kind": "combo_mask", "mode": "fp32"},
        "win32_int8": {"kind": "combo_mask", "mode": "int8"},
    }}
    got = set(cc.quantized_arms_of(cfg))
    assert got == {"int8", "int4", "gqa_int8", "win32_int8"}
