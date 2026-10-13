"""2621 · KV 预算图**剩余缺失组合**（窗口×int8 / 锚点×int4 / 头共享×窗口）的可失败单测。

契约：`axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/README.md` §4
（判据同 V1）。

钉住新语义层 `combo_mask`（`rest_arms.py`）的四件事（SCHEMA §10 每个语义开关都要有可失败
单测 + 变异验证）：

1. **横轴公式**：`combo_mask` 的 `bytes_per_position` 与 `2619`/`2620` **同一个解析乘式**
   （与 `gaps_arms` 在等价的「窗口 × 量化」字段上算出同一个数）；`value_bytes` 与 `mode`
   不一致必须报错；
2. **可见性掩码**：`combo_mask` == `depth_ar.window_anchor_keep_mask`（`anchors=0` 退化为
   `retention.recent_keep_mask`）；
3. **前向分派**：`combo_mask` 的两条退化路径**逐位还原既有仪器**（`mode="fp32"` ⇒
   `masked_logits`；全因果掩码 ⇒ `kv_quant_logits`），且量化/掩码真的改 logits；
4. **活性量**：掩码（`dropped_frac`）、量化（`rel_err`）、锚点（放行的因果对数）、
   头共享（真实张量形状）都活着。

破坏上述语义的变异（改 `VALUE_BYTES["int4"]`、让 `masked_kv_quant_logits` 跳过量化、
让 `keep_mask_of` 忽略锚点）会让对应单测抓红；变异与逐字节还原见 `README.md` §6b。
"""

import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RA_PATH = os.path.join(ROOT, "axes", "5-quant", "5.1-kv-quant",
                       "experiments", "2621-kv-budget-rest", "rest_arms.py")
GA_PATH = os.path.join(ROOT, "axes", "5-quant", "5.1-kv-quant",
                       "experiments", "2620-kv-budget-gaps", "gaps_arms.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ra = _load(RA_PATH, "rest_arms_2621")
ga = _load(GA_PATH, "gaps_arms_2620")

from pathlm.depth_ar import (DepthARConfig, DepthARModel,   # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.kv_quant import kv_quant_logits          # noqa: E402
from tools.evalkit.retention import (masked_logits,         # noqa: E402
                                     recent_keep_mask)


def tiny_model(n_kv_heads=None, d=24, L=2, H=4, T=64, vocab=12):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=d, n_layers=L, n_heads=H, seq_len=T, mlp_mult=4,
                        dense=True, n_kv_heads=n_kv_heads)
    return DepthARModel(cfg, vocab_size=vocab)


WIN32_INT8 = {"arm": "win32_int8", "kind": "combo_mask", "base_model": "trunk",
              "window": 32, "anchors": 0, "mode": "int8", "value_bytes": 1.0,
              "positions_kept": 32}
WIN64_INT8 = {"arm": "win64_int8", "kind": "combo_mask", "base_model": "trunk",
              "window": 64, "anchors": 0, "mode": "int8", "value_bytes": 1.0,
              "positions_kept": 64}
WIN16_ANCHOR16_INT4 = {"arm": "win16_anchor16_int4", "kind": "combo_mask",
                       "base_model": "trunk", "window": 16, "anchors": 16,
                       "mode": "int4", "value_bytes": 0.5, "positions_kept": 32}
WIN32_ANCHOR32_INT4 = {"arm": "win32_anchor32_int4", "kind": "combo_mask",
                       "base_model": "trunk", "window": 32, "anchors": 32,
                       "mode": "int4", "value_bytes": 0.5, "positions_kept": 64}
GQA_WIN64 = {"arm": "gqa_win64", "kind": "combo_mask", "base_model": "gqa",
             "n_kv_heads": 2, "window": 64, "anchors": 0, "mode": "fp32",
             "value_bytes": 2.0, "positions_kept": 64}
GQA_WIN128 = {"arm": "gqa_win128", "kind": "combo_mask", "base_model": "gqa",
              "n_kv_heads": 2, "window": 128, "anchors": 0, "mode": "fp32",
              "value_bytes": 2.0, "positions_kept": 128}
GQA_INT4 = {"arm": "gqa_int4", "kind": "quant_head", "n_kv_heads": 2, "mode": "int4",
            "value_bytes": 0.5, "positions_kept": 256}
MQA_INT4 = {"arm": "mqa_int4", "kind": "quant_head", "n_kv_heads": 1, "mode": "int4",
            "value_bytes": 0.5, "positions_kept": 256}


# --------------------------------------------------------------- 1. 横轴公式


def test_combo_mask_bytes_match_single_formula():
    """窗口×int8 / 锚点×int4 / GQA×窗口 的字节都由**唯一的解析乘式**给出。"""
    a = ra.bytes_per_position(WIN32_INT8, 12, 8, 32, 256)
    b = ra.bytes_per_position(WIN64_INT8, 12, 8, 32, 256)
    c = ra.bytes_per_position(WIN16_ANCHOR16_INT4, 12, 8, 32, 256)
    d = ra.bytes_per_position(WIN32_ANCHOR32_INT4, 12, 8, 32, 256)
    assert a["bytes_per_position"] == 768.0 and a["factor_positions_kept"] == 32
    assert b["bytes_per_position"] == 1536.0 and b["factor_positions_kept"] == 64
    assert c["bytes_per_position"] == 384.0 and c["factor_positions_kept"] == 32
    assert d["bytes_per_position"] == 768.0 and d["factor_positions_kept"] == 64
    assert a["factor_value_bytes"] == 1.0 and c["factor_value_bytes"] == 0.5


def test_combo_mask_matches_gaps_arms_on_equivalent_fields():
    """与 `gaps_arms`（2619/2620 的公式）在等价的『窗口 × 量化』字段上算出同一个数。"""
    mine = ra.bytes_per_position(WIN32_INT8, 12, 8, 32, 256)
    equiv = ga.bytes_per_position({"arm": "win32", "kind": "window", "d": 32,
                                   "value_bytes": 1.0, "mode": "int8"}, 12, 8, 32, 256)
    assert mine["bytes_per_position"] == equiv["bytes_per_position"] == 768.0
    mine2 = ra.bytes_per_position(WIN64_INT8, 12, 8, 32, 256)
    equiv2 = ga.bytes_per_position({"arm": "win64", "kind": "window", "d": 64,
                                    "value_bytes": 1.0, "mode": "int8"}, 12, 8, 32, 256)
    assert mine2["bytes_per_position"] == equiv2["bytes_per_position"] == 1536.0


def test_head_window_bytes_use_n_kv_heads():
    """GQA×窗口 的 `n_kv_heads_eff` 取 2（头共享），不是 8。"""
    g = ra.bytes_per_position(GQA_WIN64, 12, 8, 32, 256)
    h = ra.bytes_per_position(GQA_WIN128, 12, 8, 32, 256)
    assert g["bytes_per_position"] == 768.0 and g["factor_n_kv_heads_eff"] == 2
    assert h["bytes_per_position"] == 1536.0 and h["factor_positions_kept"] == 128


def test_quant_head_int4_bytes():
    """GQA×int4 / MQA×int4 走既有 `quant_head` 语义（量化档换 int4）。"""
    assert ga.bytes_per_position(GQA_INT4, 12, 8, 32, 256)["bytes_per_position"] == 768.0
    assert ga.bytes_per_position(MQA_INT4, 12, 8, 32, 256)["bytes_per_position"] == 384.0


def test_value_bytes_must_match_mode():
    bad = dict(WIN32_INT8, value_bytes=2.0)            # int8 却写 2 字节
    with pytest.raises(ValueError):
        ra.bytes_per_position(bad, 12, 8, 32, 256)


def test_positions_kept_bounds_rejected():
    bad = {"arm": "x", "kind": "combo_mask", "base_model": "trunk", "window": 250,
           "anchors": 16, "mode": "int4", "value_bytes": 0.5}
    with pytest.raises(ValueError):
        ra.bytes_per_position(bad, 12, 8, 32, 256)


# --------------------------------------------------------------- 2. 可见性掩码


def test_keep_mask_of_matches_shared_instruments():
    T = 64
    assert torch.equal(ra.keep_mask_of(WIN32_INT8, T), recent_keep_mask(T, 32))
    assert torch.equal(ra.keep_mask_of(WIN16_ANCHOR16_INT4, T),
                       window_anchor_keep_mask(T, 16, 16))
    assert torch.equal(ra.keep_mask_of(WIN32_ANCHOR32_INT4, T),
                       window_anchor_keep_mask(T, 32, 32))
    assert torch.equal(ra.keep_mask_of(GQA_WIN64, T), recent_keep_mask(T, 64))


def test_anchor_positions_visible_to_last_query():
    """锚点语义的可失败点：前 `anchors` 个位置对最末查询可见；纯窗口下不可见。"""
    T = 256
    ka = ra.keep_mask_of(WIN32_ANCHOR32_INT4, T)
    kw = ra.keep_mask_of(WIN32_INT8, T)
    for j in range(32):
        assert bool(ka[T - 1, j]) is True
        assert bool(kw[T - 1, j]) is False
    assert bool(ka[T - 1, 32]) is False                # 锚点之外仍不可见


# --------------------------------------------------------------- 3. 前向分派


def test_combo_mask_fp32_matches_masked_logits():
    """`combo_mask` + mode=fp32 = `retention.masked_logits`（量化恒等，逐位相等）。"""
    m = tiny_model()
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    keep = window_anchor_keep_mask(T, 16, 8)
    got = ra.masked_kv_quant_logits(m, x, keep, "fp32")[0]
    ref = masked_logits(m, x, keep)
    assert torch.equal(got, ref)


def test_combo_mask_full_causal_quant_matches_kv_quant_logits():
    """掩码 = 全因果时，int8 档必须逐位还原 `kv_quant.kv_quant_logits`。"""
    m = tiny_model()
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    keep_full = window_anchor_keep_mask(T, T, 0)
    got = ra.masked_kv_quant_logits(m, x, keep_full, "int8")[0]
    ref = kv_quant_logits(m, x, "int8")[0]
    assert torch.equal(got, ref)


def test_combo_mask_causal_fp32_matches_release_path():
    m = tiny_model()
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    got = ra.masked_kv_quant_logits(m, x, None, "fp32")[0]
    ref = kv_quant_logits(m, x, "fp32")[0]
    assert torch.equal(got, ref)


def test_combo_mask_gqa_block_matches_masked_logits():
    """头共享 × 窗口：GQA 块 + 掩码，fp32 档（量化恒等）与 `masked_logits` 逐位相等。"""
    m = tiny_model(n_kv_heads=2)
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    keep = recent_keep_mask(T, 32)
    got = ra.masked_kv_quant_logits(m, x, keep, "fp32")[0]
    assert torch.equal(got, masked_logits(m, x, keep))
    assert torch.equal(ra.masked_kv_quant_logits(m, x, None, "fp32")[0],
                       masked_logits(m, x, None))


def test_combo_mask_quant_and_mask_change_logits():
    """两个因子都必须真的改 logits：量化档 != fp32；窗口档 != 全因果。"""
    m = tiny_model()
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    keep = window_anchor_keep_mask(T, 32, 8)
    full = window_anchor_keep_mask(T, T, 0)
    fp32 = ra.masked_kv_quant_logits(m, x, keep, "fp32")[0]
    int8 = ra.masked_kv_quant_logits(m, x, keep, "int8")[0]
    int4 = ra.masked_kv_quant_logits(m, x, keep, "int4")[0]
    nocausal = ra.masked_kv_quant_logits(m, x, full, "fp32")[0]
    assert not torch.equal(fp32, int8)
    assert not torch.equal(int8, int4)
    assert not torch.equal(fp32, nocausal)


def test_combo_mask_int8_reports_real_quantization():
    """int8 档的量化活性量必须非零（不是只改标签）。"""
    m = tiny_model()
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    keep = recent_keep_mask(T, 32)
    _, stats = ra.masked_kv_quant_logits(m, x, keep, "int8")
    s = ra.summarize_stats(stats)
    assert int(s["n_values"]) > 0 and float(s["rel_err"]) > 0
    # fp32 档是恒等：无量化因子（rel_err = 0）
    _, stats0 = ra.masked_kv_quant_logits(m, x, keep, "fp32")
    assert float(ra.summarize_stats(stats0)["rel_err"]) == 0.0


def test_make_logits_fn_rejects_foreign_kind():
    m = tiny_model()
    with pytest.raises(ValueError):
        ra.make_logits_fn(m, {"kind": "window", "d": 32, "value_bytes": 2.0},
                          m.cfg.seq_len, "cpu")


def test_bad_mode_rejected():
    m = tiny_model()
    x = torch.randint(0, 12, (2, m.cfg.seq_len))
    with pytest.raises(ValueError):
        ra.masked_kv_quant_logits(m, x, None, "int2")


# --------------------------------------------------------------- 4. 活性量


def test_activity_combo_mask_reports_both_factors():
    m = tiny_model()
    T = m.cfg.seq_len
    probe = torch.randint(0, 12, (2, T))
    act = ra.activity_of(WIN32_INT8, m, T, "cpu", probe)
    assert act["kind"] == "combo_mask" and act["mode"] == "int8"
    assert act["dropped_frac"] > 0                     # 掩码活着
    assert float(act["quant"]["rel_err"]) > 0          # 量化活着
    assert act["anchor_contribution_pairs"] == 0       # 无锚点
    assert act["positions_kept_analytic"] == 32


def test_activity_anchor_contribution_pairs():
    m = tiny_model()
    T = m.cfg.seq_len
    probe = torch.randint(0, 12, (1, T))
    act = ra.activity_of(WIN16_ANCHOR16_INT4, m, T, "cpu", probe)
    assert act["anchors"] == 16 and act["positions_kept_analytic"] == 32
    # 列 j（<16）对一切 i≥j 可见 ⇒ Σ列 True = Σ_j (T−j)
    assert act["anchor_contribution_pairs"] == sum(T - j for j in range(16))
    assert act["diagonal_kept"] is True                # 不产生全掩码行


def test_activity_head_window_reports_kv_head_shape():
    """头共享 × 窗口臂：qkv 实际输出宽度 < 3d（GQA 真的生效）。"""
    m = tiny_model(n_kv_heads=2)
    T = m.cfg.seq_len
    probe = torch.randint(0, 12, (2, T))
    act = ra.activity_of(GQA_WIN64, m, T, "cpu", probe)
    assert act["base_model"] == "gqa"
    assert act["head"]["qkv_out_width"] != 3 * m.cfg.d_model
