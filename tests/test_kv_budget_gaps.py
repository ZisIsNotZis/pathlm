"""2620 · KV 预算图缺失组合（GQA×int8 / MQA×int8 / 窗口×锚点）的可失败单测。

契约：`axes/5-quant/5.1-kv-quant/experiments/2620-kv-budget-gaps/README.md`
§4（判据线逐字抄录 `.scratch/37-gaps/issues/01-gaps.md` 的「契约 V1」）。

Run: `python3 -m pytest tests/ -q`

钉住新语义层的四件事（SCHEMA §10 每个语义开关都要有可失败单测 + 变异验证）：

1. **横轴公式**：`quant_head`（`n_kv_heads_eff` 与 `value_bytes` 同时改）与
   `window_anchor`（`positions_kept = window + anchors`）都落在**同一个解析乘式**上；
   标签与事实不一致（`value_bytes` ≠ mode 对应值）必须报错，不能默默入账；
2. **可见性掩码**：`window_anchor` == `depth_ar.window_anchor_keep_mask`（训练/评测
   同一件掩码），`window` == `retention.recent_keep_mask`；锚点位置**真的**对末端
   查询可见（否则「锚点通道」是一句空话）；
3. **前向分派**：`quant_head` 臂 = **头共享模型** × 量化往返（fp32 档与释放版 dense
   前向逐位相等；int8 真改数值、真改 logits）；
4. **活性量**：两个因子都活着——头数由**真实张量形状**证明，量化由 `rel_err` 证明，
   锚点由被放行的因果对数证明。

破坏上述语义的变异（改 `VALUE_BYTES`、改锚点位置判据、把 `quant_head` 的前向换成
不量化的路径）会让对应单测抓红；变异与逐字节还原见 `README.md` §6b「变异验证」。
"""

import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GA_PATH = os.path.join(ROOT, "axes", "5-quant", "5.1-kv-quant",
                       "experiments", "2620-kv-budget-gaps", "gaps_arms.py")


def _load_gaps_arms():
    spec = importlib.util.spec_from_file_location("gaps_arms_2620", GA_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ga = _load_gaps_arms()

from pathlm.depth_ar import (DepthARConfig, DepthARModel,   # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.kv_quant import kv_quant_logits          # noqa: E402
from tools.evalkit.retention import (masked_logits,          # noqa: E402
                                     recent_keep_mask)

def tiny_model(n_kv_heads=None, d=24, L=2, H=4, T=64, vocab=12):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=d, n_layers=L, n_heads=H, seq_len=T, mlp_mult=4,
                        dense=True, n_kv_heads=n_kv_heads)
    return DepthARModel(cfg, vocab_size=vocab)


GQA_INT8 = {"arm": "gqa_int8", "kind": "quant_head", "n_kv_heads": 2, "mode": "int8",
            "value_bytes": 1.0, "positions_kept": 256}
MQA_INT8 = {"arm": "mqa_int8", "kind": "quant_head", "n_kv_heads": 1, "mode": "int8",
            "value_bytes": 1.0, "positions_kept": 256}
WIN16_ANCHOR16 = {"arm": "win16_anchor16", "kind": "window_anchor", "window": 16,
                  "anchors": 16, "value_bytes": 2.0}
WIN32_ANCHOR32 = {"arm": "win32_anchor32", "kind": "window_anchor", "window": 32,
                  "anchors": 32, "value_bytes": 2.0}
WIN32_ANCHOR8 = {"arm": "win32_anchor8", "kind": "window_anchor", "window": 32,
                 "anchors": 8, "value_bytes": 2.0}
WIN32 = {"arm": "win32", "kind": "window", "d": 32, "value_bytes": 2.0}


# --------------------------------------------------------------- 1. 横轴公式


def test_quant_head_bytes_are_gqa_times_int8():
    """GQA(2 头) × int8 ⇒ 2·12·2·32·1.0·1 = 1536；MQA(1 头) × int8 ⇒ 768。"""
    g = ga.bytes_per_position(GQA_INT8, n_layers=12, n_heads=8, head_dim=32, T=256)
    m = ga.bytes_per_position(MQA_INT8, n_layers=12, n_heads=8, head_dim=32, T=256)
    assert g["bytes_per_position"] == 1536.0
    assert g["factor_n_kv_heads_eff"] == 2 and g["factor_value_bytes"] == 1.0
    assert m["bytes_per_position"] == 768.0
    assert m["factor_n_kv_heads_eff"] == 1 and m["factor_value_bytes"] == 1.0


def test_window_anchor_bytes_add_positions():
    """窗口+锚点档的 `positions_kept` 必须是 `window + anchors`（锚点占用位置预算）。"""
    a = ga.bytes_per_position(WIN32_ANCHOR32, 12, 8, 32, 256)
    b = ga.bytes_per_position(WIN16_ANCHOR16, 12, 8, 32, 256)
    c = ga.bytes_per_position(WIN32_ANCHOR8, 12, 8, 32, 256)
    assert a["bytes_per_position"] == 3072.0 and a["factor_positions_kept"] == 64
    assert b["bytes_per_position"] == 1536.0 and b["factor_positions_kept"] == 32
    assert c["bytes_per_position"] == 1920.0 and c["factor_positions_kept"] == 40
    # 与「纯窗口」在同一公式上可比：win32(32 kept) 与 win16+anchor16(32 kept) 同字节
    w = ga.bytes_per_position(WIN32, 12, 8, 32, 256)
    assert w["bytes_per_position"] == b["bytes_per_position"] == 1536.0


def test_value_bytes_must_match_mode():
    """`value_bytes` 与 mode 不一致 ⇒ 报错（标签不能与事实不符）。"""
    bad = dict(GQA_INT8, value_bytes=2.0)          # int8 却写 2 字节
    with pytest.raises(ValueError):
        ga.bytes_per_position(bad, 12, 8, 32, 256)


def test_window_anchor_overlap_is_rejected():
    """window + anchors > T 时区间重叠、预算不相加 ⇒ 报错，不得入图。"""
    bad = {"arm": "x", "kind": "window_anchor", "window": 250, "anchors": 16,
           "value_bytes": 2.0}
    with pytest.raises(ValueError):
        ga.bytes_per_position(bad, 12, 8, 32, 256)


# --------------------------------------------------------------- 2. 可见性掩码


def test_keep_mask_of_matches_shared_instruments():
    T = 64
    assert torch.equal(ga.keep_mask_of(WIN32, T), recent_keep_mask(T, 32))
    assert torch.equal(ga.keep_mask_of(WIN32_ANCHOR8, T),
                       window_anchor_keep_mask(T, 32, 8))
    assert ga.keep_mask_of(GQA_INT8, T) is None


def test_anchor_positions_are_visible_to_last_query():
    """锚点语义的可失败点：前 `anchors` 个位置必须对**最末查询**可见；
    纯窗口（同 window）下它们必须不可见。"""
    T = 256
    ka = ga.keep_mask_of(WIN32_ANCHOR8, T)
    kw = ga.keep_mask_of(WIN32, T)
    for j in range(8):
        assert bool(ka[T - 1, j]) is True
    for j in range(8):
        assert bool(kw[T - 1, j]) is False
    # 锚点之外的最前位置（第 8..31）仍不可见 ⇒ 锚点不是「保留整个前缀」
    assert bool(ka[T - 1, 15]) is False


# --------------------------------------------------------------- 3. 前向分派


def test_quant_head_fp32_matches_release_dense_forward():
    """`quant_head` + mode=fp32 = 头共享模型的**释放版** dense 前向（逐位相等）。"""
    m = tiny_model(n_kv_heads=2)
    x = torch.randint(0, 12, (2, m.cfg.seq_len))
    spec = {"kind": "quant_head", "mode": "fp32", "n_kv_heads": 2, "value_bytes": 2.0}
    fn = ga.make_logits_fn(m, spec, m.cfg.seq_len, "cpu")
    ref = masked_logits(m, x, None)
    assert torch.equal(fn(x), ref)


def test_quant_head_int8_changes_logits_and_values():
    """int8 档必须真改数值、真改 logits（不是只改标签）；且 head 数真的是 2。"""
    m = tiny_model(n_kv_heads=2)
    x = torch.randint(0, 12, (2, m.cfg.seq_len))
    fp32 = ga.make_logits_fn(m, {"kind": "quant_head", "mode": "fp32",
                                 "n_kv_heads": 2, "value_bytes": 2.0},
                             m.cfg.seq_len, "cpu")
    int8 = ga.make_logits_fn(m, {"kind": "quant_head", "mode": "int8",
                                 "n_kv_heads": 2, "value_bytes": 1.0},
                             m.cfg.seq_len, "cpu")
    want_logits, want_raw = ga.kv_quant_head_logits(m, x, "int8")
    assert torch.allclose(int8(x), want_logits, atol=0, rtol=0)  # 与量化路径同值
    assert not torch.equal(fp32(x), int8(x))                     # 量化真的改了 logits
    summary = ga.summarize_stats(want_raw)
    assert int(summary["n_values"]) > 0 and float(summary["rel_err"]) > 0
    assert m.cfg.n_kv_heads == 2                              # head 共享确实生效


def test_reference_quant_logits_rejects_gqa_block():
    """既有参照线 `kv_quant_logits` 只支持 stock Block ⇒ 头共享臂必须走新模块。"""
    m = tiny_model(n_kv_heads=2)
    x = torch.randint(0, 12, (2, m.cfg.seq_len))
    with pytest.raises(ValueError):
        kv_quant_logits(m, x, "int8")


def test_window_anchor_forward_uses_anchor_mask():
    """`window_anchor` 前向必须走锚点掩码（与纯窗口不同）；锚点让源信息可见。"""
    m = tiny_model()
    T = m.cfg.seq_len
    x = torch.randint(0, 12, (2, T))
    fn = ga.make_logits_fn(m, WIN32_ANCHOR8, T, "cpu")
    ref = masked_logits(m, x, window_anchor_keep_mask(T, 32, 8))
    assert torch.equal(fn(x), ref)
    assert not torch.equal(fn(x), masked_logits(m, x, recent_keep_mask(T, 32)))


def test_make_logits_fn_rejects_foreign_kind():
    m = tiny_model()
    win_t = {"arm": "win32", "kind": "window", "d": 32, "value_bytes": 2.0}
    with pytest.raises(ValueError):
        ga.make_logits_fn(m, win_t, m.cfg.seq_len, "cpu")


# --------------------------------------------------------------- 4. 活性量


def test_activity_quant_head_reports_both_factors():
    m = tiny_model(n_kv_heads=2)
    x = torch.randint(0, 12, (2, m.cfg.seq_len))
    act = ga.activity_of(GQA_INT8, m, m.cfg.seq_len, "cpu", x)
    assert act["kind"] == "quant_head" and act["n_kv_heads"] == 2
    assert float(act["quant"]["rel_err"]) > 0        # 量化活着
    assert act["qkv_out_width"] != 3 * m.cfg.d_model  # 头共享活着（宽度更窄）


def test_activity_window_anchor_reports_two_counting_budgets():
    m = tiny_model()
    act = ga.activity_of(WIN32_ANCHOR8, m, m.cfg.seq_len, "cpu",
                         torch.randint(0, 12, (1, m.cfg.seq_len)))
    assert act["kind"] == "window_anchor" and act["anchors"] == 8
    assert act["dropped_frac"] > 0                    # 掩码真的挡掉了 key
    T = m.cfg.seq_len
    # 每行都放行锚点：列 j（<8）对一切 i≥j 可见 ⇒ Σ列 True = Σ_j (T−j)
    assert act["anchor_contribution_pairs"] == sum(T - j for j in range(8))
    assert act["diagonal_kept"] is True               # 不产生全掩码行
