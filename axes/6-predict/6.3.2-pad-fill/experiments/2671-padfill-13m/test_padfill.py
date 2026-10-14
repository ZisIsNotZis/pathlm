"""2671-padfill-13m · 契约 §5 的可失败单测（CPU、秒级）。

靶子 = 本实验自己的仪器 `padfill.py`：

- **L0 硬门**：`fill_bidir_*` 的模型输入在被填位置含 `[mask]`，把**真 token 扰动**后重前向，
  被填位置 logits 必须逐位不变（`max|Δ| == 0`）；`fill_causal_r25` 额外要求右侧**非 M** 真 token
  扰动后左侧 M 位置 logits 逐位不变。
- **L0b gate probe**：把 mask 构造刻意改成泄漏模式（`leak`，把真 token 喂进 M 位置）⇒ L0 必须变红；
  把 causal 臂改成 full ⇒ 第二条必须变红。两条都**真变红**才算这台仪器可失败。
- 统计层：bpc 的算术与 CI 聚合方向。
- 结构不变量：tied 读出（`logits == h @ E^T`）、实测参数量落在契约 §2 的预期区间。

    python3 -m pytest test_padfill.py -q
"""

import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import padfill as P  # noqa: E402

# 小配置：与交付结构走同一代码路径，但 CPU 秒级（L0 测的是机制）。
TV = 32          # 测试用 vocab（含 [mask] 槽）
TD = 64          # d_model
TL = 2           # n_layers
TH = 4           # n_heads
TT = 64          # seq_len


def _mk(attn_mask: str, seed: int = 0) -> P.PadFillLM:
    torch.manual_seed(seed)
    m = P.PadFillLM(vocab_size=TV, attn_mask=attn_mask, d_model=TD, n_layers=TL,
                    n_heads=TH, mlp_mult=4, seq_len=TT)
    m.eval()
    return m


def _true_tokens(B: int, T: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TV - 1, (B, T), generator=g)  # 真 token 永不含 [mask]


def _max_abs(a: torch.Tensor, b: torch.Tensor, sel: torch.Tensor | None = None) -> float:
    d = (a - b).abs()
    if sel is not None:
        d = d[sel]
    return float(d.max())


# ---------------------------------------------------------------- L0 硬门（绿）---
def test_L0_bidir_mask_blocks_true_token_at_M():
    """fill_bidir：扰动 M 位置真 token 后，M 位置 logits 逐位不变。"""
    m = _mk("full")
    true = _true_tokens(2, TT)
    M = P.sample_fill_mask(true.shape, 0.25, torch.Generator().manual_seed(7))
    assert 0 < int(M.sum())

    x = P.make_fill_input(true, M, "mask", mask_token=m.mask_token)
    assert torch.equal(x[M], torch.full_like(x[M], m.mask_token))  # M 位置确含 [mask]

    true2 = true.clone()
    true2[M] = (true[M] + 1) % (TV - 1)          # 被填位置的真值换成任意别的 token
    assert torch.equal(true2[M] - true[M] != 0, torch.ones_like(M[M]))  # 确实换了
    x2 = P.make_fill_input(true2, M, "mask", mask_token=m.mask_token)

    with torch.no_grad():
        lg, lg2 = m(x), m(x2)
    assert _max_abs(lg, lg2, M) == 0.0


def test_L0_causal_right_nonM_does_not_touch_left_M():
    """fill_causal_r25：右侧非 M 真 token 扰动后，左侧 M 位置 logits 逐位不变。"""
    m = _mk("causal")
    true = _true_tokens(2, TT)
    pos = torch.arange(TT)[None, :]
    M = P.sample_fill_mask(true.shape, 0.25, torch.Generator().manual_seed(7))
    half = TT // 2
    perturb = (~M) & (pos >= half)                # 右半的**非 M** 位置
    check = M & (pos < half)                      # 左半的 M 位置
    assert int(perturb.sum()) > 0 and int(check.sum()) > 0

    x = P.make_fill_input(true, M, "mask", mask_token=m.mask_token)
    true2 = true.clone()
    true2[perturb] = (true[perturb] + 1) % (TV - 1)
    x2 = P.make_fill_input(true2, M, "mask", mask_token=m.mask_token)
    assert not torch.equal(x, x2)                 # 扰动确实进了输入

    with torch.no_grad():
        lg, lg2 = m(x), m(x2)
    assert _max_abs(lg, lg2, check) == 0.0


# ------------------------------------------------------------ L0b gate probe（红）---
def test_L0b_probe1_leak_mask_makes_L0_red():
    """gate probe 1：泄漏 mask 构造（真 token 喂进 M 位置）⇒ L0 变红。"""
    m = _mk("full")
    true = _true_tokens(2, TT)
    M = P.sample_fill_mask(true.shape, 0.25, torch.Generator().manual_seed(7))
    true2 = true.clone()
    true2[M] = (true[M] + 1) % (TV - 1)

    x = P.make_fill_input(true, M, "leak", mask_token=m.mask_token)
    x2 = P.make_fill_input(true2, M, "leak", mask_token=m.mask_token)
    assert not torch.equal(x, x2)                 # 泄漏模式确实把真值喂进去了

    with torch.no_grad():
        lg, lg2 = m(x), m(x2)
    assert _max_abs(lg, lg2, M) > 0.0             # L0 的红


def test_L0b_probe2_full_attention_makes_causal_check_red():
    """gate probe 2：把 causal 臂改成 full ⇒ 第二条（右侧扰动）检查变红。"""
    true = _true_tokens(2, TT)
    pos = torch.arange(TT)[None, :]
    M = P.sample_fill_mask(true.shape, 0.25, torch.Generator().manual_seed(7))
    half = TT // 2
    perturb = (~M) & (pos >= half)
    check = M & (pos < half)
    assert int(perturb.sum()) > 0 and int(check.sum()) > 0

    m_full = _mk("full")                          # 同 seed ⇒ 只差可见性
    x = P.make_fill_input(true, M, "mask", mask_token=m_full.mask_token)
    true2 = true.clone()
    true2[perturb] = (true[perturb] + 1) % (TV - 1)
    x2 = P.make_fill_input(true2, M, "mask", mask_token=m_full.mask_token)

    with torch.no_grad():
        lg, lg2 = m_full(x), m_full(x2)
    assert _max_abs(lg, lg2, check) > 0.0         # 第二条的红


def test_causal_model_output_at_i_ignores_all_right_of_i():
    """因果前向正确性：任意位置 i 的 logits 不受所有 j > i 扰动影响（逐位）。"""
    m = _mk("causal")
    true = _true_tokens(2, TT)
    x = true.clone()
    cut = TT // 2
    true2 = true.clone()
    true2[:, cut:] = (true[:, cut:] + 1) % (TV - 1)
    with torch.no_grad():
        lg, lg2 = m(x), m(true2)
    assert _max_abs(lg[:, :cut], lg2[:, :cut]) == 0.0   # 左侧逐位不变
    assert _max_abs(lg[:, cut:], lg2[:, cut:]) > 0.0    # 右侧确实变了（非空步）


# ------------------------------------------------------------------- 统计层 ---
def test_bpc_of_uniform_logits_is_log2_vocab():
    """均匀 logits ⇒ 每字符 bpc = log2(V)。"""
    V = 8
    logits = torch.zeros(2, 5, V)
    targets = torch.randint(0, V, (2, 5))
    nll = P.nll_of(logits, targets)
    bpc = P.bpc_from_nll(float(nll.sum()), nll.numel())
    assert bpc == pytest.approx(math.log2(V), abs=1e-6)


def test_bpc_matches_manual_nll_over_subset():
    """bpc 只在给定位置集上平均（评测位置集的算术）。"""
    torch.manual_seed(3)
    V = 16
    logits = torch.randn(1, 6, V)
    targets = torch.randint(0, V, (1, 6))
    nll = P.nll_of(logits, targets)
    sel = torch.tensor([1, 3, 4])
    bpc = P.bpc_from_nll(float(nll[0, sel].sum()), len(sel))
    assert bpc == pytest.approx(float(nll[0, sel].sum()) / (len(sel) * math.log(2)), abs=1e-12)


def test_bootstrap_ci_direction_positive_delta_excludes_zero():
    """全正配对 Δ ⇒ CI 下界 > 0（不含 0）。"""
    mean, lo, hi = P.paired_bootstrap_ci([0.05] * 64)
    assert mean == pytest.approx(0.05)
    assert lo > 0.0 and hi > 0.0


def test_bootstrap_ci_direction_centered_delta_includes_zero():
    """零均值配对 Δ ⇒ CI 含 0。"""
    deltas = [0.05, -0.05] * 32
    mean, lo, hi = P.paired_bootstrap_ci(deltas)
    assert mean == pytest.approx(0.0)
    assert lo < 0.0 < hi


# --------------------------------------------------------- 结构 / 冻结不变量 ---
def test_tied_readout_logits_equal_h_at_E_transpose():
    """tied 读出：forward 的 logits 逐位等于 `ln_f(h) @ E^T`（E 是唯一读出基）。"""
    m = _mk("full")
    idx = _true_tokens(1, 8)
    pos = torch.arange(8)
    with torch.no_grad():
        h = m.tok_embed(idx) + m.pos_embed(pos)[None]
        for blk in m.blocks:
            h = blk(h, m.causal)
        h = m.ln_f(h)
        manual = h @ m.tok_embed.weight.t()
        auto = m(idx)
    assert torch.equal(manual, auto)
    shapes = [tuple(p.shape) for p in m.parameters()]
    assert shapes.count((TV, TD)) == 1     # 只有一个 (V, d) 表：读出与嵌入共享


def test_real_structure_param_count_in_expected_band():
    """交付结构的实测参数量落在契约 §2 预期的 12–14M。"""
    m = P.PadFillLM(attn_mask="full")
    assert m.seq_len == P.SEQ_LEN and m.vocab_size == P.VOCAB_SIZE
    n = P.count_params(m)
    assert 12_000_000 < n < 14_000_000


def test_attn_mask_switch_rejects_unknown():
    with pytest.raises(ValueError):
        P.PadFillLM(attn_mask="diagonal")


def test_sample_fill_mask_respects_min_pos_and_rate():
    M = P.sample_fill_mask((4, 512), 0.25, torch.Generator().manual_seed(0), min_pos=8)
    assert not bool(M[:, :8].any())
    assert 0.20 < float(M[:, 8:].float().mean()) < 0.30


def test_make_fill_input_modes():
    true = _true_tokens(1, 4)
    M = torch.zeros(1, 4, dtype=torch.bool)
    M[0, 1] = True
    xm = P.make_fill_input(true, M, "mask", mask_token=TV - 1)
    assert int(xm[0, 1]) == TV - 1 and int(xm[0, 0]) == int(true[0, 0])
    xl = P.make_fill_input(true, M, "leak", mask_token=TV - 1)
    assert torch.equal(xl, true)               # 泄漏模式保留真值（gate probe 的靶）


def test_eval_block_set_shapes_and_determinism():
    """评测聚合：三路 bpc 齐备、`ar_at_M`/`fill_M` 用同一批 M（配对）、同 seed 重跑逐值相同。"""
    import numpy as np
    ar = _mk("causal")
    fl = _mk("full")
    val = (np.arange(2000) % (TV - 1)).astype(np.uint16)
    offs = P.eval_offsets(len(val), T=TT, n=4)
    a = P.eval_block_set(ar, fl, val, offs, 0.25, device="cpu", T=TT)
    b = P.eval_block_set(ar, fl, val, offs, 0.25, device="cpu", T=TT)
    assert {"ar_full_bpc", "ar_at_M_bpc", "fill_M_bpc", "delta_blocks", "delta_lo"} <= set(a)
    assert len(a["delta_blocks"]) == len(offs)
    assert [r["n_m"] for r in a["blocks"]] == [r["n_m"] for r in b["blocks"]]
    assert a["delta_blocks"] == b["delta_blocks"]
    assert (a["delta_lo"], a["delta_hi"]) == (b["delta_lo"], b["delta_hi"])
    only_ar = P.eval_block_set(ar, None, val, offs, 0.25, device="cpu", T=TT)
    assert "fill_M_bpc" not in only_ar


def test_steps_from_steptime_matches_contract_rule():
    assert P.steps_from_steptime(0.05) == 3000      # 1500s / (10 × 0.05) = 3000
    assert P.steps_from_steptime(0.02) == 7500
    assert P.steps_from_steptime(1.0) == P.STEP_MIN  # 反解 < 1500 ⇒ 抬到最低档
    assert P.steps_from_steptime(0.05) % P.STEP_QUANTUM == 0


def test_has_wall_clock_flags_only_wall_semantics():
    assert P.has_wall_clock({"steps": 3000, "arms": ["ar"], "params": 12.8, "mse": 0.1}) == []
    assert P.has_wall_clock({"wall_clock_s": 1.0}) == [".wall_clock_s"]
    assert P.has_wall_clock({"nested": {"elapsed": 2}}) == [".nested.elapsed"]
