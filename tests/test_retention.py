"""KV 消融保留曲线仪器的可失败单测（ticket 31 契约 I）。

Run: `python3 -m pytest tests/ -q`

钉住三件事：
1. **删除位置集合正确**（三条曲线的掩码逐点语义、对角线恒保留、masked 对数
   的闭式计数）——契约要求「故意删错位置必须抓红」，见 `mutate.py`；
2. **掩码真的被注意力读到**：与一份独立手写的掩码注意力参考实现逐位一致，
   且身份（精确因果）掩码与释放版 `is_causal=True` 路径逐位相等；
3. **bpc 助手口径正确**（因果移位 CE、与释放版基线同一数据流）且干预活性
   非零（删除确实改变 logits / 抬高 bpc）。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.depth_ar import DepthARConfig, DepthARModel
from pathlm.data import batch
from tools.evalkit.retention import (distance_keep_mask, prefix_keep_mask,
                                     segment_keep_mask, mask_stats, masked_logits,
                                     masked_bpc, row_bpc, eval_mask, safe_radius,
                                     monotonicity, LN2)

T_SMALL = 8


def tiny_model(d=16, L=2, H=2, T=12, vocab=10):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=d, n_layers=L, n_heads=H, seq_len=T, mlp_mult=4,
                        dense=True)
    return DepthARModel(cfg, vocab_size=vocab)


def causal_keep(T, device="cpu"):
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(1) >= idx.unsqueeze(0)  # keep[i,j] = i >= j


# ------------------------------------------------------- 1. 删除位置集合（掩码）


def test_distance_mask_deletes_exactly_distance_d():
    """距离曲线的删除集 = 恰好在距离 d 上的 key：keep[i,j] ⟺ j ≤ i 且 i−j ≠ d。
    闭式计数：masked 对数 = T − d。"""
    for d in (1, 2, 3, 5, 7):
        keep = distance_keep_mask(T_SMALL, d)
        i = torch.arange(T_SMALL).unsqueeze(1)
        j = torch.arange(T_SMALL).unsqueeze(0)
        expect = (j <= i) & ((i - j) != d)
        assert torch.equal(keep, expect)
        assert int((~keep & causal_keep(T_SMALL)).sum()) == T_SMALL - d
        # 被删的正是距离恰好 d 的那些 (i, i-d) 对
        dropped = (~keep & causal_keep(T_SMALL)).nonzero()
        assert all(int(pi) - int(pj) == d for pi, pj in dropped)
        assert keep.diag().all()
    with pytest.raises(ValueError):
        distance_keep_mask(T_SMALL, 0)
    with pytest.raises(ValueError):
        distance_keep_mask(T_SMALL, T_SMALL)


def test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal():
    """前缀截断 = 删绝对位置 [0, N)：key j < N 对所有 query 不可见（自身位置
    落在前缀里的 query 只留对角）；最近的 key 一个不删。"""
    T = T_SMALL
    for n in (0, 1, 3, 5, T):
        keep = prefix_keep_mask(T, n)
        i = torch.arange(T).unsqueeze(1)
        j = torch.arange(T).unsqueeze(0)
        expect = (j <= i) & ~(j < n)
        expect = expect | torch.eye(T, dtype=torch.bool)
        assert torch.equal(keep, expect)
        assert keep.diag().all()
        # 被删的 (i,j) 对：key j < N 且 i > j（对角豁免）
        dropped = [(int(a), int(b)) for a, b in (~keep & causal_keep(T)).nonzero()]
        assert all(b < n and a > b for a, b in dropped)
        assert len(dropped) == sum(T - 1 - j for j in range(n))
        # 最近的 key（j >= N）对所有 i >= j 可见
        for jj in range(n, T):
            assert keep[jj, jj] and keep[T - 1, jj]
    with pytest.raises(ValueError):
        prefix_keep_mask(T, T + 1)


def test_segment_mask_deletes_contiguous_block_only():
    """连续段删除 = 只删 [start, start+length) 这一段，别的 key 不动。"""
    T = T_SMALL
    for start, length in ((0, 4), (2, 3), (5, 3), (T - 2, 2)):
        keep = segment_keep_mask(T, start, length)
        i = torch.arange(T).unsqueeze(1)
        j = torch.arange(T).unsqueeze(0)
        in_seg = (j >= start) & (j < start + length)
        expect = ((j <= i) & ~in_seg) | torch.eye(T, dtype=torch.bool)
        assert torch.equal(keep, expect)
        assert keep.diag().all()
        dropped = [(int(a), int(b)) for a, b in (~keep & causal_keep(T)).nonzero()]
        assert all(start <= b < start + length and a > b for a, b in dropped)
        assert len(dropped) == sum(max(0, T - 1 - b) for b in range(start, start + length))
    with pytest.raises(ValueError):
        segment_keep_mask(T, 4, 5)          # 越界
    with pytest.raises(ValueError):
        segment_keep_mask(T, -1, 2)


def test_mask_stats_reports_dropped_pair_activity():
    """活性量：dropped 对数 / 占比 / 完全不可见 key 数 / 对角保留。"""
    keep = prefix_keep_mask(T_SMALL, 3)
    st = mask_stats(keep)
    assert st["causal_pairs"] == T_SMALL * (T_SMALL + 1) // 2
    assert st["n_dropped_pairs"] == sum(T_SMALL - 1 - j for j in range(3))
    assert 0 < st["dropped_frac"] < 1 and st["diagonal_kept"]
    assert st["n_dropped_keys"] == 3          # 前缀删的正是 3 个位置
    assert mask_stats(segment_keep_mask(T_SMALL, 2, 3))["n_dropped_keys"] == 3
    assert mask_stats(distance_keep_mask(T_SMALL, 2))["n_dropped_keys"] == T_SMALL - 2
    # 全保留 ⇒ 零活性
    assert mask_stats(causal_keep(T_SMALL))["n_dropped_pairs"] == 0


# ------------------------------------------- 2. 掩码真的被注意力读到（参考实现）


def _reference_masked_logits(model, tokens, keep):
    """独立手写参考：逐 block 用显式 softmax 计算掩码注意力（不调用 SDPA）。"""
    B, T = tokens.shape
    d, H = model.cfg.d_model, model.cfg.n_heads
    dh = d // H
    h = model._input_states(tokens)
    allow = keep.view(1, 1, T, T).expand(B, H, T, T)
    for blk in model.blocks:
        x = h
        q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
        sh = lambda t: t.view(B, T, H, dh).transpose(1, 2)
        q_, k_, v_ = sh(q), sh(k), sh(v)
        scores = q_ @ k_.transpose(-2, -1) / dh ** 0.5
        scores = scores.masked_fill(~allow, float("-inf"))
        a = scores.softmax(-1) @ v_
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + blk.proj(a)
        h = x + blk.mlp(blk.ln2(x))
    return model.readout(h)


def test_masked_logits_matches_manual_reference():
    """掩码前向 == 独立手写的显式掩码注意力（公式层面钉住「这些位置真的被删」）。"""
    m = tiny_model()
    x = torch.randint(0, 10, (2, 12))
    for keep in (causal_keep(12), distance_keep_mask(12, 3), prefix_keep_mask(12, 4),
                 segment_keep_mask(12, 5, 3)):
        got = masked_logits(m, x, keep)
        ref = _reference_masked_logits(m, x, keep)
        assert torch.allclose(got, ref, atol=1e-5), (keep, (got - ref).abs().max())


def test_identity_causal_mask_bitwise_equals_release_forward():
    """身份（精确因果）掩码与释放版 forward 逐位相等 ⇒ 删除曲线与基线同引擎。"""
    m = tiny_model()
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    release = aux["depth_logits"][-1]
    assert torch.equal(masked_logits(m, x, causal_keep(12)), release)
    assert torch.equal(masked_logits(m, x, None), release)


def test_deletion_is_live_changes_logits_and_raises_bpc():
    """干预活性（SCHEMA §10）：删位置必须真的改变读数；不删 ≡ 基线。
    另外：距离 1 的删除比距离 6 的删除影响更大（近距离 key 更被读到）。"""
    m = tiny_model()
    x = torch.randint(0, 10, (2, 12))
    base = masked_logits(m, x, None)
    assert torch.equal(masked_logits(m, x, causal_keep(12)), base)
    near = masked_logits(m, x, distance_keep_mask(12, 1))
    far = masked_logits(m, x, distance_keep_mask(12, 6))
    assert not torch.allclose(near, base, atol=1e-6)
    assert not torch.allclose(far, base, atol=1e-6)
    assert (near - base).abs().max() > (far - base).abs().max()


# ------------------------------------------------------------- 3. bpc 助手口径


def test_masked_bpc_uses_causal_shift_ce():
    """bpc = 因果移位 token 平均 CE（nats → bits）；身份掩码 == 无掩码基线。"""
    m = tiny_model(T=16)
    arr = torch.randint(0, 10, (4000,)).numpy()
    gen = torch.Generator().manual_seed(2)
    x, _ = batch(arr, 2, 16, gen)
    logits = masked_logits(m, x, None)[:, :-1]
    tgt = x[:, 1:]
    manual = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tgt.reshape(-1)).item() / LN2
    got = masked_bpc(m, arr, None, n_batches=1, batch_size=2, window_seed=2,
                     device="cpu")
    assert abs(got - manual) < 1e-6
    ident = masked_bpc(m, arr, causal_keep(16), n_batches=1, batch_size=2,
                       window_seed=2, device="cpu")
    assert ident == got
    # 删掉所有可删 key（前缀 = 全删）⇒ bpc 必然变差且有限（无 NaN）
    bad = masked_bpc(m, arr, prefix_keep_mask(16, 15), n_batches=1, batch_size=2,
                     window_seed=2, device="cpu")
    assert torch.isfinite(torch.tensor(bad)) and bad > got


def test_row_bpc_matches_masked_bpc_and_decomposes():
    """行均 bpc：(a) 长度 = T−1；(b) 行均值 == masked_bpc（算术对账）；
    (c) 能把前缀删除的损伤拆成「前缀内的 query」与「前缀外的 query」。"""
    m = tiny_model(T=16)
    arr = torch.randint(0, 10, (4000,)).numpy()
    kw = dict(n_batches=2, batch_size=2, window_seed=2, device="cpu")
    base = row_bpc(m, arr, None, **kw)
    assert len(base) == 15
    assert abs(sum(base) / len(base) - masked_bpc(m, arr, None, **kw)) < 1e-6
    keep = prefix_keep_mask(16, 4)
    rows = row_bpc(m, arr, keep, **kw)
    assert abs(sum(rows) / len(rows) - masked_bpc(m, arr, keep, **kw)) < 1e-6
    # 位置 i < 4 的 query 被删光历史 ⇒ 损伤远大于前缀外的 query
    d_inside = sum(rows[:4]) / 4 - sum(base[:4]) / 4
    d_outside = sum(rows[4:]) / 11 - sum(base[4:]) / 11
    assert d_inside > 0 and d_inside > d_outside   # 未训练小模型上只看方向


def test_eval_mask_and_safe_radius_helper():
    """eval_mask 出 bpc + 活性量；safe_radius 取「|Δ| < 带」的最大参数值。"""
    m = tiny_model(T=16)
    arr = torch.randint(0, 10, (4000,)).numpy()
    row = eval_mask(m, arr, prefix_keep_mask(16, 2), n_batches=1, batch_size=2,
                    window_seed=2, device="cpu")
    assert set(row) >= {"bpc", "n_dropped_pairs", "dropped_frac", "diagonal_kept"}
    curve = [{"d": 1, "delta_bpc": -0.30}, {"d": 2, "delta_bpc": -0.005},
             {"d": 4, "delta_bpc": -0.02}, {"d": 8, "delta_bpc": -0.001}]
    sr = safe_radius(curve, band=0.014)
    assert sr["value"] == 8 and sr["n_safe"] == 2
    assert safe_radius([{"d": 1, "delta_bpc": -0.5}], band=0.014)["value"] is None


def test_monotonicity_helper_flags_violations():
    """单调性助手跑在 damage 口径上：距离曲线方向 = 非增；前缀/连续段 = 非减。
    damage = bpc_实验 − bpc_对照 = −Δ（正 = 变差）。"""
    dist = [{"d": 1, "damage_bpc": 0.30}, {"d": 2, "damage_bpc": 0.20},
            {"d": 4, "damage_bpc": 0.05}]
    assert monotonicity(dist, "d", direction="decreasing")["monotone"]
    assert not monotonicity(dist, "d", direction="increasing")["monotone"]
    bad = [{"N": 8, "damage_bpc": 0.30}, {"N": 16, "damage_bpc": 0.10},
           {"N": 32, "damage_bpc": 0.20}]
    res = monotonicity(bad, "N", direction="increasing")
    assert not res["monotone"]
    assert res["violations"][0]["damage_change"] == pytest.approx(-0.20)
    # tol 把噪点吞掉
    assert monotonicity(bad, "N", direction="increasing", tol=0.25)["monotone"]
