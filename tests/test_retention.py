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

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.depth_ar import DepthARConfig, DepthARModel
from pathlm.data import batch
from tools.evalkit.retention import (distance_keep_mask, prefix_keep_mask,
                                     segment_keep_mask, mask_stats, masked_logits,
                                     masked_bpc, row_bpc, eval_mask, safe_radius,
                                     monotonicity, recent_keep_mask,
                                     phrase_needle_batch, needle_fate,
                                     needle_recall, needle_summary, LN2,
                                     attention_stats, attention_stats_summary,
                                     priority_drop_mask, drop_mask_keep,
                                     drop_mask_overlap, drop_mask_stats,
                                     PRIORITY_RULES)

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


# ---------------------------------- 4. 检索腿（ticket 32 契约 M：保留配置 + 针召回）


def _needle_data(n=6000, n_real=48, seed=0):
    return np.random.RandomState(seed).randint(0, n_real, size=n).astype(np.uint16)


def test_recent_keep_mask_keeps_only_last_d_keys():
    """「只保留最近 d」：query i 恰好可见 min(d, i+1) 个 key（含对角）；d = T ≡ 因果。"""
    T = T_SMALL
    for d in (1, 2, 3, 5, T):
        keep = recent_keep_mask(T, d)
        i = torch.arange(T).unsqueeze(1)
        j = torch.arange(T).unsqueeze(0)
        expect = (j <= i) & ((i - j) < d)
        assert torch.equal(keep, expect | torch.eye(T, dtype=torch.bool))
        assert keep.diag().all()
        for q in range(T):
            assert int(keep[q].sum()) == min(d, q + 1)
        # 被删的 (i,j) 对：距离 >= d 且非对角
        dropped = (~keep & causal_keep(T)).nonzero()
        assert all(int(a) - int(b) >= d for a, b in dropped)
    assert torch.equal(recent_keep_mask(T, T), causal_keep(T))
    with pytest.raises(ValueError):
        recent_keep_mask(T, 0)
    with pytest.raises(ValueError):
        recent_keep_mask(T, T + 1)


def test_phrase_needle_layout_makes_the_plant_the_only_source():
    """needle v3：尾块 [mask, cue…] / 答案行 = cue 末尾；短语在窗口正文里**只出现一次**
    （关联唯一）；针的 L+1 个位置互不重叠（每根针只被放一次）；plant=False 与原批
    逐字段同一 generator 序列 ⇒ meta 完全相同（配对地板臂）。"""
    T, K, L = 48, 2, 4
    data = _needle_data(n_real=48)
    kw = dict(anchors=2, anchor_frac=0.5, n_needles=K, phrase_len=L)
    x, meta = phrase_needle_batch(3, T, data, 48, 48, torch.Generator().manual_seed(1), **kw)
    q0 = T - K * (L + 1)
    for b in range(3):
        assert len(meta[b]) == K
        ps = []
        for i, m in enumerate(meta[b]):
            assert m["q_row"] == q0 + i * (L + 1) + L
            assert int(x[b, m["q_row"] - L]) == 48          # 线索 = [mask]
            assert [int(v) for v in x[b, m["q_row"] - L + 1:m["q_row"] + 1]] == m["cue"]
            assert int(x[b, m["p"] + L]) == m["ans"]        # 副本里的答案
            assert m["ans"] < 48 and len(m["cue"]) == L
            ps.append(m["p"])
        assert len(set(ps)) == K                              # 位置不重复
        assert all(abs(a - c) >= L + 1 for k, a in enumerate(ps) for c in ps[k + 1:])
        for m in meta[b]:                                     # 关联唯一：正文里只有植入处
            ph = m["cue"] + [m["ans"]]
            hits = [s for s in range(q0 - L)
                    if [int(v) for v in x[b, s:s + L + 1]] == ph]
            assert hits == [m["p"]]
    x2, meta2 = phrase_needle_batch(3, T, data, 48, 48,
                                    torch.Generator().manual_seed(1), plant=False, **kw)
    assert meta2 == meta                                     # 同种子 ⇒ 同针位（配对）
    # 只有植入格不同（允许与填充文本巧合相等，所以只给上界）
    assert 0 < int((x != x2).sum()) <= 3 * K * (L + 1)
    with pytest.raises(ValueError):
        phrase_needle_batch(1, T, data, 48, 48, torch.Generator().manual_seed(0),
                            n_needles=40, phrase_len=L)      # 尾块放不下


def test_needle_fate_partitions_each_needle_exactly_once():
    """每根针恰好进一个桶（anchor / window / beyond），三桶计数之和 == 针数；
    全保留时没有 beyond。"""
    T, K, L = 48, 3, 4
    data = _needle_data(n_real=48)
    x, meta = phrase_needle_batch(2, T, data, 48, 48, torch.Generator().manual_seed(2),
                                  anchors=2, anchor_frac=0.5, n_needles=K, phrase_len=L)
    total = sum(len(m) for m in meta)
    fates = [f for b in range(2) for f in needle_fate(None, meta[b], anchors=2, phrase_len=L)]
    assert len(fates) == total
    assert all(f["bucket"] in {"anchor", "window", "beyond"} for f in fates)
    assert not any(f["bucket"] == "beyond" for f in fates)     # 全保留 ⇒ 无被删
    keep = prefix_keep_mask(T, 6)
    f2 = [f for b in range(2) for f in needle_fate(keep, meta[b], anchors=2, phrase_len=L)]
    counts = {k: sum(1 for f in f2 if f["bucket"] == k)
              for k in ("anchor", "window", "beyond")}
    assert sum(counts.values()) == total and len(f2) == total  # 不重不漏


def test_half_deleted_needle_counts_as_deleted_not_kept():
    """「针位置只被删一次」的语义：一根针的 L+1 个位置里**少任何一个**都不算「保住了」。
    只删掉短语中间一个位置的掩码必须把该针判成 beyond（用 any 而非 all 的写法会抓红）。"""
    T, L = 32, 3
    data = _needle_data(n_real=48, seed=3)
    x, meta = phrase_needle_batch(1, T, data, 48, 48, torch.Generator().manual_seed(3),
                                  anchors=0, anchor_frac=0.0, n_needles=2, phrase_len=L)
    assert all(m["p"] >= 0 for m in meta[0])
    keep = torch.ones(T, T, dtype=torch.bool).tril()
    assert all(f["bucket"] == "window" for f in needle_fate(keep, meta[0], anchors=0,
                                                             phrase_len=L))
    # 只抽掉第 0 根针短语中间的**一个**位置（其余位置仍可见）
    m0 = meta[0][0]
    keep2 = keep.clone()
    keep2[m0["q_row"], m0["p"] + L // 2] = False
    fates = needle_fate(keep2, meta[0], anchors=0, phrase_len=L)
    assert fates[0]["bucket"] == "beyond" and not fates[0]["visible"]
    assert all(f["bucket"] == "window" for f in fates[1:])
    # 全删该针的所有位置也只是 beyond（不会变成别的桶）
    m1 = meta[0][1]
    keep3 = keep.clone()
    for j in range(L + 1):
        keep3[m1["q_row"], m1["p"] + j] = False
    assert needle_fate(keep3, meta[0], anchors=0, phrase_len=L)[1]["bucket"] == "beyond"


def test_needle_recall_counts_every_needle_once_and_tracks_the_buckets():
    """needle_recall：n == 植入针数；recall == 逐针命中的均值；分桶计数相加 == n；
    掩码把全部针变成 beyond 时总召回 == beyond 桶召回（无泄漏）。"""
    T, K, L = 32, 2, 3
    m = tiny_model(d=16, L=2, H=2, T=T, vocab=49)
    data = _needle_data(n_real=48, seed=4)
    x, meta = phrase_needle_batch(2, T, data, 48, 48, torch.Generator().manual_seed(4),
                                  anchors=0, anchor_frac=0.0, n_needles=K, phrase_len=L)
    rec = needle_recall(m, x, meta, None, anchors=0, phrase_len=L, device="cpu")
    total = sum(len(r) for r in meta)
    assert rec["n"] == total
    assert rec["recall"] == pytest.approx(
        sum(p["hit"] for p in rec["per_needle"]) / total)
    assert sum(b["n"] for b in rec["buckets"].values()) == total
    assert set(rec["buckets"]) <= {"anchor", "window"}
    # 用一个把所有针删掉的掩码（每根针的位置对它自己的答案行不可见）
    keep = torch.ones(T, T, dtype=torch.bool).tril()
    for rows in meta:
        for mm in rows:
            keep[mm["q_row"], mm["p"]:mm["p"] + L + 1] = False
    rec2 = needle_recall(m, x, meta, keep, anchors=0, phrase_len=L, device="cpu")
    assert rec2["n"] == total and list(rec2["buckets"]) == ["beyond"]
    assert rec2["recall"] == rec2["buckets"]["beyond"]["recall"]
    assert needle_summary([])["recall"] is None


# ------------------------- 5. 优先级规则（ticket 32 契约 N：删哪些，预算相同）


@torch.no_grad()
def _hand_stats(model, x, recent_ks=(2, 6)):
    """手写参考：逐层逐头显式 softmax，独立复算契约 N 的统计量（不走被测实现）。"""
    B, T = x.shape
    d = model.cfg.d_model
    nh, dh = model.cfg.n_heads, d // model.cfg.n_heads
    mass = torch.zeros(B, T, dtype=torch.float64)
    cos = torch.zeros(B, T, dtype=torch.float64)
    kn = torch.zeros(B, T, dtype=torch.float64)
    rec = torch.zeros(B, T, dtype=torch.float64)
    recent = {k: torch.zeros(B, T, dtype=torch.float64) for k in recent_ks}
    diag = torch.zeros(B, T, dtype=torch.float64)
    h = model._input_states(x)
    for blk in model.blocks:
        q, k, v = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)
        qs = q.view(B, T, nh, dh).transpose(1, 2)
        ks = k.view(B, T, nh, dh).transpose(1, 2)
        for b in range(B):
            for hh in range(nh):
                for i in range(T):
                    w = torch.softmax(ks[b, hh, :i + 1] @ qs[b, hh, i]
                                      / dh ** 0.5, dim=-1)
                    for j in range(i + 1):
                        diag[b, j] += float(w[j]) / nh
                        if i > j:
                            mass[b, j] += float(w[j]) / nh
                            for kk in recent_ks:
                                if i >= T - kk:
                                    recent[kk][b, j] += float(w[j]) / nh
                            rec[b, j] += float(w[j]) / nh * i
        for b in range(B):
            for j in range(T):
                cnt = T - 1 - j
                if cnt > 0:
                    cs = sum(float(qs[b, hh, i] @ ks[b, hh, j]
                                   / (qs[b, hh, i].norm() * ks[b, hh, j].norm()))
                             for hh in range(nh) for i in range(j + 1, T))
                    cos[b, j] = cs / cnt / nh
                kn[b, j] = sum(float(ks[b, hh, j].norm()) for hh in range(nh)) / nh
        h, _ = blk(h)
    n_layer = len(model.blocks)
    mass, diag = mass / n_layer, diag / n_layer
    for kk in recent_ks:
        recent[kk] = recent[kk] / n_layer
    cos = cos / n_layer
    kn = kn / n_layer
    rec = torch.where(mass > 0, rec / mass.clamp_min(1e-12),
                      torch.arange(T).view(1, T).double())
    return {"mass_total": mass, "mass_total_incl_diag": diag, "cos_query": cos,
            "key_norm": kn, "soft_recency": rec,
            "mass_recent": {k: recent[k] for k in recent_ks}}


def test_attention_stats_match_manual_reference():
    """`attention_stats` 逐项等于手写参考（显式 softmax、排除自身读出、逐层逐头平均）。"""
    m = tiny_model(d=8, L=1, H=2, T=6, vocab=10)
    x = torch.randint(0, 10, (2, 6))
    st = attention_stats(m, x, recent_ks=(2, 6))
    ref = _hand_stats(m, x, recent_ks=(2, 6))
    for key in ("mass_total", "mass_total_incl_diag", "cos_query", "key_norm",
                "soft_recency"):
        assert torch.allclose(st[key].double(), ref[key], atol=1e-6), key
    for k in (2, 6):
        assert torch.allclose(st["mass_recent"][k].double(), ref["mass_recent"][k],
                              atol=1e-6), k
    # 含对角线的质量 = 排除对角线的质量 + 对角权重（逐位置 ≥）
    assert bool((st["mass_total_incl_diag"] >= st["mass_total"] - 1e-6).all())
    # 最近 T 个查询 ≡ 全窗口
    assert torch.allclose(st["mass_recent"][6], st["mass_total"], atol=1e-6)
    assert st["meta"]["recent_ks"] == [2, 6]


def test_attention_stats_are_causal_and_normalized():
    """统计量的因果性与归一化闭式：最后一个位置**收不到任何质量**（排除对角线后
    没有 query 在它后面）；含对角线的质量逐窗口求和恰为 1（每行注意力归一）；
    `mass_recent[k]` 在 k 上单调不减且 ≤ `mass_total`。变异（把因果掩码改成全 1）
    会被最后位置的不变量与手写参考一起抓红。"""
    m = tiny_model(d=8, L=2, H=2, T=8, vocab=10)
    x = torch.randint(0, 10, (2, 8))
    st = attention_stats(m, x, recent_ks=(3, 8))
    assert torch.allclose(st["mass_total"][:, -1], torch.zeros(2), atol=1e-9)
    assert torch.allclose(st["mass_recent"][3][:, -1], torch.zeros(2), atol=1e-9)
    assert torch.allclose(st["soft_recency"][:, -1], torch.full((2,), 7.0), atol=1e-5)
    # 含对角线：每一行注意力归一 ⇒ 逐窗口的（位置平均）恰好 = 1
    assert torch.allclose(st["mass_total_incl_diag"].double().mean(1),
                          torch.ones(2, dtype=torch.float64), atol=1e-5)
    # 排除对角线后每个位置的上界 = 它后面有多少个 query
    cap = torch.arange(8, 0, -1).double() - 1.0            # j: T-1-j
    assert bool((st["mass_total"].double() <= cap.view(1, 8) + 1e-5).all())
    assert bool((st["mass_recent"][3].double() <= st["mass_total"].double() + 1e-6)
                .all())
    assert bool((st["mass_recent"][3].double() <= st["mass_recent"][8].double()
                 + 1e-6).all())                            # k 越大窗口越大
    assert bool((st["cos_query"].abs() <= 1.0 + 1e-5).all())
    assert bool((st["soft_recency"] >= torch.arange(8).float().view(1, 8) - 1e-5).all())
    # 仪器活着：换输入 ⇒ 统计量变化（否则无区分度）
    st2 = attention_stats(m, (x + 1) % 10, recent_ks=(3, 8))
    assert not torch.allclose(st["mass_total"], st2["mass_total"], atol=1e-6)


def _fake_stats(B, T, seed=0):
    """构造统计量（不跑模型）：让三条规则的排序互不相同，用于钉住规则语义。"""
    g = torch.Generator().manual_seed(seed)
    return {"mass_total": torch.rand(B, T, generator=g),
            "mass_total_incl_diag": torch.zeros(B, T),
            "cos_query": torch.rand(B, T, generator=g),
            "key_norm": torch.rand(B, T, generator=g),
            "soft_recency": torch.rand(B, T, generator=g),
            "mass_recent": {64: torch.rand(B, T, generator=g)},
            "meta": {"T": T}}


def test_priority_rules_spend_the_same_budget_and_are_deterministic():
    """四条规则在**同一个** n_drop 下删**同样多的位置**（逐窗口恰好 n_drop 个、无重复），
    且完全确定（同输入 ⇒ 同删除集）。"""
    B, T, N = 3, 32, 5
    st = _fake_stats(B, T)
    for rule in PRIORITY_RULES:
        d1 = priority_drop_mask(st, N, rule, k=64)
        d2 = priority_drop_mask(st, N, rule, k=64)
        assert torch.equal(d1, d2)                     # 确定性（含平局次序）
        assert d1.shape == (B, T) and d1.dtype == torch.bool
        assert [int(v) for v in d1.sum(-1)] == [N] * B  # 预算 = 位置个数
    d = priority_drop_mask(st, N, "distance", k=64)
    assert [int(v) for v in d[0].nonzero().flatten()] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        priority_drop_mask(st, N, "nope")
    with pytest.raises(ValueError):
        priority_drop_mask(st, T + 1, "lru")
    with pytest.raises(ValueError):
        priority_drop_mask(st, N, "lru", k=128)        # 统计里没有 mass_recent[128]


def test_priority_rules_delete_by_their_own_key():
    """规则语义逐条钉死（含平局次序）：lru = 最近 k 步质量最小、content = 与查询最不相似、
    lastused = 加权平均查询下标最小。"""
    T, N = 8, 2
    st = {"mass_total": torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]]),
          "mass_total_incl_diag": torch.zeros(1, T),
          "cos_query": torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.3, 0.8, 0.7, 0.6]]),
          "key_norm": torch.tensor([[1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
          "soft_recency": torch.tensor([[3.0, 7.0, 1.0, 6.0, 5.0, 4.0, 2.0, 8.0]]),
          "mass_recent": {2: torch.tensor([[0.2, 0.0, 0.4, 0.0, 0.1, 0.3, 0.0, 0.5]])},
          "meta": {"T": T}}
    got = lambda rule: [int(v) for v in priority_drop_mask(st, N, rule, k=2).nonzero()[:, 1]]
    assert got("distance") == [0, 1]
    # lru(k=2)：质量 [0.2,0,0.4,0,0.1,0.3,0,0.5] → 最小的是 {1,3,6}（都 0），
    # 次级键 mass_total 最小者 6(3.0) → 再取 3(6.0)
    assert got("lru") == [3, 6]
    # content：cos 最小 1(0.1) → 3(0.2)（次级键 key_norm 只在 cos 相等时用）
    assert got("content") == [1, 3]
    # lastused：加权平均查询下标最小 2(1.0) → 6(2.0)
    assert got("lastused") == [2, 6]
    # content 的另一种内容读数：按 key 范数最小（契约允许）。范数最小的是 {0,2,3,4,5,6,7}
    # （都 1.0），次级键换成 cos_query 升序 ⇒ 3(0.2) → 4(0.3)
    kn = {2: torch.tensor([[1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
          "cos_query": st["cos_query"], "key_norm": st["key_norm"],
          "mass_total": st["mass_total"], "soft_recency": st["soft_recency"],
          "mass_recent": st["mass_recent"], "mass_total_incl_diag": st["mass_total_incl_diag"],
          "meta": st["meta"]}
    got_kn = priority_drop_mask(kn, N, "content", k=2, key="key_norm").nonzero()[:, 1]
    assert [int(v) for v in got_kn] == [3, 4]
    with pytest.raises(ValueError):
        priority_drop_mask(kn, N, "lru", k=2, key="key_norm")
    with pytest.raises(ValueError):
        priority_drop_mask(kn, N, "content", k=2, key="mass_total")


def test_drop_mask_keep_matches_the_prefix_baseline_and_never_empties_a_row():
    """逐行同一删除集时 `drop_mask_keep` 与 `prefix_keep_mask` **逐位一致**——
    即契约 N 的 `distance` 规则就是现基线语义；对角线恒保留（否则该行 NaN）。"""
    T, N = 12, 4
    drop = torch.zeros(3, T, dtype=torch.bool)
    drop[:, :N] = True
    keep = drop_mask_keep(drop)
    base = prefix_keep_mask(T, N)
    assert torch.equal(keep[0], base) and torch.equal(keep[2], base)
    assert bool(keep.diagonal(dim1=1, dim2=2).all())
    assert bool(keep.any(-1).all())                       # 每一行至少看得见自己
    # 逐样本不同的删除集：只影响被删的那一列
    drop2 = torch.zeros(1, T, dtype=torch.bool)
    drop2[0, 7] = True
    k2 = drop_mask_keep(drop2)
    assert not bool(k2[0, 8:, 7].any())            # 后面的 query 看不见位置 7
    assert not bool(k2[0, :7, 7].any())             # 前面的 query 本来就看不见 7
    assert bool(k2[0, 7, 7])                        # 对角线恒保留
    assert bool(k2[0, 7, 7])                              # 自己被删但保留对角线
    with pytest.raises(ValueError):
        drop_mask_keep(torch.zeros(T, dtype=torch.bool))
    # logits 层：逐样本掩码 → 每个窗口只受自己那套删除集影响
    m = tiny_model(d=8, L=1, H=2, T=T, vocab=10)
    x = torch.randint(0, 10, (2, T))
    lg_all_shared = masked_logits(m, x, prefix_keep_mask(T, N))
    lg_per_sample = masked_logits(m, x, drop_mask_keep(drop[:2]))
    assert torch.equal(lg_all_shared, lg_per_sample)
    with pytest.raises(ValueError):
        masked_logits(m, x, torch.ones(7, T, T, dtype=torch.bool))


def test_priority_candidates_keep_the_budget_and_only_shrink_the_pool():
    """`candidates` 只改**候选池**，不改预算：删的仍是 n_drop 个位置、全落在候选集内；
    候选 = 全集时逐位等于不传（诊断臂「线索块不参与压缩」靠它）。"""
    B, T, N = 2, 12, 3
    st = _fake_stats(B, T, seed=11)
    for rule in PRIORITY_RULES:
        full = priority_drop_mask(st, N, rule, k=64)
        same = priority_drop_mask(st, N, rule, k=64,
                                  candidates=torch.ones(T, dtype=torch.bool))
        assert torch.equal(full, same)
    cand = torch.zeros(T, dtype=torch.bool)
    cand[T // 2:] = True
    d = priority_drop_mask(st, N, "distance", k=64, candidates=cand)
    assert [int(v) for v in d[0].nonzero().flatten()] == [T // 2, T // 2 + 1, T // 2 + 2]
    for rule in ("lru", "content", "lastused"):
        dd = priority_drop_mask(st, N, rule, k=64, candidates=cand)
        assert [int(v) for v in dd.sum(-1)] == [N] * B
        assert not bool(dd[:, :T // 2].any())
    with pytest.raises(ValueError):
        priority_drop_mask(st, N, "lru", k=64, candidates=torch.ones(T - 1,
                                                                    dtype=torch.bool))
    with pytest.raises(ValueError):                     # 预算超过候选池
        priority_drop_mask(st, T // 2 + 1, "content", k=64, candidates=cand)


def test_drop_mask_stats_matches_mask_stats_on_a_shared_drop_set():
    """`drop_mask_stats` 与 `mask_stats` 同口径：逐行同一删除集时，两者报的
    被删位置数与被删因果对数**完全一致**（预算按位置对齐、代价按可见性对数另报）。"""
    T = 20
    for N in (3, 8):
        drop = torch.zeros(4, T, dtype=torch.bool)
        drop[:, :N] = True
        st = drop_mask_stats(drop)
        ref = mask_stats(prefix_keep_mask(T, N))
        assert st["n_dropped_keys_mean"] == ref["n_dropped_keys"]
        assert st["n_dropped_pairs_mean"] == pytest.approx(ref["n_dropped_pairs"])
        assert st["dropped_pair_frac_mean"] == pytest.approx(ref["dropped_frac"])
        assert st["n_windows"] == 4
    # 预算相同（都是 3 个位置）但位置不同 ⇒ 可见性对数不同（不是同一个代价）
    a = torch.zeros(1, T, dtype=torch.bool)
    a[0, :3] = True
    b = torch.zeros(1, T, dtype=torch.bool)
    b[0, T - 3:] = True
    assert (drop_mask_stats(a)["n_dropped_keys_mean"]
            == drop_mask_stats(b)["n_dropped_keys_mean"] == 3)
    assert drop_mask_stats(a)["n_dropped_pairs_mean"] > \
        drop_mask_stats(b)["n_dropped_pairs_mean"]
    with pytest.raises(ValueError):
        drop_mask_stats(torch.zeros(T, dtype=torch.bool))


def test_deleted_set_overlap_reports_intersection_and_jaccard():
    """重叠度：交集大小 / 并集 / Jaccard / 完全相同的窗口数（契约 N 要求报重叠度）。"""
    T = 8
    a = torch.zeros(2, T, dtype=torch.bool)
    b = torch.zeros(2, T, dtype=torch.bool)
    a[0, :3] = True
    b[0, 1:5] = True                     # 交集 {1,2}，并集 5 个 → J = 0.4
    a[1, :3] = True
    b[1, :3] = True                      # 完全相同 → J = 1
    ov = drop_mask_overlap(a, b)
    assert ov["n_windows"] == 2 and ov["n_identical_windows"] == 1
    assert ov["mean_intersection"] == pytest.approx(2.5)
    assert ov["mean_union"] == pytest.approx(4.0)
    assert ov["mean_jaccard"] == pytest.approx((0.4 + 1.0) / 2)
    assert drop_mask_overlap(a, a)["mean_jaccard"] == 1.0
    with pytest.raises(ValueError):
        drop_mask_overlap(a, torch.zeros(3, T, dtype=torch.bool))


def test_priority_rules_move_the_outcome_liveness():
    """干预活性（SCHEMA §10）：三条规则在同一批统计下删出的集合**确实不同**
    （Jaccard < 1），且逐样本掩码真的改变 logits —— 否则「规则无效」与「机制无效」
    无法区分。"""
    B, T, N = 2, 16, 4
    st = _fake_stats(B, T, seed=7)
    drops = {r: priority_drop_mask(st, N, r, k=64) for r in
             ("distance", "lru", "content", "lastused")}
    for a in drops:
        for b in drops:
            if a < b:
                ov = drop_mask_overlap(drops[a], drops[b])
                assert ov["mean_jaccard"] < 1.0, (a, b)
                assert ov["mean_intersection"] < N, (a, b)
    m = tiny_model(d=8, L=1, H=2, T=T, vocab=10)
    x = torch.randint(0, 10, (B, T))
    base = masked_logits(m, x, None)
    assert not torch.allclose(base, masked_logits(m, x, drop_mask_keep(drops["lru"])))


def test_attention_stats_summary_reports_the_distribution():
    """分布报告：分箱均值与逐窗口均值一致、分位数有序、热点位置是质量最大的那些。"""
    B, T = 3, 16
    st = _fake_stats(B, T, seed=1)
    st["mass_recent"] = {4: torch.rand(B, T, generator=torch.Generator().manual_seed(2))}
    st["meta"] = {"n_windows": B, "n_layers": 1, "T": T,
                  "layer_mean_mass_recent_bins": {}}
    s = attention_stats_summary(st, ks=(4,), bins=((0, 4), (4, 16)), top_n=3)
    mean_mass = st["mass_total"].double().mean(0)
    assert s["mass_total"]["by_bin"]["[0,4)"] == pytest.approx(
        float(mean_mass[:4].mean()), abs=1e-6)
    assert s["mass_total"]["by_bin"]["[4,16)"] == pytest.approx(
        float(mean_mass[4:].mean()), abs=1e-6)
    q = s["mass_total"]["quantiles"]
    assert q["min"] <= q["p25"] <= q["median"] <= q["p75"] <= q["max"]
    assert len(s["mass_total"]["profile_per_window_mean"]) == T
    top = [h["pos"] for h in s["mass_total"]["top_positions"]]
    assert top == [int(i) for i in torch.argsort(mean_mass, descending=True)[:3]]
    assert 0.0 <= s["mass_total"]["zero_frac"] <= 1.0
    assert "[0,4)" in s["mass_recent"]["4"]["by_bin"]


def test_per_sample_masks_reach_their_own_window_in_bpc_and_needles():
    """逐样本掩码的管路：`mask_fn` / `[B,T,T]` 的每一行只作用于**同一下标**的窗口
    （bpc 与针召回两条腿都钉）。全行同一删除集时必须与共享掩码**逐位相同**。"""
    T, L, K = 16, 2, 1
    m = tiny_model(d=8, L=1, H=2, T=T, vocab=49)
    data = _needle_data(n_real=48, seed=5)
    x, meta = phrase_needle_batch(2, T, data, 48, 48, torch.Generator().manual_seed(5),
                                  anchors=0, anchor_frac=0.0, n_needles=K, phrase_len=L)
    prefix = prefix_keep_mask(T, 3)
    same_drop = torch.zeros(2, T, dtype=torch.bool)
    same_drop[:, :3] = True
    rec_shared = needle_recall(m, x, meta, prefix, anchors=0, phrase_len=L, device="cpu")
    rec_per = needle_recall(m, x, meta, drop_mask_keep(same_drop), anchors=0,
                            phrase_len=L, device="cpu")
    assert rec_shared == rec_per                  # 逐行同一删除集 ⇒ 与共享掩码同结果
    # 只删第 0 行那根针的短语位置：第 0 行判被删，第 1 行不受影响
    drop = torch.zeros(2, T, dtype=torch.bool)
    p0 = meta[0][0]["p"]
    drop[0, p0:p0 + L + 1] = True
    rec2 = needle_recall(m, x, meta, drop_mask_keep(drop), anchors=0, phrase_len=L,
                         device="cpu")
    assert rec2["n"] == sum(len(r) for r in meta)
    assert rec2["per_needle"][0]["bucket"] == "beyond"        # 第 0 行的针被删
    assert rec2["per_needle"][1]["bucket"] == "window"        # 第 1 行不受影响
    # bpc 管路：mask_fn 下的逐样本删除集与共享掩码在「同一删除集」时逐位一致
    arr = np.random.RandomState(7).randint(0, 48, size=4000).astype(np.uint16)
    b1 = masked_bpc(m, arr, prefix, n_batches=2, batch_size=2, window_seed=1,
                    device="cpu")
    b2 = masked_bpc(m, arr, n_batches=2, batch_size=2, window_seed=1, device="cpu",
                    mask_fn=lambda xx: drop_mask_keep(
                        torch.cat([torch.ones(xx.shape[0], 3, dtype=torch.bool),
                                   torch.zeros(xx.shape[0], T - 3, dtype=torch.bool)],
                                  dim=1)))
    assert b1 == b2
    with pytest.raises(ValueError):
        masked_bpc(m, arr, prefix, n_batches=1, batch_size=2, device="cpu",
                   mask_fn=lambda xx: prefix)
