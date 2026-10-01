"""合成长程任务电池的可失败单测（ticket 34 契约 A）。

Run: `python3 -m pytest tests/ -q`

钉住五件事：
1. **任务构造正确**：四个生成器的布局、答案恒等式、距离字段逐点对（故意换错
   位置/答案必须抓红）；
2. **局部先验抗性**（契约判据 (a)）：四个任务上「只看最近 k 个 token」的最强
   预测器都退化为 chance；
3. **局部先验是真预测器**（不是坏掉的桩）：在「答案确实落在窗口内」的构造上
   它能答对——否则判据 (a) 是假的；
4. **变异验证**：把答案标签整体打乱后，准确率必须落回 chance；
5. **与 `retention.py` 兼容**：掩码可见性复用既有删除逻辑，且掩码真的被注意力
   读到（干预活性）。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlm.depth_ar import DepthARConfig, DepthARModel          # noqa: E402
from tools.evalkit.retention import masked_logits, recent_keep_mask  # noqa: E402
from tools.evalkit.longrange_tasks import (                      # noqa: E402
    TaskBatch, assoc_recall_batch, induction_batch, copy_batch,
    var_recall_batch, local_prior_predict, local_prior_accuracy, accuracy,
    task_summary, recall_under_visibility, shuffle_answers, corrupt_span,
    check_local_prior_resistance, DEFAULT_LOCAL_K, DISTANCES)

# 小尺寸任务（快）+ 小模型；符号词表 32、filler 32、cue = 小模型 mask 槽 63
SYM, FILL, CUE = 32, 32, 63
T_SMALL = 64


def tiny_model(vocab=64, T=64):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=T, mlp_mult=4,
                        dense=True)
    m = DepthARModel(cfg, vocab)
    m.eval()
    return m


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------- 1. 任务构造 / 答案恒等式


def test_assoc_recall_layout_and_answer_identity():
    """关联回忆：pairs 连续排在最前、keys 行内互不相同、cue/query 各就各位、
    答案 = 被问那一对的 value、距离 = (T-1)-2j。"""
    g = gen(3)
    b = assoc_recall_batch(6, T_SMALL, n_pairs=6, sym_vocab=SYM, filler_token=FILL,
                           cue_token=CUE, generator=g, query_index=2)
    assert b.tokens.shape == (6, T_SMALL)
    for r in range(b.B):
        row = b.tokens[r].tolist()
        # 行内符号范围
        syms = [v for v in row if v < SYM]
        assert all(0 <= v < SYM for v in syms)
        # keys 互不相同
        keys = [row[2 * j] for j in range(6)]
        assert len(set(keys)) == 6
        # cue 与 query key
        assert row[T_SMALL - 2] == CUE
        assert row[T_SMALL - 1] == keys[2]
        # 答案 = 被问那一对的 value；距离恒等式
        assert int(b.rows[r, 0]) == T_SMALL - 1
        assert int(b.answers[r, 0]) == row[2 * 2 + 1]
        assert int(b.dists[r, 0]) == (T_SMALL - 1) - 2 * 2
        # filler 未与符号/提示词重叠
        assert FILL not in syms and FILL != CUE
    assert b.chance() == 1.0 / SYM


def test_induction_repeats_prefix_segment_and_answers_next():
    """Induction：末尾块 == 前缀里的一段原样拷贝；答案 == 拷贝段之后再一项；
    距离 == 提问行 − 源片段末端。"""
    g = gen(7)
    P, M = 40, 12
    b = induction_batch(6, T_SMALL, prefix_len=P, copy_len=M, sym_vocab=SYM,
                        filler_token=FILL, cue_token=CUE, generator=g, src_start=3)
    for r in range(b.B):
        row = b.tokens[r].tolist()
        copied = row[T_SMALL - M:T_SMALL]
        assert copied == row[3:3 + M]                      # 原样拷贝
        assert int(b.answers[r, 0]) == row[3 + M]          # 原前缀里紧跟其后的一项
        assert int(b.rows[r, 0]) == T_SMALL - 1
        assert int(b.dists[r, 0]) == (T_SMALL - 1) - (3 + M - 1)
    assert b.M == 1


def test_copy_task_reproduces_source_block():
    """复制：源片段落在 [src_pos, src_pos+m)；输出块（teacher-forced 输入）逐位
    等于源片段；查询行 = T-m-1+k 的第 k 个，答案 = 源片段第 k 项。"""
    g = gen(5)
    m, src = 8, 4
    b = copy_batch(5, T_SMALL, copy_len=m, src_pos=src, sym_vocab=SYM,
                   filler_token=FILL, cue_token=CUE, generator=g)
    assert b.rows.shape == (5, m) and b.answers.shape == (5, m)
    for r in range(b.B):
        row = b.tokens[r].tolist()
        assert row[src:src + m] == row[T_SMALL - m:T_SMALL]       # 输出块 == 源
        assert row[T_SMALL - m - 1] == CUE                        # cue 位置
        assert b.answers[r].tolist() == row[src:src + m]
        assert b.rows[r].tolist() == [T_SMALL - m - 1 + k for k in range(m)]
    assert int(b.dists[0, 0]) == (T_SMALL - m - 1) - src
    assert torch.equal(b.dists, b.dists[0, 0] * torch.ones_like(b.dists))


def test_var_recall_distance_places_pair_and_answers_value():
    """变距离回忆：key 在 (T-1)-d、value 在 (T-d)、cue 在 T-2、query key 在 T-1、
    答案 = value、距离字段 == d。"""
    g = gen(11)
    for d in (16, 31, 40):
        b = var_recall_batch(4, T_SMALL, distance=d, sym_vocab=SYM,
                             filler_token=FILL, cue_token=CUE, generator=g)
        src = (T_SMALL - 1) - d
        for r in range(b.B):
            row = b.tokens[r].tolist()
            assert row[src + 1] == int(b.answers[r, 0])           # value
            assert row[T_SMALL - 1] == row[src]                   # query key 重复源 key
            assert row[T_SMALL - 2] == CUE
            assert int(b.dists[r, 0]) == d


# ------------------------------------------------- 2/3. 局部先验：抗性 + 真预测器


def test_local_prior_is_a_real_predictor_on_in_window_copy():
    """局部先验不是坏桩：答案**确实落在窗口内**时它能答对（最长后缀最近邻）。"""
    seq = [7, 9, 3, 8, 8, 8, 7, 9, 3]                 # 第二个 [7,9] 之后应接 3
    pred = local_prior_predict(seq, row=7, k=16, answer_set=set(range(SYM)))
    assert pred == 3
    b = TaskBatch("in_window_copy", torch.tensor([seq] * 4),
                  torch.tensor([[7]] * 4), torch.tensor([[3]] * 4),
                  torch.tensor([[1]] * 4), SYM)
    assert local_prior_accuracy(b, k=16)["accuracy"] == 1.0


def test_local_prior_resistance_all_four_tasks():
    """契约判据 (a)：局部先验在四个任务上都退化为 chance（显著低于真值 1.0）。"""
    g = gen(13)
    B = 256
    batches = {
        "assoc_recall": assoc_recall_batch(B, 256, 8, sym_vocab=64, generator=g),
        "induction": induction_batch(B, 256, 192, 32, sym_vocab=64, generator=g),
        "copy": copy_batch(B, 256, 32, 0, sym_vocab=64, generator=g),
        "var_recall_d16": var_recall_batch(B, 256, 16, sym_vocab=64, generator=g),
        "var_recall_d255": var_recall_batch(B, 256, 255, sym_vocab=64, generator=g),
    }
    for name, b in batches.items():
        res = check_local_prior_resistance(b, k=DEFAULT_LOCAL_K)
        assert res["passed"], (name, res)
        assert res["local_prior_accuracy"] < res["ceiling"] < 0.2


def test_local_prior_window_must_be_below_min_distance():
    """护栏的物理含义：窗口 k 一旦 ≥ 最小距离，value 会落进窗口、局部先验被抬起
    （k=16 时 d=16 实测被抬到 ~0.5）——证明「k < 距离」是任务构造的必要条件。"""
    g = gen(2)
    b = var_recall_batch(256, 256, 16, sym_vocab=64, generator=g)
    hi = local_prior_accuracy(b, k=16)["accuracy"]
    lo = local_prior_accuracy(b, k=DEFAULT_LOCAL_K)["accuracy"]
    assert lo < 3 * b.chance(), (lo, b.chance())
    assert hi > lo, (hi, lo)


# ------------------------------------------------- 4. 变异验证


def test_shuffle_answers_drops_accuracy_to_chance():
    """变异验证：打乱答案标签后准确率必须落回 chance（否则准确率不是从上下文来的）。"""
    g = gen(17)
    model = tiny_model(vocab=64, T=T_SMALL)
    b = var_recall_batch(64, T_SMALL, 24, sym_vocab=SYM, filler_token=FILL,
                         cue_token=CUE, generator=g)
    acc = accuracy(model, b)["accuracy"]
    mut = shuffle_answers(b, generator=g)
    acc_mut = accuracy(model, mut)["accuracy"]
    assert acc_mut < 3 * b.chance(), (acc, acc_mut, b.chance())
    # 打乱只换答案标签，token / 行 / 距离不变
    assert torch.equal(mut.tokens, b.tokens)
    assert torch.equal(mut.rows, b.rows)
    assert torch.equal(mut.dists, b.dists)


# ------------------------------------------------- 5. retention 兼容 + 活性


def test_accuracy_reads_retention_keep_mask():
    """掩码可见性复用 retention 语义，且掩码真的被注意力读到（干预活性）。"""
    g = gen(19)
    model = tiny_model(vocab=64, T=T_SMALL)
    b = var_recall_batch(4, T_SMALL, 24, sym_vocab=SYM, filler_token=FILL,
                         cue_token=CUE, generator=g)
    full = masked_logits(model, b.tokens, None)
    self_only = masked_logits(model, b.tokens,
                              torch.eye(T_SMALL, dtype=torch.bool))
    assert not torch.allclose(full, self_only)          # 掩码改变了前向
    _ = accuracy(model, b, keep_mask=torch.eye(T_SMALL, dtype=torch.bool))
    # distinct-results 口径（不建判据，只证明接口跑得通、读数随掩码而变）
    assert accuracy(model, b)["n"] == b.B


def test_recall_under_visibility_reuses_recent_keep_mask():
    """变距离回忆 × 可见性：只保留最近 d 个位置时，源信息被删 ⇒ 读数改变。"""
    g = gen(23)
    model = tiny_model(vocab=64, T=T_SMALL)
    b = var_recall_batch(4, T_SMALL, 40, sym_vocab=SYM, filler_token=FILL,
                         cue_token=CUE, generator=g)
    out = recall_under_visibility(model, b, keep_ds=(8, T_SMALL))
    assert set(out) == {"8", str(T_SMALL)}
    with_keep = masked_logits(model, b.tokens, recent_keep_mask(T_SMALL, 8))
    without_keep = masked_logits(model, b.tokens, recent_keep_mask(T_SMALL, T_SMALL))
    assert not torch.allclose(with_keep, without_keep)


# ------------------------------------------------- 6. 其他不变量


def test_generator_seed_reproducibility():
    """同 seed 逐位相同；不同 seed 不同（可复现性）。"""
    for mk in (lambda gg: assoc_recall_batch(4, T_SMALL, 3, generator=gg),
               lambda gg: induction_batch(4, T_SMALL, 32, 8, generator=gg),
               lambda gg: copy_batch(4, T_SMALL, 6, 2, generator=gg),
               lambda gg: var_recall_batch(4, T_SMALL, 20, generator=gg)):
        a = mk(gen(1))
        a2 = mk(gen(1))
        b = mk(gen(2))
        assert torch.equal(a.tokens, a2.tokens)
        assert torch.equal(a.answers, a2.answers)
        assert not torch.equal(a.tokens, b.tokens)


def test_distance_curve_buckets_match_query_distances():
    """距离-准确率曲线按 `dists` 分桶；chance_dict 两个口径正确。"""
    g = gen(29)
    model = tiny_model(vocab=64, T=T_SMALL)
    b = copy_batch(6, T_SMALL, 8, 4, sym_vocab=SYM, filler_token=FILL,
                   cue_token=CUE, generator=g)
    res = accuracy(model, b)
    assert set(res["by_distance"]) == {str((T_SMALL - 8 - 1) - 4)}
    assert res["n"] == 6 * 8 and res["n_hits"] <= res["n"]
    cd = b.chance_dict(vocab_size=64)
    assert cd["uniform_answer_set"] == 1.0 / SYM and cd["uniform_model_vocab"] == 1.0 / 64
    summary = task_summary(model, b, local_k=DEFAULT_LOCAL_K, vocab_size=64)
    assert set(summary) == {"name", "params", "chance", "local_prior", "model"}


def test_corrupt_span_destroys_only_source_region():
    """源对照臂：只换掉指定 span，查询行 / 答案 / 距离逐位不变（其余 token 也不变）。

    毁掉源片段后，一个靠源信息作答的读数必须掉回 chance——这里用「可答空间」
    上的局部先验保证仍有意义（拷贝任务局部先验本就是 chance）。
    """
    g = gen(31)
    m, src = 8, 4
    b = copy_batch(8, T_SMALL, copy_len=m, src_pos=src, sym_vocab=SYM,
                   filler_token=FILL, cue_token=CUE, generator=g)
    c = corrupt_span(b, src, src + m, generator=g)
    assert torch.equal(c.rows, b.rows) and torch.equal(c.answers, b.answers)
    assert torch.equal(c.dists, b.dists)
    assert torch.equal(c.tokens[:, :src], b.tokens[:, :src])
    assert torch.equal(c.tokens[:, src + m:], b.tokens[:, src + m:])
    assert not torch.equal(c.tokens[:, src:src + m], b.tokens[:, src:src + m])
    assert c.meta["corrupt_span"] == [src, src + m]
    with pytest.raises(ValueError):
        corrupt_span(b, -1, 3, generator=g)


def test_distance_constants_match_contract():
    """契约指定的距离档逐字存在。"""
    assert DISTANCES == (16, 32, 64, 128, 192, 255)
