"""2632 · 可失败单测（放在本目录，不写 tests/）。

钉住本诊断里三个「会静默出错」的语义点（SCHEMA §10 活性 / methodology §5 变异）：
1. `chain2` 的显式 condition 接线真的改变输出（条件活的）；
2. 「显式喂 v1」与「deploy 分支（condition=None 自取 node1 argmax）」逐元素相同
   （否则 chain_deploy 与 chain_oracle 的分叉是接线 bug，不是条件效应）；
3. 混合 / 分层 / top-k 召回三个统计量的定义与手算一致。
"""

from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import diag  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402
from pathlm.decode import Decoder  # noqa: E402


def tiny_model(n_mtp: int = 3):
    torch.manual_seed(0)
    m = PathLM(ModelConfig(d_model=16, n_layers=2, n_heads=2, seq_len=32, mlp_mult=2),
               PathConfig(n_mtp=n_mtp, chain_mode="token"), vocab_size=20)
    return m.eval()


def test_condition_is_live_different_conditions_differ():
    """条件接线活的：换 condition token，chain2 logits 必须变（否则是死干预）。"""
    m = tiny_model()
    h = torch.randn(1, 1, 16)
    a = m._mtp_nodes(h, condition=torch.tensor([[1]]))[1]["chain2"]["logits"]
    b = m._mtp_nodes(h, condition=torch.tensor([[7]]))[1]["chain2"]["logits"]
    assert not torch.allclose(a, b), "condition 未生效（死干预）"


def test_explicit_v1_matches_deploy_branch():
    """显式 condition=v1 与 deploy（condition=None）必须逐元素相同。"""
    m = tiny_model()
    h = torch.randn(1, 1, 16)
    nodes = m._mtp_nodes(h)                       # condition=None → deploy 分支
    v1 = nodes[1]["logits"].argmax(-1)            # [1, 1]
    nodes2 = m._mtp_nodes(h, condition=v1)
    assert torch.allclose(nodes[1]["chain2"]["logits"], nodes2[1]["chain2"]["logits"])


def test_condition_requires_explicit_and_not_overridden_by_deploy_chain():
    """确认模型默认 deploy_chain=False，否则显式 condition 会被静默覆盖。"""
    m = tiny_model()
    assert m.deploy_chain is False


def test_mean_eq_and_topk_recall_definitions():
    rows = [
        {"seg": 0, "d_direct": 3, "true_t2": 5, "t2dep": 4, "topk_direct": [4, 9, 2]},
        {"seg": 0, "d_direct": 4, "true_t2": 4, "t2dep": 4, "topk_direct": [1, 2, 3]},
        {"seg": 1, "d_direct": 1, "true_t2": 1, "t2dep": 7, "topk_direct": [7, 8, 9]},
    ]
    assert diag._mean_eq(rows, "d_direct", "t2dep") == pytest.approx(1 / 3)
    assert diag._mean_eq(rows, "d_direct", "true_t2") == pytest.approx(2 / 3)
    assert diag._topk_recall(rows, "topk_direct", 1) == pytest.approx(2 / 3)
    assert diag._topk_recall(rows, "topk_direct", 3) == pytest.approx(2 / 3)
    # k 敏感性：把 row2 的 t2dep 换成 8（top3 含 8，top1 不含）
    rows2 = [dict(r) for r in rows]
    rows2[2]["t2dep"] = 8
    assert diag._topk_recall(rows2, "topk_direct", 1) == pytest.approx(1 / 3)
    assert diag._topk_recall(rows2, "topk_direct", 3) == pytest.approx(2 / 3)


def test_strata_split_is_by_v1_correctness():
    rows = [
        {"seg": 0, "v1": 1, "true_next": 1, "d_chain": 2, "true_t2": 2, "t2dep": 2},
        {"seg": 0, "v1": 1, "true_next": 5, "d_chain": 2, "true_t2": 2, "t2dep": 2},
    ]
    e1 = [r for r in rows if r["v1"] == r["true_next"]]
    e0 = [r for r in rows if r["v1"] != r["true_next"]]
    assert len(e1) == 1 and len(e0) == 1


def test_teacher_anchor_reconstructs_four_drafts_and_targets():
    """从微型模型独立重推四种草稿 + t2dep/t2teacher，钉住 claim-critical 接线。"""
    m = tiny_model()
    val = np.random.RandomState(0).randint(0, 19, size=32).astype(np.int64)
    rows = diag.teacher_anchor_pass(m, m.pcap, val, offsets=[0], length=4)
    dec = Decoder(m)
    h, _ = dec.step(int(val[0]))
    nodes = dec.last_nodes
    v1 = int(nodes[1]["logits"][0, 0].argmax())
    dd = int(nodes[2]["logits"][0, 0].argmax())
    lc = nodes[1]["chain2"]["logits"][0, 0].float()
    dc = int(lc.argmax())
    pc, pd = lc.softmax(-1), nodes[2]["logits"][0, 0].float().softmax(-1)
    dm = int((0.5 * pc + 0.5 * pd).argmax())
    do = int(m._mtp_nodes(h, condition=torch.tensor([[int(val[1])]]))[1]["chain2"]
             ["logits"][0, 0].float().argmax())
    _, fn = dec.step_multi([v1])
    t2dep = int(fn[1]["logits"][0, 0].float().argmax())
    r0 = rows[0]
    assert (r0["v1"], r0["d_direct"], r0["d_chain"], r0["d_oracle"],
            r0["d_comb"]) == (v1, dd, dc, do, dm)
    assert r0["t2dep"] == t2dep
    assert r0["true_next"] == int(val[1]) and r0["true_t2"] == int(val[2])
    # t2teacher == 下一位置的 v1（同一 teacher 前缀）
    assert rows[0]["t2teacher"] == rows[1]["v1"]
    # 四种草稿 + 两个目标都必须落盘（否则下游聚合会 KeyError）
    for key in ("v1", "true_next", "true_t2", "d_direct", "d_chain", "d_oracle",
                "d_comb", "t2dep", "t2teacher", "topk_direct", "topk_chain"):
        assert key in rows[0], key


def test_free_gen_pass_and_summarize_aggregation():
    """free_gen_pass 的臂结构 + summarize_free_gen 聚合成有效。"""
    m = tiny_model()
    val = np.random.RandomState(1).randint(0, 19, size=64).astype(np.int64)
    per = diag.free_gen_pass(m, m.pcap, val, n_prompts=1, prompt_len=8, n_new=6)
    assert set(per) == {"native", "chain", "direct", "combined"}
    s = diag.summarize_free_gen(per, 6)
    assert s["n_prompts"] == 1 and isinstance(s["bit_identical_all_arms"], bool)
    for arm in ("chain", "direct", "combined"):
        assert 0.0 <= s[arm]["alpha1"] <= 1.0 and s[arm]["n_rounds"] >= 1


def test_cluster_bootstrap_shape_and_degeneracy():
    rows = [{"seg": 0, "x": 1}, {"seg": 0, "x": 1}, {"seg": 1, "x": 1}]
    assert diag.cluster_bootstrap(rows, lambda rs: 0.5, B=50, seed=1) == [0.5, 0.5]
    ci = diag.cluster_bootstrap(rows, lambda rs: float(np.mean([r["x"] for r in rs])), B=50)
    assert ci[0] <= ci[1]
    assert diag.cluster_bootstrap([], lambda rs: 0.0, B=10) == [None, None]


def test_wilson_ci_properties():
    """准确率走二项/Wilson：k=0 下界恰为 0，对称，且 n 越大越窄。"""
    assert diag.wilson_ci(0, 0) == [None, None]
    lo, hi = diag.wilson_ci(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.29
    lo, hi = diag.wilson_ci(5, 10)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9
    w10, w1000 = diag.wilson_ci(5, 10), diag.wilson_ci(500, 1000)
    assert (w1000[1] - w1000[0]) < (w10[1] - w10[0])


def test_committed_wilson_ci_recomputes_from_raw_records():
    """results.json 的 wilson_ci_95 必须能由 diag_records.jsonl 逐条重算。"""
    res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
    rows_by: dict = collections.defaultdict(list)
    for line in open(os.path.join(HERE, "diag_records.jsonl"), encoding="utf-8"):
        row = json.loads(line)
        rows_by[row["ckpt"]].append(row)
    assert set(rows_by) == set(res["checkpoints"])
    for ck, rows in rows_by.items():
        w = res["checkpoints"][ck]["teacher_anchor"]["wilson_ci_95"]
        assert w["quality_vs_true_t2"]["direct"] == diag._wilson_eq(rows, "d_direct", "true_t2")
        assert w["quality_vs_true_t2"]["chain_oracle"] == diag._wilson_eq(rows, "d_oracle", "true_t2")
        assert w["accept_vs_deploy_target"]["chain_deploy"] == diag._wilson_eq(rows, "d_chain", "t2dep")
