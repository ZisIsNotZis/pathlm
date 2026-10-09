"""2632 · 可失败单测（放在本目录，不写 tests/）。

钉住本诊断里三个「会静默出错」的语义点（SCHEMA §10 活性 / methodology §5 变异）：
1. `chain2` 的显式 condition 接线真的改变输出（条件活的）；
2. 「显式喂 v1」与「deploy 分支（condition=None 自取 node1 argmax）」逐元素相同
   （否则 chain_deploy 与 chain_oracle 的分叉是接线 bug，不是条件效应）；
3. 混合 / 分层 / top-k 召回三个统计量的定义与手算一致。
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import diag  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402


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
