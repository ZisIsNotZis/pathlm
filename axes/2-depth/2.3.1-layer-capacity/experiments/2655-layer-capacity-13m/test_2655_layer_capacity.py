"""2655 单测（`pytest axes/2-depth/2.3.1-layer-capacity/experiments/2655-layer-capacity-13m/
test_2655_layer_capacity.py -q`）。CPU，秒级。

每条断言都是契约里的门（README §4 的 P4/P5/P6），并各自带一个**必须失效**的探针，
所以这些测试不是恒真的装饰（`proof.gate_probe`）。
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARConfig, DepthARModel   # noqa: E402
from run import SCHEDULES, SUM_HIDDEN, analytic_cost      # noqa: E402
from smoke import _build, _forward, _sd_equal             # noqa: E402


def test_p5_explicit_uniform_schedule_is_bitwise_identical_to_engine_default():
    """P5 引擎等值门：显式 [1024]*8 与缺省（mlp_mult=4）逐位一致。"""
    m_def, m_exp = _build(None), _build(SCHEDULES["uniform"])
    assert _sd_equal(m_def, m_exp)
    assert torch.equal(_forward(m_def), _forward(m_exp))


def test_p5_probe_perturbed_schedule_breaks_bitwise_equality():
    """P5 探针：等值门会红（改一位 ⇒ 逐位不等），否则门是死的。"""
    m_def = _build(None)
    m_bad = _build(SCHEDULES["uniform"][:7] + [768])
    assert not _sd_equal(m_def, m_bad)
    assert not torch.equal(_forward(m_def), _forward(m_bad))


@pytest.mark.parametrize("arm", sorted(SCHEDULES))
def test_p6_hidden_by_layer_matches_schedule(arm):
    """P6 活性：开关真的到了模型（逐层隐藏维 == schedule）。"""
    m = _build(SCHEDULES[arm])
    assert [int(b.mlp[0].weight.shape[0]) for b in m.blocks] == SCHEDULES[arm]


def test_p4_all_arms_have_bitwise_equal_total_and_ffn_params():
    """P4 等预算门：三臂 Σh / FFN 参数 / 总参数逐位相等。"""
    vals = {}
    for arm, sched in SCHEDULES.items():
        m = _build(sched)
        ffn = sum(p.numel() for b in m.blocks for p in b.mlp.parameters())
        total = sum(p.numel() for p in m.parameters())
        assert sum(sched) == SUM_HIDDEN
        assert ffn == analytic_cost(sched)["ffn_params_analytic"]
        vals[arm] = (sum(sched), ffn, total)
    assert len(set(vals.values())) == 1


def test_p4_probe_off_budget_schedule_changes_params():
    """P4 探针：Σh 不同 ⇒ 参数必不同，否则「等预算」这句话无法失败。"""
    base = sum(p.numel() for p in _build(SCHEDULES["uniform"]).parameters())
    off = sum(p.numel() for p in _build(SCHEDULES["uniform"][:7] + [768]).parameters())
    assert off != base


def test_p6_probe_uniform_forward_is_stable_and_nonuniform_differs():
    """P6 探针：同 seed 同 schedule 前向逐位可复现；非均匀臂与 uniform 不同。"""
    ref = _forward(_build(SCHEDULES["uniform"]))
    assert torch.equal(_forward(_build(SCHEDULES["uniform"])), ref)
    for arm in ("front-big", "growing"):
        assert not torch.equal(_forward(_build(SCHEDULES[arm])), ref)
