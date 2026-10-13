#!/usr/bin/env python3
"""2641 · 等值性门（T1）、参数审计（T2）与变异验证（T3）——自包含，不改共享 `tests/`。

契约见同目录 `README.md` §8。这些门的作用：证明本实验的「逐层 FFN 前缀切片」与
引擎原生窄 FFN **实现等价**（T1），且这些门**不是恒真的摆设**（T3）。

    python3 -m pytest test_2641_ffn_ratio.py -q
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARConfig, DepthARModel      # noqa: E402
from run import (D_MODEL, MLP_MULT, N_HEADS, N_LAYERS, SEQ_LEN, SCHEDULES,
                 analytic_cost, build_model, narrow_ffn_prefix, param_audit)  # noqa: E402

VOCAB = 206


def _cfg(mlp_mult=MLP_MULT, dense=True):
    return DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                         seq_len=SEQ_LEN, mlp_mult=mlp_mult, dense=dense)


def _tokens(seed=0, b=2, t=SEQ_LEN):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, t), generator=g)


def _logits(model, x):
    model.eval()
    with torch.no_grad():
        _, aux = model(x, exit_depths=None)
    return aux["depth_logits"][-1]


# ---------------------------------------------------------------- T1 等值性门

def test_T1_uniform_r2_slice_bitwise_equals_engine_mlp_mult_2():
    """均匀 r=2 的前缀切片模型，装入引擎原生 `mlp_mult=2` 的权重后，
    最终深度 logits **逐位相同** ⇒ 切片实现 = 引擎原生窄 FFN。"""
    torch.manual_seed(0)
    stock_narrow = DepthARModel(_cfg(mlp_mult=2), VOCAB)
    sliced = build_model([2] * N_LAYERS, VOCAB, seed=7)   # mlp_mult=4 构造 + 切到 2
    sliced.load_state_dict(stock_narrow.state_dict())
    x = _tokens(3)
    a, b = _logits(stock_narrow, x), _logits(sliced, x)
    assert a.shape == b.shape
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_T1_uniform_r4_is_noop_on_stock():
    """均匀 r=4 = stock：不该动任何参数对象，logits 与原生 mlp_mult=4 逐位相同。"""
    torch.manual_seed(0)
    stock = DepthARModel(_cfg(), VOCAB)
    model = build_model([4] * N_LAYERS, VOCAB, seed=7)
    model.load_state_dict(stock.state_dict())
    x = _tokens(4)
    assert torch.equal(_logits(stock, x), _logits(model, x))


# ---------------------------------------------------------------- T2 参数/代价审计

@pytest.mark.parametrize("arm", sorted(SCHEDULES))
def test_T2_hidden_dims_and_ffn_params_match_analytic(arm):
    sched = SCHEDULES[arm]
    model = build_model(sched, VOCAB, seed=0)
    audit = param_audit(model)
    assert audit["hidden_by_layer"] == [r * D_MODEL for r in sched]
    cost = analytic_cost(sched, vocab=VOCAB)
    assert audit["params_ffn"] == cost["ffn_params_analytic"], (
        audit["params_ffn"], cost["ffn_params_analytic"])


def test_T2_equal_ffn_budget_arms_are_bitwise_equal_in_cost():
    """U2 / PYR / APYR：Σr、FFN 参数、每 token FFN MACs 三者逐位相等（契约 §3）。"""
    u2, pyr, apyr = (analytic_cost(SCHEDULES[a], vocab=VOCAB)
                     for a in ("U2", "PYR", "APYR"))
    assert sum(SCHEDULES["U2"]) == sum(SCHEDULES["PYR"]) == sum(SCHEDULES["APYR"])
    assert (u2["macs_per_token"]["ffn"] == pyr["macs_per_token"]["ffn"]
            == apyr["macs_per_token"]["ffn"])
    assert (u2["ffn_params_analytic"] == pyr["ffn_params_analytic"]
            == apyr["ffn_params_analytic"])
    m = [param_audit(build_model(SCHEDULES[a], VOCAB, seed=0))["params_ffn"]
         for a in ("U2", "PYR", "APYR")]
    assert len(set(m)) == 1, m
    assert m[0] == u2["ffn_params_analytic"]
    # U4 是它的两倍（FFN 参数与每 token FFN MACs）
    u4 = analytic_cost(SCHEDULES["U4"], vocab=VOCAB)
    assert u4["macs_per_token"]["ffn"] == 2 * u2["macs_per_token"]["ffn"]


def test_T2_prefix_slice_takes_first_rows_and_cols():
    """切片是**前缀**（前 r·d 行 / 前 r·d 列），不是后缀；且切片后是**新参数对象**。"""
    model = build_model([2] * N_LAYERS, VOCAB, seed=0)
    full = build_model([4] * N_LAYERS, VOCAB, seed=0)
    lin1_s, lin1_f = model.blocks[3].mlp[0], full.blocks[3].mlp[0]
    assert lin1_s.weight.shape == (2 * D_MODEL, D_MODEL)
    assert torch.equal(lin1_s.weight, lin1_f.weight[:2 * D_MODEL])
    assert torch.equal(model.blocks[3].mlp[2].weight,
                       full.blocks[3].mlp[2].weight[:, :2 * D_MODEL])
    assert model.blocks[3].mlp[0].weight is not lin1_f.weight


# ---------------------------------------------------------------- T3 变异（必须抓红）

def test_T3a_mutation_activation_swap_breaks_T1():
    """变异：把窄 FFN 的激活从 GELU 换成 ReLU（形状不变、state_dict 装得进）
    ⇒ T1 的逐位相等必须变红 ⇒ 证明 T1 分辨的是「接线」而不只是形状。

    注：`load_state_dict` 会把切片的**取值**整个覆盖，因此任何**保形**的切片
    约定错误在装权重后不可见；T1 真正钉住的是「形状 + 接线」与引擎原生窄 FFN
    一致（形状的变异由 T3b 抓、chedule 的变异由 T3c 抓）。"""
    import torch.nn as nn
    torch.manual_seed(0)
    stock_narrow = DepthARModel(_cfg(mlp_mult=2), VOCAB)
    model = build_model([2] * N_LAYERS, VOCAB, seed=0)
    for blk in model.blocks:
        blk.mlp[1] = nn.ReLU()                       # ← 变异点
    model.load_state_dict(stock_narrow.state_dict())
    x = _tokens(5)
    assert not torch.equal(_logits(stock_narrow, x), _logits(model, x)), \
        "接线被打错却仍逐位相同 ⇒ T1 门失灵（不能分辨错误实现）"


def test_T3b_mutation_wrong_width_shape_mismatch_is_detected():
    """变异：切片宽度切错（r=3）⇒ 装 r=2 的权重必须报形状错，不许静默通过。"""
    torch.manual_seed(0)
    stock_narrow = DepthARModel(_cfg(mlp_mult=2), VOCAB)
    wrong = build_model([3] * N_LAYERS, VOCAB, seed=0)
    assert wrong.blocks[0].mlp[0].weight.shape == (3 * D_MODEL, D_MODEL)
    with pytest.raises(RuntimeError):
        wrong.load_state_dict(stock_narrow.state_dict())


def test_T3_mutation_missing_one_layer_breaks_param_audit():
    """变异：少切一层 ⇒ 隐藏维审计与解析参数审计必须变红。"""
    sched = SCHEDULES["PYR"]
    model = build_model(sched, VOCAB, seed=0)
    # 把最后一层「还原」成全宽（模拟漏切）
    blk = model.blocks[-1]
    model2 = build_model([4] * N_LAYERS, VOCAB, seed=0)
    blk.mlp[0].weight = torch.nn.Parameter(
        model2.blocks[-1].mlp[0].weight.clone())
    blk.mlp[0].bias = torch.nn.Parameter(model2.blocks[-1].mlp[0].bias.clone())
    blk.mlp[2].weight = torch.nn.Parameter(
        model2.blocks[-1].mlp[2].weight.clone())
    audit = param_audit(model)
    assert audit["hidden_by_layer"] != [r * D_MODEL for r in sched]
    assert audit["params_ffn"] != analytic_cost(sched, vocab=VOCAB)["ffn_params_analytic"]


def test_T3_mutation_out_of_range_r_raises():
    """变异：r=0 或 r>r_max 必须报错（否则会出现「0 维 FFN」这种静默退化）。"""
    model = build_model([1] * N_LAYERS, VOCAB, seed=0)
    for bad in ([0] * N_LAYERS, [5] * N_LAYERS, [2] * (N_LAYERS - 1)):
        with pytest.raises(ValueError):
            narrow_ffn_prefix(model, bad)


def test_T3_mutation_wide_arm_differs_from_narrow_arm():
    """活性对照：同样的 seed 下，U4 与 U2 的初始前向必须**不同**（否则干预没接线）。"""
    x = _tokens(6)
    u4 = build_model(SCHEDULES["U4"], VOCAB, seed=0)
    u2 = build_model(SCHEDULES["U2"], VOCAB, seed=0)
    assert not torch.equal(_logits(u4, x), _logits(u2, x))
    # 但 trunk（attention）必须逐位相同 —— 臂间唯一差别是 FFN 宽度
    for l in range(N_LAYERS):
        assert torch.equal(u4.blocks[l].qkv.weight, u2.blocks[l].qkv.weight)
        assert torch.equal(u4.blocks[l].proj.weight, u2.blocks[l].proj.weight)
    assert torch.equal(u4.embed.weight, u2.embed.weight)
