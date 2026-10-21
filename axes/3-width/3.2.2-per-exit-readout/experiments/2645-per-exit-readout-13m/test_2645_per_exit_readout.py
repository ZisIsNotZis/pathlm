#!/usr/bin/env python3
"""2645 · 等值门（T1–T3）、变异（T4）、表达力（T5）、因果（T6）、锚对照（T7）、
电池口径（T8）——自包含，不改共享 `tests/`。

契约见同目录 `README.md` §4/§8。门的作用：
* 证明「`T` 臂 = stock tied 读出」与「`PE`/`LR` 恒等起步 = `T`」**逐位相同**；
* 证明这些门**不是恒真的摆设**：改动一出口的读出头必须让该出口 logits 变红；
* 钉住每出口参数量（P2 的「单列」口径）与四臂主干逐位相同（单因素）。

    python3 -m pytest test_2645_per_exit_readout.py -q
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, depth_curve,  # noqa: E402
                             tf_acceptance)

# 同目录 `run.py` 用**唯一模块名**载入（各实验目录都有 `run.py`，同一 pytest
# 进程收集多个实验的测试时会互相串味）。
import importlib.util as _ilu                                   # noqa: E402
_spec = _ilu.spec_from_file_location("run_2645", os.path.join(HERE, "run.py"))
_run = _ilu.module_from_spec(_spec)
sys.modules["run_2645"] = _run
_spec.loader.exec_module(_run)
from run_2645 import (ANCHOR_DIR, ARMS, D_MODEL, EXITS, LR_RANK, N_HEADS,  # noqa: E402
                      N_LAYERS, PE_READOUT_PARAMS, READOUT_ARMS, SEQ_LEN, VOCAB,
                      WIDE_MACS_DELTA, WIDE_MLP_HIDDEN, WIDE_PARAM_DELTA,
                      anchor_equivalence_table, analytic_cost, base_cfg,
                      build_model, cfg_for, equal_param_branch, param_audit,
                      tf_acceptance_paired, MLP_MULT)

ANCHOR_RESULTS = os.path.join(ANCHOR_DIR, "results.json")


def _tokens(seed=0, b=2, t=SEQ_LEN):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, t), generator=g)


def _logits_by_depth(model, x):
    model.eval()
    with torch.no_grad():
        _, aux = model(x, exit_depths=None)
    return [lg.float().clone() for lg in aux["depth_logits"]]


# ---------------------------------------------------------------- T1 等值性门

def test_T1_T_arm_readout_bitwise_equals_stock_model():
    """`T` 臂（无自有读出参数）的逐出口 logits 与 stock `DepthARModel` 逐位相同。"""
    torch.manual_seed(0)
    stock = DepthARModel(base_cfg(), VOCAB)
    torch.manual_seed(0)
    arm_t = build_model("T", VOCAB, seed=0)
    assert set(arm_t.state_dict()) == set(stock.state_dict())   # 无多余参数键
    stock.load_state_dict(arm_t.state_dict())
    x = _tokens(1)
    for a, b in zip(_logits_by_depth(stock, x), _logits_by_depth(arm_t, x)):
        assert torch.equal(a, b), (a - b).abs().max().item()


def test_T1_T_arm_readout_matches_stock_readout_call():
    """直接调 `readout(h, depth=k)` 也逐位相同（含 depth=None 的全深度出口）。"""
    torch.manual_seed(0)
    stock = DepthARModel(base_cfg(), VOCAB)
    torch.manual_seed(0)
    arm_t = build_model("T", VOCAB, seed=0)
    stock.load_state_dict(arm_t.state_dict())
    h = torch.randn(2, 5, D_MODEL, generator=torch.Generator().manual_seed(2))
    for k in (0, 3, N_LAYERS, None):
        assert torch.equal(stock.readout(h, depth=k), arm_t.readout(h, depth=k))


def test_T1_identity_start_PE_and_LR_bitwise_equal_T():
    """`PE`（M_k = I）与 `LR`（B_k = 0）的起步读出 = tied 表 ⇒ 逐位等于 `T`。"""
    x = _tokens(2)
    ref = _logits_by_depth(build_model("T", VOCAB, seed=0), x)
    for arm in ("PE", "LR"):
        m = build_model(arm, VOCAB, seed=0)
        for a, b in zip(ref, _logits_by_depth(m, x)):
            assert torch.equal(a, b), (arm, (a - b).abs().max().item())


def test_T1_IND_with_tied_tables_bitwise_equal_T():
    """`IND` 的表若全部拷成 tied 表 `E`，则 logits 逐位等于 `T`（接线正确性门）。"""
    x = _tokens(3)
    ref = _logits_by_depth(build_model("T", VOCAB, seed=0), x)
    m = build_model("IND", VOCAB, seed=0)
    with torch.no_grad():
        for mod in m.per_exit.tab:
            mod.weight.copy_(m.embed.weight.detach())
    for a, b in zip(ref, _logits_by_depth(m, x)):
        assert torch.equal(a, b), (a - b).abs().max().item()


# ---------------------------------------------------------------- T3 参数审计

def test_T3_param_counts_match_analytic():
    want = {"T": 0, "PE": EXITS * D_MODEL * D_MODEL,
            "IND": EXITS * VOCAB * D_MODEL,
            "LR": EXITS * (D_MODEL * LR_RANK + LR_RANK * VOCAB),
            "WIDE": 0}
    for arm in ARMS:
        audit = param_audit(build_model(arm, VOCAB, seed=0))
        assert audit["params_per_exit_readout"] == want[arm], arm
        assert audit["params_per_exit_readout_analytic"] == want[arm], arm
        assert audit["params_trunk_and_tied"] == (
            audit["params_total"] - want[arm])
        assert audit["n_exits"] == EXITS == N_LAYERS + 1


def test_T3_trunk_params_bitwise_equal_across_arms():
    """四支单因素臂同 seed 的主干 + tied 表逐位相同（唯一差别 = 每出口读出）。

    `WIDE` **不在**此列：它的主干被有意改变（§11），不能逐位对齐。
    """
    base = None
    for arm in READOUT_ARMS:
        sd = build_model(arm, VOCAB, seed=0).state_dict()
        trunk = {k: v for k, v in sd.items() if not k.startswith("per_exit.")}
        if base is None:
            base = trunk
        else:
            assert set(trunk) == set(base), arm
            for k in trunk:
                assert torch.equal(trunk[k], base[k]), (arm, k)


def test_T3_analytic_cost_readout_terms():
    for arm in ARMS:
        c = analytic_cost(arm, vocab=VOCAB,
                          mlp_hidden=(WIDE_MLP_HIDDEN if arm == "WIDE" else None))
        assert c["readout_params"] == param_audit(
            build_model(arm, VOCAB, seed=0))["params_per_exit_readout"]
        assert c["macs_per_token"]["total_train_all_exits"] - \
            c["macs_per_token"]["readout_all_exits_train"] == \
            c["macs_per_token"]["trunk"]


# ---------------------------------------------------------------- T9 等参数对照臂

def test_T9_WIDE_pays_the_PE_budget_on_the_trunk():
    """等参数对照的记账：`WIDE` 的参数增量 ≈ `PE` 的读出增量，训练 MACs 严格相等。"""
    t = param_audit(build_model("T", VOCAB, seed=0))
    pe = param_audit(build_model("PE", VOCAB, seed=0))
    wide = param_audit(build_model("WIDE", VOCAB, seed=0))
    assert pe["params_per_exit_readout"] == PE_READOUT_PARAMS == 589824
    assert wide["params_per_exit_readout"] == 0
    assert wide["params_total"] - t["params_total"] == WIDE_PARAM_DELTA == 590976
    # 偏差只允许是「对照臂多拿」这一侧，且 ≤ 0.2% 增量
    over = WIDE_PARAM_DELTA - PE_READOUT_PARAMS
    assert 0 <= over <= 0.002 * PE_READOUT_PARAMS, over
    # 每层 FFN 隐藏宽真的换了（不是只传了 config 没接线）
    assert wide["mlp_hidden"] == [WIDE_MLP_HIDDEN] * N_LAYERS == [1168] * 8
    assert t["mlp_hidden"] == [MLP_MULT * D_MODEL] * N_LAYERS == [1024] * 8
    # 训练期每 token MACs 与 PE 严格对齐，推理期对照臂更多
    ct = analytic_cost("T", vocab=VOCAB)["macs_per_token"]
    cp = analytic_cost("PE", vocab=VOCAB)["macs_per_token"]
    cw = analytic_cost("WIDE", vocab=VOCAB, mlp_hidden=WIDE_MLP_HIDDEN)[
        "macs_per_token"]
    assert cw["total_train_all_exits"] == cp["total_train_all_exits"]
    assert cw["trunk"] - ct["trunk"] == WIDE_MACS_DELTA == PE_READOUT_PARAMS
    assert cw["total_infer_one_exit"] > cp["total_infer_one_exit"]


def test_T9_WIDE_readout_is_bitwise_the_stock_tied_readout():
    """`WIDE` 的唯一干预在主干：同权重下它的读出与 stock `readout` 逐位相同。

    反过来：把 `WIDE` 的 FFN 宽度改回 stock（`cfg_for("T")`）后，它的参数形状与 `T`
    逐键相同 ⇒ 证明参数差确实全部在 FFN 上，不在读出上。
    """
    wide = build_model("WIDE", VOCAB, seed=0)
    x = _tokens(9)
    h = torch.randn(2, 5, D_MODEL, generator=torch.Generator().manual_seed(3))
    for k in (0, 3, N_LAYERS, None):
        # 覆写层对 `WIDE` 是恒等的：直接调父类方法结果逐位相同
        assert torch.equal(DepthARModel.readout(wide, h, depth=k),
                           wide.readout(h, depth=k)), k
    assert _logits_by_depth(wide, x) is not None      # 逐深度读出路径不报错
    # 键集逐键相同，形状只在 FFN 两处不同（up 权重 / up 偏置）
    wt = build_model("T", VOCAB, seed=0).state_dict()
    ws = wide.state_dict()
    assert set(wt) == set(ws)
    diff = sorted(k for k in wt if wt[k].shape != ws[k].shape)
    assert all(k.endswith(("mlp.0.weight", "mlp.0.bias", "mlp.2.weight"))
               for k in diff), diff
    assert len(diff) == 3 * N_LAYERS, diff


def test_T9_equal_param_branch_rule():
    """P5 的四分支阈值规则（README §4）：边界值两侧各钉一次，防事后改判。"""
    B = 0.014
    assert equal_param_branch(-0.010, -0.080, B) == "params_not_the_explanation"
    # |Δ_W| 刚过带、残余在带内 ⇒ 参数是全部解释
    assert equal_param_branch(-0.016, -0.004, B) == "params_explain_all"
    # 残余仍越带且 PE 更好 ⇒ 部分解释
    assert equal_param_branch(-0.016, -0.070, B) == "params_explain_part"
    # 对照臂反而更好 ⇒ 这笔钱花在主干上更好
    assert equal_param_branch(-0.016, +0.070, B) == "trunk_better"
    # 边界：恰好落在带上是「带内」（≤ band），不是越带
    assert equal_param_branch(-B, -0.080, B) == "params_not_the_explanation"
    assert equal_param_branch(-0.050, -B, B) == "params_explain_all"
    assert equal_param_branch(-0.050, B + 1e-9, B) == "trunk_better"


# ---------------------------------------------------------------- T4 变异（能变红）

def test_T4_mutating_one_exit_head_changes_only_that_exit():
    """把 `PE` 的第 3 出口变换改掉 ⇒ 该出口 logits 变红、其余出口逐位不变。"""
    x = _tokens(4)
    m = build_model("PE", VOCAB, seed=0)
    before = _logits_by_depth(m, x)
    with torch.no_grad():
        m.per_exit.m[3].weight.add_(0.5 * torch.eye(D_MODEL))
    after = _logits_by_depth(m, x)
    assert not torch.equal(before[3], after[3]), "第 3 出口没变红 ⇒ 门恒真"
    for k in range(EXITS):
        if k != 3:
            assert torch.equal(before[k], after[k]), k


def test_T4_mutating_IND_table_entry_turns_red():
    x = _tokens(5)
    m = build_model("IND", VOCAB, seed=0)
    before = _logits_by_depth(m, x)
    with torch.no_grad():
        m.per_exit.tab[6].weight[0, 0].add_(1.0)
    after = _logits_by_depth(m, x)
    assert not torch.equal(before[6], after[6])
    assert torch.equal(before[5], after[5])


def test_T4_LR_gradient_nonzero_at_init():
    """`LR` 的单边零起步非鞍点：∂L/∂A_k ≠ 0（零侧可学）——同引擎 `readout_perdepth`
    的 P=0 侧（SCHEMA §10 的「单边零」）。另一侧 B_k 起步非零 ⇒ ∂L/∂B_k = 0 是
    该结构的恒等式（A_k = 0），在此一并钉住以防误判为鞍点。"""
    m = build_model("LR", VOCAB, seed=0)
    x = _tokens(6)
    m.train()
    loss, _ = m(x)
    m.zero_grad()
    loss.backward()
    for k in range(EXITS):
        assert float(m.per_exit.a[k].weight.grad.norm()) > 0, k
        assert float(m.per_exit.b[k].weight.grad.norm()) == 0.0, k
    m.zero_grad()


def test_T4_PE_gradient_nonzero_at_identity_start():
    m = build_model("PE", VOCAB, seed=0)
    x = _tokens(7)
    m.train()
    loss, _ = m(x)
    m.zero_grad()
    loss.backward()
    for k in range(EXITS):
        assert float(m.per_exit.m[k].weight.grad.norm()) > 0, k
    m.zero_grad()


# ---------------------------------------------------------------- T5 表达力（README §3）

def test_T5_PE_head_family_covers_any_linear_head_at_this_tokenizer_scale():
    """V=206 < d=256 ⇒ 存在 `M` 使 `E·M = Y`（任意 `Y ∈ R^{V×d}`）⇒ `PE` 的有效
    出口头族**等于**任意线性头族（`IND` 的表也在其中），即 `PE ⊇ IND`（表达力）。"""
    g = torch.Generator().manual_seed(11)
    E = torch.randn(VOCAB, D_MODEL, generator=g)
    Y = torch.randn(VOCAB, D_MODEL, generator=g)
    M = torch.linalg.lstsq(E, Y).solution            # 解 E M = Y（欠定，取最小范数）
    assert M.shape == (D_MODEL, D_MODEL)
    assert float((E @ M - Y).abs().max()) < 1e-5


# ---------------------------------------------------------------- T6 因果性

def test_T6_PE_causal_prefix_bitwise_unchanged():
    m = build_model("PE", VOCAB, seed=0)
    g = torch.Generator().manual_seed(8)
    x = torch.randint(0, VOCAB, (2, SEQ_LEN), generator=g)
    x2 = x.clone()
    j = SEQ_LEN // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, SEQ_LEN - j), generator=g)
    b = _logits_by_depth(m, x)[-1][:, :j]
    a = _logits_by_depth(m, x2)[-1][:, :j]
    assert torch.equal(a, b)


# ---------------------------------------------------------------- T7 锚对照

@pytest.mark.skipif(not os.path.exists(ANCHOR_RESULTS), reason="锚结果缺失")
def test_T7_anchor_config_field_by_field_equal():
    hist = json.load(open(ANCHOR_RESULTS))
    rows = anchor_equivalence_table(hist)
    assert all(r["equal"] for r in rows), [r for r in rows if not r["equal"]]


@pytest.mark.skipif(not os.path.exists(ANCHOR_RESULTS), reason="锚结果缺失")
def test_T7_anchor_equivalence_gate_can_turn_red():
    """变异用例：把锚的 mlp_mult 改成 6，门必须变红（不是恒真的摆设）。"""
    hist = json.load(open(ANCHOR_RESULTS))
    hist = json.loads(json.dumps(hist))
    hist["config"]["mlp_mult"] = 6
    rows = anchor_equivalence_table(hist)
    assert not all(r["equal"] for r in rows)
    assert any(r["field"] == "mlp_mult" and not r["equal"] for r in rows)


# ---------------------------------------------------------------- T8 电池口径

def test_T8_tf_acceptance_paired_matches_engine_tf_acceptance():
    """自持的逐深度接受率（带逐批计数）与引擎 `tf_acceptance` 逐位一致。"""
    _, eval_arr, _ = (None, np.random.default_rng(0).integers(
        0, VOCAB, size=SEQ_LEN * 40, dtype=np.uint16), None)
    m = build_model("PE", VOCAB, seed=0)
    mine = tf_acceptance_paired(m, eval_arr, n_batches=2, batch_size=2,
                                device="cpu")
    ref = tf_acceptance(m, eval_arr, n_batches=2, batch_size=2, device="cpu")
    # 引擎只保留 4 位小数 ⇒ 按 4 位比（自持电池保留 6 位；舍入口径差异不算不等）
    assert [round(r["accept"], 4) for r in mine["by_depth"]] == \
        [r["accept"] for r in ref]
    assert mine["n_total"] == 2 * 2 * (SEQ_LEN - 1)
    assert mine["n_per_batch"] == 2 * (SEQ_LEN - 1)
    # 计数一致（不是只靠小数位碰巧相等）：逐深度计数 = accept × n 必为整数
    for row in mine["by_depth"]:
        c = row["accept"] * row["n"]
        assert abs(c - round(c)) < 1e-6, (row["depth"], c)


def test_T8_depth_curve_and_tf_acceptance_shapes():
    _, eval_arr, _ = (None, np.random.default_rng(1).integers(
        0, VOCAB, size=SEQ_LEN * 40, dtype=np.uint16), None)
    m = build_model("T", VOCAB, seed=0)
    curve = depth_curve(m, eval_arr, n_batches=1, batch_size=2, device="cpu")
    assert [r["depth"] for r in curve] == list(range(N_LAYERS + 1))
