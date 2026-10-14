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
                      N_LAYERS, SEQ_LEN, VOCAB, anchor_equivalence_table,
                      analytic_cost, base_cfg, build_model, param_audit,
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
            "LR": EXITS * (D_MODEL * LR_RANK + LR_RANK * VOCAB)}
    for arm in ARMS:
        audit = param_audit(build_model(arm, VOCAB, seed=0))
        assert audit["params_per_exit_readout"] == want[arm], arm
        assert audit["params_per_exit_readout_analytic"] == want[arm], arm
        assert audit["params_trunk_and_tied"] == (
            audit["params_total"] - want[arm])
        assert audit["n_exits"] == EXITS == N_LAYERS + 1


def test_T3_trunk_params_bitwise_equal_across_arms():
    """四臂同 seed 的主干 + tied 表逐位相同（唯一差别 = 每出口读出）。"""
    base = None
    for arm in ARMS:
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
        c = analytic_cost(arm, vocab=VOCAB)
        assert c["readout_params"] == param_audit(
            build_model(arm, VOCAB, seed=0))["params_per_exit_readout"]
        assert c["macs_per_token"]["total_train_all_exits"] - \
            c["macs_per_token"]["readout_all_exits_train"] == \
            c["macs_per_token"]["trunk"]


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
    assert [r["accept"] for r in mine["by_depth"]] == [r["accept"] for r in ref]
    assert mine["n_total"] == 2 * 2 * (SEQ_LEN - 1)


def test_T8_depth_curve_and_tf_acceptance_shapes():
    _, eval_arr, _ = (None, np.random.default_rng(1).integers(
        0, VOCAB, size=SEQ_LEN * 40, dtype=np.uint16), None)
    m = build_model("T", VOCAB, seed=0)
    curve = depth_curve(m, eval_arr, n_batches=1, batch_size=2, device="cpu")
    assert [r["depth"] for r in curve] == list(range(N_LAYERS + 1))
