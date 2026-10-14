#!/usr/bin/env python3
"""2643 · 逐位门（T1/T2）、机制活性与变异（T3/T5）、恒等起步（T4）、
参数账（T6）、因果性（T7）——自包含，不改共享 `tests/`。

契约见同目录 `README.md` §8。这些门的作用：证明
①「解绑开关关闭时 `T` 臂与引擎 stock 逐位相同」；
② 复用 2641 `U4-s0` 锚是**逐位忠实**的（键无缺无余 + 逐张量逐位相同）；
③ 这些门**不是恒真的摆设**（变异用例能变红）。

    python3 -m pytest test_2643_tied_untied.py -q
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

# 同目录 `run.py` 用**唯一模块名**载入：实验目录各自有一个 `run.py`，
# 若用 `import run`，同一 pytest 进程同时收集两个实验的测试时会互相串味。
import importlib.util as _ilu                                    # noqa: E402
_spec = _ilu.spec_from_file_location("run_2643", os.path.join(HERE, "run.py"))
_run = _ilu.module_from_spec(_spec)
sys.modules["run_2643"] = _run
_spec.loader.exec_module(_run)
from run_2643 import (ANCHOR_2641_RUN, ARMS, D_MODEL, MLP_MULT, N_HEADS,  # noqa: E402
                      N_LAYERS, PR_RANK, SEQ_LEN, VOCAB, analytic_cost,
                      build_model, intervention_liveness, param_audit)

ANCHOR_CKPT = os.path.join(ANCHOR_2641_RUN, "model.pt")


def _cfg():
    return DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                         seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True)


def _tokens(seed=0, b=2, t=SEQ_LEN):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, t), generator=g)


def _final_logits(model, x):
    model.eval()
    with torch.no_grad():
        _, aux = model(x, exit_depths=None)
    return aux["depth_logits"][-1]


def _prob0_rate(model, h):
    """同一 h 下 `argmax(Eh) == argmax(U_eff h)` 的位置占比（P2 探针的最小版）。"""
    E = model.embed.weight.float()
    U = model.effective_unembed()
    a = (h.float() @ E.T).argmax(-1)
    b = (h.float() @ U.T).argmax(-1)
    return float((a == b).float().mean())


# ---------------------------------------------------------------- T1/T2 逐位门

def test_T1_tied_arm_forward_bitwise_equals_stock_engine():
    """解绑开关关闭（arm='T'）时，前向最终深度 logits 与 stock `DepthARModel`
    **逐位相同**，且同 seed 下构造出的 state_dict 逐张量逐位相同。"""
    torch.manual_seed(0)
    stock = DepthARModel(_cfg(), VOCAB)
    tied = build_model("T", VOCAB, seed=0)
    a, b = stock.state_dict(), tied.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    x = _tokens(3)
    lg_a, lg_b = _final_logits(stock, x), _final_logits(tied, x)
    assert lg_a.shape == lg_b.shape == (2, SEQ_LEN, VOCAB)
    assert torch.equal(lg_a, lg_b), (lg_a - lg_b).abs().max().item()


def test_T2_reuse_anchor_checkpoint_is_faithful_on_all_keys():
    """复用 2641 `U4-s0` 锚：载入后键**无缺无余**、逐张量逐位相同。

    注：2641 的 `model.pt` 是**训练后**权重，无法与本次的**初始化**逐位对照
    （同一个 seed 只保证初始化相同）；可对照且已对照的是「构造路径相同」
    （T1：同 seed 下 init 逐位相同）与「载入复现逐位相同」（本测试）。
    """
    assert os.path.exists(ANCHOR_CKPT), f"锚 checkpoint 缺失：{ANCHOR_CKPT}"
    ckpt = torch.load(ANCHOR_CKPT, map_location="cpu")
    tied = build_model("T", VOCAB, seed=0)
    missing, unexpected = tied.load_state_dict(ckpt, strict=False)
    assert list(missing) == [], missing
    assert list(unexpected) == [], unexpected
    loaded = tied.state_dict()
    for k in ckpt:
        assert torch.equal(ckpt[k], loaded[k]), k


# ---------------------------------------------------------------- T3 变异（U 头真在用）

def test_T3_untied_head_is_actually_used_and_probe_can_go_red():
    """P2 探针：`U` 臂的 prob0 一致性 < 1（解绑真的生效）；
    把 `effective_unembed` 退化成「返回 E」（= 静默绑定）后一致性**变红为 1.0**
    ⇒ 探针能区分「解绑真生效」与「静默绑定」。"""
    model = build_model("U", VOCAB, seed=0)
    g = torch.Generator().manual_seed(11)
    h = torch.randn(128, D_MODEL, generator=g)
    good = _prob0_rate(model, h)
    assert good < 1.0, ("U 臂与 tied 解读出完全相同 ⇒ U 没有被使用（死干预）", good)
    # 变异：读出退化为 tied
    saved = model.effective_unembed
    model.effective_unembed = lambda: model.embed.weight.float()
    try:
        mutated = _prob0_rate(model, h)
    finally:
        model.effective_unembed = saved
    assert mutated == 1.0, ("退化后的探针没有变红 ⇒ 探针无法检出静默绑定", mutated)


# ---------------------------------------------------------------- T4 PR 恒等起步

def test_T4_partial_binding_starts_bitwise_equal_to_tied():
    """`A=0`、`B~N(0,0.02)` 时 `PR` 的前向最终深度 logits 与 `T` **逐位相同**
    （单边零的恒等起步），且主干也逐位相同（同 seed 同一构造路径）。"""
    tied = build_model("T", VOCAB, seed=0)
    pr = build_model("PR", VOCAB, seed=0)
    td, pd = tied.state_dict(), pr.state_dict()
    shared = [k for k in td if k not in ("pr_a", "pr_b")]
    for k in shared:
        assert torch.equal(td[k], pd[k]), k
    assert torch.equal(pr.pr_a, torch.zeros_like(pr.pr_a))
    assert float(pr.pr_b.detach().std()) > 0.0
    x = _tokens(5)
    assert torch.equal(_final_logits(tied, x), _final_logits(pr, x))


# ---------------------------------------------------------------- T5 非鞍点（可失败）

def test_T5_one_sided_zero_is_not_a_saddle_but_two_sided_zero_is():
    """`A=0,B~N(0,0.02)`：`A` 在 step 0 就有非零梯度，一步优化器后离开 0；
    把 `B` 也置零（双侧零）后 `A` 的梯度恒为 0（鞍点确实存在）⇒ 本门可失败。"""
    x = _tokens(9)
    pr = build_model("PR", VOCAB, seed=0)
    opt = torch.optim.AdamW(pr.parameters(), lr=1e-3)
    pr.train()
    loss, _ = pr(x)
    opt.zero_grad()
    loss.backward()
    assert pr.pr_a.grad is not None
    g = float(pr.pr_a.grad.abs().max())
    assert g > 0.0, ("A 起步梯度为 0 ⇒ 落在鞍点", g)
    opt.step()
    assert float(pr.pr_a.detach().abs().max()) > 0.0, "一步优化器后 A 仍全零"

    # 变异：双侧零 ⇒ 梯度恒 0（证明「单边零」不是无意义的仪式）
    pr2 = build_model("PR", VOCAB, seed=0)
    with torch.no_grad():
        pr2.pr_b.zero_()
    loss2, _ = pr2(x)
    pr2.zero_grad()
    loss2.backward()
    ga = pr2.pr_a.grad
    assert ga is not None and float(ga.abs().max()) == 0.0, float(ga.abs().max())


# ---------------------------------------------------------------- T6 参数账

def test_T6_param_counts_are_exact():
    tied = build_model("T", VOCAB, seed=0)
    u = build_model("U", VOCAB, seed=0)
    pr = build_model("PR", VOCAB, seed=0)
    assert param_audit(tied)["params_total"] == 6962688
    assert param_audit(u)["params_added_readout"] == VOCAB * D_MODEL == 52736
    assert param_audit(pr)["params_added_readout"] == (
        D_MODEL * PR_RANK + PR_RANK * VOCAB) == 14784
    assert param_audit(u)["params_total"] - param_audit(tied)["params_total"] == 52736
    assert param_audit(pr)["params_total"] - param_audit(tied)["params_total"] == 14784
    for arm in ("T", "U", "PR"):
        cost = analytic_cost(arm)
        assert cost["macs_per_token"]["attention"] == 3145728
        assert cost["macs_per_token"]["ffn"] == 4194304
    assert analytic_cost("T")["macs_per_token"]["readout_online"] == 52736
    assert analytic_cost("U")["macs_per_token"]["readout_online"] == 52736
    assert analytic_cost("PR")["macs_per_token"]["readout_online"] == 67520
    # 活性块字段齐全（SCHEMA §10）
    live = intervention_liveness(pr)
    assert set(("pr_a", "pr_b")) <= set(live)
    assert live["pr_a"]["w_fro"] == 0.0            # 起步（未训练）A 为 0
    assert live["pr_b"]["nonzero_frac"] == 1.0


# ---------------------------------------------------------------- T7 因果性

@pytest.mark.parametrize("arm", ["U", "PR"])
def test_T7_causality_future_tokens_do_not_change_past_logits(arm):
    """P3 必须不变：只改位置 ≥ j 的 token，位置 < j 的 logits 逐位不变。"""
    model = build_model(arm, VOCAB, seed=0)
    g = torch.Generator().manual_seed(21)
    x = torch.randint(0, VOCAB, (2, SEQ_LEN), generator=g)
    x2 = x.clone()
    j = SEQ_LEN // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, SEQ_LEN - j), generator=g)
    a, b = _final_logits(model, x)[:, :j], _final_logits(model, x2)[:, :j]
    assert torch.equal(a, b)
