#!/usr/bin/env python3
"""2687 · 等值门（T1）、变异（T2）、参数同量（T3）、打分器梯度活性（T4）、
嵌套恒等（T5）、直通前向精确性（T6）、计算账（T7）、因果性（T8）、负载均衡项（T9）
—— 自包含，不改共享 `tests/`。

契约见同目录 `README.md` §8。门的用途：
* `T1` 把「不筛（`D`）时前向与 stock 逐位相同」钉住；
* `T2` 钉住 `T1` **不是恒真的摆设**（半掩码必须改变前向）；
* `T5` 钉住 `L50` 从「`|pre|` 规则」起步（初始选择集 = `K50`）——「臂间唯一差异 = 这份
  预算是否被训练」这句话的证据；
* `T4` 钉住打分器在 `L50` 里确实拿到梯度、在其它臂里确实冻着（否则 `T5` 的相等就是造假）。

    python3 -m pytest test_2687_learned_selector.py -q
"""

import copy
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARConfig, DepthARModel        # noqa: E402

# 同目录 `run.py` 用**唯一模块名**载入（多个实验目录各有 `run.py`）
import importlib.util as _ilu                                  # noqa: E402
_spec = _ilu.spec_from_file_location("run_2687", os.path.join(HERE, "run.py"))
_run = _ilu.module_from_spec(_spec)
sys.modules["run_2687"] = _run
_spec.loader.exec_module(_run)
from run_2687 import (ALPHA_LB, ARMS, D_MODEL, HIDDEN, MLP_MULT, N_HEADS,  # noqa: E402
                      N_LAYERS, SCORER_PARAMS_PER_LAYER, SEQ_LEN, VOCAB,
                      analytic_cost, build_model, patch_ffn, scored_ffns)


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


def _stock():
    torch.manual_seed(0)
    return DepthARModel(_cfg(), VOCAB)


def _patched(arm):
    """stock 权重 + 指定臂的包装（打分器键是新参数，strict=False 只缺它们）。"""
    torch.manual_seed(0)
    model = _run.SelectorModel(_cfg(), VOCAB)
    patch_ffn(model, arm)
    model.aux_weight = ALPHA_LB if arm == "L50" else 0.0
    state = _run.adapt_state(_stock().state_dict())
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.endswith(("score_a", "score_b")) for k in missing), missing
    return model


# ---------------------------------------------------------------- T1 等值门

def test_T1_dense_arm_is_bitwise_equal_to_stock():
    """`D` 臂（不筛、掩码恒 1）的最终深度 logits 与 stock **逐位相同**。"""
    stock = _stock()
    dense = _patched("D")
    x = _tokens(3)
    a, b = _final_logits(stock, x), _final_logits(dense, x)
    assert torch.equal(a, b), (a - b).abs().max().item()


# ---------------------------------------------------------------- T2 变异（门能变红）

def test_T2_half_mask_changes_forward():
    """`K50` / `F50` / `L50` 都不是恒等：logits 必须与 stock 不同。"""
    stock = _stock()
    x = _tokens(4)
    ref = _final_logits(stock, x)
    for arm in ("K50", "F50", "L50"):
        got = _final_logits(_patched(arm), x)
        assert not torch.equal(ref, got), f"{arm} 与 stock 逐位相同 ⇒ 掩码没生效"
        assert (ref - got).abs().max().item() > 1e-6


# ---------------------------------------------------------------- T3 参数同量

def test_T3_params_identical_across_arms_and_selector_budget():
    """四臂总参数逐位相同 = stock + 逐层打分器；只有 `L50` 的 81 920 个是可训练的。"""
    stock = sum(p.numel() for p in _stock().parameters())
    expect = stock + N_LAYERS * SCORER_PARAMS_PER_LAYER
    for arm in ARMS:
        model = _patched(arm)
        total = sum(p.numel() for p in model.parameters())
        assert total == expect, (arm, total, expect)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if arm == "L50":
            assert trainable == expect
        else:
            assert trainable == stock, (arm, trainable, stock)


# ---------------------------------------------------------------- T4 打分器梯度活性

def test_T4_scorer_gets_gradient_in_L50_only():
    """一次真实前向 + 反传：`L50` 的打分器梯度非零；`K50`/`F50`/`D` 的打分器无梯度。

    `B ≡ 0` 起步时 `∂s/∂A ≡ 0`（`g = GELU(xA)B`，`B = 0` 把 A 的梯度乘成 0）⇒ 第一步只有
    `B` 拿梯度、`A` 恰好为 0；`B ≠ 0` 之后两个矩阵都有梯度。两个事实都钉住，否则
    「打分器被训练」这句话没有依据。
    """
    x = _tokens(5)
    for arm in ARMS:
        model = _patched(arm)
        model.train()
        loss, _ = model(x, exit_depths=None)
        model.zero_grad()
        loss.backward()
        for f in scored_ffns(model):
            if arm != "L50":
                assert f.score_a.grad is None and f.score_b.grad is None, arm
                continue
            assert f.score_b.grad is not None and f.score_b.grad.abs().sum() > 0
            assert f.score_a.grad is not None and f.score_a.grad.abs().sum() == 0
            f.score_b.data.normal_(0, 0.02, generator=torch.Generator().manual_seed(21))
        if arm == "L50":
            model.zero_grad()
            loss, _ = model(x, exit_depths=None)
            loss.backward()
            for f in scored_ffns(model):
                assert f.score_b.grad.abs().sum() > 0
                assert f.score_a.grad.abs().sum() > 0
        model.zero_grad()


# ---------------------------------------------------------------- T5 嵌套恒等（L50 从 |pre| 规则起步）

def test_T5_L50_init_selection_equals_K50_exactly():
    """`B ≡ 0` ⇒ `s = |pre|` 逐位相同 ⇒ `L50` 的掩码与 `K50` 的掩码**逐位相同**。"""
    model_k = _patched("K50")
    model_l = _patched("L50")
    x = _tokens(6)
    model_k.eval()
    model_l.eval()
    with torch.no_grad():
        # 两臂的 pre-activation 由同一份权重给出 ⇒ 同一 s
        for fk, fl in zip(scored_ffns(model_k), scored_ffns(model_l)):
            h = torch.randn(2, SEQ_LEN, D_MODEL, generator=torch.Generator().manual_seed(9))
            pre = fk.mlp[0](h)
            mk = fk.mask_for(h, pre)
            ml = fl.mask_for(h, pre)
            assert torch.equal(mk, ml), (mk - ml).abs().max().item()
        # 端到端：整模型 logits 也必须逐位相同
        a, b = _final_logits(model_k, x), _final_logits(model_l, x)
        assert torch.equal(a, b), (a - b).abs().max().item()


# ---------------------------------------------------------------- T6 直通前向精确性

def test_T6_straight_through_term_is_exactly_zero_forward():
    """`g*m + (t − t.detach())` 的前向值必须与 `g*m` **逐位相同**（残项只在反向上存在）。"""
    g = torch.randn(4, 8, 16, generator=torch.Generator().manual_seed(11))
    m = (torch.rand(4, 8, 16, generator=torch.Generator().manual_seed(12)) > 0.5).float()
    ms = torch.rand(4, 8, 16, generator=torch.Generator().manual_seed(13)).requires_grad_(True)
    t = g * ms
    out = g * m + (t - t.detach())
    assert torch.equal(out, g * m)
    out.sum().backward()
    assert ms.grad is not None and ms.grad.abs().sum() > 0


# ---------------------------------------------------------------- T7 计算账

def test_T7_analytic_cost_matches_contract_table():
    """每 token 总 MACs / 每层选择器预算与契约 §3 的表逐位一致。"""
    d, h, T, L, v = D_MODEL, HIDDEN, SEQ_LEN, N_LAYERS, VOCAB
    attn = L * (4 * d * d + 2 * T * d)
    readout = d * v
    sel = L * 2 * SCORER_PARAMS_PER_LAYER // 2          # d·r + r·H 每层
    expect = {
        "D": attn + L * 2 * h * d + readout + sel,
        "K50": attn + L * (d * h + (h // 2) * d) + readout + sel,
        "F50": attn + L * (d * (h // 2) + (h // 2) * d) + readout + sel,
        "L50": attn + L * (d * h + (h // 2) * d) + readout + sel,
    }
    for arm, total in expect.items():
        c = analytic_cost(arm)
        assert c["macs_per_token"]["total"] == total, (arm, c["macs_per_token"]["total"], total)
        assert c["params_total"] == 6962688 + L * SCORER_PARAMS_PER_LAYER
    assert analytic_cost("F50")["macs_per_token"]["total"] < \
        analytic_cost("K50")["macs_per_token"]["total"] < \
        analytic_cost("D")["macs_per_token"]["total"]


# ---------------------------------------------------------------- T8 因果性

def test_T8_causality():
    """改未来 token 不改变更早位置的 logits（`L50` 的选择器也不许偷看未来）。"""
    model = _patched("L50")
    model.eval()
    T = SEQ_LEN
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, VOCAB, (2, T), generator=g)
    x2 = x.clone()
    j = T // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, T - j), generator=g)
    with torch.no_grad():
        _, a1 = model(x, exit_depths=None)
        _, a2 = model(x2, exit_depths=None)
    lg1 = a1["depth_logits"][-1][:, :j]
    lg2 = a2["depth_logits"][-1][:, :j]
    assert torch.equal(lg1, lg2)


# ---------------------------------------------------------------- T9 负载均衡项

def test_T9_load_balance_term_range():
    """均衡项 `Σ_j f̂_j·P_j`：均匀选择 + 均匀路由概率 ⇒ 1.0；坍缩到一半通道 ⇒ 明显偏离 1。"""
    H, k = HIDDEN, HIDDEN // 2
    f_uniform = torch.full((H,), 1.0)                       # 已按 H/k 归一
    p_uniform = torch.full((H,), 1.0 / H)
    assert abs(float((f_uniform * p_uniform).sum()) - 1.0) < 1e-9
    f_collapse = torch.zeros(H)
    f_collapse[:k] = 2.0
    assert float((f_collapse * p_uniform).sum()) == 1.0     # 与均匀 P 下不可分（Switch 式的已知性质）
    p_skew = torch.zeros(H)
    p_skew[:k] = 1.0 / k
    assert float((f_collapse * p_skew).sum()) == 2.0


# ---------------------------------------------------------------- T10 键名适配

def test_T10_adapt_state_renames_only_stock_ffn_keys():
    """`adapt_state` 只把 stock 的 `blocks.i.mlp.<j>.*` 改名，**不许**碰打分器键
    `blocks.i.mlp.score_*`（改名就把打分器当成「缺键」丢掉、换回初值 ⇒ 读数变成
    「用初始打分器配已训主干」，这正是本实验最贵的一类读数错误）。"""
    stock_like = {"blocks.0.mlp.0.weight": 1, "blocks.0.mlp.2.bias": 2,
                  "blocks.0.ln1.weight": 3, "readout.0.weight": 4}
    wrapped = _run.adapt_state(stock_like)
    assert wrapped["blocks.0.mlp.mlp.0.weight"] == 1
    assert wrapped["blocks.0.mlp.mlp.2.bias"] == 2
    assert "blocks.0.ln1.weight" in wrapped and "readout.0.weight" in wrapped
    own = {"blocks.0.mlp.score_a": 5, "blocks.0.mlp.score_b": 6,
           "blocks.0.mlp.mlp.0.weight": 7}
    assert _run.adapt_state(own) == own
