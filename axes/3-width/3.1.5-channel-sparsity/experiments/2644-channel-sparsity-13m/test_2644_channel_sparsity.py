#!/usr/bin/env python3
"""2644 · 等值门（T1）、变异（T2）、参数同量（T3）、死区/梯度（T4）、计算账（T5）、
因果性（T6）——自包含，不改共享 `tests/`。

契约见同目录 `README.md` §8。门的作用：证明「掩码恒 1 时前向与 stock 逐位相同」，
且这些门**不是恒真的摆设**（`keep_frac=0.5` 时前向必须不同）。

    python3 -m pytest test_2644_channel_sparsity.py -q
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

# 同目录 `run.py` 用**唯一模块名**载入：实验目录各自有一个 `run.py`，
# 若用 `import run`，同一 pytest 进程同时收集两个实验的测试时会互相串味。
import importlib.util as _ilu                                  # noqa: E402
_spec = _ilu.spec_from_file_location("run_2644", os.path.join(HERE, "run.py"))
_run = _ilu.module_from_spec(_spec)
sys.modules["run_2644"] = _run
_spec.loader.exec_module(_run)
from run_2644 import (ARMS, D_ANCHOR_BPC, D_MODEL, HIDDEN, MLP_MULT, N_HEADS,  # noqa: E402
                      N_LAYERS, SEQ_LEN, U2_ANCHOR_BPC, VOCAB, analytic_cost,
                      build_model, patch_ffn, sparse_ffns)


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


# ---------------------------------------------------------------- T1 等值门

@pytest.mark.parametrize("arm", ["K50", "F50"])
def test_T1_full_mask_is_bitwise_equal_to_stock(arm):
    """`keep_frac = 1.0`（mask 全部通道）= 等值门：整个模型最终深度 logits
    与 stock **逐位相同**（`x * 1.0` 在 IEEE 下精确）。"""
    stock = _stock()
    patched = copy.deepcopy(stock)
    patch_ffn(patched, arm)
    for f in sparse_ffns(patched):
        f.keep = 1.0
        assert f.k_active() == HIDDEN
    x = _tokens(3)
    a, b = _final_logits(stock, x), _final_logits(patched, x)
    assert torch.equal(a, b), (a - b).abs().max().item()


# ---------------------------------------------------------------- T2 变异（门能变红）

def test_T2_half_mask_changes_forward_and_equality_is_not_vacuous():
    """`keep_frac = 0.5` 时 logits 与 stock **不同**（等值门不是恒真）；
    且 K50 与 F50 的半掩码彼此不同（条件选择 ≠ 固定前缀）。"""
    stock = _stock()
    x = _tokens(3)
    base = _final_logits(stock, x)
    outs = {}
    for arm in ("K50", "F50"):
        m = copy.deepcopy(stock)
        patch_ffn(m, arm)
        outs[arm] = _final_logits(m, x)
        assert not torch.equal(base, outs[arm]), (arm, "半掩码没有改变前向 ⇒ 门失效")
    assert not torch.equal(outs["K50"], outs["F50"]), "条件选择与固定前缀产生了同一前向"


# ---------------------------------------------------------------- T3 参数同量

@pytest.mark.parametrize("arm", ARMS)
def test_T3_params_identical_across_arms(arm):
    """总参数量逐臂逐位相同（= 6,962,688，与 dense 锚同）：`SparseFFN` 不引入参数。"""
    stock = _stock()
    model = build_model(arm, VOCAB, seed=0)
    n_stock = sum(p.numel() for p in stock.parameters())
    n_arm = sum(p.numel() for p in model.parameters())
    assert n_stock == n_arm == 6962688


# ---------------------------------------------------------------- T4 死区 / 梯度

def test_T4_f50_unused_channels_have_zero_grad_but_k50_does_not():
    """`F50` 未用（后 `H/2`）通道的 `lin1` 行/`lin2` 列梯度恒 0；
    `K50` 因选择器让**全部**通道都可能进入前向而梯度非 0 ⇒ 条件臂不是死路径。"""
    x = _tokens(9)
    for arm, expect_dead in (("F50", True), ("K50", False)):
        model = build_model(arm, VOCAB, seed=0)
        model.train()
        loss, _ = model(x)
        model.zero_grad()
        loss.backward()
        for blk in model.blocks:
            lin1, lin2 = blk.mlp.mlp[0], blk.mlp.mlp[2]
            rows_dead = int((lin1.weight.grad.abs().sum(-1)[HIDDEN // 2:] == 0).sum())
            cols_dead = int((lin2.weight.grad.abs().sum(0)[HIDDEN // 2:] == 0).sum())
            if expect_dead:
                assert rows_dead == HIDDEN // 2 and cols_dead == HIDDEN // 2
            else:
                assert rows_dead < HIDDEN // 2 or cols_dead < HIDDEN // 2
        model.zero_grad()


# ---------------------------------------------------------------- T5 计算账

def test_T5_analytic_cost_matches_contract_table():
    d = D_MODEL
    attn = N_LAYERS * (4 * d * d + 2 * SEQ_LEN * d)
    for arm, k, ffn in (("K50", 512, N_LAYERS * (d * HIDDEN + 512 * d)),
                        ("K25", 256, N_LAYERS * (d * HIDDEN + 256 * d)),
                        ("F50", 512, N_LAYERS * (d * 512 + 512 * d))):
        c = analytic_cost(arm)
        assert c["k_active_channels"] == k
        assert c["ffn_macs_impl"] == ffn
        assert c["macs_per_token"]["attention"] == attn
        assert c["macs_per_token"]["total"] == attn + ffn + d * VOCAB
    assert analytic_cost("D")["ffn_macs_impl"] == N_LAYERS * 2 * HIDDEN * d == 4194304
    assert analytic_cost("D")["macs_per_token"]["total"] == 7392768
    assert analytic_cost("D")["macs_per_token"]["total"] == 7392768
    assert analytic_cost("F50")["macs_per_token"]["total"] == 5295616


# ---------------------------------------------------------------- T6 因果性

@pytest.mark.parametrize("arm", ARMS)
def test_T6_causality_future_tokens_do_not_change_past_logits(arm):
    model = build_model(arm, VOCAB, seed=0)
    g = torch.Generator().manual_seed(31)
    x = torch.randint(0, VOCAB, (2, SEQ_LEN), generator=g)
    x2 = x.clone()
    j = SEQ_LEN // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, SEQ_LEN - j), generator=g)
    a, b = _final_logits(model, x)[:, :j], _final_logits(model, x2)[:, :j]
    assert torch.equal(a, b)
