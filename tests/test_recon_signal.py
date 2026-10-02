"""重建训练信号（ticket 36 波二 U2 线，F4）的可失败单测。Run: python -m pytest tests/ -q

钉住四件事（每条都对应契约里的硬要求，不是描述性断言）：

1. **目标定义**：行 i 的目标恰好是 `tokens[i−W]`，且在**真实窗口掩码**下
   「i 行不可见、i−1 行可见」——这是「被窗口遮住 = 必须交班」的物理前提；
2. **λ=0 逐位等价基线**：`add_recon_term` 原样返回**同一个** loss 对象、头参数的
   `grad is None`（AdamW 整条跳过）⇒ 不是「加了 0 倍」，是根本没建图；
3. **单因素与初始化**：解码头建在参数 init 循环之后 ⇒ 同 seed 下「有头/无头」
   两臂主干逐位相同；头**不参与**模型自身 logits；与其它干预显式互斥；
4. **干预活着**：λ>0 时梯度真的流到解码头与主干；活性块在 λ=0 与 λ>0 两种形态下
   给出**不同**的判读（`alive` / `grad_flowed`）。
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             window_anchor_keep_mask)
from pathlm.recon_signal import (recon_targets, recon_loss,       # noqa: E402
                                 recon_liveness, recon_logits,
                                 build_recon_head)
import train_depth_ar as T                                        # noqa: E402

T_SEQ, W = 24, 4


def recon_model(window: int = W, **kw):
    """tiny 模型（同 seed）⇒ 可逐位比较「有头/无头」两臂的主干参数。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=T_SEQ,
                        dense=True, topology="sequential", train_window=W,
                        recon_window=window, **kw)
    return DepthARModel(cfg, vocab_size=10)


# ------------------------------------------------------------------ 目标定义


def test_recon_targets_are_exactly_window_masked():
    """**契约的物理前提**：目标 `tokens[i−W]` 在行 i 时不可见、在行 i−1 时可见。

    用真实掩码仪器（`window_anchor_keep_mask`，训练/评测同一件）逐行核对，
    而不是文字声明——若有人把偏移改成 W−1（= 窗内最后一格）或 W+1（= 隔两格），
    这条测试必须抓红（前者目标可见 ⇒ 退化成平凡项；后者前一格也看不见 ⇒ 不是
    「交班」而是「多步搬运」，语义变了）。
    """
    tokens = torch.randint(0, 10, (2, T_SEQ))
    rows, targets = recon_targets(tokens, W)
    assert rows.tolist() == list(range(W, T_SEQ))
    assert torch.equal(targets, tokens[:, :T_SEQ - W])           # 目标 = tokens[i−W]
    keep = window_anchor_keep_mask(T_SEQ, W, 0)
    for i in rows.tolist():
        j = i - W
        assert not bool(keep[i, j]), (i, j)          # 行 i 看不见它（被窗口遮住）
        assert bool(keep[i - 1, j]), (i, j)          # 行 i−1 看得见它（可交班）
    # 变异方向：偏移取 W−1 时目标（tokens[i−W+1]）落在窗内（可见）⇒ 与契约语义不同
    assert bool(keep[W, W - 1])


def test_recon_targets_rejects_degenerate_window():
    with pytest.raises(ValueError):
        recon_targets(torch.zeros(1, T_SEQ, dtype=torch.long), 0)
    with pytest.raises(ValueError):
        recon_targets(torch.zeros(1, T_SEQ, dtype=torch.long), T_SEQ)


# ------------------------------------------------------------------ 逐位等价与初始化


def test_recon_default_off_matches_v1_bitwise():
    """`recon_window=0`（缺省）逐位还原 v1：无 `recon_head` 参数、前向逐位一致。"""
    m0 = recon_model(0)
    assert "recon_head.weight" not in m0.state_dict()
    assert not hasattr(m0, "recon_head")
    m1 = recon_model(W)
    x = torch.randint(0, 10, (2, T_SEQ))
    with torch.no_grad():
        l0 = m0(x)[1]["depth_logits"][-1]
        l1 = m1(x)[1]["depth_logits"][-1]
    assert torch.equal(l0, l1)                     # 解码头不进模型自身 logits


def test_recon_head_init_does_not_disturb_other_params():
    """解码头建在参数 init 循环之后 ⇒ 主干参数逐位相同（SCHEMA §15 的等参数位点）。"""
    m0, m1 = recon_model(0), recon_model(W)
    p0, p1 = m0.state_dict(), m1.state_dict()
    assert sorted(set(p1) - set(p0)) == ["recon_head.weight"]
    for k in p0:
        assert torch.equal(p0[k], p1[k]), k
    assert float(m1.recon_head.weight.detach().abs().sum()) > 0      # 头是活的
    assert sum(p.numel() for p in m1.parameters()) - \
        sum(p.numel() for p in m0.parameters()) == 10 * 16           # V·d


def test_recon_config_requires_window_and_single_factor():
    """单因素护栏：需要 dense + sequential + train_window>0，且与其它干预互斥。"""
    for kw in ({"dense": False, "topology": "sequential", "train_window": W},
               {"dense": True, "topology": "parallel", "train_window": W},
               {"dense": True, "topology": "sequential", "train_window": 0},
               {"dense": True, "topology": "sequential", "train_window": W,
                "attn_linear": True},
               {"dense": True, "topology": "sequential", "train_window": W,
                "carry_rule": "delta"},
               {"dense": True, "topology": "sequential", "train_window": W,
                "n_kv_heads": 1}):
        with pytest.raises(ValueError):
            torch.manual_seed(0)
            DepthARModel(DepthARConfig(d_model=16, n_layers=2, n_heads=2,
                                       seq_len=T_SEQ, recon_window=W, **kw),
                         vocab_size=10)
    with pytest.raises(ValueError):               # 偏移越界
        torch.manual_seed(0)
        cfg = DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=8,
                            dense=True, train_window=4, recon_window=8)
        DepthARModel(cfg, vocab_size=10)
    # 合法组合真的能建起来（防止「护栏把正路也堵死」）
    recon_model(W)(torch.randint(0, 10, (2, T_SEQ)))


def test_recon_cli_plumbed_to_config():
    """CLI 通道（可失败）：`--recon-window` 必须真的进 `DepthARConfig`。"""
    import argparse
    base = dict(d=16, layers=2, heads=2, seq_len=T_SEQ, mlp_mult=4, dense=True,
                gate_bias_init=-2.0, topology="sequential", fill_kv=True,
                proj_fill=False, readout_dims=None, readout_gated=False,
                readout_linear_resid=False, attn_sparse_k=0, attn_linear=False,
                linear_chunk=64, train_window=W, train_anchors=0,
                n_kv_heads=None, carry_chunk=64, carry_state_dim=None,
                carry_write="none")
    cfg = T.build_depthar_config(argparse.Namespace(**base, recon_window=W))
    assert cfg.recon_window == W
    torch.manual_seed(0)
    assert hasattr(DepthARModel(cfg, vocab_size=10), "recon_head")
    cfg_off = T.build_depthar_config(argparse.Namespace(**base, recon_window=0))
    assert cfg_off.recon_window == 0
    assert not hasattr(DepthARModel(cfg_off, vocab_size=10), "recon_head")


# ------------------------------------------------------------------ 损失与 λ=0 等价


def test_recon_loss_matches_manual_ce():
    """损失 = `recon_head(hidden[:, W:])` 对 `tokens[:, :T−W]` 的逐行 CE 均值。

    这条同时钉住**目标索引**（把偏移写错会立刻不等）与 fp32 口径。
    """
    m = recon_model(W)
    x = torch.randint(0, 10, (2, T_SEQ))
    with torch.no_grad():
        _, aux = m(x)
        hidden = aux["states"][-1]
        stats = recon_loss(m.recon_head, hidden, x, W, filler_token=7)
        logits = recon_logits(m.recon_head, hidden[:, W:])
        manual = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 10).float(), x[:, :T_SEQ - W].reshape(-1))
    assert torch.allclose(stats["loss"], manual, atol=1e-6)
    assert int(stats["n_targets"]) == 2 * (T_SEQ - W)
    assert int(stats["n_symbol"]) + int(stats["n_filler"]) == int(stats["n_targets"])
    assert int(stats["n_filler"]) == int((x[:, :T_SEQ - W] == 7).sum())
    # fp32：头的输出是 float32（autocast 关闭），不是 bf16
    assert logits.dtype == torch.float32


def test_recon_logits_is_fp32_under_bf16_autocast():
    """**fp32 是硬要求**（F4 教训：bf16 下这类损失两次发散）：即使外层在 bf16
    autocast 内，`recon_logits` 也必须输出 float32。"""
    m = recon_model(W)
    h = torch.randn(2, T_SEQ, 16) * 0.1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = recon_logits(m.recon_head, h)
        # 对照：同一层在 autocast 内直接调用会变 bf16（证明这条测试真的能抓）
        naive = m.recon_head(h)
    assert out.dtype == torch.float32
    assert naive.dtype == torch.bfloat16
    # device_type 不匹配的 off 上下文拦不住外层 autocast（实测教训，钉住）
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.autocast("cuda", enabled=False):
            unmatched = m.recon_head(h)
    assert unmatched.dtype == torch.bfloat16


def test_recon_loss_split_is_diagnostic_only():
    """`filler_token` 只是诊断参数：损失值（被 λ 乘的那一项）必须与它无关。"""
    m = recon_model(W)
    x = torch.randint(0, 10, (2, T_SEQ))
    with torch.no_grad():
        _, aux = m(x)
        a = recon_loss(m.recon_head, aux["states"][-1], x, W, filler_token=None)
        b = recon_loss(m.recon_head, aux["states"][-1], x, W, filler_token=3)
    assert torch.equal(a["loss"], b["loss"])
    assert int(b["n_filler"]) == int((x[:, :T_SEQ - W] == 3).sum())
    assert int(a["n_symbol"]) == int(a["n_targets"])          # 不拆时全算 symbol


def test_recon_lambda_zero_returns_same_loss_object():
    """**硬要求（λ=0 逐位等价）**：返回的是**同一个对象** ⇒ 不建图、无浮点运算。"""
    m = recon_model(W)
    x = torch.randint(0, 10, (2, T_SEQ))
    loss, aux = m(x)
    out0, stats0 = T.add_recon_term(loss, m, aux["states"][-1], x, 0.0, 7)
    assert out0 is loss and stats0 is None
    out1, stats1 = T.add_recon_term(loss, m, aux["states"][-1], x, 1.0, 7)
    assert out1 is not loss and stats1 is not None
    assert torch.allclose(out1 - loss.detach(), stats1["loss"], atol=1e-6)
    # recon_window=0 的模型即使 λ>0 也走「原样返回」分支（不是死接线）
    m0 = recon_model(0)
    loss0, aux0 = m0(x)
    out2, stats2 = T.add_recon_term(loss0, m0, aux0["states"][-1], x, 1.0, 7)
    assert out2 is loss0 and stats2 is None


def test_recon_head_dead_at_lambda_zero_grads_flow_at_lambda_one():
    """λ=0：头参数 `grad is None`（AdamW 整条跳过）；λ>0：头与主干都拿到梯度。"""
    x = torch.randint(0, 10, (2, T_SEQ))
    m = recon_model(W)
    loss, aux = m(x)
    loss.backward()                                    # 基线路径（λ=0）
    assert m.recon_head.weight.grad is None
    assert m.embed.weight.grad is not None
    m.zero_grad(set_to_none=True)
    loss2, aux2 = m(x)                                  # 重新前向（不用 retain_graph）
    total, stats = T.add_recon_term(loss2, m, aux2["states"][-1], x, 1.0, 7)
    total.backward()
    assert m.recon_head.weight.grad is not None
    assert float(m.recon_head.weight.grad.abs().sum()) > 0
    assert float(m.embed.weight.grad.abs().sum()) > 0   # 辅助信号穿回主干


def test_recon_mutation_target_shift_changes_loss():
    """变异方向（把目标偏移改成 W−1 或 W+1）必须改变损失值——索引写错抓得红。"""
    m = recon_model(W)
    x = torch.randint(0, 10, (2, T_SEQ))
    with torch.no_grad():
        _, aux = m(x)
        h = aux["states"][-1]
        good = recon_loss(m.recon_head, h, x, W)["loss"]
        bad_lo = recon_loss(m.recon_head, h, x, W - 1)["loss"]
        bad_hi = recon_loss(m.recon_head, h, x, W + 1)["loss"]
    assert not torch.allclose(good, bad_lo)
    assert not torch.allclose(good, bad_hi)


# ------------------------------------------------------------------ 活性量


def test_recon_liveness_reports_alive_and_dead():
    """SCHEMA §10 活性块：λ=0（头未被训练）与 λ>0（梯度真的流过）判读必须不同。"""
    m = recon_model(W)
    off = recon_liveness(None, 0, 0.0)
    assert off["enabled"] is False and off["alive"] is False
    assert off["params_added"] == 0 and off["w_fro"] is None
    untrained = recon_liveness(m.recon_head, W, 0.0,
                               {"ce_all": 2.0, "head_gnorm": None, "n_steps": 5})
    assert untrained["enabled"] is True and untrained["alive"] is False
    assert untrained["params_added"] == 10 * 16
    assert untrained["nonzero_frac"] > 0 and untrained["head_gnorm_mean"] is None
    assert untrained["grad_flowed"] is None and "未被训练" in untrained["alive_note"]
    alive = recon_liveness(m.recon_head, W, 1.0,
                           {"ce_all": 1.5, "ce_symbol": 2.0, "acc_symbol": 0.1,
                            "head_gnorm": 0.25, "n_steps": 5,
                            "n_symbol_frac": 0.5})
    assert alive["alive"] is True and alive["grad_flowed"] is True
    assert alive["recon_ce_mean"] == 1.5 and alive["recon_ce_symbol_mean"] == 2.0


def test_build_recon_head_rejects_off_config():
    """`build_recon_head` 自己也要护栏（它是在 `__init__` 里被调用的那一层）。"""
    cfg = DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=T_SEQ,
                        dense=True, topology="sequential", train_window=0,
                        recon_window=W)
    with pytest.raises(ValueError):
        build_recon_head(cfg, 10)              # 没有窗口就没有「刚掉出窗口的那一格」
    with pytest.raises(ValueError):
        build_recon_head(DepthARConfig(recon_window=0), 10)
