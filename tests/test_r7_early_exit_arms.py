"""R7「既有早退实现」对照臂的引擎改动单测（可失败，SCHEMA §10/§22.8）。

钉三件事：
1. `aux_mode="additive"` 的归一化语义（主目标不被摊薄）＋ `"renorm"` 逐位不变；
2. `carry_gate=False`（无门变体）的语义与初始化随机数流（臂间可比）；
3. `MoDModel` 的容量路由语义、旁路不变量、路由器损失形式与活性。

Run: python -m pytest tests/test_r7_early_exit_arms.py -q
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathlm.depth_ar import (DepthARConfig, DepthARModel, DepthARDecoder,
                             FrozenOpenGate, sample_exit_depths, token_input)
from pathlm.mod_router import MoDModel, mod_router_loss, mod_eval


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12, **kw)
    return DepthARModel(cfg, vocab_size=10)


def tiny_mod(ratio=0.5, weight=0.01, **kw):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12,
                        dense=True, mod_ratio=ratio, **kw)
    return MoDModel(cfg, vocab_size=10, router_weight=weight)


# ---------- 1. aux_mode ----------


def test_renorm_mode_matches_v1_formula_bitwise():
    """缺省 renorm 必须逐位等于 v1 加权平均式（改动的回归护栏）。"""
    m = tiny_model()
    torch.manual_seed(3)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(4))
    loss, aux = m(x, exit_depths=depths, shallow_weight=1.0, final_weight=1.0)
    ces = torch.stack(aux["depth_ce"])
    expected = (ces[:-1].sum() + ces[-1]) / (m.cfg.n_layers + 1.0)
    assert torch.equal(loss, expected)


def test_additive_mode_is_main_weight_plus_light_mean_shallow():
    """additive：loss == f·CE_last + w·mean(浅层 CE)（主目标保绝对权重）。"""
    m = tiny_model(aux_mode="additive")
    torch.manual_seed(5)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(6))
    loss, aux = m(x, exit_depths=depths, shallow_weight=0.1, final_weight=1.0)
    ces = torch.stack(aux["depth_ce"])
    expected = ces[-1] + 0.1 * ces[:-1].mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    # 主目标权重与任务无关地保持 1：λ→0 时 loss → CE_last（不是它的 1/(1+λL) 倍）
    loss0, aux0 = m(x, exit_depths=depths, shallow_weight=0.0, final_weight=1.0)
    assert torch.equal(loss0, torch.stack(aux0["depth_ce"])[-1])


def test_additive_changes_only_normalization_not_states():
    """唯一变量 = 归一化：两模式的 states/逐深度 CE 逐位相同。"""
    A = tiny_model()
    B = tiny_model(aux_mode="additive")
    seed = torch.Generator().manual_seed(11)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, A.cfg.n_layers, seed)
    _, a = A(x, exit_depths=depths)
    _, b = B(x, exit_depths=depths)
    for s1, s2 in zip(a["states"], b["states"]):
        assert torch.equal(s1, s2)
    for c1, c2 in zip(a["depth_ce"], b["depth_ce"]):
        assert torch.equal(c1, c2)


def test_additive_with_zero_weight_equals_stock_dense_arm():
    """λ=0 + 无门 + 全深度 ⇒ 与 stock dense 臂的 loss/states **逐位相同**
    （「公平配方 + 普通 transformer」确实回到 stock）。"""
    A = tiny_model(carry_gate=False, aux_mode="additive")
    B = tiny_model(dense=True)
    missing = B.load_state_dict(
        {k: v for k, v in A.state_dict().items() if k in B.state_dict()},
        strict=False)
    assert not missing.unexpected_keys
    torch.manual_seed(17)
    x = torch.randint(0, 10, (2, 12))
    l_a, aux_a = A(x, exit_depths=None, shallow_weight=0.0, final_weight=1.0)
    l_b, aux_b = B(x)
    for s1, s2 in zip(aux_a["states"], aux_b["states"]):
        assert torch.equal(s1, s2)
    assert torch.equal(l_a, l_b)


# ---------- 2. carry_gate=False ----------


def test_no_gate_arm_has_no_gate_parameters():
    g = tiny_model()
    ng = tiny_model(carry_gate=False)
    n_gate = sum(p.numel() for p in g.gates.parameters())
    assert n_gate == g.cfg.n_layers * (16 * 16 + 16)
    assert sum(p.numel() for p in ng.gates.parameters()) == 0
    assert isinstance(ng.gates[0], FrozenOpenGate)
    assert (sum(p.numel() for p in g.parameters())
            - sum(p.numel() for p in ng.parameters())) == n_gate


def test_no_gate_arm_keeps_trunk_init_rng_stream():
    """无门臂的 trunk 权重必须与有门臂同 seed **逐位相同**（否则臂间不可比：
    门少抽了 L 次签，后续参数会整体错位）。"""
    g = tiny_model()
    ng = tiny_model(carry_gate=False)
    for k in ("embed.weight", "pos_embed.weight", "blocks.0.qkv.weight",
              "blocks.2.mlp.2.weight"):
        assert torch.equal(g.state_dict()[k], ng.state_dict()[k]), k


def test_no_gate_mixed_depth_matches_manual_rule():
    """无门混合深度：h_next = where(active, block(h, kv_from=kv_in), frozen)
    （块输出原样写入，无 g⊙delta 回环）。"""
    m = tiny_model(carry_gate=False)
    torch.manual_seed(19)
    x = torch.randint(0, 10, (1, 12))
    depths = torch.tensor([[0, 1, 2, 3, 1, 0, 3, 2, 0, 1, 3, 2]])
    _, aux = m(x, exit_depths=depths)
    st = aux["states"]
    h = st[0]
    frozen = st[0]
    for k in range(1, m.cfg.n_layers + 1):
        active = depths >= k
        kv_in = torch.where(active.unsqueeze(-1), h, frozen)
        out, _ = m.blocks[k - 1](h, kv_from=kv_in)
        h = torch.where(active.unsqueeze(-1), out, frozen)
        frozen = torch.where((depths == k).unsqueeze(-1), h, frozen)
        assert torch.allclose(st[k], h, atol=1e-5), f"layer {k}"
    # 门活性量槽位：无门臂记 0，不从门开度读机制
    assert all(float(g) == 0.0 for g in aux["gate_mean"])


def test_no_gate_decoder_matches_parallel_forward():
    """无门臂的增量解码也必须与并行混合深度 forward 一致（缓存语义的端到端
    钉：`DepthARDecoder` 的门分支被绕过时不能静默换成别的语义）。"""
    m = tiny_model(carry_gate=False)
    torch.manual_seed(23)
    x = torch.randint(0, 10, (1, 12))
    depths = torch.tensor([[0, 3, 1, 2, 0, 3, 3, 1, 2, 0, 1, 3]])
    _, aux = m(x, exit_depths=depths)
    dec = DepthARDecoder(m)
    for i in range(12):
        _, logits = dec.step(token_input(m, int(x[0, i]), dec.n),
                             int(depths[0, i]))
        e = int(depths[0, i])
        assert torch.allclose(logits, aux["depth_logits"][e][0, i], atol=2e-4), \
            f"position {i} exit {e}"


def test_carry_gate_config_rejects_dense():
    """carry_gate=False 在 dense=True 下是死开关 ⇒ 必须显式报错（SCHEMA §10）。"""
    with pytest.raises(ValueError):
        tiny_model(dense=True, carry_gate=False)


def test_aux_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        tiny_model(aux_mode="mean")


# ---------- 3. MoD ----------


def test_mod_capacity_is_exact_per_sequence():
    m = tiny_mod(ratio=0.5)
    torch.manual_seed(29)
    x = torch.randint(0, 10, (4, 12))
    _, aux = m(x)
    cap = int(round(0.5 * 12))
    assert aux["mod"]["capacity_per_seq"] == cap
    for f in aux["mod"]["routed_frac_per_layer"]:
        assert f == pytest.approx(cap / 12, abs=1e-6)
    # 逐 token 的块作用次数 = 放行比例 × 层数（解析算力账）
    assert aux["mod"]["fwd_counts_mean"] == pytest.approx(cap / 12 * 3, abs=1e-6)


def test_mod_unselected_tokens_bypass_bitwise():
    """旁路不变量：每层未选中位置的状态必须与该层输入**逐位相同**。"""
    m = tiny_mod(ratio=0.25)
    torch.manual_seed(31)
    x = torch.randint(0, 10, (2, 12))
    h = m._input_states(x)
    states = [h]
    for k0, blk in enumerate(m.blocks):
        logits = m.routers[k0](h).squeeze(-1)
        cap = max(1, int(round(0.25 * 12)))
        idx = torch.topk(logits, cap, dim=1).indices
        sel = torch.zeros_like(logits, dtype=torch.bool)
        sel.scatter_(1, idx, True)
        d = h.shape[-1]
        h_sel = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
        allow = (idx.unsqueeze(-1) >= idx.unsqueeze(-2)).unsqueeze(1).expand(
            -1, m.cfg.n_heads, -1, -1)
        out, _ = blk(h_sel, attn_mask=allow)
        h_new = h.clone()
        h_new.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, d), out)
        for b in range(2):
            un = (~sel[b]).nonzero().flatten()
            assert torch.equal(h_new[b, un], h[b, un])
        h = h_new
        states.append(h)
    _, aux = m(x)
    for s1, s2 in zip(states, aux["states"]):
        assert torch.equal(s1, s2)


def test_mod_loss_is_main_objective_plus_weighted_router_loss():
    """loss == 最终深度 CE + α·router_loss（主目标不降权）；α=0 时逐位等于 CE。"""
    m = tiny_mod(ratio=0.5, weight=0.01)
    torch.manual_seed(37)
    x = torch.randint(0, 10, (2, 12))
    loss, aux = m(x)
    ce = aux["depth_ce"][0]
    assert torch.allclose(loss, ce + 0.01 * aux["mod"]["router_loss"], atol=1e-6)
    m0 = tiny_mod(ratio=0.5, weight=0.0)
    m0.load_state_dict(m.state_dict(), strict=False)
    loss0, aux0 = m0(x)
    assert torch.equal(loss0, aux0["depth_ce"][0])


def test_mod_router_loss_form_and_direction():
    """路由器损失形式：选中推向 1、未选中推向 capacity 比例（不压到 0）。
    手算 + 变异方向（把选中/未选中对调必须使损失变化）。"""
    prob = torch.tensor([[0.9, 0.2, 0.5, 0.4]])
    sel = torch.tensor([[True, False, True, False]])
    r = 0.5
    expected = (((1 - 0.9) + (1 - 0.5)) / 2 + r * (0.2 + 0.4) / 2)
    assert float(mod_router_loss(prob, sel, r)) == pytest.approx(expected, abs=1e-6)
    assert float(mod_router_loss(prob, sel, r)) != pytest.approx(
        float(mod_router_loss(prob, ~sel, r)), abs=1e-3)


def test_mod_router_params_receive_gradient():
    """活性：路由器参数真的拿到梯度（死开关检测）。"""
    m = tiny_mod(ratio=0.5)
    torch.manual_seed(41)
    x = torch.randint(0, 10, (2, 12))
    loss, _ = m(x)
    loss.backward()
    for r in m.routers:
        assert r.weight.grad is not None and float(r.weight.grad.abs().sum()) > 0
    assert m.routers[0].weight.shape == (1, 16)


def test_mod_rejects_non_dense_and_bad_ratio():
    with pytest.raises(ValueError):
        torch.manual_seed(0)
        MoDModel(DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=12,
                               dense=False, mod_ratio=0.5), vocab_size=10)
    with pytest.raises(ValueError):
        tiny_mod(ratio=0.0)
    with pytest.raises(ValueError):
        tiny_model(dense=True, mod_ratio=1.5)
    with pytest.raises(ValueError):
        tiny_model(mod_ratio=0.5)   # 非 dense ⇒ 报错


def test_mod_isoflop_analytic_account():
    """MoD 的解析算力账：每 token 块作用次数 == ratio·L（isoFLOP 设计点的定义）。"""
    m = tiny_mod(ratio=0.5)
    torch.manual_seed(43)
    x = torch.randint(0, 10, (3, 12))
    _, aux = m(x)
    assert aux["mod"]["fwd_counts_mean"] == pytest.approx(0.5 * m.cfg.n_layers,
                                                          abs=1e-6)


def test_mod_eval_reports_both_measured_and_analytic_compute():
    """`mod_eval` 必须同时给实测放行比例与解析账（防「路由器是死的」）。"""
    m = tiny_mod(ratio=0.5)
    ev = torch.randint(0, 10, (4000,))
    out = mod_eval(m, ev, n_batches=2, batch_size=2, device="cpu")
    assert out["capacity_frac"] == 0.5
    assert out["layer_apps_per_token_measured"] == pytest.approx(1.5, abs=1e-3)
    assert out["layer_apps_per_token_analytic"] == pytest.approx(1.5)
    assert out["routed_frac_mean"] == pytest.approx(0.5, abs=1e-6)
    assert out["router_params"] == m.cfg.n_layers * (16 + 1)
    assert torch.isfinite(torch.tensor(out["bpc"]))


# ---------- 4. CLI → config 接线（开关不能是死的） ----------


def test_r7_cli_flags_plumbed_to_config():
    import argparse
    from train_depth_ar import build_depthar_config
    ns = argparse.Namespace(
        d=16, layers=3, heads=2, seq_len=12, mlp_mult=4, dense=True,
        gate_bias_init=-2.0, topology="sequential", fill_kv=True, proj_fill=False,
        readout_dims=None, readout_gated=False, readout_linear_resid=False,
        attn_sparse_k=0, attn_linear=False, linear_chunk=64, train_window=0,
        train_anchors=0, n_kv_heads=None, carry_write="none", carry_chunk=64,
        carry_state_dim=None, carry_gate=False, aux_mode="additive",
        mod_ratio=0.5)
    cfg = build_depthar_config(ns)
    assert cfg.carry_gate is False and cfg.aux_mode == "additive"
    assert cfg.mod_ratio == 0.5
    # 历史 Namespace（不带这三个字段）必须落到 v1 缺省
    del ns.carry_gate, ns.aux_mode, ns.mod_ratio
    cfg2 = build_depthar_config(ns)
    assert cfg2.carry_gate is True and cfg2.aux_mode == "renorm"
    assert cfg2.mod_ratio == 0.0
