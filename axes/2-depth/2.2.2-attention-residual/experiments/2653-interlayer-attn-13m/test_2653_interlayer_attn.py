"""2653 单测（`pytest axes/2-depth/2.2.2-attention-residual/experiments/
2653-interlayer-attn-13m/test_2653_interlayer_attn.py -q`）。CPU，秒级。

钉住契约 §4 的三条门：①等值门（one-hot ⇒ 与 stock 单残差逐位一致）；②活性
（层维权重非退化，且度量有分辨力）；③参数账（额外 = L / 3d²+L）。每条各带一个
**必须失效**的探针（`proof.gate_probe`）。
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import interlayer_attn as IA                              # noqa: E402
from smoke import (build_attn, build_stock, forward,      # noqa: E402
                   trunk_equal)


def test_a_forced_one_hot_equals_stock_single_residual_bitwise():
    """① 等值门：把输出权重强制成只取紧邻上一层 ⇒ 与 stock 逐位一致（前向 + loss）。"""
    stock, forced = build_stock(), build_attn("scalar", force_last=True)
    assert trunk_equal(stock, forced, ("tau", "wq", "wk", "wv"))
    assert torch.equal(forward(stock), forward(forced))
    assert torch.equal(forward(stock, aux_out=True)[0],
                       forward(forced, aux_out=True)[0])


def test_a_probe_free_weights_differ_from_stock():
    """① 探针：不强制时前向必须不同，否则等值门测的是「什么都一样」。"""
    assert not torch.equal(forward(build_stock()),
                           forward(build_attn("scalar", force_last=False)))


@pytest.mark.parametrize("mode", ["scalar", "qkv"])
def test_b_layer_dim_weights_are_non_degenerate(mode):
    """② 活性：层维注意力权重偏离「只取上一层」（k ≥ 2 的层）。"""
    m = build_attn(mode, force_last=False)
    forward(m)
    tele = m.attn_tele
    assert len(tele) == 8
    for t in tele:
        if t["layer"] == 1:            # k=1：唯一源，平凡
            continue
        assert t["entropy"] > 0.0
        assert t["w_immediate_prev"] < 1.0


def test_b_probe_liveness_metric_has_resolution():
    """② 探针：本度量能把 one-hot / uniform 两个端点分开（否则它说明不了活性）。"""
    k = 4
    w_one = torch.zeros(1, 1, k)
    w_one[..., k - 1] = 1.0
    w_uni = torch.full((1, 1, k), 1.0 / k)
    assert abs(float(w_one[..., k - 1].mean()) - 1.0) < 1e-6
    assert abs(float(w_uni[..., k - 1].mean()) - 1.0 / k) < 1e-6
    assert abs(float(-(w_one * w_one.clamp_min(1e-12).log()).sum(-1).mean())) < 1e-6
    assert float(-(w_uni * w_uni.clamp_min(1e-12).log()).sum(-1).mean()) > 1.0


@pytest.mark.parametrize("mode,expected", [("scalar", 8), ("qkv", 8 + 3 * 256 * 256)])
def test_c_extra_params_match_analytic(mode, expected):
    """③ 参数账：额外参数 = L（scalar）/ 3d² + L（qkv）。"""
    base = sum(p.numel() for p in build_stock().parameters())
    m = build_attn(mode, force_last=False)
    assert sum(p.numel() for p in m.parameters()) - base == expected
    assert m.aggregation_cost()["extra_params"] == expected


def test_c_probe_qkv_extra_params_differ_from_scalar():
    """③ 探针：两臂额外参数必须不同，否则这条账无分辨力。"""
    base = sum(p.numel() for p in build_stock().parameters())
    a = sum(p.numel() for p in build_attn("scalar").parameters()) - base
    b = sum(p.numel() for p in build_attn("qkv").parameters()) - base
    assert a != b


def test_factory_global_is_frozen_at_construction():
    """回归：`MODE`/`FORCE_LAST` 必须在建实例时冻结（前向时读全局会让等值门失效）。"""
    forced = build_attn("scalar", force_last=True)
    free = build_attn("scalar", force_last=False)      # 改全局
    assert forced.force_last is True and free.force_last is False
    assert not torch.equal(forward(forced), forward(free))


@pytest.mark.parametrize("mode", ["scalar", "qkv"])
def test_d_topology_liveness_is_self_reported_not_dense_layer_mix(mode):
    """回归（非契约 §4 门）：本臂的层间聚合不是线性层权，构造时已删 `layer_mix`，
    而共享 `topology_liveness` 曾假定 dense ⇒ 落盘阶段抛
    `AttributeError: 'InterLayerAttentionModel' object has no attribute 'layer_mix'`
    （2653 训练完 12k 步后在此崩 781 次，每次约 9.5 分钟 GPU）。

    现在：共享入口转调本类自报的 `topology_liveness()`，活性看得见（τ 偏离初值）；
    且 τ 未动时 `alive` 必须为 False（否则该量没有分辨力）。
    """
    from pathlm.depth_ar import topology_liveness
    m = build_attn(mode, force_last=False)
    assert not hasattr(m, "layer_mix")        # 本臂真没有线性层权
    live = topology_liveness(m)               # 修复前：AttributeError
    assert live["alive"] is False             # τ 还在初值 1.0 ⇒ 未训 ⇒ 不活
    assert live["tau_absmax_dev_from_init"] == 0.0
    assert live["attn_weights"] is None       # 尚无真实前向样本
    d, L = m.cfg.d_model, m.cfg.n_layers
    assert live["params_added"] == L + (3 * d * d if mode == "qkv" else 0)
    with torch.no_grad():                     # 模拟训过的 τ：离开初值 ⇒ 活
        m.tau += torch.linspace(0.0, 0.5, L)
    forward(m)                                # 真实前向 ⇒ 聚合权重摘要
    live = topology_liveness(m)
    assert live["alive"] is True and live["tau_absmax_dev_from_init"] > 0
    assert live["attn_weights"]["n_samples"] == L - 1   # k ≥ 2 的层
    assert live["attn_weights"]["w_immediate_prev_min"] < 1.0
