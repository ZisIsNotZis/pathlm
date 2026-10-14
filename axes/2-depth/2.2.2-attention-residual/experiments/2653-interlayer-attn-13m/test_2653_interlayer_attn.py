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
