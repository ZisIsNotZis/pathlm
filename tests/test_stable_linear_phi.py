"""2685 的两处新机检：无消去 φ（`stable_linear.py`）与同前向 FLOPs 计数（`flops_count.py`）。

钉住四件事（都能红）：

1. **φ 的函数没变**：在非塌陷区（|x| ≤ 15）`stable_phi(x)` 与引擎默认 `F.elu(x)+1` 逐值相同；
2. **塌陷区被修掉**：`x ≤ −20` 时 `F.elu(x)+1 == 0`（引擎默认档的分母塌陷来源），而 `stable_phi(x) > 0`；
3. **install() 真的改了引擎**：装档后 `pathlm.depth_ar.linear_attn_phi` 就是本模块的函数、`LINEAR_ATTN_EPS`
   就是声明值（装完恢复，避免污染同进程的其它测试）；
4. **FLOPs 计数与算式一致**：`full` 臂每 token 等效 MAC = `d_model·(T+1)`；掩码臂 =
   `2·d_model·nnz/T`；同 nnz 的两张掩码给出逐位相同的计数；线性块 = 闭式
   `B·H·(2·T·dh² + 2·T·C·dh + 2·T·dh)`。

全程 CPU、秒级，不取 GPU 锁。

Run: python3 -m pytest tests/test_stable_linear_phi.py -q
"""

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             window_anchor_keep_mask)
import pathlm.depth_ar as da  # noqa: E402

EXP = os.path.join(ROOT, "axes", "1-time", "1.2.2-block-sparse",
                   "experiments", "2685-recall-flops-linear-13m")


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(EXP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stable_linear = _load("stable_linear_2685_test", "stable_linear.py")
flops_count = _load("flops_count_2685_test", "flops_count.py")

D, L, H, MLP, T = 32, 2, 2, 4, 32


def test_stable_phi_equals_elu_plus_one_away_from_the_collapse():
    x = torch.linspace(-15, 15, 601, dtype=torch.float32)
    ref = F.elu(x) + 1.0
    got = stable_linear.stable_phi(x)
    assert torch.allclose(got, ref, rtol=1e-6, atol=1e-7)   # 差在 fp32 舍入量级（≤6e-8）
    assert float((got - ref).abs().max()) <= 1e-7


def test_stable_phi_keeps_positivity_where_elu_underflows():
    x = torch.tensor([-20.0, -30.0, -60.0, -85.0], dtype=torch.float32)
    assert torch.all(F.elu(x) + 1.0 == 0)                 # 引擎默认档：φ 下溢为 0
    got = stable_linear.stable_phi(x)
    assert torch.all(got > 0)                              # 无消去档：φ 仍 > 0
    assert got[0].item() == torch.exp(torch.tensor(-20.0)).item()


def test_install_replaces_engine_globals_and_restores():
    phi0, eps0 = da.linear_attn_phi, da.LINEAR_ATTN_EPS
    try:
        stable_linear.install()
        assert da.linear_attn_phi is stable_linear.stable_phi
        assert da.LINEAR_ATTN_EPS == stable_linear.LINEAR_ATTN_EPS_DECLARED
        # 引擎块真的走到了新档：φ 的塌陷点不再产出 0
        x = torch.tensor([-20.0], dtype=torch.float32)
        assert da.linear_attn_phi(x).item() > 0
    finally:
        da.linear_attn_phi, da.LINEAR_ATTN_EPS = phi0, eps0
    assert da.linear_attn_phi is phi0 and da.LINEAR_ATTN_EPS == eps0


def _tiny(attn_linear: bool = False):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=T, mlp_mult=MLP,
                        dense=True, attn_linear=attn_linear, linear_chunk=8)
    return DepthARModel(cfg, 64).eval()


def test_flops_count_full_equals_analytic_and_masked_scales_with_nnz():
    model = _tiny()
    x = torch.randint(0, 60, (4, T))
    full = flops_count.count_attention(model, x, None)
    assert full["equiv_macs_per_token"] == D * (T + 1)        # 2·d_head·pairs / token，pairs = T(T+1)/2
    keep = window_anchor_keep_mask(T, 8, 0)
    band = flops_count.count_attention(model, x, keep)
    nnz = int(keep.sum())
    assert band["equiv_macs_per_token"] == round(2 * D * nnz / T, 3)
    assert band["kept_keys_per_row_mean"] == round(nnz / T, 3)
    assert band["executed_macs_per_token"] == full["equiv_macs_per_token"]   # 稠密 bool 掩码：算整张因果矩阵


def test_flops_count_same_nnz_gives_identical_equiv():
    model = _tiny()
    x = torch.randint(0, 60, (2, T))
    a = window_anchor_keep_mask(T, 8, 0)
    b = window_anchor_keep_mask(T, 6, 2)          # 每行 8 个键的另一种分配（同 nnz 保证见下）
    assert int(a.sum()) == int(b.sum())
    ca = flops_count.count_attention(model, x, a)
    cb = flops_count.count_attention(model, x, b)
    assert ca["equiv_macs_per_token"] == cb["equiv_macs_per_token"]


def test_flops_count_linear_block_matches_closed_form():
    model = _tiny(attn_linear=True)
    x = torch.randint(0, 60, (2, T))
    got = flops_count.count_attention(model, x, None)
    dh = D // H
    C = min(model.cfg.linear_chunk, T)
    per_layer = 2 * (T * dh * dh) + 2 * (T * C * dh) + 2 * (T * dh)
    want = round(L * per_layer / T, 3)
    assert got["equiv_macs_per_token"] == want
    assert got["executed_macs_per_token"] == want          # 线性块算的就是它自己的 O(T) 算式
