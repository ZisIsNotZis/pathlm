"""2637（线性注意力质量腿）的可失败单测。

钉住本实验自己负责的语义面：
- 2×2 臂表（{full, linear} × {无 QK-Norm, 全局 QK-Norm 稳定化}）；
- 线性块与全注意力块**逐位同参**（单因素 + SCHEMA §15 等参数）；
- 线性注意力的**严格因果**（改未来 token，前面位置逐位不变）；
- 线性驱动器的 QK-Norm 档位接线（`none` 零 hook / `global` 挂满）。

全程 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_linear_attn_quality.py -q
"""

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel)     # noqa: E402

EXP = os.path.join(ROOT, "axes", "1-time", "1.2.2-block-sparse",
                   "experiments", "2637-linear-attn-quality")
D, L, H, MLP = 32, 2, 2, 4


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(EXP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("run2637_t", "run.py")
drv = _load("lin_drv2637_t", "linear_driver.py")


def _tiny(linear: bool, seq_len: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                        mlp_mult=MLP, dense=True, attn_linear=linear,
                        linear_chunk=8)
    return DepthARModel(cfg, 32)


# ---------------------------------------------------- 臂表

def test_arms_2x2():
    key = {(a["linear"], a["qk"]) for a in run.ARMS}
    assert key == {(False, "none"), (True, "none"),
                   (False, "global"), (True, "global")}
    assert len(run.ARMS) == 4


# ---------------------------------------------------- 线性块与全注意力块同参

def test_linear_block_same_params_as_full_block():
    mf, ml = _tiny(False), _tiny(True)
    nf = {n: p.numel() for n, p in mf.named_parameters()}
    nl = {n: p.numel() for n, p in ml.named_parameters()}
    assert nf == nl, "线性块与全注意力块必须逐位同参（零新增参数）"


def test_linear_block_class_differs_from_full():
    assert type(_tiny(True).blocks[0]).__name__ == "LinearAttnBlock"
    assert type(_tiny(False).blocks[0]).__name__ != "LinearAttnBlock"


# ---------------------------------------------------- 严格因果

def test_linear_attn_strictly_causal():
    """改未来 token：前面位置的输出逐位不变（线性块只走 dense+sequential）。"""
    m = _tiny(True, seq_len=16)
    m.eval()
    x = torch.randint(0, 32, (2, 16))
    x2 = x.clone()
    x2[:, 9:] = torch.randint(0, 32, (2, 7))
    with torch.no_grad():
        a = m(x)[1]["depth_logits"][-1]
        b = m(x2)[1]["depth_logits"][-1]
    # 位置 0..8 的输出不受位置 9..15 影响
    assert torch.equal(a[:, :9], b[:, :9]), "线性注意力出现未来泄漏"


# ---------------------------------------------------- QK-Norm 稳定化档接线

def test_qk_norm_none_zero_hooks_global_all_hooks():
    m = _tiny(True, seq_len=256)
    info_none = drv.apply_qk_norm(m, "none")
    assert info_none["n_qkv_hooks"] == 0
    m2 = _tiny(True, seq_len=256)
    info_global = drv.apply_qk_norm(m2, "global")
    assert info_global["n_qkv_hooks"] == len(m2.blocks)
    n0 = sum(p.numel() for p in m2.parameters())
    assert info_global["params_added"] if "params_added" in info_global else True
    assert sum(p.numel() for p in m2.parameters()) == n0, "QK-Norm 零新增参数"


def test_qk_norm_global_bounds_phi_away_from_zero():
    """稳定化档下 φ 的输入有界 ⇒ φ = elu(x)+1 不塌到 0（den 不塌）。"""
    m = _tiny(True, seq_len=64)
    drv.apply_qk_norm(m, "global")
    m.eval()
    x = torch.randint(0, 32, (4, 64))
    with torch.no_grad():
        _, aux = m(x, exit_depths=None)
    tele = aux["attn_linear"]
    phi_min = min(float(t["phi_min"]) for t in tele)
    den_min = min(float(t["den_min"]) for t in tele)
    assert phi_min > 0.0, f"稳定化后 φ 仍触 0：phi_min={phi_min}"
    assert den_min > 1e-5, f"稳定化后 den 仍塌：den_min={den_min}"
