"""2636（长 T 四点统一口径）的可失败单测。

钉住本实验自己负责的语义面：
- **同 step 口径常量**（S=6000）与臂表（四配置 × 两 seed，窗口 = T/2，同相对刻度）；
- **窗口掩码** 在四个点（128@256 / 256@512 / 512@1024 / 1024@2048）非退化、且 w ≥ T 时为恒等；
- 成本口径 `cost_theory` 的公式值；
- 复用 `2625` 的长度条件档 QK-Norm hook 在本实验上下文里**再次钉一次**
  （T ≤ 512 零 hook 且前向逐位等于无 hook）。

全程 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_unified.py -q
"""

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,     # noqa: E402
                             window_anchor_keep_mask)

EXP = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window",
                   "experiments", "2636-longT-unified")
D, L, H, MLP = 32, 2, 2, 4


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "run2636_t", os.path.join(EXP, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load_run()


def _tiny(seq_len: int, window: int = 0, seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                        mlp_mult=MLP, dense=True, train_window=window)
    return DepthARModel(cfg, 32)


# ---------------------------------------------------- 口径常量与臂表

def test_step_budget_is_6000():
    assert run.S == 6000


def test_tokens_at_s_match_formula():
    for T in (256, 512, 1024, 2048):
        assert run.TOKENS_AT_S[T] == run.S * run.BS * T


def test_arms_table_four_configs_two_seeds():
    key = {(a["seq_len"], a["window"], a["seed"]) for a in run.ARMS}
    assert len(run.ARMS) == 8
    for T in (256, 512):
        for seed in (0, 1):
            assert (T, 0, seed) in key
            assert (T, T // 2, seed) in key


def test_points_are_four_with_matched_relative_scale():
    assert run.POINTS == ((128, 256), (256, 512), (512, 1024), (1024, 2048))
    for w, T in run.POINTS:
        assert w == T // 2, "窗口必须是 T 的同一相对刻度（50%）"


# ---------------------------------------------------- 窗口掩码：非退化 + 恒等护栏

def test_window_mask_non_degenerate_at_points():
    """四个点必须在训练分布内非恒等、也非空集（§17.16 第 2 条）。"""
    for w, T in run.POINTS:
        keep = window_anchor_keep_mask(T, w, 0, "cpu")
        ident = torch.tril(torch.ones(T, T, dtype=torch.bool))
        frac = float(keep.sum()) / float(ident.sum())
        assert 0.5 < frac < 1.0, f"{w}@{T} 退化：visible_frac={frac}"
        assert not torch.equal(keep, ident), f"{w}@{T} 是恒等算子"


def test_window_mask_identity_when_window_ge_T():
    """w ≥ T 时窗口 = 恒等（退化点，不得当真点）。"""
    T = 16
    keep = window_anchor_keep_mask(T, T, 0, "cpu")
    ident = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.equal(keep, ident)


def test_window_mask_causal_and_diagonal_open():
    """对角恒开、严格因果（未来 token 不可见）。"""
    T, w = 32, 16
    keep = window_anchor_keep_mask(T, w, 0, "cpu")
    for i in range(T):
        assert bool(keep[i, i]), "对角必须恒开"
        for j in range(T):
            if j > i:
                assert not bool(keep[i, j]), "未来位置必须不可见"
            if i - j >= w:
                assert not bool(keep[i, j]), "窗口外必须不可见"


# ---------------------------------------------------- 成本口径公式

def test_cost_theory_values():
    th = run.cost_theory((256, 512, 1024, 2048))
    for T, want in ((256, 0.1429), (512, 0.25), (1024, 0.4), (2048, 0.5714)):
        assert abs(th["attn_share_formula"][str(T)] - want) < 1e-4
    # 注意力 FLOPs 减半（窗口 = T/2）；all-in 降幅随 T 增大
    vals = []
    for T in (256, 512, 1024, 2048):
        wt = th["window_theory"][f"{T // 2}@{T}"]
        assert abs(wt["attn_flops_reduction_relative_to_full"] - 0.5) < 1e-6
        vals.append(wt["layer_macs_reduction_all_in"])
    assert vals == sorted(vals), "all-in 降幅应随 T 单调增"


# ---------------------------------------------------- 长度条件档 QK-Norm（复用 2625）

def _fwd_logits(model, x):
    with torch.no_grad():
        return model(x)[1]["depth_logits"][-1]


def test_qk_norm_inert_at_or_below_min_t_bitwise():
    x = torch.randint(0, 32, (2, 16))
    for T in (16, 256, 512):
        base = _tiny(T)
        b = _fwd_logits(base, x)
        m = _tiny(T)
        info = run._apply_qk_norm(m, T)
        assert info["n_qkv_hooks"] == 0
        assert torch.equal(_fwd_logits(m, x), b), "T≤512 路径必须逐位不变"


def test_qk_norm_active_above_min_t_and_zero_params():
    T = 1024
    x = torch.randint(0, 32, (2, 16))
    m = _tiny(T)
    n0 = sum(p.numel() for p in m.parameters())
    info = run._apply_qk_norm(m, T)
    n1 = sum(p.numel() for p in m.parameters())
    assert info["n_qkv_hooks"] == len(m.blocks)
    assert n1 - n0 == 0, "QK-Norm 必须零新增参数"
    base = _tiny(T)
    assert not torch.equal(_fwd_logits(m, x), _fwd_logits(base, x)), \
        "T>512 时 hook 必须真的改变前向"
