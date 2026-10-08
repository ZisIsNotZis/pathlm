"""ticket 45-L1（长 T 第四点 T=2048）的可失败单测。

钉住的是**本实验自己负责的语义面**：
- 长度条件档 `qk-normfrom512` 在 **T=2048（> min_t=512）** 的行为：两臂真挂 hook、
  逐头 L2 归一 q/k、**零新增参数**；
- 两臂表（1 点 × 2 臂，窗口 = T/2 = 1024，同相对刻度、非退化）；
- token 预算与 2626 干净三点**相同**（同配方 ⇒ 四点可并列）；
- `StepProbe` 的**训练门**计数（`window_mask_probe` 的 eval 前向不错位步号）。

全程 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_2048.py -q
"""

import importlib.util
import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evidence_anchor                                      # noqa: E402

from pathlm.depth_ar import (DepthARConfig, DepthARModel,     # noqa: E402
                             window_anchor_keep_mask)

EXP = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                   "experiments", "2628-longT-2048")
DRIVER = os.path.join(EXP, "clean_driver.py")

D, L, H, MLP = 32, 2, 2, 4


def _load_driver():
    spec = importlib.util.spec_from_file_location("clean_driver_2048_t", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = _load_driver()


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "run2628_t", os.path.join(EXP, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run2628 = _load_run()


def _tiny(seq_len: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                        mlp_mult=MLP, dense=True)
    return DepthARModel(cfg, 32)


# ---------------------------------------------------- 长度条件档（语义开关）

def test_min_t_constant_is_512():
    assert cd.QK_NORM_MIN_T == 512, "长度条件档阈值必须是 512（= 2625 推荐档）"


def test_qk_norm_active_at_2048():
    """T=2048 > min_t ⇒ 真的挂 hook（两臂各 8 个）。"""
    m = _tiny(2048, seed=1)
    info = cd.apply_qk_norm(m)
    assert info["active"] is True and info["n_qkv_hooks"] == L, \
        f"T=2048 该挂 {L} 个 hook，得到 {info}"


def test_qk_norm_unit_scale_and_zero_params_at_2048():
    """T=2048：每头 q/k 的 L2 范数 == √d_h；**零新增参数**。"""
    dh = D // H
    m = _tiny(2048, seed=1)
    n0 = sum(p.numel() for p in m.parameters())
    cd.apply_qk_norm(m)
    n1 = sum(p.numel() for p in m.parameters())
    assert n1 == n0, "QK-Norm 必须是零新增参数"

    seen = []
    for blk in m.blocks:                        # 记录归一后的 qkv 输出
        blk.qkv.register_forward_hook(lambda mod, i, o: seen.append(o))
    x = torch.randint(0, 32, (2, 16))
    with torch.no_grad():
        m(x)
    assert len(seen) == L
    for out in seen:
        d = out.shape[-1] // 3
        q = out[..., :d].reshape(-1, H, dh).float()
        k = out[..., d:2 * d].reshape(-1, H, dh).float()
        tgt = torch.full_like(q.norm(dim=-1), math.sqrt(dh))
        assert torch.allclose(q.norm(dim=-1), tgt, atol=1e-4)
        assert torch.allclose(k.norm(dim=-1), tgt, atol=1e-4)


def test_qk_norm_changes_forward_at_2048():
    """语义开关真的生效：T=2048 时同一输入的前向输出必须变。"""
    x = torch.randint(0, 32, (2, 16))
    b = _tiny(2048)
    with torch.no_grad():
        before = b(x)[1]["depth_logits"][-1]
    m = _tiny(2048)
    cd.apply_qk_norm(m)
    with torch.no_grad():
        after = m(x)[1]["depth_logits"][-1]
    assert not torch.equal(before, after), "qk-norm 是死的（前向输出未变）"


# ---------------------------------------------------- 两臂表 / 非退化

def test_arm_table_is_one_point_two_arms():
    arms = run2628.ARMS
    assert len(arms) == 2, f"本实验是 1 点 × 2 臂，得到 {len(arms)} 臂"
    by_T = {}
    for a in arms:
        by_T.setdefault(a["seq_len"], []).append(a["window"])
    assert set(by_T) == {2048}
    assert sorted(by_T[2048]) == [0, 1024], \
        f"T=2048 的臂必须是 {{full, T/2}}，得 {by_T[2048]}"


def test_window_point_is_same_relative_scale_and_non_degenerate():
    """窗口点占比 50%（与 2626 三点同一相对刻度），且不是恒等算子（SCHEMA §17.16）。"""
    T, w = 2048, 1024
    assert w / T == 0.5
    km = window_anchor_keep_mask(T, w, 0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert not torch.equal(km, causal), f"window{w}@{T} 是退化点（恒等算子）"
    assert int(km.sum()) < int(causal.sum())


def test_token_budget_matches_2626_clean_three_points():
    """同 token 预算 = 四点并列的前提：本实验与 2626 的 TOKEN_BUDGET 必须相等。"""
    assert run2628.TOKEN_BUDGET == 49_152_000
    p26 = evidence_anchor.require_tracked(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2626-longT-clean", "config.json"),
        "四点并列表的 token 预算基准（入 git）")
    cfg26 = json.load(open(p26))
    assert cfg26["token_budget"] == run2628.TOKEN_BUDGET, \
        "本实验与 2626 的 token 预算不同 ⇒ 四点不可并列"
    assert cfg26["batch_size"] == 8 and run2628.BS == 8, "bs 必须同为 8"


def test_steps_for_2048_is_3000():
    assert run2628.steps_for(2048) == 3000, \
        "T=2048 在 49.152M token / bs8 下应为 3000 步"


# ---------------------------------------------------- StepProbe 训练门

def test_step_probe_counts_training_forwards_only():
    """eval 模式的前向（如 `window_mask_probe`）不得让步号错位。"""
    m = _tiny(16)
    probe = cd.StepProbe(m, every=2, nl=4).attach()
    x = torch.randint(0, 32, (2, 16))
    m.eval()
    with torch.no_grad():
        m(x)
        m(x)
    assert probe.n == 0 and probe.snapshots == []
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(x)
    steps = [s["step"] for s in probe.snapshots]
    assert steps == [0, 2], f"训练步号错位：{steps}"


def test_step_probe_records_all_layer_stats():
    m = _tiny(16)
    probe = cd.StepProbe(m, every=1, nl=4).attach()
    m.train()
    with torch.no_grad():
        m(torch.randint(0, 32, (2, 16)))
    snap = probe.snapshots[0]
    assert snap["step"] == 0 and snap["resid_in"] is not None
    assert len(snap["layers"]) == L
    for row in snap["layers"]:
        for key in ("qkv_w_norm", "logit_rms", "logit_max", "top1", "entropy",
                    "resid_out"):
            assert key in row and row[key] == row[key]
        assert 0.0 < row["top1"] <= 1.0


# ---------------------------------------------------- 落盘结果的一致性

def test_results_four_points_include_2048_when_present():
    """落盘后：四点并列应含 T=2048，且 Δ 与 eval.json 重算一致。"""
    rp = evidence_anchor.require_tracked(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2628-longT-2048", "results.json"),
        "本实验的落盘结果（入 git）")
    ep = evidence_anchor.require_tracked(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2628-longT-2048", "eval.json"),
        "本实验的评测落盘（入 git）")
    R = json.load(open(rp))
    EV = json.load(open(ep))
    four = R["criteria"]["criterion_1_quality"]["four_points_native"]
    Ts = [p["T"] for p in four]
    assert Ts == [256, 512, 1024, 2048], f"四点并列的顺序/内容不对：{Ts}"
    b = EV["batteries"]
    dn = round(b["full-2048-clean"]["bpc_native"]
               - b["win1024-2048-clean"]["bpc_native"], 6)
    got = next(p["delta_native"] for p in four if p["T"] == 2048)
    assert got == dn, f"Δ2048 与服务端不一致：{got} vs {dn}"
