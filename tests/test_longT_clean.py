"""ticket 43-V1（长 T 干净三点）的可失败单测。

钉住的是**本实验自己负责的语义面**：
- 长度条件档 `qk-normfrom512` 的行为（T ≤ 512 零 hook 且前向逐位等于无 hook；
  T > 512 时逐头 L2 归一、零新增参数）——复用 2625 的实现，但**在本实验上下文里再钉一次**；
- 六臂表（3 点 × 2 臂，窗口 = T/2，同相对刻度、非退化）；
- `StepProbe` 的**训练门**计数（`window_mask_probe` 的 eval 前向不错位步号）。

全程 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_clean.py -q
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
                   "experiments", "2626-longT-clean")
DRIVER = os.path.join(EXP, "clean_driver.py")

D, L, H, MLP = 32, 2, 2, 4


def _load_driver():
    spec = importlib.util.spec_from_file_location("clean_driver_t", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = _load_driver()


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "run2626_t", os.path.join(EXP, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run2626 = _load_run()


def _tiny(seq_len: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                        mlp_mult=MLP, dense=True)
    return DepthARModel(cfg, 32)


# ---------------------------------------------------- 长度条件档（语义开关）

def test_min_t_constant_is_512():
    assert cd.QK_NORM_MIN_T == 512, "长度条件档阈值必须是 512（= 2625 推荐档）"


def test_qk_norm_inert_at_or_below_min_t_bitwise():
    """T ≤ 512：不挂任何 hook，且前向**逐位**等于无 hook 的模型。"""
    x = torch.randint(0, 32, (2, 16))
    for T in (16, 256, 512):
        base = _tiny(T)
        with torch.no_grad():
            b = base(x)[1]["depth_logits"][-1]
        m = _tiny(T)
        info = cd.apply_qk_norm(m)
        assert info["active"] is False and info["n_qkv_hooks"] == 0, \
            f"T={T} 不该挂 hook（min_t=512）"
        with torch.no_grad():
            a = m(x)[1]["depth_logits"][-1]
        assert torch.equal(b, a), f"T={T} 长度条件档不是逐位恒等"


def test_qk_norm_active_and_unit_scale_above_min_t():
    """T > 512：真的挂 hook；每头 q/k 的 L2 范数 == √d_h；**零新增参数**。"""
    dh = D // H
    T = 520                                     # > 512 ⇒ 触发长度条件档
    m = _tiny(T, seed=1)
    n0 = sum(p.numel() for p in m.parameters())
    info = cd.apply_qk_norm(m)
    n1 = sum(p.numel() for p in m.parameters())
    assert info["active"] is True and info["n_qkv_hooks"] == L
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


def test_qk_norm_changes_forward_above_min_t():
    """语义开关必须真的生效：T > 512 时同一输入的前向输出必须变。"""
    T = 520
    x = torch.randint(0, 32, (2, 16))
    b = _tiny(T)
    with torch.no_grad():
        before = b(x)[1]["depth_logits"][-1]
    m = _tiny(T)
    cd.apply_qk_norm(m)
    with torch.no_grad():
        after = m(x)[1]["depth_logits"][-1]
    assert not torch.equal(before, after), "qk-norm 是死的（前向输出未变）"


# ---------------------------------------------------- 六臂表 / 非退化

def test_arm_table_is_three_points_two_arms():
    arms = run2626.ARMS
    assert len(arms) == 6
    by_T = {}
    for a in arms:
        by_T.setdefault(a["seq_len"], []).append(a["window"])
    assert set(by_T) == {256, 512, 1024}
    for T, ws in by_T.items():
        assert sorted(ws) == [0, T // 2], f"T={T} 的臂必须是 {{full, T/2}}，得 {ws}"


def test_window_points_are_same_relative_scale_and_non_degenerate():
    """三点窗口占比同为 50%，且都不是恒等算子（SCHEMA §17.16）。"""
    for T in (256, 512, 1024):
        w = T // 2
        assert w / T == 0.5
        km = window_anchor_keep_mask(T, w, 0)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        assert not torch.equal(km, causal), f"window{w}@{T} 是退化点（恒等算子）"
        assert int(km.sum()) < int(causal.sum())


# ---------------------------------------------------- StepProbe 训练门

def test_step_probe_counts_training_forwards_only():
    """eval 模式的前向（如 `window_mask_probe`）不得让步号错位。"""
    m = _tiny(16)
    probe = cd.StepProbe(m, every=2, nl=4).attach()
    x = torch.randint(0, 32, (2, 16))
    # eval 前向：不计入
    m.eval()
    with torch.no_grad():
        m(x)
        m(x)
    assert probe.n == 0 and probe.snapshots == []
    # 训练前向：step 0 采一次，step 2 再采一次（every=2）
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


# ---------------------------------------------------- 驱动器保真度锚点

def test_driver_qknorm_step0_matches_2625_w1():
    """本实验 `full-1024-clean`（bs8/T=1024/qk-norm，seed0）的 step0 读数必须与
    2625-W1 的 `full-T1024-qknorm-s0` 逐位相同（同一 QK-Norm 实现 + 同一引擎）。

    证据：`runs/full-1024-clean/train_log.jsonl` 首行 vs
    `2625-longT-stability/runs/full-T1024-qknorm-s0/train_log.jsonl` 首行。

    F_KEY（复核 B12）：`runs/` 不入 git ⇒ 比对所需的最小证据已入 git
    （`tests/evidence/longT_step0_anchors.json`）；本地证据不在时 xfail，不再静默 skip。
    """
    anchors = evidence_anchor.load_anchors()
    ours_a = evidence_anchor.anchor(anchors, "clean_full1024_step0")
    theirs_a = evidence_anchor.anchor(anchors, "w1_qknorm_2625_step0")
    assert ours_a["step"] == theirs_a["step"] == 0
    assert ours_a["loss"] == theirs_a["loss"], "锚点记录的两侧 step0 loss 不一致"
    assert ours_a["gnorm"] == theirs_a["gnorm"], "锚点记录的两侧 step0 gnorm 不一致"
    ours = evidence_anchor.local_step0(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2626-longT-clean", "runs", "full-1024-clean",
                     "train_log.jsonl"), "clean_full1024_step0", anchors)
    theirs = evidence_anchor.local_step0(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2625-longT-stability", "runs", "full-T1024-qknorm-s0",
                     "train_log.jsonl"), "w1_qknorm_2625_step0", anchors)
    a, b = ours, theirs
    assert a["step"] == b["step"] == 0
    assert a["loss"] == b["loss"], "step0 loss 与 2625-W1 qk-norm 臂不一致"
    assert a["gnorm"] == b["gnorm"], "step0 gnorm 与 2625-W1 qk-norm 臂不一致"
