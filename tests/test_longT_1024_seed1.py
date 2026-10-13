"""ticket 41-Z1（长 T 第三点 T=1024 的 **seed=1 双臂重跑**）的可失败单测。

钉住本实验相对 2623（seed0）**唯一变化 = seed** 这一事实，以及预登记的
「gnorm 爆炸复现判据」与「带宽分类」两个纯函数。全部 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_1024_seed1.py -q
"""

import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathlm.depth_ar import (DepthARConfig, DepthARModel,      # noqa: E402
                             window_anchor_keep_mask, window_mask_liveness)

EXP = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window",
                   "experiments", "2624-longT-1024-seed1")


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "run_2624", os.path.join(EXP, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_run()

#: 三个同族点：窗口占 T 的同一相对刻度（50%，SCHEMA §17.16）
POINTS = ((128, 256), (256, 512), (512, 1024))


def _causal(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


# ------------------------------------------------------------------ 单因素 / seed

def test_seed1_arms_only_differ_in_train_window():
    """两臂逐字段一致，**只有** `train_window` 不同；seed 两臂同为 1（Z1 ②）。"""
    fields = [{"d_model": R.D, "n_layers": R.L, "n_heads": R.H, "mlp_mult": R.MLP,
               "dense": True, "topology": "sequential", "batch_size": R.BS,
               "lr": R.LR, "seed": R.SEED, "steps": R.steps_for(a["seq_len"]),
               "exit_mode": "uniform", "seq_len": a["seq_len"],
               "train_window": a["window"]} for a in R.ARMS]
    a, b = fields
    assert a["train_window"] == 0 and b["train_window"] == 512
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"train_window"}, f"两臂只允许差 train_window，实差 {diff}"
    assert a["seed"] == b["seed"] == 1, "两臂 seed 必须相同且为 1"


def test_recipe_matches_2623_except_seed():
    """相对 2623（seed0）的唯一变化 = seed；bs/token 预算/步数逐字段相同。"""
    assert R.BS == 8 and R.SEED == 1
    assert R.steps_for(1024) == 12000
    assert R.steps_for(1024) * R.BS * 1024 == 98_304_000
    assert R.TOKEN_BUDGET == 98_304_000


def test_seed_is_a_live_factor():
    """seed 是**活因子**（§10）：同配置、不同 seed ⇒ 初始参数必须不同。"""
    def init(seed):
        torch.manual_seed(seed)
        m = DepthARModel(R.cfg_for(1024, 512), 206)
        return torch.cat([p.detach().flatten() for p in m.parameters()])
    assert not torch.equal(init(0), init(1)), "seed 不改变初始化 ⇒ seed 不是活因子"


def test_steps_override_keeps_both_arms_at_same_scale():
    """§20.3：降速时只缩步数，且两臂同刻度（override 对两臂同时生效）。"""
    assert R.steps_for(1024) == 12000
    orig = R.STEPS_OVERRIDE
    try:
        R.STEPS_OVERRIDE = 6000
        assert R.steps_for(1024) == 6000
        R.STEPS_OVERRIDE = 99000          # 不得被抬高（只允许缩）
        assert R.steps_for(1024) == 12000
    finally:
        R.STEPS_OVERRIDE = orig


# ------------------------------------------------------------- 复现判据（预登记）

def test_explosion_reproduction_reproduced():
    """seed0 的签名（1740.92 vs 7.35）⇒ R1、R2 皆过 ⇒ reproduced。"""
    r = R.explosion_reproduction({"gnorm_max": 1740.92}, {"gnorm_max": 7.35})
    assert (r["R1_passed"], r["R2_passed"], r["verdict"]) == (True, True, "reproduced")


def test_explosion_reproduction_not_reproduced():
    """两臂都平稳（1.2 vs 0.5）⇒ R1、R2 皆不过 ⇒ not_reproduced（可给干净 Δ）。"""
    r = R.explosion_reproduction({"gnorm_max": 1.2}, {"gnorm_max": 0.5})
    assert (r["R1_passed"], r["R2_passed"], r["verdict"]) == (False, False,
                                                              "not_reproduced")


@pytest.mark.parametrize("full,win", [(200.0, 30.0), (50.0, 0.5)])
def test_explosion_reproduction_partial_is_not_clean(full, win):
    """只过一个判据 ⇒ partial ⇒ 保守按「不可归因」（不给干净 Δ）。"""
    r = R.explosion_reproduction({"gnorm_max": full}, {"gnorm_max": win})
    assert r["verdict"] == "partial"
    assert r["R1_passed"] != r["R2_passed"], "partial 的定义 = 判据恰过一个"


# ------------------------------------------------------------- 带宽分类（有向）

def test_band_class_boundary_and_direction():
    assert R.band_class(0.0139) == "不亏（噪声带内）"
    assert R.band_class(0.014) == "新算子更好（越界）"
    assert R.band_class(-0.014) == "新算子更差（越界）"
    assert R.band_class(-0.0139) == "不亏（噪声带内）"


def test_band_threshold_is_014():
    """方差界是 methodology §2 的 0.014，不得被改成别的值。"""
    assert R.band_class(0.014) != R.band_class(0.013)


# ------------------------------------------------------------------ 非退化/同刻度

def test_three_points_have_same_relative_scale():
    assert all(abs(w / T - 0.5) < 1e-12 for w, T in POINTS), POINTS


def test_window512_at_T1024_is_nondegenerate():
    """主臂 window512@T=1024 非退化（否则第三点是假点，§17.16）。"""
    keep = window_anchor_keep_mask(1024, 512, 0)
    assert not torch.equal(keep, _causal(1024)), "window512@T=1024 不得退化为恒等"
    live = window_mask_liveness(1024, 512, 0)
    assert live["alive"] is True and 0.0 < live["visible_frac"] < 1.0


def test_window1024_at_T1024_is_identity_and_thus_degenerate():
    """禁止凑数的点：window1024@T=1024 是恒等算子（§17.16 的退化点样例）。"""
    assert torch.equal(window_anchor_keep_mask(1024, 1024, 0), _causal(1024))
    assert window_mask_liveness(1024, 1024, 0)["alive"] is False
