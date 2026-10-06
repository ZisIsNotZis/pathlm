"""2630-tf4-samecaliber · 两臂单因素设计 + 同口径判决的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义层（`samecaliber_arms.py`）：两臂真的只差 `carry` 一个因子、
两臂**都训在 enwik8 上**（原生预训练口径、非合成）、随机退出 + proj-fill 是两臂同款
形态件、块类映射正确、单因素增量是单因素对照、`tf4_holds` 规则唯一；外加三条**能力面**
回归钉（SCHEMA §22.8）：
① `carry_rule` 与非 dense（随机退出）+ proj-fill 同开可建可前向（ENG2 解锁的组合）；
② `train_window > 0` 与 `dense=False` **仍硬互斥**（本线没有训练期窗口的引擎依据）；
③ `proj_fill` 要求 `fill_kv=True`（形态件的前提）。

引擎自身的变异覆盖另有其人（`tests/test_depth_ar.py` §ENG2）；本文件的变异验证见同目录
`mutate.py`。
"""

import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2630-tf4-samecaliber")
sys.path.insert(0, EXP)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             sample_exit_depths)
from samecaliber_arms import (ARMS, ARM_NAMES, BASELINE_ARM, CARRY_RULE,  # noqa: E402
                              DATA_SOURCE, EXIT_MODE, TARGET_ARM,
                              VARIANCE_BAND_PP, arm_spec, capability_probe,
                              data_source, expected_block_class,
                              model_config_kwargs, native_visibility,
                              single_factor_increment, single_factor_increment_pp,
                              tf4_holds, train_flags)


def _small(name, vocab=10):
    spec = arm_spec(name)
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=16,
                        **model_config_kwargs(spec))
    return DepthARModel(cfg, vocab_size=vocab)


# ---------------- ① 两臂 = 单因素设计（不是阶梯、不是与裸基线比） ----------------

def test_arm_names_and_single_factor():
    assert ARM_NAMES == ("base", "carry")
    assert BASELINE_ARM == "base" and TARGET_ARM == "carry"
    grid = [(a["carry"],) for a in ARMS]
    assert grid == [(False,), (True,)]
    for a in ARMS:
        assert set(a) == {"name", "carry"}


def test_only_carry_varies_between_arms():
    b = model_config_kwargs(arm_spec("base"))
    c = model_config_kwargs(arm_spec("carry"))
    # 唯一差异 = carry_rule；其余形态件两臂逐字段相同（非 dense / proj-fill / fill）。
    assert b["carry_rule"] is None and c["carry_rule"] == CARRY_RULE
    for k in ("dense", "topology", "fill_kv", "proj_fill"):
        assert b[k] == c[k], k
    assert b["dense"] is False and b["proj_fill"] is True and b["fill_kv"] is True


def test_train_flags_share_recipe_and_add_carry():
    b = train_flags(arm_spec("base"))
    c = train_flags(arm_spec("carry"))
    # 两臂都带形态件（随机退出 + proj-fill），且都不带 --dense（非 dense RAP 形态）。
    assert "--proj-fill" in b and "--proj-fill" in c
    assert b[1:3] == c[1:3] == ["--exit-mode", EXIT_MODE]
    assert "--dense" not in b and "--dense" not in c
    # 目标臂多且只多 --carry-write delta
    assert c[len(b):] == ["--carry-write", CARRY_RULE]
    assert b == ["--proj-fill", "--exit-mode", EXIT_MODE]


def test_both_arms_train_on_enwik8_not_synth():
    """同口径的前提：两臂的训练数据 = enwik8，且训练开关里**不带** `--synth-task`。"""
    for name in ARM_NAMES:
        assert data_source(arm_spec(name)) == DATA_SOURCE == "enwik8"
        flags = train_flags(arm_spec(name))
        assert "--synth-task" not in flags and "assoc-mqar" not in flags


def test_expected_block_class_per_arm():
    assert expected_block_class(arm_spec("base")) == "Block"
    assert expected_block_class(arm_spec("carry")) == "CarryChannelBlock"


def test_native_visibility_zero_for_both_arms():
    # 两臂都没有训练期可见窗口（train_window>0 与 dense=False 互斥）⇒ (0,0) 全因果。
    for name in ARM_NAMES:
        assert native_visibility(arm_spec(name)) == (0, 0)


def test_param_counts_and_block_classes_at_build():
    m0, m1 = _small("base"), _small("carry")
    assert type(m0.blocks[0]).__name__ == "Block"
    assert type(m1.blocks[0]).__name__ == "CarryChannelBlock"
    n0 = sum(p.numel() for p in m0.parameters())
    n1 = sum(p.numel() for p in m1.parameters())
    assert n1 > n0                                    # carry 加参数（记账，非 0）
    # 主干参数逐位相同（同 seed；carry 参数只在目标臂）⇒ 单因素对照成立（§9/§15）。
    t0 = {k: v for k, v in m0.state_dict().items() if "carry" not in k}
    t1 = {k: v for k, v in m1.state_dict().items() if "carry" not in k}
    assert set(t0) == set(t1)
    for k in t0:
        assert torch.equal(t0[k], t1[k]), k


def test_capability_probe_reports_combo_buildable_and_clean():
    cap = capability_probe()
    assert cap["combo_buildable"] is True
    assert cap["trunk_bit_identical"] is True and cap["trunk_diff_keys"] == []
    assert cap["all_extra_params_are_carry"] is True
    assert cap["param_delta"] > 0
    assert cap["arms"]["carry"]["block_class"] == "CarryChannelBlock"
    assert cap["arms"]["base"]["dense"] is False
    assert cap["arms"]["carry"]["proj_fill"] is True
    assert cap["data_source"] == "enwik8"


# ---------------- ② 单因素增量（§17.17）与 tf@4 判定 ----------------

def test_single_factor_increment_arithmetic():
    d = single_factor_increment({"base": 0.90, "carry": 0.87})
    assert d["carry_effect"] == pytest.approx(-0.03, abs=1e-9)
    assert d["net_vs_base"] == d["carry_effect"]
    p = single_factor_increment_pp({"base": 0.90, "carry": 0.87})
    assert p["carry_effect"] == pytest.approx(-3.0, abs=1e-6)


def test_tf4_holds_rule():
    # |Δ| < 1.4pp ⇒ 保持（判据①预登记）；|Δ| ≥ 1.4pp ⇒ 不保持。
    assert tf4_holds(0.0) is True
    assert tf4_holds(1.39) is True
    assert tf4_holds(-1.39) is True
    assert tf4_holds(1.4) is False
    assert tf4_holds(-1.4) is False
    assert tf4_holds(4.71) is False
    assert VARIANCE_BAND_PP == 1.4


# ---------------- ③ 能力面回归钉（引擎依据） ----------------

def test_capability_engine_carry_non_dense_proj_fill_forward():
    """① `carry_rule` × 非 dense × 随机退出 × proj-fill 可建可前向（ENG2）；通道活着。"""
    spec = arm_spec(TARGET_ARM)
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=3, n_heads=4, seq_len=12,
                        **model_config_kwargs(spec))
    m = DepthARModel(cfg, vocab_size=10)
    assert type(m.blocks[0]).__name__ == "CarryChannelBlock"
    for blk in m.blocks:
        blk.carry_scale.data.fill_(0.5)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, 3, torch.Generator().manual_seed(7))
    _, aux = m(x, exit_depths=depths)
    assert len(aux["attn_carry"]) == 3          # 非 dense 路径也收进了遥测
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0


def test_capability_train_window_still_requires_dense():
    """② 本线没有训练期窗口：`train_window > 0` 与 `dense=False` 仍硬互斥（未测因子）。

    校验在 `DepthARModel.__init__` 里（不是 `DepthARConfig`）⇒ 必须**建模型**才算数。
    """
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=16,
                        dense=False, train_window=16)
    with pytest.raises(ValueError):
        DepthARModel(cfg, vocab_size=10)


def test_capability_proj_fill_requires_fill_kv():
    """③ proj-fill 要求 fill_kv=True（形态件的前提；非 dense 前向里显式校验）。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=16,
                        dense=False, proj_fill=True, fill_kv=False)
    m = DepthARModel(cfg, vocab_size=10)
    with pytest.raises(ValueError):
        m(torch.randint(0, 10, (2, 16)))
