"""2627-carry-x-depth · 2×2 因子设计的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义逻辑（`carry_x_depth_arms.py`）：四臂真的是 (carry × depth) 的全因子格子、
深度因子只改训练信号（`--dense-aux`）不改架构、块类映射正确、可见性四臂一致（window16）、
因子分解是 2×2 单向增量 + 交互项（§17.17 禁止只看净和）；外加两条**能力面**回归钉：
① 引擎的 `dense_aux` 确实是「全深度监督」的活开关（不是标签）；② dense 主干**忽略**
`exit_depths`（这是本实验把深度因子落到 `dense_aux` 的引擎依据，§22.8/§4b）。

引擎自身的变异覆盖另有其人（`tests/test_depth_ar.py::test_dense_aux_full_depth_supervision`）；
本文件的变异验证见同目录 `mutate.py`。
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2627-carry-x-depth")
sys.path.insert(0, EXP)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             sample_exit_depths, window_anchor_keep_mask,
                             window_mask_liveness)
from carry_x_depth_arms import (ARMS, ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                  DEPTH_TRAIN_FLAG, arm_spec, expected_block_class,
                  factorial_increments, factorial_increments_pp,
                  model_config_kwargs, native_visibility, train_flags)

#: 13M 规格（与训练同配方）：d256 × 12 层 × 8 头 × mlp6 × T256。
BIG = {"d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
       "mlp_mult": 6, "dense": True, "topology": "sequential"}
T = 256


def _small(name, vocab=10):
    spec = arm_spec(name)
    torch.manual_seed(0)
    cfg = DepthARConfig(**{"d_model": 32, "n_layers": 2, "n_heads": 4,
                           "seq_len": 16, "dense": True, "topology": "sequential",
                           **model_config_kwargs(spec)})
    return DepthARModel(cfg, vocab_size=vocab)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------- ① 2×2 因子格子（不是阶梯） ----------------

def test_arm_names_and_factorial_grid():
    assert ARM_NAMES == ("base", "carry", "depth", "both")
    assert BASELINE_ARM == "base" and TARGET_ARM == "both"
    grid = [(a["carry"], a["depth"]) for a in ARMS]
    assert grid == [(False, False), (True, False), (False, True), (True, True)]
    # 每臂只由 (carry, depth) 两因子描述
    for a in ARMS:
        assert set(a) == {"name", "carry", "depth"}


def test_carry_factor_is_the_only_config_variable():
    """单因素：只有 carry 因子碰 `DepthARConfig`（depth 因子是训练信号，不是配置）。"""
    assert model_config_kwargs(arm_spec("base")) == {"carry_rule": None}
    assert model_config_kwargs(arm_spec("carry")) == {"carry_rule": "delta"}
    assert model_config_kwargs(arm_spec("depth")) == {"carry_rule": None}
    assert model_config_kwargs(arm_spec("both")) == {"carry_rule": "delta"}
    # depth 臂的模型配置 == base 臂（架构逐字段相同）
    assert model_config_kwargs(arm_spec("depth")) == model_config_kwargs(arm_spec("base"))
    assert model_config_kwargs(arm_spec("both")) == model_config_kwargs(arm_spec("carry"))


def test_depth_factor_maps_to_dense_aux_train_flag():
    """深度因子 = 训练侧 `--dense-aux`；carry 臂不带它、depth/both 带它。"""
    assert train_flags(arm_spec("base")) == []
    assert train_flags(arm_spec("carry")) == []
    assert train_flags(arm_spec("depth")) == [DEPTH_TRAIN_FLAG]
    assert train_flags(arm_spec("both")) == [DEPTH_TRAIN_FLAG]
    assert DEPTH_TRAIN_FLAG == "--dense-aux"


def test_expected_block_class_per_arm():
    for name, cls in (("base", "Block"), ("carry", "CarryChannelBlock"),
                      ("depth", "Block"), ("both", "CarryChannelBlock")):
        assert expected_block_class(arm_spec(name)) == cls
    # 真的建起来 + 落对块类（不是标签）
    for name in ARM_NAMES:
        m = _small(name)
        assert type(m.blocks[0]).__name__ == expected_block_class(arm_spec(name)), name


def test_native_visibility_is_window16_for_all_arms():
    """四臂训练/评测同一可见性（window16）——窗口是制造需求的一步，不是变量。"""
    for name in ARM_NAMES:
        assert native_visibility(arm_spec(name)) == (16, 0)


def test_window_is_nondegenerate_and_not_identity():
    liv = window_mask_liveness(T, 16, 0)
    assert liv["alive"] and liv["visible_frac"] < 1.0
    keep = window_anchor_keep_mask(T, 16, 0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert not torch.equal(keep, causal) and int(keep.sum()) < int(causal.sum())


# ---------------- ② 参数量（carry 因子加减参数；depth 因子不加） ----------------

def test_param_counts_depth_adds_none_carry_adds_264296():
    torch.manual_seed(0)
    m = {name: DepthARModel(DepthARConfig(**{**BIG, **model_config_kwargs(arm_spec(name))}),
                            vocab_size=206)
         for name in ARM_NAMES}
    n = {k: _n_params(v) for k, v in m.items()}
    assert n["base"] == 13536768 and n["carry"] == 13801064
    assert n["depth"] == n["base"]            # 深度因子是训练信号，0 参数
    assert n["both"] == n["carry"]
    assert n["carry"] - n["base"] == 264296


# ---------------- ③ 引擎侧：dense_aux 是活开关；dense 主干忽略 exit_depths ----------------

def test_dense_aux_full_depth_supervision_is_live():
    """`dense_aux=True` 真的把监督从 1 个出口扩到 L+1 个（depth 因子的引擎依据）。"""
    m = _small("base", vocab=10)
    x = torch.randint(0, 10, (2, 16))
    with torch.no_grad():
        _, aux_off = m(x, dense_aux=False)
        _, aux_on = m(x, dense_aux=True)
    assert len(aux_off["depth_ce"]) == 1          # 只监督最终深度
    assert len(aux_on["depth_ce"]) == m.cfg.n_layers + 1   # 全深度并行监督


def test_dense_trunk_ignores_exit_depths():
    """能力面钉（§4b）：dense 主干**完全忽略** exit_depths——`--exit-mode uniform`
    在 dense 上是死开关；这正是深度因子必须落到 `dense_aux` 的引擎理由。"""
    m = _small("base", vocab=10)
    x = torch.randint(0, 10, (2, 16))
    full = torch.full((2, 16), m.cfg.n_layers, dtype=torch.long)
    rand = sample_exit_depths(2, 16, m.cfg.n_layers,
                              torch.Generator().manual_seed(1))
    assert not torch.equal(full, rand)            # 两个深度张量确实不同
    with torch.no_grad():
        l_full, _ = m(x, exit_depths=full)
        l_rand, _ = m(x, exit_depths=rand)
    assert torch.equal(l_full, l_rand)            # 但 dense 前向的 loss 逐位相同


# ---------------- ④ 因子分解 = 2×2 单向增量 + 交互项（§17.17） ----------------

def test_factorial_increments_are_factorial_not_staircase():
    metric = {"base": 0.10, "carry": 0.20, "depth": 0.15, "both": 0.22}
    inc = factorial_increments(metric)
    assert inc["carry_at_depth_off"] == 0.10
    assert inc["depth_at_carry_off"] == 0.05
    assert inc["carry_at_depth_on"] == 0.07
    assert inc["depth_at_carry_on"] == 0.02
    assert inc["interaction"] == -0.03            # (both−carry) − (depth−base)
    assert inc["net_vs_base"] == 0.12
    # 净和 != 任何单步（§17.17：不得只看净和）
    assert inc["net_vs_base"] not in (inc["carry_at_depth_off"],
                                      inc["depth_at_carry_off"],
                                      inc["carry_at_depth_on"],
                                      inc["depth_at_carry_on"])
    assert factorial_increments_pp(metric)["interaction"] == -3.0


def test_interaction_is_zero_when_factors_additive():
    """可加情形：交互项必须为 0（否则「不可叠加」的判定标准是错的）。"""
    metric = {"base": 0.1, "carry": 0.2, "depth": 0.3, "both": 0.4}
    assert factorial_increments(metric)["interaction"] == 0.0
