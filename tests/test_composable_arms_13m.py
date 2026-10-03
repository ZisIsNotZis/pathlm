"""2623-composable-arms-13m · 四臂阶梯的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义逻辑（`arms.py`）：四臂真的是四个不同配置、每臂落对块类
（ENG1 的组合选择规则）、窗口臂在 T=256 上**非退化**（§17.16）、native 可见性映射
正确、叠加增量是**相邻臂**之差（§17.17）。引擎三特性本身的活性/变异已由
`tests/test_eng1_composable.py` + 2622 的 `mutate.py` 钉住（本文件不重复）。

变异验证（每处抓红 + 逐字节还原）见同目录 `mutate.py`。
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2623-composable-arms-13m")
sys.path.insert(0, EXP)

from pathlm.depth_ar import DepthARConfig, DepthARModel, kv_head_liveness, \
    window_anchor_keep_mask, window_mask_liveness          # noqa: E402
from arms import (ARMS, ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                  arm_spec, expected_block_class, model_config_kwargs,
                  native_visibility, net_vs_baseline, staircase_increments)

#: 13M 规格（与训练同配方）：d256 × 12 层 × 8 头 × mlp6 × T256。
BIG = {"d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
       "mlp_mult": 6, "dense": True, "topology": "sequential"}
#: 训练/评测序列长（§17.16 非退化点在这里判）。
T = 256


def _build(name, mcfg=None, vocab=10):
    spec = arm_spec(name)
    torch.manual_seed(0)
    cfg = DepthARConfig(**{**(mcfg or BIG), **model_config_kwargs(spec)})
    return DepthARModel(cfg, vocab_size=vocab)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------- ① 四臂 = 四个不同配置，落对块类（能力面，§22.8） ----------------

def test_arm_names_match_contract():
    assert ARM_NAMES == ("stock", "gqa", "gqa_carry", "gqa_carry_window")
    assert BASELINE_ARM == "stock" and TARGET_ARM == "gqa_carry_window"


def test_each_arm_builds_and_lands_expected_block_class():
    """四臂在引擎里都能建起来，且块类 = ENG1 的组合选择规则（不是标签）。"""
    assert expected_block_class(arm_spec("stock")) == "Block"
    assert expected_block_class(arm_spec("gqa")) == "GQABlock"
    assert expected_block_class(arm_spec("gqa_carry")) == "ComposableBlock"
    assert expected_block_class(arm_spec("gqa_carry_window")) == "ComposableBlock"
    for name in ARM_NAMES:
        m = _build(name, mcfg={"d_model": 32, "n_layers": 2, "n_heads": 8,
                               "seq_len": 16, "dense": True,
                               "topology": "sequential"})
        assert type(m.blocks[0]).__name__ == expected_block_class(arm_spec(name)), name


def test_stacked_features_grow_monotonically():
    st = [set(a["stacked"]) for a in ARMS]
    assert st[0] == set()
    assert st[1] == {"gqa"}
    assert st[2] == {"gqa", "carry"}
    assert st[3] == {"gqa", "carry", "window"}
    # 每步只**加**一个特性（阶梯口径）
    for prev, cur in zip(ARMS, ARMS[1:]):
        assert len(cur["stacked"]) == len(prev["stacked"]) + 1


# ---------------- ② 参数量差（判据 ④ 的机器可核部分） ----------------

def test_param_counts_and_gqa_reduction():
    ns = {name: _n_params(_build(name, vocab=206)) for name in ARM_NAMES}
    assert ns["gqa"] < ns["stock"]                      # GQA 减参数（KV 投影半边变窄）
    assert ns["gqa_carry"] > ns["gqa"]                  # carry 增参数
    assert ns["gqa_carry_window"] == ns["gqa_carry"]    # window 是掩码，0 参数
    red = 1 - ns["gqa"] / ns["stock"]
    assert 0.085 < red < 0.090, f"GQA 参数下降 {red:.4f} 应落在 ~8.75%"
    assert ns["stock"] == 13536768 and ns["gqa"] == 12352512
    assert ns["gqa_carry"] == 12616808


# ---------------- ③ 窗口臂非退化（§17.16） ----------------

def test_window_arm_is_nondegenerate_and_not_identity():
    w, a = native_visibility(arm_spec(TARGET_ARM))
    assert (w, a) == (16, 0)
    liv = window_mask_liveness(T, w, a)
    assert liv["alive"] and liv["visible_frac"] < 1.0   # 真的挡掉了 key
    keep = window_anchor_keep_mask(T, w, a)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert not torch.equal(keep, causal)                # 不是恒等因果（非退化点）
    assert int(keep.sum()) < int(causal.sum())


def test_non_window_arms_use_full_causal():
    for name in ("stock", "gqa", "gqa_carry"):
        assert native_visibility(arm_spec(name)) == (0, 0)


# ---------------- ④ 叠加增量 = 相邻臂之差（§17.17） ----------------

def test_staircase_increments_are_adjacent_diffs():
    acc = {"stock": 0.10, "gqa": 0.20, "gqa_carry": 0.25, "gqa_carry_window": 0.15}
    inc = staircase_increments(acc)
    assert [i["added"] for i in inc] == ["gqa", "carry", "window"]
    assert [i["delta_pp"] for i in inc] == [10.0, 5.0, -10.0]
    assert net_vs_baseline(acc) == 5.0                 # 净和 != 任何单步（避免只看净和）


# ---------------- ⑤ GQA 头形状是真的变窄（活性，不是标签） ----------------

def test_gqa_actually_narrows_kv_heads_in_forward():
    m = _build("gqa", mcfg={"d_model": 32, "n_layers": 2, "n_heads": 8,
                            "seq_len": 16, "dense": True, "topology": "sequential"})
    x = torch.randint(0, 10, (2, 16))
    out = kv_head_liveness(m, x)
    assert out["k_batch_shape"][1] == 2                # 真进 SDPA 的 K 头数
    assert out["qkv_out_width"] == 32 + 2 * 2 * 4       # d + 2·nkv·dh
    assert out["alive"] is True


def test_carry_switch_is_not_dead():
    """carry 臂的读出增益单边零起步（恒等），激活后 read_share > 0（开关不是死的）。"""
    m = _build("gqa_carry", mcfg={"d_model": 32, "n_layers": 2, "n_heads": 8,
                                  "seq_len": 16, "dense": True,
                                  "topology": "sequential"})
    assert type(m.blocks[0]).__name__ == "ComposableBlock"
    for blk in m.blocks:
        assert blk.carry_scale.abs().sum().item() == 0.0   # 单边零起步
    x = torch.randint(0, 10, (2, 16))
    with torch.no_grad():
        _, aux0 = m(x)
    assert float(aux0["attn_carry"][0]["read_share"]) == 0.0
    for blk in m.blocks:
        blk.carry_scale.data.fill_(0.5)
    with torch.no_grad():
        _, aux1 = m(x)
    assert float(aux1["attn_carry"][0]["read_share"]) > 0.0
