"""2625-ood-leg-variance · 臂定义 / 同批位置 / 方差统计 / 判据阈值 —— 可失败单测。

Run: `python3 -m pytest tests/test_ood_leg_variance.py -q`

靶子 = 本实验自己的纯逻辑（`ood_variance_arms.py`）+ 复用的评测仪器的实现锚
（`quant_carry.py` 的 `fp32` 档与释放版前向逐位相等，GQA×carry 块）。

引擎三特性（GQA × carry × window）**本线不改**，其活性/变异由 `test_eng1_composable.py`
与 2624 的测试钉住——本文件不重复。变异验证（每处抓红 + 逐字节还原）见同目录 `mutate.py`。
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             window_anchor_keep_mask, window_mask_liveness)
from tools.evalkit.retention import masked_logits  # noqa: E402

from importlib import import_module  # noqa: E402

HERE = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory",
                    "experiments", "2625-ood-leg-variance")
Z3_DIR = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory",
                      "experiments", "2624-carry-under-compression-quality")
for p in (HERE, Z3_DIR):
    sys.path.insert(0, p)
oa = import_module("ood_variance_arms")


# ① 主集 / 唯一变量 = seed / 目标配置 ---------------------------------------

def test_primary_set_is_three_seeds_of_one_config():
    """n=3 主集 = seed{0(reuse),1,2}；两条新臂 seed=1/2；复用的 s0 seed=0。"""
    assert oa.PRIMARY_ARMS == ("gqa_carry_win_s0", "gqa_carry_win_s1", "gqa_carry_win_s2")
    assert [a["name"] for a in oa.TRAINED_ARMS] == ["gqa_carry_win_s1",
                                                    "gqa_carry_win_s2"]
    assert sorted(oa.train_seed(oa.arm(n)) for n in oa.PRIMARY_ARMS) == [0, 1, 2]
    assert oa.train_seed(oa.arm("gqa_carry_win_s0")) == 0
    assert oa.train_seed(oa.arm("y1_gqa_carry_win_s0")) == 0


def test_arms_differ_only_in_seed():
    """所有臂的 `config_flags` 逐字段相同（唯一变量 = seed）。"""
    flags = [oa.model_config_kwargs(oa.arm(n)) for n in oa.ALL_ARMS]
    assert all(f == flags[0] for f in flags)
    seeds = {n: oa.train_seed(oa.arm(n)) for n in oa.ALL_ARMS}
    assert seeds == {"gqa_carry_win_s0": 0, "gqa_carry_win_s1": 1,
                     "gqa_carry_win_s2": 2, "y1_gqa_carry_win_s0": 0}


def test_target_config_is_gqa_carry_window16():
    """配置 = GQA(2 KV 头) + carry(delta) + window(16, anchors 0)（W2 明文）。"""
    assert oa.CONFIG_FLAGS == {"n_kv_heads": 2, "carry_rule": "delta",
                               "train_window": 16, "train_anchors": 0}
    for n in oa.ALL_ARMS:
        f = oa.model_config_kwargs(oa.arm(n))
        assert f["n_kv_heads"] == 2
        assert f["carry_rule"] == "delta"
        assert f["train_window"] == 16 and f["train_anchors"] == 0


def test_expected_block_class_is_composable_for_all_arms():
    for n in oa.ALL_ARMS:
        assert oa.expected_block_class(oa.arm(n)) == "ComposableBlock"
    # 反例：只有 GQA（无 carry）→ GQABlock；只有 carry → CarryChannelBlock
    assert oa.expected_block_class({"flags": {"n_kv_heads": 2}}) == "GQABlock"
    assert oa.expected_block_class({"flags": {"carry_rule": "delta"}}) == \
        "CarryChannelBlock"


# ② 评测「同一批位置」（与臂无关） -------------------------------------------

def test_eval_batch_seeds_identical_regardless_of_arm():
    """同一条腿的批生成器 seed 与臂无关（三臂评同一批位置）。"""
    assert oa.single_batch_seeds(3) == [1056, 1057, 1058]
    assert oa.multi_batch_seeds(3) == [2112, 2113, 2114]
    assert oa.var_batch_seeds(3) == [5000, 5001, 5002]
    # 函数不吃臂参数 ⇒ 同腿同 seed 列表
    assert oa.eval_batch_seeds("single", 4) == oa.single_batch_seeds(4)
    assert oa.eval_batch_seeds("multi", 4) == oa.multi_batch_seeds(4)
    assert oa.eval_batch_seeds("var", 4) == oa.var_batch_seeds(4)


def test_leg_seed_bases_match_2624_instrument():
    """与 2624 的 `quant_carry` 默认 seed 基准一致（同件仪器 ⇒ 同批位置可比）。"""
    assert (oa.SINGLE_SEED_BASE, oa.MULTI_SEED_BASE, oa.VAR_SEED_BASE) == (1000, 2000, 5000)


# ③ 方差统计与预登记判据 -----------------------------------------------------

def test_range_mean_sd_pp():
    accs = [0.009277, 0.069336, 0.117676]
    assert oa.range_pp(accs) == 10.84
    assert oa.mean_pp(accs) == 6.543
    assert oa.sd_pp(accs) == 5.43
    assert oa.range_pp([0.5]) == 0.0            # 单点 → 0
    assert oa.sd_pp([0.5]) == 0.0


def test_downgrade_threshold_is_inclusive_at_3pp():
    """判据 ②「极差 ≥ 结论效应（≥3pp）」——含等号，且只在极差上判。"""
    assert oa.downgrade_to_auxiliary(3.0) is True
    assert oa.downgrade_to_auxiliary(2.999) is False
    assert oa.downgrade_to_auxiliary(10.84) is True
    assert oa.EFFECT_THRESHOLD_PP == 3.0


def test_in_dist_control_uses_same_threshold():
    """分布内腿对照用同一阈值（≥3pp 才算「整体不可用」）。"""
    assert oa.downgrade_to_auxiliary(1.221) is False     # 实测分布内极差
    assert oa.downgrade_to_auxiliary(36.0) is True


# ④ 窗口非退化（§17.16） -----------------------------------------------------

def test_window_mask_nondegenerate_at_T256():
    live = window_mask_liveness(256, 16, 0)
    assert live["alive"] is True
    assert 0.0 < live["visible_frac"] < 1.0


# ⑤ 复用仪器的实现锚：fp32 档 == 释放版前向（GQA×carry 块） -----------------

def _gqa_carry_model(seed=0, T=12, d=16, layers=3, heads=2, window=4):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=d, n_layers=layers, n_heads=heads, seq_len=T,
                        dense=True, train_window=window, train_anchors=0,
                        carry_rule="delta", carry_chunk=64, n_kv_heads=heads // 2)
    m = DepthARModel(cfg, vocab_size=10)
    with torch.no_grad():
        for blk in m.blocks:
            blk.carry_scale.fill_(0.5)
        m.carry.w_k.normal_(std=0.3)
        m.carry.w_v.normal_(std=0.3)
        m.carry.w_q.normal_(std=0.3)
    m.eval()
    return m


def test_reused_instrument_fp32_bitwise_matches_released():
    """复用 2624 的 `quant_carry`：GQA×carry 块 fp32 档 == 释放版前向（逐位）。"""
    qc = import_module("quant_carry")
    m = _gqa_carry_model()
    assert type(m.blocks[0]).__name__ == "ComposableBlock"
    x = torch.randint(0, 10, (4, 12), generator=torch.Generator().manual_seed(3))
    keep = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        rel = masked_logits(m, x, keep)
        q, _ = qc.quant_carry_logits(m, x, "fp32", keep)
    assert torch.equal(rel, q)


def test_reused_instrument_fp16_changes_logits():
    """fp16 档真的改了数值（不是空操作）——活性量非零。"""
    qc = import_module("quant_carry")
    m = _gqa_carry_model()
    x = torch.randint(0, 10, (4, 12), generator=torch.Generator().manual_seed(3))
    keep = window_anchor_keep_mask(12, 4, 0)
    acc = qc._new_stats()
    with torch.no_grad():
        f32, _ = qc.quant_carry_logits(m, x, "fp32", keep)
        f16, _ = qc.quant_carry_logits(m, x, "fp16", keep, acc)
    st = qc.quant_liveness_of(acc)
    assert st is not None and st["changed_frac"] > 0
    assert not torch.equal(f16, f32)
