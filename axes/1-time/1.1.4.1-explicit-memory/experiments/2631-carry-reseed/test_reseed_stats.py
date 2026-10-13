"""2631-carry-reseed · 本地单测（先红后绿 + 变异靶子）。

**位置说明**：本轮并行的写入者分工里 `tests/**` 属 L1（instr）线，本线（L4）不写共享
`tests/` 目录 ⇒ 单测放在本实验目录内，用
`python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2631-carry-reseed/test_reseed_stats.py -q`
单跑（不进 `tests/` 基线；基线仍由 `python3 -m pytest tests/ -q` 全绿保证）。

覆盖：臂定义、聚类 bootstrap（B2 的修法）、配对差、极差/方向/sd、
跨种子「稳健」判定、毁源对照的作废判定，以及主腿/辅助腿的**非退化**与**分布内外**钉死。
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import reseed_arms as ra                                            # noqa: E402
from tools.evalkit.longrange_tasks import (assoc_recall_batch,      # noqa: E402
                                           assoc_recall_multi_batch)
from tools.evalkit.carry_channel import wilson_ci                   # noqa: E402

TRAIN_MAX_DIST = 240        # 训练生成器（K=8/Q=8/gap=14）能出现的最远查询距离


# ------------------------------------------------------------------ 臂定义


def test_arm_names_and_single_factor():
    assert ra.arm_name("delta", 1) == "delta_s1"
    names = [a["name"] for a in ra.ARMS]
    assert names == ["none_s0", "plain_s0", "delta_s0", "none_s1", "plain_s1", "delta_s1"]
    # 唯一变量：同 seed 内只有 carry_rule 变，同 rule 内只有 seed 变
    for seed in ra.SEEDS:
        rules = [a["rule"] for a in ra.ARMS if a["seed"] == seed]
        assert sorted(rules) == sorted(ra.RULES) and len(rules) == len(ra.RULES)


def test_model_config_kwargs_and_train_flags():
    assert ra.model_config_kwargs("none") == {"carry_rule": None}
    assert ra.model_config_kwargs("plain") == {"carry_rule": "plain"}
    assert ra.train_flags("none") == []
    assert ra.train_flags("delta") == ["--carry-write", "delta"]


def test_unknown_rule_rejected():
    for fn in (ra.arm_name, ra.model_config_kwargs, ra.train_flags):
        try:
            fn("residual") if fn is not ra.arm_name else fn("residual", 0)
        except ValueError:
            pass
        else:                                                       # pragma: no cover
            raise AssertionError(f"{fn.__name__} 未拒绝未知规则")


# ------------------------------------------------------------------ 聚类统计（B2）


def test_pooled_accuracy_and_mismatch():
    assert ra.pooled_accuracy([1, 2], [4, 4]) == 3 / 8
    try:
        ra.pooled_accuracy([1], [4, 4])
    except ValueError:
        pass
    else:                                                           # pragma: no cover
        raise AssertionError("长度不一致未被拒绝")


def test_clustered_ci_wider_than_naive_wilson_under_clustering():
    """聚类的存在必须让主 CI 比「按独立查询算」的 Wilson 宽（B2 的机制）。"""
    # 40 条序列 × 8 查询：簇内完全相关（一半序列全中、一半全不中）
    hits = [8] * 20 + [0] * 20
    ns = [8] * 40
    clus = ra.clustered_bootstrap_ci(hits, ns, n_boot=1500, seed=2631)
    naive = wilson_ci(sum(hits), sum(ns))
    assert abs(clus[0] - clus[1]) > abs(naive[0] - naive[1])
    assert clus[0] <= ra.pooled_accuracy(hits, ns) <= clus[1]
    # 完全独立（每条序列都 4/8）时，聚类 CI 不应比 Wilson 窄得离谱
    indep = ra.clustered_bootstrap_ci([4] * 40, [8] * 40, n_boot=800, seed=7)
    assert abs(indep[0] - indep[1]) < abs(clus[0] - clus[1])


def test_paired_bootstrap_diff_sign_and_null():
    a_hits, a_n = [0] * 30 + [8] * 10, [8] * 40
    b_hits, b_n = [8] * 30 + [0] * 10, [8] * 40
    ci = ra.paired_clustered_bootstrap_diff_ci(a_hits, a_n, b_hits, b_n,
                                               n_boot=1200, seed=2631)
    assert ci[0] > 0, ci                      # b 明显更好 ⇒ 差值 CI 下界 > 0
    same = ra.paired_clustered_bootstrap_diff_ci(a_hits, a_n, a_hits, a_n,
                                                 n_boot=200, seed=1)
    assert same == [0.0, 0.0]                 # 同一臂自比 ⇒ 恒为 0


def test_effect_reports_paired_ci_and_counts():
    base = ra.arm_reading([0] * 40, [8] * 40, n_boot=200, seed=1)
    tgt = ra.arm_reading([4] * 20 + [0] * 20, [8] * 40, n_boot=200, seed=1)
    eff = ra.effect(base, tgt, n_boot=200, seed=1)
    assert eff["effect_pp"] == 25.0
    assert eff["paired_boot_ci_pp"][0] > 0
    assert eff["base_n"] == 320 and eff["target_n_hits"] == 80
    assert eff["base_accuracy"] == 0.0


# ------------------------------------------------------------------ 跨种子判据


def test_pp_range_mean_sd_and_sign():
    assert ra.pp_range([1.0, 4.5, -2.0]) == 6.5
    assert ra.pp_mean([1.0, 2.0, 3.0]) == 2.0
    assert ra.pp_sd([1.0, 2.0, 3.0]) == 1.0            # 样本 sd（除 n−1），不是 0.8165
    assert ra.sign_consistent([1.0, 2.0]) and ra.sign_consistent([-1.0, -0.5])
    assert not ra.sign_consistent([1.0, -2.0])
    assert not ra.sign_consistent([1.0, 0.0])


def _row(seed, eff, lo):
    return {"seed": seed, "effect_pp": eff, "paired_boot_ci_pp": [lo, 9.9]}


def test_effect_stable_requires_threshold_and_positive_ci_lower():
    ok = ra.effect_stable([_row(0, 8.0, 6.0), _row(1, 6.5, 4.0)], 5.0)
    assert ok["stable"] and ok["all_seeds_pass"] and ok["sign_consistent"]
    assert ok["range_pp"] == 1.5 and ok["n_seeds"] == 2
    below = ra.effect_stable([_row(0, 8.0, 6.0), _row(1, 4.9, 2.0)], 5.0)
    assert not below["stable"]                          # 点估计不过 5pp
    ci = ra.effect_stable([_row(0, 8.0, 6.0), _row(1, 7.0, -0.2)], 5.0)
    assert not ci["stable"]                             # CI 下界跨 0 ⇒ 不稳
    flips = ra.effect_stable([_row(0, 8.0, 6.0), _row(1, -6.0, -8.0)], 5.0)
    assert not flips["stable"] and not flips["sign_consistent"]


def test_effect_stable_needs_two_seeds():
    one = ra.effect_stable([_row(0, 8.0, 6.0)], 5.0)
    assert one["n_seeds"] == 1 and not one["stable"]


def test_corrupt_control_flags_void_arm():
    def leg(main_lo, cor_lo):
        return {"main": {"clustered_ci": [main_lo, 0.5]},
                "corrupt": {"clustered_ci": [cor_lo, 0.5]}}
    chance = 1 / 64
    good = ra.corrupt_control_reading({"delta_s0": leg(0.08, 0.005)}, chance)
    assert good["ok"] and good["void_arms"] == []
    bad = ra.corrupt_control_reading({"delta_s0": leg(0.08, 0.30)}, chance)
    assert not bad["ok"] and bad["void_arms"] == ["delta_s0"]
    floor = ra.corrupt_control_reading({"none_s0": leg(0.001, 0.001)}, chance)
    assert floor["ok"]              # 主臂本就在地板 ⇒ 无作废风险


# ------------------------------------------------------------------ 腿的非退化/分布内外


def test_main_leg_is_in_distribution_and_nondegenerate():
    t = assoc_recall_multi_batch(4, 256, 8, 8, generator=torch.Generator().manual_seed(0))
    d = t.dists.flatten()
    assert t.meta["task"] == "assoc_recall_multi"
    assert int(d.min()) >= 16                      # 远于窗口 16
    assert int(d.max()) <= TRAIN_MAX_DIST          # **训练分布内**（≤240）
    assert t.rows.max() < 256                      # 提问位置不是 T−1（训练见过的形态）
    # 局部先验 = chance：窗口内不含任何键/值符号 ⇒ 不搬就答不出
    assert t.chance() == 1 / 64
    # 非退化：查询数 = 序列数 × 8，且每条序列问的是不同 pair
    assert t.rows.shape == (4, 8)
    assert len(set(t.rows[0].tolist())) == 8


def test_auxiliary_ood_leg_is_out_of_distribution():
    t = assoc_recall_batch(4, 256, 8, generator=torch.Generator().manual_seed(0))
    d = t.dists.flatten()
    assert int(d.min()) > TRAIN_MAX_DIST           # 2618 首验用的就是这条腿 ⇒ 越界
    assert 241 <= int(d.min()) and int(d.max()) <= 255
    assert t.rows.max() == 255                     # 提问行 = T−1（训练里从未作为提问行）


def test_generator_seed_is_model_independent_so_arms_are_paired():
    t1 = assoc_recall_multi_batch(2, 256, 8, 8, generator=torch.Generator().manual_seed(5000))
    t2 = assoc_recall_multi_batch(2, 256, 8, 8, generator=torch.Generator().manual_seed(5000))
    assert torch.equal(t1.tokens, t2.tokens)
    assert torch.equal(t1.answers, t2.answers)
    assert torch.equal(t1.dists, t2.dists)
