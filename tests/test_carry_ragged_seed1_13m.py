"""2630-carry-ragged-seed1 · seed=1 复现的跨 seed 统计层**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义层 `ragged_seed1_arms.py`：臂名映射、单因素定义**复用** 2629、
跨 seed 统计（极差 / 样本 sd / 符号一致性）、以及 W2 判据②③ 的两条强断言判据。

**不重复**测 2629 的单因素设计（两臂只差 carry / 块类 / 增量符号 / 毁源规则）——那些由
`tests/test_carry_ragged_projfill_13m.py` 钉住；本文件只加「seed 层」与「跨 seed 层」：

① 两 seed 除 `seed` 外逐字段同配方（读两份 `config.json` 直接对照）；
② 跨 seed 统计算术（极差 / 均值 / 样本 sd / 符号）；
③ 判据②：越界腿稳健**必须**同时满足「同号为正」与「每 seed 差值 CI 下界 > 0」；
④ 判据③：分布内冗余稳定 = 「每 seed 效应都在方差带内」。

变异验证见同目录 `mutate.py`。
"""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP2630 = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                       "2630-carry-ragged-seed1")
EXP2629 = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                       "2629-carry-ragged-projfill")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP2630)

import ragged_seed1_arms as sa  # noqa: E402
import ragged_projfill_arms as rp  # noqa: E402


# ---------------- ① 臂名映射与 seed ----------------

def test_arm_name_mapping():
    assert sa.SEED == 1
    assert sa.train_seed() == 1
    assert sa.ARM_NAMES_S1 == ("projfill_s1", "projfill_carry_s1")
    assert sa.seed0_to_seed1("projfill") == "projfill_s1"
    assert sa.seed0_to_seed1("projfill_carry") == "projfill_carry_s1"
    assert sa.seed1_to_seed0("projfill_s1") == "projfill"
    assert sa.seed1_to_seed0("projfill_carry_s1") == "projfill_carry"
    # 双向一致
    for n in sa.ARM_NAMES_S1:
        assert sa.seed0_to_seed1(sa.seed1_to_seed0(n)) == n


def test_single_factor_definition_reused_from_2629():
    """两臂配方直接复用 2629 语义层：唯一变量仍是 carry 的开关（非 seed 层变量）。"""
    b = rp.model_config_kwargs(rp.arm_spec("projfill"))
    c = rp.model_config_kwargs(rp.arm_spec("projfill_carry"))
    assert b["carry_rule"] is None and c["carry_rule"] == "delta"
    for k in ("dense", "topology", "fill_kv", "proj_fill"):
        assert b[k] == c[k], k
    # 本实验两臂 = 2629 两臂 + seed=1 命名（无新因子）
    assert set(sa.ARM_NAMES_S1) == {sa.seed0_to_seed1(n) for n in rp.ARM_NAMES}


# ---------------- ② 跨 seed 统计算术 ----------------

def test_pp_range():
    assert sa.pp_range([10.0, 20.0]) == pytest.approx(10.0, abs=1e-9)
    assert sa.pp_range([20.0, 10.0, 15.0]) == pytest.approx(10.0, abs=1e-9)
    assert sa.pp_range([30.0]) == 0.0
    assert sa.pp_mean([10.0, 20.0]) == pytest.approx(15.0, abs=1e-9)


def test_pp_sd():
    # 样本标准差（n−1）：[10,20]pp 的样本 sd = sqrt(50) ≈ 7.071
    assert sa.pp_sd([10.0, 20.0]) == pytest.approx(7.071, abs=1e-3)
    assert sa.pp_sd([10.0]) == 0.0


def test_sign_consistency():
    assert sa.sign_consistent([0.1, 0.2]) is True
    assert sa.sign_consistent([-0.1, -0.2]) is True
    assert sa.sign_consistent([0.1, -0.2]) is False
    assert sa.sign_consistent([0.0, 0.2]) is False
    assert sa.sign_consistent([]) is False


# ---------------- ③ 判据②：越界腿稳健 ----------------

def test_ood_robust_requires_significance():
    per_seed = {0: {"effect_pp": 6.05, "diff_ci_lo": 4.1, "diff_ci_hi": 8.0},
                1: {"effect_pp": 5.80, "diff_ci_lo": 3.9, "diff_ci_hi": 7.7}}
    out = sa.ood_effect_robust(per_seed)
    assert out["robust"] is True
    assert out["same_sign_positive"] is True
    assert out["ci_lower_positive_each"] is True
    assert out["range_pp"] == pytest.approx(0.25, abs=1e-9)
    # 效应为正但 CI 下界 ≤ 0（不显著）⇒ 不稳健
    weak = {0: {"effect_pp": 6.05, "diff_ci_lo": 4.1, "diff_ci_hi": 8.0},
            1: {"effect_pp": 0.30, "diff_ci_lo": -1.2, "diff_ci_hi": 1.8}}
    assert sa.ood_effect_robust(weak)["robust"] is False
    # 反号 ⇒ 不稳健
    flip = {0: {"effect_pp": 6.05, "diff_ci_lo": 4.1, "diff_ci_hi": 8.0},
            1: {"effect_pp": -2.0, "diff_ci_lo": -3.1, "diff_ci_hi": -0.9}}
    assert sa.ood_effect_robust(flip)["robust"] is False
    # 只有一个 seed ⇒ 不稳健（n=2 才谈跨 seed）
    assert sa.ood_effect_robust({0: per_seed[0]})["robust"] is False
    assert "statistical_test" in out["evidence_type"]


# ---------------- ④ 判据③：分布内冗余稳定 ----------------

def test_indist_redundancy_band():
    within = {0: {"effect_pp": 0.36, "diff_ci_lo": -0.66, "diff_ci_hi": 1.38},
              1: {"effect_pp": -0.20, "diff_ci_lo": -1.1, "diff_ci_hi": 0.7}}
    out = sa.indist_redundancy_stable(within)
    assert out["stable"] is True and out["all_within_band"] is True
    assert out["band_pp"] == 1.4
    # 有一个 seed 超带 ⇒ 冗余结论不稳定
    out2 = sa.indist_redundancy_stable(
        {0: within[0], 1: {"effect_pp": 2.5, "diff_ci_lo": 1.0, "diff_ci_hi": 4.0}})
    assert out2["stable"] is False and out2["all_within_band"] is False
    assert "magnitude" in out["evidence_type"]


# ---------------- ⑤ 两 seed 除 seed 外逐字段同配方 ----------------

def test_config_matches_seed0_except_seed():
    c0 = json.load(open(os.path.join(EXP2629, "config.json")))
    c1 = json.load(open(os.path.join(EXP2630, "config.json")))
    assert c0["train"]["seed"] == 0 and c1["train"]["seed"] == 1
    assert c1["seed"] == 1
    for k in ("model", "tokens"):
        assert c0[k] == c1[k], k
    for k in ("batch_size", "lr", "steps", "final_weight", "exit_mode", "tokens_per_arm"):
        assert c0["train"][k] == c1["train"][k], k
    for k in ("train_task_leg", "tf_leg", "bpc_leg"):
        # 同一批位置 / 同一评测口径（保证跨 seed 可比）
        assert {kk: vv for kk, vv in c0[k].items() if kk != "note"} == \
               {kk: vv for kk, vv in c1[k].items() if kk != "note"}, k
    # 唯一改动 = seed 与臂名（配方逐字段相同）
    assert c0["model"] == c1["model"]
    assert {a["name"] for a in c0["arms"]} == {"projfill", "projfill_carry"}
    assert {a["name"] for a in c1["arms"]} == set(sa.ARM_NAMES_S1)


def test_seed0_results_present_for_merge():
    """合并 n=2 的前提：2629 的 results.json 已有本轮要引用的腿。"""
    R = json.load(open(os.path.join(EXP2629, "results.json")))
    for key in ("train_task_leg", "train_task_leg_no_last_block",
                "main_leg_single_query_ood", "tf_synth_indist", "tf_enwik8"):
        assert key in R, key
    for mode in ("in_distribution_random_exit", "full_depth_deployment"):
        assert mode in R["train_task_leg"]
        for name in ("projfill", "projfill_carry"):
            assert name in R["train_task_leg"][mode]
            assert R["train_task_leg"][mode][name]["n"] == 16384
    # 越界腿 n=2048（W2 明写的口径）
    for name in ("projfill", "projfill_carry"):
        assert R["main_leg_single_query_ood"]["full_depth_deployment"][name]["n"] == 2048
