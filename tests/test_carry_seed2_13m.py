"""2633-carry-seed2 · 第三种子补点的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义层 `seed3_stats.py` + SCHEMA §9 的复用证据 `equality_vs_2631.py`
+ `config.json` 的臂定义：

① 臂定义：九个臂 = 三规则 × 三种子（前六个复用 2631，后三个本实验新训）；
② 统计层：种子层样本量估计（判据⑥）的算术、t 临界值、Wilson「塌回地板」比例；
③ **复用证据必须可失败**：实现层逐文件 sha256 必须等于 2631 运行时记录、配置层共享字段
   逐字段相等、训练配方 flag 只差 `--seed`/`--out-root` —— 任一不满足即 `reuse_rejected`
   （本组测试直接断言 `verdict == "reuse_ok"`，并另造一个「改了配方」的反例）；
④ `results.json` 存在时（跑完之后）：判据①②③④⑥ 的字段齐备、三 seed 齐备、
   毁源对照阈值口径正确。
"""

import copy
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2633-carry-seed2")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)

import seed3_stats as s3                                           # noqa: E402
import equality_vs_2631 as eqrep                                   # noqa: E402
import reseed_arms as ra                                           # noqa: E402

CFG = json.load(open(os.path.join(EXP, "config.json")))


# ---------------- ① 臂定义 ----------------

def test_arm_grid_is_three_rules_by_three_seeds():
    assert s3.SEEDS3 == (0, 1, 2)
    names = [a["name"] for a in CFG["arms"]]
    assert names == [f"{r}_s{sd}" for sd in (0, 1, 2) for r in ("none", "plain", "delta")]
    assert CFG["new_in_2633"] == ["none_s2", "plain_s2", "delta_s2"]
    assert CFG["reused_from_2631"] == ["none_s0", "plain_s0", "delta_s0",
                                       "none_s1", "plain_s1", "delta_s1"]
    assert s3.arm_name("delta", 2) == "delta_s2"
    # seed=2 是 2631 的 SEEDS 之外的新档：2631 的函数会拒绝，本实验的委派版不会
    with pytest.raises(ValueError):
        ra.arm_name("delta", 2)


def test_arm_names_round_trip_and_delegation():
    for sd in s3.SEEDS3:
        for rule in ra.RULES:
            assert s3.arm_name(rule, sd) == f"{rule}_s{sd}"
    # 重叠档（0/1）必须与 2631 的命名逐字相同（同一口径，不另立一套）
    for sd in ra.SEEDS:
        for rule in ra.RULES:
            assert s3.arm_name(rule, sd) == ra.arm_name(rule, sd)
    with pytest.raises(ValueError):
        s3.arm_name("delta", 3)


# ---------------- ② 统计层（判据⑥） ----------------

def test_seeds_needed_monotone_and_known_value():
    # 算术对账（SCHEMA §12）：σ=24.184pp、Δ=3pp ⇒ n = 2(1.9600+0.8416)²·24.184²/9
    #   = 2·7.8493·584.86/9 = 9182.2/9 ≈ 1020.2 ⇒ ⌈⌉ = 1021
    n = s3.seeds_needed_for_mean_effect(24.184, 3.0)
    assert n == 1021
    # σ 越大、Δ 越小 ⇒ 需要越多
    assert s3.seeds_needed_for_mean_effect(10.0, 3.0) < n
    assert s3.seeds_needed_for_mean_effect(24.184, 1.0) > n


def test_seeds_needed_rejects_bad_input():
    with pytest.raises(ValueError):
        s3.seeds_needed_for_mean_effect(-1.0, 3.0)
    with pytest.raises(ValueError):
        s3.seeds_needed_for_mean_effect(1.0, 0.0)


def test_norm_ppf_matches_known_quantiles():
    assert s3._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert s3._norm_ppf(0.8) == pytest.approx(0.841621, abs=1e-5)
    assert s3._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


def test_mean_ci_t_uses_df2_critical_value():
    out = s3.mean_ci_t([36.478, 35.971, 30.0])
    assert out["n"] == 3 and out["t"] == pytest.approx(4.303)
    assert out["ci95_pp"][0] < out["mean_pp"] < out["ci95_pp"][1]


def test_collapse_wilson_bounds():
    # 三 seed 里 1 个塌回地板
    out = s3.collapse_wilson([False, True, False])
    assert out["k"] == 1 and out["n"] == 3
    assert out["wilson_ci"][0] < 0.333 < out["wilson_ci"][1]
    assert s3.collapse_wilson([True, True])["rate"] == 1.0


def test_judge_seed3_reuses_effect_stable_semantics():
    effects = {}
    for key, vals in (("delta_vs_none", [(sd, 34.9, 34.4) for sd in (0, 1, 2)]),
                      ("delta_vs_plain", [(0, 0.37, -0.15), (1, 34.06, 33.6),
                                          (2, 1.0, 0.2)])):
        per = [{"seed": sd, "effect_pp": v, "paired_boot_ci_pp": [lo, lo + 0.5]}
               for sd, v, lo in vals]
        thr = (ra.DELTA_VS_NONE_MIN_PP if key == "delta_vs_none"
               else ra.DELTA_VS_PLAIN_MIN_PP)
        effects[key] = {"per_seed": per, "stability": ra.effect_stable(per, thr)}
    out = s3.judge_seed3(effects)
    assert out["delta_vs_none"]["stable"] is True
    assert out["delta_vs_plain"]["stable"] is False       # seed0 CI 下界 < 0


# ---------------- ③ 复用证据（SCHEMA §9） ----------------

def test_reuse_equality_report_verdict_ok():
    rep = eqrep.compare(CFG)
    assert rep["implementation_identical"] is True
    assert rep["differing_fields"] == []
    assert rep["train_recipe_differing_flags"] == []
    assert rep["verdict"] == "reuse_ok"
    # 复用臂的 checkpoint 哈希必须就是 2631 当时落盘的那一份
    assert rep["reused_arm_checkpoint_sha256_at_2631_run"]["delta_s0"]


def test_reuse_equality_rejects_recipe_change():
    """反例：把训练步数改掉（配方不等）⇒ 复用必须被拒。"""
    bad = copy.deepcopy(CFG)
    bad["train_recipe_seed2"]["cmd"] = bad["train_recipe_seed2"]["cmd"].replace(
        "--steps 12000", "--steps 6000")
    rep = eqrep.compare(bad)
    assert rep["verdict"] == "reuse_rejected"
    assert "--steps" in rep["train_recipe_differing_flags"]


def test_engine_files_unchanged_since_2631_run():
    rep = eqrep.compare(CFG)
    for rel, row in rep["implementation"].items():
        assert row["identical"], f"{rel} 在 2631 之后被改过 ⇒ §9 复用前提失效"


# ---------------- ④ results.json（跑完之后） ----------------

RESULTS = os.path.join(EXP, "results.json")


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="训练/评测尚未完成")
def test_results_criteria_and_seed_coverage():
    r = json.load(open(RESULTS))
    c = r["criteria"]
    for k in ("c1_premise_none_near_chance", "c2_delta_vs_none_stable",
              "c3_delta_vs_plain_stable", "c4_corrupt_control_ok",
              "c5_no_extrapolation", "c6_sample_size_reported"):
        assert k in c, k
    assert len(r["effects"]["delta_vs_plain"]["per_seed"]) == 3
    assert [x["seed"] for x in r["effects"]["delta_vs_plain"]["per_seed"]] == [0, 1, 2]
    assert r["corrupt_control"]["factor"] == 4.0        # 判据④的阈值口径
    assert r["corrupt_control"]["bound"] == pytest.approx(4 * r["premise_check"]["0"]["chance"])
    for name in ("none_s0", "plain_s0", "delta_s0", "none_s1", "plain_s1", "delta_s1",
                 "none_s2", "plain_s2", "delta_s2"):
        assert name in r["main_leg"] and name in r["corrupt_leg"]
    assert r["sample_size_analysis"]["target_min_pp"] == 3.0


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="训练/评测尚未完成")
def test_results_judgement_is_consistent_with_its_own_numbers():
    """判据②③的布尔值必须与逐种子数字自洽（防手工改判）。"""
    r = json.load(open(RESULTS))
    for key, thr in (("delta_vs_none", ra.DELTA_VS_NONE_MIN_PP),
                     ("delta_vs_plain", ra.DELTA_VS_PLAIN_MIN_PP)):
        row = r["effects"][key]["per_seed"]
        expect = all(x["effect_pp"] >= thr and x["paired_boot_ci_pp"][0] > 0
                     for x in row)
        assert row and expect == r["effects"][key]["stability"]["all_seeds_pass"], key
