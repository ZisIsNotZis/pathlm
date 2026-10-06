"""2630-carry-ragged-seed1 · 对账（SCHEMA §12 三道 + 跨 seed 合并对账）。

`analysis.md` 的每个数字都能在 `results.json`（及其引用的 2629 结果 / 训练日志）里找到
同值，且算术自洽：

1. **算术对账**：准确率 = hits/n、Wilson CI、单因素增量（§17.17）、bpc 差分、参数量差、
   tf@k 一致率、去伪影块——全部从 `results.json` 的原始计数重算。
2. **任务对账**：seed=1 两臂同一任务/同一位置集（n、chance、local_prior 相等）；毁源同批；
   主判据腿距离全部 ≤240（训练分布内，§17.18），越界腿距离 >240。
3. **口径对账**：基准唯一（`projfill_s1`）；两臂非 dense + proj-fill；块类/参数量声明一致；
   carry 臂活性非空、无 carry 臂为 null；clean bpc 是非退化锚（> log2 206）；收敛门禁在
   答案行 CE 上；出口资产保持性判定与 Δ 一致；seed=1。
4. **跨 seed 对账（W2 判据①②③）**：从 **2629 与本实验两份 `results.json` 的原始计数**
   （hits/n、tf 一致数/n）**独立重算**每 seed 的 carry 效应 + `two_proportion` 差值 CI，
   与 `results.json::cross_seed` 的派生字段逐项比对（极差、均值、判据②/③ 的布尔标记）；
   并核对每 seed n（分布内 16384 / 越界腿 2048）。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from tools.evalkit.carry_channel import two_proportion, wilson_ci  # noqa: E402
import ragged_seed1_arms as sa  # noqa: E402
from ragged_projfill_arms import (single_factor_increment,  # noqa: E402
                                  single_factor_increment_pp)

RAND = "in_distribution_random_exit"
FULL = "full_depth_deployment"
BASE = sa.seed0_to_seed1("projfill")
TGT = sa.seed0_to_seed1("projfill_carry")
ARMS = (BASE, TGT)

CROSS_LEGS = ("recall_indist_random_exit", "recall_indist_random_exit_no_last_block",
              "recall_full_depth_deployment", "ood_single_query_random_exit",
              "ood_single_query_full_depth", "tf4_synth_indist", "tf4_enwik8")


def _counts(row):
    return int(row["n_hits"]), int(row["n"])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_counts(R, seed_key, T):
    """从一份 results.json 提取该 seed 各腿的 (基准 hits/n, 目标 hits/n)。

    `seed_key` = "seed0"（2629 臂名 = projfill/projfill_carry）或 "seed1"（本实验臂名）。
    只读**原始计数**（hits/n、tf 一致数/n）——不引用派生字段，避免口径漂移（§14）。
    """
    if seed_key == "seed0":
        b, t = sa.BASELINE_ARM, sa.TARGET_ARM
    else:
        b, t = BASE, TGT
    in_rand, in_full = R["train_task_leg"][RAND], R["train_task_leg"][FULL]
    nlb = R["train_task_leg_no_last_block"][RAND]
    ood_rand, ood_full = R["main_leg_single_query_ood"][RAND], \
        R["main_leg_single_query_ood"][FULL]
    n_en = R["config"]["tf_leg"]["enwik8_n_batches"] * \
        R["config"]["tf_leg"]["enwik8_batch_size"] * (T - 1)
    n_synth = int(R["tf_synth_indist"][b]["n_tokens"])
    return {
        "recall_indist_random_exit": (_counts(in_rand[b]), _counts(in_rand[t])),
        "recall_indist_random_exit_no_last_block": (_counts(nlb[b]), _counts(nlb[t])),
        "recall_full_depth_deployment": (_counts(in_full[b]), _counts(in_full[t])),
        "ood_single_query_random_exit": (_counts(ood_rand[b]), _counts(ood_rand[t])),
        "ood_single_query_full_depth": (_counts(ood_full[b]), _counts(ood_full[t])),
        "tf4_synth_indist": (
            (round(R["tf_synth_indist"][b]["accept"]["4"] * n_synth), n_synth),
            (round(R["tf_synth_indist"][t]["accept"]["4"] * n_synth), n_synth)),
        "tf4_enwik8": (
            (round(R["tf_enwik8"][b]["tf4"] * n_en), n_en),
            (round(R["tf_enwik8"][t]["tf4"] * n_en), n_en)),
    }


def _effect(base_counts, tgt_counts):
    (bh, bn), (th, tn) = base_counts, tgt_counts
    tp = two_proportion(bh, bn, th, tn)
    return {"effect_pp": tp["delta_pp"],
            "diff_ci_lo": tp["diff_ci_95"][0] * 100.0,
            "diff_ci_hi": tp["diff_ci_95"][1] * 100.0}


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    R0 = json.load(open(sa.SEED0_RESULTS))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = (abs(float(got) - float(exp)) <= tol
              if isinstance(got, (int, float)) and isinstance(exp, (int, float))
              else got == exp)
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    modes = (RAND, FULL)

    # ---------------- ① 算术对账（本实验 seed=1） ----------------
    for mode in modes:
        for name in ARMS:
            arm = R["train_task_leg"][mode][name]
            ck(f"main.{mode}.{name}.acc==hits/n", arm["accuracy"], arm["n_hits"] / arm["n"], 1e-6)
            ck(f"main.{mode}.{name}.ci", arm["accuracy_ci"], wilson_ci(arm["n_hits"], arm["n"]), 1e-9)
            a = arm["corrupt_source"]
            ck(f"main.{mode}.{name}.corrupt==hits/n", a["accuracy"], a["n_hits"] / a["n"], 1e-6)
            ck(f"main.{mode}.{name}.corrupt.ci", a["accuracy_ci"],
               wilson_ci(a["n_hits"], a["n"]), 1e-9)
            o = R["main_leg_single_query_ood"][mode][name]
            ck(f"ood.{mode}.{name}.acc==hits/n", o["accuracy"], o["n_hits"] / o["n"], 1e-6)
            ck(f"ood.{mode}.{name}.ci", o["accuracy_ci"],
               wilson_ci(o["n_hits"], o["n"]), 1e-9)

    def _map_to_seed0(d):
        """把 seed=1 臂名键映射回 2629 的 seed=0 臂名（复用同一件算术）。"""
        return {sa.seed1_to_seed0(n): v for n, v in d.items()}

    def recheck_inc(key, leg, scale=100.0):
        acc = _map_to_seed0({n: leg[n]["accuracy"] for n in ARMS})
        for k, v in single_factor_increment(acc).items():
            ck(f"incr.{key}.{k}", R["increments"][key][k], round(scale * v, 6), 1e-6)

    recheck_inc("recall_random_exit", R["train_task_leg"][RAND])
    recheck_inc("recall_full_depth", R["train_task_leg"][FULL])
    recheck_inc("recall_random_exit_no_last_block",
                R["train_task_leg_no_last_block"][RAND])
    recheck_inc("single_query_ood_random_exit", R["main_leg_single_query_ood"][RAND])
    recheck_inc("single_query_ood_full_depth", R["main_leg_single_query_ood"][FULL])
    for key, leg in (("tf4_synth_indist", {n: R["tf_synth_indist"][n]["accept"]["4"]
                                           for n in ARMS}),
                     ("tf4_enwik8", {n: R["tf_enwik8"][n]["tf4"] for n in ARMS})):
        for k, v in single_factor_increment(_map_to_seed0(leg)).items():
            ck(f"incr.{key}.{k}", R["increments"][key][k], round(100 * v, 6), 1e-6)
    ck("bpc.incr.carry_effect", R["increments"]["bpc"]["carry_effect"],
       round(R["bpc_leg"][TGT]["bpc"] - R["bpc_leg"][BASE]["bpc"], 6), 1e-9)
    for name in ARMS:
        ck(f"bpc.damage.{name}", R["bpc_damage_vs_base"][name],
           round(R["bpc_leg"][name]["bpc"] - R["bpc_leg"][BASE]["bpc"], 6), 1e-9)
        d = R["param_declaration"][name]
        ck(f"param.{name}.delta==n-base", d["delta_vs_base"],
           d["n_params"] - R["param_declaration"][BASE]["n_params"])
        ck(f"tf.{name}.tf4==by_depth[depth4]", R["tf_enwik8"][name]["tf4"],
           next(r["accept"] for r in R["tf_enwik8"][name]["by_depth"]
                if r["depth"] == 4), 1e-9)
    # 去伪影块：与 by_distance（d≤223）重算一致
    for mode in modes:
        for name in ARMS:
            n = h = 0
            for d, v in R["train_task_leg"][mode][name]["by_distance"].items():
                if int(d) <= 223:
                    n += v["n"]; h += round(v["accuracy"] * v["n"])
            ck(f"no_last.{mode}.{name}.n", R["train_task_leg_no_last_block"][mode][name]["n"], n)
            ck(f"no_last.{mode}.{name}.acc",
               R["train_task_leg_no_last_block"][mode][name]["accuracy"], round(h / n, 6), 1e-6)

    # ---------------- ② 任务对账 ----------------
    for mode in modes:
        ck(f"task.{mode}.same_n", len({R["train_task_leg"][mode][n]["n"] for n in ARMS}), 1)
        ck(f"task.{mode}.same_chance",
           len({R["train_task_leg"][mode][n]["chance"] for n in ARMS}), 1)
        ck(f"task.{mode}.same_local_prior",
           len({R["train_task_leg"][mode][n]["local_prior"] for n in ARMS}), 1)
        ck(f"task.{mode}.corrupt_same_n",
           len({R["train_task_leg"][mode][n]["corrupt_source"]["n"] for n in ARMS}), 1)
        ck(f"task.{mode}.ood_same_n",
           len({R["main_leg_single_query_ood"][mode][n]["n"] for n in ARMS}), 1)
        for name in ARMS:
            ck(f"task.indist.{mode}.{name}.max_dist<=240",
               R["train_task_leg"][mode][name]["query_key_distance"]["max"], 240)
            ck(f"task.indist.{mode}.{name}.in_dist_flag",
               R["train_task_leg"][mode][name]["query_key_distance"]["in_train_distribution"], True)
            ck(f"task.ood.{mode}.{name}.min_dist>240",
               R["main_leg_single_query_ood"][mode][name]["distance_range"][0] > 240, True)
            ck(f"task.ood.{mode}.{name}.in_dist_flag_false",
               R["main_leg_single_query_ood"][mode][name]["in_train_distribution"], False)
    # W2 明写的口径 n：分布内 16384、越界腿 2048
    ck("task.indist_n_16384", R["train_task_leg"][RAND][BASE]["n"], 16384)
    ck("task.ood_n_2048", R["main_leg_single_query_ood"][FULL][BASE]["n"], 2048)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.seed_is_1", R["seed"], 1)
    ck("caliber.config_seed_is_1", R["config"]["train"]["seed"], 1)
    ck("caliber.baseline", BASE, "projfill_s1")
    ck("caliber.target", TGT, "projfill_carry_s1")
    aeq = R["architecture_equality"]
    ck("caliber.block_classes", aeq["block_class"], aeq["block_class_expected"])
    ck("caliber.both_non_dense", all(v is False for v in aeq["dense"].values()), True)
    ck("caliber.both_proj_fill", all(v is True for v in aeq["proj_fill"].values()), True)
    ck("caliber.both_fill_kv", all(v is True for v in aeq["fill_kv"].values()), True)
    ck("caliber.param_delta_positive",
       R["param_declaration"][TGT]["delta_vs_base"] > 0, True)
    ck("caliber.destroy_all_valid",
       all(R["destroy_source_validity"][m][n]["valid"] for m in modes for n in ARMS), True)
    ck("caliber.bpc_all_above_uniform",
       all(R["bpc_leg"][n]["bpc"] > math.log2(206) for n in ARMS), True)
    ck("caliber.eval_only", R["cost"]["pure_inference"], True)
    ck("caliber.carry_activity_present",
       R["train_activity"][TGT]["attn_carry_series_present"] > 0, True)
    ck("caliber.nocarry_arm_quiet", R["train_activity"][BASE]["last_attn_carry"] is None, True)
    ck("caliber.convergence_on_answer_ce",
       all(R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"] is not None
           for n in ARMS), True)
    ck("caliber.tf4_present",
       all("4" in R["tf_synth_indist"][n]["accept"] for n in ARMS), True)
    eh = R["exit_asset_holds"]
    ck("caliber.exit_delta_indist", eh["delta_pp_indist"],
       round(100 * (eh["tf4_indist_carry"] - eh["tf4_indist_base"]), 4), 1e-6)
    ck("caliber.exit_holds_indist_flag", eh["holds_indist"], abs(eh["delta_pp_indist"]) <= 1.4)
    ck("caliber.exit_indist_matches_tf_synth",
       eh["tf4_indist_base"], R["tf_synth_indist"][BASE]["accept"]["4"], 1e-9)
    ck("caliber.capability_buildable", R["capability_probe"]["combo_buildable"], True)
    ck("caliber.capability_trunk_identical", R["capability_probe"]["trunk_bit_identical"], True)

    # ---------------- ④ 跨 seed 对账（W2 判据①②③） ----------------
    T = R["config"]["model"]["seq_len"]
    c0 = _extract_counts(R0, "seed0", T)
    c1 = _extract_counts(R, "seed1", T)
    recomputed = {leg: {0: _effect(*c0[leg]), 1: _effect(*c1[leg])} for leg in CROSS_LEGS}
    cs = R["cross_seed"]
    ck("cross.n_seeds", cs["n_seeds"], 2)
    ck("cross.seeds", cs["seeds"], [0, 1])
    ck("cross.seed0_source_relpath", cs["seed0_source"],
       os.path.relpath(sa.SEED0_RESULTS, ROOT))
    ck("cross.seed0_source_sha256_matches", cs["seed0_source_sha256"],
       _sha256(sa.SEED0_RESULTS))
    # 每腿每 seed：效应与差值 CI 一字不差
    for leg in CROSS_LEGS:
        for s in (0, 1):
            got = cs["legs"][leg][str(s)]
            exp = recomputed[leg][s]
            ck(f"cross.{leg}.{s}.effect_pp", got["effect_pp"], exp["effect_pp"], 1e-9)
            ck(f"cross.{leg}.{s}.diff_ci_lo", got["diff_ci_lo"], exp["diff_ci_lo"], 1e-6)
            ck(f"cross.{leg}.{s}.diff_ci_hi", got["diff_ci_hi"], exp["diff_ci_hi"], 1e-6)
        # 极差 = pp_range
        ck(f"cross.{leg}.range", cs["criterion_1_range_pp"][leg],
           sa.pp_range([recomputed[leg][0]["effect_pp"], recomputed[leg][1]["effect_pp"]]), 1e-9)
    # 判据②③ 布尔标记从重算的 per-seed 字典复现
    ck("cross.c2_ood_full_from_recompute",
       cs["criterion_2_ood_robust_full_depth"]["robust"],
       sa.ood_effect_robust(recomputed["ood_single_query_full_depth"])["robust"])
    ck("cross.c2_ood_rand_from_recompute",
       cs["criterion_2_ood_robust_random_exit"]["robust"],
       sa.ood_effect_robust(recomputed["ood_single_query_random_exit"])["robust"])
    ck("cross.c3_indist_from_recompute",
       cs["criterion_3_indist_redundancy_stable"]["stable"],
       sa.indist_redundancy_stable(recomputed["recall_indist_random_exit"])["stable"])
    # seed=0 的效应 == 2629 自己的 increments（同口径交叉核对；2629 的 increment 是
    # 6 位小数的 acc 差，这里的两比例 delta_pp 是 3 位小数 ⇒ 容差 1e-3）
    ck("cross.seed0_indist_effect==2629_increment",
       recomputed["recall_indist_random_exit"][0]["effect_pp"],
       R0["increments"]["recall_random_exit"]["carry_effect"], 1e-3)
    ck("cross.seed0_ood_full_effect==2629_increment",
       recomputed["ood_single_query_full_depth"][0]["effect_pp"],
       R0["increments"]["single_query_ood"]["carry_effect"], 1e-3)
    # 每 seed 的 n 与 W2 明写口径一致
    for leg, want_n in (("recall_indist_random_exit", 16384),
                        ("ood_single_query_full_depth", 2048)):
        ck(f"cross.{leg}.seed0_n", c0[leg][0][1], want_n)
        ck(f"cross.{leg}.seed1_n", c1[leg][0][1], want_n)
    # 强断言用词受证据类型约束
    ck("cross.c2_evidence_type_statistical_test",
       "statistical_test" in cs["criterion_2_ood_robust_full_depth"]["evidence_type"], True)
    ck("cross.c3_evidence_type_magnitude",
       "magnitude" in cs["criterion_3_indist_redundancy_stable"]["evidence_type"], True)
    ck("cross.no_extrapolation_present", bool(cs.get("no_extrapolation")), True)

    ck("files.analysis_exists", os.path.exists(os.path.join(HERE, "analysis.md")), True)

    # ---------------- ⑤ analysis.md 引用的关键数字 → results.json -------------
    legs = cs["legs"]
    TFS, TFE = R["tf_synth_indist"], R["tf_enwik8"]
    ACT = R["train_activity"]
    TFS0, TFE0 = R0["tf_synth_indist"], R0["tf_enwik8"]
    ACT0 = R0["train_activity"]
    CITED = [
        # §2 跨 seed 主表（效应 pp / 极差 pp）
        ("s2 indist seed0 eff 0.36", legs["recall_indist_random_exit"]["0"]["effect_pp"], 0.36, 1e-9),
        ("s2 indist seed1 eff -3.259", legs["recall_indist_random_exit"]["1"]["effect_pp"], -3.259, 1e-9),
        ("s2 indist range 3.619", cs["criterion_1_range_pp"]["recall_indist_random_exit"], 3.619, 1e-9),
        ("s2 ood full seed0 eff 6.055", legs["ood_single_query_full_depth"]["0"]["effect_pp"], 6.055, 1e-9),
        ("s2 ood full seed1 eff -2.441", legs["ood_single_query_full_depth"]["1"]["effect_pp"], -2.441, 1e-9),
        ("s2 ood full range 8.496", cs["criterion_1_range_pp"]["ood_single_query_full_depth"], 8.496, 1e-9),
        ("s2 ood rand range 9.472", cs["criterion_1_range_pp"]["ood_single_query_random_exit"], 9.472, 1e-9),
        ("s2 tf4 enwik8 range 17.95", cs["criterion_1_range_pp"]["tf4_enwik8"], 17.95, 1e-9),
        # 每 seed 差值 CI（强断言证据）
        ("s2 ood full seed0 ci_lo 4.6686", legs["ood_single_query_full_depth"]["0"]["diff_ci_lo"], 4.6686, 1e-3),
        ("s2 ood full seed1 ci_hi -1.2152", legs["ood_single_query_full_depth"]["1"]["diff_ci_hi"], -1.2152, 1e-3),
        ("s2 indist seed1 ci_lo -4.2666", legs["recall_indist_random_exit"]["1"]["diff_ci_lo"], -4.2666, 1e-3),
        ("s2 indist seed1 ci_hi -2.2520", legs["recall_indist_random_exit"]["1"]["diff_ci_hi"], -2.2520, 1e-3),
        # 逐 seed 原始计数（§2 表）
        ("s2 seed0 indist base hits 5340", R0["train_task_leg"][RAND][sa.BASELINE_ARM]["n_hits"], 5340),
        ("s2 seed0 indist carry hits 5399", R0["train_task_leg"][RAND][sa.TARGET_ARM]["n_hits"], 5399),
        ("s2 seed1 indist base hits 5466", R["train_task_leg"][RAND][BASE]["n_hits"], 5466),
        ("s2 seed1 indist carry hits 4932", R["train_task_leg"][RAND][TGT]["n_hits"], 4932),
        ("s2 seed0 ood base hits 51", R0["main_leg_single_query_ood"][FULL][sa.BASELINE_ARM]["n_hits"], 51),
        ("s2 seed0 ood carry hits 175", R0["main_leg_single_query_ood"][FULL][sa.TARGET_ARM]["n_hits"], 175),
        ("s2 seed1 ood base hits 111", R["main_leg_single_query_ood"][FULL][BASE]["n_hits"], 111),
        ("s2 seed1 ood carry hits 61", R["main_leg_single_query_ood"][FULL][TGT]["n_hits"], 61),
        # §3 出口资产
        ("s3 tf4 synth seed1 base 0.9942", TFS[BASE]["accept"]["4"], 0.9942, 5e-5),
        ("s3 tf4 synth seed1 carry 0.9953", TFS[TGT]["accept"]["4"], 0.9953, 5e-5),
        ("s3 tf4 enwik8 seed1 base 0.6888", TFE[BASE]["tf4"], 0.6888, 5e-5),
        ("s3 tf4 enwik8 seed1 carry 0.8212", TFE[TGT]["tf4"], 0.8212, 5e-5),
        ("s3 tf4 synth seed0 base 0.9966", TFS0[sa.BASELINE_ARM]["accept"]["4"], 0.9966, 5e-5),
        ("s3 tf4 synth seed0 carry 0.9868", TFS0[sa.TARGET_ARM]["accept"]["4"], 0.9868, 5e-5),
        ("s3 tf4 enwik8 seed0 base 0.9369", TFE0[sa.BASELINE_ARM]["tf4"], 0.9369, 5e-5),
        ("s3 tf4 enwik8 seed0 carry 0.8898", TFE0[sa.TARGET_ARM]["tf4"], 0.8898, 5e-5),
        ("s3 enwik8 base cross-seed gap 24.81",
         round(100 * (TFE0[sa.BASELINE_ARM]["tf4"] - TFE[BASE]["tf4"]), 2), 24.81, 1e-2),
        # §4 参数 / 活性 / 收敛 / bpc
        ("s4 param base 14,326,272", R["param_declaration"][BASE]["n_params"], 14326272),
        ("s4 param carry 14,590,568", R["param_declaration"][TGT]["n_params"], 14590568),
        ("s4 param delta 264,296", R["param_declaration"][TGT]["delta_vs_base"], 264296),
        ("s4 param pct 1.845", R["param_declaration"][TGT]["delta_pct_vs_base"], 1.845, 5e-4),
        ("s4 read_share 0.402539", ACT[TGT]["last_attn_carry"]["read_share"], 0.402539, 5e-6),
        ("s4 carry series 30", ACT[TGT]["attn_carry_series_present"], 30),
        ("s4 nocarry quiet", ACT[BASE]["last_attn_carry"], None),
        ("s4 ans_ce base 2.5615", ACT[BASE]["final_synth_answer_ce"], 2.5615, 5e-5),
        ("s4 ans_ce carry 3.1789", ACT[TGT]["final_synth_answer_ce"], 3.1789, 5e-5),
        ("s4 slope base -0.18578", ACT[BASE]["answer_ce_slope_nats_per_1k_lastquarter"], -0.18578, 5e-5),
        ("s4 slope carry -0.22087", ACT[TGT]["answer_ce_slope_nats_per_1k_lastquarter"], -0.22087, 5e-5),
        ("s4 seed0 ans_ce base 3.0723", ACT0[sa.BASELINE_ARM]["final_synth_answer_ce"], 3.0723, 5e-5),
        ("s4 seed0 read_share 0.541741", ACT0[sa.TARGET_ARM]["last_attn_carry"]["read_share"], 0.541741, 5e-6),
        ("s4 bpc base 15.880892", R["bpc_leg"][BASE]["bpc"], 15.880892, 5e-6),
        ("s4 bpc carry 17.387679", R["bpc_leg"][TGT]["bpc"], 17.387679, 5e-6),
        # §7 成本
        ("s7 eval wall 0.466", R["cost"]["wall_minutes_total_eval"], 0.466, 1e-9),
        ("s7 eval peak 786.6", R["cost"]["peak_gpu_mem_mb"], 786.6, 1e-9),
        # 判据标记
        ("s2 c2 robust false", cs["criterion_2_ood_robust_full_depth"]["robust"], False),
        ("s2 c2 rand robust false", cs["criterion_2_ood_robust_random_exit"]["robust"], False),
        ("s2 c3 indist stable false", cs["criterion_3_indist_redundancy_stable"]["stable"], False),
        ("s2 c3 nlb stable true", cs["criterion_3_no_last_block"]["stable"], True),
    ]
    for name, got, exp, *rest in CITED:
        ck(name, got, exp, rest[0] if rest else 0.0)

    n_fail = sum(1 for c in checks if not c["pass"])
    report = {"n_checks": len(checks), "n_fail": n_fail, "all_pass": n_fail == 0,
              "checks": checks}
    with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for c in checks:
        if not c["pass"]:
            print(f"FAIL {c['check']}: want {c['want']} got {c['got']}")
    print(f"reconcile: {len(checks)} checks, {n_fail} fail, all_pass={n_fail == 0}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
