"""2629-carry-ragged-projfill · 对账（SCHEMA §12 三道）。

`analysis.md` 的每个数字都能在 `results.json`（及其引用的训练日志）里找到同值，且算术
自洽：

1. **算术对账**：准确率 = hits/n、单因素增量（§17.17）、bpc 差分、参数量差、tf@k 一致率、
   去伪影块——全部从 `results.json` 的原始计数重算。
2. **任务对账**：各腿内两臂同一任务/同一位置集（n、chance、local_prior 相等）；
   毁源对照同一批（n 相等）；**主判据腿 `in_distribution_random_exit` 的距离全部 ≤240
   （训练分布内，§17.18）**，单查询腿的距离 >240（训练分布外，只作外推声明）。
3. **口径对账**：基准唯一（`projfill`，唯一变量 = carry）；主判据 = 训练分布内多查询召回；
   两臂都是非 dense + proj-fill（形态件）；块类与参数量声明一致；carry 臂活性量非 null、
   无 carry 臂为 null；clean bpc 是非退化锚（> log2 206）；收敛门禁在答案行 CE 上；
   出口资产保持性判定与 Δ 一致。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from ragged_projfill_arms import (ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                                  single_factor_increment)

RAND = "in_distribution_random_exit"
FULL = "full_depth_deployment"


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = (abs(float(got) - float(exp)) <= tol
              if isinstance(got, (int, float)) and isinstance(exp, (int, float))
              else got == exp)
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    modes = (RAND, FULL)

    # ---------------- ① 算术对账 ----------------
    for mode in modes:
        for name in ARM_NAMES:
            arm = R["train_task_leg"][mode][name]
            ck(f"main.{mode}.{name}.acc==hits/n", arm["accuracy"],
               arm["n_hits"] / arm["n"], 1e-6)
            ck(f"main.{mode}.{name}.ci_brackets",
               arm["accuracy_ci"][0] <= arm["accuracy"] <= arm["accuracy_ci"][1], True)
            a = arm["corrupt_source"]
            ck(f"main.{mode}.{name}.corrupt==hits/n", a["accuracy"],
               a["n_hits"] / a["n"], 1e-6)
        for name in ARM_NAMES:
            arm = R["main_leg_single_query_ood"][mode][name]
            ck(f"ood.{mode}.{name}.acc==hits/n", arm["accuracy"],
               arm["n_hits"] / arm["n"], 1e-6)

    # 单因素增量：从原始 acc 重算
    def recheck_inc(key, leg, scale=100.0):
        acc = {n: leg[n]["accuracy"] for n in ARM_NAMES}
        exp = single_factor_increment(acc)
        got = R["increments"][key]
        for k, v in exp.items():
            ck(f"incr.{key}.{k}", got[k], round(scale * v, 6), 1e-6)

    recheck_inc("recall_random_exit", R["train_task_leg"][RAND])
    recheck_inc("recall_full_depth", R["train_task_leg"][FULL])
    recheck_inc("single_query_ood", R["main_leg_single_query_ood"][FULL])
    recheck_inc("recall_random_exit_no_last_block",
                R["train_task_leg_no_last_block"][RAND])
    tf_acc = {n: R["tf_synth_indist"][n]["accept"]["4"] for n in ARM_NAMES}
    for k, v in single_factor_increment(tf_acc).items():
        ck(f"incr.tf4_synth_indist.{k}", R["increments"]["tf4_synth_indist"][k],
           round(100 * v, 6), 1e-6)
    tf_e = {n: R["tf_enwik8"][n]["tf4"] for n in ARM_NAMES}
    for k, v in single_factor_increment(tf_e).items():
        ck(f"incr.tf4_enwik8.{k}", R["increments"]["tf4_enwik8"][k],
           round(100 * v, 6), 1e-6)
    for name in ARM_NAMES:
        ck(f"bpc.damage.{name}", R["bpc_damage_vs_base"][name],
           round(R["bpc_leg"][name]["bpc"] - R["bpc_leg"][BASELINE_ARM]["bpc"], 6), 1e-9)
        ck(f"bpc.incr.{name}", R["increments"]["bpc"]["carry_effect"],
           round(R["bpc_leg"][TARGET_ARM]["bpc"] - R["bpc_leg"][BASELINE_ARM]["bpc"], 6),
           1e-9)
        d = R["param_declaration"][name]
        ck(f"param.{name}.delta==n-base", d["delta_vs_base"],
           d["n_params"] - R["param_declaration"][BASELINE_ARM]["n_params"])
        ck(f"tf.{name}.tf4==by_depth[depth4]",
           R["tf_enwik8"][name]["tf4"],
           next(r["accept"] for r in R["tf_enwik8"][name]["by_depth"]
                if r["depth"] == 4), 1e-9)

    # 去伪影块：与 by_distance（d≤223）重算一致
    for mode in modes:
        for name in ARM_NAMES:
            n = h = 0
            for d, v in R["train_task_leg"][mode][name]["by_distance"].items():
                if int(d) <= 223:
                    n += v["n"]; h += round(v["accuracy"] * v["n"])
            ck(f"no_last.{mode}.{name}.n",
               R["train_task_leg_no_last_block"][mode][name]["n"], n)
            ck(f"no_last.{mode}.{name}.acc",
               R["train_task_leg_no_last_block"][mode][name]["accuracy"],
               round(h / n, 6), 1e-6)

    # ---------------- ② 任务对账 ----------------
    for mode in modes:
        ck(f"task.{mode}.same_n",
           len({R["train_task_leg"][mode][n]["n"] for n in ARM_NAMES}), 1)
        ck(f"task.{mode}.same_chance",
           len({R["train_task_leg"][mode][n]["chance"] for n in ARM_NAMES}), 1)
        ck(f"task.{mode}.same_local_prior",
           len({R["train_task_leg"][mode][n]["local_prior"] for n in ARM_NAMES}), 1)
        ck(f"task.{mode}.corrupt_same_n",
           len({R["train_task_leg"][mode][n]["corrupt_source"]["n"]
                for n in ARM_NAMES}), 1)
        ck(f"task.{mode}.ood_same_n",
           len({R["main_leg_single_query_ood"][mode][n]["n"] for n in ARM_NAMES}), 1)
        for name in ARM_NAMES:
            ck(f"task.indist.{mode}.{name}.max_dist<=240",
               R["train_task_leg"][mode][name]["query_key_distance"]["max"], 240)
            ck(f"task.indist.{mode}.{name}.in_dist_flag",
               R["train_task_leg"][mode][name]["query_key_distance"]
               ["in_train_distribution"], True)
            ck(f"task.ood.{mode}.{name}.min_dist>240",
               R["main_leg_single_query_ood"][mode][name]["distance_range"][0] > 240,
               True)
            ck(f"task.ood.{mode}.{name}.in_dist_flag_false",
               R["main_leg_single_query_ood"][mode][name]["in_train_distribution"],
               False)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.baseline", BASELINE_ARM, "projfill")
    ck("caliber.target", TARGET_ARM, "projfill_carry")
    aeq = R["architecture_equality"]
    ck("caliber.block_classes",
       aeq["block_class"], aeq["block_class_expected"])
    ck("caliber.both_non_dense", all(v is False for v in aeq["dense"].values()), True)
    ck("caliber.both_proj_fill", all(v is True for v in aeq["proj_fill"].values()), True)
    ck("caliber.both_fill_kv", all(v is True for v in aeq["fill_kv"].values()), True)
    ck("caliber.param_delta_positive",
       R["param_declaration"][TARGET_ARM]["delta_vs_base"] > 0, True)
    ck("caliber.destroy_all_valid",
       all(R["destroy_source_validity"][m][n]["valid"] for m in modes
           for n in ARM_NAMES), True)
    ck("caliber.bpc_all_above_uniform",
       all(R["bpc_leg"][n]["bpc"] > math.log2(206) for n in ARM_NAMES), True)
    ck("caliber.eval_only", R["cost"]["pure_inference"], True)
    ck("caliber.carry_activity_present",
       R["train_activity"][TARGET_ARM]["attn_carry_series_present"] > 0, True)
    ck("caliber.nocarry_arms_quiet",
       R["train_activity"][BASELINE_ARM]["last_attn_carry"] is None, True)
    ck("caliber.convergence_on_answer_ce",
       all(R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"] is not None
           for n in ARM_NAMES), True)
    ck("caliber.tf4_present",
       all("4" in R["tf_synth_indist"][n]["accept"] for n in ARM_NAMES), True)
    # 出口资产保持性判定与两个口径的 Δ 一致
    eh = R["exit_asset_holds"]
    ck("caliber.exit_delta_indist", eh["delta_pp_indist"],
       round(100 * (eh["tf4_indist_carry"] - eh["tf4_indist_base"]), 4), 1e-6)
    ck("caliber.exit_delta_enwik8", eh["delta_pp_enwik8"],
       round(100 * (eh["tf4_enwik8_carry"] - eh["tf4_enwik8_base"]), 4), 1e-6)
    ck("caliber.exit_holds_indist_flag", eh["holds_indist"],
       abs(eh["delta_pp_indist"]) <= 1.4)
    ck("caliber.exit_holds_enwik8_flag", eh["holds_enwik8_proxy"],
       abs(eh["delta_pp_enwik8"]) <= 1.4)
    ck("caliber.exit_holds_primary", eh["holds"], eh["holds_indist"])
    ck("caliber.exit_reproduces_flag", eh["reproduces_100m_magnitude"],
       min(eh["tf4_indist_base"], eh["tf4_indist_carry"]) >= 0.90)
    ck("caliber.exit_indist_matches_tf_synth",
       eh["tf4_indist_base"], R["tf_synth_indist"][BASELINE_ARM]["accept"]["4"],
       1e-9)
    ck("caliber.exit_enwik8_matches_tf_enwik8",
       eh["tf4_enwik8_carry"], R["tf_enwik8"][TARGET_ARM]["tf4"], 1e-9)
    ck("caliber.capability_buildable", R["capability_probe"]["combo_buildable"], True)
    ck("caliber.capability_trunk_identical",
       R["capability_probe"]["trunk_bit_identical"], True)
    ck("files.analysis_exists", os.path.exists(os.path.join(HERE, "analysis.md")), True)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    # 每条 = (名字, 实际值, 期望值, [容差])；期望值 = analysis.md 里写的数。
    RAND_H = R["train_task_leg"][RAND]
    FULL_H = R["train_task_leg"][FULL]
    NBL = R["train_task_leg_no_last_block"][RAND]
    TFS = R["tf_synth_indist"]
    TFE = R["tf_enwik8"]
    OODF = R["main_leg_single_query_ood"][FULL]
    OODR = R["main_leg_single_query_ood"][RAND]
    ACT = R["train_activity"]
    CITED = [
        # §3 参数量 / 活性
        ("s3 param base 14,326,272", R["param_declaration"][BASELINE_ARM]["n_params"], 14326272),
        ("s3 param carry 14,590,568", R["param_declaration"][TARGET_ARM]["n_params"], 14590568),
        ("s3 param delta 264296", R["param_declaration"][TARGET_ARM]["delta_vs_base"], 264296),
        ("s3 param pct 1.845", R["param_declaration"][TARGET_ARM]["delta_pct_vs_base"], 1.845, 5e-4),
        ("s3 read_share 0.5417", ACT[TARGET_ARM]["last_attn_carry"]["read_share"], 0.541741, 5e-6),
        ("s3 read_scale 0.0592", ACT[TARGET_ARM]["last_attn_carry"]["read_scale_absmean"], 0.059182, 5e-6),
        ("s3 carry series 30", ACT[TARGET_ARM]["attn_carry_series_present"], 30),
        ("s3 nocarry quiet", ACT[BASELINE_ARM]["last_attn_carry"], None),
        # §4 收敛门禁
        ("s4 slope base -0.23881", ACT[BASELINE_ARM]["answer_ce_slope_nats_per_1k_lastquarter"], -0.23881, 5e-5),
        ("s4 slope carry -0.18292", ACT[TARGET_ARM]["answer_ce_slope_nats_per_1k_lastquarter"], -0.18292, 5e-5),
        ("s4 ans_ce base 3.0723", ACT[BASELINE_ARM]["final_synth_answer_ce"], 3.0723, 5e-5),
        ("s4 ans_ce carry 2.6186", ACT[TARGET_ARM]["final_synth_answer_ce"], 2.6186, 5e-5),
        ("s4 chance_ce 4.1589", ACT[BASELINE_ARM]["synth_chance_ce"], 4.1589, 5e-5),
        # §5.1 主判据
        ("s5.1 base hits 5340", RAND_H[BASELINE_ARM]["n_hits"], 5340),
        ("s5.1 base acc 32.59%", RAND_H[BASELINE_ARM]["accuracy"], 0.325928, 5e-6),
        ("s5.1 carry hits 5399", RAND_H[TARGET_ARM]["n_hits"], 5399),
        ("s5.1 carry acc 32.95%", RAND_H[TARGET_ARM]["accuracy"], 0.329529, 5e-6),
        ("s5.1 n 16384", RAND_H[BASELINE_ARM]["n"], 16384),
        ("s5.1 chance 1.56%", RAND_H[BASELINE_ARM]["chance"], 0.015625, 1e-9),
        ("s5.1 lp 1.37%", RAND_H[BASELINE_ARM]["local_prior"], 0.013672, 5e-6),
        ("s5.1 corrupt base 1.43%", RAND_H[BASELINE_ARM]["corrupt_source"]["accuracy"], 0.014282, 5e-6),
        ("s5.1 corrupt carry 1.51%", RAND_H[TARGET_ARM]["corrupt_source"]["accuracy"], 0.015137, 5e-6),
        ("s5.1 diff 0.36pp", R["increments"]["recall_random_exit"]["carry_effect"], 0.3601, 5e-4),
        ("s5.1 p 0.487", R["increments"]["pairwise_recall_random_exit"]["p_value"], 0.48744347, 5e-6),
        ("s5.1 nbl base 24.89%", NBL[BASELINE_ARM]["accuracy"], 0.248884, 5e-6),
        ("s5.1 nbl carry 24.50%", NBL[TARGET_ARM]["accuracy"], 0.244978, 5e-6),
        # §5.2 全深度
        ("s5.2 full base 35.28%", FULL_H[BASELINE_ARM]["accuracy"], 0.352844, 5e-6),
        ("s5.2 full carry 35.78%", FULL_H[TARGET_ARM]["accuracy"], 0.357788, 5e-6),
        ("s5.2 full diff 0.49pp", R["increments"]["recall_full_depth"]["carry_effect"], 0.4944, 5e-4),
        # §5.3 出口资产
        ("s5.3 tf4 synth base 99.66%", TFS[BASELINE_ARM]["accept"]["4"], 0.9966, 5e-5),
        ("s5.3 tf4 synth carry 98.68%", TFS[TARGET_ARM]["accept"]["4"], 0.9868, 5e-5),
        ("s5.3 tf6 synth base 99.73%", TFS[BASELINE_ARM]["accept"]["6"], 0.9973, 5e-5),
        ("s5.3 tf6 synth carry 99.13%", TFS[TARGET_ARM]["accept"]["6"], 0.9913, 5e-5),
        ("s5.3 tf4 enwik8 base 93.69%", TFE[BASELINE_ARM]["tf4"], 0.9369, 5e-5),
        ("s5.3 tf4 enwik8 carry 88.98%", TFE[TARGET_ARM]["tf4"], 0.8898, 5e-5),
        ("s5.3 tf6 enwik8 base 94.98%", TFE[BASELINE_ARM]["tf6"], 0.9498, 5e-5),
        ("s5.3 tf6 enwik8 carry 94.45%", TFE[TARGET_ARM]["tf6"], 0.9445, 5e-5),
        ("s5.3 delta_indist -0.98", R["exit_asset_holds"]["delta_pp_indist"], -0.98, 5e-5),
        ("s5.3 delta_enwik8 -4.71", R["exit_asset_holds"]["delta_pp_enwik8"], -4.71, 5e-5),
        # §5.4 clean bpc（跨分布）
        ("s5.4 bpc base 15.978", R["bpc_leg"][BASELINE_ARM]["bpc"], 15.977669, 5e-5),
        ("s5.4 bpc carry 46.333", R["bpc_leg"][TARGET_ARM]["bpc"], 46.333189, 5e-5),
        # §5.5 越界腿（外推参照）
        ("s5.5 ood rand base 2.73%", OODR[BASELINE_ARM]["accuracy"], 0.027344, 5e-6),
        ("s5.5 ood rand carry 8.64%", OODR[TARGET_ARM]["accuracy"], 0.086426, 5e-6),
        ("s5.5 ood full base 2.49%", OODF[BASELINE_ARM]["accuracy"], 0.024902, 5e-6),
        ("s5.5 ood full carry 8.54%", OODF[TARGET_ARM]["accuracy"], 0.085449, 5e-6),
        ("s5.5 ood diff 6.05pp", R["increments"]["single_query_ood"]["carry_effect"], 6.0547, 5e-3),
        # §8 成本
        ("s8 eval peak 786.6", R["cost"]["peak_gpu_mem_mb"], 786.6, 1e-9),
        ("s8 eval wall 0.403", R["cost"]["wall_minutes_total_eval"], 0.403, 1e-9),
    ]
    for name, got, exp, *rest in CITED:
        ck(name, got, exp, rest[0] if rest else 0.0)

    n_fail = sum(1 for c in checks if not c["pass"])
    report = {"n_checks": len(checks), "n_fail": n_fail,
              "all_pass": n_fail == 0, "checks": checks}
    with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for c in checks:
        if not c["pass"]:
            print(f"FAIL {c['check']}: want {c['want']} got {c['got']}")
    print(f"reconcile: {len(checks)} checks, {n_fail} fail, all_pass={n_fail == 0}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
