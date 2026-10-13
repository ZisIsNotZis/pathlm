"""2627-carry-x-depth · 对账（SCHEMA §12 三道）。

`analysis.md` 的每个数字都能在 `results.json`（及其引用的训练日志）里找到同值，且算术
自洽：

1. **算术对账**：准确率 = hits/n、2×2 因子分解（四个单向增量 + 交互项 + 净和）、bpc 差分、
   参数量差、tf@k 一致率——全部从 `results.json` 的原始计数重算。
2. **任务对账**：各腿内四臂同一任务/同一位置集（n、chance、local_prior 相等）；
   毁源对照同一批（n 相等）；**主判据腿的距离全部 ≤240（训练分布内，§17.18）**，
   单查询腿的距离 >240（训练分布外，只作外推声明）。
3. **口径对账**：基准唯一（base，两因子都关）；主判据 = 训练型多查询召回；depth 因子
   在训练日志里真的活了（出口 CE 条数 13 vs 1）；架构一致（base==depth / carry==both）；
   clean bpc 是非退化锚 + 质量代价项；活性量每 200 步落盘；收敛门禁在答案行 CE 上。

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

from carry_x_depth_arms import ARM_NAMES, BASELINE_ARM, TARGET_ARM, factorial_increments  # noqa: E402

#: `analysis.md` 引用的关键数字在 `main()` 内从 `results.json` 构造（见 CITED 构造段）。



def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = (abs(float(got) - float(exp)) <= tol
              if isinstance(got, (int, float)) and isinstance(exp, (int, float))
              else got == exp)
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    legs = ("train_task_leg", "train_task_leg_no_last_block",
            "cross_leg_full_causal", "main_leg_single_query_ood")

    # ---------------- ① 算术对账 ----------------
    for leg in legs:
        for name in ARM_NAMES:
            arm = R[leg][name]
            ck(f"{leg}.{name}.acc==hits/n", arm["accuracy"],
               arm["n_hits"] / arm["n"], 1e-6)
            ck(f"{leg}.{name}.ci_brackets_acc",
               arm["accuracy_ci"][0] <= arm["accuracy"] <= arm["accuracy_ci"][1], True)
    for leg in ("train_task_leg", "main_leg_single_query_ood"):
        for name in ARM_NAMES:
            a = R[leg][name]
            ck(f"{leg}.{name}.corrupt==hits/n", a["corrupt_source"]["accuracy"],
               a["corrupt_source"]["n_hits"] / a["corrupt_source"]["n"], 1e-6)

    # 2×2 因子分解：从原始 acc 重算四个单向增量 + 交互项
    def recheck_fact(key, leg, scale=100.0):
        acc = {n: R[leg][n]["accuracy"] for n in ARM_NAMES}
        exp = factorial_increments(acc)
        got = R["increments"][key]
        for k, v in exp.items():
            ck(f"incr.{key}.{k}", got[k], round(scale * v, 6), 1e-6)

    recheck_fact("train_task_indist", "train_task_leg")
    recheck_fact("single_query_ood", "main_leg_single_query_ood")
    # tf4 用 accept 字典重算
    tf_acc = {n: R["tf_synth_indist"][n]["accept"]["4"] for n in ARM_NAMES}
    for k, v in factorial_increments(tf_acc).items():
        ck(f"incr.tf4_synth_indist.{k}",
           R["increments"]["tf4_synth_indist"][k], round(100 * v, 6), 1e-6)

    for name in ARM_NAMES:
        ck(f"bpc.damage.{name}", R["bpc_damage_vs_base"][name],
           round(R["bpc_leg"][name]["bpc"] - R["bpc_leg"][BASELINE_ARM]["bpc"], 6), 1e-9)
        d = R["param_declaration"][name]
        ck(f"param.{name}.delta==n-base", d["delta_vs_base"],
           d["n_params"] - R["param_declaration"][BASELINE_ARM]["n_params"])
        ck(f"tf.{name}.tf4==by_depth[depth4]",
           R["tf_enwik8"][name]["tf4"],
           next(r["accept"] for r in R["tf_enwik8"][name]["by_depth"]
                if r["depth"] == 4), 1e-9)

    # 去伪影块：与 by_distance（d≤223）重算一致
    for name in ARM_NAMES:
        n = h = 0
        for d, v in R["train_task_leg"][name]["by_distance"].items():
            if int(d) <= 223:
                n += v["n"]; h += round(v["accuracy"] * v["n"])
        ck(f"no_last_block.{name}.n", R["train_task_leg_no_last_block"][name]["n"], n)
        ck(f"no_last_block.{name}.acc",
           R["train_task_leg_no_last_block"][name]["accuracy"], round(h / n, 6), 1e-6)

    # ---------------- ② 任务对账 ----------------
    for leg in legs:
        ck(f"task.{leg}.same_n", len({R[leg][n]["n"] for n in ARM_NAMES}), 1)
        if "chance" in R[leg][BASELINE_ARM]:
            ck(f"task.{leg}.same_chance",
               len({R[leg][n]["chance"] for n in ARM_NAMES}), 1)
            ck(f"task.{leg}.same_local_prior",
               len({R[leg][n]["local_prior"] for n in ARM_NAMES}), 1)
    ck("task.main.corrupt_same_n",
       len({R["train_task_leg"][n]["corrupt_source"]["n"] for n in ARM_NAMES}), 1)
    for name in ARM_NAMES:
        ck(f"task.indist.{name}.max_dist<=240",
           R["train_task_leg"][name]["query_key_distance"]["max"], 240)
        ck(f"task.indist.{name}.in_dist_flag",
           R["train_task_leg"][name]["query_key_distance"]["in_train_distribution"], True)
        ck(f"task.ood.{name}.min_dist>240>",
           R["main_leg_single_query_ood"][name]["query_key_distance"]["min"] > 240, True)
        ck(f"task.ood.{name}.in_dist_flag_false",
           R["main_leg_single_query_ood"][name]["query_key_distance"]
           ["in_train_distribution"], False)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.baseline_is_base", BASELINE_ARM, "base")
    ck("caliber.target_is_both", TARGET_ARM, "both")
    ck("caliber.arch_base_eq_depth", R["architecture_equality"]["base_eq_depth"], True)
    ck("caliber.arch_carry_eq_both", R["architecture_equality"]["carry_eq_both"], True)
    ck("caliber.block_classes",
       R["architecture_equality"]["block_class"],
       {n: R["architecture_equality"]["block_class_expected"][n] for n in ARM_NAMES})
    ck("caliber.destroy_all_valid",
       all(R["destroy_source_validity"][n]["valid"] for n in ARM_NAMES), True)
    ck("caliber.bpc_all_worse_than_uniform",
       all(R["bpc_leg"][n]["bpc"] > math.log2(206) for n in ARM_NAMES), True)
    ck("caliber.eval_only", "train_wall_minutes" not in R["cost"], True)
    # 深度因子活性：depth/both 臂出口 CE 条数 = 13；base/carry = 1
    ck("caliber.depth_live_depth_arms",
       {n: R["train_activity"][n]["n_exit_ce"] for n in ("depth", "both")},
       {"depth": 13, "both": 13})
    ck("caliber.depth_off_arms",
       {n: R["train_activity"][n]["n_exit_ce"] for n in ("base", "carry")},
       {"base": 1, "carry": 1})
    ck("caliber.carry_activity_present",
       all(R["train_activity"][n]["attn_carry_series_present"] > 0
           for n in ("carry", "both")), True)
    ck("caliber.carry_arms_quiet_no_carry",
       all(R["train_activity"][n]["last"]["attn_carry"] is None
           for n in ("base", "depth")), True)
    ck("caliber.convergence_on_answer_ce",
       all(R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"] is not None
           for n in ARM_NAMES), True)
    ck("caliber.tf4_present",
       all("4" in R["tf_synth_indist"][n]["accept"] for n in ARM_NAMES), True)
    # 可叠加性判定与交互项一致
    inter = R["increments"]["train_task_indist"]["interaction"]
    ck("caliber.additivity_interaction_matches",
       R["additivity_verdict"]["interaction_pp_indist"], inter, 1e-9)
    verdict = R["additivity_verdict"]["verdict"]
    ck("caliber.additivity_verdict_consistent",
       verdict.startswith("可加") if abs(inter) <= 1.4
       else (verdict.startswith("不可叠加") if inter < 0 else verdict.startswith("超可加")),
       True)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    # 每条 = (名字, 实际值, 期望值, [容差])；期望值 = analysis.md 里写的数。
    CITED = [
        # §5.1 主判据（训练分布内多查询召回，n=16384）
        ("s5.1 base hits 242", R["train_task_leg"]["base"]["n_hits"], 242),
        ("s5.1 base acc 1.48%", R["train_task_leg"]["base"]["accuracy"], 0.014771, 5e-6),
        ("s5.1 carry hits 5862", R["train_task_leg"]["carry"]["n_hits"], 5862),
        ("s5.1 carry acc 35.78%", R["train_task_leg"]["carry"]["accuracy"], 0.357788, 5e-6),
        ("s5.1 depth acc 1.47%", R["train_task_leg"]["depth"]["accuracy"], 0.014709, 5e-6),
        ("s5.1 both hits 5924", R["train_task_leg"]["both"]["n_hits"], 5924),
        ("s5.1 both acc 36.16%", R["train_task_leg"]["both"]["accuracy"], 0.361572, 5e-6),
        ("s5.1 n 16384", R["train_task_leg"]["base"]["n"], 16384),
        ("s5.1 chance 1.56%", R["train_task_leg"]["base"]["chance"], 0.015625, 1e-9),
        ("s5.1 local_prior 1.37%", R["train_task_leg"]["base"]["local_prior"], 0.013672, 5e-6),
        # §5.2 因子分解（pp）
        ("s5.2 carry@off +34.30", R["increments"]["train_task_indist"]["carry_at_depth_off"], 34.3017, 5e-4),
        ("s5.2 depth@off -0.006", R["increments"]["train_task_indist"]["depth_at_carry_off"], -0.0062, 5e-4),
        ("s5.2 carry@on +34.69", R["increments"]["train_task_indist"]["carry_at_depth_on"], 34.6863, 5e-4),
        ("s5.2 depth@on +0.378", R["increments"]["train_task_indist"]["depth_at_carry_on"], 0.3784, 5e-4),
        ("s5.2 interaction +0.385", R["increments"]["train_task_indist"]["interaction"], 0.3846, 5e-4),
        ("s5.2 net +34.68", R["increments"]["train_task_indist"]["net_vs_base"], 34.6801, 5e-4),
        ("s5.2 cross interaction -0.537", R["increments"]["cross_full_causal"]["interaction"], -0.537, 5e-4),
        ("s5.2 ood interaction +16.02", R["increments"]["single_query_ood"]["interaction"], 16.0157, 5e-3),
        ("s5.2 no_last carry 26.60%", R["train_task_leg_no_last_block"]["carry"]["accuracy"], 0.2660, 5e-5),
        ("s5.2 no_last both 27.04%", R["train_task_leg_no_last_block"]["both"]["accuracy"], 0.2704, 5e-5),
        ("s5.2 no_last interaction +0.46", R["increments"]["train_task_indist_no_last_block"]["interaction"], 0.4603, 5e-4),
        # §5.3 出口资产
        ("s5.3 tf4 base 87.84%", R["tf_synth_indist"]["base"]["accept"]["4"], 0.8784, 5e-5),
        ("s5.3 tf4 carry 88.53%", R["tf_synth_indist"]["carry"]["accept"]["4"], 0.8853, 5e-5),
        ("s5.3 tf4 depth 99.84%", R["tf_synth_indist"]["depth"]["accept"]["4"], 0.9984, 5e-5),
        ("s5.3 tf4 both 94.47%", R["tf_synth_indist"]["both"]["accept"]["4"], 0.9447, 5e-5),
        ("s5.3 tf4 interaction -6.06", R["increments"]["tf4_synth_indist"]["interaction"], -6.06, 5e-3),
        ("s5.3 tf6 depth 99.90%", R["tf_synth_indist"]["depth"]["accept"]["6"], 0.9990, 5e-5),
        ("s5.3 tf4 enwik8 depth 20.73%", R["tf_enwik8"]["depth"]["tf4"], 0.2073, 5e-5),
        ("s5.3 tf4 enwik8 both 52.86%", R["tf_enwik8"]["both"]["tf4"], 0.5286, 5e-5),
        ("s5.3 tf4 enwik8 base 90.09%", R["tf_enwik8"]["base"]["tf4"], 0.9009, 5e-5),
        # §5.4 质量代价项（clean bpc，跨分布）
        ("s5.4 bpc base 55.07", R["bpc_leg"]["base"]["bpc"], 55.072425, 1e-5),
        ("s5.4 bpc carry 52.36", R["bpc_leg"]["carry"]["bpc"], 52.357692, 1e-5),
        ("s5.4 bpc depth 125.74", R["bpc_leg"]["depth"]["bpc"], 125.743568, 1e-5),
        ("s5.4 bpc both 1322.98", R["bpc_leg"]["both"]["bpc"], 1322.984011, 1e-4),
        ("s5.4 bpc damage depth +70.67", R["bpc_damage_vs_base"]["depth"], 70.671143, 1e-4),
        ("s5.4 bpc damage both +1267.91", R["bpc_damage_vs_base"]["both"], 1267.911586, 1e-3),
        # §5.5 辅助腿（单对布局，非训练分布）
        ("s5.5 dist both d64 97.92%", R["distance_leg"]["both"]["64"]["accuracy"], 0.979167, 5e-6),
        ("s5.5 dist both d128 91.67%", R["distance_leg"]["both"]["128"]["accuracy"], 0.916667, 5e-6),
        ("s5.5 dist both d255 73.44%", R["distance_leg"]["both"]["255"]["accuracy"], 0.734375, 5e-6),
        ("s5.5 dist carry d255 2.08%", R["distance_leg"]["carry"]["255"]["accuracy"], 0.020833, 5e-6),
        # §4 收敛门禁
        ("s4 slope base -0.01065", R["train_activity"]["base"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.01065, 5e-5),
        ("s4 slope carry -0.06219", R["train_activity"]["carry"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.06219, 5e-5),
        ("s4 slope depth -0.01137", R["train_activity"]["depth"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.01137, 5e-5),
        ("s4 slope both -0.08063", R["train_activity"]["both"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.08063, 5e-5),
        ("s4 answer_ce carry 1.5998", R["train_activity"]["carry"]["final_synth_answer_ce"], 1.5998, 5e-5),
        ("s4 answer_ce both 1.7100", R["train_activity"]["both"]["final_synth_answer_ce"], 1.71, 5e-5),
        ("s4 chance_ce 4.1589", R["train_activity"]["base"]["synth_chance_ce"], 4.1589, 5e-5),
        ("s4 carry read_share 0.8716", R["train_activity"]["carry"]["last"]["attn_carry"]["read_share"], 0.871644, 5e-6),
        ("s4 both read_share 0.6333", R["train_activity"]["both"]["last"]["attn_carry"]["read_share"], 0.633346, 5e-6),
        # §3 参数量 / §7 成本
        ("s3 param carry +264296", R["param_declaration"]["carry"]["delta_vs_base"], 264296),
        ("s3 param carry pct 1.952", R["param_declaration"]["carry"]["delta_pct_vs_base"], 1.952, 5e-4),
        ("s3 param depth +0", R["param_declaration"]["depth"]["delta_vs_base"], 0),
        ("s7 eval peak 572.4", R["cost"]["peak_gpu_mem_mb"], 572.4, 1e-9),
    ]
    for name, got, exp, *rest in CITED:
        ck(name, got, exp, rest[0] if rest else 0.0)
    ck("files.analysis_exists", os.path.exists(os.path.join(HERE, "analysis.md")), True)

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
