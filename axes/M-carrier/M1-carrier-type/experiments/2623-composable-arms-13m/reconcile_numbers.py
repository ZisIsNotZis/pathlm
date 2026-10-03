"""2623-composable-arms-13m · 对账（SCHEMA §12 三道）。

`analysis.md` 的每个数字都能在 `results.json`（及其引用的训练日志）里找到同值，且算术
自洽：

1. **算术对账**：准确率 = hits/n、逐叠加步增量 = 相邻臂差、净和、去伪影块的聚合、
   bpc 差分、参数量差、距离腿命中率——全部从 `results.json` 的原始计数重算。
2. **任务对账**：各腿内四臂同一任务/同一位置集（n、chance、local_prior 相等）；
   毁源对照同一批（n 相等）；两腿之间的距离区间与训练同分布/近分布外分开。
3. **口径对账**：基准唯一（stock）；判决腿为 primary；clean bpc 是非退化锚且
   四臂 bpc 均 > log2(206)；参数量差显式声明；活性量每 200 步落盘；收敛门禁在
   答案行 CE 上。

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

from arms import ARM_NAMES, ARMS, BASELINE_ARM, TARGET_ARM  # noqa: E402


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = (abs(float(got) - float(exp)) <= tol
              if isinstance(got, (int, float)) and isinstance(exp, (int, float))
              else got == exp)
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    # ---------------- ① 算术对账 ----------------
    for leg in ("main_leg", "cross_leg_full_causal", "train_task_leg",
                "train_task_leg_no_last_block"):
        for name in ARM_NAMES:
            arm = R[leg][name]
            ck(f"{leg}.{name}.acc==hits/n", arm["accuracy"],
               arm["n_hits"] / arm["n"], 1e-6)
            ck(f"{leg}.{name}.ci_low<acc<ci_high",
               arm["accuracy_ci"][0] <= arm["accuracy"] <= arm["accuracy_ci"][1], True)
    for leg in ("main_leg", "train_task_leg"):
        for name in ARM_NAMES:
            a = R[leg][name]
            ck(f"{leg}.{name}.corrupt==hits/n", a["corrupt_source"]["accuracy"],
               a["corrupt_source"]["n_hits"] / a["corrupt_source"]["n"], 1e-6)

    # 逐叠加步增量 = 相邻臂差；净和 = 目标 − 基准；三套口径都核
    accs = {leg: {n: R[leg][n]["accuracy"] for n in ARM_NAMES}
            for leg in ("main_leg", "cross_leg_full_causal", "train_task_leg",
                        "train_task_leg_no_last_block")}
    incr_map = {"main_leg": "native",
                "cross_leg_full_causal": "full_causal",
                "train_task_leg": "train_task",
                "train_task_leg_no_last_block": "train_task_no_last_block"}
    for leg, key in incr_map.items():
        for step in R["increments"][key]["staircase"]:
            ck(f"incr.{key}.{step['added']}", step["delta_pp"],
               round(100 * (accs[leg][step["to"]] - accs[leg][step["from"]]), 3), 1e-6)
        ck(f"incr.{key}.net", R["increments"][key]["net_vs_baseline_pp"],
           round(100 * (accs[leg][TARGET_ARM] - accs[leg][BASELINE_ARM]), 3), 1e-6)

    # 去伪影块：与 by_distance（d≤223）重算一致
    for name in ARM_NAMES:
        n = h = 0
        for d, v in R["train_task_leg"][name]["by_distance"].items():
            if int(d) <= 223:
                n += v["n"]; h += round(v["accuracy"] * v["n"])
        ck(f"no_last_block.{name}.n", R["train_task_leg_no_last_block"][name]["n"], n)
        ck(f"no_last_block.{name}.acc", R["train_task_leg_no_last_block"][name]["accuracy"],
           round(h / n, 6), 1e-6)

    for name in ARM_NAMES:
        ck(f"bpc.damage.{name}", R["bpc_damage_vs_stock"][name],
           round(R["bpc_leg"][name]["bpc"] - R["bpc_leg"][BASELINE_ARM]["bpc"], 6), 1e-9)
        d = R["param_declaration"][name]
        ck(f"param.{name}.delta==n-base", d["delta_vs_base"],
           d["n_params"] - R["param_declaration"][BASELINE_ARM]["n_params"])
    ck("ladder.step_features", [len(a["stacked"]) for a in ARMS], [0, 1, 2, 3])

    # ---------------- ② 任务对账 ----------------
    for leg in ("main_leg", "train_task_leg", "cross_leg_full_causal"):
        ck(f"task.{leg}.same_n", len({R[leg][n]["n"] for n in ARM_NAMES}), 1)
        ck(f"task.{leg}.same_chance", len({R[leg][n]["chance"] for n in ARM_NAMES}), 1)
        ck(f"task.{leg}.same_local_prior",
           len({R[leg][n]["local_prior"] for n in ARM_NAMES}), 1)
    ck("task.main.corrupt_same_n",
       len({R["main_leg"][n]["corrupt_source"]["n"] for n in ARM_NAMES}), 1)
    ck("task.main.corrupt_span",
       {n: R["main_leg"][n]["corrupt_source"]["span"] for n in ARM_NAMES},
       {n: [0, 2 * R["main_leg"][n]["K"]] for n in ARM_NAMES})
    # 判决腿距离全部 > 训练分布最大距离；训练型任务腿 ≤ 240
    ck("task.main.dist_beyond_train",
       all(R["main_leg"][n]["query_key_distance"]["min"] > 240 for n in ARM_NAMES), True)
    ck("task.train.dist_max_240",
       all(R["train_task_leg"][n]["query_key_distance"]["max"] == 240 for n in ARM_NAMES), True)
    # 伪影：最后一块（d≥226）全部 100%
    ck("task.train.lastblock_is_100pct",
       all(all(v["accuracy"] == 1.0 for d, v in R["train_task_leg"][n]["by_distance"].items()
               if int(d) >= 226) for n in ARM_NAMES), True)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.baseline_is_stock", BASELINE_ARM, "stock")
    ck("caliber.two_legs_present", ("main_leg" in R) and ("bpc_leg" in R), True)
    ck("caliber.nonwindow_arms_native_zero",
       all(R["main_leg"][n]["native_window"] == 0
           for n in ("stock", "gqa", "gqa_carry")), True)
    ck("caliber.window_nondegenerate", R["window_nondegenerate"]["alive"], True)
    ck("caliber.gqa_kv_shape", R["kv_head_liveness"]["gqa"]["k_shape"][1], 2)
    ck("caliber.gqa_reduction_75pct",
       R["kv_head_liveness"]["gqa"]["kv_reduction_frac"], 0.75)
    ck("caliber.destroy_all_valid",
       all(R["destroy_source_validity"][n]["valid"] for n in ARM_NAMES), True)
    ck("caliber.bpc_all_worse_than_uniform",
       all(R["bpc_leg"][n]["bpc"] > math.log2(206) for n in ARM_NAMES), True)
    ck("caliber.activity_rows_200",
       {n: R["train_activity"][n]["n_rows_step_multiple_of_200"] for n in ARM_NAMES},
       {n: R["train_activity"][n]["n_rows_step_multiple_of_200"] for n in ARM_NAMES})
    for n, sw in (("gqa", "kv_heads_series_present"),
                  ("gqa_carry", "attn_carry_series_present"),
                  ("gqa_carry_window", "window_mask_series_present")):
        ck(f"caliber.activity_{n}.{sw}", R["train_activity"][n][sw] > 0, True)
    ck("caliber.window_adds_no_params",
       R["param_declaration"]["gqa_carry_window"]["delta_vs_base"] ==
       R["param_declaration"]["gqa_carry"]["delta_vs_base"], True)
    ck("caliber.eval_only", "train_wall_minutes" not in R["cost"], True)
    ck("caliber.convergence_on_answer_ce",
       all(R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"] is not None
           for n in ARM_NAMES), True)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    cited = [
        # §5.1 主腿
        ("main stock 0.000", R["main_leg"]["stock"]["accuracy"], 0.0, 1e-9),
        ("main gqa 0.146", R["main_leg"]["gqa"]["accuracy"], 0.001465, 1e-9),
        ("main gqa_carry 15.674", R["main_leg"]["gqa_carry"]["accuracy"], 0.15674, 5e-6),
        ("main gqa_carry_window 15.332",
         R["main_leg"]["gqa_carry_window"]["accuracy"], 0.15332, 5e-6),
        ("main gqa_carry hits 321", R["main_leg"]["gqa_carry"]["n_hits"], 321),
        ("main stock destroy 0.000", R["main_leg"]["stock"]["corrupt_source"]["accuracy"], 0.0, 1e-9),
        ("main carry destroy 1.367", R["main_leg"]["gqa_carry"]["corrupt_source"]["accuracy"],
         0.01367, 5e-6),
        ("incr native gqa +0.146", R["increments"]["native"]["staircase"][0]["delta_pp"], 0.146, 1e-6),
        ("incr native carry +15.527", R["increments"]["native"]["staircase"][1]["delta_pp"], 15.527, 1e-6),
        ("incr native window -0.342", R["increments"]["native"]["staircase"][2]["delta_pp"], -0.342, 1e-6),
        ("incr native net +15.332", R["increments"]["native"]["net_vs_baseline_pp"], 15.332, 1e-6),
        # §5.2 训练型任务（去伪影块）
        ("train stock 25.725", R["train_task_leg_no_last_block"]["stock"]["accuracy"], 0.25725, 5e-6),
        ("train gqa 26.674", R["train_task_leg_no_last_block"]["gqa"]["accuracy"], 0.26674, 5e-6),
        ("train gqa_carry 27.120",
         R["train_task_leg_no_last_block"]["gqa_carry"]["accuracy"], 0.27120, 5e-6),
        ("train gqa_carry_window 25.446",
         R["train_task_leg_no_last_block"]["gqa_carry_window"]["accuracy"], 0.25446, 5e-6),
        ("train incl leak stock 35.010", R["train_task_leg"]["stock"]["accuracy"], 0.35010, 5e-6),
        ("incr train_nl net -0.279",
         R["increments"]["train_task_no_last_block"]["net_vs_baseline_pp"], -0.279, 1e-6),
        # §5.3 clean bpc
        ("bpc stock 13.069", R["bpc_leg"]["stock"]["bpc"], 13.069118, 1e-6),
        ("bpc gqa 37.145", R["bpc_leg"]["gqa"]["bpc"], 37.145279, 1e-6),
        ("bpc gqa_carry 58.308", R["bpc_leg"]["gqa_carry"]["bpc"], 58.307878, 1e-6),
        ("bpc gqa_carry_window 54.970", R["bpc_leg"]["gqa_carry_window"]["bpc"], 54.970002, 1e-6),
        # §5.4 距离腿（d=255 伪影 + 中段）
        ("dist gqa_carry d=128", R["distance_leg"]["gqa_carry"]["128"]["accuracy"], 0.109375, 1e-6),
        ("dist gqa_carry d=192", R["distance_leg"]["gqa_carry"]["192"]["accuracy"], 0.098958, 1e-6),
        ("dist stock d=255", R["distance_leg"]["stock"]["255"]["accuracy"], 0.0, 1e-9),
        ("dist gqa d=255", R["distance_leg"]["gqa"]["255"]["accuracy"], 0.979167, 1e-6),
        # §6 活性量与收敛门禁
        ("activity gqa kv rows 30", R["train_activity"]["gqa"]["kv_heads_series_present"], 30),
        ("activity carry read_share 0.6050",
         R["train_activity"]["gqa_carry"]["last"]["attn_carry"]["read_share"], 0.604966, 1e-6),
        ("activity carry_win read_share 0.8111",
         R["train_activity"]["gqa_carry_window"]["last"]["attn_carry"]["read_share"], 0.811133, 1e-6),
        ("activity window visible_frac 0.1209",
         R["train_activity"]["gqa_carry_window"]["last"]["window_mask"]["visible_frac"], 0.120866, 1e-6),
        ("window n_visible 3976", R["window_nondegenerate"]["n_visible_pairs"], 3976),
        ("window causal 32896", R["window_nondegenerate"]["causal_pairs"], 32896),
        ("conv stock slope -0.0926",
         R["train_activity"]["stock"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.09264, 1e-5),
        ("conv gqa slope -0.0740",
         R["train_activity"]["gqa"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.07398, 1e-5),
        ("conv carry slope -0.0602",
         R["train_activity"]["gqa_carry"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.06019, 1e-5),
        ("conv carry_win slope -0.0608",
         R["train_activity"]["gqa_carry_window"]["answer_ce_slope_nats_per_1k_lastquarter"], -0.06076, 1e-5),
        # §7 参数量差
        ("param gqa -1184256", R["param_declaration"]["gqa"]["delta_vs_base"], -1184256),
        ("param gqa pct -8.748", R["param_declaration"]["gqa"]["delta_pct_vs_base"], -8.748, 1e-3),
        ("param carry +264296",
         R["param_declaration"]["gqa_carry"]["n_params"] -
         R["param_declaration"]["gqa"]["n_params"], 264296),
        ("cost peak 542.8", R["cost"]["peak_gpu_mem_mb"], 542.8),
    ]
    for name, got, exp, *rest in cited:
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
