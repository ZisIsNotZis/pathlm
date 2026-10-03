"""2625 · 对账（SCHEMA §12 三道）：`analysis.md` 的每个数字都能在 `results.json`
（以及它引用的训练日志）里找到同值，且算术自洽。

三道对账：
1. **算术对账**：准确率 = hits/n、Wilson 95% CI 复算、极差/均值/sd 复算、毁源 = hits/n。
2. **任务对账**：每条腿各臂同一批位置（同 n / chance / local_prior / query_key_distance /
   毁源 n）；判决腿距离档 >240（近 OOD）、分布内腿 ≤240（§17.18）。
3. **口径对账**：四臂同块类/同参数量；fp32 档与释放版逐位相等；复用臂与训练臂的 sha256；
   窗口非退化；判据 ②（降级）与极差一致。
4. **analysis.md 引用的关键数字 → results.json**（逐条比对）。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit.carry_channel import wilson_ci  # noqa: E402

ARMS = ("gqa_carry_win_s0", "gqa_carry_win_s1", "gqa_carry_win_s2",
        "y1_gqa_carry_win_s0")
PRIMARY = ("gqa_carry_win_s0", "gqa_carry_win_s1", "gqa_carry_win_s2")
LEGS = ("main_leg", "train_task_leg", "cross_leg_full_causal")


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        if isinstance(got, (int, float)) and isinstance(exp, (int, float)):
            ok = abs(float(got) - float(exp)) <= tol
        else:
            ok = got == exp
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    # ---------------- ① 算术对账 ----------------
    for leg in LEGS:
        for a in ARMS:
            r = R[leg][a]
            ck(f"{leg}.{a}.acc==hits/n", r["accuracy"], r["n_hits"] / r["n"], 1e-6)
            ck(f"{leg}.{a}.ci", r["accuracy_ci"],
               wilson_ci(r["n_hits"], r["n"]), 1e-9)
            ck(f"{leg}.{a}.corrupt==hits/n", r["corrupt_source"]["accuracy"],
               r["corrupt_source"]["n_hits"] / r["corrupt_source"]["n"], 1e-6)
            ck(f"{leg}.{a}.corrupt.ci", r["corrupt_source"]["accuracy_ci"],
               wilson_ci(r["corrupt_source"]["n_hits"], r["corrupt_source"]["n"]), 1e-9)

    # 极差 / 均值 / sd 复算（判决腿 + 分布内腿 + 交叉腿；主集 n=3）
    for leg_key, variance_key in (("main_leg", "main_leg_variance"),
                                  ("train_task_leg", "train_task_leg_variance"),
                                  ("cross_leg_full_causal", "cross_leg_variance")):
        v = R[variance_key]
        vals = [R[leg_key][n]["accuracy"] for n in PRIMARY]
        ck(f"{variance_key}.names", v["names"], list(PRIMARY))
        ck(f"{variance_key}.range", v["range_pp"],
           round(100.0 * (max(vals) - min(vals)), 3), 1e-9)
        ck(f"{variance_key}.mean", v["mean_pp"],
           round(100.0 * sum(vals) / len(vals), 3), 1e-9)
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        ck(f"{variance_key}.sd", v["sd_pp"], round(100.0 * sd, 3), 1e-9)
        ck(f"{variance_key}.min", v["min_pp"], round(100.0 * min(vals), 3), 1e-9)
        ck(f"{variance_key}.max", v["max_pp"], round(100.0 * max(vals), 3), 1e-9)

    # ---------------- ② 任务对账（同批位置 / 距离档） ----------------
    for leg in LEGS:
        for field in ("n", "chance", "local_prior"):
            ck(f"task.{leg}.same_{field}",
               len({R[leg][a][field] for a in ARMS}), 1)
        ck(f"task.{leg}.same_qkd",
           len({json.dumps(R[leg][a]["query_key_distance"], sort_keys=True)
                for a in ARMS}), 1)
        ck(f"task.{leg}.same_corrupt_n",
           len({R[leg][a]["corrupt_source"]["n"] for a in ARMS}), 1)
        ck(f"task.{leg}.corrupt_span",
           R[leg][ARMS[0]]["corrupt_source"]["span"], [0, 2 * R[leg][ARMS[0]]["K"]])
    # §17.18：判决腿近 OOD（>240）、分布内腿 ≤240
    ck("task.main_leg_is_ood_over_240",
       R["main_leg"][ARMS[0]]["query_key_distance"]["min"] > 240, True)
    ck("task.main_leg_dist_max_255",
       R["main_leg"][ARMS[0]]["query_key_distance"]["max"], 255)
    ck("task.train_task_leg_within_240",
       R["train_task_leg"][ARMS[0]]["query_key_distance"]["max"] <= 240, True)
    ck("task.train_task_leg_dist_min_107",
       R["train_task_leg"][ARMS[0]]["query_key_distance"]["min"], 107)
    ck("task.main_leg_all_outside_window",
       all(R["main_leg"][a]["query_key_distance"]["frac_outside_window"] == 1.0
           for a in ARMS), True)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.all_composable",
       sorted({R["checkpoints"][a]["block_class"] for a in ARMS}), ["ComposableBlock"])
    ck("caliber.block_class_expected_matches",
       all(R["checkpoints"][a]["block_class"] == R["checkpoints"][a]["block_class_expected"]
           for a in ARMS), True)
    ck("caliber.same_n_params",
       len({R["checkpoints"][a]["n_params"] for a in ARMS}), 1)
    ck("caliber.n_params_12_616_808",
       R["checkpoints"][ARMS[0]]["n_params"], 12616808)
    ck("caliber.fp32_bitwise_all",
       all(R["instrument_selfcheck"]["fp32_bitwise"][a]["equal_to_released"]
           for a in ARMS), True)
    ck("caliber.rerun_identical",
       R["instrument_selfcheck"]["fp32_bitwise"]["rerun_identical"], True)
    ck("caliber.window_alive", R["window_nondegenerate"]["alive"], True)
    # sha256：训练臂与 config 回填一致；复用臂与臂定义的写死值一致
    for a in ("gqa_carry_win_s1", "gqa_carry_win_s2"):
        ck(f"caliber.trained_sha256.{a}", R["checkpoints"][a]["sha256"],
           R["config"]["checkpoint"]["sha256"][a])
    for a, want in (("gqa_carry_win_s0",
                     "02c390d8b1cb41a3f8c64923b45dcadeadcddc60a18b78477e533ad4020b0b09"),
                    ("y1_gqa_carry_win_s0",
                     "48c4e39ec058b8c509de88744e32f4169f993eb4f2680e6b4080b1b410f9c8a6")):
        ck(f"caliber.reused_sha256.{a}", R["checkpoints"][a]["sha256"], want)
    # 判据 ② 与极差一致
    v = R["verdict"]
    ck("caliber.downgrade_flag_consistent",
       v["downgrade_to_auxiliary"], v["range_pp"] >= v["effect_threshold_pp"])
    ck("caliber.threshold_3pp", v["effect_threshold_pp"], 3.0)
    ck("caliber.in_dist_flag_consistent",
       v["in_dist_leg"]["high_variance"],
       v["in_dist_leg"]["range_pp"] >= v["effect_threshold_pp"])
    ck("caliber.gqa_heads_2", R["kv_head_liveness"][ARMS[1]]["n_kv_heads"], 2)
    ck("caliber.eval_only", R["cost"]["pure_inference"], True)

    # ---------------- ④ analysis.md 引用的关键数字 → results.json ----------------
    CITED = [
        ("analysis 判决腿 s0/s1/s2 hits = 19/142/241",
         [R["main_leg"][a]["n_hits"] for a in PRIMARY], [19, 142, 241]),
        ("analysis 判决腿 s0/s1/s2 = 0.928/6.934/11.768 pp",
         [round(100.0 * R["main_leg"][a]["accuracy"], 3) for a in PRIMARY],
         [0.928, 6.934, 11.768]),
        ("analysis 判决腿极差 10.840 pp", v["range_pp"], 10.84),
        ("analysis 判决腿均值 6.543 pp", v["mean_pp"], 6.543),
        ("analysis 判决腿 sd 5.43 pp", v["sd_pp"], 5.43),
        ("analysis 参照臂 Y1 s0 = 15.332 pp",
         round(100.0 * R["main_leg"]["y1_gqa_carry_win_s0"]["accuracy"], 3), 15.332),
        ("analysis 含参照四跑极差 14.404 pp",
         R["main_leg_variance_reference_included"]["range_pp"], 14.404),
        ("analysis 含参照四跑均值 8.740 pp",
         R["main_leg_variance_reference_included"]["mean_pp"], 8.74),
        ("analysis 含参照四跑 sd 6.243 pp",
         R["main_leg_variance_reference_included"]["sd_pp"], 6.243),
        ("analysis 分布内含参照四跑极差 2.173 pp",
         round(100.0 * (max(R["train_task_leg"][a]["accuracy"] for a in ARMS) -
                        min(R["train_task_leg"][a]["accuracy"] for a in ARMS)), 3), 2.173),
        ("analysis 参照臂 y1_s0 CI = [13.837, 16.957] pp",
         [round(100.0 * R["main_leg"]["y1_gqa_carry_win_s0"]["accuracy_ci"][0], 3),
          round(100.0 * R["main_leg"]["y1_gqa_carry_win_s0"]["accuracy_ci"][1], 3)],
         [13.837, 16.957]),
        ("analysis 分布内 s0/s1/s2 = 35.718/36.939/36.206 pp",
         [round(100.0 * R["train_task_leg"][a]["accuracy"], 3) for a in PRIMARY],
         [35.718, 36.939, 36.206]),
        ("analysis 分布内极差 1.221 pp",
         R["train_task_leg_variance"]["range_pp"], 1.221),
        ("analysis 分布内 sd 0.614 pp",
         R["train_task_leg_variance"]["sd_pp"], 0.614),
        ("analysis 交叉全因果 s0/s1/s2 = 9.912/4.199/16.065 pp",
         [round(100.0 * R["cross_leg_full_causal"][a]["accuracy"], 3) for a in PRIMARY],
         [9.912, 4.199, 16.065]),
        ("analysis 交叉全因果极差 11.865 pp",
         R["cross_leg_variance"]["range_pp"], 11.865),
        ("analysis chance = 1/64 = 0.015625",
         R["main_leg"][ARMS[0]]["chance"], 0.015625),
        ("analysis 分布内 local_prior = 0.013672",
         R["train_task_leg"][ARMS[0]]["local_prior"], 0.013672),
        ("analysis 判决腿 s2 CI 上界 13.235 pp",
         round(100.0 * R["main_leg"]["gqa_carry_win_s2"]["accuracy_ci"][1], 3), 13.235),
        ("analysis 毁源 s0/s1/s2 = 0.098/0.586/1.270 pp",
         [round(100.0 * R["main_leg"][a]["corrupt_source"]["accuracy"], 3)
          for a in PRIMARY], [0.098, 0.586, 1.27]),
        ("analysis 训练 gnorm_max s1/s2 = 51.21/57.41",
         [R["train_activity"][a]["gnorm_max"] for a in ("gqa_carry_win_s1",
                                                        "gqa_carry_win_s2")],
         [51.21, 57.41]),
        ("analysis read_share s1/s2 = 0.863936/0.709234",
         [R["train_activity"][a]["last_read_share"] for a in ("gqa_carry_win_s1",
                                                              "gqa_carry_win_s2")],
         [0.863936, 0.709234]),
        ("analysis synth_answer_ce s1/s2 = 1.6407/1.6039",
         [R["train_activity"][a]["final_synth_answer_ce"] for a in ("gqa_carry_win_s1",
                                                                    "gqa_carry_win_s2")],
         [1.6407, 1.6039]),
        ("analysis 答案行 CE 斜率 s1/s2 = -0.07141/-0.06297",
         [R["train_activity"][a]["answer_ce_slope_nats_per_1k_lastquarter"]
          for a in ("gqa_carry_win_s1", "gqa_carry_win_s2")], [-0.07141, -0.06297]),
        ("analysis peak_gpu_mem_mb eval = 580.4", R["cost"]["peak_gpu_mem_mb"], 580.4),
        ("analysis 距离曲线 d=255 s0/s1/s2 = 6.25/46.88/56.25 pp",
         [round(100.0 * R["distance_leg"][a]["255"]["accuracy"], 2) for a in PRIMARY],
         [6.25, 46.88, 56.25]),
    ]
    for name, got, exp in CITED:
        ck(name, got, exp)

    ck("files.analysis_exists", os.path.exists(os.path.join(HERE, "analysis.md")), True)

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
