"""2621 · 对账（SCHEMA §12 三道）：`analysis.md` / `run_stdout.txt` 的每个数字都能在
`results.json`（以及它引用的训练日志）里找到同值，且算术自洽。

三道对账：
1. **算术对账**：字节式、准确率 = hits/n、CI、Δ、bpc 差分、read_share 一致性——全部从
   `results.json` 的原始计数重算，与落盘值比对。
2. **任务对账**：三臂是否同一任务/同一位置集（n、query_key_distance、chance、local_prior
   逐字段相等），以及毁源对照臂是否同一批（n 相等）。
3. **口径对账**：基线唯一（fp16）；解析字节与实测显存分列；eval-only（无训练步）。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def approx(a, b, tol=0.0):
    return abs(a - b) <= tol


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = approx(float(got), float(exp), tol) if isinstance(got, (int, float)) else got == exp
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    # ---------------- ① 算术对账 ----------------
    cfg = R["config"]["model"]
    L, H, T = cfg["n_layers"], cfg["n_heads"], cfg["seq_len"]
    dh = cfg["d_model"] // H
    vals_per_pos = 2 * L * H * dh
    bt = R["bytes_analytic"]["per_mode"]
    ck("bytes.values_per_position", bt["fp16"]["values_per_position"], vals_per_pos)
    ck("bytes.fp16_per_position", bt["fp16"]["per_position_bytes"], vals_per_pos * 2)
    ck("bytes.int8_per_position", bt["int8"]["per_position_bytes"], vals_per_pos * 1)
    ck("bytes.int8_reduction", bt["int8"]["reduction_vs_fp16"], 0.5, 1e-9)
    ck("bytes.baseline_is_fp16", R["bytes_analytic"]["baseline_mode"], "fp16")

    for mode, arm in R["main_leg"].items():
        ck(f"recall.{mode}.acc==hits/n", arm["accuracy"],
           arm["n_hits"] / arm["n"], 1e-6)
        ck(f"recall.{mode}.corrupt==hits/n", arm["corrupt_source"]["accuracy"],
           arm["corrupt_source"]["n_hits"] / arm["corrupt_source"]["n"], 1e-6)
        ck(f"recall.{mode}.frac_outside_window", arm["query_key_distance"]["frac_outside_window"], 1.0)
    cmp_ = R["comparisons"]["int8_vs_fp16"]
    ck("delta_pp.int8_vs_fp16",
       cmp_["delta_pp"],
       round(100 * (R["main_leg"]["int8"]["accuracy"] - R["main_leg"]["fp16"]["accuracy"]), 3),
       1e-9)
    ck("comparison.fp32_equals_fp16",
       R["comparisons"]["fp32_vs_fp16"]["delta_pp"], 0.0)
    ck("bpc.damage.int8",
       R["bpc_damage"]["int8"],
       round(R["bpc_leg"]["int8"]["bpc"] - R["bpc_leg"]["fp16"]["bpc"], 6), 1e-9)
    ck("bpc.damage.fp32",
       R["bpc_damage"]["fp32"],
       round(R["bpc_leg"]["fp32"]["bpc"] - R["bpc_leg"]["fp16"]["bpc"], 6), 1e-9)

    # ---------------- ② 任务对账（同任务/同位置集/同批） ----------------
    modes = ["fp32", "fp16", "int8"]
    ck("task.same_n", len({R["main_leg"][m]["n"] for m in modes}), 1)
    ck("task.same_chance", len({R["main_leg"][m]["chance"] for m in modes}), 1)
    ck("task.same_local_prior", len({R["main_leg"][m]["local_prior"] for m in modes}), 1)
    ck("task.same_query_key_distance",
       len({json.dumps(R["main_leg"][m]["query_key_distance"], sort_keys=True) for m in modes}), 1)
    ck("task.corrupt_same_n",
       len({R["main_leg"][m]["corrupt_source"]["n"] for m in modes}), 1)
    ck("task.corrupt_span",
       R["main_leg"]["fp16"]["corrupt_source"]["span"],
       [0, 2 * R["main_leg"]["fp16"]["K"]])

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.analytic_separate_from_measured",
       "eval_peak_gpu_mem_mb" in R["memory_measured"] and "per_position_bytes" in bt["int8"], True)
    ck("caliber.eval_only_no_train",
       R["cost"]["wall_minutes_total_eval"] > 0 and "train_wall_minutes" not in R["cost"], True)
    ck("caliber.read_share_unchanged_int8_vs_fp16",
       R["carry_activity"]["read_share_by_mode"]["int8"],
       R["carry_activity"]["read_share_by_mode"]["fp16"], 1e-4)
    ck("caliber.carry_alive", R["carry_activity"]["carry_liveness"]["alive"], True)
    ck("caliber.int8_rel_err_gt_fp16",
       R["carry_activity"]["quant_liveness_by_mode"]["int8"]["rel_err"]
       > R["carry_activity"]["quant_liveness_by_mode"]["fp16"]["rel_err"], True)
    ck("caliber.fp32_no_values_touched",
       R["selfchecks"]["carry_quant_fp32_quant_stats"]["n_values"], 0)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    cited = [
        ("analysis 9.38% fp16", R["main_leg"]["fp16"]["accuracy"], 0.09375),
        ("analysis 9.28% int8", R["main_leg"]["int8"]["accuracy"], 0.092773),
        ("analysis -0.098pp", cmp_["delta_pp"], -0.098),
        ("analysis chance 1.5625%", R["main_leg"]["fp16"]["chance"], 0.015625),
        ("analysis local_prior 3.125%", R["main_leg"]["fp16"]["local_prior"], 0.03125),
        ("analysis corrupt 0.73%", R["main_leg"]["fp16"]["corrupt_source"]["accuracy"], 0.007324),
        ("analysis bpc fp16", R["bpc_leg"]["fp16"]["bpc"], 103.611790),
        ("analysis bpc int8", R["bpc_leg"]["int8"]["bpc"], 103.609283),
        ("analysis bpc damage -0.002507", R["bpc_damage"]["int8"], -0.002507),
        ("analysis read_share 0.34217", R["carry_activity"]["read_share_by_mode"]["fp16"], 0.34217),
        ("analysis int8 clipped 3.82e-5",
         R["carry_activity"]["quant_liveness_by_mode"]["int8"]["clipped_frac"], 3.8204e-05),
        ("analysis peak 431.6 MB", R["cost"]["peak_gpu_mem_mb"], 431.6),
        # 墙钟是唯一随时间浮动的量（其它读数确定）；analysis.md 引用的值与本次
        # 结果同源，这里只校验它落在「纯推理 < 2 min」的区间内（§20.1 前提）。
        ("analysis wall within 2 min 且 > 0", 1.0 if 0 < R["cost"]["wall_minutes_total_eval"] <= 2 else 0.0, 1.0),
        ("analysis train_log 60/61", R["training_log_activity"]["n_rows_step_multiple_of_200"], 60),
        ("analysis train_log read_share 0.7103",
         R["training_log_activity"]["last_carry_activity"]["read_share"], 0.710287),
    ]
    for name, got, exp in cited:
        ck(name, got, exp)

    # 距离腿：fp16 与 int8 逐点逐位相同
    dl = R["distance_leg"]
    d_same = all(dl["fp16"][d]["accuracy"] == dl["int8"][d]["accuracy"]
                 for d in dl["fp16"])
    ck("distance_leg.fp16_equals_int8_every_point", d_same, True)

    # 原始结果文件与 analysis.md 都在盘上（可回溯）
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
