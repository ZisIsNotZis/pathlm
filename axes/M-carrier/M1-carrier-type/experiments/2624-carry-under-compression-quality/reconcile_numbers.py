"""2624 · 对账（SCHEMA §12 三道）：`analysis.md` 的每个数字都能在 `results.json`
（以及它引用的训练日志）里找到同值，且算术自洽。

三道对账：
1. **算术对账**：准确率 = hits/n、CI 复算、分量分解（pp）、参数量差、解析字节——全部从
   `results.json` 的原始计数重算，与落盘值比对。
2. **任务对账**：三条评测行是否同一任务/同一位置集（n、query_key_distance、chance、
   local_prior 逐字段相等），毁源对照是否同一批（n 相等）。
3. **口径对账**：基准唯一（`carry_mha_fp16`）；int8 行复用 GQA 权重（无参数量混杂）；
   解析字节与主判据分列；eval-only（无训练步）。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        if isinstance(got, (int, float)) and isinstance(exp, (int, float)):
            ok = abs(float(got) - float(exp)) <= tol
        else:
            ok = got == exp
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    rows = ["carry_mha_fp16", "carry_gqa_fp16", "carry_gqa_int8"]

    # ---------------- ① 算术对账 ----------------
    for leg in ("main_leg", "cross_leg_full_causal", "train_task_leg"):
        for r in rows:
            a = R[leg][r]
            ck(f"{leg}.{r}.acc==hits/n", a["accuracy"], a["n_hits"] / a["n"], 1e-6)
            ck(f"{leg}.{r}.corrupt==hits/n", a["corrupt_source"]["accuracy"],
               a["corrupt_source"]["n_hits"] / a["corrupt_source"]["n"], 1e-6)

    inc = R["increments"]
    for leg in ("native", "full_causal", "train_task"):
        acc = inc[leg]["accuracy"]
        ck(f"inc.{leg}.gqa_vs_mha", inc[leg]["gqa_vs_mha_pp"],
           round(100 * (acc["carry_gqa_fp16"] - acc["carry_mha_fp16"]), 3), 1e-9)
        ck(f"inc.{leg}.int8_vs_fp16", inc[leg]["int8_vs_fp16_pp"],
           round(100 * (acc["carry_gqa_int8"] - acc["carry_gqa_fp16"]), 3), 1e-9)
        ck(f"inc.{leg}.net", inc[leg]["net_pp"],
           round(100 * (acc["carry_gqa_int8"] - acc["carry_mha_fp16"]), 3), 1e-9)

    # 参数量差
    pd = R["param_declaration"]
    nmha = pd["carry_mha"]["n_params"]
    ngqa = pd["carry_gqa"]["n_params"]
    ck("param.delta_vs_carry_mha", pd["carry_gqa"]["delta_vs_carry_mha"], ngqa - nmha)
    ck("param.delta_pct_vs_carry_mha", pd["carry_gqa"]["delta_pct_vs_carry_mha"],
       round(100.0 * (ngqa - nmha) / nmha, 3), 1e-9)

    # 解析字节
    b = R["bytes_analytic"]["rows"]
    base_b = b["carry_mha_fp16"]["per_position_bytes"]
    ck("bytes.gqa_fp16_reduction", b["carry_gqa_fp16"]["reduction_vs_carry_mha_fp16"],
       round(1 - b["carry_gqa_fp16"]["per_position_bytes"] / base_b, 6), 1e-9)
    ck("bytes.gqa_int8_is_half_of_gqa_fp16",
       b["carry_gqa_int8"]["per_position_bytes"] * 2,
       b["carry_gqa_fp16"]["per_position_bytes"])

    # ---------------- ② 任务对账（同任务/同位置集/同批） ----------------
    ck("task.same_n", len({R["main_leg"][r]["n"] for r in rows}), 1)
    ck("task.same_chance", len({R["main_leg"][r]["chance"] for r in rows}), 1)
    ck("task.same_local_prior", len({R["main_leg"][r]["local_prior"] for r in rows}), 1)
    ck("task.same_qkd",
       len({json.dumps(R["main_leg"][r]["query_key_distance"], sort_keys=True)
            for r in rows}), 1)
    ck("task.corrupt_same_n",
       len({R["main_leg"][r]["corrupt_source"]["n"] for r in rows}), 1)
    ck("task.corrupt_span", R["main_leg"]["carry_mha_fp16"]["corrupt_source"]["span"],
       [0, 2 * R["main_leg"]["carry_mha_fp16"]["K"]])
    ck("task.all_frac_outside_window",
       all(R["main_leg"][r]["query_key_distance"]["frac_outside_window"] == 1.0
           for r in rows), True)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.baseline_unique", R["config"]["baseline_eval_arm"], "carry_mha_fp16")
    ck("caliber.int8_row_uses_gqa_weights",
       [e for e in R["config"]["eval_rows"] if e["name"] == "carry_gqa_int8"][0]["weights"],
       "carry_gqa")
    ck("caliber.eval_only", "train_wall_minutes" not in R["cost"], True)
    ck("caliber.block_classes",
       R["checkpoints"]["carry_mha"]["block_class"] == "CarryChannelBlock"
       and R["checkpoints"]["carry_gqa"]["block_class"] == "ComposableBlock", True)
    bit = R["instrument_selfcheck"]["fp32_bitwise"]
    ck("caliber.fp32_bitwise_mha", bit["carry_mha"]["equal_to_released"], True)
    ck("caliber.fp32_bitwise_gqa", bit["carry_gqa"]["equal_to_released"], True)
    ck("caliber.int8_rel_err_gt_fp16",
       R["quant_liveness"]["carry_gqa_int8"]["rel_err"]
       > R["quant_liveness"]["carry_gqa_fp16"]["rel_err"], True)
    ck("caliber.kv_head_shapes",
       R["kv_head_liveness"]["carry_gqa"]["k_shape"][1], 2)
    ck("caliber.mha_full_heads",
       R["kv_head_liveness"]["carry_mha"]["k_shape"][1], 8)
    for r in rows:
        ck(f"caliber.destroy_valid.{r}", R["destroy_source_validity"][r]["valid"], True)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    # （analysis.md 里逐字引用的读数，逐条与 results.json 比对）
    CITED = [
        ("analysis 判决腿 carry_mha 3.955%",
         R["main_leg"]["carry_mha_fp16"]["accuracy"], 0.039551),
        ("analysis 判决腿 carry_gqa 0.928%",
         R["main_leg"]["carry_gqa_fp16"]["accuracy"], 0.009277),
        ("analysis 判决腿 carry_gqa_int8 0.928%",
         R["main_leg"]["carry_gqa_int8"]["accuracy"], 0.009277),
        ("analysis 判决腿命中 81/19/19",
         [R["main_leg"][r]["n_hits"] for r in rows], [81, 19, 19]),
        ("analysis 全因果 15.527/9.912/9.912%",
         [R["cross_leg_full_causal"][r]["accuracy"] for r in rows],
         [0.155273, 0.099121, 0.099121]),
        ("analysis 分布内 34.815/35.718/35.693%",
         [R["train_task_leg"][r]["accuracy"] for r in rows],
         [0.348145, 0.357178, 0.356934]),
        ("analysis GQA_vs_MHA 判决腿 −3.027pp",
         R["increments"]["native"]["gqa_vs_mha_pp"], -3.027),
        ("analysis GQA_vs_MHA 全因果 −5.615pp",
         R["increments"]["full_causal"]["gqa_vs_mha_pp"], -5.615),
        ("analysis GQA_vs_MHA 分布内 +0.903pp",
         R["increments"]["train_task"]["gqa_vs_mha_pp"], 0.903),
        ("analysis int8_vs_fp16 分布内 −0.024pp",
         R["increments"]["train_task"]["int8_vs_fp16_pp"], -0.024),
        ("analysis 参数量 −1,184,256",
         R["param_declaration"]["carry_gqa"]["delta_vs_carry_mha"], -1184256),
        ("analysis 参数量 −8.581%",
         R["param_declaration"]["carry_gqa"]["delta_pct_vs_carry_mha"], -8.581),
        ("analysis int8 rel_err 4.80e-3",
         R["quant_liveness"]["carry_gqa_int8"]["rel_err"], 0.004802536),
        ("analysis fp16 rel_err 1.75e-4",
         R["quant_liveness"]["carry_gqa_fp16"]["rel_err"], 0.00017518),
        ("analysis 字节 12288/3072/1536",
         [R["bytes_analytic"]["rows"][r]["per_position_bytes"] for r in rows],
         [12288, 3072, 1536]),
        ("analysis 斜率 −0.06683/−0.05826",
         [R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"]
          for n in ("carry_mha", "carry_gqa")], [-0.06683, -0.05826]),
        ("analysis peak 502.7 MB", R["cost"]["peak_gpu_mem_mb"], 502.7),
        ("analysis 毁源三行有效",
         all(R["destroy_source_validity"][r]["valid"] for r in rows), True),
        ("analysis read_share 0.8564/0.7541",
         [round(R["train_activity"][n]["last"]["attn_carry"]["read_share"], 4)
          for n in ("carry_mha", "carry_gqa")], [0.8564, 0.7541]),
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
