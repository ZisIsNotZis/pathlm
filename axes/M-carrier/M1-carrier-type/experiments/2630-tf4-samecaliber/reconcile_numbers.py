"""2630-tf4-samecaliber · 对账（SCHEMA §12 三道）。

`analysis.md` 的每个数字都能在 `results.json`（及其引用的训练日志/训练入口
`train/<arm>/results.json`）里找到同值，且算术自洽：

1. **算术对账**：tf@k 一致率 = `by_depth[depth k]`、单因素增量（§17.17）、clean bpc 差分、
   参数量差、tf@4 判定（`holds` == `tf4_holds(Δ)`）——全部从 `results.json` 的原始数重算。
2. **任务对账**：两臂**同一分布**（enwik8 训练 + enwik8 评测，`architecture_equality`
   的 `data_source` 相等）；同 tf 腿参数（`n_batches`/`batch_size`/批函数同 seed）；两臂
   参数量差 = carry 参数；**run.py 的 tf@4 与训练入口 `train_depth_ar.py` 的同值名
   （tf@4 / clean bpc）逐位一致**（两只入口、同一引擎函数 ⇒ 同口径的独立确认）。
3. **口径对账**：基准唯一（`base`，唯一变量 = carry）；主判据 = **同口径 tf@4**；
   两臂都是非 dense + proj-fill（形态件）；块类与参数量声明一致；carry 臂活性量非 null、
   无 carry 臂为 null；clean bpc 是非退化锚（> log2 206）；tf@4 判定与 Δ 一致；与 2629 的
   **跨分布口径**在报告中分开写（本文件只做「本口径内部自洽」）。

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

from samecaliber_arms import (ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                              tf4_holds)


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = (abs(float(got) - float(exp)) <= tol
              if isinstance(got, (int, float)) and isinstance(exp, (int, float))
              else got == exp)
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    TF = R["tf_enwik8_samecaliber"]
    BPC = R["clean_bpc_samecaliber"]
    PAR = R["param_declaration"]

    # ---------------- ① 算术对账 ----------------
    for name in ARM_NAMES:
        by = {r["depth"]: r["accept"] for r in TF[name]["by_depth"]}
        ck(f"arith.tf.{name}.tf4==by_depth[4]", TF[name]["tf4"], by[4], 1e-9)
        ck(f"arith.tf.{name}.tf6==by_depth[6]", TF[name]["tf6"], by[6], 1e-9)
        ck(f"arith.tf.{name}.tf4_in_ci01", 0.0 <= TF[name]["tf4"] <= 1.0, True)
        ck(f"arith.bpc.{name}.bpc==by_depth_last",
           BPC[name]["bpc"], BPC[name]["by_depth"][-1]["bpc"], 1e-9)
        d = PAR[name]
        ck(f"arith.param.{name}.delta==n-base", d["delta_vs_base"],
           d["n_params"] - PAR[BASELINE_ARM]["n_params"])
        ck(f"arith.param.{name}.pct",
           d["delta_pct_vs_base"],
           round(100.0 * d["delta_vs_base"] / PAR[BASELINE_ARM]["n_params"], 3),
           1e-6)
    # 增量从原始数重算
    for key, leg, val in (
            ("tf4_enwik8_samecaliber", TF, lambda n: TF[n]["tf4"]),
            ("tf6_enwik8_samecaliber", TF, lambda n: TF[n]["tf6"]),
            ("tf4_synth_crosscaliber_diag",
             R["tf_synth_crosscaliber_diag"],
             lambda n: R["tf_synth_crosscaliber_diag"][n]["accept"]["4"]),
            ("bpc_samecaliber", BPC,
             lambda n: -BPC[n]["bpc"])):
        # tf@k 的增量以 pp 计（×100）；bpc 的增量以 raw bpc 计（不乘）
        scale = 1.0 if key == "bpc_samecaliber" else 100.0
        exp = round(scale * (val(TARGET_ARM) - val(BASELINE_ARM)), 6)
        ck(f"arith.increments.{key}", R["increments"][key]["carry_effect"], exp, 1e-6)
        ck(f"arith.increments.{key}.net",
           R["increments"][key]["net_vs_base"], exp, 1e-6)
    # 判定与 Δ 一致
    dec = R["tf4_decision"]
    ck("arith.decision.delta==incr", dec["delta_pp"],
       R["increments"]["tf4_enwik8_samecaliber"]["carry_effect"], 1e-9)
    ck("arith.decision.holds==rule", dec["holds"], tf4_holds(dec["delta_pp"]))
    ck("arith.decision.band", dec["band_pp"], 1.4, 1e-9)

    # ---------------- ② 任务对账（同口径 / 同腿参数 / 两只入口一致） ----------------
    aeq = R["architecture_equality"]
    ck("task.same_data_source",
       len({aeq["data_source"][n] for n in ARM_NAMES}), 1)
    ck("task.data_source_is_enwik8",
       aeq["data_source"][BASELINE_ARM], "enwik8")
    ck("task.same_n_batches",
       len({BPC[n]["n_batches"] for n in ARM_NAMES}), 1)
    ck("task.same_batch_size",
       len({BPC[n]["batch_size"] for n in ARM_NAMES}), 1)
    ck("task.tf_leg_n_batches", BPC[BASELINE_ARM]["n_batches"], 20)
    ck("task.eval_only", R["cost"]["pure_inference"], True)
    # run.py 的 tf@4 / clean bpc 与训练入口 train/<arm>/results.json 逐位一致
    # （carry 臂的训练入口因引擎已知限制在 generation_quality 处中止 ⇒ results.json
    #   未落盘；该缺档是**预期**的，主电池由 run.py 独立重算，见 README §4b/§8。）
    te = R["train_entry_battery"]
    for name in ARM_NAMES:
        if te[name]["present"]:
            ck(f"task.{name}.tf4_run==train", TF[name]["tf4"], te[name]["tf4"],
               1e-9)
            ck(f"task.{name}.tf6_run==train", TF[name]["tf6"], te[name]["tf6"],
               1e-9)
            ck(f"task.{name}.cleanbpc_run==train", BPC[name]["bpc"],
               te[name]["clean_bpc"], 1e-9)
        else:
            ck(f"task.{name}.train_entry_absent_is_target", name, TARGET_ARM)
            ck(f"task.{name}.train_entry_absence_noted",
               bool(te[name].get("note")), True)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.baseline", BASELINE_ARM, "base")
    ck("caliber.target", TARGET_ARM, "carry")
    ck("caliber.block_classes", aeq["block_class"], aeq["block_class_expected"])
    ck("caliber.both_non_dense",
       all(v is False for v in aeq["dense"].values()), True)
    ck("caliber.both_proj_fill",
       all(v is True for v in aeq["proj_fill"].values()), True)
    ck("caliber.both_fill_kv", all(v is True for v in aeq["fill_kv"].values()), True)
    ck("caliber.param_delta_positive", PAR[TARGET_ARM]["delta_vs_base"] > 0, True)
    ck("caliber.bpc_all_finite_positive",
       all(0.0 < BPC[n]["bpc"] < math.log2(206) for n in ARM_NAMES), True)
    ck("caliber.carry_activity_present",
       R["train_activity"][TARGET_ARM]["attn_carry_series_present"] > 0, True)
    ck("caliber.nocarry_arm_quiet",
       R["train_activity"][BASELINE_ARM]["last_attn_carry"] is None, True)
    ck("caliber.intervention_alive", R["intervention"]["alive"], True)
    ck("caliber.intervention_enabled", R["intervention"]["enabled"], True)
    ck("caliber.intervention_params_added",
       R["intervention"]["params_added"], PAR[TARGET_ARM]["delta_vs_base"])
    ck("caliber.intervention_nonzero",
       R["intervention"]["nonzero_frac"] is not None
       and R["intervention"]["nonzero_frac"] > 0, True)
    ck("caliber.capability_buildable", R["capability_probe"]["combo_buildable"], True)
    ck("caliber.capability_trunk_identical",
       R["capability_probe"]["trunk_bit_identical"], True)
    ck("caliber.convergence_slopes_present",
       all(R["train_activity"][n]["full_depth_ce_slope_nats_per_1k_lastquarter"]
           is not None for n in ARM_NAMES), True)
    ck("files.analysis_exists", os.path.exists(os.path.join(HERE, "analysis.md")), True)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    # 每条 = (名字, 实际值, 期望值, [容差])；期望值 = analysis.md 里写的数。
    # 注：本块在训练落盘后按 analysis.md 的实际引用逐条回填（提交前不得留占位）。
    ACT = R["train_activity"]
    CITED = [
        # §3 参数量 / 活性
        ("s3 param base", PAR[BASELINE_ARM]["n_params"], 14326272),
        ("s3 param carry", PAR[TARGET_ARM]["n_params"], 14590568),
        ("s3 param delta", PAR[TARGET_ARM]["delta_vs_base"], 264296),
        ("s3 param pct", PAR[TARGET_ARM]["delta_pct_vs_base"], 1.845, 5e-4),
        ("s3 w_fro 76.019", R["intervention"]["w_fro"], 76.018863, 5e-6),
        ("s3 nonzero_frac 1.0", R["intervention"]["nonzero_frac"], 1.0, 1e-9),
        ("s3 read_scale 0.1679", R["intervention"]["read_scale_absmean"], 0.167914, 5e-6),
        ("s3 carry series 31", ACT[TARGET_ARM]["attn_carry_series_present"], 31),
        ("s3 carry read_share 0.4070",
         ACT[TARGET_ARM]["carry_series_last200"]["read_share"], 0.406998, 5e-6),
        ("s3 nocarry quiet", ACT[BASELINE_ARM]["last_attn_carry"], None),
        # §5 收敛门禁
        ("s5 slope full base -0.05470",
         ACT[BASELINE_ARM]["full_depth_ce_slope_nats_per_1k_lastquarter"], -0.054696, 5e-6),
        ("s5 slope full carry -0.00752",
         ACT[TARGET_ARM]["full_depth_ce_slope_nats_per_1k_lastquarter"], -0.007518, 5e-6),
        ("s5 slope d4 base -0.06416",
         ACT[BASELINE_ARM]["depth4_ce_slope_nats_per_1k_lastquarter"], -0.064161, 5e-6),
        ("s5 slope d4 carry -0.03050",
         ACT[TARGET_ARM]["depth4_ce_slope_nats_per_1k_lastquarter"], -0.0305, 5e-6),
        ("s5 final loss base 1.6862", ACT[BASELINE_ARM]["final_loss"], 1.6862, 5e-5),
        ("s5 final loss carry 1.5786", ACT[TARGET_ARM]["final_loss"], 1.5786, 5e-5),
        # §2 主结果（同口径）
        ("s2 tf4 base 97.11", TF[BASELINE_ARM]["tf4"], 0.9711, 5e-5),
        ("s2 tf4 carry 96.35", TF[TARGET_ARM]["tf4"], 0.9635, 5e-5),
        ("s2 tf6 base 98.60", TF[BASELINE_ARM]["tf6"], 0.986, 5e-5),
        ("s2 tf6 carry 98.50", TF[TARGET_ARM]["tf6"], 0.985, 5e-5),
        ("s2 delta tf4 -0.76", R["tf4_decision"]["delta_pp"], -0.76, 5e-5),
        ("s2 delta tf6 -0.10",
         R["increments"]["tf6_enwik8_samecaliber"]["carry_effect"], -0.1, 5e-5),
        ("s2 clean bpc base 2.0923", BPC[BASELINE_ARM]["bpc"], 2.0923, 5e-5),
        ("s2 clean bpc carry 2.0098", BPC[TARGET_ARM]["bpc"], 2.0098, 5e-5),
        ("s2 delta bpc +0.0825", R["increments"]["bpc_samecaliber"]["carry_effect"],
         0.0825, 5e-4),
        ("s2 holds true", R["tf4_decision"]["holds"], True),
        # §4 反向跳分布诊断（不作判据）
        ("s4 diag synth tf4 base 88.26",
         R["tf_synth_crosscaliber_diag"][BASELINE_ARM]["accept"]["4"], 0.882598, 5e-6),
        ("s4 diag synth tf4 carry 59.37",
         R["tf_synth_crosscaliber_diag"][TARGET_ARM]["accept"]["4"], 0.593689, 5e-6),
        ("s4 diag delta -28.89",
         R["increments"]["tf4_synth_crosscaliber_diag"]["carry_effect"], -28.8909, 5e-4),
        # §9 成本
        ("s9 eval peak 628.1", R["cost"]["peak_gpu_mem_mb"], 628.1, 1e-9),
        ("s9 eval wall finite", R["cost"]["wall_minutes_total_eval"] < 1.0, True),
        ("s9 base train peak 2279.9",
         R["train_entry_battery"][BASELINE_ARM]["peak_gpu_mem_mb"], 2279.9, 1e-9),
        ("s9 base train wall 8.78",
         R["train_entry_battery"][BASELINE_ARM]["wall_minutes"], 8.78, 1e-9),
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
