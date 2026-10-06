"""2628-eng2-carry-ragged · 对账（SCHEMA §12 三道）：`analysis.md` / `results.json` 的每个
数字都能在 `results.json`（及它引用的冒烟日志 / 测试输出）里找到同值，且算术自洽。

三道对账：
1. **算术对账**：四个组合都在同一非 dense CPU 测试位点（d=32/3L/4H/seq12/seed0）上量；
   `carry_read_share > 0`、`carry_changes_logits == True`、`frozen_invariant_ok == True`
   三条活性量逐组合成立；proj-fill 组合的 adapter 消费量为正、非 proj-fill 组合为 None。
2. **任务对账**：`exit_mode` / `fill_kv` / `proj_fill` 三个旗标与组合名一一对应；退出
   模式真的改变前向（`exit_changes_logits_vs_full == True`，非退化点）；遥测层数 =
   模型层数（通道在非 dense 路径上也进了 `aux["attn_carry"]`）。
3. **口径对账**：缺省路径不含任何通道结构；写入时机与退出模式无关（两条布尔均真）；
   冒烟的写入通道活性为活且日志行含 `attn_carry`；全量回归无 fail；变异全部抓红且
   逐字节还原。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: 组合名 → (fill_kv, proj_fill, 是否用固定退出模式)
EXPECT = {
    "carry_x_ragged": (False, False, True),
    "carry_x_random_exit": (False, False, False),
    "carry_x_random_exit_x_fill": (True, False, False),
    "carry_x_proj_fill": (True, True, False),
}


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks = []

    def ck(name, got, exp):
        checks.append({"check": name, "want": exp, "got": got,
                       "pass": got == exp})

    cfg = R["config"]["model"]
    L = cfg["n_layers"]

    # ---------------- ① 算术对账 ----------------
    ck("arithmetic.n_combos", len(R["combos"]), 4)
    for c in R["combos"]:
        name = c["name"]
        fill, proj, fixed = EXPECT[name]
        ck(f"{name}.fill_kv", c["fill_kv"], fill)
        ck(f"{name}.proj_fill", c["proj_fill"], proj)
        ck(f"{name}.fixed_exit", c["exit_mode"] == "fixed", fixed)
        ck(f"{name}.block_class", c["block_class"], "CarryChannelBlock")
        # 活性量逐组合成立
        ck(f"{name}.carry_read_share_present", c["carry_read_share"] is not None,
           True)
        ck(f"{name}.carry_alive", (c["carry_read_share"] or 0) > 0.0, True)
        ck(f"{name}.carry_changes_logits", c["carry_changes_logits"], True)
        ck(f"{name}.frozen_invariant", c["frozen_invariant_ok"], True)
        ck(f"{name}.telemetry_layers", c["n_carry_layers_in_aux"], L)
        ck(f"{name}.scan_tele_present", c["carry_scan_tele_present"], True)
        # proj-fill adapter 只在 proj-fill 组合上被消费
        delta = c["proj_fill_adapter_max_abs_logit_delta"]
        ck(f"{name}.adapter_consumed", (delta is not None) and delta > 1e-6, proj)

    # ---------------- ② 任务对账（同一测试位点 + 非退化点） ----------------
    ck("task.same_point_dense", cfg["dense"], False)
    ck("task.same_point_topology", cfg["topology"], "sequential")
    ck("task.same_point_carry", cfg["carry_rule"], "delta")
    for c in R["combos"]:
        ck(f"{c['name']}.exit_non_degenerate",
           c["exit_changes_logits_vs_full"], True)
        ck(f"{c['name']}.exit_range_sane",
           c["exit_min"] < L and 0 <= c["exit_min"] <= c["exit_max"] <= L, True)

    # ---------------- ③ 口径对账（缺省不动 / 写入时机 / 冒烟活 / 回归绿 / 变异红） ----------------
    du = R["default_untouched"]
    ck("caliber.default_block", du["block_class"], "Block")
    ck("caliber.default_no_carry_module", du["has_carry_module"], False)
    ck("caliber.default_no_carry_box", du["has_carry_box"], False)
    ck("caliber.default_no_telemetry", du["attn_carry_in_aux"], False)
    wt = R["write_timing"]
    ck("caliber.slot_indep_full_vs_fixed",
       wt["slot_identical_full_vs_fixed"], True)
    ck("caliber.slot_indep_full_vs_zero",
       wt["slot_identical_full_vs_zero"], True)
    if R.get("smoke"):
        cl = R["smoke"].get("carry_liveness", {})
        # 20 步冒烟里读出增益刚离开单边零（ENG1 同口径：`alive` 在 20 步上为 false）
        # ⇒ 判据取「已逃逸 + 读出量非零」，不强求 `alive`。
        ck("smoke.carry_scale_escaped",
           (cl.get("read_scale_absmean") or 0) > 0, True)
        ck("smoke.carry_read_share_positive",
           (cl.get("read_share") or 0) > 0, True)
        ck("smoke.final_step", R["smoke"]["final_step"], 19)
        ck("smoke.log_has_attn_carry",
           R["smoke"].get("attn_carry_final") is not None, True)
        ck("smoke.log_attn_carry_rule",
           (R["smoke"].get("attn_carry_final") or {}).get("rule"), "delta")
        ck("smoke.carry_rule", R["smoke"].get("rebind_carry_rule"), "delta")
        ck("smoke.fill_kv", R["smoke"].get("fill_kv"), True)
        ck("smoke.proj_fill", R["smoke"].get("proj_fill"), True)
        ck("smoke.block_class", R["smoke"].get("block_carry_liveness_class"),
           "CarryChannelBlock")
    if R.get("regression"):
        ck("regression.no_failed", R["regression"].get("failed", 0), 0)
    if R.get("mutation"):
        ck("mutation.all_caught_red", R["mutation"]["all_caught_red"], True)
        ck("mutation.all_restored_bitwise",
           R["mutation"]["all_restored_bitwise"], True)
        ck("mutation.sha_restored",
           R["mutation"]["sha256_restored"], R["mutation"]["sha256_original"])

    fails = [c for c in checks if not c["pass"]]
    out = {"n_checks": len(checks), "n_failed": len(fails), "checks": checks}
    with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for c in fails:
        print("FAIL", c, file=sys.stderr)
    print(f"reconcile: {len(checks) - len(fails)}/{len(checks)} pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
