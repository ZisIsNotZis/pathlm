"""2620 · 报告数字对账（SCHEMA §12）：`analysis.md` / `README.md` / `stability.md`
里引用的每个数字都必须能回到 `results.json` / 训练日志 / 其它落盘文件。

用法：python3 reconcile_numbers.py    # 任一 mismatch 以非零码退出（写 reconcile_report.json）

**口径修正（采纳 ticket 36 复核 V4 的意见）**：报告侧的匹配**不用子串**，
而是把报告里所有「带小数点的数值 token」抽出来，按**数值容差**匹配
（`abs(float(tok) - value) <= tol`）——这样 `32.11` 不会再被 `32` 假命中，
也不允许落入无小数点的整数字样。
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import capacity_converged_lib as L          # noqa: E402

R = json.load(open(os.path.join(HERE, "results.json")))
CFG = R["config"]
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
AN = open(os.path.join(HERE, "analysis.md")).read()
RM = open(os.path.join(HERE, "README.md")).read()
ST = open(os.path.join(HERE, "stability.md")).read()

checks = []
NUM_RE = re.compile(r"\d+\.\d+")


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


def num_in(text, value, label, tol=5e-4):
    """数值容差匹配（要求报告里出现带小数点的数值 token）。"""
    if value is None:
        ck(label, False, None, "value is None")
        return
    toks = [float(t) for t in NUM_RE.findall(text)]
    ok = any(abs(t - float(value)) <= tol for t in toks)
    ck(label, ok, f"nearest={min((abs(t-float(value)) for t in toks), default=None)}",
       f"{value}±{tol}")


grid = R["grid"]
levels = [tuple(x) for x in CFG["difficulty_levels"]]
arms = CFG["arm_order"]
small, big = CFG["main_arms"]

# ---------------------------------------------------------------- 原始计数 → 比例
for key, row in grid.items():
    ck(f"{key} accuracy == n_hits/n",
       round(row["accuracy"], 6) == round(row["n_hits"] / row["n"], 6),
       row["accuracy"], round(row["n_hits"] / row["n"], 6))
    ck(f"{key} corrupt accuracy == hits/n",
       round(row["corrupt_source"]["accuracy"], 6)
       == round(row["corrupt_source"]["n_hits"] / row["corrupt_source"]["n"], 6))
    ck(f"{key} chance == 1/64", abs(row["chance"] - 1 / 64) < 1e-9, row["chance"], 1 / 64)
    ck(f"{key} n == bs*nb*Q",
       row["n"] == CFG["eval"]["batch_size"] * CFG["eval"]["n_batches"] * row["Q"],
       row["n"], CFG["eval"]["batch_size"] * CFG["eval"]["n_batches"] * row["Q"])
    ck(f"{key} dh == arm spec",
       row["dh"] == CFG["arms"][row["arm"]]["carry_state_dim"])
    ck(f"{key} corrupt span == (0, 2K)",
       row["corrupt_source"]["span"] == [0, 2 * row["K"]])

# ---------------------------------------------------------------- 结构自检
sc = R["selfchecks"]
ck("自检 release vs 显式因果掩码 == 0.0",
   sc["release_vs_explicit_causal_max_abs_logit"] == 0.0,
   sc["release_vs_explicit_causal_max_abs_logit"], 0.0)
ck("自检 模型窗口掩码前向 vs 仪器 == 0.0",
   sc["model_window_forward_vs_mask_instrument_max_abs_logit"] == 0.0,
   sc["model_window_forward_vs_mask_instrument_max_abs_logit"], 0.0)
ck("自检 注意力头数三臂恒定", sc["attention_heads_unchanged_across_arms"] is True)
ck("自检 总状态浮点 = d_model·dh",
   sc["total_state_floats_equals_d_model_times_dh"] is True)
for n in arms:
    st = sc["structure_by_arm"][n]
    ck(f"{n} 主干逐位 == 参考臂", st["trunk_bitwise_equal_to_ref"] is True)
    ck(f"{n} carry head_dim == dh",
       st["carry_state_dim"] == CFG["arms"][n]["carry_state_dim"])
    ck(f"{n} state floats == 256*dh",
       st["carry_state_floats"] == 256 * CFG["arms"][n]["carry_state_dim"])
    ck(f"{n} 注意力头 == 8", st["attn_heads"] == CFG["model"]["n_heads"])

# ---------------------------------------------------------------- 判据 ① 收敛门禁复算
conv = R["convergence"]
for n in CFG["main_arms"]:
    b = conv["by_arm"][n]
    lin, g = b["ansce_last_window_linear"], b["gate"]
    # 直接复算：从训练日志重算
    lines = [json.loads(x) for x in
             open(os.path.join(ROOT, CFG["arms"][n]["train_log"])) if x.strip()]
    recomputed = L.linear_window(lines, "synth_answer_ce", *conv["window"])
    ck(f"{n} 末窗 slope 与训练日志复算一致",
       recomputed["slope_per_1k"] == lin["slope_per_1k"],
       recomputed["slope_per_1k"], lin["slope_per_1k"])
    ck(f"{n} 末窗 rel_drop 与训练日志复算一致",
       recomputed["rel_drop"] == lin["rel_drop"],
       recomputed["rel_drop"], lin["rel_drop"])
    expect = (abs(lin["slope_per_1k"]) < L.CONVERGENCE_SLOPE_MAX
              or lin["rel_drop"] < L.CONVERGENCE_REL_DROP_MAX)
    ck(f"{n} 门禁旗标复算一致", g["converged"] == expect,
       g["converged"], expect)
    ck(f"{n} 总 CE 末窗斜率已报（对照，不作门禁）",
       b["total_loss_last_window_linear"]["slope_per_1k"] is not None)
# 「总 CE 口径恒真」的证据：U1 的 10k 日志
for arm in ("D16", "D128"):
    u = conv["u1_10k_reference"][arm]
    ck(f"U1/{arm} 总 CE 末窗斜率 |·| < 0.01（恒真的证据）",
       abs(u["total_loss_[6000,10000]_linear"]["slope_per_1k"]) < 0.01,
       u["total_loss_[6000,10000]_linear"]["slope_per_1k"], "<0.01")
    ck(f"U1/{arm} 答案行末窗斜率 |·| >= 0.01（未收敛的证据）",
       abs(u["ansce_[6000,10000]_linear"]["slope_per_1k"]) >= 0.01,
       u["ansce_[6000,10000]_linear"]["slope_per_1k"], ">=0.01")
crit = R["criteria"]
ck("c1 门禁 == 两主臂 gate 全过",
   crit["c1_convergence_gate"]["both_converged"]
   == all(conv["by_arm"][n]["gate"]["converged"] for n in CFG["main_arms"]))

# ---------------------------------------------------------------- 判据 ② 主判据复算
for (K, _) in levels:
    v = crit["c2_main_by_K"][f"K{K}"]["verdict"]
    a, b = grid[f"{small}_K{K}"], grid[f"{big}_K{K}"]
    ck(f"K={K} delta_pp 复算一致",
       v["delta_pp"] == round(100 * (b["accuracy"] - a["accuracy"]), 3),
       v["delta_pp"], round(100 * (b["accuracy"] - a["accuracy"]), 3))
    expect_eff = bool(v["meets_delta"] and not v["ci_overlap"])
    ck(f"K={K} capacity_axis_effective 复算一致",
       v["capacity_axis_effective"] == expect_eff, v["capacity_axis_effective"], expect_eff)
    ck(f"K={K} is_main_leg 与预登记主腿一致",
       v["is_main_leg"] == (K == L.MAIN_LEVEL_K))
main_leg = crit["c2_main_by_K"][f"K{L.MAIN_LEVEL_K}"]["verdict"]
ck("c2 结论 = 门禁 ∧ 主腿有效",
   crit["c2_capacity_axis_effective"]
   == bool(crit["c1_convergence_gate"]["both_converged"]
           and main_leg["capacity_axis_effective"]),
   crit["c2_capacity_axis_effective"], None)

# ---------------------------------------------------------------- 判据 ③.1 毁源
n_invalid = 0
for key, v in crit["c3_1_corrupt"].items():
    row = grid[key]
    inv = row["corrupt_source"]["accuracy_ci"][0] > 4 * row["chance"]
    ck(f"毁源判读复算 {key}", v["invalidated"] == inv)
    n_invalid += int(inv)
ck("作废格数复算一致", crit["c3_1_n_invalidated_cells"] == n_invalid,
   crit["c3_1_n_invalidated_cells"], n_invalid)

# ---------------------------------------------------------------- 判据 ③.2 重跑散度
for pair, d in crit["c3_2_training_divergence"].items():
    orig, rep = pair.split("_vs_")
    ko, kr = grid[f"{orig}_K{L.MAIN_LEVEL_K}"], grid[f"{rep}_K{L.MAIN_LEVEL_K}"]
    ck(f"{pair} K=8 delta_pp 复算一致",
       d["accuracy_at_main_leg"]["delta_pp"] == round(100 * (kr["accuracy"] - ko["accuracy"]), 3),
       d["accuracy_at_main_leg"]["delta_pp"], None)
    ck(f"{pair} ci_overlap 复算一致",
       d["accuracy_at_main_leg"]["ci_overlap"]
       == (not (kr["accuracy_ci"][0] > ko["accuracy_ci"][1]
                or ko["accuracy_ci"][0] > kr["accuracy_ci"][1])))
    lines_a = [json.loads(x) for x in
               open(os.path.join(ROOT, CFG["arms"][orig]["train_log"])) if x.strip()]
    lines_b = [json.loads(x) for x in
               open(os.path.join(ROOT, CFG["arms"][rep]["train_log"])) if x.strip()]
    rec = L.training_divergence(lines_a, lines_b)
    for key in ("loss", "synth_answer_ce"):
        ck(f"{pair} {key} max|Δ| 与日志复算一致",
           rec["per_key"][key]["max_abs_diff"] == d["per_key"][key]["max_abs_diff"],
           rec["per_key"][key]["max_abs_diff"], d["per_key"][key]["max_abs_diff"])
        ck(f"{pair} {key} 同步骤对数一致",
           rec["per_key"][key]["n_steps_compared"]
           == d["per_key"][key]["n_steps_compared"])

# ---------------------------------------------------------------- 活性量 / 成本
for n in arms:
    s = R["liveness_series"][n]
    ck(f"{n} 活性量每 200 步落盘（≥100 行）", len(s) >= 100, len(s), ">=100")
    ck(f"{n} 读出占比末值 > 0", (s[-1].get("read_share") or 0) > 0, s[-1].get("read_share"))
    ck(f"{n} 状态范数末值 > 0", (s[-1].get("state_fro") or 0) > 0, s[-1].get("state_fro"))
cost = R["cost"]
ck("n_forward_batches == nb*档数*臂数*2",
   cost["n_forward_batches"] == CFG["eval"]["n_batches"] * len(levels) * len(arms) * 2,
   cost["n_forward_batches"], CFG["eval"]["n_batches"] * len(levels) * len(arms) * 2)
ck("n_grid_cells == 臂数×档数", cost["n_grid_cells"] == len(arms) * len(levels))
ck("训练墙钟合计 == 逐臂之和",
   abs(cost["train_wall_minutes_total"]
       - sum(cost["train_wall_minutes"].values())) < 1e-6,
   cost["train_wall_minutes_total"], sum(cost["train_wall_minutes"].values()))
for n in arms:
    tr = json.load(open(os.path.join(ROOT, CFG["arms"][n]["results_json"])))
    ck(f"{n} 训练墙钟一致", cost["train_wall_minutes"][n] == tr["wall_minutes"],
       cost["train_wall_minutes"][n], tr["wall_minutes"])
    ck(f"{n} 峰值显存一致",
       R["train_results"][n]["peak_gpu_mem_mb"] == tr["peak_gpu_mem_mb"])
    ck(f"{n} 参数量一致", R["train_results"][n]["params"] == tr["params"])
    ck(f"{n} 训练步数 == 24000", tr["steps"] == CFG["train_recipe"]["steps"],
       tr["steps"], CFG["train_recipe"]["steps"])

# ---------------------------------------------------------------- 复现性 / 变异 / §22
ck("同 seed 复跑逐位一致", R["reproducibility"]["pass"] is True)
ck("变异验证全部抓红", MUT["all_caught"] is True)
ck("变异验证未改任何源文件", MUT["no_source_file_modified"] is True)
ck("自洽对账 all_consistent", R["reconciliation"]["all_consistent"] is True)
ck("无暂存文件（no staged）",
   len(subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                      capture_output=True, text=True).stdout.split()) == 0)

# ---------------------------------------------------------------- 报告里能查到数字
num_in(AN, grid[CFG["premise_cell"]]["accuracy"], "analysis 里有 S16_K8 准确率")
for n in arms:
    num_in(AN, grid[f"{n}_K8"]["accuracy"], f"analysis 里有 {n}_K8 准确率")
    num_in(AN, grid[f"{n}_K32"]["accuracy"], f"analysis 里有 {n}_K32 准确率")
num_in(AN, main_leg["delta_pp"], "analysis 里有主腿 Δpp")
for n in CFG["main_arms"]:
    num_in(AN, conv["by_arm"][n]["ansce_last_window_linear"]["slope_per_1k"],
           f"analysis 里有 {n} 答案行末窗斜率")
    num_in(AN, conv["by_arm"][n]["total_loss_last_window_linear"]["slope_per_1k"],
           f"analysis 里有 {n} 总 CE 末窗斜率")
num_in(AN, cost["train_wall_minutes_total"], "analysis 里有训练墙钟合计")
num_in(AN, cost["wall_minutes_total_eval"], "analysis 里有评测墙钟")
num_in(ST, 0.0014, "stability 里有 600 步确定性探针的 loss 差（近似）", tol=5e-4)
ck("README 含判据线与预登记字样", "预登记" in RM and "§4" in RM)
ck("README 含 GPU 锁声明", "pathlm_gpu.lock" in RM)
ck("analysis 含「恒定/恒真」总 CE 口径说明", "恒真" in AN or "无区分度" in AN)
ck("analysis 含 §19 口径（不得写「机制不可能」）",
   "不划算" in AN or "不是约束" in AN)

# ---------------------------------------------------------------- 汇总
failed = [c for c in checks if not c[1]]
report = {"n_checks": len(checks), "n_failed": len(failed), "failed": failed,
          "checks": checks,
          "note": ("报告侧用数值容差匹配（NUM_RE 抽带小数点的 token），"
                   "不再用子串匹配（采纳 ticket 36 复核 V4 的意见）")}
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"reconcile: {len(checks) - len(failed)}/{len(checks)} passed")
for c in failed:
    print("  FAIL:", c[0], "| got", c[2], "| want", c[3])
sys.exit(0 if not failed else 1)
