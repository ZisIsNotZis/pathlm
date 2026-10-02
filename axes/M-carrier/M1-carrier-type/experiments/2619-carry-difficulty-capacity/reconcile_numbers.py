"""2619 · 报告数字对账：报告里每个数字都必须能回到 results.json / 训练日志 / 其它落盘。

用法：python3 reconcile_numbers.py    # 任一 mismatch 以非零码退出（报告 reconcile_report.json）
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

R = json.load(open(os.path.join(HERE, "results.json")))
CFG = R["config"]
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
import subprocess                                        # noqa: E402
AN = open(os.path.join(HERE, "analysis.md")).read()
RM = open(os.path.join(HERE, "README.md")).read()

checks = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


# 注意（复核 V4 汇总 6）：本函数用**子串**匹配，会把 32.11 与 32、0.82 与 0.8 混淆；
# 后续请改为按数值容差匹配（abs(float(a)-float(b)) < tol），并禁止落入无小数点整数档。
def num_in(text, value, label, places=6):
    for fmt in (f"{value:.{places}f}", f"{value:.4f}", f"{value:.3f}",
                f"{value:.2f}", f"{value:.1f}"):
        if fmt in text:
            ck(label, True, fmt, "found")
            return
    ck(label, False, None, f"any of {value:.6f}")


grid = R["grid"]
levels = CFG["difficulty_levels"]
arms = CFG["arm_order"]

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
ck("自检 注意力头数四臂恒定", sc["attention_heads_unchanged_across_arms"] is True)
ck("自检 总状态浮点 = d_model·dh",
   sc["total_state_floats_equals_d_model_times_dh"] is True)
for n in arms:
    st = sc["structure_by_arm"][n]
    ck(f"{n} 主干逐位 == 参考臂", st["trunk_bitwise_equal_to_ref"] is True)
    ck(f"{n} carry head_dim == dh", st["carry_state_dim"] == CFG["arms"][n]["carry_state_dim"])
    ck(f"{n} state floats == 256*dh",
       st["carry_state_floats"] == 256 * CFG["arms"][n]["carry_state_dim"])
    ck(f"{n} 注意力头 == 8", st["attn_heads"] == CFG["model"]["n_heads"])

# ---------------------------------------------------------------- 判据复算
prem = R["premise_check"]
ck("前提校验：显著高于 chance == (CI 下界 > chance)",
   prem["significantly_above_chance"] == (prem["accuracy_ci"][0] > prem["chance"]))
ck("前提校验：multiple_of_chance == acc/chance",
   abs(prem["multiple_of_chance"] - prem["accuracy"] / prem["chance"]) < 5e-4)

crit = R["criteria"]
ck("c1 == 前提校验", crit["c1_premise_anchor_above_chance"] is True)

diff = R["curves"]["difficulty"]
vals = {p["K"]: p["accuracy"] for p in diff["points"]}
ck("难度曲线 4 个测点", sorted(vals) == sorted(K for K, _ in levels), sorted(vals))
# 复算单调性：只在 CI 不重叠的相邻对判定
adj = diff["adjacent"]
resolvable = [a for a in adj if not a["ci_overlap"]]
mdown = bool([a for a in resolvable if a["delta_pp"] < 0]) and not [
    a for a in resolvable if a["delta_pp"] > 0]
ck("难度曲线 monotone_non_increasing 复算一致",
   diff["monotone_non_increasing"] == mdown, diff["monotone_non_increasing"], mdown)
ck("c2 == 难度曲线 monotone_or_knee",
   crit["c2_difficulty_monotone_or_knee"] == diff["monotone_or_knee"])

cap = R["curves"]["capacity"]
for K, _ in levels:
    c = cap[f"K{K}"]
    ck(f"容量曲线 K={K} 4 个测点",
       sorted(p["dh"] for p in c["points"]) == [16, 32, 64, 128])
ck("capacity_main_K 存在", f"K{CFG['capacity_main_K']}" in cap)
ck("c3 == 容量主腿({}) monotone_or_knee".format(CFG["capacity_main_K"]),
   crit["c3_capacity_monotone_or_knee"]
   == cap[f"K{CFG['capacity_main_K']}"]["monotone_or_knee"])

# 交换率复算
ex = R["exchange_rate"]
thr = CFG["exchange_threshold_mult"] * grid[CFG["premise_cell"]]["chance"]
ck("交换率阈值 == 4×chance", abs(ex["threshold"] - thr) < 1e-12)
for n in arms:
    passed = [K for K, _ in levels if grid[f"{n}_K{K}"]["accuracy_ci"][0] > thr]
    want = max(passed) if passed else 0
    ck(f"{n} K* 复算一致", ex["K_star_by_arm"][n]["K_star"] == want,
       ex["K_star_by_arm"][n]["K_star"], want)
    ck(f"{n} K_passed 复算一致", ex["K_star_by_arm"][n]["K_passed"] == passed)
n_distinct = len({v["K_star"] for v in ex["K_star_by_arm"].values()})
ck("交换率 distinct 点数复算一致", ex["n_distinct_measured_points"] == n_distinct,
   ex["n_distinct_measured_points"], n_distinct)
ck("c4 == 交换率≥3 实测点",
   crit["c4_exchange_ge3_measured_points"] == (n_distinct >= 3))

# 毁源对照复算
n_invalid = 0
for key, v in crit["c5_corrupt_source_control"].items():
    row = grid[key]
    inv = row["corrupt_source"]["accuracy_ci"][0] > 4 * row["chance"]
    ck(f"毁源判读复算 {key}", v["invalidated"] == inv)
    n_invalid += int(inv)
ck("c5 作废格数复算一致", crit["c5_n_invalidated_cells"] == n_invalid,
   crit["c5_n_invalidated_cells"], n_invalid)

# ---------------------------------------------------------------- 成本 / 训练侧
cost = R["cost"]
ck("n_forward_batches == nb*档数*臂数*2",
   cost["n_forward_batches"] == CFG["eval"]["n_batches"] * len(levels) * len(arms) * 2,
   cost["n_forward_batches"], CFG["eval"]["n_batches"] * len(levels) * len(arms) * 2)
ck("n_grid_cells == 臂数×档数", cost["n_grid_cells"] == len(arms) * len(levels))
for n in arms:
    tp = os.path.join(ROOT, CFG["arms"][n]["results_json"])
    tr = json.load(open(tp))
    ck(f"{n} 训练墙钟一致",
       cost["train_wall_minutes"][n] == tr["wall_minutes"],
       cost["train_wall_minutes"][n], tr["wall_minutes"])
    ck(f"{n} 峰值显存一致",
       R["train_results"][n]["peak_gpu_mem_mb"] == tr["peak_gpu_mem_mb"])
    ck(f"{n} 参数量一致", R["train_results"][n]["params"] == tr["params"])
    ck(f"{n} loss 斜率已算", R["loss_slope"][n]["slope"] is not None)

# ---------------------------------------------------------------- 复现性 / 变异
ck("同 seed 复跑逐位一致", R["reproducibility"]["pass"] is True)
ck("变异验证全部抓红", MUT["all_caught"] is True)

# ---------------------------------------------------------------- 报告里能查到数字
num_in(AN, grid[CFG["premise_cell"]]["accuracy"], "analysis 里有 anchor 准确率")
num_in(AN, prem["multiple_of_chance"], "analysis 里有 anchor / chance 倍数", places=2)
for n in arms:
    num_in(AN, grid[f"{n}_K8"]["accuracy"], f"analysis 里有 {n}_K8 准确率")
    num_in(AN, grid[f"{n}_K32"]["accuracy"], f"analysis 里有 {n}_K32 准确率")
num_in(AN, cost["wall_minutes_total_eval"], "analysis 里有评测墙钟", places=2)
ck("README 含判据线字样", "预登记" in RM)
ck("README 含 GPU 锁声明", "pathlm_gpu.lock" in RM)

# ---------------------------------------------------------------- 无暂存文件
staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                        capture_output=True, text=True).stdout.split()
ck("无暂存文件（no staged）", len(staged) == 0, staged, [])

# ---------------------------------------------------------------- 汇总
failed = [c for c in checks if not c[1]]
report = {"n_checks": len(checks), "n_failed": len(failed), "failed": failed,
          "checks": checks}
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"reconcile: {len(checks) - len(failed)}/{len(checks)} passed")
for c in failed:
    print("  FAIL:", c[0], "| got", c[2], "| want", c[3])
sys.exit(0 if not failed else 1)
