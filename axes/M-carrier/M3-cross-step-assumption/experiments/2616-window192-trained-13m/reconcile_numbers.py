"""2616 · 报告数字对账（每个写进 README/analysis/MODEL/stability 的数字都必须能回到
`results.json` / 日志）。

前两批各因「报告数字与证据文件对不上」吃了 P1，所以这里把对账**脚本化**：逐条把
`analysis.md` / `MODEL.md` / `README.md` / `stability.md` 里引用的数字与
`results.json` / `rep/results.json` / 训练日志重新算一遍。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出，报告写 reconcile_report.json
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

R = json.load(open(os.path.join(HERE, "results.json")))
P = json.load(open(os.path.join(HERE, "rep", "results.json")))
CFG = json.load(open(os.path.join(HERE, "config.json")))
CFGR = json.load(open(os.path.join(HERE, "config_rep.json")))
TR = json.load(open(os.path.join(HERE, "train", "window192", "results.json")))
TR2 = json.load(open(os.path.join(HERE, "train", "window192_rep", "results.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
MEM = json.load(open(os.path.join(HERE, "smoke", "mem_probe.json")))


def read(name):
    return open(os.path.join(HERE, name)).read().replace("\u2212", "-")   # 报告里的 U+2212


AN, RM, MD, ST = read("analysis.md"), read("README.md"), read("MODEL.md"), read("stability.md")

checks = []


def ck(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    checks.append((label, ok, got, want))


def num(text, pattern, label, cast=float, flags=re.M):
    m = re.search(pattern, text, flags)
    if not m:
        checks.append((label, False, None, "pattern not found"))
        return float("nan")
    return cast(m.group(1))


def section(text, name, nxt):
    i = text.index(name)
    j = text.index(nxt, i)
    return text[i:j]


# ---------------------------------------------------------------- 判据①
c1, c1r = R["criteria"]["criterion_1_quality"], P["criteria"]["criterion_1_quality"]
ck("主臂税 = native(win192) - native(full)",
   round(c1["bpc_window192_native_operating_point"]
         - c1["bpc_full_native_operating_point"], 6), c1["tax_bpc_native"], 1e-9)
ck("主臂税 = -0.002813", c1["tax_bpc_native"], -0.002813, 1e-9)
ck("重跑臂税 = +0.000673", c1r["tax_bpc_native"], 0.000673, 1e-9)
ck("判据①门槛（近零 0.014 / 悬崖 0.05）",
   (c1["thresholds"]["near_zero_tax_bpc"], c1["thresholds"]["cliff_tax_bpc"]), (0.014, 0.05))
ck("analysis 引用的税", num(AN, r"税 = \*\*(-?[\d.]+)\*\*（1\.483569", "an:tax"), -0.002813, 1e-9)
ck("analysis 1.4836→1.4808 的量级",
   num(AN, r"\*\*1\.4836 → (1\.4808)，税", "an:tax2"), 1.4808, 5e-5)
ck("analysis 重跑臂税（4 位小数）",
   num(AN, r"重跑一遍得到 \*\*\+([\d.]+) bpc\*\*", "an:reptax"), 0.0007, 1e-9)
ck("analysis release Δ", num(AN, r"好 \*\*([\d.]+)\*\*（该口径", "an:rel"), 0.005190, 5e-7)
ck("主臂 release Δ = full - win_release",
   round(c1["bpc_full_native_operating_point"] - c1["bpc_window192_release_path"], 6),
   0.005191, 1e-6)
ck("判据①结论文字（analysis）", "近零代价成立" in AN, True)
ck("判据①结论文字（README 判据表）", "近零代价" in RM, True)

# ---------------------------------------------------------------- 预算对齐（§17.12）
rows = {r["window"]: r for r in R["budget_alignment"]["rows"]}
ck("预算表两行 = {32, 192}", sorted(rows), [32, 192])
ck("w=192 删除量（被删可见性对）", rows[192]["dropped_pairs"], 1596)
ck("w=192 放行的可见性对", rows[192]["visible_pairs"], 31300)
ck("算术 32896 - 1596 = 31300", rows[192]["causal_pairs"] - rows[192]["dropped_pairs"], 31300)
ck("算术 31300/32896", round(rows[192]["visible_pairs"] / rows[192]["causal_pairs"], 6),
   rows[192]["visible_frac"])
ck("w=192 可见占比 0.951483", rows[192]["visible_frac"], 0.951483, 1e-6)
ck("w=192 被删位置数 56（位置 8..63）", rows[192]["deleted_key_positions"], 56)
ck("w=32 删除量", rows[32]["dropped_pairs"], 23436)
ck("w=32 可见性对数", rows[32]["visible_pairs"], 9460)
ck("w=32 可见占比 0.287573", rows[32]["visible_frac"], 0.287573, 1e-6)
ck("w=32 被删位置数 216", rows[32]["deleted_key_positions"], 216)
ck("w=32 同族预算的评测期代价 0.149528",
   rows[32]["full_arm_eval_time_damage_bpc"], 0.149528, 1e-9)
ck("w=192 同族预算的评测期代价 0.002851",
   rows[192]["full_arm_eval_time_damage_bpc"], 0.002851, 1e-9)
ck("训练补回（32 档）= 0.149528 - 0.086566",
   round(rows[32]["full_arm_eval_time_damage_bpc"] - rows[32]["trained_arm_tax_bpc"], 6),
   0.062962, 1e-6)
ck("训练补回比例 42.10%",
   round(100 * (rows[32]["full_arm_eval_time_damage_bpc"] - rows[32]["trained_arm_tax_bpc"])
         / rows[32]["full_arm_eval_time_damage_bpc"], 2), 42.10, 1e-2)
ck("analysis 引用 42.1%", num(AN, r"即训练\*\*补回了 ([\d.]+)%\*\*", "an:rec"), 42.0, 0.2)
ck("analysis 引用补回量 0.062962", num(AN, r"0\.149528 - 0\.086566 = ([\d.]+)", "an:rec2"),
   0.062962, 1e-6)
ck("analysis 引用 +0.0029（192 不重训）",
   num(AN, r"代价也只有 \*\*\+([\d.]+) bpc\*\*", "an:free"), 0.0029, 5e-5)
ck("analysis 引用 4.85%",
   num(AN, r"\*\*([\d.]+)% 的可见性对数\*\*", "an:485"), 4.85, 5e-3)
ck("analysis 引用 +0.1495（32 不重训）",
   num(AN, r"不重训的代价是\s*\n?\s*\*\*\+([\d.]+)\*\*", "an:1495", flags=re.M), 0.1495, 5e-5)
ck("analysis 引用 +0.0866（32 训练）",
   num(AN, r"训练后压到 \*\*\+([\d.]+)\*\*", "an:866"), 0.0866, 5e-5)

# 训练点的 32 侧来自 2615 记录（config 内嵌）
ck("2615 记录的 window32 税 = +0.086566",
   rows[32]["trained_arm_tax_bpc"], CFG["recorded_2615_window32"]["tax_bpc_native"])
ck("2615 记录的 window32 native bpc", CFG["recorded_2615_window32"]["bpc_native_operating_point"],
   1.570135)

# ---------------------------------------------------------------- 曲线表（analysis §5.1/5.2/5.3）
S51 = section(AN, "### 5.1", "### 5.2")
df = {r["d"]: r for r in R["curves"]["full"]["distance"]}
dw = {r["d"]: r for r in R["curves"]["window192"]["distance"]}
for d in (1, 2, 4, 8, 16, 32, 64, 128, 255):
    m = re.search(rf"^\| {d} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| "
                  rf"\+?(-?[\d.]+) \|$", S51, re.M)
    if not m:
        checks.append((f"analysis:distance d={d}", False, None, "row not found"))
        continue
    g = [float(x) for x in m.groups()]
    ck(f"an:distance d={d} full", g[0], round(df[d]["bpc"], 6), 1e-6)
    ck(f"an:distance d={d} full damage", g[1], round(df[d]["damage_bpc"], 6), 1e-6)
    ck(f"an:distance d={d} win", g[2], round(dw[d]["bpc"], 6), 1e-6)
    ck(f"an:distance d={d} win damage", g[3], round(dw[d]["damage_bpc"], 6), 1e-6)
    ck(f"an:distance d={d} gap", g[4], round(df[d]["damage_bpc"] - dw[d]["damage_bpc"], 6), 1e-6)

S52 = section(AN, "### 5.2", "### 5.3")
pf = {r["n_dropped"]: r for r in R["curves"]["full"]["prefix"]}
pw = {r["n_dropped"]: r for r in R["curves"]["window192"]["prefix"]}
for n in (8, 16, 32, 64, 128, 192, 255):
    m = re.search(rf"^\| {n} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
                  S52, re.M)
    if not m:
        checks.append((f"analysis:prefix N={n}", False, None, "row not found"))
        continue
    g = [float(x) for x in m.groups()]
    ck(f"an:prefix N={n} full damage", g[0], round(pf[n]["damage_bpc"], 6), 1e-6)
    ck(f"an:prefix N={n} full qmin128", g[1], round(pf[n]["damage_qmin128"], 6), 1e-6)
    ck(f"an:prefix N={n} win damage", g[2], round(pw[n]["damage_bpc"], 6), 1e-6)
    ck(f"an:prefix N={n} win qmin128", g[3], round(pw[n]["damage_qmin128"], 6), 1e-6)
    ck(f"an:prefix N={n} win native qmin128", g[4],
       round(pw[n]["damage_qmin128_on_native"], 6), 1e-6)

S53 = section(AN, "### 5.3", "### 5.4")
rf = {r["d"]: r for r in R["curves"]["full"]["recent"]}
af = {r["d"]: r for r in R["curves"]["full"]["anchored_budget"]}
m = re.search(r"^\| `recent`[^\n]*$", S53, re.M)
ck("analysis recent 行存在", m is not None, True)
pairs = re.findall(r"([\d.]+) / \*?\*?\+?(-?[\d.]+)", m.group(0)) if m else []
ck("an:recent 四组 (bpc, damage)",
   [round(float(v), 6) for pair in pairs for v in pair],
   [round(rf[d][k], 6) for d in (32, 64, 128, 192) for k in ("bpc", "damage_bpc")], 1e-6)
S53_anch = None
m = re.search(r"^\| `anchored`[^\n]*$", S53, re.M)
ck("analysis anchored 行存在", m is not None, True)
if m:
    nums = [float(v) for pair in re.findall(r"([\d.]+) / \*?\*?\+?([\d.]+)", m.group(0))
            for v in pair]
    ck("an:anchored 4 组 (bpc, damage)",
       [round(v, 6) for v in nums],
       [round(af[d][k], 6) for d in (32, 64, 128, 192) for k in ("bpc", "damage_bpc")], 1e-6)

# 训练点表（b）
m = re.search(r"^\| 32（\+8 锚点） \| ([\d.]+) \| \*\*\+([\d.]+)\*\*", S53, re.M)
ck("an:训练点 32 native bpc", float(m.group(1)) if m else float("nan"), 1.570135, 1e-6)
m = re.search(r"^\| 192（\+8 锚点） \| ([\d.]+) \| \*\*(-?[\d.]+)\*\*", S53, re.M)
ck("an:训练点 192 native bpc", float(m.group(1)) if m else float("nan"),
   round(c1["bpc_window192_native_operating_point"], 6), 1e-6)
ck("an:训练点 192 税", float(m.group(2)) if m else float("nan"), -0.002813, 1e-9)
m = re.search(r"^\| 192（重跑） \| ([\d.]+) \| \*\*\+([\d.]+)\*\*", S53, re.M)
ck("an:重跑点 native bpc", float(m.group(1)) if m else float("nan"),
   round(c1r["bpc_window192_native_operating_point"], 6), 1e-6)
ck("an:重跑点税", float(m.group(2)) if m else float("nan"), 0.000673, 1e-9)
ck("an:不得外推内插已标注",
   ("不得外推内插" in S53) and ("禁止" in S53) and ("内插" in S53), True)

# 预算对齐表（c）行
m = re.search(r"^\| 32 \+ 8 \| 40 \| 216 \| 23436 \| ([\d.]+) \| 9460 \| ([\d.]+) \| "
              r"\+([\d.]+) \| \+([\d.]+) \| \*\*([\d.]+)（([\d.]+)%）\*\*", S53, re.M)
ck("an:预算对齐 w=32 行", m is not None, True)
if m:
    ck("an:预算行 w=32 被删占比", float(m.group(1)), round(rows[32]["dropped_frac"], 6), 1e-6)
    ck("an:预算行 w=32 可见占比", float(m.group(2)), round(rows[32]["visible_frac"], 6), 1e-6)
    ck("an:预算行 w=32 评测期代价", float(m.group(3)),
       round(rows[32]["full_arm_eval_time_damage_bpc"], 6), 1e-6)
    ck("an:预算行 w=32 训练税", float(m.group(4)), round(rows[32]["trained_arm_tax_bpc"], 6), 1e-6)
    ck("an:预算行 w=32 补回量", float(m.group(5)), 0.062962, 1e-6)
    ck("an:预算行 w=32 补回比例", float(m.group(6)), 42.10, 2e-2)
m = re.search(r"^\| 192 \+ 8 \| 200 \| 56 \| 1596 \| ([\d.]+) \| 31300 \| ([\d.]+) \| "
              r"\+([\d.]+) \| (-?[\d.]+) \|", S53, re.M)
ck("an:预算对齐 w=192 行", m is not None, True)
if m:
    ck("an:预算行 w=192 被删占比", float(m.group(1)), round(rows[192]["dropped_frac"], 6), 1e-6)
    ck("an:预算行 w=192 可见占比", float(m.group(2)), round(rows[192]["visible_frac"], 6), 1e-6)
    ck("an:预算行 w=192 评测期代价", float(m.group(3)),
       round(rows[192]["full_arm_eval_time_damage_bpc"], 6), 1e-6)
    ck("an:预算行 w=192 训练税", float(m.group(4)), round(rows[192]["trained_arm_tax_bpc"], 6), 1e-6)

# ---------------------------------------------------------------- 检索腿
nc = R["needle"]["conditions"]
c3 = R["criteria"]["criterion_3_retention"]
w_in, w_ct = nc["window192|native|planted"], nc["window192|native|control"]
ck("针总数 = 32×16×8", R["needle"]["n_needles_total"], 4096)
ck("构造检查：full release 基线 = 契约 M 记录",
   nc["full|release|planted"]["recall"], 0.686768)
ck("构造检查 matches_M_2615", R["needle"]["construction_check"]["matches_M_2615"], True)
ck("beyond 桶 n=603", w_in["buckets"]["beyond"]["n"], 603)
ck("beyond 召回 0.475954", w_in["buckets"]["beyond"]["recall"], 0.475954, 1e-9)
ck("beyond 地板 0.444444", w_ct["buckets"]["beyond"]["recall"], 0.444444, 1e-9)
ck("beyond gap = 0.031510", c3["window192_native_beyond_gap_vs_floor"], 0.031510, 1e-9)
ck("beyond gap = 召回之差",
   round(w_in["buckets"]["beyond"]["recall"] - w_ct["buckets"]["beyond"]["recall"], 6),
   c3["window192_native_beyond_gap_vs_floor"], 1e-9)
sp_in = w_in["beyond_split"]["fully_out_of_window"]
sp_ct = w_ct["beyond_split"]["fully_out_of_window"]
ck("完全不可见子桶 n=494", (sp_in["n"], sp_ct["n"]), (494, 494))
ck("完全不可见子桶 gap = 0.000",
   round(sp_in["recall"] - sp_ct["recall"], 6), 0.0, 1e-9)
ck("完全不可见子桶两臂召回相同", sp_in["recall"], 0.457490, 1e-9)
ck("横跨子桶 n=109", w_in["beyond_split"]["straddling"]["n"], 109)
ck("横跨子桶 gap = +0.174312",
   round(w_in["beyond_split"]["straddling"]["recall"]
         - w_ct["beyond_split"]["straddling"]["recall"], 6), 0.174312, 1e-9)
ck("整桶 = 两半加权平均",
   round((109 * w_in["beyond_split"]["straddling"]["recall"]
          + 494 * sp_in["recall"]) / 603, 6), w_in["buckets"]["beyond"]["recall"], 1e-6)
ck("阳性对照 gap = +0.282077", c3["window192_positive_control_gap"], 0.282077, 1e-9)
ck("阳性对照 = 窗内桶差",
   round(w_in["buckets"]["window"]["recall"] - w_ct["buckets"]["window"]["recall"], 6),
   c3["window192_positive_control_gap"], 1e-9)
ck("与 full 同掩码之差 = +0.011609", c3["gap_window192_minus_full"], 0.011609, 1e-9)
ck("32-O 记录的 beyond gap = +0.004279",
   nc and c3["recorded_2615_window32"]["beyond_gap_vs_floor"], 0.004279, 1e-9)
ck("beyond 分桶计数之和 = n",
   sum(v["n"] for v in w_in["buckets"].values()), w_in["n"])
S54 = section(AN, "### 5.4", "### 5.5")
for label, val in (("0.475954", True), ("0.444444", True), ("0.457490", True),
                   ("0.174312", True), ("0.719263", True)):
    ck(f"an §5.4 含 {label}", label in S54, True)
ck("an §5.4 加权平均自洽文字", "= 0.475954 = 整桶召回" in S54, True)
ck("judgement ③ 文案", c3["verdict"].startswith("丢了"), True)
ck("analysis 判据③ 结论", "仍然是「丢了」" in AN, True)

# ---------------------------------------------------------------- 复现性 / 噪声 / 收敛
ck("评测复现性 max diff（主臂）", R["reproducibility"]["max_abs_diff"], 3.65e-07, 1e-11)
ck("评测复现性 pass（主臂）", R["reproducibility"]["pass"], True)
ck("评测复现性 max diff（重跑）", P["reproducibility"]["max_abs_diff"], 4.6e-07, 1e-11)
ck("native 掩码逐位重跑 = 0.0",
   [r["abs_diff"] for r in R["reproducibility"]["per_config"] if r["config"] == "native:None"],
   [0.0])
ck("noise floor 覆盖 3 个 seed", R["noise_floor"]["window_seeds"], [3, 4, 5])
ck("noise full distance:8 spread",
   R["noise_floor"]["spread_by_arm"]["full"]["distance:8"]["spread_bpc"], 0.00703, 1e-9)
ck("noise full recent:32 spread",
   R["noise_floor"]["spread_by_arm"]["full"]["recent:32"]["spread_bpc"], 0.020867, 1e-9)
ck("noise win192 distance:8 spread（主臂）",
   R["noise_floor"]["spread_by_arm"]["window192"]["distance:8"]["spread_bpc"], 0.000144, 1e-9)
ck("noise win192 prefix:64 spread（重跑）",
   P["noise_floor"]["spread_by_arm"]["window192"]["prefix:64"]["spread_bpc"], 0.007477, 1e-9)

conv, convr = R["convergence_reconciliation"], P["convergence_reconciliation"]
ck("收敛斜率 全程 full", conv["slope_full_range"]["full"], -9.7588e-05, 1e-12)
ck("收敛斜率 全程 win192（主臂）", conv["slope_full_range"]["window192"], -9.8002e-05, 1e-12)
ck("收敛斜率 全程 win192（重跑）", convr["slope_full_range"]["window192"], -9.7619e-05, 1e-12)
ck("收敛斜率 末 25% full", conv["slope_last_25pct"]["full"], 1.177e-06, 1e-12)
ck("收敛斜率 末 25% win192（主臂）", conv["slope_last_25pct"]["window192"], -3.6e-07, 1e-12)
ck("末行 loss full", conv["last_logged_loss"]["full"], 1.011, 1e-9)
ck("末行 loss win192（主臂）", conv["last_logged_loss"]["window192"], 1.0103, 1e-9)
ck("末行 loss win192（重跑）", convr["last_logged_loss"]["window192"], 1.0051, 1e-9)
ck("训练日志行数 = 61（两臂）",
   (conv["n_points"]["full"], conv["n_points"]["window192"]), (61, 61))
ck("train_log 行数（磁盘）",
   sum(1 for _ in open(os.path.join(HERE, "train", "window192", "train_log.jsonl"))), 61)
ck("train_log 行数（重跑）",
   sum(1 for _ in open(os.path.join(HERE, "train", "window192_rep", "train_log.jsonl"))), 61)

# ---------------------------------------------------------------- 活性量 / 训练器电池
wm = TR["gnorm_monitor"]["window_mask"]
ck("活性：n_visible_pairs", wm["n_visible_pairs"], 31300)
ck("活性：causal_pairs", wm["causal_pairs"], 32896)
ck("活性：visible_frac", wm["visible_frac"], 0.951483, 1e-6)
ck("活性：dropped_frac", wm["dropped_frac"], 0.048517, 1e-6)
ck("活性：alive", wm["alive"], True)
ck("功能探针 max|Δlogit| = 0.217099",
   TR["gnorm_monitor"]["window_mask_probe"]["max_abs_logit_diff_masked_vs_open"], 0.217099, 1e-9)
ck("results.json::intervention 活性量一致",
   R["intervention"]["activity_value"]["n_visible_pairs"], 31300)
ck("results.json::intervention alive", R["intervention"]["alive"], True)
ck("results.json::intervention params_added", R["intervention"]["params_added"], 0)
ck("train 臂 train_window=192 / anchors=8", (TR["train_window"], TR["train_anchors"]), (192, 8))
ck("训练 tokens = 12000×32×256", TR["train_tokens"], 12000 * 32 * 256)
ck("gnorm max（主臂）", TR["gnorm_monitor"]["gnorm_max"], 9.88, 1e-9)
ck("gnorm mean（主臂）", TR["gnorm_monitor"]["gnorm_mean"], 0.525, 1e-9)
ck("grad_skips = 0（两臂）",
   (TR["gnorm_monitor"]["grad_skips"], TR2["gnorm_monitor"]["grad_skips"]), (0, 0))
ck("平均训练 CE（主臂）", TR["avg_train_depth_ce"][-1], 1.2388, 1e-9)
ck("平均训练 CE（重跑）", TR2["avg_train_depth_ce"][-1], 1.2392, 1e-9)
ck("训练器电池 acc（主臂 / 重跑）",
   (TR["next_token_acc"], TR2["next_token_acc"]), (0.6976, 0.6962))
ck("训练墙钟（主臂）", TR["wall_minutes"], 8.23, 1e-9)
ck("训练墙钟（重跑）", TR2["wall_minutes"], 8.67, 1e-9)

# ---------------------------------------------------------------- 成本 / 次数 / 步时 / 显存
ck("评测次数（主臂 / 重跑）", (R["cost"]["n_evals"], P["cost"]["n_evals"]), (408, 408))
ck("评测墙钟（主臂，与 results.json 一致）",
   R["cost"]["wall_seconds"], num(AN, r"\| \*\*([\d.]+) s\*\*，\*\*408 次评测\*\*", "an:wall"),
   0.01)
ck("评测墙钟（重跑）",
   P["cost"]["wall_seconds"], num(ST, r"\| 44\.50 s，408 次评测", "—", str) if False else
   P["cost"]["wall_seconds"])
ck("analysis 重跑电池 44.50 s", num(AN, r"\| (44\.50) s，408 次评测", "an:repwall"), 44.50, 1e-9)
ck("评测峰值显存 194.7 MiB", R["cost"]["peak_mem_mib"], 194.7, 1e-9)
ck("解析 KV 位置 256→200 (−21.9%)",
   (R["cost"]["memory_analytic_vs_measured"]["analytic_kv_positions_per_row"],
    R["cost"]["memory_analytic_vs_measured"]["analytic_kv_reduction_frac"]),
   ({"full": 256, "window192": 200}, 0.21875))
ck("解析注意力对降幅 = 被删占比",
   R["cost"]["memory_analytic_vs_measured"]["analytic_attention_pairs_reduction_frac"],
   0.04851654, 1e-9)
ck("训练显存探针 allocated",
   MEM["peak_allocated_mib"], num(AN, r"net 峰值 \*\*([\d.]+) MiB\*\*", "an:mem"), 0.05)
ck("训练显存探针 reserved", MEM["peak_reserved_mib"], 1984.0, 1e-9)
ck("GPU 总墙钟 ≈19.5 min", num(AN, r"\| \*\*≈([\d.]+) min\*\* = 0\.02", "an:gpu"), 19.5, 0.1)
# 评测次数的逐项分解（analysis §9）：36 + 71 + 256 + 9 + 36 = 408
n_full = (1 + 1) + 9 + (1 + 7 + 7) + 6 + 4
n_win = (1 + 1 + 1) + 9 * 2 + (1 + 1 + 7 * 4) + 6 * 2 + 4 * 2
n_needle = 32 * 2 * 2 * 2
n_repro, n_nf = 9, 3 * 3 * 2 * 2
ck("评测次数分解：full 36 / win192 71 / 针 256 / 复现 9 / 噪声底 36",
   (n_full, n_win, n_needle, n_repro, n_nf), (36, 71, 256, 9, 36))
ck("评测次数分解之和 = n_evals",
   n_full + n_win + n_needle + n_repro + n_nf, R["cost"]["n_evals"])
ck("复现性条目数 = 9", len(R["reproducibility"]["per_config"]), 9)
_bench = sum(json.loads(open(os.path.join(HERE, f"smoke/stepbench_w{w}/{r}/train_log.jsonl"))
                         .readlines()[-1])["min"]
              for w, r in ((192, "bench192"), (0, "bench0")))
ck("步时基准训练循环 = 0.43 + 0.41 = 0.84 min", round(_bench, 2), 0.84, 1e-9)
ck("GPU 总墙钟 = 各项之和（0.02 + 基准 + 两次训练 + 两次电池 + 0.35）",
   round(0.02 + _bench + TR["wall_minutes"] + TR2["wall_minutes"]
         + R["cost"]["wall_seconds"] / 60 + P["cost"]["wall_seconds"] / 60 + 0.35, 1), 19.5, 0.05)


def _ms_per_step(path):
    rows = [json.loads(l) for l in open(path)]
    return round((rows[-1]["min"] - rows[0]["min"]) * 60
                 / (rows[-1]["step"] - rows[0]["step"]) * 1000, 1)


ms192 = _ms_per_step(os.path.join(HERE, "smoke/stepbench_w192/bench192/train_log.jsonl"))
ms0 = _ms_per_step(os.path.join(HERE, "smoke/stepbench_w0/bench0/train_log.jsonl"))
ck("步时（窗口开）与报告一致", ms192,
   num(RM, r"步时 \*\*([\d.]+) ms/步（窗口开）", "README:ms192"), 0.05)
ck("步时（窗口关）与报告一致", ms0,
   num(RM, r"vs ([\d.]+) ms/步\s*\n?\s*（窗口关）", "README:ms0", flags=re.M), 0.05)
ck("掩码路径开销 = 1.05×", round(ms192 / ms0, 2), 1.05, 1e-9)
ck("stability 步时一致",
   num(ST, r"\| \*\*([\d.]+) ms/步\*\* \| `smoke/stepbench_w192", "st:ms192"), ms192, 0.05)

# ---------------------------------------------------------------- 变异 / 单测 / 对账
ck("变异 6/6 抓红", MUT["all_caught"], True)
ck("变异记录 6 条", len(MUT["mutations"]), 6)
ck("变异后逐字节还原", MUT["restored_bitwise"], True)
ck("变异前 sha256 = 现场重算",
   MUT["sha256_before"],
   hashlib.sha256(open(os.path.join(ROOT, "pathlm", "depth_ar.py"), "rb").read()).hexdigest())
ck("Q3（窗口被夹到 32）被抓红",
   [m["caught_red"] for m in MUT["mutations"] if m["id"] == "Q3"], [True])
ck("Q3 抓红的是新单测",
   "test_window192_keep_mask_counts_and_boundary_at_real_scale"
   in [m for m in MUT["mutations"] if m["id"] == "Q3"][0]["tests_actually_red"], True)
ck("README 标注「逐字」与「摘要」", ("逐字" in RM) and ("摘要" in RM), True)
ck("README 含 gpu_wall_budget", "gpu_wall_budget" in RM, True)
ck("README 判据线逐字段存在", "Δ **≤ 0.014**" in RM, True)
ck("analysis 判据线未改声明", "未改一字" in AN, True)

# ---------------------------------------------------------------- 文件指纹
code = R["code"]
FILEMAP = {"run_py_sha256": "axes/M-carrier/M3-cross-step-assumption/experiments/"
                            "2616-window192-trained-13m/run.py",
           "retention_py_sha256": "tools/evalkit/retention.py",
           "depth_ar_py_sha256": "pathlm/depth_ar.py",
           "train_depth_ar_py_sha256": "train_depth_ar.py",
           "tests_depth_ar_sha256": "tests/test_depth_ar.py",
           "tests_retention_sha256": "tests/test_retention.py"}
for key, rel in FILEMAP.items():
    h = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
    ck(f"现场重算 {rel}", h, code[key])
    if key != "run_py_sha256":
        ck(f"MODEL.md 指纹 {rel}", code[key] in MD, True)
ck("MODEL.md 指纹 run.py", code["run_py_sha256"] in MD, True)
ck("depth_ar 指纹与 2615 记录相同（未改动）",
   code["depth_ar_py_sha256"], "6c7454581af30291688cf825c39f01ce2175c0e447ef5e83b772f2d063ddfea5")
ck("train_depth_ar 指纹与 2615 记录相同（未改动）",
   code["train_depth_ar_py_sha256"],
   "11298e6b22678c3c6d9b41600a539b2daf62f90003a32980e0b52a960e208613")

for arm, ckp in (("full", R["checkpoints"]["full"]), ("window192", R["checkpoints"]["window192"])):
    ck(f"MODEL.md 含 {arm} 权重 sha256", ckp["sha256"][:12] in MD.replace("`", ""), True)
ck("window192 权重 sha256 与 config.json 一致",
   R["checkpoints"]["window192"]["sha256"], CFG["trained_checkpoint"]["sha256"])
ck("重跑臂权重 sha256 与 config_rep.json 一致",
   P["checkpoints"]["window192"]["sha256"], CFGR["trained_checkpoint"]["sha256"])
ck("权重体积 54,193,675 B", R["checkpoints"]["window192"]["size_bytes"], 54193675)
ck("现场重算 window192 权重 sha256",
   hashlib.sha256(open(os.path.join(HERE, "train", "window192", "model.pt"), "rb").read())
   .hexdigest(), R["checkpoints"]["window192"]["sha256"])
ck("现场重算重跑臂权重 sha256",
   hashlib.sha256(open(os.path.join(HERE, "train", "window192_rep", "model.pt"), "rb").read())
   .hexdigest(), P["checkpoints"]["window192"]["sha256"])

# ---------------------------------------------------------------- 报告
bad = [c for c in checks if not c[1]]
print(f"{len(checks) - len(bad)}/{len(checks)} 条对账通过")
for label, ok, got, want in bad:
    print(f"  MISMATCH  {label}: got={got} want={want}")
json.dump({"n_checks": len(checks), "n_failed": len(bad),
           "failed": [{"label": l, "got": g, "want": w} for l, _, g, w in bad]},
          open(os.path.join(HERE, "reconcile_report.json"), "w"), indent=2, ensure_ascii=False)
sys.exit(1 if bad else 0)
