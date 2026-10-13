"""2615 · 报告数字对账（每个写进 README/analysis/MODEL 的数字都必须能回到 results.json）。

上一批因「报告数字与 results.json 对不上」吃了 2 个 P1，所以这里把对账**脚本化**：
逐条把 `analysis.md` / `MODEL.md` 里引用的数字与 `results.json` / 训练日志重新算一遍。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

R = json.load(open(os.path.join(HERE, "results.json")))
TR = json.load(open(os.path.join(HERE, "train", "window32", "results.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
AN = open(os.path.join(HERE, "analysis.md")).read()
MD = open(os.path.join(HERE, "MODEL.md")).read()
RM = open(os.path.join(HERE, "README.md")).read()

checks = []


def ck(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    checks.append((label, ok, got, want))


def num(text, pattern, label, cast=float):
    m = re.search(pattern, text, re.M)
    if not m:
        checks.append((label, False, None, "pattern not found"))
        return float("nan") if cast is not str else ""
    s = m.group(1).replace("\u2212", "-")          # 报告里用 U+2212 减号
    return cast(s) if cast is not str else s


def section(text, name, nxt):
    """取 analysis.md 里某一小节正文（避免同表号跨节误匹配）。"""
    i = text.index(name)
    j = text.index(nxt, i)
    return text[i:j]


S51 = section(AN, "### 5.1", "### 5.2")
S52 = section(AN, "### 5.2", "### 5.3")
S53 = section(AN, "### 5.3", "### 5.4")


# ---- 判据 ----
c1 = R["criteria"]["criterion_1_quality"]
ck("analysis 税 1.483569→1.570135 Δ−0.086566",
   round(c1["bpc_full_native_operating_point"] - c1["bpc_window32_native_operating_point"], 6),
   -0.086566)
ck("analysis 引用的 Δ 与 results 一致",
   num(AN, r"Δ = \*\*(−?[\d.]+)\*\*；release", "analysis:crit1 delta", str),
   "-0.086566")
ck("判据① release Δ", c1["delta_bpc_release"], -0.223798)
c2 = R["criteria"]["criterion_2_dependence_shift"]
g = c2["band"]["gap"]
ck("判据② 主口径均值", c2["band"]["mean_gap"], 0.000785)
ck("判据② 主口径均值 = 三项算术平均",
   round(sum(g.values()) / len(g), 6), c2["band"]["mean_gap"])
ck("判据② 工作点口径均值", c2["secondary_native_reference"]["mean_gap"], 0.013298)
gn = c2["secondary_native_reference"]["gap"]
ck("判据② 工作点口径 = 三项算术平均",
   round(sum(gn.values()) / len(gn), 6), c2["secondary_native_reference"]["mean_gap"])
ck("analysis 引用的 d=8 主口径 gap",
   num(S51, r"^\| \*\*8\*\* \|[^\n]*\| \*\*(−?[\d.]+)\*\* \|", "an:d8", float),
   g["8"], 2e-4)
ck("analysis 引用的 d=8 工作点 gap",
   num(S51, r"\| 8 \| 0\.089877 \| 0\.060010 \| \*\*\+([\d.]+)\*\*",
       "an:native8", float), gn["8"], 2e-4)
c3 = R["criteria"]["criterion_3_retention"]
ck("判据③ beyond gap", c3["window32_native_beyond_gap_vs_floor"], 0.004279)
ck("判据③ beyond 算术", round(c3["window32_native_beyond_recall"]["recall"]
                              - c3["window32_native_beyond_control"]["recall"], 6),
   c3["window32_native_beyond_gap_vs_floor"])

# ---- 曲线（analysis §5.1/§5.2 的表逐行回算）----
df = {r["d"]: r for r in R["curves"]["full"]["distance"]}
dw = {r["d"]: r for r in R["curves"]["window32"]["distance"]}
for d in (1, 2, 4, 8, 16, 32, 64, 128, 255):
    m = re.search(rf"^\| \*?\*?{d}\*?\*? \| ([\d.]+) \| (\*?\*?−?[\d.]+).*?\| ([\d.]+) \| (\*?\*?−?[\d.]+)",
                  S51, re.M)
    if not m:
        checks.append((f"analysis:distance d={d}", False, None, "row not found"))
        continue
    ck(f"analysis:distance d={d} full bpc", float(m.group(1)), round(df[d]["bpc"], 4), 2e-4)
    ck(f"analysis:distance d={d} window32 bpc", float(m.group(3)), round(dw[d]["bpc"], 4), 2e-4)
pf = {r["n_dropped"]: r for r in R["curves"]["full"]["prefix"]}
pw = {r["n_dropped"]: r for r in R["curves"]["window32"]["prefix"]}
PREFIX6 = (r"^\| {n} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| "
           r"([\u2212\d.]+) \| \*?\*?([\d.]+)\*?\*? \|")
for n in (8, 16, 32, 64, 128, 192, 255):
    def _cell(idx):
        m = re.search(PREFIX6.format(n=n), S52, re.M)
        if not m:
            checks.append((f"an:prefix {n}", False, None, "row not found"))
            return float("nan")
        return float(m.group(idx).replace("\u2212", "-"))
    ck(f"prefix:N={n} full qmin128", round(pf[n]["damage_qmin128"], 4), _cell(2), 2e-4)
    ck(f"prefix:N={n} win release/all-query", round(pw[n]["damage_bpc"], 4), _cell(3), 2e-4)
    ck(f"prefix:N={n} win release qmin128", round(pw[n]["damage_qmin128"], 4), _cell(4), 2e-4)
    ck(f"prefix:N={n} win native qmin128", round(pw[n]["damage_qmin128_on_native"], 4),
       _cell(5), 2e-4)
# 「N ≤ 64 时 0.0118~0.0128（免费）」区间自洽
vals = [pw[n]["damage_qmin128_on_native"] for n in (8, 16, 32, 64)]
ck("analysis 声明的 0.0118~0.0128 区间", (round(min(vals), 4), round(max(vals), 4)),
   (0.0118, 0.0128), 0.0)

# ---- 检索腿 ----
nc = R["needle"]["conditions"]
ck("针总数 = 32×16×8", R["needle"]["n_needles_total"], 32 * 16 * 8)
ck("构造检查：full release 基线 = 契约 M 记录",
   nc["full|release|planted"]["recall"], 0.686768)
ck("构造检查 matches_M_2615", R["needle"]["construction_check"]["matches_M_2615"], True)
for k, label in (("window32|native|planted", "win native"), ("window32|native|control", "win ctrl"),
                 ("full|native|planted", "full native"), ("window32|release|planted", "win release")):
    ck(f"needle {label} recall 在 [0,1]", 0.0 <= nc[k]["recall"] <= 1.0, True)
ck("beyond 分桶计数 = n", sum(v["n"] for v in
                             nc["window32|native|planted"]["buckets"].values()),
   nc["window32|native|planted"]["n"])
ck("难度子集 138/2250",
   round(nc["window32|native|planted"]["hard_subset"]["recall"]
         * nc["window32|native|planted"]["hard_subset"]["n"]), 138.0, 1.0)

# ---- 复现性 / 噪声 / 成本 ----
ck("评测复现性 max diff", R["reproducibility"]["max_abs_diff"], 4.31e-07, 1e-9)
ck("复现性 pass", R["reproducibility"]["pass"], True)
ck("噪声底 full distance:8 spread", R["noise_floor"]["spread_by_arm"]["full"]["distance:8"]["spread_bpc"],
   0.00703)
ck("噪声底 window32 distance:8 spread",
   R["noise_floor"]["spread_by_arm"]["window32"]["distance:8"]["spread_bpc"], 0.01248)
ck("评测次数", R["cost"]["n_evals"], 396)
ck("评测墙钟(s)", R["cost"]["wall_seconds"], 41.95)
ck("峰值显存(MiB)", R["cost"]["peak_mem_mib"], 194.7)
ck("训练墙钟(min)", TR["wall_minutes"], 7.62)
ck("训练 tokens", TR["train_tokens"], 12000 * 32 * 256)
ck("训练 step 数逐行 = 61", sum(1 for _ in open(os.path.join(
    HERE, "train", "window32", "train_log.jsonl"))), 61)
ck("掩码活性 visible_frac", TR["gnorm_monitor"]["window_mask"]["visible_frac"], 0.287573)
ck("掩码活性 9460/32896", round(TR["gnorm_monitor"]["window_mask"]["n_visible_pairs"]
                                / TR["gnorm_monitor"]["window_mask"]["causal_pairs"], 6), 0.287573)
ck("掩码功能探针 > 0", TR["gnorm_monitor"]["window_mask_probe"]["max_abs_logit_diff_masked_vs_open"] > 0, True)
ck("梯度跳过数 = 0", TR["gnorm_monitor"]["grad_skips"], 0)

# ---- 对账 / 仪器 ----
ck("full 臂 vs 2614 max|Δ| = 0.0", R["reconciliation_vs_2614"]["max_abs_diff_bpc"], 0.0)
ck("训练掩码 == 评测掩码", R["instrument"]["train_mask_equals_eval_mask"], True)
ck("release == identity（两臂）",
   R["instrument"]["full"]["bitwise_equal"] and R["instrument"]["window32"]["bitwise_equal"], True)

# ---- 变异验证 ----
ck("变异 5/5 抓红", MUT["all_caught"], True)
ck("变异 6 条记录", len(MUT["mutations"]), 6)
ck("变异后逐字节还原", MUT["restored_bitwise"], True)
ck("M4 未被抓红（冗余防线，如实登记）",
   [m["caught_red"] for m in MUT["mutations"] if m["id"] == "M4"], [False])

# ---- MODEL.md / README.md 的哈希与体积 ----
code = R["code"]
for key, mdkey in (("run_py_sha256", "`run.py`"), ("depth_ar_py_sha256", "`pathlm/depth_ar.py`"),
                   ("train_depth_ar_py_sha256", "`train_depth_ar.py`"),
                   ("tests_depth_ar_sha256", "`tests/test_depth_ar.py`"),
                   ("retention_py_sha256", "`tools/evalkit/retention.py`"),
                   ("tests_retention_sha256", "`tests/test_retention.py`")):
    ck(f"MODEL.md 指纹 {mdkey} 与 results.json 一致", code[key] in MD, True)
    # 现场重算（run.py 自身不重算以免递归）
    fmap = {"run_py_sha256": "axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/"
                             "2615-window-trained-retention-13m/run.py",
            "retention_py_sha256": "tools/evalkit/retention.py",
            "depth_ar_py_sha256": "pathlm/depth_ar.py",
            "train_depth_ar_py_sha256": "train_depth_ar.py",
            "tests_depth_ar_sha256": "tests/test_depth_ar.py",
            "tests_retention_sha256": "tests/test_retention.py"}[key]
    if key != "run_py_sha256":
        h = hashlib.sha256(open(os.path.join(ROOT, fmap), "rb").read()).hexdigest()
        ck(f"现场重算 {fmap}", h, code[key])

for arm in ("full", "window32"):
    ck(f"MODEL.md 含 {arm} 权重 sha256",
       R["checkpoints"][arm]["sha256"][:12] in MD.replace("`", ""), True)
ck("window32 权重 sha256 与 config 一致",
   hash_ok := (R["checkpoints"]["window32"]["sha256"] == json.load(
       open(os.path.join(HERE, "config.json")))["trained_checkpoint"]["sha256"]), True)
ck("权重体积 54,193,675 B", R["checkpoints"]["window32"]["size_bytes"], 54193675)

# ---- 契约副本标注（README）----
ck("README 标注「逐字」", ("逐字" in RM) and ("摘要" in RM), True)
ck("README 含 gpu_wall_budget", "gpu_wall_budget" in RM, True)

# ---- 复现臂（rep/）与 stability.md ----
RB = json.load(open(os.path.join(HERE, "rep", "results.json")))
TR2 = json.load(open(os.path.join(HERE, "train", "window32_rep", "results.json")))
ST = open(os.path.join(HERE, "stability.md")).read()
ck("重跑臂 native bpc", RB["criteria"]["criterion_1_quality"]["bpc_window32_native_operating_point"],
   1.565637)
ck("重跑臂 release bpc", RB["criteria"]["criterion_1_quality"]["bpc_window32_release_path"],
   1.641257)
ck("重跑臂税", RB["criteria"]["criterion_1_quality"]["delta_bpc_native"], -0.082069)
ck("重跑臂判据②均值", RB["criteria"]["criterion_2_dependence_shift"]["band"]["mean_gap"],
   0.001402)
ck("重跑臂判据③ gap", RB["criteria"]["criterion_3_retention"]["window32_native_beyond_gap_vs_floor"],
   -0.004565)
ck("重跑臂 wall(min)", TR2["wall_minutes"], 8.27)
ck("重跑臂评测墙钟(s)", RB["cost"]["wall_seconds"], 27.81)
ck("两次训练 native bpc 之差 0.0045",
   round(abs(R["criteria"]["criterion_1_quality"]["bpc_window32_native_operating_point"]
             - RB["criteria"]["criterion_1_quality"]["bpc_window32_native_operating_point"]), 4),
   0.0045, 1e-4)
ck("两次训练 release bpc 之差 0.0661",
   round(abs(R["criteria"]["criterion_1_quality"]["bpc_window32_release_path"]
             - RB["criteria"]["criterion_1_quality"]["bpc_window32_release_path"]), 4),
   0.0661, 1e-4)
ck("MODEL.md 含重跑臂 sha256", TR2 is not None and
   "4890194e9b71e9b411827ceb4b55710a3cbfae52afe4c31eb23aa788ad124800"[:12] in MD, True)
ck("stability.md 引用重跑臂结论一致", ("−0.082069" in ST or "0.0821" in ST) and "0.004279" in ST, True)
ck("stability.md 含 400 步探针数字", "2.5886" in ST and "2.5841" in ST, True)

# ---- 步时实测（smoke/stepbench_w{0,32}）与报告数字一致 ----
def _ms_per_step(path):
    rows = [json.loads(l) for l in open(path)]
    return round((rows[-1]["min"] - rows[0]["min"]) * 60 /
                 (rows[-1]["step"] - rows[0]["step"]) * 1000, 1)

ms32 = _ms_per_step(os.path.join(HERE, "smoke/stepbench_w32/bench32/train_log.jsonl"))
ms0 = _ms_per_step(os.path.join(HERE, "smoke/stepbench_w0/bench0/train_log.jsonl"))
ck("步时（窗口开）与报告一致", ms32,
   num(RM, r"步时 ([\d.]+) ms/步（窗口开）", "README:ms32", float), 0.1)
ck("步时（窗口关）与报告一致", ms0,
   num(RM, r"vs ([\d.]+) ms/步（窗口关）", "README:ms0", float), 0.1)
ck("掩码路径开销 = 1.03×", round(ms32 / ms0, 2), 1.03, 1e-9)
ck("analysis 步时与 README 一致", num(AN, r"\| 掩码路径开销 \| ([\d.]+) ms/step", "an:ms", float),
   ms32, 0.1)

# ---- 报告 ----
bad = [c for c in checks if not c[1]]
print(f"{len(checks) - len(bad)}/{len(checks)} 条对账通过")
for label, ok, got, want in bad:
    print(f"  MISMATCH  {label}: got={got} want={want}")
json.dump({"n_checks": len(checks), "n_failed": len(bad),
           "failed": [{"label": l, "got": g, "want": w} for l, _, g, w in bad]},
          open(os.path.join(HERE, "reconcile_report.json"), "w"), indent=2, ensure_ascii=False)
sys.exit(1 if bad else 0)
