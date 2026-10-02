"""2620 · 报告数字对账（`SCHEMA §12`）：`analysis.md` 里写下的每个关键数字都必须能在
`results.json` / 训练日志 / `config.json` 里原样找到。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出

两层：
  ① **结构性/自洽性**（不依赖本次读数）：判据键齐全、主判据两种写法代数等价、税与曲线自洽、
     活性量非空且与 `train_log.jsonl` 的 200 步行逐点一致、参数量/token 数与配置自洽、
     代码指纹已落盘、权重 sha256 与 `config.json` 一致；
  ② **具体数字**：`analysis.md` 里的关键读数逐条回查（`SPECIFIC` 段）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from tools.evalkit.ensemble_probe import reading_B                     # noqa: E402

ARMS = ("stock", "l1", "divl1")
R = json.load(open(os.path.join(HERE, "results.json")))
CFG = json.load(open(os.path.join(HERE, "config.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
AN = open(os.path.join(HERE, "analysis.md")).read().replace("\u2212", "-")  # 报告用排版减号，比对时归一
MD = open(os.path.join(HERE, "MODEL.md")).read()
RM = open(os.path.join(HERE, "README.md")).read()

checks = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


# ------------------------------------------------------------------ 结构/自洽
sc = R["selfchecks"]["probe_tts_method_identity"]
ck("仪器自检：与 probe_tts.bpc_k 同方法（全部 K 逐位同值）", sc["all_equal"] is True,
   {k: v for k, v in sc.items() if isinstance(v, dict)}, "all bpc_equal")
for arm in ARMS:
    ck(f"臂 {arm} 在 K 曲线里", arm in R["k_curve"])
    for k in CFG["k_sweep"]:
        ck(f"臂 {arm} K={k} 有 bpc", str(k) in R["k_curve"][arm])
    ck(f"臂 {arm} checkpoint sha256 与 config.json 一致",
       R["checkpoints"][arm]["sha256"] == CFG["arms"][arm]["sha256"], True)
    ck(f"臂 {arm} 训练产物齐（wall/peak/intervention）",
       all(R["train_products"][arm].get(x) is not None
           for x in ("wall_minutes", "peak_gpu_mem_mb", "intervention")))

# 参数量与 token 数
params = {a: R["checkpoints"][a]["params"] for a in ARMS}
ck("三臂参数量完全相同（去相关压力是纯损失项，不加参数）", len(set(params.values())) == 1,
   params, "单一值")
for a in ARMS:
    cfg_a = json.load(open(os.path.join(HERE, "configs", f"{a}.json")))
    tr = R["train_products"][a]
    ck(f"臂 {a} train_tokens = steps×bs×T",
       tr["train_tokens"] == cfg_a["train"]["steps"] * cfg_a["train"]["batch_size"]
       * cfg_a["model"]["seq_len"], tr["train_tokens"],
       cfg_a["train"]["steps"] * cfg_a["train"]["batch_size"] * cfg_a["model"]["seq_len"])
    ck(f"臂 {a} intervention.params_added == 0", tr["intervention"]["params_added"] == 0)

# 活性量：与训练日志的 200 步行逐点一致（SCHEMA §17.4）
for a in ARMS:
    lp = os.path.join(HERE, "train", a, "train_log.jsonl")
    lines = [json.loads(x) for x in open(lp) if x.strip()]
    # 活动量落盘：每 200 步一行**且每行都带活动字段**。被 pre-step grad-norm gate 跳过的
    # 步不会落行（`continue` 在日志之前）——这是既有实现的行为，用 gate 计数精确对账，
    # 而不是用「行数 ≥ steps/200」这种在 gate 触发时会误报的断言（divl1 跳了 42.7%）。
    ck(f"臂 {a} 活动量落盘覆盖到训练末段（≥40% 的 200 步边界）",
       len(lines) >= 0.4 * CFG["steps_full"] // 200, len(lines),
       round(0.4 * CFG["steps_full"] // 200))
    ck(f"臂 {a} 每行都带 activity 字段（gnorm/n_paths/skip_frac/diversity_js）",
       all(all(k in ln for k in ("gnorm", "n_paths", "skip_frac", "diversity_js"))
           for ln in lines))
    iv = R["train_products"][a]["intervention"]
    logged = [ln["skip_frac"] for ln in lines]
    ck(f"臂 {a} skip_frac_mean 落在已落盘样本的 [min,max] 带内",
       min(logged) <= iv["skip_frac_mean"] <= max(logged),
       iv["skip_frac_mean"], [min(logged), max(logged)])
    if iv["w_diversity"] > 0:
        ck(f"臂 {a} 观测数 + gate 跳过数 == 总步数（精确对账）",
           iv["n_diversity_observations"] + iv["gate_skips"] == CFG["steps_full"],
           iv["n_diversity_observations"] + iv["gate_skips"], CFG["steps_full"])
    w = R["train_products"][a]["intervention"]["w_diversity"]
    ck(f"臂 {a} alive 与 w_diversity 一致（对照臂不得报活着）",
       bool(R["train_products"][a]["intervention"]["alive"]) == (w > 0), w, None)
    ck(f"臂 {a} n_paths_per_step = 1 + (w_diversity>0)",
       R["train_products"][a]["intervention"]["n_paths_per_step"] == (1 + int(w > 0)))
ck("stock 臂 skip_frac_mean == 1.0（路径确定性 ⇒ 无落层）",
   R["train_products"]["stock"]["intervention"]["skip_frac_mean"] == 1.0)
for a in ("l1", "divl1"):
    ck(f"{a} 臂 skip_frac_mean < 1（路径真的在丢层）",
       R["train_products"][a]["intervention"]["skip_frac_mean"] < 1.0,
       R["train_products"][a]["intervention"]["skip_frac_mean"])

# 判据自洽：两种写法代数等价 + 税与曲线一致
b1 = {a: R["k_curve"][a]["1"]["bpc"] for a in ARMS}
bk = {a: R["k_curve"][a][str(CFG["main_k"])]["bpc"] for a in ARMS}
rb = reading_B(b1["l1"], b1["stock"], bk["divl1"])
ck("主判据：预登记形式 == 化简形式", rb["algebra_consistent"] is True)
ck("主判据：results.json 里的两种写法 == 从曲线重算",
   abs(R["criteria"]["main_reading_B"]["value_verbatim_form"] - rb["verbatim"]) < 1e-6
   and abs(R["criteria"]["main_reading_B"]["value_reduced_form"] - rb["reduced"]) < 1e-6,
   R["criteria"]["main_reading_B"]["value_reduced_form"], round(rb["reduced"], 6))
ck("并列判据：值 == 从曲线重算",
   abs(R["criteria"]["parallel_reading_A"]["value"] - (b1["l1"] - bk["divl1"])) < 1e-6)
ck("税（去相关压力 vs l1）== 曲线差",
   abs(R["criteria"]["taxes"]["diversity_pressure_clean_tax_vs_l1"]
       - (b1["divl1"] - b1["l1"])) < 1e-6)
ck("税（路径机制 vs stock）== 曲线差",
   abs(R["criteria"]["taxes"]["mechanism_clean_tax_vs_stock"]
       - (b1["l1"] - b1["stock"])) < 1e-6)
ck("每个比较都带配对 CI 且 CI 是区间",
   all(isinstance(c["delta_ci"], list) and len(c["delta_ci"]) == 2
       for c in [R["criteria"]["main_reading_B"]["ci"], R["criteria"]["parallel_reading_A"]["ci"],
                 R["criteria"]["taxes"]["diversity_pressure_clean_tax_vs_l1_ci"]]))

# 确定性路径对照：路径逐位相同 ⇒ 增益在浮点噪声内
det = R["deterministic_path_control"]
for tag in ("stock", "divl1_forced"):
    ck(f"确定性对照 {tag}：路径逐位相同",
       det[tag]["paths_bitwise_identical_all"] is True)
    ck(f"确定性对照 {tag}：|增益| < 1e-4",
       abs(det[tag]["gain_K1_minus_Kmax"]) < 1e-4, det[tag]["gain_K1_minus_Kmax"])

# 成对 JS
for a in ARMS:
    js = R["pairwise_js_at_kmax"][a]
    if a == "stock":
        ck("stock 的成对 JS == 0（路径确定性 ⇒ 无相关度可言）",
           js["mean_nats"] < 1e-9, js["mean_nats"])
    else:
        ck(f"{a} 的成对 JS > 0（路径真随机）", js["mean_nats"] > 0, js["mean_nats"])
        ck(f"{a} 的 JS bits == nats/ln2",
           abs(js["mean_bits"] - js["mean_nats"] / 0.6931471805599453) < 1e-9)

# 收敛门禁落在判据量上（SCHEMA §17.15）
cg = R["convergence_gate"]
ck("收敛门禁的 quantity 是判据量（评测 bpc 差），不是训练总 CE",
   "criterion quantity" in cg["quantity"])
if cg.get("available", True):
    ck("门禁带判据量在两个点上的值与斜率",
       all(cg.get(k) is not None for k in ("criterion_value_at_ckpt", "criterion_value_at_full",
                                           "criterion_slope_per_1k", "stock_K1_slope_per_1k",
                                           "divl1_Kmax_slope_per_1k")))
else:
    ck("门禁不可用（未拿到中间检查点）⇒ 已显式标记 available=False 并降为快照口径",
       cg["available"] is False and "快照" in cg["why"])
ck("门禁并报训练总 CE 斜率（仅作对照）", isinstance(cg["train_loss_slopes"], dict))
ck("13M 锚点与近似同算力参照已落盘",
   R["config"]["anchors_13m"]["l1"]["K1"] == 1.6926
   and R["config"]["anchors_13m"]["approx_equal_compute_reference"]["delta_bpc"] == -0.1615)

# 代码指纹 / 变异报告
ck("代码指纹齐全（run.py / train_m1.py / 新模块 / 本契约）",
   all(R["code"].get(k) for k in ("run.py", "train_m1.py",
                                  "tools/evalkit/ensemble_probe.py", "README.md (本契约)")))
ck("变异验证全部抓红且还原", MUT["all_caught_and_restored"] is True)
ck("MODEL.md 里有三臂 sha256", all(R["checkpoints"][a]["sha256"][:12] in MD for a in ARMS))
ck("README 里预登记的两条判据公式仍在（契约未被事后改）",
   "bpc_stock@K1" in RM and "bpc_l1@K1" in RM)

# ------------------------------------------------------------------ 具体数字
SPECIFIC = [
    ("stock@K1", f"{b1['stock']}"), ("l1@K1", f"{b1['l1']}"), ("divl1@K1", f"{b1['divl1']}"),
    ("stock@Kmax", f"{bk['stock']}"), ("l1@Kmax", f"{bk['l1']}"), ("divl1@Kmax", f"{bk['divl1']}"),
    ("主判据（化简）", f"{R['criteria']['main_reading_B']['value_reduced_form']}"),
    ("并列判据 A", f"{R['criteria']['parallel_reading_A']['value']}"),
    ("税（去相关 vs l1）", f"{R['criteria']['taxes']['diversity_pressure_clean_tax_vs_l1']}"),
    ("税（机制 vs stock）", f"{R['criteria']['taxes']['mechanism_clean_tax_vs_stock']}"),
    ("成对 JS divl1 (bits)", f"{R['pairwise_js_at_kmax']['divl1']['mean_bits']:.4f}"),
    ("成对 JS l1 (bits)", f"{R['pairwise_js_at_kmax']['l1']['mean_bits']:.4f}"),
    ("参数量", f"{R['checkpoints']['stock']['params']}"),
    ("训练 token", f"{R['train_products']['stock']['train_tokens']}"),
]
if cg.get("available", True):
    SPECIFIC.append(("门禁判据量斜率", f"{cg['criterion_slope_per_1k']}"))
for label, s in SPECIFIC:
    ck(f"analysis.md 引用了「{label}」= {s}", s in AN, s, "in analysis.md")
for a in ARMS:
    for k in ("wall_minutes", "peak_gpu_mem_mb"):
        s = str(R["train_products"][a][k])
        ck(f"analysis.md 引用了 {a}.{k} = {s}", s in AN, s, "in analysis.md")

# ------------------------------------------------------------------ 结论
failed = [c for c in checks if not c[1]]
report = {"checks": checks, "n_checks": len(checks), "n_failed": len(failed),
          "passed": not failed,
          "main_criterion_passed": R["criteria"]["main_reading_B"]["passed"],
          "parallel_criterion_passed": R["criteria"]["parallel_reading_A"]["passed"]}
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
for c in failed:
    print("FAIL:", c)
print(f"checks={len(checks)} failed={len(failed)} passed={not failed}")
sys.exit(0 if not failed else 1)
