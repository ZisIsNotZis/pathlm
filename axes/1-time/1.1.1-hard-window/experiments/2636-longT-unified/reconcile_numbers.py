"""2636 · 报告数字对账（每个写进 `analysis.md` / `README.md` 的数字都必须能回到
`results.json` / `eval.json` / `smoke/smoke.json` / `runs/<arm>/train_log.jsonl`）。

三道对账（SCHEMA §12）：
① **算术**：token 计数 = steps × bs × T（同 step 四点）；同 token 副口径 49.152M；
   Δ = bpc_full − bpc_window 从 `eval.json` 或引用点重算；带宽分类逐点一致；
   注意力占比公式 / 窗口 visible_frac 闭式复算；**跨口径差（判据 C5）** = 两口径 Δ 之差，
   且其两个门限（本格判据量噪声 / 工程带）必须与 `criterion_4` / `criterion_1` 一致。
② **任务**：八臂同一数据切分与同一评测流（`window_seed=2`、20 批 × bs16）；
   逐字段对照只允许差 `seq_len` / `train_window` / `seed`。
③ **口径**：判据 C1 读数来自 `eval.json::batteries`（新臂）或引用点 `eval.json`；
   C2 来自 `runs/<arm>/results.json::gnorm_monitor`；C3 来自 `results.json::cost`。

用法：python3 reconcile_numbers.py [目录]   # 默认读同目录；任一 mismatch 非零退出；写 reconcile_report.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
#: 读入目录：默认本目录；给了参数就读该目录（默认行为不变）。这条口子是给「构造不一致
#: 输入 ⇒ 对账必须红」的可失败自测用的（`tests/test_longT_unified_cross_reading.py`）。
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else HERE
sys.path.insert(0, ROOT)

from pathlm.depth_ar import window_mask_liveness            # noqa: E402

checks = []
BAND = 0.014
S = 6000
BS = 8
EXP_ROOT = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window", "experiments")


def ck(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) \
        else (got == want)
    checks.append({"label": label, "ok": bool(ok), "got": got, "want": want})
    if not ok:
        print(f"MISMATCH {label}: got={got} want={want}", flush=True)


def load(path):
    with open(os.path.join(DATA_DIR, path)) as f:
        return json.load(f)


R = load("results.json")
EV = load("eval.json")
SM = load("smoke/smoke.json")


def band_class(x):
    if abs(x) < BAND:
        return "不亏（噪声带内）"
    return "窗口更好（越界）" if x > 0 else "窗口更差（越界）"


# ------------------------------------------------------------ ① 算术（本实验八臂）
for arm, a in R["arms"].items():
    T = a["battery"]["seq_len"]
    ck(f"{arm} tokens = 6000*8*{T}", a["tokens"], S * BS * T)
    ck(f"{arm} steps == 6000", a["steps"], S)
    ck(f"{arm} eval window_seed", a["battery"]["window_seed"], 2)
    ck(f"{arm} eval n_batches", a["battery"]["n_batches"], 20)
    ck(f"{arm} eval batch_size", a["battery"]["batch_size"], 16)

# 同 step 四点（新臂 256/512 + 引用 1024/2048）Δ 从数值重算，须与服务端一致
sp = {str(r["T"]): r for r in R["criteria"]["criterion_1_quality_matched_step"]["four_points"]}
for T in (256, 512):
    fa, wa = f"full-{T}-s6k", f"win{T // 2}-{T}-s6k"
    dn = round(EV["batteries"][fa]["bpc_native"] - EV["batteries"][wa]["bpc_native"], 6)
    ck(f"Δnative@step{T}", sp[str(T)]["delta_native"], dn, 1e-6)
    ck(f"band@step{T}", sp[str(T)]["band_class"], band_class(dn))
# 引用点：从各自 eval.json 重读
for T, (exp, fa, wa) in {1024: ("2626-longT-clean", "full-1024-clean", "win512-1024-clean"),
                         2048: ("2629-longT-2048-matched", "full-2048-clean", "win1024-2048-clean")}.items():
    b = json.load(open(os.path.join(EXP_ROOT, exp, "eval.json")))["batteries"]
    dn = round(b[fa]["bpc_native"] - b[wa]["bpc_native"], 6)
    ck(f"ref Δ@step{T}", sp[str(T)]["delta_native"], dn, 1e-6)

# 同 token 副口径四点
tp = {str(r["T"]): r for r in R["criteria"]["criterion_1b_quality_matched_token"]["four_points"]}
REF_TOKEN = {256: ("2626-longT-clean", "full-256-clean", "win128-256-clean", 24000),
             512: ("2626-longT-clean", "full-512-clean", "win256-512-clean", 12000),
             1024: ("2626-longT-clean", "full-1024-clean", "win512-1024-clean", 6000),
             2048: ("2628-longT-2048", "full-2048-clean", "win1024-2048-clean", 3000)}
for T, (exp, fa, wa, steps) in REF_TOKEN.items():
    b = json.load(open(os.path.join(EXP_ROOT, exp, "eval.json")))["batteries"]
    dn = round(b[fa]["bpc_native"] - b[wa]["bpc_native"], 6)
    ck(f"token Δ@{T}", tp[str(T)]["delta_native"], dn, 1e-6)
    ck(f"token-check tokens@{T}=49.152M", steps * BS * T, 49_152_000)

# ------------------------------------------------------------ 判据 C5：跨口径差
# 这个差值此前只存在于 analysis.md 的两个分量手算 ⇒ 不被对账覆盖，改门限时会静默失配
# （SCHEMA §24 第 11 类）。此处的断言把「块 ↔ 两个分量 ↔ 两个门限」全部钉住。
cr = (R["criteria"].get("criterion_5_cross_reading") or {}).get("per_T") or {}
for T in (256, 512):
    ck(f"cross_reading per_T 含 {T}", str(T) in cr, True)
    if str(T) not in cr:
        continue
    row = cr[str(T)]
    dn_step, dn_token = sp[str(T)]["delta_native"], tp[str(T)]["delta_native"]
    ck(f"cross@{T} delta_step == 同 step 分量", row["delta_step"], dn_step, 1e-6)
    ck(f"cross@{T} delta_token == 同 token 分量", row["delta_token"], dn_token, 1e-6)
    ck(f"cross@{T} cross_diff == Δ_step − Δ_token", row["cross_diff"],
       round(dn_step - dn_token, 6), 1e-6)
    ck(f"cross@{T} abs_cross_diff == |cross_diff|", row["abs_cross_diff"],
       abs(row["cross_diff"]), 1e-9)
    ck(f"cross@{T} cell_criterion_noise == C4 Δ 跨 seed 极差", row["cell_criterion_noise"],
       R["criteria"]["criterion_4_cell_noise"]["delta_spread"][str(T)]["abs_spread"], 1e-6)
    ck(f"cross@{T} engineering_band == C1 工程带", row["engineering_band"],
       R["criteria"]["criterion_1_quality_matched_step"]["variance_band_project"], 1e-9)
    ck(f"cross@{T} exceeds_cell_criterion_noise 与门限一致",
       row["exceeds_cell_criterion_noise"],
       bool(row["abs_cross_diff"] > row["cell_criterion_noise"]))
    ck(f"cross@{T} within_engineering_band 与门限一致", row["within_engineering_band"],
       bool(row["abs_cross_diff"] <= row["engineering_band"]))

# 注意力占比公式闭式复算
OTHER = 786432
for T in (256, 512, 1024, 2048):
    ck(f"attn_share_formula({T})",
       round((2 * T * 256) / (2 * T * 256 + OTHER), 4),
       R["cost"]["theory"]["attn_share_formula"][str(T)], 1e-4)

# 窗口 visible_frac 闭式复算（与 pathlm 的实现比对）
for T in (256, 512, 1024, 2048):
    w = T // 2
    live = window_mask_liveness(T, w, 0)
    n_keep = sum(1 for i in range(T) for j in range(T) if j <= i and (i - j) < w)
    ck(f"visible_frac({w}@{T})", round(n_keep / (T * (T + 1) // 2), 6),
       live["visible_frac"], 1e-6)

# ------------------------------------------------------------ ② 任务（逐字段）
KEYS = ("d_model", "n_layers", "n_heads", "mlp_mult", "dense", "topology",
        "batch_size", "lr", "exit_mode", "qk_norm", "qk_norm_min_t")
fields = {a["arm"]: a["fields"] for a in R["config"]["arms"]}
base = fields["full-256-s6k"]
for arm, f in fields.items():
    for k in KEYS:
        ck(f"{arm} field {k} == base", f[k], base[k])
    ck(f"{arm} steps == 6000", f["steps"], S)
    ck(f"{arm} seq_len in cfg field", f["seq_len"], next(
        a["seq_len"] for a in R["config"]["arms"] if a["arm"] == arm))

# ---- SCHEMA §9 跨 run 复用：与引用臂逐字段对照（只允许差 seq_len/train_window/steps） ----
# 修 P1-6：README/analysis 宣称的「逐字段对照 + 不等字段列出」必须机器可读地存在。
ALL_KEYS = ("d_model", "n_layers", "n_heads", "mlp_mult", "dense", "topology",
            "batch_size", "lr", "seed", "exit_mode", "qk_norm", "qk_norm_min_t",
            "data", "vocab_size", "seq_len", "train_window", "steps")
ALLOWED_DIFF = {"seq_len", "train_window", "steps"}
REF_FIELD = {
    "2626-longT-clean": ("full-1024-clean", "win512-1024-clean"),
    "2629-longT-2048-matched": ("full-2048-clean", "win1024-2048-clean"),
}
cross_ref = {}
for exp, arms in REF_FIELD.items():
    ev_ref = json.load(open(os.path.join(EXP_ROOT, exp, "eval.json")))
    for arm in arms:
        rf = ev_ref["field_by_field"][arm]
        diffs = {k: {"this_exp_base": base[k], "ref": rf[k]}
                 for k in ALL_KEYS if base[k] != rf[k]}
        cross_ref[f"{exp}:{arm}"] = {"differing_fields": sorted(diffs),
                                     "diff_values": diffs}
        ck(f"ref {exp}:{arm} 不等字段 ⊆ {sorted(ALLOWED_DIFF)}",
           set(diffs) <= ALLOWED_DIFF, True)
# 基准臂自身必须与引用臂在「只读复用允许的字段」上不同、其余相同：
# （按 §9：不等字段显式列出，等字段逐条断言）
for key, row in cross_ref.items():
    for f in KEYS:
        ck(f"cross-run {key} 等值字段 {f}", f not in row["differing_fields"], True)

# ------------------------------------------------------------ ③ 口径
gate = R["criteria"]["criterion_2_stability_gate"]
for T in (256, 512):
    ck(f"gate full-{T}-s6k pass", gate["per_arm"][f"full-{T}-s6k"]["pass"], True)
    ck(f"gate full-{T}-s6k gnorm_max<100",
       gate["per_arm"][f"full-{T}-s6k"]["gnorm_max"] < 100, True)

# qk-norm 活性：T>512 臂 8 hook（本实验没有 >512 的新臂 ⇒ 应全 0），零新增参数
for arm, a in R["arms"].items():
    T = a["battery"]["seq_len"]
    hooks = a["patch"].get("n_qkv_hooks")
    ck(f"{arm} qk hooks == {8 if T > 512 else 0}", hooks, 8 if T > 512 else 0)
    ck(f"{arm} qk params_added == 0", a["patch"].get("params_added"), 0)

# 四条腿读数来自 cost（口径）；r_time/r_mem 与 model_step 中位墙钟自洽
meas = R["cost"]["measured"]
for row in meas["ratio_by_T"]:
    ck(f"r_time@{row['T']}", row["r_time_window_over_full"],
       round(row["step_ms_window"] / row["step_ms_full"], 4), 1e-3)
    ck(f"r_mem@{row['T']}", row["r_mem_window_over_full"],
       round(row["peak_window_gib"] / row["peak_full_gib"], 4), 1e-3)

# 峰值礼貌线：训练臂（本实验）逐臂记账
mx = max(a["peak_torch_gb"] for a in R["arms"].values())
ck("本实验训练臂峰值 ≤5GB", mx <= 5.0, True)

# ------------------------------------------------------------ 报告文本读数（若已写）
REP = ""
p = os.path.join(DATA_DIR, "analysis.md")
if os.path.exists(p):
    REP = open(p).read().replace("\u2212", "-")
for T in (256, 512, 1024, 2048):
    dn = sp[str(T)]["delta_native"]
    ck(f"analysis 出现 Δ@step{T}", any(s in REP for s in
       (f"{dn:+.6f}", f"{dn:.6f}", f"{dn:+.4f}", f"{dn:.4f}")), True)

# ------------------------------------------------------------ 汇总
n_ok = sum(1 for c in checks if c["ok"])
report = {"n_checks": len(checks), "n_ok": n_ok, "n_fail": len(checks) - n_ok,
          "band": BAND, "cross_run_field_compare": cross_ref, "checks": checks}
json.dump(report, open(os.path.join(DATA_DIR, "reconcile_report.json"), "w"),
          indent=2, ensure_ascii=False)
print(f"reconcile: {n_ok}/{len(checks)} OK")
if n_ok != len(checks):
    raise SystemExit(1)
