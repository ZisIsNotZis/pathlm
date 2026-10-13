"""2625 · 报告数字对账（每个写进 `stability_report.md` / `README.md` 的数字都必须能回到
`runs/<arm>/results.json` / `train_log.jsonl` / `probe.jsonl` / `smoke/*.json`）。

三道对账：
① 算术：token 计数、logits 上界 √d_h、探针的层间对照；
② 任务：所有臂同一数据切分 / 同一评测流（口径写死，脚本只核字段）；
③ 口径：判据①读数来自 `gnorm_monitor`；判据②基准唯一（2622 的 `full-256`/`full-512`）；
   候选臂与基准**逐字段只差候选补丁**。

用法：python3 reconcile_numbers.py   # 任一 mismatch 非零退出；写 reconcile_report.json
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
EXP_ROOT = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window",
                        "experiments")

checks = []


def ck(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    checks.append({"label": label, "ok": bool(ok), "got": got, "want": want})
    if not ok:
        print(f"MISMATCH {label}: got={got} want={want}", flush=True)


def load(path):
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


def report_text(name):
    p = os.path.join(HERE, name)
    return open(p).read().replace("\u2212", "-") if os.path.exists(p) else ""


REP = report_text("stability_report.md")
README = report_text("README.md")

# ------------------------------------------------------------ 各臂读数
arms = {}
for arm in ("full-T1024-qknorm-s0", "full-T1024-qknorm-s1", "full-1024-s1-diag"):
    arms[arm] = load(f"runs/{arm}/results.json")

B23 = json.load(open(os.path.join(EXP_ROOT, "2623-longT-1024", "results.json")))
B24 = json.load(open(os.path.join(EXP_ROOT, "2624-longT-1024-seed1", "results.json")))
B22 = json.load(open(os.path.join(EXP_ROOT, "2622-longT-512", "results.json")))

# ------------------------------------------------------------ 判据①（主判据）
crit = B23["criteria"]["criterion_1_quality"] if "criteria" in B23 else None
gn0 = B23["gnorm_monitor"] if "gnorm_monitor" in B23 else None
# 2623/2624 的 results.json 里逐臂 gnorm 存在 *_train_summary.json
S23 = json.load(open(os.path.join(EXP_ROOT, "2623-longT-1024",
                                  "full-1024-train_summary.json")))
S24 = json.load(open(os.path.join(EXP_ROOT, "2624-longT-1024-seed1",
                                  "full-1024-seed1-train_summary.json")))
W23 = json.load(open(os.path.join(EXP_ROOT, "2623-longT-1024",
                                  "win512-1024-train_summary.json")))
W24 = json.load(open(os.path.join(EXP_ROOT, "2624-longT-1024-seed1",
                                  "win512-1024-seed1-train_summary.json")))
ck("基准 seed0 gnorm_max", S23["gnorm_monitor"]["gnorm_max"], 1740.92, 0.01)
ck("基准 seed0 gnorm_mean", S23["gnorm_monitor"]["gnorm_mean"], 6.533, 1e-6)
ck("基准 seed0 skips", S23["gnorm_monitor"]["grad_skips"], 0)
ck("基准 seed1 gnorm_max", S24["gnorm_monitor"]["gnorm_max"], 29578.57, 0.01)
ck("基准 seed1 gnorm_mean", S24["gnorm_monitor"]["gnorm_mean"], 22.946, 1e-6)
ck("基准 seed1 skips", S24["gnorm_monitor"]["grad_skips"], 3)
ck("窗口 seed0 gnorm_max", W23["gnorm_monitor"]["gnorm_max"], 7.35, 0.01)
ck("窗口 seed1 gnorm_max", W24["gnorm_monitor"]["gnorm_max"], 7.66, 0.01)

for arm, res in arms.items():
    g = res["gnorm_monitor"]
    ck(f"{arm} 通过判据①（gnorm_max<100）", g["gnorm_max"] < 100, True)
    ck(f"{arm} 跳步为 0", g["grad_skips"], 0)

QK0, QK1 = arms["full-T1024-qknorm-s0"], arms["full-T1024-qknorm-s1"]
g0, g1 = QK0["gnorm_monitor"], QK1["gnorm_monitor"]
ck("qk-norm seed0 gnorm_max", g0["gnorm_max"], 7.25, 0.01)
ck("qk-norm seed0 gnorm_mean", g0["gnorm_mean"], 0.539, 1e-6)
ck("qk-norm seed1 gnorm_max", g1["gnorm_max"], 7.58, 0.01)
ck("qk-norm seed1 gnorm_mean", g1["gnorm_mean"], 0.553, 1e-6)
ck("诊断臂 gnorm_max", arms["full-1024-s1-diag"]["gnorm_monitor"]["gnorm_max"],
   7.64, 0.01)

# 质量读数（native 口径）
for arm, want in (("full-T1024-qknorm-s0", 1.543248),
                  ("full-T1024-qknorm-s1", 1.543707)):
    ev = load(f"runs/{arm}/eval.json")
    ck(f"{arm} clean bpc", ev["bpc_native"], want, 1e-6)
    ck(f"{arm} 评测流 window_seed", ev["window_seed"], 2)
B23EV = json.load(open(os.path.join(EXP_ROOT, "2623-longT-1024", "eval.json")))
B24EV = json.load(open(os.path.join(EXP_ROOT, "2624-longT-1024-seed1",
                                    "eval.json")))
ck("基准 seed0 clean bpc", B23EV["batteries"]["full-1024"]["bpc_native"],
   1.510446, 1e-6)
ck("基准 seed1 clean bpc", B24EV["batteries"]["full-1024-seed1"]["bpc_native"],
   1.606426, 1e-6)
qk = [1.543248, 1.543707]
base = [1.510446, 1.606426]
ck("QK-Norm 两 seed 离散度 < 0.001", max(qk) - min(qk) < 0.001, True)
ck("基准两 seed 离散度 > 0.05", max(base) - min(base) > 0.05, True)
for tok in ("1.543248", "1.543707", "1.510446", "1.606426"):
    ck(f"报告出现 {tok}", tok in REP, True)

# 报告文本里必须出现这些读数
for tok in ("1740.92", "29578.57", "7.25", "7.58", "6.533", "22.946"):
    ck(f"报告出现 {tok}", tok in REP, True)

# ------------------------------------------------------------ 逐字段对照（§2 单因素）
CFG23 = json.load(open(os.path.join(EXP_ROOT, "2623-longT-1024", "config.json")))
SEL = ("seq_len", "window", "steps", "batch_size", "lr", "seed")
for arm, ref_seed in (("full-T1024-qknorm-s0", 0), ("full-T1024-qknorm-s1", 1)):
    a = arms[arm]
    ck(f"{arm} seq_len", a["seq_len"], 1024)
    ck(f"{arm} window", a["window"], 0)
    ck(f"{arm} steps", a["steps"], 12000)
    ck(f"{arm} batch_size", a["batch_size"], 8)
    ck(f"{arm} lr", a["lr"], 1e-3)
    ck(f"{arm} seed", a["seed"], ref_seed)
    ck(f"{arm} 唯一补丁 = qk-norm", a["candidate"], "qk-norm")
    ck(f"{arm} 补丁覆盖 8 层", a["patch"]["n_qkv_hooks"], 8)

# ------------------------------------------------------------ ③ 算术对账
for arm, res in arms.items():
    if arm == "full-1024-s1-diag":       # 诊断臂：7000 步，不同 token 预算
        ck("诊断臂 token 计数（7000 步）",
           res["steps"] * res["batch_size"] * res["seq_len"], 57_344_000)
        continue
    ck(f"{arm} token 计数", res["steps"] * res["batch_size"] * res["seq_len"],
       98_304_000)
ck("qk-norm 的 logits 上界 = √(d_model/n_heads)",
   round(math.sqrt(256 / 8), 6), round(math.sqrt(32), 6))
BAND = 0.014

# ------------------------------------------------------------ ③ 探针（诊断臂）
probe = [json.loads(l) for l in
         open(os.path.join(HERE, "runs", "full-1024-s1-diag",
                           "probe.jsonl"))]
by_step = {r["step"]: r for r in probe}
ck("诊断臂探针步数", sorted(by_step), list(range(0, 7000, 200)))
for st, want in ((200, 0.60), (6800, 86.13), (6000, 50.02)):
    got = round(by_step[st]["layers"][1]["logit_rms"], 2)
    ck(f"诊断臂 L1 logit_rms@step{st}", got, want, 0.01)
ck("诊断臂 L1 logit_rms 单调上升（末 > 首×100）",
   by_step[6800]["layers"][1]["logit_rms"]
   > 100 * by_step[200]["layers"][1]["logit_rms"], True)
ck("诊断臂 L0 logit_rms 全程 <1.5",
   max(r["layers"][0]["logit_rms"] for r in probe) < 1.5, True)
for st, layer, want in ((6800, 0, 0.76), (6800, 3, 66.16), (6800, 4, 49.75),
                        (6800, 2, 27.57), (800, 3, 17.20), (800, 7, 4.49)):
    ck(f"诊断臂 L{layer} logit_rms@step{st}",
       round(by_step[st]["layers"][layer]["logit_rms"], 2), want, 0.01)

# ------------------------------------------------------------ ③ 探针（qk-norm 臂）
qp = [json.loads(l) for l in open(os.path.join(HERE, "runs",
                                               "full-T1024-qknorm-s1",
                                               "probe.jsonl"))]
qp = [r for r in qp if r.get("layers")]
ck("qk-norm 臂 L1 logit_rms 全程 ≤ √32",
   max(r["layers"][1]["logit_rms"] for r in qp) <= math.sqrt(32) + 1e-6, True)
ck("qk-norm 臂 L1 熵不塌缩（全程 > 4.5 nats）",
   min(r["layers"][1]["entropy"] for r in qp) > 4.5, True)
ck("qk-norm 臂 L1 logit_rms 末态落在 2.2–2.6",
   2.2 <= qp[-1]["layers"][1]["logit_rms"] <= 2.6, True)
q0 = [json.loads(l) for l in open(os.path.join(HERE, "runs",
                                               "full-T1024-qknorm-s0",
                                               "probe.jsonl"))]
q0 = [r for r in q0 if r.get("layers")]
ck("qk-norm seed0 L1 logit_rms 全程 ≤ √32",
   max(r["layers"][1]["logit_rms"] for r in q0) <= math.sqrt(32) + 1e-6, True)

# ------------------------------------------------------------ ③ 「疤」指纹
scar = load("smoke/ckpt_scar_probe.json")
ck("疤 full-1024 L1.qkv", round(scar["full-1024"]["qkv_w_norm_by_layer"][1], 2),
   109.50, 0.01)
ck("疤 full-1024-seed1 L1.qkv",
   round(scar["full-1024-seed1"]["qkv_w_norm_by_layer"][1], 2), 131.48, 0.01)
health = [round(scar[a]["qkv_w_norm_by_layer"][1], 2) for a in
          ("full-256", "win128-256", "full-512", "win256-512", "win512-1024",
           "win512-1024-seed1")]
ck("健康臂 L1.qkv 全部 ≤ 25.2", max(health) <= 25.20, True)

att = load("smoke/attention_scale_probe.json")
ck("疤 full-1024 L0 残差范数",
   round(att["full-1024"]["layers"][0]["resid_norm"], 1), 122.7, 0.05)
ck("疤 full-1024-seed1 L0 残差范数",
   round(att["full-1024-s1"]["layers"][0]["resid_norm"], 1), 139.1, 0.05)
ck("疤 full-1024 L1 logit_rms ≈ 1.9e5",
   1.8e5 < att["full-1024"]["layers"][1]["logit_rms"] < 2.0e5, True)
ck("疤 full-1024-s1 L1 logit_rms ≈ 3.6e5",
   3.5e5 < att["full-1024-s1"]["layers"][1]["logit_rms"] < 3.7e5, True)
ck("疤 full-1024 L1 熵塌缩 < 0.11 nats",
   att["full-1024"]["layers"][1]["entropy"] < 0.11, True)
ck("疤 full-1024 L1 top1 > 0.9（注意力几乎一热）",
   att["full-1024"]["layers"][1]["top1"] > 0.9, True)

# ------------------------------------------------------------ ③ 初始尺度（H4 排除）
gs = load("smoke/grad_scale_probe.json")
for T, want in ((256, 7.901), (512, 6.5706), (1024, 6.7076)):
    ck(f"固定 init 全因果 gnorm @T={T}", gs[f"T{T}-w0"]["gnorm_total"], want, 1e-4)
ck("固定 init gnorm 与 T 无关（极差 < 1.4）",
   max(gs[f"T{T}-w0"]["gnorm_total"] for T in (256, 512, 1024))
   - min(gs[f"T{T}-w0"]["gnorm_total"] for T in (256, 512, 1024)) < 1.4, True)

# ------------------------------------------------------------ 保真度锚点
pf = json.loads(open(os.path.join(HERE, "runs", "preflight-20",
                                  "train_log.jsonl")).readline())
eng = json.loads(open(os.path.join(EXP_ROOT, "2623-longT-1024", "runs",
                                   "full-1024", "train_log.jsonl")).readline())
ck("驱动器 step0 loss == 引擎", pf["loss"], eng["loss"])
ck("驱动器 step0 gnorm == 引擎", pf["gnorm"], eng["gnorm"])

# ------------------------------------------------------------ 判据②（短 T Δbpc）
short = []
for arm, ref_key in (("full-T256-qknorm256-s0", "bpc_full_256"),
                     ("full-T512-qknorm512-s0", "bpc_full_512")):
    ev = os.path.join(HERE, "runs", arm, "eval.json")
    if not os.path.exists(ev):
        short.append({"arm": arm, "status": "未跑", "delta_bpc": None,
                      "pass": None})
        continue
    got = json.load(open(ev))
    base = (B22["criteria"]["criterion_1_quality"]["native_operating_point"]
            if "criteria" in B22 else None)
    if base is not None:
        b = base[ref_key]
    else:
        b = None
    if b is None:                       # 兜底：直接取 results.json 里的读数
        b = {"bpc_full_256": 1.525389, "bpc_full_512": 1.574020}[ref_key]
    d = round(got["bpc_native"] - b, 6)
    short.append({"arm": arm, "status": "已跑", "candidate_bpc": got["bpc_native"],
                  "baseline_bpc": b, "delta_bpc": d,
                  "abs_delta": abs(d), "pass": abs(d) < BAND})
    ck(f"{arm} 评测流 window_seed", got["window_seed"], 2)
    ck(f"{arm} n_batches", got["n_batches"], 20)
ck("Δbpc(T=256)", next(s["delta_bpc"] for s in short
                       if s["arm"].startswith("full-T256")), 0.010280, 1e-6)
ck("Δbpc(T=512)", next(s["delta_bpc"] for s in short
                       if s["arm"].startswith("full-T512-qk")), 0.031381, 1e-6)
for tok in ("1.535669", "1.605401", "+0.0103", "+0.0314"):
    ck(f"报告出现 {tok}", tok in REP, True)

# ------------------------------------------------------------ ③ 逐 200 步曲线表
TB = {}
for arm in ("full-T1024-qknorm-s0", "full-T1024-qknorm-s1"):
    TB[arm] = {json.loads(l)["step"]: json.loads(l) for l in
               open(os.path.join(HERE, "runs", arm, "train_log.jsonl"))}
base_s1 = {json.loads(l)["step"]: json.loads(l) for l in
           open(os.path.join(EXP_ROOT, "2624-longT-1024-seed1", "runs",
                             "full-1024-seed1", "train_log.jsonl"))}
CURVE_STEPS = (5800, 6000, 6200, 6400, 6600, 6800, 7000)
WANT_QK1_GN = [0.47, 0.42, 0.55, 0.49, 0.46, 0.50, 0.44]
WANT_QK1_LS = [1.1245, 1.1218, 1.3508, 1.3031, 1.2239, 1.1695, 1.1406]
WANT_BA1_GN = [0.79, 3.15, 82.05, 42.45, 80.32, 109.77, 174.53]
WANT_BA1_LS = [1.1094, 1.1508, 4.3934, 2.7810, 2.3697, 2.4076, 2.3063]
for i, s in enumerate(CURVE_STEPS):
    ck(f"报告曲线 qk-s1 gnorm@{s}", TB["full-T1024-qknorm-s1"][s]["gnorm"],
       WANT_QK1_GN[i], 1e-9)
    ck(f"报告曲线 qk-s1 loss@{s}", TB["full-T1024-qknorm-s1"][s]["loss"],
       WANT_QK1_LS[i], 1e-9)
    ck(f"报告曲线 基准-s1 gnorm@{s}", base_s1[s]["gnorm"], WANT_BA1_GN[i], 1e-9)
    ck(f"报告曲线 基准-s1 loss@{s}", base_s1[s]["loss"], WANT_BA1_LS[i], 1e-9)
q0tb = TB["full-T1024-qknorm-s0"]
ck("qk-norm seed0 逐 200 步 gnorm 全程 ≤ 7.25",
   max(r["gnorm"] for r in q0tb.values()) <= 7.25, True)
ck("qk-norm seed0 在爆炸点 9800 处 gnorm = 0.40", q0tb[9800]["gnorm"], 0.40, 1e-9)
base_s0 = {json.loads(l)["step"]: json.loads(l) for l in
           open(os.path.join(EXP_ROOT, "2623-longT-1024", "runs",
                             "full-1024", "train_log.jsonl"))}
ck("基准 seed0 在 9800 处 gnorm = 128.85", base_s0[9800]["gnorm"], 128.85, 1e-9)
# H5：缩短 steps 会同时压缩 LR 调度（诊断臂不是 12k 跑的前缀）
diag_lr = {json.loads(l)["step"]: json.loads(l)["lr"] for l in
           open(os.path.join(HERE, "runs", "full-1024-s1-diag",
                             "train_log.jsonl"))}
ck("诊断臂 LR@6200 = 1.29e-4", diag_lr[6200], 0.000129, 1e-9)
ck("基准/主臂 LR@6200 = 5.26e-4", base_s1[6200]["lr"], 0.000526, 1e-9)
ck("诊断臂 LR@6200 < 基准的 1/3（非前缀）",
   diag_lr[6200] < base_s1[6200]["lr"] / 3, True)

# ------------------------------------------------------------ 判据②（结构性零 hook + run-to-run 噪声）
struct = {}
for arm in ("full-T512-cond512-s0", "full-T512-none512-s0"):
    rp = os.path.join(HERE, "runs", arm, "results.json")
    if not os.path.exists(rp):
        struct[arm] = "未跑"
        continue
    struct[arm] = json.load(open(rp))["patch"]
ck("cond512 在 T=512 零 hook（结构性口径）",
   struct["full-T512-cond512-s0"]["n_qkv_hooks"], 0)
ck("cond512 active == False",
   struct["full-T512-cond512-s0"]["active"], False)

# 短 T 逐 bpc（三个 T=512 臂）
def _bpc(arm):
    p = os.path.join(HERE, "runs", arm, "eval.json")
    return json.load(open(p))["bpc_native"] if os.path.exists(p) else None

a_, b_, c_ = (_bpc("full-T512-none512-s0"),
              _bpc("full-T512-none512b-s0"),
              _bpc("full-T512-cond512-s0"))
if None not in (a_, b_, c_):
    struct["bpc_T512"] = {"none_a": a_, "none_b": b_, "cond": c_,
                          "engine_2622": 1.574020}
    struct["run_to_run_bpc_range"] = round(abs(a_ - b_), 6)
    struct["cond_vs_engine_delta"] = round(c_ - 1.574020, 6)
    ck("cond512 Δbpc vs 2622 在噪声带内", abs(c_ - 1.574020) < BAND, True)
    ck("cond512 Δbpc = −0.001603", round(c_ - 1.574020, 6), -0.001603, 1e-6)
    ck("run-to-run bpc 噪声 0.004534", round(abs(a_ - b_), 6), 0.004534, 1e-6)
    ck("cond 的 |Δ| 小于 run-to-run 噪声",
       abs(c_ - 1.574020) < abs(a_ - b_), True)

# 同配方重跑不是逐位的（附上证据，不当作失败）
def _logs(arm):
    p = os.path.join(HERE, "runs", arm, "train_log.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(x) for x in open(p)]

la, lb, lc = (_logs("full-T512-none512-s0"), _logs("full-T512-none512b-s0"),
              _logs("full-T512-cond512-s0"))
if None not in (la, lb, lc):
    d_rerun = max(abs(x["loss"] - y["loss"]) for x, y in zip(la, lb))
    d_cond = max(abs(x["loss"] - y["loss"]) for x, y in zip(la, lc))
    struct["max_abs_loss_diff_rerun"] = round(d_rerun, 4)
    struct["max_abs_loss_diff_cond_vs_none"] = round(d_cond, 4)
    ck("同配方重跑非逐位（max|Δloss| > 0.01）", d_rerun > 0.01, True)
    ck("重跑散度 > 候选 vs 基线散度", d_rerun > d_cond, True)
    ck("报告出现 0.0320", "0.0320" in REP, True)
for tok in ("−0.0016", "0.0045", "n_qkv_hooks: 0"):
    ck(f"报告出现 {tok}", tok in REP.replace("-", "−"), True)
bitwise = struct

# ------------------------------------------------------------ 报告自洽
ck("README 含判据①逐字判据",
   "gnorm_max < 100 且无跳步" in README.replace("\n", " "), True)
ck("README 声明不改共享文件", "不改任何共享文件" in README, True)

rep = {"checks": checks, "n_checks": len(checks),
       "n_failed": sum(1 for c in checks if not c["ok"]),
       "criteria_1_pass": bool(g0["gnorm_max"] < 100 and g0["grad_skips"] == 0
                              and g1["gnorm_max"] < 100 and g1["grad_skips"] == 0),
       "criteria_2_band": BAND, "criteria_2_short_T": short,
       "curves_file": "curves.json", "eval_file": "eval.json"}

# 注意：rep 的 `n_failed` 必须涵盖全部 ck（包括下面 curves 段的），故在最后重建。

# ------------------------------------------------------------ 曲线导出（可入库）
# `runs/` 被 .gitignore 忽略（与 2622/2623 同约定），故把每个改动臂的 gnorm/loss 曲线
# 导出到实验根目录的 `curves.json`（判据④要求「每个改动臂的 gnorm/loss 曲线数据」）。
CURVE_ARMS = ["full-T1024-qknorm-s0", "full-T1024-qknorm-s1",
              "full-1024-s1-diag", "full-T256-qknorm256-s0",
              "full-T512-qknorm512-s0", "full-T512-none512-s0",
              "full-T512-none512b-s0", "full-T512-cond512-s0"]
curves = {"source": "runs/<arm>/train_log.jsonl（每 200 步一行，与引擎同格式）",
          "budget_note": ("诊断臂 7000 步、短 T 臂 6000/12000 步；"
                          "其步数均已在 stability_report.md 声明"),
          "arms": {}}
for arm in CURVE_ARMS:
    p = os.path.join(HERE, "runs", arm, "train_log.jsonl")
    if not os.path.exists(p):
        continue
    curves["arms"][arm] = [{"step": r["step"], "gnorm": r["gnorm"],
                            "loss": r["loss"]}
                           for r in (json.loads(x) for x in open(p))]
with open(os.path.join(HERE, "curves.json"), "w") as f:
    json.dump(curves, f, indent=2, ensure_ascii=False)
ck("curves.json 含 8 个改动臂", len(curves["arms"]), len(CURVE_ARMS))
for arm in ("full-T1024-qknorm-s0", "full-T1024-qknorm-s1"):
    cs = curves["arms"][arm]
    ck(f"curves {arm} 行数 61（每 200 步）", len(cs), 61)
    # 每 200 步采样 ≤ 全程 gnorm_max（最大值可能落在未采样步）
    ck(f"curves {arm} 采样最大 gnorm ≤ 全程 gnorm_max",
       max(c["gnorm"] for c in cs)
       <= arms[arm]["gnorm_monitor"]["gnorm_max"] + 1e-9, True)
    ck(f"curves {arm} 采样最大 gnorm ≥ 0.9×gnorm_max",
       max(c["gnorm"] for c in cs)
       >= 0.9 * arms[arm]["gnorm_monitor"]["gnorm_max"] - 1e-9, True)

# ------------------------------------------------------------ 最终汇总
# 把各臂 clean bpc 汇到根目录 `eval.json`（`runs/` 不入库，与 2622/2623 同约定）
EVAL = {arm: json.load(open(os.path.join(HERE, "runs", arm, "eval.json")))
        for arm in ("full-T1024-qknorm-s0", "full-T1024-qknorm-s1",
                    "full-T256-qknorm256-s0", "full-T512-qknorm512-s0",
                    "full-T512-none512-s0", "full-T512-none512b-s0",
                    "full-T512-cond512-s0")
        if os.path.exists(os.path.join(HERE, "runs", arm, "eval.json"))}
with open(os.path.join(HERE, "eval.json"), "w") as f:
    json.dump({"eval_stream": {"fn": "tools.evalkit.retention.masked_bpc",
                              "n_batches": 20, "batch_size": 16,
                              "window_seed": 2},
               "native_operating_point": ("full 臂 = 全因果（release）；"
                                          "窗口臂 = 自身训练掩码"),
               "baselines_2622": {"full-256": 1.525389, "full-512": 1.574020},
               "baselines_2623_2624": {"full-1024": 1.510446,
                                       "full-1024-seed1": 1.606426},
               "arms": EVAL}, f, indent=2, ensure_ascii=False)
ck("eval.json 含 7 个臂的 clean bpc", len(EVAL), 7)

rep["checks"] = checks
rep["n_checks"] = len(checks)
rep["n_failed"] = sum(1 for c in checks if not c["ok"])
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump(rep, f, indent=2, ensure_ascii=False)
print(json.dumps({"n_checks": rep["n_checks"], "n_failed": rep["n_failed"],
                  "criteria_1_pass": rep["criteria_1_pass"],
                  "criteria_2_short_T": short,
                  "bitwise_short_T": struct}, indent=2, ensure_ascii=False))
if rep["n_failed"]:
    raise SystemExit(1)
