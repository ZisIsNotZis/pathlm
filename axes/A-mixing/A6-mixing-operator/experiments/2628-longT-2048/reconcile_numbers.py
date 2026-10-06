"""2628 · 报告数字对账（每个写进 `analysis.md` / `README.md` 的数字都必须能回到
`results.json` / `eval.json` / `runs/<arm>/train_log.jsonl` / `smoke/*.json`）。

三道对账（SCHEMA §12）：
① **算术**：token 计数 = steps × bs × T；Δ2048 = bpc_full − bpc_window（从 `eval.json` 重算）；
   带宽分类与服务端一致；四点并列 = 2626 三点（从其 `results.json` 重读）+ 本点；
   注意力占比公式闭式复算（1024/2048）；窗口 `visible_frac` 闭式复算。
② **任务**：两臂同一数据切分（enwik8 前 90M/末 10M）与**同一评测流**
   （`window_seed=2`、20 批 × bs16）；口径写死，脚本只核字段。
③ **口径**：判据①读数来自 `eval.json::batteries[*].bpc_native`；判据②来自 `results.json::gnorm_monitor`；
   逐字段对照只允许差 `seq_len` / `train_window`（同 seed/bs/lr/dense/topology/qk_norm）。

用法：python3 reconcile_numbers.py   # 任一 mismatch 非零退出；写 reconcile_report.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

ARMS = ("full-2048-clean", "win1024-2048-clean")
EXP_ROOT = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator", "experiments")

checks = []


def ck(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) \
        else (got == want)
    checks.append({"label": label, "ok": bool(ok), "got": got, "want": want})
    if not ok:
        print(f"MISMATCH {label}: got={got} want={want}", flush=True)


def load(path):
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


def report_text(name):
    p = os.path.join(HERE, name)
    return open(p).read().replace("\u2212", "-") if os.path.exists(p) else ""


R = load("results.json")
EV = load("eval.json")
SM = load("smoke/smoke.json")
REP = report_text("analysis.md")

BAND = 0.014


def band_class(x):
    if abs(x) < BAND:
        return "不亏（噪声带内）"
    return "窗口更好（越界）" if x > 0 else "窗口更差（越界）"


# ------------------------------------------------------------ ① 算术
for arm in ARMS:
    a = R["arms"][arm]
    T = a["battery"]["seq_len"]
    ck(f"{arm} tokens = steps*bs*T",
       a["tokens"], a["steps"] * R["config"]["batch_size"] * T)
    ck(f"{arm} eval window_seed", a["battery"]["window_seed"], 2)
    ck(f"{arm} eval n_batches", a["battery"]["n_batches"], 20)
    ck(f"{arm} eval batch_size", a["battery"]["batch_size"], 16)

# 本点 Δ2048 从 eval.json 重算，必须与服务端相等
srv = {p["T"]: p for p in R["criteria"]["criterion_1_quality"]["this_point_native"]}
fa, wa = "full-2048-clean", "win1024-2048-clean"
dn = round(EV["batteries"][fa]["bpc_native"] - EV["batteries"][wa]["bpc_native"], 6)
ck("Δnative@2048", srv[2048]["delta_native"], dn, 1e-6)
ck("band_class@2048", srv[2048]["band_class"], band_class(dn))
dr = round(EV["batteries"][fa]["bpc_release"] - EV["batteries"][wa]["bpc_release"], 6)
ck("Δrelease@2048", R["criteria"]["criterion_1_quality"]["this_point_release"][0]
   ["delta_release"], dr, 1e-6)

# 四点并列 = 2626 三点（重读）+ 本点，形状字符串逐点一致
three = R["criteria"]["criterion_1_quality"]["four_points_native"]
ck("four_points has 4 entries", len(three), 4)
ck("four_points T order", [p["T"] for p in three], [256, 512, 1024, 2048])
p26 = os.path.join(EXP_ROOT, "2626-longT-clean", "results.json")
if os.path.exists(p26):
    ref3 = json.load(open(p26))["criteria"]["criterion_1_quality"]["three_points_native"]
    for a26, a28 in zip(ref3, three[:3]):
        ck(f"four_points[{a26['T']}] delta matches 2626",
           a28["delta_native"], a26["delta_native"], 1e-9)
        ck(f"four_points[{a26['T']}] band matches 2626",
           a28["band_class"], a26["band_class"])
    ck("four_point_shape_native", R["criteria"]["criterion_1_quality"]
       ["four_point_shape_native"],
       " → ".join(f"{p['band_class']}@{p['T']}" for p in three))

# 注意力占比公式闭式复算（1024=0.4000、2048=0.5714）
for T, want in ((1024, 0.4000), (2048, 0.5714)):
    ck(f"attn_share_formula({T})",
       round((2 * T * 256) / (2 * T * 256 + 786432), 4), want, 1e-4)

# 窗口 visible_frac 闭式复算（1024@2048）
T, w = 2048, 1024
n_keep = sum(1 for i in range(T) for j in range(T) if j <= i and (i - j) < w)
n_causal = T * (T + 1) // 2
ck("visible_frac(1024@2048)", round(n_keep / n_causal, 6),
   round(R["arms"][wa]["battery"]["native_mask_liveness"]["visible_frac"], 6), 1e-6)

# ------------------------------------------------------------ ② 任务（同一评测流）
ck("all arms present", sorted(R["arms"].keys()) == sorted(list(ARMS)), True)
fields = {a["arm"]: a["fields"] for a in R["config"]["arms"]}
for arm, f in fields.items():
    ck(f"{arm} dense", f["dense"], True)
    ck(f"{arm} topology", f["topology"], "sequential")
    ck(f"{arm} seed", f["seed"], 0)
    ck(f"{arm} lr", f["lr"], 0.001)
    ck(f"{arm} bs", f["batch_size"], 8)
    ck(f"{arm} qk_norm", f["qk_norm"], "qk-normfrom512")
# 逐字段对照：两臂只允许差 seq_len / train_window / steps
KEYS = ("d_model", "n_layers", "n_heads", "mlp_mult", "dense", "topology",
        "batch_size", "lr", "seed", "exit_mode", "qk_norm", "qk_norm_min_t")
base = fields["full-2048-clean"]
for arm, f in fields.items():
    for k in KEYS:
        ck(f"{arm} field {k} == base", f[k], base[k])
ck("两臂同 seq_len", fields[ARMS[0]]["seq_len"], fields[ARMS[1]]["seq_len"])
ck("窗口 = T/2", fields[ARMS[1]]["train_window"], fields[ARMS[1]]["seq_len"] // 2)
ck("两臂同 steps", R["arms"][ARMS[0]]["steps"], R["arms"][ARMS[1]]["steps"])

# ------------------------------------------------------------ ③ 口径
gate = R["criteria"]["criterion_2_stability_gate"]
ck("gate gnorm_max<100", gate["full_2048_gnorm_max"] < 100, True)
ck("gate skips==0", gate["full_2048_grad_skips"], 0)
ck("gate passed", gate["passed"], True)

# qk-norm 活性：T=2048 > 512 ⇒ 两臂各 8 hook；零新增参数
for arm, a in R["arms"].items():
    ck(f"{arm} qk hooks == 8", a["patch"].get("n_qkv_hooks"), 8)
    ck(f"{arm} qk params_added == 0", a["patch"].get("params_added"), 0)

# 判据②读数来自 gnorm_monitor（口径），逐臂
for arm, a in R["arms"].items():
    ck(f"{arm} gnorm_max present",
       isinstance(a["gnorm_monitor"]["gnorm_max"], (int, float)), True)

# 口径对照：与 2626 同 bs/同 token 预算（四点并列的前提）
c5 = R["criteria"]["criterion_5_calibre"]
ck("calibre token_budget_same", c5["token_budget_same"], True)
ck("calibre batch_size_same", c5["batch_size_same"], True)
ck("calibre qk_norm_same", c5["qk_norm_same"], True)

# ---------------------------------------------------- 报告文本里必须出现这些读数
dn_val = srv[2048]["delta_native"]
ck("analysis 出现 Δnative@2048",
   any(s in REP for s in (f"{dn_val:+.4f}", f"{dn_val:.4f}",
                          f"{dn_val:+.6f}", f"{dn_val:.6f}")), True)
ck("analysis 出现 gate", ("gnorm" in REP) and ("<100" in REP or "< 100" in REP), True)

# ------------------------------------------------------------ 汇总
n_ok = sum(1 for c in checks if c["ok"])
report = {"n_checks": len(checks), "n_ok": n_ok, "n_fail": len(checks) - n_ok,
          "band": BAND, "checks": checks}
json.dump(report, open(os.path.join(HERE, "reconcile_report.json"), "w"),
          indent=2, ensure_ascii=False)
print(f"reconcile: {n_ok}/{len(checks)} OK")
if n_ok != len(checks):
    raise SystemExit(1)
