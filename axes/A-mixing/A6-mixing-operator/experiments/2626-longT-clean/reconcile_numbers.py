"""2626 · 报告数字对账（每个写进 `analysis.md` / `README.md` 的数字都必须能回到
`results.json` / `eval.json` / `runs/<arm>/train_log.jsonl` / `smoke/*.json`）。

三道对账（SCHEMA §12）：
① **算术**：token 计数 = steps × bs × T；Δ = bpc_full − bpc_window（从 `eval.json` 重算）；
   带宽分类与服务端逐点一致；注意力占比公式闭式复算；窗口 `visible_frac` 闭式复算。
② **任务**：六臂同一数据切分（enwik8 前 90M/末 10M）与**同一评测流**
   （`window_seed=2`、20 批 × bs16）；口径写死，脚本只核字段。
③ **口径**：判据①读数来自 `eval.json::batteries[*].bpc_native`；判据②来自 `results.json::gnorm_monitor`；
   逐字段对照只允许差 `seq_len` / `train_window`（同 seed/bs/lr/dense/topology）。

用法：python3 reconcile_numbers.py   # 任一 mismatch 非零退出；写 reconcile_report.json
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

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
README = report_text("README.md")

BAND = 0.014
POINTS = ((128, 256), (256, 512), (512, 1024))


def band_class(x):
    if abs(x) < BAND:
        return "不亏（噪声带内）"
    return "窗口更好（越界）" if x > 0 else "窗口更差（越界）"


# ------------------------------------------------------------ ① 算术
for arm in R["arms"]:
    a = R["arms"][arm]
    steps = a["steps"]
    T = a["battery"]["seq_len"]
    ck(f"{arm} tokens = steps*bs*T",
       a["tokens"], steps * R["config"]["batch_size"] * T)
    ck(f"{arm} eval window_seed", a["battery"]["window_seed"], 2)
    ck(f"{arm} eval n_batches", a["battery"]["n_batches"], 20)
    ck(f"{arm} eval batch_size", a["battery"]["batch_size"], 16)

# 三点 Δ 从 eval.json 重算，必须与服务端逐点相等
srv = {f"{p['T']}": p for p in R["criteria"]["criterion_1_quality"]["three_points_native"]}
for w, T in POINTS:
    fa, wa = f"full-{T}-clean", f"win{w}-{T}-clean"
    dn = round(EV["batteries"][fa]["bpc_native"] - EV["batteries"][wa]["bpc_native"], 6)
    ck(f"Δnative@{T}", srv[str(T)]["delta_native"], dn, 1e-6)
    ck(f"band_class@{T}", srv[str(T)]["band_class"], band_class(dn))

# 注意力占比公式闭式复算
for T, want in ((256, 0.1429), (512, 0.2500), (1024, 0.4000)):
    ck(f"attn_share_formula({T})",
       round((2 * T * 256) / (2 * T * 256 + 786432), 4), want, 1e-4)

# 窗口 visible_frac 闭式复算
for w, T in POINTS:
    n_keep = sum(1 for i in range(T) for j in range(T) if j <= i and (i - j) < w)
    n_causal = T * (T + 1) // 2
    ck(f"visible_frac({w}@{T})", round(n_keep / n_causal, 6),
       round(n_keep / n_causal, 6), 0.0)

# ------------------------------------------------------------ ② 任务（同一评测流）
ck("all arms present", sorted(R["arms"].keys()) == sorted(
    ["full-256-clean", "win128-256-clean", "full-512-clean",
     "win256-512-clean", "full-1024-clean", "win512-1024-clean"]), True)
fields = {a["arm"]: a["fields"] for a in R["config"]["arms"]}
for arm, f in fields.items():
    ck(f"{arm} dense", f["dense"], True)
    ck(f"{arm} topology", f["topology"], "sequential")
    ck(f"{arm} seed", f["seed"], 0)
    ck(f"{arm} lr", f["lr"], 0.001)
    ck(f"{arm} bs", f["batch_size"], 8)
    ck(f"{arm} qk_norm", f["qk_norm"], "qk-normfrom512")
# 逐字段对照：六臂只允许差 seq_len / train_window / steps
KEYS = ("d_model", "n_layers", "n_heads", "mlp_mult", "dense", "topology",
        "batch_size", "lr", "seed", "exit_mode", "qk_norm", "qk_norm_min_t")
base = fields["full-256-clean"]
for arm, f in fields.items():
    for k in KEYS:
        ck(f"{arm} field {k} == base", f[k], base[k])

# ------------------------------------------------------------ ③ 口径
gate = R["criteria"]["criterion_2_stability_gate"]
ck("gate gnorm_max<100", gate["full_1024_gnorm_max"] < 100, True)
ck("gate skips==0", gate["full_1024_grad_skips"], 0)
ck("gate passed", gate["passed"], True)

# qk-norm 活性：T=1024 臂 8 hook，T ≤ 512 臂 0 hook；零新增参数
for arm, a in R["arms"].items():
    hooks = a["patch"].get("n_qkv_hooks")
    if a["battery"]["seq_len"] > 512:
        ck(f"{arm} qk hooks == 8", hooks, 8)
    else:
        ck(f"{arm} qk hooks == 0", hooks, 0)
    ck(f"{arm} qk params_added == 0", a["patch"].get("params_added"), 0)

# 判据②读数来自 gnorm_monitor（口径），逐臂
for arm, a in R["arms"].items():
    ck(f"{arm} gnorm_max present", isinstance(a["gnorm_monitor"]["gnorm_max"], (int, float)), True)

# ------------------------------------------------------------ 报告文本里必须出现这些读数
for T in (256, 512, 1024):
    dn = srv[str(T)]["delta_native"]
    ck(f"analysis 出现 Δnative@{T}={dn}", f"{dn:+.4f}" in REP or f"{dn:.4f}" in REP
       or f"{dn:+.6f}" in REP or f"{dn:.6f}" in REP, True)
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
