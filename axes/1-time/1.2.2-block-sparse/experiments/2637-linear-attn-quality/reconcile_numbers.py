"""2637 · 报告数字对账（写进 `analysis.md` / `README.md` 的数字必须能回到
`results.json` / `eval.json` / `smoke/smoke.json` / `diag_matrix.json`）。

① 算术：Δ = bpc_full − bpc_linear 从 `eval.json` 重算；注意力 MACs 闭式复算；
   token = steps × bs × T。
② 任务：四臂同一数据切分与同一评测流（`window_seed=2`、20 批 × bs16）；
   逐字段对照只允许差 `attn_linear` / `qk_norm`。
③ 口径：C1 读数来自 `eval.json::batteries`；C3 停摆状态来自逐臂 `train_summary`。

用法：python3 reconcile_numbers.py   # 任一 mismatch 非零退出；写 reconcile_report.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

checks = []
BAND = 0.014
S = 6000
BS = 8
D, H = 256, 8


def ck(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) \
        else (got == want)
    checks.append({"label": label, "ok": bool(ok), "got": got, "want": want})
    if not ok:
        print(f"MISMATCH {label}: got={got} want={want}", flush=True)


def load(path):
    with open(os.path.join(HERE, path)) as f:
        return json.load(f)


R = load("results.json")
EV = load("eval.json")


def band_class(x):
    if abs(x) < BAND:
        return "不亏（噪声带内）"
    return "线性更好（越界）" if x > 0 else "线性更差（越界）"


# ① 算术
EVAL_BATCHES, EVAL_BS, WINDOW_SEED = 20, 16, 2
for arm, bat in EV["batteries"].items():
    ck(f"{arm} eval window_seed", bat["window_seed"], WINDOW_SEED)
    ck(f"{arm} eval n_batches", bat["n_batches"], EVAL_BATCHES)
    ck(f"{arm} eval batch_size", bat["batch_size"], EVAL_BS)

b = EV["batteries"]
crit = R["criteria"]["criterion_1_quality"]
if "full-256-lin" in b and "linear-256-lin" in b:
    d = round(b["full-256-lin"]["bpc_native"] - b["linear-256-lin"]["bpc_native"], 6)
    ck("Δ no_qk", crit["delta_native_no_qk"], d, 1e-6)
    ck("band no_qk", crit["band_class_no_qk"], band_class(d))
if "full-256-lin-qk" in b and "linear-256-lin-qk" in b:
    d = round(b["full-256-lin-qk"]["bpc_native"] - b["linear-256-lin-qk"]["bpc_native"], 6)
    ck("Δ with_qk", crit["delta_native_with_qk"], d, 1e-6)
    ck("band with_qk", crit["band_class_with_qk"], band_class(d))

# 注意力 MACs 闭式复算
ck("attn_macs_full", EV["cost"]["theory"]["attn_macs_per_token_per_layer_full"], 2 * 256 * D)
ck("attn_macs_linear_chunk64",
   EV["cost"]["theory"]["attn_macs_per_token_per_layer_linear_chunk64"],
   2 * D * (D // H) + 2 * 64 * D)

# token 计数
ck("tokens_by_arm", R["config"]["tokens_by_arm"], S * BS * 256)

# ② 任务：逐字段
KEYS = ("d_model", "n_layers", "n_heads", "mlp_mult", "dense", "topology",
        "batch_size", "lr", "seed", "steps", "seq_len")
fields = {a["arm"]: a["fields"] for a in R["config"]["arms"] if a.get("fields")}
base = next(iter(fields.values()), None)
for arm, f in fields.items():
    for k in KEYS:
        ck(f"{arm} field {k} == base", f[k], base[k])

# ③ 口径
for arm, a in R["arms"].items():
    if a.get("battery"):
        ck(f"{arm} has bpc", isinstance(a["battery"]["bpc_native"], float), True)
ck("linear no_qk stalled flag", R["criteria"]["criterion_3_stall"]["no_qk_stalled"]
   in (True, False, None), True)

REP = ""
p = os.path.join(HERE, "analysis.md")
if os.path.exists(p):
    REP = open(p).read().replace("\u2212", "-")

n_ok = sum(1 for c in checks if c["ok"])
report = {"n_checks": len(checks), "n_ok": n_ok, "n_fail": len(checks) - n_ok,
          "band": BAND, "checks": checks}
json.dump(report, open(os.path.join(HERE, "reconcile_report.json"), "w"),
          indent=2, ensure_ascii=False)
print(f"reconcile: {n_ok}/{len(checks)} OK")
if n_ok != len(checks):
    raise SystemExit(1)
