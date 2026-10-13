"""2624 · 报告数字对账（每个写进 analysis/README/MODEL 的数字都必须能回到
`results.json` / 日志）。

把对账**脚本化**：逐条把 `analysis.md` / `README.md` 里引用的数字与 `results.json`
重新算一遍，并独立复算几何/公式恒等式。

用法：python3 reconcile_numbers.py      # 任一 mismatch 以非零码退出；写 reconcile_report.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

R = json.load(open(os.path.join(HERE, "results.json")))
EV = json.load(open(os.path.join(HERE, "eval.json")))
CFG = json.load(open(os.path.join(HERE, "config.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
SMOKE = json.load(open(os.path.join(HERE, "smoke", "smoke.json")))
CPU = json.load(open(os.path.join(HERE, "smoke", "cpu_smoke.json")))
CAP = json.load(open(os.path.join(HERE, "controls", "capability_analysis.json")))

checks = []


def read(name):
    p = os.path.join(HERE, name)
    return open(p).read().replace("\u2212", "-") if os.path.exists(p) else ""


AN = read("analysis.md")
RM = read("README.md")


def ck(label, got, want, tol=0.0):
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        ok = abs(got - want) <= tol
    else:
        ok = (got == want)
    checks.append({"label": label, "ok": bool(ok), "got": got, "want": want})


BAND = 0.014
C1 = R["criteria"]["criterion_1_gnorm_explosion_reproduction"]
C2 = R["criteria"]["criterion_2_recipe_equality"]
C3 = R["criteria"]["criterion_3_answer_ce_slope_and_memory"]
C4 = R["criteria"]["criterion_4_no_extrapolation"]
C5 = R["criteria"]["criterion_5_quality_this_point"]
ST = R["convergence_and_stability"]["by_arm"]
NAT = C5["this_point_native"]
FULL_ARM, WIN_ARM = "full-1024-seed1", "win512-1024-seed1"


def _band(x):
    if abs(x) < BAND:
        return "不亏（噪声带内）"
    return "新算子更好（越界）" if x > 0 else "新算子更差（越界）"


# ---------------------------------------------------------- 判据判断点①（复现）
ck("Δ1024(seed1) = full − win（native）",
   round(NAT["bpc_full_1024_seed1"] - NAT["bpc_window512_1024_seed1"], 6),
   NAT["delta_bpc_T1024_window512_seed1"], 1e-9)
ck("R1 判定 = full.gnorm_max > 100",
   C1["R1_passed"], ST[FULL_ARM]["gnorm_max"] > C1["R1_abs_gt"])
ck("R2 判定 = full > 20×win",
   C1["R2_passed"], ST[FULL_ARM]["gnorm_max"] > C1["R2_ratio_gt"] * ST[WIN_ARM]["gnorm_max"])
_expected = ("reproduced" if (C1["R1_passed"] and C1["R2_passed"]) else
             "not_reproduced" if (not C1["R1_passed"] and not C1["R2_passed"]) else
             "partial")
ck("verdict 与两条判据一致", C1["verdict"], _expected)
ck("verdict_text 与 verdict 对应（不可归因 vs 干净 Δ）",
   ("永久不可归因" in C1["verdict_text"]) if C1["verdict"] != "not_reproduced"
   else ("未复现" in C1["verdict_text"]), True)
ck("seed0 参照 gnorm_max 未被改（1740.92）",
   C1["seed0_reference"]["full_gnorm_max"], 1740.92, 1e-9)

# ---------------------------------------------------------- 判据②（配方一致）
ck("两臂 seed 同为 1", C2["seed_both_arms"], 1, 0)
ck("两臂参数相同",
   R["arms"][FULL_ARM]["params"], R["arms"][WIN_ARM]["params"], 0)
ck("两臂 token 相同且 = 98.3M",
   C2["tokens_by_arm"][FULL_ARM], C2["tokens_by_arm"][WIN_ARM], 0)
ck("token 预算 = 12000×8×1024", C2["tokens_by_arm"][FULL_ARM], 12000 * 8 * 1024, 0)
_ff = C2["field_by_field"]
ck("逐字段仅差 train_window",
   sorted(k for k in _ff[FULL_ARM] if _ff[FULL_ARM][k] != _ff[WIN_ARM][k]),
   ["train_window"])
ck("两臂 field seed = 1", sorted({_ff[FULL_ARM]["seed"], _ff[WIN_ARM]["seed"]}), [1])
ck("batch = 8（与 Y2 一致）", _ff[WIN_ARM]["batch_size"], 8, 0)

# ---------------------------------------------------------- 判据③（斜率 + 显存）
for arm in (FULL_ARM, WIN_ARM):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "runs", arm,
                                                     "train_log.jsonl"))]
    n = C2["tokens_by_arm"][arm] // (8 * 1024)
    pts = [(r["step"], r["loss"]) for r in rows
           if max(0, n - 800) <= r["step"] <= n]
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    slope = round(sum((p[0] - mx) * (p[1] - my) for p in pts) / den * 1000, 5)
    ck(f"末窗斜率 {arm} 可复算", C3["by_arm"][arm]["last_window_slope_nats_per_1k"],
       slope, 1e-6)
    ck(f"末日志 loss {arm} 与日志一致", C3["by_arm"][arm]["final_logged_loss"],
       rows[-1]["loss"], 1e-9)
ck("Δ(末日志 loss) = full − win", C3["delta_at_final_loss"],
   round(C3["by_arm"][FULL_ARM]["final_logged_loss"]
         - C3["by_arm"][WIN_ARM]["final_logged_loss"], 4), 1e-9)
ck("评测评台 bpc 与 arms 块一致", C3["held_out_bpc_by_arm"][WIN_ARM],
   R["arms"][WIN_ARM]["battery"]["bpc_native"], 1e-9)
ck("峰值未越 5GB net", C3["memory"]["max_peak_gb"] < 5.0, True)
ck("跑前余量已记", isinstance(C3["memory"]["free_mb_before"], (int, float)), True)

# ---------------------------------------------------------- 判据④（禁外推）
ck("n=2 且未外推", (C4["n_seeds"], C4["extrapolated"]), (2, False))

# ---------------------------------------------------------- 质量/三点/带宽
ck("band_class(seed1) 与 Δ 一致", C5["band_class_T1024_seed1"], _band(
   NAT["delta_bpc_T1024_window512_seed1"]))
ck("sign(seed1)", C5["sign_T1024_seed1"],
   "0" if abs(NAT["delta_bpc_T1024_window512_seed1"]) < BAND
   else ("+" if NAT["delta_bpc_T1024_window512_seed1"] > 0 else "-"))
_trend = C5["three_point_trend_native"]
_pts = C5["points_native_seed0_and_seed1"]
ck("三点趋势 = T256 + T512 + T1024(seed1)", [p["T"] for p in _trend], [256, 512, 1024])
ck("趋势每点同相对刻度 50%",
   all(round(p["window"] / p["T"], 6) == 0.5 for p in _trend), True)
for p in _trend:
    ck(f"趋势 band_class@{p['T']}", p["band_class"], _band(p["delta_native"]))
ck("四点（含 seed0 参照）来源齐备",
   [p["source"] for p in _pts], ["X3(2622)", "X3(2622)", "Y2(2623, seed0)",
                                  "this(2624, seed1)"])
ck("趋势第三点 = 本点 seed1 Δ", _trend[-1]["delta_native"],
   NAT["delta_bpc_T1024_window512_seed1"], 1e-9)
ck("n=2 配对：seed0 Δ 与 2623 一致",
   C5["n2_delta_pair"]["seed0_2623"],
   C5["y2_seed0_reference"]["native"]["delta_bpc_T1024_window512"], 1e-9)
ck("n=2 配对：seed1 Δ = 本点 Δ", C5["n2_delta_pair"]["seed1_2624"],
   NAT["delta_bpc_T1024_window512_seed1"], 1e-9)
ck("n=2 均值 = 两点算术平均", C5["n2_delta_pair"]["mean"],
   round((C5["n2_delta_pair"]["seed0_2623"] + C5["n2_delta_pair"]["seed1_2624"]) / 2, 6),
   1e-9)
REL = C5["this_point_release"]
ck("release Δ1024(seed1) = full − win",
   round(REL["bpc_full_1024_seed1"] - REL["bpc_window512_1024_seed1"], 6),
   REL["delta_bpc_T1024_window512_seed1"], 1e-9)
ck("full/win gnorm 比值可复算", C1["ratio_full_over_win"],
   round(C1["full_gnorm_max"] / C1["win_gnorm_max"], 4), 1e-9)

# ---------------------------------------------------------- 算力/墙钟（只作对照）
for row in R["criteria"]["criterion_6_attn_share"]["measured_T1024"]:
    T = row["T"]
    ck(f"T={T} 公式值 = 512T/(512T+786432)", row["attn_share_formula"],
       round(512 * T / (512 * T + 786432), 4), 1e-9)
    ck(f"T={T} r_time = window/full", row["r_time_window_over_full"],
       round(row["step_ms_window"] / row["step_ms_full"], 4), 1e-3)
ck("墙钟 round1 非空", len(SMOKE["cost_measured_step_ms_by_T"]) >= 2, True)
ck("墙钟 round2 非空", len(EV["cost"]["measured"]["model_step"]) >= 2, True)

# ---------------------------------------------------------- 算术恒等式
for arm in CFG["arms"]:
    ck(f"{arm['arm']} tokens = steps×bs×seq", arm["tokens"],
       arm["steps"] * arm["fields"]["batch_size"] * arm["seq_len"], 0)
    ck(f"{arm['arm']} tokens == 98.3M", arm["tokens"], 98_304_000, 0)
ck("cpu_smoke param delta", CPU["invariants"]["param_delta_T1024_minus_T512"], 131072, 0)
ck("cpu_smoke visible_frac 闭式", CPU["invariants"]["window512@T1024_visible_frac"],
   393472 / 524800, 1e-6)
ck("cpu_smoke seed 活性", CPU["seed_liveness"]["alive"], True)
ck("能力面：契约组合可达", all(v["ok"] for v in CAP["forward"].values()), True)
ck("能力面：互斥校验活着",
   all(not v["ok"] for v in CAP["reverse_mutex_present"].values()), True)
ck("变异：全部抓红且逐字节还原",
   (MUT["all_caught"], MUT["restored_bitwise"]), (True, True))

# ------------------------------------------------- 报告文本与 results 一致（抽样）
if AN:
    ck("analysis 出现 seed1 Δ 数字",
       f"{NAT['delta_bpc_T1024_window512_seed1']:+.6f}" in AN, True)
    ck("analysis 出现 release Δ 数字",
       f"{REL['delta_bpc_T1024_window512_seed1']:.6f}" in AN, True)
    ck("analysis 出现 n=2 均值",
       f"{C5['n2_delta_pair']['mean']:.6f}" in AN, True)
    for arm in (FULL_ARM, WIN_ARM):
        ck(f"analysis 引用 {arm} 的 gnorm_max",
           str(ST[arm]["gnorm_max"]) in AN, True)

bad = [c for c in checks if not c["ok"]]
json.dump({"n_checks": len(checks), "n_failed": len(bad), "checks": checks},
          open(os.path.join(HERE, "reconcile_report.json"), "w"),
          indent=2, ensure_ascii=False)
print(f"{len(checks) - len(bad)}/{len(checks)} checks OK")
for c in bad:
    print(f"MISMATCH: {c['label']}: got {c['got']!r} want {c['want']!r}")
sys.exit(1 if bad else 0)
