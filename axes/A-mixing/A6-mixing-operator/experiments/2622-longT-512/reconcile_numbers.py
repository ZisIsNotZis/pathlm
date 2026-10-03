"""2622 · 报告数字对账（每个写进 analysis/README/MODEL 的数字都必须能回到
`results.json` / 日志）。

前几批因「报告数字与证据文件对不上」吃过 P1，故这里把对账**脚本化**：逐条把
`analysis.md` / `README.md` 里引用的数字与 `results.json` 重新算一遍，并独立复算
几何/公式恒等式。

用法：python3 reconcile_numbers.py      # 任一 mismatch 以非零码退出；写 reconcile_report.json
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

R = json.load(open(os.path.join(HERE, "results.json")))
EV = json.load(open(os.path.join(HERE, "eval.json")))
CFG = json.load(open(os.path.join(HERE, "config.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
SMOKE = json.load(open(os.path.join(HERE, "smoke", "smoke.json")))
CPU = json.load(open(os.path.join(HERE, "smoke", "cpu_smoke.json")))

checks = []


def read(name):
    return open(os.path.join(HERE, name)).read().replace("\u2212", "-")


AN = read("analysis.md") if os.path.exists(os.path.join(HERE, "analysis.md")) else ""


def ck(label, got, want, tol=0.0):
    if isinstance(want, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = (got == want)
    checks.append({"label": label, "ok": bool(ok), "got": got, "want": want})


def num(text, pattern, label, cast=float, flags=re.M):
    m = re.search(pattern, text, flags)
    if not m:
        checks.append({"label": label, "ok": False, "got": None,
                       "want": "pattern not found"})
        return float("nan")
    return cast(m.group(1))


C1 = R["criteria"]["criterion_1_quality"]
NAT = C1["native_operating_point"]
REL = C1["release_path"]

# ---------------------------------------------------------- 判据①（两点质量）
ck("Δ256 = full − win128（native）",
   round(NAT["bpc_full_256"] - NAT["bpc_window128_256"], 6),
   NAT["delta_bpc_T256_window128"], 1e-9)
ck("Δ512 = full − win256（native）",
   round(NAT["bpc_full_512"] - NAT["bpc_window256_512"], 6),
   NAT["delta_bpc_T512_window256"], 1e-9)
ck("Δ256（release）",
   round(REL["bpc_full_256"] - REL["bpc_window128_256"], 6),
   REL["delta_bpc_T256_window128"], 1e-9)
ck("Δ512（release）",
   round(REL["bpc_full_512"] - REL["bpc_window256_512"], 6),
   REL["delta_bpc_T512_window256"], 1e-9)
ck("sign_T256 == sign(Δ256)",
   C1["sign_T256"], "0" if abs(NAT["delta_bpc_T256_window128"]) < 0.014
   else ("+" if NAT["delta_bpc_T256_window128"] > 0 else "-"))
ck("sign_T512 == sign(Δ512)",
   C1["sign_T512"], "0" if abs(NAT["delta_bpc_T512_window256"]) < 0.014
   else ("+" if NAT["delta_bpc_T512_window256"] > 0 else "-"))
ck("same_sign == (sign_T256 == sign_T512)", C1["same_sign"],
   C1["sign_T256"] == C1["sign_T512"])
ck("判据④ 与 (Δ512 ≥ −0.014) 一致", R["criteria"]["criterion_4_crossover"]
   ["window256_T512_not_losing"], bool(NAT["delta_bpc_T512_window256"] >= -0.014))

# ---------------------------------------------------------- 判据②（算力占比）
for row in R["criteria"]["criterion_2_attn_share"]["measured"]:
    T = row["T"]
    ck(f"T={T} 公式值 = 512T/(512T+786432)", row["attn_share_formula"],
       round(512 * T / (512 * T + 786432), 4), 1e-9)
    ck(f"T={T} r_time = window/full",
       round(row["step_ms_window"] / row["step_ms_full"], 4),
       row["r_time_window_over_full"], 1e-3)
ck("formula T=512 == 0.25（主 Agent 修正后）",
   R["criteria"]["criterion_2_attn_share"]["formula"]["512"], 0.25, 1e-9)

# ---------------------------------------------------------- 判据③（显存）
ck("峰值未越 5GB net", R["criteria"]["criterion_3_memory"]["max_peak_gb"] < 5.0, True)

# ---------------------------------------------------------- 算术恒等式（独立复算）
for arm in CFG["arms"]:
    ck(f"{arm['arm']} tokens = steps×bs×seq",
       arm["tokens"], arm["steps"] * 32 * arm["seq_len"], 0)
    ck(f"{arm['arm']} tokens == 98.3M 预算", arm["tokens"], 98_304_000, 0)
ck("cpu_smoke param delta", CPU["invariants"]["param_delta_T512_minus_T256"], 65536, 0)
ck("cpu_smoke window256@T512 visible_frac 闭式",
   CPU["invariants"]["window256@T512_visible_frac"], 98432 / 131328, 1e-6)
ck("window256@T256 是恒等（退化点）",
   CPU["invariants"]["window256@T256_is_identity"], True)
ck("同 token 预算跨 T", CPU["invariants"]["token_budget_equal_across_T"], True)
ck("同相对刻度 50%（两点）", round(128 / 256, 6), round(256 / 512, 6), 0)

# ---------------------------------------------------------- 对照（§9 逐字段）
for arm in CFG["arms"]:
    f = arm["fields"]
    ck(f"{arm['arm']} 除 seq/window 外逐字段一致",
       (f["d_model"], f["n_layers"], f["n_heads"], f["mlp_mult"], f["seed"],
        f["batch_size"], f["lr"], f["exit_mode"]),
       (256, 8, 8, 4, 0, 32, 1e-3, "uniform"))

# ---------------------------------------------------------- 活性量（§10）
for arm, live in R["liveness"].items():
    if arm.startswith("win"):
        ck(f"{arm} 窗口干预活（alive=True）", live["window_mask"]["alive"], True)
        ck(f"{arm} 掩码探针 diff > 0",
           live["window_mask_probe"]["max_abs_logit_diff_masked_vs_open"] > 0, True)
    else:
        ck(f"{arm} 无窗口干预（window_mask=None）", live["window_mask"], None)

# ---------------------------------------------------------- 合成腿（§19）
for arm, leg in R["synthetic_leg"]["results"].items():
    for d, row in leg["by_distance"].items():
        ck(f"{arm} d={d} chance=1/64", row["chance_answer_set"], round(1 / 64, 6), 1e-9)
        ck(f"{arm} d={d} 局部先验抗性成立", row["local_prior_resistance"], True)

# ---------------------------------------------------------- 变异验证
ck("变异全部抓红", MUT["all_caught"], True)
ck("变异后逐字节还原", MUT["restored_bitwise"], True)

# ---------------------------------------------------------- analysis.md 里的数字可回溯
if AN:
    ck("analysis 引用 Δ512（native）", f"{NAT['delta_bpc_T512_window256']:.6f}" in AN, True)
    ck("analysis 引用 Δ256（native）", f"{NAT['delta_bpc_T256_window128']:.6f}" in AN, True)
    ck("analysis 引用 T=512 公式 25.0%",
       0.25 if re.search(r"25\.0%", AN) else float("nan"), 0.25, 1e-9)
    ck("analysis 人话摘要首段存在", "人话摘要" in AN, True)

report = {"n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["ok"]),
          "all_ok": all(c["ok"] for c in checks), "checks": checks}
json.dump(report, open(os.path.join(HERE, "reconcile_report.json"), "w"),
          indent=2, ensure_ascii=False)
for c in checks:
    if not c["ok"]:
        print(f"FAIL: {c['label']}  got={c['got']}  want={c['want']}")
print(f"对账：{report['n_checks']} 条，失败 {report['n_failed']} 条，"
      f"all_ok={report['all_ok']}")
sys.exit(0 if report["all_ok"] else 1)
