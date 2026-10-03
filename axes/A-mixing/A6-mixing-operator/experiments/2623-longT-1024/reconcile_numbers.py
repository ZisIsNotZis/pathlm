"""2623 · 报告数字对账（每个写进 analysis/README/MODEL 的数字都必须能回到
`results.json` / 日志）。

把对账**脚本化**：逐条把 `analysis.md` / `README.md` 里引用的数字与 `results.json`
重新算一遍，并独立复算几何/公式恒等式。

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


C1 = R["criteria"]["criterion_1_quality"]
NAT = C1["this_point_native"]
REL = C1["this_point_release"]
BAND = 0.014

# ---------------------------------------------------------- 判据①（第三点质量）
ck("Δ1024 = full − win512（native）",
   round(NAT["bpc_full_1024"] - NAT["bpc_window512_1024"], 6),
   NAT["delta_bpc_T1024_window512"], 1e-9)
ck("Δ1024（release）",
   round(REL["bpc_full_1024"] - REL["bpc_window512_1024"], 6),
   REL["delta_bpc_T1024_window512"], 1e-9)
def _band_class(x):
    if abs(x) < BAND:
        return "不亏（噪声带内）"
    return "新算子更好（越界）" if x > 0 else "新算子更差（越界）"


ck("sign_T1024 == 带宽分类",
   C1["sign_T1024"], "0" if abs(NAT["delta_bpc_T1024_window512"]) < BAND
   else ("+" if NAT["delta_bpc_T1024_window512"] > 0 else "-"))
ck("band_class_T1024",
   C1["band_class_T1024"], _band_class(NAT["delta_bpc_T1024_window512"]))

# ---------------------------------------------------------- 三点并列（跨 run 引用 X3）
X3 = C1["x3_reference"]
ck("X3 参考可用", X3["available"], True)
tp = {p["T"]: p for p in C1["three_points_native"]}
ck("三点含 T=256/512/1024", sorted(tp.keys()), [256, 512, 1024])
for T in (256, 512, 1024):
    ck(f"three_points T={T} 带宽分类一致", tp[T]["band_class"],
       _band_class(tp[T]["delta_native"]))
ck("three_points 每点同相对刻度 50%",
   all(round(p["window"] / p["T"], 6) == 0.5 for p in C1["three_points_native"]), True)

# ---------------------------------------------------------- 判据②（算力占比）
for row in R["criteria"]["criterion_2_attn_share"]["measured_T1024"]:
    T = row["T"]
    ck(f"T={T} 公式值 = 512T/(512T+786432)", row["attn_share_formula"],
       round(512 * T / (512 * T + 786432), 4), 1e-9)
    ck(f"T={T} r_time = window/full",
       round(row["step_ms_window"] / row["step_ms_full"], 4),
       row["r_time_window_over_full"], 1e-3)
ck("formula T=1024 == 0.40",
   R["criteria"]["criterion_2_attn_share"]["formula"]["1024"], 0.4, 1e-9)
ck("X3 实测占比 256 并列", R["criteria"]["criterion_2_attn_share"]
   ["measured_X3_256_512"]["256"], X3["attn_share_measured"]["256"], 1e-9)
ck("X3 实测占比 512 并列", R["criteria"]["criterion_2_attn_share"]
   ["measured_X3_256_512"]["512"], X3["attn_share_measured"]["512"], 1e-9)

# ---------------------------------------------------------- 混淆判定（主 Agent 裁决）
STAB = R["convergence_and_stability"]["by_arm"]
ck("full 臂 gnorm_max 记录", STAB["full-1024"]["gnorm_max"], 1740.92, 1e-6)
ck("win 臂 gnorm_max 记录", STAB["win512-1024"]["gnorm_max"], 7.35, 1e-6)
ck("基准臂不稳定（full gnorm_max > 100×win）",
   STAB["full-1024"]["gnorm_max"] > 100 * STAB["win512-1024"]["gnorm_max"], True)
ck("结论降级为「不可判」", C1["interpretation"], "不可判（基准臂不稳定）")
ck("band 分类字面保留（不得改判据）", C1["band_class_T1024"], "新算子更好（越界）")
ck("三点形状含不可判", "不可判" in C1["three_point_shape"], True)

# ---------------------------------------------------------- 判据③（显存）
ck("峰值未越 5GB net", R["criteria"]["criterion_3_memory"]["max_peak_gb"] < 5.0, True)
ck("跑前余量已记", isinstance(R["criteria"]["criterion_3_memory"]["free_mb_before"],
                            (int, float)), True)

# ---------------------------------------------------------- 判据④（墙钟两轮）
ck("墙钟 round1（preflight）非空", len(SMOKE["cost_measured_step_ms_by_T"]) >= 2, True)
ck("墙钟 round2（eval）非空", len(EV["cost"]["measured"]["model_step"]) >= 2, True)

# ---------------------------------------------------------- 算术恒等式（独立复算）
for arm in CFG["arms"]:
    ck(f"{arm['arm']} tokens = steps×bs×seq",
       arm["tokens"], arm["steps"] * arm["fields"]["batch_size"] * arm["seq_len"], 0)
    ck(f"{arm['arm']} tokens == 98.3M 预算", arm["tokens"], 98_304_000, 0)
ck("cpu_smoke param delta (T1024−T512)",
   CPU["invariants"]["param_delta_T1024_minus_T512"], 131072, 0)
ck("cpu_smoke window512@T1024 visible_frac 闭式",
   CPU["invariants"]["window512@T1024_visible_frac"], 393472 / 524800, 1e-6)
ck("window1024@T1024 是恒等（退化点）",
   CPU["invariants"]["window1024@T1024_is_identity"], True)
ck("window512@T1024 非退化",
   CPU["invariants"]["window512@T1024_is_nondegenerate"], True)
ck("同 token 预算跨两臂", CPU["invariants"]["token_budget_equal_across_arms"], True)
ck("三点同相对刻度 50%",
   all(round(p["ratio"], 6) == 0.5 for p in CPU["three_points"]), True)

# ---------------------------------------------------------- 对照（§9 逐字段）
for arm in CFG["arms"]:
    f = arm["fields"]
    ck(f"{arm['arm']} 除 seq/window 外逐字段一致",
       (f["d_model"], f["n_layers"], f["n_heads"], f["mlp_mult"], f["seed"],
        f["batch_size"], f["lr"], f["exit_mode"]),
       (256, 8, 8, 4, 0, 8, 1e-3, "uniform"))

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
    ck("analysis 引用 Δ1024（native）", f"{NAT['delta_bpc_T1024_window512']:.6f}" in AN, True)
    ck("analysis 引用 T=1024 公式 40.0%", 0.4 if re.search(r"40\.0%", AN) else float("nan"),
       0.4, 1e-9)
    ck("analysis 人话摘要首段存在", "人话摘要" in AN, True)
    ck("analysis 结论降级为不可判", "不可判" in AN, True)
    ck("analysis 三点形状「更差 → 不亏 → 不可判」", "更差 → 不亏 → 不可判" in AN, True)
    ck("analysis 含长 T 稳定性小节", "长 T 训练稳定性" in AN, True)
    ck("analysis 记基准 gnorm 1740.92", "1740.92" in AN, True)
    # 合成腿表格与 results.json 逐一对应（§12 证据可回溯）
    for arm, leg in R["synthetic_leg"]["results"].items():
        line = ", ".join(f"{d}:{row['accuracy']:.3f}"
                         for d, row in leg["by_distance"].items())
        ck(f"analysis 合成腿表 {arm} 与 results 一致", line in AN, True)

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
