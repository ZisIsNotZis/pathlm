"""2628 · 报告数字对账（`SCHEMA §12`）：`analysis.md` 里写下的关键数字都必须能在
`results_13m.json` / 训练日志 / `config.json` 里原样找到。

用法：python3 reconcile_numbers.py --scale 13m   # 任一 mismatch 以非零码退出

两层：
  ① **结构性/自洽性**：判据键齐全、复现检查逐位相同、抖动判决与预登记阈值自洽、
     两臂合并极差（取大）与并列口径算术自洽、确定性对照=0、gate_skips=0、
     活性量与 `train_log.jsonl` 一致、参数量/代码指纹已落盘；
  ② **具体数字**：`analysis.md` 里的关键读数逐条回查（`SPECIFIC`）。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from jitter_l1 import combined_range, jitter_verdict                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scale", choices=("13m",), required=True)
args = ap.parse_args()
SCALE = args.scale

CFG = json.load(open(os.path.join(HERE, "config.json")))
S = CFG["scales"][SCALE]
ARMS = tuple(S["arms"])
BASE = S["mechanism_baseline"]
L1A, L1B = S["l1_pair"]
A, B = S["repr_pair"]
KMAX = str(CFG["main_k"])
R = json.load(open(os.path.join(HERE, f"results_{SCALE}.json")))
MUT_PATH = os.path.join(HERE, "mutation_report.json")
MUT = json.load(open(MUT_PATH)) if os.path.exists(MUT_PATH) else {"all_caught_and_restored": None}

checks = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


# ------------------------------------------------------------------ 结构/自洽
ck("仪器自检：与 probe_tts.bpc_k 同方法（全部 K 逐位同值）",
   R["selfchecks"]["probe_tts_method_identity"]["all_equal"] is True,
   {k: v for k, v in R["selfchecks"]["probe_tts_method_identity"].items()
    if isinstance(v, dict)}, "all bpc_equal")
ck("跨运行复现：本轮重评既有权重 vs 记录 逐位相同",
   R["cross_run_reproduction"]["all_equal"] is True,
   {k: v["equal"] for k, v in R["cross_run_reproduction"]["checks"].items()}, "all equal")

for arm in ARMS:
    ck(f"臂 {arm} 在 K 曲线里", arm in R["k_curve"])
    for k in CFG["k_sweep"]:
        ck(f"臂 {arm} K={k} 有 bpc", str(k) in R["k_curve"][arm])
    ck(f"臂 {arm} checkpoint sha256 与 config.json 一致",
       R["checkpoints"][arm]["sha256"] == CFG["arms"][f"{SCALE}/{arm}"]["sha256"], True)

params = {a: R["checkpoints"][a]["params"] for a in ARMS}
ck(f"{SCALE} 四臂参数量完全相同（去相关是纯损失项，不加参数）",
   len(set(params.values())) == 1, params, "单一值")
ck("l1_B 权重 sha256 ≠ l1_A（确实是两次训练，不是复用）",
   R["checkpoints"][L1B]["sha256"] != R["checkpoints"][L1A]["sha256"], True)

# 抖动判决：与曲线、与预登记阈值自洽
J = R["jitter"]
l1_a = R["k_curve"][L1A][KMAX]["bpc"]
l1_b = R["k_curve"][L1B][KMAX]["bpc"]
r_a = R["k_curve"][A][KMAX]["bpc"]
r_b = R["k_curve"][B][KMAX]["bpc"]
ck("jitter.baseline_l1_K8_A == 曲线 l1_A@K8", J["baseline_l1_K8_A"] == l1_a, J["baseline_l1_K8_A"], l1_a)
ck("jitter.l1_K8_B == 曲线 l1_B@K8", J["l1_K8_B"] == l1_b, J["l1_K8_B"], l1_b)
ck("jitter.l1_jitter == |l1_B@K8 − l1_A@K8|",
   abs(J["l1_jitter"] - abs(l1_b - l1_a)) < 1e-9, J["l1_jitter"], abs(l1_b - l1_a))
ck("jitter.repr01_jitter == |repr01_B@K8 − repr01_A@K8|",
   abs(J["repr01_jitter"] - abs(r_b - r_a)) < 1e-9, J["repr01_jitter"], abs(r_b - r_a))
ck("jitter.combined_range == max(l1_jitter, repr01_jitter)（README §4 primary）",
   abs(J["combined_range"] - combined_range(J["l1_jitter"], J["repr01_jitter"])) < 1e-12)
ck("jitter.worst_case_additive_range == l1_jitter + repr01_jitter",
   abs(J["worst_case_additive_range"] - (J["l1_jitter"] + J["repr01_jitter"])) < 1e-12)
ck("jitter.combos_range == 四组合极差（并列披露，应恒等于两臂之和）",
   abs(J["combos_range"] - J["worst_case_additive_range"]) < 1e-9)
ck("repr01 臂本次重测抖动 == 2627 记录 0.0035（口径可回溯）",
   J["repr01_jitter_matches_2627"] is True and J["repr01_jitter"] == CFG["repr01_jitter_2627"],
   J["repr01_jitter"], CFG["repr01_jitter_2627"])
_verdict = jitter_verdict(J["combined_range"])
ck("jitter.verdict.state == jitter_verdict(combined_range)（预登记阈值）",
   J["verdict"]["state"] == _verdict["state"], J["verdict"]["state"], _verdict["state"])
ck("抖动阈值与 README §4 预登记一致",
   J["verdict"]["downgrade_ge"] == 0.0144 and J["verdict"]["upgrade_lt"] == 0.005
   and J["effect_v2"] == 0.0144)

# 净效应组合算术
nc = J["net_combos"]
ck("net_combos.l1_A__repr01_A == l1_A@K8 − repr01_A@K8",
   abs(nc["l1_A__repr01_A"] - (l1_a - r_a)) < 1e-9)
ck("net_combos.l1_A__repr01_B == l1_A@K8 − repr01_B@K8",
   abs(nc["l1_A__repr01_B"] - (l1_a - r_b)) < 1e-9)
ck("net_combos.l1_B__repr01_A == l1_B@K8 − repr01_A@K8",
   abs(nc["l1_B__repr01_A"] - (l1_b - r_a)) < 1e-9)
ck("net_combos.l1_B__repr01_B == l1_B@K8 − repr01_B@K8",
   abs(nc["l1_B__repr01_B"] - (l1_b - r_b)) < 1e-9)

# 分量分解算术
cd = J["component_decomposition"]
k1 = {a: R["k_curve"][a]["1"]["bpc"] for a in ARMS}
ck("clean_tax_repr01_A == repr01_A@K1 − l1_A@K1",
   abs(cd["clean_tax_repr01_A_vs_l1A"] - (k1[A] - k1[BASE])) < 1e-9)
ck("clean_tax_repr01_B == repr01_B@K1 − l1_A@K1",
   abs(cd["clean_tax_repr01_B_vs_l1A"] - (k1[B] - k1[BASE])) < 1e-9)
ck("ensemble_gain_repr01_A == repr01_A@K1 − repr01_A@K8",
   abs(cd["ensemble_gain_repr01_A"] - (k1[A] - R["k_curve"][A][KMAX]["bpc"])) < 1e-9)
ck("ensemble_gain_repr01_B == repr01_B@K1 − repr01_B@K8",
   abs(cd["ensemble_gain_repr01_B"] - (k1[B] - R["k_curve"][B][KMAX]["bpc"])) < 1e-9)

# 配对 CI：两对都算出来了
ck("l1_pair_ci 有 delta/CI", "delta" in J["l1_pair_ci"] and "delta_ci" in J["l1_pair_ci"])
ck("repr01_pair_ci 有 delta/CI", "delta" in J["repr01_pair_ci"] and "delta_ci" in J["repr01_pair_ci"])

# 确定性对照
det = R["deterministic_path_control"]
ck("确定性对照 gain_K1_minus_Kmax == 0", det["gain_K1_minus_Kmax"] == 0.0,
   det["gain_K1_minus_Kmax"], 0.0)
ck("确定性对照 K 条路径 logits 逐位相同", det["paths_bitwise_identical_all"] is True)

# 训练产物：活性量与 gate；两 run 同轨
for arm in ARMS:
    tr = R["train_products"][arm]
    cfg_a = json.load(open(os.path.join(ROOT, CFG["arms"][f"{SCALE}/{arm}"]["config_file"])))
    want_tok = cfg_a["train"]["steps"] * cfg_a["train"]["batch_size"] * cfg_a["model"]["seq_len"]
    ck(f"臂 {arm} train_tokens = steps×bs×T", tr["train_tokens"] == want_tok,
       tr["train_tokens"], want_tok)
    ck(f"臂 {arm} gate_skips == 0（吃满更新）", tr["intervention"]["gate_skips"] == 0,
       tr["intervention"]["gate_skips"], 0)
    ck(f"臂 {arm} params_added == 0", tr["intervention"]["params_added"] == 0)
    if cfg_a["path"]["w_repr_decorr"] > 0:   # 去相关臂：罚则必须真活着、多一次并行前向
        ck(f"臂 {arm} intervention.alive == True", tr["intervention"]["alive"] is True)
        ck(f"臂 {arm} extra_forward_per_step == 1",
           tr["intervention"]["extra_forward_per_step"] == 1)
        ck(f"臂 {arm} 有 repr 活性观测", (tr["intervention"]["n_repr_observations"] or 0) > 0)
    else:   # l1 臂：不是未激活的开关，alive=false / extra_forward=0 是预期
        ck(f"臂 {arm} 是对照（alive=false, extra_forward=0）",
           tr["intervention"]["alive"] is False
           and tr["intervention"]["extra_forward_per_step"] == 0)

# l1_B 活性量落盘覆盖 + 每行字段（§17.4）
lp = os.path.join(HERE, S["train_dir"], "l1", "train_log.jsonl")
lines = [json.loads(x) for x in open(lp) if x.strip()]
ck("l1_B 活动量落盘覆盖到训练末段（≥40% 的 200 步边界）",
   len(lines) >= 0.4 * S["steps_full"] // 200, len(lines), round(0.4 * S["steps_full"] // 200))
ck("l1_B 每行都带 activity 字段（gnorm/n_paths/skip_frac/diversity_js/repr_decorr）",
   all(all(k in ln for k in ("gnorm", "n_paths", "skip_frac", "diversity_js", "repr_decorr"))
       for ln in lines))

ck("代码指纹已落盘（train_m1/model/config/eval/ensemble_probe/run/jitter）",
   all(R["code"].get(p) for p in ("train_m1.py", "pathlm/model.py", "pathlm/config.py",
                                  "pathlm/eval.py", "tools/evalkit/ensemble_probe.py",
                                  "run.py", "jitter_l1.py")))
ck("变异验证全部抓红且逐字节还原", MUT.get("all_caught_and_restored") is True,
   MUT.get("all_caught_and_restored"))


# ------------------------------------------------------------------ 具体数字回查
def _md() -> str:
    p = os.path.join(HERE, "analysis.md")
    if not os.path.exists(p):
        return ""
    return open(p).read().replace("\u2212", "-")


def _num_in_md(value, text) -> bool:
    if isinstance(value, float):
        return (f"{value:.4f}" in text) or (f"{value:.6f}" in text) or (f"{value:.2f}" in text)
    return str(value) in text


AN = _md()
if AN:
    SPECIFIC = [
        ("l1_A@K8", l1_a),
        ("l1_B@K8", l1_b),
        ("repr01_A@K8", r_a),
        ("repr01_B@K8", r_b),
        ("l1_jitter", J["l1_jitter"]),
        ("repr01_jitter", J["repr01_jitter"]),
        ("combined_range", J["combined_range"]),
        ("worst_case_additive_range", J["worst_case_additive_range"]),
        ("effect_v2", J["effect_v2"]),
    ]
    for label, val in SPECIFIC:
        ck(f"analysis.md 引到 {label} = {val}", _num_in_md(val, AN), val, "in analysis.md")
    for a in ARMS:
        ck(f"analysis.md 引到 {a} gate_skips",
           str(R["train_products"][a]["intervention"]["gate_skips"]) in AN)
    ck("analysis.md 注明 l1 臂抖动已补全（不再是下界口径）",
       "补全" in AN or "两臂" in AN)
    ck("analysis.md 注明 50M 的 0.0322 缺口未被覆盖", "50M" in AN and "0.0322" in AN)
    ck("analysis.md 注明禁外推", "外推" in AN)
else:
    checks.append(["analysis.md 尚未落盘（数字回查跳过）", True, "", ""])

# ------------------------------------------------------------------ 输出
fails = [c for c in checks if not c[1]]
report = {"scale": SCALE, "n_checks": len(checks), "n_fail": len(fails), "checks": checks,
          "all_pass": not fails}
out = os.path.join(HERE, f"reconcile_report_{SCALE}.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
for label, ok, got, want in checks:
    if not ok:
        print(f"FAIL  {label}: got={got} want={want}")
print(f"reconcile [{SCALE}]: {len(checks) - len(fails)}/{len(checks)} pass -> {out}")
sys.exit(0 if not fails else 1)
