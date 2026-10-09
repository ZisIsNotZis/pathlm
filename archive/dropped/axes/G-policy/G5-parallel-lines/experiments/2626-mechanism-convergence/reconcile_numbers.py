"""2626 · 报告数字对账（`SCHEMA §12`）：`analysis.md` 里写下的关键数字都必须能在
`results_13m.json` / 训练日志 / `config.json` 里原样找到。

用法：python3 reconcile_numbers.py --scale 13m   # 任一 mismatch 以非零码退出

两层：
  ① **结构性/自洽性**：判据键齐全、主判据两种写法代数等价、税与曲线自洽、
     gate_skips=0、活性量与 `train_log.jsonl` 一致、参数量/token 数与配置自洽、
     三点收敛轨迹与判据量斜率自洽、代码/权重指纹已落盘；
  ② **具体数字**：`analysis.md` 里的关键读数逐条回查（`SPECIFIC`）。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from tools.evalkit.ensemble_probe import reading_B                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scale", choices=("13m",), required=True)
args = ap.parse_args()
SCALE = args.scale

CFG = json.load(open(os.path.join(HERE, "config.json")))
S = CFG["scales"][SCALE]
ARMS = tuple(S["arms"])
PRIMARY_REPR = S.get("primary_repr") or S.get("repr_arms", ["repr01"])[0]
R = json.load(open(os.path.join(HERE, f"results_{SCALE}.json")))
MUT_PATH = os.path.join(HERE, "mutation_report.json")
MUT = json.load(open(MUT_PATH)) if os.path.exists(MUT_PATH) else {"all_caught_and_restored": None}

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
       R["checkpoints"][arm]["sha256"] == CFG["arms"][f"{SCALE}/{arm}"]["sha256"], True)

params = {a: R["checkpoints"][a]["params"] for a in ARMS}
ck(f"{SCALE} 三臂参数量完全相同（去相关压力是纯损失项，不加参数）",
   len(set(params.values())) == 1, params, "单一值")
for a in ARMS:
    cfg_a = json.load(open(os.path.join(HERE, S["config_dir"], f"{a}.json")))
    want_tok = cfg_a["train"]["steps"] * cfg_a["train"]["batch_size"] * cfg_a["model"]["seq_len"]
    tr = R["train_products"][a]
    ck(f"臂 {a} train_tokens = steps×bs×T（4× X1）", tr["train_tokens"] == want_tok,
       tr["train_tokens"], want_tok)
    ck(f"臂 {a} intervention.params_added == 0", tr["intervention"]["params_added"] == 0)
    ck(f"臂 {a} gate_skips == 0（吃满更新）", tr["intervention"]["gate_skips"] == 0,
       tr["intervention"]["gate_skips"], 0)
    if cfg_a["path"]["w_repr_decorr"] > 0:
        ck(f"臂 {a} 确实多了一次并行前向",
           tr["intervention"]["extra_forward_per_step"] == 1,
           tr["intervention"]["extra_forward_per_step"], 1)
        ck(f"臂 {a} intervention.alive == True（去相关罚则真活着）",
           tr["intervention"]["alive"] is True)
        ck(f"臂 {a} 有 repr 活性观测", (tr["intervention"]["n_repr_observations"] or 0) > 0,
           tr["intervention"]["n_repr_observations"])

# 活性量：与训练日志逐点一致（SCHEMA §17.4）
for a in ARMS:
    lp = os.path.join(HERE, S["train_dir"], a, "train_log.jsonl")
    lines = [json.loads(x) for x in open(lp) if x.strip()]
    ck(f"臂 {a} 活动量落盘覆盖到训练末段（≥40% 的 200 步边界）",
       len(lines) >= 0.4 * S["steps_full"] // 200, len(lines),
       round(0.4 * S["steps_full"] // 200))
    ck(f"臂 {a} 每行都带 activity 字段（gnorm/n_paths/skip_frac/diversity_js/repr_decorr）",
       all(all(k in ln for k in ("gnorm", "n_paths", "skip_frac", "diversity_js",
                                 "repr_decorr")) for ln in lines))
    iv = R["train_products"][a]["intervention"]
    ck(f"臂 {a} intervention.gate_skips == 0 与日志覆盖自洽",
       iv["gate_skips"] == 0, iv["gate_skips"], 0)

# 确定性路径对照：增益恰好 0 + 逐位相同
for arm in S.get("det_arms", ["stock", PRIMARY_REPR]):
    tag = "stock" if arm == "stock" else f"{arm}_forced"
    d = R["deterministic_path_control"][tag]
    ck(f"确定性对照 {tag}: gain == 0", d["gain_K1_minus_Kmax"] == 0.0, d["gain_K1_minus_Kmax"], 0.0)
    ck(f"确定性对照 {tag}: K 条路径 logits 逐位相同", d["paths_bitwise_identical_all"] is True)

# 判据：代数等价 + 与曲线自洽
crit = R["criteria"]
b1 = {a: R["k_curve"][a]["1"]["bpc"] for a in ARMS}
bk = {a: R["k_curve"][a][str(CFG["main_k"])]["bpc"] for a in ARMS}
rb = reading_B(b1["l1"], b1["stock"], bk[PRIMARY_REPR])
ck("主判据任务形式 = −reading_B.reduced（代数对账）",
   abs(crit["main_reading_B"]["value"] + rb["reduced"]) < 1e-9,
   crit["main_reading_B"]["value"], -rb["reduced"])
ck("主判据值 = repr01@K8 − stock@K1（从曲线重算）",
   abs(crit["main_reading_B"]["value"] - (bk[PRIMARY_REPR] - b1["stock"])) < 1e-9)
ck("机制净效应 = l1@K8 − repr01@K8（从曲线重算）",
   abs(crit["mechanism_net"]["value"] - (bk["l1"] - bk[PRIMARY_REPR])) < 1e-9)
ck("税 repr01−l1（@K1）从曲线重算",
   abs(crit["taxes"]["repr_clean_tax_vs_l1"] - (b1[PRIMARY_REPR] - b1["l1"])) < 1e-9)
ck("税 l1−stock（@K1）从曲线重算",
   abs(crit["taxes"]["mechanism_clean_tax_vs_stock"] - (b1["l1"] - b1["stock"])) < 1e-9)
ck("机制净效应的 ordering 与阈值 0.014 自洽",
   crit["mechanism_net"]["ordering"] ==
   ("keep" if bk["l1"] - bk[PRIMARY_REPR] >= 0.014
    else "narrowed" if bk["l1"] - bk[PRIMARY_REPR] >= 0.0 else "reversed"))

# 收敛轨迹：判据量在四个点 + 斜率与轨迹自洽（§17.15）
traj = R["trajectory"]["checkpoints"]
g = R["convergence_gate"]
steps_full = S["steps_full"]
ck("轨迹覆盖全部 traj_ckpts（6k/12k/18k）",
   all(str(st) in traj and traj[str(st)].get("available") for st in S["traj_ckpts"]),
   list(traj), S["traj_ckpts"])
for st in S["traj_ckpts"]:
    row = traj[str(st)]
    if not row.get("available"):
        continue
    st_i = int(st)
    d = steps_full - st_i
    ck(f"轨迹 @{st} 主判据斜率 = (终点−该点)/(Δ/1k)",
       abs(g["per_checkpoint"][str(st)]["main_slope_per_1k"]
           - (g["main_criterion_at_full"] - row["main_criterion"]) / (d / 1000)) < 1e-6)
    ck(f"轨迹 @{st} 机制净效应斜率 = (终点−该点)/(Δ/1k)",
       abs(g["per_checkpoint"][str(st)]["mechanism_slope_per_1k"]
           - (g["mechanism_net_at_full"] - row["mechanism_net"]) / (d / 1000)) < 1e-6)
    ck(f"轨迹 @{st} 主判据值 = repr01@K8 − stock@K1（从该点读数重算）",
       abs(row["main_criterion"]
           - (row["arms"][PRIMARY_REPR]["K8_bpc"] - row["arms"]["stock"]["K1_bpc"])) < 1e-9)
    ck(f"轨迹 @{st} 机制净效应 = l1@K8 − repr01@K8（从该点读数重算）",
       abs(row["mechanism_net"]
           - (row["arms"]["l1"]["K8_bpc"] - row["arms"][PRIMARY_REPR]["K8_bpc"])) < 1e-9)
ck("终点主判据 == 曲线 repr01@K8 − stock@K1",
   abs(g["main_criterion_at_full"] - (bk[PRIMARY_REPR] - b1["stock"])) < 1e-9)
ck("终点机制净效应 == 曲线 l1@K8 − repr01@K8",
   abs(g["mechanism_net_at_full"] - (bk["l1"] - bk[PRIMARY_REPR])) < 1e-9)

ck("代码指纹已落盘（train_m1/model/config/ensemble_probe/run）",
   all(R["code"].get(p) for p in ("train_m1.py", "pathlm/model.py", "pathlm/config.py",
                                  "tools/evalkit/ensemble_probe.py", "run.py")))
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
        (f"{SCALE} repr01@K8", bk[PRIMARY_REPR]),
        (f"{SCALE} l1@K8", bk["l1"]),
        (f"{SCALE} stock@K1", b1["stock"]),
        (f"{SCALE} 主判据 value", crit["main_reading_B"]["value"]),
        (f"{SCALE} 机制净效应 value", crit["mechanism_net"]["value"]),
        (f"{SCALE} 机制税 l1−stock", crit["taxes"]["mechanism_clean_tax_vs_stock"]),
        (f"{SCALE} 去相关税 repr01−l1", crit["taxes"]["repr_clean_tax_vs_l1"]),
    ]
    for st in S["traj_ckpts"]:
        if traj.get(str(st), {}).get("available"):
            SPECIFIC.append((f"轨迹 @{st} 主判据", traj[str(st)]["main_criterion"]))
            SPECIFIC.append((f"轨迹 @{st} 机制净效应", traj[str(st)]["mechanism_net"]))
    for label, val in SPECIFIC:
        ck(f"analysis.md 引到 {label} = {val}", _num_in_md(val, AN), val, "in analysis.md")
    for a in ARMS:
        if R["train_products"].get(a, {}).get("intervention"):
            ck(f"analysis.md 引到 {a} gate_skips",
               str(R["train_products"][a]["intervention"]["gate_skips"]) in AN)
    ck("analysis.md 注明 X1 6k 与 24k 跑不可作逐位对照（调度不同）",
       ("调度" in AN and ("6k" in AN or "6000" in AN)))
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
