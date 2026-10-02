"""2619 · 报告数字对账：报告里每个数字都必须能回到 results.json / 训练日志 / 其它落盘。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出

两层检查：
  ① **结构性/自洽性**（不依赖本次读数）：自检值、臂齐全、判据键齐全、
     `analysis.md` 与 `results.json` 逐条一致、训练日志与 trainer 汇总一致、
     代码指纹已落盘；
  ② **具体数字**：`analysis.md` 里写下的每个关键读数必须能在 `results.json` /
     训练日志里原样找到（标为「SPECIFIC」的段落）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

R = json.load(open(os.path.join(HERE, "results.json")))
CFG = json.load(open(os.path.join(HERE, "config.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
AN = open(os.path.join(HERE, "analysis.md")).read()
MD = open(os.path.join(HERE, "MODEL.md")).read()
RM = open(os.path.join(HERE, "README.md")).read()
ARMS = ("A", "B", "B_rep")

checks = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


def track(path):
    return [json.loads(ln) for ln in open(path) if ln.strip()]


# ---------------------------------------------------------------- 结构/自洽
sc = R["selfchecks"]
ck("自检 release vs 显式因果掩码 == 0.0",
   sc["release_vs_explicit_causal_max_abs_logit"] == 0.0,
   sc["release_vs_explicit_causal_max_abs_logit"], 0.0)
ck("自检 模型窗口掩码前向 vs 仪器 == 0.0",
   sc["model_window_forward_vs_mask_instrument_max_abs_logit"] == 0.0,
   sc["model_window_forward_vs_mask_instrument_max_abs_logit"], 0.0)
eq = sc["lambda0_implementation_equivalence"]
ck("自检 λ=0 实现级等价：主干参数逐位", eq["trunk_bitwise_equal"] is True)
ck("自检 λ=0 实现级等价：logits 逐位", eq["logits_bitwise_equal"] is True)
ck("自检 新增张量只有解码头", eq["added_tensors"] == ["recon_head.weight"],
   eq["added_tensors"], ["recon_head.weight"])
ck("自检 参数差 = V·d = 52736 (=256*206)",
   eq["n_params_head"] - eq["n_params_nohead"] == 256 * 206,
   eq["n_params_head"] - eq["n_params_nohead"], 256 * 206)
tp = sc["recon_target_premise"]
ck("目标前提：行 i 的目标被窗口遮住，零违例",
   tp["target_hidden_at_row_i"]["violations"] == 0,
   tp["target_hidden_at_row_i"]["violations"], 0)
ck("目标前提：行 i−1 可见（全部）",
   tp["target_visible_at_row_i_minus_1"]["ok"] == tp["n_rows"],
   tp["target_visible_at_row_i_minus_1"]["ok"], tp["n_rows"])
ck("目标前提：target == tokens[:, :T−W]",
   tp["target_equals_tokens_i_minus_W"] is True)
ck("目标前提：局部先验抗性通过", tp["local_prior"]["passed"] is True,
   tp["local_prior"]["local_prior_accuracy"], f"< {tp['local_prior']['ceiling']}")

# 臂 / 判据齐全
for a in ARMS:
    ck(f"臂 {a} 在 assoc_recall（K=4/8/16）",
       all(f"{a}_K{k}" in R["assoc_recall"] for k in (4, 8, 16)))
    ck(f"臂 {a} checkpoint sha256 已核", R["checkpoints"][a]["sha256_verified"] is True)
    ck(f"臂 {a} config sha256 与 CONFIG 一致",
       R["checkpoints"][a]["sha256"] == CFG["arms"][a]["sha256"], True)
for key in ("c0_premise_A_near_chance", "c1_main_B_ge_A_3pp_and_ci_disjoint",
            "c2_corrupt_source_control", "c3_rerun_rep_ge_A_3pp_and_ci_disjoint",
            "c3_B_vs_Brep_ci_disjoint", "c3_interpretation"):
    ck(f"判据键 {key} 存在", key in R["criteria"])
ck("c3_interpretation 非空", bool(R["criteria"]["c3_interpretation"]))

# 训练日志 ↔ trainer 汇总一致（同 seed 重跑的散度证据必须来自日志）
for a in ARMS:
    lines = track(os.path.join(HERE, "train", a, "train_log.jsonl"))
    res = json.load(open(os.path.join(HERE, "train", a, "results.json")))
    ck(f"{a}: 训练日志末步 == 12000-1", lines[-1]["step"] == 11999, lines[-1]["step"], 11999)
    # 末行自洽：λ=0 时 loss **逐位等于** trunk CE（重建项根本没接上）；λ>0 时
    # loss == trunk + λ·recon.ce_all（对账「干预真的接在损失上」）
    if a == "A":
        ck("A: 末行 loss == 末行 trunk CE（λ=0 无附加项）",
           lines[-1]["loss"] == lines[-1]["depth_ce"][-1],
           lines[-1]["loss"], lines[-1]["depth_ce"][-1])
    else:
        want = lines[-1]["depth_ce"][-1] + lines[-1]["recon"]["ce_all"]
        ck(f"{a}: 末行 loss == trunk + recon.ce_all（±0.02）",
           abs(lines[-1]["loss"] - want) < 0.02, lines[-1]["loss"], round(want, 4))
    # 日志每 200 步一行 ⇒ 第 0 行（loss 4.7499）会把「抽样均值」抬到 0.576，
    # 与 trainer 的**全步**均值（0.5106）不可比；只用 step ≥ 200 的行比。
    tail_rows = [ln for ln in lines if ln["step"] >= 200]
    ck(f"{a}: 训练汇总与日志一致（trunk CE 均值，±0.01）",
       abs(sum(ln["depth_ce"][-1] for ln in tail_rows) / len(tail_rows)
           - res["avg_train_depth_ce"][-1]) < 0.01,
       round(sum(ln["depth_ce"][-1] for ln in tail_rows) / len(tail_rows), 4),
       res["avg_train_depth_ce"][-1])
    ck(f"{a}: recon 活性块 enabled", res["intervention"]["enabled"] is True)
    if a == "A":
        ck("A(λ=0)：每行 recon 均为 None（未建图）",
           all(ln.get("recon") is None for ln in lines))
        ck("A(λ=0)：intervention.alive == False", res["intervention"]["alive"] is False)
        ck("A(λ=0)：head grad 未流（grad_flowed is None）",
           res["intervention"]["grad_flowed"] is None)
    else:
        ck(f"{a}(λ>0)：recon 每行都有读数",
           all(ln.get("recon") is not None for ln in lines))
        ck(f"{a}(λ>0)：intervention.alive == True", res["intervention"]["alive"] is True)
        ck(f"{a}(λ>0)：head 梯度非零", res["intervention"]["head_gnorm_mean"] > 0,
           res["intervention"]["head_gnorm_mean"], "> 0")

# 变异验证
ck("变异验证：5 处全部抓红", MUT["all_caught_red"] is True)
ck("变异验证：5 处全部逐字节还原", MUT["all_restored_bitwise"] is True)
ck("变异验证：条数 = 5", len(MUT["mutations"]) == 5, len(MUT["mutations"]), 5)

# 代码指纹
for f, h in R["code"].items():
    ck(f"code[{f}] 已落盘", h is not None, h, "hash")

# ---------------------------------------------------------------- SPECIFIC 数字
def num_in(text, value, label, places=6):
    for fmt in (f"{value:.{places}f}", f"{value:.4f}", f"{value:.3f}",
                f"{value:.2f}", f"{value:.1f}"):
        # 报告里负数常用 U+2212（−），对账两种写法都要接受
        if fmt in text or fmt.replace("-", chr(0x2212)) in text:
            ck(label, True, fmt, "found")
            return
    ck(label, False, None, f"any of {value}")


a = R["assoc_recall"]
KM = CFG["K_main"]
for arm in ARMS:
    row = a[f"{arm}_K{KM}"]
    num_in(AN, row["accuracy"], f"analysis 写 {arm} K={KM} 准确率")
    ck(f"{arm} K={KM} 主腿 n=4096", row["n"] == 4096, row["n"], 4096)
    ck(f"{arm} K={KM} chance=1/64", row["chance"] == 0.015625, row["chance"], 0.015625)
    num_in(AN, row["corrupt_source"]["accuracy"], f"analysis 写 {arm} 毁源准确率")
    ck(f"{arm} K={KM} 窗口外距离占比 = 1.0",
       row["query_key_distance"]["frac_outside_window"] == 1.0,
       row["query_key_distance"]["frac_outside_window"], 1.0)
for tag, cmp_key in (("B−A", "B_vs_A"), ("B_rep−A", "Brep_vs_A"),
                     ("B−B_rep", "B_vs_Brep")):
    dp = R["comparisons"][cmp_key]["delta_pp"]
    forms = (f"{dp}pp", f"{dp:.2f}pp", f"{dp:.1f}pp",
             f"{str(dp).replace('-', chr(0x2212))}pp")
    ck(f"analysis 写 {tag} 差值 pp", any(f in AN for f in forms), dp,
       f"one of {forms}")
for k in (4, 8, 16):
    num_in(AN, a[f"B_K{k}"]["accuracy"], f"analysis 写 B K={k} 准确率")
ras = R["recon_activity_on_trained_heads"]
for arm in ARMS:
    num_in(AN, ras[arm]["ce_symbol"], f"analysis 写 {arm} 重建符号 CE")
    num_in(AN, ras[arm]["acc_symbol"], f"analysis 写 {arm} 重建符号命中率")
num_in(AN, R["assoc_recall"][f"A_K{CFG['K_main']}"]["corrupt_source"]["accuracy"],
       "analysis 写 A 毁源准确率")
# 补充专报（slope_diagnosis.py）与 2618 的外部参照值
SR = json.load(open(os.path.join(HERE, "slope_report.json")))
for arm in ARMS:
    num_in(AN, SR["trunk_ce_slope"][arm]["slope"], f"analysis 写 {arm} 主干 CE 斜率")
REF = {"2618 U2 K=8": 0.094971, "2618 U0 K=8": 0.013672, "2618 U1 K=8": 0.000732}
R18 = json.load(open(os.path.join(
    ROOT, "axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m",
    "results.json")))
for name, v in REF.items():
    armK = name.split()[1] + "_" + name.split()[2].replace("=", "")
    got = R18["assoc_recall"][armK]["accuracy"]
    ck(f"外部参照 {name} 口径一致", abs(got - v) < 1e-9, got, v)
    num_in(AN, got, f"analysis 写 {name}")
for arm in ARMS:
    t = R["train_results"][arm]
    num_in(AN, t["wall_minutes"], f"analysis 写 {arm} 墙钟")
    num_in(AN, t["peak_gpu_mem_mb"], f"analysis 写 {arm} 峰值显存")
num_in(AN, R["cost"]["wall_minutes_total_eval"], "analysis 写评测墙钟")
has_mut = f"{len(MUT['mutations'])}" in AN
ck("analysis 写变异验证条数", has_mut, len(MUT["mutations"]), "in analysis.md")
ck("MODEL.md 写参数 13,589,504", "13,589,504" in MD or "13589504" in MD)
ck("MODEL.md 写解码头参数 52,736", "52,736" in MD or "52736" in MD)
ck("README 标注判据线逐字", "逐字抄录" in RM)
ck("analysis 标注口径（快照/matched-step）",
   any(s in AN for s in ("matched-step", "快照", "snapshot")), True)

bad = [c for c in checks if not c[1]]
print(f"{len(checks) - len(bad)}/{len(checks)} 通过")
for c in bad:
    print("  MISMATCH:", c)
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump({"n_checks": len(checks), "n_failed": len(bad),
               "checks": [{"label": c[0], "ok": c[1], "got": c[2], "want": c[3]}
                          for c in checks]}, f, indent=2, ensure_ascii=False)
if bad:
    raise SystemExit(1)
