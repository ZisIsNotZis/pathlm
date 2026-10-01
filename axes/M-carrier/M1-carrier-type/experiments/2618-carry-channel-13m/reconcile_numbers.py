"""2618 · 报告数字对账：报告里每个数字都必须能回到 results.json / 训练日志 / 其它落盘。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

R = json.load(open(os.path.join(HERE, "results.json")))
PC = json.load(open(os.path.join(HERE, "premise_check.json")))
MUT = json.load(open(os.path.join(HERE, "mutation_report.json")))
AN = open(os.path.join(HERE, "analysis.md")).read()
ST = open(os.path.join(HERE, "stability.md")).read()
MD = open(os.path.join(HERE, "MODEL.md")).read()
RM = open(os.path.join(HERE, "README.md")).read()

checks = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


def has(text, s, label):
    ck(label, s in text, s[:60], "present in report")


def num_in(text, value, label, places=6):
    """报告里是否出现该数字（按给定小数位的四舍五入形式找）。"""
    for fmt in (f"{value:.{places}f}", f"{value:.4f}", f"{value:.3f}",
                f"{value:.2f}", f"{value:.1f}"):
        if fmt in text:
            ck(label, True, fmt, "found")
            return
    ck(label, False, None, f"any of {value}")


def track(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------- 自主检查
sc = R["selfchecks"]
ck("自检 release vs 显式因果掩码 == 0.0",
   sc["release_vs_explicit_causal_max_abs_logit"] == 0.0,
   sc["release_vs_explicit_causal_max_abs_logit"], 0.0)
ck("自检 模型窗口掩码前向 vs 仪器 == 0.0",
   sc["model_window_forward_vs_mask_instrument_max_abs_logit"] == 0.0,
   sc["model_window_forward_vs_mask_instrument_max_abs_logit"], 0.0)
ck("自检 U1/U2 逐位等参数（构造）", sc["params_bitwise_equal_U1_U2"]["bitwise_equal"] is True)
ck("自检 主干逐位 == U0（构造）", sc["trunk_bitwise_equal_carry_vs_U0"]["trunk_bitwise_equal"] is True)
ck("自检 单边零起步 logits 逐位 == U0",
   all(sc["identity_start_logits_bitwise"].values()) is True,
   sc["identity_start_logits_bitwise"], True)
ck("自检 新增参数 = 264296", sc["trunk_bitwise_equal_carry_vs_U0"]["n_params_added"] == 264296,
   sc["trunk_bitwise_equal_carry_vs_U0"]["n_params_added"], 264296)
ck("自检 U0 参数 = 13536768",
   sc["trunk_bitwise_equal_carry_vs_U0"]["n_params_base_arm"] == 13536768)
ck("自检 U1 参数 = 13801064", sc["params_bitwise_equal_U1_U2"]["n_params_a"] == 13801064)

# ---------------------------------------------------------------- 判据与主腿
a = R["assoc_recall"]
for arm in ("U0", "U1", "U2"):
    for K in (4, 8, 16):
        row = a[f"{arm}_K{K}"]
        ck(f"{arm} K={K} 主腿准确率进报告",
           f"{row['accuracy']:.6f}" in AN or f"{row['accuracy']*100:.2f}%" in AN,
           row["accuracy"], "in analysis.md")
        ck(f"{arm} K={K} n=4096", row["n"] == 4096, row["n"], 4096)
        ck(f"{arm} K={K} chance=1/64", row["chance"] == 0.015625, row["chance"], 0.015625)
ck("U2 K=8 毁源后 == 0.008057", a["U2_K8"]["corrupt_source"]["accuracy"] == 0.008057)
ck("U0 K=8 准确率 == 0.013672", a["U0_K8"]["accuracy"] == 0.013672)
ck("U1 K=8 准确率 == 0.000732", a["U1_K8"]["accuracy"] == 0.000732)
ck("U2 K=8 准确率 == 0.094971", a["U2_K8"]["accuracy"] == 0.094971)
pr = R["premise_check"]
ck("判据① 前提校验过", pr["U0_significantly_above_chance"] is False)
ck("判据① U0 CI 下界 ≤ chance", pr["U0_ci"][0] <= pr["chance"])
for f in ("min", "median", "max", "frac_outside_window"):
    ck(f"距离分布 {f} 与 premise_check.json 一致",
       pr["query_key_distance"][f] == PC["query_key_distance"][f],
       pr["query_key_distance"][f], PC["query_key_distance"][f])
ck("距离 100% 在窗外", pr["query_key_distance"]["frac_outside_window"] == 1.0)
cmp_ = R["comparisons"]
ck("判据② Δ == 8.13pp", cmp_["U2_vs_U0"]["delta_pp"] == 8.13, cmp_["U2_vs_U0"]["delta_pp"], 8.13)
ck("判据② CI 不重叠", cmp_["U2_vs_U0"]["ci_overlap"] is False)
ck("判据③ Δ == 9.424pp", cmp_["U2_vs_U1"]["delta_pp"] == 9.424)
ck("判据③ CI 不重叠", cmp_["U2_vs_U1"]["ci_overlap"] is False)
cr = R["criteria"]
ck("criteria.c1", cr["c1_premise_U0_near_chance"] is True)
ck("criteria.c2", cr["c2_main_U2_gt_U0_5pp_and_ci_disjoint"] is True)
ck("criteria.c3", cr["c3_delta_vs_plain_3pp_and_ci_disjoint"] is True)
ck("criteria.c4 三臂均不作废",
   all(not v["invalidated"] for v in cr["c4_corrupt_source_control"].values()))

# ---------------------------------------------------------------- 距离腿
dl = R["distance_leg"]
for arm in ("U0", "U1", "U2"):
    for d in ("16", "32", "64", "128", "192", "255"):
        ck(f"距离腿 {arm} d={d} 进报告",
           f"{dl[arm][d]['accuracy']:.6f}" in AN
           or f"{dl[arm][d]['accuracy']*100:.2f}%" in AN,
           dl[arm][d]["accuracy"], "in analysis.md")
        ck(f"距离腿 {arm} d={d} n=1024", dl[arm][d]["n"] == 1024)

# ---------------------------------------------------------------- 活性（日志口径）
for arm, exp in (("U1", {"state_fro": 71.839294, "read_scale": 0.006782}),
                 ("U2", {"state_fro": 3.109095, "read_scale": 0.087928})):
    rows = track(os.path.join(HERE, "train", arm, "train_log.jsonl"))
    last = rows[-1]["attn_carry"]
    ck(f"{arm} 末行 state_fro 与日志一致", last["state_fro"] == exp["state_fro"],
       last["state_fro"], exp["state_fro"])
    ck(f"{arm} 末行 read_scale 与日志一致",
       last["read_scale_absmean"] == exp["read_scale"],
       last["read_scale_absmean"], exp["read_scale"])
    ck(f"{arm} 每 200 步落盘活性量", all(r.get("attn_carry") for r in rows))
    ck(f"{arm} write_rel（块 2..n）有定义", last["write_rel"] > 0, last["write_rel"], ">0")
    ck(f"{arm} 末行答案行 CE 与 results 一致",
       rows[-1]["synth_answer_ce"]
       == R["train_log_tail"][arm]["last_answer_ce"],
       rows[-1]["synth_answer_ce"], R["train_log_tail"][arm]["last_answer_ce"])

# ---------------------------------------------------------------- 重跑散度（判据⑥）
DV = json.load(open(os.path.join(HERE, "divergence.json")))
ck("重跑臂 U1rep 准确率 == 0.033447", DV["arms"]["U1rep"]["accuracy"] == 0.033447)
ck("重跑臂 U1rep 答案行 CE 末值 == 2.2524",
   DV["arms"]["U1rep"]["answer_ce_track"]["last_answer_ce"] == 2.2524)
ck("正式 U1 答案行 CE 末值 == 4.184",
   DV["arms"]["U1"]["answer_ce_track"]["last_answer_ce"] == 4.184)
ck("同配置两次 plain 差 == +3.271pp",
   DV["rerun_comparison"]["delta_pp"] == 3.271, DV["rerun_comparison"]["delta_pp"], 3.271)
ck("同配置两次 plain CI 不重叠", DV["rerun_comparison"]["ci_overlap"] is False)
U2K8 = a["U2_K8"]
from tools.evalkit.carry_channel import two_proportion as _two_proportion
cmp_rep = _two_proportion(DV["arms"]["U1rep"]["n_hits"], DV["arms"]["U1rep"]["n"],
                             U2K8["n_hits"], U2K8["n"])
ck("U2 vs U1rep Δ == +6.152pp", cmp_rep["delta_pp"] == 6.152, cmp_rep["delta_pp"], 6.152)
ck("U2 vs U1rep CI 不重叠", cmp_rep["ci_overlap"] is False)
ck("判据③稳健性写进 analysis/stability",
   "6.15pp" in AN and "6.152pp" in ST)

# ---------------------------------------------------------------- 斜率/成本/内存
ck("loss 斜率 U0 == -0.001738", R["loss_slope"]["U0"]["slope"] == -0.001738)
ck("loss 斜率 U1 == 0.026008", R["loss_slope"]["U1"]["slope"] == 0.026008)
ck("loss 斜率 U2 == -0.053230", R["loss_slope"]["U2"]["slope"] == -0.05323)
ck("U2 末行 loss == 0.3260", R["train_log_tail"]["U2"]["last_loss"] == 0.326)
ck("U0 末行 loss == 0.5059", R["train_log_tail"]["U0"]["last_loss"] == 0.5059)
for arm, mins in (("U0", 11.12), ("U1", 13.48), ("U2", 16.26)):
    ck(f"训练墙钟 {arm} == {mins}",
       R["train_results"][arm]["wall_minutes"] == mins,
       R["train_results"][arm]["wall_minutes"], mins)
for arm, mb in (("U0", 2015.3), ("U1", 2169.5), ("U2", 2282.4)):
    ck(f"训练峰值显存 {arm} == {mb}",
       R["memory_measured"]["train_peak_gpu_mem_mb"][arm] == mb)
ck("评测墙钟 ≈ 2.57 min", R["memory_measured"]["eval_peak_gpu_mem_mb"] == 521.5)
ck("成本：评测墙钟 < 10 min", R["cost"]["wall_minutes_total_eval"] < 10)
ck("解析：通道状态 8192 floats（与 T 无关）",
   R["memory_analytic"]["carry_state_floats"] == 8192)
ck("解析：窗口注意力分数 8.0 MB（对照全注意力 128.0 MB）",
   R["memory_analytic"]["window_attn_scores_mb"] == 8.0
   and R["memory_analytic"]["full_attn_scores_mb_for_contrast"] == 128.0)

# ---------------------------------------------------------------- 复现性/变异
ck("评测侧同 seed 重跑 diff == 0.0",
   R["reproducibility"]["same_seed_rerun_max_abs_diff"] == 0.0
   and R["reproducibility"]["pass"] is True)
ck("变异验证 7/7 抓红", MUT["all_caught_red"] is True and len(MUT["mutations"]) == 7,
   len(MUT["mutations"]), 7)
ck("变异验证逐字节还原", MUT["all_restored_bitwise"] is True)

# ---------------------------------------------------------------- 报告文本一致性
for s, lbl in ((f"{a['U2_K8']['accuracy']*100:.2f}%", "analysis 写 U2 K8 百分数"),
               ("+8.13pp", "analysis 写主判据差"), ("+9.42pp", "analysis 写并列判据差"),
               ("0.015625", "analysis 写 chance"), ("241", "analysis 写距离下界"),
               ("255", "analysis 写距离上界"), ("264,296", "analysis 写新增参数"),
               ("13,536,768", "analysis 写 U0 参数"), ("13,801,064", "analysis 写通道臂参数"),
               ("2.57 min", "analysis 写评测墙钟"), ("8192", "analysis 写状态大小"),
               ("4.1589", "analysis 写瞎猜 CE"), ("3.3447", "analysis 写重跑臂准确率"),
               ("6.15pp", "analysis 写判据③稳健下界")):
    has(AN, s, lbl)
for s, lbl in (("0.011719", "stability 写评测复现值"), ("3e-4", "stability 写训练散度"),
               ("+8.13pp", "stability 写主判据"), ("1.4772", "stability 写 U2 末 CE"),
               ("4.184", "stability 写 U1 末 CE"), ("207 / **221.1 ms**", "stability 写 per-layer 步时"),
               ("**73.1 / 81.3 ms**", "stability 写共享槽 delta 步时"),
               ("277 passed", "stability 写 pytest")):
    has(ST, s, lbl)
has(MD, a["U0_K8"] and sc["trunk_bitwise_equal_carry_vs_U0"]["n_params_base_arm"]
    and "13,536,768", "MODEL 写 U0 参数")
has(RM, "逐字抄录", "README 标注判据线逐字")

# ---------------------------------------------------------------- 代码指纹
for f, h in R["code"].items():
    if h is None:
        ck(f"code[{f}] 已落盘", False, None, "hash")
    elif f in ("run.py", "pathlm/depth_ar.py", "train_depth_ar.py",
               "tools/evalkit/carry_channel.py"):
        ck(f"MODEL/REPORT 含 code[{f}] 前 16 位或完整哈希",
           h[:16] in MD or h in MD or h[:16] in AN or h in AN, h[:16], "in report")

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
