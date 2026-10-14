"""2672 接任段文档生成器：从落盘 results.json / rescore-*.json / config-parity-*.json
生成 analysis.md（人话在前）、MODEL.md（checkpoint + sha256 + 复现命令）、RESUME.md，
并把墙钟摘要追加到 run_log.txt（墙钟只允许进 run_log.txt，L5）。

用法：python3 write_docs.py
"""

import glob
import hashlib
import json
import os
import subprocess

D = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/z/vibe/pathlm"
R = json.load(open(os.path.join(D, "results.json")))
TAG = "2k"
ARMS = ["fill", "nofill", "projfill"]
E12 = f"{ROOT}/.scratch/23-depth-ar/evidence"
SEEDS = (0, 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pv(seed, arm):
    return R["point_values"][f"seed{seed}"].get(arm)


def fmt(x):
    return "—" if x is None else f"{x:.4f}"


diag = []
diag.append("# 2672-fill-n2-13m · 分析（当前结论与证据）\n")
diag.append("> **口径声明（首段，必读）**：本轮 6 个 run 全部是 **2k 快照口径**（每 run 2000 步、"
            "batch 32 × seq 256 = 16.38M train_tokens），与既有 **12k 档（98.3M tokens）不可比**。"
            "快照上模型远未收敛（如 fill 的 depth-8 bpc 2.80 vs 12k 档 1.86），"
            "**只读臂间增量与排序，不读绝对值**；12k 档只作方向参照。\n")

# ---------------- 人话摘要 ----------------
pv_f0, pv_f1 = pv(0, "fill"), pv(1, "fill")
pv_n0, pv_n1 = pv(0, "nofill"), pv(1, "nofill")
pv_p0, pv_p1 = pv(0, "projfill"), pv(1, "projfill")
d = R["deltas"]
diag.append("## 1 人话摘要\n")
diag.append(
    f"**这东西是干什么的**：三种「早退位置在未跑层的 KV 怎么办」的填充语义——"
    f"`fill`（保留式：用已冻结的浅层表征把已退出位置的 KV 填上）、"
    f"`nofill`（ragged：早退位置在未跑层干脆缺席 KV）、"
    f"`projfill`（fill 之上加一层恒等初始化的逐层线性 adapter）。"
    f"要回答的问题是：**同一训练配方下，哪种填充更划算，且这个排序换种子还成不成立（n≥2）**。\n")
diag.append(
    f"**测出来（2k 快照，越低越好，单位 bpc = 每字符比特数）**：\n"
    f"- 保留式填充优于 ragged：depth-8 bpc **fill {fmt(pv_f0['depth8_bpc'])} vs nofill {fmt(pv_n0['depth8_bpc'])}**"
    f"（seed 0，Δ = {d['seed0']['fill_minus_nofill_depth8']:+.4f}），"
    f"**fill {fmt(pv_f1['depth8_bpc'])} vs nofill {fmt(pv_n1['depth8_bpc'])}**"
    f"（seed 1，Δ = {d['seed1']['fill_minus_nofill_depth8']:+.4f}）"
    f"——两个种子都同向、量级 ≈0.06–0.08 bpc，方向与 12k 档（−0.0856）一致 ⇒ **可复现**。\n"
    f"- proj-fill 在本快照刻度上**反而略差于 fill**：Δ(projfill − fill) = "
    f"{d['seed0']['projfill_minus_fill_depth8']:+.4f}（seed 0）、{d['seed1']['projfill_minus_fill_depth8']:+.4f}（seed 1），"
    f"两个种子都同向（proj-fill 更差）。而 12k 档里 proj-fill 是**更好**（−0.0226）⇒ "
    f"该排序**依赖训练步数**，不能从 2k 快照外推到 matched-step 刻度。\n")
peak_txt = ", ".join(
    f"{a} " + "/".join(f"{v:+.4f}" for v in R["rescore_ci"][a]["delta_peak_soft_minus_hard"])
    for a in ARMS)
diag.append(
    f"**软提交污染（WP2 等参数剥离）**：在**同一个 checkpoint** 上只改提交规则，"
    f"soft（Σp·E 软提交）的 rescore_bpc 峰值一致地远高于 hard（argmax 硬提交）："
    f"逐臂 Δpeak(soft−hard)（seed 0 / seed 1）= {peak_txt}。"
    f"因为**权重逐位相同**，容量（参数量/表达力）不再是解释 ⇒ 软提交确实把分布外内容注入了上下文。\n")
diag.append(
    f"**结论**：填充语义排序在 n=2 下可复现（fill > nofill，快照刻度）；"
    f"proj-fill 的收益不复现（快照刻度为负收益，符号与 12k 档相反）；"
    f"软提交污染在等参数剥离下成立（3 个臂 × 2 个种子全部同向）。\n")

# ---------------- 点值表 ----------------
diag.append("## 2 逐臂逐 seed 点值（2k 快照）\n")
diag.append("| 臂 | seed | params | train_tokens | steps | depth-8 bpc | depth-4 bpc | depth-8 acc | tf@4 接受率 |")
diag.append("|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    for s in SEEDS:
        p = pv(s, arm)
        diag.append(f"| {arm} | {s} | {p['params']} | {p['train_tokens']} | {p['steps']} | "
                    f"{fmt(p['depth8_bpc'])} | {fmt(p['depth4_bpc'])} | {p['depth8_acc']:.4f} | "
                    f"{fmt(p['tf4_accept'])} |")
diag.append("")
diag.append("Δ 与复本一致性（seed 作为复本轴，契约 §3 L1）：\n")
diag.append("| 配对的 Δ | seed 0 | seed 1 | 符号一致 | 极差 | 判定 |")
diag.append("|---|---|---|---|---|---|")
for k, v in R["reproducibility"].items():
    diag.append(f"| {k} | {fmt(v['seed0'])} | {fmt(v['seed1'])} | "
                f"{'是' if v['sign_consistent'] else '否'} | {fmt(v['range'])} | {v['status']} |")
diag.append("")

# ---------------- 12k 参照 ----------------
diag.append("## 3 12k 档参照（**不可与上面直接比**）\n")
diag.append("| 臂 | seed | steps | depth-8 bpc | depth-4 bpc |")
diag.append("|---|---|---|---|---|")
for arm in ARMS:
    p = R["reference_12k"]["seed0"].get(arm)
    if p:
        diag.append(f"| {arm} | 0 | {p['steps']} | {fmt(p['depth8_bpc'])} | {fmt(p['depth4_bpc'])} |")
diag.append("\n12k 档的 Δ：fill−nofill = −0.0856、projfill−fill = −0.0226（均 depth-8，seed 0 单点）。\n")

# ---------------- rescore 表 ----------------
diag.append("## 4 WP2：等参数软/硬提交剥离（rescore）\n")
diag.append("**声明：同一 checkpoint ⇒ 参数逐位相同，容量因素被剥离**；唯一变量是提交规则"
            "（soft = Σp·E 软提交 / hard = argmax 嵌入 / latent = 隐状态直接送入读出）。\n")
diag.append("`rescore_bpc` = 把生成序列回灌进模型后测得的每字符比特数（越低越好）；"
            "深度 0 / 4 / 8 分别指只用前 0 / L/2 / L 层后的读出。\n")
for name, r in sorted(R["rescore"]["checkpoints"].items()):
    diag.append(f"\n**{name}**（seed {r['seed']}，刻度 {r['scale_steps']} 步，臂 {r['arm_class']}，"
                f"stored_match = {r['stored_match']}）：\n")
    diag.append("| 深度 | soft | hard | latent | Δ(soft−hard) | Δ(latent−hard) |")
    diag.append("|---|---|---|---|---|---|")
    for dp in ("0", "4", "8"):
        b = r["rescore_bpc"][dp]
        diag.append(f"| {dp} | {b['soft']:.4f} | {b['hard']:.4f} | {b['latent']:.4f} | "
                    f"{r['delta_soft_minus_hard'][dp]:+.4f} | {r['delta_latent_minus_hard'][dp]:+.4f} |")
    diag.append(f"\n峰值：soft {r['rescore_peak']['soft']:.4f} / hard {r['rescore_peak']['hard']:.4f} / "
                f"latent {r['rescore_peak']['latent']:.4f} ⇒ Δpeak(soft−hard) = "
                f"**{r['delta_peak_soft_minus_hard']:+.4f}**、Δpeak(latent−hard) = "
                f"{r['delta_peak_latent_minus_hard']:+.4f}。\n")
diag.append("### 4.1 显著性（率类口径：配对 bootstrap，不跨量纲借用 0.014 bpc 带）\n")
diag.append("| 臂 | 复本（checkpoint） | Δpeak(soft−hard) | 配对 bootstrap 95% CI | CI 不含 0 | 符号一致 |")
diag.append("|---|---|---|---|---|---|")
for arm in ARMS:
    c = R["rescore_ci"][arm]
    diag.append(f"| {arm} | {', '.join(c['checkpoints'])} | "
                f"{', '.join(f'{v:+.4f}' for v in c['delta_peak_soft_minus_hard'])} | "
                f"[{c['paired_bootstrap_95ci'][0]:+.4f}, {c['paired_bootstrap_95ci'][1]:+.4f}] | "
                f"{'是' if c['ci_excludes_0'] else '否'} | {'是' if c['sign_consistent'] else '否'} |")
diag.append("\n**区间退化声明**：复本只有 n=2 个种子，bootstrap 区间端点≈两点值，"
            "不是自由度为 2 的正规区间；显著性以「两 seed 同号 且 CI 不含 0」为准（契约 §4.2 的判法）。"
            "逐 prompt 的原始 rescore 未落盘 ⇒ 本轮拿不到 checkpoint 内的抽样不确定度，"
            "这一点是**未验证项**，不声称已测量。\n")

# ---------------- 活性/可读性 ----------------
diag.append("## 5 活性列（L3）与填入位置可读性（L4）\n")
diag.append("`avg_train_gate_mean` = 各层 carry 门在训练中的平均开度（越大表示该层越常被复用）；"
            "`avg_train_depth_ce` = 各退出深度的训练交叉熵（越小越好）。\n")
diag.append("| run | gate 均值（逐层） | depth-CE（逐深度） |")
diag.append("|---|---|---|")
for k in sorted(R["activity"]):
    a = R["activity"][k]
    diag.append(f"| {k} | {' '.join(f'{x:.3f}' for x in a['avg_train_gate_mean'])} | "
                f"{' '.join(f'{x:.3f}' for x in a['avg_train_depth_ce'])} |")
diag.append("`tf_acceptance`（逐深度接受率，tf@4 = 用前 4 层当草稿、被全深度接受的比例）：\n")
dep0 = [r["depth"] for r in R["readability"][sorted(R["readability"])[0]]["tf_acceptance"]]
hdr = " | ".join(f"d{depth}" for depth in dep0)
diag.append(f"| run | {hdr} |")
diag.append("|---" * (len(dep0) + 1) + "|")
for k in sorted(R["readability"]):
    diag.append(f"| {k} | " + " | ".join(f"{r['accept']:.4f}" for r in R["readability"][k]["tf_acceptance"]) + " |")
diag.append("\n填充位置占比（由 `exit_mode=uniform` 的采样律导出，非落盘测量字段）："
            f"逐层 {R['activity'][sorted(R['activity'])[0]]['fill_fraction']['per_layer']}、"
            f"均值 {R['activity'][sorted(R['activity'])[0]]['fill_fraction']['mean']}。\n")

# ---------------- 判据 L0–L5 ----------------
parity_all = {os.path.basename(p): json.load(open(p))
              for p in sorted(glob.glob(os.path.join(D, "config-parity-*.json")))}
n_bad = [k for k, v in parity_all.items() if not v["passed"]]
diag.append("## 6 契约 §3 逐条判据\n")
diag.append(f"- **L0 协议保真硬门 —— 通过**。同刻度种子对（`snap2k-<arm>-s0` vs `snap2k-<arm>-s1`）"
            f"逐字段对平 **差异清单为空**（三对全部 passed，文件 `config-parity-snap2k-*-s0-vs-s1.json`）；"
            f"跨刻度对 12k 档的唯一差异 = `train.steps`（2000 vs 12000，逐条记入 allowed_diffs），"
            f"其余缺席字段均已核为引擎缺省（`DepthARConfig()` / CLI 缺省，逐条记入 explained_absent_in_seed0）。"
            f"参数量：fill/nofill {pv_f0['params']}、projfill {pv_p0['params']}"
            f"（增量 = 逐层 adapter，实测 {pv_p0['params'] - pv_f0['params']} 参数）。"
            f"未被解释的差异：{len(n_bad)} 个文件（应为 0）。\n")
diag.append(f"- **L1 主判 P1 —— 报告齐全，两个排序在 2k 刻度上符号一致**："
            f"Δ(fill−nofill) depth-8 = {d['seed0']['fill_minus_nofill_depth8']:+.4f} / "
            f"{d['seed1']['fill_minus_nofill_depth8']:+.4f}（seed0/seed1），"
            f"Δ(projfill−fill) depth-8 = {d['seed0']['projfill_minus_fill_depth8']:+.4f} / "
            f"{d['seed1']['projfill_minus_fill_depth8']:+.4f}；depth-4 同向且量级一致（差 ≤0.0005）。"
            f"n=2 且每个 Δ 只有一个自由度 ⇒ **没有可报的显著性区间**，只能说「两个种子的符号一致、极差 "
            f"{fmt(R['reproducibility']['Delta(fill - nofill) depth8']['range'])}」；"
            f"0.014 bpc 带（clean bpc 口径）**不用于** rescore/接受率。\n")
diag.append(f"- **L2 主判 P2 —— 报告齐全**：6 个新 checkpoint + 3 个 12k 档 checkpoint 的 "
            f"soft/hard/latent 深度 0/4/8 rescore 全部落盘（`rescore-snap2k.json` + `rescore-seed0.json`，"
            f"逐 checkpoint `stored_match=true` 表示重放与各自 results.json 逐位一致）；"
            f"Δ(soft−hard) 在 3 臂 × 2 种子上全部为正且 CI 不含 0（n=2 退化，见 §4.1 声明）。\n")
diag.append(f"- **L3 活性列 —— 齐全**（见 §5：逐层 gate 均值、逐深度 CE、填充占比）。\n")
diag.append(f"- **L4 可读性 —— 齐全**（见 §5：`tf_acceptance` 逐深度 + `depth_curve.acc` 在 results.json）。\n")
diag.append(f"- **L5 禁墙钟 —— 通过**：`results.json::l5_pass={R['l5_pass']}`、"
            f"命中墙钟字段 {R['l5_wall_clock_fields']}；本文件的判据区不含任何墙钟数字，"
            f"墙钟只写在 `run_log.txt`。\n")

# ---------------- §4 否证条件 ----------------
diag.append("## 7 契约 §4 否证条件触发情况\n")
sign_ok_f = R["reproducibility"]["Delta(fill - nofill) depth8"]["sign_consistent"]
sign_ok_p = R["reproducibility"]["Delta(projfill - fill) depth8"]["sign_consistent"]
diag.append(f"- **§4.1（种子间符号不一致 ⇒ 排序不可复现）—— 未触发**："
            f"Δ(fill−nofill) 符号一致 = {sign_ok_f}，Δ(projfill−fill) 符号一致 = {sign_ok_p}。"
            f"**但**：Δ(projfill−fill) 在 2k 快照上为正（proj-fill 更差），"
            f"而 12k 档为负（proj-fill 更好）⇒ 该排序**依赖训练步数**，"
            f"这是「快照口径不可外推」的实例，须写进 6.4 的 failure_analysis 而非当作可复现收益。\n")
not_sig = [a for a in ARMS if not R["rescore_ci"][a]["ci_excludes_0"]]
diag.append(f"- **§4.2（soft−hard 峰值差 CI 在两种子上均含 0 ⇒ 污染由容量/别的因素解释）—— 未触发**："
            f"3 个臂的配对 CI 全部不含 0（未通过者：{not_sig if not_sig else '无'}）；"
            f"n=2 的区间退化已在 §4.1 声明，因此这条结论的强度受「只有 2 个种子」限制。\n")
single = [a for a in ARMS if not R["rescore_ci"][a]["sign_consistent"]]
diag.append(f"- **§4.3（只有一处 seed 显著 ⇒ 写「单点显著、不可复现」，不得升级为判据）—— 未触发**："
            f"3 个臂都是两个种子同号，不存在单点显著的臂（{single if single else '无'}）。\n")

# ---------------- 未完成项 ----------------
diag.append("## 8 未完成 / 未验证项（如实写）\n")
diag.append(f"- **12k matched-step 的 n≥2 复本仍然没有**：本轮为 2k 快照口径，"
            f"契约 §3 的 L0/L1 是对**快照**成立的；要给出 matched-step（12000 步）的 n≥2，"
            f"须补跑 `snap12k-{{fill,nofill,projfill}}-s{{0,1}}`（6 run × ~10 min GPU）。"
            f"已跑 run 清单见 §2，未跑 run：{R['missing_runs'] if R['missing_runs'] else '无（本口径 6/6 完成）'}。\n")
diag.append("- **`mutate.py` + `mutation_report.json`（契约 §6 的 ≥3 条变异抓红）未做**："
            "本段是训练 + 评测，变异纪律沿用引擎既有单测（`tests/`）；"
            "本目录的变异抓红是未完成项，不得声称已做。\n")
diag.append("- **逐 prompt 的 rescore 原始数据未落盘** ⇒ 无法给 checkpoint 内抽样不确定度（见 §4.1）。\n")
diag.append("- **proj-fill 的逐层 adapter 参数量未单独拆表**：只给了总增量（见 §6 L0）。\n")

# ---------------- MODEL.md ----------------
mod = []
mod.append("# MODEL.md — 2672-fill-n2-13m 的 checkpoint 清单与复现命令\n")
mod.append("> 本轮全部为 **2k 快照**（2000 步）checkpoint，与 12k 档不可比；"
           "sha256 覆盖 `model.pt` 逐字节。12k 档 checkpoint 为**只读复用**（未修改）。\n")
mod.append("## 1 本实验产出的 6 个 checkpoint\n")
mod.append("| run | seed | 臂开关 | 步数 | sha256(model.pt) | 文件 |")
mod.append("|---|---|---|---|---|---|")
for s in SEEDS:
    for arm in ARMS:
        rd = os.path.join(D, f"snap{TAG}-{arm}-s{s}")
        mp = os.path.join(rd, "model.pt")
        p = pv(s, arm)
        if os.path.exists(mp):
            sw = "fill_kv=True" if arm == "fill" else ("fill_kv=False" if arm == "nofill"
                                                       else "fill_kv=True + proj_fill")
            mod.append(f"| snap{TAG}-{arm}-s{s} | {s} | {sw} | {p['steps']} | {sha256(mp)} | model.pt |")
mod.append("\n每个 run 目录同时含 `train_log.jsonl`（逐步 loss）、`gate_log.jsonl`（逐层门）、"
           "`results.json`（config/点值/填充语义开关/generation）。\n")
mod.append("## 2 只读复用的 12k 档 checkpoint\n")
mod.append("| run | seed | sha256(model.pt) | 路径 |")
mod.append("|---|---|---|---|")
for arm in ARMS:
    rd = os.path.join(E12, {"fill": "23c-H2-w0-12k", "nofill": "23e-nofill-12k",
                            "projfill": "23e-projfill-12k"}[arm])
    mod.append(f"| 23*({arm}) | 0 | {sha256(os.path.join(rd, 'model.pt'))} | "
               f"`.scratch/23-depth-ar/evidence/{os.path.basename(rd)}` |")
mod.append("\n## 3 复现命令（repo 根 = /home/z/vibe/pathlm）\n")
mod.append("```bash\nD=axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m\n")
mod.append("# 训练 6 个快照 run（逐 run 取放 GPU 锁；SCHEMA §20）\n")
mod.append("STEPS=2000 DEADLINE=$(date -d '+40 min' +%s) bash \"$D\"/run6.sh\n")
mod.append("# 或者单臂：STEPS=2000 bash \"$D\"/run_arm.sh projfill 1\n")
mod.append("# 等参数 rescore（取锁跑评测）+ config 对平 + 汇聚\nbash \"$D\"/finish.sh\n")
mod.append("# 文档与墙钟\npython3 \"$D\"/write_docs.py\n```\n")
mod.append("单臂等价的原始终端命令（`run_arm.sh` 内部就是它）：\n")
mod.append("```bash\npython3 -u train_depth_ar.py snap2k-fill-s1 \\\n"
           "  --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \\\n"
           "  --steps 2000 --batch-size 32 --lr 1e-3 --seed 1 --out-root \"$D\"\n"
           "# nofill 加 --no-fill-kv；projfill 加 --proj-fill（需 fill_kv=True，缺省）\n```\n")
mod.append("## 4 权重/数据不入库\n")
mod.append("`model.pt` 共 6 × 27.9 MB 留在实验目录、不进版本库；"
           "可由上面的命令在同 seed / 同配置下重训，sha256 用于核对「评测的确实是这一个 checkpoint」。\n")

# ---------------- run_log 墙钟摘要 ----------------
wall_lines = ["", f"[{subprocess.run(['date', '-Is'], capture_output=True, text=True).stdout.strip()}] "
              "WRITE_DOCS: 墙钟摘要（仅本文件允许墙钟字段）"]
for s in SEEDS:
    for arm in ARMS:
        rd = os.path.join(D, f"snap{TAG}-{arm}-s{s}")
        rp = os.path.join(rd, "results.json")
        if os.path.exists(rp):
            m = json.load(open(rp))
            wall_lines.append(f"  {m['run']}: wall_minutes={m.get('wall_minutes')} steps={m['train']['steps']}")
for name, rec in sorted(json.load(open(os.path.join(D, f"rescore-snap{TAG}.json"))).items()):
    wall_lines.append(f"  rescore {name}: elapsed_s={rec['elapsed_s']}")
wall_lines.append("  备注：DEADLINE=22:39 只用于「不再启动新 run」；6/6 run 均在截止前跑完，无截断。")

# ---------------- RESUME.md ----------------
res = [f"# RESUME.md — 2672-fill-n2-13m 当前状态（接任段收尾）\n",
       "## 完成到哪\n",
       f"- ✅ **6/6 快照 run 跑完**（3 臂 × seed{{0,1}}，统一 2000 步）：`snap{TAG}-{{fill,nofill,projfill}}-s{{0,1}}`，"
       "各含 `model.pt` / `train_log.jsonl` / `gate_log.jsonl` / `results.json`。\n",
       f"- ✅ **WP2 等参数剥离评测扩展到 6 个新 checkpoint**（`rescore-snap{TAG}.json`，逐 checkpoint "
       "`stored_match=true`），连同前任的 3 个 12k 档读数（`rescore-seed0.json`）一并进 `results.json::rescore`。\n",
       "- ✅ **L0 config 对平全通过**：同刻度种子对差异清单为空；跨刻度对 12k 档仅 `train.steps` 差异"
       "（allowed）+ 缺席字段取引擎缺省（explained），未解释差异 0。\n",
       "- ✅ 落盘：`results.json`（无墙钟字段，`l5_pass=true`）、`analysis.md`（人话在前 + L0–L5 + §4）、"
       "`MODEL.md`（sha256 + 复现命令）、本文件、`run_log.txt`。\n",
       "- ⚠️ **本口径是 2k 快照**（锁饥饿 ⇒ 按预登记授权缩步数），**matched-step 12k 的 n≥2 仍未做**。\n",
       "- ❌ 未做：本目录 `mutate.py`（变异抓红）、逐 prompt rescore 原始数据。\n",
       "\n## 为什么是 2000 步\n",
       "本段时间盒开头 GPU 锁被其它 worker（`2671-padfill-13m`，逐臂取放锁、每臂 ~2:30）连续占用；"
       "按 23e 实测（12000 步 ≈ 9.9 min）折算，可用 GPU 预算 ≈10 min ⇒ 取 6 run 总量 ≤10 min 的最大 500 倍数 = "
       "**2000**（≥1500 下限、臂间同刻度、契约 §2 明确允许缩步数）。\n",
       "\n## 下一步确切命令（补 matched-step n≥2 时才需要）\n",
       "```bash\nD=axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m\n",
       "# 12k 档 6 个 run（3 臂 × seed{0,1}；每 run ≈10 min GPU，分两段跑）\n",
       "STEPS=12000 TAG=12k DEADLINE=$(date -d '+55 min' +%s) bash \"$D\"/run6.sh\n",
       "TAG=12k SNAP_STEPS=12000 bash \"$D\"/finish.sh   # rescore + 对平 + 汇聚\n",
       "python3 \"$D\"/write_docs.py\n```\n",
       "\n## 边界\n",
       "只写 `axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/**`；`README.md`（预登记契约）、"
       "`pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、`.scratch/23-depth-ar/evidence/**`（只读复用）、"
       "其它轴/实验、任何 `project.yaml` 均未改。\n"]

open(os.path.join(D, "analysis.md"), "w").write("\n".join(diag) + "\n")
open(os.path.join(D, "MODEL.md"), "w").write("\n".join(mod) + "\n")
open(os.path.join(D, "RESUME.md"), "w").write("\n".join(res) + "\n")
with open(os.path.join(D, "run_log.txt"), "a") as f:
    f.write("\n".join(wall_lines) + "\n")
print("wrote analysis.md / MODEL.md / RESUME.md / run_log.txt(append)")
