#!/usr/bin/env python3
"""6.3.2 的终局事实文档更新（Phase C，WP1 = 2671）——scripted，带命中计数断言。"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, "axes/6-predict/6.3.2-pad-fill/project.yaml")
TS = "2026-10-14T23:07:00+08:00"


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


CONCL = (
    "本地判决（WP1 = `experiments/2671-padfill-13m`，S = 1500 的**快照口径**、只读臂间增量与排序；"
    "同 harness、同 token 预算、唯一变量 = 注意力可见性 + 损失位置集；n≥2 = 模型种子 {0,1}）："
    "(a) **一次性填充的质量代价巨大且可分**——在同一批待填位置 `M_r` 上（bpc 越低越好，Δ = 填空 − 全因果 AR 在同一 "
    "`M_r` 上的 teacher-forced bpc），`fill_bidir_r10` **+2.1453** [+2.0400, +2.2500]、`fill_bidir_r25` **+1.0726** "
    "[+1.0034, +1.1455]、`fill_bidir_r50` **+2.3735** [+2.2821, +2.4720]、`fill_causal_r25` **+1.2362** [+1.1822, +1.2903] "
    "——block 聚类配对 bootstrap 95% CI **全部不含 0 且全部远越 0.014 bpc 参考带**；参照 = AR 全位置 teacher-forced bpc "
    "2.2704（快照刻度；收敛档为 ≈1.48）。(b) **r 形状不单调**：Δ 随 r ∈ {10%,25%,50%} 走成 +2.1453 → +1.0726 → +2.3735，"
    "**r=0.25 的代价最小** ⇒ 预登记否证条件里「随 PAD 比例单调恶化」这一子条件**未满足**（原因未查清，已入 gaps），"
    "故该否证条件整体**未触发**，判词按各臂分判而非一刀切。(c) **可见性有独立增益**：`fill_causal_r25` − "
    "`fill_bidir_r25` = **+0.1635** [+0.1096, +0.2140]（CI 不含 0）⇒ 同层互相可见（填空视图）确实换来质量，但**两种都远差于 AR**。"
    "(d) **硬门通过**：把 `M` 位置真 token 扰动后该位置 logits 逐位不变（`max|Δ| = 0.0`），且 gate probe 两条均变红"
    "（仪器可失败）⇒ 「不受未来泄漏」这条不变量成立。(e) 代价面（如实报、不作判据）：tokens-per-forward = |M| ≈ r·T、"
    "每 block 顺序前向次数 = 1（AR 为 T−1）、每生成 token 等效 FLOPs ≈ L/r（AR = L）。(f) 未测：PAD 布局的随机 vs 后段连续、"
    "与 6.3.1 的 K≥2 连续过渡、更大规模；`L5` 的**确定性未取得独立验证**（本 revision 未拿到 GPU 锁重跑，`results.json::L5` "
    "的两项为 None，如实登记为「未通过 / 未验证」，不得读成通过）。"
)

FAIL = (
    "结论含「手段不成立」成分 ⇒ 归因：**设计/结构为主，不是 baseline 错**。设计 = 「一次前向并行填出全部 PAD」这一出口"
    "把被填位置之间的条件依赖整体删除（被填位置互相不可见），在本任务上代价 +1.07…+2.37 bpc（相对 AR 在同一批位置上）；"
    "结构 = 双向可见虽有独立正增益（bidir 优于 causal 0.1635 bpc），但**不足以补回删掉的条件依赖**（两种仍远越带）。"
    "**保留的正面结论**：可见性的方向性对（双向优于纯因果），且「不受未来泄漏」的仪器可失败、成立。"
    "未查清项 = 为什么 r=0.25 的代价小于 r=0.10 与 r=0.50（非单调），登记为下一步最小实验，不作为归因依据。"
)

CR = (
    'conclusion_review: "自评：本项从「零读数」推到「有判决」——一次性填充在本快照刻度上质量代价 +1.07…+2.37 bpc（CI 不含 0、'
    "远越 0.014 带，AR 同位置参照 2.2704），可见性有独立正增益但不足以补回，`不受未来泄漏` 硬门通过；**r 形状不单调**，"
    "故预登记否证条件未整体触发、按臂分判。判词只在本设定内成立（S=1500 快照 / char enwik8 / 12.8M 结构 / 双 seed），"
    "不外推。**当前（最后一轮）独立复核判词**：见 `.scratch/63-axis6-predict/REVIEW-C2-632.md`（本轴派发的 fresh 复核，"
    "结果与处置记录在该报告与 `.scratch/63-axis6-predict/TICKET.md`）。**残余缺口（如实登记）**：`L5` 确定性未获独立验证"
    "（未取得 GPU 锁重跑，两项为 None）；r 非单调原因未查；PAD 布局只测随机位置；与 6.3.1 的 K≥2 过渡、更大规模未测——见 `gaps`。\""
)

WP1 = """    acceptance: "各臂预算对齐、未填位置因果性单测通过；序列级 bpc 相对全因果 AR 的差可分（越 0.014 噪声带）或明确「不越带」；结论不越尺度外推"
    evidence: "axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/"
    review: "跑完另派 fresh 复核（SCHEMA.md §14/§14b）——已做：`.scratch/63-axis6-predict/REVIEW-C2-632.md`（fresh、未参与写作、只读、带 shell），判词与 findings 处置见该报告与本项 `conclusion_review`"
    status: "已完成"
"""


def main():
    t = open(P, encoding="utf-8").read()

    # conclusion（原为空串）
    t = sub(t, 'conclusion: ""', 'conclusion: "%s"' % CONCL, tag="-CONCL")

    # failure_analysis
    t = sub(t, 'failure_analysis: ""', 'failure_analysis: "%s"' % FAIL, tag="-FAIL")

    # WP1
    t = sub(t, """    acceptance: "各臂齐备且预算对齐；未填位置因果性单测通过（无未来泄漏）；质量-代价曲线齐备"
    evidence: "axes/6-predict/6.3.2-pad-fill/experiments/"
    review: "跑完另派 fresh 复核（SCHEMA.md §14/§14b，复核排在执行之后）"
    status: "未开始"
""", WP1, tag="-WP1")

    # experiments
    t = sub(t, "experiments: []\n",
            'experiments:\n  - "axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m"\n', tag="-EXP")

    # gaps
    t = sub(t, '  - "本仓零经验：需先做最小判决（13M、{AR, pad 填空 r=10%/25%/50%, 因果填空}，matched-step + 同 token 预算）"\n',
            '  - "r 形状不单调（r=0.25 代价最小）的原因未查清：下一步最小实验 = 用同一 harness 扫更密 r 网格（0.05–0.75）'
            '并同时报训练 mask 密度与评测位置密度的分解，成本 ≈3 min/点 GPU"\n'
            '  - "`L5` 确定性未获独立验证（未取得 GPU 锁重跑，`results.json::L5` 两项为 None）；补做成本 = 一次同 seed 重跑 ≈3 min GPU"\n'
            '  - "本项在 **S = 1500 快照口径**可判，与仓库既有 6000 步档不可比；收敛刻度下代价是否缩小未测"\n', tag="-GAP")

    # conclusion_review 整行替换
    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("conclusion_review: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: conclusion_review 行命中 %d 次" % len(idx))
    lines[idx[0]] = CR
    t = "\n".join(lines)

    # 里程碑：规模标签与快照口径
    t = sub(t, "在 13M matched-step / 同 token 预算下的序列级 bpc",
            "在 12.8M（实测 12,813,312）matched-step / 同 token 预算下的序列级 bpc", tag="-MIL")

    # status/ended
    t = sub(t, '\nstatus: "未开始"\nstarted: "2026-10-13T10:47:00+08:00"\nended: ""',
            '\nstatus: "已完成"\nstarted: "2026-10-13T10:47:00+08:00"\nended: "%s"' % TS, tag="-STATUS")

    entry = (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "Phase C 收口（WP1 = `experiments/2671-padfill-13m`，实验目录内自带 harness）：conclusion 首次填入'
        "（(a) 填充质量代价 +1.07…+2.37 bpc、CI 不含 0 且远越 0.014 带；(b) r 形状不单调 ⇒ 预登记否证条件的「单调恶化」"
        "子条件未满足、按臂分判；(c) 可见性独立增益 +0.1635；(d) 不受未来泄漏硬门通过 + gate probe 可失败；(e) 代价轴；"
        "(f) 未测清单含 L5 确定性未验证）；failure_analysis 填入（设计/结构归因，非 baseline 错）；WP1 status 未开始→已完成；"
        "experiments 增 2671；milestone 规模标签按实测更正为 12,813,312；gaps 换成剩余项（r 非单调原因、L5 确定性、快照口径）；"
        'status 未开始→已完成、ended 落盘。"\n' % TS
    )
    t = t.rstrip("\n") + "\n" + entry
    open(P, "w", encoding="utf-8").write(t)
    print("OK: %s" % os.path.relpath(P, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
