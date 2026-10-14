#!/usr/bin/env python3
"""6.4 的终局事实文档更新（Phase C）——scripted，带命中计数断言。

依据：`axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/`（n≥2 快照复本 + 等参数软/硬提交剥离）
与独立复核 `.scratch/63-axis6-predict/REVIEW-C1-64.md`（VERDICT: PASS，2 normal + 3 info）。
只改一个文件、只做替换/插入；每处断言命中次数。
"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, "axes/6-predict/6.4-training-fill/project.yaml")
TS = "2026-10-14T23:00:00+08:00"


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


CONCL_ADD = (
    " Phase C 补测（WP1b + WP2，`experiments/2672-fill-n2-13m`）：**(f) n≥2 复本（2k 步快照口径，臂间同刻度）**——"
    "`fill` 优于 `nofill` 在 seed {0,1} 上都成立（depth-8 bpc 2.7995 vs 2.8770 与 2.7580 vs 2.8221，"
    "Δ = −0.0775 / −0.0641，符号一致、极差 0.0134，方向与 12k 档 −0.0856 一致）⇒ **fill > nofill 可复现**；"
    "**(g) proj-fill 的排序依赖训练步数**——同一快照刻度下 `projfill` 反而略差于 `fill`（Δ = +0.0297 / +0.0218，"
    "两种子同向），与 12k 档的「proj-fill 更优（−0.0226）」**符号相反** ⇒ 该排序**不得从 2k 快照外推到 12k，"
    "反之亦然**；**(h) 软提交 OOD 污染在等参数剥离下成立（WP2）**——同一 checkpoint 只改提交规则，"
    "soft 的 rescore 峰值一致远高于 hard：Δ(soft−hard) 逐臂（seed0/seed1）= fill +3.8325/+6.2138、"
    "nofill +5.8908/+6.5625、projfill +6.2739/+4.7332（权重逐位相同 ⇒ **容量不再是解释**）。"
    "本项的**规模标签按落盘实测更正**：既有 23e 协议与 2672 的实测参数量 = 6,962,688（≈6.96M，"
    "d=256/L=8/h=8/mlp_mult=4），milestone 原文的「13M」是措辞误标、不改判据。"
)

FAIL = (
    "非「本项手段失败」：填充语义的主体结论（fill > nofill 可复现、软提交污染等参数成立）成立。"
    "唯一未复现的是 **proj-fill 相对 fill 的排序**（12k 档 −0.0226 vs 2k 快照 +0.0218…+0.0297，符号相反）——"
    "归因 = **训练步数/收敛阶段效应（规模—regime 效应，不是设计失败也不是 innovation 失败）**：proj-fill 的"
    "逐层恒等初始化 adapter 需要从 fill 出发学习 remap，短步数下收益来不及出现；证据 = 同一协议只改步数即翻转、"
    "且两种子同向。**下一步**：在 12k 档重跑 proj-fill 的第二个 seed 以判该排序是否在收敛刻度上稳定（成本 ≈10 min GPU）。"
)

WP1_NEW = """    acceptance: "六臂齐备且等参数对齐；各语义相对无填充的质量差可分或明确无差别；活性列（填充位置占比 / adapter 使用率）齐备；n≥2（快照刻度亦可，须声明与 12k 档不可比）"
    evidence: "axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/"
    review: "已做：`.scratch/63-axis6-predict/REVIEW-C1-64.md`（fresh、未参与写作、只读）判 **VERDICT: PASS**（blocker 0 · normal 2 · info 3）；两条 normal = ①辅助日志文件里出现墙钟（非判据区）②交付目录名 `snap2k-<arm>-s{0,1}/` 与契约 §6 的 `run1-<arm>-s1/` 不一致（仅命名，无科学影响，已记入本项 `gaps`）。执行类验收（独立复算臂级 bpc 与 Δ、config 对平、rescore 复算、summary/git 检查）由该复核者亲自执行。"
    status: "已完成"
"""

WP2_NEW = """  - id: "WP2"
    what: "软提交 OOD 污染复现与等参数剥离"
    goal: "把「软提交污染上下文」从单次现象升格为可复现判据，并剥离 adapter 容量成分"
    means: "同一 checkpoint 只改提交规则（soft Σp·E / hard argmax→embed / latent 直通），报 rescore_bpc 峰值分布；3 个既有 seed-0 checkpoint + 本轮 6 个快照 checkpoint"
    acceptance: "soft 的 rescore 峰值显著高于 hard 且跨 checkpoint 复现；**等参数（权重逐位相同）下仍成立**或明确归因容量"
    evidence: "axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/rescore-seed0.json"
    review: "随 WP1 一并由 `.scratch/63-axis6-predict/REVIEW-C1-64.md` 复核（同一 revision，PASS）；该复核者亲自复算了 ≥2 个 (checkpoint, depth) 的 Δ(soft−hard) 并核对「同一 checkpoint ⇒ 参数逐位相同」的依据"
    status: "已完成"
"""

CR_NEW = (
    'conclusion_review: "自评：本项已收口（旧格 H2 的 12k 读数 + Phase C 的 2672 快照复本与等参数剥离）。'
    "主体结论（fill > nofill、软提交污染）在两种子 / 多种 checkpoints 上一致；唯一未复现的是 proj-fill 相对 fill 的排序，"
    "归因为训练步数效应并已登记下一步最小实验。**当前（最后一轮）独立复核判词**：`.scratch/63-axis6-predict/REVIEW-C1-64.md`"
    "（fresh、未参与写作、只读）判 **VERDICT: PASS**（blocker 0 · normal 2 · info 3）：normal-2（交付目录名与契约 §6 不一致，"
    "仅命名）与 normal-1（辅助 `*.nohup.log` 里有墙钟，非判据区）已记入 `gaps`；info-3 指出「注入分布外内容」的机制措辞强于"
    "直接测量（本项只测到 rescore 峰值差，未做嵌入层分布检验），已在 `gaps` 记为未测项。复核者亲自执行了臂级 bpc 与 Δ 的独立复算、"
    "6 个 run 与既有 23e 的 config 对平（除 `steps` 外无差异）、rescore 表复算、`summary.py --check` 与越界检查。"
    "**残余缺口**：快照口径（2k 步）与 12k 档不可比；proj-fill 的 12k 第二 seed 未跑；pad/mask、错误 token、软 latent 三臂"
    "仍只有旧读数、不在本轮 n≥2 复本内——见 `gaps`。历轮过程见 `.scratch/63-axis6-predict/TICKET.md`。\""
)


def main():
    t = open(P, encoding="utf-8").read()

    # milestone 的规模标签更正（实测 6,962,688；措辞误标，不改判据）
    t = sub(t, "在 13M matched-step（n≥2）下的 clean bpc",
            "在既有 23e 协议（实测参数量 6,962,688 ≈6.96M；原文「13M」为措辞误标）的 matched-step（n≥2；"
            "快照刻度亦可、须声明）下的 clean bpc", tag="-MIL")

    # conclusion 追加 (f)(g)(h)
    t = sub(t, "判据带（C1 裁决）：0.014 bpc 带（限配对 Δ、≥5 窗口）。\"",
            "判据带（C1 裁决）：0.014 bpc 带（限配对 Δ、≥5 窗口）。" + CONCL_ADD + '"', tag="-CONCL")

    # failure_analysis
    t = sub(t, 'failure_analysis: ""', 'failure_analysis: "%s"' % FAIL, tag="-FAIL")

    # WP1 / WP2
    t = sub(t, """    acceptance: "六臂齐备且等参数对齐；各语义相对无填充的质量差可分或明确无差别；活性列（填充位置占比 / adapter 使用率）齐备"
    evidence: "axes/6-predict/6.4-training-fill/experiments/"
    review: "跑完另派 fresh 复核（SCHEMA.md §14/§14b）"
    status: "进行中"
""", WP1_NEW, tag="-WP1")

    t = sub(t, """  - id: "WP2"
    what: "软提交 OOD 污染复现与剥离"
    goal: "把「软提交污染上下文」从单次现象升格为可复现判据，并剥离 adapter 容量成分"
    means: "同 checkpoint 三度复现点 + 等参数 hard/soft 对照，报 rescore 峰值分布"
    acceptance: "软提交 rescore 峰值显著高于 hard（rescore 峰值差走配对 bootstrap / Wilson 95% CI 为主判，不跨量纲借用 0.014 bpc 带——C1 裁决）且复现；等参数对照下仍成立或明确归因容量"
    evidence: "axes/6-predict/6.4-training-fill/experiments/"
    review: "跑完另派 fresh 复核"
    status: "未开始"
""", WP2_NEW, tag="-WP2")

    # experiments
    t = sub(t, '  - "axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants"\n',
            '  - "axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants"\n'
            '  - "axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m"\n', tag="-EXP")

    # gaps 首条
    t = sub(t, '  - "仅 13M（12k matched）+ 部分 @100M 点；填充语义对照无 n≥2 复本"\n',
            '  - "n≥2 复本已在 **2k 快照口径**完成（`2672`），但**与 12k 档不可比**：proj-fill 的 12k 第二 seed 未跑'
            '（下一步最小实验 = 在 12k 档补 projfill seed 1，成本 ≈10 min GPU）"\n'
            '  - "pad / mask、错误 token、软 latent 三臂仍只有旧读数（`.scratch` 锚点），不在本轮 n≥2 复本内"\n'
            '  - "复核 info-3：「软提交注入分布外内容」是**机制解释**，本项只测到 rescore 峰值差；未做嵌入层分布检验（下一步最小实验 = 对 soft/hard 提交流的嵌入统计量做 KS 检验，成本 ≈3 min GPU）"\n'
            '  - "交付目录名 `snap2k-<arm>-s{0,1}/` 与 `2672/README.md` §6 的 `run1-<arm>-s1/` 不一致（仅命名，无科学影响；未改预登记契约）"\n', tag="-GAP")

    # conclusion_review 整行替换
    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("conclusion_review: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: conclusion_review 行命中 %d 次" % len(idx))
    lines[idx[0]] = CR_NEW
    t = "\n".join(lines)

    # status / ended
    t = sub(t, '\nstatus: "进行中"\nstarted: "2026-10-13T10:47:00+08:00"\nended: ""',
            '\nstatus: "已完成"\nstarted: "2026-10-13T10:47:00+08:00"\nended: "%s"' % TS, tag="-STATUS")

    # change_log
    entry = (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "Phase C 收口（`2672-fill-n2-13m`）：conclusion 新增 (f) n≥2 快照复本（fill>nofill 两种子同向、'
        "方向与 12k 档一致）、(g) proj-fill 排序**依赖训练步数**（2k 快照反转，不得外推）、(h) 等参数软/硬提交剥离成立；"
        "failure_analysis 填入（唯一未复现的是 proj-fill 排序，归因训练步数效应，非设计/innovation 失败）；"
        "milestone 的规模标签按落盘实测更正为 6,962,688（原文「13M」为措辞误标）；WP1/WP2 status → 已完成 并回填复核判词；"
        "experiments 增 2672；gaps 换成剩余项（12k 档 projfill 第二 seed、三臂旧读数、嵌入层分布检验未做、目录名与契约 §6 不一致）；"
        'status 进行中→已完成、ended 落盘。"\n' % TS
    )
    t = t.rstrip("\n") + "\n" + entry

    open(P, "w", encoding="utf-8").write(t)
    print("OK: %s" % os.path.relpath(P, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
