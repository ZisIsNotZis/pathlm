#!/usr/bin/env python3
"""轴 6 · 6.3.3 的终局事实文档更新（scripted，带命中计数断言；未命中即非零退出）。

用途：把 WP2（2670-freerun-bpc-13m）的读数与复核判词写进
`axes/6-predict/6.3.3-latent-regression/project.yaml`，并把 status 推到终局。
本脚本只改一个文件、只做「替换/插入」，每处替换都断言命中次数（未命中或多次命中 ⇒ 报错退出）。
"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, "axes/6-predict/6.3.3-latent-regression/project.yaml")

TS = "2026-10-14T21:45:00+08:00"

F_CONCL = (
    " (f) **路径 (i)（估计得贪心、不验证）的 free-run 序列级质量代价明确（WP2 / `2670-freerun-bpc-13m`）**："
    "在冻结主干上按「自己发出的 token 在自家预测分布下的 NLL」计 bpc，`ar` 逐 token 贪心 = 0.504356（13m_s0）/ "
    "0.489330（13m_s1）；一次前向直接发 2 个位置的 `disc_k2` 自评 Δbpc = **+0.7389** [+0.5745, +0.8918]（13m_s0 种子合并）/ "
    "**+0.9195** [+0.7927, +1.0555]（13m_s1），`lat_k2_lam1`（λ=1.0 的 latent 匹配监督）把差缩小到 **+0.4289** "
    "[+0.2817, +0.5867] / **+0.1633** [−0.0443, +0.4052]——**除 13m_s1 的 latent 臂外，全部 CI 不含 0 且越 0.014 bpc 带**。"
    "**外部真值锚**（另一主干 node-1 的 teacher-forced bpc，复核 N1 要求补）下 Δ 更大且同向：`disc_k2` **+4.1541 / +5.0708**、"
    "`lat_k2_lam1` **+1.2611 / +1.1701**（真实 enwik8 续写同口径 1.6040 / 1.6027）⇒ 自评口径**低估**而非高估代价；"
    "客观退化统计显示候选臂的流明显退化（`lat_k2_lam1` 最长连续相同 token 平均 ≈137–154，`disc_k2` 3.8–59，`ar` 仅 2.04）"
    "⇒ 该路径在本尺度**质量面不成立**，milestone 的可用形态是路径 (ii)「当草稿再验证」（代价 ≈ L）。"
    "代价面：路径 (i) tokens-per-forward = 2、每 token 主干等效 layer-passes ≈ 6.0（`ar` = 1 / 12）。"
)

F_TAIL = (
    "故 status = 已完成（milestone 的判据齐备：WP1 给出 (i)/(ii) 两路径的可分读数，WP2 补齐序列级 bpc 并给出真值锚；"
    "剩余未测项见 gaps）。"
)

F_FAIL = (
    "路径 (i)（估计得贪心、一次前向直接发 k 个位置、不验证）在本尺度**手段不成立**；归因 = "
    "**设计问题为主 + 结构问题次之**。设计问题：「不验证直接发」这一出口设计把 k 个位置的预测误差直接写进上下文并逐轮累积"
    "（实测真值锚 Δ = +1.26…+5.07 bpc，且 `lat_k2_lam1` 的流最长连续相同 token 平均 ≈137，即进入重复退化）。"
    "结构问题：一次回归出的 k 个 latent 无法互相条件（与 chain 式逐位置条件相反），且共享 tied 读出下两臂同参数化 ⇒ "
    "独立自由度只有监督目标，λ·MSE 只能缓解、不能消除该累积误差。**不是**「latent（隐状态）匹配监督本身失败」"
    "（WP1 已证它在 teacher-forced 命中率 +0.031、草稿接受率 +0.092 上有独立增益）。"
)

WP2 = """  - id: "WP2"
    what: "路径 (i) 的 free-run 序列级 bpc 与真值锚"
    goal: "补齐 milestone.output 要求的序列级 bpc，并回答「两臂各自自评」口径是否高估了路径 (i) 的质量代价（复核 N1）"
    means: "复用 2660 的头与 2631 的冻结主干（不重训）；12 prompt × 200 token 自由生成；bpc = 自家预测分布下发出 token 的 NLL；另落 token 流 + 客观退化统计 + 另一主干 teacher-forced 打分（外部锚）"
    acceptance: "臂级 bpc 与 Δ 的 prompt 聚类配对 95% CI 齐备；外部锚与自评同向；n≥2（两主干点 × 两头种子）"
    evidence: "axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/"
    review: "已做：`.scratch/63-axis6-predict/REVIEW-C1-2670.md`（fresh、未参与写作、只读、**带 shell**）判 **PASS**（blocker 0 · normal 1 · info 5）；其 N1（主判为自评口径、需真值锚）已按建议补测并落盘（`analysis.md::真值锚`）；该复核者亲自跑过 pytest（11 passed）/ 变异（5/5 抓红 + 逐字节还原）/ 独立复算（臂级 bpc 与 Δ/CI 逐值）/ 逐张量比对头权重 / 2640 快照哈希 / 越界 / 禁墙钟"
    status: "已完成"
"""

CR_NEW = (
    'conclusion_review: "自评：WP1（`2660`）与 WP2（`2670`）两条读数互不冲突——**latent（隐状态）匹配监督**在 teacher-forced '
    "命中率（+0.031）与草稿接受率（Δα₁ +0.0916 / Δτ +0.1464）上有独立增益，而路径 (i)「一次前向直接发 k 个位置、不验证」"
    "在序列级质量面不成立（真值锚 Δ +1.26…+5.07 bpc，远超 0.014 带，且有重复退化）。**当前（最后一轮）独立复核判词**："
    "`.scratch/63-axis6-predict/REVIEW-C1-2670.md`（fresh、未参与写作、只读、带 shell）判 **VERDICT: PASS**"
    "（blocker 0 · normal 1 · info 5）；其 N1（主判是两臂各自自评自家流 ⇒ 需真值锚）已按该复核建议补测并落盘"
    "（`analysis.md::真值锚`：生成流落盘 + 客观退化统计 + 自评/外部锚/真实续写三读数；外部锚与自评 **同向且代价更大**，"
    "`ar` 流最长连续相同 token 仅 2.04 ⇒ 复核担心的伪影不成立），故原判读**成立且措辞不弱化**。该复核者亲自执行了"
    "pytest（11 passed）、变异（5/5 抓红 + 逐字节还原 + 报告确定性）、独立复算（臂级 bpc 差 ≤1.02e-9、Δ 与 CI 逐 6 位小数一致）、"
    "逐张量比对 8 份头文件、2640 三份快照哈希、越界与禁墙钟检查。**残余缺口（如实登记）**：WP1 的历轮复核者（B1/B2/B3）"
    "无 shell，其执行类验收由 writer 与 axis6-lead 自跑；路径 (i) 只测 k=2、只有 13M 一种规模、真值锚用同族主干当 scorer"
    "（同族偏差未剥离）——见 `gaps`。历轮判词与处置过程见 `.scratch/63-axis6-predict/TICKET.md`。\""
)


def sub(text, old, new, n=1):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL: 命中 %d 次（期望 %d）：%r" % (c, n, old[:80]))
    return text.replace(old, new)


def main():
    with open(P, encoding="utf-8") as f:
        t = f.read()
    orig = t

    # 1) conclusion 追加 (f) 段（插在 (e) 的收尾句之前）
    t = sub(t, "故 status = 进行中（milestone 输出包的序列级 bpc 未齐）。\"", F_CONCL + F_TAIL + '"')

    # 2) failure_analysis
    t = sub(t, 'failure_analysis: ""', 'failure_analysis: "%s"' % F_FAIL)

    # 3) WP2 条目（插在 WP1 的 status 之后、roadmap 之前）
    t = sub(t, '    status: "已完成"\n\nroadmap:', '    status: "已完成"\n' + WP2 + "\nroadmap:")

    # 4) experiments 增加 2670
    t = sub(
        t,
        '  - "axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m"\n',
        '  - "axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m"\n'
        '  - "axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m"\n',
    )

    # 5) gaps 第一条（free-run 未测）换成剩余项
    old_gap = ('  - "自由跑（free-run）序列级 bpc 未测：只有 teacher-forced 命中率 / CE 与草稿接受率；'
               "下一步最小实验 = 用同一批头做 12 prompt × 200 token 的自由生成并按 token 计 bpc"
               '（与 stock AR 同框），成本约 5–10 min GPU"\n')
    new_gaps = ('  - "free-run bpc 只测了 k=2、温度 0 贪心、路径 (i) 直接发；k=4 的 free-run 与温度采样未测'
                "（下一步最小实验 = 复用 `2670` 的驱动加 k=4 头 + 12 prompt × 200 token，成本 ≈30 s GPU）\"\n"
                '  - "真值锚只用「另一主干 node-1」当 scorer（同族偏差未剥离）：未用族外第三方模型打分；'
                '路径 (i) 只有 13M 一种规模"\n')
    t = sub(t, old_gap, new_gaps)

    # 6) conclusion_review 整行替换
    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("conclusion_review: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: conclusion_review 行命中 %d 次" % len(idx))
    lines[idx[0]] = CR_NEW
    t = "\n".join(lines)

    # 7) status / ended
    t = sub(t, '\nstatus: "进行中"\nstarted: "2026-10-13T10:47:00+08:00"\nended: ""',
            '\nstatus: "已完成"\nstarted: "2026-10-13T10:47:00+08:00"\nended: "%s"' % TS)

    # 8) roadmap 收尾行
    t = sub(t, '  - "复核通过后写 conclusion，更新 status"',
            '  - "WP2：路径 (i) 的 free-run 序列级 bpc + 真值锚（`2670-freerun-bpc-13m`）"\n'
            '  - "收口：conclusion / status / 复核判词齐备（2026-10-14 完成）"')

    # 9) change_log 追加一条（放在 change_log 列表末尾）
    entry = (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "Phase C 收口（WP2 `2670-freerun-bpc-13m`）：conclusion 新增 (f) 路径 (i) 的 free-run 序列级 bpc'
        "（自评 Δ = +0.74/+0.92 bpc；外部真值锚 Δ = +4.15/+5.07，同向且更大；候选臂流明显退化）；\n"
        'failure_analysis 首次填入（路径 (i) 手段不成立；设计问题为主 + 结构问题次之）；WP2 条目新增（含 REVIEW-C1-2670 判词）；'
        "experiments 增 2670；gaps 换成剩余项（k=4/温度采样未测、真值锚同族偏差、单一规模）；"
        'status 进行中→已完成、ended 落盘。"\n' % TS
    )
    if not t.endswith("\n"):
        t += "\n"
    t += entry

    if t == orig:
        raise SystemExit("FATAL: 没有任何替换发生")
    with open(P, "w", encoding="utf-8") as f:
        f.write(t)
    print("OK: 写入 %s" % os.path.relpath(P, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
