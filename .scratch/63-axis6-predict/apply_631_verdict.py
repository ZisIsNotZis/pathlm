#!/usr/bin/env python3
"""6.3.1 按 REVIEW-C1-631（PASS + 4 normal + 3 info）收口：status → 被否证、判词更新、K 形状按 info-3 削弱、N3 显式裁决。"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, "axes/6-predict/6.3.1-diffusion/project.yaml")
TS = "2026-10-14T23:16:00+08:00"


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


def main():
    t = open(P, encoding="utf-8").read()

    # info-3：K 形状不当作精细可分结论（主判只看 K=1 也已越带）
    t = sub(t, "(b) **提高 K 不能把质量拉回带内**（Δ 随 K 单调变差：+2.6815 → +2.6873）",
            "(b) **提高 K 不能把质量拉回带内**（Δ 随 K 单调变差：+2.6815 → +2.6873；"
            "但 K 之间的极差仅 0.006、远小于 CI 半宽 ≈0.11 ⇒ **不把 K 形状当精细可分结论**，"
            "主判只看 K=1 就已在点估计上越带约 190 倍）", tag="-I3")
    # N3：K=1 连接点的显式裁决（两 CI 不相交）
    t = sub(t, "（本刻度的 regime 差，如实报、不外推）",
            "（本刻度的 regime 差，如实报、不外推）；该连接点的**统计裁决**：两个 CI 不相交（+2.57617 > +1.14548）"
            "⇒ 二者在本刻度**可区分**，连接点判「不成立（可区分）」，差异来源列为「变密度掩码训练 vs 固定 r=0.25」", tag="-N3")
    # 未收尾项与未交付项登记进 gaps
    t = sub(t, '  - "只有 S=1500 快照一种刻度（与 6000 步档不可比）；更大规模未测"\n',
            '  - "只有 S=1500 快照一种刻度（与 6000 步档不可比）；更大规模未测"\n'
            '  - "契约 §3 的 L3（用冻结 AR 对生成序列 rescore + 配对差 + CI）**未交付**，以 gap 替代：因为 2671 的 AR checkpoint 未落盘，'
            '只能只读复用其读数；下一步最小实验 = 落一个 AR checkpoint 后补 rescore（成本 ≈3 min GPU）"\n'
            '  - "本项 `results.json` 只落了 mdiff 侧逐 block 原始量，AR 侧逐 block 量需回读 `2671/results.json` ⇒ 自足可复算依赖外部文件存活（轻微脆弱点）"\n'
            '  - "`RESUME.md` 由超时的 writer 留下、内容停在「未完成」（与本项实际已 finalize 不一致）——事实以 `analysis.md`/`results.json` 为准"\n',
            tag="-GAPS")

    # conclusion_review（当前判词 = REVIEW-C1-631 PASS）
    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("conclusion_review: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: conclusion_review 行命中 %d 次" % len(idx))
    lines[idx[0]] = (
        'conclusion_review: "自评：本项从零读数推到明确否证——掩码扩散（K 步双向去噪）在本快照刻度上相对全因果 AR 的质量差为 '
        "+2.68…+2.69 bpc（CI 不含 0、远越 0.014 带），K=1 单点即已越带约 190 倍；K 之间的差异在本数据下不可精细分辨（极差 0.006 ≪ CI 半宽 0.11），"
        "但「提高 K 拉不回」在点估计上成立且主判不依赖它。K=1 连接点与 6.3.2 的 `fill_bidir_r25` 同 M 口径：两 CI 不相交 ⇒ **可区分**，"
        "故「连接点不可区分」不成立，差异来源列为变密度掩码训练 vs 固定 r=0.25。**当前（最后一轮）独立复核判词**："
        "`.scratch/63-axis6-predict/REVIEW-C1-631.md`（fresh、未参与写作、只读、带 shell）判 **VERDICT: PASS**"
        "（blocker 0 · normal 4 · info 3）：主判与 §4 否证条件触发均被其独立复算逐值吻合；4 条 normal 全为**完整性/未收尾**类"
        "（`RESUME.md` 未收尾、L3 以 gap 替代、K=1 连接点未显式裁决、AR 侧逐 block 原始量未复制进本项）——已逐条记入 `gaps`，不推翻主判。\""
    )
    t = "\n".join(lines)

    # status → 被否证
    t = sub(t, '\nstatus: "进行中"\nstarted: "2026-10-13T10:47:00+08:00"\nended: ""',
            '\nstatus: "被否证"\nstarted: "2026-10-13T10:47:00+08:00"\nended: "%s"' % TS, tag="-STATUS")

    t = t.rstrip("\n") + (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "按独立复核 REVIEW-C1-631.md（PASS，blocker 0 / normal 4 / info 3）收口：status 进行中→**被否证**'
        "（契约 §4 否证条件整体触发：任一 K 越带 + 提高 K 拉不回）、ended 落盘；按 info-3 把「K 形状」明确降级为"
        "「不当作精细可分结论（极差 0.006 ≪ CI 半宽 0.11），主判只看 K=1」；按 N3 补 K=1 连接点的**显式统计裁决**"
        "（与 6.3.2 `fill_bidir_r25` 的 CI 不相交 ⇒ 可区分）；按 N1/N2/N4 把 RESUME 未收尾、L3 未交付、AR 侧逐 block 量未复制"
        '记入 `gaps`。数字与结论不变。"\n' % TS
    )
    open(P, "w", encoding="utf-8").write(t)
    print("OK 6.3.1 -> 被否证")
    return 0


if __name__ == "__main__":
    sys.exit(main())
