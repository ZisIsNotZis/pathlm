#!/usr/bin/env python3
"""6.3.3 的复核 #C2 返工（B1 + N2 + N3 + N4 + I3 + I4 + I5）——scripted，带命中计数断言。

依据：`.scratch/63-axis6-predict/REVIEW-C2-633.md`（VERDICT: FAIL，1 blocker + 3 normal + 6 info；
复核者明确声明 WP2 的执行/数字/证据链全部通过其独立复算，缺陷都是**文档一致性**类）。
只改两处文件、只做替换；每处断言命中次数，未命中或多次命中即报错退出。
"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P_YAML = os.path.join(ROOT, "axes/6-predict/6.3.3-latent-regression/project.yaml")
P_ANA = os.path.join(
    ROOT, "axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/analysis.md"
)


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


def main():
    y = open(P_YAML, encoding="utf-8").read()

    # B1：conclusion (e) 里「free-run 序列级 bpc 未测」与 (f)/gaps 矛盾 ⇒ 去掉该分句
    y = sub(y, "(e) 未测：自由跑（free-run）序列级 bpc 未测（只有 teacher-forced 命中率 / CE 与草稿接受率）、",
            "(e) 未测：", tag="-B1b")

    # N2：真值锚 Δ 下界（1.26 → 1.1701）
    y = sub(y, "真值锚 Δ = +1.26…+5.07 bpc", "真值锚 Δ = +1.1701…+5.0708 bpc", tag="-N2")

    # N3：退化统计改成与同句其它支同格式（min–max）
    y = sub(y, "`lat_k2_lam1` 最长连续相同 token 平均 ≈137–154",
            "`lat_k2_lam1` 最长连续相同 token 均值 99.8–153.8（四个窗口）", tag="-N3")

    # I4：真值锚数字只留 conclusion (f) 一处 ⇒ conclusion_review 改为指针式表述
    y = sub(y, "（真值锚 Δ +1.26…+5.07 bpc，远超 0.014 带，且有重复退化）",
            "（真值锚 Δ 四个窗口全部同向且远超 0.014 带，数值见 `conclusion` (f)；候选臂流有重复退化）",
            tag="-I4")

    # I3：roadmap 里去掉完成日期
    y = sub(y, '"收口：conclusion / status / 复核判词齐备（2026-10-14 完成）"',
            '"收口：conclusion / status / 复核判词齐备"', tag="-I3")

    # I5：标注该执行清单属 C1 轮
    y = sub(y, "该复核者亲自执行了pytest（11 passed）、变异（5/5 抓红 + 逐字节还原 + 报告确定性）",
            "该复核者在 C1 轮亲自执行了 pytest（11 passed）、变异（5/5 抓红 + 逐字节还原 + 报告确定性）",
            tag="-I5")

    # change_log：记本次返工
    y = y.rstrip("\n") + (
        '\n  - date: "2026-10-14T22:30:00+08:00"\n'
        '    actor: "axis6-lead"\n'
        '    change: "按复核 #C2（`.scratch/63-axis6-predict/REVIEW-C2-633.md`，VERDICT FAIL、缺陷均为文档一致性类）返工：'
        "B1 去掉 conclusion (e) 里与 (f)/gaps 矛盾的「free-run 序列级 bpc 未测」分句；N2 真值锚 Δ 下界改为 +1.1701；"
        "N3 退化统计改为与同句同格式的均值区间 99.8–153.8；I4 真值锚数字只留 conclusion (f) 一处、conclusion_review 改指针式；"
        'I3 roadmap 去掉完成日期；I5 标注执行清单属 C1 轮。数字与结论均未变。"\n'
    )
    open(P_YAML, "w", encoding="utf-8").write(y)

    # N4：analysis.md 要点里「自评 Δ」指代不明 ⇒ 点明是自评 free-run 列
    a = open(P_ANA, encoding="utf-8").read()
    a = sub(a, "故外部锚 Δ 在全部 4 个窗口都**比自评 Δ 更大**（是）",
            "故外部锚 Δ 在全部 4 个窗口都**比自评 free-run Δ 更大**（是；注意不是 teacher-forced 自评列："
            "`lat_k2_lam1 / 13m_s0` 外部锚 +1.2611 略小于 teacher-forced 自评 +1.4521）", tag="-N4")
    open(P_ANA, "w", encoding="utf-8").write(a)
    print("OK: 两处文件已返工")
    return 0


if __name__ == "__main__":
    sys.exit(main())
