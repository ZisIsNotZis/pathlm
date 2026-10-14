#!/usr/bin/env python3
"""最终复核（REVIEW-C2-FINAL）的 findings 处置：修 6.3.2 的悬空指针 + 6.3.3 的「当前判词」过时。
   scripted，带命中计数断言。"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P632 = os.path.join(ROOT, "axes/6-predict/6.3.2-pad-fill/project.yaml")
P633 = os.path.join(ROOT, "axes/6-predict/6.3.3-latent-regression/project.yaml")
TS = "2026-10-14T23:12:00+08:00"


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


def main():
    # --- 6.3.2：悬空指针 → 实际报告路径（N1） ---
    t = open(P632, encoding="utf-8").read()
    t = sub(t, "见 `.scratch/63-axis6-predict/REVIEW-C2-632.md`（本轴派发的 fresh 复核，"
               "结果与处置记录在该报告与 `.scratch/63-axis6-predict/TICKET.md`）",
            "`.scratch/63-axis6-predict/REVIEW-C2-FINAL.md`（本轴派发的 fresh 复核、未参与写作、只读、带 shell；"
            "判 **VERDICT: PASS**，blocker 0 · normal 1【本指针命名缺陷，即本处修复】· info 2；其独立复算见该报告）",
            tag="-632-CR")
    t = sub(t, "已做：`.scratch/63-axis6-predict/REVIEW-C2-632.md`（fresh、未参与写作、只读、带 shell），"
               "判词与 findings 处置见该报告与本项 `conclusion_review`",
            "已做：`.scratch/63-axis6-predict/REVIEW-C2-FINAL.md`（fresh、未参与写作、只读、带 shell；**PASS**）",
            tag="-632-WP")
    t = sub(t, "（快照刻度；收敛档为 ≈1.48）",
            "（快照刻度；收敛档 ≈1.4836 见 6.2 的 `2610-frozen-trunk-jumphead-13m`——不同步数档，仅作方向参照）",
            tag="-632-Ia")
    t = t.rstrip("\n") + (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "按最终复核 REVIEW-C2-FINAL.md 的 N1 与 info-a 修：`conclusion_review` 与 WP1.review 的复核指针'
        "从悬空的 `REVIEW-C2-632.md`（本轴收尾脚本写入时用错文件名，实际报告是 C2-FINAL）改为实际报告路径；"
        'conclusion (a) 的收敛档参照补出处（6.2/2610 clean bpc 1.4836）。不改任何数字与结论。"\n' % TS
    )
    open(P632, "w", encoding="utf-8").write(t)

    # --- 6.3.3：把「当前判词」更新为最新一轮（C2 FAIL → 返工 → C2-FINAL PASS） ---
    t = open(P633, encoding="utf-8").read()
    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("conclusion_review: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: 633 conclusion_review 行命中 %d 次" % len(idx))
    lines[idx[0]] = (
        'conclusion_review: "自评：WP1（`2660`）与 WP2（`2670`）两条读数互不冲突——**latent（隐状态）匹配监督**在 '
        "teacher-forced 命中率（+0.031）与草稿接受率（Δα₁ +0.0916 / Δτ +0.1464）上有独立增益，而路径 (i)「一次前向直接发 k 个"
        "位置、不验证」在序列级质量面不成立（真值锚 Δ +1.1701…+5.0708 bpc，远超 0.014 带，且有重复退化）。"
        "**当前（最后一轮）独立复核判词**：`.scratch/63-axis6-predict/REVIEW-C2-FINAL.md`（fresh、未参与写作、只读、带 shell）"
        "对**本项的返工**判 **VERDICT: PASS**——上一轮 `REVIEW-C2-633.md` 的 FAIL（B1 与 (f)/gaps 矛盾的「未测」旧话、N2 真值锚下界、"
        "N3 退化统计格式、N4 措辞指代、I3/I4/I5）已逐条返工并经该轮核验到位、未引入新矛盾。**残余缺口（如实登记）**："
        "WP1 的历轮复核者（B1/B2/B3）无 shell，其执行类验收由 writer 与 axis6-lead 自跑；路径 (i) 只测 k=2、只有 13M 一种规模、"
        "真值锚用同族主干当 scorer（同族偏差未剥离）——见 `gaps`。历轮判词与处置过程见 `.scratch/63-axis6-predict/TICKET.md`。\""
    )
    t = "\n".join(lines)
    t = t.rstrip("\n") + (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "按最终复核 REVIEW-C2-FINAL.md 的判词更新 `conclusion_review`：当前（最后一轮）判词由 C1 轮改为'
        '「#C2 FAIL → 返工 → C2-FINAL PASS」，并写明返工项已被核验到位、未引入新矛盾。不改数字与结论。"\n' % TS
    )
    open(P633, "w", encoding="utf-8").write(t)
    print("OK: 632 + 633 已按最终复核修")
    return 0


if __name__ == "__main__":
    sys.exit(main())
