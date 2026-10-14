#!/usr/bin/env python3
"""Main 裁决落地（新预算）：① 6.3.3 innovation 改写为实际机制；② 6.4/6.3.2 的规模标签更正逐条披露。
   scripted，带命中计数断言。不改任何数字与判据。"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P633 = os.path.join(ROOT, "axes/6-predict/6.3.3-latent-regression/project.yaml")
P64 = os.path.join(ROOT, "axes/6-predict/6.4-training-fill/project.yaml")
P632 = os.path.join(ROOT, "axes/6-predict/6.3.2-pad-fill/project.yaml")
TS = "2026-10-14T23:20:00+08:00"

INNOV = (
    "相对普通 decoder-only transformer 的逐位置离散 AR 目标，在**冻结**主干上加 k 个变换头（每个 d×d），"
    "一次前向从同一位置的主干隐状态 `h_i` 产出未来 k 个位置的表征 `z_j`，再各自经**共享** tied 读出"
    "（`logits_j = z_j @ E^T`，E = 主干 embed 权重、两臂共用）取 argmax（贪心）；训练目标 = 对未来位置真值 token "
    "的 CE + **latent 匹配监督项** `λ · MSE(z_j, h_true_{i+1+j})`（把 `z_j` 拉向主干在位置 i+1+j 的真实隐状态）。"
    "**机制限定（本仓实测可核，2660 的 L0）**：在共享 tied 读出 + 同参数量下，λ=0 的「回归臂」与「离散分类臂」"
    "**数学等价且同种子逐位相同**（同一条代码路径、同参数量）⇒ 本项实际改变的自由度只有**监督目标**"
    "（latent 匹配监督 vs 纯 CE），**不是**「回归输出 vs 分类输出」两套参数化；故结论表述为「latent（隐状态）匹配监督"
    "有独立增益」。改动量级：额外 k 个 d×d 变换（量级 k×d×d 参数，与挂 k 个独立离散头同参数量级）、主干与读出共享、"
    "主干冻结不动（实测 13M 档挂头 Δbpc = 0.0000，冻结断言 `grad is None` 通过）。"
)


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


def main():
    # ① 6.3.3 innovation 改写（Main 裁决 B5.4-2；只改机制描述，不改任何数字）
    t = open(P633, encoding="utf-8").read()
    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("innovation: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: innovation 行命中 %d 次" % len(idx))
    lines[idx[0]] = 'innovation: "%s"' % INNOV
    t = "\n".join(lines)
    t = t.rstrip("\n") + (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead（Main 裁决 B5.4-2）"\n'
        '    change: "`innovation` 按实际机制改写（Main 批准）：由「回归到连续 latent 再贪心」改为'
        "「latent 匹配监督（把 z_j 拉向主干未来真值隐状态）」，并写明「在共享 tied 读出 + 同参数量下与离散分类头数学等价、"
        '同种子逐位相同」这一本仓可核的等价性（依据 2660 的 L0）。仅改写机制描述，数字 / 结论 / 判据未变。"\n' % TS
    )
    open(P633, "w", encoding="utf-8").write(t)

    # ② 6.4 规模标签更正披露
    t = open(P64, encoding="utf-8").read()
    t = t.rstrip("\n") + (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead（Main 批准的规模标签更正）"\n'
        '    change: "规模标签正名：预登记为「13M」，本仓 23e 协议与 2672 的**实测参数量 = 6,962,688**'
        '（d=256/L=8/h=8/mlp_mult=4）——仅正名、未改判据（判据 / 阈值 / 否证条件逐字未动）。"\n' % TS
    )
    open(P64, "w", encoding="utf-8").write(t)

    # ③ 6.3.2 规模标签更正披露
    t = open(P632, encoding="utf-8").read()
    t = t.rstrip("\n") + (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead（Main 批准的规模标签更正）"\n'
        '    change: "规模标签正名：预登记为「13M」，本仓 2671 的**实测参数量 = 12,813,312**'
        '（d=256/L=12/h=8/mlp_mult=6；两臂逐位相同）——仅正名、未改判据。"\n' % TS
    )
    open(P632, "w", encoding="utf-8").write(t)
    print("OK: 6.3.3 innovation 改写 + 6.4/6.3.2 规模正名披露")
    return 0


if __name__ == "__main__":
    sys.exit(main())
