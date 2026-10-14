#!/usr/bin/env python3
"""6.3.1 的终局事实文档更新（Phase C，WP1 = 2674）——scripted，带命中计数断言。

用法：`python3 apply_631_final.py <PASS|FAIL|PENDING>`；判词文本按参数注入 status 与 conclusion_review。
"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, "axes/6-predict/6.3.1-diffusion/project.yaml")
TS = "2026-10-14T23:20:00+08:00"


def sub(text, old, new, n=1, tag=""):
    c = text.count(old)
    if c != n:
        raise SystemExit("FATAL%s: 命中 %d 次（期望 %d）：%r" % (tag, c, n, old[:90]))
    return text.replace(old, new)


CONCL = (
    "本地判决（WP1 = `experiments/2674-mdiffusion-13m`，**S = 1500 快照口径**（与 2671 同刻度）、只读臂间增量与排序；"
    "复用 6.3.2 的同 harness 并只读其全因果 AR 参照；n≥2 = 模型种子 {0,1}）：**(a) 掩码扩散在本尺度/本任务上不成立**——"
    "吸收态掩码扩散臂（训练时 `r ~ U(0,1)`、每位置独立替换为 `[mask]`、只在被 mask 位置算 CE）+ K 步置信度解掩码采样，"
    "在**同一批位置 `M`** 上相对全因果 AR 的 teacher-forced bpc 差为 K=1 **+2.6815** [+2.57617, +2.79239]、K=4 **+2.6851** "
    "[+2.57898, +2.79591]、K=8 **+2.6866** [+2.58035, +2.79774]、K=16 **+2.6873** [+2.58136, +2.79883]（NELBO 逐步均值 "
    "5.0133/5.0275/5.0423）——block 聚类配对 bootstrap 95% CI **全部不含 0 且远越 0.014 bpc 参考带**；参照 = AR 全位置 "
    "teacher-forced bpc 2.2704。(b) **提高 K 不能把质量拉回带内**（Δ 随 K 单调变差：+2.6815 → +2.6873）⇒ 契约 §4 的否证"
    "条件**整体触发**（「任一 K 越带且提高 K 拉不回」两个子条件都成立）。(c) **K=1 连接点（与 6.3.2）**：本项 K=1 的 Δ "
    "**+2.6815** 明显大于 6.3.2 固定 r=0.25 训练的 `fill_bidir_r25` 同 M 口径 Δ **+1.0726** ⇒ 在**同一 K=1 出口**上，"
    "「r ~ U(0,1) 的变密度掩码训练」比「固定 r=0.25」差约 1.61 bpc（本刻度的 regime 差，如实报、不外推）。(d) 代价面"
    "（如实报、不作判据）：一次去噪前向 = 全段一次双向前向（layer-passes = L×T），K 步 ⇒ 每 token 等效 FLOPs = K×L"
    "（AR 每 token = L）；tokens-per-forward = |M|/K；每 block 顺序前向次数 = K（AR 为 T−1）——**顺序前向确实由 T 降到 K，"
    "但质量代价远超可接受范围**。(e) 未测：离散扩散的**均匀核**臂、噪声调度/采样器扫描、与 6.3.3 同框、更大规模；"
    "`L5` 的确定性未获独立验证（见 `gaps`）。"
)

FAIL = (
    "假设被本地证据否证 ⇒ 归因（按顺序给证据）：**结构问题为主**——「双向 + 被去噪位置彼此不能条件」把条件依赖整体删掉，"
    "证据 = Δ 与 K 几乎无关（K=1→16 只从 +2.6815 变到 +2.6873），说明瓶颈不在「迭代次数/采样器」而在该出口本身；"
    "**并含设计/regime 问题**——同一 K=1 出口下 `r ~ U(0,1)` 的变密度训练比 6.3.2 的固定 r=0.25 差 1.61 bpc，说明噪声调度"
    "（本实验按契约定的默认）在本步数下未被调好；**不是** baseline 不公平（同 harness、同 token 预算、同评测口径、"
    "只读复用同一 AR 参照）也**不是**测量错误（L0 位置集对齐通过、禁墙钟 0 命中）。"
)

CR = (
    'conclusion_review: "自评：本项从零读数推到明确否证——掩码扩散（K 步双向去噪）在本快照刻度上相对全因果 AR 的质量差为 '
    "+2.68…+2.69 bpc（CI 不含 0、远越 0.014 带），且提高 K 不能拉回（Δ 随 K 单调变差）；K=1 连接点显示「变密度掩码训练」"
    "比 6.3.2 的固定 r=0.25 差 1.61 bpc，故归因以结构为主并含 regime 成分。判词只在本设定内成立（S=1500 快照 / char enwik8 / "
    "同一共享 harness / 双 seed），不外推。**当前（最后一轮）独立复核判词**：见 `.scratch/63-axis6-predict/REVIEW-C1-631.md`"
    "（本轴派发的 fresh 复核；结果与处置见该报告与 `.scratch/63-axis6-predict/TICKET.md`）。**残余缺口（如实登记）**：本 WP 的 "
    "writer agent 在 30 min 运行时上限超时退出（实验已由 detached supervisor 跑完、产物已落盘），复核者发现的未收尾项见其报告；"
    "`L5` 确定性未独立验证；均匀核臂与调度扫描未做——见 `gaps`。\""
)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "PENDING"
    t = open(P, encoding="utf-8").read()

    t = sub(t, 'conclusion: ""', 'conclusion: "%s"' % CONCL, tag="-CONCL")
    t = sub(t, 'failure_analysis: ""', 'failure_analysis: "%s"' % FAIL, tag="-FAIL")
    t = sub(t, '  status: "未开始"', '  status: "被否证"', tag="-WP")
    t = sub(t, '  - "本仓零经验：需先做最小判决（13M、{AR, 掩码扩散 K=1/4/8/16} 两族，matched-step + 同 token 预算）"\n',
            '  - "本次只做了 1 个扩散臂 × seed{0,1}；**均匀核离散扩散臂未做**（下一步最小实验 = 同 harness 把腐蚀核换成'
            '均匀随机 token，成本 ≈2.5 min/seed GPU）"\n'
            '  - "噪声调度只用了契约默认 `r ~ U(0,1)`；未扫调度（K=1 连接点提示变密度训练差 1.61 bpc ⇒ 调度是首要可疑项）"\n'
            '  - "`L5` 确定性未获独立验证；writer agent 因 30 min 运行时上限超时退出，未收尾项见 REVIEW-C1-631"\n'
            '  - "只有 S=1500 快照一种刻度（与 6000 步档不可比）；更大规模未测"\n', tag="-GAP")
    t = sub(t, 'experiments: []\n', 'experiments:\n  - "axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m"\n',
            tag="-EXP")
    t = sub(t, "在 13M matched-step / 同 token 预算下的序列级 bpc",
            "在 12.8M（同 harness、实测参数量见 2674 落盘）matched-step / 同 token 预算下的序列级 bpc", tag="-MIL")

    lines = t.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("conclusion_review: ")]
    if len(idx) != 1:
        raise SystemExit("FATAL: conclusion_review 行命中 %d 次" % len(idx))
    lines[idx[0]] = CR
    t = "\n".join(lines)

    status = {"PASS": "被否证", "FAIL": "进行中", "PENDING": "进行中"}[mode]
    t = sub(t, '\nstatus: "未开始"\nstarted: "2026-10-13T10:47:00+08:00"\nended: ""',
            '\nstatus: "%s"\nstarted: "2026-10-13T10:47:00+08:00"\nended: "%s"'
            % (status, TS if status == "被否证" else ""), tag="-STATUS")

    entry = (
        '\n  - date: "%s"\n'
        '    actor: "axis6-lead"\n'
        '    change: "Phase C（WP1 = `experiments/2674-mdiffusion-13m`，复用 6.3.2 harness）：conclusion 首次填入'
        "（K=1/4/8/16 的 Δ = +2.6815…+2.6873，CI 不含 0 且远越 0.014 带；提高 K 拉不回 ⇒ 契约 §4 否证条件整体触发；"
        "K=1 连接点显示变密度掩码训练比固定 r=0.25 差 1.61 bpc）；failure_analysis 填入（结构为主 + regime 成分）；"
        "milestone 规模标签按实测更正；gaps 换成剩余项（均匀核臂、调度扫描、L5 确定性、快照口径）；"
        'status 未开始→%s（复核 mode=%s）。"\n' % (TS, status, mode)
    )
    t = t.rstrip("\n") + "\n" + entry
    open(P, "w", encoding="utf-8").write(t)
    print("OK mode=%s status=%s" % (mode, status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
