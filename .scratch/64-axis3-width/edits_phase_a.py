#!/usr/bin/env python3
"""ticket 64 · Phase A 步骤 1：8 份 project.yaml 的脚本化改文本（`AGENTS.md` §8）。

纪律：每处替换必须**恰好命中 1 次**，否则非零退出并列出未命中清单（不许静默）。
改动内容 = role_map 实名 + 3.1.4 的「等 FFN 预算」位点写死 + 3.1.1/3.1.2 边界、
3.2.3 的 vocab 事实修正、3.1.4 的 WP1 登记。历史过程记 ticket 64，不写进事实文档。
"""
import sys

EDITS = []


def edit(path, old, new, tag):
    EDITS.append({"path": path, "old": old, "new": new, "tag": tag})


BASE = "axes/3-width"

# ---------------------------------------------------------------- role_map 实名（8/8）
ROLE_OLD = """role_map:
  owner: "轴 3 宽度轴 lead（待指派）"
  writer: "{writer}"
  reviewer: "fresh 复核者（reviewer 池，不得为本项 writer）"
  integrator: "研究树 integrator"
"""
ROLE_NEW = """role_map:
  owner: "axis3-lead（轴 3 宽度轴 lead）"
  writer: "{writer}"
  reviewer: "axis3-reviewer（fresh 独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后）"
  integrator: "tree-integrator（研究树 integrator；本轴合并与跨轴一致性核对由 axis3-lead 代行）"
"""
WRITERS = {
    "3.1.1": "axis3-w3.1.1（3.1.1 worker，单一写入者）",
    "3.1.2": "axis3-w3.1.2（3.1.2 worker，单一写入者）",
    "3.1.3": "axis3-w3.1.3（3.1.3 worker，单一写入者）",
    "3.1.4": "axis3-w3.1.4（3.1.4 worker，单一写入者；本轮 WP1 的 writer = axis3-lead）",
    "3.1.5": "axis3-w3.1.5（3.1.5 worker，单一写入者）",
    "3.2.1": "axis3-w3.2.1（3.2.1 worker，单一写入者）",
    "3.2.2": "axis3-w3.2.2（3.2.2 worker，单一写入者）",
    "3.2.3": "axis3-w3.2.3（3.2.3 worker，单一写入者）",
}
DIRS = {
    "3.1.1": "3.1.1-uniform-narrowing", "3.1.2": "3.1.2-layer-width",
    "3.1.3": "3.1.3-position-width", "3.1.4": "3.1.4-ffn-ratio",
    "3.1.5": "3.1.5-channel-sparsity", "3.2.1": "3.2.1-readout-reparam",
    "3.2.2": "3.2.2-per-exit-readout", "3.2.3": "3.2.3-tied-untied",
}
for rid, w in WRITERS.items():
    edit(f"{BASE}/{DIRS[rid]}/project.yaml",
         ROLE_OLD.format(writer=w), ROLE_NEW.format(writer=w), f"role_map {rid}")

# ---------------------------------------------------------------- 3.1.1：边界 + 剪枝类等值位点

P = f"{BASE}/3.1.1-uniform-narrowing/project.yaml"
edit(P,
     "method: \"单因素：固定数据/训练预算，只改宽度分配。臂 = {全宽 stock, uniform 减宽（d / FFN 宽按比例缩小）, width dropout（训练期随机保留宽度子集）, 弹性宽度（一次训练多档）}；各减宽臂给等参数 / 等 FLOPs 对照。基底恒为 stock decoder-only。\"",
     "method: \"单因素：固定数据/训练预算，只改宽度分配。臂 = {全宽 stock, uniform 减宽（d_model / 头维按比例缩小）, width dropout（训练期随机保留宽度子集）, 弹性宽度（一次训练多档）}；减宽属**剪枝类**（参数与算力同时下降）⇒ 按 `SCHEMA.md` §15 的例外取**同 FLOPs 的稠密前沿点**作对照（不是等参数量：等参数量在减宽臂上不可得），并报参数量与 FLOPs 两列。基底恒为 stock decoder-only。边界：本项只判 d_model / 头维整体缩小；**FFN 隐藏比 r 及其逐层变化归 `3.1.4`**，两者不得对同一读数各记一遍。\"",
     "3.1.1 method")
edit(P,
     "gap: \"本仓**零本地判决**：没有任何实验测过「uniform 减宽 / width dropout」本身。",
     "gap: \"本仓**零本地判决**：没有任何实验测过「uniform 减宽 / width dropout」本身。边界：**d_model / 头维**整体缩小归本项，**FFN 隐藏比 r**（含其逐层变化）归 `3.1.4`（`d_model` 固定）——两格不得对同一读数各记一遍。",
     "3.1.1 gap 边界")
edit(P,
     "  - \"55-review W-14：旧格引用的「用户本地独立实验（减宽）」无可核实 artifact，已剔除；若确有该实验，需补落盘证据才能引用\"",
     "  - \"55-review W-14：旧格引用的「用户本地独立实验（减宽）」无可核实 artifact，已剔除；若确有该实验，需补落盘证据才能引用\"\n  - \"「同 FLOPs 稠密前沿点」这一对照在 13M 上尚未实测（需先有前沿：`3.1.4` 的 2641 已给出 d_model 固定下的 FFN 预算-质量斜率，可先借它当量纲参照）\"",
     "3.1.1 gaps 追加")

# ---------------------------------------------------------------- 3.1.2：边界（只判 d_model）

P = f"{BASE}/3.1.2-layer-width/project.yaml"
edit(P,
     "gap: \"本仓零经验。与 `2.3.1` 逐层容量变化（深度轴）在实现上重叠",
     "gap: \"本仓零经验。边界：本项判**整体 d_model 的逐层 schedule**（金字塔 / 沙漏）；**FFN 隐藏比 r 的逐层 schedule 归 `3.1.4`**（`d_model` 固定），两格不得对同一读数各记一遍。与 `2.3.1` 逐层容量变化（深度轴）在实现上重叠",
     "3.1.2 gap 边界")
edit(P,
     "  - \"逐层宽度 schedule 与 `2.1.1` 早退的联合（浅层宽度是否影响可截断性）未测\"",
     "  - \"逐层宽度 schedule 与 `2.1.1` 早退的联合（浅层宽度是否影响可截断性）未测\"\n  - \"**实现前置**：引擎 `pathlm/depth_ar.py` 的 `mlp_mult` 是全局标量、`Block` 的 `d_model` 逐层不可配 ⇒ 本项的逐层 d_model schedule 需要引擎支持（最小 patch 描述见 `.scratch/64-axis3-width/TICKET.md` §引擎改动申请）；在引擎收编前本项不跑\"",
     "3.1.2 gaps 实现前置")

# ---------------------------------------------------------------- 3.1.4：把「等 FFN 预算」位点写死 + WP1 登记

P = f"{BASE}/3.1.4-ffn-ratio/project.yaml"
edit(P,
     "method: \"单因素：固定 d_model 与层数，只改 FFN 隐藏比。臂 = {stock r=4, 常数横扫 r∈{1,2,3,6,8}, 层间变化 r_l, GLU 式门控等价}。基底恒为 stock decoder-only。\"",
     "method: \"单因素：固定 d_model 与层数，只改 FFN 隐藏比。臂 = {stock r=4（U4，代价 100% 锚）, 常数横扫 r∈{1,2,3,6,8}, 层间变化 r_l（PYR/APYR）, GLU 式门控等价}。**层间变化臂必须与等宽臂对齐 Σ_l r_l**——FFN 参数与每 token FFN 等效算力都只依赖 Σ_l r_l ⇒ 「Σr 相等」= 「FFN 参数逐位相等 + 每 token FFN 等效算力逐位相等」，否则比的是 FFN 容量而不是分配方式。基底恒为 stock decoder-only。\"",
     "3.1.4 method Σr")
edit(P,
     "objective_criteria: \"该跌：FFN 参数量与每 token 等效 FLOPs；该不跌/可排序：clean bpc；必须不变：d_model 与层数；必须报：每层隐藏比分布（对照列）。\"",
     "objective_criteria: \"该跌：FFN 参数量与每 token 等效 FLOPs（相对 stock r=4）；该不跌/可排序：clean bpc；必须不变：d_model、层数与因果性，且**等预算臂（U2/PYR/APYR）之间 Σ_l r_l、FFN 参数量、每 token FFN 等效算力逐位相等**；必须报：每层 r_l 分布 + 每层计算量分布（对照列，因层间流水节拍由最宽层决定）。\"",
     "3.1.4 objective_criteria")
edit(P,
     "falsification: \"若隐藏比下降带来的质量损失与算力节省相对同预算 stock 不占前沿优势 ⇒ 判该手段在本尺度不产生新知识；若层间隐藏比变化相对常数比无越带差异 ⇒ 判该层间自由度无效。\"",
     "falsification: \"若隐藏比下降带来的质量损失与算力节省相对同预算 stock 不占前沿优势 ⇒ 判该手段在本尺度不产生新知识；**若层间 schedule 与同 Σr 的等宽臂的 clean bpc 差 |Δ| < 0.014（噪声带）⇒ 判「逐层 FFN 分配」在本尺度不是独立于 FFN 总量的设计维度**（该读数并入 `3.1.2` 的深度形状结论，本项不再单独主张）；若 schedule 臂越带更差 ⇒ 判非均匀分配反向有害。\"",
     "3.1.4 falsification")
edit(P,
     "validation: \"端到端：对照 = stock 常数隐藏比（归一为 100%）+ 等参数对照；",
     "validation: \"端到端：对照 = stock 常数隐藏比 r=4（归一为 100%）+ **等 FFN 预算对照（同 Σr 的等宽臂，剪枝类按 `SCHEMA.md` §15 例外取同算力位点）**；",
     "3.1.4 validation")
edit(P,
     "experiments: []",
     "experiments:\n  - \"axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m\"",
     "3.1.4 experiments")
edit(P,
     """  - id: "WP1"
    what: "前馈隐藏比前沿最小判决（含层间 schedule 与 GLU 等价臂）"
    goal: "判定隐藏比是否是有效的算力-质量旋钮、层间变化是否额外有增益"
    means: "13M、固定 d_model/层数，{r=4, 横扫 r, 层间 r_l, GLU 等价}，报 clean bpc + FFN 参数量/FLOPs + 每层隐藏比分布，等参数对照"
    acceptance: "d_model/层数逐位不变；隐藏比 → (算力,质量) 前沿点齐备；层间 schedule 有/无越噪声带差异，二选一明确"
    evidence: "axes/3-width/3.1.4-ffn-ratio/experiments/"
    review: "跑完另派 fresh 复核（`SCHEMA.md` §14/§14b）"
    status: "未开始\"""",
     """  - id: "WP1"
    what: "前馈隐藏比前沿 + **等 FFN 预算**下的逐层非均匀宽度最小判决"
    goal: "判定隐藏比是否是有效的算力-质量旋钮，以及「逐层 FFN 分配」在等 FFN 预算下是否独立有增益"
    means: "13M/dense/12k matched-step，d_model 与层数固定；臂 = {U4 r=4（100% 代价锚）, U2 r=2 等宽, PYR 前宽后窄 [4,3,2,2,2,1,1,1], APYR 镜像}，**U2/PYR/APYR 同 Σr=16**；每层 FFN 用同一次初始化的前缀切片实现（引擎无逐层开关，实现边界见 gaps）；报 clean bpc + FFN 参数/每 token 等效算力 + 每层 r_l 分布"
    acceptance: "d_model/层数/数据/步数/lr/seed 逐位不变；U2/PYR/APYR 的 Σr、FFN 参数、每 token FFN 等效算力逐位相等；layer schedule 相对等宽臂的 Δbpc 越/不越 0.014 二选一明确（n≥2 同号）"
    evidence: "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m"
    review: "跑完另派 fresh 复核（`SCHEMA.md` §14/§14b；复核件落 `.scratch/64-axis3-width/REVIEW-*.md`）"
    status: "进行中\"""",
     "3.1.4 WP1")
edit(P,
     "  - \"与 `3.1.2` 逐层宽度变化的记账重叠须由 integrator 核一遍\"",
     "  - \"与 `3.1.2` 逐层宽度变化的记账重叠须由 integrator 核一遍（边界已写死：本项判 FFN 隐藏比，`3.1.2` 判整体 d_model 的逐层 schedule）\"\n  - \"**实现边界**：引擎 `pathlm/depth_ar.py` 无「逐层 FFN 宽」开关（`mlp_mult` 为全局标量）⇒ 本项 WP1 把逐层 schedule 实现在**实验目录内**（构造后对 FFN 做前缀切片，见 2641/README.md §2.1），不碰共享 `pathlm/**`；把该开关收编进引擎的最小 patch 描述记在 `.scratch/64-axis3-width/TICKET.md`，由 Main 裁决\"\n  - \"13M/12k 快照 + 单任务（enwik8 字符级）；规模与任务外推未做\"",
     "3.1.4 gaps 实现/尺度")
edit(P,
     "status: \"未开始\"",
     "status: \"进行中\"",
     "3.1.4 status")

# ---------------------------------------------------------------- 3.2.3：vocab 事实修正（205→206）

P = f"{BASE}/3.2.3-tied-untied/project.yaml"
edit(P,
     "改动量级：+vocab×d 参数（本仓 vocab=205 / d=256 时 +52480；大词表下不可 scale）",
     "改动量级：+vocab×d 参数（本仓 vocab=206 = 205 个不同字节 + 1 个 [mask] 槽 / d=256 时 +52736；大词表下不可 scale）",
     "3.2.3 vocab innovation")
edit(P,
     "小词表（本仓 vocab=205）与大词表（100k+）",
     "小词表（本仓 vocab=206）与大词表（100k+）",
     "3.2.3 vocab argument")
edit(P,
     "  - \"大词表档未测：本仓 vocab=205，无法判",
     "  - \"大词表档未测：本仓 vocab=206，无法判",
     "3.2.3 vocab gaps")
edit(P,
     "解绑 = vocab×d（本仓小词表 +52480，大词表不可 scale）",
     "解绑 = vocab×d（本仓小词表 +52736，大词表不可 scale）",
     "3.2.3 vocab cost_model")
edit(P,
     "**必须报**：参数增量 = vocab×d（单列）",
     "**必须报**：参数增量 = vocab×d = 206×256 = 52736（单列）",
     "3.2.3 判据参数量")


def main():
    n_hits, misses = 0, []
    for e in EDITS:
        src = open(e["path"], encoding="utf-8").read()
        c = src.count(e["old"])
        if c != 1:
            misses.append((e["tag"], e["path"], c))
            continue
        open(e["path"], "w", encoding="utf-8").write(src.replace(e["old"], e["new"]))
        n_hits += 1
    print(f"命中 {n_hits}/{len(EDITS)} 处替换")
    if misses:
        print("未命中清单（必须为 0）：")
        for tag, path, c in misses:
            print(f"  [{tag}] {path} 命中 {c} 次")
        sys.exit(1)


if __name__ == "__main__":
    main()
