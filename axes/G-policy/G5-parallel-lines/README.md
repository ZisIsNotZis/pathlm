---
id: G5-parallel-lines
axis: G
grid_dir: G-policy/G5-parallel-lines
title: "并行多线机制"
status: judged
judge: "质量增益（同 FLOPs）+ 线间相关度 + 每线边际收益曲线"
cost: "13M 纯推理分钟级–单臂 25 min（若需训练多样性）"
tax: "多样性压力 mech 的税 +0.023（capped −JS）"
benefit: "去相关 voter 有效：DIVL1（capped −JS + float32 + pre-step grad-norm gate）同时改善 baseline 与集成斜率（−0.062 > −0.047，上限 +0.023）"
verdict: "共享主干的多线几乎不增益——瓶颈是相关性，不是算力。"
deps: ["G1-selector", "C6-depth-supervision", "F1-iteration-object"]
evidence: [".scratch/08-tts-probe/evidence @aa97bcf", ".scratch/09-m3-interactions/evidence @39b1838", ".scratch/02-engine-m0/evidence/REPORT.md @27418ae"]
---

<!-- AUTO:begin -->

# G5-parallel-lines · 并行多线机制

> 轴：**G 策略与调度** ｜ 状态：**✅ judged**

> **判读**：共享主干的多线几乎不增益——瓶颈是相关性，不是算力。

## 1. 命题

如何同时推进多条概率线并合并——引擎与调度的核心格。

## 2. 选项枚举（本格全部可能取值）

1. `单线（贪心）`
2. `beam search`
3. `多路共享主干`
4. `去相关 voter（显式训练多样性）`
5. `verifier 重排`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：去相关 voter 有效：DIVL1（capped −JS + float32 + pre-step grad-norm gate）同时改善 baseline 与集成斜率（−0.062 > −0.047，上限 +0.023）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：多样性压力 mech 的税 +0.023（capped −JS）
- **收益**：去相关 voter 有效：DIVL1（capped −JS + float32 + pre-step grad-norm gate）同时改善 baseline 与集成斜率（−0.062 > −0.047，上限 +0.023）
- **判读**：共享主干的多线几乎不增益——瓶颈是相关性，不是算力。

## 5. 失败路径与原因

1. 共享 trunk 的 voter 太相关：集成增益仅 +0.0008~+0.0015（可忽略）
2. 确定性路径 checkpoint 的 K-path 集成恰好得 0——beam search 产出的多线正落在这个死区
3. 深度集成失败：置信头只排序深度、不排序逐 token 可靠性

## 6. 判据与判决实验设计

- **判据**：质量增益（同 FLOPs）+ 线间相关度 + 每线边际收益曲线

## 7. 成本估计

- 13M 纯推理分钟级–单臂 25 min（若需训练多样性）

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `G1-selector`
2. `C6-depth-supervision`
3. `F1-iteration-object`

## 9. 复现手段

1. .scratch/08-tts-probe/evidence @aa97bcf
2. .scratch/09-m3-interactions/evidence @39b1838
3. .scratch/02-engine-m0/evidence/REPORT.md @27418ae

## 备注

「多线并行调度」的第一个问题不是引擎复杂度，是去相关从哪来。
本地已有答案方向：去相关来自 transports/depths，而不是 trunk 本身；
且多样性必须练进去。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
