---
id: H1-outputs-per-pass
axis: H
grid_dir: H-horizon/H1-outputs-per-pass
title: "每步输出位置数"
status: judged
judge: "税 + 草稿接受率 + 端到端 tok/s（forwards/FLOPs/墙钟三口径）+ 逐位一致（验证正确性）"
cost: "13M 单臂 ~25–30 min；100M ~32 min"
tax: "n_mtp 1→2 +0.041（B2 1.5481）；2→3 +0.0166（B3 1.5604 vs B2fresh 1.5438，递减）"
benefit: "Medusa 批量验证 1.68–1.90× tok/s @13M；@100M 接受级联稳定（a₂ 0.80、a₃|a₂ 0.767↑、2.4 tok/fwd）"
verdict: "全项目税效比最好的机制；但墙钟收益是 13M/launch-bound 现象。"
deps: ["H3-composition", "H5-verification", "C1-layers-per-token"]
evidence: [".scratch/10-gap-sweep/evidence/B2 @57d64a7", ".scratch/13-four-experiments/evidence/mtp3 @9abc181", ".scratch/20-spec-scale/evidence @a864bcf"]
---

<!-- AUTO:begin -->

# H1-outputs-per-pass · 每步输出位置数

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**✅ judged**

> **判读**：全项目税效比最好的机制；但墙钟收益是 13M/launch-bound 现象。

## 1. 命题

一次前向为几个未来位置产出预测——决定草稿的宽度。

## 2. 选项枚举（本格全部可能取值）

1. `1（纯 AR）`
2. `k（MTP，本项目 n_mtp 1→3）`
3. `整段（扩散式并行）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：Medusa 批量验证 1.68–1.90× tok/s @13M；@100M 接受级联稳定（a₂ 0.80、a₃|a₂ 0.767↑、2.4 tok/fwd）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：n_mtp 1→2 +0.041（B2 1.5481）；2→3 +0.0166（B3 1.5604 vs B2fresh 1.5438，递减）
- **收益**：Medusa 批量验证 1.68–1.90× tok/s @13M；@100M 接受级联稳定（a₂ 0.80、a₃|a₂ 0.767↑、2.4 tok/fwd）
- **判读**：全项目税效比最好的机制；但墙钟收益是 13M/launch-bound 现象。

## 5. 失败路径与原因

1. @100M 墙钟收益消失（前向 compute-bound）：k=2 仅 1.15–1.29×、k=1 转负——「spec 逐位 == 贪心」是 13M 规模性质
2. fp 平局翻转首现（top-2 gap 0.008，1/400 token）→ 逐位相等不能作为跨规模的正确性判据
3. node-k 概率朴素复合是数学错误（CE 3.96 vs 1.58，见 H3）

## 6. 判据与判决实验设计

- **判据**：税 + 草稿接受率 + 端到端 tok/s（forwards/FLOPs/墙钟三口径）+ 逐位一致（验证正确性）

## 7. 成本估计

- 13M 单臂 ~25–30 min；100M ~32 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `H3-composition`
2. `H5-verification`
3. `C1-layers-per-token`

## 9. 复现手段

1. .scratch/10-gap-sweep/evidence/B2 @57d64a7
2. .scratch/13-four-experiments/evidence/mtp3 @9abc181
3. .scratch/20-spec-scale/evidence @a864bcf

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
