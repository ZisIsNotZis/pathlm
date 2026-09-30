---
id: F2-return-channel
axis: F
grid_dir: F-iteration/F2-return-channel
title: "回传通道"
status: judged
judge: "每轮 bpc 边际 + 校准（ECE）+ 是否结构性 no-op"
cost: "13M 单臂 ~25 min"
tax: "通道税在 F1 的 C 族内（+0.146…+0.178）"
benefit: "hard 回传 + mixture 投票消除灾难（C5 −1.3pp vs R1 −4.6pp）；mixture 保持 init 质量永久下限（凸包安全）"
verdict: "直接回传是固定点（结构性 no-op）；软回传有 OOD 污染；离散回传必须门控。"
deps: []
evidence: [".scratch/02-engine-m0/evidence/REPORT.md @27418ae", ".scratch/11-retry-matrix-validity/VERDICTS.md @d6a9898", ".scratch/12-overnight/evidence @216a00a", ".scratch/13-four-experiments/evidence @9abc181"]
---

<!-- AUTO:begin -->

# F2-return-channel · 回传通道

> 轴：**F 迭代** ｜ 状态：**✅ judged**

> **判读**：直接回传是固定点（结构性 no-op）；软回传有 OOD 污染；离散回传必须门控。

## 1. 命题

一轮的结果以什么形式回到下一轮——决定重复能否带来新信息。

## 2. 选项枚举（本格全部可能取值）

1. `direct（原样回传）`
2. `linear 投影`
3. `soft 单纯形再量化`
4. `hard 离散重嵌`
5. `部分状态掩码`
6. `verifier 反馈`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：hard 回传 + mixture 投票消除灾难（C5 −1.3pp vs R1 −4.6pp）；mixture 保持 init 质量永久下限（凸包安全）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：通道税在 F1 的 C 族内（+0.146…+0.178）
- **收益**：hard 回传 + mixture 投票消除灾难（C5 −1.3pp vs R1 −4.6pp）；mixture 保持 init 质量永久下限（凸包安全）
- **判读**：直接回传是固定点（结构性 no-op）；软回传有 OOD 污染；离散回传必须门控。

## 5. 失败路径与原因

1. direct 重入 = 固定点（M0 实测，结构性 no-op）——「无传输的重入无价值」是全项目第一条铁律
2. soft 回传 OOD 污染三度复现（见 H2）
3. D1 gauge bug：direct/linear mixture 测到的是 ~30× rescale 而非 latent mean（修复后 r2 单调 2.71→1.936；pre-fix 发散至 95.2）
4. 逐位置状态掩码不省算力（需稀疏 gather/scatter）→ 逐位置门控 latent 版此路不通

## 6. 判据与判决实验设计

- **判据**：每轮 bpc 边际 + 校准（ECE）+ 是否结构性 no-op

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

（无已知依赖）

## 9. 复现手段

1. .scratch/02-engine-m0/evidence/REPORT.md @27418ae
2. .scratch/11-retry-matrix-validity/VERDICTS.md @d6a9898
3. .scratch/12-overnight/evidence @216a00a
4. .scratch/13-four-experiments/evidence @9abc181

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
