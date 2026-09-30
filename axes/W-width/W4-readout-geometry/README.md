---
id: W4-readout-geometry
axis: W
grid_dir: W-width/W4-readout-geometry
title: "读出几何"
status: partial
judge: "clean bpc（税）+ workspace 探针（null 能量占比 / functional_dirs）+ 校准（ECE）+ 传输与提交机制是否仍可用"
cost: "13M 单臂 ~15–25 min"
tax: "tied 线性 = 基底；切分式紧凑读出 = 负结果（见 W1）"
benefit: "同空间几何使 skip=恒等无需 remap、软提交 Σp·E 有凸包安全、prob0 可从同一读出取"
verdict: "tied 线性是「同空间」前提的地基；它同时是宽度约束的来源（d − 有效读出秩）。"
deps: ["W1-active-width", "A9-old-info-retention", "H4-head-trunk-relation"]
evidence: [".scratch/23-depth-ar/evidence @1586745", ".scratch/25-tax-attribution/evidence @2e86924"]
---

<!-- AUTO:begin -->

# W4-readout-geometry · 读出几何

> 轴：**W 宽度** ｜ 状态：**◐ partial**

> **判读**：tied 线性是「同空间」前提的地基；它同时是宽度约束的来源（d − 有效读出秩）。

## 1. 命题

状态到 token 概率的映射——决定「计算空间」与「读出可见空间」是否耦合。

## 2. 选项枚举（本格全部可能取值）

1. `tied 线性（U = Eᵀ，本项目基底）`
2. `解绑线性（独立 unembed）`
3. `gated 残差读出（h·Eᵀ + SiLU-gate(h)·W，保线性桥）`
4. `纯 MLP 读出`
5. `低秩读出`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：同空间几何使 skip=恒等无需 remap、软提交 Σp·E 有凸包安全、prob0 可从同一读出取

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：tied 线性 = 基底；切分式紧凑读出 = 负结果（见 W1）
- **收益**：同空间几何使 skip=恒等无需 remap、软提交 Σp·E 有凸包安全、prob0 可从同一读出取
- **判读**：tied 线性是「同空间」前提的地基；它同时是宽度约束的来源（d − 有效读出秩）。

## 5. 失败路径与原因

1. 宽度约束是硬的：d=256 / V=205 时 null 空间仅 62 维、功能性读取 26 条；d=128 < V 时结构性为 0（无余量）
2. 结构性切分（只读前 r 维）→ 能量迁出读出子空间（W1）

## 6. 判据与判决实验设计

- **判据**：clean bpc（税）+ workspace 探针（null 能量占比 / functional_dirs）+ 校准（ECE）+ 传输与提交机制是否仍可用

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：clean bpc（税）+ workspace 探针（null 能量占比 / functional_dirs）+ 校准（ECE）+ 传输与提交机制是否仍可用
- 成本：13M 单臂 ~15–25 min

## 7. 成本估计

- 13M 单臂 ~15–25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `W1-active-width`
2. `A9-old-info-retention`
3. `H4-head-trunk-relation`

## 9. 复现手段

1. .scratch/23-depth-ar/evidence @1586745
2. .scratch/25-tax-attribution/evidence @2e86924

## 备注

用户新想法所在格：读出不必线性映射到 unembedding 空间；SwiGLU/SiLU +
线性残差可保住 tied 桥（prob0/传输/提交仍走线性路径），只把质量读出解耦。
澄清：soft commit 的凸包安全不依赖读出线性度（Σp·E 恒在凸包内）；
真正会断的是「概率 = 状态在 embedding 基下的坐标」这一语义、
linear transport 的投影语义、以及 tied 表的梯度对称。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
