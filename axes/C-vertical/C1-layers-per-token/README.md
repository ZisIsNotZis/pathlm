---
id: C1-layers-per-token
axis: C
grid_dir: C-vertical/C1-layers-per-token
title: "每个 token 跑几层"
status: todo
judge: "深度-bpc 曲线 + 出口接受率 + 墙钟/FLOPs 加速 + 税"
cost: "13M 单臂 ~5–25 min；100M 单臂 ~32–70 min"
tax: "形态税 +0.6786（13M 首验）→ matched-step 真税 +0.3295 @12k → 100M +0.4362（同引擎干净归属：形态 96% / 引擎 4% / 配方 ≈0）"
benefit: "tf@4 0.9726、tf@6 0.9946；d6−d12 = 0.0003（半栈 ≈ 全栈）；exit-4 3.1× 墙钟、uniform-exit 1.8×；逐 token 变深度 + 不重 prefill 已实现并逐位测试（max|Δlogit| 9.2e-5）"
verdict: "**已有**：逐 token 变深度成立且是本项目最厚的资产；代价是底座质量税。**尚缺**：100M 端到端深度草稿 + 交错口径墙钟 + FLOPs 净账（省延迟不省算力）（待补做）。"
deps: ["C5-refinement-operator", "C6-depth-supervision", "H4-head-trunk-relation", "G1-selector"]
evidence: [".scratch/23-depth-ar/evidence @1586745", ".scratch/24-depth-ar-100m/evidence @7cdcdaa", ".scratch/25-tax-attribution/evidence @2e86924"]
---

<!-- AUTO:begin -->

# C1-layers-per-token · 每个 token 跑几层

> 轴：**C 垂直机制（深度）** ｜ 状态：**⬜ todo**

> **判读**：**已有**：逐 token 变深度成立且是本项目最厚的资产；代价是底座质量税。**尚缺**：100M 端到端深度草稿 + 交错口径墙钟 + FLOPs 净账（省延迟不省算力）（待补做）。

## 1. 命题

算力沿深度按什么粒度分配。

## 2. 选项枚举（本格全部可能取值）

1. `全深度固定`
2. `固定前缀早退`
3. `逐 token 变深度（本项目）`
4. `逐通道变深度`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：tf@4 0.9726、tf@6 0.9946；d6−d12 = 0.0003（半栈 ≈ 全栈）；exit-4 3.1× 墙钟、uniform-exit 1.8×；逐 token 变深度 + 不重 prefill 已实现并逐位测试（max|Δlogit| 9.2e-5）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：形态税 +0.6786（13M 首验）→ matched-step 真税 +0.3295 @12k → 100M +0.4362（同引擎干净归属：形态 96% / 引擎 4% / 配方 ≈0）
- **收益**：tf@4 0.9726、tf@6 0.9946；d6−d12 = 0.0003（半栈 ≈ 全栈）；exit-4 3.1× 墙钟、uniform-exit 1.8×；逐 token 变深度 + 不重 prefill 已实现并逐位测试（max|Δlogit| 9.2e-5）
- **判读**：**已有**：逐 token 变深度成立且是本项目最厚的资产；代价是底座质量税。**尚缺**：100M 端到端深度草稿 + 交错口径墙钟 + FLOPs 净账（省延迟不省算力）（待补做）。

## 5. 失败路径与原因

1. **端到端实测（ticket 38-W2，真 checkpoint、800/800 token 与贪心逐位相同）**：**深度草稿台提速是真的，但省的是延迟不是算力**——同缓存机械下普通全栈 88.21 tok/s → 深度草稿 **163.83 tok/s（1.86×）**；对仓库原解码器路径 43.77 → **3.74×**（其中 2.02× 属实现层差异，已用 `plain_cache_full` 臂剥离）。**接受率代理是准的但它只量链子第一环**：实测汇总 0.9423 vs TF 代理 0.9694（高估 2.7pp），而**第 1 位真实 0.9704 ≈ 代理 0.9694**，偏差全在链的第 2/3/4 位（0.953/0.935/0.911）。**分级验证（d4→d6→全栈）不值得**：发出流与单级逐 token 相同、只省 0.9% 算力、墙钟反而慢 13%。**双口径分歧**：等效 layer-position/token 8.45 vs 8.00（+5.6%）⇒ **墙钟正收益、FLOPs 负收益**；真 FLOP 杠杆是「直接退 4 层」（4.00/token 半价），用 2.11× FLOPs 把质量买回全栈
2. 浅层监督加权 sweep 全败且单调反向（w 越低 depth-L 越差 2.45→2.73）→「监督稀释」不是主因，浅层全深度监督是表示正则
3. 结构切分读出（readout 360）恶化 −50.4%：模型把状态能量整体迁出读出子空间
4. 2× 训练 + 冷却退火：名义恢复 +16.4%，拆解后形态特异收回 ≈0（与 dense 侧训练收益逐位镜像）
5. dense 全深度监督（LayerSkip 式）在 100M 上剂量响应恶化（corrupt 崩溃 +1.73）

## 6. 判据与判决实验设计

- **判据**：深度-bpc 曲线 + 出口接受率 + 墙钟/FLOPs 加速 + 税

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：深度-bpc 曲线 + 出口接受率 + 墙钟/FLOPs 加速 + 税
- 成本：13M 单臂 ~5–25 min；100M 单臂 ~32–70 min

## 7. 成本估计

- 13M 单臂 ~5–25 min；100M 单臂 ~32–70 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `C5-refinement-operator`
2. `C6-depth-supervision`
3. `H4-head-trunk-relation`
4. `G1-selector`

## 9. 复现手段

1. .scratch/23-depth-ar/evidence @1586745
2. .scratch/24-depth-ar-100m/evidence @7cdcdaa
3. .scratch/25-tax-attribution/evidence @2e86924

## 备注

价格是唯一未解决问题。剩余候选：读出解耦（W4）、proj-fill @100M（P3）、
规模（Q8 剩的唯一候选，但同引擎斜率是正的）。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
