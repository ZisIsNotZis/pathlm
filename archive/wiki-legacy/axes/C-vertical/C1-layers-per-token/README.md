---
id: C1-layers-per-token
axis: C
grid_dir: C-vertical/C1-layers-per-token
title: "每个 token 跑几层"
status: todo
judge: "主判据 = 序列级 perplexity/一致率 vs 全深度（单 token 一致率不足：p^N 累积）+ 算力（layer-passes/token）与跳过层数占比 + 税；墙钟不进判据"
cost: "13M 单臂 ~5–25 min；100M 单臂 ~32–70 min"
tax: "13M matched-step 真税 **+0.3291**（本波 2623 复现历史 +0.3295）；**2×2（门×配方）+ 交互重拆**：配方轴**反向**（保权更差 0.1218–0.2610；**同份额控制臂 additive λ=8 与 renorm 打平**（1.8550 vs 1.8575）⇒ 驱动量是份额、不是归一化形式，尺度未单独隔离）、门 +0.0718/+0.2110、**交互 −0.1392（两轴不独立 ⇒ 三项相加只是记账恒等式）**；形态税**按配方分列**（现配方 +0.3291 / λ=0.1 条件值 +0.5901 / 最好形态臂 无门×等权 +0.2573）。**100M 的 +0.4362 归属未重拆**：原记录「形态 96% / 引擎 4% / 配方 ≈0」是**同引擎三列拆混杂**口径，本波只覆盖 13M，**原样保留并标注**"
benefit: "tf@4 0.9726、tf@6 0.9946；d6−d12 = 0.0003（半栈 ≈ 全栈）；exit-4 3.1× 墙钟、uniform-exit 1.8×；逐 token 变深度 + 不重 prefill 已实现并逐位测试（max|Δlogit| 9.2e-5）"
verdict: "**目标口径（本线）**：逐 token 早退必须在**降低算力**的同时保住**序列级**准确率（perplexity / 自回归一致率 vs 全深度）。单 token 一致率不足：一致率 p、长度 N 的整段一致 ≈ p^N（depth-4 p=0.973、N=100 ⇒ ≈6.5%）；序列 90% 一致对应单 token ≈99.9%。**已有**：逐 token 变深度机制成立；引擎已具备逐 token 退出门控（`pathlm/decode.py::_run_stack` 的 `exit_threshold` 读 `sigmoid(model.conf(h))`）、逐 token 深度记录（`depths`）、早退位置 KV 的 fill/proj-fill 填充（`pathlm/depth_ar.py`）。teacher-forcing：100M depth-4 bpc 1.9206 vs 全栈 1.9175（+0.0031）、tf@4 0.9726 / tf@6 0.9946；13M CALM 式门控平均退到 3.43/8 层、一致率 0.959。**尚缺（本线主判据，未测）**：① 逐 token 门控退出的**自回归序列级**结果（算力与跳过占比 + 序列 perplexity/一致率 vs 全深度）从未跑——已有 e2e 只有固定退 4 层、均匀随机退、CALM 平均退三种，均未按逐 token 报序列级口径；② 退出预测头 `conf` 的预测概率与实际「退对了没有」的校准未验（已测 ECE 只是输出头 prob0）；③ 异构深度下深层 token 回看浅层冻结表征的质量代价未测；④ 深层被哪些 token 实际使用的分布未报告。**报告口径**：算力 = layer-passes/token、跳过层数占比；端到端墙钟取决于引擎实现，不进判据。"
deps: ["C5-refinement-operator", "C6-depth-supervision", "H4-head-trunk-relation", "G1-selector"]
evidence: [".scratch/23-depth-ar/evidence @1586745", ".scratch/24-depth-ar-100m/evidence @7cdcdaa", ".scratch/25-tax-attribution/evidence @2e86924", "axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/analysis.md @d77c6b4"]
---

<!-- AUTO:begin -->

# C1-layers-per-token · 每个 token 跑几层

> 轴：**C 垂直机制（深度）** ｜ 状态：**⬜ todo**

> **判读**：**目标口径（本线）**：逐 token 早退必须在**降低算力**的同时保住**序列级**准确率（perplexity / 自回归一致率 vs 全深度）。单 token 一致率不足：一致率 p、长度 N 的整段一致 ≈ p^N（depth-4 p=0.973、N=100 ⇒ ≈6.5%）；序列 90% 一致对应单 token ≈99.9%。**已有**：逐 token 变深度机制成立；引擎已具备逐 token 退出门控（`pathlm/decode.py::_run_stack` 的 `exit_threshold` 读 `sigmoid(model.conf(h))`）、逐 token 深度记录（`depths`）、早退位置 KV 的 fill/proj-fill 填充（`pathlm/depth_ar.py`）。teacher-forcing：100M depth-4 bpc 1.9206 vs 全栈 1.9175（+0.0031）、tf@4 0.9726 / tf@6 0.9946；13M CALM 式门控平均退到 3.43/8 层、一致率 0.959。**尚缺（本线主判据，未测）**：① 逐 token 门控退出的**自回归序列级**结果（算力与跳过占比 + 序列 perplexity/一致率 vs 全深度）从未跑——已有 e2e 只有固定退 4 层、均匀随机退、CALM 平均退三种，均未按逐 token 报序列级口径；② 退出预测头 `conf` 的预测概率与实际「退对了没有」的校准未验（已测 ECE 只是输出头 prob0）；③ 异构深度下深层 token 回看浅层冻结表征的质量代价未测；④ 深层被哪些 token 实际使用的分布未报告。**报告口径**：算力 = layer-passes/token、跳过层数占比；端到端墙钟取决于引擎实现，不进判据。

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

- **税**：13M matched-step 真税 **+0.3291**（本波 2623 复现历史 +0.3295）；**2×2（门×配方）+ 交互重拆**：配方轴**反向**（保权更差 0.1218–0.2610；**同份额控制臂 additive λ=8 与 renorm 打平**（1.8550 vs 1.8575）⇒ 驱动量是份额、不是归一化形式，尺度未单独隔离）、门 +0.0718/+0.2110、**交互 −0.1392（两轴不独立 ⇒ 三项相加只是记账恒等式）**；形态税**按配方分列**（现配方 +0.3291 / λ=0.1 条件值 +0.5901 / 最好形态臂 无门×等权 +0.2573）。**100M 的 +0.4362 归属未重拆**：原记录「形态 96% / 引擎 4% / 配方 ≈0」是**同引擎三列拆混杂**口径，本波只覆盖 13M，**原样保留并标注**
- **收益**：tf@4 0.9726、tf@6 0.9946；d6−d12 = 0.0003（半栈 ≈ 全栈）；exit-4 3.1× 墙钟、uniform-exit 1.8×；逐 token 变深度 + 不重 prefill 已实现并逐位测试（max|Δlogit| 9.2e-5）
- **判读**：**目标口径（本线）**：逐 token 早退必须在**降低算力**的同时保住**序列级**准确率（perplexity / 自回归一致率 vs 全深度）。单 token 一致率不足：一致率 p、长度 N 的整段一致 ≈ p^N（depth-4 p=0.973、N=100 ⇒ ≈6.5%）；序列 90% 一致对应单 token ≈99.9%。**已有**：逐 token 变深度机制成立；引擎已具备逐 token 退出门控（`pathlm/decode.py::_run_stack` 的 `exit_threshold` 读 `sigmoid(model.conf(h))`）、逐 token 深度记录（`depths`）、早退位置 KV 的 fill/proj-fill 填充（`pathlm/depth_ar.py`）。teacher-forcing：100M depth-4 bpc 1.9206 vs 全栈 1.9175（+0.0031）、tf@4 0.9726 / tf@6 0.9946；13M CALM 式门控平均退到 3.43/8 层、一致率 0.959。**尚缺（本线主判据，未测）**：① 逐 token 门控退出的**自回归序列级**结果（算力与跳过占比 + 序列 perplexity/一致率 vs 全深度）从未跑——已有 e2e 只有固定退 4 层、均匀随机退、CALM 平均退三种，均未按逐 token 报序列级口径；② 退出预测头 `conf` 的预测概率与实际「退对了没有」的校准未验（已测 ECE 只是输出头 prob0）；③ 异构深度下深层 token 回看浅层冻结表征的质量代价未测；④ 深层被哪些 token 实际使用的分布未报告。**报告口径**：算力 = layer-passes/token、跳过层数占比；端到端墙钟取决于引擎实现，不进判据。

## 5. 失败路径与原因

1. **端到端实测（ticket 38-W2，真 checkpoint、800/800 token 与贪心逐位相同）**：**深度草稿台提速是真的，但省的是延迟不是算力**——同缓存机械下普通全栈 88.21 tok/s → 深度草稿 **163.83 tok/s（1.86×）**；对仓库原解码器路径 43.77 → **3.74×**（其中 2.02× 属实现层差异，已用 `plain_cache_full` 臂剥离）。**接受率代理是准的但它只量链子第一环**：实测汇总 0.9423 vs TF 代理 0.9694（高估 2.7pp），而**第 1 位真实 0.9704 ≈ 代理 0.9694**，偏差全在链的第 2/3/4 位（0.953/0.935/0.911）。**分级验证（d4→d6→全栈）不值得**：发出流与单级逐 token 相同、只省 0.9% 算力、墙钟反而慢 13%。**双口径分歧**：等效 layer-position/token 8.45 vs 8.00（+5.6%）⇒ **墙钟正收益、FLOPs 负收益**；真 FLOP 杠杆是「直接退 4 层」（4.00/token 半价），用 2.11× FLOPs 把质量买回全栈
2. 浅层监督加权 sweep 全败且单调反向（w 越低 depth-L 越差 2.45→2.73）→「监督稀释」不是主因，浅层全深度监督是表示正则
3. **13M 的 2×2 重拆（2623，n=1）**：把「+0.329 的税」拆成「配方（目标降权）→ 反向」+「形态」是**记账恒等式**；配方轴同时改了主目标份额与全局损失尺度，同份额控制臂（additive λ=8）只排除了份额 ⇒ 尺度未隔离；「浅层监督越重越好、越轻越差」与 23b 的 sweep 同向
4. 结构切分读出（readout 360）恶化 −50.4%：模型把状态能量整体迁出读出子空间
5. 2× 训练 + 冷却退火：名义恢复 +16.4%，拆解后形态特异收回 ≈0（与 dense 侧训练收益逐位镜像）
6. dense 全深度监督（LayerSkip 式）在 100M 上剂量响应恶化（corrupt 崩溃 +1.73）

## 6. 判据与判决实验设计

- **判据**：主判据 = 序列级 perplexity/一致率 vs 全深度（单 token 一致率不足：p^N 累积）+ 算力（layer-passes/token）与跳过层数占比 + 税；墙钟不进判据

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：主判据 = 序列级 perplexity/一致率 vs 全深度（单 token 一致率不足：p^N 累积）+ 算力（layer-passes/token）与跳过层数占比 + 税；墙钟不进判据
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
4. axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/analysis.md @d77c6b4

## 备注

本线要回答的不是「单 token 出口够不够准」，而是「逐 token 早退能否在不掉序列级
perplexity 的前提下省算力」。判决实验（未跑）：逐 token 置信度门控退出、多阈值扫描；
判据 = 自回归序列 perplexity/一致率 vs 全深度、序列一致率 ≥90% 时的 layer-passes/token
与跳过占比；不用墙钟。底座形态税（+0.3291 @13M / +0.4362 @100M）是该形态的性质，
归因已记录，不属本线的未解项。
既有支线「深度草稿 + 全栈验证」（`2621-depth-draft-e2e`）是自投机解码，不是早退：
它保序列准确率（发出流逐 token 相同）但不省算力（8.45 vs 8.00 layer-position/token）。

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
