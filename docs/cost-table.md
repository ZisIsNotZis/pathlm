<!-- 本文件由 `tools/build_cost_table.py` 从 `axes/**/project.yaml` 生成，禁止手改；改事实请改 project.yaml -->

# 代价表：定律 × 是否触主干 × 观测代价 × 边界 × 出处

> 本文件由 `tools/build_cost_table.py` 从 `axes/**/project.yaml` 的 `laws` / `cost_rows` 生成，禁止手改；改事实请改 `project.yaml`（契约见 `docs/research-schema.md` §2.1 / §2.2）。
> 定律正文与适用边界只有 `docs/playbook.md` 一个家；本表只把「定律 ↔ 证据 ↔ 代价数字」的对应写出来。
> 每个数字都逐位照抄「出处」那一栏指向的文件，**未重算、未换算口径、未翻转符号**；指标的单位写在「观测代价」里，**方向写在它右侧的「Δ 方向」列**。
> **每条行自带方向**：`delta_direction` 是 `project.yaml` 里的必填字段，本表把它翻成「Δ 方向」列——`正 = 改善` / `正 = 更差`。任何一行单独摘出来都不会被读反；同一张表里两种方向并存，**不许**把列里的符号跨行比较（契约见 `docs/research-schema.md` §2.3）。
> 列义：**改动类型** = 改了什么（人话）；**触主干** = 是否改主干前向 / 残差 / 层间聚合（旁路判断头、读出、训练配方一律为「否」）；**观测代价** = 实测 `delta` + 指标 + 对照；**Δ 方向** = 该行 `delta` 的符号约定（每行自带，见 `docs/research-schema.md` §2.3）；**规模与配方** = 这个数字所处的规模与训练配方；**适用边界** = 它不能被读成什么；**研究项** = 事实源所在的 `project.yaml`。

## 按定律分组的代价事实

### 定律 1 · 元素税分割

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 1」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 1 | 注意力距离惩罚（dist_pen 0.1，全头共用同一斜率） | 是 | `-0.0068` clean bpc Δ（实验 − B0；负 = 更好，但落在 0.014 不可排序带内）；对照 = B0（m1 引擎的同引擎基线） | 正 = 更差 | 13M / m1 引擎 / 12k 步（ticket 07） | 保留 identity、只改注意力权重 ⇒ 近免费；该读数按 55-review P10 判口径不全（丢 per-head slope 与长度外推），只作旧观测，−0.0068 不得称「改善」 | `axes/1-time/1.1.2-soft-decay/experiments/2507-x1-x3-distance-penalty/README.md` | [1.1.2](../axes/1-time/1.1.2-soft-decay/project.yaml) soft 衰减 |
| 1 | 层序 shuffle（训练时按序列打乱层级顺序） | 是 | `+0.545` clean bpc 损伤（相对 B0 1.5074；正 = 更差）；对照 = B0（13M / m1 引擎） | 正 = 更差 | 13M / m1 引擎 / M2 元素税扫描（ticket 06） | 破坏顺序是组合毒药（eval 端 shuffle=1.0 崩到 3.844）；同类里 skip +0.175、redo +0.044 ⇒ 保留或显式标记 identity 的近免费、破坏 identity 或顺序的贵 | `.scratch/06-m2-sweep/evidence/SUMMARY.md` | [2.1.2](../axes/2-depth/2.1.2-skipping/project.yaml) skipping（跳层） |
| 1 | 把未来位置的 token 换成 `[mask]`（显式「我不知道」标记） | 否 | `+0.059` clean bpc 损伤（相对 B0；正 = 更差）；对照 = B0（1.5074） | 正 = 更差 | 13M / m1 引擎 / M2 元素税扫描（I2） | 显式标记最便宜（+0.059），embedding noise ±0.000 免费 ⇒ 保留或显式标记 identity 的近免费；同一扫描里破坏 identity 的 wrong-token 更贵（排序见 `project.yaml` 的 conclusion） | `.scratch/06-m2-sweep/evidence/SUMMARY.md` | [6.4](../axes/6-predict/6.4-training-fill/project.yaml) 训练时未来位置的填充语义：pad / self-fill / 错误 token |

### 定律 2 · TTA 增益 ∝ 训练路径多样性

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 2」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 2 | capped −JS 多样性压力（DIVL1 = p_skip 0.1 + w_diversity 0.1） | 否 | `-0.062` K=8 集成增益 Δbpc（K=8 − K=1；负 = 改善）；对照 = 纯 skip 臂（L1）的 −0.047 | 正 = 更差 | 13M / m1 引擎 / 12k 步 / K∈{1,4,8} | 增益来自练进去的路径多样性；绝对数 1.6227 仍远高于 B0 ⇒ 13M 上税仍占主导，集成不是免费增益来源 | `.scratch/09-m3-interactions/evidence/SUMMARY.md` | [2.1.2](../axes/2-depth/2.1.2-skipping/project.yaml) skipping（跳层） |
| 2 | 确定性路径 checkpoint 上做 K=8 路径集成（X1 距离惩罚 / X3 驱逐+惩罚） | 否 | `0` 集成增益 Δbpc（K=8 − K=1；负 = 改善）；对照 = 同 checkpoint 单路径（K=1） | 正 = 更差 | 13M / m1 引擎 / 60-batch probe 口径（B0 = 1.5074） | 8 条路径 logits 逐位相同 ⇒ 增益恰为 0；只否证确定性路径 checkpoint 的集成，beam search 候选前缀不同属另一命题 | `.scratch/08-tts-probe/evidence/SUMMARY.md` | [2.1.2](../axes/2-depth/2.1.2-skipping/project.yaml) skipping（跳层） |

### 定律 3 · 组合有选择性

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 3」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 3 | `[mask]` 损坏与上下文内修复组合（IX1） | 否 | `+0.087` clean bpc 税（相对 B0 1.5074；正 = 更差）；对照 = 两者单独使用之和（I1 +0.127 + C3 +0.150） | 正 = 更差 | 13M / m1 引擎 / 12k 步 | 共享同一台机器 ⇒ 次可加；破坏对方所需结构的组合则拮抗（IX3 shuffle×retry = 2.2293、INT 全开 = 2.2691） | `.scratch/09-m3-interactions/evidence/SUMMARY.md` | [2.1.2](../axes/2-depth/2.1.2-skipping/project.yaml) skipping（跳层） |

### 定律 4 · 辅助输出是 fallback，不是 voter

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 4」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 4 | 把逐深度输出当 voter 混合（深度集成：等权 vs 置信度加权） | 否 | `-0.005` clean bpc Δ（置信度加权 − 等权；负 = 更好）；对照 = 等权深度集成（1.6070） | 正 = 更差 | 13M / m1 引擎 / 60-batch probe 口径（B0 = 1.5074） | 两臂都差于同 checkpoint 单路径 1.5744 ⇒ 深度输出是 fallback 不是 voter；置信头只排序深度、不排序逐 token 可靠性，加权只买到 0.005 | `.scratch/08-tts-probe/evidence/SUMMARY.md` | [9.1](../axes/9-backlog/9.1-selector-scheduling/project.yaml) 调度 / 选择器（原 G 轴） |

### 定律 5 · 稳定性工程

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 5」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 5 | capped −JS 多样性损失在 bf16 autocast 下直接训练（不加 fp32 与更新前 grad-norm 门控） | 否 | `4600` 发散步（该臂在第几步崩掉；越小 = 越早报废）；对照 = 同配方加 float32 + pre-step grad-norm gate 的 v3（跑完，集成斜率 −0.062） | 正 = 改善 | 13M / DIVL1 配方 / 12k 步预算 | v1 在 step 4600 发散（bf16 + negative-JS）、v2 在 step 2200 静默 NaN（clamp 后梯度爆炸）⇒ 负/奖励类损失在 bf16 下不加门控不可训练 | `.scratch/09-m3-interactions/evidence/SUMMARY.md` | [2.1.2](../axes/2-depth/2.1.2-skipping/project.yaml) skipping（跳层） |

### 定律 6 · 整形 vs 自适应是同一梯度的两面

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 6」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 6 | dense 0.2 剂量整形（用浅层目标塑形主干） | 是 | `+0.0146` retry 门控收益 Δbpc（corrupt 输入、τ=0.98；正 = 门控失效）；对照 = 不整形（probe 头）臂的 −0.0098 | 正 = 更差 | 13M / m1 引擎 / 24000 步 / dense 剂量 0.2（EX2） | 整形换来 exit 轴（depth 6 +0.199）却把 retry 关死 ⇒ 「两全」是同一梯度的两面，只能按 profile 选 | `.scratch/18-exit-probe/evidence/SUMMARY.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |
| 6 | probe 式深度头（主干 stop-grad，不整形） | 否 | `+0.93` 半栈（depth 6）clean bpc 代价（相对全深度；正 = 更差）；对照 = 同 checkpoint 全深度 12 层 | 正 = 更差 | 13M / m1 引擎 / 24000 步 / EX1 协议 | 保 retry 就浅层贵：不塑形主干的必然代价，exit 点存在但质量差（0.5 flop 处） | `.scratch/18-exit-probe/evidence/SUMMARY.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |

### 定律 7 · 主干被改，税就来；旁路判断头可以免费

> 定律正文与适用边界见 `docs/playbook.md` 的「定律 7」；本节只列证据与数字。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 7 | 强加 readout/workspace 边界（读出结构切分） | 是 | `+0.6863` clean bpc 税（相对同引擎 B0；正 = 更差）；对照 = B0（13M，无切分） | 正 = 更差 | 13M / 24000 步（归因臂 B，2.1475） | 切分不但不收税、反向恶化：恢复比例 −50.4%（负 = 恶化），同引擎口径 +0.6662 | `.scratch/25-tax-attribution/evidence/SUMMARY.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |
| 7 | 并行子层（层内两段串行链砍成一段；`parallel-dual-ln` 与基线逐位等参数） | 是 | `+0.0458` clean bpc Δ（实验 − stock 串行；正 = 更差）；对照 = stock 串行子层（sequential） | 正 = 更差 | 13M / L=8 / 12k 步 / 98,304,000 token / seed 0（s1 同向 +0.0422） | 质量代价可排序（3.27× 带），换层内并行度翻倍（每 token 串行步数 16→8），算力只差 0.055% ⇒ 不是免费的，是否采纳看场景 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md` | [2.2.4](../axes/2-depth/2.2.4-parallel-sublayer/project.yaml) 并行子层 |
| 7 | 冻结主干上挂跳跃头（MTP 式旁路草稿头） | 否 | `0.0000` clean bpc Δ（挂头前后，同一冻结主干；正 = 更差）；对照 = 同主干不带草稿头（1.4836） | 正 = 更差 | 13M / 冻结主干 / 纯评测（2610） | 旁路判断头零税：不动主干就不付主干税；冻结断言（grad is None）钉住这条读数 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/analysis.md` | [6.2](../axes/6-predict/6.2-speculative-decoding/project.yaml) 多位置预测 = 投机解码 |

## 定律 8–11 · 用表前必须过的口径与前置条件（不是「代价」）

这四条不是代价行，而是**引用上面任何数字之前必须先过的口径与前置条件**：该不该报「翻转」、判据怎么设计、机制有没有变化空间、算力与内存按什么口径算。正文、适用边界与判定条件都在 `docs/playbook.md`，本节**不复述**，只指回它，并列出这些口径之下已经测到的具体事实。

### 定律 8 · 规模会翻转方向（必须单独列一条）

- 指回：`docs/playbook.md` 的「定律 8」（正文与边界不在本文件重复）。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 8 | 深度自适应 AR 形态（逐层 carry 门 + 全深度监督 + proj-fill 填充） | 是 | `+0.44` 形态税 Δbpc（相对同引擎 dense；正 = 更差）；对照 = 100M 同引擎 dense（24b 等权臂） | 正 = 更差 | 100M / 98.3M token = 1.09 epoch / 24b 等权臂 | 税随规模上升：13M matched +0.31 → 100M +0.44（斜率为正）；与回环族方向相反，不许拿一族的斜率当通用规律 | `axes/2-depth/2.1.1-early-exit/experiments/2524-depth-ar-100m/README.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |
| 8 | 回环 retry / corruption 族（C1 系） | 是 | `+0.122` clean bpc 税（相对 B0 1.5074；正 = 更差）；对照 = B0（13M / m1 引擎） | 正 = 更差 | 13M / 24k 步（1.1-epoch 上界为 +0.364@1k） | 税随规模下降（+0.364@1k → +0.122@24k，其它规模的口径见 2.1.1）；与深度自适应族的上升方向相反 ⇒ 报「规模翻转」前必须先排除口径与刻度差 | `axes/9-backlog/9.2-iteration-refinement/experiments/2511-retry-matrix-withdrawal/README.md` | [9.2](../axes/9-backlog/9.2-iteration-refinement/project.yaml) 迭代精化的其余（原 F 轴剩余） |

### 定律 9 · 判据设计比实验本身更值钱

- 指回：`docs/playbook.md` 的「定律 9」（正文与边界不在本文件重复）。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 9 | 内容 top-k（k=32，按内容选可见键） | 是 | `-0.0166` clean bpc Δ（全因果 − 实验；负 = 更差）；对照 = 全因果（T=256 / 13M） | 正 = 改善 | 13M / T=256 / 12k 步 / 20 批配对 | −0.0166 只是单点越界、且 0.014 带在本项被判「来源自否」、两臂差界比它更宽 ⇒ 只能称「边缘判定」，不作强结论 | `axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/analysis.md` | [1.2.3](../axes/1-time/1.2.3-content-topk/project.yaml) 内容选择 top-k |
| 9 | 用 tf@4 代理口径当部署接受率（深度草稿） | 否 | `2.7` 代理口径高估的百分点（代理 0.969 vs 实测 0.942；越低越好）；对照 = 实测接受率 0.942（同一 checkpoint 真解码） | 正 = 更差 | 13M depth-ar checkpoint / 4 段文本 / 批 1 / 纯推理（2621） | 第 1 个草稿位置 0.970 与代理 0.969 几乎相同，偏差全来自草稿链第 2–4 位 ⇒ 代理只量到链子第一环，不能当整链接受率 | `axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/analysis.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |
| 9 | 在 6 个 latent retry cell 上做格间排序（旧 retry 矩阵） | 否 | `0.011` 格间跨度 bpc（须大于单格换 seed 散布 0.0138 才可排序）；对照 = 单格换 seed 散布 0.0138 | 正 = 更差 | 13M / m1 引擎 / 6 个 cell | 跨度落在种子散布内 ⇒ 全部格间排序撤回（含「soft×mixture 最好」）；判据带必须先于排序，否则排出来的是噪声 | `axes/9-backlog/9.2-iteration-refinement/experiments/2511-retry-matrix-withdrawal/README.md` | [9.2](../axes/9-backlog/9.2-iteration-refinement/project.yaml) 迭代精化的其余（原 F 轴剩余） |

### 定律 10 · 先问「这个机制要有变化空间，设计上必须满足什么」

- 指回：`docs/playbook.md` 的「定律 10」（正文与边界不在本文件重复）。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 10 | 硬窗口 16 + 8 个锚点位置（StreamingLLM 式 sink，只改可见掩码） | 否 | `-0.005583` clean bpc Δ（锚点臂 − 同 seed 纯窗臂；负 = 质量更好）；对照 = 同 seed 纯窗口 16 臂（锚点 0） | 正 = 更差 | 13M / T=256 / enwik8 / 12k 步 / seed 0（2662） | 代价是可见键 16→24、每位置多 98,304 B KV；锚点锁的是「最前面 8 个位置恒可见」，不是把远处信息搬近（窗外桶停在配对地板） | `axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/analysis.md` | [1.1.1](../axes/1-time/1.1.1-hard-window/project.yaml) hard 窗口 |
| 10 | 显式状态通道（逐层独立 8192-float 状态 + 内容写门，γ=1） | 是 | `+0.068228` clean bpc Δ（同字节纯窗口对手 − 本臂；正 = 本臂更好）；对照 = 同字节纯窗口 win32（1.595823） | 正 = 改善 | 13M / T=256 / enwik8 / 12k 步 / seed 0（2639 逐层前沿） | 只在「源信息确实超出可见范围」的任务上可测；真实文本上通道可被忽略（2634 实测收益仅 0.2%）⇒ 读数须先满足必要性条件才可归因到机制 | `axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/analysis.md` | [1.1.4.1](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) 显式记忆槽 / 线性状态 / 递归状态（delta 规则族：DeltaNet / GLA / Mamba-2） |
| 10 | 完备显式机制（逐层状态 + 写门）对同窗口 implicit 臂的对撞 | 是 | `+0.100515` clean bpc Δ（同窗口 implicit 臂 − 完备显式臂；正 = 显式更好）；对照 = 同窗口 implicit 臂 win16_s0（1.628110） | 正 = 改善 | 13M / T=256 / enwik8 / 12k 步 / seed 0 / 6 臂纯推理 | 可检索性腿只答出 10.226%（约盲猜的 21 倍、但只有约十分之一）；无针控制六臂全答不出 ⇒ 「搬回来」是部分而不是全部 | `axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/analysis.md` | [1.1.4.5](../axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml) implicit（掩码/窗口，已证退化）vs **explicit**（有希望的方向） |
| 10 | 线性注意力（归一化线性核 φ=elu+1）替代全因果 softmax | 是 | `-0.214` clean bpc Δ（全因果 − 实验；负 = 更差）；对照 = 全因果 softmax（T=256 / 13M） | 正 = 改善 | 13M / T=256 / 12k 步 / 原实装 | 该臂带伤：归一化分母会触到 clamp 下限（`den_min` 到 1e-06）；稳定化臂把核换成 QK-Norm 后是另一个算子 ⇒ 无伤原核的质量点未取得，只能记「本设计下无信号」 | `axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/analysis.md` | [1.2.2](../axes/1-time/1.2.2-block-sparse/project.yaml) 块稀疏 / 条带 / 局部+全局混合（Longformer 式） |
| 10 | 本格 checkpoint 的深度曲线（3 层 1.8863 vs 8 层 1.8582） | 否 | `0.028` 3 层与 8 层的 clean bpc 差距（越小 = 按需少跑几层的可撬动空间越小）；对照 = 全深度 8 层（1.8582） | 正 = 更差 | 13M / enwik8 / 1 epoch 快照 / 逐位置随机退出配方（2631） | 变化空间近似为 0：逐 token 门控必然被同算力固定截断支配 ⇒ 只能记「本设计下无信号」，不得读成「逐 token 早退机制无效」 | `axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |
| 10 | 残差换成 post-LN（相对 stock pre-LN 单残差） | 是 | `+0.0026` clean bpc Δ（实验 − stock；正 = 更差）；对照 = stock pre-LN 单残差 | 正 = 更差 | 13M / L=8 / 12k 步 / 98.3M token 快照（2652） | \|Δ\| < 0.014 ⇒ 终质维不可排序（不是「无效应」）；可截断性维越带且方向与预登记相反，post-LN 让中间深度读出更差 | `axes/2-depth/2.2.1-residual-form/experiments/2652-residual-form-13m/analysis.md` | [2.2.1](../axes/2-depth/2.2.1-residual-form/project.yaml) 残差的形式 |
| 10 | 层间 causal attention（可学习共享 W_q/W_k/W_v + 温度） | 是 | `-0.0177` clean bpc Δ（对照 − 实验；越带、负 = 更差）；对照 = stock 单残差（1.5254） | 正 = 改善 | 13M / L=8 / 12k 步 / 98.3M token / seed 0（2653） | 额外 196 616 参数、每 token 额外 5 261 312 MACs（+167.3%）换来越带变差；标量温度臂落带内但层 2–3 退化成 one-hot（活性门未过）⇒ 无信号不等于机制无效 | `axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/analysis.md` | [2.2.2](../axes/2-depth/2.2.2-attention-residual/project.yaml) 用 causal attention 代替残差 |
| 10 | 层间学习加权和（DenseFormer 式密集跨层聚合） | 是 | `-0.0008` clean bpc Δ（dense − sequential；负 = 更好）；对照 = stock 单残差（sequential，1.5254 @seed 0） | 正 = 更差 | 13M / L=8 / 12k 步 / 98,304,000 token / seed 0（s1 同号 −0.0015） | \|Δ\| < 0.014 不可排序，且中间深度一致更差（depth 1–7 Δ = +0.206…+0.544）⇒ L=8 太浅、这个机制要缓解的层间瓶颈不存在，属结构条件未满足 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md` | [2.2.3](../axes/2-depth/2.2.3-learned-layer-sum/project.yaml) 层间学习加权和（DenseFormer 式） |
| 10 | 逐层 FFN 隐藏宽前大后小（front-big，等 FFN 预算） | 是 | `-0.0140` clean bpc Δ（uniform − front-big；负 = 前大后小更差）；对照 = stock uniform（1.5291） | 正 = 改善 | 13M / 8 层 / 12k 步 / Σh = 8192 等 FFN 预算 | 等 FFN 预算下非均匀分配无正面增益；−0.0140 名义略超带 1.6×10⁻⁶、按 4dp 恰在带界 ⇒ 判「可排序更差（边界）」 | `axes/2-depth/2.3.1-layer-capacity/experiments/2655-layer-capacity-13m/analysis.md` | [2.3.1](../axes/2-depth/2.3.1-layer-capacity/project.yaml) 逐层容量变化：前大后小 / 逐渐放大 |
| 10 | uniform 减宽（d_model 256→192、头数 8→6、head_dim 恒 32） | 是 | `+0.0527` clean bpc Δ（窄 − 宽；正 = 更差）；对照 = stock d=256（1.5314） | 正 = 更差 | 13M / 8 层 / 12k 步 / 单 seed（2642） | 换来每 token 等效算力 −40.96%、参数 −43.4%（每省 1% 算力付 0.001287 bpc）；与 FFN 比例减半在等算力位点互不支配 | `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/analysis.md` | [3.1.1](../axes/3-width/3.1.1-uniform-narrowing/project.yaml) uniform 减宽 / width dropout |
| 10 | FFN 隐藏比 4→2（Σr 32→16） | 是 | `+0.0395` clean bpc Δ（实验 − stock；正 = 更差）；对照 = stock mlp_mult=4（1.5314） | 正 = 更差 | 13M / 8 层 / 12k 步 / matched-step（2641） | 换每 token 等效算力 −28.4%、参数 −30.2%；层间分配（PYR/APYR）四个 Δ 全在 0.014 带内 ⇒ 总量是旋钮、分配不是 | `axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/analysis.md` | [3.1.4](../axes/3-width/3.1.4-ffn-ratio/project.yaml) 前馈隐藏比：FF/MLP 的 hidden ratio，及其逐层变化 |
| 10 | gated 读出重参数化（旁路读出加非线性，+131k 参数） | 否 | `-0.0812` clean bpc Δ（相对同形态基线；负 = 改善）；对照 = 同形态基线（depth-ar 12k 档，1.8582） | 正 = 更差 | 13M depth-ar / 12k 步 / seed 0（24k 复测见 `experiments/2612-gated-readout-24k-13m`） | 等参数纯线性对照（`experiments/2611-equalparams-linear-control-13m`）的 Δ = −0.0888 与主臂不可排序 ⇒ 增益归读出容量而非非线性；同一读出重参数化在 stock dense 上远小于 0.014（模型主动不用该自由度），不是通用改进 | `axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/analysis.md` | [3.2.1](../axes/3-width/3.2.1-readout-reparam/project.yaml) 读出的重参数化：加非线性 / GLU 式 / 线性 |
| 10 | latent 匹配监督的头（λ=1.0）当草稿再验证（冻结主干、旁路） | 否 | `+0.0916` 草稿接受率 α₁ 配对 Δ（正 = 更好）；对照 = 同参数量 `disc_k2` 头（同 harness 配对基线） | 正 = 改善 | 13M 冻结主干 / 3000 步 Adam / char enwik8 / 种子 0 | 增益来自监督项（把隐状态拉向主干未来真值）而不是头数：k=4 同向 +0.0284；共享 tied 读出下 λ=0 两臂数学等价且逐位相同 ⇒ 独立自由度只是监督目标 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/analysis.md` | [6.3.3](../axes/6-predict/6.3.3-latent-regression/project.yaml) latent regression 批量生成（估计得贪心） |

### 定律 11 · 算力与内存的口径

- 指回：`docs/playbook.md` 的「定律 11」（正文与边界不在本文件重复）。

| 定律 | 改动类型 | 触主干 | 观测代价 | Δ 方向 | 规模与配方 | 适用边界 | 出处 | 研究项 |
|---|---|---|---|---|---|---|---|---|
| 11 | 线性注意力的纸面省算力（T=256 整模型实测） | 是 | `2.58` 整模型一步耗时倍数（相对全因果；>1 = 更慢）；对照 = 全因果 softmax 同批对照 | 正 = 更差 | 13M / T=256 / 12k 步（2616） | 纸面 FLOPs 与推理缓存节省（−93.6%@T=256）不等于实测：一步慢 2.58×、训练峰值显存 1.41× ⇒ 不换核时省幅不兑现 | `axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/analysis.md` | [1.2.2](../axes/1-time/1.2.2-block-sparse/project.yaml) 块稀疏 / 条带 / 局部+全局混合（Longformer 式） |
| 11 | 深度草稿（第 4 层出草稿、第 8 层验证） | 否 | `5.6` 每 token 等效算力超出百分比（相对普通全栈 8.0 层×位置；正 = 更费算力）；对照 = 普通全栈贪心（8.0 层×位置、88.2 tok/s） | 正 = 更差 | 13M depth-ar / 4 段文本 / 批 1（2621） | 墙钟快 1.86× 而算力反而多 5.6% ⇒ 这是启动开销的收益，不得当 FLOPs 收益报，也不得用墙钟比当算力比 | `axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/analysis.md` | [2.1.1](../axes/2-depth/2.1.1-early-exit/project.yaml) early exit（最重要） |
| 11 | KV int4 量化 + 块标尺 | 否 | `+0.022522` clean bpc Δ（实验 − fp16；正 = 更差），且每位置部署字节 3456（省 71.875%）；对照 = fp16 KV（12288 B/位置） | 正 = 更差 | 13M trunk 冻结 / 纯推理 / 3 窗口配对（2632） | 有损、越本格在力带 0.036324 ⇒ 4 位是唯一两腿都掉的档位；逐值标尺的「量化反占内存 ×1.50」是标尺粒度的产物，不是量化的性质 | `axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/analysis.md` | [5.1](../axes/5-quant/5.1-kv-quant/project.yaml) KV 量化 |
| 11 | KV int8 量化 + 块标尺（沿 head_dim 每 32 值一把标尺） | 否 | `+0.000101` clean bpc Δ（实验 − fp16；正 = 更差），且每位置部署字节 6528 = fp16 的 ×0.53；对照 = fp16 KV（12288 B/位置） | 正 = 更差 | 13M trunk 冻结 / 纯推理 / 3 窗口配对 / seed 2–4（2632） | 近无损（远低于本格在力带 0.036324）且省 46.875%；严格前缀不变成立、不需要 staging | `axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/analysis.md` | [5.1](../axes/5-quant/5.1-kv-quant/project.yaml) KV 量化 |
| 11 | 权重 int8 逐通道量化（主干冻结、纯推理） | 否 | `0.000305` max\|配对 Δbpc\|（5 窗口，非负）与每参数部署字节 1.005515 = fp16 的 ×0.5028；对照 = fp16 权重（2.0 B/参数、bpc 1.483569） | 正 = 更差 | 13.5M trunk 冻结 / 纯推理 / CPU / 5 窗口 seed 2–6（2651） | 不越 0.014 带 ⇒ 字节约减半而质量不可分辨；int4（每 32 值一把标尺）Δ 均值 0.013946 贴边 ⇒ 按契约回退 int8 | `axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/analysis.md` | [5.2](../axes/5-quant/5.2-weight-quant/project.yaml) 权重量化 |
| 11 | 只把激活压到 4 位（权重保持 fp16） | 否 | `3.598298` max\|配对 Δbpc\|（5 窗口，非负）与每 token 激活字节 13920 = fp16 的 ×0.2517；对照 = fp16 权重与激活基线（同 harness 5 窗口配对） | 正 = 更差 | 13M trunk 冻结 / 纯推理 / CPU / 5 窗口（2652） | 4 位激活在本规模崩溃：质量与 copy 召回同时掉出带与地板（召回 0.0202）；只压到 8 位的 A8 反而不可分辨（max\|Δ\| 0.003530） | `axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/analysis.md` | [5.3](../axes/5-quant/5.3-activation-quant/project.yaml) 激活量化 |
| 11 | fp8 E4M3 KV（真免标尺，n_scales = 0） | 否 | `0.005363` max\|配对 Δbpc\|（fp8 E4M3 − int8 块标尺；非负）与每位置部署字节 6144 vs int8 的 6528；对照 = int8 + 块标尺 G=32（6528 B/位置） | 正 = 更差 | 13M trunk 冻结 / 纯推理 / CPU / 5 窗口（2653） | 不越 0.014 带 ⇒ 同一「每位置字节」轴上 fp8 E4M3 比 int8 省 384 B（−5.9%）且质量不可分辨；E5M2 越带（max\|Δ\| 0.014496） | `axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m/analysis.md` | [5.5](../axes/5-quant/5.5-fp8-head-dim/project.yaml) fp8 / 大 head_dim |
| 11 | 多 token 预测（γ=1 草稿位、批内验证） | 否 | `1.065` 每 token 等效算力比 R=(1+γ)/(1+τ)（>1 = 比逐 token 前向更费算力）；对照 = 逐 token 前向（R = 1） | 正 = 更差 | 13M B3 / random 前缀 / γ=1 / τ 0.878（2520） | R>1 是 τ ≤ γ 的范式定义性质（代价轴），不是「亏损发现」；两规模 1.065–1.253 / 1.090–1.268 都 >1、不构成规模翻转 | `axes/6-predict/6.2-speculative-decoding/experiments/2520-spec-at-scale/analysis.md` | [6.2](../axes/6-predict/6.2-speculative-decoding/project.yaml) 多位置预测 = 投机解码 |
| 11 | 掩码扩散（每位置独立替换为 `[mask]`、K 步置信度解掩码） | 是 | `+2.6815` teacher-forced bpc Δ（扩散 − 全因果 AR；正 = 更差）；对照 = 全因果 AR（同批位置 teacher-forced bpc 2.2704） | 正 = 更差 | 13M / S=1500 快照 / 模型种子 {0,1} / K=1（2674） | 提高 K 拉不回（K=16 为 +2.6873）；顺序前向确实由 T 降到 K，但质量代价远超 0.014 参考带 ⇒ 代价面收益换不回质量 | `axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m/analysis.md` | [6.3.1](../axes/6-predict/6.3.1-diffusion/project.yaml) diffusion（离散扩散 / 掩码扩散） |
| 11 | 一次性批量 pad 填空（双向可见、只算被填位置，r=0.25） | 是 | `+1.0726` teacher-forced bpc Δ（填空 − 全因果 AR；正 = 更差）；对照 = 全因果 AR（同一批待填位置，2.2704） | 正 = 更差 | 13M / S=1500 快照 / 模型种子 {0,1}（2671） | r 形状不单调（r=0.10 +2.1453、r=0.25 +1.0726、r=0.50 +2.3735）；同层互相可见另有 +0.1635 的独立增益，但两种视图都远差于 AR | `axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/analysis.md` | [6.3.2](../axes/6-predict/6.3.2-pad-fill/project.yaml) 一次性批量 pad 填充（填空式） |

## 覆盖情况（本表生成时统计）

- 代价事实 **41** 条，来自 **27** 个研究项（46 个 `project.yaml` 全量扫描）。
- 已被 `cost_rows` 覆盖的定律：1、2、3、4、5、6、7、8、9、10、11。
- 尚无覆盖的定律：（无）。
