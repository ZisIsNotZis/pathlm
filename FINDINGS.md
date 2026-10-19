# FINDINGS

已确立的结论，每条一行：结论 + 关键数字 + 出处（`文件:行`）+ 状态（成立 / 部分 / 已撤回）。
行号绑定 revision `03d78d4`。数字均为仓内文档已记录的值，本文件未重算、未做新实验。
"成立"只表示"该记录所陈述的实测结论在本仓库口径下成立"，**不含跨尺度外推**——跨口径比较会得出相反结论（`README.md:25-26`）。

## 一、已测出的通用约束（`docs/playbook.md` 的定律，每条带适用边界）

| 结论 | 关键数字 | 出处 | 状态 |
| --- | --- | --- | --- |
| 定律 1：破坏 identity / 顺序结构的机制贵，保留或显式标记 identity 的近免费 | 层序 shuffle `+0.545`、skip `+0.175`、wrong-token `+0.127`、`[mask]` `+0.059`、redo `+0.044`、embedding noise `±0.000` | `docs/playbook.md:9-21` | 成立 |
| 定律 1 附条：距离惩罚的 `−0.007` **只能**称"未观察到超带损伤"，不得称"唯一负税/改善" | `−0.007` 落在 `Δ<0.014` 不可排序带内 | `docs/playbook.md:20` | 部分（带内，不可排序） |
| 定律 1 在深度自适应形态上再次应验（形态税 = 改动主干本体） | 形态税 `+0.44` | `docs/playbook.md:23` | 成立 |
| 定律 2：确定性路径的 checkpoint，K 路径集成**增益恰为 0** | 8 条路径 logits 逐位相同，增益 `0`；练进去的多样性才有效（DIVL1 `−0.062 > −0.047`）；共享主干的多条线集成仅 `+0.0008~+0.0015` | `docs/playbook.md:27-30` | 成立 |
| 定律 2 边界：只否证确定性路径 checkpoint 的 K 路集成，**不**否证 beam search | — | `docs/playbook.md:32-33` | 部分（作用域受限） |
| 定律 3：组合有选择性——共享同一台机器则次可加，破坏对方所需结构则拮抗 | 损坏 × 修复 `+0.087` < 两者之和 `0.277` | `docs/playbook.md:35-41` | 部分（只用于解释已观测的耦合方向） |
| 定律 4：辅助输出是 fallback，不是 voter；深度集成也失败（置信头只排序深度、不排序逐 token 可靠性） | — | `docs/playbook.md:43-46` | 成立（定性，无单点数字） |
| 定律 5：bf16 autocast 下任何 negative / reward 类损失必须 float32 + pre-step grad-norm gate，且门控须在更新前生效 | — | `docs/playbook.md:48-51` | 成立（工程约束） |
| 定律 6：整形 vs 自适应是同一梯度的两面，"两全"被证伪，只能按 profile 选择；剂量响应随规模变化 | dense 0.2 在 13M 温和、@100M corrupt 崩溃 `+1.73` | `docs/playbook.md:53-57` | 成立 |
| 定律 7：主干被改则税来，旁路判断头可以免费（四个独立设计 × 两个规模，同引擎对照） | 回环 retry/corruption 族 `+0.12~0.19`；深度自适应 AR `+0.44 bpc @100M`（96% 干净归属）；读出结构切分 `−50.4%`；不动主干：probe 式早退 `1.5928` vs `1.5871` @100M | `docs/playbook.md:59-71` | 成立 |
| 定律 7 边界：这是**深度自适应形态**的判决，"速度走成熟旁路"是倾向而非普遍禁令 | — | `docs/playbook.md:73-74` | 部分（不作普遍结论） |
| 定律 8：规模会翻转方向——回环族税随规模**下降**，深度自适应族税随规模**上升** | 回环 `+0.122 → +0.082`（渐近口径）；深度自适应 13M `+0.31` → 100M `+0.44`（同引擎） | `docs/playbook.md:78-79` | 成立 |
| 定律 8 边界：报"规模翻转"前必须先排除口径与刻度差；多 token 预测的 FLOPs 口径 `R=(1+γ)/(1+τ)` 在两规模都 >1，**不翻转** | 13M `1.065–1.253`、100M `1.090–1.268` | `docs/playbook.md:81-87` | 成立 |
| 定律 9：判据设计比实验本身值钱——保留类结论必须用 KV 消融保留曲线，不能用窗口外命中率 | 「窗口外 90.3% 命中」的陷阱 | `docs/playbook.md:89-94` | 成立 |
| 定律 10：任何实验契约先问"要让机制产生可测效应，设计上必须满足哪几个条件"；缺条件只能记"本设计下无信号"，不得读成"机制无效" | 例：2634 真实文本收益仅 `0.2%` ⇒ 无必要性损失、无干扰，疑为设计无效；MoD 等解析算力但参数 ×2 ⇒ 归因未隔离；窗口 16 + 源信息在窗外才有意义 | `docs/playbook.md:96-110` | 成立 |
| 定律 11：算力不得用墙钟；多猜几个词只省前向次数，不省算力 | `R=(1+γ)/(1+τ)`：13M `1.065–1.253`、100M `1.090–1.268`，两规模都 >1 | `docs/playbook.md:112-116` | 成立 |
| 定律 11：KV int8 块标尺（沿 `head_dim`，每 32 值共用一把标尺）打败 KV 量化基线 | `6528 B/位置` = fp16 的 `×0.53`（省 `46.9%`）、Δbpc `+0.0001`、严格前缀不变 | `docs/playbook.md:117-118` | 成立 |
| 定律 11：逐值标尺会**反占内存**——那是实现产物，不是量化性质 | `×1.50` | `docs/playbook.md:119` | 成立（对手 fp8 / 更大 `head_dim` 仍未测） |
| 定律 11：纸面 ≠ 实测——不换核时窗口/稀疏的纸面 FLOPs 节省不兑现 | 实测墙钟不省、峰值显存反而更费且随 T 恶化 | `docs/playbook.md:120-121` | 成立 |
| 方差界先于排序：`Δ < 0.014` 即不可排序 | nondeterminism `0.0019`；种子散布 `≤0.0138` | `docs/methodology.md:17-19` | 成立 |
| n=1 结论的采信规则：n=1 正结论默认不可采信，n=1 负结果必须复现后入账 | — | `docs/methodology.md:21-24` | 成立 |

## 二、已下的研究项判决（深度轴）

| 结论 | 关键数字 | 出处 | 状态 |
| --- | --- | --- | --- |
| `2.2.2` attention-residual：把层间线性法换成层间 causal attention 在本尺度**无净增量**，`status: 被否证` | `attn-scalar` clean bpc `1.5336`，Δ vs stock `1.5254` = `−0.0082`（`\|Δ\| < 0.014` 不可排序，且层 2–3 训练后退化成 one-hot，活性门未过）；`attn-qkv` `1.5431`，Δ = `−0.0177`（越带、方向更差）；额外参数 `+8` / `+196 616`，额外 MACs `18 432` / `5 261 312` | `docs/plan.md:13-15`、`axes/2-depth/2.2.2-attention-residual/project.yaml:17`、`:56` | 成立（n=1 seed，不越尺度外推） |
| `2.2.2` 的相对 2.2.3 线性聚合也成立 | 两臂 Δ `−0.0090` / `−0.0185`（对 `dense` = `1.5246`） | `axes/2-depth/2.2.2-attention-residual/project.yaml:17` | 成立 |
| `2.2.3` 层间学习加权和（`dense`）相对 stock 单残差**无可排序差异**，按预登记 falsification 判**无净增量**，`status: 被否证` | 配对 Δ：s0 `−0.0008`、s1 `−0.0015`，两 seed 同号但都未越 `0.014` 带，且小于基线臂自身换 seed 散度 `0.0034`；配对窗口级 SE `0.0031` 大于偏差本身 | `docs/plan.md:17-19`、`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:287-302`、`axes/2-depth/2.2.3-learned-layer-sum/project.yaml:66` | 成立（尺度受限） |
| `2.2.3` 的否证**不是**"DenseFormer 普遍无效" | 边界：13M / L=8 / 12k 步 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:303-304` | 部分（尺度受限的否证） |
| `dense` 的中间深度可读性实测**更差**（新观测） | s0 depth 1–7 Δ = `+0.226 / +0.544 / +0.448 / +0.312 / +0.206 / +0.257 / +0.279`；s1 同向更大 `+0.472 / +0.613 / +0.502 / +0.445 / +0.283 / +0.298 / +0.268`；只有 depth 8 回到 `−0.0008 / −0.0015` | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:297-300` | 成立 |
| `2.2.4` 并行子层：以**可排序的质量代价**换层内并行度翻倍，`status: 已完成`（不是免费的，是否采纳取决场景） | 配对 Δ：s0 `+0.0458 / +0.0402`、s1 `+0.0422 / +0.0374`（两 seed 同号且都越 `0.014` 带）；层内串行链 `2 段 → 1 段`、每 token 串行步数 `16 → 8`、算力只差 `0.055%`（`parallel-dual-ln` 与基线**逐位等参**） | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:314-326`、`axes/2-depth/2.2.4-parallel-sublayer/project.yaml:53` | 成立（n=2 seed，不越尺度外推） |
| M1/M2 的落盘读数与判词一致（`attn-scalar` 1.5336 / `attn-qkv` 1.5431；四臂 seed=1 clean bpc 1.5288 / 1.5710 / 1.5662 / 1.5273，Δ vs `sequential` = `0 / +0.0422 / +0.0374 / −0.0015`） | 见左 | `docs/plan.md:13-19` | 成立 |
| 上述判决都**没有**事后改判据线 | — | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:285`、`docs/plan.md:37` | 成立 |

## 三、负结果（已登记为"这条路在这里输了"）

| 结论 | 关键数字 | 出处 | 状态 |
| --- | --- | --- | --- |
| 层间 causal attention（scalar 与 qkv 两臂）在本尺度无净增量，qkv 那份额外算力不值 | 见上表；qkv 额外 MACs `+167.3%`、Δ `−0.0177` | `axes/2-depth/2.2.2-attention-residual/project.yaml:17` | 成立 |
| 层间密集线性聚合（DenseFormer 形态）在 L=8 的 13M 上无净增量，且中间深度变差 | 见上表 | `axes/2-depth/2.2.3-learned-layer-sum/project.yaml:17`、`:66` | 成立 |
| 确定性路径的多路径集成注定为 0，集成不是免费增益来源 | 8 条路径 logits 逐位相同 | `docs/playbook.md:27` | 成立 |
| 深度自适应形态两全（浅层 exit 质量 + retry 精炼）被证伪；主干一动就有税 | 形态税 `+0.44 bpc @100M`（96% 干净归属） | `docs/playbook.md:53-57`、`:67` | 成立 |
| 纸面省算力不兑现：不换核的窗口/稀疏不省墙钟，峰值显存反而更费 | — | `docs/playbook.md:120-121` | 成立 |
| "构造不好"会造出假失败（needle v1 信号饥饿，全桶 `0.0`） | 全桶 `0.0` | `docs/playbook.md:93` | 成立 |
| 设计未满足必要性条件时收益可忽略（通道可被忽略 ⇒ 疑为设计无效） | 2634 真实文本收益仅 `0.2%` | `docs/playbook.md:107` | 部分（"疑为"，仓内未定论） |

## 四、撤回记录

| 被撤回的说法 | 原因 | 出处 | 状态 |
| --- | --- | --- | --- |
| 整个 **retry 矩阵的格间排序结论** | 6 个 cell 跨度 `0.011` < 种子散布 `0.0138` ⇒ 排序无效（撤回型） | `docs/methodology.md:20`、`:91`、`CONTRIBUTING.md:30` | 已撤回 |
| "13M prob0 生成段反向" | 系**单 prompt 假象**，n=6 复刻未复现 | `docs/methodology.md:21-22` | 已撤回 |
| KV 量化的**逐值标尺口径**及其派生数字（含"量化省一半 ⇒ 内存负收益"，以及"同 −75% 档 GQA 便宜 3.2×"） | 逐值标尺把量化算成内存负收益；本项结论统一改用块标尺（G=32），旧口径与数字一律不迁入 | `axes/5-quant/5.1-kv-quant/project.yaml:17-18` | 已撤回 |
| 旧 `INDEX.md` / `catalog.yaml` 作为状态入口 | 已退役，改由 `python3 tools/summary.py` 生成全表 | `README.md:28-30` | 已撤回（文档退役） |

## 五、敞开的缺口（仓内自述，未填）

| 结论 | 关键数字 | 出处 | 状态 |
| --- | --- | --- | --- |
| `2.2.3` milestone 要求的"更大规模点"未跑 | 最小可判定规模约 4× 参数（~55M）；层间加权和成本 ≈ O(L²·d) | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:330-331` | 部分（未做） |
| `dense` 的 α 无单纯形约束，与 DenseFormer 严格凸组合语义的对照臂未做 | 成本 = 1 臂 9830 万 token | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:332-333` | 部分（未做） |
| 中间深度可读性与 `2.2.1` 残差形式 / `2.1.1` 早退的耦合未联合检验 | — | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:334-335` | 部分（未做） |
| 两份 s0 等值门只有一行日志存证，未落成结构化字段 | 需 `4 × 9830 万 token` 一次全跑 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:336-338` | 部分（未做） |
| 长度 / 层数外推未测（并行子层收益按层内串行链线性累加，本尺度只到 8 层） | L=8 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:339` | 部分（未做） |
