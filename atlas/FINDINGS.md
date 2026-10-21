# FINDINGS

已确立的结论，每条一行：**结论 + 出处 + 状态（成立 / 部分 / 已撤回）**。本文件是**路由器**，不持有数字：每个数字都在「出处」指到的那一处——按定律归的代价数字的家是 `atlas/cost-table.md`（按定律分节，逐行指向原始落盘），逐项结论与其读数的家是 `axes/**/project.yaml` 与对应实验的 `analysis.md`。改数字改它指向的家，本文件只改指针。
指向字段的引用用键锚点（`project.yaml::conclusion` 一类，不随目标文件增行而漂），叙事行仍用 `文件:行`。
"成立"只表示"该记录所陈述的实测结论在本仓库口径下成立"，**不含跨尺度外推**——跨口径比较会得出相反结论（`README.md` 的「不能只看结论」一条）。

## 一、已测出的通用约束（`atlas/playbook.md` 的定律，每条带适用边界）

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| 定律 1：破坏 identity / 顺序结构的机制贵，保留或显式标记 identity 的近免费 | `atlas/playbook.md:9-21`、`atlas/cost-table.md`（「定律 1」节） | 成立 |
| 定律 1 附条：距离惩罚的 `−0.007` **只能**称"未观察到超带损伤"，不得称"唯一负税/改善" | `atlas/playbook.md:20`、`atlas/cost-table.md`（「定律 1」节） | 部分（带内，不可排序） |
| 定律 1 在深度自适应形态上再次应验（形态税 = 改动主干本体） | `atlas/playbook.md:23`、`atlas/cost-table.md`（「定律 1」节） | 成立 |
| 定律 2：确定性路径的 checkpoint，K 路径集成**增益恰为 0** | `atlas/playbook.md:27-30`、`atlas/cost-table.md`（「定律 2」节） | 成立 |
| 定律 2 边界：只否证确定性路径 checkpoint 的 K 路集成，**不**否证 beam search | `atlas/playbook.md:32-33`、`atlas/cost-table.md`（「定律 2」节） | 部分（作用域受限） |
| 定律 3：组合有选择性——共享同一台机器则次可加，破坏对方所需结构则拮抗 | `atlas/playbook.md:35-41`、`atlas/cost-table.md`（「定律 3」节） | 部分（只用于解释已观测的耦合方向） |
| 定律 4：辅助输出是 fallback，不是 voter；深度集成也失败（置信头只排序深度、不排序逐 token 可靠性） | `atlas/playbook.md:43-46`、`atlas/cost-table.md`（「定律 4」节） | 成立（定性，无单点数字） |
| 定律 5：bf16 autocast 下任何 negative / reward 类损失必须 float32 + pre-step grad-norm gate，且门控须在更新前生效 | `atlas/playbook.md:48-51`、`atlas/cost-table.md`（「定律 5」节） | 成立（工程约束） |
| 定律 6：整形 vs 自适应是同一梯度的两面，"两全"被证伪，只能按 profile 选择；剂量响应随规模变化 | `atlas/playbook.md:53-57`、`atlas/cost-table.md`（「定律 6」节） | 成立 |
| 定律 7：主干被改则税来，旁路判断头可以免费（四个独立设计 × 两个规模，同引擎对照） | `atlas/playbook.md:59-71`、`atlas/cost-table.md`（「定律 7」节） | 成立 |
| 定律 7 边界：这是**深度自适应形态**的判决，"速度走成熟旁路"是倾向而非普遍禁令 | `atlas/playbook.md:73-74`、`atlas/cost-table.md`（「定律 7」节） | 部分（不作普遍结论） |
| 定律 8：规模会翻转方向——回环族税随规模**下降**，深度自适应族税随规模**上升** | `atlas/playbook.md:78-79`、`.scratch/19-asym-100m/evidence/SUMMARY.md`、`atlas/cost-table.md`（「定律 8」节） | 成立 |
| 定律 8 读法约束：回环族 `+0.364@1k → +0.122@24k` 是**同规模内的训练进度**曲线，**不是**规模证据；它只说「1.1-epoch 快照是渐近值的上界、约高 20%」（两条轴不得并列成一条趋势的两个端点） | `atlas/playbook.md:81-83`、`.scratch/11-retry-matrix-validity/VERDICTS.md`、`harness/methodology.md:44-45`、`atlas/cost-table.md`（「定律 8」节） | 成立（读法约束） |
| 定律 8 边界：报"规模翻转"前必须先排除口径与刻度差；多 token 预测的 FLOPs 口径 `R=(1+γ)/(1+τ)` 在两规模都 >1，**不翻转** | `atlas/playbook.md:85-91`、`atlas/cost-table.md`（「定律 8」节） | 成立 |
| 定律 9：判据设计比实验本身值钱——保留类结论必须用 KV 消融保留曲线，不能用窗口外命中率 | `atlas/playbook.md:93-98`、`atlas/cost-table.md`（「定律 9」节） | 成立 |
| 定律 10：任何实验契约先问"要让机制产生可测效应，设计上必须满足哪几个条件"；缺条件只能记"本设计下无信号"，不得读成"机制无效" | `atlas/playbook.md:100-114`、`atlas/cost-table.md`（「定律 10」节） | 成立 |
| 定律 11：算力不得用墙钟；多猜几个词只省前向次数，不省算力 | `atlas/playbook.md:116-120`、`atlas/cost-table.md`（「定律 11」节） | 成立 |
| 定律 11：KV int8 块标尺（沿 `head_dim`，每 32 值共用一把标尺）打败 KV 量化基线 | `atlas/playbook.md:121-122`、`atlas/cost-table.md`（「定律 11」节） | 成立 |
| 定律 11：逐值标尺会**反占内存**——那是实现产物，不是量化性质 | `atlas/playbook.md:123`、`atlas/cost-table.md`（「定律 11」节） | 成立（对手 fp8 / 更大 `head_dim` 仍未测） |
| 定律 11：纸面 ≠ 实测——不换核时窗口/稀疏的纸面 FLOPs 节省不兑现 | `atlas/playbook.md:124-125`、`atlas/cost-table.md`（「定律 11」节） | 成立 |
| 方差界先于排序：`Δ < 0.014` 即不可排序 | `harness/methodology.md:17-19` | 成立 |
| n=1 结论的采信规则：n=1 正结论默认不可采信，n=1 负结果必须复现后入账 | `harness/methodology.md:21-24` | 成立 |

## 二、已下的研究项判决（深度轴）

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `2.2.2` attention-residual：把层间线性法换成层间 causal attention 在本尺度**不值得采纳**——`attn-scalar` 臂落 `0.014` 不可排序带内、且层 2–3 训练后退化成 one-hot（活性门未过），`attn-qkv` 臂**越带且方向更差**（多付的参数与逐 token MACs 不值）⇒ 无净增量 ≠ 机制无效，`status: 被否证` | `axes/2-depth/2.2.2-attention-residual/project.yaml::conclusion`、`::status` | 成立（n=1 seed，不越尺度外推） |
| `2.2.2` 的相对 2.2.3 线性聚合也成立 | `axes/2-depth/2.2.2-attention-residual/project.yaml::conclusion` | 成立 |
| `2.2.3` 层间学习加权和（`dense`）相对 stock 单残差**无可排序差异**，按预登记 falsification 判**无净增量**，`status: 被否证` | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:288-303`、`axes/2-depth/2.2.3-learned-layer-sum/project.yaml::status` | 成立（尺度受限） |
| `2.2.3` 的否证**不是**"DenseFormer 普遍无效" | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:304-305` | 部分（尺度受限的否证） |
| `dense` 的中间深度可读性实测**更差**（新观测） | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:298-301` | 成立 |
| `2.2.4` 并行子层：以**可排序的质量代价**换层内并行度翻倍，`status: 已完成`（不是免费的，是否采纳取决场景） | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:315-327`、`axes/2-depth/2.2.4-parallel-sublayer/project.yaml::status` | 成立（n=2 seed，不越尺度外推） |
| M1/M2 的落盘读数与判词一致（`attn-scalar` 1.5336 / `attn-qkv` 1.5431；四臂 seed=1 clean bpc 1.5288 / 1.5710 / 1.5662 / 1.5273，Δ vs `sequential` = `0 / +0.0422 / +0.0374 / −0.0015`） | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:91-133`、`axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/results.json` | 成立 |
| 上述判决都**没有**事后改判据线 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:286` | 成立 |

## 三、负结果（已登记为"这条路在这里输了"）

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| 层间 causal attention（scalar 与 qkv 两臂）在本尺度无净增量，qkv 那份额外算力不值 | `axes/2-depth/2.2.2-attention-residual/project.yaml::conclusion` | 成立 |
| 层间密集线性聚合（DenseFormer 形态）在 L=8 的 13M 上无净增量，且中间深度变差 | `axes/2-depth/2.2.3-learned-layer-sum/project.yaml::conclusion`、`::status` | 成立 |
| 确定性路径的多路径集成注定为 0，集成不是免费增益来源 | `atlas/playbook.md:27` | 成立 |
| 深度自适应形态两全（浅层 exit 质量 + retry 精炼）被证伪；主干一动就有税 | `atlas/playbook.md:53-57`、`:67` | 成立 |
| 纸面省算力不兑现：不换核的窗口/稀疏不省墙钟，峰值显存反而更费 | `atlas/playbook.md:124-125` | 成立 |
| "构造不好"会造出假失败（needle v1 信号饥饿，全桶 `0.0`） | `atlas/playbook.md:97` | 成立 |
| 设计未满足必要性条件时收益可忽略（通道可被忽略 ⇒ 疑为设计无效） | `atlas/playbook.md:111` | 部分（"疑为"，仓内未定论） |

## 三之二、量化轴与宽度轴的新判决

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `5.5` fp8 / 大 head_dim：在同一「每位置部署字节」轴上，免标尺的 **fp8 E4M3 比 int8+块标尺更省且质量不可分辨**，`head_dim`=32 / 64 / 128 三个点上同向复现；`head_dim` 抬高只把标尺摊薄（省幅趋近 50%）、不省值，因此 fp8 相对 int8 的字节优势随 `head_dim` 变大而对折收敛；到 `head_dim`=128 时它已小到按实现与内核便利选即可，而 int8 把标尺摊到 `G=head_dim` 也没有可分辨的质量代价。三个点的省幅与标尺密度都与解析逐位一致（实测 = 解析） | `axes/5-quant/5.5-fp8-head-dim/project.yaml::conclusion`、`::status` | 成立（13M 冻结 trunk / 纯推理 / CPU / 5 窗口配对；不外推 `head_dim`=256 与更长 T） |
| `5.5` 的 `head_dim` 是「标尺摊薄」的免费杠杆：头维度每翻一倍只多买一次递减的省幅，且质量代价始终在噪声带内 ⇒ 「大 `head_dim` 让 int8 逼近 fp8 且不付质量」 | `axes/5-quant/5.5-fp8-head-dim/project.yaml::conclusion`（WP2 / WP3）、`axes/5-quant/5.5-fp8-head-dim/experiments/2657-fp8-hd128-head-dim/analysis.md` | 成立（`head_dim`=32/64/128 三点；单尺度、T=256） |
| `5.5` 的 `fp8 E5M2` 相对 int8 块标尺**越带更差** ⇒ 「用哪个 fp8 格式」有取舍，不是「fp8 一律免费」 | `axes/5-quant/5.5-fp8-head-dim/project.yaml::conclusion` | 成立（只在 `head_dim`=32 上测过） |
| `3.1.5` 通道稀疏激活：总参数不变、每 token 只算部分 FFN 通道，在本尺度**是真实但很贵的旋钮**——三个稀疏臂都省下实打实的每 token 算力却全部越出噪声带；把「哪些通道」交给逐 token 按内容选比固定砍一半**更差且更费算力**，坍缩门未命中（选择确实逐 token 变化）；保参数换计算与「直接减半 FFN 参数」的同算力位点对比**本项无配对读数、只作参考线、不下判词**。**第二轮（可学习选择器）把「是实现还是机制」分开答了**：给选择器参数（秩 8 低秩打分器）、直通 top-k 梯度与 Switch 式负载均衡项之后，打分器两 seed 都被训练动（相对位移 4.08/3.96、**99.8% 的 (层,位置) 换了通道**），但 Δ(`L50`−`K50`) 只 +0.0087/+0.0028（落 0.014 带内 ⇒ 不可排序），而相对 dense 锚 +0.1068、相对固定前缀 +0.0704/+0.0666（越带 4.8–7.6 倍、两 seed 同向）⇒ **预登记映射实测落「本尺度不可判」（G2 带内，只对两实现之间的排序表态）；「机制不成立」在本尺度成立——这一支是 G3 支配面的映射外补充判读（n=2、两 seed 同向、相对 `F50` 越带 5.0×/4.8×，而 `F50` 同参数更省算力）；「本实现不成立」被否证（不是无参数的毛病）**；验收两读法（C2）**字面成立、实质否证**——`L50` 被同 7 044 608 参数、每 token 更少 MACs（×0.719433 对 ×0.859716）却好 0.0704/0.0666 bpc 的固定前缀臂 `F50` **支配**。`status: 被否证` | `axes/3-width/3.1.5-channel-sparsity/project.yaml::conclusion`、`::status`、`axes/3-width/3.1.5-channel-sparsity/experiments/2687-learned-channel-selector-13m/analysis.md` | 成立（两轮主判 n=2 同向；matched-step 快照；两实现之间那一档不可排序；不外推到更大规模） |

| `3.2.2` 逐出口读出：多出口模型里「每个出口各自映射到词表」**不是相对接受率（`tf@k`）的杠杆，而且绝对层数—质量前沿更好**——共享 tied 表 + 每出口 256×256 变换让末层 clean bpc 更好，且它的 4 层就已优于共享表用满 8 层；浅出口的相对接受率两 seed 同向下降，但 `tf@k` 是**相对**全深度出口的量，给末层加固会机械压低它。它买到的是「多给读出容量」，不是「出口→词表形态」的好处。**这批容量不是「参数总量」买来的**：把同一笔参数（+8.5%）改给主干（每层 FFN 加宽 144）后，末层 clean bpc 在 seed 0 只低 0.0039（落带内）、seed 1 反而高 0.0248，而 `PE` 相对该等参数对照的残余 −0.0818 / −0.1059 两 seed 同向越带 5.8 / 7.6 倍；预登记的四分支因两 seed 不同分支判「不可判」，但「参数解释」这一读法在任何有限样本量下都不成立。代价与全部读数见出处的 `analysis.md` 与 `atlas/cost-table.md` | `axes/3-width/3.2.2-per-exit-readout/project.yaml::conclusion`、`axes/3-width/3.2.2-per-exit-readout/experiments/2645-per-exit-readout-13m/analysis.md`（等参数对照见同文件 §6） | 成立（n=2；13M / 8L / 9 出口 / V=206；`WIDE` 只覆盖「FFN 等宽加宽」一种放法、其跨 seed 散布 0.0245 需更多 seed；`IND`/`LR` 未跑，真词表不可判） |

## 三之三、时间轴（轴 1）的判词

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `1.2.3` 内容选择 top-k：在**同等每行可见键数**下，「按内容挑」的 clean bpc 可排序优于「按最近位置挑」（k=32 那一对两个 seed 同向、k=64 同向，剂量一致，配对 20/20 同向）⇒ 内容选择是**质量杠杆**；但它必须先算完整 `T×T` 分数矩阵，每位置 KV 字节**结构性省不下来**（内容选择本身不是字节杠杆）。旧读数里的「净亏」应读成「每 token 只看 k 个键」的窗口税，不是「按内容挑」的错 | `axes/1-time/1.2.3-content-topk/project.yaml::conclusion`、`axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m/analysis.md` | 部分（k=32 为 n=2；k=64 仍 n=1；T=256 一点、k∈{32,64} 两点；随机选键的毁源对照、长程召回腿与长 T 未测） |
| `1.2.1` 空洞 / dilated：**尺度依赖**的三条。①**T=256 上摊开不换质量（n=2）**——同等每行可见键数（32）下把可见位置摊开到更远，clean bpc 差两 seed 同向但都落 0.014 带内，可达距离抬到 255 换不到质量；**需要远处信息时机制可用（n=2）**——多查询关联回忆上空洞掩码把远处答案搬了回来（远高于 chance），同构造的窗口臂贴 chance（限定：那对臂的差别是「能不能覆盖到」，召回电池在本尺度无分辨力）。②**T=2048 上摊开明确更差（n=2）**——同 token 预算、同参数量下 Δ = −0.02794 / −0.03147（两 seed 同向、各自越负带 2.0 / 2.2 倍），T=1024 那点单 seed 且跨带边界不可判；判据③ 显示远处键被用得更多（D 从 0.10204 涨到 0.31570）⇒ 不是「不需要远处信息」，是「摊开挤掉近端键的代价随 T 变大」。③**逐层递增 dilation 把亏损抵平（n=1）**——深层按 stride `2^ℓ` 摊远、浅层留连续窗口，比每层都摊开便宜 +0.02947（t = 19.33、20/20 批同向），与窗口臂的差缩到 −0.00153（落带内）。**密度补救未答**：远处点 16 → 128 的`2690` 六个臂次里只有两次过稳定性门、且这两次方向相反（+0.02347 / −0.11563）⇒ 不作结论。数字与口径见出处的 `analysis.md` 与 `atlas/cost-table.md` | `axes/1-time/1.2.1-dilated/project.yaml::conclusion`、`axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m/analysis.md`、`axes/1-time/1.2.1-dilated/experiments/2689-longT-layerdil-13m/analysis.md` | ①n=2、②n=2、③n=1（只见一次）；不外推 T>2048 / 53.9M / k≠32 / 真稀疏核；无 T ≥ 1024 的全因果参照臂 |
| `1.2.2` 局部+全局（Longformer 式）：在**同等 24 个可见键**下，把 8 个键切给「最前 8 个恒可见全局槽」比把 24 个键全给最近位置**更差**（两 seed 同向、各自越带）⇒ 紧预算下「用全局槽换远端可达性」这笔交易在本尺度不划算；同一模式在 200 键时与全因果不可分辨（Δ 落带内）⇒ 是交易不划算，不是掩码坏 | `axes/1-time/1.2.2-block-sparse/project.yaml::conclusion`、`axes/1-time/1.2.2-block-sparse/experiments/2677-local-global-vs-band-13m/analysis.md` | 成立（n=2；13M / T=256 / 24 键；块稀疏（BigBird 式）掩码与率类召回腿已由同格后续实验测过，见本表下两行）；**成立与否不受定序影响，但它的读数入库早于 `tools/prereg_bind.py` 引入 ⇒「契约先于读数」定序不可判**（披露见 `docs/plan.md`） |
| `1.2.2` 块粒度（BigBird 式块稀疏的确定性成员）：在**同等 24 个可见键**下，把 8 个键换成「滞后 4 块的一块对齐远块」相对把 24 个键全给最近位置**没有质量增量**（Δ = −0.011552 / −0.010784，两 seed 同向但都落 0.014 带内 ⇒ 不可排序）；评测期反事实（0 训练 token）只给一条**可观察事实**：同一 checkpoint 拆掉远块会掉 0.028888 / 0.027640 bpc（两 seed 同向、越带）；该操作是评测期换掩码的分布外操作，且同一实验里 24 键**严格超集**（可见键一个都没少）反而崩 0.398799 / 0.939319（13.8× / 34.0×）、这一不对称未解释 ⇒ 它同时兼容「远键承载可用信息」与「换掩码的分布外罚项」，**远键是否被用上归因不可判**（与 `2679` 同处理；可区分两者的对照记在 `project.yaml::gaps`）；分布内率类腿（同一 checkpoint、不换掩码，`2685`）在只可能经远块看见的 `far_slot` 桶（n=33）上 `block8l4_s0` 高于地板 +27.273pp（两条 CI 都过）、`block8l4_s1` 只过配对 bootstrap（+21.212pp，`n*`=83）⇒ 远块这条路径在率类上有一条正面读数、但没在两个 seed 上立住，仍记「不可判 + 所需样本量」；成本腿：24 键掩码的每 token 等效注意力 MAC 是全因果的 0.178381 倍、实耗仍是 1.000 倍（稠密 bool 掩码）⇒ 按 `atlas/playbook.md` 定律 10 记为「本设计下无信号」 | `axes/1-time/1.2.2-block-sparse/project.yaml::conclusion`、`axes/1-time/1.2.2-block-sparse/experiments/2683-block-stripe-vs-band-13m/analysis.md`、`axes/1-time/1.2.2-block-sparse/experiments/2683-block-stripe-vs-band-13m/attribution.json::per_seed.block8l4_s0.far_block_off`、`axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m/results.json::recall_leg.per_arm.block8l4_s0` | 质量腿成立（n=2；13M / T=256 / 24 键、块粒度只测 LAG=4 一个设定；判词是「无质量增量」，不得读成「更差」或「块稀疏机制无效」）；**机制层归因不可判**（反事实是分布外操作、超集更差的不对称未解释；分布内率类腿只有一个 seed 立住）；随机块 / 多块 / 跨块条带未测；契约先于读数已机械核过：`prereg_binding.json` 的绑定提交早于读数提交 |
| `1.2.2` 率类（召回腿，`2685`，与 `2662` 同一台针仪器与同一判据、每臂 8192 针）：**率类收益只属于「能看见全长」的那一档**——全因果臂高于同掩码地板 24.341 / 25.903pp，线性核臂也高于地板但只有 8.936 / 9.180pp（约三分之一），而三个「每位置 24 键」的掩码模式在窗外桶（`beyond`）上贴地板（−0.056 ~ +0.574pp，两比例差 CI 全含 0）⇒ 「少看键」买到的是省钱，不是更强的远距可达；线性核的 O(T) 可达性是真的、但弱得多 | `axes/1-time/1.2.2-block-sparse/project.yaml::conclusion`、`axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m/analysis.md`、`axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m/results.json::recall_leg` | 成立（率类，n=2、每臂 8192 针；不借用 0.014 bpc 带；13M / T=256 / 6 字节短语 top-1，不等于生成质量） |
| `1.2.2` 线性核的**无伤规范质量点**（`2685`）：只换注意力算子、同配方同 seed 的配对下，线性核比全因果**多付 0.076059 / 0.071275 bpc**（两 seed 同向、都越 0.014 带 5.1–5.4 倍），而四条健康门全过（跳步 0、`gnorm_max` 9.87 / 10.73、评测前向 `den` 触底行占比 0、φ 零占比 0，引擎默认 φ 求值档同一次前向的 φ 零占比逐层最大 0.2202 / 0.2447）⇒ **税不是数值塌陷造成的假象**；省幅那一面同时只有一半：T=256 上每 token 等效注意力 MAC 是全因果的 0.506 倍（对 T 线性），每位置字节是常数但未实测 | `axes/1-time/1.2.2-block-sparse/project.yaml::conclusion`、`axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m/analysis.md`、`axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m/results.json::main_verdict` | 成立（n=2；13M / T=256 / enwik8 / 12k 步；只到 T=256、不外推到更长 T 或更大规模；每位置缓存字节未实测；旧带伤点 −0.214 / 核已改点 −0.624 只作负观测） |
| `1.1.1` 评测期锚点消融：同一个 checkpoint 只改评测期掩码（0 训练 token）⇒ **锚点通道在推理期真的被模型用起来了**（把它关掉，质量下降，幅度越带），但**这个消融拆不开「推理期可见」与「训练时多见键」两半**——反向对照（纯窗臂评测期白加锚点）直接崩塌，说明评测期换掩码本身就是分布外操作 ⇒ 该 Δ 同时兼容两种解释。原先「收益无法区分锚点槽本身与训练多见键」的缺口降级为更精确的形态，残余是「哪 8 个键」 | `axes/1-time/1.1.1-hard-window/project.yaml::conclusion`、`axes/1-time/1.1.1-hard-window/experiments/2679-anchor-eval-ablation-13m/analysis.md` | 部分（n=2；归因**不可判**；等键数 24 键对照臂未跑） |
| `1.1.1` 锚点召回腿（n=4）：把种子数从 2 补到 4 之后，锚区植入召回**稳定高于同掩码地板**——四个 seed 的差全为正、种子层均值与两个跨 seed 区间（t 区间与两级 bootstrap）的下界均 > 0 ⇒ 按预登记的 n=4 主判**种子层可判**。改判的全部内容是**判定单位**从「逐 seed 合取」换成「种子层均值」；单 seed 的分辨力边界（未配对两比例差半宽远大于该效应，`passes_both` 仍为假）作为**仪器边界**登记，不得读成「无信号」 | `axes/1-time/1.1.1-hard-window/project.yaml::conclusion`、`axes/1-time/1.1.1-hard-window/experiments/2681-anchor-recall-seeds23-13m/analysis.md`、`axes/1-time/1.1.1-hard-window/experiments/2681-anchor-recall-seeds23-13m/results.json::criteria.c3_cross_seed_n4` | 成立（种子层；n=4；13M / T=256 / 锚点臂 k_eff=24；不外推到「每个 seed 单独显著」，也不外推到更大规模/更长 T） |
| `1.1.1` 窗口 k 全曲线与两列口径（本目录自包含复算）：把训练期窗口宽度 k 从 16 推到 192，本目录重训 k∈{32,64,128,192} × seed 0/1 八条臂、联合 k=16 的 n=4 复本，**训练期累计预算列**（窗口税 `tax(k)=bpc(win-k)−bpc(full)`）单调下降（0.124061 → 0.082197 → 0.052907 → 0.011208 → −0.000387 bpc，四对方向全降），但按预登记的三分类，补 seed 之后：k≤64 三档**确定越本格自测带**、**k=128 补到 n=5 也判出 `above_band`**（税 0.012607/0.009808/0.002762/0.005849/0.010376、均值 +0.00828、`sd` 0.003931、区间 [+0.003400, +0.013161] ⇒ 税**确定存在**且**确定小于 0.014 跨格带**，`2688`）、只剩 **k=192 一档不可判**（n=2、`n_req = 32`、按实际区间约 n≈17，本格不取得）；判据⑥ 的 k=64 锚点杠杆补到 n=4 仍不可判（价 −0.007652/−0.008982/−0.003166/**+0.000478**、区间 [−0.011713, +0.002052] 含 0、`n_req = 7`）。两条读数卡住时的约束是**区间宽度**——配对差 `sd` 0.003–0.005、n=3 半宽（0.0126 / 0.0076）均大于效应量本身，而判据⑥ 的规则不含带、判据② 是「区间下界 > 带」带越小越容易判越带 ⇒ **带不是主因**；`2688` 是直接反证：判据线一格未改、只补 2 个 seed，k=128 就从不可判走到 `above_band`；「以配对差散布替代 σ_cell」因此降为口径登记项（框架性取舍），不再是任何一条判据的收口路径；**评测期累计预算列**（只留最近 D）与**单距离列**（删掉距离恰为 d 的键）在同一批权重上分开报、各带口径标注，且「两列不可互换」在 `full` 臂上被量出来：只留最近 32 的损伤 0.211679 是「删掉距离恰为 32/64/128/255 那四个键」之和 0.004720 的 **44.8 倍** ⇒ 单距离列只描述形状、不得求和或外推（`harness/SCHEMA.md` §17.11 记的入账错误在此被量化） | `axes/1-time/1.1.1-hard-window/project.yaml::conclusion`、`axes/1-time/1.1.1-hard-window/experiments/2682-window-kcurve-13m/analysis.md`、`axes/1-time/1.1.1-hard-window/experiments/2682-window-kcurve-13m/results.json::criteria` | 成立（主判：k≤64 三档越带、k=128 补到 n=5 判出 `above_band`（`2688`）、k=192 仍不可判；复现门 9/9（`2682`）与 10/10（`2688`）逐位相同 ⇒ 同一仪器；13M / T=256 / enwik8 / 12k 步；k=16 的 seed 0/1 是跨项复本、seed 2/3 是本项 `2681`；k=128 的 `above_band` 只对「不剔除任何 seed 的 n=5」成立——去掉 seed 1 或 seed 4 会重新跨带；不外推到更大规模、更长 T、别的分词口径）；**代价侧在本实装下为负**（`2636`：同 step 墙钟比 r_time 两轮 1.028/0.992、0.984/0.915、1.076/1.069、1.273/1.784、峰值显存 1.169…3.305 倍，只有解析 FLOPs 省）⇒ 本行判词只覆盖质量价格，「值不值」在算力侧取决于实装 |
| `1.1.4.1` 窗外召回轴铺满整条窗口预算网格：同一台针仪器、同一 generator 序列下，九个纯窗臂（窗口 16→128）与四个 carry 臂在一个**对每个臂都在窗外**的固定子集（距离 > 128 ∩ 十三臂 control 全答不出，n=1206）上比召回 ⇒ **纯注意力把窗口从 16 开到 128 换不回窗外信息**（0.33%~0.91%，不单调、CI 重叠），**只有完备的显式状态把它带回来**（6.4677%），且**召回增益可归因到机制完备度**（单共享槽退化实现只有 0.83%/0.91%） | `axes/1-time/1.1.4.1-explicit-memory/project.yaml::conclusion`、`axes/1-time/1.1.4.1-explicit-memory/experiments/2684-recall-across-budget-points-13m/analysis.md` | 成立（限 enwik8 / T=256 / 13M / 12k 步 / seed 0 这一格；完备臂只有 seed 0；召回口径是 6 字节短语 top-1，不等于生成质量） |
| `1.1.4.1` 量化档叠图 + 同字节格：把 KV 的 int8 + 块标尺档叠到同一条「每位置部署字节」横轴（13 条已训 checkpoint 中的纯窗臂做评测期量化，0 训练 token）⇒ **int8 在纯窗族自己的 checkpoint 上本尺度无质量增量**、纯窗前沿因此整体左移；**把量化后真正同字节的那一格补上（k=60）以后，显式状态点仍不被支配**——同字节最强纯窗挑战者比它差一个可排序带，多给纯窗 6.25% 字节仍追不上；**再给 `CarryChannelBlock` 写出推理期量化前向（`2685`，KV 精度 × 状态精度两轴分开）以后，carry 在自己的 int8 全档（208,896 B）与 int4 全档（110,592 B）上都不被支配**——同字节纯窗点分别差 0.068241 与 0.093485，而 int4 档在纯窗长窗口上越带（k=32/60/64/128 的 +0.030122 / +0.037082 / +0.035901 / +0.019789）⇒「int4 近无损」只在短窗与状态这条路上成立 | `axes/1-time/1.1.4.1-explicit-memory/project.yaml::conclusion`、`axes/1-time/1.1.4.1-explicit-memory/experiments/2680-quant-overlay-frontier-13m/analysis.md`、`axes/1-time/1.1.4.1-explicit-memory/experiments/2681-int8-equalbytes-frontier-13m/analysis.md` | 成立（限这张 47 点网格：9 条纯窗臂 × {fp16,int8,int4} + 4 条 carry 臂 × {fp16, 只KV int8, int8 全档, 只KV int4, int4 全档}；不外推到 T ≫ 256、G≠32 的标尺方案与多 seed——除 `win16` / `carry16` 各两 seed 外均 n=1；carry 的 int4 全档近无损不外推到长窗口，纯窗族在 k ≥ 32 上 int4 越带） |

## 三之四、专家轴（轴 4）的判词

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `4.1` 稀疏激活专家（top-1 路由）：**同每 token 激活算力下，本尺度买不到质量**。13M matched-step（6000 步 / 98.3M token / seed 0 与 1）下，E=4 / E=8 的 top-1 专家臂每 token 专家 MACs 与 stock dense **逐位相等**（比值恰 1.000000，路由税 +0.13% / +0.26%），配对 Δ（对照 − 实验，正 = MoE 更好）= E4 **−0.01057 / −0.00186**（两 seed 都落 0.014 带内）、E8 **−0.02280 / −0.01315**（一越负带、一落带内）；专家数 E4−E8 **−0.01223 / −0.01128**（两 seed 同向但都落带内）；参数限定必须同段读：**13.6M / 42.0M / 79.8M，token/param 7.23 / 2.34 / 1.23** ⇒ 「容量买到了但没喂饱」与「容量根本兑现不了」在本 n 下分不开。路由必须报且**池化会骗人**：跨 12 层池化的熵 0.985–0.999，但**逐层**口径下四条 MoE 臂全部命中坍缩登记（浅层 0–3 最集中，E4-s1 层 1 一个专家吃掉 83%、另一专家只剩 393 个 token）⇒ 判词停在「本设计下无信号」，**不写「MoE 无效」** | `axes/4-moe/4.1-sparse-experts/project.yaml::conclusion`、`axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/analysis.md`、`results.json::main_verdict` | 成立（n=2；13M / 98.3M token / enwik8 字符级 / 无负载均衡的 top-1；E∈{4,8} 是**低端两点**，不外推到更多专家、大 token 预算或更大规模） |
| `4.1` 的归因旁证：**稀疏与路由本身没有可测的独立贡献**——等参数稠密对照（41.97M、`mlp_mult=24`、每 token 4× 激活算力，**不参与主判**）比 `moe-top1-E4` 好 **−0.01806 / −0.01673** bpc（两 seed 同向、各自越负带 1.29× / 1.20×）⇒ 「多给参数但每 token 走狭窄路径」本身不产生收益，不是单纯的欠拟合 | `axes/4-moe/4.1-sparse-experts/project.yaml::conclusion`、`axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/analysis.md`（§4.1 判据③）、`atlas/cost-table.md`（定律 10 节） | 部分（旁证：参数量相等而算力不等，**不能**当等算力对照读；只回答「差异是不是纯容量」） |

## 四、撤回记录

| 被撤回的说法 | 原因 | 出处 | 状态 |
| --- | --- | --- | --- |
| 整个 **retry 矩阵的格间排序结论** | 6 个 cell 跨度 `0.011` < 种子散布 `0.0138` ⇒ 排序无效（撤回型） | `harness/methodology.md:20`、`:91`、`CONTRIBUTING.md#三种可复用的审计模板` | 已撤回 |
| "13M prob0 生成段反向" | 系**单 prompt 假象**，n=6 复刻未复现 | `harness/methodology.md:21-22` | 已撤回 |
| KV 量化的**逐值标尺口径**及其派生数字（含"量化省一半 ⇒ 内存负收益"，以及"同 −75% 档 GQA 便宜 3.2×"） | 逐值标尺把量化算成内存负收益；本项结论统一改用块标尺（G=32），旧口径与数字一律不迁入 | `axes/5-quant/5.1-kv-quant/project.yaml::conclusion`、`::conclusion_review` | 已撤回 |
| 旧 `INDEX.md` / `catalog.yaml` 作为状态入口 | 已退役，改由 `python3 tools/summary.py` 生成全表 | `docs/structure.md` §15 | 已撤回（文档退役） |

## 五、敞开的缺口（仓内自述，未填）

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `2.2.3` milestone 要求的"更大规模点"未跑 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:331-332` | 部分（未做） |
| `dense` 的 α 无单纯形约束，与 DenseFormer 严格凸组合语义的对照臂未做 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:333-334` | 部分（未做） |
| 中间深度可读性与 `2.2.1` 残差形式 / `2.1.1` 早退的耦合未联合检验 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:335-336` | 部分（未做） |
| 两份 s0 等值门只有一行日志存证，未落成结构化字段 | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:337-339` | 部分（未做） |
| 长度 / 层数外推未测（并行子层收益按层内串行链线性累加，本尺度只到 8 层） | `axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:340` | 部分（未做） |
