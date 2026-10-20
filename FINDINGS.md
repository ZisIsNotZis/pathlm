# FINDINGS

已确立的结论，每条一行：**结论 + 出处 + 状态（成立 / 部分 / 已撤回）**。本文件是**路由器**，不持有数字：每个数字都在「出处」指到的那一处——按定律归的代价数字的家是 `docs/cost-table.md`（按定律分节，逐行指向原始落盘），逐项结论与其读数的家是 `axes/**/project.yaml` 与对应实验的 `analysis.md`。改数字改它指向的家，本文件只改指针。
指向字段的引用用键锚点（`project.yaml::conclusion` 一类，不随目标文件增行而漂），叙事行仍用 `文件:行`。
"成立"只表示"该记录所陈述的实测结论在本仓库口径下成立"，**不含跨尺度外推**——跨口径比较会得出相反结论（`README.md` 的「不能只看结论」一条）。

## 一、已测出的通用约束（`docs/playbook.md` 的定律，每条带适用边界）

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| 定律 1：破坏 identity / 顺序结构的机制贵，保留或显式标记 identity 的近免费 | `docs/playbook.md:9-21`、`docs/cost-table.md`（「定律 1」节） | 成立 |
| 定律 1 附条：距离惩罚的 `−0.007` **只能**称"未观察到超带损伤"，不得称"唯一负税/改善" | `docs/playbook.md:20`、`docs/cost-table.md`（「定律 1」节） | 部分（带内，不可排序） |
| 定律 1 在深度自适应形态上再次应验（形态税 = 改动主干本体） | `docs/playbook.md:23`、`docs/cost-table.md`（「定律 1」节） | 成立 |
| 定律 2：确定性路径的 checkpoint，K 路径集成**增益恰为 0** | `docs/playbook.md:27-30`、`docs/cost-table.md`（「定律 2」节） | 成立 |
| 定律 2 边界：只否证确定性路径 checkpoint 的 K 路集成，**不**否证 beam search | `docs/playbook.md:32-33`、`docs/cost-table.md`（「定律 2」节） | 部分（作用域受限） |
| 定律 3：组合有选择性——共享同一台机器则次可加，破坏对方所需结构则拮抗 | `docs/playbook.md:35-41`、`docs/cost-table.md`（「定律 3」节） | 部分（只用于解释已观测的耦合方向） |
| 定律 4：辅助输出是 fallback，不是 voter；深度集成也失败（置信头只排序深度、不排序逐 token 可靠性） | `docs/playbook.md:43-46`、`docs/cost-table.md`（「定律 4」节） | 成立（定性，无单点数字） |
| 定律 5：bf16 autocast 下任何 negative / reward 类损失必须 float32 + pre-step grad-norm gate，且门控须在更新前生效 | `docs/playbook.md:48-51`、`docs/cost-table.md`（「定律 5」节） | 成立（工程约束） |
| 定律 6：整形 vs 自适应是同一梯度的两面，"两全"被证伪，只能按 profile 选择；剂量响应随规模变化 | `docs/playbook.md:53-57`、`docs/cost-table.md`（「定律 6」节） | 成立 |
| 定律 7：主干被改则税来，旁路判断头可以免费（四个独立设计 × 两个规模，同引擎对照） | `docs/playbook.md:59-71`、`docs/cost-table.md`（「定律 7」节） | 成立 |
| 定律 7 边界：这是**深度自适应形态**的判决，"速度走成熟旁路"是倾向而非普遍禁令 | `docs/playbook.md:73-74`、`docs/cost-table.md`（「定律 7」节） | 部分（不作普遍结论） |
| 定律 8：规模会翻转方向——回环族税随规模**下降**，深度自适应族税随规模**上升** | `docs/playbook.md:78-79`、`.scratch/19-asym-100m/evidence/SUMMARY.md`、`docs/cost-table.md`（「定律 8」节） | 成立 |
| 定律 8 读法约束：回环族 `+0.364@1k → +0.122@24k` 是**同规模内的训练进度**曲线，**不是**规模证据；它只说「1.1-epoch 快照是渐近值的上界、约高 20%」（两条轴不得并列成一条趋势的两个端点） | `docs/playbook.md:81-83`、`.scratch/11-retry-matrix-validity/VERDICTS.md`、`docs/methodology.md:44-45`、`docs/cost-table.md`（「定律 8」节） | 成立（读法约束） |
| 定律 8 边界：报"规模翻转"前必须先排除口径与刻度差；多 token 预测的 FLOPs 口径 `R=(1+γ)/(1+τ)` 在两规模都 >1，**不翻转** | `docs/playbook.md:85-91`、`docs/cost-table.md`（「定律 8」节） | 成立 |
| 定律 9：判据设计比实验本身值钱——保留类结论必须用 KV 消融保留曲线，不能用窗口外命中率 | `docs/playbook.md:93-98`、`docs/cost-table.md`（「定律 9」节） | 成立 |
| 定律 10：任何实验契约先问"要让机制产生可测效应，设计上必须满足哪几个条件"；缺条件只能记"本设计下无信号"，不得读成"机制无效" | `docs/playbook.md:100-114`、`docs/cost-table.md`（「定律 10」节） | 成立 |
| 定律 11：算力不得用墙钟；多猜几个词只省前向次数，不省算力 | `docs/playbook.md:116-120`、`docs/cost-table.md`（「定律 11」节） | 成立 |
| 定律 11：KV int8 块标尺（沿 `head_dim`，每 32 值共用一把标尺）打败 KV 量化基线 | `docs/playbook.md:121-122`、`docs/cost-table.md`（「定律 11」节） | 成立 |
| 定律 11：逐值标尺会**反占内存**——那是实现产物，不是量化性质 | `docs/playbook.md:123`、`docs/cost-table.md`（「定律 11」节） | 成立（对手 fp8 / 更大 `head_dim` 仍未测） |
| 定律 11：纸面 ≠ 实测——不换核时窗口/稀疏的纸面 FLOPs 节省不兑现 | `docs/playbook.md:124-125`、`docs/cost-table.md`（「定律 11」节） | 成立 |
| 方差界先于排序：`Δ < 0.014` 即不可排序 | `docs/methodology.md:17-19` | 成立 |
| n=1 结论的采信规则：n=1 正结论默认不可采信，n=1 负结果必须复现后入账 | `docs/methodology.md:21-24` | 成立 |

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
| 确定性路径的多路径集成注定为 0，集成不是免费增益来源 | `docs/playbook.md:27` | 成立 |
| 深度自适应形态两全（浅层 exit 质量 + retry 精炼）被证伪；主干一动就有税 | `docs/playbook.md:53-57`、`:67` | 成立 |
| 纸面省算力不兑现：不换核的窗口/稀疏不省墙钟，峰值显存反而更费 | `docs/playbook.md:124-125` | 成立 |
| "构造不好"会造出假失败（needle v1 信号饥饿，全桶 `0.0`） | `docs/playbook.md:97` | 成立 |
| 设计未满足必要性条件时收益可忽略（通道可被忽略 ⇒ 疑为设计无效） | `docs/playbook.md:111` | 部分（"疑为"，仓内未定论） |

## 三之二、量化轴与宽度轴的新判决

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `5.5` fp8 / 大 head_dim：在同一「每位置部署字节」轴上，免标尺的 **fp8 E4M3 比 int8+块标尺更省且质量不可分辨**，`head_dim`=32 与 64 两个点上同向复现；`head_dim` 抬高只把标尺摊薄（省幅趋近 50%）、不省值，因此 fp8 相对 int8 的字节优势随 `head_dim` 变大而缩小。`head_dim`=64 的省幅与解析逐位一致（实测 = 解析） | `axes/5-quant/5.5-fp8-head-dim/project.yaml::conclusion`、`::status` | 成立（13M 冻结 trunk / 纯推理 / CPU / 5 窗口配对；不外推 `head_dim`=128 与更长 T） |
| `5.5` 的 `fp8 E5M2` 相对 int8 块标尺**越带更差** ⇒ 「用哪个 fp8 格式」有取舍，不是「fp8 一律免费」 | `axes/5-quant/5.5-fp8-head-dim/project.yaml::conclusion` | 成立（只在 `head_dim`=32 上测过） |
| `3.1.5` 通道稀疏激活：总参数不变、每 token 只算部分 FFN 通道，在本尺度**是真实但很贵的旋钮**——三个稀疏臂都省下实打实的每 token 算力却全部越出噪声带；把「哪些通道」交给逐 token 按内容选比固定砍一半**更差且更费算力**，坍缩门未命中（选择确实逐 token 变化）；保参数换计算与「直接减半 FFN 参数」的同算力位点对比**本项无配对读数、只作参考线、不下判词**。`status: 被否证` | `axes/3-width/3.1.5-channel-sparsity/project.yaml::conclusion`、`::status` | 成立（主判 n=2 同向、`K25` n=1；matched-step 快照；否证只覆盖「无参数、按 `|pre-activation|` 选」这一形态） |

| `3.2.2` 逐出口读出：多出口模型里「每个出口各自映射到词表」**不是早退可用性的杠杆**——共享 tied 表 + 每出口 256×256 变换让末层 clean bpc **更好**（Δ = 逐出口读出 − 共享表 = −0.0857 / −0.0811，两 seed 同向、各自越带），但浅出口的接受率 `tf@2..4` **两 seed 同向下降**（−0.0468/−0.0159、−0.0278/−0.0207、−0.0204/−0.0219），代价是读出参数 +589,824（占总参数 +8.5%）。读的时候必须带一个口径：`tf@k` 是**相对**全深度出口的接受率，给末层加固会机械压低它——绝对质量几乎每层都更好，但相对收益落在末层，早退这笔交易的账变差；它买到的是「多给读出容量」，不是「出口→词表形态」的好处| `axes/3-width/3.2.2-per-exit-readout/project.yaml::conclusion`、`axes/3-width/3.2.2-per-exit-readout/experiments/2645-per-exit-readout-13m/analysis.md` | 成立（n=2；13M / 8L / 9 出口 / V=206；等参数对照臂与 `IND`/`LR` 未跑，真词表不可判） |

## 三之三、时间轴（轴 1）的两条判词

| 结论 | 出处 | 状态 |
| --- | --- | --- |
| `1.2.3` 内容选择 top-k：在**同等每行可见键数**下，「按内容挑」的 clean bpc 可排序优于「按最近位置挑」（k=32 那一对两个 seed 同向、k=64 同向，剂量一致，配对 20/20 同向）⇒ 内容选择是**质量杠杆**；但它必须先算完整 `T×T` 分数矩阵，每位置 KV 字节**结构性省不下来**（内容选择本身不是字节杠杆）。旧读数里的「净亏」应读成「每 token 只看 k 个键」的窗口税，不是「按内容挑」的错 | `axes/1-time/1.2.3-content-topk/project.yaml::conclusion`、`axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m/analysis.md` | 部分（k=32 为 n=2；k=64 仍 n=1；T=256 一点、k∈{32,64} 两点；随机选键的毁源对照、长程召回腿与长 T 未测） |
| `1.2.1` 空洞 / dilated：**同等每行可见键数（32）下把可见位置摊开到更远，在本尺度不换来质量**——「近端 16 + 16 个按行铺满前缀的空洞点」相对「取最近 32 个」的 clean bpc 差 +0.01195 / +0.00696（Δ = 窗口 − 空洞，两 seed 同向但都落 0.014 带内）；可达距离从 31 抬到 255 是结构事实、换不到质量；相对全因果的少看键代价窗口 −0.09462、空洞 −0.08267 ⇒ 只回收 0.0120。长程召回电池四臂全贴 chance（仪器在 13M 上无分辨力），所以这是「本设计下无信号」而不是「机制无效」 | `axes/1-time/1.2.1-dilated/project.yaml::conclusion`、`axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m/analysis.md` | 成立（n=2；13M / T=256 / k=32 一点；长 T、逐层递增 dilation、真稀疏核与召回仪器的分辨力未测） |
| `1.2.2` 局部+全局（Longformer 式）：在**同等 24 个可见键**下，把 8 个键切给「最前 8 个恒可见全局槽」比把 24 个键全给最近位置**更差**（两 seed 同向、各自越带）⇒ 紧预算下「用全局槽换远端可达性」这笔交易在本尺度不划算；同一模式在 200 键时与全因果不可分辨（Δ 落带内）⇒ 是交易不划算，不是掩码坏 | `axes/1-time/1.2.2-block-sparse/project.yaml::conclusion`、`axes/1-time/1.2.2-block-sparse/experiments/2677-local-global-vs-band-13m/analysis.md` | 成立（n=2；13M / T=256 / 24 键；块稀疏（BigBird 式）掩码与率类召回腿未测）；**成立与否不受定序影响，但它的读数入库早于 `tools/prereg_bind.py` 引入 ⇒「契约先于读数」定序不可判**（披露见 `docs/plan.md`） |

## 四、撤回记录

| 被撤回的说法 | 原因 | 出处 | 状态 |
| --- | --- | --- | --- |
| 整个 **retry 矩阵的格间排序结论** | 6 个 cell 跨度 `0.011` < 种子散布 `0.0138` ⇒ 排序无效（撤回型） | `docs/methodology.md:20`、`:91`、`CONTRIBUTING.md#三种可复用的审计模板` | 已撤回 |
| "13M prob0 生成段反向" | 系**单 prompt 假象**，n=6 复刻未复现 | `docs/methodology.md:21-22` | 已撤回 |
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
