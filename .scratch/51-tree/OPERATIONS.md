# OPERATIONS — 研究树清理与立项决策表（51 格）

- 产出：本文件（`.scratch/51-tree/OPERATIONS.md`），日志文档；本轮**只产出这一个文件**，不改 `catalog.yaml`、不改任何格目录、不改任何数字。
- 依据：主 Agent 政策 A–F（见下发任务）；`AGENTS.md`（含编排模式/事实-日志文档地图）；`SCHEMA.md` §1/§19/§21/§23；起点材料为 `.scratch/50-critical-audit/{A,C,HW,FMG,RP}-AUDIT.md` 五份汇总 + 每格 `axes/**/CRITIQUE.md`（51 份，本表逐格读过）。
- 现状取值：按工作树 `catalog.yaml` 的 `status`（当前只有 `judged`/`todo` 两值；HEAD 里残留的 `reference`/`partial` 已被规范化为 `todo`/`judged`——**本表不复活这两个已废状态**）。

---

## 0. 人话总判

审计没有推翻任何**核心负结论**（形态税量级、窗口丢长程、出口资产换延迟不换算力、量化参照线等），但推翻了大量**收益数字与机械口径**：投机解码的"省算力"符号算反、量化参照线被自造标尺反转、墙钟倍率在噪声里、"累计预算"被单点曲线冒充、跨家族 Pareto 用了已作废旧数、有 7 格（A1/W5/G5/R2/R3/R5/P2）既不是对手也不影响任何判断。清理方向：**修 3 台坏天平（R1 量化 / A9 长程腿 / H1·H3 算力口径）→ 补 4 条必须打败的基线（R1/R7/R4/W2←R6）→ 重判 12 格"方法有错、改后可能改判"的格 → 删 7 格白痴项**。

---

## 0.1 全局裁决（跨格 / 跨汇总的矛盾，先在此定，表内不再重复）

1. **"改 reference"这个选项已不存在。** 政策 A 只有 `judged`/`todo`，政策 F 又禁止以"业界已做透"免做。因此 A-AUDIT/C-AUDIT/HW/FMG/RP 里所有"降为 reference / 改 stub"的建议**一律不成立**：凡审计认为"业界已做透"的格，本表按政策 D 处理为**立项**（业界结论是必须核实并对齐的基线，见 F），或按政策 B **删除**，**不留 reference 空壳**。
2. **W2 ↔ R6（MoE / 专家化）两汇总结论相反，裁决为"并入 R6 到 W2 后立项"。**
   - HW 方（`W2-expertization/CRITIQUE.md` ④⑤）：本格 PRIOR_ART 自判 (a)，13M 上 MoE 收益是规模现象，"不必跑实验"，保留为参照线即可。
   - RP 方（`R6-moe-scale/CRITIQUE.md` ③④）：R6 与 W2 同格；`R/PRIOR_ART.md §4` 自认"R6 在 13M 尺度是否有公开对照——**未找到**"；它决定 W2/W5 的判据线，落到 D2（必须打败的基线必须本地可测）。
   - **裁决：采纳 RP。** 理由：B2 的删除要求"与本项目设定**无实质差异**"——13M char-LM 上 MoE 无公开对照（RP 已核），设定差异实质存在；且它落在"省算力"主线（D3）。HW 的"规模现象"是**外推判断**，不是可核结论。故 R6 并入 W2 并立项（见 §2② P1 的 W2 行）。
3. **W5 ↔ R3（剪枝 vs MoE 分类）两汇总互指对方，裁决为"两格都删、分类事实挂 W2"。**
   - HW 方：删/降 W5（零本地实验、结论定义式重言、判据全未测）。
   - RP 方：删 R3、并入 W5。
   - **裁决：W5、R3 都删**（理由见 §1 W 轴 W5 行与 R 轴 R3 行）；保留的唯一内容"剪枝=减容量、MoE=保容量减计算"作为 **W2 的 industry 一行 note**（保留内容、不留空壳）。依据：两格互指的"对方"本身都在删除判据内，合并后无独立残余；且无任何保留格需要它们当必须打败的基线。
4. **R1 的新口径使 W3/R4/M2 的旧数字作废（仓内自相矛盾，非审计互斥但必须一并处理）。** R1 用标准 head_dim 块标尺后 int4 = 1.25× fp16 内存（负收益）；W3 的"同 −75% 档 GQA 便宜 3.2×"与 R4 的"就用个窗口几乎为零"都建立在旧口径上（`W3/CRITIQUE.md ②`、`R4/CRITIQUE.md ②`）。裁决：W3/R4 随 R1 重算（已列入立项），在 R1 修好前它们的跨家族/累计数字**不得引用**。

---

## 1. 决策表

列的读法：`现状` = 工作树 `catalog.yaml status`；`决策` ∈ {保留 / 删除 / 并入X / 立项}；立项行给"必须回答的问题"与"验证口径（该涨／该跌／必须不变）"；删除行给"命中条款"。

### A 轴（序列混合机制）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| A1-causality | todo | **删除** | 非因果 LM 与本项目 AR 不变量（`docs/design.md §1`）不相容，业界定论，本仓零数据（`catalog.yaml:205 local` 空；A1 CRITIQUE ③-4"可跑实验永远不会改变判断"）；且被写成 `reference` 是标签误用 | **命中 B1（构造前提类）+ B3（标签误用无独立内容）** |
| A2-full-attention | judged | **保留** | A 轴唯一的税零点/锚点，审计明写"不删（A2 是锚点）"（A-AUDIT §1）；结论"基底成立"未被推翻 | —（补外部锚出处；注 B0 含 tied 读出+无 final LN 的底座偏离） |
| A3-window-anchor | judged | **立项** | 头号数字（+0.058~0.121、76.8→93.3%）只存在于 `.scratch/07-x1-x3/evidence/SUMMARY.md`（20 行 markdown，无 results.json/run.py，A3 CRITIQUE ③-2）；"评测期丢前缀"与"累计 keep-32=+0.212"混列；它是"任何压缩/摘要必须打败的价格上界"（`catalog.yaml:268`）⇒ D2+D1 | **问**：同引擎/同数据流/n≥2 自包含复算，并拆"评测期丢前缀"与"累计预算"两列。**该涨**：锚桶检索腿召回（相对同掩码地板）；**该跌**：无；**必须不变**：clean bpc 的累计预算曲线（keep k 全曲线，禁单点外推 §17.11） |
| A4-soft-distance-penalty | judged | **立项** | 固定 w、全头共用丢掉 ALiBi 的 per-head slope ⇒"全头被推向局部、长不出分工"是自设条件下的必然；judge 写"长度泛化"却从未测（A4 CRITIQUE ②③⑤）；benefit 的 93.3% 实为 X3=惩罚+驱逐组合（③-2） | **问**：per-head 可学习 slope + 长度外推（训练短测长）是否成立。**该涨**：训练 512→测 2048+ 的 bpc 相对优势；**该跌**：无；**必须不变**：同分布 bpc 无差异（若 >0.014 先排容量混杂）；**必须报**：per-head 平均注意距离的分布 |
| A5-dilated-window | todo | **并入 A6** | 审计：固定稀疏在 13M/T=256 不可能改变判断，信息已被 A3+A6 覆盖（A-AUDIT §1）；A6 已有内容 top-k，位置稀疏应作"同预算对照"并入（A6 CRITIQUE ⑤） | **并入 A6**（保留"位置稀疏 vs 内容稀疏同预算"选项；顺带修 `catalog.yaml:306` 的坏 flow-seq） |
| A6-mixing-operator | judged | **立项**（并 A5/A7） | 唯一有真实验的算子格，但三重坏证据叠加：长 T 系列（2622–2629）实为 `train_window=T/2` 位置窗（属 A3）；steps 不等 24000/12000/6000（2629 §279 自认两臂均未收敛）；长程腿用末位布局伪影（2623 §9 full=0.0000）；k=32 −0.0166 仅 1.19× 界、n=1（A-AUDIT §4-①②）⇒ D1 | **问**：真稀疏核 + T≥2048 + 同 step + n≥2 下内容稀疏是否回本（位置稀疏迁 A3）。**该涨**：T≥2048 真核吞吐；**该跌**：无（期望持平）；**必须不变**：内容稀疏 KV 缓存字节恒定/位置稀疏下降（分报）、同位置 logits 因果性单测、成本交叉点**实测** |
| A7-addressing | todo | **并入 A6** | 与 A6 是同一研究的重复拆格：A6 选项含 NSA/MoBA top-k，A7 选项含"学习式 top-k"（A7 CRITIQUE ③-4；AGENTS.md"不要重复建格"） | **并入 A6**（"寻址方式"降为 A6 一个选项维） |
| A8-positional-semantics | todo | **立项** | 绝对 PE 给 ring-buffer 解码硬上界 T≤seq_len，是长上下文主线的工程前提；judge 写"长度泛化"零实验（A8 CRITIQUE ③-2/⑤）⇒ D3 | **问**：RoPE+PI 与 ring-buffer 的可组合性（带可失败单测）。**该涨**：训练 512→测 2048/4096 的 bpc 与可跑长度；**该跌**：无；**必须不变**：T=训练长时 bpc 无差异 |
| A9-old-info-retention | todo | **立项** | A 轴最有决策价值的格（"右移压缩"主线落点）；in-distribution 长程腿是真增量仪器；但主腿在分布外、n=1、结论靠累计预算（A9 CRITIQUE ③-2；A-AUDIT P1-1）⇒ D1+D3 | **问**：窗口丢多少长程 + 显式通道救回多少，同契约同台。**该涨**：给显式通道后窗外召回（相对同掩码地板）；**该跌**：无（窗口臂窗外应停在 chance）；**必须不变**：clean bpc 对照口径；**必须报**：同配置重跑 bpc 极差 |
| A10-compression-trigger | judged | **立项** | 判据算术不可达（"比 distance 高 ≥10pp"而全保留上界只有 0.6868，2615 §6.1.1）；统计含未来查询（偏强负结果）；被测对象≠格子命题；"业界无人占位"与自家 `PRIOR_ART §A10` 直接矛盾（A10 CRITIQUE ②③）⇒ D1+D2 | **问**：流式（只看过去）统计 + 同"可见性对数"预算下驱逐优先级是否有可排序收益。**该涨**：低预算保留腿召回；**该跌**：无（clean bpc 应最差）；**必须不变**：被删位置数（对照列）；**必须报**：两条预算口径 + 真 ρ 曲线，阈值先验算可达 |

### C 轴（垂直机制 / 深度）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| C1-layers-per-token | judged | **立项** | 墙钟倍率三 run 1.57/3.08/5.53×（极差 3.5×，`25c/25b/100m-s0`），"3.1×/1.8×"是单次抽样；FLOPs 净账 **+5.6%**（8.45 vs 8.00）未进结论；"最厚的资产"藏负账（C1 CRITIQUE ③）⇒ D1 | **问**：100M 端到端深度草稿 + 交错口径墙钟 + FLOPs 净账。**该涨**：交错 tok/s 比值；**该跌**：等效 layer-position/token（**≤1 才算省算力**）；**必须不变**：发出流与贪心逐 token 相同（质量护栏） |
| C2-layer-selection-order | judged | **保留** | 元素代价表与原始 json 逐一一致（`L1 1.6822⇒+0.1748`、`L3 2.0527⇒+0.5453`，C2 CRITIQUE ③"可核为真"），是 C1/C5/C6 的定价输入，结论未被推翻 | —（删层序 shuffle 子项；修 benefit：TTS K=8 仍 +0.138 @8× 评估；"定律"→"本口径代价表"） |
| C3-inter-layer-topology | judged | **保留** | 审计判"全 C 轴方法学最干净的一格"：等参数对照逐位（6,962,688）、Δ 复算、同方法收敛对账、变异 3/3 抓红（C3 CRITIQUE ②）；"13M/12k 无便宜可捡"未被推翻 | —（子项：parallel 补吞吐腿否则删；dense 加"均匀起步 α≡1"臂才能谈 DenseFormer） |
| C4-weight-sharing | todo | **立项** | 本轴唯一真空白，直接做"参数 ↔ 循环次数"换算，对 carry-门形态关键；本地零经验却被标成业界结论（`status:reference` vs 自己的 `prior_art_verdict:(c)`，三处互相矛盾，C4 CRITIQUE ③）⇒ D3 | **问**：深度方向权重共享是否值。**该涨**：同参数量质量/同 FLOPs 质量/tf@k；**该跌**：参数量、每 token FLOPs；**必须不变**：循环=1 与不共享臂逐位等价（可失败单测） |
| C5-refinement-operator | judged | **立项** | "门是税承载者"是链式分解的**余项**而非隔离量（0.2280），且其前提"退出+填充 +0.200"在 23e matched 12k 下反号（fill=−0.0856）；回答它的现成臂 `23d-H1-12k` 缺 `tf@4`（C5 CRITIQUE ②③）⇒ D1 | **问**：出口资产是否必须付门堆栈税（补 `23d-H1-12k` 的 tf@4/6 ≈5 min + gate-off 臂 ≈10 min）。**该涨**：clean bpc / tf@k / 门活性；**该跌**：gate_mean（"渐进开门"假设的反证点）；**必须不变**：除门外逐字段同、g≡1 与 dense 等价单测 |
| C6-depth-supervision | judged | **立项** | headline"出口资产只能由深度监督造出"恰是 LayerSkip 前提；0.12 vs 0.97 的对照是**未训练 tied 读出 proxy**（`24b-decode-wall-dense.json::tf_note` 自述）；tax 行跨引擎混写；+9% benefit 是墙钟噪声（C6 CRITIQUE ②③）⇒ D1 | **问**：同引擎税 + 出口资产是否需门堆栈（与 C5 共用 23d-H1-12k 的 tf@4）。**该涨**：tf@k（须与**同监督量**对照比）；**该跌**：全深度监督底座税；**必须不变**：判据损失口径（若改归一化须双报两套斜率 §17.15）、w=0 时逐位回到"只监督最终深度"基线 |

### H 轴（视界）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| H1-outputs-per-pass | judged | **立项** | 把宽度-K 前向的**墙钟 min 比**当"等效算力"（H1 CRITIQUE ③-3）：实测 `c₃≈1.25` 是 launch-bound；按层-位置/线性 FLOPs 一次宽度-(1+γ) 前向 ≈(1+γ)× ⇒ 每 token 算力 `(1+γ)/(1+τ)≈1.25×` **恒为亏损**；且 SUMMARY 内 `3.4–5.3×` 与 `1.15–1.29×` 反解冲突（②）⇒ D1+D2 | **问**：spec 解码的真实算力账（解析比值）+ 干净 c₃ 实测。**该涨**：13M 墙钟 tok/s（**仅**该硬件/单流/空卡）；**该跌**：任何"省算力/省 FLOPs"主张须写成 ≥1.25× 亏损；**必须不变**：接受级联 a₂、a₃|a₂、tok/fwd≈2.4；层-位置口径与墙钟口径**分列** |
| H2-training-fill | judged | **保留** | 主结论 fill>nofill（+0.0856，超 ±0.03 带）成立；proj-fill"族内最优"实为 **0.0226 落在自述平局带 ±0.03 内且多 7.6% 参数**（7,489,024 vs 6,962,688，H2 CRITIQUE ②）——主结论未被推翻 | —（benefit 删"族内最优"或加"平局带内/+7.6% 参数/无等参数对照"；§15 对照低优先） |
| H3-composition | judged | **立项** | 算力口径符号错（"0.48/≈2× 效率"实为 `3/2.58≈1.16–1.25×` 亏损）；三臂共用同一 checkpoint ⇒ 结论只能限"同 checkpoint 的解码期选头"；缺 `chain-oracle` 臂（H3 CRITIQUE ③-3/②）⇒ D1（用户 P-02 核心） | **问**：α₁ 差是"部署条件错"还是"头本身差"。**该涨**：chain-oracle 的 α₁（应从 0.657 升到≈0.84）；**该跌**：等效算力 0.48/2× 效率说法；**必须不变**：三臂发出流逐位相同、`direct≈combined>chain` 排序（本轴唯一独立复核仍站得住的结论） |
| H4-head-trunk-relation | judged | **保留** | 本地结论保守自洽：jump-head 0.4667 = 诚实对手 0.5489 的 85%，落 [0.80,0.90) 中性带、接受率 0.5998<0.90 未过（H4 CRITIQUE ③）；无符号级错误 | —（就地撤回残留旧数 0.3128 / 0.4822：2610 §2/§4、2611:175、H4/PROBLEMS.md:9） |
| H5-verification | judged | **保留** | 两条本地增量（@100M fp 平局使"逐位相等"失效、exit×spec 不可组合）保守可核（H5 CRITIQUE ③）；主结论"批量验证==逐步贪心"是业界定理，需改标签 | —（改标签；`tax` 补成本口径"验证复用同一次宽度-(1+γ)前向"） |

### W 轴（宽度）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| W1-active-width | judged | **立项** | 命题-证据错配：唯一本地实验是"读出限前 360 维"（W4 域），README 自写"本地已测：（无）"；"减宽"根本未测（W1 CRITIQUE ③）⇒ D1 | **问**：真正的减宽（推断期确定性裁通道 / 训练期 width dropout）值不值。**该涨**：仅在"保容量减计算"侧；**该跌**：clean bpc；**必须不变**：§15 等参数对照逐字段 + 状态能量占比探针（防把"能量迁出"当"宽度有效"） |
| W2-expertization | todo | **立项**（并入 R6） | 两汇总矛盾（见 §0.1-2），裁决采纳 RP：R6 与 W2 同格、13M 无公开对照（`R/PRIOR_ART §4`）、决定"保容量减计算在本尺度是否成立"⇒ D2+D3 | **问**：13M 同 FLOPs MoE 是否 ≥ dense。**该涨**：同 FLOPs MoE clean bpc；**该跌**：无；**必须不变**：每 token FLOPs=dense、路由不坍缩（专家使用率熵 >0、top-1 最差专家占比≠0） |
| W3-kv-head-sharing | judged | **立项** | 跨家族 Pareto"同 −75% 档 GQA 便宜 3.2×"建立在 R1 已作废的 int4 数字上，与同仓 R1 直接矛盾（W3 CRITIQUE ②）⇒ D1+D2 | **问**：GQA 相对"因果口径下 int4"的同字节 Pareto（或删该列）。**该涨**：无（代价线）；**该跌**：clean bpc（与等参数稠密比）；**必须不变**：解析 KV 字节 `L·2·n_kv·d·2B`、可见性对数不变 |
| W4-readout-geometry | judged | **保留** | 审计判"本轴最健康的一格"：增益归容量/重参数化、随训练缩 43%、不迁移 dense、不采纳为质量机制，与证据一致（W4 CRITIQUE ③）；仅缺 §16 的 24k 点 | —（benefit 标"快照、疑为收敛加速"；补 24k 点低优先） |
| W5-capacity-vs-compute | judged | **删除** | 零本地实验、declared judge（同 FLOPs 质量/同参数量 FLOPs）一项未测、结论定义式重言、唯一证据借用 W1 的读出切片（W5 CRITIQUE ③④）；与 R3 重复 | **命中 B2（业界定论 + 无实质差异 + 无基线角色）+ B3（标签误用无独立内容）** |

### F 轴（迭代）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| F1-iteration-object | todo | **并入 F2** | 选项 3–6 是本仓 PRIOR_ART 自判 (a) 的成熟路线；税值取自 pre-D1/D3 作废引擎；唯一留存 = M0 gate(d) 负结果 + 通道复测（本就是 F2 的活）（F1 CRITIQUE ③④） | **并入 F2**（负结果"direct 重入无新信息"归 F2） |
| F2-return-channel | judged | **立项** | D1 gauge bug 使"soft>linear>direct"排序从未在可比口径测过（它是 cap 破坏量的单调），D6（linear 实为放大 71.7×）未勾选；"结构性 no-op"越证据（direct 实为 `f∘f≠f`，`model.py:398-401`）（F2 CRITIQUE ②③）⇒ D1 | **问**：修 D6 后同批/同 seed/同 token 下 direct/linear/soft/hard 的真实通道排序。**该涨**：能注入新信息通道的逐轮 bpc 边际；**该跌**：direct 边际应停 0（须先证 `alive=true`）；**必须不变**：`transport=none` clean bpc 逐位一致、各档参数量逐位相同 |
| F3-stopping | judged | **并入 G1** | "严格占优点"用带内 Δ（2.0502 vs 2.0545 = **0.0043<0.014**）；判据被事后替换（原判据"低置信桶 ≥ 高置信桶 2×"被改成单调下降读法）；evidence 与 G1 完全重合（F3 CRITIQUE ②③） | **并入 G1**（选择器与预算分配一条线）；该线立项见 G 表 |
| F4-training-signal | judged | **立项** | 把重建损失接到"无载体"dense（业界该损失从不单独使用），其失败不能推出损失无效；对照非等参数（+264,296 vs +52,736）；n=1 且重跑落"0/4096"另一退化模式（F4 CRITIQUE ③）⇒ D1 | **问**：接显式载体（carry/摘要槽）× 训练信号是否有效。**该涨**：有载体臂窗外可重建性/召回；**该跌**：无；**必须不变**：λ=0 与 λ>0 主干逐位等价（`test_recon_default_off_matches_v1_bitwise`） |

### M 轴（载体 / 信息右移）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| M1-carrier-type | judged | **立项** | 入口结论主腿在分布外（241–255，已永久降级）、跨种子符号翻转（seed0 +6.05pp / seed1 −2.44pp）、n=1、状态灯与自家结语"未收口"矛盾；但"13M dense 本无检索能力"是最有价值产物（M1 CRITIQUE ③）⇒ D1（信息右移主线落点） | **问**：显式通道能否跨 seed 稳健把信息搬过窗。**该涨**：分布内多查询腿召回（三 seed 的**最小值**也显著>chance）；**该跌**：无；**必须不变**：U0≈chance、毁源掉回 chance、U1/U2 参数量逐位同、同一物理预算 |
| M2-carrier-compression | todo | **立项** | 量化"负收益"是逐值标尺粒度产物（应 head_dim 块标尺）；2614 首屏仍给已撤 `d*=32`（5 处）；保留/依赖天平 + 检索腿是真产物（M2 CRITIQUE ②③）⇒ D1+D2 | **问**：同字节预算下量化/驱逐/窗口对照（含 per-channel 标尺）。**该涨**：相对 fp16 总字节下降（含标尺）；**该跌**：Δbpc 须落 ±0.014；**必须不变**：严格因果两条测试；2614 补 ERRATA 指针 |
| M3-cross-step-assumption | judged | **并入 M1** | 两点读数被自己判"不可排序/无头寸"（192 免训也 +0.0029）；主命题在 13M 是行业旋钮；唯一跨条件稳定结论"掩码逼不出搬运"（窗外 494 根差恰好 0.000，2616 §5.4）是信息右移主线的负结果（M3 CRITIQUE ③④） | **并入 M1**（负结果"窗口外信息是丢了不是搬了"归 M1；"192 免费"措辞删除） |

### G 轴（策略）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| G1-selector | judged | **立项**（并 F3） | "严格占优"用带内 Δ；`tax`"不加税"误导（门控开火 τ=0.98 付 22.16% 前向）；无自有实验目录；与 F3 证据完全重合（G1 CRITIQUE ②③）⇒ D1 | **问**：按预登记判据（分桶单调 + 低置信桶 ≥ 高置信桶 2×，**照原样**跑）选择性问题是否获证。**该涨**：低置信分桶每 FLOP 修复收益；**该跌**：高置信带收益 ≈0/负；**必须不变**：`fire split gap ≤0.004`、单轮 mixture≡overwrite 逐位一致 |
| G2-currency | judged | **并入 docs/methodology.md** | 不是可证伪命题、是记账纪律；无自有实验；借用他格（`.scratch/15/20/25`）却无 §9 等值证明（G2 CRITIQUE ③④） | **并入 `docs/methodology.md`**（纪律，非研究格）；删"研究格"身份 |
| G3-signal | judged | **保留** | 三信号互补关系与塌缩域失灵形态是真产物（js 假阴 0.000/假阳 0.373、prob0 塌缩域盲 0.976–0.985，G3 CRITIQUE ③）；结论未被推翻 | —（写明 js 是 **2× 前向**成本；distinct-2 标注扩样低优先） |
| G4-calibration | judged | **保留** | 实测支持"TF 窗口 ECE≤0.009、自由生成段失效"；"最稳资产"措辞过强但结论未推翻（G4 CRITIQUE ③） | —（verdict 加口径限定"TF 窗口、逐尺度不同模型"；自由生成失效写首句） |
| G5-parallel-lines | judged | **删除** | 用户已判自设命题无意义（`OPEN_PROBLEMS P-01`；`catalog.yaml:1168`）；部署判据两档皆负（13M +0.1052 / 50M +0.0322，8× 推理）；机制结论贴门槛（+0.0144 vs 0.014，只高 0.0004）（G5 CRITIQUE ②④） | **命中 B2 + 通用判据（不改变任何路径判断）**；资产（确定性对照模板/对账脚本）迁方法学与他格 |

### R 轴（参照线）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| R1-quantization | todo | **立项** | 自写**逐值**标尺把"量化省一半"改成"反占内存"；标准 head_dim 块标尺 int8=1.0625 B/值（省 46.9%）且严格因果，本仓自己算过（`2631/analysis.md:176`）；这是全仓省内存判据基准（R1 CRITIQUE ③④）⇒ D2 | **问**：head_dim 块标尺 / KIVI 式下 int8/int4 的质量与总字节。**该跌**：解析字节 int8≈1.0625（−46.9%）/ int4≈0.5625（−71.9%）；**该涨**：块标尺取整误差上界、int4 Δbpc；**必须不变**：前缀不变性两条=True、非量化臂逐位不变 |
| R2-distillation | todo | **删除** | 零本地实验、零可核实出处、本项目无 teacher、判据无可执行、不改变任何判断（R2 CRITIQUE ③④） | **命中 B2 + B3** |
| R3-pruning-sparsity | todo | **删除** | 与 W5 同结论重复；唯一"本地证据"是错配的 W1 读出切片（换变量）；判据"同 FLOPs 质量"一点未测（R3 CRITIQUE ③④） | **命中 B3（+B2）**；"剪枝=减容量"作为 W2 一行 note |
| R4-window-kv-quant-retrieval | todo | **立项** | 税"几乎为零"与自家 ERRATA 冲突（keep32=+0.212、keep64=+0.081，M2 ERRATA）；量化继承 R1 错口径；检索腿零本地；它是右移压缩必须打败的对手（R4 CRITIQUE ②③）⇒ D2 | **问**：按保留预算的窗口曲线 + 检索腿是否显著>chance。**该跌**：窗口 bpc 损伤随预算单调（keep32→192：+0.212→+0.004）；**该涨**：检索腿召回>chance(0.0156) 且>同掩码地板；**必须不变**：T=256 净亏结论、同拓扑 fp16 基线 |
| R5-bestofn-verifier | todo | **删除** | 零本地实验；"侧证"用 G5（用户判无关，换命题）；本项目无 verifier（R5 CRITIQUE ③④） | **命中 B3 + B2** |
| R6-moe-scale | todo | **并入 W2**（并入后立项） | 与 W2 同格；13M 无公开对照；决定 W2/W5 的判据（R6 CRITIQUE ③④） | **并入 W2**（见 §0.1-2）；W5 已删，分类事实挂 W2 |
| R7-existing-early-exit | todo | **立项** | 本地对照只有 LayerSkip 式 dense 监督臂，无同 FLOPs 的 CALM/MoD 臂；它决定 C1 的 +0.4362 是"新问题"还是"早退共性"（R7 CRITIQUE ③④）⇒ D2 | **问**：同 FLOPs dense 早退基线下本形态增量是否成立。**该涨**：同 FLOPs 下本形态质量应 ≥ 基线；**该跌**：本形态形态税相对基线不应更大；**必须不变**：tf@4、逐位一致门禁 |

### P 轴（修饰符）

| 格 | 现状 | 决策 | 一句话理由（审计证据） | 立项问题 · 验证口径／删除条款 |
|---|---|---|---|---|
| P1-precision | judged | **保留** | 证据真实（两次发散：v1 step 4600 bf16+负JS、v2 step 2200 silent-NaN，`09-m3-interactions/SUMMARY.md:21-24`）；结论是通用常识非本仓发现，但作为 modifier note 有防复发价值，结论未推翻 | —（降为 note 口径；补 `prior_art_*` 字段） |
| P2-conditioning | todo | **删除** | 无对照臂、`failure:[]`、判据借 C1 副产物、"置信度门控"实为 G1 的命题（重复登记）；剩余选项（噪声级/任务条件）在本项目无用例（P2 CRITIQUE ③④） | **命中 B3**（重复登记）；门控归 G1 |
| P3-param-insertion-init | judged | **保留** | 真增量"全零初始化在乘性结构上是鞍点"有独立证据；"族内最优"0.0226 落在自述 ±0.03 平局带内，只能报"方向为正"（P3 CRITIQUE ③）——主体结论未推翻 | —（序关系降级；补 ≥2 seed 或 2× 点） |
| P4-normalization | todo | **立项** | 漏记本仓已有 QK-Norm 实测（gnorm 1740.9→7.25 / 29578.6→7.58，推荐 `qk-normfrom512`）；"无 final norm 是前提"被当结论（P4 CRITIQUE ③）⇒ D3 | **问**：QK-Norm 稳定化档位回填 + 有/无 final norm 消融。**该涨**：开 QK-Norm 后 gnorm_max 从 1e3–1e4 降到 <100；**该跌**：全局档短 T clean bpc（T=512 +0.0314 越界）须用长度条件档；**必须不变**：T≤512 零 hook、`logits=h·Eᵀ` 同空间 |
| P5-loss-weighting | judged | **保留** | 浅层降权"单调反向"数字真实，但两组数字跨口径（legacy 1.1-epoch vs 新引擎）并列；判据未被推翻（P5 CRITIQUE ③） | —（删跨口径并列；现代口径重跑两臂低优先） |
| P6-data-construction | judged | **保留** | §19 局部先验抗性与 A4 假保留陷阱是本仓真增量；needle v2 结论成立（in-window 84.9%/anchor 76.8%/beyond 0.4%）；结论未被推翻（P6 CRITIQUE ③） | —（§19 升格为契约必填；补"同一权重两套口径"成对检验；摄入 42-W2 极差 10.84pp 警示） |

---

## 2. 四张清单

### ① 删除清单（7 格）

| 格 | 命中的政策条款 | 关键证据 |
|---|---|---|
| A1-causality | B1（构造前提类）+ B3（标签误用） | 零数据（`catalog.yaml:205`）；非因果 LM 与 AR 不变量不相容；A1 CRITIQUE ③-4 |
| W5-capacity-vs-compute | B2 + B3 | 零本地实验、判据全未测、结论定义式重言、evidence 借 W1（W5 CRITIQUE ③④） |
| G5-parallel-lines | B2 + 通用判据 | 用户判自设命题无意义（`OPEN_PROBLEMS P-01`）；部署判据两档皆负；贴门槛 +0.0144（G5 CRITIQUE ④） |
| R2-distillation | B2 + B3 | 无 teacher、零实验、无判据（R2 CRITIQUE ③④） |
| R3-pruning-sparsity | B3（+B2） | 与 W5 重复、证据是错配的 W1 读出切片（R3 CRITIQUE ③④） |
| R5-bestofn-verifier | B3 + B2 | 无 verifier、侧证用无关的 G5（R5 CRITIQUE ③④） |
| P2-conditioning | B3 | 重复登记 G1 的门控；剩余选项无用例（P2 CRITIQUE ③④） |

> 执行提示（供主 Agent）：按政策 B，删除 = 从 `catalog.yaml` 移除该格 + 目录移入 `archive/dropped/<原路径>` + 在 `archive/README.md` 加一句话原因。**本文件不改这些**。

### ② 立项清单（24 格，按优先级；主线优先）

**P0 — 坏天平，不修则其它线的判据都站在错口径上（4 格）**

| 格 | 必须回答的问题 | 一句话为什么是 P0 |
|---|---|---|
| R1-quantization | head_dim 块标尺 / KIVI 式下 int8/int4 的质量与总字节 | 全仓省内存判据基准；错一格带偏 A3/A9/M1/M2/W3/R4（R1 CRITIQUE ③） |
| H1-outputs-per-pass | spec 解码的解析算力账 + 干净 c₃ 实测 | "省算力"符号被算反（≈1.25× 亏损），直接决定用户 P-02 的部署经济性（H1 ③-3） |
| H3-composition | α₁ 差是"部署条件错"还是"头本身差"（补 chain-oracle 臂） | 用户 P-02 核心；算力口径符号错（H3 ③-3） |
| R7-existing-early-exit | 同 FLOPs dense 早退基线下本形态增量是否成立 | 决定 C1 的 +0.4362 是"新问题"还是"早退共性"（R7 ③④）——决定整条 C 线相对谁谈判 |

**P1 — 主线必须回答 / 必须打败的基线（8 格）**

| 格 | 必须回答的问题 |
|---|---|
| M1-carrier-type | 显式通道能否**跨 seed 稳健**把信息搬过窗（3 臂×3 seed，分布内多查询腿）——"信息右移"主线落点 |
| A3-window-anchor | 自包含复算窗口/锚点，拆"评测期丢前缀"与"累计预算"——压缩/摘要线的价格上界 |
| A9-old-info-retention | 窗口丢多少长程 + 显式通道救回多少（同契约、in-distribution 长程腿） |
| W1-active-width | 真正的减宽（裁通道 / width dropout）值不值 |
| W2-expertization（并 R6） | 13M 同 FLOPs MoE 是否 ≥ dense——"保容量减计算"在本尺度是否成立 |
| R4-window-kv-quant-retrieval | 按保留预算的窗口曲线 + 检索腿——右移压缩必须打败的对手 |
| C1-layers-per-token | 100M 端到端深度草稿 + 交错口径墙钟 + FLOPs 净账（省延迟不省算力） |
| M2-carrier-compression | 同字节预算下量化/驱逐/窗口对照（含 per-channel 标尺） |

**P2 — 方法/结论有错，改正后可能改判（8 格）**

| 格 | 必须回答的问题 |
|---|---|
| C5-refinement-operator | 出口资产是否必须付门堆栈税（补 23d-H1-12k tf@4 + gate-off 臂） |
| C6-depth-supervision | 同引擎税 + 出口资产是否需门堆栈（与 C5 共用一次 eval） |
| A6-mixing-operator | 真稀疏核 + T≥2048 + 同 step + n≥2 下内容稀疏是否回本 |
| A10-compression-trigger | 流式统计 + 同可见性对数预算下驱逐优先级是否有可排序收益 |
| A4-soft-distance-penalty | per-head slope + 长度外推是否成立 |
| A8-positional-semantics | RoPE+PI 与 ring-buffer 可组合性（长上下文工程前提） |
| W3-kv-head-sharing | GQA 相对"因果口径下 int4"的同字节 Pareto（或删该列） |
| C4-weight-sharing | 深度方向权重共享值不值（参数 ↔ 循环） |

**P3 — 补最严判据 / 低优先（4 格）**

| 格 | 必须回答的问题 |
|---|---|
| F2-return-channel | 修 D6 后 direct/linear/soft/hard 的真实通道排序 |
| F4-training-signal | 接显式载体 × 训练信号是否有效 |
| G1-selector（并 F3） | 按原预登记判据（分桶单调 + 2×）选择性问题是否获证 |
| P4-normalization | QK-Norm 档位回填 + 有/无 final norm 消融 |

### ③ 并入清单（7 格）

| 被并格 | 并入 | 理由 |
|---|---|---|
| A5-dilated-window | A6-mixing-operator | 位置稀疏应作 A6 的"同预算对照"选项；信息已被 A3+A6 覆盖 |
| A7-addressing | A6-mixing-operator | 与 A6 是同一研究的重复拆格（都是"用谁/怎么选"） |
| F1-iteration-object | F2-return-channel | 选项 3–6 属业界成熟路线；残余 = M0 负结果 + 通道复测（本属 F2） |
| F3-stopping | G1-selector | 判据用带内 Δ、证据与 G1 完全重合；合并为"选择器与预算分配"一条线 |
| M3-cross-step-assumption | M1-carrier-type | 两点读数不可排序；唯一稳定负结果"掩码逼不出搬运"归信息右移主线 |
| R6-moe-scale | W2-expertization | 与 W2 同格；13M 无公开对照 ⇒ 并为一格后立项（见 §0.1-2） |
| G2-currency | docs/methodology.md | 是记账纪律，不是可证伪研究格 |

> 执行提示：并入后在 `catalog.yaml` 互指（指针），**不留空壳**；被并格目录按并入方式处理（合并内容 or 移入 `archive/dropped/`）。

### ④ 保留清单（13 格，审计未推翻结论，原样保留含数字）

| 格 | 保留依据（审计未推翻的点） |
|---|---|
| A2-full-attention | 税零点/锚点，结论"基底成立" |
| C2-layer-selection-order | 元素代价表与原始 json 逐一一致，是 C1/C5/C6 定价输入 |
| C3-inter-layer-topology | 全 C 轴方法学最干净的一格；"13M/12k 无便宜可捡"可复现快照 |
| H2-training-fill | fill>nofill（+0.0856 超带）成立 |
| H4-head-trunk-relation | 本地结论保守自洽（85% / 未过 0.90 门） |
| H5-verification | 两条本地增量（fp 平局、exit×spec 不可组合）保守可核 |
| W4-readout-geometry | 本轴最健康的一格；增益归容量/重参数化、不迁移 dense |
| G3-signal | 三信号互补关系与塌缩域失灵形态是真产物 |
| G4-calibration | "TF 窗口 ECE≤0.009 / 自由生成段失效"实测成立 |
| P1-precision | 两次发散证据真实 |
| P3-param-insertion-init | "全零初始化=鞍点"诊断有独立证据 |
| P5-loss-weighting | 浅层降权单调反向数字真实（口径需标注） |
| P6-data-construction | §19 局部先验抗性 + A4 假保留陷阱是真增量 |

---

## 3. 计数核对

- 删除 7：A1, W5, G5, R2, R3, R5, P2
- 立项 24：A3, A4, A6, A8, A9, A10; C1, C4, C5, C6; H1, H3; W1, W2, W3; F2, F4; M1, M2; G1; R1, R4, R7; P4
- 并入 7：A5→A6, A7→A6, F1→F2, F3→G1, M3→M1, R6→W2, G2→文档
- 保留 13：A2; C2, C3; H2, H4, H5; W4; G3, G4; P1, P3, P5, P6
- 合计 7+24+7+13 = **51** ✓（与 `axes/**/CRITIQUE.md` 份数一致）

## 4. 遗留与不可核实项（如实标注，供主 Agent 决策）

- **外部文献一律未核**：五份汇总均自述"本环境无联网检索"，Medusa/KIVI/CALM/LayerSkip 等业界数字按审计口径只作"待核假设"；政策 F 要求把业界结论当**必须核实并对齐的基线**，因此本表把相关格列为立项而非删除，但它们的第一步应是"核实出处"。
- **证据锚点真伪无法核**：`check_schema.py` 只查路径存在、丢弃 `@commit`（A-AUDIT §6、HW 独立确认）；C 轴同一路径挂多个 `@sha`（C-AUDIT §4-③）。本表按内容判，未按锚点判。
- **本文件只做树管理建议**：删除的落盘（移 `archive/dropped/` + 改 `archive/README.md`）、并入的互指、立项的契约与排期，均由主 Agent 按政策 B/E 执行；本文件未改 `catalog.yaml` 与任何格目录。
