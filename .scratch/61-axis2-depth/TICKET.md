# ticket 61 · 轴 2 深度轴 lead（Phase A）

> 性质：**过程工单**（轴 2 的编排 + 执行 + 复核轨迹）。事实文档 = `axes/2-depth/**/project.yaml`
> 与 `axes/2-depth/README.md`；本文件只记「我们一步一步怎么走的」。
> 用词以 `docs/structure.md` §12 为准（轴 2 的分组 2.1/2.2/2.3 与 10 个研究项）。
> 记录纪律：显式时间戳 + actor + 命令与结果 + 证据锚点；框架性取舍上报 Main（见 §5）。

## 1 范围与契约（轴 2 lead）

| 项 | 值 |
|---|---|
| 轴 | 2 深度轴（`axes/2-depth/`） |
| owner | 轴 2 深度轴 lead |
| writer | 轴 2 lead（本轴唯一写入者；WP 若派发 worker，则 writer = 该 WP worker，单一写入者） |
| reviewer | fresh 复核者（独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后） |
| integrator | 研究树 integrator（M3 集成；本轮由轴 2 lead 代行本轴合并与跨轴一致性核对） |
| 规范输入 | `docs/structure.md` §12（节点/用词）、`docs/research-schema.md`（字段契约）、`SCHEMA.md` §6–§24 |
| 本轴节点 | 分组 2.1 / 2.2 / 2.3；研究项 `2.1.1 2.1.2 2.1.3 2.2.1 2.2.2 2.2.3 2.2.4 2.2.5 2.3.1 2.3.2`（其中 `2.1.3` / `2.2.5` = Backlog） |
| 边界 | 只动 `axes/2-depth/**` + `.scratch/61-axis2-depth/**`；不碰其它轴、`docs/structure.md` / `docs/research-schema.md` 用词、共享工具 |
| 并发/锁 | 并发子 Agent ≤3；GPU 工作按 `SCHEMA.md` §20 互斥锁**串行**，纯 CPU 不取锁 |
| 预算（Phase A） | 本轮为**有限预算**：目标 = ①10 项 role_map 实名；②逐项「想清楚」复核并修 `project.yaml`；③WP 拆分与排序落盘；④执行 2.1.1 的最高价值 WP（实现 + 序列级实测）；⑤fresh 独立复核至无问题。GPU 训练**不在**本轮（见 §4 预算理由）。 |

## 2 Phase A 计划与逐项复核（2026-10-13T12:40+08:00，actor：轴 2 lead）

### 2.0 轴级共同口径（沿用已冻结项，不自选）

- 目标函数：**用更少的算力拿到序列级类似效果**；代价口径**只**用等效理论量
  （layer-passes/token、FLOPs、tokens-per-forward，考虑 pipelining），**墙钟/吞吐不进判据**
  （`docs/research-schema.md` §3、`axes/2-depth/README.md`）。全部 10 项 `cost_model` 复核 = 无墙钟 ✓。
- 对照 = 同尺寸 stock 全深度（归一 100%）+ 同预算固定深度（+ 各项特有对手）。
- 噪声带 0.014（本仓 13M 跨 run bpc 漂移界；同 checkpoint 内的比较另报**窗口间 SE**，见 §4.4）。

### 2.1 逐项「想清楚」复核（10/10）

复核维度（对齐 `docs/research-schema.md` §2 与 `SCHEMA.md` §23 七问）：
①痛点 ②业界 SOTA 与出处 ③我们相对 SOTA 的增量 ④方法（单因素） ⑤实现 ⑥**客观判据
（该涨/该跌/必须不变）** ⑦做不出来怎么归因（设计/结构/innovation） ⑧等效理论代价 ⑨单一 milestone
⑩WP 与推进顺序。

| 研究项 | 复核结论 | 处置 |
|---|---|---|
| 2.1.1 early exit | 十项齐全；命题已按 `55-review` C-P01/02/03/04 改成序列级 e2e，缺口（主判据未跑、校准未验、异构深度 KV 未测、缺省算力对手）显式入 `gaps`；**本轮执行 WP1** | 补 role_map 实名；WP 排序见 §3；WP1 本轮执行 |
| 2.1.2 skipping | 十项齐全；旧 C2 三条硬伤（无目的/跨代引擎/p_skip 逐序列非逐 token/self-review）已改；`gaps` 齐 | 补 role_map 实名 |
| 2.1.3 repeating | Backlog 身份成立（本仓零经验 + 出资人 C4 先验否定）；**判据与 2.2.5 重复度高**（皆「同参数量 vs 非共享 dense」） | 补 role_map 实名；把「与 2.2.5 的记账边界」写入 `gaps`（避免两格同题） |
| 2.2.1 残差的形式 | 十项齐全；**缺一条「可截断性」判据的可操作化**（逐深度 bpc 只是代理） | 补 role_map 实名；在 `gaps` 写明「与 2.1.1 联合的可截断性未测」已存在 ✓（保持） |
| 2.2.2 用 causal attention 代替残差 | 十项齐全；出资人 C3 批注的设计从未测（C-P10）已入 `gap` | 补 role_map 实名 |
| 2.2.3 层间学习加权和 | 十项齐全；旧 C3 中性读数（−0.0008，n=1）已如实记为观测非定论 | 补 role_map 实名 |
| 2.2.4 并行子层 | 十项齐全；「只测质量面就判有害」已改为「质量面+代价面同框」（C-P18） | 补 role_map 实名 |
| 2.2.5 权重共享 | Backlog 身份成立；旧 C4 三处口径打架（todo/(a)/不判(a)）已消除 | 补 role_map 实名；`gaps` 补与 2.1.3 的边界 |
| 2.3.1 逐层容量变化 | 十项齐全；与 3.1.2 的边界已写 | 补 role_map 实名 |
| 2.3.2 逐层分辨率 | 十项齐全；长程召回作损伤对照列已写；与 1.2.4 边界已写 | 补 role_map 实名 |

**复核发现的框架性缺口（不改规范用词，只记此处并上报 Main）**：

1. **口径操作化未冻结**：`validation` 冻结的是「序列级 perplexity/一致率 vs 全深度」。Phase A 采用
   的操作化 = **teacher-forcing、逐位置在线决策下的整段 bpc**（每个位置按自己到达的深度读出，
   填充语义按因果顺序传播），见 §4.1。理由：这是唯一同时含「退出读出的预测质量」与「异构深度 KV
   语义代价」的序列级量；旧 `depth_curve` 是「全深度上下文下的逐深度读出」，**不含**填充代价
   （C-P06），不能替代。**待 Main 确认**（已上报，见 §5）。
2. **`falsification` 的门槛措辞**：10 项 `falsification` 均写「越噪声带 0.014」，但该带是**跨 run
   训练漂移**，而 Phase A 是**同 checkpoint 纯推理**（无训练漂移，只有窗口抽样误差）。故同
   checkpoint 内的比较另报**窗口间 SE** 并同时报「相对跨 run 噪声带的比值」，避免把抽样噪声
   当成训练噪声（口径归因，上报 Main）。

## 3 WP 拆分与推进顺序（按价值/风险排序，2026-10-13T12:45+08:00，actor：轴 2 lead）

排序原则：① 先补**能承载主判据的承载体**（无承载体则一切数字都是代理）；② 先做**低成本高判决力**
（纯推理 / 协议设计，无 GPU）；③ 训练类 WP 打包后置（昂贵、需 GPU 锁、需排队）。

| 序 | WP | 所在项 | 类型 | 判决力 | 成本（等效） | 本 Phase 是否执行 |
|---|---|---|---|---|---|---|
| 1 | **WP1-impl**：逐 token 退出决策头**实装进生产解码器** + 校准 + 序列级协议 | 2.1.1 | 实现+纯推理 | 高（主判据的承载体） | 0 GPU；CPU 推理 | ✅ 执行 |
| 2 | **WP1-e2e**：多阈值**序列级**（teacher-forcing 在线决策）bpc/一致率 vs 全深度 + layer-passes/token | 2.1.1 | 纯推理 | 高（主判据本身） | 0 GPU；CPU 推理 | ✅ 执行 |
| 3 | WP2：异构深度 KV 语义（fill / proj-fill / no-fill）质量代价 | 2.1.1 | 纯推理 | 高（决定深层是否失真） | 0 GPU（复用 WP1 口径） | 下一轮（口径已在 WP1 预留） |
| 4 | WP5：**省算力对手**（同预算更浅 dense / 深度剪枝） | 2.1.1 `gaps` C-P12 | 训练+纯推理 | 高（前沿的横轴锚点） | GPU 训练（13M ×2 臂） | 下一轮（需 GPU 锁排队） |
| 5 | WP1-cal：退出决策头校准（ECE / risk-coverage，两种标签） | 2.1.1 | 纯推理 | 中（回答 C-P05） | 0 GPU | ✅ 随 WP1 执行 |
| 6 | WP3：深度监督配方（additive λ 剂量 + 补 `23d-H1-12k` 的 tf@4） | 2.1.1 | 训练 | 中（税的可比价签） | GPU 训练（臂多） | 后置 |
| 7 | WP4：精化门的税与出口资产可替代性 | 2.1.1 | 训练 | 中 | GPU 训练 | 后置 |
| 8 | 2.1.2 WP1/WP2；2.2.1/2.2.2/2.2.3/2.2.4/2.3.1/2.3.2 各 WP1 | 其它 7 项 | 训练为主 | 中 | GPU 训练 | 后置（本轴下一 Phase） |

**本轮预算理由**：GPU 被外部任务占用（`nvidia-smi` 12:40 起 94% util / 19.2GB used，`/tmp/pathlm_gpu.lock`
不存在但 `/tmp/llama_gpu.lock` 存在）⇒ 训练类 WP 强行挤锁只会拖慢且违反 §20 礼貌；
而 2.1.1 的**主判据缺口（C-P01）是纯推理就能补的**（同一 checkpoint、只改解码策略）⇒ 本轮把预算
压在「承载体 + 主判据」上，训练类后置。

## 4 执行记录

### 4.1 口径操作化（Phase A 采用，待 Main 确认）

**A. 序列级 bpc（主判据）**：`score_stream` —— 在整段真实文本上**逐位置在线**决定退出深度
（位置 i 在第 k 层后的决策只依赖 (k, 该位置第 k 层 logits)，严格因果），得到每位置的退出深度
`d_i`、退出读出 `logits_i`（`depth=d_i`），bpc = `mean CE(logits_i, t_{i+1}) / ln2`。
它同时含 ①退出读出的预测质量 ②异构深度 KV 填充语义对后续位置的影响。
**与旧 `depth_curve` 的区别**（必须讲清）：`depth_curve[k]` = 全深度上下文下第 k 层读出的 bpc
（所有位置都跑到 L 层，填充不生效）；本口径是**真的按 `d_i` 停机**，故不同。二者在
「所有位置都退到 k」时**应相等**（本实验用它做口径自洽锚点）。

**B. 自回归一致率（描述列，非必须不变）**：自由生成（`commit="soft"`）下，逐 token 贪心流
与全深度贪心流逐 token 相同的比例。early exit 本就会改流，故它是**诊断列**；仅退化臂
（阈值=1.0，永不过早退出）必须为 1.000。

**C. 算力账**：`layer-passes/token = mean(d_i)`、跳过层占比 = `1 − mean(d_i)/L`、
退出深度直方图。**墙钟一律不报进判据**。

**D. 校准（回答 C-P05）**：在全深度前向上取每个深度 k 的 top-1 概率 `p_k`，配两种标签：
①`tf`（`argmax_k == argmax_L`，即「退出会不会改变答案」）②`true`（`argmax_k == 真实下一字节`）。
报分箱 ECE（10 等宽）与 risk–coverage（阈值扫描下的保留比例 × 保留部分准确率）。
**两种标签并列**是刻意的：`55-review` C-P02 指出单 token `tf@k` 是自洽代理而非正确性，
并排报才能看出「自洽高、正确性低」的差距。

### 4.2 执行动作（时间戳 + 命令 + 结果）

1. `2026-10-13T12:38:29+08:00` 读权威输入：`docs/structure.md` §12–§15、`docs/research-schema.md`、
   `axes/2-depth/**/project.yaml`（10 份）、`.scratch/55-review/C-PROBLEMS.md`、
   `.scratch/56-restructure/MAPPING.md`、`.scratch/57-program/TICKET.md`、`SCHEMA.md` §20/§22。
2. `2026-10-13T12:40:01+08:00` `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合
   docs/research-schema.md`（基线：本轴 10 份在改前即过契约）。
3. `2026-10-13T12:41:00+08:00` `nvidia-smi` ⇒ RTX 4090, used 19240 MiB / 24564 MiB, util 94%；
   `/tmp/pathlm_gpu.lock` 不存在、`/tmp/llama_gpu.lock` 存在 ⇒ GPU 被外部任务占用（**本轮不取锁**）。
4. `2026-10-13T12:42:00+08:00` 复核结论：10 份 `project.yaml` 十项齐全（痛点/创新/SOTA 基线/增量/方法/
   判据/否证/归因/代价/milestone/WP），无「手段枚举」「无目的」「墙钟进判据」类硬伤；只补 role_map 实名与
   两处边界（2.1.3 ↔ 2.2.5 记账边界；口径操作化的同 checkpoint 噪声列）。逐项见 §2.1 表。
5. `2026-10-13T12:42:00+08:00` 核验证据锚点：`.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt`
   存在，`sha256 = 2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6`
   （与 `2621/MODEL.md` 登记一致）；`data/enwik8_full.npz` 存在 ⇒ **主判据可纯推理复现，无需训练**。

### 4.3 WP1-impl：退出决策头实装（结果见 §4.5）

- 落点：新模块 `pathlm/exit_policy.py`（策略 + 序列级口径 + 校准 + 算力账）；
  生产解码器 `pathlm/depth_ar.py::DepthARDecoder.step_auto`（**唯一**的共享文件接线，§22.7）。
- 单测：`tests/test_axis2_exit_decision.py`，钉住 ①prescribed 排程下与并行 `forward` 逐位一致、
  ②在线策略下「增量解码链」与「批量逐层」一致、③阈值边界（min/max 深度、τ=1.0 永不过早退）、
  ④算力账、⑤no-fill 语义。

### 4.4 WP1-e2e：序列级实测（结果见 §4.5）

- checkpoint：`23c-H2-w0-12k`（13M；d=256/L=8/h=8/seq=256/mlp=4；v1 形态：carry 门 + fill_kv、
  proj_fill=False、exit_mode=uniform、aux_weight=1.0）。**不训练**。
- 数据：`data/enwik8_full.npz` val 尾段，固定 offset 窗口（非随机）⇒ 可复现；
  报**窗口间 SE** 作为同 checkpoint 抽样的离散度。
- 臂：`stock-full`（全深度基线）、`prescribed-4` / `prescribed-6`（固定前缀退出，口径自洽锚点）、
  逐 token `Top1ProbPolicy` 的阈值扫描（τ ∈ {0.3,0.6,0.9,0.95,0.99,0.999}）。
- 设备：**CPU**（纯推理，不取 GPU 锁；§20 明文「纯 CPU 工作不取锁」）。同一设备跑全部臂 ⇒ 臂间可比。

### 4.5 结果（2026-10-13T13:15:00+08:00 起，actor：轴 2 lead）

**承载体（WP1-impl）**：
- 新模块 `pathlm/exit_policy.py`（策略 `Top1ProbPolicy` / `PrescribedDepths` / `NeverExit`；序列级口径
  `score_stream` / `stream_bpc`；校准 `exit_calibration` / `risk_coverage`；算力账 `layerpass_ledger` /
  `depth_histogram`）；共享文件只加**一处**接线：`pathlm/depth_ar.py::DepthARDecoder.step_auto`
  （逐 token 退出判断，`SCHEMA.md` §22.7 的最小接线原则）。
- 单测 `tests/test_axis2_exit_decision.py`：16 条，`python3 -m pytest tests/test_axis2_exit_decision.py -q`
  ⇒ **16 passed**。钉住 ①prescribed 排程下 `score_stream` 逐位等于并行 `forward`；②`step_auto` 链与批量
  逐层给出同一套退出深度与读出；③在线策略下二者一致；④阈值边界（min/max、τ=1.0 永不过早退、τ=0 → min_depth）；
  ⑤因果性（改后面的 token 不影响前面位置的退出深度与读出）；⑥no-fill / proj-fill 语义；⑦算力账与直方图；
  ⑧`NeverExit` 的自由生成与既有全深度贪心生成逐 token 相同。
- 全量回归：`python3 -m pytest tests/ -q` ⇒ **831 passed**（改动共享文件 `depth_ar.py` 后无回归；
  该套件默认 CPU，未使用 GPU ⇒ 未触发 §20）。

**序列级实测（WP1-e2e）**：`axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/`
- 协议：13M checkpoint（sha256 校验通过）、`batch(seed=2)×20×16`、T=256 ⇒ 320 窗口 / 81,600 预测位置、
  CPU 纯推理、不取 GPU 锁；11 条臂 + 4 类归因诊断；全部数字在 `results.json`。
- **口径自洽锚点（决定性）**：`prescribed-k`（k=2..7）与全深度臂逐步复现历史 `depth_curve[k]`，
  **Δ = ±0.0000**（7 个深度全中）⇒ 新口径是既有判据的严格扩展，不是另起一套量。
- **主判据**：固定前缀前沿（折线插值）同算力比较 ⇒ 门控超出量
  τ=0.3 **+0.2593**、τ=0.6 **+0.1168**、τ=0.9 **+0.0117**、τ=0.95 +0.0035、τ=0.99 **+0.0000**、
  τ=0.999 −0.0002 bpc。⇒ 省得多时门控显著更差（0.25–18.5 倍噪声带），省得少时打平。
  τ=0.9（5.967 层）**连 prescribed-4（4.000 层）都不如** ⇒ 被固定截断支配。
- **归因（四个替代解释全被对照排掉）**：同位置配对显示损伤全部在早退位置自己的读出
  （各桶加权 +0.12046 vs 总 Δ +0.120461，逐位闭合；跑到满深度的那一桶只 +0.0033）；
  深度置换臂更差（+0.095 / +0.080，排除选位错）；no-fill 臂更差（+0.051 / +0.024，排除填充伪影）；
  结构零假设 `prescribed-4 × {fill, no-fill}` Δ = 0.000000（活性判据）。
- **校准（回答 C-P05/C-P02）**：top-1 概率对正确性校准良好（`ECE_true` ≤ 0.030，深度 ≥3 时 ≤ 0.005），
  对「是否与全深度 argmax 一致」严重失准（`ECE_tf` 0.20–0.37 且方向为**低估**）⇒ 拿 `tf@k` 反推
  「可以放心退」站不住。risk–coverage 显示门控有区分力（深度 1/τ=0.6：保留 0.877 vs 筛掉 0.341），
  但换不来前沿。
- **判决**：`project.yaml` 的 `status` 由「进行中」改 **被否证**（对照本项预登记的 falsification ②），
  `failure_analysis` 归类 **结构问题为主因 / 设计问题为次因 / 非 innovation 本身失败 / 非实现口径统计假象**；
  新增 `gaps` 6 条（WP6 策略感知训练、WP5 省算力对手、proj-fill、自由生成质量、规模复本、C-P13 对照）
  与 WP5/WP6 两个新 work package。

### 4.6 证据锚点

- 实验目录：`axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/`
  （`README.md` 预登记契约 / `config.json` / `run.py` / `results.json` / `analysis.md` / `MODEL.md`）。

## 5 上报 Main 的框架性取舍（不自选）

| # | 取舍 | 我的处置（暂定） | 理由 | 状态 |
|---|---|---|---|---|
| F1 | 「序列级 perplexity」的操作化 = teacher-forcing 在线决策 bpc（§4.1A） | 按此执行，同时保留 depth_curve 作自洽锚点 | 唯一同时含退出读出质量与异构深度填充代价的序列级量；旧 depth_curve 不含填充代价（C-P06） | 已上报 `progress_update`，待 Main 确认 |
| F2 | 同 checkpoint 比较的噪声口径：报窗口间 SE + 标注与 0.014 跨 run 带的关系 | 双列并报，判据仍以 0.014 为可比门槛 | 避免把抽样误差读成训练噪声 | 已上报 |
| F3 | 2.1.1 的 WP 顺序：先纯推理主判据、训练类（省算力对手/配方/门）后置 | 见 §3 | GPU 被外部占用 + 主判据缺口纯推理可补 | 已上报 |
| F5 | `2.1.1` 的 milestone 验收文字（「至少一条策略省算力且 Δ ≤ 0.014」）在本轮**字面成立**（τ≥0.99 省 6–12% 且 Δ ≤ 0.0002），但实质判决是否证（被同算力固定截断支配）。是否把验收口径从「Δ ≤ 0.014」改成「相对同算力固定深度前沿的超出量 ≤ 0」 | 不改 milestone 文字（预登记契约），只在 `conclusion` 里同时写明「字面成立 / 实质否证」；是否收紧口径留 Main 裁 | 口径选择属框架性取舍，不由执行方自选；且平坦深度曲线下 0.014 门槛对「省零头」的档位过松 | 已上报 `progress_update`，待 Main 裁 |
| F4 | 2.1.3 与 2.2.5 判据近乎同题（皆「同参数量 vs 非共享 dense」，且都不省 FLOPs） | 两格均维持 Backlog，`gaps` 互指边界；是否合并留给 Main | 不改变「某算力路径值不值」的项按 §5 判据不该占两格 | 待 Main 裁 |

## 6 复核（SCHEMA.md §14/§14b）

- 排程：writer（轴 2 lead）停止写入 → fresh 复核者（独立上下文 subagent）复核 WP1 代码 + 协议 +
  `project.yaml` 改动 → 有缺陷 ⇒ 派 writer 修 ⇒ **新的** fresh 复核 → 至无问题。
- 记录见 §7。

## 7 复核记录

### 7.1 fresh 复核 round 1（2026-10-13T13:40:00+08:00 派发）

- 复核者：独立上下文 subagent（**非**本项 writer）；范围 = ①`pathlm/exit_policy.py` 与
  `pathlm/depth_ar.py::step_auto` 的实现正确性；②`tests/test_axis2_exit_decision.py` 是否真的能失败
  （变异式质疑）；③`2631` 的协议与 `results.json` 数字是否支持 `analysis.md` 与 `project.yaml` 的结论；
  ④10 份 `project.yaml` 的 role_map 与 `2.1.1` 的状态/结论/归因是否与证据一致。
- 报告落 `.scratch/61-axis2-depth/REVIEW-1.md`；缺陷处置见 §7.2（修完派**新的** fresh 复核）。
