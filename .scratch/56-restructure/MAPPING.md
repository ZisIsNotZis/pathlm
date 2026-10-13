# ticket 56 · WP-M1.0 · 旧 → 新映射（MAPPING）

> 用途：把现有 37 个旧格、参照线、修饰符逐条映射到新研究树（`docs/structure.md` §12），
> 并声明**证据去向**，保证 `axes/**` 的 2434 个文件里**没有任何研究证据丢失**。
> 处置取值：`迁移` / `拆` / `并` / `降参照` / `工程纪律` / `Backlog`。
> 新目录命名见 `docs/structure.md` §14；新研究项粒度见 §13。
> **本文件只冻结映射与证据归属；实际移动文件由各轴 worker 执行。**

## 0. 证据不丢失的口径（可核）

1. **计数**：迁移前 `axes/**` 共 **2434** 个文件（`find axes -type f | wc -l`），迁移后不得减少。
2. **一等依据**：每个旧格目录 `axes/<旧轴>/<旧格>/**`（`README.md` + `experiments/**`）在迁移时
   **整目录搬到**其新家 `axes/<新轴>/<新研究项>/experiments/**`（保留子路径名）。
3. **拆并格**：一个旧格的证据按「**它测的是什么**」**只**落到一个主新家（**一格一份家，不重复复制**）；
   另一新家只**指针**指向它（`see <主家路径>`），不复制文件。
4. **`.scratch/**/evidence/**` 原地不动**：它是证据锚点（`@<sha>`）指向的原始落盘证据，
   迁移时**只更新 `project.yaml` 里的指针**，不搬动 `.scratch`。
5. **删除项只删结构位**：任何「删」都只删旧表格/旧页的**结构位置**，证据进 Backlog（`status: 未开始`）保留。
6. **`archive/` 不动**。

## 1. 旧格 → 新家（37 格）

| 旧格 | 旧路径 | 新家（id → 目录） | 处置 | 证据去向 | 55-review 条目 |
|---|---|---|---|---|---|
| A2 全注意力基线 | `axes/A-mixing/A2-full-attention` | 参照 → `axes/7-references/README.md` | 降参照 | 目录整搬 `axes/7-references/`；`.scratch/04-m1-runs/evidence/B0` 留原地作锚 | A-P08, A-R05 |
| A3 窗口 / 锚点 / sink | `axes/A-mixing/A3-window-anchor` | **拆** → `1.1.1` + `1.1.3` | 拆 | `experiments/2505-m1-followups` + `.scratch/07-x1-x3/evidence` → **1.1.1**（窗口/锚点；主家）；1.1.3 只指针 | A-P07, A-P09, A-P14, A-P18, A-P21, A-R05, A-R06 |
| A4 软距离惩罚（ALiBi 式加性偏置） | `axes/A-mixing/A4-soft-distance-penalty` | `1.1.2` soft 衰减 | 迁移（手段/参照性质） | `experiments/2507-x1-x3-distance-penalty` + `.scratch/07-x1-x3/evidence/attn_dist_telemetry.json` | A-P10, A-R05 |
| A6 混合算子 | `axes/A-mixing/A6-mixing-operator` | **拆** → `1.2.2` + `1.2.3`（+ 长T窗口系列归 `1.1.1`） | 拆 | `2614-topk` → **1.2.3**；`2616-linear-attention`、`2637-linear-attn-quality` → **1.2.2**；`2622/2623/2624/2625/2626/2628/2629/2636`（长T=位置窗口，A3 族）→ **1.1.1** | A-P11, A-P12, A-P19, A-R02, A-R03, 55-R07 |
| A8 位置语义 | `axes/A-mixing/A8-positional-semantics` | 参照 → `axes/7-references/`（另见 `1.3` 寻址前提） | 降参照 | `.scratch/07-x1-x3/evidence`；无自有 experiments | A-P13 |
| A9 旧信息处理策略 | `axes/A-mixing/A9-old-info-retention` | `1.1.4.5` implicit vs explicit | 迁移 | `2615-window-trained-retention-13m`、`2623-window-on-longrange-13m`；`.scratch/13-four-experiments/evidence` | A-P14 |
| A10 压缩触发与优先级 | `axes/A-mixing/A10-compression-trigger` | `1.1.4.3` 压缩触发与优先级 | 迁移 | `2615-priority-shootout-13m` | A-P15, A-P16, A-P23 |
| C1 每个 token 跑几层 | `axes/C-vertical/C1-layers-per-token` | `2.1.1` early exit | 迁移（核心） | `2523-depth-ar-first`、`2523d-residual-tax-decomposition`、`2621-depth-draft-e2e`；R7 的 `2623-standard-early-exit` 作对照指针 | C-P01, C-P02, C-P03 |
| C2 选哪些层与执行顺序 | `axes/C-vertical/C2-layer-selection-order` | `2.1.2` skipping（跳层） | 并 | `2506-m2-element-tax-law`、`2509-m3-composition-law` | C-P0x（C 轴重要项） |
| C3 层间拓扑（残差之外的连接） | `axes/C-vertical/C3-inter-layer-topology` | `2.2.3` 层间学习加权和 + `2.2.4` 并行子层 | 并（负结论入 Backlog note） | `2613-inter-layer-topology-13m` | C-P0x |
| C4 权重共享 | `axes/C-vertical/C4-weight-sharing` | `2.2.5` 权重共享（Backlog） | Backlog | 无自有 experiments（本地零经验） | C 轴批注 C4 |
| C5 精化算子 | `axes/C-vertical/C5-refinement-operator` | `2.1.1` WP「精化门」 | 并（手段） | `2523c-three-ablations` | C-P0x |
| C6 训练期深度分布与监督密度 | `axes/C-vertical/C6-depth-supervision` | `2.1.1` WP「深度监督配方」 | 并（手段） | `2518-exit-retry-law`、`2524-depth-ar-100m` | C-P0x |
| H1 每步输出位置数 | `axes/H-horizon/H1-outputs-per-pass` | `6.2` 多位置预测 = 投机解码 | 迁移（核心） | `2510-gap-sweep-b2-c1m`、`2520-spec-at-scale` | H-F01, H-F03, H-F04, H-I1–I8 |
| H2 训练时未来位置的内容填充 | `axes/H-horizon/H2-training-fill` | `6.4` 训练时未来位置的填充语义 | 并 | `2523e-fill-variants` | H-I7, H-S5 |
| H3 多位置预测的组合方式 | `axes/H-horizon/H3-composition` | `6.2` WP「6.2.2 概率多候选 / 树」 | 并（手段，当前被简化，需补） | `2512-overnight-causal-chain`、`2613-self-gated-composition-13m`、`2631-spec-three-arms`、`2632-alpha1-attribution` | H-F02, H-F05, 批注 B |
| H4 头与主干的关系 | `axes/H-horizon/H4-head-trunk-relation` | `6.2` WP「6.2.1 草稿头」 | 并（手段） | `2521-exit-scale-confirm`、`2610-frozen-trunk-jumphead-13m`、`2611-jumphead-ar-control-13m` | H-F04, H-I6, 批注 A |
| H5 验证机制 | `axes/H-horizon/H5-verification` | `6.2` WP「6.2.3 验证」 | 并（手段） | `2513-four-experiments` | H-I6 |
| W1 激活宽度 | `axes/W-width/W1-active-width` | `3.1.1` uniform 减宽 / width dropout | 迁移 | `2525-tax-attribution` | W-06, W-07, W-14, W-17 |
| W2 专家化（MoE） | `axes/W-width/W2-expertization` | `4.1` 稀疏激活专家（top-k 路由） | 迁移（并入 R6-moe-scale） | 无自有 experiments | W-03, W-04, W-13 |
| W3 KV 头共享 | `axes/W-width/W3-kv-head-sharing` | 参照 → `axes/7-references/` | 降参照 | `2616-gqa-mqa-13m` | W-03, W-08, W-16 |
| W4 读出几何 | `axes/W-width/W4-readout-geometry` | `3.2.1` 读出的重参数化 | 迁移 | `2610-gated-readout-13m-depthar`、`2611-equalparams-linear-control-13m`、`2612-gated-readout-24k-13m`、`2613-readout-reparam-dense-13m`、`2621-per-depth-readout-13m` | W-10, W-11 |
| F2 回传通道 | `axes/F-iteration/F2-return-channel` | `9.2` 迭代精化的其余（原 F 轴剩余） | Backlog | `2511-retry-matrix-withdrawal`（旧引擎表述） | F-01, F-12, F-19, F-21, F-22 |
| F4 迭代/重复的训练信号 | `axes/F-iteration/F4-training-signal` | `9.2` 迭代精化的其余（原 F 轴剩余） | Backlog | `2619-reconstruction-loss-13m` | F-07, F-08, F-15 |
| M1 状态载体类型 | `axes/M-carrier/M1-carrier-type` | `1.1.4.1` 显式记忆槽 / 线性状态 / 递归状态 | 迁移（核心） | 全部 M1 `experiments/**`（22 个目录，`2503`…`2635`） | M-01…M-25 |
| M2 载体压缩 | `axes/M-carrier/M2-carrier-compression` | `1.1.4.2` 压缩：摘要槽 / 低秩投影 / pooling | 迁移 | `2614-kv-ablation-retention-13m`、`2615-retention-retrieval-leg-13m` | M-E（M2 一致性项） |
| G1 路径选择器 | `axes/G-policy/G1-selector` | `9.1` 调度 / 选择器（原 G 轴） | Backlog | `2508-tts-probes`、`2514-allocator-selectivity`、`2515-slider-rung3`、`2517-slider-at-scale` | G-P01–P18 |
| G3 观测信号 | `axes/G-policy/G3-signal` | `9.1`（并） | Backlog | `2522-gen-quality-signals` | G-P05, G-P14 |
| G4 校准 | `axes/G-policy/G4-calibration` | `9.1`（并） | Backlog | 无自有 experiments | G-P05, G-P07 |
| R1 量化 | `axes/R-references/R1-quantization` | `5.1` KV 量化（内容另分 `5.2`/`5.3`） | 迁移 | `2619`–`2632` 六个 kv 目录（`2614-kv-quant-eval`、`2619-kv-budget-chart`、`2620-kv-budget-gaps`、`2621-kv-budget-rest`、`2631-kv-budget-causal`、`2632-kv-block-scale`） | （55-review 未单列 R 轴；口径见 `.scratch/51-tree/OPERATIONS.md` R1 行） |
| R4 窗口注意力 + KV 量化 + 检索 | `axes/R-references/R4-window-kv-quant-retrieval` | `5.6` 组合：窗口 + 量化 + 检索 | 迁移 | `2617-prior-art-hybrid-kv` | （同上） |
| R7 既有早退实现（CALM / LayerSkip / MoD） | `axes/R-references/R7-existing-early-exit` | 参照 → `axes/7-references/` | 降参照 | `2623-standard-early-exit-13m`（同时作 `2.1.1` 的对照指针） | 55-R07 |
| P1 精度（修饰符） | `axes/P-modifiers/P1-precision` | `8` 工程纪律 | 工程纪律 | 无自有 experiments | 55-TICKET §2（R5/R6 残留） |
| P3 参数插入与初始化（修饰符） | `axes/P-modifiers/P3-param-insertion-init` | `8` 工程纪律 | 工程纪律 | 无自有 experiments | 同上 |
| P4 归一化位置（修饰符） | `axes/P-modifiers/P4-normalization` | `8` 工程纪律 | 工程纪律 | 无自有 experiments | 同上 |
| P5 损失权重与调度（修饰符） | `axes/P-modifiers/P5-loss-weighting` | `8` 工程纪律 | 工程纪律 | `2523b-shallow-weight-sweep` | 同上 |
| P6 数据与任务构造（修饰符） | `axes/P-modifiers/P6-data-construction` | `8` 工程纪律 | 工程纪律 | 无自有 experiments | 同上 |

## 2. 缺（新建研究项，无旧格）→ 新家

| 新家 | 来源 |
|---|---|
| `1.2.1` dilated / 空洞 | 无旧格；F 轴/情报缺口 |
| `1.2.4` 动态合并：越往下层 token 越少 | 无旧格 |
| `1.1.4.4` 外部检索 | 无旧格（A9 options 覆盖方向） |
| `2.2.1` 残差的形式 | 无旧格 |
| `2.2.2` 用 causal attention 代替残差 | 无旧格（C 轴批注 C3） |
| `2.3.1` 逐层容量变化 | 无旧格 |
| `2.3.2` 逐层分辨率 | 无旧格 |
| `3.1.2` 逐层宽度变化 | 无旧格 |
| `3.1.3` 同层内位置相关宽度 | 无旧格 |
| `3.1.4` 前馈隐藏比 | 无旧格 |
| `3.1.5` 通道极化 / 稀疏激活 | 无旧格 |
| `3.2.2` 逐出口读出 | 无旧格 |
| `3.2.3` tied vs untied embedding | 无旧格 |
| `4.2` 路由与负载均衡 | 无旧格（W2 只覆盖 4.1） |
| `4.3` 专家粒度 / 异构 | 无旧格 |
| `5.2` 权重量化 / `5.3` 激活量化 / `5.5` fp8 / 大 head_dim | 无旧格 |
| `6.3.1` diffusion / `6.3.2` 一次性批量 pad 填充 / `6.3.3` latent regression 批量生成 | 无旧格 |
| `2.1.1` WP「退出决策头」「异构深度回看」 | 无旧格（出资人 C 轴批注的缺口） |
| `6.2` WP「6.2.2 概率多候选 / 树」 | 由 H3 补全（当前被简化） |

## 3. 拆 / 并的证据归属（一格一份家）

- **A3 → 1.1.1 + 1.1.3**：`A3/experiments/**` 全部 → **1.1.1**（窗口/锚点主家）；
  `1.1.3` 仅取「可见量上限」的判据/选项叙述，**不接收文件**，用指针 `see 1.1.1`。
- **A6 → 1.2.2 + 1.2.3 + 1.1.1**：`2614-topk` → **1.2.3**（内容选择 top-k）；
  `2616-linear-attention`、`2637-linear-attn-quality` → **1.2.2**（算子）；
  长T窗口系列 `2622/2623/2624/2625/2626/2628/2629/2636` → **1.1.1**（位置窗口，A3 族）。
  这三个新家之外，**不复制**任何 A6 文件。
- **H 族 → 6.2**：H1（本体）+ H3/H4/H5（手段）的 experiments 全部落 `6.2/experiments/**`；
  H3/H4/H5 作为 `6.2` 的 `work_packages` 记录，不各立研究项目录。
- **M1 → 1.1.4.1**：M1 的全部 22 个实验目录落 `1.1.4.1/experiments/**`。

## 4. 参照 / 工程 / Backlog 的落点

- **参照（轴 7）**：A2、A8、W3、R7 —— 证据整搬 `axes/7-references/`（含各自 experiments），
  在 `README.md` 台账里登记为「必须打败的跑分线」。
- **工程纪律（轴 8）**：P1/P3/P4/P5/P6 —— 台账 `axes/8-engineering/README.md`；
  P5 的 `2523b-shallow-weight-sweep` 一并搬入。
- **Backlog（轴 9 + 本轴）**：2.1.3 repeating、2.2.5 权重共享（挂 `axes/2-depth/`）；
  `9.1`（原 G）、`9.2`（原 F 剩余）。状态 `未开始`，证据一律保留。

## 5. 55-review 根因 → 处置（逐条）

| 根因（55 TICKET §1） | 处置 |
|---|---|
| R1 目的丢失（手段当 topic） | 新 schema 强制 `purpose` + `innovation` + `sota_baseline` + `gap`（`docs/research-schema.md` §2）；缺则校验 fail |
| R2 无 pareto 口径 | `validation` 强制「与 SOTA/stock 基线对照 + 客观判据 + 口径」 |
| R3 判据与目的错位 / 恒真 | `objective_criteria` 强制写「该涨/该跌/必须不变」；`falsification` 强制写否证条件 |
| R4 造轮子 | `sota_baseline` + `gap` 强制声明「相对 SOTA 的增量」 |
| R5 结论悬空 / 证据不指向实验 | `status ∈ {已完成,被否证}` 强制 `conclusion` + **存在的** `experiments` |
| R6 上一时代残留 | 各轴 worker 按本表迁入新家并清残留（`*.md.md`、旧引擎引用、死数据等） |
| R7 结构错（一格塞多点 / 该并 / 参照混入） | 本文件 §1 的拆/并/降参照；轴 7 收参照、轴 8 收工程、轴 9 收 Backlog |
| R8 跨引擎 / 跨口径混用 | `cost_model` 禁墙钟；口径纪律沿用 `SCHEMA.md` §6–§24 |

## 6. 工作树既有未提交改动的处置

迁移开始时工作树已有未提交改动（**保留、不回滚**）：`catalog.yaml`、`INDEX.md`、
`axes/C-vertical/{README,C1-layers-per-token/README}.md`（C1 的判据/结论修订）。
处置：这些内容**已被新树取代**（C1 → `2.1.1`）；其修订文本**保留在 git 工作树**，
由**轴 2 worker** 在迁 `2.1.1` 时**照准迁入**（判据 = 序列级 perplexity/一致率 vs 全深度、
算力 = layer-passes/token、墙钟不进判据），然后随旧格一并退役。本轮 foundation **不提交、
不回滚**这些文件。

## 7. 本轮 foundation 已建的脚手架

- `axes/1-time/1.1.1-hard-window/project.yaml`（exemplar，按 A3 迁移 + 55-review 修正）。
- `axes/9-backlog/9.1-selector-scheduling/project.yaml`（Backlog exemplar）。
- 轴 worker **扩展**这两个文件，**不要重建**；其余研究项由轴 worker 按本表新建。
