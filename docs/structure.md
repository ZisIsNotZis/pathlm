# 研究结构（SSOT · 规范用词）

> 本文件是**研究轴的唯一权威定义**：轴名、研究项名、子项名，一律以本文件为准。
> **exact wording 不得改**；后续文档（catalog / axes / ticket / 实验契约）只能引用本文件的用词。
> 目录沿用 `axes/`；每个研究项一份 YAML 文档（字段见 §10）。
> 目标：用更少的计算量拿到**序列级**类似效果；代价口径 = 等效理论时间/计算量（FLOPs / layer-passes /
> tokens-per-forward，考虑 pipelining），**不用墙钟**。

---

## 1 时间轴
一次前向里，一个 token「能看到谁、怎么用」。

### 1.1 可见范围（窗口族）
- **1.1.1 hard 窗口**：滑动窗口、sink、锚点槽
- **1.1.2 soft 衰减**：距离加性偏置（ALiBi 式）、指数衰减、软惩罚（软窗口）
- **1.1.3 可见量上限**：每个 token 最多能看多少
- **1.1.4 信息右移**（窗口族的子项：窗口丢掉的，怎么搬近）
  - 1.1.4.1 显式记忆槽 / 线性状态 / 递归状态（delta 规则族：DeltaNet / GLA / Mamba-2）
  - 1.1.4.2 压缩：摘要槽 / 低秩投影 / pooling
  - 1.1.4.3 压缩触发与优先级
  - 1.1.4.4 外部检索
  - 1.1.4.5 implicit（掩码/窗口，已证退化）vs **explicit**（有希望的方向）

### 1.2 可见结构（非均匀的可见模式）
- **1.2.1 dilated / 空洞**
- **1.2.2 块稀疏 / 条带 / 局部+全局混合**（Longformer 式）
- **1.2.3 内容选择 top-k**
- **1.2.4 动态合并**：越往下层 token 越少

### 1.3 寻址（前提，不是手段选择）
- 位置语义：绝对 / 相对 / RoPE / 无位置
- 寻址方式：按内容 vs 按位置

---

## 2 深度轴
### 2.1 按需深度
- **2.1.1 early exit**（最重要）
  - 退出决策头：概率 + 校准
  - 异构深度回看：前浅后深，往前 attend 看到什么
  - 深度监督配方
  - 精化门
- **2.1.2 skipping**（跳层）
- （2.1.3 repeating 循环 transformer：多算力换少参数 —— Backlog，不进编号）

### 2.2 层间怎么聚合（残差族）
- **2.2.1 残差的形式**：pre/post-norm、缩放残差、门控残差、残差随深度衰减
- **2.2.2 用 causal attention 代替残差**：层间对前面所有层的输出做一次 causal attention 聚合
- **2.2.3 层间学习加权和**（DenseFormer 式）
- **2.2.4 并行子层**
- （2.2.5 权重共享 —— Backlog）

### 2.3 深度的形状
- **2.3.1 逐层容量变化**：前大后小 / 逐渐放大
- **2.3.2 逐层分辨率**：先粗后细、层级式（层级合并）

---

## 3 宽度轴
「非均匀宽度」的 5 种读法（定义，防歧义）：
(a) 逐层不同（第 1 层宽、后面窄，或反过来）｜(b) 同层内不同位置不同（前 1024 vs 后 1024）｜
(c) 同层内通道极化｜(d) 前馈隐藏比（FF/MLP hidden ratio）｜(e) 只激活一部分通道

### 3.1 每 token 的激活宽度
- **3.1.1 uniform 减宽 / width dropout**
- **3.1.2 逐层宽度变化**：金字塔 / 沙漏
- **3.1.3 同层内位置相关宽度**：前 1024 与后 1024 意义不同
- **3.1.4 前馈隐藏比**：FF/MLP 的 hidden ratio，及其逐层变化
- **3.1.5 通道极化 / 稀疏激活**：只激活部分通道

### 3.2 读出
（读出 = 把隐藏状态变成预测的那一层，也就是 unembedding / 输出层。）
- **3.2.1 读出的重参数化**：加非线性 / GLU 式 / 线性
- **3.2.2 逐出口读出**：early exit 或多出口时，每个出口怎么映射到词表
- **3.2.3 tied vs untied embedding**

---

## 4 专家轴（MoE）
- 4.1 稀疏激活专家（top-k 路由）
- 4.2 路由与负载均衡
- 4.3 专家粒度 / 异构（细粒度、共享专家）

---

## 5 量化轴
- 5.1 KV 量化
- 5.2 权重量化
- 5.3 激活量化
- 5.4 低秩压缩（MLA 式）
- 5.5 fp8 / 大 head_dim
- 5.6 组合：窗口 + 量化 + 检索

---

## 6 预测轴
- **6.1 自回归**（一次一个位置）
- **6.2 多位置预测 = 投机解码**
  - 6.2.1 草稿头（应免费、不动主干）
  - 6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合）
  - 6.2.3 验证
- **6.3 整段并行生成（不止 AR）**
  - 6.3.1 diffusion（离散扩散 / 掩码扩散）
  - 6.3.2 一次性批量 pad 填充（填空式）
  - 6.3.3 latent regression 批量生成（估计得贪心）
- **6.4 训练时未来位置的填充语义**：pad / self-fill / 错误 token

---

## 7 不研究（只当对照组）
全注意力基线、ALiBi、RoPE、GQA/MQA、量化基线、CALM/LayerSkip/MoD。
说人话：它们拿来当跑分线，不立项。

## 8 工程纪律（不是研究）
精度、初始化、归一化、损失权重、数据构造。

## 9 Backlog（不删，移出主线）
循环 transformer（重复层）、层间拓扑、权重共享、调度 / 选择器（原 G 轴）、迭代精化的其余（原 F 轴剩余）。

---

## 10 每个研究项的 YAML 字段
```yaml
id:            # 编号，如 "1.1.4.1"
axis:          # 轴名（本文件用词）
title:         # 研究项名（本文件用词）
purpose:       # 要解决的痛点，一句话
innovation:    # 相对普通 decoder-only transformer 改了什么
expected_effect: # 预期效用
validation:    # 端到端验证：判据 + 对照 + 口径
cost_model:    # 等效理论时间/计算量（FLOPs / layer-passes / tokens-per-forward），考虑 pipelining；不用墙钟
milestone: []  # 里程碑与验收点
roadmap: []    # 推进步骤
status:        # 未开始 / 进行中 / 已完成 / 被否证
conclusion:    # 结论
conclusion_review: # 这个结论看上去对吗
experiments: []# 实验目录
argument:      # 怎么论证的
gaps: []       # 遗漏
started:       # 时间
ended:         # 时间
change_log: [] # 变更
```

> 上表是**最小字段集**；**完整、权威**的字段/取值/纪律见 `docs/research-schema.md`。

## 11 汇总脚本
小脚本扫所有 `project.yaml` → 输出表：`id / 轴 / 起 / 止 / 完成? / 状态 / 结论 / 结论对不对 / 实验 / 论证 / 遗漏`。
无网页、无缓存、无 fancy。

（实现：`python3 tools/summary.py`；`--check` 校验字段/证据，非零退出。校验器 `tools/check_tree.py`；
旧入口 `tools/check_schema.py` 转调它。）

---

## 12 节点清单（冻结 LAYOUT · 轴 → 研究项 → 子项）

> 本表是研究树的**冻结目标布局**：每个节点的**精确 `id` / 精确标题用词 / 轴 / 目录**。
> 用词一律等于本文件 §1–§6 的 exact wording，不得改名、不得自造。
> 「粒度」= `研究项`（写 `project.yaml`）/ `分组`（只写聚合 `README.md`）/ `参照`（进 `7-references`）/
> `工程`（进 `8-engineering`）/ `Backlog`（研究项形式、`status: 未开始`）。粒度判据见 §13。

### 轴 1 · 时间轴（目录 `axes/1-time/`）

| id | 标题（exact wording） | 目录 | 粒度 |
|---|---|---|---|
| 1.1 | 可见范围（窗口族） | `axes/1-time/1.1-visibility/` | 分组 |
| 1.1.1 | hard 窗口 | `axes/1-time/1.1.1-hard-window/` | 研究项 |
| 1.1.2 | soft 衰减 | `axes/1-time/1.1.2-soft-decay/` | 研究项 |
| 1.1.3 | 可见量上限 | `axes/1-time/1.1.3-visibility-cap/` | 研究项 |
| 1.1.4 | 信息右移 | `axes/1-time/1.1.4-right-shift/` | 分组 |
| 1.1.4.1 | 显式记忆槽 / 线性状态 / 递归状态（delta 规则族：DeltaNet / GLA / Mamba-2） | `axes/1-time/1.1.4.1-explicit-memory/` | 研究项 |
| 1.1.4.2 | 压缩：摘要槽 / 低秩投影 / pooling | `axes/1-time/1.1.4.2-compression/` | 研究项 |
| 1.1.4.3 | 压缩触发与优先级 | `axes/1-time/1.1.4.3-compression-trigger/` | 研究项 |
| 1.1.4.4 | 外部检索 | `axes/1-time/1.1.4.4-external-retrieval/` | 研究项 |
| 1.1.4.5 | implicit（掩码/窗口，已证退化）vs **explicit**（有希望的方向） | `axes/1-time/1.1.4.5-implicit-vs-explicit/` | 研究项 |
| 1.2 | 可见结构（非均匀的可见模式） | `axes/1-time/1.2-visibility-structure/` | 分组 |
| 1.2.1 | dilated / 空洞 | `axes/1-time/1.2.1-dilated/` | 研究项 |
| 1.2.2 | 块稀疏 / 条带 / 局部+全局混合（Longformer 式） | `axes/1-time/1.2.2-block-sparse/` | 研究项 |
| 1.2.3 | 内容选择 top-k | `axes/1-time/1.2.3-content-topk/` | 研究项 |
| 1.2.4 | 动态合并：越往下层 token 越少 | `axes/1-time/1.2.4-dynamic-merge/` | 研究项 |
| 1.3 | 寻址（前提，不是手段选择） | `axes/1-time/1.3-addressing/` | 参照 |

### 轴 2 · 深度轴（目录 `axes/2-depth/`）

| id | 标题（exact wording） | 目录 | 粒度 |
|---|---|---|---|
| 2.1 | 按需深度 | `axes/2-depth/2.1-ondemand-depth/` | 分组 |
| 2.1.1 | early exit（最重要） | `axes/2-depth/2.1.1-early-exit/` | 研究项 |
| 2.1.2 | skipping（跳层） | `axes/2-depth/2.1.2-skipping/` | 研究项 |
| 2.1.3 | repeating 循环 transformer：多算力换少参数 | `axes/2-depth/2.1.3-repeating/` | Backlog |
| 2.2 | 层间怎么聚合（残差族） | `axes/2-depth/2.2-layer-aggregation/` | 分组 |
| 2.2.1 | 残差的形式 | `axes/2-depth/2.2.1-residual-form/` | 研究项 |
| 2.2.2 | 用 causal attention 代替残差 | `axes/2-depth/2.2.2-attention-residual/` | 研究项 |
| 2.2.3 | 层间学习加权和（DenseFormer 式） | `axes/2-depth/2.2.3-learned-layer-sum/` | 研究项 |
| 2.2.4 | 并行子层 | `axes/2-depth/2.2.4-parallel-sublayer/` | 研究项 |
| 2.2.5 | 权重共享 | `axes/2-depth/2.2.5-weight-sharing/` | Backlog |
| 2.3 | 深度的形状 | `axes/2-depth/2.3-depth-shape/` | 分组 |
| 2.3.1 | 逐层容量变化：前大后小 / 逐渐放大 | `axes/2-depth/2.3.1-layer-capacity/` | 研究项 |
| 2.3.2 | 逐层分辨率：先粗后细、层级式（层级合并） | `axes/2-depth/2.3.2-layer-resolution/` | 研究项 |

### 轴 3 · 宽度轴（目录 `axes/3-width/`）

| id | 标题（exact wording） | 目录 | 粒度 |
|---|---|---|---|
| 3.1 | 每 token 的激活宽度 | `axes/3-width/3.1-active-width/` | 分组 |
| 3.1.1 | uniform 减宽 / width dropout | `axes/3-width/3.1.1-uniform-narrowing/` | 研究项 |
| 3.1.2 | 逐层宽度变化：金字塔 / 沙漏 | `axes/3-width/3.1.2-layer-width/` | 研究项 |
| 3.1.3 | 同层内位置相关宽度：前 1024 与后 1024 意义不同 | `axes/3-width/3.1.3-position-width/` | 研究项 |
| 3.1.4 | 前馈隐藏比：FF/MLP 的 hidden ratio，及其逐层变化 | `axes/3-width/3.1.4-ffn-ratio/` | 研究项 |
| 3.1.5 | 通道极化 / 稀疏激活：只激活部分通道 | `axes/3-width/3.1.5-channel-sparsity/` | 研究项 |
| 3.2 | 读出 | `axes/3-width/3.2-readout/` | 分组 |
| 3.2.1 | 读出的重参数化：加非线性 / GLU 式 / 线性 | `axes/3-width/3.2.1-readout-reparam/` | 研究项 |
| 3.2.2 | 逐出口读出：early exit 或多出口时，每个出口怎么映射到词表 | `axes/3-width/3.2.2-per-exit-readout/` | 研究项 |
| 3.2.3 | tied vs untied embedding | `axes/3-width/3.2.3-tied-untied/` | 研究项 |

### 轴 4 · 专家轴（MoE）（目录 `axes/4-moe/`）

| id | 标题（exact wording） | 目录 | 粒度 |
|---|---|---|---|
| 4.1 | 稀疏激活专家（top-k 路由） | `axes/4-moe/4.1-sparse-experts/` | 研究项 |
| 4.2 | 路由与负载均衡 | `axes/4-moe/4.2-routing-balance/` | 研究项 |
| 4.3 | 专家粒度 / 异构（细粒度、共享专家） | `axes/4-moe/4.3-expert-granularity/` | 研究项 |

### 轴 5 · 量化轴（目录 `axes/5-quant/`）

| id | 标题（exact wording） | 目录 | 粒度 |
|---|---|---|---|
| 5.1 | KV 量化 | `axes/5-quant/5.1-kv-quant/` | 研究项 |
| 5.2 | 权重量化 | `axes/5-quant/5.2-weight-quant/` | 研究项 |
| 5.3 | 激活量化 | `axes/5-quant/5.3-activation-quant/` | 研究项 |
| 5.4 | 低秩压缩（MLA 式） | `axes/5-quant/5.4-low-rank/` | 研究项 |
| 5.5 | fp8 / 大 head_dim | `axes/5-quant/5.5-fp8-head-dim/` | 研究项 |
| 5.6 | 组合：窗口 + 量化 + 检索 | `axes/5-quant/5.6-window-quant-retrieval/` | 研究项 |

### 轴 6 · 预测轴（目录 `axes/6-predict/`）

| id | 标题（exact wording） | 目录 | 粒度 |
|---|---|---|---|
| 6.1 | 自回归（一次一个位置） | `axes/6-predict/6.1-autoregressive/` | 参照 |
| 6.2 | 多位置预测 = 投机解码 | `axes/6-predict/6.2-speculative-decoding/` | 研究项 |
| 6.3 | 整段并行生成（不止 AR） | `axes/6-predict/6.3-parallel-generation/` | 分组 |
| 6.3.1 | diffusion（离散扩散 / 掩码扩散） | `axes/6-predict/6.3.1-diffusion/` | 研究项 |
| 6.3.2 | 一次性批量 pad 填充（填空式） | `axes/6-predict/6.3.2-pad-fill/` | 研究项 |
| 6.3.3 | latent regression 批量生成（估计得贪心） | `axes/6-predict/6.3.3-latent-regression/` | 研究项 |
| 6.4 | 训练时未来位置的填充语义：pad / self-fill / 错误 token | `axes/6-predict/6.4-training-fill/` | 研究项 |

> `6.2` 的 `work_packages` 用 `6.2.1 草稿头（应免费、不动主干）`、`6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合）`、
> `6.2.3 验证`（见 §13 粒度规则：这三者是**同一个投机解码研究问题的手段**，不另立研究项）。

### 轴 7 · 不研究（只当对照组）（目录 `axes/7-references/`）

| 参照（exact wording） | 台账 |
|---|---|
| 全注意力基线 | `axes/7-references/README.md` |
| ALiBi | 见 `1.1.2`（软衰减手段） |
| RoPE | `axes/7-references/README.md`（位置语义族） |
| GQA/MQA | `axes/7-references/README.md`（KV 头共享） |
| 量化基线 | 见 `5.1`–`5.6` |
| CALM/LayerSkip/MoD | `axes/7-references/README.md`（既有早退实现） |
| 自回归（一次一个位置） | `6.1` |

> 轴 7 是**参照台账**（`README.md`）：它们拿来当跑分线，不立项、不写 `project.yaml`。

### 轴 8 · 工程纪律（不是研究）（目录 `axes/8-engineering/`）

| 工程纪律 | 台账 |
|---|---|
| 精度 | `axes/8-engineering/README.md` |
| 初始化 | 同上 |
| 归一化 | 同上 |
| 损失权重 | 同上 |
| 数据构造 | 同上 |

> 轴 8 是**工程纪律台账**（`README.md`）：不是研究，不写 `project.yaml`。

### 轴 9 · Backlog（不删，移出主线）（目录 `axes/9-backlog/`）

| id | 标题 | 目录 | 粒度 |
|---|---|---|---|
| 9.1 | 调度 / 选择器（原 G 轴） | `axes/9-backlog/9.1-selector-scheduling/` | Backlog |
| 9.2 | 迭代精化的其余（原 F 轴剩余） | `axes/9-backlog/9.2-iteration-refinement/` | Backlog |

> §9 另点名的「循环 transformer（重复层）」见 `2.1.3`、「权重共享」见 `2.2.5`、「层间拓扑」
> 的负结论已入 `2.2.3` / `2.2.4`。Backlog 项**证据一律保留**（`status: 未开始`），不删。

## 13 粒度规则（什么写 `project.yaml`，什么只写 README）

1. **研究项（写 `project.yaml`）**：能**独立推进、可独立否证**、有独立 `purpose` + `innovation` +
   `validation` + 单一 `milestone` 的节点。§12 标「研究项 / Backlog」者。
2. **分组（只写聚合 `README.md`）**：只把子节点收拢的类目——轴本身，以及 `x.y`（如 `1.1`、`2.1`、
   `3.2`、`6.3`）。§12 标「分组」者。
3. **子做法 / 手段**：不另立节点，写进**父研究项的 `work_packages`**。已按此判定的有：
   `2.1.1` 的 退出决策头 / 异构深度回看 / 深度监督配方 / 精化门；`6.2` 的 6.2.1 / 6.2.2 / 6.2.3。
   判定理由：它们是**同一个研究问题的手段**（一个 `milestone` 就能收口），不是可各自独立收口的问题。
4. **参照 / 前提**：不写 `project.yaml`，进 `axes/7-references/README.md`（§12 标「参照」者：`1.3`、`6.1`）。
5. **工程纪律**：进 `axes/8-engineering/README.md`（§12 标「工程」者）。
6. **Backlog**：保留研究项形式（`status: 未开始`），`2.1.3` / `2.2.5` 挂本轴、其余进 `9-backlog`。

## 14 目录命名（`axes/` 布局）

- 轴目录 slug（统一记录，不得再改）：`1-time/ 2-depth/ 3-width/ 4-moe/ 5-quant/ 6-predict/ 7-references/ 8-engineering/ 9-backlog/`。
- 研究项目录：`axes/<轴 slug>/<id>-<slug>/`，`project.yaml` 放在其中（例 `axes/1-time/1.1.1-hard-window/project.yaml`）。
- 子项目录沿用父目录（如 `1.1.4.1` 的目录即 `axes/1-time/1.1.4.1-explicit-memory/`）。
- 旧目录 `axes/<A-mixing|C-vertical|…>/` 的**证据文件由迁移 worker 按 `MAPPING.md` 搬到新目录**（本文件不移动文件）。

## 15 退役与去留（冻结）

- **研究项 SSOT = `axes/**/project.yaml`**（格式见 `docs/research-schema.md`）。
- **退役**（不再维护、不再是 SSOT、不再生成网页）：
  - `catalog.yaml`：旧「目录完备」枚举。**迁移窗口内暂留在仓**——它的 `local:` 块是轴 worker
    迁移旧结论的**内容源**，中途删除会丢证据；**迁移完成后整体移入 `archive/wiki-legacy/`**（证据保留，不删）。
  - `tools/build_wiki.py`、`tools/check_catalog_strict.py`、`tools/backfill_legacy.py`：生成式
    wiki 工具，退役后一并移入 `archive/wiki-legacy/`。
  - `INDEX.md`：由 `build_wiki.py` 生成的旧状态灯，退役（状态改由 `tools/summary.py` 输出）。
- **取代**：
  - 校验入口 `tools/check_schema.py` → **转调** `tools/check_tree.py`（新校验器）。
  - 汇总入口 = `tools/summary.py`（一条命令出全表；`--check` 校验）。
- 出资人原话：「我不需要你给我 build 网页。或者一些 fancy 的玩意。你直接写一个小脚本，可以统计出
  所有这个子项目的状态就完事了。」
