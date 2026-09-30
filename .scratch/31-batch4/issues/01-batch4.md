# 01 — 第四批（ticket 31）：先造天平（保留曲线） + 稀疏算子首探 + 参照线

用户离场授权：主 Agent 编排、子 Agent 执行、小规模单因素、判据预登记、每研究必独立复核。
统一纪律见前批，并**先读 `SCHEMA.md §17 契约模板易错点`**（本批契约已按它检查过：
不引入需要深度监督的判据、Δ 符号统一为 `Δ = bpc_对照 − bpc_实验`（正=改善）、
活性量要求每 200 步落盘、不写 20 步外推时间字段）。

---

## 契约 I：KV 消融保留曲线（仪器建设，纯推理）

- 目录：`axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/`
- 文件所有权：可新增 `tools/evalkit/retention.py`（**新建**，不碰既有文件）、
  `tests/test_retention.py`、本实验目录。
- 动机：`A9/A10/M2` 三个格子都需要同一件天平——「丢掉某些位置的信息，损伤有多大」。
  本批先造这件天平并给出**基线数字**。它同时正面回答用户最初的问题：
  **左前缀到底能不能丢、丢多少才开始疼。**
- 被测量模型（复用，核对 sha256）：`axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
  （13.5M dense，12k 步，bpc 1.4836）；若预算允许再加一个 depth-ar 形态的 checkpoint 作对照。
- 方法（同一数据流、同一位置集、只改「哪些位置的 KV 被删」）：
  1. **距离曲线**：对每个距离 d ∈ {1,2,4,8,16,32,64,128,255}，删除「与查询距离恰为 d」的位置的 KV，测 Δbpc；
  2. **前缀截断曲线**：删除最远的 N 个位置（N ∈ {8,16,32,64,128,192,255}），测 Δbpc → 这是「左前缀可丢多少」的直接答案；
  3. **连续段删除**：删一段连续位置（随机起点，长度 L ∈ {4,16,64}），测 Δbpc（区分「按距离」与「按内容重要性」）；
  4. 可选：同一实验下的 attention-mass 统计（哪些位置真的被读到）。
- **判据（预登记；仪器类，判据是「仪器是否可信 + 基线数字」）**：
  - (a) 三条曲线**单调**（距离越远/丢得越多，损伤不应更小）；
  - (b) **复现性**：同 seed 重跑 Δbpc 差异 < 0.005；
  - (c) 输出两个基线数字：**安全丢弃半径** d*（Δbpc < 0.014 的最大 d）与**「丢最远 N 个位置」的 Δbpc 曲线**；
  - (d) 若任一曲线非单调且无法用噪声解释 ⇒ 仪器不可信，先修仪器（记 ERRATA）。
- 成本：纯推理 ≈15–25 min（T=256，20 个 eval batch 级别）。
- 交付：契约 README / `tools/evalkit/retention.py` / `run.py` / `config.json` /
  `results.json`（三条曲线 + 安全半径）/ `analysis.md`（人话摘要 + 多维表 + 对账 + 口径）/
  `tests/test_retention.py`（≥4 可失败单测，含「删除位置集合是否正确」的变异验证）。
- 注：本格 `M2` 的既有 evidence 与 A3/A9/A10 三格将在主 Agent 收到结果后回填。

## 契约 J：Top-K 稀疏注意力 vs 全注意力（A6 混合算子首探）

- 目录：`axes/A-mixing/A6-mixing-operator/experiments/2614-topk-sparse-attn-13m/`
- 文件所有权：可改 `pathlm/depth_ar.py`（新增 top-k 稀疏注意力开关）、`train_depth_ar.py`、
  `tests/test_depth_ar.py`、本实验目录。
- 干预（唯一变量 = 注意力算子）：`dense` 模型（`--dense`）下，
  - `full`（基线）：全因果 softmax 注意力；
  - `topk`：每个查询只保留**与自身内容最相似的前 k 个**键（k = 32 / 64 两档；因果掩码内选择），
    其余置 −inf；选择按内容（不按位置）。除了注意力算子，其余逐字段一致（seed=0、12k 步）。
- **判据（预登记）**：
  - 主判据：`topk` 的 clean bpc **不劣于** `full` 超过 0.014（即 |Δ| < 0.014 ⇒ 与全注意力同质量）；
  - 并列（成本）：报告注意力算力/显存的**实测**下降比例（k/T 的理论值与实现实测值都要）；
  - 若 bpc 改善 ≥0.014 ⇒ 更好（少见，但记录）；
  - 若 bpc 劣化 ≥0.014 ⇒ 稀疏选择在该规模有害，记录 k 与劣化幅度。
  - 必须附两臂**同方法** loss 斜率（§16）与重跑散度。
- 成本：2 臂 × ≈7 min（dense 实测 6.5 min/臂）+ 评测 ≈30 min 总。
- 交付：统一清单 + 成本口径（理论/实测）。

## 契约 K：KV 量化（参照线 R1/M2，纯推理）

- 目录：`axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/`
- 文件所有权：可新增 `tools/evalkit/kv_quant.py`、`tests/test_kv_quant.py`、本实验目录。
- 动机：**任何「省内存/丢信息」的主张都必须打败「把 KV 量化」。** 先测出这条参照线的代价。
- 方法：复用契约 I 的同一 checkpoint（核对 sha256），只改 KV cache 的数值精度：
  `fp16`（基线）/ `int8`（对称 per-channel 量化）/ `int4`（同上）；测 Δbpc 与内存占用。
  实现须是**评测期**量化（不改模型权重、不训练）。
- **判据（预登记）**：给出 int8/int4 的 Δbpc 与内存下降；若 int8 的 |Δbpc| < 0.014 ⇒
  记「近无损」（写进参照线，后续所有压缩主张必须与之比较）；若 int4 劣化 ≥0.014 ⇒ 记其代价。
- 成本：纯推理 ≈10 min。
- 交付：契约 README / `tools/evalkit/kv_quant.py` / `run.py` / `config.json` / `results.json` /
  `analysis.md` / 单测（≥3，含「量化-反量化往返误差上界」与变异验证）。

## 契约 L：独立复核（fresh context，只读）

- agent：`reviewer`；对象：契约 I/J/K。
- 清单：`SCHEMA.md` §14 七项 + 本轮特别关注：
  ① 仪器类判据（单调性/复现性）是否有数据支持；② 删除位置集合是否**真的**只删了该位置
  （变异验证是否覆盖）；③ top-k 选择是否真按内容、k 是否真的生效（活性证据）；
  ④ 量化是否真的降低了 KV 精度（活性证据）而非只改标签；⑤ 是否有算术不自洽的数字。
- 输出：逐条 pass/fail + 可疑数字清单；用 `contact_supervisor` 回传，由主 Agent 落盘。
