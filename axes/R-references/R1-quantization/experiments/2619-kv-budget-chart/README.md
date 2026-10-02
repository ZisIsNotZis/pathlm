# 2619 · 统一 KV 预算参照线总图（ticket 36 波二 · U3 线）—— 实验契约

> 来源契约：`.scratch/36-parallel/issues/01-parallel.md` §「波二 U3 线」。
> 本文件是该线的**执行契约**；判据线**逐字**抄录在 §4，执行期间未改。
> 入口：`run.py`（纯推理 + 汇总，**不训练**）；仪器：`tools/evalkit/longrange_tasks.py`
> （合成长程任务电池）、`tools/evalkit/kv_quant.py`（KV 量化往返）、
> `tools/evalkit/retention.py`（可见性掩码 / 检索腿）。
> 被测量模型：三族 checkpoint 复用（只读，运行时断言 sha256）；见 `MODEL.md`。

---

## 1. 命题

把已经**分别测过**的四类 KV 预算手段（int8/int4 量化、GQA/MQA 头共享、驱逐/窗口、
不压缩基线）放到**同一个横轴「每位置 KV 字节（解析口径）」**上，并同时给出
**两条纵轴腿**（clean bpc 与合成长程召回），我们能不能得到一条**可比较的参照线**，
以及图上**哪些格子是空的**（下一批候选）？

可证伪点：若横轴口径不唯一（例如驱逐用「保留比例」、量化用「每值字节」各算各的），
或召回腿在各点落在地板/天花板（不可分辨），则这张总图**不成立**，必须如实报「不可分辨」。

---

## 2. 单因素声明

- 本实验**不训练**、不加参数、不改权重：它是**测量 + 汇总**类实验。
- 本实验**不是组合实验**：图上每个点都是**单机制、单因素**的已测（或本次重测）配置；
  本实验只把它们映射到**同一物理量**（每位置 KV 字节）与**同一仪器**（bpc / 合成召回）上。
- 组合配置（如 `GQA × int8`、`窗口 × 量化`、`窗口 × 锚点`）**未测**，是**显式声明**的
  留白，列在 `results.json::missing_combinations` 里作为下一批候选。
- 基线 = 主流标准 decoder-only transformer（13.5M dense：12 层 × d256 × 8 头 × mlp6 ×
  seq256，tied 读出、仅最终深度 CE）。三族 checkpoint 各自的自带基线：
  - **trunk 族**（量化 / 窗口）：`trunk.pt`（clean bpc 1.483569）；
  - **head 族**（MHA/GQA/MQA）：`mha-12k-s0/model.pt`（clean bpc 1.482026）。
  两族基线差 **0.0016**（同配置同 seed 重训的 run-to-run 散度），已单列，未混。

---

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| 横轴（唯一口径） | `bytes_per_position` = `2(K,V) × n_layers × n_kv_heads_eff × head_dim × value_bytes × (positions_kept / T)`，T = 256 | 本实验解析计算（`results.json::points[*].bytes_per_position`） |
| 代价（质量腿 1） | clean bpc（越低越好）+ `damage_bpc = bpc_arm − bpc_own_family_baseline` | 本实验**同进程统一路径**重测（`bpc_of`），与来源实验逐一对照 |
| 代价（质量腿 2） | 合成长程召回（copy / induction / var_recall，越高越好）+ Wilson 95% CI | `tools/evalkit/longrange_tasks.py`，本实验测 |
| 参照系 | chance（答案集均匀）/ 局部先验地板（只看最近 8 个 token 的最强预测器） | 同仪器内置 |
| 收益（内存） | 相对本族基线的每位置字节下降（**解析口径**） | 解析公式 |
| 口径隔离 | **实测显存峰值**单列为一节，**不与解析字节同列** | 本实验 GPU 实测 |
| 可控性 | 复现性：同 seed 重测的最大绝对差 | 本实验重测 |
| 活性 | KV 量化的 `rel_err`/`changed_frac`；窗口掩码的 `dropped_frac`；GQA 的 `k_tensor_shape` | 复用来源实验 + 本实验重测 |

**三条腿的分工必须写清**：bpc 腿量的是「语言建模依赖」，合成召回腿量的是
「不靠局部先验能不能取回远处信息」（SCHEMA §19）。**bpc 不是主判据**——本项目
最贵的一次入账错误（假保留）正是「bpc 改善与长程检索退化同时发生」。

---

## 4. 判据线（预登记，逐字）

> 契约原文（`.scratch/36-parallel/issues/01-parallel.md` §「U3 线」）：
>
> - 把已实测的四类点画到**同一横轴「每位置 KV 字节」**：int8/int4 量化（R1）、
>   GQA/MQA（W3）、驱逐/窗口（M2）、以及「不压缩」基线；纵轴至少两条腿：clean bpc
>   与**合成长程检索召回**（用 `tools/evalkit/longrange_tasks.py`）。
> - **判据**：产出**一张图 + 一张表**（每点含：字节/位置、Δbpc、召回、来源路径、
>   口径），并明确标出**哪些点缺**（我们没测的组合）作为下一批候选。禁止把不同口径
>   （解析/实测内存）混在一列。
> - 成本：≈20 min。

**本契约把它展开为可失败判据**：

- **C1 横轴唯一性**：所有点的 `bytes_per_position` 由**同一个解析公式**算出
  （`(positions_kept/T)` 与 `value_bytes` 两个因子显式化）。任何点若无法用该公式
  表达 ⇒ 不入图。
- **C2 双列隔离**：解析字节（`bytes_per_position`）与**实测显存峰值**
  （`measured_gpu_peak_mib`）必须是**两列/两节**；不得把实测显存写进横轴。
- **C3 召回腿全覆盖**：每一个点都用**同一件仪器、同一 seed、同一任务定义**测出合成
  召回；同时报 chance 与局部先验地板。
- **C4 不可分辨如实标注**：某点的召回落在 `chance + 2σ_wilson` 内 ⇒ 标
  `at_floor: true`，读作「在该任务上不可分辨」，**不得**写成「机制无效」（SCHEMA §19）。
- **C5 缺失组合**：`results.json::missing_combinations` 非空且每项带「为什么缺 +
  若补应怎么做」。
- **C6 数字对账**：`python3 reconcile_numbers.py` 退出码 0（`analysis.md`/`table.md`
  的每个数字都能在 `results.json` 里找到同值）。
- **C7 三道对账**（SCHEMA §12）通过并落进 `results.json::reconciliation`。

---

## 5. 对照与方差

- **对照臂 1（同族基线）**：trunk 族 = `fp16`（无压缩、全位置、fp16 存储）；
  head 族 = `mha`。所有 `damage_bpc` 都相对**本族**基线，图上单列两条基线。
- **对照臂 2（召回地板）**：
  - `chance` = 1/64 = 1.5625%（答案集均匀）；
  - `local_prior` = 只看最近 8 个 token 的最强预测器（答案在窗口外只能退化为 chance）；
  - `corrupt_source` = 把 copy 任务的源片段整段换随机符号（其余逐位不变）——
    测的是「读数真的来自那段被搬运的信息吗」。
- **方差界**：bpc 的排序门槛 = 0.014（`docs/methodology.md`）；召回用 **Wilson 95% CI**，
  两点 CI 重叠 ⇒ 不可排序。
- 复现性：同 seed 重测全部臂（`reproducibility.rerun`），报最大绝对差。

---

## 6. 成本

```
gpu_wall_budget_min: 20
```

- 组成：14 个臂 ×（bpc 24×16 窗口序 + 3 件合成任务）≈ 8–12 min（共享卡 2× 折扣已含）。
- 显存预算：13.5M 模型 + `[16,256]` 批 ⇒ 峰值 << 1 GB（远低于 5 GB net 上限）。
- 可中断性：完全可中断（无可学状态）；重跑一次为分钟级。
- 冒烟：`python3 run.py --smoke`（缩小批量/档数，实测步时与显存，不外推时间）。
- **预算不够时砍什么**（SCHEMA §8）：砍任务档数（先砍 `var_recall`，再砍 `induction`），
  保 `copy` 主腿与全部 14 个预算点；**不砍判据线**。
- 冲突处理：若预算与判据不可兼得 ⇒ 停下向主 Agent 裁决，禁止自行改判据线。

---

## 7. 数据与脚本清单

- 数据集：`data/enwik8_full.npz`（前 90M 训练 / 末 10M 评测；本实验只用**评测尾段**
  跑 bpc，窗口序 `window_seed=2` + 24×16，与 2610/2614/2615/2616 同序）。
  合成任务**不使用** enwik8 文本，只用它的词表大小（206）。
  **不使用 enwik8 当长程主任务**（SCHEMA §19）。
- 入口脚本：`run.py`（`--smoke` / 默认全跑 / `--out-dir`）。
- resolved config：`config.json`。
- 落盘：`results.json`、`run_log.txt`、`table.md`、`chart.png`、`chart.txt`、
  `reconcile_report.json`、`analysis.md`、`MODEL.md`。

---

## 8. 失败决策树（无效时按此顺序排除）

1. **测量**：bpc 与来源实验不一致 ⇒ 先查窗口序/代码路径（本实验用 `bpc_of` 与
   `retention.masked_bpc` 逐位对照自检）；召回读数为 0 ⇒ 先查任务构造
   （`check_local_prior_resistance`、`corrupt_source` 对照）。
2. **实现**：量化臂读数与 `fp16` 完全相同 ⇒ 查 `quantize_kv` 是否真的改了数值
   （活性量 `rel_err` 必须 > 0）。
3. **规模**：若 13M 主干根本不会长程召回 ⇒ 召回腿落地板 —— 这是**真结论**
   （SCHEMA §19：不得外推成「机制不可能」），改用**有信号的任务档**（copy）并在
   `analysis.md` 显式标注不可分辨。
4. **设计**：若某点在两条腿上给出矛盾读数（bpc 好、召回差）⇒ 这正是「假保留」，
   必须两腿并报，**不得只报 bpc**。

---

## 9. 产出

- `results.json`（点表 + 召回 + 自检 + 对账 + 缺失组合）
- `run_log.txt`（stdout 全量）／`smoke/`（冒烟）
- `table.md`（一张表）／`chart.png` + `chart.txt`（一张图，两种载体）
- `analysis.md`（判读，**第一段 = 人话摘要**）
- `MODEL.md`（三个 checkpoint 的哈希 / 体积 / 生成命令；权重不入库）
- `reconcile_numbers.py` + `reconcile_report.json`（报告数字逐一对应）

---

## 10. 既有做法（SCHEMA §18）

> **检索能力声明（本轮硬约束）**：本环境**联网检索不可用**（上一轮 researcher 因 web
> 工具未注册而失败）。因此本节所有条目一律标 **`记忆·未核`**，**只用于设计方向**，
> **不得当判据**（SCHEMA §18）。§11 给出「若可检索应查什么」的精确关键词。

| # | 谁做过 | 为什么这么设计 | 它的实测结果（`记忆·未核`） | 我们的增量 |
|---|---|---|---|---|
| 1 | **KV cache 量化**：KIVI、KVQuant、Atom、QServe 等 | KV cache 随上下文线性增长，成为长上下文的显存/带宽瓶颈；把 K/V 存成低精度即可省内存 | int8 KV 在多数长上下文基准上近无损；int4 需要分组/逐 token/异常值处理才近无损，朴素逐通道 int4 有可测退化。**数字一律未核** | 13M、enwik8 clean bpc **+ 合成长程召回**双口径、per-channel 对称量化单因素、与「丢 KV」路线同轴可比 |
| 2 | **KV 头共享**：MQA（Shazeer 2019）、GQA（Ainslie 等 2023）、MLA（DeepSeek-V2） | 推理时 KV cache 与 head 数线性相关 ⇒ 让多个 query 头共享一组 K/V，直接砍 KV 内存与带宽 | 业界结论：GQA 在数十倍内存下降下**质量近无损**；MQA 更省但质量退化更明显；MLA 以低秩压缩做得更好。**数字一律未核** | 本格测的正是「量化 vs 头共享在**同一字节预算**下谁便宜」——这在本项目的 13M/256 长度/同一仪器下没有公开可比数 |
| 3 | **驱逐 / 窗口**：StreamingLLM（attention sink）、H2O（heavy hitters）、SnapKV、TOVA | 注意力质量高度集中（近期 + 少数 sink/重击位置）⇒ 只留这些位置，KV 有界 | 业界结论：小预算驱逐在长上下文 QA 上近无损；但**损坏与精确复制类任务**上会崩。**数字一律未核**；本项目的 A3/M2 本地实测（+0.212 bpc @ 只留最近 32）方向一致 | 本项目已有本地实测价签（A3/M2），本实验把它与量化/头共享放到同一字节轴 |
| 4 | **合成回忆 / 局部先验抗性**：MQAR 类基准（Zoology / Zoology-2 等）、copy/induction 任务文献 | 语言建模损失会被局部统计主导，看不出「信息能不能搬」⇒ 用随机键值关联回忆等合成任务直接测 | 业界结论：小模型在 MQAR 上显著低于 1.0，需要专门结构（线性注意力/状态空间）才稳。**数字一律未核** | 本项目 `longrange_tasks.py` 内置 chance **与局部先验**双基线 + 毁源对照；本实验把它当**第二条纵轴** |
| 5 | **统一预算对比**：KV 压缩综述 / 各方法论文内部对比表 | 各种压缩比口径不一（元素数/字节/保留比例），读者无法直接比 | **未找到**一份把「量化 / GQA-MQA / 驱逐窗口 / 基线」画到同一**每位置字节**轴、并同时给 clean bpc 与合成长程召回的公开表（检索不可用，**未能确认**） | 本实验的增量：**同一横轴口径 + 同一仪器 + 双纵轴 + 显式缺失组合**（见 §11 分诊） |

---

## 11. 分诊表（SCHEMA §21，(a)/(b)/(c) 三选一，机器可读）

| grid_id | 判定 | 一句话理由 | 若要写进账本的结论 |
|---|---|---|---|
| `R1-quantization` | **(a) 业界已研究透** | KV 量化（int8 近无损 / int4 需分组）是全行业标准做法，我们的 int8/int4 逐通道数字与业界方向一致 | 保持 `status: reference`（已标「业界结论，非本地实测」）；本实验只补**本地同仪器的字节轴坐标** |
| `W3-kv-head-sharing` | **(a) 业界已研究透** | GQA/MQA/MLA 已被全行业采纳，KV 内存降数倍、质量近无损 | 保持 `status: reference`；本实验只补**同字节预算下与量化的直接对比** |
| `M2-carrier-compression`（驱逐/窗口） | **(a) 业界已研究透** | StreamingLLM / H2O / SnapKV 已把驱逐/窗口研究透 | 保持 `reference` 口径的结论；本实验只把 A3/M2 的本地价签放上同一字节轴 |
| `M2-carrier-compression`（量化/摘要/低秩） | **(c) 查不到（检索不可用）** | 量化子项已由 R1 覆盖；**摘要压缩 / MLA 低秩**本项目零经验且无法检索确认 | 按 `记忆·未核` 记为方向，**不得**据此判 (a) |
| **本办法（统一预算总图 + 双纵轴 + 合成召回腿）** | **(b) 我们的设定实质不同** | 13M / 局部先验抗性合成任务 / 单因素受控 / 同一仪器同一字节轴 —— 业界没有同口径公开表 | **做**这个实验（本目录） |

---

## 12. 「若可检索应查什么」关键词清单（本轮未核）

中文/英文混名列出的检索词（供下一轮有联网能力的 researcher 直接使用）：

1. `KV cache quantization int8 int4 per-channel near-lossless long context`
2. `KIVI per-channel key quantization group-wise value`
3. `KVQuant / Atom / QServe KV cache 4-bit outlier handling`
4. `grouped-query attention ablation quality MQA MLA memory reduction`
5. `StreamingLLM attention sink eviction budget`
6. `H2O heavy hitter oracle KV eviction copy task failure`
7. `KV cache compression unified benchmark bytes per token comparison`
8. `multi-query associative recall MQAR small model associative recall failure`
9. `copy induction head long range synthetic task small transformer`
10. `needle in a haystack vs language modeling perplexity eviction correlation`

---

## 13. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——组合留白是显式声明，
不是遗漏。`deps` 字段只记录已知耦合方向。组合取舍属具体产品方 scope。
本实验是**测量/汇总**类（无训练、无新参数），因此不涉及「干预活性」的训练侧量；
活性证据改为「量化往返确实改了数值」与「窗口掩码确实删了位置」。
