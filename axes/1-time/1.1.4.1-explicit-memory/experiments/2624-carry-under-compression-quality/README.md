# 2624 · 搬运 × 压缩的**训练质量腿**（13M · 合成长程任务 · carry × {MHA,GQA} × {fp16,int8}）

> 契约来源：`.scratch/41-followups/issues/01-followups.md` 的「**Z3**」。
> 本文件是该线的**执行契约**；§4 判据线**逐字**抄录 Z3，执行期间未改。
> **§4b 记录本线的 DESIGN 裁决（window16 作常量、fp16 作「全精度」基线）**——由执行者
> 依 SCHEMA §19/§17.14 从 Z3 的明文推出并在报告中显式声明，非事后放宽判据。
> 入口：`train_arms.sh`（训练两臂）+ `run.py`（评测电池）；语义层 `quant_carry.py`
> （评测期量化前向，**本线唯一的新语义层**）；三臂定义 `compression_arms.py`；
> 单测 `tests/test_carry_under_compression_quality.py`；变异 `mutate.py`；
> 对账 `reconcile_numbers.py`；原始结果 `results.json`；判读 `analysis.md`；
> 权重清单 `MODEL.md`；对照说明 `controls/README.md`。

---

## 1. 命题

**ticket 38-W3 只测了「评测期 int8 量化对搬运通道的干扰」（载体 = MHA + carry，结论：
不破坏），且因引擎硬互斥丢掉了 GQA 因子。ENG1（39-X2）解除限制后，把 GQA 放回
「carry 通道」里——**对一个「delta 写通道 + GQA 头共享」已被训练真的用起来的模型，
评测期把 KV cache 降成 int8，长程关联回忆会不会崩？头共享本身又付/得多少？**

一句话可证伪的形式：*「`carry_gqa_int8` 的关联回忆召回 ≥ `carry_gqa_fp16` − 3pp；
若二项 95% CI 不重叠且更低则记『崩』」*（与 W3 的 C1 同一操作化，只把载体从 MHA 换成
GQA）。

本线填的正是 W3 的缺口：W3 的载体是 MHA（`CarryChannelBlock`），GQA 因子未测；
本线把「头共享」与「评测期 int8」两个因素各配一条干净对照，并量出两腿的增量。

## 2. 单因素声明

- **基底**：本仓 stock dense decoder-only trunk（13.5M：d256 × 12 层 × 8 头 × mlp6 ×
  T256，tied 读出 + 顺序残差），`DepthARModel(dense=True, topology="sequential")`。
- **唯一变量 = 两个因素**（Z3 明文），拆成**三条评测行**（§17.14 的基准唯一）：
  - 因素①**头投影**：`carry_mha`（8 KV 头）vs `carry_gqa`（2 KV 头）；
  - 因素②**评测期 KV 精度**：`fp16`（基线）vs `int8`（压缩）。

| 评测行 | 训练臂（权重） | 训练 flags | 评测 KV 精度 | 相对谁读 |
|---|---|---|---|---|
| `carry_mha_fp16`（基准） | `carry_mha` | carry+window16 | fp16 | —（W3 的载体形态） |
| `carry_gqa_fp16` | `carry_gqa` | carry+window16 | fp16 | 比基准（分离「头共享」） |
| `carry_gqa_int8` | `carry_gqa`（**同**权重） | （同 `carry_gqa`） | int8 | 比 `carry_gqa_fp16`（分离「int8」） |

- **被训练的臂 = 2 条**（`carry_mha` / `carry_gqa`），逐字段同配方（同 seed=0、同步数/
  批大小/学习率/优化器/数据），只差 `n_kv_heads`；第 3 行**不训练**、不加参数，只把
  `carry_gqa` 的权重在 int8 KV 下评一遍。
- **carry（delta 写）+ window（16 / anchors 0）是常量**（不是变量）——见 **§4b**。
- **无「必要性例外」**。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

- **GQA（`--n-kv-heads 2`）**：注意力 K/V 的头数从 8 降到 2（`ComposableBlock` 的 KV 头
  投影，`enable_gqa=True` 按组广播）。真进 SDPA 的 K/V 形状是 `[B, 2, T, 32]`（活性量）。
  ENG1 前该组合与 carry 硬互斥、块类二选一 ⇒ **不存在**；ENG1 后落在 `ComposableBlock`。
- **carry（`--carry-write delta`）**：单共享记忆槽 `S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；
  每层读出 `a ← a + γ ⊙ mem_t`，γ **单边零初始化**。
- **window（`--train-window 16 --train-anchors 0`）**：`attn_mask` 是与块类正交的装饰层，
  `keep[i,j] = (j≤i) ∧ ((i−j)<16)`。它把源 key/value 段 `[0, 2K)` 推到窗外 ⇒ 不搬就答不出。
- **评测期 int8（`quant_carry.py`）**：逐行复刻 `CarryChannelBlock.forward` /
  `ComposableBlock.forward`（GQA 头投影 + carry 读出 + 正交窗口掩码），**唯一的差别 =
  SDPA 之前对 K/V 做一次 `quantize_kv` 往返**。量化算子**复用**
  `tools/evalkit/kv_quant.py::quantize_kv`（对称 per-channel，与 2614/2619/2620/2621
  同一件仪器）；`fp32` 档**逐位恒等**（单测 + `run.py` 自检对两种块类各钉一条）。
  2621 的 `carry_quant.py` **显式拒绝 GQA/`ComposableBlock`** ⇒ 这是本线需要的新接线。

**字节口径（辅助，非判据）**：解析每位置字节 =
`2(K,V) × n_layers × n_kv_heads × head_dim × 每值字节`（int8 的 scale 开销单列、与 T 无关）。

## 4. 判据线（**逐字抄录**自 Z3 契约，一字未改）

> ## Z3 · 搬运 × 压缩的**训练质量腿**（≈40 min GPU）
> - 目录：`axes/1-time/1.1.4.1-explicit-memory/experiments/2624-carry-under-compression-quality/`
> - 动机：ticket 38-W3 只测了**评测期** int8 量化对搬运的干扰（结论：不破坏）。**训练期**是否也兼容未测；
>   且 ENG1 后 `GQA × carry` 已可组合 ⇒ 终于能做**质量腿**。
> - 干预：`carry + MHA(全精度)` vs `carry + GQA(头共享)` vs `carry + GQA + 评测期 int8`（同 token 预算、同 seed）。
> - 判据：合成关联回忆召回（主腿，准确率 + chance + CI）、毁源对照臂、参数量差声明、活性量；
>   **注明 GQA 的参数混杂**（§15）。

**执行时的操作性定义（不改判据，只把阈值/口径落定；`config.json::criteria` 同名落盘）**：

- **主腿**：`assoc_recall_batch`（单查询，K=8，T=256，chance=1/64）在 **native 可见性
  window16** 下评测（训练/评测同一掩码）；被问 pair 的 key 距离 **241–255**（窗外 ⇒
  不搬就答不出）。并报**全因果交叉口径**与**训练型任务（多查询）口径**两列。
- **C1（主判据）**：`carry_gqa_int8` 相对 `carry_gqa_fp16` 的召回准确率 `≥ −3pp`；
  **「崩」的操作化 = `int8_vs_fp16_pp < −3` 且两行 Wilson 95% CI 不重叠**。
- **C2（头共享增量，§17.17 分量分解）**：`carry_gqa_fp16 − carry_mha_fp16`（pp）；
  净和 `carry_gqa_int8 − carry_mha_fp16`（pp）。
- **C3（毁源对照，强制）**：任一评测行毁源后若准确率 CI 下界仍 `> 4×chance` ⇒ 该行读数
  不是来自被搬运的信息，**该行作废**（与 2618/2621 同款操作化）。
- **C4（参数量差，§15）**：两条训练臂参数量逐臂实测并声明；GQA 为**剪枝类**（减参数）
  ⇒ 按 §15 例外与「同 FLOPs」比不适用；归因声明见 `analysis.md`。
- **C5（活性量）**：KV 头形状（`kv_heads`）/ carry（`attn_carry.read_share`）/ window mask
  每 200 步落盘 + 量化活性（`changed_frac`/`rel_err`）。
- **C6（收敛门禁，§17.15）**：答案行 CE（`synth_answer_ce`）末窗斜率；仍在陡降 ⇒ 结论
  只读臂间增量、绝对水平是快照。

## 4b. DESIGN 裁决：window16 作常量、fp16 作「全精度」基线（**执行者声明，非事后放宽**）

Z3 明文只列了「carry + MHA / +GQA / +GQA+int8」三个因素，**没有列 window**。把它们落到
可跑的配置时需要补两处口径——两处都从 Z3 的明文与 SCHEMA 推出，**未改判据线**：

1. **window16/anchors0 作三行共同常量（不是新变量）**。理由：Z3 的「搬运」只有在
   **源信息落在窗口外**时才有意义（SCHEMA §19：局部先验抗性；2618/W3/Y1 的同一件仪器）。
   Z3 的动机段直接引用 **W3**——W3 的载体 `U2` 就是 `carry + MHA + window16`；要让本线是
   W3 的「训练质量腿」，载体必须同日同构 ⇒ 三条评测行**都**带 window16，它被**固定为常量**
   （三条行逐字段相等），因而不是「引入了一个未声明变量」。判据线、臂数、因素数均未变。
2. **「全精度」= `fp16`（工业默认 KV 精度）**，与 2614/2619/2620/2621 的基线同口径；
   `fp32`（释放版）作**实现等值锚**（逐位相等自检），不作主基线。理由：W3 的 C1 就是
   「int8 相对 fp16 ≥ −3pp」，本线沿用同一口径才能与 W3 的结论拼起来；fp16→fp32 的差在
   本尺度近无损（2614 的 int8 相对 fp16 Δbpc +0.00015）。

**未测/留白（显式声明，不是遗漏）**：`carry_mha + int8` 这一行 **Z3 未列**（W3 已单独回答
它），本线不做；MQA（`n_kv_heads=1`）与更粗精度（int4）未测。

## 4d. 执行中的方差发现与契约口径修正（**主 Agent 裁决**）

- **判决腿（距离 241–255）在 6k 下方差极大**：本线的 `carry_gqa`（0.93%）与 Y1 同配置的
  `gqa_carry_window`（15.33%）差 **15×**，四类解释（仪器/配置/能力/接线）已逐一排除 ⇒
  **该腿不得作为此类比较的主判据**。详见 `stability.md` 与 `analysis.md §7D`。
- **契约口径修正（主 Agent 认领，非执行者问题）**：Z3 契约写「主腿 = 合成关联回忆召回」
  但**未钉死距离档**，导致用了超额距离腿（241–255，> 训练上限 240）当主判据——这是**契约
  缺陷**；今后该类比较的主腿一律用**训练分布内**的距离档。本报告的**主结论**（int8 vs fp16，
  **同权重确定性评测**）不受影响；**次要结论**（GQA 增量）标 n=1、不可排序。
- **下一步最小实验（已排，≈26 min）**：两臂 seed=1 复现，量化该腿方差（`stability.md §4`）。

## 4e. 能力面分析（SCHEMA §22.8，派发前做）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 列出全部校验，逐条核对本线需要的 flag 组合：

| 组合 | 引擎校验 | 本线 |
|---|---|---|
| `carry_rule` × `n_kv_heads<8` | ENG1 前**硬互斥**；ENG1 后允许（`ComposableBlock`，`depth_ar.py:1853-1860`） | `carry_gqa` 臂 |
| `n_kv_heads<8` × `train_window>0` | ENG1 前硬互斥；ENG1 后允许（窗口是正交装饰层） | `carry_gqa` 臂 |
| `carry_rule` × `train_window>0` | 允许（2618 已用） | 两条臂 |
| 三者齐开 | ENG1 前不存在；ENG1 后可建可跑（2622 钉住） | `carry_gqa` 臂 |
| `attn_sparse_k` / `attn_linear` ⊕ 上述任一 | 仍互斥 | 本线不用 |

块类选择（`DepthARModel.__init__`）：`attn_sparse_k>0`→`TopKBlock`；`attn_linear`→`LinearAttnBlock`；
`carry ∧ gqa`→`ComposableBlock`；`gqa`→`GQABlock`；`carry`→`CarryChannelBlock`；否则 `Block`。
本线两条臂恰好命中 `carry` 与 `carry∧gqa` ⇒ **不存在不可达组合**（`run.py` 自检里
`block_class` 逐臂与 `expected_block_class` 对齐，CPU 前置已验）。

## 4c. 成本与外部争用（§8 唯一预算数字 + §20.3）

`gpu_wall_budget: 40 min`（含冒烟 + 两臂训练 + 臂间窗口 + 评测）。

- **先测步时再定预算**（§17.8）：执行前实测 200 步——`carry_mha ≈ 172 ms/step`、
  `carry_gqa ≈ 159 ms/step`（共享 4090 上两个 `llama-server` 常驻，实测 ~19.5GB / util
  ~99%，§20.3 的**外部占用**）。
- **§20.3 沿用 6k**：两条被训练的臂一律 `6k × 32 × 256 = 49.0M token`（臂间严格同预算），
  与 2618/Y1 同口径 ⇒ 快照可横向对照。**不砍臂、不换配方、不等卡**（§20.3 优先级）。
- **§20.1 锁**：`train_arms.sh` **按臂**取/放锁、臂间 65s 窗口（时间戳见 `train_arms.runlog`）；
  两臂实际训练 13.74 + 11.16 min。
- **§20.1 第 2 条并发例外登记**：正式评测（`run.py` 全电池，墙钟 **0.69 min**、峰值显存
  **502.7 MB < 1 GB**）为纯推理，运行时 GPU 锁由 **Z2（`2624-composable-followups-13m`）**
  持有（本线原本取锁等待 >10 min 未得，遂改用例外）⇒ 按 §20.1 第 2 条、满足
  `nvidia-smi` 空闲 ≥3GB（跑前 15890 MiB、跑后 15889 MiB）**并发插入**，
  **未取锁、未动锁目录**。**锁异常已在最终汇报里显式列出。**
- **预算不够时砍什么**（§17.7）：砍评测批数/距离档；**不砍判据线**。
- 实测墙钟/峰值显存逐臂见 `results.json::cost` 与 `analysis.md`。

## 5. 对照与方差

- **基准 = `carry_mha_fp16`**（W3 的载体形态 + 工业默认 KV 精度；本仓「基底恒为 stock
  对照」口径的等价落地）。
- **方差带 = 1.4pp**（`docs/methodology.md`）；召回判据用 **Wilson 95% CI** 与
  `two_proportion` 差值 CI。
- **局部先验地板** = `local_prior_accuracy(k=8)`（§19：合成任务上必须 ≈ chance）；
  **chance** = `1/sym_vocab` = 1/64 = 0.015625。
- **逐行毁源对照**（C3）。
- **纯推理、无随机采样**：同 seed 重跑应逐位相同（`run.py` 自检钉住）。
- **逐字段对照表**（唯一变量 = `n_kv_heads` + 评测 KV 精度）：

| 字段 | carry_mha | carry_gqa | 说明 |
|---|---|---|---|
| seed | 0 | 0 | 同 |
| 训练数据 | assoc-mqar K8/Q8/gap14 | 同 | 同 |
| 步数×批×T | 6000×32×256 | 同 | 臂间严格同预算 |
| lr / 优化器 | 1e-3 / AdamW+cosine | 同 | 同 |
| `carry_rule` | delta | delta | 常量 |
| `train_window/anchors` | 16/0 | 16/0 | 常量（§4b） |
| `n_kv_heads` | 8(=MHA) | **2** | **唯一变量①** |
| 评测 KV 精度 | fp16（锚 fp32） | fp16 / **int8** | **唯一变量②** |

## 6. 指标表（逐项分析）

| 指标 | 解决什么问题 | 代价 | 有效性（vs 对照） | 收益与用法 | 可控性 | 副产品 |
|---|---|---|---|---|---|---|
| 判决腿召回（native window16） | 搬运在压缩下是否还在 | 一次评测 | 头共享 vs MHA、int8 vs fp16 | 主判据 | 全可控 | 距离曲线 |
| 全因果交叉口径 | 去掉窗口需求后是否可比 | 一次评测 | 同臂自比 | 口径对账 | — | — |
| 训练型任务（多查询）召回 | in-distribution 口径 | 一次评测 | 同臂自比 | 并列 | — | — |
| 参数量 | GQA 的结构量级 | 0 | −8.75% 量级 | §15 声明 | `n_kv_heads` | — |
| KV 头形状 `k_shape` | 头共享是真的（不是标签） | 一次前向 | `[B,2,T,32]` | 活性证据 | — | — |
| carry `read_share` | 通道真的被读出 | 来自日志 | > 0 | 活性证据 | `carry_scale` | — |
| 量化活性 `changed_frac/rel_err` | 压缩真的发生 | 一次前向 | int8 > fp16 | 活性证据 | `mode` | — |
| 解析每位置字节 | 压缩的部署价签 | 0 | GQA×int8 = 12.5% of MHA×fp16 | 辅助 | — | — |

## 7. 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向、**不得当判据**；本地条目给文件路径。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果（记忆·未核 / 本地） | 我们的增量 |
|---|---|---|---|---|
| 1 | **GQA / MQA**：Shazeer 2019（MQA）、Ainslie 等 2023（GQA，Llama-2 采用） | KV cache 随 `n_kv_heads` 等比缩小、解码省访存 | 同尺寸质量近似 MHA | 本地：Y1（2623）13M 上 GQA 省 8.75% 参数、质量在方差带内；**本线**把 GQA 与 carry 组合后再压 int8 |
| 2 | **KV cache 低比特量化**：KIVI 2024、SmoothQuant 2023、vLLM `kv_cache_dtype=int8` | 解码访存瓶颈在 KV cache，2B→1B 省访存 | int8 近无损、int4 有损（本地 2614 一致） | **本线**：把 int8 压在「GQA + 显式搬运通道已工作」的模型上，问组合是否互相干扰 |
| 3 | **显式写入通道 / delta rule**：Schmidhuber 1992；Schlag 2021；Yang 2024 Gated DeltaNet | 用固定状态把窗口外信息搬到可达范围 | 本地 2618：无通道 0.0137≈chance、delta 写 0.0950（6.08×）；W3：int8 下不崩 | **本线**：在「头共享 + int8」双重压缩下复核通道质量（W3 只测了 MHA 载体） |
| 4 | **压缩 × 记忆的交互** | — | **未找到直接前例**（检索关键词见下） | 本线是 C2 因子（头共享）× C1 因子（int8）× 搬运的三元组合，13M 尺度、局部先验抗性任务 |

**若可检索应查**：`GQA int8 KV cache quality`、`KV cache quantization long-context
retrieval`、`delta rule fast weight MQA compression`、`GQA memory channel interaction`。

## 8. 数据与脚本清单

- **合成长程任务**（主腿，SCHEMA §19：**不用 enwik8 当长程主任务**）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_batch`（判决腿，K=8）与
  `assoc_recall_multi_batch`（训练型任务，Q=8）；毁源用 `corrupt_span`。
- **入口**：`train_arms.sh`（训练两臂）、`python3 run.py`（评测，`config.json`）。
- **落盘**：`results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}`、
  `train_<arm>.log`、`train_arms.runlog`、`analysis.md`、`MODEL.md`、`controls/README.md`、
  `reconcile_numbers.py → reconcile_report.json`、`mutate.py → mutation_report.json`、
  `smoke/`（缩档日志）。
- **测试**：`tests/test_carry_under_compression_quality.py`（12 条可失败单测 + 变异验证）。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：native 口径与全因果口径是否分列？chance/局部先验是否同值？毁源是否同一批？
   → 先修 `run.py`/`quant_carry.py` 的接线，**不得**动模型。
2. **实现**：`carry_gqa` 的 `k_shape[1] ≠ 2`（只改标签）或 carry `read_share = 0`（通道没
   被用）→ 跑变异验证排除死开关。
3. **规模**：三行都 ≈ chance ⇒ 该任务/该尺度上通道本就没起作用（读 2618 的 U0/U2、
   Y1 的 gqa_carry 作前提）；**不得**表述为「压缩机制不可能」（SCHEMA §19）。
4. **设计**：int8 崩但毁源也崩 ⇒ 读数混杂、该行作废；头共享增量落方差带内 ⇒ 只报
   「不可分」，不得据此说 GQA 更好或更差。

## 10. 产出与复核

见 §8 落盘清单；本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context
**只读复核者**独立检查。

## 11. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验唯一变量 = 头投影（MHA/GQA）与
评测期 KV 精度（fp16/int8），carry+window 是常量。`carry_mha + int8` 行、MQA、int4
与本组合与其它注意力算子（`attn_sparse_k`/`attn_linear`）的交互是**显式留白**，不是遗漏。
`deps` 只记录已知耦合方向（M1 载体类型 ← ENG1 引擎可组合性 ← M2 载体压缩；与 2618 的
U2、2618-W3 的 2621 参照线耦合）。
