# 2621 · 搬运通道 × KV 精度压缩（评测期；13M · 合成长程任务）—— 实验契约

> 契约来源：`.scratch/38-auto-parallel/issues/01-auto-parallel.md` 的「**W3**」。
> 本文件是该线的**执行契约**；§4 判据线**逐字**抄录 W3，执行期间未改。
> **§4b 记录与契约原设计的偏离（GQA 因子未测）及其裁决**——这是本次执行的唯一契约变更，
> 由主 Agent 明确裁决（不是执行者自行放宽）。
> 入口：`run.py`（纯推理评测，**不训练**）；新语义层：`carry_quant.py`（carry-aware
> 量化前向）；仪器：`tools/evalkit/kv_quant.py`（KV 量化往返，**复用不改**）、
> `tools/evalkit/longrange_tasks.py`（合成长程任务电池）、`tools/evalkit/carry_channel.py`
> （统计与对照）。被测量模型：ticket 35 的 `U2` 检查点（只读 + 运行时 sha256 断言）。

---

## 1. 命题

**对一个「delta 写（残差写）搬运通道已被训练真的用起来」的模型，评测期再把注意力
KV cache 降成 int8（对称 per-channel），关联回忆召回会不会崩、字节/位置省多少？**

一句话可证伪的形式：*「int8 KV 的关联回忆准确率 ≥ fp16（全精度基线）− 3pp；若二项
95% CI 不重叠且更低则记『崩』」*。

## 2. 单因素声明

- **基底**：本仓 stock dense decoder-only trunk（13.5M：d256 × 12 层 × 8 头 × mlp6 ×
  T256，tied 读出、仅最终深度 CE），加载 **ticket 35 的 `U2` 检查点**
  （`carry_rule="delta"`，`train_window=16 / train_anchors=0`，MHA）。
- **唯一变量**：**评测期注意力 K/V 的存储精度**——`fp32`（释放版，实现等值锚）/
  `fp16`（**基线**）/ `int8`（压缩档）。权重、训练数据、可见性掩码、位置集、评分配置
  逐字段一致（见 §6 对照表）。
- **不训练、不加参数、不改权重**：本实验是**评测 + 汇总**类，`params_added = 0`。
- **GQA 因子未测**（契约原设计的一部分，因引擎硬互斥而无法执行）⇒ 见 **§4b**。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

`carry_quant.py::carry_quant_logits(model, tokens, mode, keep_mask)`：逐行复刻
`pathlm.depth_ar.CarryChannelBlock.forward`（`_input_states` 先算好单共享记忆槽，
每层 `a = SDPA(q,k,v; 窗口掩码)`，之后照旧加通道读出 `γ ⊙ mem`、`proj`、`mlp`），
**唯一的差别 = SDPA 之前对 K/V 做一次 `quantize_kv(x, mode)` 往返**。

- 量化算子**复用** `tools/evalkit/kv_quant.py::quantize_kv`（对称 per-channel：通道轴
  = `head_dim`、每 `(head, channel)` 一个 fp16 scale、沿 batch/token 取 amax、
  `qmax = 2^(bits-1)-1`）⇒ 与 2614/2619/2620 是**同一件仪器**；本模块不重新定义量化。
- `mode="fp32"` 档**逐位恒等**：`quantize_kv(x, "fp32")` 原样返回、不触碰数值
  （`n_values = 0`），因此 `carry_quant_logits(..., "fp32")` 必须与释放版前向逐位相等
  （单测 `test_carry_quant_fp32_bitwise_matches_released_forward` + `run.py` 自检钉住）。
- 因果性/可见性不变：`keep_mask` 经与 `retention.masked_logits` 同款的 expand 进
  SDPA；`keep_mask=None` 时走释放版 `is_causal=True` 路径。

**字节口径（唯一横轴）**：解析每位置字节 =
`2(K,V) × n_layers × n_heads × head_dim × 每值字节`（复用 `kv_quant.kv_cache_footprint`
的同一式；MHA ⇒ `n_heads_eff = n_heads`）。int 档的 scale 开销每 `(head,channel)` 一个
fp16、与 T 无关，**单列**。

## 4. 判据线（**逐字抄录**自 W3 契约，一字未改）

> - 动机：我们已有两个独立正面结果——**delta 写通道**（关联回忆 0.0137→0.0950）与
>   **GQA×int8 近无损压缩**（1536 B/位置）。问题是：**叠加后是否互相干扰**（压缩的 KV 是否让搬运失效）。
> - 干预：`delta 通道 + 全精度 KV` vs `delta 通道 + GQA×int8 KV`（其余逐字段一致；**不改引擎**：
>   头数用既有 `--n-kv-heads 2`，量化用 `tools/evalkit/kv_quant.py` 的评测期路径）。
> - 判据（预登记）：① **关联回忆召回不崩**（压缩臂 ≥ 全精度臂 −3pp 且 CI 不重叠才算「崩」）；
>   ② 字节/位置下降比例；③ 毁源对照臂；④ 活性量每 200 步。**注意参数量混杂**（GQA −8.75%）须显式声明。

（以下为**摘要**，非逐字：三臂规格表、预算、交付物清单、scope——见 §5–§11。）

**可失败展开**（执行前写下，不得事后调整；`config.json::criteria` 同名落盘）：

- **C1（主判据）**：`int8` 的关联回忆准确率相对 **fp16 基线** `≥ −3pp`；**「崩」的操作化 =
  `delta_pp < −3` 且两臂 Wilson 95% CI **不重叠**。
- **C1b（前提）**：被问的 pair 落在窗口外（`frac_outside_window = 1`）⇒ 需要通道搬运，
  否则任务无区分度（前提校验，来自契约 §① 的口径）。
- **C2**：`int8` 的解析每位置字节相对 fp16 **下降 50%**（解析口径，单列实测显存）。
- **C3（毁源对照，强制）**：任一 mode 毁源后若准确率 CI 下界仍 `> 4×chance` ⇒ 该读数
  不是来自被搬运的信息，**该 mode 判读作废**（与 2618 判据 4 的操作化同款）。
- **C4（活性）**：量化活性（`changed_frac > 0 ∧ rel_err > 0`）与通道活性（`read_share > 0`）
  都必须非零（SCHEMA §10）。
- **C5（参数量混杂）**：`int8` 量化新增 **0 参数** ⇒ 无参数量混杂；原契约提醒的 GQA −8.75%
  混杂因子**未测**（§4b）。

## 4b. 与契约原设计的偏离：GQA 因子未测（**主 Agent 裁决**）

契约原设计的两臂是 `delta 通道 + 全精度 KV` vs `delta 通道 + GQA×int8 KV`，并写明
「不改引擎：头数用既有 `--n-kv-heads 2`」。**执行前实测发现该组合在引擎里被硬禁止**
（`pathlm/depth_ar.py`，在碰任何共享文件之前已停下请示）：

| 组合 | 结果 | 位置 |
|---|---|---|
| `carry_rule` ⊕ `n_kv_heads < n_heads` | `ValueError: carry_rule 与 n_kv_heads < n_heads 互斥` | `depth_ar.py:1690-1711` |
| `n_kv_heads < n_heads` ⊕ `train_window > 0` | `ValueError: n_kv_heads 与 train_window 互斥` | `depth_ar.py:1712-1734` |

且块类是二选一的（`GQABlock` 或 `CarryChannelBlock`），不存在同时支持 GQA + carry +
窗口掩码的块。窗口 16 是「制造必须搬运需求」的前提（ticket 35 靠它把 key 100% 推出窗外），
因此 **GQA 无法与搬运通道、也无法与窗口需求共存**。放开它 = 动 `depth_ar.py` 的架构校验
+ 新增组合块，既违反本契约「不改引擎」，也违反 SCHEMA §22（本波 `depth_ar.py`/
`train_depth_ar.py` 的唯一写入者是 W4 线，代码锁冲突）。

**裁决**：主 Agent 裁定走 **A 方案**——**丢掉 GQA 因子，只测 int8 因子**，并在报告里
显式声明；扩引擎（B）明确不做，另记为后续有界任务 `ENG1`。因此：

1. 本实验**只回答「int8 量化对搬运的干扰」**，**不外推**到 GQA/MQA/组合压缩；
2. 若召回在 int8 下不崩，只能表述为「**int8 级量化与搬运通道兼容（该任务/该尺度）**」；
3. 契约的「全精度 KV」臂在本实验中落为 **fp16（工业默认 KV 精度，与 2614/2619/2620
   的基线同口径）**，`fp32`（释放版）作实现等值锚并并报。

## 5. 既有做法（prior art，SCHEMA §18）

> **检索条件声明**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向，**不得当判据**；本地条目给文件路径。可检索时应查的关键词见
> 每块末尾。

1. **KV cache 低比特量化（int8/int4）**。谁做过：KIVI（Liu 等 2024）、SmoothQuant
   （Xiao 等 2023）、以及 vLLM/TensorRT-LLM 的 `kv_cache_dtype=int8` 一类生产栈。
   为什么：解码访存瓶颈在 KV cache，把每值 2 B → 1 B（int8）/0.5 B（int4）直接省访存。
   实测（`记忆·未核`）：int8 近无损、int4 有明显质量损失——与**本地** R1 参照线一致
   （`2614`：int8 Δbpc +0.00015、int4 +0.0457）。**我们的增量**：不是复现量化，而是把
   量化**叠加到一个「显式搬运通道正在工作」的模型上**，问压缩是否破坏长程搬运。
   §21 分诊：属 **(b) 设定实质不同**（13M / 局部先验抗性任务 / 旁路搬运通道）⇒ 值得本地补测。
2. **把 memory/SSM 状态与 KV cache 组合**（混合架构，如 Jamba、MambaFormer、Zamba）。
   为什么：用固定状态补长程、用 KV 保精度。实测（`记忆·未核`）：混合架构在长上下文
   吞吐/质量上有效。**我们的增量**：我们测的不是「组合两种状态载体」，而是「**对 KV 这一
   侧额外施加数值压缩**，通道侧不动」——即压缩与搬运的两因子交互。本地无前例。
3. **显式写入通道（delta rule / fast weight）**。谁做过：Schmidhuber 1992 fast weight；
   Schlag 等 2021（delta rule = 只用残差的写）；Yang 等 2024（Gated DeltaNet）。
   本地：ticket 35（`2618`）在合成长程任务上首验——无通道 0.0137 ≈ chance、delta 写
   0.0950（6.08× chance）。**我们的增量**：把该通道的既有正面结果放到**压缩 KV 的部署
   约束**下复核，问「省内存的代价是不是搬运失效」。

**若可检索应查**：`KIVI KV cache 2bit quantization`、`SmoothQuant`、`GQA int8 KV cache
serving vLLM`、`KV cache quantization long context retrieval`、`linear attention delta rule
KV cache compression interaction`。

## 6. 对照与方差

- **主对照 = fp16（全精度 KV 基线）**，不是 U0：本实验问的是「同一张图的压缩代价」，
  2618 已单独回答「有无通道」。并报 `fp32`（释放版）作实现等值锚。
- **chance** = `1/词表子集大小`（sym_vocab = 64）；**局部先验地板** = `local_prior_accuracy(k=8)`；
  **逐 arm 毁源对照**。
- **方差带** = `0.014 bpc`（`harness/methodology.md`）：`|Δbpc| < 0.014` 标「⚠不可排序」；
  召回判据用 Wilson 95% CI（`1.4pp` 的召回方差带仅作参考）。
- **纯推理、无随机采样**：同 seed 重跑应逐位相同（`results.json::reproducibility`）。
- **逐字段对照表**（唯一变量 = KV 精度）：

| 字段 | fp32 | fp16 | int8 | 说明 |
|---|---|---|---|---|
| 检查点 | U2 | U2 | U2 | 同 sha256 |
| 权重/参数 | 冻结 | 冻结 | 冻结 | 不训练、不加参数 |
| `train_window/anchors` | 16/0 | 16/0 | 16/0 | 同一掩码 |
| 数据/位置集 | 同 | 同 | 同 | 同 seed、同 `pathlm.data.batch` 序 |
| K/V 每值字节 | 4（无处理） | 2 | 1 | **唯一变量** |

## 7. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 10 min`（含冒烟 + 全电池 + 评测）。峰值 **≤1 GB 净**（纯推理、13M、
bs≤64、T256）；若共享卡繁忙按 SCHEMA §20 取 GPU 锁（整程远小于 10 min）。

- **预算不够时砍什么**：砍距离腿档数 > 砍 bpc 批数 > 砍主腿批数；**不砍判据线**。
  （本实验实测远低于预算，未触发砍臂。）
- **实测**：评测墙钟 **0.92 min**、峰值显存 **431.6 MB**（`results.json::cost`）。
- **§20.1 并发例外登记**：运行期间 GPU 锁由 W4 线持有；本实验为纯推理、墙钟 < 2 min、
  峰值 431.6 MB < 1 GB，且 `nvidia-smi` 空闲 ≥ 3 GB ⇒ 按 SCHEMA §20.1 第 2 条短任务插队，
  **未取锁、未动锁目录**（前后空闲读数见 `analysis.md §11`）。

## 8. 数据与脚本清单

- **合成长程任务**（主腿，SCHEMA §19：**不用 enwik8 当长程主任务**）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_batch`（K=8、窗口 16）与
  `var_recall_batch`（距离腿）；毁源用 `corrupt_span`。
- **clean bpc 腿**：`data/enwik8_full.npz` 的**评测尾段**（末 10M，vocab 206，与模型
  词表一致）；只用词表大小与 LM 质量口径，不是长程主任务。
- 入口：`python3 run.py`（配置 `config.json`）；结果 `results.json`。
- 落盘：`results.json`、`run_log.jsonl`、`run_log.txt`、`analysis.md`、`MODEL.md`、
  `controls/README.md`、`reconcile_numbers.py` → `reconcile_report.json`、
  `mutate.py` → `mutation_report.json`、`smoke/`（缩档日志）。
- 测试：`tests/test_carry_quant.py`（新语义层的可失败单测 + 变异验证）。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：`fp32` 档与释放版前向不逐位相等？→ 先修 `carry_quant` 的接线（窗口掩码/
   通道读出/量化调用点），**不得**动模型。
2. **实现**：量化活性为 0 或 `read_share = 0`？→ 跑变异验证排除死开关（量化没真的发生 /
   通道被丢掉）。
3. **规模**：int8 与 fp16 都 ≈ chance ⇒ 该任务/该尺度上通道本就没起作用（读 2618 的
   U0/U2 作前提）；**不得**表述为「压缩机制不可能」（SCHEMA §19）。
4. **设计**：int8 崩但毁源也崩 ⇒ 读数混杂；任一 mode 毁源后仍高 ⇒ 该 mode 作废。

## 10. 产出与复核

见 §8 落盘清单；本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context
**只读复核者**独立检查。

## 11. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验唯一变量 = 评测期 KV 存储精度。
`deps` 只记录已知耦合方向（M1 载体类型 ← M2 载体压缩；与 ticket 35 的 U2 检查点耦合）。
**GQA/MQA 因子与本实验的组合压缩留白是显式声明**（§4b），不是遗漏。
