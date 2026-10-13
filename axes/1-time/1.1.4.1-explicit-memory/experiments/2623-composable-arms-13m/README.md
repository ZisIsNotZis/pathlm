# 2623 · ENG1 解锁的四臂质量实验（13M · 组合阶梯 · 合成长程任务）—— 实验契约

> 契约来源：`.scratch/40-next-wave/issues/01-next-wave.md` 的「**Y1**」。
> 本文件是该线的**执行契约**；§4 判据线**逐字**抄录 Y1，执行期间未改。
> **§4c 记录唯一与契约明文的偏离（6k 快照 token 预算）及其裁决**——那是本线唯一的
> 契约参数变更，由主 Agent 明确裁决（不是执行者自行放宽）。
> 入口：`train_arms.sh`（训练四臂）+ `run.py`（评测电池）；语义层 `arms.py`（四臂定义）；
> 单测 `tests/test_composable_arms_13m.py`；变异 `mutate.py`；对账 `reconcile_numbers.py`；
> 原始结果 `results.json`；判读 `analysis.md`；权重清单 `MODEL.md`；对照说明 `controls/README.md`。

## 1. 命题

**ticket 38-W3 因引擎硬互斥丢掉了 GQA 因子——ENG1（39-X2）解除限制后，把 GQA 放回
「carry 通道 + 可见窗口」的组合里：叠加 GQA / carry / window 后，长程召回与主干质量
各付多少、得多少？** 一句话可证伪的形式：*把特性一格格叠上去，每一步的合成召回增量
（相邻臂差）能不能被二项 CI 区分。*

本线填的正是 W3 的缺口：W3 只测了「MHA + delta carry + window16」在评测期 int8 量化下的
表现（2618 的 U2 检查点），**GQA 因子未测**（W3 README §4b）。ENG1 之后，`GQA × carry ×
window` 这个组合从「引擎里不存在」变成「可建可跑」（2622 已钉住可测性），本线把它**训练并
评测**——W3 缺的那一格，现在有数。

## 2. 单因素声明（阶梯口径）

- **基底**：本仓 stock dense decoder-only trunk（13.5M：d256 × 12 层 × 8 头 × mlp6 × T256，
  tied 读出 + 顺序残差），`DepthARModel(dense=True, topology="sequential")`。
- **唯一变量 = 特性的叠加**（§17.14 的 `A→B→C→D` 阶梯）。四臂按**同一份配方**逐字段训练
  （同 seed=0、同数据生成器、同步数/批大小/学习率/优化器、同评测协议），相邻两臂只差
  一个特性：

  | 臂 | 叠加到这一步的特性 | 引擎位点 | 参数量 |
  |---|---|---|---|
  | `stock`（基准） | — | `Block` | 13,536,768 |
  | `gqa` | `+GQA`（2 KV 头） | `GQABlock` | 12,352,512 |
  | `gqa_carry` | `+carry`（delta 写） | **`ComposableBlock`** | 12,616,808 |
  | `gqa_carry_window` | `+window`（16 / anchors 0） | `ComposableBlock` + `attn_mask` | 12,616,808 |

- **阶梯的正确读法（写死在契约里，防止事后挑读法）**：
  - **增量是逐叠加步的相邻两臂之差**（`stock→gqa`、`gqa→gqa_carry`、
    `gqa_carry→gqa_carry_window`），**不是**只看净和（§17.17）；
  - `window` 那一步**同时**改训练掩码与评测掩码（它是「制造长程需求」的那一步，
    见 §3），因此 `gqa_carry_window − gqa_carry` 的口径是「在已装好 GQA+carry 的模型上
    **施加长程需求**的净效应」，不是纯算法增量——报告里必须这样写。
- **训练/评测可见性一致**：前三臂全程全因果；只有 `gqa_carry_window` 用 `window=16 /
  anchors=0`（训练与评测**同一个** `window_anchor_keep_mask`）。这条决定了每臂的 native
  评测口径（`run.py` 的 `native_visibility`）。
- **无「必要性例外」**：四臂的任务、token 预算、评测批逐字段相同；窗口不是例外而是第 4 臂的
  特性本身。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

四臂**不改任何引擎代码**（本线不写 `pathlm/`、不写 `train_depth_ar.py`），只是把 ENG1 已
解开的三个既有开关**组合起来**：

- **GQA（`--n-kv-heads 2`）**：注意力 K/V 的头数从 8 降到 2（`GQABlock` / `ComposableBlock`
  的 KV 头投影，`enable_gqa=True` 按组广播）。真进 SDPA 的 K/V 形状是 `[B, 2, T, 32]`
  （活性量，§5 判据 ⑤）。
- **carry（`--carry-write delta`）**：单共享记忆槽，`S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；
  每层读出 `a ← a + γ ⊙ mem_t`，γ **单边零初始化**（起步恒等于无通道臂、非鞍点）。
- **window（`--train-window 16 --train-anchors 0`）**：`attn_mask` 是**与块类正交的装饰层**，
  `keep[i,j] = (j≤i) ∧ ((i−j)<16)`。它把源 key/value 段 `[0, 2K)` 推到窗外 ⇒ 不搬就答不出。

**ENG1 的作用点**：`gqa × carry` 同时开时走 `ComposableBlock`（ENG1 前该组合引擎报
`ValueError`、块类二选一）。ENG1 把 KV 头投影从块类解耦、窗口做成正交装饰层，三特性才可
共存（2622 `tests/test_eng1_composable.py` 钉住）。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向、**不得当判据**；本地条目给文件路径。可检索时应查的关键词见
> 每块末尾。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果（记忆·未核 / 本地） | 我们的增量 |
|---|---|---|---|---|
| 1 | **GQA / MQA**：Shazeer 2019（MQA）、Ainslie 等 2023（GQA，Llama-2 采用） | KV cache 随 `n_kv_heads` 等比缩小、推理访存下降 | 同尺寸下质量近似 MHA、解码更快 | 本地 2618/2619 未测 GQA；**本线**在 13M、局部先验抗性合成任务上，把 GQA 与 carry+window **组合**起来量其增量 |
| 2 | **滑动窗口注意力**：Mistral 2023、Longformer 2020 | 长上下文下把注意力限制在局部窗口 | 局部任务近无损、长程靠分层/门控补齐 | 我们的窗口是**训练期可见性掩码 + 制造长程需求的手段**（§19），不是为了省算 |
| 3 | **显式写入通道 / delta rule（fast weight）**：Schmidhuber 1992；Schlag 等 2021（delta rule）；Yang 等 2024（Gated DeltaNet） | 用固定状态把窗口外信息搬到可达范围 | 本地 2618：无通道 0.0137≈chance、delta 写 0.0950（6.08×） | **本线**在该正面结果上叠加 GQA，问「KV 头共享会不会破坏搬运」 |
| 4 | **GQA 与低比特/组合压缩的交互**（KIVI / SmoothQuant / vLLM `kv_cache_dtype`） | KV 访存瓶颈下省内存 | 本地 R1（2614）：int8 Δbpc +0.00015、近无损 | 2618-W3 想测「GQA×int8×carry」但因引擎互斥未成；ENG1 后本线补上组合因子的**质量**面（W3 的压缩面仍属其自身 scope） |

**若可检索应查**：`GQA MQA quality-MHA-equivalent small scale`、`sliding window attention
long-range retrieval`、`delta rule fast weight memory MQAR`、`GQA delta net combination`。

## 4. 判据线（**逐字**抄录 Y1 契约，一字未改）

> ## Y1 · ENG1 解锁的四臂质量实验（13M，同 token 预算，≈40–60 min）
> - 目录：`axes/1-time/1.1.4.1-explicit-memory/experiments/2623-composable-arms-13m/`
> - 动机：ticket 38-W3 因引擎硬互斥只能丢掉 GQA 因子；**ENG1（ticket 39-X2）已解除限制**。
> - 四臂（唯一变量 = 特性叠加，逐字段同配方）：`stock` / `+GQA(2 KV 头)` / `+GQA+carry(delta)` /
>   `+GQA+carry+window(16, anchors 0)`。任务 = `assoc-mqar` 合成关联回忆（**不要**用 enwik8 当主任务）。
> - 判据（预登记）：① **两腿**（合成召回 + clean bpc）；② 每个叠加步的**增量**（避免只看净和，§17.17）；
>   ③ 毁源对照臂；④ 参数量差显式声明（GQA −8.75% 量级，按 §15 说明是否影响归因）；
>   ⑤ 活性量（KV 头形状 / carry read_share / window mask）每 200 步。

**执行时的操作性定义（不改判据，只把口径落定）**：

- **(1) 的「两腿」**：
  1. **主腿 = 合成关联回忆准确率**（`assoc_recall_batch`，K=8，T=256，chance=1/64）；
     每臂在 **native 可见性**下评测（训练/评测同一掩码），并**另跑**一套**全因果交叉口径**
     （所有臂 `keep_mask=None`）并报——两口径下逐臂读数都出。
  2. **第二腿 = clean bpc**（`retention.masked_bpc`，enwik8 评测尾段，全因果）。
     合成训练 ⇒ 真实文本上的通用 LM 能力本就未建立，此腿是**非退化锚**（证明「叠加特性
     没把主干弄坏」），**不作质量判据**（口径限制见 `analysis.md`）。
- **(2) 的「每个叠加步的增量」**：见 §2 的阶梯读法；同时报**净和**与**三列相邻差**（§17.17
  的分量分解：净和不可排序时不得只报净和）。判据线用 **pp**，方差带 = 1.4pp。
- **(3) 的「毁源对照臂」**：把源 KV 段 `[0, 2K)` 换成同分布新随机符号（`corrupt_span`），
  其余（查询行/答案/距离）逐位不变；**任一臂毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废**。
- **(4) 的「参数量差」**：四臂参数量逐臂实测并声明（GQA −1,184,256 = −8.75%；
  carry +264,296 = +1.95%；window +0）。GQA 是**剪枝类**（减参数）⇒ 按 §15 例外，与
  「同 FLOPs」比不适用；归因声明见 `analysis.md §5`。
- **(5) 的「活性量每 200 步」**：KV 头形状（`kv_heads`）、carry（`attn_carry.read_share` 等）、
  window mask（`window_mask`）都由 `train_depth_ar.py` 每 200 步写进同一行日志；
  `run.py` 逐臂核对「该开关对应行数 > 0」并摘末行。

## 4b. 能力面分析（SCHEMA §22.8，派发前逐条做）

契约要求的每个 flag 组合，引擎是否支持？`grep -n "ValueError\|互斥" pathlm/depth_ar.py`
列出的校验逐条核对（ENG1 后）：

| 组合 | ENG1 前 | ENG1 后 | 本线 |
|---|---|---|---|
| `n_kv_heads<8` ⊕ `carry_rule` | 硬互斥 | **允许**（`ComposableBlock`，`depth_ar.py:1853-1860`） | `gqa_carry` 臂 |
| `n_kv_heads<8` ⊕ `train_window>0` | 硬互斥 | **允许**（窗口是正交装饰层） | `gqa_carry_window` 臂 |
| `carry_rule` ⊕ `train_window>0` | 允许（2618 已用） | 允许 | `gqa_carry_window` 臂 |
| 三者齐开 | 不存在 | **可建可跑**（2622 钉住） | `gqa_carry_window` 臂 |
| 另两种注意力算子（`attn_sparse_k` / `attn_linear`）⊕ 上述任一 | 仍互斥 | **仍互斥**（本线不用） | — |

块类选择（`DepthARModel.__init__`）：`attn_sparse_k>0`→`TopKBlock`；`attn_linear`→
`LinearAttnBlock`；`carry ∧ gqa`→`ComposableBlock`；`gqa`→`GQABlock`；`carry`→
`CarryChannelBlock`；否则 `Block`。四臂恰好命中后四条，**不存在不可达组合**（CPU 前向
四臂全部 OK，见 `tests/test_composable_arms_13m.py::test_each_arm_builds_and_lands_expected_block_class`）。

## 4c. 与契约明文的唯一偏离：6k 快照 token 预算（**主 Agent 裁决**）

契约写 `≈40–60 min`、**未**写同步数（「同 token 预算」只约束臂间相等）。**先测步时**后
发现契约隐含的 12k（= 2618 口径）在本卡当前环境下超预算：

| 臂 | 实测步时（300 步稳态，ms/step） | 12k×4 推算 |
|---|---|---|
| `stock` | 78.8 | — |
| `gqa` | 78.8 | — |
| `gqa_carry` | 163.6 | — |
| `gqa_carry_window` | 151.5 | — |
| 合计 | 472.7 ms/step-全臂 | **≈95 min 训练**（+ 臂间窗口 + 评测 ≈ 102 min） |

**环境争用（外因）**：共享 4090 上另有两个 llama-server 常驻（合占 ~18.7 GB、util 100%），
本线实测步时比 2618 当年空卡（U0≈45 / U2≈73 ms/step）慢 **1.7–2.2×**。这是**共享卡礼貌线**
之外的负载，不是本线的排队问题。

**裁决**：(A) **6k 步**——四臂一律 `6k × 32 × 256 = 49.0M token`（臂间严格同预算）。
理由：Y1 是四臂**内部**阶梯对照（基准 = `stock` 臂），token 预算只要**臂间相等**即可；
契约未写 12k（Y3 才写 12k）⇒ 不算改判据。四条记账（见 `analysis.md`）：① 臂间严格同预算、
任何一臂不得单独加步；② 显式声明「**6k 快照 token 预算，与 2618 的 12k 不可比**」；
③ 附**收敛门禁**（§17.15，落在答案行 CE 上）；④ 如实记账环境争用与降速倍数。

## 5. 对照与方差

- **基准 = `stock` 臂**（无任何叠加的 stock decoder-only trunk；本仓的「基底恒为 stock 对照」口径）。
  阶梯的每一列都相对**相邻前一臂**，净和相对 `stock`。
- **方差带 = 0.014 bpc**（`docs/methodology.md`）；召回判据用 **Wilson 95% CI** 与
  `two_proportion` 的差值 CI（1.4pp 召回方差带仅作参考）。
- **局部先验地板** = `local_prior_accuracy(k=8)`（§19：合成任务上局部先验必须 ≈ chance）；
  **chance** = `1/sym_vocab` = 1/64 = 0.015625。
- **逐臂毁源对照**（§4 (3)）。
- **纯推理、无随机采样**：同 seed 重跑应逐位相同。
- **逐字段对照表**（唯一变量 = 特性叠加）：

| 字段 | stock | gqa | gqa_carry | gqa_carry_window |
|---|---|---|---|---|
| seed | 0 | 0 | 0 | 0 |
| 训练数据 | assoc-mqar | 同 | 同 | 同 |
| 步数×批×T | 6000×32×256 | 同 | 同 | 同 |
| lr / 优化器 | 1e-3 / AdamW+cosine | 同 | 同 | 同 |
| `n_kv_heads` | 8(=MHA) | **2** | 2 | 2 |
| `carry_rule` | None | None | **delta** | delta |
| `train_window/anchors` | 0/0 | 0/0 | 0/0 | **16/0** |
| 评测口径（native） | 全因果 | 全因果 | 全因果 | 窗口 16 |

## 6. 成本（§8 的唯一预算数字 + 砍什么）

```
gpu_wall_budget: 60 min   # 含四臂训练 + 评测
```

- **实测步时**（先测后定，§17.8）：见 §4c；训练四臂 + 臂间 65s 窗口 + 评测实测量见
  `results.json::cost` 与 `train_arms.runlog`。峰值显存逐臂实测 ≪5GB net（≈1.5–2.2GB）。
- **预算不够时砍什么**（§17.7）：砍步数（**四臂同步砍**，不砍单臂）> 砍评测批数/距离档；
  **不砍判据线**（两腿、逐叠加步增量、毁源对照、参数量声明、活性量都必须出）。
  本次触发：12k→6k（§4c，主 Agent 裁决）。
- **§20.1 锁记账**：`train_arms.sh` **按臂**取/放锁、臂间 65s 窗口（不整程独占）；
  本波另有 GPU 线（Y2/Y3）在同一锁上串行，等待时长与先后记入 `train_arms.runlog`（本线
  四臂实际训练合计 39.8 min，其余为等锁）。
- **§20.1 第 2 条并发例外登记**：两次冒烟（`run.py --smoke`）为**纯推理、墙钟 0.14 min、
  峰值 290.8 MB < 1 GB**，且 `nvidia-smi` 空闲 ≥3 GB（跑前 4618 MiB、跑后 4623 MiB）⇒
  **未取锁、未动锁目录**。正式评测（`run.py` 全电池，0.79 min、峰值 542.8 MB）经
  `/tmp/pathlm_gpu.lock` 排队后运行。

## 7. 数据与脚本清单

- **训练数据**（§19：**不用 enwik8 当长程主任务**）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`（K=8 对、Q=8 查询、gap=14）。
- **评测**（合成长程任务）：`assoc_recall_batch`（主腿 + 毁源对照）、
  `var_recall_batch`（距离腿）。
- **clean bpc 腿**：`data/enwik8_full.npz` 的评测尾段（vocab 206，与模型词表一致）。
- **入口**：`train_arms.sh`（训练）、`python3 run.py`（评测，`config.json`）。
- **落盘**：`results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}`、
  `train_<arm>.log`、`analysis.md`、`MODEL.md`、`controls/README.md`、
  `reconcile_numbers.py → reconcile_report.json`、`mutate.py → mutation_report.json`、
  `run_stdout.txt`、`smoke/`。
- **测试**：`tests/test_composable_arms_13m.py`（9 条可失败单测 + 变异验证 `mutate.py`）。

## 8. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：native 口径与全因果口径是否分列？chance/局部先验是否同值？毁源是否同一批？
   → 先修 `run.py` 的接线，**不得**动模型。
2. **实现**：carry 臂 `read_share = 0`（通道没被用）或 GQA 臂 `k_shape[1] ≠ 2`（只改标签）
   → 跑变异验证排除死开关；read_scale 仍单边零 = **步数**问题（6k 未逃逸），如实记录、
   不外推。
3. **规模**：四臂都 ≈ chance ⇒ 该任务/该尺度上机制本就没起作用（读 2618 的 U0/U2 作前提）；
   **不得**表述为「组合不可能」（SCHEMA §19）。
4. **设计**：某臂毁源后仍高 ⇒ 读数混杂、该臂作废；`window` 步的负增量若只来自评测口径变难
   （而非机制失效）⇒ 用全因果交叉口径与 2618 参照线分开读，并在结论里写清是设计还是口径。

## 9. 产出与复核

- `analysis.md` 给**人话摘要 + 六问**（§23）+ 逐格表格 + 判读；`MODEL.md` 给权重哈希与
  代码指纹；`controls/README.md` 给对照臂与变异说明。
- 本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context **只读复核者**独立检查。

## scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验是 M1 载体类型格内的**组合阶梯**
（ENG1 解锁的那一格），唯一变量 = 特性的叠加。评测期 KV 量化（W3 的压缩面）不在本线 scope；
`attn_sparse_k` / `attn_linear` 与组合的交互仍受引擎互斥限制，是显式留白、不是遗漏。
`deps` 只记录已知耦合方向（M1 载体类型 ← ENG1 引擎可组合性；与 2618 的 U2 参照线耦合）。
