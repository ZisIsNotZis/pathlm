# 2652 · 激活量化最小判决（13M 受控）—— 实验契约

> **契约来源**：本文件的判据线落盘自 `config.json`（**运行前**写入，mtime 早于 `results.json`）的
> `criterion` / `quantizer` / `arms` / `cost` / `eval` 字段，以及
> `axes/5-quant/5.3-activation-quant/project.yaml`（本项 SSOT：`purpose` / `method` / `validation` /
> `objective_criteria` / `falsification` / `gaps`）。口径定义与判据线以 `config.json::criterion` 为准；
> 本文件是它的落盘副本 + 设计说明。
>
> 入口：`run.py`（纯推理 + 汇总，**不训练**；`--smoke` 估时；`--table-only` 纯 CPU 重画表）；
> 语义层：`aq_arms.py`（激活假量化的**唯一**新模块，`tools/evalkit/*` 与其它实验目录**只读复用**）；
> 单测 / 变异：`selftest_aq.py` → `mutation_report.json`（7 变异全抓红 + 逐字节还原）。
> 判读：同目录 `analysis.md`；表：`table.md`；结果：`results.json`；日志：`run_log.txt`；
> 权重清单：`MODEL.md`；过程流水：`JOURNAL.md`。

---

## 1. 命题（可证伪）

**在 13M 受控设定下，把前向的中间激活以 int8 / 4 位存放（含每 token 动态标尺），
「激活字节 → 序列级质量（clean bpc + 长程召回）」的代价是什么？离群通道吃掉多少位？**

三个必须回答的子命题：

1. **激活字节**（含每 token/通道标尺）从 fp16 的 2 B/元素降到多少？每 token 的**部署**字节是多少？
2. **质量代价**：int8 / 4 位激活相对 fp16 的 clean bpc 代价是否越过 **0.014 bpc 带**（配对 Δ、≥5 窗口）？
   长程召回（copy / induction / var_recall）是否可分辨下降？
3. **离群通道**：逐层激活的通道量级分布把多少位吃掉了？在 6×median 的回退阈值下有多少通道
   需要回退到高精度、会吃掉标称省幅的多少？

## 2. 单因素声明

- 被测模型是**主流标准 decoder-only 变压器**（13.5M dense：12 层 × d256 × 8 头 × mlp6 × seq256，
  tied 读出、仅最终深度 CE），复用 2610 的冻结 checkpoint（`checkpoints.trunk.sha256` 运行时断言），
  **结构不改、参数不增（`params_added = 0`）、不训练、KV 精度不动、主 FLOPs 不变**。
- **唯一改动的一个因素** = 前向 `nn.Linear` **输入激活的存储精度**（对称假量化往返，per-token
  动态标尺）。张量集固定为每层 4 个 Linear 的输入（`qkv` / `proj` / `mlp0` / `mlp2`；见 §3）。
- **臂（先隔离激活因子，再把组合作为部署点并列）**：

  | 臂 | 激活 | 权重 | 作用 |
  |---|---|---|---|
  | `fp16` | fp16（恒等，不引入往返） | fp16 | 基线；**必须与历史锚 1.483569 逐位一致**（历史锚口径 = 24×16@seed2；主表缩口径见 §6） |
  | `A8` | int8 per-token | fp16 | **主因子臂**：隔离纯激活因子（缺它则结论被权重因子混杂） |
  | `A4` | int4 per-token | fp16 | 激活位宽的单因素效应（与 A8 只差位数） |
  | `W8A8` | int8 per-token | int8 per-channel | **组合臂（显式标注为组合）**：部署组合点；字节账分列 |

- **实现层 equality（SCHEMA §9 / §22）**：`AqForward(activation_mode="fp16")` 逐行复刻
  `Block.forward` 的 dense 分支，`run.py::instrument` 钉住它与既有仪器
  `retention.masked_logits` **逐位相同**，且 `bpc_of` 与独立路径 `retention.masked_bpc`
  在主表缩口径与历史锚口径下**各自逐位相同**（`results.json::anchor_single_point`），
  并在历史锚口径（24×16@seed2）下与历史锚 1.483569 一致（实测 1.483568730，差 2.70e-07）
  ⇒ 「唯一改动的因子只有激活精度」有实现层证据。
- **组合声明**：`W8A8` 是 `project.yaml` 预登记的部署组合点（`innovation` / `method` 字面写入），
  本实验把它与两个纯激活臂**并列**，并把它与 `A8` 的差单列为「权重因子」，不混进激活因子。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| **横轴（主口径）** | `bytes_per_token_activation = Σ_tensor in_features·bits/8·L + (整型 ? 2·4·L : 0)`，L=12、d=256、mlp=6 ⇒ 27648 元素/token | `aq_arms.activation_bytes_per_token`（与 `fake_quant_activation` 的 `scale.numel()==B·T` 逐张量对账） |
| 权重字节（**分列**） | 被量化 Linear 的整模型字节（`int8 = out·in·1 B + out·2 B`） | `aq_arms.weight_bytes`（组合臂；占全模型参数 93.1%） |
| 质量腿 1（该不跌/可排序） | clean bpc（越低越好）+ `Δbpc vs fp16` + `max|配对Δ|` | 同进程、同一路径（`b21.b19.bpc_of`，复用 2621/2619） |
| 质量腿 2（序列级） | 合成长程召回 copy / induction / var_recall + Wilson 95% + `at_floor` + 毁源对照 | `tools/evalkit/longrange_tasks.py`（SCHEMA §19：不用 enwik8 当长程主任务） |
| 参照系 | chance = 1/64 = 0.0156；局部先验（最近 8 token）；逐臂毁源对照 | 同仪器内置 |
| **活性（该稳定）** | 逐层激活标尺分布（min / median / max）、`changed_frac` / `rel_err` / `clipped_frac`、`n_scales`；**离群通道**：`outlier_index = max/median` 通道峰值、`effective_bits_median_channel = bits − log2(outlier_index)`、`revert_chan_frac_6x`、回退字节代价 | `aq_arms.summarize_stats` / `revert_byte_penalty` |
| 判据带 | **0.014 bpc 带**（预登记常数；口径 = 配对 Δ、≥5 窗口） | `config.json::criterion` |
| 可控性 | ① 复刻前向 ↔ 既有仪器逐位；② fp16 臂 ↔ 历史锚 / `masked_bpc` 独立路径；③ 变异测试全抓红；④ 因果性探针；⑤ §20 取锁/释放记账 | `results.json::instrument` / `gpu_lock` |

**明确不在本轴的东西**：KV 每位置字节（那是 `5.1` 的账）、权重每参数字节（`5.2`）、
实测显存峰值（评测实现把激活立刻反量化 ⇒ 峰值不随档下降，只报解析列，与 KV 轴同一纪律）、
墙钟 / 吞吐（不进判据、不进 `cost_model`）。

## 4. 判据线（预先写下，不许事后调整）

**逐字抄自 `config.json::criterion.verbatim_contract`（运行前写入）：**

1. 单因素：固定被测量模型/checkpoint/数据流/评测窗口，只改中间激活精度。臂 = {fp16 激活（基线）,
   int8 激活（W8A8）, 4 位激活（W4A8）}；不训练权重、不改 KV 精度、不新增参数。基底恒为 stock
   decoder-only。
2. 该跌：中间激活字节/访存（含每 token/通道标尺）；该不跌/可排序：clean bpc（须越 0.014 噪声带）；
   该稳定：逐层激活标尺与离群通道分布（活性对照列）；必须不变：因果性与非量化臂逐位不变；
   必须报：标尺粒度与开销。
3. 若激活量化的 clean bpc 代价越噪声带且吃掉全部字节收益（相近字节点位上不占优）⇒ 判该档在本尺度
   不成立；若离群通道迫使部分层回退到高精度、使省幅远小于标称 ⇒ 判该实现在该粒度下不可用；
   若非量化臂不能逐位还原 ⇒ 判实现有误（对照失效）。用哪个带（C1 裁决）：0.014 bpc 带（限配对 Δ、
   ≥5 窗口）。

**判据带口径（预登记 + 在力声明）**：主判 = **0.014 bpc 带**，适用条件是**配对 Δ、≥5 窗口**。
本实验跑 **5 个窗口 seed（2,3,4,5,6）** ⇒ **满足 ≥5**，因此**直接用预登记的 0.014**，不需要退用
自测在力带。**为什么这不循环**：0.014 是与本批数据无关的预登记常数（`config.json` 运行前写入），
不是从本批的 `max|Δ|` 反推出来的（对比 `2632` 的「用本批 3 点 max 当带」之弊；那里 ≤3 窗口不满足
标准）。本实验另报「单臂跨窗口 bpc 极差」作**描述性**对照，它**不是**判据带。
**率类指标**（召回）走 Wilson 95% CI，**不借用** 0.014。

**召回地板规则**：`recall ≤ chance + 2σ_wilson(chance)` ⇒ `at_floor = true`（读作「该任务上
不可分辨」，不得写成「机制无效」，SCHEMA §19）。
**禁外推**：单点只描述；结论只作这 4 个点（13M / enwik8 尾段 / T=256 / 单 seed 冻结 checkpoint）。
**符号约定**：`damage_bpc = bpc_臂 − bpc_fp16`（正 = 更差）。

## 5. 既有做法（prior art，SCHEMA §18）

| # | 谁做过 | 为什么这么设计 | 我们的增量 |
|---|---|---|---|
| 1 | **SmoothQuant**（Xiao et al. 2023, `arXiv:2211.10438`）：把激活离群难度**数学上迁移**到权重上（`Y = (X·diag(s)⁻¹)·(diag(s)·W)`），从而做到 W8A8 | 激活的难点是离群通道把整条激活的量化台阶拉粗；迁移后权重承担难度 | **本项不重造该迁移算法**（那部分是纯造轮子）。增量 = 在 13M 受控设定下把**含标尺的激活字节账**与「激活字节 → 序列级质量」的前沿量化到可判定，并把「离群通道」做成**可失败的活性读数** |
| 2 | **LLM.int8()**（Dettmers et al. 2022, `arXiv:2208.07339`）：离群通道（magnitude > 6.0）走 fp16 的**混合精度分解**，其余走 int8 | 离群通道主导动态范围，混精度比整条提精度便宜 | 本实验**不实现回退**，而是把「回退需要多少通道、吃掉多少省幅」做成**反事实判别**（`revert_chan_frac_6x` + `saving_eaten_frac`）——回答否证条件里那条「离群通道是否迫使部分层回退」 |
| 3 | **Atom**（Zhao et al. 2023, `arXiv:2310.19102`）：W4A4 低比特服务系统 | 低比特服务希望同时压权重与激活 | 只作背景（低比特服务的可行性参照），**不作判据**；本实验的判据是本仓自测的 bpc/召回带 |
| 4 | 本仓 `5.1` KV 量化（`2631`/`2632`/`2642`）：块标尺 KV 量化的字节-质量前沿 | 同一「同字节天平」方法在 KV cache 上的实例 | **口径划界**：`5.3` 只压中间激活，KV 与权重不动；**激活的难点是动态性（每个 token 不同）** ⇒ 标尺粒度（per-token）与离群通道分布必须是可失败读数 |

> **检索说明**：以上 arXiv 编号由 `ticket 66 §4` 本轮联网核实标题；论文内数字一律**不引用、不作判据**。
> `fp8`（1 B/值、无显式标尺）是 `5.5` 的课题，本实验**未测**，不得据此断言 int8 是最优 8 位方案。

## 6. 对照与方差

- **诚实对照**：`fp16` 恒等臂同场；`A8` 与 `W8A8` 的差 = 纯权重因子；`A4` 与 `A8` 的差 = 纯位数因子。
- **毁源对照**：copy 主腿的源片段整段换随机符号（其余逐位不变），逐臂报 —— 测读数是否真来自被搬运的信息。
- **方差与判据带**：`config.json::eval` 的窗口序 = seed 2（主窗）+ seed 3/4/5/6。**实跑口径是 CPU
  预算裁剪后的缩批量**（主窗 **3×16**、附加窗口 **2×16**，见 `config.json::eval.n_batches` /
  `extra_seed_n_batches` 与 `run_log.txt` 的「缩批量：main3/extra2/copy1/ind1/var1」行）⇒ 与 5.1 的
  历史锚 1.483569（**24×16**@seed2）**不同批数口径、不可直接比**（3×16 与锚的 |Δ| = 8.95e-03）；
  这也是本轮**超出** §7 预登记裁剪条款范围的缩口（§7 只授权额外 seed 的 `n_batches` 24→8 与砍副腿），
  如实登记见 `analysis.md` §5.3。每臂在**同一 seed 上用同一批数据**测 ⇒ **配对 Δ 仍然成立**
  （`batch()` 的取数只依赖 seed 与窗口、与批数无关 ⇒ 窗口/子集效应在差分中被消掉）；5 个窗口都
  满足「配对 Δ」口径 ⇒ 0.014 带的 ≥5 窗口条件满足，但 5 个窗口**批数不等**（主窗 3、附加 2）
  ⇒ **不得**称「同口径、无 `n_batches` 混杂」。判据量只用**同缩口径、同窗口序**的配对 Δ 的
  `max|Δ|`，与预登记 0.014 带比较；另报「单臂跨窗口极差」作描述。历史锚口径的 `fp16` 单点
  （24×16@seed2）单列在 `results.json::anchor_single_point`，**只作口径对照、不进主表**。
- **seed**：模型是单 seed 冻结 checkpoint（训练侧散布未知）⇒ 本实验**不**据此宣称结构优劣。
- **重跑**：纯推理、无随机采样；同 seed 重跑应逐位相同。

## 7. 成本

- **GPU 墙钟预算 ≤ 30 min**（`config.json::cost.gpu_wall_budget_min`，只作过程预算，不进判据 /
  `cost_model`）。整程 < 10 min ⇒ SCHEMA §20.1 允许整程持锁；否则在臂间释放锁（`gpu_lock.segments`）。
- **显存预算 ≤ 5 GiB**；实测峰值见 `results.json::cost.peak_mem_mib`。
- **§20 共享卡记账（机器可读）**：`results.json::gpu_lock` = `holder_pid` / `wait_seconds` /
  `free_mib_before` / `free_mib_after` / `release_ok` / `held_whole_run` / `segments` /
  `peak_mem_mib_this_run`。等锁上限 10 min（40×15s），**不杀任何别的进程**；拿不到锁则如实返回。
- **预算不够时砍什么**（预登记）：先砍额外 seed 的 `n_batches`（24→8），再砍副腿
  （induction / var_recall），保 `fp16`/`A8`/`A4`/`W8A8` 四臂、clean bpc 主腿与全部判据线；
  **不砍判据线**（SCHEMA §8/§17.7）。**本轮实跑的裁剪超出本条款的授权范围**（主窗 24→3、
  额外 seed 24→2、copy 主腿 2→1），已如实登记为「超出预登记裁剪条款的缩口」，见 §6 与
  `analysis.md` §5.3。
- **可中断性**：`--table-only` 只从 `results.json` 重画表（纯 CPU，不取 GPU 锁）。

## 8. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | `data/enwik8_full.npz`（末 10M 评测尾段跑 clean bpc）；**不入库** |
| 合成任务 | `tools/evalkit/longrange_tasks.py`（copy / induction / var_recall） |
| 被测量模型 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`（sha256 运行时断言；见 `MODEL.md`） |
| 入口 | `python3 run.py` / `--smoke` / `--table-only` |
| 语义层（新文件） | `aq_arms.py`（激活假量化 + 字节账 + 活性量） |
| 单测 / 变异 | `selftest_aq.py`（17 单测全绿；7 变异全抓红）→ `mutation_report.json` |
| 结果 | `results.json`（4 臂 + 仪器自检 + 配对 Δ + 离群判定 + GPU 记账 + 代码哈希） |
| 表 / 判读 | `table.md`（表 1–4）· `analysis.md`（人话摘要 + 三道对账 + 失败与局限） |
| 日志 | `run_log.txt`（逐行读数） |

## 9. 失败决策树

按此顺序排除（前一档不成立才进下一档）：

1. **对照失效**：`fp16` 臂若与历史锚 / `masked_bpc` 不逐位一致 ⇒ **本实验作废**，先修实现
   （不许用「大概是浮点误差」搪塞；复刻前向逐位是硬门禁）。**口径限定**：该判据只在历史锚口径
   （`window_seed=2` + 24×16）下成立，由 `results.json::anchor_single_point` 的补跑单点承载
   （实测 1.483568730，与锚差 2.70e-07）；主表缩口径（3×16）与锚的差是**口径差**而非
   「对照失效」（§6 / `analysis.md` §2②）。
2. **测量**：`--smoke` 复检；量化臂 `changed_frac == 0` ⇒ 干预是死的（死开关），先修语义层；
   变异未抓红 ⇒ 单测无效，先修单测。
3. **字节账**：`aq_arms.activation_bytes_per_token` 与实现测到的 `n_scales` /
   `n_tensors_observed` 对不上 ⇒ 直接报错（不许出数）。
4. **尺度**：只有 13.5M / 单 T=256 / 单 seed / enwik8 尾段 ⇒ 结论只作**这些点**的描述；
   不得外推到更大模型、别的 checkpoint 或别的 T。
5. **离群通道的可分辨性**：`outlier_index` 用 `max/median`（单点极值统计）⇒ 对单个离群通道敏感，
   样本量（4096 token × 每张量 in_features）足够时稳定；若某层 `median` 极小（> 1e-8 clamp），
   该层的 `outlier_index` 会失真 ⇒ 读 `chan_amax_median` 原值判断，并在 `analysis.md` 如实标注。

## 10. 产出

- `results.json`：4 臂 × {激活字节（含标尺）、clean bpc（5 窗口）、三条召回腿 + Wilson CI、
  逐层/逐张量活性与离群分布、因果性探针、权重活性} + 仪器自检 + 配对 Δ + 离群判定 +
  GPU 锁记账 + 代码哈希。
- `table.md`：表 1 臂总表 / 表 2 逐层字节 + 离群活性 / 表 3 配对 Δ / 表 4 三道对账 + 四条硬护栏。
- `analysis.md`：人话摘要（第一段）+ 正面对照 + **三道对账** + 失败与局限 + `gaps`（下一步最小
  实验 + 等效算力成本）。
- `mutation_report.json`：单测与变异的可失败性证据。
