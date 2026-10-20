# 2620 · KV 预算图的缺失组合（ticket 37 · V1 线）—— 实验契约

> 来源契约：`.scratch/37-gaps/issues/01-gaps.md` §「契约 V1」。
> 本文件是该线的**执行契约**；判据线**逐字**抄录在 §4，执行期间未改。
> 入口：`run.py`（纯推理 + 汇总，**不训练**）；新组合臂的唯一语义层：`gaps_arms.py`；
> 仪器：`tools/evalkit/longrange_tasks.py`（合成长程任务电池，**不用 enwik8 当长程主任务**）、
> `tools/evalkit/kv_quant.py`（KV 量化往返）、`tools/evalkit/retention.py`（可见性掩码）。
> 参考臂评估管线直接复用 `2619` 的 `run.py`（`importlib` 载入，**不改它**）⇒ 与 U3 的图
> **同一条路径、同一 seed、同一任务**。
> 被测量模型：trunk / head 两族 checkpoint 复用（只读，运行时断言 sha256）；见 `MODEL.md`。

---

## 1. 命题

U3 的统一 KV 预算图（`2619`）在同一横轴「每位置 KV 字节」上画出了四类手段，但留了
**10 个显式缺失组合**。本线补其中**最优先的三个**，每个都用**两条腿**（clean bpc +
合成长程召回）在同一仪器上测：

1. **GQA × int8**：两个因子（KV 头数、每值字节）叠加，能否在 **~1536 B/位置**做到
   **两腿近无损**？（U3 的第一推荐）
2. **窗口 × 锚点/sink**：给窗口加一个「前 `anchors` 个位置恒可见」的锚槽，锚点是不是
   **拉动窗口召回最便宜的杠杆**？（A3 的锚点通道在字节预算轴上的定位）
3. **MQA × int8**：若便宜则补。（768 B/位置）

可证伪点：若组合臂的**两腿结论不一致**（bpc 近无损但召回掉，或反之），必须显式标注；
若锚点只在**锚区源**（`src_pos=0`）有效、对通用召回（`src_pos=96/191`）无效，则
「锚点是通用召回杠杆」这一说法不成立——锚点只是**靶向**手段。

---

## 2. 单因素 / 组合声明（本批被授权的例外）

- 本实验**不训练**、不加参数、不改权重：它是**测量 + 汇总**类实验。
- 仓库总 scope 是「只做单轴单格、不做组合实验」。**本批由契约 V1 明确授权**补三个
  **组合**缺口（GQA×int8 / 窗口×锚点 / MQA×int8）；每个组合只叠加**两个已单测的因子**，
  且每个因子各自的语义都由既有仪器或本目录 `gaps_arms.py` 的单测钉住。
- 基线 = 主流标准 decoder-only transformer（13.5M dense：12 层 × d256 × 8 头 × mlp6 ×
  seq256，tied 读出、仅最终深度 CE）。各族的**全精度基线**：
  - **trunk 族**（量化 / 窗口 / 窗口+锚点）：`trunk.pt`（clean bpc 1.483569）；
  - **head 族**（MHA/GQA/MQA）与 **quant_head 族**（GQA/MQA × int8）：`mha-12k-s0/model.pt`
    （clean bpc 1.482026）。组合臂的主 damage 列相对 mha，以便与 U3 的 GQA/MQA 档**同轴**；
    另单列「量化增量」= 组合臂 − 同架构 fp16 臂，隔离 int8 的叠加代价。
- 两族基线差 **0.0016**（同配置同 seed 重训的 run-to-run 散度），已单列，未混。

---

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| 横轴（唯一口径） | `bytes_per_position` = `2(K,V) × n_layers × n_kv_heads_eff × head_dim × value_bytes × (positions_kept / T)`，T = 256；`window_anchor` 档 `positions_kept = window + anchors` | 本实验解析计算（`gaps_arms.bytes_per_position`，与 2619 同一式） |
| 代价（质量腿 1） | clean bpc（越低越好）+ `damage_bpc = bpc_arm − bpc_全精度基线` + `quant_increment_bpc` | 本实验**同进程统一路径**重测（`2619.bpc_of`） |
| 代价（质量腿 2） | 合成长程召回（copy / induction / var_recall，越高越好）+ Wilson 95% CI + **copy 分 src_pos** | `tools/evalkit/longrange_tasks.py`，本实验测 |
| 参照系 | chance（1/64）/ 局部先验地板（只看最近 8 个 token 的最强预测器）/ **逐臂毁源对照** | 同仪器内置 |
| 收益（内存） | 相对全精度基线的每位置字节下降（**解析口径**） | 解析公式 |
| 口径隔离 | **实测显存峰值**单列一节，**不与解析字节同列** | 本实验 GPU 实测 |
| 可控性 | ① 同 seed 内置自检；② 与 2619 参考臂**逐位**对照（跨 run） | 本实验 + `reproducibility_vs_2619` |
| 活性 | `quant_head`：头共享**真实张量形状** + 量化 `rel_err`/`changed_frac`；`window_anchor`：`dropped_frac` + 锚点放行的因果对数 | `gaps_arms.activity_of` |

**两条腿的分工必须写清**：bpc 腿量的是「语言建模依赖」，合成召回腿量的是「不靠局部先验
能不能取回远处信息」（SCHEMA §19）。**bpc 不是主判据**——本项目最贵的入账错误（假保留）
正是「bpc 改善与长程检索退化同时发生」。

---

## 4. 判据线（预登记，逐字）

> 契约原文（`.scratch/37-gaps/issues/01-gaps.md` §「契约 V1」）：
>
> - 目录：`axes/5-quant/5.1-kv-quant/experiments/2620-kv-budget-gaps/`
> - 目标：补齐 ticket 36-U3 列出的 10 个缺失组合里最优先的三个：
>   1. **GQA × int8**（两者叠加是否能在 ~1536 B/位置做到两腿近无损——U3 的第一推荐）；
>   2. **窗口 × 锚点/sink**（锚点是拉动窗口召回的**最便宜**办法吗？这正是 A3 的锚点通道在
>      字节预算轴上的定位）；
>   3. **MQA × int8**（若便宜）。
> - **判据（预登记）**：每点两腿（clean bpc + 合成长程召回，用 `tools/evalkit/longrange_tasks.py`），
>   与 U3 的既有图**同横轴同口径**；**禁止单点外推**；**毁源对照臂必做**；解析内存与实测显存分开列。
>   若某组合的两腿结论**不一致**（bpc 近无损但召回掉），必须显式标注——这正是我们已两次踩到的坑。
> - 成本 ≈20 min（纯 eval + 汇总）；峰值 ≤5GB。

**本契约把它展开为可失败判据**：

- **C1 横轴唯一性**：所有点的 `bytes_per_position` 由**同一个解析公式**算出
  （`quant_head` 同时改 `n_kv_heads_eff` 与 `value_bytes`；`window_anchor` 的
  `positions_kept = window + anchors`）。任何点若无法用该公式表达 ⇒ 不入图。
- **C2 双列隔离**：解析字节（`bytes_per_position`）与**实测显存峰值**
  （`measured_gpu_peak_mib`）必须是**两列/两节**；不得把实测显存写进横轴。
- **C3 两腿全覆盖 + 同口径**：每个点都用**同一件仪器、同一 seed、同一任务定义**测出
  clean bpc 与合成召回；参考臂的 bpc/召回必须与 `2619` **逐位一致**（阈值 1e-5）。
- **C4 不可分辨如实标注**：某腿落在 `chance + 2σ_wilson` 内 ⇒ 标 `at_floor: true`，
  读作「在该任务上不可分辨」，**不得**写成「机制无效」（SCHEMA §19）。
- **C5 毁源对照臂必做**：copy 主腿的源片段整段换随机符号（**逐臂**，源区间 =
  该档自己的 `[src_pos, src_pos+32)`）；若读数来自被搬运的信息，毁源后必须落回 chance。
- **C6 禁止单点外推**（SCHEMA §17.11）：单点只描述；趋势需 ≥2 个直接测点。锚点杠杆
  只在 `win32_anchor8(40)` / `win16_anchor16(32)` / `win32_anchor32(64)` 三个直接测点
  与各自的纯窗口等字节对照之间读。
- **C7 两腿不一致显式标注**：若某组合 bpc 近无损但召回掉（或反之），`analysis.md`
  必须单列该点并解释。
- **C8 数字对账**：`python3 reconcile_numbers.py` 退出码 0（`analysis.md`/`table.md`
  的每个数字都能在 `results.json` 里找到同值）。

---

## 5. 既有做法（prior art，SCHEMA §18）

> 检索条件声明：**本环境联网检索不可用**（未注册 web / 检索工具）。因此外部条目一律标
> `记忆·未核`，只作设计方向，**不得当判据**；本地条目给文件路径。可检索时应查的关键词见
> 每块末尾。

1. **头共享（GQA / MQA）**。谁做过：Shazeer 2019（MQA）、Ainslie 等 2023（GQA）。
   为什么：解码时 KV cache 的**访存**是推理瓶颈，把 K/V 头数 8→2/1 直接把每位置字节按比例
   降。实测结果（`记忆·未核`）：GQA 在质量上接近 MHA、显著省内存，已是主流 LLM 默认。
   **我们的增量**：不是复现机制，而是把它放到**同一字节横轴**、用**局部先验抗性任务**测
   （本地 W3 `2616`：GQA damage +0.0143、MQA +0.0117，召回 CI 与 mha 重叠）。§21 分诊：
   属 (a) 已做透 ⇒ 本地数只作**同尺度复核**（`reference`），本图不重测机制。
2. **KV 量化（int8/int4）**。谁做过：KIVI、SmoothQuant 等 KV cache 量化谱系。
   为什么：用每值 1/0.5 字节替代 2 字节，同样省访存。实测（`记忆·未核`）：int8 近无损、
   int4 有明显质量损失——与本地 R1 `2614` 一致（int8 +0.00015、int4 +0.0457）。
   **我们的增量**：本地在**同一图**上把 int8/int4 与窗口/头共享对撞（`2619`）。
3. **窗口 + 锚点 / attention sink**。谁做过：StreamingLLM（Xiao 等 2023，`记忆·未核`）
   用「保留前几个 sink token + 滑动窗口」让流式解码不需重 prefill。为什么：注意力对
   最前几个 token 有异常大的「汇」质量，保留它们稳定分布。实测（`记忆·未核`）：保留 sink
   即可用固定窗口跑长流。**我们的增量**：A3 已在本地测过锚点通道（needle 76.8%→93.3%，
   真实文本针口径）；本图给锚点档**解析字节**与**合成长程召回**，问它是不是**最便宜**的
   召回杠杆——这是 U3 表 4 明确缺失的一格。
4. **GQA/MQA × int8 的组合**。谁做过：生产推理栈普遍**同时**用 GQA + int8/int4 KV
   （vLLM/TensorRT-LLM 一类，`记忆·未核`）。为什么：两个省内存因子正交、叠加最省。
   实测（`记忆·未核`）：组合是默认做法。**我们的增量**：本地从未在**受控单变量 + 局部
   先验抗性任务**上量过叠加后的**两条腿**（业界组合多以吞吐/困惑度报告，不控局部先验）。
   §21 分诊：属 **(b) 设定实质不同**（13M / 受控探针 / 旁路形态）⇒ 值得本地补测。

**若可检索应查**：`GQA grouped query attention Ainslie`, `MQA Shazeer`,
`KIVI KV cache 2bit quantization`, `SmoothQuant`, `StreamingLLM attention sink`,
`attention sink window anchor streaming`, `GQA int8 KV cache serving vLLM`,
`KV cache quantization long context retrieval`.

---

## 6. 对照与方差

- **chance** = `1/64`；**局部先验地板** = `local_prior_accuracy(k=8)`（copy / induction /
  var_recall 各一）；**毁源对照**逐臂。
- **方差带** = `0.014`（`harness/methodology.md`）：`|damage_bpc| < 0.014` 标「⚠不可排序」。
- **召回判据** = Wilson 95% CI；`at_floor` = `chance + 2σ_wilson`。
- 纯推理、无随机采样；同 seed 重跑应逐位相同。参考臂与 `2619` 的对照是**跨 run 可控性**
  的直接证据（`results.json::reproducibility_vs_2619`）。

---

## 6b. 变异验证（新语义层的可失败单测 + 抓红 + 逐字节还原）

新语义层 `gaps_arms.py` 的 13 个单测见 `tests/test_kv_budget_gaps.py`。**变异验证**：
把语义开关故意改坏 → 对应单测抓红 → 从备份恢复 → 逐字节校验 sha256 一致。

| 变异 | 抓红的单测 | 结果 |
|---|---|---|
| `positions_kept` 丢掉 `anchors`（只算 `window`） | `test_window_anchor_bytes_add_positions` | 1 failed |
| `keep_mask_of` 忽略锚点（`anchors=0`） | `test_anchor_positions_are_visible_to_last_query` | 1 failed |
| `quant_head` 前向不量化（固定 fp32） | `test_quant_head_int8_changes_logits_and_values` | 1 failed |

`gaps_arms.py` sha256 变异前后：
`2787e99678bcfd624eee3ff90d52a8dcadeef70a45c9d225c0d632318d791b6e`（一致，逐字节还原）。

## 7. 文件清单

| 文件 | 内容 |
|---|---|
| `README.md` | 本契约（判据线逐字） |
| `run.py` | 复现入口（复用 2619 管线 + 本目录新臂） |
| `gaps_arms.py` | 新组合臂的唯一语义层（`quant_head` / `window_anchor`） |
| `config.json` | resolved config（含 13 臂、任务、seed、判据） |
| `results.json` | 原始结果（逐臂 bpc / 召回 / 活性 / 对照 / 对账） |
| `table.md` / `chart.txt` / `chart.png` | 一张表 + 一张图（与 U3 同横轴同口径） |
| `analysis.md` | 判读（逐格表格；失败也写） |
| `reconcile_numbers.py` → `reconcile_report.json` | 三道对账（SCHEMA §12） |
| `controls/` | 对照臂与方差界 |
| `MODEL.md` | 权重清单：哈希 / 体积 / 生成命令（权重本身不入库） |

## 8. 成本

≈20 min（纯 eval + 汇总）预算；实测见 `results.json::cost`。峰值 ≤5GB。
预算不够时砍任务档数（先 var_recall，再 induction），保 copy 主腿与全部预算点。

## 9. 复现

```bash
# GPU 工作（先取 /tmp/pathlm_gpu.lock，见 SCHEMA §20）
python3 run.py            # 全跑
python3 run.py --smoke    # 冒烟（缩档，实测步时/显存，不外推时间）
python3 run.py --charts-only   # 只重画图/表（纯 CPU，不取锁）
python3 reconcile_numbers.py   # 三道对账
```

## 10. 权重与共享文件

- 被测量 checkpoint 全部**复用**（trunk / mha / gqa / mqa），只读 + 运行时 sha256 断言；
  权重不入库。清单见 `MODEL.md`。
- 本实验**不修改任何共享文件**（`2619-kv-budget-chart/run.py` 只以 `importlib` 载入；新增逻辑全在
  `gaps_arms.py`），故不涉及 SCHEMA §22 的代码编辑锁。
