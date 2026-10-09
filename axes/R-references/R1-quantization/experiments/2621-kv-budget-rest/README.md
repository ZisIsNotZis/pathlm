# 2621 · KV 预算图剩余缺失组合（ticket 38 · W1 线）—— 实验契约

> 来源契约：`.scratch/38-auto-parallel/issues/01-auto-parallel.md` §「W1 · KV 预算图剩余缺失组合」。
> 本文件是该线的**执行契约**；判据线**逐字**抄录在 §4，执行期间未改。
> 入口：`run.py`（纯推理 + 汇总，**不训练**）；新组合臂的唯一语义层：`rest_arms.py`（`combo_mask`）；
> 既有组合语义复用 `../2620-kv-budget-gaps/gaps_arms.py`（`quant_head` / `window_anchor`，**只读不改**）；
> 仪器：`tools/evalkit/longrange_tasks.py`（合成长程任务电池，**不用 enwik8 当长程主任务**）、
> `tools/evalkit/kv_quant.py`（KV 量化往返）、`tools/evalkit/retention.py`（可见性掩码）。
> 参考臂评估管线直接复用 `2619` 的 `run.py`（`importlib` 载入，**不改它**）⇒ 与 U3 的图
> **同一条路径、同一 seed、同一任务**。被测量模型：trunk / head 两族 checkpoint 复用（只读，
> 运行时断言 sha256）；见 `MODEL.md`。

---

## 1. 命题

`2620` 补齐了 U3 列出的最优先三个组合（GQA×int8 / MQA×int8 / 窗口×锚点）；本线补齐**仍缺的
五个优先组合**，每个都用**两条腿**（clean bpc + 合成长程召回）在同一仪器上测：

1. **GQA × int4**：把 2620 的「头共享 × int8 近无损」外推到 int4，头共享是否仍能承受 int4？
2. **MQA × int4**：同上，最激进的头共享×量化组合（384 B/位置）。
3. **窗口 × int8**：int8 能否叠在窗口上而不额外掉质量？（两因子在同一字节轴上）
4. **GQA × 窗口**：窗口因子能否救回被头共享削掉的召回？（两因子在同一字节轴上）
5. **锚点 × int4**：锚点通道在 int4 下是否幸存？（锚点是 A3 的「靶向」手段）

可证伪点：若 `int4` 组合的**量化增量**（相对同拓扑 fp16 臂）显著大于 int4 单用（trunk 的
+0.0457），则「组合更省」不成立、int4 不是通用可叠因子；若某组合**两条腿结论不一致**
（bpc 近无损但召回掉，或反之），必须显式标注。

## 2. 单因素 / 组合声明（本批被授权的例外）

- 本实验**不训练**、不加参数、不改权重：它是**测量 + 汇总**类实验。
- 仓库总 scope 是「只做单轴单格、不做组合实验」。**本批由契约 W1 明确授权**继续补五个
  **组合**缺口；每个组合只叠加**两个已单测的因子**，且每个因子各自的语义都由既有仪器或
  `rest_arms.py` / `gaps_arms.py` 的单测钉住。
- 单因素无法独立生效的说明：`combo_mask` 把「可见性掩码」与「KV 量化」放进**同一前向**；
  单测证明其两条退化路径逐位还原既有仪器（`mode="fp32"` ⇒ `retention.masked_logits`；
  全因果掩码 ⇒ `kv_quant.kv_quant_logits`）⇒ 这不是「另一个模型」。
- 基线 = 主流标准 decoder-only transformer（13.5M dense：12 层 × d256 × 8 头 × mlp6 ×
  seq256，tied 读出、仅最终深度 CE）。各族全精度基线：
  - **trunk 族**（量化 / 窗口 / 窗口+锚点 / 窗口×量化 / 锚点×量化）：`trunk.pt`
    （clean bpc 1.483569）；
  - **head 族**（MHA/GQA/MQA、GQA×窗口）与 **quant_head 族**（GQA/MQA × int4）：
    `mha-12k-s0/model.pt`（clean bpc 1.482026）。组合臂的主 damage 列相对 mha，
    以便与 U3 的 GQA/MQA 档**同轴**；另单列「量化增量」= 组合臂 − 同架构 fp16 臂。
- 两族基线差 0.0015（同配置同 seed 重训的 run-to-run 散度），已单列，未混。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 数据来源 |
|---|---|---|
| 横轴（唯一口径） | `bytes_per_position` = `2(K,V) · n_layers · n_kv_heads_eff · head_dim · value_bytes · (positions_kept / T)`，T = 256；`window_anchor` / `combo_mask` 档 `positions_kept = window + anchors` | 本实验解析计算（`rest_arms.bytes_per_position` / `gaps_arms.bytes_per_position`，与 2619 同一式） |
| 代价（质量腿 1） | clean bpc（越低越好）+ `damage_bpc = bpc_arm − bpc_族基线` + `quant_increment_bpc` + `window_increment_bpc` | 本实验**同进程统一路径**重测（`2619.bpc_of`） |
| 代价（质量腿 2） | 合成长程召回（copy / induction / var_recall，越高越好）+ Wilson 95% CI + **copy 分 src_pos** | `tools/evalkit/longrange_tasks.py`，本实验测 |
| 参照系 | chance（1/64）/ 局部先验地板（只看最近 8 个 token 的最强预测器）/ **逐臂毁源对照** | 同仪器内置 |
| 收益（内存） | 相对全精度基线的每位置字节下降（**解析口径**） | 解析公式 |
| 口径隔离 | **实测显存峰值**单列一节，**不与解析字节同列** | 本实验 GPU 实测 |
| 可控性 | ① 同 seed 内置自检；② 与 2619 参考臂 / 2620 组合臂**逐位**对照（跨 run） | `reproducibility_vs_2619` / `reproducibility_vs_2620` |
| 活性 | `combo_mask`：掩码挡掉的因果对（`dropped_frac`）+ 量化 `rel_err` + 锚点放行的因果对数；`quant_head`：真实张量形状 + 量化 `rel_err` | `rest_arms.activity_of` / `gaps_arms.activity_of` |

**两条腿的分工必须写清**：bpc 腿量的是「语言建模依赖」，合成召回腿量的是「不靠局部先验
能不能取回远处信息」（SCHEMA §19）。**bpc 不是主判据**——本项目最贵的入账错误（假保留）
正是「bpc 改善与长程检索退化同时发生」。

## 4. 判据线（预登记，逐字）

> 契约原文（`.scratch/38-auto-parallel/issues/01-auto-parallel.md` §「W1」）：
>
> **W1 · KV 预算图剩余缺失组合（纯 eval ≈20 min）**
> - 目录：`axes/R-references/R1-quantization/experiments/2621-kv-budget-rest/`
> - 补齐 U3/V1 之后仍缺的组合（U3 列了 10 项、V1 补了 3 项）：优先 `GQA×int4`、`MQA×int4`、
>   `窗口×int8`、`GQA×窗口`、`锚点×int4`；**判据同 V1**（两腿 + 毁源 + 同横轴 + 解析/实测分列 + 单点不外推）。
> - 交付：一张更新后的总表（含**所有已测点**，标出来源实验路径与口径）。
>
> 判据同 V1（`.scratch/37-gaps/issues/01-gaps.md` §「契约 V1」，逐字）：
>
> - **判据（预登记）**：每点两腿（clean bpc + 合成长程召回，用 `tools/evalkit/longrange_tasks.py`），
>   与 U3 的既有图**同横轴同口径**；**禁止单点外推**；**毁源对照臂必做**；解析内存与实测显存分开列。
>   若某组合的两腿结论**不一致**（bpc 近无损但召回掉），必须显式标注——这正是我们已两次踩到的坑。
> - 成本 ≈20 min（纯 eval + 汇总）；峰值 ≤5GB。

**本契约把它展开为可失败判据**：

- **C1 横轴唯一性**：所有点的 `bytes_per_position` 由**同一个解析公式**算出（`combo_mask`
  同时改 `value_bytes` 与 `positions_kept`；`quant_head` 同时改 `n_kv_heads_eff` 与 `value_bytes`）。
  任何点若无法用该公式表达 ⇒ 不入图；`value_bytes` 与 `mode` 不一致 ⇒ 直接报错。
- **C2 双列隔离**：解析字节（`bytes_per_position`）与**实测显存峰值**
  （`measured_gpu_peak_mib`）必须是**两列/两节**；不得把实测显存写进横轴。
- **C3 两腿全覆盖 + 同口径**：每个点都用**同一件仪器、同一 seed、同一任务定义**测出
  clean bpc 与合成召回；参考臂的 bpc/召回必须与 2619 / 2620 **逐位一致**（阈值 1e-5）。
- **C4 不可分辨如实标注**：某腿落在 `chance + 2σ_wilson` 内 ⇒ 标 `at_floor: true`，
  读作「在该任务上不可分辨」，**不得**写成「机制无效」（SCHEMA §19）。
- **C5 毁源对照臂必做**：copy 主腿的源片段整段换随机符号（**逐臂**，源区间 =
  该档自己的 `[src_pos, src_pos+32)`）；若读数来自被搬运的信息，毁源后必须落回 chance。
- **C6 禁止单点外推**（SCHEMA §17.11）：单点只描述；趋势需 ≥2 个直接测点。窗口×int8 在
  win32/win64 两档、GQA×窗口在 win64/win128 两档、锚点×int4 在 (16,16)/(32,32) 两档上读；
  `gqa_int4` / `mqa_int4` 是单点，只描述该点。
- **C7 两腿不一致显式标注**：若某组合 bpc 近无损但召回掉（或反之），`analysis.md`
  必须单列该点并解释。
- **C8 数字对账**：`python3 reconcile_numbers.py` 退出码 0（`analysis.md`/`table.md`
  的每个数字都能在 `results.json` 里找到同值）。

## 5. 既有做法（prior art，SCHEMA §18）

> 检索条件声明：**本环境联网检索不可用**（未注册 web / 检索工具）。因此外部条目一律标
> `记忆·未核`，只作设计方向，**不得当判据**；本地条目给文件路径。

1. **KV 量化（int8 / int4）**。谁做过：KIVI、SmoothQuant 等 KV cache 量化谱系。
   为什么：值字节口径上用每值 1 / 0.5 字节替代 2 字节、看起来省内存。实测（`记忆·未核`）：
   int8 近无损、int4 有明显质量损失——与本地 R1 `2614` 一致（int8 +0.00015、int4 +0.0457）。
   **部署口径（见 `2631-kv-budget-causal`）**：严格因果/流式量化必需**逐值 fp16
   标尺**（每值 +2 B），int8 实为 3 B/值、int4 为 2.5 B/值 ⇒ 反比 fp16 基线（2 B/值）多占
   50%/25%，**「省内存」在严格因果口径下不成立**。
   **我们的增量**：本批问的是 int4 **叠加**（叠头共享 / 锚点）后代价是否被放大——本地新测。
2. **头共享（GQA / MQA）**。谁做过：Shazeer 2019（MQA）、Ainslie 等 2023（GQA）。
   为什么：解码时 KV 的访存是瓶颈，把 K/V 头数 8→2/1 按比例降字节。实测（`记忆·未核`）：
   GQA 质量接近 MHA、显著省内存，已是主流默认。**我们的增量**：放到**同一字节横轴**、
   用**局部先验抗性任务**测（本地 W3 `2616`：GQA damage +0.0143）。
3. **窗口 + 锚点 / attention sink**。谁做过：StreamingLLM（Xiao 等 2023，`记忆·未核`）。
   为什么：注意力对最前几个 token 有「汇」质量，保留它们稳定分布。**我们的增量**：A3 已测
   锚点通道（needle 76.8%→93.3%）；本批给锚点档 × int4 的两腿读数。
4. **GQA/MQA × int8/int4 的组合**。谁做过：生产推理栈普遍同时用 GQA + int8/int4 KV
   （vLLM/TensorRT-LLM 一类，`记忆·未核`）。为什么：两个省内存因子正交、叠加最省。
   **我们的增量**：本地从未在**受控单变量 + 局部先验抗性任务**上量过 int4 叠加后的两腿；
   本批测出「int4 叠加在头共享上代价被放大 2–4×」——与「组合必然更省」的直觉相反。
5. **窗口 × 量化**。谁做过：长上下文服务常见「滑窗 + KV 量化」。**我们的增量**：把两者放在
   同一字节轴、给两腿读数。
6. **头共享 × 窗口**。谁做过：滑动窗口注意力 + GQA 的组合在长上下文模型里常见
   （Mistral 一类，`记忆·未核`）。**我们的增量**：在 13M 受控设定与合成召回腿上量它。

**若可检索应查**：`KIVI 2bit KV cache quantization`, `KV cache int4 quality loss`,
`GQA + int4 KV cache combination quality`, `StreamingLLM attention sink`,
`sliding window attention GQA long context retrieval`, `KV cache quantization retrieval degradation`.

## 6. 对照与方差

- **chance** = `1/64`；**局部先验地板** = `local_prior_accuracy(k=8)`（copy / induction /
  var_recall 各一）；**毁源对照**逐臂。
- **方差带** = `0.014`（`docs/methodology.md`）：`|damage_bpc| < 0.014` 标「⚠不可排序」。
- **召回判据** = Wilson 95% CI；`at_floor` = `chance + 2σ_wilson`。
- 纯推理、无随机采样；同 seed 重跑应逐位相同。参考臂与 2619/2620 的对照是**跨 run 可控性**
  的直接证据（`results.json::reproducibility_vs_2619` / `reproducibility_vs_2620`，实测
  `max_bpc_abs_diff = 0.0`）。

## 6b. 变异验证（新语义层的可失败单测 + 抓红 + 逐字节还原）

新语义层 `rest_arms.py` 的 19 个单测见 `tests/test_kv_budget_rest.py`。**变异验证**：
把语义开关故意改坏 → 对应单测抓红 → 从备份恢复 → 逐字节校验 sha256 一致。

| 变异 | 抓红的单测 | 结果 |
|---|---|---|
| `VALUE_BYTES["int4"]` 0.5 → 1.0 | `test_combo_mask_bytes_match_single_formula` | 1 failed, 3 passed |
| `masked_kv_quant_logits` 前向跳过量化（`mode` 固定 fp32） | `test_combo_mask_quant_and_mask_change_logits` / `test_combo_mask_int8_reports_real_quantization` / `test_activity_combo_mask_reports_both_factors` | 3 failed |
| `keep_mask_of` 忽略锚点（`anchors` 固定 0） | `test_keep_mask_of_matches_shared_instruments` / `test_anchor_positions_visible_to_last_query` / `test_activity_anchor_contribution_pairs` | 3 failed |

`rest_arms.py` sha256 变异前后：
`973b80338e011f3790ce9e7c368b640d522f3a08daea4288fa54c49de9618571`（一致，逐字节还原）。

## 7. 文件清单

| 文件 | 内容 |
|---|---|
| `README.md` | 本契约（判据线逐字） |
| `run.py` | 复现入口（复用 2619 管线 + 本目录新臂） |
| `rest_arms.py` | 新组合臂（`combo_mask`）的唯一语义层 |
| `config.json` | resolved config（含 27 臂、任务、seed、判据） |
| `results.json` | 原始结果（逐臂 bpc / 召回 / 活性 / 对照 / 对账） |
| `table.md` / `chart.txt` / `chart.png` | 更新后的总表（含所有已测点）+ 一张图（与 U3 同横轴同口径） |
| `analysis.md` | 判读（人话摘要 + 逐点表格 + 仍缺组合；失败也写） |
| `reconcile_numbers.py` → `reconcile_report.json` | 三道对账（SCHEMA §12），实测 587/587 通过 |
| `controls/` | 对照臂与方差界（口径定义） |
| `MODEL.md` | 权重清单：哈希 / 体积 / 生成命令（权重本身不入库） |

## 8. 成本

`gpu_wall_budget: 20 min`（纯 eval + 汇总；契约 W1「≈20 min」）。实测墙钟 **2.373 min**、
峰值 **462.5 MiB**（27 臂，全在预算内）。预算不够时砍任务档数（先 var_recall，再 induction），
保 copy 主腿与全部预算点；**不砍判据线**（SCHEMA §8/§17.7）。

## 9. 复现

```bash
# GPU 工作（先取 /tmp/pathlm_gpu.lock，见 SCHEMA §20）
python3 run.py            # 全跑（≈2.4 min）
python3 run.py --smoke    # 冒烟（缩档，实测步时/显存，不外推时间）
python3 run.py --charts-only   # 只重画图/表（纯 CPU，不取锁）
python3 reconcile_numbers.py   # 三道对账（587/587）
python3 -m pytest tests/test_kv_budget_rest.py -q   # 19 绿
```

## 10. 权重与共享文件

- 被测量 checkpoint 全部**复用**（trunk / mha / gqa / mqa），只读 + 运行时 sha256 断言；
  权重不入库。清单见 `MODEL.md`。
- 本实验**不修改任何共享文件**：新逻辑全部放在**新模块** `rest_arms.py`；`gaps_arms.py`
  （2620）与 `2619/run.py`、`tools/evalkit/*` 只被**读取 / `importlib` 载入**。故本线未持
  `/tmp/pathlm_code.lock`，也不涉及「回读邻线标记 / 禁止回滚他人改动」（SCHEMA §22 第 7 条
  的「更优解」路径）。
- GPU 面上本线持有 `/tmp/pathlm_gpu.lock`（取锁 → 写 `holder_pid` + `heartbeat` → 释放先清
  内容再 `rmdir`）；整程 **<10 min** ⇒ 允许整程持锁（SCHEMA §20.1 第 1 条的例外）。
  **未发现**等锁超时 / 锁泄漏 / 显存超预算。

> **等字节 ≠ 等形状**：跨形状排名须带该声明（详见 `analysis.md` 末尾的口径说明）。
