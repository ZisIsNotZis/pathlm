# 2622 · ENG1 引擎可组合性（GQA × carry × window）· 代码 + 单测 + 冒烟

> 契约来源：`.scratch/39-close-gaps/issues/01-close-gaps.md` §「X2 · ENG1 引擎可组合性」。
> **副本标注**：§4 判据与 §2 交付段**逐字**抄录契约；其余小节为**摘要**（并注明）。
> 判据线一律未改。
> 入口：`run.py`；单测：`tests/test_eng1_composable.py`；变异：`mutate.py`；
> 对账：`reconcile_numbers.py`；原始结果：`results.json`；冒烟：`smoke/master/`；
> 权重清单：`MODEL.md`；判读：`analysis.md`。

## 1. 命题

**ENG1 前，`GQA × carry × window` 这个组合在引擎里根本不存在**（三特性硬互斥 +
块类二选一），所以它**从未被测过**；本线把它变成**可测**——`GQA`（KV 头投影）从
block 类解耦、窗口掩码做成与两者正交的装饰层，缺省路径逐位还原 v1。

本线**不追求性能**，只追求**可测性**（SCHEMA §23 第 6 问的工程面）。

## 2. 单因素声明

唯一被改动的引擎结构 = **block 的组装方式**（KV 头投影 / 写入通道 / 窗口掩码三者的
可组合性），**不改任何单特性的语义**：

- 缺省（MHA、无 carry、无 window）与 stock `Block` **逐位一致**（权重 + 前向，单测钉住）；
- 单特性路径的块类**不变**：GQA-only → `GQABlock`、carry-only → `CarryChannelBlock`、
  缺省 → `Block`；只有 `GQA × carry` 同时开时走新块 `ComposableBlock`；
- `train_depth_ar.py` **未改动**（本线只 grep 断言其邻线标记仍在）。

无「必要性例外」。

**交付（逐字抄录契约 §X2「交付」段）**：

> 交付：把「KV 头投影」从 block 类解耦、**窗口掩码做成与两者正交的装饰层**；缺省路径**逐位还原 v1**
> （§22：先声明锚点、改完 grep 断言邻线标记仍在——本轮**唯一**写 `pathlm/depth_ar.py`/`train_depth_ar.py` 的线）。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本节的结论**全部是「记忆（未核）」**——执行者（worker）本批**无 web
> 检索工具**。它们**只用于设计方向，不得当判据**（SCHEMA §18）。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果（记忆·未核） | 我们的增量 |
|---|---|---|---|---|
| 1 | **GQA / MQA**：Shazeer 2019（MQA）、Ainslie 等 2023（GQA，Llama-2 采用） | KV cache 随 `n_kv_heads` 等比缩小、推理访存下降 | 同尺寸下质量近似 MHA、解码更快 | 我们把「KV 头投影」抽成**可组合组件**，使它与写入通道/窗口可同开（W3 的组合在引擎里原本不存在） |
| 2 | **滑动窗口注意力**：Mistral 2023、Longformer 2020 | 长上下文下把注意力限制在局部窗口 | 局部任务近无损、长程靠分层窗口/门控补齐 | 我们的窗口是**训练期可见性掩码**（评测与训练共用同一函数），且作为与块类正交的装饰层 |
| 3 | **可组合 / 参数化架构搜索式的 block 组装**：如 HuggingFace 的 attention 接口 `enable_gqa`、FlashAttention 的 mask 参数 | 把一个注意力特性做成参数，而非一类模型 | 接口参数化后，特性间可自由组合 | 本工程里原本是「块类二选一 + 互斥校验」；我们改成「投影组件 + 掩码装饰」，让组合可测 |
| 4 | **线性注意力 / 状态空间混合（Hybrid）**：Jamba、Zamba 等 | 交替纯注意力层与 SSM 层 | 混合在长上下文上省内存 | 与本线无关（本线只统一**已有**三特性的组装，不引入新机制） |

**要查什么才能确定**（留给有检索能力的情报线）：`GQA MQA arithmetic intensity KV cache`、
`Mistral sliding window attention`、`FlashAttention mask parameterization`、
`attention variant composition interface`。

## 3. 指标表（逐项分析）

| 指标 | 解决什么问题 | 代价 | 有效性（vs 对照） | 收益与用法 | 可控性 | 副产品 |
|---|---|---|---|---|---|---|
| 组合可建（块类 + qkv 宽度） | 「组合存在」这一事实 | 无（CPU 组装） | vs 契约 W3 的组合（原本报错） | 后续「组合实验」的入场券 | flag 组合全可控 | — |
| qkv 输出宽度 | KV 头投影<u>真的</u>变窄（不是标签） | 无 | MHA = 3d / GQA = d+2·nkv·dh | 形状类活性证据 | `n_kv_heads` | — |
| `kv_telemetry.k_shape` | 真进 SDPA 的 K/V 张量形状 | 一次前向 | GQA = `[B,nkv,T,dh]` | 证明是内核行为不是文档承诺 | — | — |
| `attn_carry.read_share` | 写入通道真的被读出 | 一次前向 | 单边零起步 = 0；激活 γ 后 > 0 | 通道活性的运行证据 | `carry_scale` | — |
| 窗口 vs 全窗口 logits 差 | 窗口掩码真的生效 | 两次前向 | 同 seed 同权重 ⇒ 差异只来自掩码 | 掩码活性证据 | `train_window` | — |
| 缺省逐位一致 | ENG1 不改缺省行为 | 无 | vs stock `Block`（权重 + 前向） | 「逐位还原 v1」的护栏 | — | — |

## 4. 判据线（**逐字**抄录契约 §X2「验收（硬）」段，未改一字）

> **验收（硬）**：① 组合可失败单测：`GQA×carry`、`GQA×window`、`carry×window`、三者各一条（**变异验证抓红**）；
> ② 既有 361 测试全绿；③ 20 步冒烟证明三者共存可训练（活性量落盘）；④ 报告明确写出「ENG1 前/后，哪些组合从未被测过、现在可测」。
> - 说明：本线**不追求**性能提升，只追求**可测性**（这是 SCHEMA §23 第 6 问的工程面）。

**执行时的操作性定义（不改判据，只把阈值落定）**：

- **(1) 的「组合可失败单测」**：`tests/test_eng1_composable.py` 四条组合各一条 + 缺省逐位
  一致 + 快路径块类不变；「可失败」= 变异验证（`mutate.py`，见 `controls/`）四下破坏全部抓红。
- **(2) 的「既有 361 测试全绿」**：基线 361 + 本次新增 6 = 367，`pytest tests/ -q` 全绿
  （原始输出 `pytest_full.txt`）。**注**：两条断言「旧护栏」的既有用例（`test_kv_heads_…`
  的 `train_window` 互斥、`test_carry_config_rejects_combinations` 的 `n_kv_heads` 互斥）
  与 ENG1 目标**直接冲突**，按裁决把它们改为**正向断言**（组合现在合法）；判据线未改。
- **(3) 的「20 步冒烟」**：`run.py --smoke` 在 M1 引擎上跑 `--train-window 16 --carry-write
  delta --n-kv-heads 2`（三特性齐开，ENG1 前不存在的组合）20 步；「活性量落盘」= 末行
  `train_log.jsonl` 同行为 `kv_heads`（GQA）+ `attn_carry`（carry）+ `window_mask`（window）
  三块。**只证明共存可训练，不外推**。
- **(4) 的「ENG1 前/后」**：见 §9 产出与 `analysis.md` 的人话摘要表。

## 5. 对照与方差

- **对照臂**（每个写明：输入分布 / 要预测什么 / 部署对应）：
  1. `default`（MHA、无 carry、无 window）：stock `Block`。输入 = CPU 随机 token；要预测 =
     逐深 next-token；部署对应 = 「什么都不开」的基底。
  2. `full-window`（`train_window=12=seq_len`）：同 seed、同权重、窗口恒等因果。输入分布 =
     同组合；要预测 = 同目标；部署对应 = 「窗口开关关掉」的对手——窗口臂与它之比只含掩码。
  3. `GQABlock` / `CarryChannelBlock` 快路径：同配置下单特性走的块类。部署对应 = 既有实现。
- **方差**：本实验是**确定性**的（同 seed 组装与 CPU 前向逐位一致）。窗口活性判据是
  「logits 是否逐位不同」（布尔），不涉及 bpc 方差界（Δ<0.014）。**无**多 seed——
  本线不比较质量，只验证可测性。
- **噪声界**：不适用（无训练质量读数）。冒烟 20 步的 loss 只作「可训练」的证据，无统计含义。

## 6. 成本

```
gpu_wall_budget: 20 min   # 契约上限（≈20 min）；含冒烟 + 全量 pytest（CPU）
```

- 冒烟：M1 引擎（13.08M）× 20 步 × bs8 × seq256 = 0.041M token；峰值显存实测见
  `results.json::cost`（≪5GB net）；**先测步时**再定预算（SCHEMA §17.8），**不外推**训练时间。
- 组合自检 = CPU 前向（秒级）；全量 `pytest` = CPU（不占 GPU）。
- **预算不够时砍什么**（SCHEMA §17.7）：砍组合自检的重复档与冒烟批大小；**不砍判据线**
  （四条组合单测、361 回归、20 步冒烟、ENG1 前后清单都必须出）。
- 契约 §8：本文件只给一个 GPU 预算数字（20 min）。

## 7. 数据与脚本清单

- **数据**：冒烟用合成多查询关联回忆（`tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`），
  **不用 enwik8**（SCHEMA §19）。
- **入口脚本**：`run.py`（`--smoke` 跑冒烟；缺省摄取产物）。
- **resolved config**：`config.json`（组合位点 + 冒烟命令 + seed）。
- **落盘文件**：`results.json`、`smoke/master/{train_log.jsonl,gate_log.jsonl,results.json,
  model.pt,run_stdout.txt}`、`pytest_eng1.txt`、`pytest_full.txt`、`mutation_report.json`、
  `reconcile_report.json`、`analysis.md`、`MODEL.md`、`controls/README.md`。

## 8. 失败决策树（无效时的处置路径：测量 → 实现 → 规模 → 设计）

1. **测量**：组合自检是否用同一 CPU 位点与 seed？活性量是否只在对应旗标为真时出现？
2. **实现**：缺省是否逐位还原 v1？快路径块类是否未变？变异是否抓红？
3. **规模**：冒烟未激活 carry（读出增益仍在单边零）是**步数**问题而非机制问题——记录实际
   `read_scale_absmean`，**不外推**；必要时加步数只作说明，不改「20 步」判据口径。
4. **设计**：只有前三条都排除后，才怀疑可组合块本身的接线（如窗口未进 SDPA、KV 头未广播）。

## 9. 产出（ENG1 前 / 后：哪些组合从未被测过、现在可测）

| 组合 | ENG1 前 | ENG1 后 | 现在怎么测 |
|---|---|---|---|
| `carry × window` | **可测**（旧护栏已允许） | 保持可测 | `CarryChannelBlock` + `attn_mask` |
| `GQA × window` | **硬互斥 ⇒ 不可测** | **可测** | `GQABlock` + `attn_mask`（正交装饰层） |
| `GQA × carry` | **硬互斥 + 块类二选一 ⇒ 不可测** | **可测** | `ComposableBlock`（KV 头投影从块类解耦） |
| `GQA × carry × window` | **不存在 ⇒ 从未被测** | **可测** | `ComposableBlock` + `attn_mask` |
| `attn_sparse_k / attn_linear` 与上述任一 | 仍互斥（本线未开放） | 仍互斥 | — |

> **不外推**：本表只说「可测」，**不说**组合的性能/质量——那需要新的训练实验。

- `results.json`：`combos`（4 组合 × 活性量）/ `default_bitwise_v1` / `fast_paths` /
  `smoke` / `regression` / `eng1_tests` / `mutation` / `code` / `cost`。
- `analysis.md`：人话摘要 + 逐格表格 + 判读。
- `MODEL.md`：冒烟权重不入库（体积/哈希/生成命令）；代码指纹。
- `controls/README.md`：对照臂与变异验证说明。

## scope 声明

> 本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——
> 组合留白是显式声明，不是遗漏。`deps` 字段只记录已知耦合方向。
> 组合取舍属具体产品方 scope。

> **ENG1 是对 scope 条款的工程例外说明**：本线不「跑组合实验」，而是**让组合可测**
> （把三特性的组装从互斥改成可组合）。它不产出任何组合的质量/性能结论。
