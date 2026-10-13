# 2628 · ENG2 引擎：`carry` 支持非 dense ragged 路径 · 代码 + 单测 + 冒烟

> 契约来源：`.scratch/44-wave17/issues/01-wave17.md` 的「后续项（已立，不占本波）：
> `ENG2 · 让 carry 支持非 dense ragged 路径`」与任务下发文本（本波 E1 线）。
> **副本标注**：§4 判据线**逐字**抄录任务「验收（硬）」段；§1–§3、§5–§9 为**摘要 +
> 执行记录**（并注明）。判据线一律未改。
> 入口：`run.py`；单测：`tests/test_depth_ar.py` §ENG2 段；变异：`mutate.py`；
> 对账：`reconcile_numbers.py`；原始结果：`results.json`；冒烟：`smoke/master/`；
> 权重清单：`MODEL.md`；判读：`analysis.md`。

## 1. 命题

**ENG2 前，`carry`（显式写入通道）硬要求 `dense=True`**（`pathlm/depth_ar.py` 的
配置校验），且 `CarryChannelBlock.forward` 显式拒绝 `kv_from` / `gate_vec`（非 dense
的混合深度调用约定）⇒ **「carry × 随机退出 × proj-fill」这个组合在引擎里根本不存在**，
它**从未被测过**。本线把它变成**可测**：解耦 dense 硬校验、把带通道块的调用约定对齐
stock `Block`、把非 dense 路径的块遥测收进 `aux`，缺省路径逐位还原 v1。

本线**不追求性能**，只追求**可测性**（SCHEMA §23 第 6 问的工程面）。

## 2. 单因素声明

唯一被改动的引擎结构 = **写入通道在非 dense 路径上的可组合性**，**不改任何单特性
语义**：

- 缺省（`carry_rule=None`）与 stock `Block` 路径**逐位一致**（无通道模块、无槽、
  无遥测；单测 + `run.py::default_untouched` 钉住）；
- 单特性路径的块类**不变**：carry-only（dense 或非 dense）→ `CarryChannelBlock`、
  GQA×carry → `ComposableBlock`、缺省 → `Block`；
- dense 路径的 carry 前向**未变**（`CarryChannelBlock` 新增的 `kv_from`/`gate_vec`
  只在非 dense 调用时非 None）。

**必要性例外（显式记账）**：为了让组合真正可用，本线顺带修了两处**只有一个解释的**
接线，均不改任何单特性数值语义：

1. **proj-fill adapter 构造不再消耗随机数**（构造 `nn.Linear` 会抽签，故先存回 CPU
   RNG 状态）。原注释已声称「不消耗随机数」，但 carry 写门 init 在它之后 ⇒ 两者同开
   时 carry 权重被错位（同 seed 下 `carry×fill` 与 `carry×proj-fill` 的主干不再逐位
   一致，破坏 SCHEMA §9 的单因素对照）。修后两者同 seed 逐位一致（单测
   `test_eng2_carry_x_proj_fill_identity_starts_equal_to_fill` 钉住）。
2. **非 dense 前向把块遥测收进 `aux["attn_carry"]`**（原来只在 dense 分支收）——
   否则写入通道在退出/填充路径上「只有权重、没有活性量」，SCHEMA §10 的干预活性
   判据断档，20 步冒烟也无从证明「通道活着」。

**明确的 scope 边界（不静默）**：增量解码路径（`DepthARDecoder`，只用于非 dense
变深度）**未接入**写入通道读出 ⇒ 显式 `raise`（同 GQA 的处置），不交付「看起来能
用、实际少了一路」的解码路径。解码侧 carry 是**后续项**（见 §9）。

## 3. 指标表（逐项分析）

| 指标 | 解决什么问题 | 代价 | 有效性（vs 对照） | 收益与用法 | 可控性 | 副产品 |
|---|---|---|---|---|---|---|
| 组合可建（块类 + 调用约定） | 「组合存在」这一事实 | 无（CPU 组装） | vs ENG2 前的组合（原本报错） | 后续「组合实验」的入场券 | flag 组合全可控 | — |
| `aux["attn_carry"]` 层数 | 通道在非 dense 路径上真进了前向遥测 | 0（复用块遥测） | ENG2 前恒空 | 训练日志每 200 步落活性量的前提 | — | — |
| `carry_read_share` | 记忆读出占注意力输出量级的比例 | 一次前向 | γ 置零 ⇒ 恒等；激活后 > 0 | 「通道被读而不是摆设」 | `carry_scale` | — |
| `carry_changes_logits` | 通道真的改前向 | 两次前向 | γ=0 vs γ>0 逐位不同 | 运行侧活性证据 | `carry_scale` | — |
| `frozen_invariant_ok` | 退出即冻结在非 dense 路径成立 | 一次前向 | d_i<k 的状态 == 其退出深度状态 | 退出语义护栏 | `exit_depths` | — |
| `exit_changes_logits_vs_full` | 退出模式真的生效（非退化点） | 两次前向 | 混合退出 vs 全深度逐位不同 | 防「死开关」（dense 上 `exit_depths` 被忽略，见 ticket 44-X2 记录） | `exit_depths` | — |
| `proj_fill_adapter_max_abs_logit_delta` | 混合填充 KV 源（含 adapter）被真的消费 | 两次前向 | 换方向扰动 adapter ⇒ 必非零 | 防「kv_from 被忽略」 | `proj_fill` | — |
| `slot_identical_*` | 写入时机与退出模式无关 | 三次前向 | 三种退出模式槽内容逐位相同 | 写入时机语义钉 | — | — |

## 4. 判据线（**逐字**抄录任务「验收（硬）」段，未改一字）

> **验收（硬）**：① 新增组合可失败单测：`carry×ragged`、`carry×随机退出`、`carry×随机退出×fill`、`carry×proj-fill`（各至少一条）+ 变异验证抓红且逐字节还原；② 既有 522 测试全绿；③ 20 步冒烟证明「carry × 随机退出 × proj-fill」四者共存可训练（活性量落盘）；④ 报告写明「ENG2 前/后哪些组合从未被测过、现在可测」+ 写入时机的语义选择与理由。

**执行时的操作性定义（不改判据，只把阈值落定）**：

- **(1) 的「组合可失败单测」**：`tests/test_depth_ar.py` §ENG2 段，四条组合各一条
  （`test_eng2_carry_x_ragged_layer_matches_manual_rule` /
  `test_eng2_carry_x_random_exit_composes` /
  `test_eng2_carry_x_random_exit_x_fill_consumes_frozen_kv` /
  `test_eng2_carry_x_proj_fill_identity_starts_equal_to_fill`），另加写入时机、缺省
  不动、解码拒绝三条；「可失败」= 变异验证（`mutate.py`，见 `controls/`）五处破坏
  全部抓红并逐字节还原。
- **(2) 的「既有 522 测试全绿」**：**如实记账**——本仓库当前 `pytest tests/` 收集
  基线是 **549**（522 是下发时的快照；期间 44 批与在途线又加了用例）。本线新增 7 条
  （§ENG2 段）。在**本线提交修订**上实测：
  - `pytest tests/ -q`（完整收集）⇒ **549 passed, 0 failed**（`pytest_full.txt`）；
  - `pytest tests/test_depth_ar.py -q` ⇒ **134 passed, 0 failed**。

  **口径注**：这个「全量」含**其它在途线**的未跟踪测试文件（它们不在本线提交里，
  数字随其变动）。本线提交前的两次跑中，曾因为 `tests/test_jitter_l1.py` 与
  `tests/test_mechanism_jitter.py` 都写 `from jitter import …`（各自把**不同实验
  目录**的 `jitter.py` 插进 `sys.path`，先导入者赢）而出现 1 个 fail；那是**跨线
  模块名撞车**，与该线自身的提交一起消失了，**与本线改动无关**（本线不碰那两个
  文件，也不新增同名模块）。
- **(3) 的「20 步冒烟」**：`run.py --smoke` 在 M1 引擎上跑
  `--carry-write delta --proj-fill --exit-mode uniform`（四者齐开，ENG2 前不存在的
  组合）× 20 步；「活性量落盘」= `train_log.jsonl` 的 `attn_carry` 块（step 0 与
  step 19 各一行，`n_layer_samples` 12 → 240）+ `results.json::carry_liveness`。
  实测：读出增益已逃离单边零（`read_scale_absmean` 0 → **0.000225**）、读出量非零
  （`read_share` 0 → **1e-06**）、写入真的改状态（`write_rel` 0.473→0.462）、β≈0.5、
  峰值显存 861.3MB、墙钟 0.05 min。**注意口径**：20 步太短，`carry_liveness.alive`
  仍为 **false**（`rel_dev_vs_channel_off` 四舍五入到 0.0）——这与 ENG1 的 20 步冒烟
  同口径，**只证明共存可训练，不外推**通道质量。
- **(4) 的「ENG2 前/后」**：见 §9 与 `analysis.md` 的人话摘要表。

## 5. 对照与方差

- **对照臂**（每个写明：输入分布 / 要预测什么 / 部署对应）：
  1. `default`（`carry_rule=None`，非 dense）：stock `Block`。输入 = CPU 随机 token
     `[2,12]`，vocab 10；要预测 = 逐深 next-token（本实验只用前向一致性）；部署
     对应 = 「什么都不开」的基底；ENG2 不许改它。
  2. `full-depth`（`exit_depths` 全 = L）：同一非 dense 模型、同权重。输入分布 = 同
     组合；要预测 = 同目标；部署对应 = 「退出开关关掉」的对手——混合退出臂与它之比
     只含退出/填充语义。
  3. `γ=0`（读出增益置零）：同一权重。部署对应 = 「通道读出关掉」的对手——通道臂与
     它之比只含记忆读出。
  4. `adapter=identity` vs `adapter 换方向`：同权重，只改 proj-fill adapter。部署
     对应 = 「填充 KV 源 remap 关掉」的对手。
- **方差**：本实验是**确定性**的（同 seed 组装与 CPU 前向逐位一致），判据是布尔量
  （通道是否改前向 / 退出是否生效 / 槽是否与退出模式无关），不涉及 bpc 方差界
  （Δ<0.014）。**无**多 seed——本线不比较质量，只验证可测性。
- **噪声界**：不适用（无训练质量读数）。冒烟 20 步的 loss 只作「可训练」证据，无统计
  含义；`synth_answer_ce` 只作「任务在学」参考。

## 6. 成本

```
gpu_wall_budget: 20 min   # 契约上限（≈20 min）；含冒烟 + 全量 pytest（CPU）
```

- 冒烟：M1 引擎（d=256/12L/8H **非 dense**；形态件 carry 写门 + proj-fill adapter 共
  入参 ⇒ 实测 **14.59M** 参数，不是 13.08M）× 20 步 × bs8 × seq256 = 0.041M token；
  实测峰值显存 **861.3MB**（≪5GB net）、墙钟 **0.05 min**（见 `results.json::cost`）；
  **先测步时**再定预算（SCHEMA §17.8），**不外推**训练时间。
- 组合自检 = CPU 前向（秒级）；全量 `pytest` = CPU（不占 GPU）。
- **预算不够时砍什么**（SCHEMA §17.7）：砍组合自检的重复档与冒烟批大小；**不砍判据线**
  （四条组合单测、全量回归、20 步冒烟、ENG2 前后清单都必须出）。
- 契约 §8：本文件只给一个 GPU 预算数字（20 min）。

## 7. 数据与脚本清单

- **数据**：冒烟用合成多查询关联回忆（`tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`），
  **不用 enwik8**（SCHEMA §19：长程类用局部先验抗性任务）。
- **入口脚本**：`run.py`（`--smoke` 跑冒烟；缺省摄取产物）。
- **resolved config**：`config.json`。
- **落盘文件**：`results.json`、`smoke/master/{train_log.jsonl,gate_log.jsonl,results.json,model.pt}`、
  `smoke/run_stdout.txt`、`pytest_eng2.txt`、`pytest_full.txt`、`mutation_report.json`、
  `reconcile_report.json`、`analysis.md`、`MODEL.md`、`controls/README.md`。

## 8. 失败决策树（无效时的处置路径：测量 → 实现 → 规模 → 设计）

1. **测量**：组合自检是否用同一非 dense CPU 位点与 seed？活性量是否只在对应旗标为真
   时出现？退出模式是否非退化（vs 全深度逐位不同）？
2. **实现**：缺省是否不含任何通道结构？dense 路径的 carry 是否未变？变异是否全抓红？
3. **规模**：冒烟 20 步通道读出量尚小（写出在逃逸初期）是**步数**问题而非机制问题——
   记录实际 `read_scale_absmean`/`read_share`，**不外推**；不改「20 步」判据口径。
4. **设计**：只有前三条都排除后，才怀疑写入时机/填充语义的接线（例如写入被错误地
   绑定到退出模式、或 kv_from 未进 SDPA）。

## 9. 产出（ENG2 前 / 后：哪些组合从未被测过、现在可测）

| 组合 | ENG2 前 | ENG2 后 | 现在怎么测 |
|---|---|---|---|
| `carry × 非 dense 随机退出` | **硬互斥（dense=False 直接报错）⇒ 不可测** | **可测** | `CarryChannelBlock` + `gate_vec` + `kv_from`/`attn_mask` |
| `carry × ragged（no-fill）` | **块拒绝 `attn_mask`+`gate_vec` ⇒ 不可测** | **可测** | 同上，走 `_ragged_mask` |
| `carry × 随机退出 × fill` | **不可测**（同上） | **可测** | `kv_from = where(active, h, frozen)` |
| `carry × 随机退出 × proj-fill`（100M 出口资产配方的那一半） | **从未被测** | **可测** | `kv_from = where(active, h, adapter(frozen))` |
| `carry × 增量解码（KV cache）` | 不适用（carry 只在 dense） | **仍不可用**（显式报错，未接入读出） | — 后续项 |
| `carry × GQA × 非 dense` | 不可测（GQA 仍 `dense=True`） | **仍不可测**（本线只解耦 carry） | — 后续项 |

**写入时机的语义选择与理由**（本引擎的显式决定，单测
`test_eng2_carry_write_timing_ignores_exit_depths` 钉住）：

> 记忆槽在 `DepthARModel._input_states` 里对**输入流 h₀** 因果扫描一次，发生在**任何
> 层与任何退出之前** ⇒ **所有位置（含已退出位置）都写各自 h₀（depth-0 表征）**，
> 不按退出深度跳过、也不改写成冻结表征。

- **理由①（可缓存/可部署）**：槽内容与退出模式**无关** ⇒ 同一个 token 在任何退出
  模式下得到同一段记忆，可在 prefill 一次算好，decode 时不必因退出模式重扫。
- **理由②（语义一致）**：读取的生效范围由调用方的 `where(active, out, frozen)` 决定
  ——位置 i 读到的记忆只影响 `depth ≤ d_i` 的层（更深层的块输出被冻结表征替换），
  所以「已退出位置也写」**不会**让已退出位置继续参与更深层的自注意力。
- **理由③（与 fill 对齐）**：fill 让已退出位置的**冻结表征**进 KV；通道则让它们的
  **h₀** 进记忆。两者都是「已退出位置也贡献」，语义自洽；而 `proj_fill` 的 adapter 是
  **注意力 KV 源**的 remap，**不**经过通道（通道有自己的 w_k/w_v/w_q/β）。
- **取舍（另一条路及其代价）**：若改成「按退出深度写冻结表征」，则槽必须逐层重算、
  且内容依赖退出模式 ⇒ 既不能 prefill 一次算好，也会让同一序列在两种退出模式下得到
  不同记忆（破坏 §9 的单因素对照）。本线选前者，并把取舍写在这里。

- `results.json`：`combos`（4 组合 × 活性量）/ `write_timing` / `default_untouched` /
  `smoke` / `regression` / `eng2_tests` / `mutation` / `code` / `cost`。
- `analysis.md`：人话摘要 + 逐格表格 + 判读。
- `MODEL.md`：冒烟权重不入库（体积/哈希/生成命令）；代码指纹。
- `controls/README.md`：对照臂与变异验证说明。

## scope 声明

> 本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——
> 组合留白是显式声明，不是遗漏。`deps` 字段只记录已知耦合方向。
> 组合取舍属具体产品方 scope。

> **ENG2 是对 scope 条款的工程例外说明**：本线不「跑组合实验」，而是**让组合可测**
> （把写入通道从「dense 硬校验 + 块拒绝混合深度调用」改成「可组合且写入时机有定义」）。
> 它**不产出**任何组合的质量/性能结论。
