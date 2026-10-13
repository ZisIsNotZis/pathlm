# ticket 67 · 受控引擎改动通道（engine-change channel）

> 性质：**程序级台账**（`.scratch/**`，允许日期/过程叙事），不是事实文档。
> 用途：登记「不扩建共享引擎（`pathlm/**`）就测不了**完整机制**」的最小 patch 需求，并在出资人授权后
> 用**单一 engine-writer + 独立复核**执行。
> 契约：`.scratch/57-program/FRAMEWORK-DECISIONS.md` C3（裁决）+ `docs/research-schema.md` §12.3 +
> `SCHEMA.md` §20（GPU 锁）/ §22（代码编辑锁、锚点声明）/ §14b（复核顺序）。

## 0. 人话摘要（先读这段）

三个研究轴（1 时间轴 / 3 宽度轴 / 6 预测轴）各自发现：**不扩建共享引擎，就测不了完整机制**——
测出来只能是被引擎能力削过的退化版（会把「机制不成立」误判成「实现不成立」）。按边界纪律，它们
**都没有擅自改** `pathlm/**`，只把「需要什么 + 为什么 + 最小 patch」报了上来。

出资人裁决 **C3：授权建立受控通道**（本轮**不改引擎代码**，只建通道与队列）。即：把三项最小 patch
登记在案，约定「**单一 engine-writer + 独立复核 + `SCHEMA.md` §20/§22 锁**」的执行纪律；具体
「谁改、谁复核、是否本轮做」待出资人在通道内逐项授权。

## 1. 通道纪律（执行前必读）

- **单一写入者**：同一实现空间（`pathlm/**`）同一时刻**只允许一个 engine-writer**；不得两个并行
  写入者（`SCHEMA.md` §22.1）。
- **先声明锚点再改**：只做**局部追加**、禁止整块重写；改完回读校验（`SCHEMA.md` §22.3/§22.4）。
- **独立复核排在执行之后**：`SCHEMA.md` §14 / §14b，复核者不得是本 patch 的 writer。
- **锁顺序**：先 `/tmp/pathlm_code.lock`（代码编辑锁）、再 `/tmp/pathlm_gpu.lock`（GPU 锁），
  防死锁（`SCHEMA.md` §22.2）；GPU 训练/评测按 §20 串行。
- **能力面分析**：执行前做 `SCHEMA.md` §22.8 能力面分析（flag 组合是否互斥、是否换 block 类），
  避免「开关开着、部分臂没接」的死干预（`SCHEMA.md` §10）。
- **本通道不代表授权**：`C3` 授权的是**建通道**；每一项 patch 的 `授权状态` 默认为**未授权**，
  须出资人逐项批准后才排期执行。

## 2. 队列（三项已上报的最小 patch 需求）

### E1 · 轴 1 · `CarryChannel` 逐层状态 + 内容相关写入门

| 项 | 内容 |
|---|---|
| 来源 | `.scratch/62-axis1-time/TICKET.md` §4 F5；55-review M-05/M-06 |
| 需要什么 | `pathlm/depth_ar.py::CarryChannel` 目前是**单共享槽**（`shared_slots=true`：一个状态覆盖全部 12 层、只从输入流 h0 写、无门控衰减）。要测完整机制需：① 逐层独立状态；② 写入门内容相关；③ 读出增益逐层可学（已存在）；④ 可选衰减 γS。 |
| 为什么 | 单共享槽是**被引擎/墙钟开销逼出的退化实现**（per-layer 12 状态实测 207–425 ms/step，超预算 2.4×）；业界 Gated DeltaNet / GLA 都是**逐层**。不修就无法区分「机制不成立」与「本实现不成立」。 |
| 最小 patch 描述 | 把 `CarryChannel` 的状态从 `[1, n_heads, hd, hd]` 扩到 `[n_layers, n_heads, hd, hd]`；`CarryChannelBlock` 各层持层号并索引自己的槽；`write_read` 增 `layer_idx` 参数；训练侧 `carry_scan_*` 按层循环（或一次性 batch）。**约 60–120 行，只动 `pathlm/depth_ar.py`**，不新增公开开关（先内部实验）。风险：墙钟会再超，需按 §20.3 缩步数。 |
| 授权状态 | **已执行**（`2026-10-14T07:17:34+08:00`，engine-writer，fresh 上下文；出资人「按默认处置推进」授权下由 Main 排期）。实际落点与最小 patch 描述的差异（显式实现决策）：**逐层状态由各层在本层当前 hidden 上自算**（不是把 h₀ 一次扫描按层索引——同一 h₀、同一套写投影下逐层槽会逐位相同，测不出「独立」）；`carry_per_layer` 与 `carry_decay` 两个 DepthARConfig 字段（缺省 `False`/`1.0`，未加 CLI，符合「不新增公开开关」）。见 `.scratch/67-engine-changes/E1-CARRYCHANNEL.md`（执行报告，含能力面/单测/复核待办）。 |

### E2 · 轴 3 · 逐层 FFN 宽开关（`mlp_mult` 全局标量 → 逐层 schedule）

| 项 | 内容 |
|---|---|
| 来源 | `.scratch/64-axis3-width/TICKET.md` §4 F2 / §5；55-review 宽度轴 |
| 需要什么 | `DepthARConfig` / `ModelConfig` 的 `mlp_mult` 由全局标量扩展为**逐层 schedule**（如 `mlp_mult_sched: list[int] | None`）。 |
| 为什么 | 宽度轴 `3.1.2`（逐层 d_model）与 `3.1.4`（逐层 FFN 隐藏比）都以「逐层宽度」为核心干预；引擎没有该杠杆时，各实验只能在**实验目录内**用「构造后切片」自造（本轮 `3.1.4` 的 2641 即如此），代价是每个实验都要自证实现等值、且不同实验的切片实现可能不一致。 |
| 最小 patch 描述 | `pathlm/depth_ar.py::DepthARModel.__init__` 构造 `self.blocks` 处逐层传入（`Block.__init__` 增可选 `mlp_mult: int | None = None`，缺省回落 `cfg.mlp_mult`）；`train_depth_ar.py` 加 `--mlp-mult-sched "4,3,2,2,2,1,1,1"`（缺省 None = 逐位还原现状）；配套单测 2 条（缺省与现状逐位同前向；schedule 真到达每层 `mlp[0].weight.shape[0]`）。**约 30–50 行，只动 `pathlm/depth_ar.py` + `train_depth_ar.py`**。风险：`Block` 有 4 个变体（`ParallelBlock`/`GQA`/`Composable`/`LinearAttn`）各自内联 `mlp_mult`，需一并接线。 |
| 授权状态 | **未授权**（C3 已建通道；待出资人逐项批准） |

### E3 · 轴 6 · 二级条件头 + 树注意力验证（需重训头）

| 项 | 内容 |
|---|---|
| 来源 | `.scratch/63-axis6-predict/TICKET.md` §5；WP 6.2.2 的未做项 |
| 需要什么 | `pathlm/model.py::PathLM._mtp_nodes` 目前只有一级 token 条件链（预测 t_{i+2}、条件只能取 t_{i+1}）。完整树要求「位置 N+3 以**被接受的候选 token** 为条件」，即需**新的二级条件头**（预测 t_{i+3}、条件 = t_{i+2} 的任意候选）。 |
| 为什么 | 完整二级树（N+3 以被接受的 N+2 候选为条件）是本 WP 预登记设计的一部分；当前只做了「位置 N+2 的 K 候选 + 一次前向精确验证」的可判最小树。 |
| 最小 patch 描述 | 新增 `transforms_chain2`（d×d）与 `chain_cat2`（2d×d），输入 `chain_cat2(cat([h_{N+1}, embed(c)]))`（`h_{N+1}` = 验证前向已算出的 N+1 位置主干状态，EAGLE 式特征复用、不需额外主干前向）；损失 = 对 t_{i+3} 的 CE + conf BCE（复用 `_node_loss`），加进 `PathConfig` 开关（默认关）；配套 `Decoder`/`decode_spec` 的树验证路径（B=K 行 + 树注意力）。**须重训头**（~1 GPU 小时量级）。 |
| 授权状态 | **未授权**（C3 已建通道；待出资人逐项批准） |

## 3. 与既有文档的关系

- 判据/字段纪律：`docs/research-schema.md` §12.3（C3 通道指针）。
- 各轴的原始上报与最小 patch 全文：`.scratch/62-axis1-time/TICKET.md` §4 F5、`.scratch/64-axis3-width/TICKET.md` §4 F2/§5、`.scratch/63-axis6-predict/TICKET.md` §5。
- 本通道**不实现任何引擎代码**；实现属「新的派发」，需出资人逐项授权与覆盖预算。

## 4. 变更记录

- `2026-10-14T06:44:03+08:00` · framework-adjudication-writer（ticket 57）· 按 C3 裁决建立受控通道，登记 E1/E2/E3 三项最小 patch 需求（未授权）；不改引擎代码。
- `2026-10-14T07:17:34+08:00` · engine-writer（fresh，`pathlm/**` 单一写入者，code 锁 `/tmp/pathlm_code.lock`）· **E1 已执行**：`pathlm/depth_ar.py` 追加 `carry_per_layer`（逐层独立状态，写入取各层当前 hidden）+ `carry_decay`（可选衰减 γ ∈ (0,1]）；`carry_scan*` 加 `decay` 参数（缺省 1.0 逐位不变）；`CarryChannel.write_read` 加 `layer_idx`/`decay`；`CarryChannelBlock`/`ComposableBlock` 逐层支路；`carry_liveness`/`carry_read_mass_stats` 口径对齐逐层模式。配套单测 9 条（含默认逐位不变、逐层独立、内容写门、γ 分块==逐位置 fp64 1e-12、活性/梯度、能力面），`tests/test_engine_capability.py` + `docs/engine_capability.md` 事实表补 C17–C22。`pytest tests/ -q` 849 passed。**独立复核待 Main 派 fresh 复核者**（本 worker 无派发权，见报告「复核待办」）。
