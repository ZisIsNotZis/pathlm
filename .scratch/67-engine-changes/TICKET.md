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
| 授权状态 | **已执行**（`2026-10-14T21:15:00+08:00`，engine-writer，`pathlm/**` 单一写入者，code 锁 `/tmp/pathlm_code.lock`；Main 在本 run 授权 E2+E3）。实际落点（与 E4 统一，见下）：字段 = `DepthARConfig.mlp_hidden_sched: list[int] | None`（**逐层 FFN 隐藏宽，任意正整数**；缺省 None = 各层回落 `cfg.mlp_mult·d_model`，逐位还原现状）；值 = `mult·d` 即逐层 `mlp_mult`，任意整数即 E4 所需等参数宽度；CLI = `--mlp-hidden-sched`（逗号分隔）；`Block` 基类 + 4 变体（`ParallelBlock`/`GQABlock`/`ComposableBlock`/`LinearAttnBlock`）与继承类（`TopKBlock`/`CarryChannelBlock`）全部接线。见 `.scratch/67-engine-changes/E2-E3-ENGINE-REPORT.md`。 |

### E3 · 轴 6 · 二级条件头 + 树注意力验证（需重训头）

| 项 | 内容 |
|---|---|
| 来源 | `.scratch/63-axis6-predict/TICKET.md` §5；WP 6.2.2 的未做项 |
| 需要什么 | `pathlm/model.py::PathLM._mtp_nodes` 目前只有一级 token 条件链（预测 t_{i+2}、条件只能取 t_{i+1}）。完整树要求「位置 N+3 以**被接受的候选 token** 为条件」，即需**新的二级条件头**（预测 t_{i+3}、条件 = t_{i+2} 的任意候选）。 |
| 为什么 | 完整二级树（N+3 以被接受的 N+2 候选为条件）是本 WP 预登记设计的一部分；当前只做了「位置 N+2 的 K 候选 + 一次前向精确验证」的可判最小树。 |
| 最小 patch 描述 | 新增 `transforms_chain2`（d×d）与 `chain_cat2`（2d×d），输入 `chain_cat2(cat([h_{N+1}, embed(c)]))`（`h_{N+1}` = 验证前向已算出的 N+1 位置主干状态，EAGLE 式特征复用、不需额外主干前向）；损失 = 对 t_{i+3} 的 CE + conf BCE（复用 `_node_loss`），加进 `PathConfig` 开关（默认关）；配套 `Decoder`/`decode_spec` 的树验证路径（B=K 行 + 树注意力）。**须重训头**（~1 GPU 小时量级）。 |
| 授权状态 | **已执行**（`2026-10-14T21:55:00+08:00`，engine-writer，`pathlm/**` 单一写入者，code 锁 `/tmp/pathlm_code.lock`；Main 授权）。落点：`PathConfig.chain_second: bool = False`（缺省关，逐位还原现状）；`PathLM` 追加 `transforms_chain2`（d×d）/`chain_cat2`（2d×d）/`conf_chain3`（建在参数 init 循环**之后** ⇒ 不扰动既有随机数流）；`_mtp_nodes` 的 node 1 追加 `chain3`（预测 t_{i+3}，条件 = t_{i+2} 候选，context = h 右移一位 = h_{N+1}，EAGLE 式复用）；`forward` 接 `_node_loss(targets[:, 3:])`；`Decoder.step_tree_multi`（B=K 行 + 树注意力，`_run_stack_multi(tree_parents=...)`）+ `decode_spec` 二级链走树路径。见 `.scratch/67-engine-changes/E2-E3-ENGINE-REPORT.md`。 |

### E4 · 轴 5 · 等参数对照的构造能力（逐层 FFN 隐藏宽 schedule，任意整数）

| 项 | 内容 |
|---|---|
| 来源 | `.scratch/66-axis5-quant/TICKET.md` §9.1 / §13 F4（Phase B 登记）；`axes/5-quant/5.1-kv-quant/project.yaml::WP3`、`axes/5-quant/5.4-low-rank/project.yaml::gaps` |
| 需要什么 | 一个能**逐位复原参数差**的构造杆：头共享族（GQA/MQA）比同形 MHA 少参数（实测 GQA −1 184 256 = −8.75%、MQA −1 381 632 = −10.21%），要把 Δparams 还给 FFN。GQA 需 +1 184 256 参数 = 每层 +98 688 = 每层 FFN 隐藏宽 +192.75（**非整数**）⇒ 需**逐层**隐藏宽 schedule（如 9 层 +193 / 3 层 +192 ⇒ Σ 精确相等），或等价的其他等参数补偿项。 |
| 为什么 | `SCHEMA.md` §15 要求增益归因过等参数对照；`5.1` 的 7/7 同字节组最优臂全带头族参数混杂（−8.75%/−10.21%，`2642` 已精确拆开），若没有等参数臂，「同字节纯机制」读数不可得，`5.1::WP3` 与 `5.4::WP1` 的对照臂都做不出来。 |
| 最小 patch 描述 | 把 FFN 隐藏宽由全局标量 `mlp_mult`（× d）扩展为可选**逐层隐藏宽 schedule** `mlp_hidden_sched: list[int] | None`（缺省 None = 逐位回落现状）；`train_depth_ar.py` 加 `--mlp-hidden-sched "1728,1729,..."`；单测 2 条（缺省与现状逐位同前向；schedule 真到达每层 `mlp[0].weight.shape[0]`）。**与 E2 同面**（E2 是 `mlp_mult` 的整数 schedule；本项要任意整数隐藏宽）——实现时宜一并做，只动 `pathlm/depth_ar.py` + `train_depth_ar.py`，`Block` 的 4 个变体（`ParallelBlock`/`GQA`/`Composable`/`LinearAttn`）各自内联 `mlp_mult` 需一并接线。风险：改 `mlp_mult` 面会动 E2 的接线 ⇒ 两项须同一 engine-writer 串行，不得两个写者并行。 |
| 授权状态 | **由 E2 统一承接**（`2026-10-14T21:15:00+08:00`，Main 协调 + engine-writer 执行）：E2 的 schedule 已做成「逐层 FFN 隐藏宽，接受任意正整数」，字段 `mlp_hidden_sched`、CLI `--mlp-hidden-sched`，同时满足本项（轴 5 等参数宽度）；不另开第二套开关、不另派写者。 |

> 备注（能力面，派发前必读）：**不改引擎就没有等参数臂** —— 现有 flag 只有 `--mlp-mult`（整数倍 d=256）、`--heads`、`--n-kv-heads`、`--d`、`--layers`，无法把 Δparams 精确还给 FFN。轴 5 本轮把该缺口如实登记为受阻项，**未擅改引擎**。

## 3. 与既有文档的关系

- 判据/字段纪律：`docs/research-schema.md` §12.3（C3 通道指针）。
- 各轴的原始上报与最小 patch 全文：`.scratch/62-axis1-time/TICKET.md` §4 F5、`.scratch/64-axis3-width/TICKET.md` §4 F2/§5、`.scratch/63-axis6-predict/TICKET.md` §5。
- 本通道**不实现任何引擎代码**；实现属「新的派发」，需出资人逐项授权与覆盖预算。

## 4. 变更记录

- `2026-10-14T06:44:03+08:00` · framework-adjudication-writer（ticket 57）· 按 C3 裁决建立受控通道，登记 E1/E2/E3 三项最小 patch 需求（未授权）；不改引擎代码。
- `2026-10-14T07:17:34+08:00` · engine-writer（fresh，`pathlm/**` 单一写入者，code 锁 `/tmp/pathlm_code.lock`）· **E1 已执行**：`pathlm/depth_ar.py` 追加 `carry_per_layer`（逐层独立状态，写入取各层当前 hidden）+ `carry_decay`（可选衰减 γ ∈ (0,1]）；`carry_scan*` 加 `decay` 参数（缺省 1.0 逐位不变）；`CarryChannel.write_read` 加 `layer_idx`/`decay`；`CarryChannelBlock`/`ComposableBlock` 逐层支路；`carry_liveness`/`carry_read_mass_stats` 口径对齐逐层模式。配套单测 9 条（含默认逐位不变、逐层独立、内容写门、γ 分块==逐位置 fp64 1e-12、活性/梯度、能力面），`tests/test_engine_capability.py` + `docs/engine_capability.md` 事实表补 C17–C22。`pytest tests/ -q` 849 passed。**独立复核待 Main 派 fresh 复核者**（本 worker 无派发权，见报告「复核待办」）。
- `2026-10-14T20:56:00+08:00` · axis5-lead（轴 5 量化轴 lead，ticket 66 Phase B）· 登记 **E4**（等参数对照的构造能力：逐层 FFN 隐藏宽 schedule / 任意整数），来源 `5.1::WP3` + `5.4::gaps`（等参数头共享臂）。**未授权**（C3 已建通道）；本轴未改引擎代码，5.1-WP3 如实登记为受阻项。
- `2026-10-14T21:15:00+08:00` · engine-writer（fresh，`pathlm/**` 单一写入者，code 锁 `/tmp/pathlm_code.lock`；纯 CPU，未取 GPU 锁）· **E2 已执行（与 E4 统一为一个实现）**：`pathlm/depth_ar.py` 追加 `DepthARConfig.mlp_hidden_sched: list[int] | None`（逐层 FFN 隐藏宽，任意正整数；缺省 None = 各层回落 `cfg.mlp_mult·d`，逐位不变）+ 建时校验（长度 == n_layers、每项 >= 1）；`pathlm/model.py::Block` 与 `pathlm/depth_ar.py` 的 `ParallelBlock`/`GQABlock`/`ComposableBlock`/`LinearAttnBlock`/`TopKBlock`/`CarryChannelBlock` 均加 `mlp_hidden` 可选参数（缺省 None 回落现状）；`train_depth_ar.py` 加 `--mlp-hidden-sched`（逗号分隔）+ `_parse_int_sched` 接入 `build_depthar_config`。配套单测 4 条（缺省逐位同参数/前向、schedule 真到达每层 `mlp[0].weight.shape[0]`、7 个块类全部接线、校验 + CLI 接通）+ 能力面事实表补 C23–C25。**E4 标记为由 E2 统一承接**。提交 `4638882`（7 文件）；全量 `pytest tests/ -q` = **855 passed, 1 failed**（既有失败：`axes/6-predict/6.3.3` 的 mutate 脚本未接 harness，与 E2 无关）。**独立复核待 Main 派 fresh 复核者**。
- `2026-10-14T21:55:00+08:00` · engine-writer（同上一行同一 writer，同 code 锁）· **E3 已执行**：`pathlm/config.py` 加 `PathConfig.chain_second: bool = False`（默认关）；`pathlm/model.py::PathLM` 加二级条件头 `transforms_chain2`（d×d）/`chain_cat2`（2d×d）/`conf_chain3` + 建时校验（需 `chain_mode="token"` 且 `n_mtp>=2`）；`_mtp_nodes` 的 node 1 追加 `chain3` 节点（预测 t_{i+3}，条件 = t_{i+2} 候选，context = h 右移一位 = h_{N+1}）；`forward` 接 CE+conf BCE（`targets[:, 3:]`）；`pathlm/decode.py` 加 `Decoder.step_tree_multi`（B=K 行 + 树注意力，`_run_stack_multi(tree_parents=...)`）+ `decode_spec` 二级链走树路径。配套单测 7 条（默认关逐位不变、只新增 3 个头且不扰动其余参数、二级条件对齐 t_{i+2}（非 t_{i+1}）、链父指针==因果、树注意力不看兄弟、死组合报错、decode_spec 树路径==逐 token 贪心）+ 能力面事实表补 C26–C28。另：新建 PathConfig 字段使 2620/2622/2626 三份归档阶梯护栏多出一个未钉住键 ⇒ 按既有 `w_repr_decorr` 先例在护栏里显式豁免（缺省关、不改臂行为，未动 `archive/**`）。**E3 只交付引擎能力；训练/重训头由轴 6 负责**。提交 `fa6a1ad`（11 文件）；全量 `pytest tests/ -q` = **865 passed, 1 failed**（同一既有失败）。**独立复核待 Main 派 fresh 复核者**。
- `2026-10-14T22:16:00+08:00` · engine-writer（收尾，fresh，`pathlm/**` 单一写入者，code 锁 `/tmp/pathlm_code.lock`；纯 CPU，未取 GPU 锁）· **E2/E3 复核 non-blocker 收尾（N1/N2/N3）**（复核 `REVIEW-E2E3.md` 判 E2/E3 = PASS、blocker = 0）：
  - **N2（死代码 + 逐行一致）**：`tests/test_engine_capability.py` 删除 E3 后不再使用的常量 `SRC_REL`/`SRC`（`grep` 全仓零引用 = 真实死代码）；把 E2 顺带并到同一行的能力面事实条目 `C17`/`C18` 恢复为**逐条一行**（`docs/engine_capability.md` 的 C17/C18 本就逐行；`os` 仍被 `DOC`/`os.path.join` 使用故保留）。功能无变化；`pytest tests/test_engine_capability.py -q` = 30 passed。
  - **N1（计数口径）**：`E2-E3-ENGINE-REPORT.md` §4 更正——pytest 汇总的 `xfailed` **不计入** `passed`；同一命令随本地 `runs/` 证据有无而异：带证据的开发工作树 = `865 passed, 1 failed`，干净检出 = `863 passed, 2 xfailed, 1 failed`（差额 = 2 条证据锚测试 `test_longT_clean`/`test_longT_stability`）。未改任何测量数字。
  - **N3（只读扫描，未越界改 axes）**：新增 `.scratch/67-engine-changes/STALE-MUTATION-ANCHORS.md`（脚本 `scan_stale_anchors.py`）。扫描全部 46 个 `axes/**/mutate.py`（66 条 `old=` 锚点）对 E2/E3 触碰的 4 个引擎文件（`depth_ar.py`/`model.py`/`decode.py`/`config.py`）：**E2/E3 判定态（HEAD `8e80b7c`）1 条失配** = `axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py` 的 `k-ignored`（对应引擎行 `pathlm/depth_ar.py:2201-2202` 的 `_mh(i)`/`for i in range`）；**当前工作树态 0 条失配**——该锚点已由轴 1 修复（commit `676c858`，已提交；该提交另修一处与 E2/E3 无关的早前失配 `tools/evalkit/retention.py`，不在本扫描的引擎文件范围）。**未改任何 `axes/**`**。
  - **本次全量 `pytest tests/ -q`（当前工作树）**：`1 failed, 865 passed`（失败 = `test_all_repo_mutate_scripts_use_the_harness`，当前工作树列出 3 个未接 harness 脚本 `{2660,2670,2671}`，其中 `2660` 为既有跟踪文件、`2670`/`2671` 为并发轴 6 未跟踪脚本，均与本引擎 diff 无关）；`git diff --check` 无空白告警。
  - **复核状态**：E2/E3 本体已独立复核 PASS。本次收尾只改测试格式/删死代码 + `.scratch/**` 记录，**未触碰引擎行为** ⇒ 引擎面复核不因本收尾失效；**是否再派 fresh 复核由 Main 定**（本收尾无可执行引擎改动，属非代码改动）。
  - **收尾提交**：`b7f1a34`（主收尾，5 文件）；紧随 `bc4c684` 补记哈希；轴 1 的锚点修复 = `676c858`。
- `2026-10-14T22:18:00+08:00` · engine-writer（收尾，同 code 锁；纯 CPU）· **通道闭环记录**：N1/N2 属**非行为改动**（测试格式 / 删死代码 / 报告计数口径），**不触发引擎面复核重跑**；最终 M3 集成复核会再对齐。E2/E3 独立复核件 `REVIEW-E2E3.md` 入库（提交 `0f77564`）。**E2/E3 受控引擎通道到此关闭。**
