# ticket 63 · 轴 6 预测轴 lead（Phase A）

> 性质：**过程工单**（轴 6 的编排 + 执行 + 复核轨迹）。事实文档 = `axes/6-predict/**/project.yaml`
> 与 `axes/6-predict/README.md`、`axes/6-predict/6.3-parallel-generation/README.md`；本文件只记
> 「我们一步一步怎么走的」。用词以 `docs/structure.md` §12 为准（轴 6 的研究项 `6.2 / 6.3.1 / 6.3.2 /
> 6.3.3 / 6.4`；分组 `6.3`；`6.1 自回归` = 参照、不写 `project.yaml`）。
> 记录纪律：显式时间戳 + actor + 命令与结果 + 证据锚点；框架性取舍上报 Main（见 §4）。

## 1 范围与契约（轴 6 lead）

| 项 | 值 |
|---|---|
| 轴 | 6 预测轴（`axes/6-predict/`） |
| owner | axis6-lead（轴 6 预测轴 lead） |
| writer | 本轴单一写入者 = axis6-lead（文档/`project.yaml`）；WP 执行 writer = ticket 63 派发的 WP worker（每 WP 一名） |
| reviewer | axis6-reviewer（fresh 独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后） |
| integrator | tree-integrator（研究树 integrator；本轮本轴合并与跨轴一致性核对由 axis6-lead 代行） |
| 规范输入 | `docs/structure.md` §12（节点/用词/粒度）、`docs/research-schema.md`（字段契约）、`SCHEMA.md` §7/§8（成本）、§14/§14b（复核）、§20（GPU 锁） |
| 本轴节点 | 研究项 `6.2`（WP 6.2.1/6.2.2/6.2.3）、`6.3.1`、`6.3.2`、`6.3.3`、`6.4`（共 5 份 `project.yaml`）；分组 `6.3`；`6.1` 参照 |
| 前情输入 | `.scratch/55-review/H-PROBLEMS.md`（F1–F5、I1–I8）、`.scratch/56-restructure/MAPPING.md`（§1 H1→6.2、H2→6.4、H3→WP 6.2.2、H4→WP 6.2.1、H5→WP 6.2.3；§2 6.3.x 无旧格） |
| 边界 | 只动 `axes/6-predict/**` + `.scratch/63-axis6-predict/**`；不碰其它轴、`docs/**`、共享 `pathlm/**`/`train_*.py`/`tools/**`（需改则上报） |
| 并发/锁 | 并发子 Agent ≤3；GPU 工作按 `SCHEMA.md` §20 互斥锁**串行**；纯 CPU/文档不取锁 |
| 预算（Phase A） | 硬时间盒 ≤2h：①5 项 `role_map` 实名 + 逐项「想清楚」复核；②**只执行 1 个最高价值 WP = 6.2.2（概率多候选 / 树 + 置信加权）**；③fresh 独立复核（累计 ≤2 轮）；④兄弟字段一致性扫描 + `summary.py --check` 全绿；⑤更新 `status/conclusion/experiments/change_log` |

## 2 轴级共同口径（沿用已冻结项，不自选）

- 目标函数：**用更少的顺序前向次数**（等效理论算力）拿到**序列级**类似效果；代价口径**只**用等效理论量
  （FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**墙钟/吞吐不进判据**
  （`docs/research-schema.md` §3）。
- 对照 = 同尺寸 **stock 全因果逐 token 贪心**（`axes/7-references/README.md`，13M B0 快照 bpc = 1.5074；
  2610 冻结主干 = 1.4836）归一为 100%；投机解码另有诚实对手「1-step AR 两次前向」，oracle 只作上界附报
  不作判据（`SCHEMA.md` §11）。
- 判据 = 平均接受长度 τ / tokens-per-forward（该涨）+ 每 token 顺序前向次数（该跌）+ 序列级质量
  （该不跌，须越 0.014 噪声带才算可排序）+ 无损性（必须不变）；每 token 等效 FLOPs 是**代价轴**、如实报，
  `R=(1+γ)/(1+τ)>1` 是范式定义性质、**不得当判据**（55-review H-F03）。
- 「0.014 噪声带」（`docs/methodology.md:19` 的 13M 跨 run bpc 漂移界）**是否可当接受率（α/τ）阈值**
  = 框架性取舍，见 §4 F1（上报 Main，不自选；本轮 WP 6.2.2 暂按预登记写法用它，并在判读里同时给 CI）。

## 3 Phase A 逐项「想清楚」复核（5/5）

复核维度（对齐 `docs/research-schema.md` §2 与 `SCHEMA.md` §23 七问）：①痛点 ②业界 SOTA 与出处
③相对 SOTA 的增量 ④方法（单因素）⑤实现 ⑥**客观判据（该涨/该跌/必须不变）**⑦做不出来怎么归因
（设计/结构/innovation）⑧等效理论代价 ⑨单一 milestone ⑩WP 与推进顺序。

| 研究项 | 复核结论（actor=axis6-lead，2026-10-13T16:27+08:00） | 处置 |
|---|---|---|
| **6.2 多位置预测 = 投机解码** | 十项齐全；`gap` 自认「本质重造 Medusa」，增量落在「τ 与每 token FLOPs 分开量化 / 补回概率多候选 / 免费路线」三处，诚实。三处实质问题：**(a)** `objective_criteria` 的「必须不变：发出流**逐位** == 逐步贪心」与 `conclusion`(e) 的「@100M fp 平局翻转 1/400 token、逐位相等不可作跨规模判据」**自相矛盾**（同一事实两处口径不一致）；**(b)** WP `6.2.2` = 55-review F2/F5 点名的致命缺口（树 / 多候选整条没做、加权组合被简化成固定 λ=0.5）——**本轮执行**；**(c)** WP `6.2.2` 的 `evidence` 曾指向 `experiments/`（目录级、无锚点），须钉到具体新实验目录 | 修 (a)：判据加口径限定（13M 逐位相等硬门禁；更大规模按渐近一致 + 翻转率判）；(b) WP 6.2.2 `status: 未开始 → 进行中`、`evidence` 钉到 `experiments/2640-tree-conf-13m/`、acceptance 写死预登记判据；补 role_map 实名 |
| **6.3.1 diffusion（离散扩散 / 掩码扩散）** | 十项齐全；本仓零经验（MAPPING §2 无旧格），判据/否证条件已预登记；`cost_model` = K×L / (T/K，除 K) 解析式正确；外部基线条目均标 `记忆·未核` 且声明不作判据 ✓；「必须报每 token 等效 FLOPs」与「该涨 tokens-per-forward」分开写死 ✓ | 仅补 role_map 实名 |
| **6.3.2 一次性批量 pad 填充（填空式）** | 十项齐全；把「未填位置因果性（不得泄漏未来真值）」写成**必须不变**的硬门 ✓（防伪成功）；`cost_model` = L/r 正确；与 6.3.1 的边界（K=1 特例）以指针相连、不重复事实 ✓ | 仅补 role_map 实名 |
| **6.3.3 latent regression 批量生成（估计得贪心）** | 十项齐全；把「同参数量离散跳头对照 + 读出共享」写死（`SCHEMA.md` §15 等参数纪律）✓；本地对手指向 6.2 的 `2631-spec-three-arms`（指针，不复制数字）✓ | 仅补 role_map 实名 |
| **6.4 训练时未来位置的填充语义** | 十项齐全；`conclusion`/`conclusion_review`/`gaps` 三处自洽（「milestone 未达标 ⇒ 进行中」）✓；判据（clean bpc 税 / 填入位置可读性 / rescore OOD）与「必须报活性」齐备 ✓ | 仅补 role_map 实名 |

- 全 5 项 `cost_model` 复核 = 纯等效理论量、无墙钟/日期/吞吐 ✓。
- 全 5 项 `milestone` = 单数 map、含 `output` + `acceptance` ✓；`work_packages[].review` 非空 ✓。
- 兄弟字段一致性扫描（同一事实不重复 / 指针 / 区间同步）见 §6。

## 4 框架性取舍（上报 Main，不自选）

| # | 取舍 | 现状 | 待 Main 裁决 |
|---|---|---|---|
| F1 | **0.014 噪声带能否当接受率（α/τ）阈值** | 该带来源 = 13M 跨 run **bpc** 漂移界（`docs/methodology.md:19`）；轴 6 用它判 α₁ 可排序性属跨量纲借用。本轴本轮**先按 `project.yaml` 里已预登记的写法**给出 CI + 带内/带外双读数，不自选阈值 | 是否把「α/τ 的可排序门」单独定（例如按 prompt 聚类 bootstrap CI 不含 0 为主、0.014 仅作参考） |
| F2 | **树 / 多候选验证的代价口径**：树用 K 倍并行 FLOPs 换更高 τ，同 size 下 `R_tree` 通常仍 >1 | 本轴按「τ / tokens-per-forward 是硬指标、FLOPs 是代价轴」如实双报，**不**把 `R_tree` 与单臂 `R` 的对比写成硬判据 | 是否要求所有多候选 / 树结论一律附「等 FLOPs 对照臂」（本引擎 `n_mtp=3` 上限下等 FLOPs 臂不可得，见 §5 记录） |
| F3 | **引擎改动边界**：完整二级树（N+3 以被接受的 N+2 候选为条件）需要**新的二级条件头 + 重训**，属共享 `pathlm/**` 改动 | 本 WP 不擅改；把「需要什么 + 最小 patch 描述」记入 §5 并上报 | 是否授权（谁改、谁复核、是否本轮做） |

## 5 WP 排序、预算与执行

| 序 | WP | 价值 | 预算 | 状态（2026-10-13T16:27+08:00） |
|---|---|---|---|---|
| 1 | **6.2 / WP 6.2.2 概率多候选 / 树 + 置信加权组合** | 轴 6 旗舰：55-review F2/F5 两条**致命**缺口（出资人点名的设计被简化掉、树整条没测）+ 预登记 cross-K「树可能翻转」判据已触发 | ≤60 min GPU（含冒烟） | **进行中**（worker 已派，2026-10-13T16:26:27+08:00） |
| 2 | 6.2 / WP 6.2.3 验证（无损性不变量按规模分档） | 主判据的前置（无损性是硬门禁） | 未派 | 未开始（本轮不派，超时间盒） |
| 3 | 6.2 / WP 6.2.1 草稿头免费基础设施 | 已有 13M 读数（Δbpc 0.0000），n≥2 与 @100M jump head 待补 | 未派 | 进行中（沿用） |
| 4 | 6.4 / WP2 软提交 OOD 污染剥离、6.3.1/6.3.2/6.3.3 WP1 | 各自 milestone 未达 | 未派 | 未开始（后续 Phase B） |

- 排序理由：WP 6.2.2 是唯一「预登记判据已触发 + 致命缺口 + 可在冻结主干上免费跑」的项；其余项要么已
  有读数（6.2.1）、要么需要新训练预算（6.3.x 需重训主干 / 新头）。
- **引擎改动需求（不擅改，上报 Main）**：
  - **需要什么**：`pathlm/model.py::PathLM._mtp_nodes` 目前只有一级 token 条件链
    （`nodes[1]["chain2"] = transforms[0](chain_cat([h, embed(cond)]))`，预测 t_{i+2} 且条件只能取 t_{i+1}）。
    完整树要求「位置 N+3 以**被接受的候选 token** 为条件」，即需要二级条件头（预测 t_{i+3}、条件 = t_{i+2} 的任意候选）。
  - **最小 patch 描述**：新增 `transforms_chain2`（d×d）与 `chain_cat2`（2d×d），输入
    `chain_cat2(cat([h_{N+1}, embed(c)]))`（`h_{N+1}` = 验证前向已算出的 N+1 位置主干状态，属 EAGLE 式特征复用、不需额外主干前向），
    损失 = 对 t_{i+3} 的 CE + conf BCE（复用 `_node_loss`），并加进 `PathConfig` 开关（默认关）；配套
    `Decoder`/`decode_spec` 的树验证路径（B=K 行 + 树注意力）。**须重训头**（~1 GPU 小时量级）。
  - 该改动**本轮不做**：本 WP 的树只做「位置 N+2 的 K 候选 + 一次前向精确验证」，是**可判的最小树**。

## 6 命令、校验与一致性扫描记录

（本节由 axis6-lead 追加；每条含 `date -Iseconds` 时间戳、命令与结果）

- `2026-10-13T16:27:43+08:00` · axis6-lead · 逐项复核 5/5 完成（见 §3）；role_map 由「（待指派）/reviewer 池」
  占位改为实名（owner=axis6-lead，writer=每项 worker，reviewer=axis6-reviewer，integrator=tree-integrator）；
  6.2 `objective_criteria` 加无损性口径限定（消除与 `conclusion`(e) 的矛盾）；WP 6.2.2 `status: 未开始 → 进行中`、
  `evidence` 钉到 `experiments/2640-tree-conf-13m/`。
- 兄弟字段一致性扫描（待本轮 WP 结果落盘后补跑，见收口记录）。

## 7 时间线

- `2026-10-13T16:24:22+08:00` · axis6-lead · 接手轴 6；读 `docs/structure.md` §12/§13、`docs/research-schema.md`、
  5 份 `project.yaml`、`.scratch/55-review/H-PROBLEMS.md`、`.scratch/56-restructure/MAPPING.md`、`.scratch/57-program/TICKET.md`。
- `2026-10-13T16:26:27+08:00` · axis6-lead · 派 WP 6.2.2 worker（fresh，`axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/`）；
  复用 `2631-spec-three-arms` 的 checkpoint 与 prompt 口径；臂 = {native, direct1, chain1, combined_fixed, combo_pw, tree_union2, tree_topk4, oracle 上界}；
  预登记判据 P1–P5；禁墙钟；GPU 按 §20 取锁。
- `2026-10-13T16:27:43+08:00` · axis6-lead · Phase A 步骤 1 落盘（本文件 + 5 份 `project.yaml` role_map 实名）。
