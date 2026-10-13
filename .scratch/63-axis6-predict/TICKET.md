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
| F1 | **0.014 噪声带能否当接受率（α/τ）阈值** | 该带来源 = 13M 跨 run **bpc** 漂移界（`docs/methodology.md:19`）；同一文件 `:25-28` **明文**写「0.014 是 **bpc** 口径的方差带，**不得**直接当准确率的 1.4pp 界……准确率的可排序性应用**二项 / Wilson 95% CI**」⇒ 轴 6 用它判 α₁ 可排序性属**跨量纲借用**。本轴本轮**先按 `project.yaml` 里已预登记的写法**给出 CI + 带内/带外双读数（CI 为主、0.014 为参考），不自选阈值 | 是否把「α/τ 的可排序门」单独定（建议：按 prompt 聚类 bootstrap / Wilson CI 不含 0 为主判，0.014 仅作参考线）；本轴已如此并报 |
| F2 | **树 / 多候选验证的代价口径**：树用 K 倍并行 FLOPs 换更高 τ，同 size 下 `R_tree` 通常仍 >1 | 本轴按「τ / tokens-per-forward 是硬指标、FLOPs 是代价轴」如实双报，**不**把 `R_tree` 与单臂 `R` 的对比写成硬判据 | 是否要求所有多候选 / 树结论一律附「等 FLOPs 对照臂」（本引擎 `n_mtp=3` 上限下等 FLOPs 臂不可得，见 §5 记录） |
| F3 | **引擎改动边界**：完整二级树（N+3 以被接受的 N+2 候选为条件）需要**新的二级条件头 + 重训**，属共享 `pathlm/**` 改动 | 本 WP 不擅改；把「需要什么 + 最小 patch 描述」记入 §5 并上报 | 是否授权（谁改、谁复核、是否本轮做） |
| F4 | **6.3.3 latent regression 的判据归因**：`milestone`/`validation` 同时要「tokens-per-forward 与 τ（该涨）」+「草稿接受率」，但 `cost_model` 写的是「一次前向回归 k 个 latent 再各自贪心」——即**不经验证直接发**（估计得贪心） | 两条路是**不同的东西**：(i) 不验证直接发（无「接受率」概念，判据只能是质量 + 每 token 主干 FLOPs ≈ L/k）；(ii) 当**草稿**再验证（有接受率，但每 token 主干 FLOPs 变成 ≈ (1+k)/k×L，即 ≈L）。本轴不自选口径，6.3.3 本轮**不执行**；已记入本表 | 6.3.3 的判据与 `cost_model` 选哪条（或两条各作一臂并分开报代价）；确认后由 axis6-lead 改 `project.yaml` |

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
- `2026-10-13T16:31:28+08:00` · axis6-lead · 修 6.2 两处证据保真（`tokens-per-forward` 区间归因、`1/400 token` 自造摘要数）→ 见下方扫描第 4 条；`python3 tools/summary.py --check` 复跑全绿。
- `2026-10-13T16:31:28+08:00` · axis6-lead · 兄弟字段一致性扫描（第一遍，逐条可核）：
  1. `1.5074`（13M B0 快照 bpc）的唯一家 = `axes/7-references/README.md:13` ✓；6.2/6.3.1 只是指针式引用，无二家。
  2. `dir` 字段 vs 实际目录：5/5 一致 ✓；`experiments` 列出的 10 个 6.2 目录 vs 磁盘 10/10 一致 ✓；6.4 = 1/1 ✓。
  3. 6.2 experiments/README.md 表 10 行 vs 磁盘 10 目录 ✓（新实验 `2640` 待落盘后补行，见收口记录）。
  4. **6.2 `conclusion` 两处证据保真问题已修**（commit 见 §7 时间线）：(i) `tokens-per-forward 2.37–2.40（13M 与 100M）` 把 2520 的 γ=2 四点压成一个区间、并漏掉 13M random 2.6846 与 2631 部署三臂 2.07–2.58 ⇒ 改为**逐点列源**（2520：13M enwik8 2.3952 / 100M random 2.3669 / 100M enwik8 2.3952；2631 另为 2.07–2.58）；`expected_effect` 同步。(ii) `@100M fp 平局翻转（1/400 token）` 是**自造摘要数**，源件（2520 analysis §fp 平局）写的是「**1 次**翻转、首错位 169、级联 68 token、翻转处 top-2 gap 0.0079 > 预登记 1e-3 并列容差」⇒ 改为逐字对齐源件。
  5. 6.4 `conclusion` 引「>0.014 可排序」，而其源件 `2523e-fill-variants/README.md:24` 用「±0.03 平局带」：二者不冲突（0.014 = `docs/methodology.md:19` 的 SSOT 口径；±0.03 是源件自定的旧平局带），且 +0.0856 同时越两者 ⇒ 记 **info**，不改。
  6. `6.2` 的 16 条 `experiments` 中含 10 个仓内目录 + 6 个 `.scratch/**@sha` 锚点；锚点可达性由 `summary.py --check` 校验 ✓（全绿）。

## 7 时间线

- `2026-10-13T16:24:22+08:00` · axis6-lead · 接手轴 6；读 `docs/structure.md` §12/§13、`docs/research-schema.md`、
  5 份 `project.yaml`、`.scratch/55-review/H-PROBLEMS.md`、`.scratch/56-restructure/MAPPING.md`、`.scratch/57-program/TICKET.md`。
- `2026-10-13T16:26:27+08:00` · axis6-lead · 派 WP 6.2.2 worker（fresh，`axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/`）；
  复用 `2631-spec-three-arms` 的 checkpoint 与 prompt 口径；臂 = {native, direct1, chain1, combined_fixed, combo_pw, tree_union2, tree_topk4, oracle 上界}；
  预登记判据 P1–P5；禁墙钟；GPU 按 §20 取锁。
- `2026-10-13T16:27:43+08:00` · axis6-lead · Phase A 步骤 1 落盘（本文件 + 5 份 `project.yaml` role_map 实名）。
- `2026-10-13T17:23:00+08:00` · axis6-lead · WP 6.2.2 前任 worker 在 **16:56 被运行时 30 min 上限**杀死（`Subagent timed out after 1800000ms`），run #2 未产出；产物齐备度审计 → 见 §8。
- `2026-10-13T17:24:00+08:00` · axis6-lead · 派**接任 writer**（fresh，≤20 min 预算；前任无法 resume：session 路径不匹配）；同时自写第三方复算 `recompute_check.py`（结果 `recompute_check.json`）。
- `2026-10-13T17:27:18+08:00` · 接任 writer · T1–T4 完成（`analysis.md` / `criteria_fixed.json` / `recompute_criteria.py` / `mutate.py` / `mutation_report.json` / `controls/README.md`）；日志见 `WP-6.2.2-LOG.md`。
- `2026-10-13T17:31:00+08:00` · axis6-lead · 派 fresh reviewer（≤25 min，只读，未参与写作）→ `.scratch/63-axis6-predict/REVIEW-1.md`。
- `2026-10-13T17:32:11+08:00` · axis6-lead · 登记新实验到 6.2 的 `experiments` 与 `6.2/experiments/README.md`；run #1 快照降为哈希指针（`run1-snapshot/README.md`）；`summary.py --check` 全绿。

## 8 WP 6.2.2 执行记录（Phase A 步骤 2：只执行 1 个最高价值 WP）

| 项 | 值 |
|---|---|
| WP | 6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合） |
| 实验 | `axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/` |
| 预登记 | `2640.../README.md`（先写契约再跑；臂定义 / P1–P5 + C1 / 否证条件 / 禁墙钟） |
| writer | 前任 WP worker（16:26:27 派 → 16:56 被运行时上限杀死）→ **接任 writer**（17:24→17:27，完成 T1–T4） |
| 读数来源 | **run #1（16:54 落盘）**：`results.json` + `cycle_log.jsonl` + `run_log.txt`（3 个 checkpoint 完整）；判据 = 离线重算 `criteria_fixed.json` |
| GPU | §20 锁按 checkpoint 取放（`run_log.txt` 内有 `[lock] acquired/released` 与 `§20.1 让出窗口：sleep 60s`）；run #1 GPU 实耗 963 s（`wrote results.json (963s)`），在 60 min 盒内 |
| 独立复核 | fresh reviewer，报告 = `.scratch/63-axis6-predict/REVIEW-1.md`（见 §9） |

**axis-lead 自己的第三方复算（不复用 writer 判据代码）**：`.scratch/63-axis6-predict/recompute_check.py` → `recompute_check.json`（按 prompt 聚类 bootstrap，B=2000/seed=2640）：
- **臂级 α₁/τ 与 `results.json` 逐位一致**（10 臂 × 3 ckpt）⇒ 评测驱动与落盘一致。
- **P1/P2 实测成立**（配对差 = 树臂 − 该 ckpt 实测最强单臂）：13m_s0 `tree_direct_k2` Δα₁ **+0.0934** CI [0.0657,0.1217]、Δτ **+0.1857** CI [0.1251,0.2507]；`tree_topk4` Δα₁ **+0.1269**、Δτ **+0.2361**；`tree_union2` Δα₁ **+0.0855**；`tree_qpw_k2` Δα₁ **+0.0618**。13m_s1 同类（Δα₁ +0.0638…+0.1123）；53m_s0 `tree_direct_k2` Δα₁ **+0.0905**（该 ckpt 唯一跑的树臂）。⇒ run #1 `criteria` 里 P1/P2 全 false 确系**判据取键缺陷**（配对键方向相反 ⇒ null），读数本身未变。
- **P3 否证**：`combo_pw` − `combined_fixed` 的 Δα₁ = −0.0048（13m_s0）/ +0.0039（13m_s1），CI 均含 0 ⇒ 「路径概率加权 vs 固定 λ=0.5」在本尺度**不可区分**（逐轮草稿分歧率 ~5.7%/5.8% ≠ 0，说明两臂确实不同）。
- **代价轴**：`R = K·(1+γ)/(1+τ)` 从单臂 1.16–1.24 升到 K=2 的 ~2.30 / K=4 的 ~4.51 ⇒ 树用 ~1.9×（K=2）到 ~3.7×（K=4）FLOPs 换 τ +0.13…+0.24；τ 边际收益随 K 递减。**R>1 不作判据**（范式定义性质）。
- **无损性**：全部臂 × 3 ckpt × 12 prompt 的发出流逐位 == native 贪心流（P4 全过；53m_s0 也未出现 fp 平局翻转）。
- **run #1 快照处置**：快照三份与实验目录同名文件 sha256 逐字节相同 ⇒ 不入库，只留 `.scratch/63-axis6-predict/run1-snapshot/README.md`（哈希 + 主家指针），副本文件已删（证据主家在实验目录）。

**待上报 Main 的引擎改动（不擅改共享 `pathlm/**`）**：§5 的 F3（二级条件头 + 树注意力，需重训，本轮未做）。

## 9 独立复核

- `.scratch/63-axis6-predict/REVIEW-1.md`：fresh reviewer（未参与写作），预算 ≤25 min，只读；
  范围 = WP 6.2.2 实验目录全量 + 5 份 `project.yaml` 的 Phase A 改动 + 本工单 + 复算留证；
  必须尝试证伪：实现是否符合声称（概率加权 / K 候选 / 每轮 1 次顺序前向 / 无损接受判据 / 缓存回写）、
  数字能否独立复现（自写脚本复算 α₁/τ 与配对 CI）、是否越界改共享引擎、是否越证据声明。
- 结论与处置见 REVIEW-1.md 的 `VERDICT` 行（本节不重复其内容；历轮判词留本工单，不上 `conclusion_review`）。
