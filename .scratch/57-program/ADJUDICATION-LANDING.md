# ticket 57 · 框架性取舍裁决 · 落地报告（ADJUDICATION-LANDING）

> 交付物：本文件（subagent 输出）。写入者 = framework-adjudication-writer（fresh，单一写入者）。
> 权威输入：`.scratch/57-program/FRAMEWORK-DECISIONS.md`（21 条）、`docs/structure.md`、
> `docs/research-schema.md`、`docs/methodology.md`、各轴 `TICKET.md`。
> 边界：只动文档 / SSOT / 台账与 `project.yaml` 字段；**不改任何测量数字、不删证据、不实现引擎代码**。
> 出资人裁决：**按各条建议 / 默认处置推进**（2026-10-13 授权）。落地提交见下「门与提交」。

---

## 0 人话摘要（先读这段）

出资人把六个研究大类报上来的 **21 条框架性取舍**一次性裁了，统一口径是「**按各条建议 / 默认处置推进**」。
本轮做的是**把裁决写进项目的「共同规矩」（SSOT）和受影响的每个研究项的档案（`project.yaml`）**，并把
台账标记成「已裁决」。**没有碰任何测量数字、没有删任何证据、没有写一行引擎代码**。

三大件落地：

1. **「0.014 噪声带」到底能不能当判决阈值（C1）** → 写进规矩：它**只对 bpc 口径、且只在「配对 Δ、≥5
   窗口」时**能当参考门槛；概率 / 接受率类指标**各自定门**（用 bootstrap / Wilson 置信区间不含 0 为主判）；
   **某格自己测出来的带优先**于跨格的 0.014；每个研究项的 `falsification` 必须写明「**用哪个带**」。
   落到文件：`docs/research-schema.md` §12.1 + `docs/methodology.md` §2；8 个研究项补了声明。
2. **验收到底怎么算过关（C2）** → 写进规矩：**实质验收 = 「不被同算力的朴素方案支配」**（比同算力最强
   基线，超出量 ≤ 0）；预登记的原话**保留**，但 `conclusion` 必须**同时**写「**字面成立 / 实质成立或否证**」。
   落到文件：`docs/research-schema.md` §5；8 个研究项（1.1.4.1 / 2.1.1 / 3.1.4 / 4.1 / 4.2 / 4.3 / 5.1 / 6.2）。
3. **共享引擎能不能动（C3）** → **授权建一条受控通道**（**本轮不改代码**）：新建
   `.scratch/67-engine-changes/TICKET.md`，把三个轴上报的「最小 patch 需求」登记成队列（轴 1 的
   CarryChannel 逐层状态 + 写入门；轴 3 的逐层 FFN 宽开关；轴 6 的二级条件头 + 重训），并约定
   「单一 engine-writer + 独立复核 + §20/§22 锁」。三项**授权状态均记「未授权」**，等出资人逐项批。

另两条共性：**C4** 旧 wiki / catalog 退役 **追认**（文档保持现状）；**C5** `conclusion_review` 里累积的
**4 条历轮判词保持原样**（纪律只约束其后修订）。

各轴 16 条默认处置**逐条落地**（不改结论方向）：编号改动见 §2。「MoE 判决降级」（65-F1）这条有歧义
（「参照」按 `docs/structure.md` §13.4 要删 `project.yaml`，与任务「同步三份 `project.yaml`」冲突），
**已请示主 Agent 并按其裁决**：以 **Backlog** 落地（保留 `project.yaml` + 本仓等效代价口径记录），
同时在 `axes/7-references/README.md` 增一行业界标准 MoE 跑分线，承接「参照」那层含义。

---

## 1 门命令结果（全绿）

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。` |
| `python3 tools/check_tree.py` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。` |
| `git diff --check` | 无输出（干净） |
| YAML 解析（46 份 `project.yaml`） | 全部 `yaml.safe_load` 通过 |
| 跨轴一致性 | 4.1/4.2/4.3 的 `id`/`axis`/`title` 与 `docs/structure.md` §12 逐字一致；`axes/7-references/README.md` 的 MoE 指针路径可达；`project.yaml` 的 `conclusion_review` 状态引用已与 `status` 对平 |

> 说明：本轮只改**文本 / 指针**（文档 + `project.yaml` 正文 + `.scratch/**` 留痕），按
> `docs/research-schema.md` §11.4 **不必**重跑全量 `pytest tests/`；未改 `pathlm/**`、`tests/**`、`tools/**`。

## 2 提交（显式路径 `git add` + heredoc commit）

| 提交 | 内容 | staged 文件数（已人工核对） |
|---|---|---|
| `e98a26c` | ticket 57 · 落地 21 条裁决（SSOT + 受影响 `project.yaml`） | 18 |
| `ea8b5c2` | ticket 57 · C3 受控引擎改动通道 | 1 |

提交后 `git status` 干净、`git diff --cached --name-only` = 0。

## 3 逐条裁决落地指针

### 跨轴共性（C1–C5）

| # | 裁决内容 | 落地文件:字段 |
|---|---|---|
| **C1** | 0.014 仅限 bpc + 仅「配对 Δ、≥5 窗口」作参考门槛；概率/接受率类单独定门；单格自测带优先；falsification 声明「用哪个带」 | `docs/research-schema.md` §12.1；`docs/methodology.md` §2；`project.yaml` × 8：`1.1.4.1` / `2.1.1` / `3.1.4` / `4.1` / `4.2` / `4.3` / `5.1` / `6.2` 的 `validation`+`falsification` |
| **C2** | 实质验收 = 不被同算力朴素方案支配；保留预登记文字；`conclusion` 并列写「字面成立 / 实质成立或否证」 | `docs/research-schema.md` §5；`project.yaml` × 8：`1.1.4.1` / `2.1.1` / `3.1.4` / `4.1` / `4.2` / `4.3` / `5.1` / `6.2` 的 `milestone.acceptance`+`conclusion` |
| **C3** | 授权建受控引擎改动通道（本轮不改代码） | `.scratch/67-engine-changes/TICKET.md`；`docs/research-schema.md` §12.3 |
| **C4** | 旧 wiki 退役追认（已知悉 / 追认） | `.scratch/57-program/FRAMEWORK-DECISIONS.md` §6 标记（`docs/structure.md` §15 / `AGENTS.md` §2.3 现状，无改） |
| **C5** | `conclusion_review` 累积保持现有 4 条原样 | `.scratch/57-program/FRAMEWORK-DECISIONS.md` §6 标记（`docs/research-schema.md` §11.1 现状，无改；未动任何判词） |

### 各轴独有（16 条）

| # | 裁决内容 | 落地文件:字段 |
|---|---|---|
| 62-F2 | 代价主轴 = 每位置等效 KV 字节，FLOPs 副轴（确认） | `axes/1-time/1.1.4.1-explicit-memory/project.yaml` : `method` |
| 62-F3 | 只报前沿位置与支配关系，不把「形态税」写成结论 | `1.1.4.1-explicit-memory/project.yaml` : `gaps` |
| 62-F4 | 保留 milestone 文字；`gaps` 记「业界实现对齐未做」 | `1.1.4.1-explicit-memory/project.yaml` : `gaps`（现状已具备，仅标记） |
| 61-F1 | 确认「序列级 perplexity = teacher-forcing 在线决策整段 bpc」 | `axes/2-depth/2.1.1-early-exit/project.yaml` : `validation` |
| 61-F3 | 知悉排期（先纯推理主判据、训练类后置） | `FRAMEWORK-DECISIONS.md` §6 标记（无字段改） |
| 61-F4 | 维持两格 Backlog、`gaps` 互指、**不合并** | `axes/2-depth/2.1.3-repeating/project.yaml` : `gaps`；`axes/2-depth/2.2.5-weight-sharing/project.yaml` : `gaps` |
| 64-F1 | 标准对照 = 等 FFN 预算 / 等算力位点（确认） | `docs/research-schema.md` §12.2；`3.1.1` : `method`；`3.1.4` : `validation` |
| 64-F3 | 接受部分臂 n=1，按既定递减 | `3.1.1` / `3.1.4` 的 `validation` |
| **65-F1** | MoE 判决降为**参照/Backlog**，只留本仓等效代价口径记录（按主 Agent 裁决以 **Backlog** 落地 + 业界 MoE 跑分线行） | `docs/structure.md` §12 轴 4 表（粒度→Backlog）+ §13.6；`axes/7-references/README.md`（新增行）；`4.1` / `4.2` / `4.3` 的 `status`（→未开始）+ `gaps` |
| 65-F2 | 记录 GPU 协调事项 | `axes/4-moe/4.1-sparse-experts/project.yaml` : `gaps` |
| 66-F1 | 用本格在力带（0.036324），两带关系写清、不得混用（确认） | `axes/5-quant/5.1-kv-quant/project.yaml` : `validation`/`falsification`（现状已具备 + C1 声明） |
| 66-F2 | 头族优势记为「参数混杂的可见后果」，不算机制发现 | `axes/5-quant/5.1-kv-quant/project.yaml` : `gaps` |
| 66-F3 | `5.1` 保持 `已完成` | `axes/5-quant/5.1-kv-quant/project.yaml` : `status`（无改，仅标记） |
| 66-F4 | 等参数臂 = `5.1` 与 `5.4` 共用 **WP**，登记归属 | `5.1` : `work_packages`（新增 `WP3`）+ `gaps`/`roadmap`；`5.4` : `gaps`/`roadmap` |
| 63-F2 | 多候选结论附等 FLOPs 对照要求，不可得则注明 | `axes/6-predict/6.2-speculative-decoding/project.yaml` : `validation`/`falsification` |
| 63-F4 | `6.3.3` 采用「(i) 直接发 / (ii) 当草稿再验证」两条各作一臂分报代价 | `axes/6-predict/6.3.3-latent-regression/project.yaml` : `method`/`milestone`/`validation`/`objective_criteria`/`falsification`/`cost_model` |

### 台账标记（21 条全部）

`.scratch/57-program/FRAMEWORK-DECISIONS.md` 新增 **§6「裁决落地（21 条全部已裁决）」**，逐条标
「**已裁决**（2026-10-13 出资人授权按建议推进；落地记录 2026-10-14T06:44:03+08:00）」+ 内容 + 落地指针。

## 4 未决 / 残留

- **65-F1 的表述取舍**：任务原话「降为**参照**……同步三份 `project.yaml`」与 `docs/structure.md` §13.4
  「参照 = 不写 `project.yaml`」冲突。已上报主 Agent，裁决 = **Backlog**（保留 `project.yaml`），并在
  `axes/7-references/README.md` 增业界 MoE 跑分线行承接「参照」含义。**若出资人本意是真删 `project.yaml`
  并迁 `experiments/**` 到 7-references，则需改判**（本轮按主 Agent 裁决执行）。
- **C1 字段声明的覆盖范围**：按 C1 来源（轴 1/2/3/4/5/6）对**已判/在跑且显式使用该带**的 8 个研究项补了
  「用哪个带」声明；`未开始` 的研究项未逐一补（它们直接受新 SSOT 纪律约束）。若要求「全部 46 项一律声明」，
  可另派一轮批量补。
- **C5**：`1.1.4.1` / `2.1.1` / `5.1` / `6.2` 的 `conclusion_review` 累积历轮判词**按 C5 保持原样、未回填**；
  其后每次修订按 `docs/research-schema.md` §11.1 只留当前判词。
- **C3 队列**：E1/E2/E3 三项 patch **未授权**；实现属「新派发」，需出资人逐项批准 + 覆盖预算。
- **本 WP 只落地文档/字段**，未运行任何 GPU 工作、未跑全量 `pytest`（§11.4：本轮未改共享文件）。
