# 轴 2（深度轴）迁移 · 独立复核（WP-M2.3 / commit `d6f099b`）

> 复核者：fresh context subagent（fork=false），未参与迁移；**只读**（仅在 `/tmp/inj` 注入测试，未改仓库）。
> 复核对象：工作树当前 revision（`HEAD = bacaaeba6464f75b24fd28e30bca621e18527462`）+ 轴 2 迁移提交 `d6f099b`。
> 复核时间：2026-10-13（`date -Iseconds` = `2026-10-13T11:37:24+08:00` 起）。语言：中文。
> 前提：**以「迁移做砸了」为前提去证伪**，逐条找反例。

---

## 0 · 人话结论

**PASS（就本次迁移的验收面）**。把旧 C 轴（深度/早退）搬成新深度轴这件事，**该做的都做了、能证伪的都试过、没找到阻塞级错误**：
13 个节点（3 分组 + 10 研究项）齐全且标题/目录与 `docs/structure.md` §12 **逐字相等**；10 份
`project.yaml` 字段齐全、单 milestone、每个 work package 都有验收与独立复核；旧证据 48 个文件**整目录搬齐、零丢失**；
55-review 的 C-P01…C-P19 逐条有处置且**没有照抄旧错误结论**；4 项坏 YAML 注入全部被校验器抓住。

**但有两处要如实说明，不能算满分**：
1. `python3 -m pytest tests/ -q` 在我这轮**没跑完**（跑另一轴复核的并发 pytest 抢资源，到 78% 时被叫停；已观测到的
   78% 全是 `.`，无 F/E）。WP 报告自记的「827 passed」是它当时的结果，我**无法在本轮独立复现整轮**——这是**证据缺口（unverified）**，不是 WP 的错。
2. 几处**非阻塞**瑕疵（迁移后证据文档里仍写着旧 `axes/C-vertical/...` 路径；`role_map` 用占位而非实名；`2.2.4` 指针-only 进行中项）。

未发现**阻塞级**缺陷，故判 **PASS**，并把上述缺口与瑕疵原样留给 integrator。

---

## 1 · 判据逐条结果（✅/❌ + 复现）

| # | 判据 | 结果 | 复现 / 证据 |
|---|---|---|---|
| A | **节点齐全 + exact wording**：§12 轴 2 的 13 节点（3 分组 README + 10 研究项 `project.yaml`） | ✅ | `/tmp/verify_axis2.py`：13 行 id/title/axis/dir 与 §12 表逐字相等；分组 2.1/2.2/2.3 只有 `README.md`、无 `project.yaml`；轴 README 存在。见 §1.1 |
| B | **每个 `project.yaml` 字段齐全**（含 `role_map` 四角色、`sota_baseline/gap/method/validation/objective_criteria/falsification/failure_analysis/cost_model/conclusion_review/argument/gaps/...`） | ✅ | `/tmp/fields_axis2.py`：10/10 `BAD=0`；`REQUIRED` 27 字段无缺、5 个非空字段无空串 |
| C | **单 milestone（有且仅有一个 map：output+acceptance）** | ✅ | 同上脚本：10/10 `milestone` 为 map 且 output/acceptance 非空；无 list 型 milestone |
| D | **work_packages 每条含 `acceptance` + 独立 review** | ✅ | 同上脚本：全部 WP 的 `acceptance` 非空；`review` 均为「跑完另派 fresh 复核（SCHEMA.md §14/§14b）」或「另派 fresh 复核」→ 独立复核，非 self-review |
| E | **`cost_model` 无墙钟/日期**（只等效理论量） | ✅ | `grep cost_model axes/2-depth/*/project.yaml` 无「墙钟/秒/分钟/小时/ms//s」；`tools/check_tree.py` 墙钟正则全过；每份均含 layer-passes / FLOPs / tokens-per-forward + pipelining 说明 |
| F | **证据按 MAPPING 迁移、无丢失**（迁前/迁后计数，含 git 跟踪数） | ✅ | `/tmp` 脚本：旧 C-vertical 的 **48 个实验内容文件 48/48 落在 MAPPING 指定新家，LOST=0**；`git ls-files axes` 在 `d6f099b~1`=1675 → `d6f099b`=1692（+17=10 YAML+7 README）；`git ls-tree -r axes` 1675→1692；旧 C-vertical 跟踪文件 62→14（只留各格 README + `experiments/README.md`，供 integrator 退役） |
| G | **55-review C-PROBLEMS 逐条处置（不照抄）** | ✅ | 见 §3 表：C-P01…C-P19 全部有落点；关键 4 条致命项（P01/P02/P03/P04）在 `2.1.1` 内显式改正，未照抄旧「引擎已具备逐 token 门控」「tf@4 保质量」 |
| H | `python3 tools/check_tree.py` | ✅ | `OK：46 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| I | `python3 tools/summary.py --check` | ✅ | 同上（exit 0） |
| J | `python3 -m pytest tests/ -q` | ⚠️ **未跑完（unverified）** | 827 collected；并发 pytest 抢资源，到 **78%（约 620/827）全为 `.`，无 F/E** 时按编排指示停止；未取到最终 pass/fail 行。**不列为判据失败，列为证据缺口**（见 §4） |
| K | **/tmp 注入坏 YAML 证伪（4 类）** | ✅ | 控制组通过；4 类缺陷各自被抓住（见 §5） |

### 1.1 节点逐条（exact wording 比对，脚本产出）

| id | §12 标题（逐字） | 目录 | 粒度 | 实际 `project.yaml` |
|---|---|---|---|---|
| 2.1 | 按需深度 | `2.1-ondemand-depth/` | 分组 | 无（✓ 只 README） |
| 2.1.1 | early exit（最重要） | `2.1.1-early-exit/` | 研究项 | ✓ title/axis/dir 相等 |
| 2.1.2 | skipping（跳层） | `2.1.2-skipping/` | 研究项 | ✓ |
| 2.1.3 | repeating 循环 transformer：多算力换少参数 | `2.1.3-repeating/` | Backlog | ✓（status 未开始） |
| 2.2 | 层间怎么聚合（残差族） | `2.2-layer-aggregation/` | 分组 | 无（✓） |
| 2.2.1 | 残差的形式 | `2.2.1-residual-form/` | 研究项 | ✓ |
| 2.2.2 | 用 causal attention 代替残差 | `2.2.2-attention-residual/` | 研究项 | ✓ |
| 2.2.3 | 层间学习加权和（DenseFormer 式） | `2.2.3-learned-layer-sum/` | 研究项 | ✓ |
| 2.2.4 | 并行子层 | `2.2.4-parallel-sublayer/` | 研究项 | ✓ |
| 2.2.5 | 权重共享 | `2.2.5-weight-sharing/` | Backlog | ✓（status 未开始） |
| 2.3 | 深度的形状 | `2.3-depth-shape/` | 分组 | 无（✓） |
| 2.3.1 | 逐层容量变化：前大后小 / 逐渐放大 | `2.3.1-layer-capacity/` | 研究项 | ✓ |
| 2.3.2 | 逐层分辨率：先粗后细、层级式（层级合并） | `2.3.2-layer-resolution/` | 研究项 | ✓ |

---

## 2 · Material defects

### Blocking

**无。** 我按「迁移做砸了」逐项找反例（缺节点、错用词、缺字段、证据丢、照抄旧错误、校验器漏网），**没有找到阻塞级缺陷**。

### Non-blocking（位置 + 违反判据 + 复现）

| ID | 位置 | 问题 | 违反 | 复现 | 严重度 |
|---|---|---|---|---|---|
| **N1** | 迁移后的证据文档：`2.1.1/experiments/2621-depth-draft-e2e/MODEL.md`（第 32/37/39/43 行）、`2.2.3/experiments/2613-inter-layer-topology-13m/README.md`（21/138 行）、同目录 `MODEL.md`（27 行）、`analysis.md`（337 行）、`smoke/smoke.json`（357–359 行） | 这些已搬迁文件内部的**运行命令/绝对路径仍写旧家** `axes/C-vertical/...`，读者按命令会命中已搬空/已删的旧路径 | MAPPING §0.2「整目录搬到新家」未强制改写内部路径，但**可发现性**受损；WP 报告§三.5 只披露了 `smoke.json` 一处，**未提 `MODEL.md`/`README.md`/`analysis.md`** | `grep -rn "C-vertical" axes/2-depth`（11 命中，其中 3 份为迁移说明/旧家台账，其余为陈旧命令） | 非阻塞（证据本体在新家，仅指针陈旧） |
| **N2** | `2.2.4-parallel-sublayer/project.yaml`（`experiments: []` + `status: 进行中`） | 该研究项有质量面读数（parallel +0.0458）但其证据按「一格一份家」只落在 `2.2.3`，本项 `experiments: []` 靠文字指针 `see 2.2.3`。校验器接受（DONE 才要求 experiments 非空），但「指针-only 的进行中项」是否为可接受形态**需 integrator 裁决**（MAPPING §1 只写「C3 → 2.2.3 + 2.2.4」，未指定主家） | 无硬判据被违反（校验器通过），属口径待决 | 读 `2.2.4/project.yaml` 的 `experiments` 与 `conclusion_review` | 非阻塞（WP 报告§三.1/§三.2 已自曝，交 integrator） |
| **N3** | 全部 10 份 `project.yaml` 的 `role_map` | `owner`/`reviewer`/`integrator` 为**占位描述**（「轴 2 深度轴 lead（待指派）」「fresh 复核者（reviewer 池…）」），未填**实名**；M1 ticket §5.5 明写「M2 派发时需填实名」 | 判据 A「role_map 四角色齐」已满足；但 M1 计划的「M2 填实名」未执行 | `grep -n "待指派" axes/2-depth/*/project.yaml` | 非阻塞（字段齐、reviewer≠writer 文本上成立） |
| **N4** | `2.1.1` 等全部 `validation`/判据沿用 **0.014 噪声带** | 55-review 根因 R8 记「0.014 全局方差带被其自身来源否证」，本轴把它当**可排序阈值**使用 | 属**跨轴口径（框架性取舍）**，按 `AGENTS.md` §3 应由主 Agent/user 集中裁决，非 WP 可自决 | `grep -n "0.014" axes/2-depth/*/project.yaml` | 非阻塞（跨轴，交 integrator/主 Agent） |
| **N5** | 旧 `axes/C-vertical/C1-layers-per-token/README.md`（工作树未提交修订） | 该旧文件仍含 C-P03 断言的**错误说法**「引擎已具备逐 token 退出门控（`pathlm/decode.py::_run_stack`…）」，与新 `2.1.1` 的更正**并存矛盾** | 事实文档「同一事实恰好一个家」暂未成立（旧家待退役）；WP 已显式保留旧文件不改 | `git diff -- axes/C-vertical/C1-layers-per-token/README.md`（新增段仍含该句） | 非阻塞（旧家由 integrator 统一退役；**新 SSOT 已改正**，这正是任务要求） |
| **N6** | 56 TICKET §2.4「去残留」清单（含 C-P14 代码悬空引用 `mental_model.md`、旧引擎黑话、100M 税混杂口径） | WP-M2.3 报告**未逐条核销** C-P14；`2.1.1` 未承接旧 C1「100M 税 96%/4%/≈0」混杂口径（**新树里已无此结论**，属良性） | 56 §3.5「残留清单逐条消失」在**轴 2 范围内**未显式给证据（代码侧残留未必属本 WP） | `grep -rn "C-P14\|mental_model" .scratch/56-restructure axes/2-depth`（无） | 非阻塞（代码残留疑属独立 WP；新树本身干净） |

---

## 3 · 55-review（C-PROBLEMS）逐条处置核对

> 依据 `.scratch/55-review/C-PROBLEMS.md`（致命 4 / 重要 9 / 次要 4，C-P01…C-P19）。

| 问题 | 严重度 | 处置落点（新树） | 判定 |
|---|---|---|---|
| C-P01 核心命题从未端到端检验 | 致命 | `2.1.1` status=进行中；`gaps[0]`=主判据未测；`validation` 改序列级；milestone 钉在「逐 token 门控的自回归序列级结果」 | ✅ 处置 |
| C-P02 偷换命题：tf@k 单 token 自洽代理 | 致命 | `2.1.1` `argument` 明写 tf@4 是「浅层 argmax==深层 argmax」自洽、非正确性、p^N 累积；判据改序列级 | ✅ 处置 |
| C-P03 引擎张冠李戴 | 致命 | `2.1.1` `conclusion_review`/`gap`/WP1 goal 明写 `DepthARDecoder.step(inp, forced_depth)` 只支持固定退出深度；删除旧误述 | ✅ 处置（未照抄） |
| C-P04 代价口径自相矛盾（墙钟当卖点） | 致命 | `2.1.1` `validation`「墙钟不进判据」；`cost_model` 只用 layer-passes/tokens-per-forward；旧「exit-4 3.1× 墙钟」卖点剔除 | ✅ 处置 |
| C-P05 退出决策机制未落地/未校准 | 重要 | WP1「退出决策头实装 + ECE 校准 + 多阈值扫描」 | ✅ 处置 |
| C-P06 fill/no-fill 语义两套、代价未测 | 重要 | WP2「异构深度回看：fill / proj-fill / no-fill 三语义臂」 | ✅ 处置 |
| C-P07 概率头两套实现无对照 | 重要 | WP1 means「对照两套头定义（专用 conf 头每层复用 vs per-depth 读出 top-1）」 | ✅ 处置 |
| C-P08 造轮子 + 未对齐巨人 | 重要 | `sota_baseline`=ACT/UT/CALM/LayerSkip/MoD + R7；`gap`「不是重造轮子」；WP3 对齐业界配方 | ✅ 处置 |
| C-P09 C2 手段枚举/跨代引擎/p_skip 逐序列/self-review | 重要 | `2.1.2` purpose 有目的；gap 列跨代+逐序列；WP1 在 depth-ar 按逐 token 重测 | ✅ 处置 |
| C-P10 C3 无目的、判据零区分度、跨层注意力未测却 judged | 重要 | `2.2.3` 降为进行中、判据改可排序 bpc；出资人「跨层 causal attention」新建 `2.2.2` 承接 | ✅ 处置 |
| C-P11 C4 三处口径打架 | 重要 | `2.2.5` 收口为 Backlog、判据改「同参数量质量+参数量」 | ✅ 处置 |
| C-P12 缺省算力对手（更浅 dense/MoE） | 重要 | `2.1.1` `gaps` 列「缺真正的省算力对手（C-P12）」并写入 validation 对照 | ✅ 处置 |
| C-P13 C5/C6 是 C1 手段、C6 结论证据不成立 | 重要 | C5→`2.1.1` WP4 精化门、C6→WP3 深度监督配方；WP3 acceptance 补 `23d-H1-12k` tf@4 对照 | ✅ 处置 |
| C-P14 上一时代残留未 migrate | 重要 | 新树未含旧残留（100M 税混杂口径未承接）；**代码侧悬空引用（`decode.py`/`config.py`/`slider.py`→`mental_model.md`）未见处置证据** | ⚠️ 部分（见 N6） |
| C-P15 快照当结论、墙钟不可复现 | 重要 | `2.1.1`/`2.2.3` 记 n=1/单快照于 gaps；status 降进行中；墙钟不进判据 | ✅ 处置 |
| C-P16 判据无变化空间 | 次要 | `2.2.3` gap/conclusion_review 明写 `tf@4≥0.95` 对 stock dense 零区分度 | ✅ 处置 |
| C-P17 命名/目的（手段式标题） | 次要 | 标题沿用 §12 冻结 exact wording；`purpose` 写目的而非手段 | ✅ 处置 |
| C-P18 无决策价值/半边结论 | 次要 | `2.1.2` 不立「层序 shuffle」研究项；`2.2.4` 补并行子层代价面（不再只测质量面） | ✅ 处置 |
| C-P19 证据锚点不一致 | 次要 | `2.1.1` 锚点收敛为 `@1586745`/`@7725891`/`@2e86924`；全部 `git cat-file -t` = commit | ✅ 处置 |

---

## 4 · 命令与真实输出

| 命令 | 真实结果 |
|---|---|
| `git rev-parse HEAD` | `bacaaeba6464f75b24fd28e30bca621e18527462` |
| `git status --short` | ` M INDEX.md` / ` M axes/C-vertical/C1-layers-per-token/README.md` / ` M axes/C-vertical/README.md` / ` M catalog.yaml` / `?? .scratch/54-subproject-schema/` / `?? .scratch/55-review/`（迁移前既有未提交改动，WP 保留未动） |
| `python3 tools/check_tree.py` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） |
| `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） |
| `python3 -m pytest tests/ -q` | **未跑完**：`827 tests collected`；并发 pytest 抢资源，观测到 **78%（约 620 项）全 `.`、无 F/E**，按编排指示停止（未取到最终 tail 行）。**本项 unverified** |
| `git diff --check` | 无输出（exit 0，无空白错误） |
| `git diff --cached --name-only \| wc -l` | 0（本复核未 stage 任何文件） |
| `git ls-tree -r --name-only d6f099b~1 axes \| wc -l` | 1675 |
| `git ls-tree -r --name-only d6f099b axes \| wc -l` | 1692 |
| `git ls-files axes \| wc -l`（HEAD） | 1733 |
| `find axes -type f \| wc -l`（HEAD，工作树） | 2510 |

---

## 5 · /tmp 注入证伪（仅写 `/tmp/inj`，不动仓库）

把 `tools/check_tree.py` 复制到 `/tmp/inj/tools/`，在 `/tmp/inj/axes/2-depth/2.1.1-early-exit/project.yaml`
放一份**自洽的最小合法 YAML**作控制组，再各注入一类缺陷：

| 注入 | 预期 | 实测（退出码 / 报错） |
|---|---|---|
| 控制组（未破坏） | 通过 | `OK：1 个 project.yaml …`（exit 0） |
| 删 `role_map` | FAIL | `缺字段 role_map` + `role_map 必须是 map`（exit 1） |
| 删 `failure_analysis` | FAIL | `缺字段 failure_analysis`（exit 1） |
| `cost_model` 写墙钟（「exit-4 快 3× 墙钟，耗时约 2 小时」） | FAIL | `cost_model 含墙钟/日期指标 '墙钟'…`（exit 1） |
| `status=被否证` 但 `failure_analysis` 为空（conclusion/experiments 已补齐） | FAIL | `status=被否证 但 failure_analysis 为空（必须区分 设计问题 / 结构问题 / innovation 本身失败 并给证据）`（exit 1） |

**4 类缺陷全部被抓、控制组通过 ⇒ 校验器对这些门确实有效，非空转。**

---

## 6 · 未复核范围（明确排除）

1. **轴 2 之外的 9 个轴**（1/3/4/5/6/7/8/9 的迁移质量）——不在本 WP 范围。
2. **`python3 -m pytest` 整轮最终结论**——本轮未跑完（并发争抢），其「827 passed」未由我独立复现。
3. **证据文件的**测量数字正确性**（内容级）**——本复核只核**路径齐备/无丢失**与「路径常量改动未改测量数字」；
   未逐条重算实验数字。
4. **旧 `axes/C-vertical/**`、`catalog.yaml`、`INDEX.md` 的最终退役**——按设计交 integrator（本次未动、未评）。
5. **跨轴一致性**（用词重叠、`2.3.1`↔`3.1.2`、`2.3.2`↔`1.2.4`、`2.1.3`↔`2.2.5` 边界）——WP 已记 gaps，
   按 57-program §4 属 integrator 职责，未在本 WP 核。
6. **`role_map` 实名填充、0.014 口径裁决、C-P14 代码残留**——属框架性/跨轴取舍，未由本复核裁定。

---

## 7 · 一句话给 integrator

轴 2 迁移**可接受**：13 节点、10 份合格 YAML、48 份证据零丢失、55-review 逐条改正（尤其**没有照抄**
「引擎已支持逐 token 门控」与「tf@4 保质量」两个致命旧误述）。合并前请处理：`2.2.4` 主家归属裁决、
`role_map` 填实名、旧 `axes/C-vertical/**` 退役（退役后 N5 矛盾自动消失）、以及可选地清理 N1 的陈旧内部路径。
