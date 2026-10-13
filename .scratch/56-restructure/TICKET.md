# ticket 56 · 按新结构迁移研究树（执行）

> 状态：**已就绪未发车**。由新 session 自动推进。契约 = `docs/structure.md`（规范用词，不得改）。
> 问题依据 = `.scratch/55-review/`（7 份 `*-PROBLEMS.md` + `TICKET.md`）。

## 0. 目标

把现有 37 格、平铺、含残留的仓库，迁移成 `docs/structure.md` 定义的**研究树**（轴 → 研究项 →
子项），每个研究项一份 YAML，`axes/` 布局沿用。

## 1. 硬约束

1. **用词 = `docs/structure.md` 的 exact wording**（轴名、研究项名、子项名），不得改名、不得自造。
2. 代价口径 = 等效理论时间/计算量（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**。
3. 研究项必须有 `purpose / innovation / expected_effect / validation / milestone / roadmap`。
4. **不删任何研究证据**；多余/存疑的进 Backlog。
5. 主项目 = meta，无推进规则；每个研究项有 Milestone + roadmap。

## 2. 迁移任务

### 2.1 建树 + 拆并
- 按 `docs/structure.md` §1–§6 建轴与节点。
- 合并：A9→1.1.4.5、A10→1.1.4.3、M1→1.1.4.1、M2→1.1.4.2、C2→2.1.2、C5→2.1.1 精化门、
  C6→2.1.1 深度监督配方、C3→2.2.3、C4→2.2.5(Backlog)、H1→6.2、H2→6.4、H3→6.2.2、
  H4→6.2.1、H5→6.2.3、W1→3.1.1、W2→4、W3→7、W4→3.2.1、R1→5.1、R4→5.6、R7→7、
  P1/P3/P4/P5/P6→§8、F2/F4→Backlog（或 2.1.3）、G→Backlog。
- 拆：A3→1.1.1 + 1.1.3；A6→1.2.2 + 1.2.3。
- 消除：每格只保留一个 `purpose`。

### 2.2 补缺（新建研究项）
`docs/structure.md` 中**无对应旧格**的节点：1.2.1 dilated / 空洞、1.2.4 动态合并、
2.1.1 退出决策头、2.1.1 异构深度回看、2.2.1 残差的形式、2.2.2 用 causal attention 代替残差、
2.3.1 逐层容量变化、2.3.2 逐层分辨率、3.1.2 逐层宽度变化、3.1.3 同层内位置相关宽度、
3.1.4 前馈隐藏比、3.1.5 通道极化 / 稀疏激活、3.2.2 逐出口读出、5.5 fp8 / 大 head_dim、
6.2.2 概率多候选 / 树、6.3.1 diffusion、6.3.2 一次性批量 pad 填充、6.3.3 latent regression 批量生成。

### 2.3 每个研究项写一份 YAML
- 路径：`axes/<轴>/<研究项>/project.yaml`，字段见 `docs/structure.md` §10。
- 迁移旧结论时，必须按 `.scratch/55-review/` 的问题台账**修正**（目的缺失、结论悬空、
  判据错位、造轮子、口径混用），不得照抄错误结论。

### 2.4 去残留（`.scratch/55-review/` 已列）
- F2 主题只存在于旧引擎的表述；`decode.py`/`config.py`/`slider.py` 悬空引用已归档
  `mental_model.md`；被 `#` 注释的死数据；已删格引用（A5/A7 等）；`*.md.md` 双扩展名；
  旧 schema stub 实验目录；作废的 int4 数字；100M 税混杂口径。

### 2.5 汇总脚本
- 小脚本扫所有 `project.yaml` → 状态表（见 `docs/structure.md` §11）。无网页、无缓存、无 fancy。

### 2.6 一致性
- 更新 `catalog.yaml` / `axes/**/README.md` / `INDEX.md` / `AGENTS.md` 文档地图 / `SCHEMA.md`，
  使其与新树、新用词一致；`build_wiki.py` / `check_schema.py` 的去留随迁移决定。

## 3. 验收

1. 目录 = `docs/structure.md` 的树；用词逐字一致。
2. 每个研究项一份 `project.yaml`，字段齐全（含 milestone/roadmap）。
3. 旧研究证据一条不丢（删除项仅指结构位，证据进 Backlog 或对应节点）。
4. 汇总脚本一条命令出全表。
5. 残留清单（§2.4）逐条消失。
6. `.scratch/55-review/TICKET.md` 的 8 条根因逐条有对应处置。

## 4. 流程纪律

- 遵守 `/home/z/vibe/AGENTS.md` 的 Automatic Execution Mode（ticket、Work Package、独立复核）。
- GPU 串行按 `SCHEMA.md` §20；共享文件按 §22。

---

## 5. WP-M1.0 规范冻结（foundation）— 执行日志

时间：2026-10-13T09:48:38+08:00 ｜ actor：worker（fresh 上下文，单一写入者）

### 5.1 本会话出资人 L1（逐字）

> 「开始自动推进，从 Migrate 开始……首先做这个迁移，把已有的内容全部按照最标准的格式、最标准的 Layout 给它迁移过去。然后开始 spawn 你这一层的活，就是大类。然后每个大类再去管理自己……
> 每一项研究它都是得有明确的 role map……每一个这个具体的研究项都是只有，有且只有最终一个 milestone，就是最终输出。其中有一堆 work package，每个 package 都要定义它这步是干什么的，目标是什么，手段是什么，需要怎么验收。然后都需要有独立的 review。然后每一个这个任务都需要清楚自己是在研究些啥，自己尝试引入一个啥概念……这个事情应该怎么验收才能证明比之前的传统化是有效的。
> 一定要想清楚……现在的这个行业的 SOTA 是怎么样的？在这个 SOTA 之上，我们尝试解决怎么样一个问题？我们提出了哪些方法？我们要怎么去实现？我们要怎么去证明这玩意是否有效？是否有一些客观清晰一些标准去证明这玩意的结论应该是啥？如果做不出来……到底是哪里出问题？……是设计的问题？还是这个结构的问题？还是你的 Innovation 本身就是失败的？它的代价会是些啥？
> 所有关于时间的指标都没有意义……按照理论去计算。
> 先把在 Migrate 的时候，把所有的文档都更新一下，把所有的规则都说清楚，然后就开始吧。」

### 5.2 决策与理由

1. **树冻结**：`docs/structure.md` 追加 §12 节点清单（轴 1–9 每个节点的精确 id/标题/轴/目录/粒度）、
   §13 粒度规则、§14 目录命名、§15 退役与去留。原 §1–§11 用词**一字未改**（仅追加指针行）。
   轴 slug 采用 `1-time/ 2-depth/ 3-width/ 4-moe/ 5-quant/ 6-predict/ 7-references/ 8-engineering/ 9-backlog/`。
2. **粒度判定**：能独立推进/可独立否证的 = 研究项（写 `project.yaml`）；只聚合子节点的 = 分组（`README.md`）；
   同一研究问题的**手段** = 父研究项的 `work_packages`。
   - 判定 `2.1.1` 的 退出决策头/异构深度回看/深度监督配方/精化门 = **work_packages**；
     `6.2` 的 6.2.1 草稿头 / 6.2.2 概率多候选·树 / 6.2.3 验证 = **work_packages**（一个 milestone 即收口）。
   - `1.1.4.1`–`1.1.4.5` 判为**独立研究项**（不同搬运机制，各自可独立收口）。
   - 参照（`1.3` 寻址、`6.1` 自回归）进轴 7 台账；工程（P1/P3/P4/P5/P6）进轴 8；G/F 剩余进轴 9 Backlog。
   - 理由：L1 明确「一个研究项只有一个 milestone + 一堆 work package」；把手段误立为研究项会重复计里程碑。
3. **schema 冻结**：新建 `docs/research-schema.md`（`project.yaml` 全字段、role_map 四角色、单 milestone、
   work_packages 八字段、status 四值、`cost_model` 禁墙钟/日期、`validation` 三要素）。
4. **工具**：新建 `tools/check_tree.py`（新校验器）+ `tools/summary.py`（一条命令出全表，`--check` 校验）；
   `tools/check_schema.py` **改为转调** `check_tree.py`（保留旧 import 路径与锚点工具）。
5. **旧 wiki 去留（裁决）**：`catalog.yaml` / `tools/build_wiki.py` / `tools/check_catalog_strict.py` /
   `tools/backfill_legacy.py` / `INDEX.md` **退役**（不再是 SSOT、不再生成网页）。
   - **但不在 foundation 阶段物理删除/搬走**：`catalog.yaml` 的 `local:` 块是轴 worker 迁移旧结论的
     **内容源**，中途删除会丢证据。迁移窗口内三份工具加「已退役」docstring，迁移完成后整体移入
     `archive/wiki-legacy/`（证据保留，不删）。
   - 理由：出资人「不要 build 网页，小脚本统计状态就完事」；同时硬约束「不删任何研究证据」。
6. **文档一致性**：`AGENTS.md`（文档地图 / §2.3 生成物 / §8 命令）、`SCHEMA.md`（顶部迁移通告 + §5）、
   `README.md`（是什么 / 怎么读 / 轴记号）、`CONTRIBUTING.md`、`docs/design.md` §4、`docs/methodology.md`、
   `docs/playbook.md` 同步指向新规范；生成式 wiki 条款标注为已被取代。
7. **exemplar**：建 `axes/1-time/1.1.1-hard-window/project.yaml`（由 A3 迁移 + 按 55-review 修正）、
   `axes/9-backlog/9.1-selector-scheduling/project.yaml`（Backlog）。
8. **映射**：`.scratch/56-restructure/MAPPING.md`（37 格 + 缺项 + 参照/工程/Backlog + 拆并证据归属 +
   55-review 根因处置 + 工作树未提交改动处置 + 证据不丢失声明）。
9. **程序骨架**：新建 `.scratch/57-program/TICKET.md`（M1/M2/M3 里程碑 + 轴 lead 契约骨架 + M1 边界）。

### 5.3 改动文件

- 新增：`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`、`.scratch/56-restructure/FOUNDATION-REPORT.md`、
  `tools/summary.py`、`tools/check_tree.py`、`.scratch/57-program/TICKET.md`、
  `axes/1-time/1.1.1-hard-window/project.yaml`、`axes/9-backlog/9.1-selector-scheduling/project.yaml`。
- 修改：`docs/structure.md`（追加 §12–§15 + §10/§11 指针）、`AGENTS.md`、`SCHEMA.md`、`README.md`、
  `CONTRIBUTING.md`、`docs/design.md`、`docs/methodology.md`、`docs/playbook.md`、
  `tools/check_schema.py`（转调）、`tools/build_wiki.py`/`tools/check_catalog_strict.py`/`tools/backfill_legacy.py`（加退役标注）、
  `tests/test_check_schema_anchor.py`（改指新校验器与 project.yaml 证据源）。

> **更正（2026-10-13T09:58:18+08:00）**：上面「修改：`docs/structure.md`（追加 §12–§15…原 §1–§11 用词一字未改）」
> 与实际不符。git 记录显示 `docs/structure.md` 在 `ab1d8d8` 里是**新建文件**（`git cat-file -e
> ce2610a:docs/structure.md` 失败；全仓无其它 ref 含该文件），并非在既有文件上「追加」。
> 因此该文件的 §1–§11 是否为「未改动」**无法从 git 独立核验**（无先前版本），且「追加」对一个
> git 新增文件是误导性描述。正确处理（**不改写历史**）：本工单如实记为「**本次为新建文件，内容含
> 既有 §1–§11 表述 + 新 §12–§15**」，并以一个**更正提交**在提交信息里说明「上一条信息超出实际改动」。
> 详见 `.scratch/56-restructure/REPAIR-foundation.md`。

### 5.4 命令与结果

- `python3 tools/check_tree.py` → `OK：2 个 project.yaml 全部符合 docs/research-schema.md`（exit 0）。
- `python3 tools/summary.py` → 打印 2 行全表（id/轴/起/止/完成?/状态/结论/结论对不对/实验/论证/遗漏）。
- `python3 tools/check_schema.py` → 转调，同样 exit 0。
- `python3 -m pytest tests/ -q` → **827 passed in 94.85s**（0 失败）。
- `find axes -type f | wc -l` = 2434（迁移前基线，供 M2 对比）。

### 5.5 剩余未决（待裁决）

1. **轴 lead / reviewer / integrator 实名**：role_map 骨架已建，M2 派发时需填实名（当前占位
   「轴 N lead（待指派）」）。默认：由主 Agent 在 M2 派发时填。
2. **架构性口径**（框架性取舍，未自决，给出默认）：
   - **pareto frontier 的「100% 全量标准」如何定义**（是全因果 attention 的 bpc，还是同参数 dense 的
     质量-代价前沿）。默认：以 `7-references` 的 stock 全因果基线为 100%，见 `1.1.1` exemplar 的
     `validation`；如出资人要另一种参照系，请裁决。
   - **`2.1.3`/`2.2.5` 的 Backlog 身份**：默认保留研究项形式（`status: 未开始`），不删。
3. **`catalog.yaml` 物理归档时机**：默认 M2 全轴迁完后移入 `archive/wiki-legacy/`；如出资人要求立即搬，请裁决。

---

## 6. WP-M1.0 复核 FAIL → 修复（round 2）

时间：2026-10-13T10:00:28+08:00 ｜ actor：worker（fresh 上下文，单一写入者）

### 6.1 触发

独立复核（`.scratch/56-restructure/REVIEW-foundation.md`，已**原样复制入库**）判 **FAIL**：
阻塞 B1 = 冻结 schema 缺「失败归因」字段（出资人 L1 明确要求「区分设计问题 / 结构问题 / innovation 失败」）；
另 8 项非阻塞（N1–N8）。

### 6.2 改动

- **B1（阻塞）**：`docs/research-schema.md` 新增必填字段 `failure_analysis`（失败归因：设计问题 /
  结构问题 / innovation 本身失败，可并列 + 证据），§2 字段表加行、§5/§6/§7 说明触发（`status ∈ {被否证}`
  或 `conclusion` 判失败时必填）、§9 校验项同步；`tools/check_tree.py` 加入 `REQUIRED` + 硬门
  「`status=被否证` ⇒ `failure_analysis` 非空」；两份 exemplar（`1.1.1`/`9.1`）补键。
- **N1**：选方案 **(a)**——`docs/research-schema.md` §9 收窄为「`experiments` 路径必须存在；
  `work_packages[].evidence` 允许未来路径（只收锚点）」，与校验器现状一致。理由：WP `evidence` 是
  「证据将落在哪」，预登记阶段天生是未来路径，加严会与「契约先于证据」冲突。
- **N2**：`SCHEMA.md` 开篇不再把 `build_wiki.py`/`catalog.yaml` 当现行；§0/§3/§4/§12/§17.13/§21/§22
  逐处加「已退役，见顶部迁移通告 / `docs/structure.md` §15」指针。**未删任何仍生效的分析纪律**。
- **N3/N4**：`tools/check_tree.py` 补 `validation` 的「判据」关键词门 + 关键字段
  （`purpose`/`innovation`/`validation`/`objective_criteria`/`falsification`）非空门。
- **N5**：`CONTRIBUTING.md` 硬性规则编号 `1.2.4.5.6.7.` → 连续 `1.2.3.4.5.6.`。
- **N6**：本节 §5.3 追加更正说明（`docs/structure.md` 本轮为**新建**文件，含既有 §1–§11 内容），
  并以**更正提交**如实说明，**未 amend/改写历史**。
- **N7**：`MAPPING.md` §0 与 `.scratch/57-program/TICKET.md` M2 行注明 **2434 是工作树文件计数**
  （含约 777 个 gitignore 产物），非跟踪文件计数，清 cache 会漂移。
- **N8**：`docs/structure.md` §9「调度/选择器」→「调度 / 选择器」，与 §12 及 exemplar 标题一致。

### 6.3 命令与结果

- `python3 tools/check_tree.py` → `OK：2 个 project.yaml 全部符合 docs/research-schema.md`（exit 0）。
- `python3 tools/summary.py --check` → 同上（exit 0）。
- `python3 tools/check_schema.py` → 转调，同上（exit 0）。
- `python3 -m pytest tests/ -q` → **827 passed in 99.79s**（0 失败）。
- `/tmp/repair-foundation-inject/` 注入 `status=被否证` + 空 `failure_analysis` ⇒ **FAIL**（新门生效）；
  补非空 ⇒ OK；删键 ⇒ 缺字段 FAIL；`purpose: ""` ⇒ FAIL；`validation` 无判据 ⇒ FAIL。

### 6.4 未决

- 本 WP 的 PASS 判定须由**新的 fresh 独立复核**给出（复核者不得为本次 writer）；
  详见 `.scratch/56-restructure/REPAIR-foundation.md` §4。
