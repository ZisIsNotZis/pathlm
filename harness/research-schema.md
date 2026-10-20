# 研究项格式契约（`project.yaml` · SSOT）

> 本文件是**每个研究项的 YAML 文档的权威格式定义**。研究树（轴 → 研究项 → 子项）的
> 用词、编号、目录命名见 `docs/structure.md`（**exact wording 不改**）；本文件只定义
> **字段、取值域与纪律**。
>
> 迁移通告：新结构下 **`axes/**/project.yaml` 是研究项的 SSOT**。生成式 wiki
> （`catalog.yaml` / `tools/build_wiki.py` / `INDEX.md`）**退役**；旧的课题页九节契约
> （`harness/SCHEMA.md` §1–§3）随旧 `axes/**/README.md` 退役（旧页归档于 `archive/wiki-legacy/axes/**`），由本文件取代。
> `harness/SCHEMA.md` §6–§24 的**分析纪律仍然生效**（判据取向、方差界、等参数对照、独立复核、
> GPU 锁、并发写等），本文件不重复，只引用。

---

## 1. 一个研究项 = 一份 `project.yaml`

- **位置**：`axes/<轴 slug>/<id>-<slug>/project.yaml`（轴 slug 与目录命名见
  `docs/structure.md` §13）。
- **粒度**：一个研究项 = **一个可独立推进、可独立否证的研究问题**。它的**手段、子做法、
  出口、变体**不另开研究项，写成该 `project.yaml` 的 `work_packages`。纯分组节点（轴、
  「可见范围」这类上位类目）不写 `project.yaml`，只写聚合用 `README.md`。判据见
  `docs/structure.md` §12。
- 一个研究项**有且只有一个 `milestone`**（最终输出 + 验收）；推进过程拆成任意多个
  `work_packages`，每个 WP 自带 `acceptance` 与 `review`。

## 2. 字段（必须齐全；表里「必填」= 校验器会 fail）

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `id` | str | ✅ | 全局唯一编号，等于 `docs/structure.md` §12 的节点编号（如 `"1.1.1"`） |
| `axis` | str | ✅ | 轴名，用 `docs/structure.md` 的 **exact wording**（如 `"1 时间轴"`） |
| `title` | str | ✅ | 研究项名，用 `docs/structure.md` 的 **exact wording**（如 `"hard 窗口"`） |
| `dir` | str | ✅ | 仓库内相对目录，等于 `axes/<轴 slug>/<id>-<slug>`，必须与文件实际位置一致 |
| `purpose` | str | ✅ | **要解决的痛点**，一句话（目的，不是手段） |
| `innovation` | str | ✅ | 相对**普通 decoder-only transformer** 改了什么（结构/公式层面）+ **改动量级**（参数量/计算量） |
| `expected_effect` | str | ✅ | 该创新设计的**预期效用**（对准哪个痛点） |
| `sota_baseline` | str | ✅ | 对标的 **SOTA / stock 基线**（巨人）与出处；没有它就没有参照系 |
| `gap` | str | ✅ | 「我们相对 SOTA 的增量/空白在哪」——说明为什么不是重造轮子 |
| `method` | str | ✅ | 怎么做（机制/干预/手段）；单因素声明见 `harness/SCHEMA.md` §6 |
| `validation` | str | ✅ | **端到端验证**：必须同时含 ①**与 SOTA/stock 对照**、②**客观判据**、③**口径** |
| `objective_criteria` | str | ✅ | 可证伪的客观判据：**什么该涨、什么该跌、什么必须不变** |
| `falsification` | str | ✅ | **否证条件**：出现什么即判失败 / 被否证（不得只有正面路径） |
| `cost_model` | str | ✅ | **只能**是等效理论时间/计算量（FLOPs / layer-passes / tokens-per-forward，**考虑 pipelining**）。**禁墙钟与日期指标**（见 §3） |
| `role_map` | map | ✅ | `{owner, writer, reviewer, integrator}` 四个角色都必填（见 §4） |
| `milestone` | map | ✅ | **有且仅有一个**：`{output, acceptance}`（见 §5） |
| `work_packages` | list | ✅ | 每条 `{id, what, goal, means, acceptance, evidence, review, status}`（见 §6） |
| `roadmap` | list | ✅ | 推进步骤（有序字符串；可空列表，但键必须存在） |
| `status` | str | ✅ | `未开始` / `进行中` / `已完成` / `被否证`（见 §7） |
| `conclusion` | str | 条件 | `status ∈ {已完成, 被否证}` 时必填；否则可空 |
| `failure_analysis` | str | ✅ | **失败归因**：研究判失败/被否证时，必须先区分是 **设计问题 / 结构问题 / innovation 本身失败** 中的哪一类（可并列）并给证据，不得只说「没做出来」。触发与非空要求见 §7；未判失败时写 `""` |
| `conclusion_review` | str | ✅ | 这个结论**看上去对吗**：自评 + **当前（最后一轮）**独立复核判词；历轮判词写对应 ticket，不在本字段累积（见 §11.1） |
| `laws` | list[int] | ✅ | 本项证据支持 `atlas/playbook.md` 的哪几条定律（编号即该文件的「定律 N」）；无证据支撑写 `[]`。语义与判据见 §2.1 |
| `cost_rows` | list[map] | ✅ | 本项可引用的「代价事实」，一条一行（`law / change / touches_trunk / metric / delta / comparator / scale / boundary / source`）；没有可引用数字写 `[]`。纪律见 §2.2 |
| `experiments` | list | ✅ | 实验目录（仓库内相对路径，必须存在）；空列表合法 |
| `argument` | str | ✅ | 结论**是怎么论证出来的**（逻辑链 + 对照） |
| `gaps` | list | ✅ | 遗漏 / 未测项（列成「下一步最小实验 + 成本」或显式空列表） |
| `started` | str | ✅ | 起始时间（ISO8601，**溯源元数据**，允许日期） |
| `ended` | str | ✅ | 结束时间（ISO8601；未结束写 `""`） |

> 未知字段合法（不 fail），但不得与上表字段冲突、也不得与 `docs/structure.md` 用词冲突。
>
> **`change_log` 与其条目里的 `actor` 已退役**：`project.yaml` 是事实文档，只写**当前**状态；过程史
> （谁在哪一轮改了什么、为什么）的行是 `.scratch/doc-log/axes/<轴>/<格子>/CHANGELOG.md`
> （`AGENTS.md` §2.2）。留着这两个键 ⇒ `tools/check_tree.py` fail。

### 2.1 `laws`：本项证据支持哪几条定律

- **编号**指 `atlas/playbook.md` 的「定律 N」（1–11），**不复制**定律正文——正文只有那一个家，本字段只放编号。
- **判定标准**：该研究项的证据**正是**这条定律的依据之一，或该定律**明确引用**了它。主题相近、手段同类、结论方向一致都不算；填不出来的就写 `[]`，不要为了凑覆盖而发明映射。
- 一个研究项可以支撑多条定律；一条定律由多个研究项共同支撑是常态。
- 「每条定律被哪些项支撑」是**生成物**（`atlas/cost-table.md` 由 `tools/build_cost_table.py` 生成），本字段是它的**唯一事实源**；不得在别处再维护一份映射。
- 校验：编号必须是 1–11 的整数、不得重复；`cost_rows[].law` 必须是 `laws` 的成员。

### 2.2 `cost_rows`：可引用的代价事实

一条 `cost_rows` 是一个**已经落盘、能被别人核对**的代价数字，不是观点、不是计划、不是推算。

| 键 | 含义 |
|---|---|
| `law` | 这条事实归于哪条定律（必须是本项 `laws` 的成员） |
| `change` | 改了什么（人话，如「层间线性加权和（DenseFormer 式）」），不写代号主语 |
| `touches_trunk` | 是否触碰**主干本体**（bool，判定口径见下） |
| `metric` | 指标 + **单位与增长方向**，如 `clean bpc Δ（实验 − 对照；正 = 更差）` |
| `delta` | 实测代价：**带符号数字**，逐位照抄 source 里已经写着的那个数 |
| `comparator` | 对照是谁（同算力基线 / stock / 同字节对手 / 同 checkpoint 单路径） |
| `scale` | 规模与配方（如 `13M / L=8 / 12k 步 / 98.3M token`） |
| `boundary` | 适用边界一句话：这个数字**不能**被读成什么 |
| `delta_direction` | **必填枚举**：`comparator_minus_arm`（正 = 改善，规范方向）/ `arm_minus_comparator`（正 = 更差）。名字即算式，见 §2.3 |
| `source` | 该数字真正落脚的文件（`analysis.md` / `results.json` / `project.yaml` 的 `conclusion` / `.scratch/**/evidence/**`），可带 `#§标题` |

纪律（校验器逐条检查后两项，其余靠复核）：

- **不得新算**：`delta` 必须是 source 里已有的那个数——不重算差值、不换算口径、不取平均、不把 `×1.41` 改写成 `+0.41`、不把百分比换成 bpc。`metric` 说明单位与方向（本仓不同落盘文件的 Δ 方向不一致，方向写在 `metric` 里才不会被读反）。
- **照抄格式**：位数、千分位保留落盘写法，Unicode 减号 `−` 统一写成 ASCII `-`（唯一允许的格式归一）。
- **一个事实一条**：同一个数字只出现在一条 `cost_rows` 里（作为 `comparator` 被引用不算重复）。
- **没有就空着**：本项没有可引用的落盘数字时写 `cost_rows: []`，只写 `laws`——**不得**为凑表而估一个。
- **`touches_trunk` 判定口径**：改主干前向 / 残差 / 层间聚合 = `true`；旁路判断头 / 读出 / 训练配方 = `false`。
- **`source` 必须存在**：路径不存在即校验失败；数字在文件里找不到，视为该项缺陷，不得留成「待补」。
- **`delta_direction` 与 `delta` 的符号必须一致**：`delta` 的符号按该行声明的方向算出来是什么就写什么，**不得为了好看翻转数字**；要「让读者看到正数」只在展示层（`atlas/cost-table.md` 的「Δ 方向」列）做规范化，原始 `delta` 一个数字都不动（见 §2.3）。
### 2.3 `delta_direction` / `delta_directions`：Δ 方向必须机器可读

同一个 `Δ` 在本仓不同落盘文件里方向并不统一（`harness/SCHEMA.md` §17 第 3 条规定了规范方向，但已预登记的契约不许改字面）。文字修一次不够，所以方向是一个**枚举字段**，让读表和读 JSON 的人都不可能读反。同一个文件里质量腿与代价腿**各用一套方向**是常态，所以 `results.json` 用的是**字段级映射**，不是文件级单值。

- **只有三个取值**：
  - `comparator_minus_arm` = `Δ = 对照 − 实验`，**正 = 改动更好**（`harness/SCHEMA.md` §17 的规范方向）；
  - `arm_minus_comparator` = `Δ = 臂 − 对照`，**正 = 改动更差**；
  - `undeclared` = **声明「缺失」本身**：这个读数在仓内找不到方向声明，不替它选一个。
- **`cost_rows` 用单值**（§2.2）：一行一条事实，必填，不许写 `undeclared`。
- **`results.json` 用映射**：`"delta_directions": {"<字段名或 JSON 路径>": "<枚举>"}`。
  - **覆盖**：每个 `delta*` 读数（键名含 `delta` / `Δ` 且值里带数字，含它的整棵子树）都必须出现；**缺项即失败**。同一个键名在文件里出现多次（逐臂 `delta_bpc`）是一个待声明项，要分开写就写路径键，解析时先路径后键名。
  - **逐字段**：同文件不同字段可以不同方向；一个字段的方向与它所在判据块（键名含 `criterion` / `criteria`）的声明矛盾即 fail。判据块自己就并存两种约定（`Δ_schema` 与 `damage`）时，该字段列为「不判 · 待人裁决」，不替它选。
  - **`undeclared` 必配原因**：出现 `undeclared` 时必须写 `"delta_direction_note": "…"`，一句话说清为什么判不了（如「同一臂新旧口径之差，不是实验 vs 对照」）。
  - **旧的文件级单值 `delta_direction` 已废弃**：`results.json` 顶部留着它即 fail，两种形状不许并存。
- **`undeclared` 被统计，不许静默通过**：它不进自洽校验，但 `tools/check_tree.py` 每次都打印一行统计（报 Δ 的文件数 / 全字段已声明的文件数 / 含 `undeclared` 的文件数与条数），缺口必须看得见。
- **数字不动，只加标签**：`delta` 的符号按该行声明的方向算出来是什么就写什么，**不得为了好看翻转数字**；展示层规范化只在 `atlas/cost-table.md` 的「Δ 方向」列做（翻成 `正 = 改善` / `正 = 更差`），同一张表里两种方向并存，**不许**跨行比较符号。
- **机检的边界（判不了就不判，但要写清）**：
  1. `metric` 里没有方向短语（如「越低越好」「越小 = 越早报废」这类只讲量纲的写法）⇒ 符号无法机判，检查器把该行打到 stderr 的「不判」清单，`cost_rows[].delta_direction` 仍必填；
  2. 行本身报的是绝对值 / 比值（`|Δ|`、`非负`、`>1`）⇒ 只校枚举合法性；比值声明（`>1 = …`）不参与同块 Δ 字段的方向判定（它说的是另一个量的符号）；
  3. 一个字段在仓内有**两种互相矛盾的声明**且分不出哪个对这个字段 ⇒ 写 `undeclared`，`note` 里点明两处出处，不选一个；
  4. 文件里根本没有方向声明（只有数字）⇒ 全部字段写 `undeclared` + `note`，登记为缺口而不是回填一个值。

```bash
python3 tools/check_tree.py                  # 枚举、映射形状、delta 字段全覆盖、与判据串不矛盾；"不判"与 undeclared 统计打到 stderr
python3 tools/build_cost_table.py --check     # 表与 project.yaml 逐字节比对（含「Δ 方向」列）
```

- 定律 **8–11 不是「代价」**，而是**用表前必须过的口径与前置条件**；它们的 `cost_rows` 记的是「该口径下已经测到的事实」，`atlas/cost-table.md` 把它们单列一节，不复制定律正文。

生成与检查：

```bash
python3 tools/build_cost_table.py           # 生成 atlas/cost-table.md（未覆盖的定律打到 stderr）
python3 tools/build_cost_table.py --check    # 与磁盘逐字节比对；不一致或有定律未覆盖 ⇒ 非零退出
python3 -m pytest tests/ -q                  # 含「手改生成物必须被抓红」与 laws/cost_rows 字段校验
```

## 3. 代价口径：禁墙钟 / 禁日期

- `cost_model` **只**能用等效理论量：FLOPs、layer-passes（每 token 的层通过数）、
  tokens-per-forward（一次前向产出几个 token），并**说明 pipelining 影响**。
- **明文禁止**：墙钟（秒 / 分钟 / 小时 / 「快 3×」）、吞吐（tok/s）、日期型成本。
  出现即**校验失败**。理由：共享单卡 RTX 4090 的墙钟不可复现，出资人明确「所有关于
  时间的指标都没有意义……按照理论去计算」。
- 与 `harness/SCHEMA.md` §8（成本预算唯一性）、§20（GPU 锁）配套：墙钟可以**作为过程指标**
  出现在实验产物里，但**不得进 `cost_model`、不得进判据**。

## 4. `role_map`（每个研究项必须有明确 role map）

```yaml
role_map:
  owner: "轴 lead 名（对该研究项负最终责任）"
  writer: "执行/写入者（单一写入者）"
  reviewer: "独立复核者（fresh 上下文，批判性）"
  integrator: "合并与跨文档一致性负责人"
```

- `owner` 与 `writer` 可为同一人；`reviewer` **不得**是本项 `writer`（独立复核，
  见 `harness/SCHEMA.md` §14 / §14b）。
- 迁移窗口内 `owner` / `writer` / `reviewer` / `integrator` 写**角色占位**（如「轴 N lead（待指派）」）；
  **实名在轴 lead 指派时填（M3）**，本轮不臆造（未决登记见 `.scratch/56-restructure/TICKET.md`）。

## 5. `milestone`（单数、有且仅有一个）

```yaml
milestone:
  output: "最终输出物（可交付、可看：可跑 demo / 渲染图 / 实测证据 / 结论文档）"
  acceptance: "验收：怎么算过（客观判据 + 与基线的对照）"
```

- 一个研究项**只有一个最终 milestone**——即最终输出。过程节点不叫 milestone，叫
  `work_packages`。校验器要求 `milestone` 是 map 且有 `output` + `acceptance`。
- 验收不达标即判研究失败 ⇒ 按 §7 填顶层 `failure_analysis`（失败归因），不得只在 milestone 文本里写一句「没做出来」。
- **验收口径（跨轴共性裁决 C2）**：凡「用更少算力拿类似质量」类研究项，milestone 的**实质验收** =
  **不被同算力的朴素方案支配**（相对同算力最强基线，超出量 ≤ 0）。**预登记文字保持不变**；
  `conclusion` 须**同时**写明「**字面成立 / 实质成立或否证**」（两读法并列），不得只写其一。

## 6. `work_packages`（一堆 WP，每个都要定义清楚）

```yaml
work_packages:
  - id: "WP1"
    what: "这一步干什么"
    goal: "目标是什么"
    means: "手段是什么"
    acceptance: "怎么验收"
    evidence: "证据落在哪（仓库内路径）"
    review: "独立复核怎么排"
    status: "未开始"   # 未开始 / 进行中 / 已完成 / 被否证
```

- 若某 WP 的 `status` 判 `被否证`，其失败归因写进**顶层** `failure_analysis`（见 §7），
  WP 内不得只留一句「没做出来」结案。

## 7. `status`（最新状态）

| 值 | 含义 |
|---|---|
| `未开始` | 尚无本地工作 |
| `进行中` | 有本地工作、结论未定 |
| `已完成` | 本地实测已有明确结论（含失败/否证结论） |
| `被否证` | 假设被本地证据否证 |

- `status ∈ {已完成, 被否证}` ⇒ `conclusion` 必填，且 `experiments` 非空（至少一条
  存在的实验目录）作为证据。**判据悬空、无证据的结论不入账**。
- **失败归因（`failure_analysis`）**：`status = 被否证`，或 `conclusion` 判失败（含 `已完成`
  但结论是「手段不成立」）⇒ `failure_analysis` **必填且非空**，且必须**先区分**是
  **设计问题 / 结构问题 / innovation 本身失败**中的哪一类（可并列）并给证据。
  校验器对 `status = 被否证` **硬性强制**（空 ⇒ fail）；判失败但状态非 `被否证` 时由复核把关。
- 状态枚举**只回答「做完没有」**；「做到多少」写在 `conclusion` / `gaps` 正文。

## 8. 时间与变更

- `started` / `ended` 是**溯源元数据**，允许日期。
- 正文（`purpose` / `conclusion` / …）**不带流水叙事**、不写「原本是 A、后来改成 B」；变更叙事
  （谁在哪一轮改了什么、为什么）的行是 `.scratch/doc-log/axes/<轴>/<格子>/CHANGELOG.md`（家 =
  `AGENTS.md` §2.2），本文件只写当前状态。
- 正文里**不得出现内部工单号与复核编号**（`ticket NN` / `NN-review XX` / `WP-xxxx`）：读者打不开一个仓外
  的工单，而它要判断的实质已经在同一句里；写实质（「按复核意见改读数口径」）而不写编号，编号与过程
  一起留在 `.scratch/doc-log/**`。机检：`tools/check_tree.py`（出现即 fail）。

## 9. 校验与汇总

```bash
python3 tools/summary.py            # 一条命令打印全表（SSOT 状态表）
python3 tools/summary.py --check    # 缺字段 / 无证据 / 非法状态 ⇒ 非零退出
python3 tools/check_tree.py         # 同上：纯校验（summary.py --check 转调它）
python3 tools/check_schema.py       # 兼容入口：转调 check_tree.py（生成式 wiki 已退役）
```

- 校验项：字段齐全（含 `failure_analysis`）、关键字段非空（`purpose` / `innovation` /
  `validation` / `objective_criteria` / `falsification` 不允许空串/空列表）、`milestone` 单数、
  `role_map` 四角色齐、`status` 合法、`cost_model` 无墙钟/日期、`已完成/被否证` 有
  `conclusion` + 存在的 `experiments`、`被否证` ⇒ `failure_analysis` 非空、`validation`
  含对照/判据/口径、`id` 全局唯一、`dir` 与实际路径一致。
- **证据路径与锚点**：`experiments` 路径必须存在；`work_packages[].evidence` 是
  「证据将落在哪」，允许**未来路径**（只收锚点、不强制已存在）。如需钉到修订，用
  `"<路径> @<7–40 位十六进制 sha>"`，锚点须是可达真实提交（沿用旧 `check_schema.py` 的
  锚点纪律）。

## 10. 与既有纪律的关系（不重复，只引用）

- 判据取向、方差界、等参数对照、活性判据：`harness/SCHEMA.md` §10–§17。
- 独立复核与顺序屏障：`harness/SCHEMA.md` §14 / §14b。
- 不重复造轮子 / 先站巨人肩膀：`harness/SCHEMA.md` §18 / §21（`sota_baseline` 字段即此处产物）。
- 一格的「预算内极致」七问：`harness/SCHEMA.md` §23。
- 协作/编排制度与文档地图：`AGENTS.md`。

## 11. 复核与收口纪律（字段级）

> 与 `harness/SCHEMA.md` §14 / §14b 的分工：§14/§14b 规定「**要不要**复核、**什么时候**复核」；
> 本节规定复核**返工几轮、判词写哪、收口前扫什么、改动跑哪些门**。

### 11.1 `conclusion_review` 只写当前判词

- `conclusion_review` 写**自评 + 当前（最后一轮）**独立复核的判词；**历轮**判词及其处置过程写
  对应 ticket（`.scratch/<ticket>/**`），**不得**在同一字段累积多轮判词。
- 理由：同一字段累积多轮时，历轮的旧数字/旧结论会与正文（`conclusion` / `argument` / WP）冲突，
  产生「账实不符」的收口级缺陷。
- 既有已累积多轮的条目保持原样、不回填；本纪律约束其后的**每次修订**。

### 11.2 返回前做「兄弟字段一致性扫描」

- 改动某个字段后，**返回前**把**同一事实**在各兄弟字段逐处对平：`purpose` / `innovation` /
  `expected_effect` / `validation` / `objective_criteria` / `falsification` / `conclusion` /
  `failure_analysis` / `conclusion_review` / `argument` / `status` / `experiments`，以及该研究项的
  `analysis.md`。数字区间一致、指针不悬空、同一事实不在两处各写一遍（一处写、他处指针，见
  `AGENTS.md` §2.1）。

### 11.3 复核轮数上限：累计 ≤2 轮

- 一次派发内的独立复核（含返工后的再复核）**累计 ≤2 轮**；2 轮仍未 PASS ⇒ 如实登记 blocker 返回，
  不再开第 3 轮；续做由**新的派发**（新预算）承担。
- 每轮复核必须是**新的 fresh 上下文**（`harness/SCHEMA.md` §14 / §14b）；返工后不得由同一个复核者自裁。

### 11.4 改动的验证范围（按被改对象定门）

- 改**共享文件**（`pathlm/**`、`tests/**`、`tools/**`）⇒ 必跑全量 `python3 -m pytest tests/ -q`
  作为无回归门。
- 只改**文本/指针**（`project.yaml` 正文、`.scratch/**` 留痕、文档，且不含上面三类共享文件）⇒
  writer 自检的最低门 = `python3 tools/check_tree.py` + `python3 tools/summary.py --check` +
  命中文件的目标单测 + §11.2 的兄弟字段扫描；**不必**重跑全量 `pytest tests/`。
- 两类改动都必须过 `git diff --check`（文本）并如实记录命令与真实输出。

## 12. 跨轴共性裁决的判据纪律（C1 / 64-F1 / C3）

> 本节落地 `.scratch/57-program/FRAMEWORK-DECISIONS.md` 里三条跨轴取舍的裁决（C1 / 64-F1 / C3）。
> 「0.014 噪声带」本体的定义与适用纪律见 `harness/methodology.md` §2（本节不重复、只引用）。

### 12.1 判据带与可排序门（C1）

- **0.014 仅限 bpc 口径**，且**仅在「配对 Δ、≥5 窗口」**条件下作参考门槛；它是 13M 跨 run 的 bpc
  漂移界，不是通用可排序门槛。
- **不得跨量纲借用**：概率 / 接受率 / 准确率类指标**单独定门**——按 **prompt 聚类的 bootstrap /
  Wilson 95% CI 不含 0** 为主判，0.014 只作参考线。
- **单格自测带优先于跨格 0.014**：某格有自测「在力带」时以本格带为准；两带不得混用。
- **声明义务**：`falsification` 字段必须显式声明**用哪个带**（跨格 0.014 / 本格自测带 / 概率类 CI）；
  未声明者判据不可判。

### 12.2 剪枝类实验的标准对照口径（64-F1）

- 减宽 / 减 FFN 宽等**剪枝类**改动使**参数与算力同时下降** ⇒ `harness/SCHEMA.md` §15「等参数对照」在字面上
  不可得（等参数就没有减宽）。
- 标准对照改取**等 FFN 预算 / 等算力位点**（同 `Σ_l r_l` ⇒ FFN 参数与每 token FFN 等效算力逐位相等），
  并以 stock 位点为 100% 代价锚；此外必须**并列报参数量与每 token 等效算力**（防把「减参数」记成「减宽度」）。

### 12.3 共享引擎改动通道（C3）

- 需要改共享引擎（`pathlm/**`）才能测**完整机制**时，**执行方不得擅改**；把「需要什么 + 为什么 +
  最小 patch 描述」登记到受控通道 `.scratch/67-engine-changes/TICKET.md`，授权后由**单一 engine-writer**
  执行。
- 通道纪律：**单一 engine-writer**（同一实现空间不得有两个并行写入者）+ **独立复核** +
  `harness/SCHEMA.md` §22（代码编辑锁、锚点声明）与 §20（GPU 锁）串行。

## 13. 研究目录的 `RESEARCH.md`（研究级派发契约）

`axes/<轴>/<研究项>/RESEARCH.md` 是**研究级的派发契约**：一份研究项被派活时，执行方读它，
不用读别的东西就能知道「做什么、做到什么算完、谁复核、证据在哪」。它与 `project.yaml` 的关系是
**投影与事实源**，不是两份事实：

| 文件 | 它是什么 | 谁写 | 谁读 |
|---|---|---|---|
| `project.yaml` | 研究项的**事实之家**（字段见 §1–§12） | 研究级 Agent | 校验器、生成器（`docs/catalog.md` / `atlas/cost-table.md`）、复核者 |
| `RESEARCH.md` | `project.yaml` 做不到的三件事：**派发口径**（每条工作包的 owner / 预算 / 复核人）、**证据指针排成一条可走的路**、**把研究级验收映射到具体命令** | 研究级 Agent（本项 writer 或它所属轴的主 Agent） | 被派活的执行方、复核者 |

规矩：

- **不复制读数**：`RESEARCH.md` 需要数字时按指针去读（`project.yaml::conclusion`、实验 `analysis.md`），
  不把读数誊进本文——誊一份就是第二个家，必然先过期。
- **不占事实之家**：一份 `RESEARCH.md` 的开头必须说明「本文是 projected 视图，事实的家是 `project.yaml`」。
- **不挡元级目录**：`docs/catalog.md` 的七列取自 `project.yaml`，**不转发** `RESEARCH.md`；
  因此某个研究项还没写 `RESEARCH.md` 不会让元级目录缺行（`tools/build_catalog.py --check`）。
- **与 `project.yaml` 冲突时以 `project.yaml` 为准**，并当场改 `RESEARCH.md`（`@ssot`：冲突靠层归属判，
  不靠谁更新）。
- 逐项是否已落盘、跑了哪几条命令，由研究级 Agent 在本文里记；元级只在 `docs/plan.md` 记「谁还没落盘」。
