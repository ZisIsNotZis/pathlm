# 研究项格式契约（`project.yaml` · SSOT）

> 本文件是**每个研究项的 YAML 文档的权威格式定义**。研究树（轴 → 研究项 → 子项）的
> 用词、编号、目录命名见 `docs/structure.md`（**exact wording 不改**）；本文件只定义
> **字段、取值域与纪律**。
>
> 迁移通告：新结构下 **`axes/**/project.yaml` 是研究项的 SSOT**。生成式 wiki
> （`catalog.yaml` / `tools/build_wiki.py` / `INDEX.md`）**退役**；旧的课题页九节契约
> （`SCHEMA.md` §1–§3）在迁移窗口内仍适用于旧 `axes/**/README.md`，迁移完成后由本文件取代。
> `SCHEMA.md` §6–§24 的**分析纪律仍然生效**（判据取向、方差界、等参数对照、独立复核、
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
| `method` | str | ✅ | 怎么做（机制/干预/手段）；单因素声明见 `SCHEMA.md` §6 |
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
| `conclusion_review` | str | ✅ | 这个结论**看上去对吗**：自评 + 独立复核的判词 |
| `experiments` | list | ✅ | 实验目录（仓库内相对路径，必须存在）；空列表合法 |
| `argument` | str | ✅ | 结论**是怎么论证出来的**（逻辑链 + 对照） |
| `gaps` | list | ✅ | 遗漏 / 未测项（列成「下一步最小实验 + 成本」或显式空列表） |
| `started` | str | ✅ | 起始时间（ISO8601，**溯源元数据**，允许日期） |
| `ended` | str | ✅ | 结束时间（ISO8601；未结束写 `""`） |
| `change_log` | list | ✅ | 变更记录，每条 `{date, actor, change}`（见 §8） |

> 未知字段合法（不 fail），但不得与上表字段冲突、也不得与 `docs/structure.md` 用词冲突。

## 3. 代价口径：禁墙钟 / 禁日期

- `cost_model` **只**能用等效理论量：FLOPs、layer-passes（每 token 的层通过数）、
  tokens-per-forward（一次前向产出几个 token），并**说明 pipelining 影响**。
- **明文禁止**：墙钟（秒 / 分钟 / 小时 / 「快 3×」）、吞吐（tok/s）、日期型成本。
  出现即**校验失败**。理由：共享单卡 RTX 4090 的墙钟不可复现，出资人明确「所有关于
  时间的指标都没有意义……按照理论去计算」。
- 与 `SCHEMA.md` §8（成本预算唯一性）、§20（GPU 锁）配套：墙钟可以**作为过程指标**
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
  见 `SCHEMA.md` §14 / §14b）。

## 5. `milestone`（单数、有且仅有一个）

```yaml
milestone:
  output: "最终输出物（可交付、可看：可跑 demo / 渲染图 / 实测证据 / 结论文档）"
  acceptance: "验收：怎么算过（客观判据 + 与基线的对照）"
```

- 一个研究项**只有一个最终 milestone**——即最终输出。过程节点不叫 milestone，叫
  `work_packages`。校验器要求 `milestone` 是 map 且有 `output` + `acceptance`。
- 验收不达标即判研究失败 ⇒ 按 §7 填顶层 `failure_analysis`（失败归因），不得只在 milestone 文本里写一句「没做出来」。

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

- `started` / `ended` 与 `change_log[].date` 是**溯源元数据**，允许日期。
- `change_log` 每条：`{date: ISO8601, actor: "谁", change: "改了什么、为什么"}`。
- 正文（`purpose` / `conclusion` / …）**不带流水叙事**、不写「原本是 A、后来改成 B」；
  历史进 `change_log` 或对应 ticket（见 `AGENTS.md` §2）。

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

- 判据取向、方差界、等参数对照、活性判据：`SCHEMA.md` §10–§17。
- 独立复核与顺序屏障：`SCHEMA.md` §14 / §14b。
- 不重复造轮子 / 先站巨人肩膀：`SCHEMA.md` §18 / §21（`sota_baseline` 字段即此处产物）。
- 一格的「预算内极致」七问：`SCHEMA.md` §23。
- 协作/编排制度与文档地图：`AGENTS.md`。
