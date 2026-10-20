# 迁移台账（元级收身的无损证明）

> 本文件证明**这次分层搬迁没有丢事实**：迁移前那一版树上的每个元级 / 根级事实文档，
> 在这里都有且只有一行，写清它**迁到哪**或**为什么删除**。
> 机检入口：`python3 tools/check_ledger.py --check`——它按 `BASE_FILES`（迁移前那一版树的
> 文件清单）逐个对账，新家必须是**真实存在的文件**，删除行必须写明理由，本次新增的文档也要有一行。
> 分层的规矩（三层各拥有什么、谁可以改、派发契约给什么）在 `AGENTS.md`；
> 元级计划与工作包在 `docs/plan.md`。

## 1 逐文件台账

三列：**原文**（迁移前的位置）· **处置**（迁到的新家 / 删除 + 理由）· **说明**（这条事实现在住在哪一层）。

| 原文 | 处置 | 说明 |
|---|---|---|
| `AGENTS.md` | `AGENTS.md`（原地重写） | 元级：三层契约 + 派发模板。旧内容是研究级协作制度与文档地图，三层表接替它，地图收进本文件的一节 |
| `README.md` | `README.md`（原地重写） | 元级：这是什么元仓、目录怎么读、一个研究怎么被派发与验收。旧的「五条最硬结论」与状态读数迁到 `atlas/FINDINGS.md`（索引）与 `atlas/cost-table.md`（数字），本文件不再转述 |
| `CONTRIBUTING.md` | `CONTRIBUTING.md`（原地保留，指针改到新家） | 元级：贡献流程。它引用的格式契约与研究树布局改指 `harness/research-schema.md` 与 `docs/structure.md` |
| `LICENSE.md` | `LICENSE.md`（原地不动） | 对外许可，与分层无关 |
| `STATUS.md` | 删除：它是纯路由器（自己写明「不拥有事实」），而它转述的每一件事都已有家——里程碑与工作包状态在 `docs/plan.md`，逐项结论在 `axes/**/project.yaml`，结论索引在 `atlas/FINDINGS.md`，研究目录在生成物 `docs/catalog.md`；元级不再留第二份路由 | 无新家（事实原本就不在它身上） |
| `FINDINGS.md` | 迁到 `atlas/FINDINGS.md` | 研究级：已确立结论的**路由器**（只指向，不持有数字） |
| `SCHEMA.md` | 迁到 `harness/SCHEMA.md` | 工级：实验与课题页的格式契约、分析纪律 |
| `docs/cost-table.md` | 迁到 `atlas/cost-table.md` | 研究级（生成物）：定律 ↔ 证据 ↔ 代价数字的对照表，由 `tools/build_cost_table.py` 生成 |
| `docs/design.md` | 迁到 `atlas/design.md` | 研究级：核心设计（RAP）与共享底座 |
| `docs/engine_capability.md` | 迁到 `harness/engine_capability.md` | 工级：引擎能力面（哪些组合真跑得起来、拒绝在哪一行） |
| `docs/methodology.md` | 迁到 `harness/methodology.md` | 工级：测量纪律（三列账本、方差界、双货币口径、冒烟与变异、预登记绑定） |
| `docs/paper-plan.md` | 迁到 `atlas/paper-plan.md` | 研究级：把定律写成论文的 claim / 结构 / 验收（计划，不是结果） |
| `docs/plan.md` | `docs/plan.md`（原地重写为元级计划） | 元级：唯一里程碑、工作包（owner / 验收 / 预算 / 状态）与元级证据指针。旧文件里的研究级测试计数、逐格判词与复现清单三节分别迁到 `docs/plan.md` 自身的元级口径、`axes/**/project.yaml` 与生成物 `atlas/repro.md` |
| `docs/playbook.md` | 迁到 `atlas/playbook.md` | 研究级：已测定律的正文与适用边界（定律 1–11 的唯一家） |
| `docs/repro.md` | 迁到 `atlas/repro.md` | 研究级（生成物）：复现清单（run / 产物 / 重跑命令 / 三态可复现判定），由 `tools/inventory_runs.py` 生成 |
| `docs/research-schema.md` | 迁到 `harness/research-schema.md` | 工级：研究项 `project.yaml` 的字段、取值域与纪律 |
| `docs/structure.md` | `docs/structure.md`（原地收瘦到约定） | 元级：轴 / 用词 / 粒度 / 目录命名。轴 7 与轴 8 的**逐实验清单**原本在本文件 §16，已随事实源归位到 `axes/7-references/README.md` 与 `axes/8-engineering/README.md`，本文件只留约定与指针 |
| `docs/triage.md` | 迁到 `atlas/triage.md` | 研究级：研究优先级排序判据、逐格打分与批次草案 |
| （本次新增） | `atlas/README.md` | 研究级：本层索引（跨格汇总有哪几份、各自拥有什么事实） |
| （本次新增） | `harness/README.md` | 工级：本层索引（有哪几份契约、各自约束什么、机检入口） |
| （本次新增） | `docs/catalog.md` | 元级（生成物）：一行一个研究项的研究目录，由 `tools/build_catalog.py` 从 `axes/**/project.yaml` 生成 |
| （本次新增） | `docs/ledger.md` | 元级：本文件（迁移台账） |
| （本次新增） | `tools/build_catalog.py` | 工级：`docs/catalog.md` 的生成器与 `--check` |
| （本次新增） | `tools/check_meta_layer.py` | 工级：元级文档零研究级读数的机检 |
| （本次新增） | `tools/check_ledger.py` | 工级：本台账的机检 |
| （本次新增） | `tests/test_catalog.py` | 工级：研究目录的生成 / 过期 / 手改 / 可失败对照 |
| （本次新增） | `tests/test_ledger.py` | 工级：台账的覆盖 / 新家存在 / 删除理由 / 可失败对照 |
| （本次新增） | `tests/test_meta_layer_isolation.py` | 工级：元级隔离的注入探针与白名单口径 |

## 2 与旧规则冲突的处置（规则优先于现状）

方向与某条既有机器检查冲突时，**规则优先**：改检查、改既有文档，并在这里记下改了什么。
反之，不得为了让检查变绿而删除事实——事实只能搬家（见上表）。

| 位置 | 改了什么 | 为什么 |
|---|---|---|
| `tools/check_docs.py` | 扫描面与三层文档地图按新分层重写（`docs/**` 元级、`atlas/**` 研究级、`harness/**` 工级，各认一份索引） | 旧地图是「单层一张表」，新分层里每层自己拥有索引；`atlas/README.md` 与 `harness/README.md` 接替旧表的研究级 / 工级行 |
| `tools/check_docs.py` | `PENDING_TARGETS` 表逐条清空 | 它声明的是「计划里点名但尚未落盘」的交付物；这些路径落盘后必须从该表删除，否则检查会对**存在**的文件静默放行 |
| `.github/workflows/wiki.yml` | CI 闸门从已退役的 wiki 生成器改为元级四条机检 | 旧生成器随 `catalog.yaml` / `INDEX.md` 一起退役（`archive/wiki-legacy/`）；元级不能把闸门挂在一个已退役的工具上 |
| `docs/plan.md` | 重写为元级计划：唯一里程碑 + 五个工作包 + 按层分工的机检清单 | 旧计划把研究级测试计数、逐格证据与复现清单写在同一份文件里，元级读者读状态要穿过它们；这是本次要治的层的混淆本身 |

## 3 本工具不判（honest limits）

1. **搬迁是否真的无损**：本检只证明「每个原文都有归宿、每个新家都存在」，证明不了新家**内容**
   完整——那是复核者逐条读 `git diff` 的活（`2d68eb4` 里 12 个文件是 `R100` 纯移动，可逐字节复核）；
2. **删除理由是否成立**：本检只要求理由非空且不短于一句，判不了它说服力够不够；
3. **新家是否选对了层**：`docs/cost-table.md` 迁到 `harness/` 一样能通过本检，层归属的正确性靠
   `AGENTS.md` 的三层表与复核；
4. **历史中间态**：只对账 `BASE_FILES` 这一个时点。搬迁中途删掉又加回来的文件不会被本检看见。
