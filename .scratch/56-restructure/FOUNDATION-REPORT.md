# WP-M1.0 规范冻结（foundation）· 交付报告

时间：2026-10-13T09:48:38+08:00 ｜ actor：worker（fresh 上下文，单一写入者）
工单：`.scratch/56-restructure/TICKET.md` §5 ｜ 程序：`.scratch/57-program/TICKET.md`

## 0. 人话摘要

这次只干一件事：**把「迁移的规矩」定死，让后面每个大类的 worker 照着做，不再各写各的。**
具体说，我把「这棵研究树到底长什么样」（轴 → 研究项 → 子项，每个节点叫什么、编号几、放哪个目录）
写死在 `docs/structure.md`；把「每个研究项那份 `project.yaml` 必须有哪些字段、什么不能写」
写成权威文件 `docs/research-schema.md`；把「哪个旧格子搬到哪儿、证据跟着谁走」逐条列进
`MAPPING.md`（保证一条证据都不丢）。工具方面，写了一个 `tools/summary.py`——一条命令打印全树状态表，
和 `tools/check_tree.py`——格式/证据不对就直接报错；旧的生成式网页工具（`build_wiki` / `catalog`）
**退役**，但不马上删（旧 catalog 里还存着要迁的结论），等各大类迁完再整体进 `archive/`。
最后给了两份 `project.yaml` 样板（`1.1.1` 和一个 Backlog 项），供各大类照抄。
**没有**替各大类把格子都建出来——那是下一阶段（M2）各大类自己的活。

## 1. 交付物清单

| # | 交付物 | 路径 |
|---|---|---|
| 1 | 冻结研究树（节点清单 + 粒度规则 + 目录命名 + 退役去向） | `docs/structure.md`（新增 §12–§15，§10/§11 加指针） |
| 2 | `project.yaml` 权威格式与纪律 | `docs/research-schema.md`（新建） |
| 3 | 旧 → 新映射 + 证据归属 | `.scratch/56-restructure/MAPPING.md`（新建） |
| 4 | 全表汇总脚本 | `tools/summary.py`（新建；`--check`） |
| 5 | 研究树校验器 | `tools/check_tree.py`（新建）；`tools/check_schema.py`（改为转调） |
| 6 | 样板 `project.yaml`（研究项 / Backlog） | `axes/1-time/1.1.1-hard-window/project.yaml`、`axes/9-backlog/9.1-selector-scheduling/project.yaml` |
| 7 | 文档一致性 | `AGENTS.md`、`SCHEMA.md`、`README.md`、`CONTRIBUTING.md`、`docs/design.md`、`docs/methodology.md`、`docs/playbook.md` |
| 8 | 工单 + 程序骨架 | `.scratch/56-restructure/TICKET.md` §5、`.scratch/57-program/TICKET.md` |

## 2. 逐项验收自证

1. **`docs/structure.md` 完整节点清单 + 粒度规则 + 目录命名；exact wording 未改**
   - §12 逐条列出轴 1–9 的每个节点（id / exact title / axis / 目录 / 粒度）；§13 粒度规则；
     §14 目录命名；原 §1–§11 正文未改（只追加了 §10/§11 的指针行）。
2. **`docs/research-schema.md` 完整 schema**
   - 字段含 `role_map{owner,writer,reviewer,integrator}`、**单数 milestone `{output,acceptance}`**、
     `work_packages[]`（id/what/goal/means/acceptance/evidence/review/status）、`cost_model` **禁墙钟/日期**、
     `validation` 必含「对照 + 判据 + 口径」。校验器逐条强制。
3. **`MAPPING.md` 覆盖全部旧格、无证据丢失声明可核**
   - §1 逐条覆盖 37 格（旧路径 → 新家/处置/证据去向/55-review 条目）；§0 给出可核口径
     （2434 文件数、整目录搬、拆并只落一个主家、`.scratch` 证据不动）；§3 拆并证据归属。
4. **`tools/summary.py` 一条命令出全表；validator 能抓缺字段/无证据**
   - `python3 tools/summary.py` 打印 11 列全表；`--check` / `check_tree.py` 校验字段、单 milestone、
     role_map、status、cost_model 口径、`已完成/被否证` 必须有 `conclusion` + 存在的 `experiments`。
5. **exemplar YAML 通过 validator**
   - `python3 tools/check_tree.py` → `OK：2 个 project.yaml 全部符合`（exit 0）。
6. **AGENTS/SCHEMA/README 与新结构一致、无矛盾（生成式 wiki 条款已处置并指向新规范）**
   - `AGENTS.md` §2.1 文档地图加入 `structure.md`/`research-schema.md`/`project.yaml`，§2.3 标注
     `catalog/INDEX/build_wiki/...` 退役，§8 命令改为 `summary.py`/`check_tree.py`；`SCHEMA.md` 顶部迁移通告
     + §5 新命令；`README.md` 的「是什么 / 怎么读 / 轴记号」改为新树与新工具。
7. **工单记录完整、可审计**
   - `.scratch/56-restructure/TICKET.md` §5：L1 逐字、决策与理由、改动文件、命令与结果、剩余未决（含时间戳）。
8. **变更已用显式路径提交**
   - 见 §4（提交哈希）。

## 3. 未决项（框架性取舍，已给默认，未自决）

1. **pareto「100% 全量标准」的定义**：默认以 `7-references` 的 stock 全因果基线为 100%
   （见 `1.1.1` exemplar 的 `validation`）；如出资人要另一参照系，请裁决。
2. **轴 lead / reviewer / integrator 实名**：role_map 骨架已建、占位「待指派」，M2 派发时填。
3. **`catalog.yaml` 物理归档时机**：默认 M2 全轴迁完后移入 `archive/wiki-legacy/`（现在保留作迁移内容源）。
4. **`2.1.3`/`2.2.5` 的 Backlog 身份**：默认保留研究项形式（`status: 未开始`），不删。
5. **工作树既有未提交改动**（`catalog.yaml`/`INDEX.md`/`axes/C-vertical/**`）：已保留不回滚；
   内容由轴 2 worker 在迁 `2.1.1` 时照准迁入，随后随旧格退役（详见 `MAPPING.md` §6）。

## 4. 给轴 worker 的下一步

1. 读 `docs/structure.md` §12（你的轴的节点清单）+ §13（粒度）+ `docs/research-schema.md`（字段契约）。
2. 照抄样板 `axes/1-time/1.1.1-hard-window/project.yaml`（研究项）、
   `axes/9-backlog/9.1-selector-scheduling/project.yaml`（Backlog）。
3. 按 `MAPPING.md` §1/§3 把旧格证据**整目录搬**到对应新家的 `experiments/`（拆并格只落一个主家）。
4. 迁移旧结论时按 55-review 的问题台账**修正**（目的、判据、口径、证据锚点），不照抄。
5. 自检：`python3 tools/summary.py --check`（必须全绿）；交付后另派 fresh 独立复核（`SCHEMA.md` §14/§14b）。

## 5. 命令与结果

- `python3 tools/check_tree.py` → exit 0，`OK：2 个 project.yaml 全部符合`。
- `python3 tools/summary.py` → 打印全表（2 行）。
- `python3 tools/check_schema.py` → 转调 check_tree，exit 0。
- `python3 -m pytest tests/ -q` → **827 passed**（0 失败）。
- `find axes -type f | wc -l` = **2434**（迁移前基线，供 M2 对比）。
