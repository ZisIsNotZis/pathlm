# 研究级跨格汇总（atlas）

研究级里**跨格**的那一半：从本仓库已落盘的研究项（清单见生成物 `docs/catalog.md`——它是
`axes/**/project.yaml` 的逐行投影）里汇总出来的事实。
逐格事实（`project.yaml` / `experiments/**`）在 `axes/**`；元级（这是什么元仓、层契约、计划、
迁移台账）在仓库根与 `docs/`；工级（格式契约、测量纪律、引擎能力面）在 `harness/`。

本目录是**唯一权威**：每一条跨格事实只在这里有一个家，别处只放指针。

## 索引

| 文件 | 它拥有什么事实 | 生成物 |
|---|---|---|
| `README.md` | 本索引：研究级跨格汇总有哪几份、各自拥有什么 | — |
| `playbook.md` | 已测定律的正文与适用边界（定律 1–11 的唯一家） | — |
| `cost-table.md` | 定律 ↔ 证据 ↔ 观测代价 ↔ 边界的对照表；数字逐位照抄出处，不重算 | **是**：`tools/build_cost_table.py` |
| `repro.md` | 每个实验的 run / 已入库产物 / 重跑命令 / 声明墙钟 / **能否字节级复现**（三态带依据） | **是**：`tools/inventory_runs.py` |
| `triage.md` | 研究优先级：逐格「能否改变判断」的排序判据、打分与批次草案 | — |
| `design.md` | 核心设计（RAP）契约、共享底座与仓库布局 | — |
| `paper-plan.md` | 把定律写成一篇论文的 claim / 结构 / 验收（计划，不是结果；零读数） | — |
| `FINDINGS.md` | 已确立结论的索引（路由器：结论 + 出处 + 状态，不持有数字） | — |

改动规矩：

- 两份生成物**禁止手改**：改事实请改 `axes/**/project.yaml` 后重跑生成器；`--check` 与手改副本逐字节比对。
- 本目录的文档不复制 `axes/**` 的读数，只放指针（行号式引用只留给具体叙事行，指向字段用键锚点）。

## 边界

- 不拥有**逐项结论**与实验读数：那是 `axes/**/project.yaml` 与各实验 `analysis.md` / `results.json`。
- 不拥有**格式契约与测量纪律**：`harness/SCHEMA.md`、`harness/research-schema.md`、
  `harness/methodology.md`、`harness/engine_capability.md`。
- 不拥有**优先级从哪来**以外的计划事实：里程碑与工作包在 `docs/plan.md`。
