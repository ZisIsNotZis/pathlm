# 计划（元级 · 唯一里程碑与工作包）

> 本文件是**元级计划的唯一权威**：唯一里程碑、工作包（owner / 交付物 / 验收 / 预算 / 状态）与
> 元级证据指针写在这里。元级只持有**研究目录与契约**，不持有任何研究级事实——实验读数、判据推导、
> 逐格分析、代价事实、复现三态计数都在研究级（`axes/**`、`atlas/**`）与工级（`harness/**`）。
> 分层归属见 `AGENTS.md`；研究目录见 `docs/catalog.md`；迁移台账见 `docs/ledger.md`。

## 0 这一版计划在治什么

**类：层混淆**——元级文档里写着研究级细节（实验读数、判据推导、逐格 analysis、逐条代价事实、复现清单、
分诊打分），元级主 Agent 读状态要穿过几百行读数，才能判断「哪条研究该被派发、它现在到哪一步」。
治法是**把层切开**，不是把细节写短：细节整体下沉到它本来就属于的那一层，元级只留一张生成的目录。

判据从「能不能机检」出发：能生成的就生成（目录），能机检的就机检（元级零读数、引用可解析、
台账无遗漏），只有机检不了的才写成规矩。

## 1 里程碑

**唯一里程碑 = 七条验收全部满足**（「完成」不是别的意思）：

| # | 验收（可观测） |
|---|---|
| 1 | `docs/catalog.md` 是**生成物**，一行一个研究，只有七列：id / 标题 / 目的 / 验收 / 期望产出 / 状态 / 指针；行数 = 现有研究项数；状态与各研究自己的 SSOT 一致；指针可解析；`tools/build_catalog.py --check` 手改即红 |
| 2 | 每个研究目录下有 `RESEARCH.md`（目标 / 目的 / **唯一里程碑** / 工作包（owner / 验收 / 预算 / 状态）/ 证据指针），由**研究级** Agent 按 `harness/research-schema.md` 的契约写；`docs/catalog.md` 的七列取自 `project.yaml`（研究项事实的家），不由 `RESEARCH.md` 转发；`project.yaml` 与 `experiments/**` 原样留在研究级 |
| 3 | 元级零研究级读数：`docs/**` 与根级事实文档里不出现实验读数（bpc / Δ / 显存 / 墙钟 / 逐格判词）、不出现逐实验目录路径、不出现逐条代价事实与复现三态计数；`tools/check_meta_layer.py --check` 把这类模式判红，白名单只留定义性说明 |
| 4 | `AGENTS.md` 写成**三层契约**（元级 / 研究级 / 工级：各层拥有什么事实、谁可以改、派发契约给什么、什么不许下沉 / 上浮）+ 一份可复制的派发模板；`README.md` 按元级重写（这是什么元仓、目录怎么读、一个研究怎么被派发与验收） |
| 5 | 迁移无损：现有元级文档按归属迁到研究级或工级，留下**迁移台账**（一源文件一行：原文 → 新家 → 或「删除并说明理由」），`tools/check_ledger.py --check` 证明无遗漏、新家存在、删除有理由 |
| 6 | 机检按层分工：元级只查元级事实（目录与研究的对应、目录 / 指针可解析、元级文档不含研究级读数）；研究级检查留在研究级且仍由研究级 Agent 自己跑；原六条机检一条不弱、迁走的检查在新层仍生效 |
| 7 | 全绿：元级检查与测试通过（唯一允许的失败是已登记的预存失败），工作区干净并推送 |

状态：**元级四条机检与测试已落地**（工作包见 §3）。

## 2 分层决定（元级自定，执行方不自选）

### 2.1 三层各拥有什么

| 层 | 拥有的事实 | 谁可以改 | 不拥有 |
|---|---|---|---|
| **元级**（仓库根 + `docs/`） | 这是什么元仓、研究目录、层契约与派发模板、元级里程碑与工作包、迁移台账 | 元级主 Agent 派发的 worker；元级主 Agent 自写本文件 | 任何实验读数、判据、逐格分析、代价事实、复现状态、引擎能力 |
| **研究级**（`axes/**`、`atlas/**`） | 每项研究的目标 / 目的 / 唯一里程碑 / 工作包 / 证据指针（`axes/**/RESEARCH.md`）；研究项事实（`project.yaml`）；跨格汇总（定律正文、代价表、复现清单、排序打分、论文口径、结论索引） | 研究级主 Agent 与它派发的 worker / reviewer；元级只读 | 元级目录与层契约 |
| **工级**（`harness/**`、`tools/**`、`tests/**`） | 格式契约与分析纪律、测量纪律、引擎能力面、机检实现与共享底座 | 底座维护者；研究级与元级只调用 | 任何一项研究的结论 |

### 2.2 目录落位（迁移目标）

```
元级：README.md  AGENTS.md  CONTRIBUTING.md  LICENSE.md
      docs/catalog.md（生成物）  docs/plan.md（本文件）  docs/structure.md（目录 / 命名 / 粒度约定）
      docs/ledger.md（迁移台账）
研究级：axes/<轴>/<格子>/{RESEARCH.md, project.yaml, experiments/**}
        atlas/{README.md, playbook.md, cost-table.md*, repro.md*, triage.md, paper-plan.md,
               design.md, atlas/FINDINGS.md}       （* = 生成物）
工级：  harness/{README.md, harness/SCHEMA.md, research-schema.md, methodology.md, engine_capability.md}
        tools/**  tests/**
```

### 2.3 机检按层分工

- **元级**：`tools/build_catalog.py --check`（目录与研究的对应）、`tools/check_meta_layer.py --check`
  （元级文档不含研究级读数）、`tools/check_ledger.py --check`（迁移台账）、`tools/check_docs.py --check`
  （引用可解析 + 每层索引完整性）。
- **研究级**（由研究级 Agent 自己跑，元级不代跑）：`tools/summary.py --check` /
  `tools/check_tree.py`、`tools/build_cost_table.py --check`、`tools/inventory_runs.py --check`、
  `python3 -m pytest tests/ -q`。
- 元级 CI 只跑元级那几条；研究级检查在迁到新家之后仍以同样口径生效，不降级。

## 3 工作包

owner 一律是元级主 Agent 派发的 worker；每个包另配一名 fresh reviewer 复核（执行方不自评）。

| 工作包 | 交付物 | 验收（可观测） | 预算 | 状态 |
|---|---|---|---|---|
| **WP1 分层搬迁与指针重写** | 新建 `harness/`（`harness/SCHEMA.md` / `research-schema.md` / `methodology.md` / `engine_capability.md` / `README.md`）与 `atlas/`（`playbook.md` / `cost-table.md` / `repro.md` / `triage.md` / `paper-plan.md` / `design.md` / `atlas/FINDINGS.md` / `README.md`）；`STATUS.md` 删除；`docs/` 只留 `catalog.md` / `plan.md` / `structure.md` / `ledger.md`；全仓指针（`*.md` / `tools/**` / `tests/**` / CI）改到新家 | 原六条机检命令在新路径下 0 违规；`pytest` 只剩已登记的预存失败；`grep` 全仓无旧路径残留（`atlas/playbook.md` 这类）且旧文件确实不在原位；生成物生成器写出新家路径 | 60 min | 已完成（搬迁提交 `2d68eb4` 里 12 个文件是 `R100` 纯移动；`tools/check_docs.py --check` 的文档地图三份齐全）。**未闭环**：研究级 Agent 仍在产出的实验目录会让 `check_docs` 报悬空引用，逐条归属见 §6 |
| **WP2 研究级 SSOT 与生成目录** | `tools/build_catalog.py`（生成 + `--check`）；`docs/catalog.md`；`tests/test_catalog.py`；`harness/research-schema.md` 的 `RESEARCH.md` 契约节；46 × `axes/**/RESEARCH.md`（研究级 Agent 写） | `docs/catalog.md` 行数 = 46 且恰七列；每条状态与 `project.yaml::status` 一致；指针逐条可解析；手改目录副本（含改一行状态、加一行、删一行）`--check` 必红；测试里至少一条可失败对照 | 60 min | **元级一半已完成**（生成器 / 目录 / 测试，10 条用例含手改·删行·加行三条反向对照）；**研究级一半**（逐项 `RESEARCH.md`）owner = 研究级 Agent，已落盘的研究项各一份，未落盘的不由元级代造（`@absent`：缺失是一种状态） |
| **WP3 元级隔离机检与层契约文档** | `tools/check_meta_layer.py` + `tests/test_meta_layer_isolation.py`（元级文档禁实验读数 / 逐实验目录路径 / 逐条代价事实 / 复现三态计数，白名单只留定义性说明）；`AGENTS.md` 三层契约 + 可复制派发模板；`README.md` 元级重写；`docs/structure.md` 收瘦到「约定」 | 注入一条 `bpc` / 一条实验 run 目录路径 ⇒ 机检变红，移除后恢复绿；`grep` 元级文档零命中（白名单外）；`AGENTS.md` 有三层表与派发模板；`pytest` 全绿（除预存失败） | 55 min | 已完成（`check_meta_layer.py --check` 零命中；注入探针两条在 `tests/test_meta_layer_isolation.py` 里；白名单为空——仓库里没有一条必须留在元级的读数） |
| **WP4 迁移台账与无损证明** | `docs/ledger.md`（一源文件一行：原文 → 新家 → 或「删除并说明理由」，含与旧规则冲突的处置）；`tools/check_ledger.py` + `tests/test_ledger.py` | 台账覆盖迁移前时点全部元级文档（`docs/**` + 根级事实文档）与本次全部新增文档；每行新家存在（或删除理由非空且新家为空）；删掉一行 / 指向不存在的新家 ⇒ `--check` 必红 | 35 min | 已完成（18 个原文：17 迁到新家 + 1 删除带理由；本次新增 10 个各有登记；`tests/test_ledger.py` 覆盖漏行·悬空新家·无理由删除·未登记新增四条反向对照） |
| **WP5 最终 fresh 复核** | 逐条复核 §1 七条验收的可观测证据 | 每条给 PASS / FAIL + 复现命令；FAIL 项必须点名文件与行 | 30 min | todo |

## 4 证据指针

- 元级机检（元级自己的红线）：
  ```bash
  python3 tools/build_catalog.py --check
  python3 tools/check_meta_layer.py --check
  python3 tools/check_ledger.py --check
  python3 tools/check_docs.py --check
  ```
- 研究级机检（迁走后仍生效，由研究级 Agent 跑）：`tools/summary.py --check` / `tools/check_tree.py` /
  `tools/build_cost_table.py --check` / `tools/inventory_runs.py --check` / `python3 -m pytest tests/ -q`。
- 目录与台账：`docs/catalog.md`、`docs/ledger.md`。
- 每条工作包的原始证据（命令输出、注入探针、复核结论）落在提交信息与该提交的 diff 里，
  本文件只留指针，不转述读数。

## 5 需要元级裁决（未裁决前按默认走）

- 许可（`LICENSE.md` = AGPL-3.0，而研究级 `atlas/paper-plan.md` 目标是公开可引用）仍待用户裁决，
  与本次分层无关，不挡里程碑。
- `CONTRIBUTING.md` 不在本轮元级重写的写域内，所以它仍带着研究级细节；要不要按同一口径收瘦，
  等下一轮元级裁决。

## 6 本轮未闭环项（逐条点名，归属写清）

**研究级的逐项 `RESEARCH.md`**：`axes/1-time/1.1.1-hard-window/RESEARCH.md` 与
`axes/1-time/1.1.4.1-explicit-memory/RESEARCH.md` 已由研究级 Agent 落盘，其余研究项尚未写。
元级目录不依赖它（七列取自 `project.yaml`），所以这一项不挡元级里程碑；owner = 研究级 Agent。

**悬空引用只来自研究级尚未产出的证据路径**（`tools/check_docs.py --check` 逐条列出，元级无法代造）：

- `axes/1-time/1.1.1-hard-window/RESEARCH.md` → 本轮的锚点召回实验目录（分析未出）
- `axes/1-time/1.1.4.1-explicit-memory/RESEARCH.md` 与同项实验的 `README.md` → 本轮的 int8 前沿实验（分析与上游键锚点未出）
- `axes/5-quant/5.5-fp8-head-dim/**` → 本轮 `head_dim=128` 腿的 run 产物（未出）

这些路径的存在与否由那三个研究级单元决定；它们一落盘，`check_docs` 即自动转绿。
`tests/test_docs_refs.py::test_all_doc_references_resolve` 的失败就是上表的三项，非元级引入。
