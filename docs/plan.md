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
| 2 | **`project.yaml` 是每个研究的 SSOT**（目的 / 创新 / 基线 / 判据 / 结论 / 状态 / 工作包 / 证据列表）——`docs/catalog.md` 的七列就是由它逐字生成的；`axes/**/RESEARCH.md` **只在该研究已有研究级 Agent 接手时创建**（现在 3 个：1.1.1 / 1.1.4.1 / 5.5），写的是 `project.yaml` 做不到的派发口径，不复制读数、不占事实之家。理由：再要求每个研究都补一份 `RESEARCH.md` 就是给同一个事实造第二个家（`@ssot`）；`project.yaml` 与 `experiments/**` 原样留在研究级 |
| 3 | 元级零研究级读数：`docs/**` 与根级事实文档里不出现实验读数（bpc / Δ / 显存 / 墙钟 / 逐格判词）、不出现逐实验目录路径、不出现逐条代价事实与复现三态计数；`tools/check_meta_layer.py --check` 把这类模式判红，白名单只留定义性说明 |
| 4 | `AGENTS.md` 写成**三层契约**（元级 / 研究级 / 工级：各层拥有什么事实、谁可以改、派发契约给什么、什么不许下沉 / 上浮）+ 一份可复制的派发模板；`README.md` 按元级重写（这是什么元仓、目录怎么读、一个研究怎么被派发与验收） |
| 5 | 迁移无损：现有元级文档按归属迁到研究级或工级，留下**迁移台账**（一源文件一行：原文 → 新家 → 或「删除并说明理由」），`tools/check_ledger.py --check` 证明无遗漏、新家存在、删除有理由 |
| 6 | 机检按层分工：元级只查元级事实（目录与研究的对应、目录 / 指针可解析、元级文档不含研究级读数）；研究级检查留在研究级且仍由研究级 Agent 自己跑；原六条机检一条不弱、迁走的检查在新层仍生效 |
| 7 | 全绿：元级检查与测试通过（唯一允许的失败是已登记的预存失败），工作区干净并推送；**例外须逐条点名**：研究级未产出导致的级联红（`check_docs` 与 `test_docs_refs` 的同源红）在 §6 逐条写明 owner；工作区里由具名研究级 owner 在跑的改动不算脏，但每一条都必须能追到那个 owner |

状态：**元级四条机检与测试已落地**（工作包见 §3）。

## 2 分层决定（元级自定，执行方不自选）

### 2.1 三层各拥有什么

| 层 | 拥有的事实 | 谁可以改 | 不拥有 |
|---|---|---|---|
| **元级**（仓库根 + `docs/`） | 这是什么元仓、研究目录、层契约与派发模板、元级里程碑与工作包、迁移台账 | 元级主 Agent 派发的 worker；元级主 Agent 自写本文件 | 任何实验读数、判据、逐格分析、代价事实、复现状态、引擎能力 |
| **研究级**（`axes/**`、`atlas/**`） | 每项研究的事实（`project.yaml`：目的 / 基线 / 判据 / 结论 / 状态 / 工作包 / 证据列表）；已有研究级 Agent 接手的项额外带 `RESEARCH.md`（派发口径，不占事实之家）；跨格汇总（定律正文、代价表、复现清单、排序打分、论文口径、结论索引） | 研究级主 Agent 与它派发的 worker / reviewer；元级只读 | 元级目录与层契约 |
| **工级**（`harness/**`、`tools/**`、`tests/**`） | 格式契约与分析纪律、测量纪律、引擎能力面、机检实现与共享底座 | 底座维护者；研究级与元级只调用 | 任何一项研究的结论 |

### 2.2 目录落位（迁移目标）

```
元级：README.md  AGENTS.md  CONTRIBUTING.md  LICENSE.md
      docs/catalog.md（生成物）  docs/plan.md（本文件）  docs/structure.md（目录 / 命名 / 粒度约定）
      docs/ledger.md（迁移台账）
研究级：axes/<轴>/<格子>/{project.yaml（该项 SSOT）, RESEARCH.md（已有研究级 Agent 接手时才有）, experiments/**}
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
| **WP1 分层搬迁与指针重写** | 新建 `harness/`（`harness/SCHEMA.md` / `research-schema.md` / `methodology.md` / `engine_capability.md` / `README.md`）与 `atlas/`（`playbook.md` / `cost-table.md` / `repro.md` / `triage.md` / `paper-plan.md` / `design.md` / `atlas/FINDINGS.md` / `README.md`）；`STATUS.md` 删除；`docs/` 只留 `catalog.md` / `plan.md` / `structure.md` / `ledger.md`；全仓指针（`*.md` / `tools/**` / `tests/**` / CI）改到新家 | 原六条机检命令在新路径下 0 违规；`pytest` 只剩已登记的预存失败；`grep` 全仓无旧路径残留（`atlas/playbook.md` 这类）且旧文件确实不在原位；生成物生成器写出新家路径 | 60 min | 已完成（搬迁提交 `2d68eb4` 里 11 个文件是 `R100` 纯移动，逐对 blob 哈希相同 = 字节等价，见 `docs/ledger.md` §3；`tools/check_docs.py --check` 的文档地图三份齐全）。**未闭环**：研究级 Agent 仍在产出的实验目录会让 `check_docs` 报悬空引用，逐条归属见 §6 |
| **WP2 研究级 SSOT 与生成目录** | `tools/build_catalog.py`（生成 + `--check`）；`docs/catalog.md`；`tests/test_catalog.py`；`harness/research-schema.md` 的 `RESEARCH.md` 契约节；研究级 Agent 接手的研究项的 `axes/**/RESEARCH.md` | `docs/catalog.md` 行数 = `axes/**/project.yaml` 的个数且恰七列；每条状态与 `project.yaml::status` 一致；指针逐条可解析；手改目录副本（含改一行状态、加一行、删一行）`--check` 必红；测试里至少一条可失败对照 | 60 min | **已完成**：元级一半（生成器 / 目录 / 测试，10 条用例含手改·删行·加行三条反向对照）；研究级一半是 `RESEARCH.md`，它**不是**每个研究的必备件（`project.yaml` 才是 SSOT，见 §1 验收 2），已有研究级 Agent 接手的 3 项各一份，其余不由元级代造（`@absent`：缺失是一种状态） |
| **WP3 元级隔离机检与层契约文档** | `tools/check_meta_layer.py` + `tests/test_meta_layer_isolation.py`（元级文档禁实验读数 / 逐实验目录路径 / 逐条代价事实 / 复现三态计数 / 裸读数小数，白名单只留定义性说明）；`AGENTS.md` 三层契约 + 可复制派发模板；`README.md` 元级重写；`docs/structure.md` 收瘦到「约定」 | 注入一条 `bpc` / 一条实验 run 目录路径 / 一行复现三态计数 / 一个裸读数小数 ⇒ 机检变红，移除后恢复绿；`grep` 元级文档零命中（白名单外）；`AGENTS.md` 有三层表与派发模板；`pytest` 全绿（除预存失败） | 55 min | 已完成（`check_meta_layer.py --check` 零命中；注入探针四条在 `tests/test_meta_layer_isolation.py` 里；扫描面是根级 `*.md`（只排除许可证正文）+ `docs/**`，白名单为空——仓库里没有一条必须留在元级的读数） |
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

- 许可：用户已确认 `LICENSE.md` 维持 AGPL-3.0（CONFIRMED，2026-10 收口时）。研究级
  `atlas/paper-plan.md` 的目标「公开可引用」不受影响——AGPL 只覆盖代码仓库，论文文档可另行授权；
  若将来要给论文单独换许可，那是另一件事，另行裁决。
- 许可已闭环，本节只剩下面这条待裁决。
- `CONTRIBUTING.md` 已纳入元级扫描面（`tools/check_meta_layer.py` 的 `META_GLOBS` 是根级 `*.md`）
  并拿掉了它唯一那条研究级读数（改成指向 `harness/methodology.md` §2）；整份文档要不要按同一口径继续收瘦，
  等下一轮元级裁决。

## 6 本轮未闭环项（逐条点名，归属写清）

**研究级的逐项 `RESEARCH.md`**：它**不是**每个研究的必备件——每个研究的 SSOT 是 `project.yaml`（§1 验收 2），
`RESEARCH.md` 只在该研究已有研究级 Agent 接手时创建。现在 3 份：
`axes/1-time/1.1.1-hard-window/RESEARCH.md`、`axes/1-time/1.1.4.1-explicit-memory/RESEARCH.md`、
`axes/5-quant/5.5-fp8-head-dim/RESEARCH.md`。元级目录不依赖它（七列取自 `project.yaml`），它不挡里程碑。

**`tools/check_docs.py --check` 当前 0 处违规**：此前唯一的一条悬空引用（研究级 `1.1.4.1` 的实验分析未产出）已随该分析落盘而自动转绿，`tests/test_docs_refs.py` 的级联红同期消失。

**这四处上过红名单，但它们不是缺陷**（下一轮复核不要再当红报）：

- `1.1.1` 与 `5.5` 两条：对应实验的分析与 run 产物已落盘，不存在「未出」这个缺口
  （`2679-anchor-eval-ablation-13m/analysis.md`、`2657-fp8-hd128-head-dim/` 的 `analysis.md` 与 `results.json`）。
- `atlas/triage.md` 的 `2657` 键锚点与 `2680` 的 `2681` 键锚点两条：前者在 `config.json` 顶层键上真实存在，
  后者是点号键路径 `rows.win32_s0.bpc_fp16`，按结构解析通过（上游实验落盘的 `results.json` 里 `win32_s0` 与其内 `bpc_fp16` 真实存在），
  不是通配锚点，两条从未是违规。

**`1.1.4.1` 的让位说明曾带一个仓外路径，已改成不带路径的写法**：那句话说的是**别的项目**在卡上跑的
SFT 训练作业（不在本仓），不是本仓「未产出的证据」，因此它永远不会转绿；修法是改写句子而**不是**
给 `check_docs.SKIP_PREFIXES` 加白名单——加一条永久白名单等于让这条禁令对那类路径永久失效。

**`pytest` 剩下的逐条归属**（`python3 -m pytest tests/ -q` → 964 passed / 6 failed）：

- `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`：**已登记的预存失败**
  （本仓部分 mutate 脚本还没接 harness，不属元级收身范围）；原始登记在日志工单
  `.scratch/63-axis6-predict/NEXT-WP-mutation-harness.md`（日志，不是结论依据，只是找到那一条的地方）。
- 剩下 5 条全是同一条研究级未产出的级联，非元级引入：
  `test_all_doc_references_resolve` 就是上面那条悬空引用；
  `test_dangling_reference_turns_red` 与 3 条 `test_unmapped_doc_turns_red[...]` 的探针本身都判红了，
  死在收尾的「删掉探针后必须全绿」——而全树现在有一条真违规，所以它们只能红。
  那一份分析一落盘，这 5 条同时转绿。
- `tests/test_repro_inventory.py` 的 2 条**已转绿**：当时是 `atlas/repro.md` 落后于磁盘（2681 的新臂落盘后
  没重跑 `tools/inventory_runs.py`），提交 `6a657cc` 按磁盘重生成后一致；现在连两条「临时副本」对照都通过
  （`python3 -m pytest tests/test_repro_inventory.py -q` → 6 passed），所以不是「副本的入库集合与真树不一致」——
  那一种重跑仍会红。
