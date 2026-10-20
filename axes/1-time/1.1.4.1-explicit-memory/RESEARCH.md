# 1.1.4.1 显式记忆槽 / 线性状态 / 递归状态 · 研究契约

> 层：**研究级**（研究项 `1.1.4.1`，轴 1「时间轴」）。本文件只写这项研究**要做什么、怎么验收、谁做、证据在哪**；
> 判词、读数、代价事实、缺口清单的家是 `project.yaml`（本项 SSOT），本文件不重述数字、不新增事实。
> 格式契约见 `harness/research-schema.md`；目录布局见 `docs/structure.md`；研究目录见 `docs/catalog.md`。

## 1 目标（一句话）

把「显式状态折算成每位置 KV 字节后，这张价签值不值」判到底：显式记忆槽 / 线性状态 / 递归状态
（delta 规则族）在**同一每位置部署字节**上，是否相对同预算的纯窗口（含它的量化档）改善质量-代价前沿，
并且这个改善在多长的上下文里撑得住。

## 2 目的（为什么做）

普通 decoder-only transformer 的长上下文账单按序列长度线性增长。这条路线想让**远处已看过的信息
被主动写进一个与序列长度无关的固定状态**，从而把前缀真正丢掉、把账单压成常数，同时让长程质量与
可检索性尽量贴近全量 attention。它对准的痛点是「上下文长度的边际成本」，不是「再加一个模块顺便涨点分」；
所以本项唯一的成败判据是**前沿位置**（每位置字节换多少质量），不是臂间增量。

业界已证 delta / 线性读这一族（Gated DeltaNet / GLA / Mamba-2 / SSD）；本项的增量在读数的**受控性与同图性**：
真实文本 clean bpc 的单因素读数、把状态放进与窗口 / 量化同一条字节轴的对照、以及容量否证。

## 3 唯一里程碑

**「显式状态点在（每位置部署字节, 质量 [+ 可检索性]）图上，不被同预算最强基线支配，且该读数的网格边界写明」。**

验收（可观测，逐条对应 `project.yaml::milestone.acceptance` 与 `work_packages`）：

| # | 验收 | 状态 |
|---|---|---|
| 1 | 至少一个预算点（393,216 B）上，显式状态点与**同字节**纯窗口直接点比（不插值），差距与其判据带一并报出 | 已达成（`2639`） |
| 2 | 该点的可检索性（质量之外的第二轴）有读数，且相对同窗口无通道臂的差及其 CI 报出 | 已达成（`1.1.4.5` 的 `2661` 对撞） |
| 3 | 「不被支配」的结论带**明确网格限定**（列出在这张网格上、禁止外推到未测档） | 已达成（`2680` 报网格缺口；`2681` 把网格扩到 int8 下 k=60/k=64） |
| 4 | 量化档（int8 块标尺）与前沿同图；int4 档、carry 族 int8 档仍未叠 | 部分达成（状态以 `project.yaml::conclusion` 的「量化档叠图」段为唯一权威） |
| 5 | 与真实 DeltaNet / GLA 实现对齐 | 未做（本仓无外部实现，F4） |

## 4 工作包

owner 一律是轴 1 lead 派发的 worker，另配 fresh 复核者（`harness/SCHEMA.md` §14/§14b）；执行方不自评。

| 工作包 | 它要交付什么 | 验收 | 预算 | 状态 |
|---|---|---|---|---|
| **WP1 把状态放进质量-代价前沿** | 每位置等效字节 × clean bpc（+ 一个点的召回轴）的前沿图；显式状态点与同字节纯窗口直接点比 | 见 §3 验收 1–3 | 已被 `2638`/`2639`/`2680`/`2681` 消化；残余见 `gaps` | 已完成（残余项已转成 `gaps` 里的独立最小实验） |
| **WP2 完整机制（逐层独立状态 + 内容写门 + 可选 γ）** | 把被单共享槽降级的机制修回与业界同量级，并给出质量读数 | 实现不发散 + 有质量读数；与业界基线对照未做 ⇒ 仍未完全达标 | 引擎改动已交付（E1），质量读数已交付（`2639`）；余项 = 归因拆分（2 臂 ≈60 min GPU） | 进行中 |
| **WP3 撤销两处错误归因并重设任务需求** | 结论表述与证据强度匹配；任务侧「必须丢掉前缀才有区分力」作为判据前置写明 | 结论不再引用零样本 OOD 与「+0.2% 说明有用」这两处误读 | 写作 + 现有证据重读，无 GPU | 未开始 |

## 5 证据指针

- **本项 SSOT**：`axes/1-time/1.1.4.1-explicit-memory/project.yaml`（判词 / 缺口 / 代价行 / 实验清单）。
- **前沿与量化叠图**：`axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2680-quant-overlay-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2681-int8-equalbytes-frontier-13m/analysis.md`。
- **召回轴**：`axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/analysis.md`。
- **机制实现（引擎侧）**：`.scratch/67-engine-changes/E1-CARRYCHANNEL.md`、`.scratch/67-engine-changes/REVIEW-E1.md`。
- **实验索引**：`axes/1-time/1.1.4.1-explicit-memory/experiments/README.md`；权重清单见各实验目录 `MODEL.md`（权重不入库）。
- **复现三态**：`atlas/repro.md`（生成物）；代价事实：`atlas/cost-table.md`（生成物，源在 `project.yaml::cost_rows`）。

## 6 检查状态（机检与 pytest）

本项自己的机检口径与全树一致，由研究级 Agent 自己跑：

| 检查 | 结果 | 说明 |
|---|---|---|
| `python3 tools/summary.py --check` | 本项 0 违规 | 46 个 `project.yaml` 全部符合 `harness/research-schema.md`；本项字段 / 证据锚点 / Δ 方向均齐 |
| `python3 tools/check_tree.py` | 本项 0 违规 | 预登记绑定：本项 `2680` / `2681` 的读数晚于绑定、分提交有先后 |
| `python3 tools/check_docs.py --check` | 红处按归属判由谁改 | 引用可解析 + 三层索引完整性。本单元只修自己目录下的引用；元级文档与旧路径的归属见 `docs/ledger.md` 的台账 |
| `python3 tools/build_cost_table.py --check` | 绿 | 生成物与 `project.yaml::cost_rows` 逐字节一致 |
| `python3 tools/inventory_runs.py --check` | 绿 | 复现清单每条路径存在 |
| `python3 -m pytest tests/ -q` | 绿，除 6 条非本单元失败 | 预存失败 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`（`axes/6-predict/**` 四个 `mutate.py` 未接框架，已登记）；其余 5 条都是 `tests/test_docs_refs.py` 对上述 14 处悬空引用的转述，所有者同上 |

## 7 边界（non-goals）

- **不做组合实验**：carry × depth、carry × 量化、carry × GQA × 窗口都超出单轴单格（`harness/SCHEMA.md` §4）。
- **不做「形态税」判据**：只报前沿位置与支配关系，不把「显式状态点相对同预算纯窗口的质量差」写成结论（62-F3 裁决）。
- **不外推到真实长上下文**：状态是固定成本，T ≫ 256 的摊平优势未测，结论的适用范围永远是
  enwik8 / T=256 / 13M / 12k 步这一格。
- **不写研究级读数到元级文档**：本文件的数字一律以指针形式给出，数的家是 `project.yaml` 与各 `analysis.md`。
