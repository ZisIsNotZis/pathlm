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

验收（可观测，逐条对应 `project.yaml::milestone.acceptance` 与 `work_packages`）。**状态只有三种：PASS（已成立 + 证据指针）/ 部分 / 阻塞**：

| # | 验收 | 状态 |
|---|---|---|
| 1 | 至少一个预算点（393,216 B）上，显式状态点与**同字节**纯窗口直接点比（不插值），差距与其判据带一并报出 | **PASS**（`2639`）：`experiments/2639-carry-perlayer-frontier-13m/results.json::key_readings.pl_minus_win32` 逐字段 `bpc_carrypl16_s0` 1.527595 / `bpc_win32_s0` 1.595823 / `same_bytes` true / `bytes` 393,216 / 差 +0.068228（41 σ_cell、4.9× 0.014 带）；同文件 `frontier.dominated_by.carrypl16_s0` = 空，`frontier.lower_envelope` = win16→win17→win18→win24→win32→**carrypl16_s0**→win128→full；`location.carrypl16_s0.recovery_ratio_r` = 0.8014（补回窗口 16→全量差距的 80.1%）；换评测窗口序（seed 7）`robustness_alt_window_seed.pl_minus_win32_alt_seed` = 0.071912 同号同量级 |
| 2 | 该点的可检索性（质量之外的第二轴）有读数，且相对同窗口无通道臂的差及其 CI 报出 | **PASS**（`1.1.4.5` 的 `2661`）：`experiments/2661-implicit-vs-explicit-recall-13m/results.json::needle_leg.primary_hard_subset`（同索引硬子集 n=2699，`carrypl16_s0` 276 hits = 10.226%，Wilson [0.091389, 0.114262]；对 `win16_s0` +9.522pp，两比例差 CI [8.336, 10.708]）；同文件 `far_source_probe.carrypl16_s0.delta_bpc_far` = **+0.032965**，逐簇配对 CI [0.026986, 0.038727]（γ=0.99 版只剩 +0.007906、CI [0.00638, 0.009463]）；**并已扩到整条网格**：`experiments/2684-recall-across-budget-points-13m/results.json::per_arm`（固定远端集 n=1206，完备机制 6.4677%、九个纯窗臂 0.33%~0.91%，`contrasts` 九条差 +5.556~+6.136pp 且 CI 全不含 0） |
| 3 | 「不被支配」的结论带**明确网格限定**（列出在这张网格上、禁止外推到未测档） | **PASS**（`2680` 报网格缺口；`2681` 把网格扩到 int8 下 k=60/k=64）：`2681-int8-equalbytes-frontier-13m/results.json::frontier.points`（22 点 / 13 臂）+ `frontier.dominating_points` = 空；最弱的同字节挑战者 `win60_s0@int8` 391,680 B（比 carry 少 1,536 B）/ 1.558523，差 **+0.030928**（18.6 σ_cell、2.2× 0.014 带）；多给纯窗 6.25% 字节（`win64_s0@int8` 417,792 B）仍差 +0.026280；禁外推面写在 `experiments/2681-int8-equalbytes-frontier-13m/analysis.md` §3 与 `experiments/2681-int8-equalbytes-frontier-13m/README.md` §9（carry 族 int8 档、int4 档、T ≫ 256） |
| 4 | 量化档（int8 块标尺）与前沿同图；int4 档、carry 族 int8 档仍未叠 | **部分**：int8 档已叠两轮（`2680`：7 条纯窗臂 Δ +0.000010 ~ +0.000050，字节精确 ×0.53125；`2681`：两条新臂 Δ +0.000079 / +0.000066，CI95 不含 0 但 ≪ 0.014 带）；**int4 档未叠**，**carry 族 int8 档未叠**（`CarryChannelBlock` 不在量化前向的支持范围 ⇒ 要给 `CarryChannelBlock` 写量化前向，属新代码 > 1 个开关）。状态与读数以 `project.yaml::conclusion` 的「量化档叠图」段为唯一权威 |
| 5 | 与真实 DeltaNet / GLA 实现对齐 | **阻塞**：本仓无外部实现（`project.yaml::gaps` 第三条 F4），要补就得把外部代码 / 依赖引入本仓或新环境——那是依赖变更与外部执行，**需上一层裁决**；本轮未自选 |
| 6 | `milestone.acceptance` 的字面第 1 条：前沿图上**每个**预算点有质量与可检索性两轴数字 | **PASS**（`2684`）：九个窗口预算点（16/17/18/24/32/60/64/128）+ 四个 carry 臂在**同一组**对每个臂都在窗外的固定子集上都有召回数字——`experiments/2684-recall-across-budget-points-13m/results.json::per_arm`（n=1206；完备机制 78 hits = 6.4677%、Wilson [5.21%, 8.00%]；纯窗臂 4–11 hits = 0.33%~0.91%），`contrasts` 九条差 +5.556~+6.136pp（pp = 百分点）且 95% CI 全不含 0，退化实现（单共享槽）0.83%/0.91% ⇒ 召回轴可归因到机制完备度。质量轴各自的家仍是 `2638` / `2639` / `2681`（本实验 0 个 bpc 读数） |

**结论强度边界**：验收 1–3 都成立，但它们共同限定在 enwik8 / T=256 / 13M / 12k 步 / seed 0 这一格；
「不被支配」只在这张（含 int8 同字节格的）网格上成立，不外推到 carry 族自己的 int8 档、int4 档与 T ≫ 256。

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
- **召回轴**：`axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/analysis.md`（一个预算点）与 `axes/1-time/1.1.4.1-explicit-memory/experiments/2684-recall-across-budget-points-13m/analysis.md`（整条窗口网格）。
- **机制实现（引擎侧）**：`.scratch/67-engine-changes/E1-CARRYCHANNEL.md`、`.scratch/67-engine-changes/REVIEW-E1.md`。
- **实验索引**：`axes/1-time/1.1.4.1-explicit-memory/experiments/README.md`；权重清单见各实验目录 `MODEL.md`（权重不入库）。
- **复现三态**：`atlas/repro.md`（生成物）；代价事实：`atlas/cost-table.md`（生成物，源在 `project.yaml::cost_rows`）。

## 6 检查状态（机检与 pytest）

本项自己的机检口径与全树一致，由研究级 Agent 自己跑。下表是**本项**这一轮实测的结果，
命令与结果一并写出，别人可原样重跑：

| 检查 | 结果 | 说明 |
|---|---|---|
| `python3 tools/summary.py --check` | 本项 0 违规（rc=0） | 46 个 `project.yaml` 全部符合 `harness/research-schema.md`；本项字段 / 证据锚点 / Δ 方向均齐（`2684` 的 11 个 Δ 读数已在 `results.json::delta_directions` 逐字段声明） |
| `python3 tools/check_tree.py` | 0 违规（rc=0） | 预登记绑定：已绑定 11（读数晚于绑定 8、定序不可判 3）、unbound 92（历史产物，只计数）、缺绑定 0。本项的 `2684` 属「读数晚于绑定」：最终绑定 `bound_at` 2026-10-21T07:55:54 早于最后一次读数的落盘，且那次读数与重跑前逐位相同（`analysis.md` §4/§6 记全过程） |
| `python3 tools/check_docs.py --check` | 全树 0 处违规（472 个文档 / 8899 处引用） | 本单元目录 0 处；文档地图缺项 0 个 |
| `python3 tools/build_cost_table.py --check` | 绿 | 生成物与 `project.yaml::cost_rows` 逐字节一致（59 条代价事实） |
| `python3 tools/inventory_runs.py --check` | 绿 | 重新生成后 131 个实验 / 449 个 run 与磁盘一致；本项 `2684` 的实验级汇总行已收进（0 run 子目录、`rebuildable_only`），`2681` 的「声明墙钟」列取到 `budget.gpu_wall_budget_min` = 39 |
| `python3 -m pytest tests/ -q` | 见本节末的实测一行 | 全树测试；既往只有一条已登记的预存失败（`test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`），同树内其它研究级 owner 在跑的改动会让 `test_docs_refs` / `test_repro_inventory` 的注入式用例间歇变红（它们断言「注入后必须变红」，而并发写入者的临时文件正好也会让它变红） |
| `python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2681-int8-equalbytes-frontier-13m/test_2681_frontier.py -q` | 6 passed | 本目录单测（纯 CPU）：字节账、k=60 那一格的选取、支配定义与阈值算术 |

## 7 边界（non-goals）

- **不做组合实验**：carry × depth、carry × 量化、carry × GQA × 窗口都超出单轴单格（`harness/SCHEMA.md` §4）。
- **不做「形态税」判据**：只报前沿位置与支配关系，不把「显式状态点相对同预算纯窗口的质量差」写成结论（62-F3 裁决）。
- **不外推到真实长上下文**：状态是固定成本，T ≫ 256 的摊平优势未测，结论的适用范围永远是
  enwik8 / T=256 / 13M / 12k 步这一格。
- **不写研究级读数到元级文档**：本文件的数字一律以指针形式给出，数的家是 `project.yaml` 与各 `analysis.md`。
