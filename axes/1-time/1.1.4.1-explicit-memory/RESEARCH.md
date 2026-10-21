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

验收（可观测，逐条对应 `project.yaml::milestone.acceptance` 与 `work_packages`）。**状态只有四种：PASS（已成立 + 证据指针）/ 部分 / 未取得（provisional，转独立项）/ 阻塞**：

| # | 验收 | 状态 |
|---|---|---|
| 1 | 至少一个预算点（393,216 B）上，显式状态点与**同字节**纯窗口直接点比（不插值），差距与其判据带一并报出 | **PASS**（`2639`）：`experiments/2639-carry-perlayer-frontier-13m/results.json::key_readings.pl_minus_win32` 逐字段 `bpc_carrypl16_s0` 1.527595 / `bpc_win32_s0` 1.595823 / `same_bytes` true / `bytes` 393,216 / 差 +0.068228（41 σ_cell、4.9× 0.014 带）；同文件 `frontier.dominated_by.carrypl16_s0` = 空，`frontier.lower_envelope` = win16→win17→win18→win24→win32→**carrypl16_s0**→win128→full；`location.carrypl16_s0.recovery_ratio_r` = 0.8014（补回窗口 16→全量差距的 80.1%）；换评测窗口序（seed 7）`robustness_alt_window_seed.pl_minus_win32_alt_seed` = 0.071912 同号同量级 |
| 2 | 该点的可检索性（质量之外的第二轴）有读数，且相对同窗口无通道臂的差及其 CI 报出 | **PASS**（`1.1.4.5` 的 `2661`）：`experiments/2661-implicit-vs-explicit-recall-13m/results.json::needle_leg.primary_hard_subset`（同索引硬子集 n=2699，`carrypl16_s0` 276 hits = 10.226%，Wilson [0.091389, 0.114262]；对 `win16_s0` +9.522pp，两比例差 CI [8.336, 10.708]）；同文件 `far_source_probe.carrypl16_s0.delta_bpc_far` = **+0.032965**，逐簇配对 CI [0.026986, 0.038727]（γ=0.99 版只剩 +0.007906、CI [0.00638, 0.009463]）；**并已扩到整条网格**：`experiments/2684-recall-across-budget-points-13m/results.json::per_arm`（固定远端集 n=1206，完备机制 6.4677%、九个纯窗臂 0.33%~0.91%，`contrasts` 九条差 +5.556~+6.136pp 且 CI 全不含 0） |
| 3 | 「不被支配」的结论带**明确网格限定**（列出在这张网格上、禁止外推到未测档） | **PASS**（`2680` 报网格缺口；`2681` 把网格扩到 int8 下 k=60/k=64）：`2681-int8-equalbytes-frontier-13m/results.json::frontier.points`（22 点 / 13 臂）+ `frontier.dominating_points` = 空；最弱的同字节挑战者 `win60_s0@int8` 391,680 B（比 carry 少 1,536 B）/ 1.558523，差 **+0.030928**（18.6 σ_cell、2.2× 0.014 带）；多给纯窗 6.25% 字节（`win64_s0@int8` 417,792 B）仍差 +0.026280；禁外推面写在 `experiments/2681-int8-equalbytes-frontier-13m/analysis.md` §3 与 `experiments/2681-int8-equalbytes-frontier-13m/README.md` §9（carry 族 int8 档、int4 档、T ≫ 256） |
| 4 | 量化档（块标尺，G=32）与前沿同图：int8 与 **int4** 两档、纯窗族与 **carry 族**两族都在同一条「每位置部署字节」轴上 | **PASS**（`experiments/2685-quant-tiers-frontier-13m`）：给 `CarryChannelBlock` 新写一条只做推理期、不改训练的量化前向（KV 精度 × 状态精度两轴分开），47 点 = 9 纯窗臂 × {fp16,int8,int4} + 4 carry 臂 × {fp16, 只KV int8, int8 全档, 只KV int4, int4 全档}，0 训练 token。① 两道复现门 `results.json::equality_gate` 13/13（两轴都不量化时对上 `2681` 的 bpc_fp16，1e-6）+ 9/9（纯窗臂 int8 对上 bpc_int8）⇒ 新前向是已入库仪器的忠实复刻；② `byte_ledger` 用 `config.json::analytic` 的 G=32 与位宽解析出 12,288 / 6,528 / 3,456 B/位置/键，int8 = ×0.53125、int4 = ×0.28125；③ carry 的 int8 全档 393,216 → **208,896 B**、bpc 1.527616（只换 KV +0.000021 / 再换状态 −0.000001），同字节纯窗最强点 `win32@int8` 差 **+0.068241**，`dominated_by` = 空；④ carry 的 int4 全档 **110,592 B** / 1.532460（+0.004864，带内），同字节 `win32@int4` 差 **+0.093485**，`dominated_by` 亦为空；⑤ int4 档在纯窗 k=32/60/64/128 上越带（+0.0301 / +0.0371 / +0.0359 / +0.0198），已如实登记为代价行。状态与读数以 `project.yaml::conclusion` 的「量化档四腿齐」段为唯一权威 |
| 5 | 与真实 DeltaNet / GLA 实现对齐 | **未取得——转独立研究项（provisional，待用户确认）**：本仓无外部实现、无检索（`project.yaml::gaps` 的「与真实 DeltaNet/GLA 对齐未做」一条，M-04）；补齐要把外部代码 / 依赖或新环境引入本仓——那是依赖变更 + 外部执行，**须用户裁决**，本轮未自选。裁决选项 A/B/C 与 A 的代价（新依赖、可能要装环境、1–3 小时、许可与来源要记进该实验 `MODEL.md` 与 `docs/ledger.md` 台账）写在同一条 gap 里。本格在不引入外部代码的前提下已自证 = 同字节轴（`2639` / `2681` / `2685`）+ 两条读出轴（质量 clean bpc、召回 `2661` / `2684`） |
| 6 | `milestone.acceptance` 的字面第 1 条：前沿图上**每个**预算点有质量与可检索性两轴数字 | **PASS**（`2684`）：九个窗口预算点（16/17/18/24/32/60/64/128）+ 四个 carry 臂在**同一组**对每个臂都在窗外的固定子集上都有召回数字——`experiments/2684-recall-across-budget-points-13m/results.json::per_arm`（n=1206；完备机制 78 hits = 6.4677%、Wilson [5.21%, 8.00%]；纯窗臂 4–11 hits = 0.33%~0.91%），`contrasts` 九条差 +5.556~+6.136pp（pp = 百分点）且 95% CI 全不含 0，退化实现（单共享槽）0.83%/0.91% ⇒ 召回轴可归因到机制完备度。质量轴各自的家仍是 `2638` / `2639` / `2681`（本实验 0 个 bpc 读数） |

**结论强度边界**：验收 1 / 2 / 3 / 4 / 6 都成立，第 5 条未取得（provisional，转独立项），成立的部分共同限定在 enwik8 / T=256 / 13M / 12k 步 / seed 0 这一格；
「不被支配」只在这张（含 int8 / int4 两档、两族都在场的 47 点）网格上成立，不外推到 T ≫ 256、G≠32 的标尺方案与多 seed（除 `win16` / `carry16` 各两 seed 外均 n=1）。

## 4 工作包

owner 一律是轴 1 lead 派发的 worker，另配 fresh 复核者（`harness/SCHEMA.md` §14/§14b）；执行方不自评。

| 工作包 | 它要交付什么 | 验收 | 预算 | 状态 |
|---|---|---|---|---|
| **WP1 把状态放进质量-代价前沿** | 每位置等效字节 × clean bpc（+ 召回轴）的前沿图；显式状态点与同字节纯窗口直接点比 | 见 §3 验收 1–4 | 已被 `2638`/`2639`/`2680`/`2681`/`2684`/`2685` 消化；残余见 `gaps` | 已完成（残余项已转成 `gaps` 里的独立最小实验） |
| **WP2 完整机制（逐层独立状态 + 内容写门 + 可选 γ）** | 把被单共享槽降级的机制修回与业界同量级，并给出质量读数 | 实现不发散 + 有质量读数；与业界基线（真实 DeltaNet/GLA）对照**未取得——转独立研究项（provisional）**⇒ 机制与读数两半达标，业界对齐这一半不本格做 | 引擎改动已交付（E1），质量读数已交付（`2639`）；余项 = 归因拆分（2 臂 ≈60 min GPU） | 进行中 |
| **WP3 撤销两处错误归因并重设任务需求** | 结论表述与证据强度匹配；任务侧「必须丢掉前缀才有区分力」作为判据前置写明 | 结论不再引用零样本 OOD 与「+0.2% 说明有用」这两处误读 | 写作 + 现有证据重读，无 GPU | 未开始 |

## 5 证据指针

- **本项 SSOT**：`axes/1-time/1.1.4.1-explicit-memory/project.yaml`（判词 / 缺口 / 代价行 / 实验清单）。
- **前沿与量化叠图**：`axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2680-quant-overlay-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2681-int8-equalbytes-frontier-13m/analysis.md`、
  `axes/1-time/1.1.4.1-explicit-memory/experiments/2685-quant-tiers-frontier-13m/analysis.md`（int4 档 + carry 族的 int8/int4 档）。
- **召回轴**：`axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/analysis.md`（一个预算点）与 `axes/1-time/1.1.4.1-explicit-memory/experiments/2684-recall-across-budget-points-13m/analysis.md`（整条窗口网格）。
- **机制实现（引擎侧）**：`.scratch/67-engine-changes/E1-CARRYCHANNEL.md`、`.scratch/67-engine-changes/REVIEW-E1.md`。
- **实验索引**：`axes/1-time/1.1.4.1-explicit-memory/experiments/README.md`；权重清单见各实验目录 `MODEL.md`（权重不入库）。
- **复现三态**：`atlas/repro.md`（生成物）；代价事实：`atlas/cost-table.md`（生成物，源在 `project.yaml::cost_rows`）。

## 6 检查状态（机检与 pytest）

本项自己的机检口径与全树一致，由研究级 Agent 自己跑。下表是**本项**这一轮实测的结果，
命令与结果一并写出，别人可原样重跑：

| 检查 | 结果 | 说明 |
|---|---|---|
| `python3 tools/build_catalog.py --check` | 绿（rc=0） | `docs/catalog.md` 与 46 个研究项的 `project.yaml` 逐字节一致（七列：id / 标题 / 目的 / 验收 / 期望产出 / 状态 / 指针） |
| `python3 tools/check_ledger.py --check` | 绿（rc=0） | 迁移台账 18 个原文（删除 1 个）+ 本次新增 10 个，0 处违规 |
| `python3 tools/check_meta_layer.py --check` | 绿（rc=0） | 元级 6 个文档 0 处研究级读数 |
| `python3 tools/summary.py --check` | 本项 0 违规；全树的红不在本项 | 46 个 `project.yaml` 全部符合 `harness/research-schema.md`；本项字段 / 证据锚点 / Δ 方向均齐（`2684` 的 11 个 Δ 读数已在 `results.json::delta_directions` 逐字段声明）。全树的红都是 `check_docs` 的引用违规，归属见下表与本节末段 |
| `python3 tools/prereg_bind.py --check` | 绿（rc=0，「OK：预登记绑定全部可判」） | 已绑定 14（读数晚于绑定 11、读数早于绑定即定序不可判 3）、unbound 92（历史产物，只计数）、缺绑定 0。本项 `2684`（`bound_at` 2026-10-21T07:55:54+08:00，契约 = `README.md` / `config.json` / `run.py`）与 `2685`（08:11:30+08:00，契约另含同目录 `carry_quant.py`）都在「读数晚于绑定」里，契约文件在绑定后未被改过 |
| `python3 tools/check_docs.py --check`（主工作树） | 本项 0 处；别的格的红按名字列出 | 工作树里有别的格的在飞改动，扫描数与违规数随它们变，所以这一行只记**归属**：`1.2.2` 的 `RESEARCH.md:46` 键锚点 `linear_health` 不存在、`2683` 的 `analysis.md` 指向 `2685` 一个尚未产出的键；`1.1.1` 的 `RESEARCH.md` 曾指向 `2686` 未产出的 `analysis.md`（该格在处理中）。本项目录一次都没进过违规名单 |
| `python3 tools/check_docs.py --check`（干净副本，`git worktree` 于当轮 `HEAD`） | 本项 0 处；全树 1 处 | 干净副本只含已提交事实，所以它的数是可复核的定值：扫描 477 个文档 / 9062 处引用，唯一违规是 `1.2.2` 的 `linear_health` 键锚点 |
| `python3 tools/build_cost_table.py --check` | 干净副本绿；主工作树红 | 干净副本上 62 条代价事实 / 30 个研究项与生成物逐字节一致 ⇒ 本项的代价行在索引里是最新的；主工作树的红来自 `1.2.2` 的 `project.yaml` 未提交改动（生成物滞后 2 行），**不许把未提交事实写进生成物**，所以留给该格的提交 |
| `python3 tools/inventory_runs.py --check` | 主工作树 1 行滞后，属 `1.2.2` | 干净副本上重生**只**动 `1.2.2` 的 `2685-recall-flops-linear-13m` 一行（该实验入库后没重生成索引），本项 13 个实验行逐字节相同；本项 `2684` 的实验级汇总行在索引里（0 run 子目录、`rebuildable_only`），`2681` 的「声明墙钟」列取到 `budget.gpu_wall_budget_min` = 39 |
| `python3 -m pytest axes/1-time/1.1.4.1-explicit-memory -q` | **54 passed**（4.2 s） | 本项六个实验目录的全部单测，**一次会话**跑完：2620 的 16、2631 的 14、2638 / 2639 / 2681 / 2685 各 6。2638 / 2639 / 2681 的单测原来用 `import run as R` 取本目录的 `run.py`，而同一次 pytest 会话里 `sys.modules` 只保留其中一个 ⇒ 整个目录一起跑时 4 个用例拿到别的实验的 `run` 而红（单个文件跑全绿）；改成按文件路径以唯一模块名加载后，批量跑也全绿 |
| `python3 -m pytest tests/test_carry_*.py tests/test_composable_*.py tests/test_eng1_composable.py tests/test_recursive_carrier_13m.py tests/test_tf4_samecaliber_13m.py tests/test_ood_leg_variance.py tests/test_engine_capability.py -q` | **192 passed**（86 s） | 本项在仓级单测目录 `tests/**` 里的那部分（14 个文件）全绿 |
| `python3 -m pytest tests/ -q` | **未跑完**（900 s 超时被杀） | 机器上同时有别的格的训练在跑（load average ≈21 / 12 核 ⇒ 套件从 215 s 涨到 >900 s），这一轮没拿到完整计数。按文件名点名跑出的红全部属别的格：`test_docs_refs.py` 5 个（4 处 `check_docs` 违规的基线断言 + 三个注入式用例的「注入后必须变红」）、`test_cost_table.py` 2 个（`1.2.2` 的 `project.yaml` 未提交改动）、`test_attribution_direction.py` 1 个（`1.2.2` 的 `attribution.py`） |

**全树红的归属（本项不修，越界）**：`check_docs` 的引用违规与 `test_attribution_direction.py` 的失败分别属 `1.1.1`（`RESEARCH.md` 指向 `2686` 未产出的分析，归该格在处理）与 `1.2.2`（`linear_health` 键锚点、`2683` 的分析指向 `2685` 一个尚未产出的键、`attribution.py`）；主工作树的 `build_cost_table` 红属 `1.2.2` 的未提交 `project.yaml`，`inventory_runs` 的那 1 行滞后属 `1.2.2` 的 `2685-recall-flops-linear-13m` 入库后未重生成索引。

## 7 边界（non-goals）

- **不做组合实验**：carry × depth、carry × 量化、carry × GQA × 窗口都超出单轴单格（`harness/SCHEMA.md` §4）。
- **不做「形态税」判据**：只报前沿位置与支配关系，不把「显式状态点相对同预算纯窗口的质量差」写成结论（62-F3 裁决）。
- **不外推到真实长上下文**：状态是固定成本，T ≫ 256 的摊平优势未测，结论的适用范围永远是
  enwik8 / T=256 / 13M / 12k 步这一格。
- **不写研究级读数到元级文档**：本文件的数字一律以指针形式给出，数的家是 `project.yaml` 与各 `analysis.md`。
