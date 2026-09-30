# 2616 · 有界内存的悬崖——窗口 192 训练（ticket 33 契约 Q）· 13M · dense

> 契约来源：`.scratch/33-batch6/issues/01-batch6.md` §「契约 Q」+ 开头统一纪律段。
> **副本标注**：§3 干预、§4 判据线（①–③）与 §2 动机段**逐字**抄录；其余小节为
> **摘要**（摘要处已注明）。判据线一律未改。
> 入口：`run.py`；训练入口：仓库根 `train_depth_ar.py`（本实验**未改它**，只用既有
> `--train-window/--train-anchors`）；仪器：`tools/evalkit/retention.py`（依赖腿 = 2614 的
> 三条曲线；检索腿 = 契约 M 的针）；单测：`tests/test_depth_ar.py`（本实验新增 3 条，
> 钉在真实尺度 T=256 / window=192）；变异验证：`mutate.py` + `mutation_report.json`；
> 稳定性专报：`stability.md`；原始结果：`results.json`；逐配置日志：`retention_log.jsonl`；
> 数字对账：`reconcile_numbers.py` + `reconcile_report.json`；权重清单：`MODEL.md`。

## 1. 命题

**窗口长度是不是一条带悬崖的杠杆？**——把训练期可见窗口从 32 放到 192，
训练税能不能从 +0.0866 压到 ≈0？三个可分别证伪的子命题：① 质量税的量级；
② 把「评测期套预算」的四个点与两个训练点并列定位（**不得外推内插**）；
③ 窗口放宽后，窗口外的针是否仍然只是「丢了」（而不是被搬到近处 / 被锚点通道救回）。

## 2. 动机（**逐字**抄录契约 Q 的动机段）

> 动机：ticket 32-O 测出「训练成只看最近 32」的税 = **+0.0866**；而同一仪器测「把全注意力模型
> 限制到最近 192」的**评测期**代价只有 **+0.004**、最近 128 是 +0.020、最近 64 是 +0.081。
> 于是产生一个可判定、且**不能靠外推**回答的问题：**窗口长度是不是一条带悬崖的杠杆？
> 把窗口放到 192，训练税能不能压到 ≈0？**

「同一仪器」的含义（摘要）：两条腿都跑在**同一份**评测窗口序（`window_seed=2` + 24×16，
与 2614/2615 一致）、同一份针构造（契约 M 的 needle v3，`seed=3` + 32×16）上；
删除动作一律是**改写注意力可见集**（`keep[i,j]=False`），不是事后对 logits 做手脚。

## 3. 干预（**逐字**抄录契约 Q 的干预段）

> - 干预（唯一变量 = 训练期窗口长度）：`window192`（只看最近 192 + 8 锚点）vs
>   `full`（复用 2614/32-O 的全注意力基线，给配置等值性逐字段对照）。
>   其余逐字段一致（`--dense`、seed=0、12k 步、bs32、seq256）。

**实现（摘要）**：掩码 `keep[i,j] = (j ≤ i) ∧ ((i−j) < 192 ∨ j < 8)`，对角恒开；
唯一实现点 = `pathlm.depth_ar.window_anchor_keep_mask`，**训练侧与评测侧共用同一函数**。
`train_window=0`（缺省）= 关闭 ⇒ 走 stock 因果 SDPA 路径，逐位还原 v1（2615 已用单测钉住）。
**本实验没有新增开关**：契约 Q 的文件所有权允许改 `pathlm/depth_ar.py`，但既有实现
（窗口长度是参数）已经覆盖 `window=192`，所以**实际未改任何 `pathlm/` 文件、未改
`train_depth_ar.py`**（代码指纹见 `MODEL.md`；与 2615 记录一致）。

## 4. 判据线（**逐字**抄录契约 Q 的判据段，未改一字）

> - **判据（预登记）**：
>   1. **主判据**：`window192` 的 clean bpc 相对 full 的 Δ **≤ 0.014** ⇒ 「有界内存可近零代价」成立
>      （这是本批可能出现的正面结论）；Δ ≥ +0.050 ⇒ 悬崖在 192 以下，记录；
>      中间值如实报。
>   2. **曲线定位**：把已有的 `recent:32/64/128/192` 评测期四点与本次训练点并列成表，
>      给出「训练税 vs 窗口长度」的**两个数据点**（32、192），并**明确标注不得外推内插**。
>   3. 附：`window192` 的评测期依赖曲线（仪器）与窗口外针召回（检索腿），与 32 的结果对照——
>      验证「窗口放宽后，窗口外的针是否仍然只是「丢了」」。

### 执行时的操作性定义（**不改判据**，只把符号与「显著」落定）

- **符号**：契约 Q 的判据用**税**：`Δ = bpc_window192(native 工作点) − bpc_full`
  （正 = 变差）；SCHEMA §17.3 的规范符号 `Δ_schema = bpc_对照 − bpc_实验 = −税`
  （正 = 改善）。`results.json::criteria` **两个都报**，本文件与 `analysis.md` 以「税」为主。
- **bpc 口径（native 工作点）**：`full` = 全注意力（其原生配置就是全可见）；
  `window192` = 其训练掩码（最近 192 + 8 锚点）。**并列**报 release 口径
  （window192 在全可见下的 bpc），两者都写进 `results.json::criteria`。
  （与 32-O 的判据①同一口径，因此 32 与 192 两个训练点可直接并列。）
- **判据 ③ 的「仍然只是丢了」** = 沿用 32-O 的操作性定义（不改判据）：窗掩码下
  `beyond` 桶的召回相对**同一掩码、同一批针的 `plant=False` 无针地板**的差 < 10pp。
- 地板不可判读的例外：若 `full` 臂 release 口径的针基线召回 < 50%（契约 M 的构造门槛），
  视为构造问题，先修构造再判读（`construction_check` 会把它标红）。

## 5. 单因素声明与配置等值性（SCHEMA §9）

唯一因素 = **训练期可见窗口长度**（192 vs 全可见）。两臂同 runner（`train_depth_ar.train`）、
同架构、同数据、同优化器/调度/步数/批大小/seed；`full` 臂**复用 2610 的 `trunk.pt`**
（sha256 断言），逐字段对照表见 `analysis.md` §3（不等字段：仅 `train_window`/`train_anchors`）。

**实现层等值性证据**：`tests/test_depth_ar.py::test_train_window_default_off_matches_v1_bitwise`
（关闭开关 ⇒ 与显式全窗口 / 无掩码裸堆叠逐位一致；2615 已钉住）+ 本实验 20 步冒烟与
1000/600 步步时基准跑的都是**未改动的** `train_depth_ar.py`。

## 6. 指标表（多维，逐项）

| 维度 | 指标 | 与谁比 | 结果（见 `analysis.md`） |
|---|---|---|---|
| 解决什么问题 | 「有界内存能不能近零代价」的直接检验 | — | 判据 ① |
| 代价 | 训练期窗口 192 对 clean bpc 的税 | `full` 臂（native 口径） | 判据 ① |
| 有效性（依赖） | 距离曲线 / 前缀曲线 / 最近 d 曲线的 damage | 各臂 release 基线 + 2614/2615 记录 | 判据 ②③ |
| 有效性（保留） | 针召回（总 / anchor / window / **beyond** 分桶 + 距离分带） | 同掩码无针地板；`full` 臂同掩码；32-O 的 32 臂 | 判据 ③ |
| **预算对齐（§17.12）** | 同一族掩码下 w ∈ {32,192} 的「删除量」与「可见性对数」两套口径 + full 臂在同预算下的评测期 bpc | 两个训练点并列 | `results.json::budget_alignment` |
| 收益与用法 | 部署侧 KV 位置/行（解析内存）与可见对份额 | 全注意力（256 位置 / 32896 对） | §analysis 成本节 |
| 可控性 | 评测侧复现性（同 seed 重跑） + 跨窗口 seed 散布 | 阈值 0.005 / 方差界 0.014 | §analysis |
| 副产品 | 3 条可失败单测（真实尺度 192）+ 变异验证 + 数字对账脚本 | — | §analysis |
| 干预是否活着 | 掩码活性量（可见因果对占比，每 200 步落盘）+ 一次功能探针（掩码开/关最大 logit 差） | 0 参数增量 | `results.json::intervention`（= `train/window192/results.json::gnorm_monitor.window_mask`） |

## 7. 对照与方差

- **对照臂 = `full`（2610 `trunk.pt`，12k 步全可见）**，身份 = 复用而非重训；配置等值性
  逐字段对照见 `analysis.md` §3；口径标注：「历史同配置对照（2610）+ 等值性证明 +
  已知口径（同窗口序、同仪器）」。full 臂的同配置数字必须复现 2614/2615 的记录
  （`results.json::reconciliation_vs_2614`，容差 5e-4 = 记录精度）。
- **同一模型内的对照**：距离/前缀/最近 d 曲线的参照物是**同一臂自己的 release 路径**；
  检索腿的参照物是**同一掩码下的 `plant=False` 地板**。
- **同配置重跑臂（`window192_rep`）**：为满足「正/负结论都要能站住」与「n=1 结论需要
  一个 run-to-run 量级」，整条 window192 臂同 seed 重跑一次（成本 +≈8 min），
  结果见 `stability.md §3.2` 与 `rep/results.json`；它**不参与判据口径**（主臂仍是
  `train/window192/model.pt`），只作训练侧噪声与结论复现的证据。
  **预登记的执行规则（不是判据改动）**：若主臂结果落在近零门槛附近、或与 32-O 的
  0.0866 相差不足 5× 训练噪声，则**必须**跑这一条重跑臂；否则允许省略并如实记录。
- 方差界：Δ < 0.014 不可排序（methodology §2）。评测侧另测同 seed 重跑差（阈值 0.005）
  与跨窗口 seed 散布（3 个 seed）。**训练侧**：固定 seed 训练在本引擎上不可逐位复现
  （项目级已知事实，`.scratch/11-retry-matrix-validity` D4）；本实验的重跑臂实测给出量级。

## 8. 成本（唯一数字 = 契约 §「成本」行）

- 契约写的预算：**训练 ≈7 min + 仪器 ≈5 min ⇒ ≈15–25 min**（`gpu_wall_budget: 25 min`，
  含冒烟、步时基准与训练后电池）。
- **实测**（共享卡，其他任务占 6–9 GB）：步时 **42.1 ms/步（窗口开）** vs 40.1 ms/步
  （窗口关）⇒ 掩码路径 **1.05×**（日志 `smoke/stepbench_w{192,0}/`，600 步实测、
  **不外推时间字段**，遵循 SCHEMA §17.5/§17.8）。12k 步训练实测墙钟见
  `analysis.md §9`（与 `results.json::cost` 逐一对齐）。
- 峰值显存：训练 **1840.7 MiB（net，allocated；30 步探针实测）**、评测净显存见
  `results.json::cost`（预算 ≤5GB net）。
  **解析内存 vs 实测显存分开报**（契约纪律「省内存」类要求）：
  解析上滑动窗口把 KV 位置/行从 256 降到 200（−21.9%），但**本仪器的实测显存不下降**
  ——它仍是 dense SDPA + bool 掩码，照算整张 T×T 分数矩阵；真正的省要在部署侧实现
  ring-buffer KV。
- 可中断性：训练中可杀（重跑 ≈8–9 min）；评测纯推理、可重跑（≈1 min）。
- 冒烟计划：`python3 run.py --smoke`（2 个窗口批次 + 少量配置 + 针 2 批，**不留外推
  时间字段**）；训练侧 20 步冒烟用同一 runner（`--steps 20`）。
- 预算不够时砍什么：砍「同配置重跑臂 → 变异验证的破坏点个数 → 检索腿 batch 数」，
  **不砍判据线**。

## 9. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`；前 90M train / 末 10M eval；vocab 206） |
| `full` 臂权重 | `2610-frozen-trunk-jumphead-13m/trunk.pt`，sha256 `86953dcb…159d2`（运行时断言） |
| `window192` 臂权重 | `train/window192/model.pt`，sha256 见 `MODEL.md`（运行时断言；不入库） |
| 训练入口 | `python3 train_depth_ar.py window192 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --train-window 192 --train-anchors 8 --eval-batches 24 --out-root <expdir>/train` |
| 评测入口 | `python3 run.py`（`--smoke` 冒烟）；仪器 `tools/evalkit/retention.py`（只增不改） |
| 落盘 | `results.json` / `retention_log.jsonl` / `run_log.txt` / `train/window192/{train_log.jsonl,gate_log.jsonl,results.json}` / `mutation_report.json` / `reconcile_report.json` |
| resolved config | `config.json`（含 seed、两臂逐字段配置、数据切分、评测窗口序） |

## 10. 失败决策树

| 症状 | 诊断 | 处置 |
|---|---|---|
| 税与 32 的 0.0866 同量级 | 窗口长度不是杠杆（悬崖在 192 以下不存在，或杠杆在别处） | 记为否证；写清「在 192 这个取值上没有头寸」；不叠加第二个机制 |
| 税 ≈0 但归因可疑（例如模型根本没看远处） | 干预无头寸 / 口径问题 | 查 `budget_alignment` 两套口径与依赖曲线；若 release 口径也几乎不退化，说明该规模上 192 窗口与全可见**几乎等价**（这本身就是结论） |
| 判据 ③ 有召回但 `window` 桶也崩 | 实现/掩码不一致或针构造带病 | 先查 `instrument.train_mask_equals_eval_mask` 与 `construction_check`；构造 < 50% 门槛时不得判读 |
| 曲线非单调、超出噪声容差 | 测量问题 | 先修仪器/重跑；记 ERRATA |
| 两臂 Δbpc 与训练日志矛盾 | 口径问题（native vs release、快照 vs 渐近） | 双报口径，检查是否假失败（冷启动伪影 / 前缀内 query 掉对角） |
| 结果不可重复 | 方差（训练侧不可逐位复现是已知事实） | 按 Δ < 0.014 不可排序；结论用重跑臂复核（`stability.md`） |

**原则**：先排除测量问题与实现问题，再排除规模问题，最后才归因设计问题（methodology §14）。

## 11. 产出

`README.md`（本文件）/ `run.py` / `config.json` / `results.json` / `analysis.md`
（人话摘要在最前 + 多维指标表 + 逐字段对照 + 三道对账 + 失败决策树 + 口径 + 实测墙钟与
次数）/ `MODEL.md`（sha256）/ `stability.md` / `retention_log.jsonl` / `run_log.txt` /
`mutation_report.json` / `reconcile_numbers.py` + `reconcile_report.json`；
权重 `train/window192/model.pt` 不入库（哈希见 `MODEL.md`）。
