# 2615 · 窗口训练后依赖是否「右移」（ticket 32 契约 O）· 13M · dense

> 契约来源：`.scratch/32-batch5/issues/01-batch5.md` §「契约 O」+ 开头统一纪律段。
> **副本标注**：§3 干预、§4 判据线（①–③）与 §2 判据取向段**逐字**抄录；其余小节为
> **摘要**（摘要处已注明）。判据线一律未改。
> 入口：`run.py`；训练入口：仓库根 `train_depth_ar.py`（本实验只用它 + 一个新开关）；
> 仪器：`tools/evalkit/retention.py`（依赖腿 = 2614 的三条曲线；检索腿 = 契约 M 的针）；
> 单测：`tests/test_depth_ar.py`（新增 8 条）+ `tests/test_retention.py`（M/N 既有）；
> 变异验证：`mutate.py` + `mutation_report.json`；稳定性专报：`stability.md`；
> 原始结果：`results.json`；逐配置日志：`retention_log.jsonl`；权重清单：`MODEL.md`。

## 1. 命题

**把模型训练成「只用近处」，信息会不会被搬到近处（依赖右移），从而前缀可丢而不掉质量？**
三个可分别证伪的子命题：① 窗口训练的**质量代价**有多大；② 依赖是否**局部化到更近的
距离**（同一件仪器测距离曲线，与全可见臂比）；③ 窗口外的针**还能不能答出来**——
能答才是「搬了」，只是答不出就是「丢了」（参照 X3 假保留陷阱）。

这是用户最初设想的**直接检验**：如果模型被训练成只看近处，靠隐藏状态把远处信息带到近处，
那么左侧前缀就是可以安全丢弃的。

## 2. 判定取向（**逐字**抄录契约 O 的动机段）

> 动机（用户最初的设想）：若模型被训练成只用近处，它是否会把信息**往右压**（即用近处的
> 状态承载远处内容），从而「前缀可丢而不掉质量」？

「同一件仪器」的含义（摘要）：两条腿都跑在**同一份**权重、同一份评测窗口序
（`window_seed=2` + 24×16，与 2614 一致）、同一份针构造（契约 M 的 needle v3，
`seed=3` + 32×16）上；删除动作一律是**改写注意力可见集**（`keep[i,j]=False`），
不是事后对 logits 做手脚。

## 3. 干预（**逐字**抄录契约 O 的干预段）

> - 干预（唯一变量 = 训练期可见窗口）：
>   - `full`（基线）：全可见（**复用契约 I 的曲线作为基线**，不重训；需给出配置等值性对照）；
>   - `window32`：训练时只允许看最近 32 个位置 + 8 个锚点（评测时用同一掩码）。
>   其余逐字段一致（`--dense`、seed=0、12k 步、bs32、seq256）。

**实现（摘要）**：掩码 `keep[i,j] = (j ≤ i) ∧ ((i−j) < 32 ∨ j < 8)`，对角恒开；
唯一实现点 = `pathlm.depth_ar.window_anchor_keep_mask`，**训练侧与评测侧共用同一函数**
（契约 P 第 ③ 项：训练掩码与评测掩码必须一致，否则量的是别的东西）。
`train_window=0`（缺省）= 关闭 ⇒ 走 stock 因果 SDPA 路径，逐位还原 v1（单测钉住）。

## 4. 判据线（**逐字**抄录契约 O 的判据段，未改一字）

> - **判据（预登记）**：
>   1. **质量**：`window32` 的 clean bpc 相对 `full` 的 Δ（阈值 0.014；历史 X2 曾测 +0.058~0.121）；
>   2. **依赖是否右移**：用同一仪器测 `window32` 的**距离曲线**，与 `full` 比较——
>      若 `window32` 在 d ∈ [8,32] 上的伤害显著**小于** `full`（依赖被压到更近），
>      记「依赖右移/局部化成立」；
>   3. **保留是否成立**：用契约 M 的检索腿测 `window32`——若它在窗口内能答出**原本位于
>      窗口外**的针（即信息确实被搬到近处），这才是「右移」的直接证据；若只是答不出，
>      则说明是「丢了」而不是「搬了」（必须区分，参照 X3 假保留陷阱）。

### 执行时的操作性定义（**不改判据**，只把符号与「显著」落定）

- 符号统一为 SCHEMA §17.3：`Δ = bpc_对照 − bpc_实验`（正 = 改善）；本实验
  「对照」= `full` 臂，「实验」= `window32` 臂。距离曲线的「伤害」= `damage =
  bpc_实验配置 − bpc_同臂基线`（正 = 变差）。
- 判据 ① 的 bpc = **各自运行点**（native operating point）：`full` = 全注意力（其原生
  配置就是全可见）；`window32` = 其训练掩码（最近 32 + 8 锚点）。**并列**报 release 口径
  （window32 在全可见下的 bpc），两者都写进 `results.json::criteria`。
- 判据 ② 的距离曲线基线 = **各臂自己的 release 路径**（`keep_mask=None`）——与 2614
  同口径，跨臂可比。判据 ② 的「显著小于」= `damage_full(d) − damage_window32(d)` 在
  d ∈ {8,16,32} 上的**均值 ≥ 0.014**（方差界，methodology §2）；逐 d 明细一并报。
- 判据 ③ 的「能答出窗口外的针」= 窗掩码下 `beyond` 桶的召回显著高于**同一掩码、
  同一批针的 `plant=False` 无针地板**（阈值 10pp）；并同时与「`full` 臂在同一窗掩码下
  的 beyond 召回」比较。答不出（与地板同水平）= 「丢了」。
- 地板不可判读的例外：若 `full` 臂 release 口径的针基线召回 < 50%（契约 M 的构造门槛），
  视为构造问题，先修构造再判读（`construction_check` 会把它标红）。

## 5. 单因素声明与配置等值性（§9）

唯一因素 = **训练期可见窗口**（0 vs 32+8 锚点）。两臂同 runner（`train_depth_ar.train`）、
同架构、同数据、同优化器/调度/步数/批大小/seed；`full` 臂**复用 2610 的 `trunk.pt`**
（sha256 断言），逐字段对照表见 `analysis.md` §3（不等字段：仅 `train_window`）。

**实现层等值性证据**：`tests/test_depth_ar.py::test_train_window_default_off_matches_v1_bitwise`
（关闭开关 ⇒ 与显式全窗口 / 无掩码裸堆叠逐位一致）+ `--train-window 0` 的 400 步重跑在
step 0 的 loss/gnorm 与 2610 记录**逐位相符**（`analysis.md` §3）。关不开开关时新代码
不增加任何计算、不消耗任何随机数。

## 6. 指标表（多维，逐项）

| 维度 | 指标 | 与谁比 | 结果（见 `analysis.md`） |
|---|---|---|---|
| 解决什么问题 | 「信息能否往右压、前缀能否丢」的直接检验 | — | 见判据 ①②③ |
| 代价 | 训练期窗口对 clean bpc 的税 | `full` 臂（native 口径） | 判据 ① |
| 有效性（依赖） | 距离曲线 / 前缀曲线 / 最近 d 曲线的 damage | 各臂 release 基线 + 2614 记录 | 判据 ② |
| 有效性（保留） | 针召回（总 / anchor / window / **beyond** 分桶 + 距离分带） | 同掩码无针地板；`full` 臂同掩码 | 判据 ③ |
| 收益与用法 | 「丢最远 N 个位置」的 damage（左前缀能丢多少） | N ∈ {8,…,255}，qmin128 口径 | §analysis 前缀曲线 |
| 可控性 | 评测侧复现性（同 seed 重跑） + 跨窗口 seed 散布 | 阈值 0.005 / 方差界 0.014 | §analysis |
| 副产品 | 训练期窗口开关（`pathlm/depth_ar.py`，缺省逐位还原 v1）+ 8 条可失败单测 + 变异验证 | — | §analysis |
| 干预是否活着 | 掩码活性量（可见因果对占比，每 200 步落盘）+ 一次功能探针（掩码开/关最大 logit 差） | 0 参数增量 | `results.json::intervention` |

## 7. 对照与方差

- **对照臂 = `full`（2610 `trunk.pt`，12k 步全可见）**，身份 = 复用而非重训；配置等值性
  逐字段对照见 `analysis.md` §3；口径标注：「历史同配置对照（2610）+ 等值性证明 +
  已知口径（同窗口序、同仪器）」。full 臂的同配置数字必须复现 2614 的记录
  （`results.json::reconciliation_vs_2614`，容差 5e-4 = 2614 记录精度）。
- **同一模型内的对照**：距离/前缀/最近 d 曲线的参照物是**同一臂自己的 release 路径**
  （不是跨 run 对照）；检索腿的参照物是**同一掩码下的 `plant=False` 地板**。
- **同配置重跑臂（`window32_rep`）**：为满足「n=1 负结果必须复现后入账」（methodology §2），
  整条 window32 臂同 seed 重跑一次（成本 +8.27 min），结果与判据结论见 `stability.md §3.2`
  与 `rep/results.json`；它**不参与判据口径**（主臂仍是 `train/window32/model.pt`），
  只作训练侧噪声与结论复现的证据。
- 方差界：Δ < 0.014 不可排序（methodology §2）。评测侧另测同 seed 重跑差（阈值 0.005）
  与跨窗口 seed 散布（3 个 seed）。**训练侧**：固定 seed 训练在本引擎上不可逐位复现
  （项目级已知事实，`.scratch/11-retry-matrix-validity` D4）；本实验实测了两个尺度：
  400 步重跑差 0.0156 bpc、**整条 12k 重跑差 0.0045 bpc（native 口径）**，
  并把量级写进 `results.json::convergence_reconciliation` 与 `stability.md §3`。

## 8. 成本（唯一数字 = 契约 §「成本」行）

- 契约写的预算：**训练 ≈7 min + 两条仪器 ≈15 min ⇒ ≈25 min**（`gpu_wall_budget: 25 min`，
  含冒烟与训练后电池）。
- **实测**：步时 38.4 ms/步（窗口开）vs 37.2 ms/步（窗口关）⇒ 掩码路径 **1.03×**（未离开
  融合核的显著回退；日志 `smoke/stepbench_w{32,0}/`，1000 步实测、**不外推时间字段**，
  遵循 SCHEMA §17.5/§17.8）；12k 步
  训练 **7.62 min**（复现臂 8.27 min）；两次评测电池 41.95 s + 27.81 s，各 396 次评测。
  **GPU 总墙钟 ≈17.0 min**（含复现臂；仅主臂则 ≈8.3 min）< 25 min 预算。
  峰值显存见 `results.json::cost`（预算 ≤5GB net）。
- 可中断性：训练中可杀（重跑 8 min）；评测纯推理、可重跑（重跑 ≈3 min）。
- 冒烟计划：`python3 run.py --smoke`（2 个窗口批次 + 少量配置 + 针 2 批，**不留外推
  时间字段**，遵循 SCHEMA §17.5）；训练侧 20 步冒烟用同一 runner（`--steps 20`）。
- 预算不够时砍什么：砍「变异验证的破坏点个数 → 噪声底 seed 数 → 检索腿 batch 数」，
  **不砍判据线**。

## 9. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`；前 90M train / 末 10M eval；vocab 206） |
| `full` 臂权重 | `2610-frozen-trunk-jumphead-13m/trunk.pt`，sha256 `86953dcb…159d2`（运行时断言） |
| `window32` 臂权重 | `train/window32/model.pt`，sha256 见 `MODEL.md`（运行时断言；不入库） |
| 训练入口 | `python3 train_depth_ar.py window32 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --train-window 32 --train-anchors 8 --eval-batches 24 --out-root <expdir>/train` |
| 评测入口 | `python3 run.py`（`--smoke` 冒烟）；仪器 `tools/evalkit/retention.py` |
| 落盘 | `results.json` / `retention_log.jsonl` / `run_log.txt` / `train/window32/{train_log.jsonl,results.json}` / `mutation_report.json` |
| resolved config | `config.json`（含 seed、两臂逐字段配置、数据切分、评测窗口序） |

## 10. 失败决策树

| 症状 | 诊断 | 处置 |
|---|---|---|
| 判据 ①② 无差异、判据 ③ 与地板同水平 | 训练期窗口**无头寸**（13M 规模上窗口训练没改变依赖） | 记录否证 + 写清「在什么规模/什么窗口长度下才可能有效」；不叠加第二个机制 |
| 判据 ③ 有召回但 `window` 桶也崩 | **实现/掩码不一致**（训练掩码≠评测掩码）或针构造带病 | 先查 `instrument.train_mask_equals_eval_mask` 与 `construction_check`；构造 < 50% 门槛时不得判读 |
| 曲线非单调、超出噪声容差 | **测量问题** | 先修仪器/重跑；记 ERRATA |
| 两臂 Δbpc 与训练日志矛盾 | 口径问题（native vs release、快照 vs 渐近） | 双报口径（§6），检查是否假失败（冷启动伪影 / 前缀内 query 掉对角） |
| 结果不可重复 | 方差（训练侧不可逐位复现是已知事实） | 按 Δ < 0.014 不可排序；n=1 负结果复现后入账（`stability.md` 记录） |

**原则**：先排除测量问题与实现问题，再排除规模问题，最后才归因设计问题（methodology §14）。

## 11. 产出

`README.md`（本文件）/ `run.py` / `config.json` / `results.json` / `analysis.md`
（人话摘要在最前 + 多维指标表 + 逐字段对照 + 三道对账 + 失败决策树 + 口径 + 实测墙钟与
次数）/ `MODEL.md` / `stability.md` / `retention_log.jsonl` / `run_log.txt` /
`mutation_report.json`；权重 `train/window32/model.pt` 不入库（哈希见 `MODEL.md`）。
