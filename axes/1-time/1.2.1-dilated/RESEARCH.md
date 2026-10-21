# 1.2.1 dilated / 空洞 · 研究级契约（目标 / 唯一里程碑 / 工作包 / 证据指针）

> **本文件是一份 projected 视图，不是事实的家。** 目标、目的、判据、结论、里程碑与工作包字段的唯一之
> 家是 `project.yaml`（本目录）；读数之家是各实验目录的 `analysis.md` / `results.json`。本文件只做三件
> `project.yaml` 做不到的事：把**派发口径**（每条工作包的 owner / 预算 / 复核人）写清、把**证据指针**
> 排成一条可走的路、把本轮「为什么这样切两条腿」写清。数值一律不在此复制；要数字时按指针去读。

## 1 目标与目的

- **它是干什么的**：在每 token 可见量固定（远小于 T）的前提下，让可见集合覆盖到比「最近 k 个」远得多
  的上下文，从而在省算力的同时尽量少丢远距离可达性。见 `project.yaml::purpose` / `::innovation`。
- **为什么值得做**：窗口族只改可见集、不加参数，是长上下文最便宜的一档；空洞采样是同一档里唯一
  「不换预算就能扩感受野」的选项。要回答的是**这笔交易值不值**：同每 token 可见键数下，把位置摊开
  相对取最近 k 个换来多少质量。见 `project.yaml::objective_criteria` / `::falsification`。
- **它不是什么**（非目标）：不是组合实验库（`harness/SCHEMA.md` §4 的组合留白）；不是真稀疏核
  工程（纸面省算力未兑现为实测省算力，这一条与 `1.2.2` 共用缺口）；不是「换成内容选择」（那是
  `1.2.3`）。

## 2 唯一里程碑（验收）

一字不改地取自 `project.yaml::milestone`：一份「dilated vs 硬窗口」的同预算判决包——
{全因果, 硬窗口 k, 硬窗口+空洞} 在相同每 token 可见键数下的 clean bpc 与长程召回，对照 stock 全因果，
含 n≥2；三臂的每 token 可见键数严格相等；dilated 相对同宽硬窗口的长程召回可排序提升或无提升，
二选一明确；结论不越尺度外推；口径按 matched-step 报。

**当前状态**：唯一之家是 `project.yaml::status` 与 `::milestone.acceptance_evidence` 的逐条证据，本文件不复制。
里程碑的两条腿（真实文本的质量腿、需要远处信息时的机制腿）在 T=256 上已各有落盘证据（`2678`）；本轮把质量腿推到长 T
（`2689`）并加了逐层递增 dilation（同实验的机制腿）与远处点密度的支持读数（`2690`）。三条已交付、一条**没答**：
长 T 腿判出「效应随 T 变号」（T=2048 上摊开更差，n=2）、逐层递增腿判出「分布方式值钱」（n=1）、
密度腿（`2690`）坐在稳定边缘上（三次训练-评测六个臂次里只有两次过预登记入场门，且这两次过门分属不同 seed、方向相反）⇒ **未取得可用读数**，
该岔口仍开着、登记在 `project.yaml::gaps`。判词的方向与「与 `2678` 是尺度依赖不是矛盾」的写法见 `project.yaml::conclusion`。
**独立复核**：本格的复核人 `axis1-reviewer`（fresh 上下文，非本项 writer）排在执行之后，按 `harness/SCHEMA.md` §14/§14b
只看落盘产物；本轮尚未复核。

## 3 工作包（owner / 验收 / 预算 / 状态）

每条工作包的**内容、验收与状态**的唯一之家是 `project.yaml::work_packages`；本表只放**派发口径**
（owner / 预算 / 复核人），不复制任何事实。

| id | owner | 预算 | 复核人 | 状态（只放指针） |
|---|---|---|---|---|
| WP1 · T=256 的最小判决（同可见键数） | `axis1-w1.2.1` | 已花：2 臂 × 12k 步 ≈ 6.85 min/臂 + 评测；追补两条读数 ≈14 min | `axis1-reviewer` | → `project.yaml::work_packages[0].status` |
| WP2 · 长 T 腿（T=1024 / T=2048 的同一对臂） | `axis1-w1.2.1` | 已花：7 支新臂（T=2048 五支 ≈4.2 min/臂、T=1024 两支 ≈2.2 min/臂）≈26 min GPU + 评测；上限 120 min | `axis1-reviewer` | → `project.yaml::work_packages[1].status` |
| WP3 · 逐层递增 dilation（机制侧补充） | `axis1-w1.2.1` | 已花：1 支臂（T=2048、seed 0）≈4.2 min GPU；含在 WP2 的同一次取锁序列里 | `axis1-reviewer` | → `project.yaml::work_packages[2].status` |
| WP4 · 远处点密度的支持读数（16 → 128 个远处点） | `axis1-w1.2.1` | 已花：2 臂（T=2048、两 seed）≈9 min GPU | `axis1-reviewer` | → `project.yaml::work_packages[3].status` |

## 4 这一轮的派发口径：为什么是「长 T + 逐层递增」两条腿，以及为什么另开 `2690`

分诊表给本格点的动作是「长 T + 逐层递增 dilation 两条腿（长 T 是核心；逐层递增是机制侧补充）」。
两条腿**照做**，没有替换；下面是每条腿对应的判决问题与判据落点：

1. **长 T 腿（核心，`2689`）**：上一轮只在 T=256 判过「摊开位置不换质量」。本轮把同一对臂推到
   T=1024 与 T=2048，**每个 T 点同 98,304,000 token 预算**，问配对 Δ(T) 是否越 0.014 带——若越，
   起点与斜率；若一直带内，用判据③（远处键的边际价值 `D(T)`）区分「长 T 上不需要远处信息」与
   「本尺度 / 本 k 分辨不出」。判据线与解释规则写在 `experiments/2689-longT-layerdil-13m/README.md` §4。
2. **逐层递增腿（机制侧，`2689` 的 `lad32` 臂）**：上一轮各层用同一张摊满前缀的掩码。本轮让第 ℓ 层
   的远处点按 stride `2^ℓ` 推远（LongNet 式），第 0 层因此**逐位等于**窗口 32 臂 ⇒ 这条腿测的是
   「只要深层摊开、浅层保持连续窗口」是否比「每层都摊开」更值。
3. **另开 `2690` 的理由**：`2689` 的判据③ 只能说明「模型对它读到的那 16 个远处点吃得更多还是更少」，
   **不能**说明远处信息在长 T 上到底值不值钱——16 个远处点在 2048 长的前缀上间距 ≈136、整行只覆盖
   1.6%。`2690` 只改远处点的数目（16 → 128，近端 16 窗口逐位不变、每行可见键数 32 → 144），把
   `2689` 的「带内」拆成两种解释：远处信息不值钱，还是 32 键太稀。它是**支持读数**，不是第三条腿：
   它不开新的判决问题，只给主判的「不需要 / 分辨不出」二选一提供独立证据。
   **它是另开的一次预登记**（不是事后改 `2689` 的契约）——因为 `2689` 的绑定在读数之前就落了，
   事后加臂会让它「契约先于读数」失效；所以 `2690` 有自己的契约与绑定，写它的时候
   `2689` 的读数一条都还没落盘。

两处**没有**做的事，写下来免得读成默许：① 没有加重「全因果」参照臂——`full` 臂在 T ≥ 1024 上要
长度条件的 qk-norm 才稳（`1.1.1` 的长 T 线观测到 gnorm 爆炸），那是第二个因素，会破坏「唯一变量
= 可见集」；本轮的「远处键值不值」改用 `D(T)` 消融与 `2690` 的密度臂两条独立读数回答。
② 没有做真稀疏核，纸面 `O(T·k)` 仍未兑现为实测省算力（与 `1.2.2` 共用缺口）。

## 5 证据指针（按顺序走就能复核本项的全部结论）

| 结论 | 指针 |
|---|---|
| **T=256 的两半结论（上一轮）** | `experiments/2678-dilated-vs-window-13m/analysis.md`；键路径 `results.json::main_verdict`、`::synth_recall` |
| **长 T 上有没有质量增量（本轮核心）** | `experiments/2689-longT-layerdil-13m/analysis.md` §1；键路径 `results.json::criteria.c1_quality_by_T`、`::main_verdict` |
| **逐层递增 dilation 相对平铺** | 同上 `analysis.md` §2；键路径 `results.json::criteria.c2_layered_arm` |
| **远处键的边际价值随 T 的走向** | 同上 `analysis.md` §3；键路径 `results.json::criteria.c3_far_key_dependence` |
| **远处点加密到 128 能不能买到质量（本轮未答）** | `experiments/2690-dilated-density-k144-13m/analysis.md` §0/§2；键路径 `results.json::criteria.c1_density`（机械判词，**不作结论**）、`::criteria.c4_stability_gate.arms_over_threshold`（破门名单） |
| **结构 / 参数量 / 等值门 / 稳定性前置** | `2689` 与 `2690` 的 `results.json::structure_checks`、`::criteria.c4_must_not_change`、`::criteria.c5_reuse_equivalence`、`::criteria.c6_stability_gate` |
| 可引用的代价事实（进 `atlas/cost-table.md` 的那几行） | `project.yaml::cost_rows` |

## 6 派发与收口纪律（本轮的检查单）

- 判据线与常数在**训练之前**绑定：`2690` 的绑定在 `2689` 的任何读数之前，`2689` 的第一次绑定
  （14:02:34）在它自己的任何读数之前。此后为修三处**汇总阶段**缺陷（`run.py` 是契约文件）各自重绑过一次，
  重绑把当时已存在的读数记进 `readings_existing_at_bind`（定序不可判），并要求**全部读数重产出**才进判据；
  现存读数因此全部由最终绑定代码产出（`prereg_bind --check`：与绑定不符 0）。
- 每一份 `results*.json` 都过 `tools/run_provenance.write_results`（自带 `code_hash_run_py` /
  `code_hashes`），且落盘时间晚于绑定时间。
- 单臂 timeout 20 min、逐臂取/放 GPU 锁、臂间 ≥60 s；掉臂如实登记「未跑」，不当负结论。
- 收口前跑 `summary --check` / `check_tree` / `check_docs --check` / `build_cost_table --check` /
  `inventory_runs --check` / `prereg_bind --check` / `pytest tests/ -q`。
