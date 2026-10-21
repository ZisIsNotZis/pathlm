# 1.1.1 hard 窗口 · 研究级契约（目标 / 唯一里程碑 / 工作包 / 证据指针）

> **本文件是一份 projected 视图，不是事实的家。** 目标、目的、判据、结论、里程碑与工作包字段的
> 唯一之家是 `project.yaml`（本目录），读数之家是各实验目录的 `analysis.md` / `results.json`。
> 本文件只做三件 `project.yaml` 做不到的事：把**派发口径**（每条工作包的 owner / 预算 / 复核人）
> 写清、把**证据指针**排成一条可走的路、把「研究级第 7 条验收」映射到具体命令。
> 数值一律不在此复制；需要数字时按指针去读。

## 1 目标与目的

- **它是干什么的**：把每个位置能看见的范围压成「最近 k 个位置 + 少数恒可见锚点/sink」，使长上下文的
  KV 与算力随序列线性而非平方增长。见 `project.yaml::innovation`。
- **为什么值得做**：窗口族只改可见集、不加参数，是长上下文最便宜的一档；本项要回答的是**这档便宜
  到什么程度**——质量税多少、代价是否随 T 消失、以及「锚点槽」到底买到了什么。见
  `project.yaml::purpose` / `::gap`。
- **它不是什么**（非目标）：不是「把远处信息搬到当前行」的机制（本项已用窗外桶与毁源探针把这条
  否证，见 `project.yaml::conclusion` ③），也不是组合实验库（组合默认留白）。

## 2 唯一里程碑（验收）

一字不改地取自 `project.yaml::milestone`：一份自包含的「硬窗口保留曲线」（k 全曲线 + 累计预算列
与单距离列分开 + n≥2 复本 + 长 T 同 step 子线 + 锚点通道召回相对同掩码地板可排序）。

**当前状态**：唯一之家是 `project.yaml::status` 与 `::milestone.acceptance` 的逐条证据——本文件不复制。
里程碑四条验收现在各有落盘证据：k 全曲线两列（`2682`）、锚点召回腿 n=4（`2681`）、长 T 同 step（`2636`）、
判词不越界（`project.yaml::conclusion`）。**独立复核已派一次（fresh 上下文）**：四条验收里 ①②③ 成立、
④ 不成立（两处 P1 都在 `project.yaml::conclusion`：把 `undecidable` 的 k≥128 写成「≈0」，以及「值不值」缺代价侧），
两处已按盘上读数改正；同时把判据② 的 k=128 与判据⑥ 的 k=64 各补到 n=3（`2686`）——**两条补完仍不可判**，
卡住的是**区间宽度**（配对差 `sd` 0.005073 / 0.003047，n=3 半宽 0.0126 / 0.0076 均大于效应量本身），
**不是带的量纲**（判据⑥ 的规则不含带、判据② 的带越小越容易判越带）；按其未改的规则只差 seed
（⑥ 补 1 个到 n=4、② 补 2 个到 n=5），由 `2688` 承接。
**重核尚未派**：本次改正后需要 fresh 复核只看 `project.yaml::conclusion` 一字段才算过。

**本轮交付（`2682` + `2686`）的派发口径**：`2682` 训 8 条纯窗臂 + 2 条可选锚点臂（逐臂 rc=0，实耗 88.7 min GPU，
其中 8 条纯窗臂 71.1 min），连同 9 条复用臂一次评测算出四条判据；判据① 复现门 9/9 逐位相同 ⇒ 跨项复本的四个
k=16 种子与本目录新臂同仪器。`2686` 另训 3 条 seed-2 臂（`win128_s2` / `win64_s2` / `win64a_s2`）把两条
「不可判」补到 n=3：判据① 复现门重评 `2682` 的 7 条 checkpoint 要求逐位相同。
读数与判读只看 `experiments/{2682-window-kcurve-13m,2686-window-kcurve-seed2-13m}/analysis.md`，本文件不复述数字。

## 3 工作包（owner / 验收 / 预算 / 状态）

每条工作包的**内容、验收与状态**的唯一之家是 `project.yaml::work_packages`（状态看
`work_packages[*].status`）；本表只放**派发口径**（owner / 预算 / 复核人），不复制任何事实。

| id | owner | 预算 | 复核人 | 状态（只放指针） |
|---|---|---|---|---|
| WP1 | `axis1-w1.1.1` | 已花：≈30 min GPU（2 臂 × 12k 步）+ 评测；`2682` 另 8 臂 71.1 min；`2686` 另 3 臂（seed 2） | `axis1-reviewer` | → `project.yaml::work_packages[0].status` |
| WP2 | `axis1-w1.1.1` | 实耗：8 臂 × 12k 步 = 71.1 min GPU（+ 可选 2 臂 17.6 min）+ 评测 79.1 s（`2682` §7 的预算是 ≤110 + ≤12 min）；`2686` 的 3 臂另计 | `axis1-reviewer` | → `project.yaml::work_packages[1].status` |
| WP3 | `axis1-w1.1.1` | 已花：4 臂 × 12k 步 ≈ 36 min GPU（8.2–10.5 min/臂）+ 评测 2.84 min | `axis1-reviewer` | → `project.yaml::work_packages[2].status` |
| WP4 | `axis1-w1.1.1` | 已花：0 GPU（迁移读数统一口径） | `axis1-reviewer` | → `project.yaml::work_packages[3].status` |

**复核人**：`axis1-reviewer`（fresh 上下文，非本项 writer）；按 `harness/SCHEMA.md` §14/§14b，
复核**排在执行之后**，只看落盘产物与证据。
## 4 WP3 本轮的判定口径（为什么不是「照抄分诊表那一句」）

分诊表给本格的单一动作是「补 seed 2/3，使召回腿从 n=2 不可判变成可判」。**补 seed 保留**，
但**判定单位必须同时改**，依据是 `2662` 已落盘读数的功效算术（不涉及本轮任何新读数）：

- 预登记门是「两个 seed 都过两套 CI」——一个**逐 seed 的合取式**。锚点桶 1,003 根针、召回 ≈47–48%
  ⇒ 未配对两比例差的 95% 半宽 ≈ ±4.37pp；要把下界推过 0，1–3pp 的效应需要 **2,850–15,900 根/seed**
  （详见 `experiments/2681-anchor-recall-seeds23-13m/README.md` §2 的功效表）。
- 加 seed 只给合取式**加约束项** ⇒ 无论补到 n=4 还是 n=40，那条门都不会变成「可判」。
- 所以判定单位改成**种子层**：4 个 seed 的 `Δ_pp` 均值 + 两个跨 seed 区间（t 区间 df=3 与两级
  bootstrap）下界都 > 0 且 ≥3/4 同向 = 可判。**判据线的常数、CI 的两套实现、配对口径、方差带
  一项未改**，旧读数一项未删，全部并列报出（`experiments/2681-anchor-recall-seeds23-13m/README.md` §5a）。
- 另一条被评估后**否决**的动作是「把每 seed 的针数加到 15,900」：`needle_raw.json` 会从 ~8.5 MB
  涨到 ~135 MB，且改的是针密度口径；而种子才是「换成别的训练随机性还会不会为正」的重复单位。

## 5 证据指针（按顺序走就能复核本项的全部结论）

| 结论 | 指针 |
|---|---|
| **窗口的价格（k 全曲线，本目录自包含）** | `experiments/2682-window-kcurve-13m/analysis.md` §3 判据②；键路径 `results.json::criteria.c2_train_budget_column`；**k=128 补到 n=3 的判读**看 `experiments/2686-window-kcurve-seed2-13m/analysis.md` |
| **累计预算列与单距离列分开 + 两列不可互换** | `experiments/2682-window-kcurve-13m/analysis.md` §3 判据③④⑤；键路径 `results.json::criteria.c3_eval_budget_column` / `::c4_single_distance_column` / `::c5_columns_not_interchangeable` |
| **锚点杠杆在第二个 k（k=64）** | `experiments/2682-window-kcurve-13m/analysis.md` §3 判据⑥（n=2）；**n=3 判读**看 `experiments/2686-window-kcurve-seed2-13m/analysis.md` |
| 锚点的质量价格（n=2） | `experiments/2662-hard-window-anchor-13m/analysis.md` §3 |
| 锚点通道在推理期被用起来（0 训练 token 消融） | `experiments/2679-anchor-eval-ablation-13m/analysis.md` |
| **锚点召回腿 n=4 判词（本轮）** | `experiments/2681-anchor-recall-seeds23-13m/{analysis.md,results.json}` |
| 窗口代价随 T（同 step 四点） | `experiments/2636-longT-unified/analysis.md` |
| 锚点不搬运窗外内容（毁源探针） | `experiments/2662-hard-window-anchor-13m/analysis.md` §2 判据④ |

## 6 研究级第 7 条验收：具体命令

```bash
python3 tools/summary.py --check
python3 tools/check_tree.py
python3 tools/check_docs.py --check
python3 tools/build_cost_table.py --check
python3 tools/inventory_runs.py --check
python3 -m pytest tests/ -q
```

唯一允许失败：已登记的预存失败 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`。

## 7 显式未解缺口（不在本轮 scope，留给下一轮）

- **「锚点收益归推理期还是训练侧」仍不可分**：`2679` 的两个方向都破坏分布；拆开它需要**等键数
  24 键对照臂**（窗口 24、无锚点，同为 24 个最近键；1 臂 ≈10 min GPU + 复用 `experiments/2679-anchor-eval-ablation-13m/run.py` 评测）。
  这是分诊表 4.0 档的动作，**不在本轮**（本轮动作是召回腿的 n=4）。
- 分布内消融需要训练期 anchor dropout（改共享训练入口那一档，另立项）。
- **k 曲线自包含与两列拆分已闭**（`2682`）；两条「补 1 个 seed 就能判」的已补（`2686`：k=128 与判据⑥ 的 k=64，各 n=2→3）
  但**补完仍不可判**：上一轮被否定的是**基于 n=2 两点 `sd` 的半宽估计**（0.001979 / 0.00094 涨到 0.005073 / 0.003047），
  不是补 seed 这条路本身 ⇒ 按未改的规则，⑥ 补到 n=4（`sd ≤ 0.004148`）、② 补到 n=5（`sd ≤ 0.005968`）即可判（`2688`）。
- **「换带」是口径登记项、不是当前收口路径**：σ_cell 是两点极差、本格 k=16 的 n=4 配对差 `sd` 已 0.002188（2.2×），
  是否改以配对差散布作可排序线属框架性取舍（`AGENTS.md` §3.3 纪律 3），须先走完补 seed；且按字面改带的两种读法
  一种把 k=128 的 `n_req` 推到 20、另一种直接判「税存在」，结论相反 ⇒ 该路径连定义都未齐。
- k=192 要 n≈17（≈135 min GPU）⇒ **本格不取得**；卡住的是半宽而非带，按未改的规则补到 n≈17 也仍然太贵。
- T=1024/2048 无 seed 复本；per-head 平均注意距离分布未报。
