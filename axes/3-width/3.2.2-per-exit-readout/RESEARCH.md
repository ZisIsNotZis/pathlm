# RESEARCH · `3.2.2` 逐出口读出：多出口时每个出口怎么映射到词表

研究级**计划视图**：目标、目的、唯一里程碑、工作包（owner / 验收 / 预算 / 状态）与证据指针。
**事实与数字的家不在这里**——研究项事实（判据线、结论、`cost_rows`、`gaps`）在
`axes/3-width/3.2.2-per-exit-readout/project.yaml`，读数在各实验目录的 `results.json` /
`analysis.md`。本文件只放指针，不复制读数。

## 0 这是什么

普通 decoder-only transformer 只有一个出口（末层），一张 tied 读出表就够。早退 / 多出口模型里
每一步都要把「该深度的隐藏状态」映射到整张词表，于是有一个选择：**九个出口共用一张表，还是每个
出口自己一套**。本项判的就是这条「出口 → 词表」的形态，并放进以共享 tied 线性读出为 100% 的
质量-代价前沿。

第一轮（`WP1`）的判词是「**被否证**」：`PE`（共享表 + 每出口 d×d 变换）让末层与各出口的**绝对**
质量都更好，但浅出口的**相对**接受率 `tf@k` 两 seed 同向下降 ⇒ 被否证的是「浅出口相对接受率该涨」
这条读法，不是「逐出口读出让早退不可用」。那一轮留下一个没答的问题：**`PE` 多出来的约 +8.5% 参数，
扣掉参数因素后还剩多少**——即这笔交易是「多了参数所以更好」还是「读出结构本身有额外收益」。
本轮（`WP2`）只闭这一个问题，不重开「参数总量是旋钮」（已由 `3.1.1` / `3.1.4` 判定）。

它和相邻项的分界：`3.2.1` 判**单出口**读出的重参数化；「跑不跑这层 / 退不退」归深度轴 `2.1.1`；
本项只判**多出口**时「出口 → 词表」的映射形态。

## 1 目的（对着谁）

- **痛点**：早退 / 投机 / 多出口这条路线的每一个出口都要一次「状态 → 词表」映射。共用 tied 表省参数
  但把所有出口绑在同一个读出头族上；逐出口映射解耦但要付参数与算力。**该付哪一笔**是本仓能付得起的问法，
  而第一轮的读数无法把「多给容量」与「出口 → 词表形态」分开——这正是本轮的靶子。
- **对标**：early-exit 的多出口读出一族（CALM / LayerSkip / MoD，指针见 `axes/7-references/`）与逐深度
  输出头。业界条目按本地实测为准（`harness/SCHEMA.md` §21），**不作结论依据**。本地锚 = 共享 tied 线性
  读出（stock，归一为 100%）；参数总量这条轴的本地读数在 `3.1.1` / `3.1.4`。
- **本项不做**：组合实验（出口形态 × 量化 × 投机，`harness/SCHEMA.md` §4 留白）；真词表（100k+，
  V > d 时 `PE` 的表达力优势消失）；推理期早退调度与墙钟。

## 2 唯一里程碑

**输出 / 验收口径的家**：`project.yaml::milestone`（原文在那里，本文件不转述判据数字）。
本节只登记进度：

- `WP1`（四臂 + `T`/`PE` 两 seed）**已交付并判「被否证」**：`PE` 末层与各出口绝对质量更好，
  浅出口**相对**接受率两 seed 同向下降；「契约先于读数」对本格**不可判**（读数入库早于绑定，已披露）。
- `WP2`（等参数对照臂 `WIDE` × 2 seed，本轮）把第一轮留下的归因缺口闭掉：**参数是不是全部解释**，
  若不是，残余多大、是否越 0.014 带；若不可判，给所需样本量。判词与读数见
  `experiments/2645-per-exit-readout-13m/analysis.md` §6 与 `project.yaml::conclusion`。

## 3 工作包

owner = `axis3-lead`（轴 3 宽度轴 lead）；writer = `axis3-w2`（3.2.2 worker，单一写入者）；
reviewer = `axis3-reviewer`（fresh 独立上下文，不得为 writer，复核排在 writer 停止之后）；
integrator = `tree-integrator`（本轴合并与跨轴一致性核对由 `axis3-lead` 代行）。角色定义见
`project.yaml::role_map`；逐包的 `what/goal/means/acceptance/evidence/review/status` 原文在
`project.yaml::work_packages`，本表只给计划口径的 owner / 验收 / 预算 / 状态。

| WP | 交付物 | 验收（可观测） | 预算 | 状态 |
|---|---|---|---|---|
| **WP1** 四臂最小判决（`T` / `PE` / `IND` / `LR`，13M matched-step） | `experiments/2645-per-exit-readout-13m/`（`T`/`PE` 各 2 seed；`IND`/`LR` 只到冒烟） | 四支单因素臂预算对齐、主干逐位相同；各出口 `tf@k` 与末层 clean bpc 齐备；n≥2 出判词 | 2 臂 × 2 seed ≈10 min/臂 + 评测 | 被否证（判词见 `project.yaml::conclusion`，读数见该实验 `analysis.md` §1/§2） |
| **WP2** 等参数对照臂 `WIDE`（把 `PE` 的 589,824 读出参数给主干 FFN 隐藏宽） | 同实验目录第二批：`run.py` 第五臂 + `README.md` §4/§11 预登记 P5 + `config.json::p5_rule` + `test_2645_per_exit_readout.py` 的 `T9` | 契约先绑后跑（`prereg_binding.json`）；`WIDE` 参数增量 590,976（比 `PE` 多 1,152 = +0.2% 增量、+0.015% 总参数）、每 token 训练 MACs 与 `PE` 逐位相等（单测 `T9`）；四分支判词只由 0.014 带与符号定；两 seed 同分支才出 n=2 判词 | 2 臂 × ≈11 min GPU + 评测（单臂 timeout 20 min，逐臂取/放 GPU 锁） | **已完成**（判词见 `project.yaml::conclusion`，读数见 `analysis.md` §6） |

**WP2 为什么就是分诊表那一条动作**：`atlas/triage.md` 给本格的后续行写的是「等参数对照臂（把读出参数额
给主干），回答『这笔参数给主干还剩多少』」，打分 2.0、档位「补一条最小实验」。本实验与之一致：**不加新
判据形态、不改 0.014 带**，只多一支把同一笔参数花在主干上的臂，把「多给容量 vs 出口 → 词表形态」这件事
分开。

## 4 证据指针

- 研究项事实与判据线：`axes/3-width/3.2.2-per-exit-readout/project.yaml`
  （`milestone` / `work_packages` / `conclusion` / `failure_analysis` / `cost_rows` / `gaps`）。
- `WP1` + `WP2`：`experiments/2645-per-exit-readout-13m/{README.md,config.json,run.py,analysis.md,results.json}`
  ——契约与判据线在 `README.md` §4（P1–P4）与 §4/§11（P5）、判据常数在 `config.json::{band_bpc,p5_rule,n2_rule}`、
  门在 `test_2645_per_exit_readout.py`（20 项，含 `T9`）；定序证据 `prereg_binding.json`
  （第二批绑定早于 `WIDE` 读数；第一轮的读数早于第一轮绑定，已在 `readings_existing_at_bind` 里披露）。
- 跨格排序与该格动作：`atlas/triage.md`（§2 该行 + 后续行）；结论索引行：`atlas/FINDINGS.md`。
- 代价事实（生成物，不手改）：`atlas/cost-table.md`；复现清单：`atlas/repro.md`。

## 5 状态（就地更新）

- `WP1`：**被否证**（`PE` 绝对质量更好但浅出口相对接受率两 seed 同向下降；判据设计缺陷已写进
  `project.yaml::failure_analysis` ①）。
- `WP2`：**已交付**——判词与读数在 `project.yaml::conclusion` 的归因段与
  `experiments/2645-per-exit-readout-13m/analysis.md` §6。
- 本格**里程碑达成**：唯一里程碑的三条（归因隔离 / 判据先绑后用 / 结论明确回答参数是不是全部解释）
  都已落盘；`atlas/triage.md` 该行与后续行标为已闭、退出排序。
- **复核状态：待复核。** 本格产出属执行侧；`axis3-reviewer`（fresh 上下文、不得为 writer）**尚未派**，
  所以本文与 `project.yaml` 都**不写「已复核通过」**（`project.yaml::conclusion_review` 同写「独立复核：未做」）。
