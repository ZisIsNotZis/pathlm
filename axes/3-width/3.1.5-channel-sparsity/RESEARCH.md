# RESEARCH · `3.1.5` 通道极化 / 稀疏激活：只激活部分通道

研究级**计划视图**：目标、目的、唯一里程碑、工作包（owner / 验收 / 预算 / 状态）与证据指针。
**事实与数字的家不在这里**——研究项事实（判据线、结论、`cost_rows`、`gaps`）在
`axes/3-width/3.1.5-channel-sparsity/project.yaml`，读数在各实验目录的 `results.json` /
`analysis.md`。本文件只放指针，不复制读数。

## 0 这是什么

普通 decoder-only transformer 每个 token 都激活 FFN 的全部隐藏通道，不管这个 token 难不难。
本项判的是「**保参数、少算通道**」这条算力路径值不值：总参数一个不少地留着，只让每个 token 的一部分
通道参与前向（通道粒度稀疏激活 / 极化），看它能不能在等参数的沙盘里换来「省算力而不亏质量」，
并放进以 stock 全通道 dense 为 100% 的质量-代价前沿。

它和专家轴 `4`（MoE）的分界：MoE 是**专家粒度**的 token 级路由（归 `4.1`/`4.3`，已降 Backlog），
本项是**通道粒度**的稀疏激活；两者记账口径不同，边界写进 `project.yaml::gap`。

## 1 目的（对着谁）

- **痛点**：MoE / 条件计算许诺「参数不变、每 token 算力下降」，但它的证据几乎都来自路由本身很贵的
  大模型；把「选哪部分通道」这件事单独拿出来、在 13M 上以等参数 / 等预算隔离，是本仓能付得起的问法。
- **对标**：Switch Transformer（top-k 路由 + 负载均衡辅助项）、ST-MoE（直通 top-k 估计）、
  Mixture-of-Depths（条件计算），以及本仓自己的跑分线（stock dense 100% = `2641` 的 `U4-s0`）。
  业界条目按本地实测为准（`harness/SCHEMA.md` §21），**不作结论依据**。
- **本项不做**：组合实验（通道稀疏 × MoE × 量化，见 `harness/SCHEMA.md` §4 留白）；真稀疏核的实测
  收益（本仓无核，纸面 MACs 记账）；与专家轴的整合（`gaps`）。

## 2 唯一里程碑

**输出 / 验收口径的家**：`project.yaml::milestone`（原文在那里，本文件不转述判据数字）。
本节只登记进度：

- 四臂最小判决（`D` / `K50` / `K25` / `F50`）**已交付并判「条件性有害」**（`WP1`，实验 `2644`）；
- 里程碑验收里「稀疏相对 dense 的代价与算力节省同框可分」与「通道使用率非坍缩」两条**已闭**；
  剩下的那条实质缺口是**「条件选通道」这个机制本身有没有被否证**——`WP1` 只否证了「无参数、按
  `|pre-activation|` 选」这一形态。⇒ `WP2`（可学习选择器）就是闭这一条的唯一便宜路径，
  也是本格达成里程碑的最后一步。

## 3 工作包

owner = `axis3-lead`（轴 3 宽度轴 lead）；writer = `axis3-w3.1.5`（3.1.5 worker，单一写入者）；
reviewer = `axis3-reviewer`（fresh 独立上下文，不得为 writer，复核排在 writer 停止之后）；
integrator = `tree-integrator`（本轴合并与跨轴一致性核对由 `axis3-lead` 代行）。角色定义见
`project.yaml::role_map`；逐包的 `what/goal/means/acceptance/evidence/review/status` 原文在
`project.yaml::work_packages`，本表只给计划口径的 owner / 验收 / 预算 / 状态。

| WP | 交付物 | 验收（可观测） | 预算 | 状态 |
|---|---|---|---|---|
| **WP1** 四臂最小判决（`D`/`K50`/`K25`/`F50`，13M matched-step） | `experiments/2644-channel-sparsity-13m/`（四臂 + 主判两臂 seed 1） | 四臂总参数逐位相等；稀疏相对 dense 的代价与算力节省同框；坍缩门齐备；n≥2 | 3 臂 ≈24 min GPU + 评测 | 被否证（判词见 `project.yaml::conclusion`，读数见该实验 `analysis.md`） |
| **WP2** 可学习选择器一臂（打分器 + 负载均衡项，`L50`），与无参数 `|pre|` 规则（`K50`）和稠密（`D`）在**同一每位置算力/参数预算**下比 | `experiments/2687-learned-channel-selector-13m/`（新训 `L50` × 2 seed + 复用 `D`/`K50`/`F50` 共 5 条重评读数） | 契约先绑后跑（`prereg_binding.json`）；四臂总参数逐位相等（7 044 608）、每 token MACs 逐位（`L50` = `K50`）；`D` 臂前向与 stock 逐位相同、`L50` 初始选择集 = `K50`（等值门）；G1 活性门 + G2/G3 质量门按预登记映射出**一个**判决（机制不成立 / 本实现不成立 / 本尺度不可判）；负结论 n≥2 | 2 臂训练 ≈16 min GPU + 5 支重评 ≈8 min（单臂 timeout 20 min） | 见 `project.yaml::work_packages[WP2].status` |

**WP2 为什么就是分诊表那一条动作**：`atlas/triage.md` §2b 给本格点的是「`3.1.5` 后续：可学习选择器
一臂」（分 2.0），判据是 D=2 / U=2 / C=2，动作是「补可学习选择器一臂（打分器 + 负载均衡项），
把结论从『本实现』推向『机制』」。本实验与之一致：**不加新臂、不改判据形态**（同一 0.014 配对带 +
率类 CI），只把「选择信号」这一个因素从无参数规则换成可学习打分器，并把参数/算力预算对齐。

## 4 证据指针

- 研究项事实与判据线：`axes/3-width/3.1.5-channel-sparsity/project.yaml`
  （`milestone` / `work_packages` / `conclusion` / `failure_analysis` / `cost_rows` / `gaps`）。
- `WP1`：`experiments/2644-channel-sparsity-13m/{README.md,analysis.md,results.json}`（判词与读数；
  预登记绑定为历史 `unbound`——该实验的读数入库早于 `tools/prereg_bind.py` 引入）。
- `WP2`：`experiments/2687-learned-channel-selector-13m/{README.md,config.json,run.py,analysis.md,results.json}`
  ——契约与判据线在 `README.md` §4、判据常数在 `config.json::criterion`、单测与门在
  `test_2687_learned_selector.py`；定序证据 `prereg_binding.json`（绑定早于读数）。
- 跨格排序与该格动作：`atlas/triage.md`（§2 该行 + §2b 候选）；结论索引行：`atlas/FINDINGS.md`。
- 代价事实（生成物，不手改）：`atlas/cost-table.md`；复现清单：`atlas/repro.md`。

## 5 状态（就地更新）

- `WP1`：**被否证**（「无参数、按 `|pre-activation|` 选」相对「固定砍一半」更差 +0.0617 bpc / 4.4× 带）。
- `WP2`：执行中（契约已写、单测 9 条全绿；训练与重评在 GPU 队列里）。判决落
  `project.yaml::conclusion` 与 `experiments/2687-.../analysis.md`，本文只跟着改这一行的状态词。
