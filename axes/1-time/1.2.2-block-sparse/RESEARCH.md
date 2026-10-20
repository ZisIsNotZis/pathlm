# RESEARCH · `1.2.2` 块稀疏 / 条带 / 局部+全局混合（Longformer 式）

研究级**计划视图**：目标、目的、唯一里程碑、工作包（owner / 验收 / 预算 / 状态）与证据指针。
**事实与数字的家不在这里**——研究项事实（判据线、结论、`cost_rows`、`gaps`）在 `project.yaml`，
读数在各实验目录的 `results.json` / `analysis.md`。本文件只放指针，不复制读数。

## 0 这是什么

一次前向里，一个 token「能看到谁」。stock 全因果注意力让每个位置看到全部历史（`T²` 成本）；
**预先固定、非均匀**的可见掩码想用远少于 `T` 的可见键覆盖整条序列：最近的局部窗口之外，只留少数
被精心放置的位点（最前面的全局槽、或一块块对齐的远块）。本项判的是：**这种「非均匀性」本身值不值**
——在同样的每位置字节预算下，它比把同样的键全花在最近位置（均匀条带）更好吗？

## 1 目的（对着谁）

- **痛点**：长上下文里注意力与 KV 读带宽按 `T`（时间轴）与 `T²`（算力）涨；业界的主流省钱办法
  都是「非均匀可见集」（Longformer 的 local+global、BigBird 的 block sparse + random blocks、ETC），
  但**同预算下非均匀本身是否买得到质量**这一层，通常在论文里被「相同 FLOPs 下更好」的端到端数字一起报。
- **对标**：Longformer（local window + global token）、BigBird（块稀疏 + 条带 + 全局）、ETC，
  以及本仓自己的跑分线（stock 全因果 13M B0 = 1.5074 bpc 快照）。业界条目按 `记忆·未核` 标注。
- **本项不做**：组合实验（窗口 × 量化 × 检索，见 `harness/SCHEMA.md` §4 留白）；更远滞后的块、
  随机块、以及 T≥2048 的第二规模点（`gaps`）。

## 2 唯一里程碑

**输出**：一份「非均匀可见结构 vs 全因果」的同预算判决包——`{全因果, Longformer 式 local+global,
BigBird 式块稀疏, 线性核}` 在**同一条每位置字节轴**上的 clean bpc + 长程召回 + 每 token 等效 FLOPs，
含 n≥2。

**验收口径的家**：`project.yaml::milestone.acceptance`（原文在那里，本文件不转述判据数字）。
本节只登记进度：**质量腿的三个固定模式已齐并各自出判词**（`2677` 的两个实例 + `2683` 的块粒度），
仍缺**线性核无伤质量点**（`WP2`）与率类召回腿。

## 3 工作包

owner = `axis1-lead`（轴 1 时间轴 lead）；writer = `axis1-w1.2.2`（单一写入者）；reviewer =
`axis1-reviewer`（fresh 上下文，不得为 writer）；integrator = `tree-integrator`。角色定义见
`project.yaml::role_map`；逐包的 `what/goal/means/acceptance/evidence/review/status` 原文在
`project.yaml::work_packages`，本表只给计划口径的 owner / 验收 / 预算 / 状态。

| WP | 交付物 | 验收（可观测） | 预算 | 状态 |
|---|---|---|---|---|
| **WP1-a** local+global 对均匀条带（同 24 键） | `experiments/2677-local-global-vs-band-13m/`（4 条读数 + 1 个宽剂量点） | 两臂逐行可见键数逐行相等；两 seed 各自越 0.014 带且同向；复用臂等值门；参数量逐位相等 | 2 臂 ≈16 min GPU + 评测 ≈3 min | 已完成（n=2，判「非均匀更差」） |
| **WP1-b** 块条带对均匀条带（同 24 键）＝**本项 `gaps` 第一条与分诊表的单一动作** | `experiments/2683-block-stripe-vs-band-13m/`（2 条新臂 + 4 条复用臂 + 1 条 `full`） | 逐行等键 `mismatch_rows == 0`；训练/评测掩码同源三道；两 seed 同向且都越带（或都落带内）；复用臂逐位复现 `2677` | 2 臂 ≈18 min GPU + 评测 <4 min（单臂 timeout 20 min） | 已完成（n=2，判「无质量增量」；判词见 `project.yaml::conclusion`，读数与归因见该实验 `analysis.md` §3 / §7） |
| **WP2** 线性核无伤质量点 | `experiments/2616` / `2637` 的读数只作负观测 | 线性臂 `gnorm` 健康、跳步 0、`den` 不触 clamp；给出无伤 Δbpc 或明确记为未取得 | 2 臂 × n≥2 ≈40 min GPU | 未开始 |

**WP1-b 为什么就是分诊表那一条动作**：`atlas/triage.md` §2 给本格的建议动作是「补一条最小实验：
只做块稀疏掩码那一臂组；条带 / 局部+全局不必重跑」——本实验正是它：一条新臂组（块条带 × 2 seed）
对同字节的条带臂（复用 `2677`）与稠密臂（复用 `2610` 的 `full`）。**没有任何替换**：动作、判据形态
（同每位置字节、配对 Δ、0.014 带只用于配对读数、n≥2）与分诊表的描述一致。

## 4 证据指针

- 研究项事实与判据线：`project.yaml`（`milestone` / `work_packages` / `conclusion` / `cost_rows` / `gaps`）。
- WP1-a：`experiments/2677-local-global-vs-band-13m/{README.md,analysis.md,results.json}`（预登记绑定
  `prereg_binding.json` 为历史 `unbound`：该实验的读数入库早于 `tools/prereg_bind.py` 引入）。
- WP1-b：`experiments/2683-block-stripe-vs-band-13m/{README.md,config.json,analysis.md,results.json}`（判词与门、六问都在 `analysis.md`；契约外归因读数 `attribution.json`，不参与判词）；
  掩码实现与自证 `block_sparse.py`（`python3 block_sparse.py` 打印逐行等键与指纹）；
  契约与定序证据 `prereg_binding.json`（绑定早于读数，6 份契约文件含掩码与判据实现）。
- WP2：`experiments/2616-linear-attention-13m/analysis.md`、`experiments/2637-linear-attn-quality/analysis.md`
  （带伤 / 核已改的负观测，不是规范质量点）。
- 跨格汇总：代价事实见 `atlas/cost-table.md`（由 `project.yaml::cost_rows` 生成）；复现三态见
  `atlas/repro.md`（由 `tools/inventory_runs.py` 生成）；结论索引见 `atlas/FINDINGS.md`；
  本格在下一次分诊里的位置见 `atlas/triage.md`。

## 5 当前状态与残留

- 状态字的家是 `project.yaml::status`；本文件不重述。
- 三个固定模式的**质量腿**已齐：局部+全局（`2677`，判更差）、块条带（`2683`，判词见
  `project.yaml::conclusion`）、均匀条带（同键对照）。同一字节水平（294,912 B/位置）上三者可比。
- 残留缺口（原文在 `project.yaml::gaps`）：线性核无伤质量点（WP2）；率类长程召回腿（两轮实验都只
  指针引用 `2662` 的落盘，不重跑）；真稀疏核把纸面 FLOPs 兑现为等效节省；T≥2048 的第二规模点；
  更远滞后 / 随机块的块稀疏变体（本实验只测 `LAG=4`，是最有利于非均匀性的设定）。
