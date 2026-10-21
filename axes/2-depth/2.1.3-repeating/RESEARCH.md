# 2.1.3 循环深度（同权重重复前向）· 研究级契约（目标 / 唯一里程碑 / 工作包 / 证据指针）

> **本文件是一份 projected 视图，不是事实的家。** 目的、判据、结论、里程碑与工作包字段的
> 唯一之家是 `project.yaml`（本目录）；读数之家是实验目录的 `analysis.md` / `results.json`。
> 本文件只做 `project.yaml` 做不到的三件事：把**派发口径**（工作包的 owner / 预算 / 复核人）
> 写清、把**证据指针**排成一条可走的路、把纪律条款映射到具体命令。**数值一律不在此复制**。

## 1 目标与目的

- **它是干什么的**：让**同一组层参数**在同一个 token 上被重复前向若干遍，用**算力**换**参数**
  —— 参数受限时拿到更大的有效深度，推理期还能按重复次数调算力。见 `project.yaml::innovation`。
- **为什么要做（判什么）**：它要把「深度」与「参数」两个旋钮**解耦**：
  **若同每 token 算力下循环层不输于同算力稠密**，说明参数是瓶颈（多给参数不划算、重复更划算）；
  **若循环更差**，说明层间异质性（每层一套独立权重）在质量上是必需的，这条省参数的路不改变
  「算力该花在哪」的判断。见 `project.yaml::purpose` / `::falsification`。
- **它不是什么**（非目标）：不是「省算力」的机制（它花更多算力），也不是组合实验库
  （组合默认留白，`harness/SCHEMA.md` §4）。与 `2.2.5` 权重共享同题但不同判据方向，两格
  `gaps` 互指、不合并。

## 2 唯一里程碑（验收）

一字不改地取自 `project.yaml::milestone`：一份「循环深度值不值」的判决——在**同每 token 激活
算力**与**同参数量**两个口径下，`repeat2x`（6 组参数跑 12 次）相对同算力稠密（12 层）与同参数
稠密（6 层）的序列级 bpc 与代价列（参数 / 每 token MACs / 峰值显存），含 `n≥2` 的重复次数与
一次「只重复后半段」的归因臂；两问各自有明确答案。

**当前状态**：唯一之家是 `project.yaml::status`。本文件不复制判词。

**工作包闭环方式**：唯一里程碑由一个工作包落地（WP1，见 §3），实验目录
`experiments/2691-repeating-vs-dense-13m/`；判据线 P0–P5 在 `README.md` 里预登记，
训练前用 `python3 tools/prereg_bind.py <实验目录> --contract README.md run.py config.json`
冻结（哈希进 `prereg_binding.json`），读数落盘之后不得改判据。

## 3 工作包（owner / 验收 / 预算 / 状态）

每条工作包的**内容、验收与状态**的唯一之家是 `project.yaml::work_packages`；本表只放
**派发口径**（owner / 预算 / 复核人），不复制任何事实。

| id | owner | 预算（墙钟 / GPU） | 复核人 | 状态（只放指针） |
|---|---|---|---|---|
| WP1 | 轴 2 深度轴 lead（本格单一写入者） | 4.5 h 墙钟 / ≈70 min GPU（四臂 × 12k 步 + 评测 + P0 重评），臂级最多 8 次重试、单臂 timeout 20 min | fresh 复核者（独立上下文，**未派**；本轮判词为 writer 自评，见 `project.yaml::conclusion_review`） | → `project.yaml::work_packages[0].status` |

## 4 证据指针（按顺序走就能复核本项的全部结论）

| 结论 | 指针 |
|---|---|
| **契约与判据线（预登记、绑定后冻结）** | `experiments/2691-repeating-vs-dense-13m/README.md` §4；绑定记录 `experiments/2691-repeating-vs-dense-13m/prereg_binding.json` |
| **臂设计与单因素声明（含设计条件 4 条）** | 同上 §2；`experiments/2691-repeating-vs-dense-13m/config.json::arms` |
| **问一：同算力（`repeat2x` vs `dense12`）** | `experiments/2691-repeating-vs-dense-13m/analysis.md` §3 判据 P2-1；键路径 `results.json::comparisons.q1_same_compute` |
| **问二：同参数（`repeat2x` vs `dense6`）** | 同上 §3 判据 P2-2；键路径 `results.json::comparisons.q2_same_params` |
| **诊断：深度不足 vs 异质缺失（`repeat2x-tail`）** | 同上 §3 判据 P2-3；键路径 `results.json::comparisons.d1_tail_vs_repeat2x` / `::d2_tail_vs_dense12`；逐深度曲线 `results.json::arm_summary.<臂>.depth_bpc` |
| **代价列：参数 / 每 token MACs / 峰值显存 / 墙钟** | 同上 §5；键路径 `results.json::arm_summary`、`::arms.<臂>.macs_per_token` |
| **仪器可比（P0 重评历史锚）** | 同上 §2 判据 P0；键路径 `results.json::gates.p0_engine` |
| **干预活性（层调用次数 / 重复是否活着 / 恒等等价）** | 同上 §2 判据 P1；键路径 `results.json::gates.p1_liveness`、`::gates.structure`；单测 `experiments/2691-repeating-vs-dense-13m/test_repeat_schedule.py` |

## 5 纪律映射（本格点名的条款）

- **判据先绑后用**：`python3 tools/prereg_bind.py <实验> --contract README.md run.py config.json`
  在训练之前；`tools/check_tree.py` 的预登记机检覆盖「臂级读数也晚于绑定、代码哈希等于绑定值」。
- **方差带只用于配对读数**：0.014 bpc 带限**配对逐窗口 Δ、20 窗口**；率类/曲线类（逐深度 bpc、
  参数量、MACs、显存）只报数与口径（`harness/methodology.md` §2）。
- **墙钟不进判据**：只作过程量落 `results.json::arm_summary.<臂>.wall_minutes`。
- **资源纪律**：一次一个 GPU 作业；`/tmp/pathlm_gpu.lock` 按臂取/放（`harness/SCHEMA.md` §20.1）；
  单臂 timeout 20 min；卡上更高优先级的常驻项目（`local_lm_research` / `nomos` / `llama-server`）优先。
- **单 seed 的边界**：`n=1`（每臂一个训练 seed）⇒ 只做机制与方向判决，不做量级判决
  （`atlas/playbook.md` 定律 8 的方法论推论）。
