---
id: 2655-layer-capacity-13m
axis: "2 深度轴"
node: "2.3.1 逐层容量变化"
date: "2026-10-14"
status: preregistered
---

# 2655 · 逐层容量 schedule 最小判决（固定总参数 / 固定 FFN 预算）· 13M · dense

> 预登记契约（`SCHEMA.md` §6 九节）。判据线 P1–P6 **写在跑之前，跑完不得改**（`SCHEMA.md` §8）。
> 口径（`TICKET.md` §9.1）：判据只用**等效理论算力**（MACs/token、layer-passes/token）；
> **墙钟不得进判据或 `cost_model`**（可作过程量留在产物里）。

## 1 命题

在**总参数逐位相等**、且**每 token FFN 等效 MACs 逐位相等**的前提下，把每层 FFN 隐藏宽
沿深度重分配（前大后小 / 逐渐放大）相对 uniform，能否把 clean bpc 拉低越噪声带。
可证伪：若两者相对 uniform 的配对 Δbpc 都落在 ±0.014 内 ⇒ 逐层容量分配在本尺度**不是**
独立于总容量的设计维度，应并入宽度轴读数（`2.3.1.falsification`）。

## 2 单因素声明

唯一变量 = **每层 FFN 隐藏宽 `h_l`**（引擎 `DepthARConfig.mlp_hidden_sched`，ticket 67 E2 实装）。
基底 = 主流 stock decoder-only（pre-LN、单残差、`--dense --topology sequential`、tied 读出）。

- 全部臂 `Σ_l h_l = 8192`（逐位相等）⇒ FFN 参数 `Σ_l (2·h_l·d + h_l + d)` 与每 token FFN MACs
  `Σ_l 2·h_l·d` **逐位相等**；attention / embedding / LN / 读出参数逐位相同。
- 逐层容量**只由 CLI 表达**（`--mlp-hidden-sched`），无实验本地模型类、无共享代码改动 ⇒
  「训练整段走共享代码」。
- **单因素无法独立生效的部分**：本格只改 FFN 宽度，不改注意力头数 / d_model / 分辨率
  （后者是 `2.3.2` 的轴，未测，见 `gaps`）。

## 3 指标表

| 维度 | 内容 |
|---|---|
| 解决什么问题 | 「每层同等容量」是否为无根据的默认；容量该不该按深度重分配 |
| 代价 | 各臂总参数逐位相等、每 token FFN MACs 逐位相等 ⇒ **代价面中性**；真实代价是**层间不等宽**导致的层间流水节拍差（须以每层 FLOPs 分布评估，本实验落 `ffn_macs_per_token_by_layer`） |
| 有效性 | 相对 uniform（stock）的配对 Δbpc，0.014 带（限配对 Δ、≥5 窗口） |
| 收益与用法 | 若 `front-big`/`growing` 显著更优 ⇒ 深度形状是可兑现的设计维度（同预算免费增益）；若无差别 ⇒ 结论「本尺度中性」，不再追 |
| 可控性 | schedule 是显式整数列表，逐层可审计（`hidden_by_layer` 实测 == schedule） |
| 副产品 | 逐深度 bpc 曲线、每层 FFN 相对幅度（活性）、loss 斜率 |

## 4 判据线（预登记，不可改）

符号约定（`SCHEMA.md` §17.3）：`Δ = bpc_对照 − bpc_实验`，正 = 实验臂更好。

- **P1** `Δ_front = bpc_uniform(s0) − bpc_front-big(s0)`；**P2** `Δ_grow = bpc_uniform(s0) − bpc_growing(s0)`。
  Δ ≥ +0.014 ⇒ 该非均匀臂可分更优；Δ ≤ −0.014 ⇒ 可分更差；|Δ| < 0.014 ⇒ **不可排序（H0）**。
- **P3** 逐窗配对 Δ（5 窗口 × 4 批，所有臂共用同一 generator seed 的同一批序）+ **窗间 SE**；
  窗间 SE 是抽样噪声，与 0.014 的跨 run 训练漂移带**分开报**。
- **P4** 等预算门：各臂 `Σh`、FFN 参数、FFN MACs **逐位相等**；否则该臂作废（不是同预算比较）。
- **P5** 引擎等值门：显式 `uniform` schedule 相对引擎缺省（`--mlp-mult 4`）**逐位一致**
  （模型层：state_dict + 前向逐位，见 `smoke.py` 单测；训练层旁证：与同配方历史臂之差 < 0.005）。
- **P6** 活性（`SCHEMA.md` §10）：逐层 `mlp[0].weight.shape[0]` 逐位等于 schedule；
  逐层 `‖mlp_out‖/‖mlp_in‖ > 0`（每层 FFN 真的参与前向，不是死路径）。
- **该跌 / 该不变**（本项 `objective_criteria`）：clean bpc **该跌或明确无差别**；
  **必须不变**：总参数量、Σh、FFN MACs、因果性（顺序 block 不变）。

## 5 对照与方差

- 对照 = 同实验 `uniform` 臂（= stock uniform 全深度，引擎缺省形态）；另有历史同配方锚
  （2613 sequential-s0 = 1.5254、2641 U4-s0 = 1.5314）作训练层旁证。
- seed 数 = 1（s0）；噪声带 0.014（`docs/methodology.md` §2 跨 run 界）。n=1 ⇒ 判词只到
  「可排序 / 不可排序」二分，不做 trend 外推。
- 方差界口径：bpc 类 = 0.014 带（限配对 Δ）；同 checkpoint 内的窗间 SE 另报（P3）。

## 6 成本

- `gpu_wall_budget: 35 min`（含冒烟 + 3 臂 × 12k 步 + 评测）。
- 硬件 = 单卡 RTX 4090（24GB）共享；GPU 按 `SCHEMA.md` §20 取锁、**按臂释放**（§20.1，臂间 ≥60 s）。
- 峰值显存目标 ≤5GB net（13M 模型，batch 32×256）。
- 每臂等效算力 = 12k × 32 × 256 = **98.3M token**；三臂合计 **294.9M token**。
- 可中断性：`run.py --arms <单臂>` 可续；结果**按臂增量落盘**（`results_<arm>_s0.json`）。
- 冒烟计划：`smoke.py`（CPU，等值门 + 活性 + 变异探针）→ 20 步 CLI 小跑。
- 预算不够时**砍 arm 不砍判据线**：先砍 `growing`（只留 uniform 对 front-big），把「逐渐放大未测」写进 `gaps`。

## 7 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M train / 后 10M eval，vocab=206，与全轴同切分）。
- 入口：`run.py`（`--arms` / `--no-train` / `--aggregate` 一条命令复现全臂）。
- 等值门与冒烟：`smoke.py`；单测：`test_2655_layer_capacity.py`（pytest 可跑）。
- resolved config：本目录 `config.json` + 每臂训练产物内 `results.json.config`。
- 落盘：`results_<arm>_s0.json`（逐臂）、`results.json`（汇总）、`run_log.txt`、
  `analysis.md`、`MODEL.md`、`runs/<arm>-12k-s0/`（`model.pt`、`train_log.jsonl`、`results.json`）。
  本目录根下不再保留 `uniform-s0-*.jsonl/json` 逐臂训练副本（它们与 `runs/uniform-s0-12k/` 下
  同名文件逐字节相同，不属本清单；已按复核 finding 2.3.1-N2 从树中移除，内容仍在 `runs/`）。

## 8 失败决策树（按序排除）

1. **实现**：`hidden_matches_schedule=False` 或 P6 活性为零 ⇒ 开关没到模型，臂作废，先修接线。
2. **测量**：`clean_bpc_two_paths_agree=False`（窗口法 vs `depth_curve` 不一致）⇒ 评测路径有错，重测。
3. **规模**：两臂都不可排序且 loss 斜率仍显著非零（未收敛）⇒ 只作快照结论，标注「疑为收敛加速」，
   不外推；登记 2× 步数点作下一步。
4. **设计**：全部判据齐备且不可排序 ⇒ 如实判「逐层容量在本尺度无独立增益」，写进 `2.3.1.conclusion`。

## 9 产出

`results.json` / `results_<arm>_s0.json` / `run_log.txt` / `analysis.md` / `MODEL.md`；
`project.yaml` 的 `status`/`conclusion`/`failure_analysis`/`conclusion_review`/`argument`/`gaps`/
`experiments`/WP `status`/`change_log` 更新；`.scratch/61-axis2-depth/WP-2655-REPORT.md`。
独立复核（fresh 上下文，`SCHEMA.md` §14/§14b）**排在 writer 停止之后**，本目录只写「独立复核未做」。
