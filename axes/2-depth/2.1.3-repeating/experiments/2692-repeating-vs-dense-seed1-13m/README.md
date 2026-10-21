---
experiment: 2692-repeating-vs-dense-seed1-13m
grid: axes/2-depth/2.1.3-repeating
date: 2026-10-21
engine: pathlm/depth_ar.py（depth-ar/dense）+ train_depth_ar.py（共享 main）
scale: 13M（d=256 / seq=256 / heads=8 / mlp_mult=4；层数见臂表）
---

# 2692 · 循环深度四臂的 **seed 1 复本** · 13M

## 1 命题

`2691` 在 seed 0 上判了「同算力」与「同参数」两问（`repeat2x` 同算力更差 0.040、同参数更好
0.021，诊断臂把亏损归到「6 组参数」）。本实验是同一设计、同一判据线的 **seed 1 复本**：
它只回答一个问题——**这两条可排序的 Δ 换一个训练随机种子还同向、同量级吗**。
两 seed 的合并判读写在 `2691` 的 `analysis.md`（n=2），本目录只落自己的读数。

## 2 单因素声明与设计条件

**唯一变量 = 训练随机种子**（相对 `2691` 的 seed 0）。臂、层调度表、数据、步数、批大小、
lr、评测入口、判据线与 `2691` 逐字相同（`2691` 的 `README.md` §2 / §4 是本实验的完整契约）：

| 臂 | 组数（建几层） | 每 token 层传递 | 每 token MACs（解析） | 参数（解析） | 角色 |
|---|---|---|---|---|---|
| `dense6` | 6 | 6 | 5,480,960 | 5,251,584 | 问二对照：同参数稠密 |
| `dense12` | 12 | 12 | 10,909,184 | 10,384,896 | 问一对照：同算力稠密 |
| `repeat2x` | 6 | 12（1–6 再来一遍） | 10,909,184 | 5,251,584 | 主臂：全栈重复两遍 |
| `repeat2x-tail` | 6 | 12（后 3 层各 3 次） | 10,909,184 | 5,251,584 | 诊断臂：只重复后半段 |

设计条件四条与 `2691` 相同（变化空间、等参数/等算力对照都是真训练、第二遍是唯一被优化的路径、
**不加步嵌入**）。实现 = 实验内工厂替换（monkey-patch `train_depth_ar.DepthARModel` 返回
`RepeatModel`），`pathlm/**` 与 `train_depth_ar.py` 零改动；`RepeatModel` 不新增参数。

## 3 指标表

与 `2691` 相同：clean bpc（最终深度读出、序列级、越低越好）、每 token 解析 MACs、参数量（实测）、
峰值显存（实测）、逐深度 bpc、层调用次数、`gnorm` max 与跳步数；**墙钟只申报、不进判据**。

## 4 判据线（预登记、不可事后改）

符号：`Δ(候选 vs 对照) = bpc_对照 − bpc_候选`（**正 = 候选更好**）；bpc 判据用**配对逐窗口 Δ**
（同 `EVAL_SEED=2` ⇒ 各臂同位置）、**0.014 带**、20 窗口。墙钟不进判据。

- **P0 引擎等值门**：用本实验的评测代码重评 `2654` 的 stock checkpoint
  （`axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m/runs/stock-12k-s0/`），
  其 clean bpc 与其落盘值之差 `< 0.002`；不过门则其余读数不下判。
- **P1 活性门**：每臂每前向的层调用次数 = 设计传递次数；恒等调度与 stock 逐位等值；
  循环臂「完整调度」对「只跑前 6 次」的最终 logits 偏差 > 0。
- **P2 判据（与 `2691` 同一组比较）**：`Δ(repeat2x vs dense12)`（问一，同算力）、
  `Δ(repeat2x vs dense6)`（问二，同参数）、`Δ(repeat2x-tail vs repeat2x)`（诊断，同参数同算力）。
- **P3 复本判据（本实验特有）**：把本实验的三条 Δ 与 `2691` 的同名 Δ 并排，按**同向 ×
  同量级**读：两 seed 都越带且同号 ⇒ 该条从 n=1 抬到 n=2；一 seed 越带、另一 seed 落带内 ⇒
  **该条降级为「跨 seed 不稳定」**，不得读成「可排序」；两 seed 反号 ⇒ 判「不可复现」。
  这一条**不改** `2691` 的任何判据线，只决定「n=1 的结论能不能说成跨 seed 成立」。
- **P4（预算不够时砍什么）**：先砍 `repeat2x-tail`（诊断写进 `gaps`），不砍判决三臂与判据线。
- **P5（不外推）**：结论不越 13M / 12k 步 / T=256 / enwik8 / seed ∈ {0,1}。

## 5 对照与方差

与 `2691` 相同：方差带 0.014 bpc（限配对 Δ、≥5 窗口；本实验 20 窗口），另报
`delta_window_se` 与 `delta_halves`；跨 seed 的读法见 §4 P3。

## 6 成本

- **gpu_wall_budget: 40 min**（P0 重评 ≈1 min + 四臂 12k 步训练 ≈29 min + 四臂评测 ≈4 min）。
- 单臂 timeout 20 min（launcher `timeout 1200`）；显存峰值 ≤5 GB。
- 可中断性：按臂增量落盘（`completed_arms` / `pending_arms`），中断后 `--arms <剩余臂>` 续跑。
- GPU 纪律：`/tmp/pathlm_gpu.lock` 按臂取/放（`harness/SCHEMA.md` §20.1），等锁上限 10 min/轮。

## 7 数据与脚本清单

- 数据：`data/enwik8_full.npz`（同 `2691`；切分缓存 `.tmp/enwik8`）。
- 入口：`run.py`（`--arms` / `--no-train` / `--smoke` / `--p0-only` / `--aggregate`）；
  驱动：`run_arms.sh`；resolved config：`config.json`；逐臂 run config 在
  `runs/<臂>-12k-s1/results.json` 的 `config`。
- 落盘：`results.json`、`runs/<臂>-12k-s1/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}`、
  `run_log.txt`；单测 `test_repeat_schedule.py`（与 `2691` 同一份，CPU 秒级）。

## 8 失败决策树

1. **测量**：P0 不过门 ⇒ 核评测口径与重建配置（同 `2691` §8 第 1–2 条）。
2. **实现**：P1 不过 ⇒ 查调度表与 `RepeatModel.forward`（单测钉住恒等等价）。
3. **复本**：P3 判「跨 seed 不稳定」⇒ 按 P3 如实降级 `2691` 的对应结论，不调判据线。
4. **规模**：四臂都异常高 ⇒ 查 loss 斜率与 gnorm，必要时**同时**缩步数保臂间同刻度，不砍臂。

## 9 产出

- 本契约 / `config.json` / `run.py` / `run_arms.sh` / `test_repeat_schedule.py` /
  `prereg_binding.json`。
- `results.json`（逐臂增量 + P0/P1 + 三对比较 + 逐深度曲线）/ `run_log.txt` / `MODEL.md`。
- `analysis.md`（复本判读：两 seed 并排、P3 的三条判定、口径与边界）；合并判读在 `2691`。
