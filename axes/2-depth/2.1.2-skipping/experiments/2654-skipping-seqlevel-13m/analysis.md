# 2654 · 跳层的序列级重测 · 判读（analysis）

## 0. 人话摘要

**本轮没跑成训练：冒烟拿到过锁并跑了一次，`layerdrop-pt` 臂的活性门抓红，launcher 就此退出。**
从 `22:03` 起 `/tmp/pathlm_gpu.lock` 一直在别的 agent 手里轮转（`wp6.3.2-*`、
`wp2655-layer-capacity`、`2651`、`2652`），有界 launcher（`run_remaining.sh`）按 `SCHEMA.md` §20
礼貌等待，前 3 轮各等满 10 min 超时；第 4 轮（`[2026-10-14T22:37:39]`，`run_log.txt:5`）取到锁
并跑了 20 步冒烟：`stock` / `skip-mid` / `layers6` 三臂的活性门**过**（实测 layer-passes
8.0 / 6.0 / 6.0，分别见 `run_log.txt:576` / `:1144` / `:1618`），但 `layerdrop-pt` 实测
layer-passes **8.0** 与设计 7.2 不符（|Δ| = 0.8 ≥ 0.2，断言抛自 `run.py:451`，`run_log.txt:2194`），
`[2026-10-14T22:38:50] SMOKE FAILED`（`run_log.txt:2195`）⇒ launcher 按 `run_remaining.sh:53`
的 `|| { echo …; exit 1; }` 退出。

按判据线 `README.md` §4-P1「实测 layer-passes 与设计值不一致（|实测 − 设计| ≥ 0.2）⇒ 该臂作废」
与 `README.md` §8 决策树第 2 条，`layerdrop-pt` 臂**作废待修**：须先在 CPU 上修 `SkipBlock`
的逐 token 跳层接线/记账、让活性门转绿，才谈得上花 GPU。因此**四臂 12k 一个都没训**、
`results.json` 不存在（冒烟只写 `smoke/runs/<arm>-0k-s0/`），判据线 P0–P5 **一条都没有读数**。

**这不是科学结论，也不是失败读数** —— 事实上是两件事：「实现未过活性门」（与 GPU 无关）
与「资源未获得」。判据线未被触碰（预登记在 `README.md` §4，逐字未改）。

## 1. 状态

| 项 | 状态 |
|---|---|
| 实现（工厂替换 + `SkipBlock` 逐 token 旁路） | ✅ 落盘，CPU 单元验证通过（恒等旁路语义、layer-passes 账、state_dict 重建一致） |
| 预登记契约 `README.md`（SCHEMA §6 九节） | ✅ 落盘，判据线 P0–P5 预登记 |
| `config.json` / `run.py`（`--arms` 续跑 + 逐臂增量落盘） | ✅ 落盘 |
| 20 步冒烟 `run.py --smoke` | ❌ **已跑并失败**：2026-10-14T22:37:39 第 4 轮取到锁开跑（`run_log.txt:5`）；`stock` / `skip-mid` / `layers6` 活性门过（lp = 8.0 / 6.0 / 6.0），`layerdrop-pt` 活性门红（实测 lp = 8.0 ≠ 设计 7.2，断言 `run.py:451`，`run_log.txt:2194`）⇒ 2026-10-14T22:38:50 `SMOKE FAILED`（`run_log.txt:2195`） |
| 四臂训练 + 序列级评测 | ❌ 未跑（冒烟红门后 launcher 即 `exit 1`，`run_remaining.sh:53`；原因不止「无 GPU 锁」） |
| 判据 P0–P5 读数 | ❌ 无读数 |

## 2. 方法（已实现；20 步冒烟跑过一次、四臂 12k 未跑）

见 `README.md` §2 / §4。要点：四个 13M/12k/seed0 臂只改跳层策略 ——
`stock`（layer-passes 8）、`skip-mid`（0-indexed block {3,4} = 人类第 4、5 层恒等旁路，
layer-passes 6）、`layers6`（`--layers 6` 同预算固定深度对手）、`layerdrop-pt`
（逐 token 逐层独立 Bernoulli(0.1)，layer-passes ≈7.2）。跳层 = 该 token 在该层的输出 = 输入，
`out = where(skip, x, block_out)`，因果不破。

## 3. 数字表

**无实测数字**（未训练）。表位保留，等续跑臂落盘后填：

| 臂 | clean bpc | layer-passes/token（实测 / 设计） | 参数量 | 逐层跳过占比 |
|---|---|---|---|---|
| stock | — | — / 8.0 | — | — |
| skip-mid | — | — / 6.0 | — | — |
| layers6 | — | — / 6.0 | — | — |
| layerdrop-pt | — | — / 7.2 | — | — |

冒烟（20 步、远未收敛）另落了读数，只作过程留痕、**不作任何判据读数**：`run_log.txt:576` /
`:1144` / `:1618` 记 `stock` bpc 5.157922 / `skip-mid` 5.310313 / `layers6` 5.290606，
`layerdrop-pt` 在活性断言处中止、未打印该臂读数（训练产物在 `smoke/runs/`）。

## 4. 口径标注

- clean bpc 口径 = 最终深度读出、序列级、`EVAL_SEED=2`、20 窗口 × batch16 × seq256
  （与 `2613` / `2641` 的 `depth_curve` 最终深度同口径）。
- 判据符号：`Δ(x vs y) = bpc_y − bpc_x`（正 = x 更好）。bpc 判据走 0.014 带（配对 Δ、≥5 窗口）；
  率类指标（跳过占比、分布一致率）走配对 bootstrap / Wilson 95% CI。
- **墙钟不进判据**；GPU 阻塞只记为资源事实。

## 5. 残余缺口（下一步最小动作）

1. **先修实装、再排 GPU（最小实验，纯 CPU、无需 GPU 锁）**：按 `README.md` §8 决策树第 2 条查
   `SkipBlock._sample_skip` —— `drop` 模式的跳层掩码须**逐 token 独立** Bernoulli(0.1)
   （不是逐层/逐 batch 采一次）、`torch.where` 须以**层输入**做恒等旁路、layer-passes 记账须统计
   该 token 实际经过的非跳层数；修完后 `python3 run.py --smoke --steps 20` 必须**四臂全绿**
   （`stock` / `skip-mid` / `layers6` 已绿，只差 `layerdrop-pt` 的 lp ≈ 7.2 ± 窗口间 SE）。
2. **冒烟全绿后才排四臂**：`bash run_remaining.sh`（或
   `python3 run.py --arms stock,skip-mid,layers6,layerdrop-pt`）—— 优先序
   `stock → skip-mid → layers6 → layerdrop-pt`，预算不够先砍 `layerdrop-pt`（README §4-P4）。
   预计 4 臂 × 12k 步 ≈ 30–45 min GPU；等效算力账 = 每臂 12k × 32 × 256 = **98.3M token**、
   四臂合计 **393.2M token**。
3. 臂齐后填本文件 §3 表 + 写 P2 判决线判词（由后续 worker / lead 收口）。
4. 未做（修好前不做）：`layerdrop-pt` 的训练/评测跳层分布一致性列（README §4-P1b）；已做并失败：
   `--smoke` 的 GPU 端活性断言（见 §1），修复后必须重跑。
