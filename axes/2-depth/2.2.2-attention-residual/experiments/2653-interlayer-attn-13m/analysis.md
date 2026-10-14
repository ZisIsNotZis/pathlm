# 2653 · 层间 causal attention（2.2.2）· 判读

状态（截至本轮窗口结束）：**实现 + 等值门/活性门/参数账全部做完并落盘；两臂训练各 12k 步中，
本轮因共享 4090 被外部任务占用（`/tmp/pathlm_gpu.lock` 被他人 launcher 持续取用）未能跑完，
数值列如实标「未测」**（契约允许：`TICKET.md` §9.2 WP-2653「砍 arm 不砍判据线」，
本轮砍的是 GPU 排期而不是判据线）。

## 人话摘要

把「每层只接上一层」换成「每层对**之前所有层输出**做一次 causal attention 聚合」
（越早的层也能被后面直接读到），这件事本实验已经**实装并证明接线正确**：
把聚合权重强制成「只取紧邻上一层」时，前向与损失跟原版单残差**逐位一致**（不是「差不多」，
是每一位都相同），所以这个改动**没有偷偷改别的东西**——唯一的差别就是聚合本身。
两臂的参数账、逐层注意力权重活性、以及等效算力代价都已实测：

| 臂 | 额外参数 | 每 token 额外 MACs（层间聚合） | 相对 stock 每层注意力 | 等值门 | 活性 |
|---|---|---|---|---|---|
| `attn-scalar`（q/k/v 恒等 + 每层一个温度 τ） | **+8** | 18 432 | **+0.59%** | ✅ 逐位一致 | ✅ 非退化 |
| `attn-qkv`（可学习共享 W_q/W_k/W_v + τ） | **+196 616** | 5 261 312 | **+167.3%** | ✅（同聚合） | ✅ 非退化 |

对照臂 = 2.2.3 的 `dense`（层间**线性**学习加权和，+36 参数，2613 s0 clean bpc 1.5246）与
stock 单残差（`sequential`，1.5254）。判据线（预登记、未改）：Δ = 对照 − 本臂，
|Δ| < 0.014 ⇒ 不可排序（判「在本尺度无净增量，退化为线性聚合」）。

**B 表（主判据，未测）**：两臂的 clean bpc 与配对 Δ 见表 B；本轮未跑出，故不填数字。

## A 表：实测（已落盘）

| 项 | `attn-scalar` | `attn-qkv` | 出处 |
|---|---|---|---|
| stock 参数量（同配置 sequential） | 6 962 688 | 6 962 688 | `smoke_report.json:C_param_accounting` |
| 本臂参数量 | 6 962 696 | 7 159 304 | 同上 |
| `params_added`（= 解析值） | +8 | +196 616 | 同上 |
| 额外 MACs/token（层间聚合，解析） | 18 432 | 5 261 312 | `interlayer_attn.aggregation_cost()` |
| 额外 / stock 每层注意力（3 145 728 MACs/token） | 0.005859 | 1.672526 | 同上 |
| 等值门（one-hot ⇒ 与 stock 逐位一致，前向 + loss） | ✅ `true` | ✅ 同一聚合路径 | `smoke_report.json:A_equivalence_gate` |
| trunk 参数逐位相同 | ✅ | ✅ | 同上 |
| 活性：层维权重非退化（k ≥ 2 的层 `w_immediate_prev < 1`、熵 > 0） | ✅ | ✅ | `smoke_report.json:B_liveness_gate` |
| 单测 | 9 passed | — | `test_2653_interlayer_attn.py` |

等值门探针（证明这条门会红）：不强制时前向**必须**与 stock 不同 —— `probe_free_weights_forward_differs=true`；
`MODE`/`FORCE_LAST` 在建实例时冻结（前向时读全局会让这条门静默失效，已修并有回归单测）。

## B 表：主判据（未测，如实标记）

| 臂 | clean bpc | Δ vs stock（2613 s0 = 1.5254） | Δ vs dense 线性聚合（= 1.5246） | 判词 |
|---|---|---|---|---|
| `attn-scalar` | 未测 | 未测 | 未测 | 待跑 |
| `attn-qkv` | 未测 | 未测 | 未测 | 待跑 |

- 对照口径（`SCHEMA.md` §9 三条）：①**实现层等值** = 上面的 one-hot 逐位单测（`smoke_report.json`）
  **且**两臂与 stock 走同一构造顺序（`tau`/`W=init` 不消耗随机数流）⇒ trunk 权重逐位相同；
  ②**配置层等值** = `run.py baseline_equivalence()` 与 2613 s0 的 resolved config 逐字段对照
  （d/layers/heads/seq/mlp_mult/dense/exit_mode/dropout/norm_cap/steps/batch/lr/seed/optimizer/
  data/train_tokens）；③**口径标注** = 2613 s0 与 2653 均为「全深度、无退出、teacher-forcing、
  序列级 bpc、批 16 × 20 批」，同口径。
- 噪声带 0.014（跨 run 训练漂移）；同一 checkpoint 内的比较另报**窗口间 SE**（5 窗口 × 4 批，
  同一 generator seed 批序）。
- **对账（`SCHEMA.md` §12）**：算术对账——额外参数与额外 MACs 都是解析式并可实测核对（已一致）；
  任务对账——两臂与基线在同一任务定义、同一输入分布、同一位置集打分（同一评测批序）；
  口径对账——全深度 bpc，无墙钟量进判据。

## 实现要点（为什么这样接线）

- 复用引擎 `topology="dense"` 的前向路径（`h, _ = blk(self._layer_mix_input(states, k))`），
  只覆写 `_layer_mix_input` = 层间注意力聚合；`pathlm/**` 未改（实验内工厂替换
  `train_depth_ar.DepthARModel`，训练整段走共享 `main()`）。
- **层索引因果**：聚合只取 `states[0..k−1]`（`j ≤ k−1`），看不到当前层之后的信息。
- 初始化：`τ = 1`、`W_q = W_k = W_v = I`。注意一个**如实观测**：由于 `q = h_{k-1}` 而 key 里
  恰好也有 `h_{k-1}`（自相似），训练起点处层维权重**已经**偏离「只取上一层」
  （k ≥ 5 的层熵 ≈ 1.6–2.1、`w_immediate_prev` ≈ 0.15–0.24；层 2–4 更低，熵 0.69–1.38；
  全 k ≥ 2 区间为熵 0.688–2.075、`w_immediate_prev` 0.152–0.551，与 `smoke_report.json`
  的 `B_liveness_gate.per_layer` 一致；最大权重落在紧邻上一层上）。
  也就是说两臂的起点并不等于 stock 起点（不等值起步），这是该参数化固有的、已落盘的性质；
  结论只能来自训练后的 B 表。

## 缺口 / 下一步最小实验（本轮未做完）

1. **两臂训练（必须）**：`python3 run.py --arms attn-scalar,attn-qkv --seed 0 --steps 12000`
   （后台 launcher `run_remaining.sh` 已在跑，最多 8 轮、每轮每臂最多等锁 10 min；按臂增量落盘可续）。
   等效成本 = 每臂 12k × 32 × 256 = **98.3M token**，两臂 **196.6M token**（≈ 7 min 卡时/臂）。
2. **基线自包含双跑（备选）**：若 P5 的配置逐字段对照出现不等字段，回退自包含
   `--dense --topology sequential` 与 `--dense --topology dense` 两臂（+2 × 98.3M token）。
3. **n≥2**：本尺度只跑 s0；两 seed 同号才敢下「可排序」，否则只作快照观测。
4. **外推**：长度/层数外推未测（`gaps` 已记）。

## 独立复核

**第 1 轮独立复核（fresh 上下文）判 PASS**（判词由 lead 落盘于 `.scratch/61-axis2-depth/**`
与 `TICKET.md`；`SCHEMA.md` §14/§14b，`docs/research-schema.md` §11.3 累计 ≤2 轮）。
其 non-blocking 编号清单 N1–N3 本轮已处置：N1（参数倍率 2.7 万 → **2.5 万倍**，`project.yaml::argument`）、
N2（活性区间限定词 → 「k ≥ 5（层 2–4 更低，熵 0.69–1.38）」，本文件与 `project.yaml` 同步）、
N3（`README.md §7/§9` 落盘清单删去并不存在的 `run_log.txt`；两臂未训的阻塞证据只在
`.scratch/61-axis2-depth/TICKET.md` §9.11）。两臂训完后再派新一轮。

## 回归与可复算的校验命令

| 命令 | 结果 |
|---|---|
| `python3 smoke.py` | 三道门（等值 / 活性 / 参数账）全过，报告落 `smoke_report.json` |
| `python3 -m pytest …/test_2653_interlayer_attn.py -q` | 9 passed |
| `python3 tools/summary.py --check` | OK（46 份 `project.yaml`） |
| `python3 -m pytest tests/ -q` | 865 passed / 1 failed —— 失败为 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`，涉 `axes/6-predict/**` 的三个 `mutate.py`（**非本 WP 引入、与本 WP 无关**，预存失败分开登记） |
