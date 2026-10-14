# 2654 · 跳层的序列级重测 · 判读（analysis）

## 0. 人话摘要

**本轮没有跑成实验。** 代码、契约、20 步冒烟脚本、按臂增量落盘的驱动都已写好并落盘，
但**没有拿到 GPU 锁**：从 `22:03` 起 `/tmp/pathlm_gpu.lock` 一直在别的 agent 手里轮转
（`wp6.3.2-*`、`wp2655-layer-capacity`、`2651`、`2652`），有界 launcher（`run_remaining.sh`）
按 `SCHEMA.md` §20 礼貌等待，第 1 轮等满 10 min 超时后继续重试，**始终没有获得锁**。
因此**四臂（stock / skip-mid / layers6 / layerdrop-pt）一个都没训**，`results.json` 里
`arms = {}`，判据线 P0–P5 **一条都没有读数**。

**这不是科学结论，也不是失败读数** —— 它只是「资源未获得」。判据线未被触碰（预登记在
`README.md` §4，逐字未改）。

## 1. 状态

| 项 | 状态 |
|---|---|
| 实现（工厂替换 + `SkipBlock` 逐 token 旁路） | ✅ 落盘，CPU 单元验证通过（恒等旁路语义、layer-passes 账、state_dict 重建一致） |
| 预登记契约 `README.md`（SCHEMA §6 九节） | ✅ 落盘，判据线 P0–P5 预登记 |
| `config.json` / `run.py`（`--arms` 续跑 + 逐臂增量落盘） | ✅ 落盘 |
| 20 步冒烟 `run.py --smoke` | ⏳ 待 GPU 锁（launcher 已排入，未执行） |
| 四臂训练 + 序列级评测 | ❌ 未跑（无 GPU 锁） |
| 判据 P0–P5 读数 | ❌ 无读数 |

## 2. 方法（已实现，未运行）

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

## 4. 口径标注

- clean bpc 口径 = 最终深度读出、序列级、`EVAL_SEED=2`、20 窗口 × batch16 × seq256
  （与 `2613` / `2641` 的 `depth_curve` 最终深度同口径）。
- 判据符号：`Δ(x vs y) = bpc_y − bpc_x`（正 = x 更好）。bpc 判据走 0.014 带（配对 Δ、≥5 窗口）；
  率类指标（跳过占比、分布一致率）走配对 bootstrap / Wilson 95% CI。
- **墙钟不进判据**；GPU 阻塞只记为资源事实。

## 5. 残余缺口（下一步最小动作）

1. **取 GPU 锁续跑**：`bash run_remaining.sh`（或 `python3 run.py --arms stock,skip-mid,layers6,layerdrop-pt`）
   —— 优先序 `stock → skip-mid → layers6 → layerdrop-pt`，预算不够先砍 `layerdrop-pt`（README §4-P4）。
   预计 4 臂 × 12k 步 ≈ 30–45 min GPU。
2. 臂齐后填本文件 §3 表 + 写 P2 判决线判词（由后续 worker / lead 收口）。
3. 未做：`--smoke` 的 GPU 端断言（layer-passes 实活）未运行。
