# 2654 · 跳层的序列级重测 · 判读（analysis）

## 0. 人话摘要

**根因已修、CPU 回归测试已把红钉住；四臂仍没训 —— 这一轮不是「实现坏了」，而是「修好的实现没抢到 GPU」。**

上一轮冒烟在 `layerdrop-pt` 臂抓红：实测 layer-passes 8.0 ≠ 设计 7.2（`run_log.txt:2194`）。
根因 = `run.py` 的 `ARMS["layerdrop-pt"]` 字典**没有 `"p"` 键**，而 `brand_layers` 用
`spec.get("p", 0.0)` 建 `SkipBlock` ⇒ 该臂的逐 token 跳过概率 p 被悄悄取成 0.0 ⇒
`(r < 0.0)` 恒为 False ⇒ 一个 token 都不跳 ⇒ layer-passes 恒等于层数 8.0。修法是给该臂
spec 显式加 `"p": LAYERDROP_P`（0.1，`run.py:77`），**没有改动任何已落盘的测量数字、没有改判据线、
没有改跳层语义**。并加一条 CPU、秒级的**可失败**回归测试（`test_skipblock_layerdrop.py`）：
① 该臂 spec 必须显式声明 p；② 用该 p 套出的 `SkipBlock` 做一次前向时逐层跳过占比 ≈ 0.1
（落 [0.05, 0.15]）。该测试经**变异验证**：把 spec 的 p 改成 0.0 ⇒ 两条断言当场全红
（`2 failed in 1.75s`），改回即绿（`2 passed in ~3s`）。

修好之后，本轮起了**后台有界重试驱动** `run_supervise.sh`（冒烟闸门 + 四臂逐臂取锁），
自 `2026-10-14T23:27:53` 起在后台排队。**但它整轮没抢到锁**：`/tmp/pathlm_gpu.lock` 从
23:17 起一直被别的 agent（`5-quant/5.4-lowrank/2656-lowrank-vs-headshare`，pid `2620551`）
连续持有，同时另有 6 条别的 launcher 在排同一把锁（2643 / 2644 / 2645 / 2646 / 2647×2）。
本轮 20 分钟跟进窗口内两次等锁各满 10 分钟超时（`run_log.txt` 的 `23:37:54` 起两条 timeout），
冒烟**没重跑**、四臂 12k **一个都没训**、`results.json` 不存在、判据线 P0–P5 **无读数**。

**这不是科学结论，也不是失败读数**：它是三件事——「实现缺陷已定位并修复（CPU 可证）」、
「修复的证据已落（回归测试 + 变异红）」、与「资源未获得」三者的分离。判据线预登记在
`README.md` §4，**逐字未改**；够不够算力时禁止事后调判据，本轮也没调。

## 1. 状态

| 项 | 状态 |
|---|---|
| 实装（工厂替换 + `SkipBlock` 逐 token 恒等旁路） | ✅ 落盘 |
| P0 根因修复（`layerdrop-pt` spec 补 `"p": LAYERDROP_P`，`run.py:77`） | ✅ 落盘 |
| P0 回归测试（CPU，可通过也可失败） | ✅ 落盘：`test_skipblock_layerdrop.py`，`2 passed ~3s`；变异（p→0.0）`2 failed 1.75s` |
| 预登记契约 `README.md`（SCHEMA §6 九节 + 判据线 P0–P5） | ✅ 落盘，逐字未改 |
| `config.json` / `run.py`（`--arms` 续跑 + 逐臂增量落盘） | ✅ 落盘 |
| 后台有界重试驱动 `run_supervise.sh`（冒烟闸门 + 逐臂取锁） | ✅ 起（pid `2630109`，deadline 240 min），本轮**未抢到锁** |
| 20 步冒烟 `run.py --smoke`（修复后重跑） | ⏳ **待跑**：驱动已排队，锁被 pid `2620551`（2656-lowrank）连续持有 ⇒ 本轮未重跑 |
| 四臂 12k 训练 + 序列级评测 | ⏳ **未训**（同因：无 GPU 锁） |
| 判据 P0–P5 读数 | ⏳ 无读数 |

历史（修复前）：`2026-10-14T22:37:39` 第 4 轮取到锁跑 20 步冒烟，`stock` / `skip-mid` / `layers6`
三臂活性门过（实测 lp = 8.0 / 6.0 / 6.0），`layerdrop-pt` 实测 lp = 8.0 ≠ 设计 7.2
（断言 `run.py` 的 `smoke`，`run_log.txt:2194`），`22:38:50 SMOKE FAILED`（`:2195`）；
该红门即 §0 所述的缺 `"p"` 键，本轮已修。

## 2. 方法（已实现；20 步冒烟待重跑、四臂 12k 未跑）

见 `README.md` §2 / §4。要点：四个 13M/12k/seed0 臂只改跳层策略 ——
`stock`（layer-passes 8）、`skip-mid`（0-indexed block {3,4} = 人类第 4、5 层恒等旁路，
layer-passes 6）、`layers6`（`--layers 6` 同预算固定深度对手）、`layerdrop-pt`
（逐 token 逐层独立 Bernoulli(0.1)，layer-passes ≈7.2）。跳层 = 该 token 在该层的输出 = 输入，
`out = where(skip, x, block_out)`，因果不破。

## 3. 数字表

**无实测数字**（四臂 12k 未训练）。表位保留，等续跑臂落盘后填：

| 臂 | clean bpc | layer-passes/token（实测 / 设计） | 参数量 | 逐层跳过占比 |
|---|---|---|---|---|
| stock | — | — / 8.0 | — | — |
| skip-mid | — | — / 6.0 | — | — |
| layers6 | — | — / 6.0 | — | — |
| layerdrop-pt | — | — / 7.2 | — | — |

判据线读数：`Δ(skip-mid vs stock)` —；**`Δ(skip-mid vs layers6)`（判决线）** —；
`Δ(layerdrop-pt vs skip-mid)` —。P0 引擎等值门（stock vs 2613 s0 = 1.5254）— 未跑。

冒烟（20 步、远未收敛）只作过程留痕、**不作任何判据读数**；修复前的读数见 §1 末。

## 4. 口径标注

- clean bpc 口径 = 最终深度读出、序列级、`EVAL_SEED=2`、20 窗口 × batch16 × seq256
  （与 `2613` / `2641` 的 `depth_curve` 最终深度同口径）。
- 判据符号：`Δ(x vs y) = bpc_y − bpc_x`（正 = x 更好）。bpc 判据走 0.014 带（配对 Δ、≥5 窗口）；
  率类指标（跳过占比、分布一致率）走配对 bootstrap / Wilson 95% CI。
- **墙钟不进判据**；GPU 阻塞只记为资源事实。

## 5. 残余缺口（下一步最小动作）

1. **等锁 → 冒烟 → 四臂**（驱动已 detached 在跑，deadline 240 min）：`run_supervise.sh` 会先
   跑 20 步冒烟闸门（四臂全绿才继续），再逐臂取/放锁（§20.1，≥60 s 窗口）训 12k + 评测，
   臂齐后 `--aggregate`。判活看末行横幅 `SUPERVISE DONE` / `SUPERVISE GIVEUP`。
   等效算力账 = 每臂 12k × 32 × 256 = **98.3M token**、四臂合计 **393.2M token**
   （与姊妹项 2.2.2 / 2.3.2 同账）。
2. 臂齐后填本文件 §3 表 + 写 P2 判决线判词（§12 三对账 + §13 三要素），更新 `MODEL.md`
   权重 sha256 与 `project.yaml`。
3. 未做（待修复后重跑）：`layerdrop-pt` 的训练/评测跳层分布一致性列（README §4-P1b）——
   修复前的红门见 §1 末；本次修复只改了 spec 的 `"p"` 键，直接决定该列能否成立。
4. 独立复核：本轮**未做**新一轮 fresh 复核；复核排在 writer 停止、四臂跑齐之后。
