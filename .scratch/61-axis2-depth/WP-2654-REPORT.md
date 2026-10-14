# WP-2654 报告 · 2.1.2 skipping 的序列级重测

> 契约：`.scratch/61-axis2-depth/TICKET.md` §9.1（共同契约）+ §9.2 **WP-2654**。
> actor：WP-2654 worker（fresh 上下文，本 WP 唯一 writer）。性质：**过程工单**（工单与报告），
> 不是事实文档；事实文档 = `axes/2-depth/2.1.2-skipping/project.yaml`。

## 0. 人话摘要（结论先行）

**本轮冒烟跑过一次并抓红：三轮没拿到锁之后，第 4 轮（`[2026-10-14T22:37:39]`）取到了锁、
跑了 20 步冒烟，`layerdrop-pt` 臂的活性门抓红，launcher 当场 `exit 1`。** 22:03 起
`/tmp/pathlm_gpu.lock` 一直被别的 agent 的任务轮转持有（`wp6.3.2-*`、接着 `wp2655-layer-capacity`、
以及本仓 2651/2652 的在途臂），有界 launcher 按规矩排队等锁，前 3 轮各等满 10 分钟超时。第 4 轮
取到锁后的真实结果（`run_log.txt`）：`stock` / `skip-mid` / `layers6` 三臂活性门过（实测
layer-passes 8.0 / 6.0 / 6.0，`:576` / `:1144` / `:1618`），`layerdrop-pt` 实测 layer-passes
**8.0** 与设计 **7.2** 不符（断言抛自 `run.py:451`，`:2194`），`[2026-10-14T22:38:50] SMOKE FAILED`
（`:2195`）⇒ launcher 按 `run_remaining.sh:53` 的 `|| { …; exit 1; }` 退出。所以四个臂
（全深度对照 / 固定跳中间两层 / 同预算 6 层 dense / 逐 token 跳层）**一个都没训**，
`results.json` 不存在（冒烟产物只落在 `smoke/`），判据线 P0–P5 **一条也没读到**。

**这不是「跳层没价值」的结论，也不是「实验失败」**：它是两件事——**「实现未过活性门」**
（`layerdrop-pt` 按 `README.md` §4-P1「实测与设计不符 ⇒ 该臂作废」判作废，须先修接线/记账）
与**「资源没拿到」**。判据线预登记在 `README.md` §4，**逐字未改**；够不够算力时禁止事后
调判据，本轮也没调。

**已经落地的东西**（可续跑，不浪费）：

- **实验实现**：`run.py` —— 在 depth-ar 共享引擎里用**工厂替换**（monkey-patch
  `train_depth_ar.DepthARModel`，沿用 2641 的既有模式）把「逐 token 恒等旁路」接进训练整段；
  `pathlm/**` 与 `train_depth_ar.py` **零改动**。
- **预登记契约**：`README.md`（SCHEMA §6 九节 + 判据线 P0–P5）、`config.json`、`gpu_lock.py`。
- **按臂增量落盘**：`run.py --arms <臂>` 支持只跑/续跑指定臂，每臂训完即可写 `results.json`
  的 `completed_arms` / `pending_arms` ⇒ 中断不丢进度（本仓 2651/2652 已验证该模式必要）。
- **有界 launcher**：`run_remaining.sh` —— 冒烟→逐臂，逐臂按 `SCHEMA.md` §20.1 取/放锁
  （≥60s 礼貌窗口），最多 6 轮等锁，`LAUNCHER DONE` 横幅判活。

## 1. 交付物清单（已落盘）

| 路径 | 内容 |
|---|---|
| `axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m/README.md` | 预登记契约（命题/单因素/指标表/判据线 P0–P5/对照方差/成本/数据脚本/失败树/产出） |
| `.../run.py` | 唯一入口：工厂替换 + `SkipBlock` 逐 token 旁路 + `--arms` 续跑 + 逐臂增量落盘 + `--smoke` + `--aggregate` |
| `.../config.json` | resolved config（四臂定义 / 数据 / 训练 / 判据 / 预算） |
| `.../run_remaining.sh` | 有界后台 launcher（逐臂取锁） |
| `.../gpu_lock.py` | §20 锁协议（mkdir + holder_pid + heartbeat + 自愈） |
| `.../analysis.md` | 判读（本轮：四臂 12k 无实测数字；已登记 20 步冒烟的 `layerdrop-pt` 红门 + GPU 阻塞） |
| `.../MODEL.md` | 权重清单（整仓 12k 训练未跑，尚无 12k 权重；冒烟产物在 `smoke/`） |
| `.../run_log.txt` | launcher 真实 stdout（3 轮等锁超时 + 第 4 轮冒烟红门，共 2195 行） |
| `.scratch/61-axis2-depth/WP-2654-REPORT.md` | 本报告 |

## 2. 数字表

**无实测数字**（四臂 12k 未训练）。表位保留（冒烟 20 步的读数见 §3.3，未收敛、不作判据）：

| 臂 | 跳层策略 | layer-passes/token（设计） | clean bpc（实测） |
|---|---|---|---|
| `stock` | 8 层全活 | 8.0 | — |
| `skip-mid` | block {3,4}（0-idx）= 人类第 4、5 层恒等旁路 | 6.0 | — |
| `layers6` | `--layers 6` 同预算更浅 dense | 6.0 | — |
| `layerdrop-pt` | 逐 token 逐层独立 Bernoulli(0.1) | 7.2（≈8·0.9） | — |

判据线读数：`Δ(skip-mid vs stock)` —；**`Δ(skip-mid vs layers6)`（判决线）** —；
`Δ(layerdrop-pt vs skip-mid)` —。P0 引擎等值门（stock vs 2613 s0 = 1.5254）— 未跑。

## 3. 命令与真实输出

### 3.1 落盘与校验

```
$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。
```

### 3.2 GPU 锁现状（等锁记录，真实输出）

```
$ cat /tmp/pathlm_gpu.lock/holder_pid; cat /tmp/pathlm_gpu.lock/name
2420657
wp6.3.2-fill_bidir_r10-s1            # 反复轮转：wp6.3.2-fill_bidir_r10/r25/r50、wp2655-layer-capacity

$ nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
7122 MiB, 16986 MiB, 100 %
```

### 3.3 launcher 的真实 stdout（`run_log.txt`，共 2195 行）

```
[2026-10-14T22:03:08] launcher start (pid 2445457)
[2026-10-14T22:13:24] lock timeout for smoke (round 1)
[2026-10-14T22:24:09] lock timeout for smoke (round 2)
[2026-10-14T22:34:54] lock timeout for smoke (round 3)
[2026-10-14T22:37:39] === smoke (round 4) ===
[smoke] stock: lp=8.0 ok, train 3.6s, bpc=5.157922        # run_log.txt:576
[smoke] skip-mid: lp=6.0 ok, train 0.8s, bpc=5.310313     # run_log.txt:1144
[smoke] layers6: lp=6.0 ok, train 0.6s, bpc=5.290606      # run_log.txt:1618
AssertionError: layerdrop-pt: layer-passes 实测 8.0 与设计 7.2 不符   # :2194（抛自 run.py:451）
[2026-10-14T22:38:50] SMOKE FAILED                        # run_log.txt:2195
```

（前 3 轮 acquire 各等满 40×15s = 10 min 超时；第 4 轮取到锁、冒烟跑到 `layerdrop-pt` 的活性
断言处中止；`SMOKE FAILED` 之后 `run_remaining.sh:53` 的 `|| { …; exit 1; }` 让 launcher 直接
退出，四臂训练一段都没启动。）

### 3.4 已做（CPU，不取锁）

- `run.py` 结构单测（CPU，脱离 GPU）：`SkipBlock` 恒等旁路语义（全跳 ⇒ 每层 state == 输入 embedding，
  实测 `True`）、layer-passes 账（4 层跳 idx3 ⇒ 3.0；全跳 ⇒ 0.0）、`state_dict` 两次工厂重建键一致
  且 `load_state_dict` 成功、`paired_delta` 配对 Δ 计算。
- `DepthARConfig(**asdict(cfg))` 往返 OK（评测侧用训练 run 的 resolved config 重建模型）。

## 4. 口径与边界（未变）

- 判据只用**等效理论算力**（layer-passes/token）；**墙钟不进判据**，GPU 阻塞只记为资源事实。
- bpc 判据走 **0.014 bpc 带**（配对 Δ、≥5 窗口；本实验 20 窗口）；率类指标（跳过占比、
  分布一致率）走配对 bootstrap / Wilson 95% CI，不借 0.014。
- 续跑优先序：`stock → skip-mid → layers6 → layerdrop-pt`；预算不够**先砍 `layerdrop-pt`**
  （写进 `gaps`），**不砍判据线**。
- 结论不越 13M/12k 尺度与 enwik8 文本分布外推。

## 5. 残余缺口 / 下一步最小动作

1. **先修实装（纯 CPU，不取 GPU 锁）**：按 `README.md` §8 决策树第 2 条查
   `SkipBlock._sample_skip`（`drop` 是否逐 token 独立、`torch.where` 是否用层输入做旁路、
   layer-passes 记账是否统计该 token 实际非跳层数），修到 `python3 run.py --smoke --steps 20`
   **四臂全绿**（`stock` / `skip-mid` / `layers6` 已绿，只差 `layerdrop-pt` 的 lp ≈ 7.2）。
2. **冒烟全绿后才取锁续跑四臂**（预计 30–45 min GPU）：
   `cd axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m && bash run_remaining.sh`
   或 `python3 run.py --arms stock,skip-mid,layers6,layerdrop-pt`；等效算力账 = 每臂 12k × 32 × 256
   = 98.3M token，四臂合计 393.2M token。
3. 臂齐后：填 `analysis.md` §3 表 → `run.py --aggregate` 出 P0–P5 → 写 P2 判决线判词 →
   更新 `project.yaml`（`status` / `conclusion` / `failure_analysis` / `conclusion_review`）。
4. **独立复核**：本轮已过第 1 轮 fresh 复核——判决 **1 blocking**（B1：冒烟结果与文档不符，
   由本报告所在轮次处置）；修完红门、并跑完四臂后再派**新的** fresh 复核（`SCHEMA.md` §14/§14b）。
5. 已跑并失败：GPU 端 `--smoke`（layer-passes 活性断言，见 §3.3），修复后必须重跑；
   `layerdrop-pt` 的训练/评测分布一致性列在修复前不做。
