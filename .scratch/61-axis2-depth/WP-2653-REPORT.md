# WP-2653 · 2.2.2 层间 causal attention 最小判决 · 执行报告

> 过程工单（`ticket 61 · 轴 2 深度轴`）。事实文档 = `axes/2-depth/2.2.2-attention-residual/project.yaml`
> + 同目录实验 `experiments/2653-interlayer-attn-13m/`。
> writer = WP-2653 worker（fresh 上下文，唯一写入者）。actor/时间见 `change_log` 与本报告。
> 本报告取代上一轮（22:24）的同名报告；实现与门的事实**未变**（见 §3），新增的是**本轮驱动启动与锁实况**。

## 0 人话摘要（先读这段）

这一格问的是：**把「每层只接上一层」换成「每层对之前所有层输出做一次随内容加权的聚合」，能不能把预测质量拉好到能分辨的程度。** 本实验的实现早已做完并证明接线正确（把聚合权重强制成「只取紧邻上一层」时，前向与损失跟原版**逐位一致**——不是差不多，是每一位都相同），两臂的参数代价与活性也已实测。**唯一缺的就是把两臂各训 12k 步，量出 clean bpc（每字符比特数，越低越好）。**

本轮（`2026-10-14 23:27` 起）按 Main 裁决起了 **detached supervisor-retry 驱动**（不放弃重试），把「等锁就重试」这件事交给后台常驻进程。**结果是：驱动起来了，但本机 GPU 被轴 5 的 2656 多臂驱动独占（同一把 `/tmp/pathlm_gpu.lock`），本窗口内一次也没抢到锁**，两臂仍未训。**这不是失败，是排期**：两条 12k 臂 = 每条 98.3M token 的等效算力，**一个 token 都没消耗**。驱动会继续自己重试；等 2656 让出锁，它会自动跑完并逐臂增量落盘。

## 1 本轮交付物

| 文件 | 状态 | 说明 |
|---|---|---|
| `supervise.sh` | ✅ 新增（本提交） | **无界** supervisor-retry 驱动（`while :;` 不放弃）；每轮调 `run.py`，已训臂自动跳过；`setsid nohup` 后台常驻 |
| `run_stdout-supervise.txt` | ✅ 新增（本提交，驱动在写） | 驱动的每轮 start/rc 流水 + `run.py` 的 stdout/stderr |
| `run_log.txt` | ⏳ 驱动生成（本轮未 commit） | `run.py` 的逐臂结构化状态（JSONL）；本轮只写了一条 `gpu_lock_timeout` 记录，留待收口随全量证据一并入库 |
| `analysis.md` / `MODEL.md` / `results_*.json` | ⏳ 未动 | 判据线（B 表 clean bpc）未测 ⇒ 不填占位数字；两臂训完再写（Main 指令） |

## 2 驱动命令与真实输出

驱动脚本：`axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/supervise.sh`
（`setsid nohup bash supervise.sh >/dev/null 2>&1 &`，启动于 `2026-10-14T23:27:54+08:00`，PID 2630132）。

每轮实际执行的命令（驱动 `run.py` 用的就是任务给的按臂续跑形态）：

```bash
cd /home/z/vibe/pathlm            # run.py 用仓库根相对路径读 data/enwik8_full.npz
python3 <EXP>/run.py --arms attn-scalar,attn-qkv --seed 0 --steps 12000 \
    --lock-wait-s 600 --arm-gap-s 65
```

真实输出（`run_stdout-supervise.txt` 原文，截至 `23:40`）：

```
[sup] attempt 1 start 2026-10-14T23:27:54+08:00
[2653] arm=attn-scalar status=gpu_lock_timeout
```

`run_log.txt`（`run.py` 逐臂状态，JSONL）：

```json
{"arm": "attn-scalar", "seed": 0, "steps": 12000, "started_at": "2026-10-14T23:27:57+0800",
 "held_lock": false, "status": "gpu_lock_timeout",
 "error": "GPU lock held by pid 2620551 > 600.0s; 上报，不抢"}
```

含义：`attn-scalar` 等锁 600s（`SCHEMA.md` §20 上限）没等到 ⇒ **记 timeout、不抢锁、不杀进程**（`held_lock=false`）。`run.py` 接着对 `attn-qkv` 再等一轮（attempt 1 在快照时仍在跑第二臂的等锁）。

## 3 实现与门（前轮已落盘，本轮未改）

`smoke.py`（CPU，不取锁）本轮**已复跑**，三道门全绿（`smoke_report.json` 重写后与库中版本逐字节相同）：

| 门 | 结果 |
|---|---|
| A 等值门（one-hot ⇒ 与 stock 单残差前向 + loss 逐位一致；探针：不强制时必须不同） | ✅ |
| B 活性门（k ≥ 2 层 `w_immediate_prev < 1`、熵 > 0；探针：one-hot 端点 = 1） | ✅ |
| C 参数账（`attn-scalar` +8；`attn-qkv` +196 616，与解析值一致） | ✅ |

代价列（实测量，不依赖训练）：`attn-scalar` 每 token 层间聚合 +18 432 MACs（相对 stock 每层注意力 +0.59%）；`attn-qkv` +5 261 312 MACs（+167.3%）。详见 `analysis.md` A 表。

## 4 锁实况（谁持锁 / 队列里有谁 / 等效成本）

- **持锁**：`/tmp/pathlm_gpu.lock`，`holder_pid=2620551` = `axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/run_arms.py`（快照时已存活 21+ min，子进程 `train_mla.py mla-c128-s0` 正在训练）。该锁合法有效，本 WP **不回收、不杀进程**。
- **队列（约 28 个匹配 `run.py|run_arms|train_arms|train_mla|supervise` 的进程）**：本 WP 两个驱动（2653 pid 2630132、2651 pid 2630133）之外，还有轴 3 的 2643/2644/2645/2646/2647 `train_arms.sh` launcher、2654 的 `run_supervise.sh`、轴 5 的 2655/2653 run.py 等。**共享 4090 高度拥塞**。
- **GPU 快照**：`3009–3386 MiB used / ~21 GB free / 18–24% util`（只有 2656 一个计算进程在跑）。
- **等效算力成本（不进判据）**：每臂 12k × 32 × 256 = **98.3M token**；本 WP 两臂合计 **196.6M token**，**本轮消耗 0 token**（只消耗排期）。

## 5 残余缺口

1. **两臂未训**（本窗口全部等锁超时）⇒ B 表（clean bpc / 配对 Δbpc）无数据，判词未下。
2. **10 步 CLI 冒烟未跑**：任务提到「先跑 20 步冒烟」，但 2653 的 `run.py` 没有独立冒烟模式（`--steps 20` 会写 `results_<arm>_s0.json` 并污染后续跳过逻辑），而 CPU 冒烟 `smoke.py` 已覆盖「等值门/活性门」，故本轮以 `smoke.py` 为准，CLI 小跑留给首个抢到锁的尝试。
3. **README §7 事实漂移（登记）**：该节写「本目录无 `run_log.txt`」，而驱动本轮已按设计生成之；本轮不改预登记 README，留待收口统一（判据线 P1–P6 未动）。

## 6 续跑入口

- 驱动已在后台自跑：`ps -p 2630132`；日志 `run_stdout-supervise.txt`。
- 手动续跑：`bash axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/supervise.sh`。
- 单臂复现：`cd /home/z/vibe/pathlm && python3 <EXP>/run.py --arms attn-scalar,attn-qkv --seed 0 --steps 12000`，跑完 `--aggregate`。

## 7 复核

本轮**未做独立复核**（writer 仍在跑，且判据未产出可复核数字）。`SCHEMA.md` §14/§14b 要求复核排在 writer 停止之后、fresh 上下文；等两臂训完出具 B 表后再派。
