# WP-2651-s1 · 2.2.3 / 2.2.4 升 n≥2（seed=1 三条臂）· 执行报告

> 过程工单（`ticket 61 §9.2 WP-2651`）。事实文档 = `axes/2-depth/2.2.3-learned-layer-sum/project.yaml`
> （2.2.4 证据指针指向同一目录）+ `axes/2-depth/2.2.4-parallel-sublayer/project.yaml`
> + 同目录实验 `experiments/2651-inter-layer-topology-s1-13m/`。
> writer = WP-2651-s1 worker（fresh 上下文，唯一写入者）。actor/时间见本报告与 `run_stdout-supervise.txt`。

## 0 人话摘要（先读这段）

2.2.3（层间**线性**学习加权和，DenseFormer 式）与 2.2.4（**并行子层**：注意力与 MLP 同时读层输入）此前只有 **seed=0 一个种子**的读数，弱证据。这一 WP 就是把四种层间接法在 **seed=1** 上再跑一遍，凑够两个种子（n=2），好回答「换个种子结论是否还成立」。四条臂里 **基线 `sequential` 上一轮已经训完（clean bpc 1.5288）**，本轮要补的是另外三条：`parallel` / `parallel-dual-ln` / `dense`。

本轮（`2026-10-14 23:27` 起）按 Main 裁决起了 **detached supervisor-retry 驱动**（不放弃重试、已训臂自动跳过）。**结果是：驱动起来了，但本机 GPU 被轴 5 的 2656 多臂驱动独占（同一把 `/tmp/pathlm_gpu.lock`），本窗口内一次也没抢到锁**，三条臂仍未训。**这不是失败，是排期**：三条 12k 臂 = 每条 98.3M token 的等效算力，**一个 token 都没消耗**。驱动会继续自己重试，抢到锁就自动按臂跑完并增量落盘。

## 1 命令形态（一处必要偏差，如实登记）

任务的驱动命令是 `python3 run.py --arms parallel,parallel-dual-ln,dense --skip-s0-anchor`。**实测这条命令 rc=1 直接退出**——`run.py` 的 `--arms` 校验硬要求必须含 `sequential`（判读需要它当基线做逐臂 Δ 与跨 seed 配对），缺则：

```
--arms 无效（[]）：必须含 sequential，可选 ['sequential', 'parallel', 'parallel-dual-ln', 'dense']
```

因此本轮驱动改用**等价且不消耗额外 token** 的形态（`sequential` 已训完，`run.py::arm_trained` 会跳过重训，只重算一次电池 ≈3s）：

```bash
cd /home/z/vibe/pathlm            # run.py 用仓库根相对路径读 data/enwik8_full.npz
python3 <EXP>/run.py --arms sequential,parallel,parallel-dual-ln,dense \
    --skip-s0-anchor --caps-min 14 --budget-min 42
```

这既满足校验、又满足「`sequential-12k-s1` 须跳过重训」的约束；`--skip-s0-anchor` 仍跳过 2613 的 s0 锚重测。其余字段与任务命令一致。**若 Main 要的是改 `run.py` 代码而非这样绕，请指出**（本报告已把该偏差显式登记）。

驱动脚本：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/supervise-s1.sh`
（`setsid nohup bash supervise-s1.sh >/dev/null 2>&1 &`，启动于 `2026-10-14T23:27:54+08:00`，PID 2630133）。

## 2 真实输出

`run_stdout-supervise.txt`（原文，截至 `23:40`）：

```
[sup] attempt 1 start 2026-10-14T23:27:54+08:00
[cfg] vs 2613 unequal fields = ['seed', 'experiment']
Traceback (most recent call last):
  ...
File ".../2651-inter-layer-topology-s1-13m/run.py", line 215, in gpu_lock_acquire
    raise TimeoutError("等锁超过 10 min，未取到 GPU 锁")
TimeoutError: 等锁超过 10 min，未取到 GPU 锁
[sup] attempt 1 rc=1 end 2026-10-14T23:37:58+08:00
[sup] attempt 2 start 2026-10-14T23:38:18+08:00
[cfg] vs 2613 unequal fields = ['seed', 'experiment']
```

`[cfg] unequal fields = ['seed', 'experiment']` 是**预期**：s1 相对 2613 s0 的唯一科学差异就是 seed（另一个是实验目录名）。等锁 600s（`SCHEMA.md` §20 上限）没等到 ⇒ 进程超时退出、rc=1 ⇒ 驱动醒 20s 后重开 attempt 2。**全程不抢锁、不杀进程。**

## 3 臂状态与等效成本（每臂 98.3M token）

| 臂 | seed | 状态 | clean bpc | 等效 token |
|---|---|---|---|---|
| `sequential`（基线） | 1 | ✅ 已训完（上一轮） | **1.5288** | 98.3M（上轮已消耗） |
| `parallel` | 1 | ⏳ 未训（本轮等锁超时） | missing | 98.3M（未消耗） |
| `parallel-dual-ln` | 1 | ⏳ 未训（本轮等锁超时） | missing | 98.3M（未消耗） |
| `dense` | 1 | ⏳ 未训（本轮等锁超时） | missing | 98.3M（未消耗） |

- 三条待训臂合计 **294.9M token**，**本轮消耗 0**。
- `results.json` **未改动**（本轮无臂完成）：`completed_arms=['sequential']`、`pending_arms=['parallel','parallel-dual-ln','dense']`（与上轮流经一致）。
- 因臂未跑齐，**2.2.3/2.2.4 的 n=2 升级（判词重算）本轮未做**——S1 升级未发生，故未触动既有 s0 测量数字。

## 4 锁实况

- **持锁**：`holder_pid=2620551` = `axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/run_arms.py`（存活 21+ min，子进程 `train_mla.py mla-c128-s0` 在训）。合法有效，本 WP 不回收、不杀进程。
- **队列**：本 WP 两个驱动（2651 pid 2630133、2653 pid 2630132）之外，另有轴 3 的 2643–2647 `train_arms.sh` launcher、2654 `run_supervise.sh`、轴 5 的 2655/2653 run.py 等（约 28 个匹配进程）。共享 4090 高度拥塞。
- **GPU 快照**：`3009–3386 MiB used / ~21 GB free / 18–24% util`。

## 5 残余缺口

1. **三条干预臂未训** ⇒ seed=1 质量面缺 **missing**；2.2.3 的 `dense−sequential`（两 seed 配对）与 2.2.4 的 `parallel(±dual-ln)−sequential` 代价判词**均未升级到 n=2**。
2. 升级判词须在两个 seed 上同号且都越 0.014 带才可排序（见 `run.py::cross_seed_summary`）；未跑齐前不下结论。

## 6 续跑入口

- 驱动已在后台自跑：`ps -p 2630133`；日志 `run_stdout-supervise.txt`。
- 手动续跑：`bash axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/supervise-s1.sh`。
- 抢到锁后驱动自动按臂训 + 逐臂增量落盘（`results.json` + `run_log.txt`）；三臂齐后 `results.json` 会写出 `cross_seed_s0_s1` / `verdict_s1`，再做 n=2 判词重算。

## 7 复核

本轮**未做独立复核**（writer 仍在跑，且无新测量数字可复核）。`SCHEMA.md` §14/§14b 要求复核排在 writer 停止之后、fresh 上下文；等三臂训完出具跨 seed 配对后再派。
