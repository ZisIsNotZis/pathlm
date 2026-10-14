# WP-2651 报告 · 轴 2 深度轴 · 2.2.3 + 2.2.4（seed=1 复核 + 并行子层代价面）

- 实验目录：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/`
- 契约：同目录 `README.md`（预登记，判据线逐字来自 `TICKET.md` §9.2 WP-2651）
- 口径锚：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2613-inter-layer-topology-13m/`（s0, seed=0）
- GPU 预算（本 WP）：45 min；单次命令 ≤180s；锁 = `/tmp/pathlm_gpu.lock`（§20，按臂释放 §20.1）
- 本轮 writer：fresh 上下文续做（上一轮 writer 在 `run.py` 崩后中断）

## 0. 人话摘要（先读这一段）

这一轮要回答的是**「把变压器层与层之间的接法换个花样，质量会不会变、代价是什么」**这条线上的两个问题。
上一轮把四条臂里的一条（基线 `sequential`，也就是「先注意力、再 MLP」的普通接法）在 seed=1 上训完了，
但在训练收尾时要写一行日志、去读一个**导师产物里并不存在的字段**（`results.json` 里没有 `bpc` 键），
于是整个脚本崩掉——**四臂只训完了一条，结果文件一个都没落盘**。

本轮做了三件事：**（1）修掉那个读错字段的崩溃**，并把「缺失就报错、绝不静默填 0」写进代码；
**（2）给脚本加上逐臂增量落盘**——每跑完一条臂立刻把 `clean_bpc` / `tf@4` / 参数量写进 `results.json`
并往 `run_log.txt` 追加一行，这样即使中途被打断，已完成的臂也留得下可读结果；
**（3）把不需要 GPU 的 2.2.4 代价面（层内串行依赖长度 / 每层算力 / 参数量）先算出来**，
不必等训练。

**剩余缺口很明确**：三条待训臂（`parallel` / `parallel-dual-ln` / `dense`，seed=1）已由后台 launcher
排队等 GPU 锁（每臂约 10 min，三臂约 30 min）。本机此刻有多个 worker 抢同一把锁，本轮窗口内未取到，
因此**四臂齐后的判词、`analysis.md`、`MODEL.md` 与两份 `project.yaml` 的判词字段尚未落**——
它们是「四臂齐」的下游，不能提前编造。代价面已算完，数字在下文，随时可用。

> **接手者先看这里**：GPU 锁在本轮窗口内先后被 `2319397`（轴 1 的 1.1.1 hard-window）与
> `2383697`（`wp2652-2.2.1-residual-form`，用同一锁目录但写 `name` 而非 `holder_pid`）持有；
> 后台 launcher `run_remaining.sh`（setsid 脱离）仍在重试（最多 12 轮，每轮最多等锁 10 min）。
> **不要另起新跑**——直接用它的产物：`run_stdout-smoke.txt` / `run_stdout-full.txt` /
> `run_log.txt` / `results.json`（逐臂增量）。

## 1. 本轮改动

| 文件 | 改动 | 目的 |
|---|---|---|
| `run.py` | `run_battery()` 读 `clean_bpc` 时显式校验 `depth_curve[-1]` 含 `bpc` 键，缺失即 `raise KeyError`（附实有键列表），**不再**从导师产物 `results.json` 取（那里本就没有 `bpc`）；`train_arm()` 收尾日志改成打印「clean_bpc 唯一来源 = run_battery → depth_curve[-1]['bpc']」 | 修上一轮的 `KeyError: 'bpc'`；缺失即失败，不静默填 0 |
| `run.py` | 新增 `ALL_ARMS` / `dump_results()` / `append_run_log()`；每臂跑完立刻写 `results.json`（含 `completed_arms` / `pending_arms`）并 append 一行到 `run_log.txt`；最终落盘复用同一函数 | 增量落盘，避免「跑完三臂后崩溃 ⇒ 一个结果都不留」 |
| `run_remaining.sh` | 新增后台 launcher：先在仓库根跑 20 步冒烟（`--arms sequential,parallel --skip-s0-anchor`），成功后跑全跑四臂；每次失败等 20s 重试（最多 12 轮） | 在锁被抢占 / 崩溃时自动续跑，不空转 |
| `test_run_persistence.py` | 新增 4 条回归测试（CPU，无 GPU）：① `run.py` 全文不得再出现 `res['bpc']`；② `depth_curve` 缺 `bpc` 时抛**自己消息**的 KeyError；③ `dump_results` 如实报 `completed_arms` / `pending_arms`；④ `append_run_log` 是 append 语义 | 把本轮的修复与增量落盘钉成可失败的检查 |

> `config.json` 注意：`config.json` 由每次 `run.py` 启动时重写。当前（冒烟）为 `--steps 20`，
> 全跑启动时会写回 `--steps 12000`（此时 `vs_2613` 唯一不等字段应为 `seed` + `experiment`）。
> 若全跑未启动，接手者先核对 `config.json::steps == 12000`。

`run.py:405` 原崩溃点：`res['bpc']`（`res` = `train_depth_ar.py` 产物 `runs/<arm>/results.json`，
该文件**没有** `bpc` 键）。正确来源是 `run.py:347` 早就写对的 `out["depth_curve"][-1]["bpc"]`。
两个来源不是一回事：前者是训练脚本的汇总，后者是本实验评测电池（同一 20 个 eval batch、同一 eval 切分）的读数。

## 2. 状态表

| 项 | 状态 |
|---|---|
| `run.py` KeyError 修复 | **已完成**（`python3 -m py_compile` 通过；回归测试 4 passed） |
| 20 步冒烟（四臂可训起） | **未跑成**：`run.py` 需取 GPU 锁，本轮两次尝试各等满 10 min 均未取到（见 §5）。三个非基线臂的 20 步 CLI 训练此前已由 `smoke.py` 验证通过（`smoke/smoke.json`） |
| `sequential-12k-s1` 训练 | **已完成**（`runs/sequential-12k-s1/`，train_log 到 11999 步，`model.pt` 已落） |
| `parallel-s1` / `parallel-dual-ln-s1` / `dense-s1` 训练 | **未跑**（后台 launcher 已起步，抢锁中） |
| 四臂电池 + 判词 + `analysis.md` / `MODEL.md` | **未做**（依赖四臂齐） |
| 2.2.4 代价面（CPU） | **已完成**（见 §4） |
| 两份 `project.yaml`：`experiments` 加新目录 + WP1 `status`→`进行中` + `change_log` | **已完成**（`tools/check_tree.py` 通过：46 个 project.yaml 全合规） |
| 两份 `project.yaml` 的判词字段（`conclusion` 等） | **未做**（依赖四臂齐；避免提前编造判词） |

## 3. 已有数字：s0（seed=0，2613 对照）四臂

口径：enwik8 char 级 AR，d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，dense 形态，12000 步
（98.3M token），matched-step 快照；`clean_bpc` = 全深度（depth-8）逐位 clean 每字符比特数，
越低越好；`tf@4` = 用前 4 层当草稿被全深度接受的比例（本形态无浅层出口监督，历史实测 ≈0.44 为上限参考）。

| 臂 | clean_bpc | Δ vs 基线 | tf@4 | next-token acc | 参数量 |
|---|---|---|---|---|---|
| `sequential`（基线） | 1.5254 | 0 | 0.4560 | 0.6893 | 6,962,688 |
| `parallel` | 1.5712 | **+0.0458** | 0.3486 | 0.6815 | 6,958,592 |
| `parallel-dual-ln` | 1.5656 | **+0.0402** | 0.3149 | 0.6824 | 6,962,688 |
| `dense` | 1.5246 | **−0.0008** | 0.4538 | 0.6893 | 6,962,724 |

解读（人话）：基线每字符 1.5254 bit；`dense`（层间学习加权和）1.5246，只低 0.0008 bit，
**远小于 0.014 的噪声带**，所以 seed=0 这一把看不出差别；`parallel` 和 `parallel-dual-ln`
（把注意力与 MLP 改成并行读同一输入）分别高 0.0458 / 0.0402 bit，**是噪声带的 3 倍以上**，
seed=0 这一把是明确更差。

## 4. 2.2.4 代价面（解析等效量，**不含任何时间量**，CPU 可算）

公式（`run.py::topology_cost_face`）：注意力 = 4d² + 2T·d；MLP = 2·mlp_mult·d²；
一个 pre-LN ≈ 2d MACs/token；读出 = d·vocab；`dense` 另加层间加权和 Σ(k−1)·d（全栈 L(L−1)/2·d）。
d=256, L=8, mlp_mult=4, T=256, vocab=206。

| 臂 | 层内串行依赖 | 每 token 全栈串行步数 | 每层 MACs/token | 全栈 MACs/token（含读出） | vs 基线 | 参数量 Δ |
|---|---|---|---|---|---|---|
| `sequential` | 2 段（attn → MLP） | 16 | 918,528 | 7,400,960 | 100.000% | 0（基线） |
| `parallel` | **1 段**（两分支同读块输入） | **8** | 918,016 | 7,396,864 | 99.945% | **−4096** = −L·2d |
| `parallel-dual-ln` | **1 段** | **8** | 918,528 | 7,400,960 | 100.000% | **0**（逐位等参） |
| `dense` | 2 段 | 16 | 918,528 + 896（层间加权和摊派） | 7,408,128 | 100.097% | **+36** = +L(L+1)/2 |

解读（人话）：**并行子层买到的东西是「层内串行链从 2 段砍到 1 段」——同一时间能并起来的活儿多一倍**
（每 token 的串行步数 16 → 8），而算力几乎没变（差 0.055%，且那点差就是少了一个 pre-LN 的 4096 个参数）。
`parallel-dual-ln` 是**等参数对照**：串行依赖同样砍半，参数量与基线**逐位相同**，
把「并行的收益」和「参数账」彻底解耦。`dense` 的代价面几乎可以忽略（+36 个标量、+0.097% 算力），
但它**不缩短**层内串行依赖（仍是 attn → MLP 两段），只是多了一条层间加权和。

**这张代价面必须和质量面同框读**：`parallel` / `parallel-dual-ln` 换来的是「层内并行度翻倍」，
代价是 seed=0 质量面 +0.0458 / +0.0402 bit。seed=1 的读数未出，判「划算与否」要等它。
按契约，墙钟不进判据——「划算」只能由「串行依赖缩短（等效算力/流水收益）」对「clean bpc 代价」这一对量回答。

## 5. 命令与真实输出

```bash
# 1) 修 bug 后语法核对
python3 -m py_compile axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/run.py
# -> COMPILE OK

# 1b) 新增回归测试（4 passed）——并做变异验证（删掉显式守卫 / 把 pending 恒置空 ⇒ 2 failed，恢复后 4 passed）
python3 -m pytest axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/test_run_persistence.py -q
# -> 4 passed in 3.76s
# 变异后 -> 2 failed, 2 passed in 2.77s（fail: missing_bpc_key_raises_explicit_error / dump_results_reports_completed_and_pending）

# 1c) 代价面 = run.py 自带函数（非手抄）
python3 -c "...run.py::topology_cost_face(vocab=206)..."
# -> sequential segs=2 serial=16 per_layer=918528 total=7400960 frac=1.0
# -> parallel   segs=1 serial=8  per_layer=918016 total=7396864 frac=0.999447
# -> parallel-dual-ln segs=1 serial=8 per_layer=918528 total=7400960 frac=1.0
# -> dense      segs=2 serial=16 per_layer=919424 total=7408128 frac=1.000969

# 2) 本轮之前：三个非基线臂的 20 步 CLI 训练（smoke.py，已通过；见 smoke/smoke.json）
#    parallel / parallel-dual-ln / dense 各训 20 步 + 参数量账 + 恒等起步 + 活性
# -> params_delta = {sequential:0, parallel:-4096, parallel-dual-ln:0, dense:+36}
# -> identity_start_bitwise_vs_sequential = {sequential:True, parallel:False,
#                                            parallel-dual-ln:False, dense:True}

# 3) 本轮：后台 launcher（在仓库根运行，run.py 的相对数据路径要求如此）
setsid nohup bash run_remaining.sh >/dev/null 2>&1 &
# 冒烟：python3 run.py --steps 20 --arms sequential,parallel --skip-s0-anchor --caps-min 6 --poll-s 15
# 全跑：python3 run.py --arms sequential,parallel,parallel-dual-ln,dense --caps-min 14 --budget-min 40
```

真实输出（取本轮窗口内实况，逐步补）：

```
[cfg] vs 2613 unequal fields = ['steps', 'train_tokens', 'seed', 'experiment']   # 冒烟 --steps 20 所致
# ↑ 全跑（--steps 12000）时该行应为 ['seed', 'experiment']（唯一的科学字段差异 = seed 0→1）
```

（第一次启动 launcher 时用 `cd <exp dir>` 而非仓库根，`run.py` 报
`FileNotFoundError: .tmp/enwik8`；已改 launcher 为 `cd` 仓库根 + 绝对脚本路径，第二次启动正常。）

```
# 冒烟尝试 1（21:29:47 起）等锁 10 min 未取到：
TimeoutError: 等锁超过 10 min，未取到 GPU 锁
[sh] smoke attempt 1 smoke_ok=no 2026-10-14T21:39:51+08:00
[sh] smoke attempt 2 2026-10-14T21:40:06+08:00    # 仍在等锁
# 锁 holder 全程 = wp2652-2.2.1-residual-form（pid 2383697，training postln）
```

```
# 4) 恢复 config.json 到规范态（冒烟把 --steps 20 / 2 臂写进去了；run.py 启动时会重写）
python3 - <<'EOF'  # 复用 run.py 的 config_vs_2613 重建
# -> unequal_fields = ['seed', 'experiment']
# -> restored: steps 12000 arms ['sequential','parallel','parallel-dual-ln','dense']
```

> `config.json` 现为规范态（12000 步 / 四臂）。下一次 `run.py` 启动仍会重写它；
> 全跑启动时会立即写回同一规范态（先写 config 再等锁，故锁是否取到不影响该文件）。

## 6. 残余清单与下一步（给接手者）

1. **取 GPU 锁 → 20 步冒烟**（`run_remaining.sh` 会自动做；锁被占时 `run.py` 自带最多 10 min 等待，
   launcher 再重试最多 12 轮）。本轮窗口内锁先后被 `2319397`（轴 1 的 1.1.1）、`2383697`
   （`wp2652-2.2.1-residual-form`，用同一把锁但写 `name` 而非 `holder_pid`）持有，未取到；
   后台 launcher 仍在重试。
2. **三条待训臂**：`parallel-s1` / `parallel-dual-ln-s1` / `dense-s1`，seed=1，
   其余与 2613 s0 逐字段相同（`--d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --dense
   --steps 12000 --batch-size 32 --exit-mode uniform --eval-batches 20`）。
   `sequential-s1` 已训完，`arm_trained()` 会自动跳过重训（但仍会重算电池）。
3. **每臂跑完自动增量落盘**（`results.json` + `run_log.txt`），无需人工。
4. 四臂齐后：写 `analysis.md`（config 逐字段对照表 + `SCHEMA.md` §9 等值证明 + 配对 Δ 表 +
   代价面同框表）与 `MODEL.md`（权重哈希）；把 §4 的代价面并入 `analysis.md`。
5. 判词（**口径已预登记，不得事后调整**）：
   - 2.2.3：s0/s1 两 seed 的 `dense − sequential` 配对 Δ 是否越 **0.014 bpc 带**；
     不越 ⇒「无差异」；两 seed 同号且越带 ⇒「可排序」。
   - 2.2.4：`parallel` / `parallel-dual-ln` 相对 `sequential` 的质量代价（>0.014 即代价可排序）
     与代价面同框，明确回答「并行子层在本尺度是否划算」。
   - **墙钟不得进判据或 `cost_model`。**
6. 更新 `axes/2-depth/2.2.3-learned-layer-sum/project.yaml` 与
   `axes/2-depth/2.2.4-parallel-sublayer/project.yaml` 的 `status`/`conclusion`/
   `failure_analysis`/`conclusion_review`/`argument`/`gaps`/`experiments`/`work_packages`。
7. 挂 fresh 复核（`SCHEMA.md` §14），复核排在 writer 停止之后。
