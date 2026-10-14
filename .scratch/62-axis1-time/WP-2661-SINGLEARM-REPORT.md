# WP-2661-SINGLEARM · 单臂并发插入尝试报告

## 0 人话摘要

我按 `SCHEMA.md` §20.1.2 的「短任务并发插入」试了一把：在 GPU 空闲 16 GB（远超 3 GB 闸门）、
**不取锁**的前提下，把 `2661` 的 6 臂电池砍成单臂跑。结果**一条读数都没拿到**——`run.py` 在
**任何测量开始之前**就自己崩了。原因很具体：它在「自检①」里要核对「臂集内同时存在无通道臂和
通道臂、且各自参数量一致」，只放一条臂时这个核对必然失败（只留通道臂 ⇒ 无通道臂列表为空；
只留无通道臂 ⇒ 通道臂列表为空）。所以本次**没有** `bpc_win16`、没有 `Δbpc_far`、没有 needle
读数，**也不是**「实验失败」或「机制无效」——这只是一次**跑不通**的尝试，`run.py` 单臂路径本身
不成立。产物只有两份单臂配置（已核对与 `config.json` 只差 `arms` 字段）和两份崩溃日志。

## 1 标签纪律（本读数的地位：补充证据，不是判据）

这份报告里的东西**不得**当作 `1.1.4.5` 的判据：

1. 它是**单臂**读数路径（预登记是 6 臂）——跨臂硬同索引子集、σ_cell、explicit−implicit 对撞
   全部拿不到；
2. 即便跑通，批数仍是 96/64，但**臂集残缺**；本次连批数都没跑到；
3. 它**不得**用于「1.1.4.5 被否证」这类否定结论，只用于报方向与 CI（Main 2026-10-14 裁决）。

本次实际结论更弱：**零读数**，连方向都报不了。

## 2 任务边界（我只写了这些地方）

- 允许写：`axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/single_arm/`（新建）+ 本报告。
- **未改**：`README.md`、`analysis.md`、`run.py`、`config.json`、任何 `project.yaml`、`pathlm/**`、`tools/**`。
- **未做**：取 / 放 GPU 锁、动锁目录、杀进程、跑任何 `git` 命令。

## 3 单臂配置生成（步骤 1）

以 `config.json` 为模板，用 python 改副本（不手写）：

| 文件 | `arms` |
|---|---|
| `single_arm/config_carrypl16_only.json` | `["carrypl16_s0"]` |
| `single_arm/config_win16_only.json` | `["win16_s0"]` |

核对（脚本化，非目测）：`config.json` 与两份副本做深比较，仅把 `arms` 换成副本的 `arms` 后
`identical_except_arms: True`（两份都 True）⇒ `data / model / visibility / eval / needle /
retention / bands / arm_roles / reproduction / seeds` 一字未动。

`arm_roles` **故意保留原样**（含 6 臂）；因此即使下面的断言不崩，`arm_roles` 驱动的跨臂块
（`needle.primary_hard_subset` 等）仍会 KeyError——单臂路径在 `run.py` 里没有可用 bypass，
而 ticket 明令不许改 `run.py`。

## 4 并发插入前置检查 + 运行（步骤 2/3）

每次运行前查 `nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader`，
free ≥ 3 GB 才跑。全部达标，未发生退避重试。

| 时刻（`date -Iseconds`） | 事件 | nvidia-smi free/used (MiB) |
|---|---|---|
| 2026-10-14T22:09:13+08:00 | 会话首查 | 16450 / 7659 |
| 2026-10-14T22:09:40+08:00 | 写两份单臂配置 | — |
| 2026-10-14T22:09:43+08:00 | 复核副本 only-arms-differ | 16466 / 7643 |
| 2026-10-14T22:09:46+08:00 | **carrypl16_only** 前置检查（记录在日志内） | 16449 / 7660 |
| 2026-10-14T22:09:57+08:00 | **carrypl16_only** rc=1（崩溃） | 16443 / 7665 |
| 2026-10-14T22:10:16+08:00 | **win16_only** 前置检查（记录在日志内） | 16442 / 7667 |
| 2026-10-14T22:10:26+08:00 | **win16_only** rc=1（崩溃） | — |

命令（原文，每次运行前已过 3 GB 闸门）：

```
python3 run.py --config single_arm/config_carrypl16_only.json --out-dir single_arm/   > single_arm/run_log_carrypl16.txt 2>&1
python3 run.py --config single_arm/config_win16_only.json   --out-dir single_arm/   > single_arm/run_log_win16.txt   2>&1
```

**未取锁**（`§20` 锁目录未被触碰）。

## 5 实测：单臂不可跑（失败要 loud）

两条都崩在「自检①」，**早于** bpc / Δbpc_far / needle / 因果任何一条腿 ⇒ 没有可交的测量读数。

**carrypl16_only**（`single_arm/run_log_carrypl16.txt`）：

```
[    1.7s]   carrypl16_s0: 382a74aa0e41… kind=carry rule=delta per_layer=True chunk=256
                              state_floats=98304 params=13801064
[    1.7s] 自检①：窗口掩码非退化 + none/delta(共享+逐层) 主干逐位 + 单边零起步 + 参数量账
Traceback (most recent call last):
  File ".../2661-implicit-vs-explicit-recall-13m/run.py", line 706, in <module>
    main()
  File ".../2661-implicit-vs-explicit-recall-13m/run.py", line 296, in main
    assert len({ck[n]["n_params"] for n in none_names}) == 1, "无通道臂参数量不一致"
AssertionError: 无通道臂参数量不一致
```

**win16_only**（`single_arm/run_log_win16.txt`）：

```
[    1.4s]   win16_s0: c3f6f08a4fd1… kind=window rule=None per_layer=False chunk=None
                        state_floats=0 params=13536768
[    1.4s] 自检①：窗口掩码非退化 + none/delta(共享+逐层) 主干逐位 + 单边零起步 + 参数量账
Traceback (most recent call last):
  File ".../2661-implicit-vs-explicit-recall-13m/run.py", line 706, in <module>
    main()
  File ".../2661-implicit-vs-explicit-recall-13m/run.py", line 297, in main
    assert len({ck[n]["n_params"] for n in carry_names}) == 1, "通道臂参数量不一致"
AssertionError: 通道臂参数量不一致
```

根因（读 `run.py:293-297`）：

```python
none_names  = [s["name"] for s in cfg["arms"] if s["rule"] in (None, "none")]
carry_names = [s["name"] for s in cfg["arms"] if s["rule"] not in (None, "none")]
assert len({ck[n]["n_params"] for n in none_names})  == 1, "无通道臂参数量不一致"
assert len({ck[n]["n_params"] for n in carry_names}) == 1, "通道臂参数量不一致"
```

单臂时其中一组必为空集，`len(set()) == 0 != 1` ⇒ 断言失败。**这是 run.py 的臂集前置条件，
不是实验失败信号**，与机制是否有效无关。

两条 checkpoint 的 sha256 断言**都通过了**（日志里打印的前缀
`382a74aa0e41…` / `c3f6f08a4fd1…` 与 `config.json` 一致），即文件与权重没问题。

## 6 墙钟 / 峰值显存 / §20.1.2 判定

- **墙钟（过程指标，不进任何判据）**：carrypl16_only ≈ 11 s（22:09:46 → 22:09:57，含解释器启动）；
  win16_only ≈ 10 s（22:10:16 → 22:10:26）。两者都 **< 2 min**。
- **峰值显存**：**未取得**。`run.py` 只在 `results.json.cost.eval_peak_gpu_mem_mb` 里报
  `torch.cuda.max_memory_allocated()`，而它在写盘之前就崩了 ⇒ 无此数（外部 `nvidia-smi` 前后
  读数 7659→7665 / 7667 MiB used，只反映「进程已退出、几乎全释放」，不构成峰值证据）。
- **§20.1.2 判定**：墙钟上限**名义上未触及**，但两次运行**都没产出任何测量**，因此这**不是**
  「满足 §20.1.2 的并发插入」，只能记作**空跑**。峰值上限无法判定（未取得）。

## 7 结论与建议

1. **结论**：`run.py` 的单臂路径**不可跑**；本轮**零读数**，`carrypl16_s0` / `win16_s0` 的
   `bpc_win16`、`Δbpc_far`(+CI)、`max_abs_logit_diff_on_eval_rows`、`channel_off` 抓力、
   因果性探针、needle 臂内 `beyond`「植入 vs 配对地板」读数**全部未跑**（原因同一：早崩于自检①）。
2. **若要拿到这条补充读数**，唯一不改 `run.py` 的路子是在 `single_arm/` 放一个**只读助手**
   （import `run.py` 既有 helper + 同一台仪器 `realtext_probe` / `tools/evalkit/*`，
   kwargs 一律经 `rp.probe_kwargs(cfg)` 与 `config` 派生，口径一字不改），跳过跨臂断言，
   只跑单臂的 bpc / Δbpc_far / 抓力 / 因果 / needle 臂内配对读数；跨臂硬同索引子集如实标
   「单臂拿不到」。**这是一次方法取舍，按 §4a 与 `AGENTS.md` §3 归 lead 决定**，执行方不自行决定。
3. 本次已就上述选项向 lead 发了 `need_decision`（A=授权助手 / B=如实收尾），两次均超时未回。
   按「未回复即走保守」的自我声明，本报告按 **B** 收尾（与 ticket 原指示一致）；A 作为建议留档。
4. **不建议**为单臂去改 `run.py`（改判据前的自检 = 动预登记仪器）；如需单臂，应让 lead 决定
   是「助手旁路」还是「按 6 臂取锁排队」。

## 8 产物清单

| 路径 | 内容 |
|---|---|
| `axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/single_arm/config_carrypl16_only.json` | 单臂配置（arms=[carrypl16_s0]） |
| `.../single_arm/config_win16_only.json` | 单臂配置（arms=[win16_s0]） |
| `.../single_arm/run_log_carrypl16.txt` | 前置检查 + 全过程 + 崩溃 traceback + rc=1 + 时间戳 |
| `.../single_arm/run_log_win16.txt` | 同上（win16_only） |
| `.scratch/62-axis1-time/WP-2661-SINGLEARM-REPORT.md` | 本报告 |

未产出 `single_arm/results_*.json`（崩溃早于落盘；已确认目录内无 `results*.json`）。
