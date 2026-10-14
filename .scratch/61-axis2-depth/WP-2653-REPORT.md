# WP-2653 · 2.2.2 层间 causal attention 最小判决 · 执行报告

> 过程工单（`ticket 61 · 轴 2 深度轴`）。事实文档 = `axes/2-depth/2.2.2-attention-residual/project.yaml`
> + `axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/`。
> writer = WP-2653 worker（fresh 上下文，唯一写入者）。actor/时间见 `change_log` 与 `run_log.txt`。

## 0 一句话

层间 causal attention **已实装并证明接线正确**（把层维权重强制成「只取紧邻上一层」时，
前向与损失与 stock 单残差**逐位一致**），两臂的**代价面与活性已实测落盘**；
**训练判决（clean bpc / 配对 Δbpc）本轮未跑**——共享 4090 被外部任务占用、等锁未取到，
等效成本 2 × 98.3M token 已登记。判据线**未改、未砍**（按契约砍的是排期，不是判据线）。

## 1 交付物（落盘清单）

| 文件 | 状态 |
|---|---|
| `README.md`（预登记契约，SCHEMA §6 九节，P1–P6 判据线） | ✅ |
| `config.json`（resolved config + 两臂定义 + 复用基线口径） | ✅ |
| `interlayer_attn.py`（层间聚合实现；实验内工厂替换，不改 `pathlm/**`） | ✅ |
| `run.py`（一条命令复现全臂；`--arms` 按臂续跑；`--no-train`；`--aggregate`；按臂取/释放锁） | ✅ |
| `smoke.py` + `smoke_report.json`（等值门 / 活性门 / 参数账，各带必红探针） | ✅ 全过 |
| `test_2653_interlayer_attn.py`（9 passed） | ✅ |
| `gpu_lock.py`（`SCHEMA.md` §20 协议） | ✅ |
| `run_remaining.sh`（有界后台 launcher：8 轮、每轮每臂最多等锁 10 min、已完成臂自动跳过） | ✅（在跑） |
| `analysis.md`（判读；未测项如实标「未测」） | ✅ |
| `MODEL.md`（权重清单；本轮无权重 ⇒ 标「未生成」） | ✅ |
| `results_<arm>_s0.json` / `results.json` / `run_log.txt` | ⏳ 待臂跑完（run.py 按臂增量落盘） |

## 2 实测（已落盘，可独立复算）

| 项 | `attn-scalar` | `attn-qkv` |
|---|---|---|
| 额外参数 | +8 | +196 616 |
| 每 token 层间聚合额外 MACs | 18 432（+0.59% vs stock 每层注意力 3 145 728） | 5 261 312（+167.3%） |
| 等值门（one-hot ⇒ 与 stock 逐位一致） | ✅ 前向 + loss | ✅ 同一聚合路径 |
| 活性（k ≥ 2：`w_immediate_prev < 1`、熵 > 0） | ✅ | ✅ |
| 单测 | 9 passed | — |

探针（证明门会红）：不强制时前向**必须**与 stock 不同 ⇒ `true`；`MODE`/`FORCE_LAST` 在建实例时
冻结（前向时读全局会让等值门静默失效——本轮**实际踩到过这个 bug 并修掉**，回归单测 `test_factory_global_is_frozen_at_construction`）。

## 3 机制性发现（如实记录，影响后续解读）

1. **起点不等于 stock 起点**：`q = h_{k−1}` 与 key 集合里的自相似使训练起点处层维权重**已经**
   偏离「只取上一层」（k ≥ 2 的层熵 ≈ 1.6–2.1、`w_immediate_prev` ≈ 0.15–0.24，且最大权重
   落在紧邻上一层）。即 attn 臂不是「从单残差出发再学偏离」，而是「从近似均摊出发」。
   这是该参数化固有的性质，已落盘；结论只能来自训练后的 B 表。
2. **代价面很不对称**：`attn-scalar` 与 2.2.3 的 `dense`（+36）几乎同价（+8 参数、+0.59% 注意力算力），
   `attn-qkv` 则要 +3d² 参数与 +167% 注意力算力（因为共享投影对每层的 k 个源各做一次）。
   故「内容相关权重值不值」有一个很干净的价签。

## 4 阻塞与剩余（诚实登记）

- **阻塞**：`/tmp/pathlm_gpu.lock` 在本轮窗口内被他人 launcher 持续取用（`wp6.3.2-*` 系列、
  `2651` 的 smoke 进程、另有进程 `2420657` 长时间训练）；本 WP 的 `run.py` 每臂最多等锁 10 min
  （`SCHEMA.md` §20），等不到即记 `gpu_lock_timeout` 而非抢锁/杀进程。
- **环境异常（登记，不改他人）**：本 WP 的 uniform 臂实际训练期间，观察到另一进程
  （pid 2420657，4.9GB）与我的进程（1.5GB）**同时占用 GPU**，而锁目录的 `name` 由他人 launcher
  改写。锁协议在多方 launcher 并发释放/重取时存在竞态（`rmdir` 后双方各自继续）——本 WP 按
  §20 自愈规则**不回收、不动他人锁**，只如实登记；该现象会让我方的实测步时变慢（外因，只影响规模/排期，不改科学设定）。
- **等效成本（不进判据）**：`attn-scalar` + `attn-qkv` 各 12k 步 × batch32 × seq256 = **98.3M token/臂**，
  合计 **196.6M token**。等锁超时**没有消耗任何 token**（只消耗排期）⇒ 判「受阻」，不是「失败」。
- **剩余步骤（一条命令，可续跑）**：
  `bash axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/run_remaining.sh`
  （或 `python3 run.py --arms attn-scalar,attn-qkv --seed 0 --steps 12000`），跑完再 `--aggregate`。
- **收口决定（22:24，lead 指令）**：GPU 锁被轴 6 的多臂驱动长期独占，等锁不会在本轮窗口内解决
  ⇒ **不再等锁**，本 WP 收口（两臂未训如实登记）。`run_remaining.sh` **未启动**（避免与 2655 的
  launcher 自相争抢），保持可在有窗口时按臂续跑。

## 5 复核

**独立复核未做**（`SCHEMA.md` §14/§14b：复核排在 writer 停止之后，且须为 fresh 上下文；
本轮 window 内未启动）。writer 自我核对项：等值门/活性门/参数账/单测/探针各有落盘证据，
未测数值一律标「未测」，无占位数字。
