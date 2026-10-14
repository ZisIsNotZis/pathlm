# WP-2655 · 2.3.1 逐层容量 schedule 最小判决 · 执行报告

> 过程工单（`ticket 61 · 轴 2 深度轴`）。事实文档 = `axes/2-depth/2.3.1-layer-capacity/project.yaml`
> + `axes/2-depth/2.3.1-layer-capacity/experiments/2655-layer-capacity-13m/`。
> writer = WP-2655 worker（fresh 上下文，唯一写入者）。

## 0 一句话

三臂的相对宽 schedule 与**等预算门/引擎等值门/活性门**都已做完并落盘；`uniform` 臂训完
（clean bpc **1.5291**，与同配方历史锚差 <0.005）；**`front-big` / `growing` 两臂未跑**
（共享 4090 被外部任务持续占用、等锁未取到；等效成本 2 × 98.3M token 已登记），
故主判据（配对 Δbpc）本轮无读数。判据线**未改、未砍**。

## 1 交付物（落盘清单）

| 文件 | 状态 |
|---|---|
| `README.md`（预登记契约，SCHEMA §6 九节，P1–P6 判据线） | ✅ |
| `config.json`（resolved config + 三臂 schedule + 历史锚） | ✅ |
| `run.py`（一条命令复现全臂；`--arms` 按臂续跑；`--no-train`；`--aggregate`；按臂取/释放锁） | ✅ |
| `smoke.py` + `smoke_report.json`（等预算门 / 引擎等值门 / 活性门 / CLI 接线，各带必红探针） | ✅ 全过 |
| `test_2655_layer_capacity.py`（8 passed） | ✅ |
| `gpu_lock.py`（`SCHEMA.md` §20 协议） / `run_remaining.sh`（有界 launcher：8 轮、每臂最多等锁 10 min、已完成臂自动跳过） | ✅（在跑） |
| `results_uniform_s0.json`（逐臂增量） / `results.json`（汇总，已含 P4/P5 复核块） | ✅（仅 uniform 臂） |
| `analysis.md` / `MODEL.md` / `run_log.txt` | ✅ |
| `results_front-big_s0.json` / `results_growing_s0.json` | ⏳ 待两臂跑完（run.py 按臂增量落盘） |

## 2 实测（已落盘）

| 项 | 结果 |
|---|---|
| 三臂 Σh | 8192（逐位相等） |
| 三臂总参数 | 6 962 688（逐位相等） |
| 三臂 FFN 参数 | 4 204 544（逐位相等；解析式与实测互核一致） |
| 三臂每 token FFN MACs | 4 194 304（逐位相等；总 MACs 7 392 768） |
| **引擎等值门**（显式 `1024,…,1024` == 引擎缺省 `--mlp-mult 4`） | ✅ state_dict + 前向**逐位一致**；探针（改一位 ⇒ 必不等）✅ |
| 活性 | ✅ 逐层隐藏维 == schedule；逐层 ‖FFN_out‖/‖FFN_in‖ = 0.484–2.014 |
| `uniform` 臂 clean bpc | **1.5291**（窗口法 == `depth_curve` 末深度；逐深度 acc 0.6894） |
| `uniform` 逐窗 bpc（配对 Δ 用） | [1.5163, 1.5740, 1.5349, 1.5105, 1.5098] |
| 同配方历史锚对照（P5 训练层旁证） | 2613 sequential-s0 1.5254（Δ=+0.0037）；2641 U4-s0 1.5314（Δ=−0.0023）；均 <0.005 ✅ |
| 非零 loss 斜率 | 末 25% 窗 +0.00219 /1k 步（仍非零 ⇒ 只作快照口径） |
| 过程量（**不进判据**） | 训练 98.3M token；峰值显存 932.6 MB；墙钟 8.18 min/臂 |

## 3 阻塞与剩余（诚实登记）

- **阻塞**：`/tmp/pathlm_gpu.lock` 在本轮窗口内被他人 launcher 持续取用（`wp6.3.2-*` 系列、
  2651 的 smoke 进程、另有进程 `2420657` 长时间训练）。本 WP 的 `run.py` 每臂最多等锁 10 min
  （`SCHEMA.md` §20），等不到即记 `gpu_lock_timeout`，**不抢锁、不杀进程、不动他人锁**。
- **环境异常（登记，未处置）**：uniform 臂训练期间观察到**两个进程同时占用 GPU**
  （我的 `run.py` 1.5GB + 他人 pid 2420657 4.9GB），而锁目录的 `name` 由他人 launcher 改写：
  多方 launcher 并发「按臂释放/重取」时锁协议存在竞态（一方 `rmdir` 后另一方继续训练）。
  我方按 §20 不回收、不动他人锁，仅如实登记；该现象只让我方步时变慢（外因，只影响排期）。
- **等效成本（不进判据）**：每臂 12k × batch32 × seq256 = **98.3M token**；两臂剩余 **196.6M token**
  （≈ 2 × 7–9 min 卡时）。等锁超时**没有消耗任何 token** ⇒ 判「受阻」，不是「失败」。
- **剩余步骤（一条命令，可续跑）**：
  `bash axes/2-depth/2.3.1-layer-capacity/experiments/2655-layer-capacity-13m/run_remaining.sh`
  （或 `python3 run.py --arms front-big,growing --seed 0 --steps 12000`），跑完 `results.json`
  的 P1/P2/P3 与 `P4…pass` 会自动补齐（`--aggregate`）。

## 4 复核

**独立复核未做**（`SCHEMA.md` §14/§14b：复核排在 writer 停止之后、须 fresh 上下文；本轮未启动）。
writer 自我核对：所有已报数字都有 `results_uniform_s0.json` / `smoke_report.json` / `run_log.txt`
可回溯；未跑出的臂一律标「未测」，未填占位数字；判据线与 README 预登记原文逐条一致。
