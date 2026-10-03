# 2625 · 权重清单（MODEL.md）

**权重不入库**（`.gitignore` 覆盖 `runs/**/model.pt`；本文件只登记路径与用途）。

## 1. 本实验产出的 checkpoint（不入库）

| 臂 | 路径 | 用途 |
|---|---|---|
| `full-T1024-qknorm-s0` | `runs/full-T1024-qknorm-s0/model.pt` | 候选 `qk-norm` @ seed0，T=1024，12000 步（判据①） |
| `full-T1024-qknorm-s1` | `runs/full-T1024-qknorm-s1/model.pt` | 候选 `qk-norm` @ seed1，T=1024，12000 步（判据①） |
| `full-1024-s1-diag` | `runs/full-1024-s1-diag/model.pt` | 诊断臂（seed1、7000 步、候选 `none`）——**注意**：`steps` 变短会同时压缩余弦调度，故本臂不是 12000 步跑的「前缀」 |
| `full-T256-qknorm256-s0` | `runs/full-T256-qknorm256-s0/model.pt` | 候选 `qk-norm` @ T=256，bs32，12000 步（判据②） |
| `full-T512-qknorm512-s0` | `runs/full-T512-qknorm512-s0/model.pt` | 候选 `qk-norm` @ T=512，bs32，6000 步（判据②） |
| `full-T512-none512-s0` / `...b-s0` | `runs/full-T512-none512*-s0/model.pt` | 候选 `none` @ T=512（同配方重跑两次——用于量 run-to-run 散度） |
| `full-T512-cond512-s0` | `runs/full-T512-cond512-s0/model.pt` | 候选 `qk-normfrom512` @ T=512（零 hook、判据②的结构性口径） |
| `preflight-20` / `tm300` | `runs/preflight-20`、`runs/tm300` | 步时/保真度锚点 |

> 全部需从本目录 `runs/` 重新生成（驱动器 `stability_driver.py` + `run.py`）。

## 2. 本实验**只读引用**的既有 checkpoint（跨 run）

| 臂 | 路径 | 用途 |
|---|---|---|
| `full-256` / `win128-256` / `full-512` / `win256-512` | `../2622-longT-512/runs/*/model.pt` | 短 T 基准（判据②）与「疤」指纹对照 |
| `full-1024` / `win512-1024` | `../2623-longT-1024/runs/*/model.pt` | seed0 爆炸基准与对照 |
| `full-1024-seed1` / `win512-1024-seed1` | `../2624-longT-1024-seed1/runs/*/model.pt` | seed1 爆炸基准与对照 |

## 3. 代码指纹（本实验**未改动**任何共享文件）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` |
| `pathlm/model.py` | `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f` |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` |
| `tests/test_depth_ar.py` | `d50ca38cc6bc5a9f71be543d73672e39ab7f6531a9e3f953fb2f7431c0e63c15` |

> 候选补丁全部在实验本地驱动器 `stability_driver.py` 的**进程内**完成
> （forward hook / 构造后重初始化 / 优化器与调度的实参替换），
> 因此上表四个文件在本实验全程逐位不变（`mutation_report.json::restored_bitwise` 佐证）。

## 4. 保真度锚点

驱动器候选 `none` 的 step0 读数与引擎 2623 的 `full-1024` **逐位相同**：
`loss 5.2979`、`gnorm 7.28`（`runs/preflight-20/train_log.jsonl` 首行 vs
`../2623-longT-1024/runs/full-1024/train_log.jsonl` 首行；由
`tests/test_longT_stability.py::test_driver_preflight_step0_matches_engine_2623` 钉住）。

⚠️ **但仅 step 0 逐位相同**：本任务的训练**不是 run-to-run 逐位可复现的**。同配方重跑两次
（`runs/full-T512-none512-s0` vs `runs/full-T512-none512b-s0`）的 `train_log` 从 step 200 起
最大 `|Δloss| = 0.0320`；T=512 的 clean bpc 极差 **0.004534**（1.569772 / 1.574306）。
本格因此**不用「逐位」作判据**（详见 `stability_report.md §4.1`）。
