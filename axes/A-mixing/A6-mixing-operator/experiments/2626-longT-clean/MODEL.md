# 2626 · 权重清单（MODEL.md）

> 权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）。本文件记录哈希 / 体积 / 生成命令，供复核与复现。
> 契约：同目录 `README.md`；判读：`analysis.md`；对账：`reconcile_report.json`。

## 1. 生成的 checkpoint（六臂）

| 臂 | 参数 | 体积 (B) | sha256 | 生成命令 |
|---|---|---|---|---|
| `full-256-clean` | 6,962,688 | 27,882,155 | `6eba2dcc62711569559dd34b65a7df3116bf7e967a605074fb73d752cc9ac420` | `run.py --stage train --arm full-256-clean` |
| `win128-256-clean` | 6,962,688 | 27,882,155 | `db1d59b58bc4533b26449b90b4b9f02786c3ed7235cff219ec434917a6fb676f` | `run.py --stage train --arm win128-256-clean` |
| `full-512-clean` | 7,028,224 | 28,144,299 | `9bd9c233f542b7b2691958e81e36b712456e016e64caf76e75ae6b5cc80d077f` | `run.py --stage train --arm full-512-clean` |
| `win256-512-clean` | 7,028,224 | 28,144,299 | `3b5b0dad44b279f4a207d27612560907f8f592fa66e02926cc87a46cabb947dd` | `run.py --stage train --arm win256-512-clean` |
| `full-1024-clean` | 7,159,296 | 28,668,587 | `e5009af94470ecdef6303d3c361686282451d7198de391168ed3fe5483667756` | `run.py --stage train --arm full-1024-clean` |
| `win512-1024-clean` | 7,159,296 | 28,668,587 | `0c2779a262252ad10794ae6b22424889a5c75b68a20423f575927a2b48b6fdeb` | `run.py --stage train --arm win512-1024-clean` |

- 每个臂的训练命令由 `run.py` 生成：`python3 clean_driver.py <arm> --seq-len T --window w --steps S --batch-size 8 --lr 1e-3 --seed 0 --candidate qk-normfrom512`。
- 路径：`runs/<arm>/model.pt`（**不入库**）；resolved config 见 `config.json::arms[].fields`。
- 权重哈希与 `config.json::checkpoints[*].sha256` 一致（`sha256_verified: true`）。

## 2. 复现入口与参数

| 阶段 | 命令 |
|---|---|
| 冒烟 | `python3 run.py --stage preflight` |
| 训练（单臂/全部） | `python3 run.py --stage train --arm <arm>` / `--stage train-all` |
| 评测 | `python3 run.py --stage eval` |
| 汇总 | `python3 run.py --stage report` |
| 对账 | `python3 reconcile_numbers.py` |
| 变异 | `python3 mutate.py` |

固定参数：`d=256, L=8, H=8, mlp_mult=4, dense=True, topology=sequential, bs=8, lr=1e-3, seed=0,
exit_mode=uniform, token_budget=49,152,000/臂`（T=256→24000 步、T=512→12000 步、T=1024→6000 步）。
数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval / vocab 206）。

## 3. 共享文件指纹（本实验**未改**，见 README §9）

| 文件 | sha256（本实验全程） |
|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` |
| `pathlm/model.py` | `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f` |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` |
| `experiments/2625-longT-stability/stability_driver.py`（**只读引用**：QK-Norm hook） | `b7f49df7de6ea1d8aa9f2663c9c4d09272bae5dfb1389d5f211cc5c33acc3f20` |

`config.json::code_fingerprints` 记录同一组哈希；`mutate.py` 前后 sha256 证明三个本实验相关文件
逐字节还原（`mutation_report.json::restored_bitwise = true`）。

## 4. 可入库证据副本（`runs/` 被 `.gitignore` 忽略）

本目录下的 `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` / `<arm>-probe.jsonl`
（六个臂 × 3 个文件）是 `runs/<arm>/` 下同名文件的**可入库副本**（与 2623 同约定）。
权重 `model.pt` **不入库**，只留上面的 sha256。
