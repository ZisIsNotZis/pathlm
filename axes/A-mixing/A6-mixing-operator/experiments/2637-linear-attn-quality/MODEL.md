# 2637 · 权重清单（MODEL.md）

> 权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）。本文件记录哈希 / 体积 / 生成命令，供复核与复现。
> 契约：同目录 `README.md`；判读：`analysis.md`；对账：`reconcile_report.json`；变异：`mutation_report.json`。

## 1. 生成的 checkpoint（四臂）

| 臂 | 算子 | QK-Norm | 参数 | 体积 (B) | sha256 | 生成命令 |
|---|---|---|---|---|---|---|
| `full-256-lin` | full | none | 6,962,688 | 27,882,155 | `21fa6df2044e552ec0bca10f41e7b4063833a3abff2346008cba34733e8dc051` | `run.py --stage train --arm full-256-lin` |
| `linear-256-lin` | linear | none | 6,962,688 | 27,882,155 | `67966cbc26cdfa6fe8fd35fd2fae76234f3cffa6da13886bd12033f870bf8bbe` | `run.py --stage train --arm linear-256-lin` |
| `full-256-lin-qk` | full | global | 6,962,688 | 27,882,155 | `e7ac0f3c3f4045513289d2901a2bf0ee75b9f9a924620bff764a68acac24d915` | `run.py --stage train --arm full-256-lin-qk` |
| `linear-256-lin-qk` | linear | global | 6,962,688 | 27,882,155 | `75d0ce27e8e4e1181b962721a62e3aad096973bde69be3f504bcc52f7d54982f` | `run.py --stage train --arm linear-256-lin-qk` |

- 路径：`runs/<arm>/model.pt`（**不入库**）；resolved config 见 `config.json::arms[].fields`。

## 2. 复现入口与参数

| 阶段 | 命令 |
|---|---|
| 冒烟 | `python3 run.py --stage preflight` |
| 训练（单臂/全部） | `python3 run.py --stage train --arm <arm>` / `--stage train-all` |
| 停摆诊断矩阵 | `python3 run.py --stage diag` |
| 评测 | `python3 run.py --stage eval` |
| 汇总 | `python3 run.py --stage report` |
| 对账 | `python3 reconcile_numbers.py` |
| 变异 | `python3 mutate.py` |

固定参数：`d=256, L=8, H=8, mlp_mult=4, dense=True, topology=sequential, bs=8, lr=1e-3, seed=0,`
`seq_len=256, steps=6000, linear_chunk=64`。数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval / vocab 206）。

## 3. 共享文件指纹（本实验**未改**）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `f2e72361eecc022e7ccf8a7951f6a6f9a411e12d6da9348a4d8fb70023b9440e` |
| `train_depth_ar.py` | `218b63179e623d6a938e7cabe85a71c5c4a36f1831c6930e9c65b2e0a6673022` |
| `experiments/2625-longT-stability/stability_driver.py` | `b7f49df7de6ea1d8aa9f2663c9c4d09272bae5dfb1389d5f211cc5c33acc3f20` |

> 本实验未改这些文件（只读）
