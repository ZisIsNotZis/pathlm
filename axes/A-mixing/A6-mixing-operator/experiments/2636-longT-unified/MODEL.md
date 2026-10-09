# 2636 · 权重清单（MODEL.md）

> 权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）。本文件记录哈希 / 体积 / 生成命令，供复核与复现。
> 契约：同目录 `README.md`；判读：`analysis.md`；对账：`reconcile_report.json`；变异：`mutation_report.json`。

## 1. 生成的 checkpoint（八臂 = 四配置 × 两 seed）

| 臂 | seed | 参数 | 体积 (B) | sha256 | 生成命令 |
|---|---|---|---|---|---|
| `full-256-s6k` | 0 | 6,962,688 | 27,882,155 | `21fa6df2044e552ec0bca10f41e7b4063833a3abff2346008cba34733e8dc051` | `run.py --stage train --arm full-256-s6k` |
| `win128-256-s6k` | 0 | 6,962,688 | 27,882,155 | `6415c3c183048f0034a2d2231f86d221a5f361d74450381dc65b946af18e7aad` | `run.py --stage train --arm win128-256-s6k` |
| `full-512-s6k` | 0 | 7,028,224 | 28,144,299 | `56c10419ea310a65f32f3e0b0b0aa39c626efbf90513d9aab6001ae49545ea2d` | `run.py --stage train --arm full-512-s6k` |
| `win256-512-s6k` | 0 | 7,028,224 | 28,144,299 | `7287a05ba0fe6ec1c72c05992fa8d70b1e48fd9d70d8687b66faa00448e53d49` | `run.py --stage train --arm win256-512-s6k` |
| `full-256-s6k-s1` | 1 | 6,962,688 | 27,882,155 | `9ae7a7d79a6c498e279c6e294a8b52df6c0ead3411bd6d586f87a9522ba8fb73` | `run.py --stage train --arm full-256-s6k-s1` |
| `win128-256-s6k-s1` | 1 | 6,962,688 | 27,882,155 | `536fe5a67557edd2cf69f21b29e819077f9466c966eec7ce5a4dcb96a096d52a` | `run.py --stage train --arm win128-256-s6k-s1` |
| `full-512-s6k-s1` | 1 | 7,028,224 | 28,144,299 | `59230e55054f9b7456f880df8201e7e08f258b1f15c510bded02bfddb984a9f5` | `run.py --stage train --arm full-512-s6k-s1` |
| `win256-512-s6k-s1` | 1 | 7,028,224 | 28,144,299 | `c6eaaa928206464246b804922dbc94b9409dce20bf0ac1f5f6cd2a8c1cf28ff2` | `run.py --stage train --arm win256-512-s6k-s1` |

- 路径：`runs/<arm>/model.pt`（**不入库**）；resolved config 见 `config.json::arms[].fields`。
- 权重哈希与 `config.json::checkpoints[*].sha256` 一致（`sha256_verified: true`）。
- **T=1024/2048 的同 step 点不入本实验**（只读引用 `2626`/`2629` 的 `model.pt`，哈希见其 `MODEL.md`）。

## 2. 复现入口与参数

| 阶段 | 命令 |
|---|---|
| 冒烟 | `python3 run.py --stage preflight` |
| 训练（单臂/全部，断点续跑） | `python3 run.py --stage train --arm <arm>` / `--stage train-all` |
| 评测 | `python3 run.py --stage eval` |
| 汇总 | `python3 run.py --stage report` |
| 对账 | `python3 reconcile_numbers.py` |
| 变异 | `python3 mutate.py` |

固定参数：`d=256, L=8, H=8, mlp_mult=4, dense=True, topology=sequential, bs=8, lr=1e-3,`
`exit_mode=uniform, steps=6000（同 step 口径）, qk-normfrom512`。
数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval / vocab 206）。

## 3. 共享文件指纹（本实验**未改**，见 README §3.1）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `b304c234b82b51a49ba477285169b41ebf3fc4e24b087ec2bafdd9caa4423192` |
| `pathlm/model.py` | `75bd84a013c71f2e6eef479517cc99b96d8cebff672383f2728f2752812655d3` |
| `train_depth_ar.py` | `218b63179e623d6a938e7cabe85a71c5c4a36f1831c6930e9c65b2e0a6673022` |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` |
| `experiments/2625-longT-stability/stability_driver.py` | `b7f49df7de6ea1d8aa9f2663c9c4d09272bae5dfb1389d5f211cc5c33acc3f20` |

> 本实验未改这些文件（只读）；qk-norm hook 从 2625 的 stability_driver.py 只读引用（单一实现来源）
