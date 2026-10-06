# 2629 · 权重清单（MODEL.md）

> 权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）。本文件记录哈希 / 体积 / 生成命令，供复核与复现。
> 契约：同目录 `README.md`；判读：`analysis.md`；对账：`reconcile_report.json`；
> 变异：`mutation_report.json`。

## 1. 生成的 checkpoint（两臂，各 6000 步）

| 臂 | 参数 | 体积 (B) | sha256 | 生成命令 |
|---|---|---|---|---|
| `full-2048-clean` | 7,421,440 | 29,717,163 | `5e0006a3f4e01af4b6ca35d2bcc7379fa3bee230e49760d4582441a0fc2072da` | `run.py --stage train --arm full-2048-clean` |
| `win1024-2048-clean` | 7,421,440 | 29,717,163 | `68f14bc4eeed04c46b3186b7a9dc8b85320f94eb6f75d375e2ebdd22fa576f62` | `run.py --stage train --arm win1024-2048-clean` |

- 每个臂的训练命令由 `run.py` 生成：
  `python3 clean_driver.py <arm> --seq-len 2048 --window {0,1024} --steps 6000 --batch-size 8 --lr 1e-3 --seed 0 --candidate qk-normfrom512`。
- 路径：`runs/<arm>/model.pt`（**不入库**）；resolved config 见 `config.json::arms[].fields`。
- 权重哈希与 `config.json::checkpoints[*].sha256` 一致（`sha256_verified: true`）。
- 与 2628 的关系：**同配方、只差步数**（3000 → 6000）⇒ 哈希必然不同；`full` 臂与 2628 的
  `full-2048-clean` 是**同一个模型的两次快照**。

## 2. 复现入口与参数

| 阶段 | 命令 |
|---|---|
| 能力面 | `python3 capability_check.py` |
| 冒烟 | `python3 run.py --stage preflight` |
| 训练（单臂/全部） | `python3 run.py --stage train --arm <arm>` / `--stage train-all` |
| 评测 | `python3 run.py --stage eval` |
| 汇总 | `python3 run.py --stage report` |
| 证据副本 | `python3 run.py --stage promote` |
| 对账 | `python3 reconcile_numbers.py` |
| 变异 | `python3 mutate.py` |

固定参数：`d=256, L=8, H=8, mlp_mult=4, dense=True, topology=sequential, bs=8, lr=1e-3, seed=0,
exit_mode=uniform, steps=6000, token_budget=98,304,000/臂`（= 2 × 该系列其它点的 49.152M，
因为是 **matched-step** 口径）。
数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval / vocab 206）。

## 3. 共享文件指纹（本实验**未改**，见 README §9；与 2628 记录值一致）

| 文件 | sha256（本实验全程） |
|---|---|
| `pathlm/depth_ar.py` | `45f946793835f2afadf3a10f727b02ee879682ef6024b23fa179d3351c28c585` |
| `pathlm/model.py` | `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f` |
| `train_depth_ar.py` | `2d1863bbc4403e026d12b597d98a1da2b85f3606110073f2117a7dd1db2a1f35` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` |
| `experiments/2625-longT-stability/stability_driver.py`（**只读引用**：QK-Norm hook） | `b7f49df7de6ea1d8aa9f2663c9c4d09272bae5dfb1389d5f211cc5c33acc3f20` |

`config.json::code_fingerprints` 记录同一组哈希。
**口径说明**：这组值来自**仓库当前版本**；截至本实验落盘时刻 `git diff HEAD -- pathlm/
train_depth_ar.py tools/` 为**空** ⇒ 本实验**未改**共享文件。与本轮并行的另一条线
（`2629-carry-ragged-projfill`）的未提交改动只在其**自己的目录**与
`tests/test_carry_ragged_projfill_13m.py`（`git status --short` 可核），**不在上表路径下**
⇒ 两线无共享文件重叠（§22）。

## 4. 本实验自身文件指纹（§22.6）

| 文件 | sha256 |
|---|---|
| `clean_driver.py` | `5484cba82f138582f3c10d4cb4ddf54c71b8551021e349a7110789bc0bb284d6` |
| `run.py` | `a7c03207f02c2319e30b9b8e08dc475115238c9b76c2e926a27ba7f1516d2069` |
| `tests/test_longT_2048_matched.py` | `1be89f186a19e1dd67f1f7c35a547e1e3bffcf909df47829a55c2263be75ac22` |

`mutate.py` 前后 sha256 证明这些文件**逐字节还原**（`mutation_report.json::restored_bitwise = true`）。

## 5. 可入库证据副本（`runs/` 被 `.gitignore` 忽略）

本目录下的 `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` / `<arm>-probe.jsonl`
（两个臂 × 3 个文件，由 `run.py --stage promote` 生成）是 `runs/<arm>/` 下同名文件的
**可入库副本**（与 2623/2626/2628 同约定）。权重 `model.pt` **不入库**，只留上面的 sha256。
