# 2628 · 权重清单（MODEL.md）

> 权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）。本文件记录哈希 / 体积 / 生成命令，供复核与复现。
> 契约：同目录 `README.md`；判读：`analysis.md`；对账：`reconcile_report.json`。

## 1. 生成的 checkpoint（两臂）

| 臂 | 参数 | 体积 (B) | sha256 | 生成命令 |
|---|---|---|---|---|
| `full-2048-clean` | 7,421,440 | 29,717,163 | `0af4e484a085d968e0da8aa86be634fca549d294bc910034f1d8d9606012907e` | `run.py --stage train --arm full-2048-clean` |
| `win1024-2048-clean` | 7,421,440 | 29,717,163 | `5f28875016afee66821515c711b71578f66f819751980c7918ea13d8c3d6ca59` | `run.py --stage train --arm win1024-2048-clean` |

- 每个臂的训练命令由 `run.py` 生成：`python3 clean_driver.py <arm> --seq-len 2048 --window {0,1024} --steps 3000 --batch-size 8 --lr 1e-3 --seed 0 --candidate qk-normfrom512`。
- 路径：`runs/<arm>/model.pt`（**不入库**）；resolved config 见 `config.json::arms[].fields`。
- 权重哈希与 `config.json::checkpoints[*].sha256` 一致（`sha256_verified: true`）。

## 2. 复现入口与参数

| 阶段 | 命令 |
|---|---|
| 能力面 | `python3 capability_check.py` |
| 冒烟 | `python3 run.py --stage preflight` |
| 训练（单臂/全部） | `python3 run.py --stage train --arm <arm>` / `--stage train-all` |
| 评测 | `python3 run.py --stage eval` |
| 汇总 | `python3 run.py --stage report` |
| 对账 | `python3 reconcile_numbers.py` |
| 变异 | `python3 mutate.py` |

固定参数：`d=256, L=8, H=8, mlp_mult=4, dense=True, topology=sequential, bs=8, lr=1e-3, seed=0,
exit_mode=uniform, token_budget=49,152,000/臂`（T=2048 → 3000 步）。
数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval / vocab 206）。

## 3. 共享文件指纹（本实验**未改**，见 README §9）

| 文件 | sha256（本实验全程） |
|---|---|
| `pathlm/depth_ar.py` | `45f946793835f2afadf3a10f727b02ee879682ef6024b23fa179d3351c28c585` |
| `pathlm/model.py` | `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f` |
| `train_depth_ar.py` | `2d1863bbc4403e026d12b597d98a1da2b85f3606110073f2117a7dd1db2a1f35` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` |
| `experiments/2625-longT-stability/stability_driver.py`（**只读引用**：QK-Norm hook） | `b7f49df7de6ea1d8aa9f2663c9c4d09272bae5dfb1389d5f211cc5c33acc3f20` |

`config.json::code_fingerprints` 记录同一组哈希；`mutate.py` 前后 sha256 证明三个本实验相关文件
逐字节还原（`mutation_report.json::restored_bitwise = true`）。

**口径说明（与 2626 的差异）**：2626 记录的 `pathlm/depth_ar.py` = `3a0db552…`、
`train_depth_ar.py` = `1035565281…`；本实验运行时的对应文件已由**其他 ticket**（第 44 批及其后）
更新为 `45f9467…` / `2d1863bb…`（`git diff HEAD -- pathlm/ train_depth_ar.py` 为空 ⇒ 本实验**未改**
这两个文件，只是仓库当前版本已前进）。该差异不影响本实验的判据（窗口掩码 `window_anchor_keep_mask`
与 qk-norm hook 的语义未变，单测 + 变异钉住）。

## 4. 可入库证据副本（`runs/` 被 `.gitignore` 忽略）

本目录下的 `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` / `<arm>-probe.jsonl`
（两个臂 × 3 个文件）是 `runs/<arm>/` 下同名文件的**可入库副本**（与 2623/2626 同约定）。
权重 `model.pt` **不入库**，只留上面的 sha256。
