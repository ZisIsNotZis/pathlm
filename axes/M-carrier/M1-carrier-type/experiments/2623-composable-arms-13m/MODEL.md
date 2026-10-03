# 2623 · 权重清单与代码指纹

> 权重（`*.pt`）**不入库**（`.gitignore: *.pt`）；本文件记录体积/哈希与生成命令，供复核逐次核。

## 1. 四臂权重（每臂自己训练，6k 步合成 assoc-mqar）

| 臂 | 体积 (B) | sha256 |
|---|---|---|
| `stock` | 54193675 | `72ed2438e99c41ed5c139ee31b7cd9cda13ce6a440524418293c3a9162f2d54d` |
| `gqa` | 49456651 | `411f962f0b0acc8f40b7d8e5927a787600141f442610f61d43dfa8d189303e2a` |
| `gqa_carry` | 50518595 | `5ec1e0c5df6ffe0c371996bbe0c5d02e6eb7bd57385355321c666359c8e26c1c` |
| `gqa_carry_window` | 50518595 | `48c4e39ec058b8c509de88744e32f4169f993eb4f2680e6b4080b1b410f9c8a6` |

生成命令（`train_arms.sh`，四臂逐字段同配方，只有特性叠加不同）：
```bash
python3 train_depth_ar.py stock              --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --steps 6000 --batch-size 32 --seed 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14
python3 train_depth_ar.py gqa                ... --n-kv-heads 2
python3 train_depth_ar.py gqa_carry          ... --n-kv-heads 2 --carry-write delta
python3 train_depth_ar.py gqa_carry_window   ... --n-kv-heads 2 --carry-write delta --train-window 16 --train-anchors 0
```

## 2. 代码指纹（本线产物）

| 文件 | sha256 |
|---|---|
| `axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/run.py` | `39e44accf1e51d24d9b4213e8e7e2e0f5c5954b46d0ffa66803fbc1b0693812e` |
| `axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/arms.py` | `13973d98316326979c6e2be7e21fafcc7a95e98661589a12c7f7d96be7a0be05` |
| `axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/train_arms.sh` | `882212cbd0ab1cd2c42925e47f4800c7617ecd4bd72a2beffb07dd8e3fe83002` |
| `axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/reconcile_numbers.py` | `fd61239e16a05914332bf11618919596630ac5b2bea711527c50b9b969fbcb89` |
| `axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/mutate.py` | `68e0a6f8ef615fa8a140d0911b72da52a69e043f1472390d6338a37db5d8b06e` |
| `axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/config.json` | `e6e40280711bc71f5fbbf1a639b2cad311252bed28a48fa3865bd0fe964155d1` |
| `tests/test_composable_arms_13m.py` | `1022e31e6fbc21e9e06a989915fcb233d04c1ae3cdf39013e9eba200d4eb5bee` |
## 3. 共享文件（本线**未改**，只记指纹以划清口径重叠）

> SCHEMA §22.6：把共享文件的 sha256 记进 MODEL.md，并在报告里写明口径重叠。

| 文件 | sha256 | 本线是否改动 |
|---|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` | **否**（本线不写引擎/训练入口；只组合 ENG1 已解开的开关） |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` | **否**（本线不写引擎/训练入口；只组合 ENG1 已解开的开关） |

> 口径重叠核对：评测落盘时（2026-10-03 18:32）`git status` 显示 `pathlm/depth_ar.py` 与
> `train_depth_ar.py` **没有未提交改动**（本波其它 GPU 线 Y2/Y3 只新增自己的实验目录与
> `tests/`，不改引擎/训练入口）⇒ 上表指纹即提交态基线；本线**未触碰**共享引擎文件。

## 4. 生成与校验命令

```bash
bash axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/train_arms.sh   # 训练四臂（每臂 6.8–13.6 min，按臂取锁）
python3 axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/run.py     # 评测（纯推理 0.74 min）
python3 axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/reconcile_numbers.py
python3 axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m/mutate.py
python3 -m pytest tests/test_composable_arms_13m.py -q
```
