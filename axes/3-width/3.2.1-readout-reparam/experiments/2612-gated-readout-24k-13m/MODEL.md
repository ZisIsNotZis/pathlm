# MODEL — 2612 gated 读出 24k 收敛复测 · 13M · depth-ar

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、
哈希、体积与生成命令。

## 1. 权重清单

| 臂 | 文件 | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|
| **A 臂**（tied 线性读出，readout_gated=False） | `runs/A-24k-s0/model.pt` | 6,962,688 | 27,882,155 | `863f6fdef43a5cece3bf9611c7df340f5d4cb71087d17cd419640ad5516f535a` |
| **gated 臂**（非线性读出状态变换，readout_gated=True） | `runs/gated-24k-s0/model.pt` | 7,093,760 | 28,407,011 | `87314bec0a57a5b9f05442c90a4213f28c22e2d96ccbfc6ea32a8a116f4bcdd0` |

两臂参数量差 = 2·d² = 2·256² = **131,072**（读出几何唯一新增参数）。
两臂均为**全新训练 24000 步**（非 2610 的 12k checkpoint 续训）。

## 2. 生成命令

两臂训练 + 全电池 + 判读（后台运行 + ≤120s 轮询日志）：

```bash
cd /home/z/vibe/pathlm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 axes/3-width/3.2.1-readout-reparam/experiments/2612-gated-readout-24k-13m/run.py \
    --steps 24000 --seed 0 --eval-batches 20 --caps-min 50 \
    > axes/3-width/3.2.1-readout-reparam/experiments/2612-gated-readout-24k-13m/run.log 2>&1 &
# 轮询：tail -n 40 <exp>/run.log
```

`run.py` 内部对两臂各以子进程调用仓库入口（等价命令，唯一差别 = `--readout-gated`）：

```bash
# A 臂（tied 线性）
python3 train_depth_ar.py A-24k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
    --steps 24000 --batch-size 32 --seed 0 --exit-mode uniform \
    --eval-batches 20 --out-root <exp>/runs

# gated 臂（非线性读出状态变换）
python3 train_depth_ar.py gated-24k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
    --steps 24000 --batch-size 32 --seed 0 --exit-mode uniform \
    --readout-gated --eval-batches 20 --out-root <exp>/runs
```

冒烟（20 步 × 两臂 + CPU 恒等检查）：

```bash
python3 axes/3-width/3.2.1-readout-reparam/experiments/2612-gated-readout-24k-13m/smoke.py
```

数据集：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 后 10M eval，词表由
全文件统计 = 206），本地已有，无需下载。

## 3. 复现资产（入 git）

- `results.json`：两臂电池 + 干预活性 + 逐字段对照 + 收敛块 + 对账 + 判读。
- `train_log_{A,gated}-24k-s0.jsonl` / `gate_log_{A,gated}-24k-s0.jsonl`：两臂训练日志
  （每 200 步一落）；`stdout_{A,gated}-24k-s0.txt`：训练进程原始 stdout。
- `run_log.txt`：`run.py` 总日志（含两臂训练墙钟、峰值显存、判读）。
- `smoke/smoke.json`：20 步冒烟结果（恒等起步逐位、参数量 +2·d²、开关接线、活性）。
- `analysis.md`：多维指标表 + 对账 + 收敛证据 + 失败决策树。
- `config.json`：resolved config（含 seed）。

权重位于 `.gitignore` 覆盖的 `runs/` 下，需重跑上面的 `run.py` 生成。
