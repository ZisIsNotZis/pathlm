# MODEL — 2610 gated 残差读出 · 13M · depth-ar

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、
哈希、体积与生成命令。

## 1. 权重清单

| 臂 | 文件 | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|
| **A 臂**（tied 线性，readout_gated=False） | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt` | 6,962,688 | 27,882,155 | `2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6` |
| **主臂**（gated 残差读出，readout_gated=True） | `runs/gated-12k-s0/model.pt` | 7,093,760 | 28,407,011 | `26e976ec3fc8e16b0ae1dd1faace197bb0489f1d576f7193e533eda30bfceece` |
| 零初始鞍点对照（作废臂，readout_gated=True，W1=W2=0） | `runs/gated-12k-s0/model.pt`（已被主臂覆盖；哈希见 `zero-init-saddle/model.pt.sha256`） | 7,093,760 | 28,407,011 | `0f893d2a64f89fc2c70191afc9e2e5d2b21cb323e184cb7b3877ccca93807de1` |

A 臂 checkpoint 系**历史同配置对照**（ticket 23c，seed 0 / d256 / L8 / bs32 / 12k /
fill），位于只读 `archive/` 之外的 `.scratch/` 证据目录；已过「等值性门」（见
`results.json::A_arm_equivalence` 与 `analysis.md` §1），并在当前评测代码下重测电池，
逐位复现历史数字。

主臂与 A 臂的参数量差 = 2·d² = 2·256² = **131,072**（读出几何唯一新增参数）。

## 2. 生成命令

A 臂（不需要重新训练；从历史 checkpoint 载入并重测）：

```bash
# A 臂等值性门 + 当前代码重测电池（不训练）
python3 axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/run.py \
    --eval-a-only --eval-batches 20
```

主臂（全新训练 12k 步 + 全电池 + 判读，后台运行 + 轮询日志）：

```bash
cd /home/z/vibe/pathlm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/run.py \
    --steps 12000 --seed 0 --eval-batches 20 --caps-min 16 > run.log 2>&1 &
tail -f run.log
```

`run.py` 内部对主臂以子进程调用仓库入口（等价命令）：

```bash
python3 train_depth_ar.py gated-12k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
    --steps 12000 --batch-size 32 --seed 0 --exit-mode uniform \
    --readout-gated --eval-batches 20 --out-root <exp>/runs
```

数据集：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 后 10M eval，词表由
全文件统计 = 206），本地已有，无需下载。

## 3. 复现资产

- `train_log.jsonl` / `gate_log.jsonl`：主臂训练日志（每 200 步一落）。
- `trainer_results.json`：`train_depth_ar.py` 写出的主臂原始结果（子集于
  `results.json::main_arm`）。
- `readout_gate_diag.json`：主臂读出变换幅度诊断（W1/W2 范数 + 相对幅度）。
- `zero-init-saddle/`：作废臂（零初始鞍点）的 `results.json` / `train_log.jsonl` /
  `gate_log.jsonl` / `model.pt.sha256` / `intervention.json`，作为机制发现的证据保留。
