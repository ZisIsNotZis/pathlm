# MODEL — 2611 等参数纯线性读出对照 · 13M · depth-ar

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、
哈希、体积与生成命令。

## 1. 权重清单

| 臂 | 文件 | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|
| **A 臂**（tied 线性，两开关均 False） | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt` | 6,962,688 | 27,882,155 | `2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6` |
| **gated 臂**（`readout_gated=True`，2610 主臂） | `../2610-gated-readout-13m-depthar/runs/gated-12k-s0/model.pt` | 7,093,760 | 28,407,011 | `26e976ec3fc8e16b0ae1dd1faace197bb0489f1d576f7193e533eda30bfceece` |
| **本对照臂**（`readout_linear_resid=True`，本次新训） | `runs/linear-12k-s0/model.pt` | 7,093,760 | 28,407,011 | `f8ae40869ac0439c0567d7e68cef7749886e655ea93aa150ebcd95ab3bc6073f` |
| 冒烟臂（20 步，仅用于接线/活性检查） | `smoke/smoke-linear/model.pt` | 7,093,760 | 28,407,011 | （20 步中间态，不作为结果证据） |

- 三臂参数差：gated 与线性对照均为 **+2·d² = +131,072**（相对 A 臂**逐位相等**）。
- A 臂 checkpoint 系**历史同配置对照**（ticket 23c，seed 0 / d256 / L8 / bs32 / 12k /
  fill），位于 `.scratch/` 证据目录；已过等值性门（`results.json::A_arm_equivalence`）
  并在当前评测代码下重测电池，逐位复现历史数字（`analysis.md` §1）。
- gated 臂 checkpoint 复用 2610（同配置、唯一变量 = 非线性读出），同样过等值性门
  （`results.json::gated_arm_equivalence`）并逐位复现 2610 数字。

## 2. 生成命令

三臂电池 + 本对照臂全新训练 12k（后台运行 + ≤120s 轮询 `run.log`）：

```bash
cd /home/z/vibe/pathlm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 axes/3-width/3.2.1-readout-reparam/experiments/2611-equalparams-linear-control-13m/run.py \
    --steps 12000 --seed 0 --eval-batches 20 --caps-min 20 > run_log.txt 2>&1 &
tail -f run_log.txt
```

`run.py` 内部对训练以子进程调用仓库入口（等价命令）：

```bash
python3 train_depth_ar.py linear-12k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
    --steps 12000 --batch-size 32 --seed 0 --exit-mode uniform \
    --readout-linear-resid --eval-batches 20 --out-root <exp>/runs
```

20 步冒烟（恒等起步 + 活性 + 参数量）：

```bash
python3 axes/3-width/3.2.1-readout-reparam/experiments/2611-equalparams-linear-control-13m/smoke.py
```

只重算三臂电池、不训练（核对既有结果）：

```bash
python3 axes/3-width/3.2.1-readout-reparam/experiments/2611-equalparams-linear-control-13m/run.py --eval-only
```

数据集：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 末 10M eval，词表由
全文件统计 = 206），本地已有，无需下载。

## 3. 复现资产

| 文件 | 内容 |
|---|---|
| `results.json` | 三臂全部电池 + 等值性门 + 对账块 + 判读 + 稳定性块 |
| `train_log.jsonl` / `gate_log.jsonl` | 本对照臂训练日志（每 200 步） |
| `runs/linear-12k-s0/stdout.log` | 本对照臂训练子进程原始输出 |
| `smoke/smoke-linear/smoke.json` | 20 步冒烟：恒等起步 / 活性 / 参数量三项检查 |
| `analysis.md` | 判读（含 gated vs 线性逐项对照、三道对账） |
| `stability.md` | 三臂 gnorm / NaN / loss 突跳 / 逐深度 ECE |

`run.py` 用同一 CLI 同时可重测 A 臂与 gated 臂（`--eval-only`），因此本实验**自持**：
判读数字不依赖任何未落盘的中间态。
