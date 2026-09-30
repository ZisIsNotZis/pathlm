# 2613 · 权重清单（权重本身不入库）

> `.gitignore` 已覆盖 `*.pt`。本文件记录每个 checkpoint 的 sha256 / 体积 / 生成命令，
> 供复核者独立重算校验。训练与评测的完整入口 = `run.py`。

## 训练配置（两臂逐字段一致，唯一变量 = 读出几何）

```
python3 run.py --steps 12000 --arms baseline,linres --seed <seed> --caps-min 22
```
内部对每臂调用（等价命令）：

```
python3 train_depth_ar.py <run_name> \
  --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
  --steps 12000 --batch-size 32 --seed <seed> --exit-mode uniform \
  --dense [--readout-linear-resid] --eval-batches 20 \
  --out-root axes/W-width/W4-readout-geometry/experiments/2613-readout-reparam-dense-13m/runs
```

数据：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 末 10M eval，词表 206）。

## seed = 0（主口径）

| 臂 | 文件 | 体积 (bytes) | sha256 |
|---|---|---|---|
| baseline（tied 线性） | `runs/dense-base-12000-s0/model.pt` | 27,882,155 | `fe2123bf4e603d9d80bc267ed3ca0f6e9111ec55f14ec3a57204eb53dcc5d188` |
| linres（线性重参数化） | `runs/dense-linres-12000-s0/model.pt` | 28,407,011 | `6a7b2276bad32983bd693acfbf56227f32152d6b7f757bba6b7156173b3256b7` |

参数量：baseline 6,962,688；linres 7,093,760（+131,072 = 2·d²）。

## seed = 1（n=1 负结果复现）

| 臂 | 文件 | 体积 (bytes) | sha256 |
|---|---|---|---|
| baseline（tied 线性） | `runs/dense-base-12000-s1/model.pt` | 27,882,155 | `38722eebaeeabec49d2a1672c0c04ee3d61c1e2ac2a2d9c578a3f70642a9e5cf` |
| linres（线性重参数化） | `runs/dense-linres-12000-s1/model.pt` | 28,407,011 | `dd4db57450eb6aef003ef1d93a3e6c555fc852f09fc73b60d9a8603a63de8627` |

生成命令：`python3 run.py --steps 12000 --arms baseline,linres --seed 1 --results-file results-seed1.json`
（日志 `run_seed1.log`，原始结果 `results-seed1.json`）。

## 历史对照 checkpoint（引擎复现门，只读、不重训）

| 臂 | 文件 | 用途 |
|---|---|---|
| 23d-dense-B-12k | `.scratch/23-depth-ar/evidence/23d-dense-B-12k/model.pt` | 当前评测代码重测得 bpc 1.5287，逐位复现历史数字（`results.json::historical_reproduction`） |
