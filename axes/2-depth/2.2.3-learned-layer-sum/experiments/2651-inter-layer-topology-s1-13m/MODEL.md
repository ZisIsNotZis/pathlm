# MODEL — 2651 层间拓扑 seed=1 复核 + 并行子层代价面 · 13M · depth-ar/dense

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录本实验权重的
清单、哈希、体积与生成命令。s0（seed=0）权重的主家在 `../2613-inter-layer-topology-13m/MODEL.md`
（此处只指针，不复制）。

## 1. 权重清单

### 1.1 s1（seed=1，本实验）

| 臂 | 文件 | 参数量 | 体积 (bytes) | sha256 | 状态 |
|---|---|---|---|---|---|
| `sequential`（基线） | `runs/sequential-12k-s1/model.pt` | 6,962,688 | 27,882,155 | `daeb1ec9ec68e35ff8d0fa2fe2ee9b37202fd6e87f743f5bd55c5bd65c8bb172` | 已训完（12k 步 / seed=1） |
| `parallel` | `runs/parallel-12k-s1/model.pt` | — | — | — | **未生成**（等 GPU 锁超时，未跑） |
| `parallel-dual-ln` | `runs/parallel-dual-ln-12k-s1/model.pt` | — | — | — | **未生成**（同上） |
| `dense` | `runs/dense-12k-s1/model.pt` | — | — | — | **未生成**（同上） |

> `as of 2026-10-14T22:32+08:00`：三条未生成臂的等效成本 = 每条 98.3M token（≈6–8 min 卡时）。
> 参数量期望值（`smoke/smoke.json::checks.params_delta_vs_sequential`，已实测）：
> `parallel` 6,958,592（−4096）；`parallel-dual-ln` 6,962,688（与基线逐位相同）；`dense` 6,962,724（+36）。

### 1.2 冒烟（20 步 ×3 拓扑，非结果证据）

| 文件 | 说明 |
|---|---|
| `smoke/smoke-{parallel,parallel-dual-ln,dense}/model.pt` | 20 步冒烟中间态，不入判据、不入清单哈希 |
| `smoke/smoke.json` | 参数量账 + 恒等起步 + α 活性（CPU 侧证据） |

### 1.3 s0（seed=0，对照）——只指针

s0 四条臂的权重主家在 `../2613-inter-layer-topology-13m/MODEL.md`（含 sha256 与体积）；
本实验的 `run.py::remeasure_s0` 用当前电池重测这四条 s0 checkpoint，逐位复现 2613 记录
（`run_log-attempt1-crash.txt:3` `[s0] remeasured all_bitwise_equal=True`）。

## 2. 生成命令

本实验全臂一条命令复现（契约要求）：`run.py` 按 2613 s0 配方逐臂调用仓库入口
`train_depth_ar.py`（唯一科学变量 = seed 0→1）。

```bash
cd /home/z/vibe/pathlm
EXP=axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 "$EXP/run.py" --arms sequential,parallel,parallel-dual-ln,dense \
  > "$EXP/run_stdout-full.txt" 2>&1 &
```

逐臂等价命令（`run.py::train_arm` 内部调用，`<topo>` ∈ {sequential, parallel, parallel-dual-ln, dense}）：

```bash
python3 train_depth_ar.py <topo>-12k-s1 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --dense \
    --topology <topo> \
    --steps 12000 --batch-size 32 --seed 1 --exit-mode uniform \
    --eval-batches 20 --out-root <exp>/runs
```

已训臂的实际命令（`run_log-attempt1-crash.txt` / `run_stdout-smoke.txt`）：
`sequential-12k-s1` 由 `run.py` 于 2026-10-14 以 `--steps 12000 --seed 1` 启动并训完
（`runs/sequential-12k-s1/train_log.jsonl` 到 step 11999）。

只用已有 checkpoint 重算电池（不训练、不取锁）：

```bash
cd /home/z/vibe/pathlm
python3 "$EXP/run.py" --skip-train --arms sequential
```
