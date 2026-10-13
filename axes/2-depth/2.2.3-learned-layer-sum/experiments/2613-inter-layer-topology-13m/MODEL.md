# MODEL — 2613 层间拓扑 · 13M · depth-ar/dense

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、
哈希、体积与生成命令。

## 1. 权重清单（四臂，各 12k 步 / seed 0）

| 臂 | 文件 | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|
| `sequential`（基线） | `runs/sequential-12k-s0/model.pt` | 6,962,688 | 27,882,155 | `bbdc7c59d4469173e7b0620808543785013b95fd300f63f2e44be1d2c725ed26` |
| `parallel`（共享 pre-LN） | `runs/parallel-12k-s0/model.pt` | 6,958,592 | 27,861,613 | `0109572dd94e0aa04b6881dbe6107e2251addef0e84ef25b308d84e542d58b0a` |
| `parallel-dual-ln`（等参数并行） | `runs/parallel-dual-ln-12k-s0/model.pt` | 6,962,688 | 27,882,155 | `832940a97a0de70d746e114f2ae94656de48975df75b84e0b16b2652916875ce` |
| `dense`（层间学习加权和） | `runs/dense-12k-s0/model.pt` | 6,962,724 | 27,882,599 | `2e973fe5dcb4bf114841afc50ccca375f687705948a845068007b20ea0325185` |
| 冒烟（20 步 ×3 拓扑，非结果证据） | `smoke/smoke-{parallel,parallel-dual-ln,dense}/model.pt` | 同上 | — | （不记：中间态） |
| 历史交叉核对（非本次训练） | `.scratch/23-depth-ar/evidence/23d-dense-B-12k/model.pt` | 6,962,688 | 27,882,155 | （历史资产，只读引用） |

参数账（`results.json::reconciliation.arithmetic`，逐位核对通过）：

- `parallel-dual-ln` − `sequential` = **0**（等参数位点，§15）；
- `parallel` − `sequential` = **−4096** = −L·2d（少一个共享 pre-LN）；
- `dense` − `sequential` = **+36** = L(L+1)/2 = 5.17e-6 相对量。

## 2. 生成命令

```bash
cd /home/z/vibe/pathlm
EXP=axes/2-depth/2.2.3-learned-layer-sum/experiments/2613-inter-layer-topology-13m
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 $EXP/run.py --steps 12000 --seed 0 > $EXP/run_log.txt 2>&1 &
tail -f $EXP/run_log.txt          # ≤120s 轮询
```

`run.py` 对每臂以子进程调用仓库入口（等价命令，逐臂一次）：

```bash
python3 train_depth_ar.py <topo>-12k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --dense \
    --topology <sequential|parallel|parallel-dual-ln|dense> \
    --steps 12000 --batch-size 32 --seed 0 --exit-mode uniform \
    --eval-batches 20 --out-root <exp>/runs
```

20 步冒烟（参数量账 + 恒等起步 + 接线活性 + 步时/显存）：

```bash
python3 $EXP/smoke.py
```

变异验证（关键开关故意破坏必须抓红，<1 min）：

```bash
python3 $EXP/mutate.py
```

§10 活性块从 checkpoint **等价重算**（不重训；`topology_liveness`
只是 `cfg.topology` + checkpoint 里 `layer_mix` 的纯函数）：

```bash
python3 $EXP/rederive_liveness.py
```

（活性块用两个可逐位核对的布尔量 `params_identical_to_sequential` /
`start_bitwise_equals_sequential` 表达起步等价性：对**无专属参数**的 `parallel-dual-ln` 臂
前者为 True 而后者不是——「与基线参数逐位相同」不等于「起步逐位等价」，见
`smoke/smoke.json::identity_start_bitwise_vs_sequential`。单测
`test_topology_liveness_params_added_matches_param_delta` 钉住 `params_added` 与实测
参数量增量一致。重算只动这四个臂的活性字段，不动任何质量数字。）

只重算电池、不训练（复用 `runs/` 下 checkpoint；需从仓库根目录运行）：

```bash
python3 $EXP/run.py --skip-train
```

数据集：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 末 10M eval，词表由
全文件统计 = 206），本地已有，无需下载。

## 3. 复现资产

| 文件 | 内容 |
|---|---|
| `results.json` | 四臂全部电池 + 逐字段对照 + 对账/收敛块 + 判读 + 稳定性块 + 活性块 |
| `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` | 四臂训练日志（每 200 步） |
| `runs/<arm>/stdout.log` | 四臂训练子进程原始输出 |
| `smoke/smoke.json` | 20 步冒烟：参数量账 / 恒等起步 / 接线活性 / 步时 / 峰值显存 |
| `mutation_report.json` | 三处变异抓红记录 + 还原后 sha256 一致性 |
| `rederive_liveness.py` | §10 活性块的等价重算工具（幂等，<10 s） |
| `analysis.md` / `stability.md` | 判读（含三道对账）/ 稳定性专报 |

`run.py --skip-train` 用同一 CLI 可重算全部臂的电池与判读，因此本实验**自持**：
判读数字不依赖任何未落盘的中间态。
