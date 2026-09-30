# MODEL — 2614 内容 top-k 稀疏注意力 · 13M · depth-ar/dense（契约 J）

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、
哈希、体积与生成命令。权重参数总量三臂**逐位相同**（`params_all_equal`）。

## 1. 权重清单（12k 步 / seed 0）

| 臂 | 文件 | attn_sparse_k | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|---|
| full（基线，全注意力） | `runs/full-12k-s0/model.pt` | 0 | 6,962,688 | 27,882,155 | `a5cfad5363b675d0f873fc1075733048d589b1fb3bff7164f96d5ac59910e3f9` |
| topk32（内容 top-k，k=32） | `runs/topk32-12k-s0/model.pt` | 32 | 6,962,688 | 27,882,155 | `3d5a1630d44c2438d6a9994484f0065430aa2491097d0956d33225f787ffb5ab` |
| topk64（内容 top-k，k=64） | `runs/topk64-12k-s0/model.pt` | 64 | 6,962,688 | 27,882,155 | `961d08f201a92d4afca59d606edc0be44c2830ccd9e8664d7ef387b90b35dfef` |
| full 重跑（同 seed 第二次训练） | `runs/full-12k-s0-r2/model.pt` | 0 | 6,962,688 | 27,882,155 | `08dc6304900fc8485732ddf9bb18865bc4f0d8339aa47846c5dbbc70ff0d7a17` |

冒烟权重（20 步 × k∈{0,32,64}，非结果证据）：`smoke/smoke-k{0,32,64}/model.pt`（不记哈希）。

参数量账（`results.json::reconciliation.arithmetic`，逐位核对通过）：
`full` = `topk32` = `topk64` = **6,962,688** ⇒ 本干预**不新增参数**（`params_added = 0`），
因此不存在容量/重参数化混杂（SCHEMA §15 的等参数要求自动满足，见
`analysis.md` §3）。同 seed 下三臂初始化**逐位相同**（`smoke/smoke.json::
init_bitwise_identical_to_full`）。

## 2. 代码 revision（本实验绑定的实现版本）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `3d2aabe70922336a2b8fc4b779dfbd5f913d816dc29602cc5179acd39ebd8085` |
| `train_depth_ar.py` | `a867ad5e4699805f9bf461391114e936451754434b9591c9729b7cd5ea2d5d87` |
| `tests/test_depth_ar.py` | `3b872dea05491bbff5ef87b1f1001135d0def9fcc50a15df7869c47ea1497f9c` |

- `git rev-parse HEAD` = `f7bc8ad26a754bdadb0ff986dfd3a709360865d7`（本实验的代码改动在**工作树**中，未由执行者提交；
  上表 sha256 是判读数字所绑定的确切实现版本，复核者可直接逐字节核对）。
- 变异验证（`mutate.py`）记录：5 处故意破坏全部抓红，且每次还原后
  `pathlm/depth_ar.py` 的 sha256 与上表**逐字节一致**
  （`mutation_report.json::restored_byte_identical = true`）。

## 3. 生成命令

```bash
cd /home/z/vibe/pathlm
EXP=axes/A-mixing/A6-mixing-operator/experiments/2614-topk-sparse-attn-13m
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 $EXP/run.py --steps 12000 --seed 0 --rerun-arms full > $EXP/run_log.txt 2>&1 &
tail -f $EXP/run_log.txt          # ≤110s 轮询
```

`run.py` 对每个臂以子进程调用仓库入口（等价命令，逐臂一次）：

```bash
python3 train_depth_ar.py <arm>-12k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --dense \
    --attn-sparse-k <0|32|64> --steps 12000 --batch-size 32 --seed 0 \
    --exit-mode uniform --eval-batches 20 --out-root <exp>/runs
```

20 步冒烟（关→v1 逐位还原 / 三 k 参数逐位相等 / 掩码不变量 / CLI 接线 + 活性 /
步时显存；**无 20 步外推时间字段**）：

```bash
python3 $EXP/smoke.py
```

变异验证（5 处破坏必须抓红 + 逐字节还原，<3 min，纯 CPU；**不要与训练子进程并发**）：

```bash
python3 $EXP/mutate.py
```

配对评测（阈值附近的噪声量化，20 批配对差，<1 min）：

```bash
python3 $EXP/paired_eval.py
```

只重算电池、不训练（复用 `runs/` 下 checkpoint；需从仓库根目录运行）：

```bash
python3 $EXP/run.py --skip-train
```

数据集：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 末 10M eval，
词表由全文件统计 = 206），本地已有，无需下载。

## 4. 复现资产

| 文件 | 内容 |
|---|---|
| `results.json` | 三臂 + 重跑臂全部电池 + 逐字段对照 + 三道对账/收敛块 + 判读 + 稳定性块 + 成本口径 |
| `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` | 每臂训练日志（每 200 步，含 `attn_sparse` 活性量） |
| `runs/<arm>/stdout.log` | 每臂训练子进程原始输出 |
| `smoke/smoke.json` | 20 步冒烟：v1 还原 / 参数账 / 掩码不变量 / CLI 接线与活性 / 步时显存 |
| `mutation_report.json` | 5 处变异抓红记录 + 还原后 sha256 一致性 |
| `paired_eval.json` | 20 批配对差（k=32 / k=64 / 同配置重跑散度） |
| `analysis.md` / `stability.md` | 判读（含三道对账 + 成本交叉点）/ 稳定性专报 |

`run.py --skip-train` 用同一 CLI 可重算全部臂的电池与判读，因此本实验**自持**：
判读数字不依赖任何未落盘的中间态。
