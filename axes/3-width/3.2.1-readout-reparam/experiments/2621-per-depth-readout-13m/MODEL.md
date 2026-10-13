# MODEL.md · 2621-per-depth-readout-13m（权重清单 / 复现哈希）

> 权重（`*.pt`）**不入库**（`.gitignore` 覆盖）。本文件记录：参与本实验的每个
> checkpoint 的路径、sha256、体积、生成命令与口径。哈希在 `results.json` 落盘
> 时的磁盘状态上计算。

## 1. 本实验新训的两臂

| 臂 | 路径 | sha256 | 体积 | 生成命令（cwd = 仓库根） |
|---|---|---|---|---|
| **PD**（逐深度读出） | `runs/perdepth/model.pt` | `d7f00441b38183d3795b1bb723051ba9545a4be70cd3ab73559c5c178a1158e3` | 32M | `python3 train_depth_ar.py perdepth --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --steps 12000 --batch-size 32 --seed 0 --exit-mode uniform --readout-perdepth --eval-batches 20 --out-root <HERE>/runs`（由 `run.py --stage perdepth` 驱动） |
| **EP**（等参数共享对照） | `runs/wide_shared/model.pt` | `121cd32c858f874d32d1db15a1952a305f099fbef0270ebce288b0d6a72d441a` | 32M | 同上，把 `--readout-perdepth` 换成 `--readout-wide-shared 2304`（由 `run.py --stage wide_shared` 驱动） |

两臂同 seed=0、同 token 预算（98,304,000）、同超参，唯一差别 = 读出几何。
**训练步 0 的 log loss 两臂相同（5.2764）**，与 A 臂亦相同 ⇒ 主干初始化逐位一致
（单因素）。

## 2. 复用臂（SCHEMA §9；本轮**不重训**，只按当前代码同口径重测）

| 臂 | 路径 | sha256 | 备注 |
|---|---|---|---|
| **A**（共享 tied 读出） | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt` | `2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6` | 历史 run（ticket 23c），同配置同 seed |
| **gated**（共享 2d² 非线性） | `axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/runs/gated-12k-s0/model.pt` | `26e976ec3fc8e16b0ae1dd1faace197bb0489f1d576f7193e533eda30bfceece` | 2610 主臂 |
| **linear**（共享 2d² 线性） | `axes/3-width/3.2.1-readout-reparam/experiments/2611-equalparams-linear-control-13m/runs/linear-12k-s0/model.pt` | `f8ae40869ac0439c0567d7e68cef7749886e655ea93aa150ebcd95ab3bc6073f` | 2611 等参数对照臂 |

复用三道门（§9）：① **实现层**——开关关闭逐位等值单测（`tests/test_depth_ar.py::
test_readout_perdepth_off_matches_v1_bitwise` 等）；② **配置层**——`arm_*.json::equivalence`
逐字段对照（`equivalence_ok=True`）；③ **口径**——全部臂用同一 `run.py::run_battery`
重测，clean bpc 与历史值逐位复现（A 1.8582 / gated 1.7770 / linear 1.7694）。

## 3. 源码指纹（本轮共享文件，§22 要求如实记录）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `16a27e96ac30a8958891d7fdc5c7de252c0becb949c9542897ee7faa416b4d48` |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` |
| `tests/test_depth_ar.py` | `ffeab7df937dfc67d43b8cb972a8ef29d68542da2253ae20d0042692175138d1` |

本轮共享文件**只含本线（W4）的改动**（本波契约声明：这三文件唯一写者是 W4 线）。
改动是**局部追加**：新增两个配置字段 + 两个读出分支 + `readout(depth=)` 形参 +
互斥门 + CLI 开关 + 每 200 步 `readout` 活性量 + 9 条单测；邻线标记
（`carry_state_dim` / `n_kv_heads` / `attn_sparse_k` / `topology` / `recon_window`）
改动前后均在（§22 回读断言见 `analysis.md` §7）。

## 4. 复现

```bash
# 冒烟 + 变异验证（CPU；--train 追加 20 步 GPU 冒烟）
python3 axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/smoke.py --train
# 分阶段（每阶段单独取 GPU 锁，§20.1）：arms → perdepth → wide_shared → assemble
python3 axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/run.py --stage arms
python3 axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/run.py --stage perdepth
python3 axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/run.py --stage wide_shared
python3 axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/run.py --stage assemble
python3 axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/reconcile_numbers.py
```

数据：char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / 词表 206，
本地已有，不入库）。
