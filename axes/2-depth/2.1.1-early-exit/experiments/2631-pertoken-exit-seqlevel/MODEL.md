# MODEL — 2631 逐 token 退出决策的序列级实测

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录本次用到的
checkpoint 清单、哈希与复现命令。

## 1 权重清单（本轮**不训练**，复用已入账的 depth-ar checkpoint）

| 用途 | 文件 | 参数量 | sha256 |
|---|---|---|---|
| 主 checkpoint（全部臂共用，`run.py` 启动时重算校验） | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt` | 6,962,688 | `2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6` |
| no-fill 诊断臂（同一 state_dict，只改 `DepthARConfig(fill_kv=False)`） | 同上（加载同一权重） | 6,962,688 | 同上 |

该 checkpoint 的登记出处：`axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/MODEL.md`；
本仓库内第一次使用它的实验是 `axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e`（同为
纯推理口径），本实验的启动校验与登记值逐位相同。

## 2 配置

`d_model=256 / n_layers=8 / n_heads=8 / seq_len=256 / mlp_mult=4`；
训练口径：12k 步、`exit_mode=uniform`（逐位置随机退出深度）、v1 配方（`aux_mode=renorm`、
`exit_aux_weight=1.0`）、`fill_kv=True`、`proj_fill=False`、`carry_gate=True`、`dense=False`、
seed 0、98.3M tokens（≈1 epoch 快照）。

> 关键含义：该 checkpoint **训练时见过**逐位置随机退出（并带填充语义）⇒ 它对「上下文里混有
> 不同退出深度的位置」是**被训练到鲁棒**的。本实验测到的「异构深度上下文代价很小」这一条，
> 只对「随机退出训练过的模型」成立（`analysis.md` §6 已列为边界）。

## 3 数据

`data/enwik8_full.npz`（byte 级 enwik8，前 90M train / 后 10M val，V=206）；
评测样本 = `pathlm.data.batch` + `torch.Generator().manual_seed(2)` × 20 批 × 16 行，`T=256`
⇒ W=320 窗口、81,600 个预测位置。**与历史 `depth_curve` 同一个样本约定**（`depth_curve` 用 `manual_seed(2)`；`tf_acceptance` 用
`manual_seed(3)` ⇒ 那条只是**量级参考**，不用于对齐）。本实验的 `prescribed-k` 臂逐深度复现历史
`depth_curve[k]`（Δbpc = 7e-06…4.6e-05，见 `analysis.md` §3）。

## 4 复现命令

```bash
cd /home/z/vibe/pathlm
# 承载体单测（CPU，秒级）
python3 -m pytest tests/test_axis2_exit_decision.py -q
# 主实验（CPU 纯推理；不取 GPU 锁）
python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py \
        --out axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel
```
