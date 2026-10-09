# MODEL — 2619 · 写入通道的难度—容量曲线（权重清单）

> 权重（`*.pt`）**不入库**（`.gitignore` 覆盖）。本文件是哈希与生成命令的唯一来源。
> 评测入口 `run.py` 加载权重前**逐次断言 sha256**（不符即 `SystemExit`）。

## 1. 四臂

| 臂 | `carry_state_dim` | 记忆槽头数 | 状态浮点/样本 | 参数量 | sha256 | 体积 |
|---|---|---|---|---|---|---|
| `D16` | 16 | 16 | 4,096 | 13,803,216 | `b8799d315a808a5a0df052ef5f0b79fe666af8cfd6211bb405ef507f89e20a39` | 55,263,811 B |
| `D32` | 32 | 8 | 8,192 | 13,801,064 | `1181a9988aa587dae97b58151a868679f79f149d5f36b30e7bd913980e6f5d52` | 55,255,619 B |
| `D64` | 64 | 4 | 16,384 | 13,799,988 | `b792d7e2912e24acecbc6c486f6d92a79f0ef246a7507191869422d15c83d6a0` | 55,251,523 B |
| `D128` | 128 | 2 | 32,768 | 13,799,450 | `b29d1fc71bf9a1641cd32988544fff395d28435d1d17c8b96804e8428c772715` | 55,249,475 B |

（sha256 由 `fill_config_hashes.py` 回填 `config.json`；`run.py` 加载时逐臂断言。）
四臂训练墙钟合计 **59.55 min**（见 `results.json::cost`）；峰值净显存 2.24–2.38 GB。

## 2. 生成命令（逐字，四臂只差 `--carry-state-dim`）

```bash
python3 train_depth_ar.py <arm> \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 10000 --batch-size 32 --seed 0 --lr 0.001 \
  --train-window 16 --train-anchors 0 \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 \
  --carry-write delta --carry-state-dim {16,32,64,128} --out-root train
```

（实跑由 `train_all.sh` 驱动，持 `/tmp/pathlm_gpu.lock`。）

## 3. 评测

```bash
python3 run.py          # 全电池（本目录；持 GPU 锁）
python3 run.py --smoke  # 冒烟
```

## 4. 共享文件指纹（SCHEMA §22.6，如实记录重叠归属）

本轮 `pathlm/depth_ar.py` 与 `train_depth_ar.py` 是**共享文件**：U1（本实验）追加
`carry_state_dim` 接线，U2 线（F4 重建损失）追加 `recon_window` 接线。两者在同一工作树里
**共存**（U1 的改动已被编排方提交，U2 的改动当时仍未提交）。本实验在写这份报告时
记录的工作树 sha256：

```
pathlm/depth_ar.py   248a92754154d65e0857ca5ceccda91ef63ffd9b90b3cfde2199da29b6b59135
train_depth_ar.py    8c333e5405baf23812cd8d389e69063a3b0129fa17993db7fe1ddb50e71828a0
tests/test_depth_ar.py 433d4ef4dc48de4c40d70a068a1c916004cd35fc38aad11be377af7fd22bdead
```

（运行期真实指纹另见 `results.json::code`；上表是训练启动时刻的快照，两者若不同以
`results.json::code` 为准并说明差异原因。）

## 5. 权重来源与口径

- 训练数据**全是合成**（`tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`），
  **不含 enwik8**、不含任何外部语料 ⇒ 无数据授权/隐私问题。
- 权重只用于本目录的评测（`run.py`），不进入任何发布产物。
