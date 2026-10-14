# MODEL · 2674-mdiffusion-13m

本实验**只读复用** 2671-padfill-13m 的 harness `padfill.py`（不修改），在自己的 `mdiff.py` 里
新增**吸收态掩码扩散训练目标**与 **K 步置信度解掩码采样器**。模型结构逐字段同 2671（唯一结构变量
= 注意力可见性 `attn_mask`；本项用 `full`=双向）。

## 结构（同 2671，契约 §2）

| 项 | 值 |
|---|---|
| 形态 | pre-LN decoder-only char LM |
| d_model | 256 |
| n_layers | 12 |
| n_heads | 8 |
| mlp_mult | 6 |
| seq_len | 512 |
| 读出 | 共享 tied：`logits = LN_F(h) @ E^T` |
| 注意力 | SDPA；`attn_mask=full`（本项唯一结构变量） |
| 实测参数量 | 12,813,312 |

## 只读依赖 / sha256

- `padfill.py`（只读 import，来源 = `2671-padfill-13m`）：`67be5cc63536ffdfc87e35ab6276f5de844810efdf218cbb4b00fda85e7f2982`
- `mdiff.py`（本项库）：`0c1227d9458744ff85a8d0009df52fb43b9f183ba266e1d2d77a47e323281d94`
- `run.py`（本项驱动）：`cc38fcab565e5c583ddd24b8da075d35ba56425d6017640e123d8a8ccabcb3e0`

## AR 参照臂来源

- `2671-padfill-13m AR checkpoint 未落盘（无 ckpt）；只读复用其 results.json 读数`
- 策略：reuse 2671 读数

## 数据

`data/enwik8_full.npz`（仓库既有）：train 90M / val 10M 字节、`vocab_size=206`
（205 字节值 + 1 个 `[mask]` 槽，`[mask]` = 末槽 = id 205）。

权重不入库（`.gitignore` 覆盖 `*.pt`）：`mdiff` 臂由 `torch.manual_seed(seed)` + 固定数据/mask
生成器重建；同 seed 采样器重跑逐值相同（见 `analysis.md` L5）。
