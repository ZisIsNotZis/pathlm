# MODEL · 2610 冻结主干 + 跳跃头（13M）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。
主干 `trunk.pt` 与头 `head_s*.pt` 只用于本实验：主干训练后**冻结**，头只是
冻结主干表征上的两个 d→d 变换 + tied 读出。

## 主干（13.5M dense decoder-only，冻结）

| 项 | 值 |
|---|---|
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0 |
| 主干 clean bpc / t+1 acc | **1.4836** / **0.6979**（eval 24×16 窗口） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 体积 | 54,193,675 B（54.2 MB） |
| 文件 | `trunk.pt`（`capacity-h4x2/trunk.pt` 为同一哈希的复用副本） |

生成命令（复用现有 dense 臂，不另起架构）：

```bash
python3 train_jumphead.py \
  --config axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m
```

## 跳跃头 / 直接头（主臂，hidden=2d × 1 block = `TransformHead` 形）

两条头共用一个 `HeadBank`（t1 = 直接头 t+1，t2 = 跳跃头 t+2），
读出复用冻结主干表（`logits = transform(h) @ Eᵀ`）。

| 臂 | seed | 训练 | 文件 | 体积 | SHA-256 |
|---|---|---|---|---|---|
| 直接头 t+1 + 跳跃头 t+2 | 0 | 2000 步 × bs32，lr 1e-3，冻结主干 | `head_s0.pt` | 2,106,893 B | `0a90537dc93d34aba6fd72f4d3cded6248d5ef6a3be5e1b01358d7682daf077f` |
| 直接头 t+1 + 跳跃头 t+2 | 1 | 2000 步 × bs32，lr 1e-3，冻结主干 | `head_s1.pt` | 2,106,893 B | `aa07b532813c2c258e1ae4d5fefa763c13ab7832aab78de2a434aa5400d44174` |

## 容量消融头（宽头，hidden=4d × 2 block，同一冻结主干）

`capacity-h4x2/` 目录，复用上面的 `trunk.pt`（`--trunk-ckpt`），只训头 3000 步。

| 臂 | seed | 文件 | 体积 | SHA-256 |
|---|---|---|---|---|
| 宽直接头 + 宽跳跃头 | 0 | `capacity-h4x2/head_s0.pt` | 8,415,283 B | `1ea4f6de938559b0b898c56869c278a48a1d1a21d7e154cf78cc2999d98e7a32` |
| 宽直接头 + 宽跳跃头 | 1 | `capacity-h4x2/head_s1.pt` | 8,415,283 B | `456e6643c5a368129963cad94786e339d56f972e4d4082c3c617b520c879c5f4` |

生成命令：

```bash
python3 train_jumphead.py \
  --config axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/capacity-h4x2 \
  --trunk-ckpt axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt \
  --trunk-steps 0 --head-hidden-mult 4 --head-blocks 2 --head-steps 3000
```

## 联合臂（对照 3，未单独存盘）

主命令加 `--joint-steps 1500`：从已训主干出发、放开参数联合微调 1500 步。
该臂改动了主干（bpc 1.48 → 1.85），结果只作记录（见 `analysis.md` §2），
不另存权重哈希（其意义受主干损坏限制）。
