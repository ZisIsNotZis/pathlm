# MODEL · 2611 跳跃头对照臂修正（纯推理，复用 2610 权重）

本实验**不训练、不产出新权重**；只加载 2610 的冻结主干与两个跳跃头做推理。
权重（`*.pt`）不入库（`.gitignore` 屏蔽），本文件给出复用权重的哈希、体积、
来源与生成命令。`run.py` 启动时用这些 sha256 断言核对（对不上直接报错）。

## 复用权重（来源：`../2610-frozen-trunk-jumphead-13m/`）

| 文件 | 角色 | 体积 | SHA-256 |
|---|---|---|---|
| `trunk.pt` | 13.5M dense decoder-only 主干（冻结） | 54,193,675 B | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| `head_s0.pt` | 直接头 t+1 + 跳跃头 t+2（seed 0） | 2,106,893 B | `0a90537dc93d34aba6fd72f4d3cded6248d5ef6a3be5e1b01358d7682daf077f` |
| `head_s1.pt` | 直接头 t+1 + 跳跃头 t+2（seed 1） | 2,106,893 B | `aa07b532813c2c258e1ae4d5fefa763c13ab7832aab78de2a434aa5400d44174` |

主干：d256 × 12 层 × 8 头 × mlp6 × seq256，tied 读出、无 final LN；
训练 12000 步 × bs32 × seq256（seed 0）。头：`HeadBank`（t1/t2 各一个
`TransformHead` 形 d→d 变换 + tied 读出），各 262,912 参数；训练 2000 步
× bs32（seed 0/1），冻结主干。全部细节见 `../2610-frozen-trunk-jumphead-13m/MODEL.md`。

## 生成命令（本实验 = 纯推理）

```bash
# 冒烟（2 批 × bs4）
python3 axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/run.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m --smoke

# 全跑（24 批 × bs16，与 2610 评测口径一致）
python3 axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/run.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m
```

复用权重的生成命令（在 2610 目录，本实验不重跑）：

```bash
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```
