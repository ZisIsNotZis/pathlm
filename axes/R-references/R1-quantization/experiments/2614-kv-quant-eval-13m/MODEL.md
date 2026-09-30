# MODEL · 2614 KV 量化参照线（13M）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。
本实验**不训练、不改权重**——只复用契约 I 的同一 checkpoint。

## 被测量主干（复用，只读）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0 |
| clean bpc | **1.4836**（eval 24×16 窗口；本实验释放版复算 1.483569，差 3.1e-05） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 体积 | 54,193,675 B（54.2 MB） |

**本实验的强制项**：`run.py::load_trunk` 在加载前断言该 sha256，不符即 `SystemExit`。

## 生成命令（复用，不重新生成本实验的权重）

该主干由契约 I 之前的实验生成，生成命令见
`axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/MODEL.md`：

```bash
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验产出的权重

**无**。KV 量化是**评测期**量化：K/V 在注意力前做量化-反量化往返后即用即弃，
不落盘、不改权重、不新增参数（`results.json::intervention.params_added = 0`）。
