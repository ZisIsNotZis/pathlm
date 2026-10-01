# MODEL · 2617 合成长程任务电池（无新权重）

本实验**不训练、不产出任何新权重**：它是纯推理的合成任务天平，只**复用**既有的
13M dense 主干作被测量模型。权重（`*.pt`）不入库（`.gitignore` 屏蔽）。

## 被测量主干（复用 2610，未改动）

| 项 | 值 |
|---|---|
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 词表 | 206（enwik8 全文件 205 个不同 byte + 1 个 mask 槽 205） |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0（enwik8） |
| 主干 clean bpc / t+1 acc | 1.4836 / 0.6979（eval 24×16 窗口，口径见 2610/MODEL.md） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 体积 | 54,193,675 B（54.2 MB） |
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |

`run.py::load_trunk` 在加载前**断言 sha256**（不匹配直接 `SystemExit`）。
锚点来源：`axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/MODEL.md`。

## 生成命令

```bash
python3 axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/run.py
```

被复用的主干由 2610 生成：

```bash
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验产出的其它落盘

- `results.json`：原始结果（含 `code` 块的 sha256：run.py / longrange_tasks.py / test / retention.py）。
- `run_log.jsonl`：逐任务/逐 K 的简短日志。
- `smoke/`：冒烟结果（不覆盖正式 results.json）。
