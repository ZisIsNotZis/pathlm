# MODEL — 2510-gap-sweep-b2-c1m

权重本身不入库（`*.pt` 已 gitignore）。

| 臂 | checkpoint | sha256 | 状态 |
|---|---|---|---|
| B2（n_mtp=2） | `.tmp/p3/B2`（历史路径） | **不可得** | `.tmp/` 已清 ⇒ **无法重跑**，只引用原始 `results.json` |
| B0（对照，n_mtp=1） | `.scratch/04-m1-runs/evidence/B0/model.pt` | 见该目录 | 已训练，本目录不重训 |
| B2fresh（ticket 13 重训对照） | `.tmp/p3/B2fresh` | **不可得** | 只引用 `B2fresh.results.json` |

架构（B2，`results.json:config.model`）：d_model=256, n_layers=12, n_heads=8, seq_len=512, mlp_mult=6；
路径：`n_mtp=2, transport="none"`；训练：6000 步 × batch 32，seed 0。

生成命令（历史 ticket 10；本目录不重训）：

```
# 训练（已发生，权重为派生物）：
#   python3 train_m1.py <config>   → .tmp/p3/B2（已清）
# 接受率探针：
#   python3 probe_mtp.py .scratch/10-gap-sweep/evidence/B2
```

原始产物（入库，唯一可核）：`.scratch/10-gap-sweep/evidence/{B2,C1M,C2M}/results.json`、
`B2/mtp_accept.json`、`eval_ablations.json`。本目录 `results.json` 是从上述文件**逐字抽取**的记录。
