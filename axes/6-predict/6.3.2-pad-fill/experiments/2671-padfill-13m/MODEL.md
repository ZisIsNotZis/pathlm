# MODEL · 2671-padfill-13m

本实验自带 harness（`padfill.py`），因为共享引擎 `pathlm/model.py` 的注意力是**硬编码因果**的，
而主判需要 `full`（双向）可见性的填空臂（不得改 `pathlm/**`）。AR 参照臂用**同一 harness 的因果**
配置，保证唯一结构变量只有注意力可见性（+ 损失位置集）。

## 结构（契约 §2 写死）

| 项 | 值 |
|---|---|
| 形态 | pre-LN decoder-only char LM |
| d_model | 256 |
| n_layers | 12 |
| n_heads | 8（head_dim = 32） |
| mlp_mult | 6 |
| seq_len | 512 |
| 读出 | 共享 tied：`logits = LN_F(h) @ E^T` |
| 注意力 | `F.scaled_dot_product_attention`；`attn_mask ∈ {causal, full}` 是唯一结构变量 |
| 实测参数量 | 12,813,312（两可见性逐位相同） |

## 数据

`data/enwik8_full.npz`（仓库既有，`pathlm/data.py::load_enwik8_full`）：train 90M / val 10M 字节、
`vocab_size = 206`（205 字节值 + 1 个 `[mask]` 槽，`[mask]` = 末槽 = id 205）。

## 五臂（契约 §2）与本次读数

| 臂 | 注意力 | r | 种子 | ar_at_M bpc | fill_M bpc | Δbpc |
|---|---|---|---|---|---|---|
| `fill_bidir_r10` | full | 0.10 | 0 | 2.22093 | 3.86745 | +1.64348 |
| `fill_bidir_r10` | full | 0.10 | 1 | 2.39539 | 5.04788 | +2.64708 |
| `fill_bidir_r25` | full | 0.25 | 0 | 2.20235 | 3.43149 | +1.22497 |
| `fill_bidir_r25` | full | 0.25 | 1 | 2.37170 | 3.29578 | +0.92031 |
| `fill_bidir_r50` | full | 0.50 | 0 | 2.16433 | 4.20803 | +2.04465 |
| `fill_bidir_r50` | full | 0.50 | 1 | 2.34445 | 5.04468 | +2.70235 |
| `fill_causal_r25` | causal | 0.25 | 0 | 2.20235 | 3.51846 | +1.31107 |
| `fill_causal_r25` | causal | 0.25 | 1 | 2.37170 | 3.53802 | +1.16128 |
| `ar` | causal | — | 0 | — | — | —（参照 ar_full=2.18422） |
| `ar` | causal | — | 1 | — | — | —（参照 ar_full=2.35653） |

权重不入库（`.gitignore` 覆盖 `*.pt`）：模型由 `torch.manual_seed(seed)` + 固定数据/mask 生成器
重建（`padfill.train_arm`），确定性见 `analysis.md` 的 L5。

## 复现命令

```bash
cd /home/z/vibe/pathlm/axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m
python3 padfill.py --params
python3 -m pytest test_padfill.py test_run_s2.py -q
python3 run.py --stage all --out . --seeds 0 1 --steps 1500
python3 mutate.py            # 变异抓红 + 逐字节还原
```
