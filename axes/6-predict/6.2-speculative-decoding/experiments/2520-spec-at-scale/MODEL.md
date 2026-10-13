# MODEL — 2520-spec-at-scale

权重本身不入库（`*.pt` 已 gitignore）。本实验只用**一个**已训练 checkpoint（纯推理，无训练）。

| 项 | 值 |
|---|---|
| checkpoint | `.scratch/20-spec-scale/evidence/INT3_100M_s0/model.pt` |
| sha256 | `29001bd0bb4c2d7fec49b6889f025d3dd1750a5377c192dd1ce2bfbbf4fd13f0` |
| 体积 | 425,687,955 bytes（≈406 MiB，fp32） |
| 参数量 | 106,410,245（`results.json:params`） |
| 架构 | d_model=720, n_layers=12, n_heads=12, seq_len=512, mlp_mult=6 |
| 路径配置 | n_mtp=3, chain_mode=latent, window=128, anchors=8, transport=soft |
| 训练 | 32000 步 × batch 6 × seq 512 = 98.3M token，seed 0（`results.json:train_tokens`） |

生成命令（历史 ticket 20；本目录不重训）：

```
# 训练（已发生，权重为派生物）
#   协议 INT_100M + n_mtp=3；产物落在 .scratch/20-spec-scale/evidence/INT3_100M_s0/
# 本实验的推理复现：
python3 axes/H-horizon/H1-outputs-per-pass/experiments/2520-spec-at-scale/run.py \
        --ckpt .scratch/20-spec-scale/evidence/INT3_100M_s0 --reps 5
```

13M 参照点（checkpoint 已不可得，只引用原始 `spec_speed.json`）：

| 项 | 值 |
|---|---|
| checkpoint | `.tmp/p3/B3`（**不可得**，`.tmp/` 已清） |
| 架构 | d_model=256, n_layers=12, n_heads=8, n_mtp=3（`B3.results.json`） |
| sha256 | 不可得 |
| 原始产物 | `.scratch/13-four-experiments/evidence/mtp3/spec_speed.json`（sha256 `8d310153…b710cf`） |
| 独立复现 | `.scratch/13-four-experiments/evidence/mtp3/spec_speed_independent_repro.json`（sha256 `d3024d23…ea4e8d`） |

共享文件指纹：本实验未改 `pathlm/**` 或 `tools/evalkit/**`；`probe_spec.py` / `probe_tts.py` 只读复用。
