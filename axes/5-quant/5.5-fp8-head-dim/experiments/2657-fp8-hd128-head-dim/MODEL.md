# MODEL · 2657-fp8-hd128-head-dim

## 被测量模型

| 项 | 值 |
|---|---|
| 权重（seed 0 主腿，`results.json` 出的那一个） | `runs/hd128-s0/model.pt` |
| sha256（seed 0） | `a41d5eaac208ed5febf296818c0710bebf3f867b3ef4216d0e3a5bd20ef50ad2`（评测时现算，与训练驱动记录逐位一致） |
| 第二 seed 腿（读数 `hd128-s1/results.json`；n=2 只给方向一致性） | `runs/hd128-s1/model.pt` · sha256 `70ca528def94e2bec88c6e62b5779408ff95d4e10c9b0f959b5416af66bf70d0` |
| size | 54 193 675 B（两个 checkpoint 同） |
| params | 13 536 768 |
| 结构 | 12 层 × `d_model` 256 × **2 头** × `head_dim` **128** × `mlp_mult` 6，dense decoder-only（stock `Block`，MHA） |
| 训练 | 12k 步 × bs 32 × seq 256 = 98.3M token，`seed 0` / `1`，`lr 1e-3`；配方与 `2616` mha 臂逐字段一致，**唯一改动 = 注意力头数 8→2**（⇒ `head_dim` 32→128） |
| 复现命令 | `python3 train_depth_ar.py hd128-s0 --d 256 --layers 12 --heads 2 --mlp-mult 6 --dense --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --eval-batches 24 --out-root <本目录>/runs` |
| clean bpc（本 run，CPU） | 见 `results.json::points`（seed 0：`fp16` 1.481189 / `int8_block_g128` 1.481199 / `fp8_e4m3` 1.481548）与 `hd128-s1/results.json::points`（seed 1：1.486259 / 1.486309 / 1.486802） |

## 权重与数据是否入库

- **权重不入库**：`runs/**` 被 `.gitignore` 排除；本文件是它们的哈希清单（唯一入库的口径记录）。
- **数据不入库**：`data/enwik8_full.npz` 的评测尾段（`load_enwik8_full`）。
- **不新增参数**：KV 量化/标尺是推理期算子，`params_added = 0`；评测不改权重、不训练。
- 训练侧的机器可读状态在 `train_driver_state.json`（逐臂 sha256 / 墙钟 / GPU 锁记账），与上表逐位一致。
