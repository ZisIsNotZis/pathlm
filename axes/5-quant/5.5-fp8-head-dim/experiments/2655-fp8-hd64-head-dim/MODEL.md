# MODEL · 2655-fp8-hd64-head-dim

## 被测量模型

| 项 | 值 |
|---|---|
| 权重（出读数的那一个） | `runs/hd64-s0/model.pt` |
| sha256 | `ce40e246614f2f50471e6bc66e6468269af83bef3c9ab3e383f04831d433038f`（评测时现算，与训练驱动记录逐位一致） |
| 第二 seed 腿（读数 `hd64-s1/results.json`；n=2 只给方向一致性） | `runs/hd64-s1/model.pt` · sha256 `931df39bb961ee5e178c2eca977732ad962ac63154d630b70cbb60f92946ff5a` |
| size | 54 193 675 B（两个 checkpoint 同） |
| params | 13 536 768 |
| 结构 | 12 层 × `d_model` 256 × **4 头** × `head_dim` **64** × `mlp_mult` 6，dense decoder-only（stock `Block`） |
| 训练 | 12k 步 × bs 32 × seq 256 = 98.3M token，`seed 0`/`1`，`lr 1e-3`；配方与 `2616` mha 臂逐字段一致，**唯一改动 = kv 头数 8→4**（⇒ `head_dim` 32→64） |
| 复现命令 | `python3 train_depth_ar.py hd64-s0 --d 256 --layers 12 --heads 4 --mlp-mult 6 --dense --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --eval-batches 24 --out-root <本目录>/runs` |
| clean bpc（本 run，CPU） | 见 `results.json::points`（`fp16` 1.483031 / `int8_block_g64` 1.483140 / `fp8_e4m3` 1.484297） |

## 权重与数据是否入库

- **权重不入库**：`runs/**` 被 `.gitignore` 排除；本文件是它们的哈希清单（唯一入库的口径记录）。
- **数据不入库**：`data/enwik8_full.npz` 的评测尾段（`load_enwik8_full`）。
- **不新增参数**：KV 量化/标尺是推理期算子，`params_added = 0`；评测不改权重、不训练。
- 训练侧的机器可读状态在 `train_driver_state.json`（逐臂 sha256 / 墙钟 / 锁记账），与上表逐位一致。
