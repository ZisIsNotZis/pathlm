# MODEL · 2653-fp8-vs-int8-13m

## 被测量模型

| 项 | 值 |
|---|---|
| 权重 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| sha256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2`（运行时断言，`load_trunk` 校验） |
| size | 54,193,675 B |
| 结构 | 12 层 × d_model 256 × 8 头 × head_dim 32 × mlp_mult 6，dense decoder-only（stock `Block`） |
| 训练 | 12k 步；配方与 `2616` 同源，见 `2610-frozen-trunk-jumphead-13m/MODEL.md` |
| clean bpc（GPU 历史锚） | 1.483569（`window_seed=2`, 24×16, T=256, device=cuda；`2614`/`2610` 口径） |
| clean bpc（本 run, CPU） | 见 `results.json::points[0].bpc`（**同设备口径**，与上面的 GPU 锚并列但不当判据） |

## 权重与数据是否入库

- **权重不入库**：`trunk.pt` 只在 `2610` 实验目录里，本实验**只读**引用 + 运行时 sha256 断言。
- **数据不入库**：`data/enwik8_full.npz` 的评测尾段（`load_enwik8_full`）；合成召回任务只用词表大小。
- **不新增参数**：KV 量化/标尺是**推理期**算子，`params_added = 0`，不改权重、不训练。
- 本实验**没有**任何训练步骤 ⇒ 无优化器状态、无 checkpoint 产出。

## WP2 的 head_dim=64 模型（已交付）

- 该腿的模型与读数在 `experiments/2655-fp8-hd64-head-dim/`（`runs/hd64-s0/model.pt`，sha256
  `ce40e246614f2f50471e6bc66e6468269af83bef3c9ab3e383f04831d433038f`；第二 seed 的 checkpoint 与其读数同在 `2655-fp8-hd64-head-dim/`（`runs/hd64-s1/model.pt` + `hd64-s1/results.json`）。
- 那里的配方＝`--d 256 --layers 12 --heads 4 --mlp-mult 6 --dense`（⇒ `head_dim = 64`），除 kv 头数外与
  `2616` 的 mha 臂逐字段一致。权重、训练命令与派生口径只有那一个家：`2655-fp8-hd64-head-dim/MODEL.md`。
- 本目录（`2653`）仍只负责 `head_dim=32` 的 WP1 四臂；WP2 的 hd64 读数**不在本目录重复**。
