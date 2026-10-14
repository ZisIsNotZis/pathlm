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

## WP2 的 head_dim=64 模型（未交付）

- 若补齐该腿，需要另训 1 个 13M：`--d 256 --layers 12 --heads 4 --mlp-mult 6 --dense`（⇒ head_dim = 64）。
- 该 checkpoint **未产出**（GPU 锁被占，见 `wp2_hd64_status.json`）；命令与后续三臂已在 `config.json::wp2` 与状态文件里写死。
- **不得**把未跑的训练数字当成已测（契约 §4）。
