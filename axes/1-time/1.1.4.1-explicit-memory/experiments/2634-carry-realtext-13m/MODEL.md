# MODEL · 2634-carry-realtext-13m 权重清单

> 真实文本口径：{无通道, delta 写} × seed {0,1}，各 12k 步 / bs32 / T256 = 98.3M token，数据 = enwik8（真实文本）。

> 权重（`model.pt`）**不入库**（`.gitignore` 的 `*.pt`）；本文件记录 sha256 / 体积 / 训练墙钟与峰值显存，
> `run.py` 加载时**逐次断言** sha256（`results.json::checkpoints`）。

| 臂 | 参数量 | 体积 (B) | 训练墙钟 (min) | 训练峰值显存 (MB) | sha256 |
|---|---:|---:|---:|---:|---|
| `rt_none_s0` | 13536768 | 54193675 | 11.94 | 2015.1 | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` |
| `rt_delta_s0` | 13801064 | 55255619 | 16.93 | 2282.5 | `d92114822c09f2f7538ea6a59802eed21b1297409fe91f66c031dec9d6c305eb` |
| `rt_none_s1` | 13536768 | 54193675 | 8.48 | 2015.1 | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` |
| `rt_delta_s1` | 13801064 | 55255619 | 15.11 | 2282.5 | `b3d3cdbc57266109b6b461d18c2188c7d1712170f6760b35a48ff9b6170d6843` |

## 结构

- 主仪器：`realtext_probe.far_source_probe`（被毁前缀 [0,64)、被评行 [244,255)，`rf = L(w−1) = 180` 的多跳感受野必须整段落在未毁区）。
- 离散答案探针：`retention.phrase_needle_batch`（needle v3）+ 配对 `plant=False` 地板；逐针原始记录在 `needle_raw.json`。
- 评测期通道消融：`realtext_probe.channel_off`（置零 `gate_w` + 每层 `carry_scale`，退出即还原）。

## 复现路径

```bash
bash axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train_*.sh      # 训练（按臂释放 GPU 锁，臂间 ≥60s）
bash axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/eval.sh          # 评测（纯推理，整程 <10 min 持锁）
```

## 说明

- 主干与 2631/2633 同规格（13.54M）；通道臂 +264,296 参数（13.80M）。
- 训练与评测都用 `window_anchor_keep_mask(256, 16, 0)`。
- 无通道臂与通道臂的**主干逐位相同**（同 seed 重建断言，`results.json::selfchecks`）。
