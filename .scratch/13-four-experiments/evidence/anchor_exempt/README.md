# 实验 2 — 锚区豁免损坏（`corrupt_spare_anchors`）

## 结论（一句话）

**判据不成立：锚区召回只从 X2C 的 1.29% 回到 3.38%，远低于 50%。**
把驱逐设计里“前 `anchors` 个位置是可靠长程通道”的逻辑原样扩展到损坏
（锚区免疫 stage-0 损坏），并不能恢复损坏训练摧毁的精确复制能力。干净 bpc
1.6841 对比 X2C 1.6797 劣化 +0.0044（大于同种子噪声底 0.0019，但落在
per-cell 种子散布 0.0001–0.0138 内）。

`in_window` 桶（非锚 needle）如预期仍然低（0.52%），说明“模型学会不信任
非锚内容”的预期成立——但锚区通道本身也没有学到复制，所以设计意图未实现。

## 动机（已测量）

隔离实验 X2C（`configs/X2.json` 路径块 + `corrupt_mask 0.075` +
`corrupt_wrong 0.075`，6000 步，seed 0）把 needle 召回从无损坏对照 X2v2 的
in-window 84.85% / anchor 76.84% 直接打到 0.47% / 1.29%。损坏训练（容忍任意
token 错误）与复制任务（精确抄锚点）目标冲突。

## 设计

- 新 `PathConfig` 旋钮 `corrupt_spare_anchors: bool = False`（默认 False =
  现状，逐位一致）。
- `_input_latents`：当旋钮开启且 `pc.anchors > 0` 时，在应用之前把
  `mask_pos`/`wrong_pos` 在 `position < anchors` 处清零，使返回的
  `corrupt_mask` 反映真实施加的损坏（修复指标保持诚实）。

## 训练对照（6000 步，seed 0，enwik8）

| run | corrupt | spare_anchors | clean bpc | needle in_window | needle anchor | needle beyond |
|---|---|---|---|---|---|---|
| X2v2 | 关 | — | 1.5652 | 0.8485 | 0.7684 | 0.0044 |
| X2C | 0.075+0.075 | 关 | 1.6797 | 0.0047 | 0.0129 | 0.0059 |
| **X2C-v2** | 0.075+0.075 | **开** | **1.6841** | **0.0052** | **0.0338** | 0.0067 |

- 判据 1（anchor recall ≥ 50%）：**不成立**（3.38%）。
- 判据 2（clean bpc 不劣于 1.6797 超种子方差）：+0.0044，略超同种子底
  0.0019，但在 per-cell 种子散布内，判为“基本持平/边缘劣化”。

## 解读与教训

- 锚区豁免只保护了 stage-0 的 **key/value** 原样，但 needle 查询块
  `[mask, x_i, y_i, ...]` 在尾部（非锚），仍受损坏；且 80% 的 needle 放在
  正文区，训练信号仍以“损坏=不可信”为主，模型整体抑制了复制行为。
- 结论：**“锚=免疫驱逐”不能直接延伸为“锚=免疫损坏”来救 needle**。损坏
  训练与复制能力的拮抗比单点豁免更深（怀疑是全局的“别抄”策略）。
- 不建议继续在同一问题上盲试；若要做，下一步应改为**按位置/按样本屏蔽
  损坏**（例如 needle batch 完全不做损坏），而不是只豁免锚区。

## 复现

```
python3 train_m1.py X2C-v2 --config .tmp/X2C-v2.json --seed 0 --out-root .tmp/p3
```

配置见 `.tmp/X2C-v2.json`（gitignored）。`X2C_ref/` 为对照 X2C 的
`results.json` + `train_log.jsonl`。
