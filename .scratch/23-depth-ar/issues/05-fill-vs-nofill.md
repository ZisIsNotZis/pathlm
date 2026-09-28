# 23e — Q4 fill 变体对决（12k matched 步）

- **Status:** done（fill 胜：nofill-12k 1.9438 vs fill-12k 1.8582，Δ=+0.0856 ≫
  ±0.03 平局带——「填充必须」成立，3.6k 反向信号不复现；可选 proj-fill 臂
  1.8356 全深度支配 fill 为族新最优；判读见 evidence/SUMMARY.md §12）
- **Related:** 23/23b/23c/23d（形态税链条）；23 的意外线索：3.6k 步时 no-fill
  （2.2496）优于 fill（2.4501）——填充是负分量 +0.200

## 背景

用户的"填充必须"假设（未来 token 深层要看早退位置）在 3.6k 实测中方向
相反：ragged no-fill 语义（我们的原引擎语义，早退位置在未跑的层中缺席）
反而更好。Q4 在 12k matched 步下重判。

## 协议（sanity-B 配置 d=256/L=8，各 12000 步）

| run | 填充策略 | 状态 |
|---|---|---|
| fill-12k | 退出深度表征填充所有深层（=W0-12k） | 已有：1.8582 |
| nofill-12k | ragged（早退位置在未跑层缺席，原引擎语义） | **本 ticket 跑** |
| （可选）proj-fill-12k | 填充 + 逐层线性 adapter（填充表征→该层空间的学得 remap） | 时间允许再跑 |

统一：随机退出深度采样、aux 等权、soft 提交训练（同 W0-12k 其余全部设置）。

## 测量与判据

- 主判据：no-fill-12k vs fill-12k 的 depth-8 bpc（±0.03 内算平局，bf16 噪声）；
- 副判据：depth 曲线、depth-4 出口 bpc、快道可用性（tf@4）；
- 若 no-fill 胜 → 填充分量（≈+0.2）可整体砍除，形态税降至 ≈+0.13，
  且部署语义回到原引擎（实现成本零）；
- 若 fill 胜 → "填充必须"成立，转 proj-fill 或二元路由 fork。

## 产出

证据续 `.scratch/23-depth-ar/evidence/`（标 23e）；SUMMARY 增判读；
findings 票据行更新；commit 不 push。GPU ≤30 min。
