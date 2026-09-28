# 23d — 残税 +0.121 分解（dense-B-12k / H1-12k 对照）

- **Status:** done（dense-B-12k 1.5287 / H1-12k 1.6362：残税 +0.121 = {dense 欠拟合 −0.209，aux 残余 +0.108，门/填充固有 +0.222}，matched 真税 +0.33，「堆栈固有」强解读证伪；100M 4.4-epoch 外推 +0.21，过不了 <0.05 判据，详见 evidence/SUMMARY.md §11）
- **Related:** 23c（税分解：12k 步后残税 +0.121 归属开放）

## 协议

全部 sanity-B 配置（d=256/L=8），补两个 12k 对照：

| run | 堆栈 | 步 | 目的 |
|---|---|---|---|
| dense-B-12k | dense trunk，无 aux | 12000 | dense 自身欠拟合项：12k 的 dense bpc（对照 3.6k 的 1.7375） |
| H1-12k | dense + aux（梯度开） | 12000 | aux 分量在 12k 的贡献 |

判读：结合已有的 W0-12k（1.8582），把 +0.121 分解为 {dense 欠拟合、aux 残余、
门/填充残余}；判据：若 dense-B-12k ≈ 1.34-1.38 且 W0-12k 1.858 的差 ≈ 残税，
则残税主要是堆栈固有（门/填充），收敛预测按 ticket-19 渐近经验外推。

## 产出

证据续 `.scratch/23-depth-ar/evidence/`（标 23d）；SUMMARY 增补；findings
票据行更新；commit 不 push。GPU ≤40 min（两 run 各 ~10-13 min + 评测）。
