# 23c — 形态税因果分解：{aux 梯度, 门瓶颈, 欠拟合} 三消融

- **Status:** done（欠拟合=税主因 +0.592 可由 12k 步收回（残税 +0.121）；aux 非无罪但其在门控臂穿 trunk 的梯度是净收益，stop-grad 排除；判读见 evidence/SUMMARY.md §10）
- **Related:** 23（形态税 +0.713 vs dense-B 1.7375）、23b（加权 sweep 反向：
  浅层监督是表示正则；税主嫌升级为 carry 门乘性瓶颈/欠拟合）

## 目标

把形态税 +0.713 分解到三个嫌疑组件，找出具名原因与配方修复：

- **H1（aux 梯度嫌疑）**：dense trunk（无门无出口）+ 浅层 aux 监督（梯度开启，
  穿 trunk）。对照 dense-B 1.7375。若 H1 ≈ 基线 → aux 无罪，门是嫌疑；
  若 H1 显著劣化 → 逐深度 aux 监督本身在此配置有问题。
- **H2（欠拟合嫌疑）**：W0 配置训练 12000 步（vs 3600）。若税随步数显著
  收缩 → 欠拟合，解法=更长训练；若不收缩 → 排除。
- **H3（门瓶颈嫌疑）**：carry 门 + 浅层 aux 头 **stop-grad**（纯读出，无
  梯度穿 trunk）。对照 W0（2.4501）。若 H3 ≈ W0 → 门架构本身（乘性瓶颈）
  是主因（配方 fork：改二元跳层路由）；若 H3 显著好于 W0 → aux 梯度经门
  的路径有问题（解法=浅头 stop-grad 化）。

三者正交分解：H1−基线 = aux 梯度贡献；W0−H3 = 门×aux 交互；H2 = 时间项。
全部复用 train_depth_ar.py 现有开关（shallow_weight 已支持；stop-grad 与
dense+aux 需 additive 小改 + 单测）。

## 协议

全部 sanity-B 配置（d=256/L=8）：
- H1：3600 步
- H2：12000 步（W0 配置）
- H3：3600 步
总 GPU ≤40 min。测量：全深度 bpc（vs dense-B 1.7375）、depth-4 出口 bpc、
门均值曲线。

## 产出

证据续 `.scratch/23-depth-ar/evidence/`（标 23c）；SUMMARY 增判读（因果
分解表）；findings 票据行更新（不回溯压缩他人内容）；commit 不 push。
若三嫌疑全排除 → 如实记录（税源未知，升级为开放问题，给下一步假设排序）。
