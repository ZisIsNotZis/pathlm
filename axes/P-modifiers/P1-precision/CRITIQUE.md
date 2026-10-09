# CRITIQUE — P1-precision（独立批判审计，fork=false）

## ① 是什么
哪些部分必须高精度：主体 bf16 autocast、读出/损失头 fp32。落点是一句稳定性规则。

## ② 对着什么 sota，我们偏离在哪
业界标准：混合精度训练（fp32 master weights + loss scaling，Micikevicius 2017，`记忆·未核`）；奖励/对数类损失单独走 fp32 + 梯度裁剪是社区惯例（`axes/P-modifiers/PRIOR_ART.md` 块① 自己列了这三条）。**偏离**：本仓把这条通用工程纪律当成本地研究结论（`status: judged`）。

## ③ 五类问题逐条判定
- **开倒车：有（轻度）。** 把业界 2017 年以来的标准写法当"本项目定律"（`docs/playbook.md` 定律 5 的出处）。`axes/P-modifiers/PRIOR_ART.md` 的判定自己写了"通用部分应改「业界结论」口径"，但 `catalog.yaml:1308–1316` 的 P1 **没有任何 `prior_art_*` 字段**、`status` 仍是 `judged` ⇒ 分诊与账本不一致（P 轴 6 格全部如此）。
- **过程错误：无。** 证据真实存在且可核：`.scratch/09-m3-interactions/evidence/SUMMARY.md:21–24`"v1 diverged at step 4600 (bf16 + negative-JS), v2 silent-NaN after step 2200, v3 stable with the grad-norm guard … reward terms need float32 + gradient sanity gates under bf16 autocast"——两次发散是实观测。
- **结论错误：无**，但结论是通用常识，不是本项目的机制发现。
- **无意义：部分。** 它不改变任何"路径值不值"的判断；作为 **modifier note** 有防复发的价值（尤其"可失败地抓"这一点），作为**研究格**没有价值。
- **值得做：否。**

## ④ 建议动作：保留，但降级为 note /「业界结论」口径
- 按 P 轴 PRIOR_ART 的分诊把通用部分改「业界结论」口径，本地只保留"两次发散 + pre-step grad-norm gate 的可复现修法"；补 `prior_art_*` 字段以消除"分诊写了、账本没写"的不一致。
- 不做新实验。

## ⑤ 若重做（不建议）
判据 = 发散步数与 gnorm 轨迹；必须不变 = 非 reward 项的行为、同 seed 复现。
