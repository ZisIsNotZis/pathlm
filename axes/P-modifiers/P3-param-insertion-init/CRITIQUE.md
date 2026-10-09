# CRITIQUE — P3-param-insertion-init（独立批判审计，fork=false）

## ① 是什么
新参数插在哪、初始化成什么——决定"恒等起步"能否成立（proj-fill / 负偏置门 / LoRA / prefix）。

## ② 对着什么 sota，我们偏离在哪
业界标准：LoRA 的 B=0（单边零）、ReZero、Fixup、ControlNet zero-conv（`记忆·未核`；`axes/P-modifiers/PRIOR_ART.md` 块③ 已列出并自判"通用部分应改 reference"）。我们：`catalog.yaml:1339–1347` 仍 `status: judged`、无 `prior_art_*` 字段。本地真实增量是"**全零初始化在乘性结构上是鞍点**"的可复现诊断。

## ③ 五类问题逐条判定
- **开倒车：有（轻度）。** 单边零/恒等起步是业界成熟技法，本格把它写成"本地结论"；账本未按分诊降级（同 P1 的制度问题）。
- **过程错误：有（效应落在噪声带内被当结论）。**
  - `catalog.yaml:1343`（`tax`）写"proj-fill **1.8356** < fill 1.8582 < nofill 1.9438（**族内最优**，全深度逐深度支配 fill）"。原始证据自己写着这 0.0226 的差**在平局带内**：`.scratch/23-depth-ar/evidence/23e-decomposition.json:150–151` `"projfill_minus_fill_depth8": -0.0226, "note": "平局带内正向；全深度曲线逐深度支配 fill…"`；`.scratch/23-depth-ar/evidence/SUMMARY.md:392` 同句。
  - 本仓方法学带是 **0.014**（`docs/methodology.md §2`），而该族自述的 bf16 平局带是 **±0.03**（`.scratch/23-depth-ar/evidence/SUMMARY.md:372`）。0.0226 落在两带之间 ⇒ **按任一现役口径都不能排序**，只能报"方向为正"。
  - n=1 seed（`23e` 三个臂都是 sanity-B / seed 0）。
- **结论错误：有（序关系越证据）。** 正确结论：**proj-fill ≈ fill（不可排序，方向为正）；两者都显著优于 nofill（+0.0856，>0.03 带）**。
- **无意义：否。** 它给出可复用的写法（单边零、恒等起步不消耗随机数 ⇒ 可严格对照）。
- **值得做：是（补做）。**

## ④ 建议动作：保留，补 seed / 更长训练点
- 把"族内最优"改为"平局带内、方向为正"；若要升级为序关系，补 ≥2 seed（或一个 2× 训练点，`SCHEMA.md §16`）。
- 保留"全零=鞍点"的诊断（这是真的本地增量，且有 `zero-init-saddle/` 目录证据）。
- 按分诊补 `prior_art_*` 字段。

## ⑤ 正确的验证方式
- **该涨**：proj-fill 的 bpc 应 ≤ fill（同 seed、同 token 预算）。
- **该跌**：nofill 应显著差于 fill（+0.0856 量级）。
- **必须不变**：恒等起步的**逐位还原**、以及"同 seed 下 proj-fill 与 fill 逐位一致（不消耗随机数）"这条对照属性。
