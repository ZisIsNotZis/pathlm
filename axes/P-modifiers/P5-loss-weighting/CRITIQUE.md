# CRITIQUE — P5-loss-weighting（独立批判审计，fork=false）

## ① 是什么
多目标损失如何加权与调度：等权 / 浅层降权 sweep / 退火 / 梯度裁剪。

## ② 对着什么 sota，我们偏离在哪
业界标准：deep supervision（GoogLeNet aux heads）、loss annealing、gradient clipping（Pascanu）、uncertainty weighting（Kendall）（`记忆·未核`）。偏离：本格把两批**不同口径**的实验并进同一个 `tax` 字段。

## ③ 五类问题逐条判定
- **开倒车：无**（分诊已列业界做法）。但同 P1/P3：`status: judged`、无 `prior_art_*` 字段，与 `axes/P-modifiers/PRIOR_ART.md` 的"通用部分改 reference"不一致。
- **过程错误：有（口径混用 + 证据来自 legacy 备份）。**
  - `catalog.yaml:1400`（`tax`）写"浅层降权全败且单调反向（w∈{1,0.1,0.02,退火}：depth-L **2.45→2.57→2.73**）→ 浅层全深度监督是表示正则"，`benefit` 又写"退火…使 C 配方成为族内最优（**1.8429**）"。两组数字来自**不同实验、不同口径**，被并列在同一字段。
  - 证据链：`axes/P-modifiers/P5-loss-weighting/experiments/2523b-shallow-weight-sweep/README.md` 末句自注"— 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成"，而 `archive/README.md` 明写"**所有早期公布的税是 1.1-epoch 上界**…不要拿旧税与新实验直接比较，除非口径相同"。本格却把 legacy 臂（`.scratch/23-depth-ar/evidence @b8c9f86`）与新引擎臂（`.scratch/25-tax-attribution/evidence @2e86924`）并列。
  - 单 seed（ticket 23b 的 sanity-B / seed 0）。
- **结论错误：有（口径越界）。** 正确写法：**"浅层降权在 3.6k 快照下单调反向（2.45→2.73）"**（必须标快照口径）；**"退火是 C 配方的组成"是旧口径 + 单 seed**，不能与 matched-step 数字同列。
- **无意义：否。** 这是真配方（退火进了 C 配方）。
- **值得做：是（按现代口径重跑）。**

## ④ 建议动作：保留，重做两臂带口径
- 在现引擎上重跑：① 浅层降权 sweep（w ∈ {1, 0.1, 0.02, 退火}，**同 step、同 token 预算、同 seed≥2**）；② 退火 vs 等权（同 step 的 2× 点，`SCHEMA.md §16` 收敛对账）。
- 每个数字标 matched-step/快照口径；删掉跨口径并列。

## ⑤ 正确的验证方式
- **该涨**：w 越低深度 L 的质量应越差（单调反向应保持）。
- **该跌**：无。
- **必须不变**：退火配方的底座质量（族内最优的结论必须与等权臂同口径）；活性量（各深度 aux 权重）每 200 步落盘。
