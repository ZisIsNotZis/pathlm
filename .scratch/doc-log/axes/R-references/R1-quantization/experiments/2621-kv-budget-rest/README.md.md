# axes/R-references/R1-quantization/experiments/2621-kv-budget-rest/README.md · 过程史（自事实文档迁出）

> 性质：日志文档。此处是自事实文档 `axes/R-references/R1-quantization/experiments/2621-kv-budget-rest/README.md` 迁出的**过程叙述**（日期锚定说明 / 更正 / ERRATA /
> 变更叠加 / 已废台账指代）。事实以该事实文档的当前版本为准；本文件只作可回溯记录。

## 「部署口径更正」+ 日期

- 原文：「**部署口径更正（2026-10-02，`2631-kv-budget-causal`）**：严格因果/流式量化必需逐值 fp16 标尺…」
- 现状：改为「**部署口径（见 `2631-kv-budget-causal`）**：…」，去掉「更正」与日期，结论不变。

## 复核 X4 P1 口径声明注

- 原文（§10 末）：「> 主 Agent 补注（复核 X4 P1）：等字节 ≠ 等形状，跨形状排名须带该声明
  （见 analysis.md 末尾补写）。」
- 现状：事实文档改为当前口径陈述「**等字节 ≠ 等形状**：跨形状排名须带该声明（详见 `analysis.md`
  末尾的口径说明）」；过程（复核 X4 P1、末尾补写）迁出。
