# 过程史 · 自 `axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/MODEL.md` 迁出

> 性质：日志文档。原文把活性块字段的拆分记成一次「字段名修正」；SSOT 化时改为陈述式。

- 原有：`§10 活性块的字段名修正后，从 checkpoint 等价重算活性块…`
- 原有（括号内更正说明）：`（修正内容：原来的 identity_init 字段对无专属参数的 parallel-dual-ln 臂
  给出 True，读起来像「该臂起步与基线逐位等价」——实际不是…现拆成两个可逐位核对的布尔量
  params_identical_to_sequential / start_bitwise_equals_sequential…）`
- 资产表原有：`rederive_liveness.py | §10 活性块字段名修正后的等价重算工具（幂等，<10 s）`
