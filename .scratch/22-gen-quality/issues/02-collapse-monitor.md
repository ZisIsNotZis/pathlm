# 22b — 塌缩域监测信号（文本重复率）

- **Status:** done（判据成立：distinct-2 区分塌缩/健康窗 AUC 0.9815 ≥ 0.9，
  且与 js 解耦——塌缩窗 js 不升反降 0.035<0.046、告警方向 AUC 0.389、
  rank_corr 0.14；js 双向失灵：假阴 js=0.000 与假阳 js=0.373 并存。
  诚实记录：契约阈值标注规则退化为全塌缩（greedy 全局重复偏置），
  主标签改用契约允许的人工语义标注（摘录入档供审计）。监测栈定形：
  prob0+js+distinct-2 三信号正交互补）。
- **Related:** 22-gen-quality（路径分歧度 js 的盲区：吸引子塌缩窗 js→0 且
  agree=1.0，注入 15% 也不动——js 不能单独监测塌缩域）

## 假设（可证伪）

**文本重复率是塌缩域的监测信号**：吸引子塌缩的生成窗（特征：低 n-gram
多样性/高重复）即使 prob0 高、js≈0，重复率指标也应显著区别于健康窗。
与 js 互补后形成完整监测：js 管"语义分歧"，重复率管"塌缩"。

## 协议（纯推理，INT2_r3 权重，复用 ticket 22 探针基建）

1. **构造窗集合**：free-running 贪心生成 ≥24 窗（3 prompt 类型 × 8，
   400 token，含已知的吸引子塌缩窗——random prompt 下常见；人工标记：
   unique-token 比率或 distinct-2-gram 比率 < 阈值的窗为"塌缩"）。
2. **指标**：每窗计算 (a) distinct-2 比率、(b) 最高单 token 占比、
   (c) mean prob0、(d) js。判据：重复率指标在塌缩窗 vs 健康窗可分
   （AUC 或分离裕度），且与 js 正交（塌缩窗 js 不分、重复率分）。
3. **对照**：真实 enwik8 前缀窗的重复率基线（enwik8 本身有自然重复，
   需要报告自然水平）。

## 判据

重复率指标 AUC ≥ 0.9 区分塌缩/健康窗，且塌缩窗上 js 与重复率解耦
（js 低、重复率高）→ 成立；否则如实证伪（塌缩域监测仍开放）。

## 产出与约束

- 代码进 `pathlm/gen_quality.py` 或 probe_gen_quality.py 扩展 + 可失败单测；
- 证据 `.scratch/22-gen-quality/evidence/`（续用 ticket 22 目录，标 22b）；
- findings.md（≤200 行）与 WORKSPACE 补行；commit 中文；**不 push**；
- `python3 -m pytest tests/ -q`（88 绿基线）全绿；
- GPU 预算 ≤40 min。
