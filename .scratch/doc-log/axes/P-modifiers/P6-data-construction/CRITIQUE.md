# CRITIQUE — P6-data-construction（独立批判审计，fork=false）

## ① 是什么
训练/评测数据与任务如何构造（enwik8 / 损坏增广 / needle / 去噪 / 合成），决定结论能否迁移。

## ② 对着什么 sota，我们偏离在哪
业界标准：NIAH（Kamradt 2023）、MQAR/copy/induction 合成任务（`记忆·未核`）。本地增量是真的：**§19 局部先验抗性**（"越小越弱的模型对远距离 context 越不敏感"）及其在 A4 的"假保留陷阱"实证（`catalog.yaml:273–302`）。

## ③ 五类问题逐条判定
- **开倒车：无。**
- **过程错误：有（本格被判 `judged` 但没有实验；且协议未强制）。**
  - `axes/P-modifiers/P6-data-construction/` 下**没有任何实验目录**（`experiments/` 只有 README），`status: judged` 的 evidence 全在别的轴上（`.scratch/05-m1-followup`、`.scratch/07-x1-x3`）。
  - 本仓最重要的 P6 争议——"enwik8 上所有长程结论都被局部先验污染"——只在少数线执行；`docs/OPEN_PROBLEMS.md` P-11 自认"**合成任务与真实文本之间的口径裂缝**：真实文本代理口径的**自身噪声（同一模型只换种子：24.81pp）大于要测的效应**"。
  - needle v2 的结论已核到并成立：`.scratch/05-m1-followup/issues/01-mixture-needle-followup.md:39`"X2v2 PASS — needle redesign works: in-window 84.9%, anchor 76.8%, beyond 0.4%（allowed to fail）；v1 failed on signal starvation only"。
- **结论错误：n/a**（元结论），但"任务构造是本项目最大的单点偏差来源"**尚未落成强制协议**，所以它现在是"知道了但没执行"。
- **无意义：否。** 这是方法学核心。
- **值得做：是（升格为强制协议而非"格"）。**

## ④ 建议动作：保留，升格为协议 + 补成对检验
1. 把 §19 落成硬性协议：凡长程/记忆/压缩类结论，**必须**同时给"强局部先验（enwik8）"与"弱局部先验（合成）"两套口径；两者结论不同时**以合成为准**（`SCHEMA.md §19` 已有此条，需在契约模板里做成必填项）。
2. 补"同一权重、两套口径"的**成对**检验（现多为跨模型拼接）。
3. 摄入已有方差证据：ticket 42-W2 的近 OOD 腿同配置三跑极差 **10.840pp**（`catalog.yaml:1013–1035`）——这是"该腿不可作主判据"的实证，应作为 P6 的标准警示。

## ⑤ 正确的验证方式
- **该涨**：合成任务上长程机制应显著 > chance（chance=1/64）。
- **该跌**：无。
- **必须不变**：毁源后掉回 chance；局部先验基线（0.0098–0.0217）低于 chance 上限；评测前向逐位一致。
