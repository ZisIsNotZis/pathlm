# CRITIQUE — R5-bestofn-verifier（独立批判审计，fork=false）

## ① 是什么
用额外推理算力做"采样 n 个候选 + verifier/自一致性重排"——test-time compute 的标准答案。

## ② 对着什么 sota，我们偏离在哪
业界标准：Cobbe verifiers、self-consistency、PRM/Lightman（`记忆·未核`）。我们**零本地实验**；`catalog.yaml:1264–1270` 的"本地侧证"用的是 **G5 的「共享主干的多线增益可忽略」**（evidence `.scratch/08-tts-probe/evidence`）。

## ③ 五类问题逐条判定
- **开倒车：无。**
- **过程错误：有（命题错配）。** G5 测的是"同一权重、确定性路径的 K-path 集成"，用户已在 2026-10-02 明确判定该命题是**执行方自设、且与用户要的投机解码无关**（`docs/OPEN_PROBLEMS.md` P-01："跳层的多线肯定没有意义"；`catalog.yaml:1169–1197` 同步标注）。用 G5 的结果当 best-of-n 的侧证＝**换了一个命题**（`SCHEMA.md §13` 三要素里"部署中对应什么"就不成立）。
- **结论错误：有。** "本地侧证：共享主干的多线增益可忽略"被用来支撑"best-of-n 与 G5 多线直接竞争"——它们不竞争：best-of-n 的候选来自**随机采样且前缀不同**，G5 的候选是**确定性路径**。`catalog.yaml:1174–1176` 已把 beam search 的同类推论撤回（C9），R5 的这条推论同样应撤。
- **无意义：是。** 本项目**不存在 verifier**（无学习式打分器、无 PRM），"rank 候选"这一环零经验、零用例；作为参照线它不改变任何判断。
- **值得做：否**（除非用户把 test-time compute 列为目标）。

## ④ 建议动作：删除或重写为"待核基线 + 与 G5 解绑"
- 建议**删除**该格，把"best-of-n + verifier 是同 FLOPs 的标准手段（待核）"列为 `G5-parallel-lines` 的 industry 一句，并明写"本项目无 verifier，未测"。
- 理由：现状是"用一个被用户判为无关的自设命题的结论，冒充另一条线的侧证"。

## ⑤ 若重做（需先立用户目标）
判据 = 同 FLOPs 命中率 / verifier 校准（ECE）；必须不变 = 单线 greedy 基线；必须双报 n 与采样温度。
