# CRITIQUE · F2-return-channel（回传通道）— 独立批判审计（2026-10-04）

## ① 这是什么（我的理解）

研究「上一轮的结果以什么形式回到下一轮」，即决定重复能否带来**新信息**；本仓扫了
direct / linear / soft / hard / 掩码 / verifier 六种通道。

## ② 对着什么 sota，我们偏离在哪

- 业界标准：Universal Transformer 每步把输入加回状态；DEQ / gated recurrence 是同一族；
  软 vs 硬潜变量（VQ/straight-through）的「软提交凸包安全、外插 OOD 有风险」是通用结论
  （`axes/F-iteration/PRIOR_ART.md:60-62`，均 `记忆·未核`）。
- 偏离：我们的「通道组单因素扫描」被自己判为未找到直接前例（`PRIOR_ART.md:161`），
  但**扫描本身没跑在有效引擎上**（见 ③）。

## ③ 五类问题逐条判定

1. **开倒车：有（轻）。** 「循环需输入注入，否则退化为固定点」= 业界教训
   （`PRIOR_ART.md:45,72`），被写成 `verdict` 的第一句。

2. **过程错误：有，且比 F1 更直接。**
   - 本格的税字段仍是 F1 的 C 族数字（`catalog.yaml:946`：「通道税在 F1 的 C 族内
     （+0.146…+0.178）」）⇒ 继承 F1 的 pre-D1/D3 引擎问题。
   - **D1 的 gauge bug 直接废掉了通道对比**：`issues/01-retry-matrix-validity.md` D1 段实测
     「`_mixture_reentry(direct)` 输出 norm 1.00 vs direct overwrite 28.7；linear 投影
     放大 71.7×再被压回 1.00；soft 的 cap 是 no-op」⇒ 「reported ordering soft > linear > direct
     是单调于 *cap 破坏了多少*，不是保留了多少信息」。⇒ 在 D1 修复前后，本格的通道排序
     **从未在可比口径上被测过**。
   - D6（linear 不是投影，是放大 72×）未修。

3. **结论错误：有。**
   - `verdict`：「直接回传是固定点（**结构性** no-op）」（`F2-return-channel/README.md:11,22,47`）。
     代码事实：`pc.transport == "direct"` 时 `reentry = h`，随后 `self._run_layers(h, path)`
     **把上一轮的输出再喂回同一栈**（`pathlm/model.py:398-401` + `:446`）——这在数学上
     **不是恒等**（f(f(x)) ≠ f(x) 一般成立）。
     M0 的解释只是「no new information enters」（信息论意义），并不能推出「固定点」。
     实测也只是 1M 探针上 ±0.001（`REPORT.md` gate d）。
   - **正确结论应该是**：「direct 回传**不注入任何外部新信息**；在 M0（1M、2 层）实测
     第二轮对修复无增益（0.1889→0.1897）。这是经验事实，不是结构性不可能」。
   - 「软回传有 OOD 污染」有独立证据（H2），这条可留。

4. **无意义：无（通道问题本身是真问题）**，但「通道税」这一数字字段在修复前**不可引用**。

5. **值得做：有（重做）** —— 这是 F 轴唯一值得再做的一件事。

## ④ 建议动作

- **重做**：修 D6（真正交投影或显式归一化标尺）+ 用 D3 的几何重试，在**同一批、同一 seed、
  同 token 预算**下重扫 `direct / linear / soft / hard` 四档，并落 `intervention.alive` 活力量。
  成本：4 臂 × 13M × 12k ≈ 40 min GPU（含冒烟/评测）。
- **改措辞**：`verdict` 与轴 README 里「结构性 no-op」「第一条铁律」两处必须降级。
- **保留**：D1 gauge bug 与 D6 作为方法学教训（它们本身就是有价值的产物）。

## ⑤ 正确验证方式

- **该涨**：能注入新信息的通道（soft/hard/verifier）相对 `direct` 的**逐轮 bpc 边际**
  （在损坏输入上、同轮数对照下）；以及 `hard + mixture 投票` 相对单轮 gambled 的修复准确率。
- **该跌**：`direct` 的逐轮边际应停在 0（若真的停了，要把「无新信息」与「实现退化」分开：
  必须先证明 `alive=true`、`rel_dev_vs_channel_off > 0`）。
- **必须不变**：修复前后 `transport=none` 的 clean bpc 必须逐位一致（否则不是同一引擎）；
  各档参数量与初始化必须逐位相同（本仓 U1/U2 已用单测钉住，可复用同法）。
