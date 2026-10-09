# PROBLEMS-TODOS — 问题台账仍有效行动项汇总

- 性质：日志文档（过程记录）。原文为事实树里的 `axes/**/PROBLEMS.md`（问题台账），
  已按文档 SSOT 化移出事实树到 `.scratch/doc-log/<原相对路径>`；本文件只汇总其中
  **仍有效的行动项**，供后续立项/派发使用。
- 归属：问题台账归主 Agent；本文件不改任何已入账结论与数字。

## M1-carrier-type（原 `axes/M-carrier/M1-carrier-type/PROBLEMS.md`）

- P-15（最高优先级）：递归 / 线性载体（SSM、Mamba、RWKV 一族）质量完全空白。
- P-10：单种子是系统性缺陷 ⇒ 跨种子复现升为前置条件（现结论 n=1，且唯一跨种子项已反号）。
- P-11：合成任务与真实文本之间的口径裂缝 ⇒ 补真实文本口径的同口径复现
  （现结论几乎全在合成关联回忆上）。

## R1-quantization（原 `axes/R-references/R1-quantization/PROBLEMS.md`）

- P-14：量化边界未定位 ⇒ 补 5/6 位与旋转 / 投影类量化（TurboQuant 一族）；
  权重与激活量化未测。
- P-17：本格为业界结论且**未经核实**（联网不可用）；核实后才能作为免做依据。

## A4-soft-distance-penalty（原 `axes/A-mixing/A4-soft-distance-penalty/PROBLEMS.md`）

- P-03（归因错配）：本格属于信息右移族（注意力按距离加权），与层序无关 ⇒ 更正汇报归属。
- P-13：「保留型近乎免费」目前是单点证据（−0.0068，落在不可排序带内）⇒ 要么补点，
  要么把措辞降级为「倾向」。

## A6-mixing-operator（原 `axes/A-mixing/A6-mixing-operator/PROBLEMS.md`）

- P-16：长 T 四点口径不齐（256/512/1024 为半预算快照，与旧点不可比）⇒ 补齐四点口径。
- P-15：线性注意力质量数据缺失（当前实装停摆）⇒ 先解决实现，再谈质量。

## A9-old-info-retention（原 `axes/A-mixing/A9-old-info-retention/PROBLEMS.md`）

- P-10（同 M1）：单种子是系统性缺陷 ⇒ 结论 n=1，待跨种子复现。
- P-11（同 M1）：合成任务与真实文本之间的口径裂缝 ⇒ 补真实文本口径的同口径复现。
- 「给通道」正结论需补复现（「靠掩码逼不出搬运」的负结论边界已写清）。

## C1-layers-per-token（原 `axes/C-vertical/C1-layers-per-token/PROBLEMS.md`）

- P-02：投机线缺口 ⇒ 补「链式 vs 直接预测 + 概率组合」的显式实验与 100M 规模收益。
- P-12：形态税是否值得付属框架性取舍 ⇒ 请用户裁决。

## C2-layer-selection-order（原 `axes/C-vertical/C2-layer-selection-order/PROBLEMS.md`）

- P-13：「元素税定律」措辞需按证据强度校准。
