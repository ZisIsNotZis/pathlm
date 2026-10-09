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
