# CRITIQUE · F1-iteration-object（迭代对象）— 独立批判审计（2026-10-04）

定位：本文件只读产出，不改代码、不改数字、不改 catalog。证据均为 `文件:行`／原始读数。

## ① 这是什么（我的理解，一句话）

研究「同一份算力应该重复施加在**什么对象**上」；本仓实际只做过「同一 token 的隐状态
再过一遍」这一种，其余五种（同段多遍／整段去噪／候选集／隐式推理步／权重迭代）
在本地是零实验。

## ② 对着什么 sota，我们偏离在哪

- 业界标准：looped / recurrent depth（Universal Transformer、Geiping 2024、Huginn 系）
  的标准配方是「**共享层 + 输入注入 + 自适应步数**」；扩散 LM（Diffusion-LM/SEDD）、
  best-of-n、pause token（Goyal 2024）、fast-weight/TTT（Sun 2024）各自成线。
- 偏离：本仓把「direct latent 重入 = 结构性 no-op」当成**自创的第一条铁律**
  （`axes/F-iteration/README.md:19`、`F2-return-channel/README.md:51`），
  而同一条在业界只是「循环式重复若无输入注入会退化为固定点」的**既有教训**
  （`axes/F-iteration/PRIOR_ART.md:33,45`）。我们把它当新发现，且用词比证据强（见 ③-3）。

## ③ 五类问题逐条判定

1. **开倒车：有。**
   - 选项 3–6（整段去噪／候选集／隐式推理步／权重迭代）在 `PRIOR_ART.md:143` 的分诊
     里已被本仓自己判为「业界成熟路线（`记忆·未核`）」，却仍以「后四项零经验」
     （`F1-iteration-object/README.md:8`）留在目录里当待做项。
   - 「循环需输入注入」是业界公论，我们做成了本地「铁律」并写成参照线（`PRIOR_ART.md:72-73`）。

2. **过程错误：有，且是本格最重的。**
   - 本格税值 `+0.146…+0.178`（`catalog.yaml:918`、`F1-iteration-object/README.md:6`）取自
     **pre-D1/D3 引擎**。ticket 11 自己写死：「no retry number in findings.md survives as-is」、
     「at p=0.5 the mean retry count goes 0.5 → 1.0, so the retry runs get ~33% more expected
     rounds and **all prior retry numbers must be re-measured**」
     （`.scratch/11-retry-matrix-validity/issues/01-retry-matrix-validity.md`「IMPORTANT scoping note」段），
     且 `.scratch/11-retry-matrix-validity/evidence/README.md` D5 结尾明写
     「the fixed engine is unmeasured」。
   - 同 ticket 的 D6（linear transport 实为 `EᵀE`，谱范数 72.9，用 `cap_norm` 掩盖）
     在验收单里**仍未勾选**；D4（无重复）也未勾。⇒ 税、通道排名、「结构性 no-op」
     三条结论都落在已作废或未复现的引擎上。
   - 单种子：`evidence/README.md` D4 给出 seed 散布 0.0001–0.0138（cell 依赖），
     C2M_n1 的种子散布（0.0138）**大于整个六格跨度（0.011）**，这是「排序撤回」的根因。

3. **结论错误：有。**
   - 轴结语「可靠的是『**多一遍有价值**，回传几何无价值』」（`axes/F-iteration/README.md:19`）
     两半都过了：
     · 「回传几何无价值」的实际证据是「6 个 latent 回传 cell 跨度仅 0.011 < 种子散布
       0.0138 → 全部排序结论撤回」（`catalog.yaml:920`）——**不可排序 ≠ 无价值**。
     · 「多一遍有价值」在本仓 M0 上是**反例**：gate (d)「one latent-retry round improves
       repair」判 **FAIL**（self-acc 0.1889 → 0.1897）（`.scratch/02-engine-m0/evidence/REPORT.md`
       的 Gate verdicts 表）。真正的正向读数只出现在「13M + 15% 损坏输入 + prob0 门控」
       （`.scratch/14-allocator/issues/01-step1-completion.md` 实验 1 裁决表）。
   - **正确结论应该是**：「13M／损坏输入／门控条件下，第二轮有小幅修复收益（INT2
     门控 15% +0.0211 vs 均匀 −0.0193，bpc 口径）；重入与回传通道的**几何**效应在本
     方差带内不可分辨，且旧引擎已作废，需在修复后引擎上重测」。

4. **无意义：部分。**
   - 选项 3–6 在「单卡 13M char-enwik8 / 自回归 bpc」这个唯一基底上不会改变「该算力路径
     值不值」的判断（去噪、隐式步、TTT 都是另立基底的问题）⇒ 建议按 SCHEMA §21(a)
     写「业界结论」 后从「待做」里移除。
   - 「回传几何」这个子命题：无论结论正负都不会改变任何部署判断（各 cell 差 < 方差带）
     ⇒ 建议只保留「不可分辨」这一句。

5. **值得做：几乎无（本格范围内）。** 唯一值得做的是把 D1/D3/D6 修好后**重测通道**
   ——但那属于 F2/F1 交界，且属于「重做」而非「补做」。

## ④ 建议动作

- **重做（仅限与 F2 交界的通道复测）**：见 F2 的 CRITIQUE。
- **改「业界结论」 并移出待做**：选项 3、4、5、6（去噪／best-of-n／隐式推理步／权重迭代）。
- **保留**：M0 gate (d) 的负结果（retry 在 1M 是 no-op）——它是本格最硬的一条。
- **保留但改措辞**：「回传几何」= 不可分辨，不是无价值。

## ⑤ 若重做（通道复测）的正确验证方式

- **该涨**：修复引擎下，损坏输入上 r1→r2 的**修复准确率 / 答案行 CE**随轮次单调改善
  （对照 = 同引擎 `transport=none`）。
- **该跌**：无（本格不主张任何「更好」）。
- **必须不变**：clean 输入上的 bpc 税**符号与时序平台**（不应因引擎修复而从「真实税」
  变成「收敛加速免费」）；`p_retry` 的期望轮次要显式报出（D3 改后 0.5→1.0），
  否则跨批不可比。
