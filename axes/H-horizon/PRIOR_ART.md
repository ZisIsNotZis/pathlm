# PRIOR_ART — 视界（MTP / 投机解码 / 多位置预测）的既有做法

> 轴：**H 视界（一次前向产生几个位置）** ｜ 格子：`H1-outputs-per-pass` … `H5-verification`
> 角色：researcher（只读 + 本文件）｜ 本波**无 GPU、不跑训练/评测、不取锁**
> 依据：`SCHEMA.md` §18（既有做法）、§19（局部先验抗性）、§21（不重复造轮子）

---

## 0. 检索条件声明（**必读，决定本文证据等级**）

**本环境联网检索不可用**（上一轮 researcher 因 web 工具未注册而失败）。因此本文**无法在线核证**
任何外部数字。遵守：

- 凡来自我记忆的论文/系统，一律标 **`记忆·未核`**，**只允许作「设计方向」，不得当判据**（§18）。
- 凡来自本仓库的数，标 **`本地实测`** 并给路径（`catalog.yaml` 的 `local:` 块及其 `evidence`）。
- 不确定的一律写 `不确定` + 「要查什么才能确定」。**不编数字**：记忆中的数字降级为定性。
- 术语就地解释：**MTP**（multi-token prediction，一次前向预测多个未来位置）；
  **投机解码 / spec**（用便宜草稿先猜几个 token，再由全深度前向批量验证）；
  **税**（相对普通 transformer 的质量代价，用 clean bpc 差衡量）；
  **bpc**（每字符比特数，越低越好）；**`tf@k`**（用前 k 层当草稿、被全深度接受的比例）；
  **批量验证**（一次前向同时验证全部草稿位置）。

**若可检索应查的关键词清单**见 §3。

---

## 1. 既有做法（按格）

### H1 — 每步输出位置数（MTP：一次前向预测 k 个未来位置）

**谁做过 / 为什么（`记忆·未核`）**

- **投机解码**：`Leviathan et al., 2023`（Fast Inference via Speculative Decoding）、
  `Chen et al., 2023`（Speculative Sampling）。动机：自回归逐 token 解码是**访存带宽受限**，
  用便宜草稿 + 一次批量验证，把多个 token 塞进一次全深度前向。**核心定理：投机解码无损**
  （输出分布与目标模型逐步贪心一致）——这一条**是本轴 H5 的行业基石**。
- **块并行解码**：`Stern et al., 2018`（Blockwise Parallel Decoding）——MTP 的早期形态。
- **Medusa**：`Cai et al., 2024`——在**冻结主干**上挂多个解码头 + 树注意力批量验证。
  分 Medusa-1（冻结主干）/ Medusa-2（联合微调）。
- **多 token 预测（质量侧）**：`Gloeckle et al., 2024`（Meta，Better & Faster LLM via MTP）
  报告 MTP **同时**改善质量并自带自投机能力；**DeepSeek-V3（2024）**把 MTP 作为训练目标并
  用于推理加速。`不确定`：各自的**确切税值与接受率数字**，需查原文表。

**它声称解决**：解码吞吐（草稿从哪来、接受率多高、验证多贵），以及（附属）训练信号更密。

**实测结果（定性，`记忆·未核`）**：MTP/spec 的**加速比随批次与模型规模变化**——
小批次/小模型是 **launch-bound（核启动受限）**，加速比高；大批次/大模型转 **compute-bound
（算力受限）**，加速比被摊薄。接受率是主旋钮。`数字未核`。

**我们的增量**：**13M / char-enwik8 / 同尺寸同 token 预算下的税曲线**（n_mtp 1→2→3）
**+ 同引擎三口径（forwards/FLOPs/墙钟）**。业界报加速比但**少报「草稿从哪来的税」**、
**很少在极小尺度上给完整税-接受率-三口径表**。

**「业界是否已有此结论」判定（`H1` 是 `status: judged`）**：
- 结论一「MTP 税小、spec 加速比可观」→ **业界已有**（`记忆·未核`）⇒
  **该条标「业界结论」口径**：来源类型 = **业界结论，非本地实测**。
- 结论二「墙钟收益是 13M/launch-bound 现象、@100M 前向 compute-bound 后收益塌」→
  **业界机制早已知道**（compute-bound vs launch-bound，`记忆·未核`）⇒
  **该条标「业界结论」口径**。
- 结论三「逐位一致（spec 输出 == 贪心）在 @100M 因 fp 平局翻转失效」→ 这是**本地规模特有**
  的发现（`本地实测`），业界讨论 fp 平局但少见此具体判据 ⇒ **保留 `judged`**。

**可复现的判据建议**：横轴 n_mtp∈{1,2,3}；纵轴三色 = ① Δclean bpc（税，扫同 seed）；
② 草稿接受率（含 `a₂`、`a₃|a₂`）；③ 端到端 tok/s（**必须同时给 FLOPs 与墙钟两口径**）。
**正确性腿**：spec 逐位 vs 贪心（但按本地教训，@大尺度不能只靠逐位相等）。

---

### H2 — 训练时未来位置的内容填充

**谁做过 / 为什么（`记忆·未核`）**

- **主流 MTP 训练不「填充」**：DeepSeek-V3 MTP 用**独立的顺序头**直接预测 t+1/t+2 的目标；
  Medusa 头直接拟合并行草稿——**都不填右侧内容**。⇒ 我们的「填充内容对照」不是行业标准做法。
- **填空/去噪类目标**：`fill-in-the-middle`（FIM，如 Codex 类）、masked-token 预测、
  扩散语言模型（见 F1）——这些**用 [mask]/噪声填充**，动机是生成/补齐而非 MTP。
- `不确定`：是否有工作系统比较过「self-fill / proj-fill / mask / 错误 token / 软 latent」
  这一组填充方案的税。

**我们的增量**：**同引擎、同 seed 单因素扫描**填充内容（mask/wrong-token/noise/soft
latent/self-fill/proj-fill），并给出「proj-fill（逐层恒等初始化 adapter）最优」的受控结论。
这是**结构性新对照**，业界少见。

**「业界是否已有此结论」判定（`H2` 是 `status: judged`）**：填充内容对照**未找到**直接前例
⇒ **判 (b)**，保留 `judged`。但「soft 提交 OOD 污染」与 ML 通用现象（软标签/凸包外插）同源，
`记忆·未核` 可能有旁证。

**可复现的判据建议**：两臂/多臂同 seed；判据 = Δclean bpc（税）+ 出口可读性 + 提交质量
（rescore）。**必须**报活性量（填充位置占比、adapter 是否真被用到）。

---

### H3 — 多位置预测的组合方式

**谁做过 / 为什么（`记忆·未核`）**

- **链式条件（concat+proj，顺序头）**= DeepSeek-V3 MTP 的配方。
- **并行头** = Medusa。
- **朴素概率乘积**（把各位置边缘概率相乘当联合概率）——**这是数学错误**：
  联合概率 ≠ 边缘概率之积（除独立外）。⇒ 这一条**属数学事实，非实验结论**。
- **beam/lattice 重排**：通用解码技术。

**我们的增量**：**用户提出的「自门控组合深度」**（一次跳 k vs 跳 1 走 k 次，sigmoid 门决定）
在本项目被**实测否决为主张**（门坍缩到连跳，增益归容量不归门控，SCHEMA §15）。这是**新设计
的受控否证**，业界没有。

**「业界是否已有此结论」判定（`H3` 是 `status: judged`）**：
- 「朴素概率复合是数学错误」→ **数学事实，业界显然已知**（`记忆·未核`）⇒
  **该条标「业界结论」口径**（来源类型 = 业界公认/数学事实）。
- 「自门控组合无结构优势」→ **(b)** 新设计否证，保留 `judged`。

**可复现的判据建议**：等 FLOPs 判决（1 次前向 + 跳跃头预测 t+2 vs 2 次前向推到 t+2）+
**等参数无门对照**（§15，否则会把容量增益记成门控增益）+ 门分布（是否坍缩到单支）。

---

### H4 — 头与主干的关系（联合训练 vs 冻结主干只训头）

**谁做过 / 为什么（`记忆·未核`）**

- **Medusa-1 = 冻结主干只训头 → 零税**；`EAGLE-1` 在冻结主干上训特征级草稿。
  动机：挂头不扰动主干、单臂便宜。
- **联合训练** = Medusa-2 / DeepSeek MTP（把 MTP 目标并进预训练）。
- **权重绑定 embedding / 独立解码头**：通用设计选择。

**我们的增量**：13M 上**「冻结主干挂跳跃头 Δbpc = 0.0000」逐位坐实** + **诚实对手口径**
（vs 1-step AR 两次前向的 0.5489，跳跃头 = 85%，落中性带）+ 冻结表征不可达 t+2 的**归因**
（排除 off-by-one/冻结失败/头映射/步数容量）。

**「业界是否已有此结论」判定（`H4` 是 `status: judged`）**：
- 「冻结主干只训头 → 税≈0」→ **业界已有（Medusa-1/EAGLE-1）**（`记忆·未核`）⇒
  **该条标「业界结论」口径**（来源类型 = 业界结论，非本地实测）。
- 「冻结表征只被 t+1 目标塑形 ⇒ t+2 映射不可达（85% 而非 ≥90%）」→ **(b)** 本地受控新结论，
  保留 `judged`。

**可复现的判据建议**：预登记 **deploy rollout 为对手**，oracle 只作上界附报（SCHEMA §11）；
判据 = 头质量 vs 1-step AR 两次前向 + 草稿接受率 vs 主干自身输出 + 冻结断言（grad is None）。

---

### H5 — 验证机制

**谁做过 / 为什么（`记忆·未核`）**

- **投机解码的批量验证无损性**：Leviathan 2023 / Chen 2023 证明「一次前向验证全部草稿、
  按拒绝点截断」与目标模型**逐步采样的输出分布一致**。Medusa 用**树注意力**做批量验证。
  ⇒ 「批量验证逐位 == 贪心」是**行业定理级结论**。

**我们的增量**：本引擎的**批量验证 + 定稿不变量（全深度重打分）**实现，及
**@100M fp 平局翻转（1/400 token）使逐位一致失效**的规模特有发现。

**「业界是否已有此结论」判定（`H5` 是 `status: judged`）**：
- 「批量验证可用且逐位一致」→ **业界已有**（spec 无损定理）⇒
  **该条改「业界结论」口径**（来源类型 = 业界结论，非本地实测）。
  应写入结论句：**「批量验证草稿在数学上与逐步贪心等价（投机解码无损性，`记忆·未核`：
  Leviathan 2023 / Chen 2023）；本地复现一致（13M，`本地实测`）。」**
- 「@100M fp 平局使逐位相等不可作跨规模正确性判据」→ **(b)** 本地增量，保留 `judged`。
- 「exit × spec 在本引擎不可组合」→ **(b)** 本引擎实现约束，本地结论。

**可复现的判据建议**：强制全接受/强制拒绝/窗口驱逐三个专项单测；正确性腿按规模设阈值
（不能只靠逐位相等）。

---

## 2. 机器可读分诊表

> 判定取值：**(a)** 业界已研究透（不跑实验，写结论 + `status: todo`）｜
> **(b)** 需本地实测（业界没有/结论不一致/我们设定实质不同）｜
> **(c)** 查不到（含检索不可用，按 `记忆·未核` 记方向，**不得据此判 (a)**）。
> `+b` 表示主判为 (a)/(c) 但含本地增量子结论需保留 `judged`。

| grid_id | 判定 | 一句话理由 | 拟写进 `catalog.yaml` 的结论 |
|---|---|---|---|
| `H1-outputs-per-pass` | (a)+b | MTP 税小、spec 加速、规模决定墙钟收益——业界已知；fp 翻转规模细节是本地增量 | 参照线句：「MTP/MTP-spec 能减小质量税并加速解码，但墙钟收益随规模由 launch-bound 转 compute-bound（`记忆·未核`，Gloeckle 2024 / Medusa 2024）；本地 13M 复现且 @100M 墙钟收益消失（`本地实测`）。」本地 fp 平局翻转保留 `judged` |
| `H2-training-fill` | (b) | 填充内容单因素对照未找到直接前例；主流 MTP 不填充 | （实验结论）「同引擎下 proj-fill（逐层恒等 adapter）优于 fill/nofill/mask/wrong-token/noise；soft 提交有 OOD 污染。」保留 `judged` |
| `H3-composition` | (a)+b | 朴素概率乘积错=数学事实（业界已知）；自门控组合否证=本地新设计 | 参照线句：「多位置预测的正确组合是顺序条件头或并行头；朴素概率乘积在数学上不等价于联合概率（业界公认，`记忆·未核`）。」自门控否证保留 `judged` |
| `H4-head-trunk-relation` | (a)+b | 冻结主干只训头→零税=Medusa-1/EAGLE-1 已知；冻结表征 t+2 不可达=本地受控新结论 | 参照线句：「在冻结主干上挂头可做到零税（Medusa-1 / EAGLE-1，`记忆·未核`）；本地 13M 逐位坐实 Δbpc=0（`本地实测`）。」质量中性带结论保留 `judged` |
| `H5-verification` | (a)+b | 批量验证逐位等价=投机解码无损定理（业界已知）；fp 翻转=本地规模细节 | 参照线句：「投机解码的批量验证与逐步贪心在数学上等价（无损性定理，`记忆·未核`：Leviathan 2023 / Chen 2023）。」fp 平局与 exit×spec 约束保留 `judged` |

---

## 3. 若可检索应查的关键词

| 格 | 关键词 |
|---|---|
| H1 | `speculative decoding fast inference Leviathan`, `speculative sampling Chen 2023`, `blockwise parallel decoding Stern 2018`, `Medusa multiple decoding heads frozen backbone`, `multi-token prediction Gloeckle Meta`, `DeepSeek-V3 MTP`, `speculative decoding batch size speedup compute bound launch bound` |
| H2 | `multi-token prediction training target fill`, `fill-in-the-middle training`, `masked token prediction auxiliary loss`, `self-fill next-token teacher forcing vs self-feed` |
| H3 | `chained MTP head DeepSeek conditional`, `parallel medusa heads`, `joint probability product of marginals error`, `gated composition depth router`, `speculative tree attention rerank` |
| H4 | `Medusa-1 frozen backbone zero tax`, `EAGLE-1 feature draft frozen`, `draft head representation frozen mapping`, `direct vs composed jump head acceptance` |
| H5 | `speculative decoding lossless exact distribution`, `tree attention verification Medusa`, `greedy equivalence fp tie flip`, `spec decoding correctness at scale` |

**核证优先级**：`H5 投机解码无损定理`（决定 H5 是否改「业界结论」）与
`H1 规模-加速比 regime`（决定 H1 参照线措辞）。

---

## 4. 来源清单

**来源类型图例**：`官网/论文（含链接）`｜`记忆（未核）`｜`本地实测`。
（本波无检索工具 ⇒ **无一条「官网/论文（含链接）」**；所有外部条目均为 `记忆·未核`。）

**本地实测（可直接当证据；路径见 `catalog.yaml` `H1`–`H5` 的 `local.evidence`）**
- `H1-outputs-per-pass`：税 n_mtp 1→2 +0.041、2→3 +0.0166；Medusa 批量验证 1.68–1.90×@13M；
  @100M 墙钟收益消失（1.15–1.29×）；fp 平局翻转 1/400 token。证据 `.scratch/10-gap-sweep`、
  `.scratch/13-four-experiments`、`.scratch/20-spec-scale`。
- `H2-training-fill`：mask/wrong-token +0.059/+0.127；self-fill 1.8582；proj-fill 1.8356 最优；
  soft 提交 OOD 污染三度复现。证据 `.scratch/06-m2-sweep`、`.scratch/23-depth-ar`、
  `.scratch/24-depth-ar-100m`。
- `H3-composition`：gated 0.4738 vs 直跳 0.4672（+0.0020 < 0.014）；等参数无门对照 0.4785；
  门坍缩 96.7% 偏两跳。证据 `.scratch/12-overnight`、`.scratch/04-m1-runs`。
- `H4-head-trunk-relation`：冻结主干挂跳跃头 Δbpc=0.0000；诚实对手 1-step AR 0.5489；
  跳跃头 0.4667（85%）；草稿接受率 0.5998。证据 `.scratch/18-exit-probe`、
  `.scratch/21-exit-at-scale`、`axes/H-horizon/H4-head-trunk-relation/experiments/*/analysis.md`。
- `H5-verification`：Medusa 批量验证逐位 == 贪心（三个专项测试）；稳态 2.40 tok/fwd。
  证据 `.scratch/13-four-experiments`、`.scratch/20-spec-scale`。

**记忆·未核（仅作设计方向，不得当判据）**
- 投机解码：Leviathan 2023、Chen 2023、Stern 2018（块并行）、Medusa（Cai 2024）、
  EAGLE（Li 2024）、Gloeckle 2024（Meta MTP）、DeepSeek-V3 MTP。
- 填充/去噪：fill-in-the-middle、masked-token 预测。

**不确定 / 未找到（明确列出，防误当已知）**
1. MTP/Medusa/EAGLE 的**确切税值与接受率数字**：`数字未核`，需查原文表。
2. 「填充内容」这一组（self-fill/proj-fill/mask/wrong/soft/noise）的系统对照：`未找到`。
3. 自门控组合深度：`未找到`先例（疑为真空白，但检索不可用，不能断言）。
4. 「冻结表征 t+2 不可达」是否有同结论的公开工作：`不确定`。
5. `fp 平局翻转`作为正确性判据失效前例：`不确定`。

**诚实清单**：本文无任何训练/评测，无 `results.json`，**只是情报**；所有外部内容均未在线核证。
