# A 轴（序列混合机制）批判性审计 · 2026-10-03

审查者：独立子代理，`fork=false`，只读，未跑 GPU、未跑 pytest、未改任何文件/数字。
范围：`axes/A-mixing/**` 全部 10 格 + 其 in-repo 实验（19 个目录）+ `catalog.yaml` 的 A 段 + `PRIOR_ART.md` + `docs/{methodology,playbook,design,OPEN_PROBLEMS}.md`。
起点：`.scratch/49-adversarial-review/REPORT.md`（本轮**独立复核了其中与本轴相关的部分**，并另立了 49-REPORT 未涵盖的发现）。

---

## 0. 一句话总判

**不能用。** 这 10 格里只有 3 格有仓内可核实验（A6/A9/A10），而这三格的**头号结论都站在坏仪器或错口径上**（A6 用伪影长程腿 + 步数不等的跨 T 比较；A9 的头号数字位置特异；A10 用算术不可达的判据 + 与自己命题不符的实验）；另外 **A3/A4 两格的"结论"完全没有仓内实验**，只有 `.scratch` 里一份 20 行 markdown，且其机制是业界标准件；A1/A5/A7/A8 是记忆级条目，其中 6 条的 `prior_art_note` **被写进 YAML 注释且整体错位一格**（新发现，见 §3）。可以当"待重做的索引与线索"，**不能当"已独立验证的结论库"**。

---

## 1. 建议删除清单（按 AGENTS.md「研究树的判据」）

| 格 | 建议 | 理由（证据） |
|---|---|---|
| **A5-dilated-window** | **删除"实验"意图**（保留枚举维持完备性） | 固定稀疏模式在 13M/T=256 上的结果不可能改变"该路径值不值"；信息量已被 A3+A6 覆盖；`local` 为空（`catalog.yaml:314`）；且其 `options` YAML 已坏（`catalog.yaml:306` → 生成页把"跨步(k"/"stride s)" 当两个选项） |
| **A7-addressing** | **并入 A6**（同一条研究被拆两格） | A6 选项已含"Top-K 稀疏（内容选择式，NSA/MoBA 族）"，A7 选项含"内容式（QK 相似度）/学习式 top-k"（`catalog.yaml:325` vs `:367`）；AGENTS.md 明令"不要重复建格" |
| **A1-causality** | **删掉"判决实验设计"**（保留 reference 条目） | 非因果 LM 与 RAP 的 AR 不变式（`docs/design.md` §1）不相容；业界定论，本仓零数据（`catalog.yaml:205`） |
| **A8-positional-semantics** | **降级为 reference + 写进契约的必需条件** | "RoPE+PI 是长上下文标准件、绝对 PE 有硬上界"是教科书结论；本仓零实验；唯一真需求是"长上下文线必须换位置编码"这一**工程前提** |
| A2 / A3 / A4 / A6 / A9 / A10 | **不删**（A2 是锚点；A9 是本轴主格；A6/A10 有真实验；A3/A4 保留但降级机制结论） | — |

> 说明：删除/建格由主 Agent 决定；我**没有**动 `catalog.yaml`。若为了"目录完备"必须保留格子，则 A5/A7/A1/A8 请至少把"判决实验设计"段改为「不跑，理由 X」。

---

## 2. 建议重做清单（按优先级，写明重做要点）

**P1（不修则 A 轴结论不可用）**
1. **A9-2623 的长程腿仪器化**（最便宜、解锁别的格）：`var_recall`/单查询 `assoc_recall_batch` 把查询键放在**末位**，而已有文档证明该布局是位置外推伪影（`2623/analysis.md` §9：同权重在该布局 `full = 0.0000`，在训练同族多查询布局 0.3665）。把 **in-distribution 多查询布局**改成全仓默认长程腿，并把"末位布局"标为禁用。**成本**：改工具 + 复算已有权重，~CPU。
2. **A6 长 T 系列重判**：① 机制归属迁回 A3（`2622–2629` 是位置窗口）；② 三点改**同 step** 或同退火刻度（现在 `steps=24000/12000/6000`，`train_depth_ar.py:201-202` 的 lr 以 `steps` 为分母）；③ 补 n≥2 seed；④ 用 P1-1 的合法长程腿。**成本**：6–12 臂 × ~10 min，≈60–100 min GPU。
3. **每格种子散布**：把 A 轴所有"越界/带内"分类重判（见 §3-①）。要点：先测**本格**的跨种子 bpc 极差，再定界；不要再用全局 0.014。

**P2（结论措辞与证据链）**
4. **A3/A4 仓内自包含复算**：把 X1/X2/X3（惩罚 × 驱逐）重跑成自包含实验（同引擎、同数据流、n≥2），或把机制结论降为 `reference` 并删掉"真产物"措辞。要点：拆开"惩罚"与"驱逐"两个因素（现在 93.3% 是组合）。
5. **A10 重做**：① 流式（只看过去）统计；② 只按**可见性对数**做预算；③ 测真 ρ 曲线；④ 判据阈值先验算可达性（现在 10pp 门槛 > 上界 0.6868）。
6. **A6 top-k 补真核**：位置稀疏 vs 内容稀疏**同预算**对照；真块稀疏核把"理论省 FLOPs"兑现；T≥2048 的质量点。

**P3（低优先）**
7. A4 用**per-head slope** 的 ALiBi 测长度外推（唯一有决策价值的增量）；A2 补同 epoch 的 13M/100M 锚。

---

## 3. 开倒车清单（业界标准是什么、我们错在哪）

| 格 | 业界标准 | 我们的错 |
|---|---|---|
| **A3** | StreamingLLM（保 sink+滑窗即恢复）、Longformer/BigBird（global token）、gist token、Mistral SWA | 把"窗口+锚点便宜/锚通道有用"当本地发现示人（`status: judged`，`catalog.yaml:250,252`）；其 triage 判 `(a) 业界已做透` 却**写在 YAML 注释里**且错位（`catalog.yaml:237` 装的是 A2 的文本）⇒ 页面完全看不到"这是标准件" |
| **A4** | ALiBi（Press 2022，**per-head slopes**，卖点是**长度外推**） | ① 用**固定 w、全头共用**，丢掉 per-head slope ⇒ "全头被推向局部、长不出分工"是自设条件下的必然；② 只测同分布 bpc（ALiBi 论文自己报的就是"同分布中性"）⇒ `−0.0068` 无新增信息；③ **没测长度外推**，却把 judge 写成"bpc + 长度泛化"（`catalog.yaml:281`） |
| **A5** | Sparse Transformer / Longformer dilated / BigBird | 无实验，纯记忆条目，可接受；但 `options` YAML 已坏 |
| **A6** | NSA/MoBA（长上下文省算力）、Katharopoulos 线性注意力、Mamba/RWKV、Jamba/Zamba 混合 | ① 在 T=256（注意力仅占 14.3% MACs）测稀疏 ⇒ **结构性必亏**，业界从不这么卖；② 实装走稠密打分+稠密混合 ⇒ **实装 FLOPs 节省 = 0**（`pathlm/depth_ar.py:377-380`），"−43.75%"是假想核数字；③ 把"A6 注意力占比实测 vs 公式"当"公式高估"（单位错，实测 ≈ 公式/8） |
| **A7** | kNN-LM / RETRO / RAG / Memorizing Transformer | 无实验；与 A6 重复建格 |
| **A8** | RoPE + PI/NTK/YaRN | 本地仍用绝对可学习位置嵌入 ⇒ 长上下文 `T ≤ seq_len` 硬上界；结论正确但不是本仓的 |
| **A10** | H2O / Scissorhands / SnapKV / PyramidKV（访问/重要性感知驱逐） | 备注写"业界无人占位"（`catalog.yaml:415`），与本仓自己的 `PRIOR_ART.md` §A10 直接矛盾 |

---

## 4. 本轮最严重的三件事（P0/P1）

### ① 全场用来判"越界/带内"的 0.014 界，被它自己的来源否证（最重，影响整轴）
- `docs/methodology.md:19`、`SCHEMA.md:74,105`、`CONTRIBUTING.md:25` 把 `Δ < 0.014 不可排序` 写成**单一全局界**。
- 其唯一来源 `.scratch/11-retry-matrix-validity/evidence/README.md:69-71` 的原话是：`Seed variance is cell-dependent (0.0001 to 0.0138) and for C2M_n1 alone exceeds the entire 0.011 bpc spread of the six latent cells. **There is no single global floor: ordering claims need per-cell replicates.**`
- ⇒ 规则**与来源相反**。而 A 轴**每一个**"越界"判定都紧贴这个界：A6 top-k k=32 `−0.0166`（**1.19×**）、A6 长 T T=256 `−0.015752`（1.13×）/ T=512 `−0.015101`（1.08×）、A10 N=8 lru `+0.015914`（1.14×）。**全部落在 1.05–1.19× 区间**。
- 附加缺陷：0.0138 是**单臂**跨种子极差，而 A 轴判的是**两臂之差**；若两臂独立，差值界应是 ≈√2×0.0138 ≈ **0.0195**。按此读法，上列"越界"全部回到带内。且 n=1 ⇒ 本格种子散布**从未被测量**（即"没有 single global floor"这一条正好命中）。
- **最小修**：把 A 轴每条"越界"改标为"贴边、n=1、未过双种子 ⇒ 不可排序"；或按 ticket 11 原意，先测**该格**的跨种子极差再排序。

### ② A6 的长 T 结论：机制错格 + 步数不等 + 坏仪器，三者叠加
- 机制错格：`2622–2629` 全部是 `train_window=T/2`（位置窗口），属 A3；A6 却用它支撑"稀疏的收益要到长上下文才可能出现"（`catalog.yaml:335`）。
- 步数不等：三点 `steps = 24000/12000/6000`（同 token 预算），lr 以 `steps` 为分母（`train_depth_ar.py:201-202`），`2629/analysis.md:279-280` 自认两臂**都未收敛**；A9 自己结论"更弱模型损伤曲线更平"（`2615/analysis.md` §8.3）⇒ **"窗口代价随 T 消失"是欠训练的预期形状**。
- 坏仪器：合成长程腿用 `var_recall` 末位布局，全在 chance，被解释成"§19 局部先验"，而同仓 `2623/analysis.md` §9 已证该布局是**位置外推伪影** ⇒ A6 的长程结论**没有任何合法证据**。
- 附带：`2626/analysis.md` §4 的"实测注意力占比远低于公式 ⇒ 公式高估"是**单位错**（实测 = 单层块/8 层整模型 ≈ 公式/8，我逐点复算）。

### ③ A3/A4 这两个 `judged` 格，没有可独立复核的仓内实验
- A3 `tax/benefit`（+0.058~0.121、76.8→93.3%）与 A4（−0.0068）的证据是 `.scratch/07-x1-x3/evidence/SUMMARY.md`（20 行）与 `attn_dist_telemetry.json`；`A3/experiments/2505-m1-followups/README.md`、`A4/experiments/2507-x1-x3-distance-penalty/README.md` 只是 legacy 回填 stub（无 `results.json`/`run.py`/`analysis.md`）。
- 而且 A4 的"锚通道 93.3%"是**组合**（X3 = 惩罚 + 驱逐 vs X2v2 = 驱逐），未按 SCHEMA §4/C15 登记为组合例外。
- 下游影响：A3 被当作"任何压缩/摘要求设计必须打败的价格上界"（`catalog.yaml:268`），而这个上界的数字来自不可复核的旧 markdown。

---

## 5. 其余具体发现（P2，可核查）

1. **A 轴页头与 A3 verdict 互相矛盾**：`axes/A-mixing/README.md`（由 `catalog.yaml:43` 渲染）"丢前缀在 bpc 上很便宜（+0.058~0.121）" vs A3 `verdict`"并不便宜（+0.212）"（`catalog.yaml:252`）。
2. **6 条 prior-art triage 被注释掉且错位一格**（新发现）：`catalog.yaml:237/270/300/361/396/429` 分别在 A3/A4/A5/A7/A9/A10 的 id 之上，装的却是 A2/A3/A4/A6/A8/A9 的文本 ⇒ 机器不可读 + 归错格；`check_schema.py` 查不到（只看 evidence 路径存在）。
3. **证据锚不全**（`check_schema.py` 也查不到）：A6 只锚 `2614/2616/2622`，bullets 4–10 引用的 `2623–2629` 无锚（`catalog.yaml:352`）；A9 只锚 `.scratch`，两个 in-repo 实验无锚（`catalog.yaml:409`）；A3/A8 同样只锚 `.scratch`。
4. **A5 options YAML 坏**（`catalog.yaml:306` → 生成页把"跨步(k"/"stride s)" 当两个选项）。
5. **A10 三处自相矛盾**：judge 字段（ρ 曲线）≠ 实测量；命题（per-token 吸收率）≠ 被测（驱逐优先级）；"业界无人占位"≠ 自己的 `PRIOR_ART.md` §A10。
6. **A6 top-k 的显著性口径**：`catalog.yaml:337` 用"20/20 批同向，t=−7.7"当真实性证据，而该 t 只量评测批内配对差（不含训练重跑方差）——同类问题（49-REPORT B2）在此格**未修**。
7. **A6 线性注意力"不可行"是工程债**（执行者的二分实验在独立进程跑通 3000 步，`2616/analysis.md` §3 行 5），却被 `catalog.yaml:341` 与 `PRIOR_ART.md` 的 A6 行并列成机制级结论。

---

## 6. 对 49-REPORT 的独立确认/推翻

**确认（我独立核到，非复述）**：`check_schema.py` 不校验 `@commit`（读源码：只 `(ROOT/p).exists()`）；`catalog.yaml` 存在会使页面自相矛盾的重复/坏键（A 段即一例）。
**本轮新增（49-REPORT 未涵盖）**：§4-① 的 0.014 全局界与 ticket 11 原文相反（**并含双臂 √2 问题**）；§4-② 的 A6 机制错格 + steps 不等 + 长程腿伪影；§4-③ 的 A3/A4 无仓内实验；§5 的 prior-art 注释错位、证据锚不全、A5 options 坏、A10 自相矛盾、A6 §4 单位错。
**未复核**：49-REPORT 的 A1/A2/A3/A4/A5（H/M/W 轴）与提交可信度部分——不在本次 A 轴范围。

---

## 7. 一句话给主 Agent
A 轴的**枚举**可以留，**结论**要重做；重做的顺序是"先修仪器（P1-1）→ 再修口径（P1-3）→ 再重判 A6 长 T（P1-2）→ A3/A4 要么自包含复算要么降 reference"；在此之前，**不要把 A3/A4 的数字当价格上界**，也不要把 A6 的"窗口在长上下文不亏"当结论用。
