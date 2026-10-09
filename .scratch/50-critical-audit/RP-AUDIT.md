# R/P 轴批判审计（2026-10-02，独立复核者 fork=false）

> 范围：`axes/R-references/**`（R1–R7，7 格）+ `axes/P-modifiers/**`（P1–P6，6 格）。
> 依据：`AGENTS.md`（含最新「编排模式」节）、`SCHEMA.md`、`catalog.yaml`、`docs/methodology.md`、
> `docs/playbook.md`、`docs/OPEN_PROBLEMS.md`、`docs/REMEDIATION.md`、各实验目录原始 `results.json`/代码，
> 并对 `.scratch/49-adversarial-review/REPORT.md` 做**独立**核对。
> 只读，未改任何实验代码/数字/`catalog.yaml`。逐格详版见各目录 `CRITIQUE.md`。

---

## 0. 一句话总判

**这批研究目前不能直接用**：R 轴 7 格里只有 R1 有本地实验，而 R1 的"部署价签"被**自造口径**反转、把业界标准的"int8 近无损且省一半"改写成"量化反占内存"；R2/R5/R6 是**零数据或错命题的 `reference`**，R4/R7 的"必须打败的对手"价签过期或无对照。P 轴 6 格中 4 格无独立判据或与其它格重复，1 格漏记了本仓自己最相关的实测。

---

## 1. 本轮最严重的三件事

### 严重①：R1 用自写玩具量化器把核心参照线从"省一半"改成"反占内存"（开倒车 + 结论错误）
- **事实**：`tools/evalkit/kv_quant.py` 是自写对称量化器；2026-10-02 的 F1"因果化"把标尺做成**逐值**（`kv_quant.py:127` 对 token 轴 `cummax`，`scale` 形状 `[B,H,T,head_dim]`；`kv_quant.py:33` 自述 int8 **3 B/值**）。据此 `catalog.yaml:1205–1219` 与 `axes/R-references/R1-quantization/README.md` 写下"**量化在任何拓扑上都是内存负收益**"、"旧的『近无损省内存』参照线**作废**"。
- **为什么错（不依赖任何外部数字）**：同一仓里"标尺摊薄"本来就有——per-channel 标尺的总开销是 **T 无关的 12288 B**（`2614/analysis.md:49`；`kv_quant.py:273`），T=256 上只有 48 B/位置 ⇒ int8 ≈ **6192 B/位置 ≈ −49.6%**。F1 把"因果"与"逐值标尺"划了等号，而正确解法是**把分块放到 head_dim 轴上**：同一个 `b_grouped` 算术（`budget_caliber.py`，G=32 ⇒ **+2/32 = 0.0625 B/值**）沿 head_dim 施用时**严格因果、无需回填、无需 staging**。本仓甚至在 `2631/analysis.md:202` 与 `budget_caliber.py::mixed_axis_note` 里**自己写下了**这条（"V 按 token 取 head_dim 维 amax ⇒ 严格因果且 +2/32 B/值"），紧接着断言"**K 侧做不到**"——没有理由，且不成立（profession 标准实现正是按 head_dim 分块，需在可联网时核证 `q8_0`/`q4_0` 的块布局与 KIVI 的 K per-channel / V per-token）。
- **后果**：所有"省内存/丢信息"主张的**入场费**被改成了错误的量。正确参照线应恢复为：**int8 KV ≈ 1.0625 B/值（省 46.9%）且近无损（+0.0001 量级）；int4 ≈ 0.5625（省 71.9%），代价 +0.025~+0.046**（本仓自己在 `2631/analysis.md:176`、`table.md:266` 算出了 46.9%/71.9%）。
- **修法**：见 R1 `CRITIQUE.md` ④⑤（实现 head_dim 块标尺 / KIVI 式，逐值标尺降为对照臂；前缀不变性必须为 True；非量化臂逐位不变）。

### 严重②：R2/R5/R6 三格是"零本地数据 + 记忆当结论"，而 R6 正是决定 W 轴值不值的关键判据（制度违规 + 判据缺失）
- `axes/R-references/PRIOR_ART.md` §0 自述"本环境未注册任何 web / 检索工具 ⇒ **无法在线核证任何外部数字**"，§2 却把 **R1–R7 七格全部判 `(a) 业界已研究透`**，并据此写 `status: reference`。这违反 `SCHEMA.md §21`（`(a)` 只能由可核实出处支撑；检索不可用 ⇒ 只能记 `(c)` 待核），也与本仓 2026-10-02 对 A/W/C 若干格执行的 **C14 更正**自相矛盾（`docs/REMEDIATION.md:60`；A1/A5/A7/C4/W2 已被改判 `(c)`，R/P 未改）。
- **R2/R6**：`local: {}`、**零实验目录**。R6 的 `industry`+PRIOR_ART §4 自认"13M 尺度是否有公开对照——**未找到**"。而 R6（MoE 同 FLOPs 更多参数）**直接决定 W2/W5 的判据线**：没有 13M 点，"保容量减计算是不是正路"在本尺度无判据。
- **R5**：`benefit` 的"本地侧证"用的是 **G5**（`.scratch/08-tts-probe/evidence`），而 G5 的命题已被用户 2026-10-02 明确判为**执行方自设、与用户目标无关**（`docs/OPEN_PROBLEMS.md` P-01；`catalog.yaml:1169–1197` 同步标注）。用 G5 当 best-of-n 的侧证是换命题（`SCHEMA.md §13`）。本仓**不存在 verifier**。
- **修法**：R6 立项重做（13M、同 FLOPs、top-1/top-2、活性量=专家使用率）；R2 删除（本项目无 teacher）；R5 删除或与 G5 解绑后重立。

### 严重③：9/13 格仍挂着已被废除的状态，且 R4 的判据句与本仓自己的 ERRATA 冲突（状态语义 + 数字一致性）
- `AGENTS.md`「编排模式」明写：**没有「部分」和「参照」这两种概念**，每格要么"做完并给出结论"、要么"从树中删除"。而 `SCHEMA.md §1` 的 status 列表仍含 `partial | reference`。本轮范围内 **9 格**踩这条：`R1–R7`（`reference`）+ `P2`/`P4`（`partial`）。
- **R4 的数字冲突**：`catalog.yaml:1255–1262` 写"窗口部分 **+0.058~0.121 bpc**"、"『就用个窗口』的 bpc 代价**几乎为零**"；而 M2 的 ERRATA 已实测 **keep-32 = +0.212**、keep-64 +0.081、keep-128 +0.020、keep-192 +0.004（`catalog.yaml:1052–1066`），M3 实测窗口 32 训练税 **+0.0866**（`catalog.yaml:1074–1086`）。A3 已带 ERRATA，R4 未同步 ⇒ 参照线的判据句是错的（"几乎为零"只在 keep ≥ 128 成立）。
- **修法**：按新状态语义重判全部 13 格（做完给结论 / 删除）；R4 同步累计口径。

---

## 2. 建议删除清单（主 Agent 树管理用）

| 课题 | 理由（证据） |
|---|---|
| **R2-distillation** | 零本地实验、零可核实出处；本项目**无 teacher**（单卡 13M 自训）；不改变任何"路径值不值"的判断（`AGENTS.md`「编排模式」删除判据）。 |
| **R3-pruning-sparsity** | 与 **W5-capacity-vs-compute** 同结论重复；唯一"本地证据"是 **W1 读出切分**（错配命题，`catalog.yaml:1239–1247`）；本格判据"同 FLOPs 质量"一个点都没测。 |
| **R5-bestofn-verifier** | 零本地实验；"侧证"来自被用户判为无关的 **G5**（P-01）；本项目无 verifier ⇒ 无判据可执行。 |
| **P2-conditioning** | `failure: []`、无对照臂、无独立判据（收益借自 C1）；"置信度门控"已属 **G1-selector**；剩余选项（噪声级/任务条件）无任何用例。 |

（可选：**P1** 若主 Agent 认为"修饰符页"不承载研究，也可从"研究格"降为 note——本审计建议**保留**，因为它有防复发的可复现价值。）

---

## 3. 建议重做清单（按优先级）

| # | 课题 | 重做要点 |
|---|---|---|
| 1 | **R1-quantization**（P0） | 实装 **head_dim 块标尺**（G=16/32/64）+ KIVI 式 K per-channel/V per-token；逐值标尺降为对照臂；**前缀不变性必须 True**；横轴改部署口径；重测 27 臂；修正 R1 结论为"int8 ≈ 省 46.9% 且近无损"。 |
| 2 | **R6-moe-scale**（立项） | 13M、**同 FLOPs** 对照 dense、top-1/top-2、2–4 专家；活性量=专家使用率（不坍缩）；成本按实测步时估；给等参数或同 FLOPs 对照（`SCHEMA.md §15`）。 |
| 3 | **R4-window-kv-quant-retrieval** | 同步 M2/M3 的累计口径（keep 32/64/128/192 = +0.212/+0.081/+0.020/+0.004），删"几乎为零"；量化部分随 R1 重测；补**检索腿**（可复用 `longrange_tasks.py` + `retention.py`）；横轴改部署口径。 |
| 4 | **R7-existing-early-exit** | 补一臂**同 FLOPs 的 dense 早退基线**；把"我们的增量在哪"从"已证"降为"假设"。 |
| 5 | **P4-normalization** | 回填 ticket 42-W1/43-V1 的 **QK-Norm** 结果（gnorm 1740.9→7.25、推荐 `qk-normfrom512`、短 T 越界）；明确"无 final norm"是**前提**不是结论；可选做有/无 final norm 的最小消融。 |
| 6 | **P3-param-insertion-init** | 把"族内最优"降为"平局带内方向为正"（0.0226 < 自述 ±0.03 带，`23e-decomposition.json:150–151`）；补 ≥2 seed 或一个 2× 训练点；补 `prior_art_*` 字段。 |
| 7 | **P5-loss-weighting** | 在现引擎重跑降权 sweep + 退火，**带 matched-step/快照口径与 ≥2 seed**；删掉 legacy 1.1-epoch 与新引擎数字的并列。 |
| 8 | **P6-data-construction** | 把 §19 落成**强制协议**（长程类结论必须两套口径、以合成为准）；补"同一权重两口径"成对检验；摄入 42-W2 的极差 10.84pp 警示。 |

---

## 4. 开倒车清单（业界标准是什么 / 我们错在哪）

| 格 | 业界标准 | 我们的偏离 | 证据 |
|---|---|---|---|
| **R1** | KV 量化的标尺按**块/通道摊薄**（每 G 值 1 个标尺，开销 `2/G B/值`、**与 T 无关**），代表实现按 **head_dim** 分块（llama.cpp 类）或 KIVI 的 K per-channel / V per-token。**待核**：需联网核 `q8_0/q4_0` 块布局与 KIVI 细节。 | 改成逐值标尺（`kv_quant.py:127`），并据此宣布"量化反占内存"；把"因果"与"逐值标尺"划等号；只把分块沿 **token** 轴考虑（需回填 ⇒ 认定违反前缀确定）。 | `kv_quant.py:33,127,273–276`；`2631/analysis.md:176,202`；`2631/budget_caliber.py` (`DEFAULT_GROUP`, `mixed_axis_note`)；`2614/analysis.md:49` |
| **R3** | 结构化剪枝收益依任务、需重训；"减结构=减容量" | 把 **W1 的读出切分**当剪枝侧证（错配命题），结论与 W5 重复。 | `catalog.yaml:1239–1247`、`885–895` |
| **R5** | best-of-n + verifier（候选来自**随机采样、前缀不同**） | 用 **G5 的确定性路径集成**当侧证（G5 命题已被用户判为无关）。 | `catalog.yaml:1264–1270`；`docs/OPEN_PROBLEMS.md` P-01 |
| **P1** | 混合精度：fp32 master weights + loss scaling；奖励/对数类损失单独 fp32（Micikevicius 2017 起） | 把该通用纪律当本地研究结论（`status: judged`、无 `prior_art_*`）。 | `catalog.yaml:1308–1316`；`axes/P-modifiers/PRIOR_ART.md` 块① |
| **P3** | 单边零/恒等起步：LoRA `B=0`、ReZero、Fixup、ControlNet zero-conv | 同上（把通用技法当本地结论），且把"族内最优"当序关系（0.0226 在自述平局带内）。 | `axes/P-modifiers/PRIOR_ART.md` 块③；`23e-decomposition.json:150–151` |
| **P2** | FiLM / AdaLN(DiT) | 重复登记 G1 的置信门控；剩余选项无用例。 | `catalog.yaml:1326–1334`；`1087–1103` |

---

## 5. 值得做（真 promising，当前做错/不够）

1. **R1 量化（P0，最该立刻修）**：这是全仓"省内存"判据的基准，错一格带偏 A3/A9/M1/M2/W3/R4 五格。
2. **R6 MoE @13M（立项）**：唯一能决定"保容量减计算在本尺度是否成立"的实验，成本低（25–40 min/臂），现在完全空白。
3. **R4 的检索腿**：右移压缩主线的对手只有窗口一半被测过，检索那一半（RAG/锚点）零经验。
4. **R7 的同 FLOPs 早退基线**：决定 C1 的 +0.4362 形态税是"新问题"还是"早退的共性"。
5. **P4 的 QK-Norm 回填**：本仓已有可用的稳定化实测却不在 P4 页上，属最便宜的修补。

---

## 6. 独立核到的东西（与 `.scratch/49-adversarial-review/REPORT.md` 的关系）

- **确认**（我独立读到原始证据，不是复述）：A2 的非因果 scale 问题确实存在于工具层（`kv_quant.py` 的旧归约），且 2026-10-02 的 F1 已改；但**F1 的修法本身制造了新问题**（逐值标尺 ⇒ 内存负收益），这是上一轮复核**没有**指出的（它只要求"因果/流式"，未审"标尺摊薄"）。
- **推翻/限定**：上一轮把 A2 的后果写成"量化价签只能记为离线扰动实验"——**方向对但诊断不完整**：真正的错误不是"因果化本身"，而是**沿 token 轴分块**这一选择；沿 head_dim 分块可同时满足严格因果与便宜标尺（本仓自己的 `budget_caliber.py` 算术已经给出）。
- **补充新发现**：R4 的判据句与 M2 ERRATA 冲突；P3 的"族内最优"落在自述平局带内；R 轴 PRIOR_ART 的 7 格 `(a)` 在 C14 更正后被漏改；P 轴 6 格的分诊未回写账本。
- **无法核实项**（如实标注）：① `@<sha>` 证据锚点是否为真实提交（本会话无 git 访问，`watchdog_diff` 亦不可用 ⇒ 无法核）；② 所有外部论文/数字（KIVI/llama.cpp 块布局、GQA/蒸馏/MoE 的规模效应）——离线不可核，一律按"待核假设"处理。

---

## 7. 给主 Agent 的最小动作建议

1. 先修 **R1**（把参照线恢复正常），再动其它省内存线，否则它们的判据都在错的天平上。
2. 按第 2 节删除 R2/R3/R5/P2（四格），按第 3 节排 R6 立项 + R4/R7/P4/P3/P5/P6 的补做。
3. 全 R/P 轴按新状态语义重判状态（消除 `reference`/`partial`），并给 R 轴 PRIOR_ART 的 7 格改判 `(c)` 或补可核实出处。
4. 若要真核证业界结论，需要用户提供检索工具（`docs/OPEN_PROBLEMS.md §6` 已挂账）；在提供之前，不得再把 `记忆·未核` 当"免做依据"。
