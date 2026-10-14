<!-- 落盘说明（axis5-lead，ticket 66 Phase B）：第 2 轮复核者（fresh 独立上下文 `reviewer`，未参与第 1 轮、非任何包 writer）所在会话**无写入工具** ⇒ 判词由其返回、axis5-lead **原样落盘**（未改动其结论、未补写内容）。 -->
> 来源：subagent run `92da7b0f-256b-402c-9229-93ff9a91d59d`；落盘时间：2026-10-14（+08:00）。

## 复核报告（第 2 轮 · 轴 5）

**落盘说明**：本会话无写工具 ⇒ 报告全文如下，由 runtime 落盘到 `.scratch/66-axis5-quant/REVIEW-ROUND2.md`；未改任何产物、未 `git add/commit`、未取 GPU 锁。复核者：fresh 独立上下文（未参加第 1 轮、非任何包 writer）。

**基线与限制**：`watchdog_diff` 不含已提交区间，我未核 commit/范围纪律与 mtime（第 1 轮已登记为不可核项，本轮同样不可核）。我核的是**当前磁盘上的文件内容**（各读一次）。

---

### 人话结论

**第 1 轮的 blocker 全部落地。** `5.3` 的三条 blocker（补跑 24×16 单点 / 改 `eval.note` 的不实口径 / 修 `anchor_check_applicable`）与 `5.5` 的 blocker（缺「fp8 vs int8 二选一」明文 + `project.yaml` 未更新）都真修好了，`5.2` 的两条 normal（表格不渲染 / 设备字段不符）也修好了。我**独立复算**了 5.5 的两条差值（`e4m3−int8` max=**0.005363**、`e5m2−int8` max=**0.014496**），与产物一致，也和第 1 轮的 0.005363 一致。**三包的主表数字与判据线（0.014 带、配对 Δ、≥5 窗）全部未被事后改动。** 但返工**引入 2 条新 normal**（都是文本自相矛盾，不动数字）：① `5.5` 的 `project.yaml` 结论句把 `e5m2−int8` 也写成「< 0.014」，实际是 0.014496 **越带**，与同包 `analysis.md §3.1`/`README §6` 直接矛盾；② `5.3` 的 `results.json::caliber_note.open_defect` 仍写「`config.json::eval.note` 未改、交 owner 处置」，而它**已经改了**，成陈旧矛盾。**无新 blocker。**

---

### (A) 逐条清单

| 轮次项 | 判定 | 证据（file:line / 字段） |
|---|---|---|
| **5.3 B1** 补跑 24×16@seed2 fp16 | **已修** | `run_log.txt` `[单点]` 行 `bpc_of 24×16@seed2 = 1.483568730`；`results.json::anchor_single_point.bpc = 1.48356873`、`abs_diff_vs_anchor = 2.700488819229463e-07`；`analysis.md:84-85`、`table.md:66`、`README.md:50/165`；`anchor_single_point.log` 存在。**主表数字未改**（§1：1.474621/1.478151/5.040267/1.477052；配对 0.003530/3.598298/0.002955 = 第 1 轮） |
| **5.3 B2** `eval.note` 三句不实叙述 | **已修** | `config.json:109` = 实跑 3×16/2×16 + 明写「与 24×16 锚**不可直接比**」+「不得称同口径、无批量混杂」；`results.json:113` 副本同；`README.md:115-125`（§6）同 |
| **5.3 B3** `anchor_check_applicable` | **已修** | `results.json:302 = false`；`:304 anchor_check_applicable_rule`「按实跑 `n_batches` 门控（不只看 `--smoke`）」；`:305` 指向单点承载 |
| 5.3 N1「原因已定」 | **已修** | `analysis.md:82-92` 改为实测并列表（3×16 / 24×16 / GPU 锚）+「实测确认」 |
| 5.3 N2 10% 阈值出处 | **已修** | `analysis.md:22`「**10% 是实现层 `run.py` 自设的活性阈值**（未写进 `criterion`）」；§4 同 |
| 5.3 N3 裁剪超授权 | **已修** | `analysis.md §5.3`「**超出** README §7 的预登记裁剪条款范围」+ 影响 + 补齐成本 |
| 5.3 N4 表格不渲染 | **已修** | `table.md:8-13` 行间已无空行，表 1–4 连续可渲染 |
| 5.3 I1（张量集表述） | **未修·已登记** | `analysis.md §0` 略更具体（「每层 4 个 `nn.Linear` 的输入激活」）但未显式排除注意力矩阵/logits；`JOURNAL.md:143` 登记「未动，不在返工契约内」 |
| 5.3 I2（`caliber_note` 手改） | **未修·已登记（但有新矛盾）** | `JOURNAL.md:143-144` 登记「原样保留」；见 (B) 第 2 条 |
| **5.2 N1** `table.md` 不渲染 | **已修** | `table.md:8-12` 连续无空行 |
| **5.2 N2** device 不符 | **已修** | `config.json:93 device_resolved:"cpu"`；`results.json:114`、`:10442 device_override_note`、`:10447 caliber_note`（§20 不适用）；`README.md:134`（§8）、`:167`（§10）加「CPU 口径 ⇒ GPU 墙钟/§20 条款不适用」 |
| **5.2 I1**（`Δ锚` 标签） | **部分修** | `run_log.txt` `[门禁] …G2=True(Δ锚(6位，参5.2 REVIEW I1)=0.00e+00)`：加了指向复核的注记，但**标签仍叫「Δ锚」**（该值实为两 bpc 路径之差，非与锚之差）⇒ 歧义减轻未消除 |
| **5.2 I2**（`dominates` 措辞） | **已修** | `analysis.md:17/19`「不构成严格支配」+ 定义「即质量不可分辨、不构成 Pareto 严格改进」 |
| **5.2 I3**（int4 越带幅度） | **已修** | `analysis.md:13`「超出带 +0.000904；5 窗均值 0.013946 落在带内 ⇒ 依赖 max 聚合」 |
| **5.5 B1** fp8-vs-int8 二选一 + SSOT | **已修（但 SSOT 句写错，见 B）** | `analysis.md §3.1`（:95-119）逐窗表 + 「**二选一明确**：e4m3 不越带/不可分（max 0.005363）、e5m2 越带（0.014496）」；`README.md:75-77` 指针；`project.yaml`：`status: 进行中`、`experiments` 已登记、`conclusion_review` 已填。**我独立复算**：e4m3−int8 = [0.002229,0.002025,0.005363,0.001024,0.001833] max **0.005363**；e5m2−int8 = [0.011607,0.008847,0.008257,0.012915,0.014496] max **0.014496** ⇒ 与第 1 轮/产物一致 |
| **5.5 N3** 越带聚合口径 | **已修** | `config.json:112 band_aggregation`（max 取法 + 披露 e5m2 均值 0.011375）；`analysis.md:115-119`；`README.md:74-75` |
| **5.5 N1/N2** | **未修·已登记** | `JOURNAL.md`「未处置（不在派单范围）：N1、N2（`selftest_fp8.py` 缺运行留证）、I1、I3」；无 `mutation_report.json`（确认缺） |
| 5.5 I2/I4 | **已修** | `analysis.md:29`（去掉「最稳」排序义）、`:115`（措辞） |

**数字/判据线核查（item 8）**：三包 `config.json::criterion` 与 `README` 判据段**均未放宽**（5.2 带 0.014 未变；5.3 把 10% 明确降为「实现层阈值」而非写进 `criterion`；5.5 仅**新增** `band_aggregation` 声明键，预登记数组未改）。三包原有臂 bpc 与配对 Δ 与第 1 轮报告**逐项一致**，无数字被改。

---

### (B) 返工引入的新缺陷

- **new-normal-1（`5.5`）**：`project.yaml::conclusion` 写「`fp8_e4m3 − int8_block_g32` 与 `fp8_e5m2 − int8_block_g32` 的 `max|配对 Δ|` **均 < 0.014** ⇒ 字节更省但质量不可分辨」。**证据**：我由 `results.json::paired_delta.per_arm` 复算 e5m2−int8 max = **0.014496 > 0.014**（= `analysis.md §3.1` 的 `:110-111` 与 `README.md:77` 的读数）。**该句与同包 `analysis.md §3.1`/`README §6` 直接矛盾**（二者都写 e5m2 越带），也与 `project.yaml` 自己前半句「E5M2 越带（vs fp16）」不一致。**最小修法**：把该句改为「`fp8_e4m3 − int8_block_g32` max|Δ| = 0.005363 < 0.014（不越带、不可分）；`fp8_e5m2 − int8_block_g32` = 0.014496 > 0.014（越带）」。仅改文本，不动数字/结论。
- **new-normal-2（`5.3`）**：`results.json::caliber_note.open_defect`（`:8281`）仍断言「`config.json::eval.note` … 仍写『主表 … 24×16』…属陈旧文字。本次**未改**，交 5.3 owner 处置」。**证据**：该 note 已在返工中改为实跑口径（`config.json:109`）⇒ 该 `open_defect` 现在是**假陈述**，且与同一 `results.json` 的新 `anchor_single_point` 语义相反。**最小修法**：删除或改写 `caliber_note.open_defect` 为「已于本轮修复」，或把整段 `caliber_note` 迁出生成物。
- **info（`5.2`）**：I1 只做「部分修」（`run_log` 标签仍叫 `Δ锚`）。

无新 blocker。三包也未出现新的数字改动或不实结论。

---

### VERDICT

- **`5.2`（`2651-weight-quant-13m`）：`PASS`** — 两条 normal 全部落地，info 已处置（I1 部分修，非阻塞）。
- **`5.3`（`2652-act-quant-13m`）：`PASS`**（第 1 轮 3 条 blocker 全部落地，主表数字与判据线未动）；**附 1 条新 normal**：`results.json::caliber_note.open_defect` 成陈旧矛盾，建议顺手改/删。
- **`5.5`（`2653-fp8-vs-int8-13m`）：`PASS`**（blocker 两条落地，二选一已明文且复算一致）；**附 1 条新 normal**：`project.yaml::conclusion` 的 e5m2−int8「均 < 0.014」写错（应为 0.014496 越带），建议改这一句。

> 说明：按派单判定规则（第 1 轮 blocker 全落地且无新 blocker ⇒ PASS），三包均给 PASS；上述两条新 normal 是**文本自相矛盾**，建议在收口前各改一句（无需重跑实验）。

---

**返回摘要（≤12 行）**
1. `5.2` = PASS；`5.3` = PASS；`5.5` = PASS。
2. (A) 第 1 轮 blocker：`5.3` B1/B2/B3 **全部落地**；`5.5` B1 **全部落地**（含 `project.yaml` 状态/experiments/`conclusion_review`）；`5.2` 无 blocker。
3. (A) normal/info：`5.2` N1/N2 落地、I1 部分修、I2/I3 落地；`5.3` N1–N4 落地、I1/I2 未修但已登记；`5.5` N3 落地、N1/N2 未修但已登记。
4. (B) 新缺陷 2 条，均 normal：`5.5` `project.yaml::conclusion` 把 e5m2−int8 误写「< 0.014」（实为 0.014496 越带）；`5.3` `results.json::caliber_note.open_defect` 陈旧矛盾（称 `eval.note` 未改，实已改）。新 blocker **0** 条。
5. 独立复算：`e4m3−int8` max = 0.005363（同第 1 轮）；`e5m2−int8` max = 0.014496（writer 值正确）。
6. 未落地项：`5.3` I1/I2、`5.5` N1/N2（均已如实登记为返工范围外）；`5.2` I1 部分修。
7. 数字/判据线：三包**均未被事后改动**（criterion 未放宽，原有臂 bpc/配对 Δ 与第 1 轮一致）。
