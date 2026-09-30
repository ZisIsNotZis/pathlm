# 独立复核（契约 P，fresh context，只读）

# 契约 P · 独立复核报告（只读，fresh context）

复核对象：`.scratch/32-batch5/issues/01-batch5.md` 契约 M / N / O
复核方式：对照契约原文逐字比对 + 三份 `results.json`/日志/训练日志/仪器源码/单测文件逐一核对（无 shell，pytest 与 check_schema 未跑，见末节）

---

## Review

### Correct（已核对正确的部分）

**判据线实质未被改**
- M：README §2/§3/§4 与契约「动机/方法/判据」逐字一致（唯一差异为空白：`>30pp` vs `> 30pp`），符号与口径只在「操作性定义」里落定，未动阈值。
- O：README §3「干预」、§4「判据 ①–③」与契约逐字一致（含「d ∈ [8,32]」「阈值 0.014」「参照 X3 假保留陷阱」）。
- N：阈值（0.014 / 10pp）、析取结构（`lru 或 content`）、两条腿、必须报分布与重叠度**全部保留**，功能性改动 = 0（标注问题见 P2-1）。

**⑦ 回归面（可检查部分）**：tracked 改动只有 5 个文件——`pathlm/depth_ar.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`、`tools/evalkit/retention.py`、`tests/test_retention.py`；`catalog.yaml`、`INDEX.md`、`axes/**/README.md` **均未被碰**（`watchdog_diff --stat`），文件所有权合规。权重 `*.pt` 与 `data/` 未入库（新增目录全部以未跟踪状态存在，`train/window32/model.pt` 仅落 `MODEL.md` 哈希+体积）。

**③ 对照臂三要素与同一任务定义**：三线均写明「输入分布 / 要预测什么 / 部署对应」（M §5、N §6.2、O §6.2），且依赖腿与保留腿共用同批窗口、同批针、同位置集、同读出（M 无针地板 `plant=False` 同 generator 状态、`meta` 逐字段断言）。

**④ 机制活性**：O 训练日志每条含 `window_mask`（`train/window32/train_log.jsonl` 61 行，step 0/200/…/11999，`visible_frac=0.287573`、功能探针 `max|Δlogit|=1.011293`）；M/N 无训练步，逐配置活性量（`n_dropped_pairs`/`dropped_frac`/分桶）全量落 `retention_log.jsonl`。

**⑥ 口径标注**：三线各有专节（M §9、N §6.3+§8+§11、O §6.3），native/release 双口径、qmin128/qmin200、快照口径都写了。

**a) 两条腿分开、判据不混用**：pass。M 用 `|Δbpc|` 定低损伤、用 `全保留基线召回 − 配置召回` 定保留腿崩掉，判据为合取、逐口径扫描；N 分别给 `passes_dependency_leg` / `passes_recall_leg` 再按契约析取裁决；O ①=质量、②=依赖距离曲线、③=同掩码无针地板的召回差。三线均未把两腿合并成单一分数。

**b) 三种优先级删除预算真的相等**：pass。`retention.py::_rank` 每窗口严格取 `order[:n_drop]`；`results.json::criteria.per_N.*.same_budget_n_dropped_keys` = 8/32/64（三规则相同）；`retention_log.jsonl` 末行 `n_dropped_keys_mean` 恰为 8.0/32.0/64.0（384 窗口）。删掉的**可见性对数**确实不同（N=8：2012 / 1561.85 / 1859.43），已如实另报（这是本实验的口径发现，不是预算不等）。

**c) `window32` 训练掩码 == 评测掩码**：pass。`results.json::instrument.train_mask_equals_eval_mask = true`、`train_vs_eval_mask_max_abs_diff = 0.0`；`run.py:175-185` 用 `masked_logits(model, native)` 与 `model(probe_x)` 逐位比对，不等即 `SystemExit`；训练侧 `pathlm/depth_ar.py::_window_mask` 与评测侧共用 `window_anchor_keep_mask`。掩码计数算术自洽：9460/32896 = 0.287573。

**d) needle 基线召回达标**：pass。三线全保留基线 **0.686768 ≥ 0.50**（M/N/O 逐位相同，同 seed=3 构造序列）；地板 0.433105；契约指定的 v2 构造先测出 0.015079（n=1260）后按契约「<50% 先修构造」改用 v3，过程落盘。

**f) 复用基线配置等值性**：pass。O 给 13 行逐字段对照（唯一不等字段 = `train_window/train_anchors`），并以 `reconciliation_vs_2614`（9 距离点 + 7 前缀点 + 基线）**max|Δ| = 0.0** 复现 2614；N 给对 M 的逐字段表并在同进程重算 `prefix:N=8/32/64`（max 2.59e-07）；M 与 2614 `prefix` 曲线逐点相同（同进程重算）。补记：M 的产物 code anchor 指向旧版 `retention.py`（`f1c6a8f3`），其数字由 N/O 用新版（`8320f03a`）重算证明等价（≤2.6e-7），链条闭合并有证据，但 M 自己的 `results.json` 不含该交叉验证字段。

**⑤ 抽查 ≥8（成本/次数/等式，全部一一对应）**
M：0.686768/0.433105/0.253663（差=0.253663 ✓）；`recent:d=32` recall 0.343018、drop 0.343750；低损伤最大降 0.103272、配对 0.214876；prefix N=8 = 0.196528（= 1.680097−1.483569）；1425 次/39 配置/56.98 s/142.7 MiB；σ 0.001401/0.004157、0.025788/0.035066；82 行日志。
N：2012/1561.85/1859.43（对数）、8/32/64（位置）；+0.015914/+0.024363/−0.055697/−0.020782/+0.037807；10.13pp/…/−1.61pp；单位代价 0.0836–0.0977 / 0.0864–0.0926 / 0.1041–0.1156（逐点复算通过）；p25 0.7366813→9.37433/0.73668=12.7×、`zero_frac=0.003906`；4444×22.854 ms = 101.56 s ≈ 墙钟 101.57 s ✓；92 = 9+1+14+56+12 ✓。
O：1.483569−1.570135 = −0.086566；1.483569−1.707367 = −0.223798；gap 均值 (−0.01307+0.009413+0.006013)/3 = 0.000785、工作点 (0.029867+0.006627+0.0034)/3 = 0.013298；beyond 0.458773−0.454494 = 0.004279（n=3505）；0.061333×2250 = 138、0.037134×1912 = 71；0.011778/0.011761/0.01185/0.012772；训练 7.62 min + 复现 8.27 min + 0.699 + 0.464 ≈ 17.0 min；rep 27.81 s/396 次。

### Finding

**P1-1 · M：一句「结构上不可达」与本实验自己的数据矛盾（算术不自洽）**
- 位置：`2615-retention-retrieval-leg-13m/analysis.md:40`（人话摘要表）、`:291-295`（§4 原因①）；复述于 `README.md:133`、`stability.md:29`。
- 断言：「仪器动态范围（基线 − 地板）= 25.4pp < 30pp ⇒ 结构上就达不到」「哪怕把针全删光，召回也只能掉 25pp」。
- 反证（同一产物内）：`results.json::criteria.main_delta_bpc_full_query.max_recall_drop_overall = 0.453125`、`…_no_query_block_overlap = 0.34375`、`main_on_paired_deleted_subset… = 0.590494`；`recent:d=32` 召回 0.343018（基线 0.686768，**下降 34.38pp**，`overlaps_query_block = false`，删针 3505/4096）。即：无损上下文里的「记忆净贡献」是 25.4pp，但**删除本身会把召回压到无针地板以下**，故 30pp 并非结构性不可能。主判据未达成的真实原因是 §4 原因②（低损伤配置删针太少），该条成立。
- 最小修法：删掉「结构上不可能/只能掉 25pp」两句，改为「未受损上下文下记忆净贡献 25.4pp；非重叠配置实测最大下降 34.4pp（配对口径 59.0pp），故 30pp 可及，未达成的结构原因是低损伤配置只删掉 6–19% 的针」；`README.md:133` 与 `stability.md:29` 同步改。

**P1-2 · M：报告中有一批数字在本实验产物里查无出处**
- 位置：`analysis.md:105-107`（induction 0.116/0.108/0.107/0.098/0.091、任意复制 0.058、自然续写 0.700）、`analysis.md:440`（「0.359 vs 0.411」）；复述于 `README.md:47-48`、`MODEL.md:17`（10% / 6%）、`analysis.md:62,295,436`。
- 证据：全目录检索（`results.json`、`run_log.txt`、`config.json`、`stability.md`、`smoke/*`、`mutation_report.json`）无这些字段；`run.py` 只实现 `v2_construction_diagnosis`；`run_log.txt` 的构造诊断只有 `v2 基线召回 = 0.0151`。0.359375/0.411765 在目录里只以 `recall_no_needle`(batch 1) 与 anchor 桶值出现，语义不同。
- 影响：本批硬性纪律「报告里的每个成本/次数/等式都要与 `results.json` 或日志逐一对得上」，且这些数字正被用来支撑「主干没有任意内容复制能力」这一关键论证。
- 最小修法：把探针落成 `results.json` 字段（n/seed/命令）或从三份文档删除，只保留有据的 `v2=0.015079`（`results.json::v2_construction_diagnosis`，n=1260）与 `v3=0.686768`。

**P2-1 · N：「逐字」标注不实（判据未被改）**：`README.md:4,7,§4`、`config.json::criteria.note`、`results.json::criteria_verbatim` 称逐字，但抄录被压缩——缺「相对 `distance`」、`（用户设计的前提成立，值得往下做）`→`（前提成立）`、`< 上述阈值`→`低于阈值`、`不带来可见收益`→`无可见收益`、去「（哪些位置被读得多）」。阈值与析取完整、操作定义补全了被删字句 ⇒ 不构成判据改动，但违反 SCHEMA §17.6 的标注要求。最小修法：改标「摘要（阈值/析取逐字）」或补全原文。

**P2-2 · N：单测条数与文件不符**：`README.md:7,132` 写「既有 16 条 + 新增 9 条」（=25），`tests/test_retention.py` 实含 **27** 个 `def test_`（11 基线 + 5 M + 11 N）。最小修法：改 9→11（或重算总数）。

**P2-3 · N：误引 M 的噪声底**：`stability.md §5` 引「M 跨窗口 seed σ 0.0041–0.0185」；M 的产物是 0.0014–0.0042（`M/results.json::noise_floor.dependency_damage.max_std_damage_bpc = 0.004157`，M README §6 同）。最小修法：改成 M 的实际区间。

**P2-4 · N：核对声称无落盘证据**：`README.md §12` 称「引用的 278 个数字已用 `reconcile_numbers.py` 逐条核对通过」，但该脚本只 `print`（`reconcile_numbers.py:225`），目录内无报告文件（对照 O 有 `reconcile_report.json`）。最小修法：落一份报告 JSON（我已手工抽查的项全部对得上）。

**P2-5 · N：「按内容…算术不可达」用了非严格上界**：`analysis.md §6.1` 以「全保留召回 0.6868 = 可达上界」推出召回腿 10pp 不可达；该上界非严格（子桶已见删除后召回**升高**：`recent:d=32` window 桶 0.7952 vs 全保留 0.6945）。结论（N∈{8,32,64} 上 +10pp 不可达）仍稳健，措辞应改成经验上界。

**P2-6 · O：对账条数与报告不符**：`stability.md:4` 与 `MODEL.md:75` 写「105/105 通过」，落盘 `reconcile_report.json = {n_checks: 121, n_failed: 0}`。最小修法：改 121/121。

**P2-7 · M：一个计数与限定语不自洽**：`analysis.md §3.3` 注「32 个含 beyond 桶的**非覆盖**配置」；非覆盖配置共 31 个，其中含 beyond 桶者 **28** 个（含 beyond 桶的配置总数才是 32，其中 4 个是 ⚠️ 覆盖查询块）。报告的区间 0.3332–0.5556 与那 28 个一致。最小修法：28（或去掉「非覆盖」）。

**P2-8 · M：逐批活性量的落盘范围小于其描述**：`README.md §7`/`analysis.md §7` 称「逐批判定的针命中与分桶，每批量即写」，实际 `retention_log.jsonl` 前 43 行只对基线全保留与无针地板逐批（10 行 v2 + 1 行 needle_start + 32 行 needle_batch），39 个配置各只有 1 行汇总。活性量本身完整（每配置 `n_dropped_pairs`/分桶），故只是描述偏宽。

### 可疑数字清单（逐条）

| # | 数字 | 出处 | 状态 |
|---|---|---|---|
| 1 | 「动态范围 25.4pp ⇒ 30pp 结构上不可达」 | M analysis:40,291-295 / README:133 / stability:29 | **与自身 results.json 矛盾**（0.453125 / 0.34375 / 0.590494）→ P1-1 |
| 2 | 0.116/0.108/0.107/0.098/0.091、0.058、0.700、0.359 / 0.411 | M analysis:105-107,440；README:47-48；MODEL:17 | **无产物出处** → P1-2 |
| 3 | 「32 个含 beyond 桶的非覆盖配置」 | M analysis §3.3 注 | 应为 28 → P2-7 |
| 4 | 「M 跨窗口 seed σ 0.0041–0.0185」 | N stability §5 | M 实为 0.0014–0.0042 → P2-3 |
| 5 | 「既有 16 条 + 新增 9 条」 | N README:7,132 | 实际 27 函数（16+11）→ P2-2 |
| 6 | 「105/105 通过」 | O stability:4 / MODEL:75 | 实为 121 条 → P2-6 |
| 7 | 「278 个数字已核对通过」 | N README §12 | 无落盘报告 → P2-4 |
| 8 | 「上界 = 全保留召回 0.6868」 | N analysis §6.1 | 非严格上界（子桶反例）→ P2-5 |

其余抽查数字（成本、次数、等式、复现性、噪声底、分桶计数、闭式删对数 d=32→25200、9460/32896=0.287573、D=0.253663、4444×22.854≈101.57 s 等）逐一对得上，**无未标注的外推**：N 的对数重对齐在 8 点曲线上插值，超出区间者标 `above_range_clamped`/`below_range_clamped`，属插值+标记而非静默外推。

### 各线结论（供主 Agent 定状态灯）

- **契约 M**：主判据 **未达成（中性）**——仪器成立（基线召回 0.686768 ≥ 0.50；动态范围 25.4pp；被删针落到地板、可达针不动；同 seed 逐位复现），39 配置两腿同向（Pearson r = −0.47）。附带跨实验 ERRATA（2614「安全半径 d\*=32」不可加，累计代价 +0.212 = 15×）。**Merge = BLOCK**（2 × P1 均为文本级修法，不需重跑）。
- **契约 N**：契约**字面**裁决 `priority_rules_informative`（N=8：lru +0.015914、content +0.024363 ≥ 0.014）；按被删可见性对数重对齐后撤回（lru 转 −0.0170、content 0.0132/0.0092 < 0.014）⇒ **净结论：优先级规则无可见收益，LRU 前提在该规模未获支持**；两腿未混用、预算逐窗口相等、10/10 变异抓红。**Merge = OK with notes**（P2 标注/计数类）。
- **契约 O**：判据 ①②③ **全部未过**（税 −0.086566 的 6×阈值、gap 均值 +0.0008/+0.0133、窗外针 +0.0043 vs 同掩码地板）⇒「是丢了不是搬了」；负结果已同配置 12k 重跑复现（税 0.0821、gap +0.0014、beyond −0.0046）；每 200 步活性量、训练=评测掩码逐位一致均有据。**Merge = OK with notes**（P2-6 计数）。
- 三线共同的**未完成/留给主 Agent**：`pytest tests/ -q`（本批应 200+5+11+8 = 224 绿；注意 N 的 README 把它写成 25）与 `tools/check_schema.py`（新增目录未刷索引而红属预期）我无法运行（无 shell）；M 的 2614/M2「安全半径」ERRATA 需主 Agent 决定是否改 catalog/M2 索引文字。

---

```
## 逐条 pass/fail（§14 七项 + 本轮 a–f）
① 判据线是否被改：M pass（逐字一致）／N fail-标注（功能性未改，标「逐字」不实，P2-1）／O pass
② 三道对账：M fail（做了但漏掉 P1-1 的自相矛盾）／N pass（含 ERRATA 落盘）／O pass
③ 对照臂三要素+同一任务定义：M/N/O 全 pass
④ 机制活性：O pass（每 200 步）／M/N pass（无训练步，逐配置活性全量；M 逐批范围偏窄 P2-8）
⑤ analysis 数字 ↔ results.json：抽查 20+ 项全对；但 M 的探针数字无出处（P1-2）、O 105/121、N 9/11 → **部分 fail**
⑥ 口径标注：M/N/O 全 pass
⑦ pytest / check_schema：**未跑（无 shell）**，留主 Agent
a 两条腿分开：pass  b 删除预算相等：pass  c 训练掩码=评测掩码：pass
d needle 基线达标：pass  e 外推/不自洽数字：**fail（P1-1/P1-2，P2-5）**  f 复用等值性逐字段：pass
```

<contact_supervisor 已按契约 P 要求回传完整报告：reason=progress_update，内容含上述 P1/P2 清单与各线 Merge 裁定。>