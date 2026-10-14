<!-- 落盘说明（axis5-lead，ticket 66 Phase B）：复核者（fresh 独立上下文 `reviewer`，非本包 writer）所在会话**无写入工具** ⇒ 判词由其返回、axis5-lead **原样落盘**（未改动其结论、未补写内容）。 -->
> 来源：subagent run `9ccd56c9-0983-4c25-a52e-4f107395d691`；落盘时间：2026-10-14（+08:00）。内容为复核者返回原文。

## 复核报告 · 轴 5 研究项 `5.5 fp8 / 大 head_dim`（ticket 66 · Phase B 独立复核）

**报告落盘**：`.scratch/66-axis5-quant/REVIEW-5.5.md`（本会话无写工具 ⇒ 全文如下，由 runtime 落盘；未改任何产物、未执行 GPU/锁、未动 git）
**复核对象**：`axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m/**` + `axes/5-quant/5.5-fp8-head-dim/project.yaml`
**派单模板**：`.scratch/66-axis5-quant/REVIEW-CONTRACT-TEMPLATE.md`（占位已按派单填好：研究项 = 5.5，报告路径 = `.scratch/66-axis5-quant/REVIEW-5.5.md`，实验目录 = 上式）

---

### 0. 人话结论（≤10 行）

1. **VERDICT：`OK with notes`**（blocker **1** / normal **3** / info **4**）。
2. 数据面很干净：7 个臂全部前缀不变性 True/True；`int8 G=32 = 6528 B/位置`、`fp16 = 12288 B/位置` 与 `5.1/2632` 逐项对上（`fp8_arms.py:238` 用断言硬钉）；fp8 的 `scale_mode="none"` 确实**没有标尺、没有隐藏 clip/scale**，实测 `n_scales = 0`；字节账每个因子都能由 `fp8_arms.py` 反推。
3. 最重要的一条：**判据要求的「fp8 相对 int8+块标尺 越/不越带，二选一明确」在产物里找不到**（`analysis.md:62` 的「越带」列是 **vs fp16**，不是 vs int8）；我按 `results.json` 的 5 窗口配对读数自己补算：`fp8_e4m3 − int8_block_g32` 的 max|Δ| = **0.005363 < 0.014 ⇒ 不可分**，但这句话必须由产物自己写出来（`README.md:67`、`project.yaml:36` 都明写要它）。
4. 第二条：**milestone 验收未闭 ⇒ 不得判 `已完成`**。`project.yaml` 现在还是 `status: 未开始`（`:61`）+ `experiments: []`（`:19`）+ `conclusion: ""`（`:17`），SSOT 与磁盘产物矛盾；`head_dim=64` 实测腿缺位（降级**如实**，`wp2_hd64_status.json` 与 `TICKET.md` §19.3 一致）。
5. 必须修：① 在 `analysis.md`/`README` 补一行 fp8-vs-int8 的带判定；② `project.yaml` 更新为 **`进行中`**（WP1 已判、WP2 实测腿缺位）+ 登记 `experiments` + 填本轮 `conclusion_review`。
6. 我建议 status：**`进行中`**（不是 `已完成`，也不是 `未开始`）。
7. 范围纪律与修订绑定 **我没能核**（无 exec；`watchdog_diff` 报 “Reviewer-launch HEAD changed or is unavailable”）。
8. 结论强度上限：**单个 13M checkpoint（seed 0/12k 步）、CPU、T=256、质量差在 0.014 带内；字节账是解析价签、不是实测峰值**；`head_dim=64` 只有解析。

---

### 1. 逐条判定（模板 9 条 + 派单必核 6 条）

| # | 检查项 | 判定 | 证据（file:line / results.json 字段） |
|---|---|---|---|
| 1 | 预登记与事后调整 | **PASS** | 判据线逐字同源于 `config.json::criterion`（`README.md:63-80` ↔ `config.json:104-116`）；`README.md` 明写 config 的 mtime 必须早于 `results.json`（`README.md:5`）。我**无法核 mtime**（无 exec）⇒ 该项按文本一致性判，mtime 顺序未核。未发现跑后改写判据的迹象（`results.json` 内嵌的 `criterion` 与 config 逐字相同）。 |
| 2 | 单因素与实现等值 | **PASS** | `instrument.fp32_forward_eq_2619 = true`、`int8_block_eq_2632 = true`、`bpc_path.bitwise_equal = true`（`bpc_of = masked_bpc = 1.470954`）；`run.py:407-409` 三条 `assert` 会对不相等直接崩（真门禁，不是恒真）。唯一改动 = 数值格式；活性列非空（`changed_frac` 0.97–1.0、`rel_err` 1.8e-4…4.5e-2）。 |
| 3 | 字节账与实现一一对应 | **PASS**（含下条 1 的必核） | 见 §2.1。 |
| 4 | 判据带适用性 | **不通过（缺一条明文判定）** | 见 blocker B1。带口径本身成立：5 窗 {2,3,4,5,6} × `n_batches=2` × bs16 × 同 CPU，`paired_delta.per_arm[*].n_windows = 5`、`ranking_band.standard_met = true`，与 config 声明逐字对上；无单点外推（`wp2_head_dim_64.empirical = null`）。 |
| 5 | 否证路径 | **基本 PASS** | 三条否证条件在 `analysis.md:200-206` 逐条回答：否证① 命中 E5M2、否证② 未命中、否证③ 仅在 head_dim=32 上未命中且如实标「head_dim=64 未测 ⇒ 不判」。不利结果如实报（E5M2 越带、两召回腿落地板、`at_floor` 未被写成「机制无效」）。见 normal N1（否证规则的字面触发未按字面宣布）。 |
| 6 | 成本口径 | **PASS** | `cost_model`（`project.yaml:30`）只有等效理论量，无墙钟/吞吐/日期；`validation`（`:28`）明写「**墙钟不进判据**」；`objective_criteria`（`:29`）无数值时间量。`results.json::cost` 的 `wall_minutes=5.695` 是 run 记账（§3 溯源元数据），非判据，允许。 |
| 7 | 结论强度 | **PASS with notes** | `analysis.md` 的表/对账/局限与 `results.json` 逐条一致（bpc、配对 Δ、Wilson CI、`at_floor`、字节数我都抽核过）。见 info I2/I4 两处措辞超范围。 |
| 8 | 范围纪律 | **未能核**（工具缺失） | 无 `git status --short` / `git diff --stat`；`watchdog_diff` 返回 “Reviewer-launch HEAD changed or is unavailable. Committed changes are unsupported”。只能确认实验目录自身只含约定产物（无 `smoke/` 残留、未发现对 `tools/**`、`pathlm/**`、`tests/**`、`docs/**` 的写入证据）。**committed 区间不在我的视野**。 |
| 9 | §20 记账 | **PASS**（CPU 口径适配正确） | `results.json::gpu_lock = null` + `gpu_lock_note`（`config.json::device.reason`）解释「纯 CPU，§20 不适用」；`cost.peak_mem_mib_this_run = null` 并注明「无显存足迹」⇒ 是**声明式缺失**，不是静默省略。无「整程持锁 >10 min」问题（全程未取锁）。 |
| 必核 1 | 同字节轴与 `5.1` 对账 | **PASS** | 见 §2.1。 |
| 必核 2 | fp8 数值格式 | **PASS**（运行面未核） | 见 §2.2。 |
| 必核 3 | 配对 Δ 与带 | **部分不通过** | 窗口数/口径全部对上（逐 arm 5 条 `bpc_paired_windows`）；但「fp8 vs int8 二选一」缺明文（B1）；E5M2 的「越带」依赖未声明的 max 聚合（N3）。 |
| 必核 4 | 前缀不变性两条 | **PASS** | 见 §2.4。 |
| 必核 5 | WP2 降级是否诚实 | **PASS（诚实）；但 status 不成立** | 见 §2.5。 |
| 必核 6 | 成本口径与范围纪律 | 成本 **PASS** / 范围 **未核** | 见上表 6、8。 |

---

### 2. 必核 6 条的独立结论

#### 2.1 同字节轴与 `5.1` 对账 —— 一致，无口径混用
- `int8_block_g32`：`results.json::deployment_ledger.per_arm.int8_block_g32 = {value 1.0 + scale 0.0625 = 1.0625 B/值，6528 B/位置}`，`values_per_position = 6144`，`fp16 = 12288`。逐项因子由 `fp8_arms.py` 反推成立：`deployment_ledger`（`fp8_arms.py:238-...`）先断言 `6144 / 12288 / 6528`（对不上直接 `raise ValueError`），`selftest_fp8.py::test_deployment_ledger_reconciles_5_1` 再钉 `values × bytes_per_value = bytes_per_position` 与 `value + scale = bytes_per_value`。
- 与 `5.1` 的独立对账（我亲自看的**另一份**产物）：`axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/results.json:3610-3611` 中 `deployment_bytes_per_position: 6528.0`、`fp16_deployment_bytes_per_position: 12288.0`，`:3638` 记 fp8「1 B/值 …（质量未测）」⇒ 6528/12288/6144 三个锚在 `5.1` 侧真实存在，不是本包自造。
- **fp8 真免标尺吗**：`fp8_arms.py:71` `if scale_mode == "none": return x.to(dt).to(x.dtype), None, n_sat` —— 无 scale 张量、无乘除、无 clip；实测 `activity.n_scales = 0`（e4m3/e5m2 两臂）；字节账 `value 1.0 + scale 0.0 = 1.0 B/值 = 6144 B/位置`。**没有「带了标尺却不算进字节账」**。
- **有没有仅值/部署口径混用**：没有。`config.json::arms[*].value_bytes` 是**仅值**口径（int8 写 1.0），但 `deployment_ledger` 不读这个字段、自己按 mode 重算并分列 `value_bytes_per_value` / `scale_bytes_per_value` / `bytes_per_value`（resolved 表 3 列齐全）。`analysis.md:28` 的「再省 384 B/位置（−5.9%）」也算对（6528−6144=384；384/6528=5.88%）。

#### 2.2 fp8 数值格式 —— 真用了 E4M3/E5M2，`饱和=0.0` 是弱但可失败的读数
- 真格式：`_FP8_DTYPE = {torch.float8_e4m3fn, torch.float8_e5m2}`；上限由 `torch.finfo(...).max` 取（448 / 57344），`selftest_fp8.py` 把它钉死（`FP8_MAX["fp8_e4m3"] == 448.0`、`fp8_e5m2 == 57344.0`）。
- 舍入/溢出语义：声明为「CPU cast，round-to-nearest，E4M3 超范围**饱和**到 ±448、E5M2 超范围**溢出为 inf**」（`fp8_arms.py:41-43`），并有 3 个针对性单测（`test_e4m3_saturates_at_448_and_e5m2_at_57344`、`test_over_range_semantics_differ_between_e4m3_and_e5m2`（含 600→640 的取整点）、`test_e4m3_flushes_subnormals_below_2_pow_minus_9`）。**我没能运行它们**（无 exec）⇒ 「真的这么取整」我按源码 + 断言结构判，未按运行结果判。
- `饱和=0.0` 是什么意思、能否失败：`n_saturated` 数的是 **cast 之前** `|x| > 格式 max` 的个数（`fp8_arms.py:70`），所以它是**可失败的真读数**（若 KV 值 >448 就会 >0）。本数据最大绝对误差量级 0.4998/0.9999 ⇒ KV 值量级 ~8–16，离 448 很远 ⇒ 0.0 是「本数据从未触界」，**不是恒真式**，但对「免标尺前提」它只是**弱证据**：真正支撑免标尺的是质量不越带（E4M3 配对 Δ max 0.005415 < 0.014）。`analysis.md:82` 自己也把话说对了：「不是饱和问题而是尾数分辨率问题」。
- 有没有偷偷 clip/scale：没有（`scale_mode="none"` 只做 cast；`n_saturated` 只是计数，不参与数值）。E5M2 的 inf 风险在 `analysis.md:187` 如实登记「本数据未触发，换模型/尺度时是真的」。

#### 2.3 配对 Δ 与带 —— 口径成立，但缺「fp8 vs int8」的明文判定
- 成立部分：`config.json::eval` 声明 5 窗 {2,3,4,5,6}、`pairing_n_batches=2`、bs16、`device=cpu`；`results.json::paired_delta.caliber` = 「同窗口 seed、同 n_batches=2、同 batch_size=16、同设备（CPU）」，每个臂 `delta_by_window` 5 条、`n_windows=5`，与 `config.json::criterion.band_primary_caliber` 的适用前提（配对 Δ、≥5 窗口）逐字对上。**没有单点外推**（`head_dim=64` 只给解析，`results.json::wp2_head_dim_64.empirical = null`）。
- 缺的部分（blocker B1）：判据要的是「**fp8 相对 int8+块标尺** 有/无越带差异，二选一明确」，而产物只给了「臂 vs **fp16**」（`analysis.md:62` 表头「越 0.014 带」的定义在 `render_table`（`run.py:162`）里就是 `paired_delta`，其 base 是 fp16）。我用 `results.json` 的两臂 5 窗口读数自己补算（这是我**推导**的值，不是产物里的数）：
  - `fp8_e4m3 − int8_block_g32` = +0.002229 / +0.002025 / +0.005363 / +0.001024 / +0.001833 ⇒ **max|Δ| = 0.005363 < 0.014 ⇒ 不可分**（主表口径 Δ = 1.486669−1.483681 = 0.002988，同向）。
  - `fp8_e4m3_tokscale − int8_block_g32` ⇒ max|Δ| = 0.004532 ⇒ 同字节（6528 B/位置）下**不可分**；`fp8_e4m3_tokscale − fp8_e4m3` ⇒ max|Δ| = 0.004657，符号在 5 个窗口里翻转 ⇒ 「加标尺无收益」这个结论方向正确。
  ⇒ 二选一的正确答案是「**无越带差异（不可分），因此按每位置字节取 fp8 E4M3 更省**」，但这句话必须由产物写出来，不能留给读者自己算。

#### 2.4 严格前缀不变性两条 —— True，且是真检验（有牙）
- 7/7 臂 `future_token_past_logits_equal = true` 且 `peer_request_logits_equal = true`（`results.json::points[*].prefix_invariance`、`side_points[*].prefix_invariance`）；`probe = tasks["var_recall"].tokens[:4]`（B=4 ≥ 2、T=256 ≥ 2）。
- 可失败性：仪器实现在 `.../2631-kv-budget-causal/causal_check.py:38-66`，做法是「改第 0 个请求的最后一个 token ⇒ 比 `base[0, :-1]` vs `got_future[0, :-1]`」与「改同批第 1..B-1 个请求 ⇒ 比 `base[0]` vs `got_peer[0]`」，**不是**恒真式；`selftest_fp8.py::test_prefix_invariance_instrument_catches_noncausal_forward` 构造一个非因果前向并断言仪器两条都报 False ⇒ 仪器确有牙（这条是包内自证，我未运行）。
- 注意口径：这两条只覆盖**同批内**的因果性（batch 邻居污染），不覆盖「流式跨步缓存」场景；本包是 dense 全深度重算，够用。

#### 2.5 WP2 降级是否诚实 —— 诚实；`已完成` 不成立
- 诚实（我逐条核过）：`wp2_hd64_status.json::status = "not_completed"`；`empirical = null`；`analysis.md:118-124` 把 `head_dim=64` 的 6366/6336 B 明确标为**解析**、`head_dim=32` 的 G 阶梯才是实测；`analysis.md:181` 明写「是解析价签不是实测量」；`MODEL.md` 末尾「**不得**把未跑的训练数字当成已测」。给出精确复现命令与 `then_arms`、能力面检查（`engine_change_required = false`）、受影响验收项。**没有**把解析当实测、**没有**把未跑的训练数字写成已测。
- 该点无实测支撑这件事本身：`milestone.acceptance`（`project.yaml:36`）含「**大 `head_dim` 省幅与解析一致**」。当前只有「`head_dim=32` 的 G ∈ {8,16,32} 实测（`n_scales/n_values` = 1/G，与解析逐位一致）+ `head_dim=64` 解析」⇒ 验收项**未满足**。
- 我的独立判定：`status` 应为 **`进行中`**（`docs/research-schema.md:48,122-124` 的枚举里，`进行中` = 有本地工作、结论未定）；WP1 已有明确判决可记 `已完成`，WP2 实测腿缺位 ⇒ **项级不得 `已完成`**（`已完成` = 「本地实测已有明确结论」）。lead 的 `TICKET.md:374-376` 自认「该项 milestone **未完全满足**」，与我一致。

---

### 3. Finding

**blocker（1）**

**B1 · milestone 验收未闭，且判据要的「fp8 vs int8」二选一判定缺失；SSOT 仍写 `未开始`。**
- 位置：`project.yaml:36`（acceptance 原文）、`project.yaml:61` / `:19` / `:17`、`analysis.md:62`、`README.md:67`、`README.md:123`、`run.py:162`（`paired_delta` 的 base 恒为 fp16）。
- 违反的判据：`milestone.acceptance`「fp8 相对 int8+块标尺有/无越带差异，**二选一明确**」+「大 `head_dim` 省幅与解析一致」；`objective_criteria` 同文；`AGENTS.md` §2.1（`project.yaml` 是「有哪些研究项、每项当前状态/结论/证据」的唯一权威）。
- 复现/证据（我读到的原文，命令我**未能执行**）：
  - `grep -n "fp8 vs int8" analysis.md` ⇒ 只命中 `analysis.md:62` 表头，其「越 0.014 带」列的值来自 vs `fp16`（`run.py:162` 用 `paired_delta`，`results.json::paired_delta.base_arm = "fp16"`）。
  - 我按 `results.json::points[1].bpc_paired_windows`（int8）与 `points[2].bpc_paired_windows`（e4m3）相减得到的 5 个值（见 §2.3）⇒ max|Δ| = 0.005363 < 0.014。
  - `project.yaml:61 status: "未开始"`、`:19 experiments: []`、`:17 conclusion: ""`，而 `experiments/2653-fp8-vs-int8-13m/results.json::partial = false`、`arms_completed` 4/4、`side_arms_completed` 3/3。
- 建议修法（最小）：① 在 `analysis.md` §0 与 §8 各加一句「`fp8_e4m3` vs `int8_block_g32`：配对 Δ max|Δ| = 0.005363（5 窗，同口径，由 `results.json` 两臂读数相减得到）⇒ **不越 0.014 带 = 不可分**，故二选一按每位置字节取 fp8 E4M3（6144 B）」，并在 `README`/`config::criterion` 的对应字段补同一句指针；② `project.yaml`：`status: 进行中`、`experiments` 登记该目录、`conclusion_review` 填本轮判词；**不得**判 `已完成`（WP2 实测腿缺位）。

**normal（3）**

**N1 · 预登记的 `fp8_scale_claim_rule` 的触发分支没按字面执行。**
- 位置：`config.json:116`（规则原文）、`analysis.md:26`（结论）、`results.json::paired_delta.per_arm.fp8_e5m2.exceeds_band_0_014 = true`。
- 违反：规则写「『fp8 免标尺』只在 **fp8_e4m3 / fp8_e5m2 两臂**质量不越带时成立；**任一臂越带** ⇒ … 并**宣布该前提失败**」。E5M2 越带（max +0.014880 > 0.014）⇒ 触发「宣布前提失败」分支；`analysis.md:26` 却直书「『fp8 免标尺』这个前提在本尺度**成立**」。
- 证据：`config.json:116` 与 `results.json::criterion.fp8_scale_claim_rule` 逐字相同；`analysis.md:26` 与 `:206`（否证②「未命中」）两处都在讲「不需要标尺」，而不是「两臂都不越带」。
- 建议修法：把结论改成「『免标尺』在本尺度**对 E4M3 成立、对 E5M2 被质量否掉**（按预登记规则，E5M2 越带已触发『前提失败』分支）」；或明确写「本项把该规则窄化为『fp8 是否需要标尺』回答（实测 0 标尺），两臂越带与否分列」。任一写法都行，但不能留「规则说 A、结论说 B」。

**N2 · 变异/单测只有「声称」没有运行证据。**
- 位置：实验目录**没有** `mutation_report.json`（模板 §复核对象列的产物之一）；`README.md:12`「`selftest_fp8.py`（可失败单测 + 变异）」、`README.md:130` `python3 -m pytest selftest_fp8.py -q`、`selftest_fp8.py` 各 docstring 的「变异：… ⇒ 本测试红」。
- 违反：`docs/methodology.md` §5 的变异纪律 + `proof.gate_probe`（每个门禁要有让它变红的实例并有留证）。
- 证据：`ls` 该目录只有 12 个文件 + `__pycache__`，无 `mutation_report.json`；`run_log.txt` 只记了主流程与逐臂读数，没有 pytest/变异输出；`results.json` 也没有 `pytest` 字段。
- 修复：跑 `python3 -m pytest selftest_fp8.py -q`，把（含变异红跑）输出落到该目录的 `mutation_report.json`/`JOURNAL.md`，或在 `analysis.md` §6 如实写「单测未跑/未留证」。**该命令必须由有 exec 的 supervisor 代跑**（我无 exec）。

**N3 · E5M2 的「越带」判定依赖未声明的聚合规则，且贴近边界。**
- 位置：`run.py:423-425`（`exceeds_band_0_014 = max(|Δ|) > band`）、`results.json::paired_delta.per_arm.fp8_e5m2`（`mean = 0.0113752`、`max = 0.01488`）、`analysis.md:16`/`:200-202`（把 E5M2 判为「越带 ⇒ 不能用」）。
- 违反：判据只声明「0.014 带（限配对 Δ、≥5 窗口）」，**未声明**用 max 还是均值判定越带 ⇒ 结论对聚合规则敏感（用均值读 = 0.0114 < 0.014 ⇒ 不越带）。
- 证据：`config.json::criterion` / `README.md:73` 均无聚合规则；`analysis.md:200` 的「E5M2 +0.0149 越带」用的是 max。
- 建议修法：在 `config.json::criterion`（下轮预登记）或 `analysis.md` 明写「越带 = 任一窗口 max|Δ| > 带（保守取法）」并披露「5 窗均值 0.0114 在带内、仅 seed 6 = 0.01488 越界」，别让边界结论裸奔。**不要再改数字/结论**，只加声明（符合本轴「只加声明句」惯例）。

**info（4）**

- **I1 · fp8 臂的 `clipped_frac = 0.0` 是默认值填出来的，不是测量值。** `tools/evalkit/kv_quant.py:93` 把 `n_clipped` 初始化为 0，`:217` 输出 `clipped_frac = n_clipped/n_values`；fp8 路径（`fp8_arms.py:96` 只另加 `n_saturated`）从不设置 `n_clipped` ⇒ `results.json::points[fp8_*].activity.clipped_frac = 0.0` 是**未测量却报 0**（`AGENTS.md` @fabricated）。真实诊断量 `saturated_frac` 是分开报的、有值。建议：fp8 臂不输出 `clipped_frac`，或改名为 `n_clipped: null`。
- **I2 · 「每位置字节随 G 的实测与解析逐位一致」措辞偏大。** `analysis.md:115`（与 `:44`）——实测的是**标尺个数/值数 = 1/G**（`n_scales` 786432 / 393216 / 196608），`7680/6912/6528` 三个字节数仍是**解析**值；`analysis.md:181` 已自己声明「是解析价签不是实测量」，两处口径应一致措辞（建议改成「实测标尺计数与解析 1/G 逐位一致 ⇒ 字节账的 `2/G` 项成立（字节数仍为解析）」）。
- **I3 · 同一个锚在两个家写了两个值。** `project.yaml:10` 写 `trunk 1.483571 bpc`，而 `config.json::checkpoints.trunk.role` / `MODEL.md` / `analysis.md:26` 全写 **1.483569**（本 run 的 CPU 读数为 1.483570）。同一事实两处数值 ⇒ 按 @ssot 统一（`project.yaml` 是权威，需确认哪个是 `2610/2614` 的原值）。
- **I4 · 「同字节下 int8+块标尺仍是最稳的」是在带内做的排序。** `analysis.md:29`/`:44` —— int8 的 +0.0004 与 fp8+标尺的 +0.0048 都在带内，二者之差 ≈0.0044 也在带内 ⇒ 按判据它们**不可排序**；正确说法是「6528 B/位置处两者质量不可分，而 fp8+标尺不比 int8 省字节 ⇒ **加标尺无收益**」（结论方向不变，只需去掉「最稳」的排序含义）。

---

### 4. 我无法执行的检查项（如实登记，未猜结论）

| 检查 | 原因 | 需要谁跑什么 |
|---|---|---|
| `git rev-parse HEAD` 基线绑定、`git status --short`、`git diff --stat`、`git diff --name-only <sha>..HEAD -- axes/5-quant/5.5-fp8-head-dim/` | 本会话**无 exec**；`watchdog_diff` 返回「Reviewer-launch HEAD changed or is unavailable. Committed changes are unsupported」⇒ **committed 区间与工作树范围纪律都不在我的视野** | 有 exec 的 supervisor 代跑并把原文贴进本轮 ticket |
| `python3 -m pytest selftest_fp8.py -q`（及其变异红跑） | 无 exec | supervisor 代跑；结果决定 N2 是否关闭 |
| `python3 tools/summary.py --check` / `python3 tools/check_tree.py` | 无 exec | supervisor 代跑（`project.yaml` 合规性 + 是否因 `experiments: []` 报错） |
| `README`/`config` 的 mtime 早于 `results.json`（预登记时序） | 无 exec | supervisor 用 `stat` 代核 |
| `grep -n "add_argument" train_depth_ar.py`（WP2 能力面） | 无 exec；我未读该引擎文件 | 按 `wp2_hd64_status.json::capability_check` 采信，未独立复核 |
| `5.1/2632` 的 bpc 数字 | 契约明写不复用其 bpc 当判据（设备不同） | 不需要跑；我按契约未使用 |

---

### 5. 本包结论的强度上限（我认可的最强说法）

1. **WP1（可以说的）**：在**这一个** 13M checkpoint（seed 0、12k 步）、**CPU 口径**、T=256、单次评测窗口序下，把 KV 存成 1 B/值的 fp8：`fp8_e4m3` 的 clean bpc 为 1.486669（fp16 = 1.483570），配对 Δ（5 窗）max = 0.005415 **在 0.014 带内**；`fp8_e5m2` max = 0.014880 **越带**且 copy 召回 0.1419 → 0.1289；`int8_block_g32` = 6528 B/位置、max Δ = 0.000384。**「免标尺」在本数据的实况是 `n_scales = 0`**。
2. **不可以说的**：不能外推到别的尺度/T/模型（`analysis.md:184-185` 自己也这么声明）；不能说「fp8 免标尺「全局成立」（E5M2 已被质量否掉，见 N1）；不能把 6144/6528 B/位置当成「已实现的显存收益」（是解析价签，评测实现立刻反量化，`analysis.md:181`）；不能说 `head_dim` 更大就更好（`head_dim=64` 无实测）；不能按「int8 最稳」排序（I4）。
3. **WP2 上限**：解析阶梯 + `head_dim=32` 上 `2/G` 标尺计数的实测对账（三档 1/G 逐位一致）；**`head_dim=64` 的 48.4375% 只是解析**。
4. 召回腿强度上限：copy n=1536（显著高于地板 0.025756）；`induction`/`var_recall` 全臂 = 0、`at_floor` ⇒ **在本主干上不可分辨**（不得读作「机制无效」，产物已如此声明）。

---

### 6. 对 lead 的处置建议

1. **必须修（闭合验收前）**：B1 —— ① `analysis.md` 补 fp8-vs-int8 的带判定一句（不可分，Δ max 0.005363）；② `project.yaml` 更新为 **`进行中`** + 登记 `experiments` + 填 `conclusion_review`（含本轮判词）；**不要**判 `已完成`（`head_dim=64` 实测腿缺位 + 上面那条判定缺失）。这两项是文档面改动，**无需重跑实验**。
2. **必须修（规则一致性）**：N1（`fp8_scale_claim_rule` 字面触发）与 N3（越带聚合规则/边界披露）；都只加声明句，不动数字与结论。
3. **必须留证**：N2 —— 由有 exec 的一方跑 `python3 -m pytest selftest_fp8.py -q`、`python3 tools/check_tree.py`、`python3 tools/summary.py --check`，把输出 + `git status --short` / `git diff --stat` 落进 `.scratch/66-axis5-quant/`（我这一轮无法执行，故范围纪律与单测两项对我**仍是未核**）。
4. **建议登记为局限、不必修**：`head_dim=64` 实测腿未完成（GPU 锁争用，降级路径按 `DISPATCH-5.5` §4 执行且已如实登记）；字节账是解析口径；`induction`/`var_recall` 落地板；单 seed 冻结 checkpoint ⇒ 不宣称结构优劣。
5. **建议 status**：**`进行中`**（WP1 = `已完成`；WP2 = `进行中`，缺口 = 1 个 13M `--heads 4` 训练 + 3 臂 CPU 评测，`analysis.md:212` 已给 ≈9 min GPU + ≈4 min CPU）。若 lead 想让整项 `已完成`，则**必须**先补 `head_dim=64` 实测腿（或把 milestone.acceptance 的该条按 `replan_lower` 明确改写并说明理由，不能默认放宽）。
6. 判词：**`OK with notes`**（blocker 1 / normal 3 / info 4）。数据面可信、门禁有牙、降级诚实；真正挡「收口」的是那条没写出来的二选一判定与 SSOT 未更新。
