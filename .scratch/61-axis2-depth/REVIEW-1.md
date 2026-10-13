# REVIEW-1 · 轴 2 Phase A 交付独立复核（revision `233e978`）

> 复核者：**fresh 独立复核者**（独立上下文 subagent，未参与本线任何写作；本轮以 `reviewer`
> 角色派发，只读工具集）。报告由复核者产出，轴 2 lead **原样落盘**（未改动一字）。
> 复核者能力限制（其自述）：本 run 只有只读工具（read/grep/find/ls/diff），**没有 shell 与写文件
> 能力** ⇒ `pytest`、变异测试、实验重跑、`summary.py --check` **均未执行**；结论建立在
> **静态复核 + 逐数字对账**上。运行证据由 round 2 复核（有 shell）补齐。
> 处置记录见 `.scratch/61-axis2-depth/TICKET.md` §7.2。

---

I have no shell or write tools in this run (only read/grep/find/ls/diff/contact), so I could not execute the four mandatory commands nor write `REVIEW-1.md`. Per the task's own fallback clause ("若你没有写文件的工具，就把完整报告放在最终回复里，我会原样落盘"), the **full report** is below, unmodified. No file was changed; no git operation was performed.

---

# 轴 2 Phase A 交付独立复核 — revision `233e978`（fresh 复核 round 1）

复核者：独立上下文 subagent，未参与本线任何写作。工作区状态：`watchdog_diff` = "No working-tree changes against reviewer-launch HEAD 233e978f4d46"。
**能力限制（必须先声明）**：本 run 只有只读工具（read/grep/find/ls/diff），**没有** shell 与写文件能力 ⇒ ①`pytest`、②变异测试、③实验重跑、④`summary.py --check` **均未执行**，仅完成静态复核 + 逐数字对账（对账是我能做的最强证据层，见 §5、§6）。另：`watchdog_diff` **不覆盖已提交区间**，且本任务未提供 commit diff 工件 ⇒ 我复核的是 `233e978` 的**最终文件树**，无法判断「哪几行是本次新增」（凡涉及"是否本次引入"的地方我都明确标注）。

## ① 人话摘要

这段交付在说：普通 transformer 每个字符都跑满 8 层，这条线主张让**每个字符自己决定跑几层**；55 号独立复核判过「这个主张从没端到端验过，公开的 0.97 只是『第 4 层答案和第 8 层答案一样不一样』的自洽指标」。本轮补了两件东西：**承载体**（把逐字符退出判断真的装进生产解码器 `DepthARDecoder.step_auto`，新模块 `pathlm/exit_policy.py`，16 条单测）和**主判据的第一次实测**（8.16 万个字符位置上逐位置在线决策，量整段 bpc 与每字符平均层数）。

我判：**技术结论成立，而且是被证据逼出来的、双向诚实的结论**——「逐字符门控最多只能不输简单的整段截断，从未赢」。这一点我按数字逐条对过 `results.json`（14 条臂、锚点、前沿、校准、risk–coverage、同位置配对、四类诊断，**全部命中**，见 §5）。决定性的那条比较（τ=0.9 用 5.967 层却比「固定退 4 层」还差 0.0074 bpc）**不依赖任何插值**，是直接实测对撞，很硬。

**最要紧的一条问题（按父级判据属 blocking）**：SSOT 里两个界限数字被写紧了。`project.yaml` 的 `conclusion` 和 `analysis.md` §0/§6 都写「**异构深度上下文的代价 ≤ 0.003 bpc**」，但同目录 `results.json` 与它自己引用的数字是 **+0.003296**（`same_position_paired`，τ=0.6 的 d8 桶，n=34452）；`analysis.md` §5 写「ECE_true ≤ 0.005（深度 ≥ 3）」而 depth 3 实测 **0.0051**。前者把 0.0033 四舍五入成上界（差 10%），后者压在 0.0051 的取整边界上。**结论方向不受影响**（两者都远在 0.014 带内），但按父级判据「数字与 `results.json` 矛盾 ⇒ blocking」，这属于必须一行修掉的硬缺陷。

其余是措辞/区间标注/一个潜在的解码器语义坑（见 §3、§4），没有第二条 blocking。

## ② 逐范围块判词

### 块 1 —— `pathlm/exit_policy.py` + `DepthARDecoder.step_auto` 实现正确性：**成立（有 2 条范围性保留）**

我按「逐句对照并行 `forward` 的非 dense 分支」静态核过（`depth_ar.py:2262-2310` vs `exit_policy.py:141-203`）：

- **活性判定一致**：parallel `active = exit_depths >= k` ↔ streaming `active = ~decided`；退出位置在层 k 的 KV 源 `where(active, h, kv_src)`（`depth_ar.py:2284`）与 `exit_policy.py:185-187` 逐句相同；`proj_fill` 走 `kv_adapters[k-1](frozen)` 两边一致（`2283` vs `exit_policy.py:183-184`）。
- **冻结态定格位置一致**：parallel 在层末 `frozen = where(exit_depths == k, h_next, frozen)`（`2300-2301`），streaming 在 `_apply` 里 `frozen = where(want, state, frozen)`（`exit_policy.py:167`）；两处存的都是**深度 k 的状态**（该位置在层 k 是 active ⇒ `h_next = out`）。手工验算三种情形（全同深度、混合深度、depth-0 退出）都对齐。
- **`step_auto` 边界**：depth-0（`_stop(0, readout(h,0))` ⇒ `exit_depth=0, frozen_self=inp`，与 `step(inp,0)` 的 `frozen_self = inp` 同款，`depth_ar.py:2708`）；`min_depth`（策略侧，批量/增量同源）；到顶必停与 `max_depth=None`（循环跑完走 `frozen_self is None ⇒ exit_depth, frozen_self = L, h`，`depth_ar.py:2809-2810`，与 `step` 收口一致）；`_apply` 的兜底 `if not decided.all()`（`exit_policy.py:196-197`）保证自定义策略不留半截状态。
- **`stream_bpc` 口径**：`bpc = mean CE(logits[:, :-1], tokens[:, 1:])/ln2`（`exit_policy.py:218-222`）；`layer_passes_per_token = mean(d)` 覆盖**全部 T 个位置**、`by_exit_depth` 覆盖 **T−1 个预测位置**，两者的计费口径在代码注释与单测里都写明了（`exit_policy.py:255-262`；测试断言 `sum(by_exit_depth.n) == depths.numel()-depths.shape[0]`）。
- **未取 GPU 锁、纯 CPU**：`run.py:196 --device cpu` 默认，`results.json.protocol.device = "cpu"` ⇒ `SCHEMA.md` §20 合规（纯 CPU 不取锁）；`results.json` 内**无墙钟/吞吐字段**（墙钟只进 `run_log.txt` 的过程行，`run.py:151-153` 只写 log，不进 row）⇒ §7/§8 与 `docs/research-schema.md` §3 合规。

保留（范围，均**不被本实验触发**，且我无法判断是否本 diff 引入）：
1. **`_layer_kv` 无视 `proj_fill`**：`depth_ar.py:2696-2700` 的填充始终是 `block.qkv(block.ln1(self.frozen))`，而并行 forward 的填充源是 `kv_adapters[k-1](frozen)`（`2281-2284`）。`DepthARDecoder.__init__` 为 GQA 与 `carry_rule` 都写了显式 `ValueError`（`2666-2683`），却**没为 `proj_fill=True` 把关** ⇒ 对非恒等 adapter 的模型，`step_auto`/`exit_policy.generate_with_exit` 会**静默**给出与训练前向不同的语义。`tests/test_depth_ar.py:197`（`test_decoder_matches_parallel_forward`）用的是 `tiny_model()` 默认（`proj_fill=False`）；新测试 `test_projfill_identity_matches_fill`（`tests/test_axis2_exit_decision.py:182`）只拿 `score_stream` 对 `score_stream`（恒等初始化），抓不到。本实验 checkpoint `proj_fill=False`，且分析已说明 proj-fill 将由 `score_stream` 做（`analysis.md` §8 U3）⇒ **今天不构成错误结论**，但 WP2/WP4 一旦改用 `generate_with_exit` 会踩到。
2. **`_layer_kv` 也无视 `fill_kv=False`**：它无条件对 `depths <= k0` 的位置做填充（`2699-2703`），所以解码器对 no-fill 模型不成立。同样未被本实验触发（no-fill 臂走 `stream_bpc`，`run.py:396`）。

### 块 2 —— `tests/test_axis2_exit_decision.py`（16 条）能否抓住实现错误：**成立（判词基于静态变异推理，未执行）**

- 计数核实：16 个 `def test_`（`grep` 全文件，行号 52…248），与 README/MODEL/ticket 声称的「16 条」一致；无同名、无 `skip`。
- **没有同义反复的测试**：每一条都是「两个独立实现互相对拍」（`score_stream` ↔ `forward(exit_depths=D)` ↔ `step_auto` 链）或「语义边界断言」，不存在把函数和自身比较的空转测试。
- 三条指定变异**应当**抓红（我按代码逐条推理，**未执行**）：
  - **变异 A**：`exit_policy.py:163` 把 `want = want & ~decided` 改成 `want = want`。⇒ **必然**红 `test_threshold_zero_exits_at_min_depth`：τ=0 时被冻结位置在更深层的读出仍是 `p ≥ 0`，`want` 恒真 ⇒ `depths` 被覆写成 L，`assert set(out["depths"][0].tolist()) == {2}` 失败（无需依赖任何概率偶然）；很可能同时红 `test_top1_policy_decoder_matches_batched`（批量口径与 `step_auto` 链分叉）。
  - **变异 B**：`depth_ar.py` `step_auto` 里把 `exit_depth, frozen_self = depth, h` 改成 `frozen_self = inp`（冻结态取错深度）。⇒ 后续位置的填充 KV 用错表征 ⇒ `test_step_auto_matches_prescribed_schedule` 的 `allclose(atol=2e-4)` 与 `test_top1_policy_decoder_matches_batched` 应当红（`tests/test_depth_ar.py:3132` 的既有活性测试已证明"填充源换值 ⇒ 底层 logits 必变"）。
  - **变异 C**：`Top1ProbPolicy.decide` 的 `min_depth` 判断反向（`if depth < self.min_depth` → `>=`）。⇒ 红 `test_threshold_zero_exits_at_min_depth`（深度变 0）与 `test_min_depth_is_respected_for_high_threshold`（`min ≥ 1` 失败）。
  - 结论：**这套测试对"实现改坏一处"是有抓力的**；弱项是 `p >= τ` 的等号边界与 `test_min_depth_is_respected_for_high_threshold`（只断言上下界的宽松单测），属正常强度差异，不构成缺陷。
- **不可替代性缺口**：没有任何一条测试覆盖 `proj_fill=True` 的非恒等 adapter 解码路径，也没有 `fill_kv=False` 的解码器对拍（见块 1 保留项）；若父级希望闸住块 1 的那个坑，最小补法是**在 `DepthARDecoder.__init__` 里像 `carry_rule` 一样拒绝 `proj_fill`（或先修 `_layer_kv`）**，并加一条 `fill_kv=False` 的 `step_auto` ↔ `forward` 对拍。

### 块 3 —— `2631` 实验（README / run.py / results.json / analysis.md / config.json / MODEL.md）协议与数字：**成立（1 条 blocking 数字界限 + 若干区间标注缺陷）**

预先登记与协议（成立）：
- 契约 `README.md` §1–§3 在跑前冻结（口径表、臂表、三条否证条件），`analysis.md` §7 逐条对照 ①未触发（字面）/②触发（实质）/③未触发，**判据线未被改过** ⇒ `SCHEMA.md` §14 清单第 1 项通过。
- 样本与历史锚点：`HIST_DEPTH_CURVE`（`run.py:59-60`）与我直接读到的历史证据 `.scratch/23-depth-ar/evidence/23c-H2-w0-12k/results.json::depth_curve` **逐值相同**（5.4191/2.3819/1.9988/1.8863/1.8628/1.8593/1.8585/1.8584/1.8582）；`HIST_TF` 与历史 `tf_acceptance` 逐值相同。锚点行 `sanity_anchor` 的 Δbpc = 7e-06…4.6e-05（不是"零"，见 §4-F6），`Δacc_true` ≤ 2.2e-05 ⇒ 新口径与既有逐深度曲线**同刻度**（这是本次最硬的一条自洽证据，且由我独立对到历史文件，不依赖 writer 的自述）。
- 三块诊断的排他性检查：`结构零假设` Δ=0.000000（全同深度下填充开关不该有影响 ⇒ 开关只影响混合深度，活性方向正确）；置换臂（+0.094727/+0.079843）与 no-fill 臂（+0.051373/+0.024385）都比原臂**更差** ⇒ 排除"选位错"与"填充伪影"。数值全部命中 `results.json`（`attribution_diagnostics`）。
- **前沿对比的公平性（父级点名质疑）**：折线端点 = `prescribed-2..7` + 全深度（8 层，Δ=0），区间外返回 `None`（`run.py:268-278`）；Δbpc(lp) 曲线是**凸**的（差商 −0.1125→−0.0235→−0.0035→−0.0008→−0.0002→−0.0001 单调递增），**弦在图上 ⇒ 插值把对手的 Δ 高估 ⇒ 对门控更有利**，不构成对结论有利的偏置。更强的是：主判据**根本不需要插值**——`thr-0.9`（lp 5.9668, Δ +0.011954）与直接实测的 `prescribed-4`（lp 4.000, Δ +0.004578）比，**少用 1.97 层还更好 0.0074 bpc**。⇒ 判词：前沿比较**公平且保守**，结论不依赖插值。
- **同位置配对是否循环论证（父级点名质疑）**：`base_ce` 来自独立的 `full_ce_per_pos`（`run.py:106-116`，全深度前向、逐位置 CE），臂 CE 来自臂自己的 `ce_per_pos`，两者在**同一批位置**上相减 ⇒ 桶内 Δ 是有效配对对比，**不循环**。但"**各桶加权 +0.12046 ≈ 总 Δ +0.120461，逐位闭合**"我按 `results.json` 的 8 个桶（n/Δ）实算：Σ n_k·Δ_k / Σ n_k = **0.120461**，与总 Δ **精确相等**（分桶是全体预测位置的一个划分）⇒ **这条"闭合"是恒等式，不是证据**（详见 §4-F3）。有信息量的是桶型：d1 +0.260644（n=30323）、d2 +0.144223、d3 +0.052002、d8 +0.003296（n=34452）。
- **"异构深度上下文代价 ≤0.003"的支撑强度（父级点名质疑）**：不是"只由一个桶一次偶然支持"——**两条臂**各自的最深桶都给小值：τ=0.6 d8 = +0.003296（n=34452，占 42.2%），τ=0.9 d8 = +0.000552（n=54457，占 66.7%），互相一致；d5 桶 τ=0.9 甚至为 −0.000234。但报告**没有给这两个桶的 SE，也没做分块/半样本稳健性**；我把"分块稳健性未知"登记为未覆盖（§6）而不是缺陷。
- **`acc_tf` 是否口径偷换（父级点名质疑）**：不是偷换，但有一处语境混杂须写明。定义在契约 README §1 表与 `analysis.md` §2 脚注都披露为「退出位置 top-1 vs **全深度 argmax**」，且 `analysis.md` §2 脚注主动点出「`never-exit` 的 `acc_tf` 恒为 1，而 `acc_true` 只有 0.627」——**退化臂的自我参照是被作者认领的，不是被藏起来的**。语境混杂之处：标签 `full` 是在**全深度上下文**前向上算的（`run.py:100-105`），而门控臂的读出处在**异构上下文**里，故该列同时含"深度截断"与"上下文"两个效应；锚点臂（prescribed-k，全同深度、填充不触发）不受影响，所以锚点那一行是干净的（`prescribed-4` 0.969767 = 校准表 depth4 0.969767，两处相等即证）。
- **`depth0`/直方图/算力账**：`depths_hist` 桶数固定 0..L（`n_layers` 传入，`exit_policy.py:270-282`），`skip_frac = 1 − mean(d)/L`。

**逐数字对账结果（我逐条核过，全部命中）**：`analysis.md` §0 人话表、§2 臂表（14 行 bpc/Δ/配对 SE/acc_tf/acc_true）、§3 锚点表、§4 前沿表（含"超出"列 0.2593/0.1168/0.0117/0.0035/0.0000/−0.0002/−0.0000 与 18.5×/8.3×/0.84×/0.25×/−0.02×）、§5 校准表（逐深度 p̄/acc_tf/acc_true/ECE_tf/ECE_true）与 risk–coverage 4 行（0.372/0.877/0.341；0.142/0.984/0.467；0.479/0.892/0.338；0.255/0.980/0.475）、§6 诊断表（2.0734/2.0301/1.9500/1.8946/Δ=0；桶值 +0.261/+0.144/+0.052/+0.0033 与 +0.0006）、§9 生成表（36/128…91/128）——**与 `results.json` 一致**（唯一例外见 §3 的 blocking 项与 §4 的区间标注）。

### 块 4 —— 10 份 `project.yaml` 的 `role_map` 与 `2.1.1` 的 SSOT 字段：**成立**

- `role_map`：10/10 四角色齐（`tools/check_tree.py:53` 只查"齐"、不查"异"，我人工核过）：`owner`/`writer` = 「轴 2 深度轴 lead（本项单一写入者…）」，`reviewer` = 「fresh 复核者（独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后）」，`integrator` = 「研究树 integrator…」。⇒ **非占位**（无"待指派"），**reviewer 由定义独立于 writer**，且实际独立有据：`.scratch/61-axis2-depth/TICKET.md` §7.1 记录了本次 fresh 复核的派发（复核者 = 独立上下文 subagent、非本项 writer），change_log 的 actor 是「轴 2 深度轴 lead（Phase A）」——与 reviewer 不同一。唯一可加严之处：`reviewer` 记的是**角色定义**而非具体 agent 身份，单看 `project.yaml` 无法验证"这一次的复核者是新的"（这一条由 TICKET §7.1 承载）。
- **`status: 被否证` 与证据一致，且没有双向谎报**：`failure_analysis` 非空，且**先分类**为①结构问题（主因）②设计问题（次因、可修）③非 innovation 本身失败④非实现/口径/统计假象，每类都带证据（`docs/research-schema.md` §7 要求，`check_tree.py:239-241` 硬性强制）✓。**关键**：分析**没有**把"字面验收成立"写成 PASS，也**没有**把"实质否证"夸大成"做不出来"——`analysis.md` §7 明确写①字面未触发（τ≥0.99 省 6–12% 且 Δ≤0.0002）、②实质触发（对照 falsification ②：被同算力固定截断支配）、③填充语义未触发；`project.yaml` 的 `conclusion` 复述同一结构，并把口径收紧问题（是否把验收从「Δ≤0.014」改成「相对同算力前沿超出量≤0」）作为**框架性取舍**上报（TICKET §5 F5），未自行改判据线——符合 `AGENTS.md`「口径选择属框架性取舍、由主 agent 记录并请用户裁决」。`falsification` 预登记②本来就写着「若门控被同算力固定前缀支配 ⇒ 判…无净增量（这是更实质的否证，比 0.014 更严）」⇒ `被否证` 是**预登记条件被触发**，不是事后改判。
- `work_packages`：7 个（WP1–WP6，其中 WP1 = 退出决策头，WP2 = 异构深度回看，WP3 = 深度监督配方，WP4 = 精化门 —— 与 `docs/structure.md` §12/§295 列的 2.1.1 四个手段**一一对应**，未另立节点 ✓；WP5/WP6 由 C-P12 与"策略感知训练"缺口引出 ✓）。WP1 `status: 已完成`、`evidence` 指向本实验目录，其余"未开始"。
- `experiments`：11 条**全部存在**（7 个本项实验目录 + `.scratch/23-depth-ar/evidence`、`.scratch/24-depth-ar-100m/evidence`、`.scratch/25-tax-attribution/evidence` + `axes/7-references/…/2623-standard-early-exit-13m/analysis.md`）⇒ `check_tree.py:243-246` 的存在性硬查应过。
- `cost_model` 无墙钟/日期（`check_tree.py:56-58` 正则词全不命中；10 份都只写 FLOPs / layer-passes / tokens-per-forward / pipelining）✓；`axis`/`title`/`dir` 用词与 `docs/structure.md` §12 冻结表一致（`2 深度轴` / `early exit（最重要）` / `axes/2-depth/2.1.1-early-exit/`）✓；`status` 值 ∈ 合法枚举 ✓；`change_log` 两条含 date/actor/change ✓。

## ③ blocking 缺陷清单（位置 + 违反判据 + 复现）

**F1（P1，按父级判据 blocking）数字界限被写紧：`≤0.003` 与 `≤0.005` 与其自身证据矛盾。**
- 位置：`axes/2-depth/2.1.1-early-exit/project.yaml:17`（`conclusion`）「**异构深度上下文的代价 ≤ 0.003 bpc**（同位置配对…）」；同一文件 `:16`（`failure_analysis`）却引「d8 桶 +0.0033」；`.../2631-pertoken-exit-seqlevel/analysis.md` §0 第 5 点「同位置配对 Δ ≤ +0.003 bpc」、§6 同一段「异构深度上下文代价 ≤ 0.003 bpc」而紧邻表行「d8 桶（…42%）只有 +0.0033」。
- 证据：`experiments/2631-pertoken-exit-seqlevel/results.json::same_position_paired` → `{arm: token-thr-0.6, depth_bucket: 8, n: 34452, delta_same_positions: 0.003296}`（另 τ=0.9 d8 = 0.000552）。
- 违反判据：父级「**数字与 `results.json` 矛盾**」（blocking）。本质是把 0.003296 四舍五入成"上界"。
- 最小修法（一行）：把 `≤ 0.003` 改为 `≤ 0.004`（或直接写「实测 0.0033（τ=0.6 d8 子集）/0.0006（τ=0.9）」），三处（project.yaml 的 `conclusion` + analysis §0/§6）同改。
- 复现：读 `results.json` 的 `same_position_paired`（如上），对照 `analysis.md` §6 表行与 §0 第 5 点的 `≤ +0.003`。

**同族副例（P2，建议一并对齐）**：`analysis.md` §5 与 `project.yaml:17` 写「`ECE_true ≤ 0.030`、深度 ≥ 3 时 **≤ 0.005**」，而 `results.json::calibration.rows[depth=3].ece_true = 0.0051`（深度 4–7 为 0.0028/0.0028/0.0024/0.0024，故真实界是 ≤0.0051）。修法：改 `≤ 0.006` 或写「≈0.005（0.0051）」。

## ④ non-blocking 清单（P2）

| # | 位置 | 问题 | 最小修法 |
|---|---|---|---|
| F2 | `analysis.md` §7 ②、`project.yaml:16-17`；`failure_analysis` ④ | **区间与其标注的集合不符**：写「省 ≥25% 算力的档位…差 **0.0035–0.2593** bpc（0.25–18.5 倍噪声带）」，但 0.0035（0.25×）属 `token-thr-0.95`（省 20%）；≥25% 档的实际超出是 **0.0117–0.2593**（0.84×–18.5×）。另 `failure_analysis` ④「主结论 **Δ** 是 8–18 倍噪声带」用的是**超出量**的倍数（Δ 的倍数是 8.6×/25.9×）。注意 `analysis.md` §0 写的是对的（0.012–0.26 / 0.8–18×） | 把 §7 与 project.yaml 的区间改成 0.0117–0.2593（0.84–18.5×），或把限定词改成「省 ≥20%」；④ 的 8–18 改成「超出量是 8.3–18.5 倍」 |
| F3 | `project.yaml:18`（`conclusion_review`）、`:16`（`failure_analysis`）、`analysis.md` §6 | 「同位置配对各桶加权 +0.12046 ≈ 总 Δ +0.120461，**逐位闭合**」在 `conclusion_review` 里被当作佐证（「这种闭合性**很难由巧合产生**」）。我实算：Σ n_kΔ_k/Σ n_k = 0.120461（用 `results.json` 8 个桶）＝总 Δ **精确相等** ⇒ 这是**分桶划分的算术恒等式**，不能佐证任何东西（信息量全在桶型：d1 +0.261 / d8 +0.0033） | 把该句改成"分桶分解（恒等式，用于确认归因口径）"，删掉"很难由巧合产生"这半句 |
| F4 | `run.py:290-293`（→ `results.json::frontier_comparison`） | `dominated_by_fixed` 只判「∃ 固定臂 Δ ≤ d」，**忽略横轴** ⇒ 会被"更贵的固定臂"判成支配；配对的 `extra_layers_vs_dominated_fixed_arm` 因此可为**负数**（`token-thr-0.999-min4`: −0.362231；`token-thr-0.999`: `null`）。分析未引用这两个字段，所以不影响结论，但 `results.json` 是证据 SSOT，下游读者会误读 | 改成 `∃ (x,y): x <= lp and y <= d`（或删掉这两个字段，保留 `frontier_excess`） |
| F5 | `MODEL.md` §3、`README.md` §1、`run.py:58` 注释 | 「样本 = 与历史 `depth_curve` / **`tf_acceptance`** 完全同一个样本约定」**对 tf 不成立**：`depth_ar.py:2442` `depth_curve` 用 `manual_seed(2)`，而 `depth_ar.py:2471` `tf_acceptance` 默认 `manual_seed(3)`。经验证据：同一样本下 Δbpc ≤ 4.6e-05、Δacc_true ≤ 2.2e-05，但 `run_log.txt` 的锚点行 `prescribed-2: Δtf=-0.0054`（`results.json::sanity_anchor.acc_tf_hist=0.8597` vs 本次 0.854326）——远大于浮点差 ⇒ 历史 tf 是**另一个抽样**。`analysis.md` §3「历史 tf_acceptance 也对上」偏松 | 把该锚点写成「tf@k 为跨抽样量级对照（历史记录用不同 seed）」；bpc 锚点保持不变 |
| F6 | `analysis.md` §3「与既有判据**逐位重合**」、§6/§8 多处「逐位一致」 | 锚点差值是 **7e-06…4.6e-05**（`results.json::sanity_anchor.delta`），不是 bitwise；三方对拍容差也是 `atol=2e-4` | 改「在浮点噪声内重合（Δbpc ≤ 4.6e-05）」 |
| F7 | `project.yaml:17`、`analysis.md` §4 | 「门控**从未**超过同算力固定截断」与自己的表冲突：τ=0.999 的超出 = **−0.0002**（配对 SE 0.000136，不显著） | 改「从未**显著**超过」 |
| F8 | `analysis.md` §4 表末列（`token-thr-0.99` 行） | 该列定义是「超出量/0.014」（其邻居 τ=0.95 = 0.003493/0.014 = 0.25× 对得上），但 0.99 行写 `0.01×`，按定义应为 `0.00×`（0.01× 是 Δ/0.014 = 0.011）——一列混了两个量 | 该格改成 `0.00×`（或统一列定义） |
| F9 | `analysis.md` §6「≤0.003」的稳健性 | 该界由**两条臂的最深桶**支持（+0.0033 / +0.0006，互相一致，比"单桶一次"强），但**无 SE、无分块/半样本检验** | 建议补一个 0-GPU 的最小稳健性检查（按 CHUNK=64 的 5 块分别算 d8 桶 Δ 与 SE），落进 `results.json` 后再写界 |
| F10 | `analysis.md` §5 校准表、§0 第 6 点与 `project.yaml:12`（`gaps`） | §5 表**跳过 depth 5**（`results.json` 有该行：p̄ 0.6276 / tf 0.9865 / true 0.6270 / ECE_tf 0.3589 / ECE_true 0.0028，不含任何反例）；「生成一致率 **28–55%**」把最高的 `token-thr-0.999-min4 = 0.711` 排除在区间外（§9 表里却列了它） | §5 补 depth 5 行；区间改成 28–71%（或注明只算 τ 扫描臂） |
| F11 | `pathlm/depth_ar.py:2696-2703`（+ `2746` `step_auto` 依赖它）、`tests/test_axis2_exit_decision.py:182` | 见块 1 保留项 1/2：解码器填充**无视 `proj_fill` 与 `fill_kv=False`**，而 `__init__` 对 GQA/carry 都有显式拒绝。本实验不触发（ckpt `proj_fill=False`、no-fill 臂走 `stream_bpc`），但 `step_auto`/`generate_with_exit` 的"与训练前向逐位一致"能力边界未被把关 | 在 `DepthARDecoder.__init__` 加 `if model.cfg.proj_fill: raise ValueError(...)`（与 carry 同款处置），或先修 `_layer_kv`；并补一条 `fill_kv=False` 的 `step_auto`↔`forward` 对拍 |
| F12 | `project.yaml:18`（`conclusion_review`）、`work_packages[WP1].review` | 指向 `.scratch/61-axis2-depth/REVIEW-*.md`，该目录**当前只有 `TICKET.md`**（我确认过）⇒ 悬空前向引用；`conclusion_review` 的"修完后本字段改为复核者判词摘要"也尚未兑现 | 本轮复核报告落盘后，把 `conclusion_review` 的自评段替换为复核判词摘要（不要保留"已派复核"的悬空指针） |
| F13 | `TICKET.md` §4.5「11 条臂」；`run.py:200`；`run.py:262` | ① 实际臂数 = **14**（`results.json::arms` 与 `analysis.md` §2 的 14 行；1 never-exit + 6 prescribed + 6 阈值 + 1 min4）；② `run_log.txt` 以 `"a"` 打开，README §4 教的原地重跑会**把两次运行流水混在一起**；③ `results.json` 同时存 `bpc_per_window` 与 `bpc_per_window_rounded`（各 320 值 × 14 臂，占文件绝大部分） | ① 改 14；② 改 `"w"` 或按 `--out` 分目录；③ 删掉 rounded 副本 |
| F14 | `exit_policy.py:287-295`（`generate_with_exit` docstring） | `stats["depths"]` 被描述为「每个**新生成位置**的退出深度」，实际是**产出该 token 的那个位置**的深度（prompt 尾对齐），且 `layer_passes_per_token` 覆盖 prompt + (n_new−1) 个位置（最后一个生成位置没有前向）。诊断列用不到，但 U4（自由生成质量）会依赖它 | docstring 改成"产出各生成 token 的位置的退出深度；均值覆盖 prompt + n_new−1 个位置" |

## ⑤ 实际跑过的命令与结果（诚实登记）

**我一条都没跑**（本 run 无 shell）。以下是**必须由父级/supervisor 执行**的命令与预期，我已把预期值的来源与容差写清，便于直接比对：

1. `python3 -m pytest tests/test_axis2_exit_decision.py -q` → 预期 **16 passed**（我按 `grep '^def test_'` 核过确实 16 条）。
2. **变异测试（必做，做完必须恢复）**：
   - A：`pathlm/exit_policy.py:163` `want = want & ~decided` → `want = want`；预期红：`test_threshold_zero_exits_at_min_depth`（必然）、可能 `test_top1_policy_decoder_matches_batched`。
   - B：`pathlm/depth_ar.py` `step_auto` 内 `exit_depth, frozen_self = depth, h` → `frozen_self = inp`；预期红：`test_step_auto_matches_prescribed_schedule`、`test_top1_policy_decoder_matches_batched`。
   - C：`Top1ProbPolicy.decide` 的 min_depth 判断反向；预期红：`test_threshold_zero_exits_at_min_depth`、`test_min_depth_is_respected_for_high_threshold`。
   - 恢复后必须 `git diff --exit-code`（工作区应回到 `233e978` 干净；不要 `checkout`/`stash` 别人的东西）。
3. `python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py --out .scratch/61-axis2-depth/repro-review1`（纯 CPU，**不取 GPU 锁**；`run_log.txt` 记录上次 ~443s 过程量，仅作时长参考）。预期关键数字（与 `results.json` 6 位小数一致）：
   - 臂 bpc：`never-exit 1.858246` / `prescribed-2 1.998807` / `-3 1.886340` / `-4 1.862824` / `-5 1.859336` / `-6 1.858512` / `-7 1.858358` / `thr-0.3 ≈2.220760` / `thr-0.6 1.978707` / `thr-0.9 1.870200` / `thr-0.95 1.861944` / `thr-0.99 1.858397` / `thr-0.999 1.858073` / `min4 1.858249`；`layer_passes` 8.000 / 2.000 … / 7.637769。
   - `sanity_anchor` Δbpc ≤ 4.6e-05；`frontier_comparison` 超出量：0.259346 / 0.116819 / 0.011661 / 0.003493 / 0.000042 / −0.000231 / −0.000038。
   - `same_position_paired`：τ=0.6 d8 `n=34452, Δ=0.003296`；τ=0.9 d8 `n=54457, Δ=0.000552`。
   - `attribution_diagnostics`：shuffled-0.6 `2.073434`(+0.094727)、nofill-0.6 `2.030080`(+0.051373)、shuffled-0.9 `1.950043`(+0.079843)、nofill-0.9 `1.894585`(+0.024385)、结构零假设 `1.862824/1.862824, Δ=0.0`。
4. `python3 tools/summary.py --check` → 预期 OK（46 个 `project.yaml`）。我静态核过 2.1.1 的必需字段集、`status` 枚举、`被否证 ⇒ failure_analysis` 非空、`cost_model` 无墙钟、`experiments` 11 条路径**全部存在**、`role_map` 四角色齐 ⇒ 它应过；但**未运行**。
5. `python3 -m pytest tests/ -q` → writer 自报 831 passed；**未复核**（父级已豁免整仓回归）。

## ⑥ 未覆盖部分（诚实登记）

1. **未执行任何命令**（无 shell）：`pytest`、变异测试、实验重跑、`summary.py --check`、整仓回归——全部只给了命令与预期（§5）；因此"测试真的能失败"目前是**静态变异推理**，不是运行证据。
2. **未看 commit diff**：`watchdog_diff` 不含已提交区间、任务也未给 diff 工件 ⇒ 无法把"块 1 保留项（`_layer_kv` 的 proj_fill/no-fill 语义）"归因为"本次引入"或"既有债"；只能报为**当前文件树里的事实**。
3. **未做的稳健性检查**：d8 桶 Δ 的 SE / 分块（CHUNK=64 五块）与子样本稳定性；置换臂的多次置换（只有 1 个置换 seed=1234，`run.py:404`）——都需重跑，0 GPU 可做。
4. **未核的历史面**：历史 `tf_acceptance` 的生成脚本/seed 没有留在仓内（`.scratch/23-depth-ar/` 只有 `evidence/` 与 `issues/`），所以 F5 是"由 `depth_ar.py` 默认 seed + 观测差推断"，不是直接读到那条命令。
5. **未触及**：其它轴、`docs/**`、共享工具、无关既有代码（按契约排除）。
6. **未写入 `REVIEW-1.md`**：本 run 无写文件工具，完整报告在本回复里，请原样落盘到 `/home/z/vibe/pathlm/.scratch/61-axis2-depth/REVIEW-1.md`（`TICKET.md` §7.1 已把该路径写成约定）。

## 判词（合并结论）

- 块 1 实现正确性：**成立**（2 条范围性保留：`proj_fill` / `fill_kv=False` 的解码器语义未把关，本实验不触发）。
- 块 2 16 条单测：**成立**（三条指定变异应当抓红；未执行，登记为未运行）。
- 块 3 实验协议与数字：**成立但有一处数字界限与证据不符（F1）**；其余为区间/措辞级。
- 块 4 `role_map` 与 SSOT 字段：**成立**（`status: 被否证` 双向诚实、归因分类合规、无谎报 PASS/FAIL）。
- **Merge verdict：BLOCK（仅 F1 一条；一处一行数字修正后，其余判 OK with notes）**。若父级认定「≤0.003 vs 0.0033 / ≤0.005 vs 0.0051」属措辞而非实质，则本交付可判 **OK with notes**——我在 14 条臂、锚点、前沿、校准、risk–coverage、同位置配对、四类诊断上**没有找到结论与证据不符、口径偷换、无根据外推或测试抓不住实现错误**的证据。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "完整复核报告已给出：①人话摘要与总体判词（技术结论成立、1 条 blocking 数字界限）；②四个范围块逐块判词（含实现逐句对拍、变异推理、前沿公平性与循环论证分析、SSOT 字段核对）；③blocking 清单 F1（位置 project.yaml:17 + analysis.md §0/§6「≤0.003」vs results.json same_position_paired=0.003296）；④14 条 non-blocking；⑤命令未执行的原因与必须由 supervisor 执行的 5 条命令及预期值；⑥未覆盖部分 6 条；⑦residualRisks 已列。"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 -m pytest tests/test_axis2_exit_decision.py -q",
      "result": "not-run",
      "summary": "本 run 无 shell 工具；已静态核实确有 16 条 test_（行号 52..248），预期 16 passed，需 supervisor 执行"
    },
    {
      "command": "mutation: exit_policy.py:163 `want = want & ~decided` -> `want = want`（另有 step_auto frozen_self、min_depth 反向两处）",
      "result": "not-run",
      "summary": "未执行；静态推理判定 test_threshold_zero_exits_at_min_depth 必然红，test_top1_policy_decoder_matches_batched 很可能红；恢复后需 git diff --exit-code 验干净"
    },
    {
      "command": "python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py --out .scratch/61-axis2-depth/repro-review1",
      "result": "not-run",
      "summary": "未执行（纯 CPU、不取锁）；已给出应与 results.json 一致的逐项预期值（臂 bpc / 锚点 Δ≤4.6e-05 / frontier_excess / d8 桶 0.003296 与 0.000552 / 诊断四条 / 结构零假设 Δ=0）"
    },
    {
      "command": "python3 tools/summary.py --check",
      "result": "not-run",
      "summary": "未执行；静态核对 2.1.1 字段齐、status 合法、被否证⇒failure_analysis 非空、cost_model 无墙钟、experiments 11 条路径全部存在，预期 OK"
    },
    {
      "command": "python3 -m pytest tests/ -q",
      "result": "not-run",
      "summary": "父级已豁免整仓回归；writer 自报 831 passed，未复核"
    }
  ],
  "validationOutput": [
    "对账：analysis.md §0/§2/§3/§4/§5/§6/§9 的每条数字与 results.json 逐一比对，全部命中（14 臂 bpc/Δ/配对SE/acc_tf/acc_true、sanity_anchor、frontier_comparison、calibration 8 行、risk_coverage 4 行、same_position_paired 16 行、attribution_diagnostics 5 条、generation_agreement 7 行）。",
    "独立锚点核验：run.py HIST_DEPTH_CURVE/HIST_TF 与 .scratch/23-depth-ar/evidence/23c-H2-w0-12k/results.json 的 depth_curve/tf_acceptance 逐值相同；本次 prescribed-k 复现 depth_curve 至 Δbpc ≤ 4.6e-05、Δacc_true ≤ 2.2e-05。",
    "同位置配对恒等式实算：Σ n_k·Δ_k / Σ n_k = 0.120461 = 总 Δ（用 results.json 8 个桶）⇒「逐位闭合」是算术恒等式，不能作佐证。",
    "前沿插值公平性：Δbpc(lp) 差商单调递增（−0.1125/−0.0235/−0.0035/−0.0008/−0.0002/−0.0001）⇒ 曲线凸、弦在图上、对门控有利（保守）；主结论另有不依赖插值的直接对撞（thr-0.9 lp=5.9668 Δ=+0.011954 vs prescribed-4 lp=4.000 Δ=+0.004578）。",
    "工作区状态：watchdog_diff = No working-tree changes against reviewer-launch HEAD 233e978；本次复核未修改任何文件、未提交 git。"
  ],
  "residualRisks": [
    "未执行 pytest / 变异测试 / 实验重跑 / summary.py --check：『16 条都能失败』与『数字可复现』目前只有静态推理与逐数字对账支撑，不是运行证据；必须由 supervisor 跑 §5 的 5 条命令。",
    "watchdog_diff 不含已提交区间且未提供 diff 工件 ⇒ 无法区分『本次引入』与『既有债』；块 1 的两条保留项（DepthARDecoder 填充无视 proj_fill 与 fill_kv=False，depth_ar.py:2696-2703）在文件树里成立，但归属未定。",
    "F1（≤0.003 vs 实测 0.003296；≤0.005 vs 0.0051）按父级判据字面属 blocking；若父级认定非实质，可判 OK with notes。",
    "异构深度上下文代价界仅由两条臂的 d8 桶支持（0.0033 / 0.0006，互相一致），无 SE、无分块/子样本稳健性；置换诊断只用了一个置换 seed。",
    "F5：历史 tf_acceptance 的抽样 seed 无法从仓内直接读到，结论由 depth_ar.py:2471 默认 seed=3 与观测 Δtf=−0.0054 推出。",
    "结论的外推边界（单 checkpoint / 单 seed / 13M / 1 epoch / 均匀随机退出配方）已由交付自己登记，我核对了其表述未越界；但 100M、策略感知训练的未测项本身仍是残余风险。"
  ],
  "noStagedFiles": true,
  "diffSummary": "未查看 commit diff（无 diff 工具与工件，watchdog_diff 仅报工作区相对 233e978 干净）。复核对象为该 revision 的文件树：pathlm/exit_policy.py（新模块）、pathlm/depth_ar.py::DepthARDecoder.step_auto、tests/test_axis2_exit_decision.py（16 条）、axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/**、axes/2-depth/*/project.yaml（10 份 role_map）、.scratch/61-axis2-depth/TICKET.md。本次复核未修改任何文件。",
  "reviewFindings": [
    "blocker: axes/2-depth/2.1.1-early-exit/project.yaml:17 与 2631/analysis.md §0+§6 - 「异构深度上下文的代价 ≤ 0.003 bpc」与其自身证据 results.json::same_position_paired(τ=0.6, d8)=0.003296 矛盾（同文件 failure_analysis 自己写的是 +0.0033）；同族副例 analysis.md §5「深度≥3 时 ECE_true ≤ 0.005」vs 实测 0.0051。修法：≤0.003→≤0.004（三处）、≤0.005→≤0.006。",
    "no other blockers: 14 条臂 / 锚点 / frontier_comparison / calibration / risk_coverage / same_position_paired / attribution_diagnostics 逐数字与 results.json 一致；status=被否证 双向诚实（字面验收成立已披露、实质否证对应预登记 falsification ②）；10 份 role_map reviewer 独立于 writer；cost_model 无墙钟；纯 CPU 未取 GPU 锁；未发现测试抓不住实现错误的证据（三条指定变异静态推理均应抓红）。",
    "non-blocking: F2 Δ/超出量区间标注混用；F3 分桶『逐位闭合』是算术恒等式却被当作佐证；F4 run.py dominated_by_fixed 忽略横轴、extra_layers 可为负；F5 历史 tf_acceptance 是不同抽样（seed 3 vs 2）；F6『逐位重合』实为 ≤4.6e-05；F7『从未超过』应为『从未显著超过』；F8 §4 表 0.99 行混了两个量；F9 d8 桶无 SE/分块稳健性；F10 §5 漏 depth 5 行、生成一致率区间漏 0.711；F11 解码器填充无视 proj_fill/fill_kv=False（本实验不触发）；F12 REVIEW-*.md 悬空引用；F13 TICKET『11 条臂』实为 14、run_log 追加模式、bpc_per_window 重复存储；F14 generate_with_exit stats['depths'] 语义标注偏差。"
  ],
  "manualNotes": "本 run 只有只读工具，无法执行命令也无法写文件：请把本回复原样落盘为 /home/z/vibe/pathlm/.scratch/61-axis2-depth/REVIEW-1.md（TICKET.md §7.1 已约定该路径），并至少运行 §5 的第 1、2、3 条命令（pytest / 变异 / 重跑）以把『测试能失败』与『数字可复现』从静态推理升级为运行证据。我没有修改任何被复核文件、没有暂存、没有提交、没有触碰 git 历史；被复核 revision 233e978 的工作区在我复核前后均为干净。"
}
```