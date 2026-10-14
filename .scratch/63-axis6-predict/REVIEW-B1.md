<!-- 本报告作者 = fresh 复核者 #1（未参与写作）；其上下文**无 shell、无写文件工具**，
     故由 axis6-lead 于 2026-10-14T10:16:57+08:00 把其返回的完整报告**逐字**落盘（内容未改一字），
     并附其自报的约束与 not-run 清单。轴 lead 已代跑其 shell 类补证：见 TICKET.md §B5。 -->

# REVIEW-B1.md

**VERDICT: PASS**

复审对象：WP 6.2.3 `axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/**`（revision = 当前工作树，未提交）。契约 = 同目录 `README.md`。
复核者 #1，fresh 上下文，未参与写作。**本上下文无 shell、无写工具**，故所有需执行的动作标 `not-run`；数值复算用 `read`/`grep` 从落盘 JSON 逐位读出后手工核。

> 时间戳：无法读取系统时钟（无 shell），故本报告不写 `date -Iseconds` 读取值；下文中凡引用时刻均来自被审文档内已落盘的记录。**这是本报告唯一的证据缺口**，其余各条均有源文件行号。

---

## 正确项（已核，动手复算）

**C1 · V0 复现门 PASS（逐位相同，我自己比对，不引用自报值）**
`2650/results.json` ↔ `2640/results.json`：
- `repro_topk4`@13m_s0/prompt0 accepts = `[1,1,0,2,2,2,1,1,2,1,1,2,2,1,2,2,1,…]`（`2650` L166–236）与 2640 `tree_topk4`@13m_s0/prompt0（`2640` L10325–10396）**逐位相同**，`n_rounds=77` 两侧一致，`gen_head=[78,78,91,2,85,87,82,82]` 一致。
- `single_k1_ctl`@13m_s0/prompt0（`2650` L2593–2663）= `[0,2,0,2,2,0,2,0,0,2,0,1,0,2,0,2,0,0,2,0,1,1,2,2,0,2,2,2,2,2,0,2,2,2,2,2,1,2,2,1,2,2,1,…]`，与 2640 `combo_pw`@13m_s0/prompt0（`2640` L4350–4430）**逐位相同**，`n_rounds=82` 一致。
- `criteria.V0_harness_faithful.{bit_identical:true, pass:true, all_checked:true}`、`V3_degenerate_equivalence.{bit_identical:true, pass:true}`（`2650` L12624–12702），四对 (ckpt,臂) 均 `n_prompts_differing:0`。

**C2 · V1 无损硬门 PASS**
`all_arms_stream_identical:true`、`mismatching_arms:[]`、`n_forward_eq_n_rounds_all:true`（L12632–12634、L21514–21516）；`grep "stream_identical_to_native": false` 无命中。逐臂 `n_forward_total == n_rounds`：908/908、990/990、990/990、990/990、882/882、898/898、2325/2325、2400/2400（13m_s0）；896/896、963/963、963/963、963/963、876/876、2310/2310（13m_s1）。全部为 1 次顺序前向/轮。含对抗臂。

**C3 · V2 负对照 PASS（复算 Wilson 与量级）**
`adv_random_k4` α₁=0.018495（43/2325），Wilson 95% 复算 = [0.013759, 0.024819] ↔ 落盘 [0.0138, 0.0248]（L12647–12651），上界 <0.06；`adv_bottomk4` α₁=0.0（0/2400），Wilson 上界 = 1.96²/(2·2400·(1+1.96²/2400)) ≈ 0.00160 ↔ 落盘 [0.0, 0.0016]（L12661–12665），≤0.02。二者均 <0.05 且 `verdict_breach:false`。均匀随机假接受率 `pseudo_accept_4_over_V` = 4/206 = 0.019417，与实测 0.018495 同量级 ✓。门禁是真验证。

**C4 · V3 退化等价 PASS**
`single_k1_ctl` 的逐 prompt `n_rounds` 表 (82,73,84,94,85,86,79,86,81,82,78,80) 与 `repro_combo_pw` 完全相同（`run_log.txt`），α₁/τ 逐位相等（0.800000/1.432323；13m_s1 0.845275/1.503634），`dup_k4` 三者在落盘上完全同值（L5036/5046）。K 行机制 bookkeeping 无误。

**C5 · V4 算得对（复算 Δα₁）**
13m_s0：`aggr_topk8` 0.978458−0.931718 = **+0.046740** ✓；`aggr_union_heads` 0.951002−0.931718 = **+0.019284** ✓；13m_s1：0.990868−0.957589 = **+0.033279** ✓。配对 CI 落盘 13m_s0 [0.029604,0.062894]/[0.004336,0.035441]、13m_s1 [0.017135,0.048628]，与 `analysis.md` 表逐位一致；`ci_has_zero:false`、`abs_delta_le_0p014:false` ⇒ 按 README §4「CI 含 0 **或** |Δα₁|≤0.014 才判饱和」判**未饱和**，与 README 规则一致。（bootstrap 具体 CI 端点无法独立重算——无 shell；我核的是 Δα₁ 点值与 `paired_diff` 口径闭环。）

**C6 · V5 算得对（复算台账与边际）**
- `R_K（2640 口径）=3K/(1+τ)`：1.2334 / 2.3547 / 4.5137 / 8.7725 ✓（我按 τ=1.432323/1.548148/1.65859/1.735828 手算一致）。
- `R_K（README 原式）=9K/(1+τ)`：3.7002 / 7.0640 / 13.541 / 26.3174 ✓。
- `τ_K/K` 1.432/0.774/0.415/0.217、`τ_K/τ_oracle`（τ_oracle=1.767202）0.811/0.876/0.939/0.982 ✓。
- 边际 `Δτ/ΔK`：0.115825 / 0.055221 / **0.019309**；阈值 = 0.5×0.115825 = **0.057913**；`k4_to_k8_delta_tau_per_delta_K`=0.019309 < 0.057913 ⇒ `k4_to_k8_worth_it:false` ✓，与 README §4 V5 预登记规则逐字相符。13m_s1 同判（0.014807 < 0.05446，`worth_it:false`，L21677–21682）。
- K 同族取样**未跨分布**：`single_k1_ctl`=`combo_pw`草稿=argmax(q)（`spec_tree.py:153` `first_slot_distribution("combo_pw")` 返回 `pw_mix(nodes)["q"]`），`tree_qpw_k2`=q top-2（`spec_tree.py:82,175`），`repro_topk4`=q top-4，`aggr_topk8`=q top-8 ⇒ 四点是**嵌套同族**（top-1⊂top-2⊂top-4⊂top-8）。K=2 取 2640 指针合法，因为 C1/C4 已证两 harness 逐位等价。
- `aggr_union_heads`（K 均值 3.163448，非整数）**未被误用**：不在 V5 K 族内（L12797–12814 只有 1/2/4/8），V4 也未用其 K。

**C7 · 口径冲突核查（重要）—— 结论：README 原式确为因子 3 笔误，worker 处置恰当，判词未被污染**
- 2640 落盘口径逐字为 `R_equivalent_flops = round((mean_k or 1.0)*(1+gamma)/(1+tau), 4)`（`2640/run.py:333`），即 `R = B_verified·(1+γ)/(1+τ)`，B_verified=每轮行数。对本实验 K 行平铺，B_verified=K ⇒ `R = 3K/(1+τ)`。
- README §4 原式 `R_K = 3K·(1+γ)/(1+τ_K)` = `9K/(1+τ)`：其中 `3K` **已经**是「每轮主干位置数」（README §4 V5 自己列出 `3K`），再乘 `(1+γ)=3` ⇒ **重复计入位置因子**，恰为 2640 口径的 **3 倍**。这是出题方（axis6-lead）的笔误，**不是** worker 的错。
- README **§8 已自己裁定**（L148–158，2026-10-14T09:51:48 记录）：「本文件原文写 `R_K = 3K·(1+γ)/(1+τ_K)`…正确式 = `K·(1+γ)/(1+τ_K)=3K/(1+τ_K)`…该笔误是出题（axis6-lead）的错，由 WP worker 在开跑前发现并拒绝擅改」。⇒ 与我独立复算一致。
- worker 的处置（两套并列、不擅改口径、上报）**恰当**：`results.json::criteria.V5_cost_ledger.R_K_2640consistent` 与 `R_K_readme_literal` 并存，`R_formula_note` 明写相差因子 (1+γ)=3。
- `analysis.md` 判词**未被污染**：`k4_to_k8_worth_it` 只依赖 τ（直接测量量），`analysis.md` 人话摘要里的「4.5 倍→8.8 倍」用的是 2640 口径（4.5137→8.7725，L1297 与 L6185），即**已被 §8 确认的正确式**。

**C8 · V6 失败模式目录（机器可核）**
- (g) `window>0`：源码确凿 `raise NotImplementedError(...仅实现在 window<=0/anchors==0...)`（`2640/spec_tree.py:263–266`），落盘 `arm_failures.win64_tree_k2.error="NotImplementedError"`（L12596–12601），单测 `test_win64_tree_not_implemented` 存在 ⇒ 登记诚实（是「引擎未实装」，非「未测得读数」）。
- (f) `n_mtp=3` 上限：README §4 V6(f) 与 `analysis.md` 均登记为引擎改动需求、**本实验不做**，`results.json` 未声明任何相关臂 ⇒ 诚实。
- (a)–(e) 均有对应臂/变异（`dup_k4`、`adv_*`、M1/M2/M3）。未发现漏项。
- `53m_s0` **未测**如实登记（`results.json.checkpoints` 只有 13m_s0/13m_s1；`analysis.md` 给下一步最小实验 + 成本）。

**C9 · 越界核查（用 `watchdog_diff`，非 shell）**
工作树相对 reviewer-launch HEAD：唯一**已跟踪改动** = `.scratch/63-axis6-predict/TICKET.md (+42)`；其余全为**新增未跟踪**。**无** `pathlm/**`、`tools/**`、`docs/**` 改动；**无** `2640-tree-conf-13m/**` 任何改动；**无**其它轴的已跟踪改动。变异补丁三处原文片段在 `verify_gate.py` 各**唯一**出现（L156 / L161 / L168），可唯一定位 ✓。

**C10 · 禁墙钟**
`grep -i "wall|speedup|\"ms\"|seconds|tok_s|wallclock|elapsed|\"toks\""` 在 `results.json` **零命中**；`timing_measured:false`、`timing_note` 明确。GPU 实耗只在 `run_log.txt`（401 s）。

---

## Findings

### Finding N1 — normal：`results.json::config.budget_min=25.0` 与 README 预登记硬预算 `≤15 min` 冲突
- 位置：`2650/results.json` L32（`"budget_min": 25.0`）vs `2650/README.md` L7（`budget: "GPU ≤15 min（硬时间盒）"`）。
- 被违反判据：README 前沿 `budget` 预登记（硬时间盒）。`run_log.txt` 首行也记 `budget_min=25.0`；TICKET B3 写「WP 6.2.3 worker（≤25 min）」。
- 复现：`grep budget_min 2650/results.json`（=25.0）；`read 2650/README.md` L7。
- 影响：实测整程 401 s（≈6.7 min）**未超 15 min**，故**不影响任何读数**；但落盘参数与契约文字不一致，属记录/契约层面缺陷。
- 最小修法（axis6-lead）：把 README 前沿 `budget` 改为与实际一致（如 `GPU ≤15 min 实测 401 s；运行参数 budget_min=25` 或直接改为 25），或让 `results.json` 的 `config.budget_min` 记 15。二选一，勿两处并存。

### Finding N2 — normal：README §4 V5 错误原式与 §8 更正并存 = 同一事实两个家
- 位置：`2650/README.md` §4 V5（原式 `R_K = 3K·(1+γ)/(1+τ_K)`）与 §8（L148–158 更正为 `3K/(1+τ_K)`）同时存在且冲突。
- 被违反判据：`AGENTS.md` §2.1「每类事实恰好有一个家…同一事实若在两处冲突：以权威处为准，另一处删除或改为指针」；以及「事实文档正文不带变更记录 / ERRATA」。
- 复现：`grep -n "3K·(1+γ)" 2650/README.md`（§4 仍有）与 `grep -n "更正记录" 2650/README.md`（§8）。
- 影响：数字与判词均未受影响（worker 两套并列，C7 已核）；纯文档一致性。
- 最小修法（axis6-lead，勿由 worker 改）：在 §4 V5 **就地**把 `R_K = 3K·(1+γ)/(1+τ_K)` 改为 `R_K = 3K/(1+τ_K)`，§8 收缩为一行指针或删除；保持一事实一家。

### Finding I1 — info（not-run，逐条给命令）：本上下文无 shell，下列须由主管跑
- `not-run`：`python3 -m pytest …/2650-verify-gate-13m/test_verify_gate.py -q` —— 补齐命令：`cd /home/z/vibe/pathlm && python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/test_verify_gate.py -q`（预期 `11 passed`；`mutation_report.json` 各条 `pytest_summary` 均为 `1 failed, 10 deselected` ⇒ 全集 11 条自洽）。
- `not-run`：`python3 …/2650-verify-gate-13m/mutate.py`（须全抓红 + 逐字节还原 + 报 sha256）。报告侧自洽：`all_caught_red:true`、`all_restored_bitwise:true`、三条 `sha256_restored` 均 = `targets.sha256` = `3d2ee007…c6c224`；但**该目标哈希与当前 `verify_gate.py` 是否仍相等无法在本上下文核**。补齐命令：`sha256sum …/2650-verify-gate-13m/verify_gate.py`（应 = `3d2ee00772995658a40b9c866ba00f4a08e8869465062931d7ce3a4b19c6c224`）。
- `not-run`：`python3 tools/summary.py --check`（若因未提交改动失败，记录不修）。
- `not-run`：V4 配对 bootstrap CI 的端点独立重算（须 2640 的 `paired_diff`/`stat_alpha1` + numpy；本上下文无执行能力）。已核的是 Δα₁ 点值与口径闭环。

### Finding I2 — info：`results.json` 非纯 run 输出（`criteria.V5_cost_ledger` 被事后离线回写，已披露）
- 位置：`2650/recompute_v5.py`（就地把 `criteria.V5_cost_ledger` 重算回写）；`results.json::v5_recomputed_note`。
- 说明：回写只涉派生量（τ/α₁ 逐 prompt 未动，`recompute_v5.py::_arm_sum_from_stored` 仅取 `tau`/`alpha1`），并在文件中披露 + `analysis.md` 注明「修一个 int/str 键 bug，不含重新测量」。可接受，但 README §5 称 `results.json` 为「原始」，措辞与「被事后回写」略有张力。最小修法（axis6-lead）：README §5 或 analysis 加一句「`criteria.V5_cost_ledger` 为离线派生字段」，不改数。

### Finding I3 — info：`imported_2640_sha256.run.py` 缺独立旁证
- 位置：`2650/results.json::imported_2640_sha256` = `{spec_tree.py: 71d98b26…8fb62f, run.py: fba5a16c…b9468}`。
- 核验：`spec_tree.py` 的 `71d98b26…` 在 `.scratch/63-axis6-predict/{PHASE-A-REPORT.md:127, WP-6.2.2-LOG.md:44, REVIEW-1.md:10, REVIEW-2.md:26, TICKET.md:164}` 多处独立记录 ⇒ **交叉核对 PASS**。`run.py` 的 `fba5a16c…` 在 `.scratch/**` 内**无**第二处记录 ⇒ 无法交叉核对。补齐命令：`sha256sum axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/{spec_tree.py,run.py,results.json,cycle_log.jsonl}` 并与 `imported_2640_sha256` 及 `run1-snapshot/README.md`（`results.json=1302b678…`、`cycle_log=d2073370…`）对照。
- 注：`run1-snapshot/README.md` 记录的是 2640 的 `results.json/cycle_log.jsonl/run_log.txt` 哈希，**与 `imported_2640_sha256` 覆盖的文件（spec_tree.py/run.py）不重叠**，故两者之间本无可比项；`spec_tree.py`/`results.json` 的核对需上条命令实际执行。

### Finding I4 — info：`2650/README.md` 的 mtime **不早于** run 开始时间（未放宽判据）
- 位置：`2650/README.md` §8（L148–158）为**跑之后**追加（记录时刻 2026-10-14T09:51:48），而 `run_log.txt` 首行 run 开始 = 2026-10-14T09:27:28 ⇒ README 于 run 后仍被写。
- 核验范围：§8 追加的内容**仅**更正 V5 的 `R_K` 笔误，并明写「只改口径笔误、不动判据」；我逐条比对了 §4 V0–V6 判据与 §3 臂定义，**与 `analysis.md`/`results.json` 实际执行的判据一致，未发现事后放宽**（否证阈值 0.05/0.02/0.014、饱和规则、K=1→2 半阈值规则均未变）。TICKET B3 也记录该笔误与两处时间戳更正。
- 结论：任务要求的「mtime < run 开始时间」这一**形式不变量不成立**，但「预登记不得事后放宽」的**实质意图满足**。我无法直接 `stat` mtime（无 shell）；上述判定依据是 §8 内的实测时刻 + `run_log.txt` 时刻。补齐命令：`stat -c '%y %n' …/2650-verify-gate-13m/README.md …/run_log.txt`。

### Finding I5 — info：工作树含同轮兄弟 WP 的未跟踪文件
- `watchdog_diff` 显示 `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/**` 为新增未跟踪（属 WP 6.3.3-WP1，同一 ticket 的另一 WP）。**不是** WP 6.2.3 的写入，不构成越界；仅提示提交时须按显式路径、勿用 `-A`（`AGENTS.md` §8），且 WP 6.2.3 的路径清单里不得混入 2660。

---

## Merge verdict: **OK with notes**

- 复现硬门 V0、无损硬门 V1、负对照 V2、退化等价 V3 我独立复算/逐位比对**全部成立**；V4/V5 的数值与判词口径正确，未受 README 因子 3 笔误污染；V6 失败模式目录诚实、无漏项；越界与禁墙钟均通过。
- 两条 normal 均为**记录/文档一致性**（`budget_min` 与契约冲突；README §4 与 §8 同一事实两家），**不阻断实验结论**；建议 axis6-lead 定稿前修掉，且这两条都属 axis6-lead 拥有的文件，勿派 worker 改契约。
- 已声明限制：本上下文无 shell，`pytest`/`mutate.py`/`summary.py --check`/`sha256sum`/`stat`/V4 bootstrap 端点重算均 `not-run`，补齐命令见 I1/I3/I4；`mutation_report.json`、`run_log.txt`、`results.json` 之间自洽，但未经我复跑。

---
