<!-- 落盘说明（axis5-lead，ticket 66 Phase B）：复核者（fresh 独立上下文 `reviewer`，非两包 writer）所在会话**无写入工具** ⇒ 判词由其返回、axis5-lead **原样落盘**（未改动其结论、未补写内容）。 -->
> 来源：subagent run `223b4d98-fb37-4f10-a601-9499f56f9f03`；落盘时间：2026-10-14（+08:00）。内容为复核者返回原文。

Since I have no write-capable tool in this session, I return the complete artifact below; the runtime persists it to the configured path.

---

# 复核报告 · 轴 5 WP1（`5.2` = `2651-weight-quant-13m` / `5.3` = `2652-act-quant-13m`）

复核者：fresh 独立上下文 reviewer（非两包 writer）。只读 + 只写本报告；未取 GPU 锁、未改任何产物、未 `git add/commit`。
修订基线：`git rev-parse HEAD` = **`e8f0d4233e45`**（由工具在启动时报告，见下「无法执行的检查项」）。

---

## 0. 对 lead 的处置建议（先给人话结论）

| 包 | VERDICT | blocker / normal / info |
|---|---|---|
| `5.2`（`2651-weight-quant-13m`） | **OK with notes** | 0 / 2 / 3 |
| `5.3`（`2652-act-quant-13m`） | **FAIL** | 3 / 4 / 2 |

**结论强度上限（两包共有）**：单 checkpoint / 单 seed / 单 T=256 / 单数据流 / CPU 设备；**无法核 mtime**（本会话无 exec 工具）、**无法重跑**单测/变异与 `check_tree`/`summary --check`；`watchdog_diff` 不含已提交范围。所以本判词只覆盖「我直接读到的文件内容」，不能宣称穷尽了仓库状态。

**必须修（5.3）**：
1. **补跑 `24×16@seed2` 的 `fp16` 单点（≈82 s CPU）并并列报告** —— 这是 `DISPATCH-FINISH-52-53.md` 步骤 4 明确要求的交付项，未跑；不跑就无法把「`|Δanchor|=8.95e-03` 是批数口径差」从推断变为实测，而该包的**预登记硬护栏②**（`config.json::criterion.must_hold.gate_2_bitwise_anchor`）与 `README.md` §9 失败决策树第 1 条正是以「`fp16` == 锚」为对照是否失效的判据。
2. **改掉 `config.json::eval.note`（及 `results.json::config.eval.note` 的副本、`README.md` §6）里「主表 24×16…与历史锚同口径可比 / 另 4 个窗口用同一 24×16 / 无批量混杂」这三句**：实跑是 main `3×16` + extra `2×16`（`config.json::eval.n_batches=3`、`extra_seed_n_batches=2`；`run_log.txt:15` = `缩批量：main3/extra2/copy1/ind1/var1`），且 `table.md:19` 与 `analysis.md §2②/§5.3` 自己都写的是 3×16。
3. **把 `results.json::instrument.bpc_path.anchor_check_applicable` 由 `true` 改为 `false` 或按实际 `n_batches` 门控**：该字段自己的 note 定义「只有 `window_seed=2`/`24×16` 才与锚可比」，字段却为 true（`run.py:539-542` 只按 `--smoke` 判）。

**可如实登记为局限（不必改数）**：5.3 的 `analysis.md §2②` 把「原因已定」写成定论（实为推断，§6 自认「推断」）、§0/§4 把「迫使回退」的 10% 阈值说成「判据里」（预登记只写「远小于标称」，10% 只存在于 `run.py:682/705`）、预算裁剪超出预登记裁剪条款（`README.md` §7 只授权 extra seeds 24→8 + 砍副腿）且被误述为「依 README §7」、`table.md` 表格因行间空行不渲染；5.2 的 `int4` 「越带」贴边（超出带仅 +0.000904）应在 §0 明写幅度、`dominates` 措辞应说明质量腿是「不可分辨」而非「不劣」。

---

## 1. 包 A（`5.2` · `axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/`）

### 人话结论

**VERDICT：`OK with notes`（blocker 0 / normal 2 / info 3）。** 结论：「int8（per-channel 与 group-32）在本尺度是白赚——字节约减半、配对 Δbpc 落在预登记 0.014 带内；int4 group-32 越带（`max|配对Δ|`=0.014904），按契约判『本规模有损、回退 int8』」。我核了它的横轴、字节账、配对 Δ、门禁与活性，**没有找到会作废结论的错误**；但它有两处「记录/呈现与世界不一致」的毛病（前者是执行记录字段与设备不符，后者是主表在 Markdown 里根本不渲染），以及三处措辞/呈报强度问题。

### Correct（已核对为真）

- **单因素隔离成立**：`fp16` 臂不装补丁（`results.json::instrument.patch_reversibility.fp16.logits_changed_when_patched = false`），三个量化臂「装上改 logits=True / 拆掉逐位还原=True / 无残留=True」；`gates.*` 12 模块 789504 参数被逐位证明为死权重（`zeroed_logits_bitwise_equal=True`），只进字节账不进质量读数 —— 这挡住了「开关开着、部分臂没接」与「把死权重当质量证据」两类混淆。
- **预登记 vs 事后调整**：`config.json::eval` 与实跑口径一致（`run_log.txt:1` = `窗口 = 24×16 @ seed 2；配对窗口 = [2,3,4,5,6]`），判据带 0.014 来自 `docs/research-schema.md §12.1` 的预登记常数而**不是**本批 `max|Δ|` 反推（不构成循环），5 个配对窗口各 8×16（`config.json::criterion.bpc_band.this_batch`）满足标准口径 ⇒ 在力带 0.014。
- **字节账与实现一一对应**：`int8_per_channel` 标尺字节/参数 0.005515 ⇒ Σparams ≈ 1.337e7，与 `8×report` 的 group 账（0.0625 = 2·417792/N ⇒ N ≈ 1.337e7）以及 `run_log.txt:12` 的 `params=13536768` 自洽；`scales_match_analytic` 全 True、`max_abs_err ≤ 0.5·s_max` 违例 0 个矩阵（G3+G4 全 True）。
- **判据带适用性与一致性**：5 窗口、配对 Δ（同窗同批、同设备）、取 `max|Δ|` —— 与 `5.1` 的 C1 声明（0.014、配对 Δ、≥5 窗口）一致；召回走 Wilson 不借 0.014 带（`config.json::criterion.recall_rule.no_dimension_borrowing`），率类未跨量纲借用。
- **成本口径**：`project.yaml::cost_model` 是纯等效理论量（无墙钟/吞吐/日期）；墙钟只出现在 `run_log`/`results.json::cost`（`wall_seconds=1670.41`）与 `README.md` §8 的过程读数 —— 符合它自己 `criterion.forbidden` 的豁免。判据里 0 条墙钟。
- **否证路径被真的走查**：否证④（int4 越带 ⇒ 回退 int8）走完并落到 §0；否证②（字节账对不上 ⇒ 不出数）由断言硬化；`int4_per_channel` 未测被明确记为未测（`results.json::notes.int4_pc_not_measured`，不冒充）。
- **变异电池非恒真**：M1..M9 全部 `caught_red=true` 且每条给了具体失败点（如 M6「还原不拆补丁 ⇒ 补丁没拆掉」、M9「fp16 臂不该装任何补丁」），是可变红的断言而非「读数透传」式循环论证。
- **范围纪律**：`watchdog_diff` 显示这两个包的全部文件都是**新增未跟踪文件**（`axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/**`），工作树内 **`tools/**` 零改动**、`pathlm/**`、`tests/**`、`docs/**`、`train_*.py` 零改动，`5.2/project.yaml` 与 `5.3/project.yaml` 零改动（工作树内仅 `5.5`/`5.6` 的 `project.yaml` 被轴 lead 改，不属本包）。

### Finding（normal）

- **N1（normal）· 主表在 Markdown 里不渲染**。位置：`table.md:8-9`（表头/分隔行后紧跟空行，之后每两行数据之间也有空行，同样模式在 `:23-24`、`:154-155`、`:167-168`、`:190-191`）。证据：`table.md` 第 8 行 `|---|---|…|`、第 9 行空、第 10 行才是 `| \`fp16\` | …`（`grep '^\|---\|---' table.md` 与逐行 read 一致）；对照 `analysis.md` 的同名表**行间无空行**、可正常渲染。违反：交付物应可被人类直接读（`AGENTS.md` §7 / `SCHEMA.md` 表纪律）。最小修法：删掉生成器里的空行（只影响 `table.md` 渲染，不改任何数字），必要时 `run.py --charts-only` 重画。
- **N2（normal）· 运行记录里的设备字段与实跑不符，且契约副本仍以 GPU 口径措辞**。位置：`results.json::config.eval.device = "cuda"` / `config.json::eval.device = "cuda"`，而 `results.json::caliber.device = "cpu"`、`run_log.txt:8` = `[口径] device=cpu … 不取 GPU 锁`；`README.md` §8（GPU 墙钟预算 + §20 取锁/释放记账）与 §10 失败决策树①（「`fp16` 臂不逐位等于 stock 锚 1.483569 ⇒ 本实验作废」）仍是转 CPU 前的文本。违反：`docs/research-schema.md` 要求 resolved config 反映实跑口径。严重性下调理由：`results.json::caliber.cross_device_anchor.in_force=false`、`instrument.apply_anchor_check=false` 已显式声明「跨设备不构成判据」，而**数值上 G2 反而按字面满足**（`bpc_of=1.483569`，`|Δ vs 锚|=2.70e-07 < 1e-6`，两条 bpc 路径逐位相等）。最小修法：在 `eval` 里补 `device_resolved: "cpu"`（或在 `caliber` 里注明 device 由 CLI 覆盖），并在 `README.md` §8/§10 加一句「本轮改为 CPU 口径（ticket 66 §13bis F5），本节的锁/墙钟条款不适用」。

### Finding（info，报告级）

- **I1（info）· `run_log.txt:26` 的 `G2=True(Δ锚=0.00e+00)` 标签歧义**：`0.00e+00` 是两条 bpc 路径之差（`masked_bpc − bpc_of`），不是与锚之差（实为 `2.70e-07`，见 `results.json::instrument.fp16_identity.abs_diff_vs_anchor`）。读者可能误读成「与锚完全一致」。修法：该行标注为 `Δ跨路径`。
- **I2（info）· `dominates` 措辞强于定义**。位置：`analysis.md §0`「`int8_per_channel` … **支配** fp16」，定义在同段术语行「`dominates` = 字节不更多、且质量代价在带内不更差」；`results.json::notes.*.dominates_fp16=true`（`int8_*`）/`false`（`int4_group32`）。问题：三个 int8 臂的 Δ 均值/主窗 Δ 都是**正的**（+1.1e-4 / +7.7e-5），「不更差」严格读为假，实际是「不可分辨」。定义已写明、判读一致，故不算隐藏夸大，但「支配」一词会让人读成 Pareto 严格改进。修法：改为「字节更省 **且质量不可分辨**（不构成严格支配）」。
- **I3（info）· int4「越带」的幅度未在主结论里给出**（贴边问题，见下 §1.1）。`analysis.md §0` 只说「掉出噪声带」，未写「超出 0.000904（+6.5%）」也未写「Δ 均值 0.013946 恰落在带内」。修法：§0 补一句幅度与聚合口径（见下）。

### §1.1 独立判定：`int4_group32` 的「越带」是否稳健

**结论：稳健（成立），但应在报告里标明它是「刚越带」且依赖预登记的 max 聚合。** 证据与推理：

1. **逐窗符号一致**：`results.json::paired_delta.per_arm.int4_group32.by_seed = [0.011115, 0.014904, 0.014196, 0.014851, 0.014665]`（`analysis.md §2` 同）⇒ **5/5 同号为正**，且 **5 窗中有 4 窗单独 > 0.014**（只有 seed 2 的 0.011115 在带内，为 0.79×带）。判定不是被单一窗口拉过去的。
2. **5 窗满足标准口径**：`config.json::criterion.bpc_band.this_batch = {windows:5, seeds:[2,3,4,5,6], n_batches_per_window:8, standard_met:true}`，与 `run_log.txt:28` 的实跑配对行一致；与 `5.1` 的「配对 Δ、≥5 窗口」口径同构。
3. **无「用 `max|Δ|` 当带」的循环**：带值 0.014 来自预登记常数（`provenance: docs/research-schema.md §12.1`），不是本批统计量。
4. **量级远在配对噪声之上（我的推断，标注来源）**：同 run 内两个 int8 臂的逐窗配对 Δ 极差为 3.66e-4 / 2.10e-4（⇒ 逐窗配对 Δ 的散布约 1–2e-4 量级），而 int4 的 4 个越带窗口超出带 +2e-4…+9e-4（约 1.3σ…6σ）。以「真实 Δ 恰好 = 0.014」为零假设，5 窗取 max 的期望上偏约 +1.2σ ≈ +2e-4 < 观测的 +9e-4 ⇒ 观测 max 高得不是带边抖动可以解释的。
5. **须声明的两点**：(a) 判定用的是预登记的 `max` 聚合，若改用 Δ 均值（0.013946）则落在带内 ⇒ 「越带」是规则内的确定判定，但幅度只有带的 1.07 倍；(b) 5.2 的窗口均为 8×16，故不存在 5.3 那种窗口间批数不等的问题。

---

## 2. 包 B（`5.3` · `axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/`）

### 人话结论

**VERDICT：`FAIL`（blocker 3 / normal 4 / info 2）。** 数字本身我认为**是对的、四臂齐全、无伪造**（A8 省 49.8% 激活字节、`max|配对Δ|`=0.00353 在带内；A4 崩坏到 bpc 5.04、`max|Δ|`=3.598 越带、copy 落地板；W8A8 不越带）；**但「判据能否生效」的那一环没有过**：它自己的预登记硬护栏②要求「`fp16` == 历史锚 1.483569（`window_seed=2`, 24×16）」，实跑主窗是 **3×16**，`|Δ|=8.95e-03`；而派单明确要求补跑的 `24×16@seed2` 单点（≈82 s）**没有跑**。加上预登记文件里那句「主表 24×16…与历史锚同口径可比」与实跑不符，这份包现在**不能**以「判决」形式落地，先修这三处（最小成本 ≈ 2 分钟 CPU），A4 的越带结论不受影响。

### Correct（已核对为真）

- **主因子臂齐备、组合臂显式分列**：`A8`（只改激活）与 `W8A8`（激活+权重 int8）都在表里，且 `table.md` 表 1 有 `组合?` 列、权重字节单独一列（12.06 MiB）与每 token 激活字节分列、`analysis.md §3` 明说「它与 A8 的差 = **权重因子**，不计入激活因子」——没有被权重因子混杂。
- **实现层等值不是自证**：`AqForward(fp16)` 与既有仪器 `retention.masked_logits` 逐位相同（`instrument.replicated_vs_library.vs_masked_logits_bitwise_equal=true`，probe 批），`bpc_of` 与独立路径 `masked_bpc` 逐位相同（`1.4746206610207933` 两侧一致）；四个量化臂 `changed_frac`=0.997/0.998/0.997、`fp16`=0.000（活干预，非死开关）。
- **字节账与实现一一对应**：每层 4 张量 `len ∈ {256,256,256,1536}` ⇒ 2304 元素/token/层、27648/token、fp16 55296 B/token、A8 27648+96=27744、A4 13824+96=13920（`results.json::bytes_account.activation`），标尺数 4×12×16×256=196608 与解析式一致；`W8A8` 权重 12.58M×1 B + 36864×2 B = 12.65 MB = 12.06 MiB 亦自洽。**权重字节与激活字节分列**（未混入同一横轴）。
- **「用哪个带」与实跑一致**：`project.yaml::falsification` 声明「0.014 bpc 带（限配对 Δ、≥5 窗口）」⇒ 实跑就是 5 窗口配对 Δ、0.014；召回只用 Wilson CI，未借 bpc 带。
- **否证条件「离群通道是否迫使部分层回退」被真的做成可失败读数**：`aq_arms.revert_byte_penalty`（`revert_chan_frac_6x` / `saving_eaten_frac`），读数 0.03% / 0.07% ≪ 阈值，且 `README.md §9.5` 事先给出该读数失真的条件（median 触 clamp）并检查过；`outlier_idx`（均值 3.36、最坏 `L01.qkv` 10.96）与 `eff_bits`（8→6.49，4 位→2.36）如实报。
- **变异电池非恒真**：7 个变异全部有非零 returncode 的失败输出（M1 位数定义、M2 per-token 退化、M3 标尺忘除 qmax、M4 标尺当免费、M5 死干预…），并附「源文件逐字节还原」。
- **成本口径**：`project.yaml::cost_model` 只有等效理论量；墙钟只出现在 `run_log`/`results.json::cost`（512.241 s）/`analysis.md §5.3` 的局限段 —— 判据与 `cost_model` 里 **0 条**墙钟/吞吐/日期。
- **范围纪律**：整个 `experiments/2652-act-quant-13m/**` 为新增未跟踪文件；`watchdog_diff` 里 `tools/**`、`pathlm/**`、`tests/**`、`docs/**`、`train_*.py`、其它轴、`5.3/project.yaml` **零改动**（改动只出现在这两个包自己的目录内；`5.5`/`5.6/project.yaml` 由轴 lead 改，不属本包）。

### Finding（blocker）

- **B1（blocker）· 预登记硬护栏②未达，且派单要求补跑的对照点没跑 ⇒「对照成立」未验证**。位置：`config.json::criterion.must_hold.gate_2_bitwise_anchor`（「`fp16` 臂 == 历史锚 1.483569（`window_seed=2`, **24×16**）且 == `retention.masked_bpc`（逐位）」）、`results.json::instrument.bpc_path.abs_diff_vs_anchor = 0.00894833897920666`；证据：`config.json::eval.n_batches = 3`（`extra_seed_n_batches = 2`）、`run_log.txt:15` = `缩批量：main3/extra2/copy1/ind1/var1`、`run_log.txt` 的 `[自检] … |Δanchor|=8.95e-03`。`README.md` §9 决策树第 1 条自己写着「`fp16` 臂若与历史锚…不逐位一致 ⇒ **本实验作废**」。`DISPATCH-FINISH-52-53.md` 步骤 4 明确要求「把 `24×16@seed2` 的 `fp16` 值**也跑一遍并列**（约 82 s）」，`analysis.md §5/§6` 自认「**预算内未能补跑**」。违反：`proof.execution.gate_probe` / `unverified`（未跑的步骤不算通过；未验证的验收不是 PASS）。最小修法：跑 `24×16@seed2` 的 `fp16` 单点（≈82 s，CPU，不需要 GPU 锁）并把两值并列 —— 若得 ≈1.4746，则批数解释被证实、B1 降为「已登记局限」；若得 ≈1.4836，则 3×16 的 8.95e-03 另有他因，A8/A4 的口径须重查。
- **B2（blocker）· 预登记文件里的口径叙述与实际批数不符，且正是「与锚可比」那句话**。位置：`config.json::eval.note`（= `results.json::config.eval.note`，`results.json:113`）与 `README.md` §6：「主表 `window_seed=2` + **24×16** 与 2610/2614/2619/2621/2632 同一窗口序 ⇒ 与历史锚 1.483569 同口径可比；另 4 个窗口 seed 用**同一** 24×16 批量（… 5 个窗口的统计量同口径，**无批量混杂**）」。实测：主窗 3×16、附加窗 2×16（`config.json:102/111`、`run_log.txt:15`），而 `table.md:19` 与 `analysis.md §2②/§5.3` 自己写的是 3×16 —— 同一份产物里两种口径叙述并存。**我的判断（不照抄派单）**：这**构成不可复核的错述**，因为 ① 它与同一文件里的字段直接矛盾（读者无法判定哪个是口径），② 它恰好是唯一会让读者相信「主表 bpc 与历史锚同口径可比」的句子，而实际差 8.95e-03（≈0.64×判据带），③ 「无批量混杂」也被自己的实跑推翻（main 3 / extra 2）。它不改数字，但改「这些数字凭什么可比」的结论。最小修法：把 note 改成实跑口径 + 明写「不与 1.483569 同口径、跨口径不作判据」；若口径是转 CPU 时被改小的，把「原预登记 24×16、实跑 3×16（理由）」写进 note，而不是留着旧句子。
- **B3（blocker）· 机器可读标志与它自己的定义相反**。位置：`results.json::instrument.bpc_path.anchor_check_applicable = true`，同字段 `note` 定义「**只有 `window_seed=2 / 24×16` 的口径下才与锚可比**；`--smoke` 缩小批量 ⇒ `anchor_check_applicable=False`」；`run.py:539-542` 只按 `--smoke` 决定该标志，**没有把 `n_batches` 纳入**。实跑 3×16 ⇒ 该标志按自己的定义应为 false。违反：`falsifiable.gate_probe`（门禁必须能红；这里它在本批事实上无法按定义正确置值）。最小修法：把判据改为 `anchor_check_applicable = (not smoke) and n_batches == 24 and window_seed == 2`（或直接置 false + 注释），重跑 `--table-only` 落盘。

### Finding（normal）

- **N1（normal）· `analysis.md §2②` 把推断写成「已定」**。原文：「与历史锚 1.483569 的差 = 8.95e-03，**原因已定**：口径不同，不是实现缺陷」，论据是「同 run 内 `masked_bpc` 与 `bpc_of` 同批逐位相同」（这只证明评估路径无误）+「跨 run 对照 2621 的 fp16 点（24×16）仍是 8.95e-03，与该解释一致（差的正是批数）」——后者与锚比较是**同一个比较的重述**，无法把设备因子与批数因子分开；`README.md` §9.5 的规定是「若口径相同而仍不一致 ⇒ 查明或如实登记**原因未定**」。同一份 `analysis.md` §6 gaps 又把它列为「把 §2② 的口径差从『**推断**』变成『实测并列』」⇒ 自相矛盾。我的判定：**①② 的答案是「部分如实」**——归因方向（批数不同 ⇒ bpc 不同）是**正确**的（`bpc_of` 是批内 token 平均 CE，批数不同取样子集不同），但「原因已定」超出了证据；本仓另有**独立**支持它的证据未被引用：包 A 在 **CPU 24×16@seed2** 上得 `1.483569`（与 GPU 锚差 2.70e-07，见 `…/2651-…/results.json::instrument.fp16_identity`）⇒ 设备不是原因，批数几乎必然是原因。最小修法：§2② 改「原因**推断**为批数口径不同（并列证据：`2651` 在 CPU 24×16 上与锚 6 位小数一致；直接证据待补跑单点）」，并删掉 §2② 与 §6 的自相矛盾表述。
- **N2（normal）· 10% 阈值被说成「判据里」，实际只存在于实现里**。位置：`analysis.md:24`「远小于**判据里**「迫使回退」的 10% 阈值」、`analysis.md:125` 同；`JOURNAL.md:107` 同。核查：`config.json::criterion.verbatim_contract`（`config.json:150`）与 `project.yaml::falsification` 只写「使省幅**远小于标称**」，**没有任何 10%**；10% 只出现在实现 `run.py:682`（`saving_eaten_frac > 0.10`）与它自己生成的 `results.json::outlier_verdict.criterion`（`run.py:705`）。违反：判据不得事后由实现自定（`precedence` / `proof.execution.only_authorized`）。实质无影响（0.03%/0.07% 在任何合理阈值下都「不迫使回退」），但这句陈述不准确。最小修法：把该阈值写进 `config.json::criterion`（作为对「远小于标称」的显式操作化）或在 analysis 里标注「实现层操作化（post-hoc operationalization）」。
- **N3（normal）· 预算裁剪超出预登记条款，且被误述为「依 README §7」**。预登记条款（`README.md` §7 / `config.json::cost.budget_cut_rule`）：「先砍**额外窗口 seed** 的 `n_batches`（24→8），再砍**副腿**（induction / var_recall），保 `fp16`/`A8`/`A4`/`W8A8` 四臂、clean bpc 主腿与全部判据线」。实跑（`run_log.txt:15`）：main 24→**3**、extra 24→**2**、copy 2→**1**、副腿 16→**1** —— 主窗与 copy 主腿的削减**不在**授权范围内。`analysis.md §5.3` 写「依 README §7 的预算条款把主窗 24→3 批」属于误述。违反：`change.outdated`/`disclose_changes`（协议改动要按改动登记、如实说明改的是什么）。它已作为「局限」披露（值得肯定），但**来源**记错、也未见 ticket 层面的 plan-change 记录（ticket §19.2 只把 note 不符登记为「复核核对项」）。最小修法：在 `analysis.md §5.3` 与 ticket 里写清「本轮裁剪超出 README §7 授权范围，改动项与理由为 X」，或恢复主窗 `n_batches` 后重测。
- **N4（normal）· `table.md` 表 1–4 在 Markdown 里不渲染**。位置：`table.md:10-11`（分隔行后空行）、`:27-28`、`:58-59`、`:73-74` 同样模式；对照 `analysis.md` 同名表行间无空行。违反：交付表须可被人类直接读。最小修法：去掉生成器里数据行间的空行并 `--table-only` 重画。

### Finding（info）

- **I1（info）· 横轴的覆盖面没写在人话摘要里**：「激活字节」只覆盖每层 4 个 matmul 的输入（27648 元素/token）；注意力分数矩阵（12×8×256×256 ≈ 6.3e6 元素/批，与已计 7.08e6 同量级）与 logits 不在账内，`analysis.md §0` 开头「把模型前向里的『中间激活』从 fp16 压成 int8」读起来像覆盖了全部中间激活（同段随后给了 27648 的限定，`config.json::axis_definition.tensor_set` 也列明）。对比：包 A 报了权重覆盖 93.1%。最小修法：摘要加一句「本账只含每层 4 个 matmul 的输入，不含注意力矩阵与 logits」。
- **I2（info）· 口径注释被手工追加进生成物**：`results.json::caliber_note`（`results.json:8268-8279`，含 `open_defect`、`grep` 提示）是在 `run.py` 落盘后追加的说明，而 `2851`/`2652` 的 `results.json` 其他部分都是机器生成；一旦有人重跑，这段会消失或被覆盖，且「生成物不得手改」（`change.write_only`）要求它有自己的家。内容本身**准确**（与我在 `config.json:102/111/113`、`run_log.txt:15` 核到的一致）。修法：把这段移进 `analysis.md §2②` / `JOURNAL.md`，或让 `run.py` 原生生成它。另注：包 A 的 `JOURNAL.md` 末段既说这是「**只读**发现，未越界」又记「已**在** 5.3 的 `results.json` 里**追加**一条 `caliber_note`」——两句话互斥，读者无法判断到底写没写（写本身在收尾派单的授权目录内，但表述要一致）。

### §2.1 独立判定：口径缩口是否影响判据有效性（派单补充 3③）

**结论：判定规则仍然有效（配对性质不因批数而破坏），但「不越带」一侧的结论强度下降、锚可比性失效；「越带」一侧不受影响。** 理由：

1. **配对有效性不受批数影响**：判据量是**同一 `window_seed` 下同批数据**的两臂 bpc 之差（`config.json::criterion.primary_band_caliber`），`batch()` 的取数只依赖 seed/窗口、与设备与批数无关；两臂在同窗用同批 ⇒ 窗口/子集效应在差分中被消掉。所以 3×16 与 24×16 都是「合法的配对窗」，5 窗 ≥5 满足标准口径 ⇒ **规则本身在力**。
2. **噪声放大 ⇒ 对「不越带」是保守方向（但对判据的灵敏度是削弱）**：主窗 3×16=48 条序列、附加窗 2×16=32 条，而预登记口径是 24×16=384 条 ⇒ 逐窗配对 Δ 的噪声更大。因此 `A8`/`W8A8` 的「不越带」只能读作「**在本缩口径下未能分辨**」，不能读作「已证明无代价」——`analysis.md §5.3` 已如此声明（这是本包做得对的地方）。
3. **「越带」一侧稳健**：`A4` 的 Δ≈3.48–3.60，是带的 ≈250 倍，任何合理的批数噪声都不可能造成；`W8A8` 的 0.002955 也远离带。
4. **被缩口破坏的不是判据，是「口径声明」与「对照前提」**：`config.json::eval.note` 的「24×16、与锚同口径、无批量混杂」三句都不成立（`main 3` / `extra 2`），且硬护栏②的成立条件（24×16）没被满足、对照点没跑 ⇒ 「这些数字与历史锚处在同一把尺子上」这件事**没有被验证**。这正是 B1/B2/B3 的由来。
5. **一个附带的同口径瑕疵**：5 个窗口本身**不同口径**（seed2 = 3×16，seed3–6 = 2×16），而 `config.json::eval.note` 声称「5 个窗口的统计量同口径，无批量混杂」⇒ 该句也假。对 `max|Δ|` 判定影响很小（各窗内配对仍成立），但「同口径」的说法须删。

---

## 3. 我**无法执行**的检查项（不得当作通过）

本会话**没有 exec/shell 工具**，以下项只能列为「无法执行」，我未用猜测替代：

| # | 项 | 状态 |
|---|---|---|
| 1 | **mtime 判「预登记早于 `results.json`」**（派单检查 1、契约要求两者判据线一致） | **无法执行**。替代：我核到 `config.json::criterion` 与 `README.md` §4/§5 的判据线文本一致、且 `run_log.txt` 开头的 `[cfg]` 行早于读数；但「文件写入时间」这一事实我无法核 —— 对 5.3 而言这尤其重要（`eval.note` 与 `n_batches` 的时间先后决定它是「陈旧文字」还是「事后改字段」）。 |
| 2 | `python3 tools/check_tree.py` / `python3 tools/summary.py --check` | **无法执行**（两包 `JOURNAL.md` 各记 EXIT=0，属自报，未独立验证）。 |
| 3 | 重跑 `selftest_wq.py` / `selftest_aq.py` 与变异电池 | **无法执行**。我只审了 `mutation_report.json` 的**形式**（每条变异 → 一个非零 returncode 的失败测试 + 具体失败点），未复现 9/9、7/7。 |
| 4 | `sha256sum` 核 trunk/源文件哈希、`git status --short` / `git diff --stat` / `git log` | **无法执行**。替代：`watchdog_diff`（工作树 vs HEAD `e8f0d4233e45`，**不含已提交范围**）显示两包目录全为新增未跟踪文件、`tools/**` 等共享面零改动。**若这两个包的内容在更早的 commit 里被提交过，我这一份判词覆盖不到** —— 请在收口前由能用 git 的一方补 `git log --stat` 一次。 |
| 5 | CPU 上跑任何一步复现（如 `run.py --table-only`） | **未做**（会写产物；按边界只读）。 |
| 6 | 两包 `README.md` 的「能力面分析」（`SCHEMA.md §22.8`） | **未在派单的 9 条内**，未查（只顺带看到 5.2 `JOURNAL` 记了两次自查 bug；5.3 未见同类记录 —— 但这属另一条线索，不在本契约检查项内，故不计入 findings）。 |

另外两条**边界说明**：① 本报告只覆盖我读到的文件（两包目录内 12+11 个文件与两份 `project.yaml`、`DISPATCH-FINISH-52-53.md`、`TICKET.md` 的相关段），**未遍历仓库**；② 工作树里存在其它轴的未跟踪/已改文件（其它轴并发工作），与本次复核对象无关，未列入 finding。

---

## 4. 逐条对照派单 9 项的判定汇总

| 派单检查项 | 5.2 | 5.3 |
|---|---|---|
| ① 预登记 vs 事后调整（判据线一致、口径变更如实记） | 通过（口径变更在 `caliber`/`analysis §5.2` 显式声明；契约副本仍用 GPU 措辞 ⇒ N2） | **不通过**（B2：`eval.note` 与实跑不符；B1：硬护栏②未达） |
| ② 单因素与实现等值（活干预、`fp16` 逐位复原） | 通过 | 通过（等值链有仪器级证据） |
| ③ 字节账与实现一一对应、跨口径不借用 | 通过（解析式与参数数自洽） | 通过（激活/权重分列；I1 只涉表述覆盖） |
| ④ 判据带适用性（配对 Δ、≥5 窗、无单点外推） | 通过（见 §1.1，稳健但贴边） | 规则在力、配对有效；但锚可比性未验证（§2.1 判定） |
| ⑤ 否证路径被走查（含离群回退） | 通过 | 通过（N2 只涉阈值出处） |
| ⑥ 成本口径（`cost_model`/判据 0 条墙钟） | 通过 | 通过 |
| ⑦ 结论强度（analysis ↔ results 一致、不外推） | 基本通过（I2/I3 措辞） | 数字一致；N1 把推断写成熟知（部分如实） |
| ⑧ 范围纪律（只改各自 `experiments/**`） | 通过（工作树内；见「无法执行」#4 的限制） | 同左（`results.json::caliber_note` 手改见 I2） |
| ⑨ 变异/单测能真失败 | 形式通过（9/9 有具体失败点） | 形式通过（7/7 有非零 returncode） |

---

## 5. 对 lead 的处置建议（收口清单）

1. **必须修（5.3，≈2 分钟 CPU + 3 处文本）**：补跑 `24×16@seed2` 的 `fp16` 点并并列（B1）→ 改 `config.json::eval.note`（含 `results.json` 副本）与 `README.md` §6 为实跑口径 + 明写不与锚同口径（B2）→ 修 `anchor_check_applicable` 的判据（B3）→ 重生成 `analysis.md`/`table.md`（顺带修 N4 的表格空行与 N1 的「原因已定」措辞）。**B 系列不修之前，`5.3` 不应写成 `judged` 结论**（`README.md` §9 第 1 条按字面会判「本实验作废」）。
2. **可如实登记为局限（不必改数）**：主窗/附加窗缩口径与「A8 不越带 = 未能分辨」（5.3 已自登，补上 N3 的真实来源即可）、`A4` 单 seed 崩坏点、副腿在地板、`int4_per_channel` 未测、覆盖张量集不含注意力矩阵与 logits。
3. **建议顺手修（5.2，非阻塞）**：`table.md` 空行（N1）、`eval.device` 与 README §8/§10 的口径措辞（N2）、`run_log` 的 `Δ锚` 标签（I1）、`dominates` 措辞与 int4 贴边幅度的明确呈报（I2/I3）。
4. **不要做的事**：不要因为「`fp16` 的 CPU bpc 不等于 GPU 锚」而要求 5.2 或 5.3 逐位对齐 1.483569（跨设备在本批不构成判据，两包都已声明）；也不要把 5.3 的 3×16 当成「另一套结果」而丢弃——它与 5.2 的 24×16 读数各自内部一致，缺的只是那条 82 秒的并列点。
