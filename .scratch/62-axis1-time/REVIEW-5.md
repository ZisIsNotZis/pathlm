# REVIEW-5 · 轴 1 时间轴第二轮（实验 `2639` 完整机制重测）独立复核

> 复核者：fresh、只读、**无 shell/执行工具**（可用能力仅 读文件 / grep / find / ls / 工作树 diff），
> 未参与本线任何实验、文档或前几轮复核。
> 复核对象：`axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/` 全套产物
> + `axes/1-time/1.1.4.1-explicit-memory/project.yaml` 改动；HEAD = `c77cc0b`（工作树干净，见 §6）。
> 复核者声明（重要）：本报告是唯一写入物，本复核体**没有写文件工具**，故报告以最终回复交付，
> 由运行时落盘到 `.scratch/62-axis1-time/REVIEW-5.md`。

---

## §0 人话摘要（先读这段）

**判词：FAIL，但范围极窄——科学主张 PASS，阻断项只有 1 条、且是「记录与文件不一致」，不是数字或方法。**

1. **主结论成立（我独立复算过，不是复述 writer 的 prose）**：完备机制点在等字节（393,216 B）上
   `win32` 1.595823 vs `carrypl16` 1.527595 ⇒ **+0.068228**，与 `analysis.md`/`project.yaml`
   一字不差；gap-closing `r = 0.100515 / 0.125421 = 0.8014`（80.1%）；完备度增量
   `1.624979 − 1.527595 = 0.097384`；换窗口序 +0.071912 同号；摊平口径 +0.100265 同号；
   前沿下包络里 `carrypl16` 确实在包络上、`win32` 确实被支配。**我未能推翻这个主张**（§1–§4、§7）。
2. **最致命的疑点不是数字，是两条记录说了没发生的事**：`c77cc0b` 的 commit message 第 7 行与
   `project.yaml` 的 `change_log`（`2026-10-14T09:10:00` 条）都写着「**experiments 新增 2639**
   （analysis.md/results.json @ccdc91a）」，但 `project.yaml` 的 `experiments:` 列表（第 19–46 行）
   **根本没有 2639**（`grep 2639` 只命中 `gap` / `WP*.evidence` / `roadmap` / `change_log`）。
   这正是本轴 REVIEW-1…4 反复抓的同一形态（声称 > 实际落地），按 `AGENTS.md` §8 应**立刻补更正提交**。
   —— 修法 1 行：在 `experiments:` 补 `2639/analysis.md @ccdc91a`、`2639/results.json @ccdc91a`
   （或删掉两处「experiments 新增 2639」的字样）。
3. **我自己的最大局限（必须说清）**：本复核体没有执行工具，任务书点名要「不得只读 prose」的 6 项
   执行类检查（sha256sum / pytest / `summary.py --check` / GPU 重跑 `run.py` / 自写因果探针 /
   `git diff d3d33eb..HEAD --stat`）**我一条也没跑**（§末给出确切命令）。因此本报告对「主判据」
   的核验强度 = **读盘 + 手工复算 + 交叉一致性**，不是执行级证据。
4. **真实残余不确定性（writer 已自认、我独立确认成立）**：单一 seed；完备臂比纯窗口多 264,296 参数
   （+1.95%）的等参数对照未做；E1 把「逐层状态」与「写入源取自本层 hidden」绑在一个开关 ⇒
   0.0974 bpc 不可归因到哪一半。这三条都不足以推翻等字节结论（§7 ③ 量化了）。

---

## §1 主判据复算（手工，逐位与 `results.json` 对齐）

**做法**：直接读 `results.json` 的 `rows[*].bpc/total_bytes/state_bytes/k_eff`（`:1470–1690`）、
`frontier`（`:1710–1755`）、`location`（`:1756–1830`）、`key_readings`（`:1832–1845`）、
`gap_closing`、`sigma_within_config_2634`、`reuse_equivalence`（`:1941–2005`），再手算差值。

| 量 | 我的复算 | 落盘键（`2639/results.json`） | 文档值 | 判 |
|---|---|---|---|---|
| `pl_minus_win32` | 1.595823 − 1.527595 = **0.068228** | `key_readings.pl_minus_win32.pl_minus_win32` `:1483` | 0.068228 | ✅ |
| 同字节 | win32 `total_bytes`=393216（`:1577`）；carrypl16 `total_bytes`=393216（`:1641`） | `key_readings.*.same_bytes=true` | 是 | ✅ |
| `Δc`（vs win16_s0） | 1.62811 − 1.527595 = **0.100515** | `location.carrypl16_s0.delta_bpc_vs_win16_s0` `:1817` | 0.100515 | ✅ |
| `Δg`（win16→full） | 1.62811 − 1.502689 = **0.125421** | `gap_closing.gap_win16_full` `:1888` | 0.125421 | ✅ |
| `r = Δc/Δg` | 0.100515/0.125421 = **0.80143** | `location…recovery_ratio_r = 0.8014` `:1818` | 80.1% | ✅ |
| 完备度增量 | 1.624979 − 1.527595 = **0.097384** | `key_readings.pl_vs_carry16_s0.delta` `:1800` | 0.0974 | ✅ |
| chunk 匹配口径 | 1.628940 − 1.527595 = **0.101345** | `key_readings.pl_vs_shared_c256.delta` `:1805` | 0.1013 | ✅ |
| chunk 敏感性 | 1.628940 − 1.624979 = **0.003961** | `key_readings.chunk_sensitivity_shared.delta` `:1809` | 0.003961 | ✅ |
| γ=0.99 残留 | 0.99²⁵⁶ = e^(−2.5728) = **7.63%** | —（分析文字） | 7.6% | ✅ |
| γ 臂 vs win32 | 1.595823 − 1.585531 = **0.010292** | `location.carrypl16d_s0.carry_minus_win32_s0` `:1840` | 0.0103 | ✅ |
| γ=1 优于 γ=0.99 | 1.585531 − 1.527595 = **0.057936** | —（两 `rows` 之差） | 0.0579 | ✅ |
| 41×σ_cell / 4.9×0.014 | 0.068228/0.00166 = **41.10**；/0.014 = **4.873** | `key_readings.pl_minus_win32.{sigma_cell,cross_cell_band}` `:1487` | 41×/4.9× | ✅ |
| 换窗口序 | 1.594147 − 1.522235 = **0.071912** | `robustness_alt_window_seed.pl_minus_win32_alt_seed` `:1926` | 0.071912 | ✅ |
| 窗 16→24 收益 | 1.62811 − 1.599856 = **0.028254**；0.097384/0.028254 = **3.45×** | `rows` | 0.0283 / 3.4× | ✅ |
| 「≈窗口 111」 | 0.068228/(1.595823−1.513325)=0.827 → 32+0.827×96 = **111.4** | `robustness…`/`rows` | ≈111 | ✅ |
| 摊平口径（fp32 脚注） | win48 = 1.595823−(16/96)×0.082498 = **1.582073**，仍 > 1.527595 | `sensitivity` 只在 fp16 口径报数 | 1.582 | ✅（脚注自算，我复算同值） |

**状态字节记账**（判据②/③的代价轴）：`results.json.checkpoints.carrypl16_s0.state_floats_actual
= 98304`（`:410`）× 2 B = **196,608 B**；`run.py:load_arm()` 是从**加载好的模型**上算
`n_heads × head_dim² × (n_layers if per_layer else 1)`（`run.py:86–92`）——即**不是**写死的常量。
`train/pl16/results.json.carry_liveness` 与之一致：`per_layer=true`（`:664`）、`shared_slots=false`
（`:663`）、`carry_chunk=256`（`:661`）、`block_python_class="CarryChannelBlock"`（`:662`）、
`params_added=264296`（`:666`）、`alive=true`（`:680`）。单共享槽对照 `train/c16`：`shared_slots=true`
（`:663`）。K-V 单位 `12×2×256×2 = 12288 B/位置`；win32 = 12288×32 = **393,216**；
carrypl16 = 196,608+196,608 = **393,216** ⇒ 逐字节相等成立。

**复用臂逐位相同**：`reuse_equivalence[*].equal` 10/10 true（`:1941–2005`）；我另**独立**读了
`2638-carry-frontier-13m/results.json` 的 `rows`（`:799` full 1.502689、`:813` win16_s0 1.62811、
`:826` win16_s1 1.629093、`:839` win17 1.624104、`:852` win18 1.619953、`:865` win24 1.599856、
`:878` **win32_s0 1.595823**、`:891` win128 1.513325、`:904` carry16_s0 1.624979、`:917` carry16_s1 1.62664），
与 2639 `rows` **逐位相同** ⇒ 换引擎（E1 默认路径）未扰动基线。

---

## §2 口径诚实性（预登记 vs 执行期补充；数字去向）

**(a) 判据线**：`README.md` §4 表格 + §4a 逐条给出 ①口径一致 ②代价 ③主判据
`pl_minus_win32 = bpc(win32_s0) − bpc(carrypl16)`（正=carrypl 更好）④必要条件（Δbpc>0 + live）
⑤机制 vs 实现 ⑥禁外推，并写明「写下后不得改」；§5 臂表把 `carrypl16_s0` 标为**主判据臂**、
`carrypl16d_s0`/`c16c256_s0` 为**补充臂**且「缺任一个都不影响③的成立（不得事后以『缺 γ/缺 chunk
对照』为由软化判词）」。**我没有能力核验文件 mtime/历史**（无 git、无 shell）⇒「训练启动前写死、
之后未软化」这一点我只能核到**文本自洽 + ticket 时间线自洽**：`TICKET.md` §9.2 记修订在
`07:35–07:39`，`§9.4` 训练起于 `07:40:07`；`2639/README.md` §5 与 `run.py` 的判定线
（`SIGMA_CELL=0.00166`、`if d > SIGMA_CELL → 改善前沿`，`run.py:29–32,~305`）与 ③ 的文字**方向一致**。

**(b) 执行期补充是否加强而非软化** —— 逐条核到落盘：

| 补充 | 位置 | 加强/软化 | 可查证据 |
|---|---|---|---|
| `carry_chunk=256` 显式记账 + **另训 chunk 匹配的单共享槽对照** | `README.md` §2/§5 | **加强**（多一个下界对照 + 敏感性探针） | `key_readings.chunk_sensitivity_shared`、`pl_vs_shared_c256`；`train/c16` 真实存在 |
| `sensitivity` 摊平口径（**明示非判据**） | `README.md` §4a 末条；`run.py` 注释 | **中性偏加强**（同号 +0.100265，且明示「判据一律用冻结口径」） | `results.json.sensitivity.carrypl16_s0`（`:1846–1885`） |
| `robustness_alt_window_seed=7` | 评测脚本内 `run.py:~432–452` | **加强**（换数据流仍同号 +0.071912） | `results.json.robustness_alt_window_seed`（`:1913–1940`）、`run_log.jsonl:30–33` |
| `causality`（必须不变项探针） | `run.py::causality_probe`（`run.py:105–140`） | **加强**（判据 `objective_criteria` 的必须不变项） | `liveness[*].causality`（12 个掩码臂全 `max_abs_logit_diff_prefix=0.0`） |
| 新增两探针是**看过第一次评测后**才加的（如实登记） | `TICKET.md` §9.7 | 诚实（自认后置） | 同上 |

**诚实性瑕疵**：后置添加的鲁棒性检查「同号」只能算「不构成反证」，不能算独立确认——writer 在
`TICKET.md` §9.7 已如实写明「看到第一轮评测后的两个补充探针…判据/臂集/数字均未改」，这一条我判**诚实**。

**(c) `analysis.md` 数字去向抽样（≥8，逐一落到 `results.json` 键）**：
1. §0 表 1.6281/1.6241/1.6250/1.5999/1.5958/1.5276/1.5855/1.5133/1.5027 → `rows[*].bpc`（`:1478,1494,1510,1526,1542,1558,1574,1590,1606,1622,1638,1654,1670`）✅
2. 「补回了多少差距」列 0/3.2/2.5/22.5/25.8/80.1/33.9/91.5/100% → `location[*].recovery_ratio_r`（`:1818` 等）✅
3. 「41×σ_cell / 4.9×0.014」→ `key_readings.pl_minus_win32`（`:1487`）✅
4. 「换窗口序 +0.0719」→ `robustness_alt_window_seed`（`:1926`）✅
5. 「0.0974 / chunk 匹配 0.1013」→ `key_readings.pl_vs_carry16_s0 / pl_vs_shared_c256`（`:1798–1807`）✅
6. §1 判据④「Δbpc=+0.100515、r=0.8014」→ `location.carrypl16_s0`（`:1817–1818`）✅
7. §1「12/12 `selfchecks.ok`、`nondegenerate` 全真」→ `selfchecks`（`:449–521`）✅
8. §2「下包络 = win16…win32→carrypl16→win128→full；win32 被 carrypl16/carrypl16d 支配」→ `frontier.lower_envelope`（`:1748–1757`）、`frontier.dominated_by.win32_s0`（`:1735–1738`）✅
9. §3「0.000983 / 0.001661 / chunk 0.003961 / 摊平 +0.100265 / k_eff_amort 16.06」→ `sigma_within_config_2634`、`key_readings.chunk_sensitivity_shared`、`sensitivity.carrypl16_s0`（`:1885–1900`）✅
10. §4「read_share 0.386、beta_mean 0.157、state_fro 24.9、rel_dev 0.258、params_added 264,296」→ `liveness.carrypl16_s0.carry_liveness_training`（`:1205–1220`）✅；「12 个逐层读出占比 0.011–0.647」→ `train_carry_read_mass.read_share_by_layer`（`train/pl16/results.json:688–701`：0.011381…0.64722）✅；「66/66 对 mem 不同、max_abs_mem_diff 690.45」→ `perlayer_state_probe`（`:1140–1155`）✅；「掩码反事实 13.88」→ `liveness._mask_probe`（`:1462–1466`）✅
11. §5「退化实现把收益压掉 0.0974（32×）」→ 32× 是相对 2638 的 0.003131（`2638` 的 carry16 Δbpc）：0.0974/0.003131=31.1 ✅（表述可更明确，非错误）
12. §8 成本表 → **两处对不上**，见 §5 P2-1。

**结论**：口径线没有被软化；执行期补充全部落在 `results.json` 可查；`analysis.md` 抽查的 12 组数字
**无一处与 `results.json` 冲突**（唯一不一致在 §8 成本表，属记账）。

---

## §3 「必须不变」：无未来泄漏（我核到探针逻辑与落盘读数，**未跑模型**）

- **探针实现**（`run.py:105–140`）：取 `x[B,T]`，把位置 `≥ T//2`（=128）换成随机 token 得 `x2`；
  用 `masked_logits(model, ·, keep_mask)` 两次前向，比较**最后一层 logits** 在位置 `<128` 的
  `max|a−b|`，要求 `== 0`；同时报 `changed_suffix_max_abs_logit_diff` 作抓力证据。
  `masked_logits` 走 `model._input_states → blocks → readout`（`tools/evalkit/retention.py:149–174`），
  carry 块在 `blocks` 内 ⇒ 该探针**确实穿过 carry 路径**（不是绕过）。
- **落盘读数**：12 个带掩码臂全部 `max_abs_logit_diff_prefix = 0.0`、`causal_no_future_leak=true`，
  抓力读数为 21.64–26.7（如 `carrypl16_s0` 24.631809 `:1168`、`carrypl16d_s0` 21.640949 `:1384`、
  win16_s0 25.33 `:548`、win16_s1 25.65 `:606`），另有掩码反事实 13.88（`:1464`）。
- **我未能独立复算**（无执行工具）。父级若要执行级证据，跑：
  ```bash
  python3 - <<'EOF'
  import sys, torch; sys.path.insert(0,'/home/z/vibe/pathlm')
  from pathlm.depth_ar import DepthARConfig, DepthARModel, window_anchor_keep_mask
  from tools.evalkit.retention import masked_logits
  cfg=DepthARConfig(d_model=256,n_layers=12,n_heads=8,seq_len=256,mlp_mult=6,dense=True,
                    topology="sequential",train_window=16,train_anchors=0,carry_rule="delta",
                    carry_chunk=256,carry_per_layer=True,carry_decay=1.0)
  m=DepthARModel(cfg,206); m.load_state_dict(torch.load("axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/train/pl16/model.pt",map_location="cpu")); m.eval()
  x=torch.randint(0,206,(2,256)); km=window_anchor_keep_mask(256,16,0)
  x2=x.clone(); x2[:,128:]=torch.randint(1,206,(2,128))
  with torch.no_grad(): a=masked_logits(m,x,km); b=masked_logits(m,x2,km)
  print("prefix max|Δ| =",(a[:,:128]-b[:,:128]).abs().max().item(), "suffix max|Δ| =",(a[:,128:]-b[:,128:]).abs().max().item())
  # 期望 ≈ 0.0 与 >20（探针有抓力）；我自己没跑，此期望来自 results.json 的落盘读数
  EOF
  ```
  变异（可选）：把 `carry_scan` 的读出改成读到 `t+1` 的写入，应先让 `prefix max|Δ| > 0`——我没做（无工具）。

---

## §4 反否证：换口径 / 换序 / 换读数

- **(a) 换评测窗口序**：`window_seed=7` 下 win32 1.594147、carrypl16 1.522235 ⇒ **+0.071912**
  （`:1913–1940`），与主评价 +0.068228 **同号同量级**（差 0.0037，≈1×chunk 敏感量级）；
  同一次换序里 full 1.502689→1.496001、win16 1.62811→1.625221（漂移 ≤0.005）
  ⇒ 主判据不是单次窗口采样的运气。
- **(b) 换折价口径**：冻结口径（判据）= 状态整块计字节 ⇒ carrypl16 k_eff **32.00**（`:1642`）；
  摊平口径（只报数）= `(kv + state/T)/12288` ⇒ **16.0625**（`:1859`），此时同点纯窗口插值 1.62786、
  `carry_minus_interp_window_amortized = +0.100265`（`:1867`）**同号且更大** ⇒ 主结论**不依赖折价口径**。
  第三口径（状态按 fp32 记，= 引擎内部精度）：我自算 k_eff=48 → win48≈1.582073 > 1.527595
  ⇒ **仍占优**（与 `analysis.md` §3 脚注一致）。
- **(c) 换基线强度**：见 §7 ②。**（d）漏报风险**：`sensitivity` 只覆盖 carry 臂、未给等价“摊平后的窗口族”；
  它是非判据读数，不构成缺陷。

---

## §5 口径诚实性之外的三处非阻断记账问题（P2）

**P2-1｜`analysis.md` §8 成本表与落盘证据不一致（2 处）**
- 位置：`2639/analysis.md` §8 表。
- 证据：表里写「`pl16d` **30.0 min**（`08:21:40–08:52:27`）」，但
  `train/pl16d/results.json.wall_minutes = **30.7**`（`:871`），且两个时间戳之差 = 30 min 47 s = **30.78 min**
  ⇒ 两边都不是 30.0（`TICKET.md` §9.6 同错）。另表里「评测 **0.49 min**」，
  而 `results.json.cost.wall_minutes_total_eval = **0.5**`（`:2004` 附近）、`run_log.jsonl` 末条 t=29.96 s。
- 影响：不影响任何科学结论（合计 ≈75.5 min 仍 <85 min 预算）；属记账。
- 最小修：`30.0 → 30.7`；`0.49 → 0.5`（或去掉第二位小数）。

**P2-2｜`results.json` 含裸 `NaN`，不是严格 JSON**
- 位置：`2639/results.json:1429`（`liveness.carrypl16d_s0.carry_liveness_training.rel_dev_vs_other_rule`），
  源自 `train/pl16d/results.json:679`。
- 证据：`json.dump` 默认 `allow_nan=True` 写出裸 `NaN`；Python `json.load` 能读，但 `jq`/`JSON.parse`
  会失败。其余 12 个臂同键是实数（如 `:1215` 0.292065）。
- 影响：SSOT 原始数字文件对严格 JSON 工具不可解析；不影响本实验使用的量。最小修：
  `json.dump(..., allow_nan=False)` + 该字段置 `null`（或让引擎侧对 `decay≠1` 返回 `null`）。

**P2-3｜术语「同算力」vs 判据体的「同字节」**
- 位置：`project.yaml:16`（`milestone.acceptance`「不被**同算力**最强基线支配」）、`:14`
  （`objective_criteria` 同句），而 `method:12` / `README.md` §4 ③ 判据体一律是**每位置等效 KV 字节**。
- 影响：C2 的实质判据按 62-F2 走「字节」，此处「同算力」是口径模糊的简写（纯窗口与线性状态的
  FLOPs 本来就不同）。属历史文本（`FRAMEWORK-DECISIONS` C2 落地时写入），非本轮新引入。
  最小修：把「同算力」改成「同预算点（每位置等效字节）」。**report-only**。

---

## §6 越界 / 卫生（我能核到的部分）

- **工作树**：`watchdog_diff`（对 `c77cc0b`）= `No working-tree changes`（含无 untracked）⇒ 与任务书
  「工作树干净」一致 ✅。**但该工具明确不含已提交区间** ⇒ 「本轮是否只动了 `axes/1-time/**` 与
  `.scratch/62-axis1-time/**` + 两份 E1 报告」**我核不了**（需要 `git diff d3d33eb..HEAD --stat`，见 §末）。
  我能给的旁证：`2639` 目录内**没有任何**共享路径的拷贝/补丁；`run.py` 只 `import` 共享模块；
  `train_pl.py` 只**运行期**把 `train_depth_ar.DepthARConfig` 换成子类（`train_pl.py:33–43`），
  **未改共享文件**；单测放在实验目录内（`test_2639_frontier.py`）而非 `tests/` ✅。
- **共享引擎指纹**：`MODEL.md:52–59` 与 `results.json.code_hashes`（`:2010–2019`）8 条**逐字一致**；
  其中 `pathlm/depth_ar.py = 2f4b8887…` 与 E1 的两份独立件对得上
  （`.scratch/67-engine-changes/E1-CARRYCHANNEL.md:118`「提交后 sha256：`depth_ar.py 2f4b8887…`」、
  `REVIEW-E1.md:124`「还原后 `2f4b8887…`」；`E1-CARRYCHANNEL.md:86` 记改前 `1f8f9eb4…`，
  与 `2638/MODEL.md:37`、`2638/results.json:1027` 一致）⇒「2638 记的是 E1 前、2639 记的是 E1 后」这条链条自洽；
  资产面我也确认 `pathlm/depth_ar.py` 磁盘版本**确实含** E1 字段
  （`:237 carry_per_layer`、`:244 carry_decay`、`:1783–1791` 遥测、`:2104–2115` 非法组合报错）
  ⇒ 至少是 E1-or-later。**磁盘重算 sha256 我做不到**（见 §末）。
- `.gitignore` 覆盖 `*.pt`/`*.log`/`__pycache__/` ⇒ 实验目录里的 `*.log` 与 `__pycache__/` 是本地
  忽略产物，**不是**入库噪声；每个臂的训练证据 `train/{pl16,c16,pl16d}/{results.json,train_log.jsonl,gate_log.jsonl}`
  已入库 ✅。
- **缺少 `.scratch` 之外的过程噪声**：未发现写入 `axes/` 其它轴或 `tools/`/`tests/` 的痕迹（限于可见面）。

---

## §7 要否证的三个关键点（逐一结论）

### ① 「等字节点」比较是否公平（状态 12×8192 floats 的字节记账是否被夸大/缩小）
**结论：公平，且记账偏向「利 carry」而非「利窗口」——方向不被动摇。**
- 记账 = `n_heads × hd²`（8192 floats/层）× `n_layers`（逐层）或 1（单共享槽），由
  `run.py:86–92` 从**加载后的模型**读，不是我或 writer 手填；`checkpoints.state_floats_actual`
  12 臂中 carry 臂分别为 98304（逐层）/8192（单共享槽）✅。
- 状态是 **fp16 计**（2 B/元素）而引擎内部一律升 fp32（`analysis.md` §3 脚注、writer 自记）⇒
  按 fp32 记 k_eff=48、同预算纯窗口 1.582073，**carrypl16 仍更好**（差 0.0545）⇒ 记账**没有**夸大 carry 的优势。
- 对照臂 `win32` 是**同一 `n_layers/d_model`、同参数族**的纯窗口模型，两者 `total_bytes` 逐字节相等
  （393,216），比较是「同 footprint 直接点比」而不是插值 ✅。
- **唯一的记账外差异**：carrypl16 比 win32 多 264,296 参数（13.80M vs 13.54M，+1.95%）——这**不**
  进字节轴，但对「增益来自机制还是来自容量」是一个未消混淆，writer 已列 `gaps`（下一步 ≈10 min）。
  **我判：不足以降级结论**（见 ③ 的量化 + 下面这条我另外发现的事实）。

> **我独立发现的一条有利于结论的事实（writer 未强调）**：完备度对照 `carry16_s0` 与 `carrypl16_s0`
> 的**参数量完全相同**（`checkpoints.*.n_params` 均为 **13,801,064**，`:373`/`:407`；`train/c16` 与
> `train/pl16` 的 `carry_liveness.params_added` 均 264,296）⇒ 「单共享槽 → 逐层 + 当前 hidden 写」
> 那 **0.0974 bpc** 是**等参数**对照，不能被「多参数」解释 ⇒ 「上一轮的否证落在该实现」这一归因更硬。

### ② `win32` 是不是「同预算最强基线」；`carrypl16` 是否真在包络上
**结论：是；是。**
- 下包络（`frontier.lower_envelope` `:1748–1757`，我按 `(total_bytes↑, bpc↓)` 手算复现）：
  `win16_s0 → win17_s0 → win18_s0 → win24_s0 → win32_s0 → carrypl16_s0 → win128_s0 → full`。
- 在所有 `total_bytes ≤ 393,216` 的臂里，`win32_s0`（1.595823）是 bpc 最低的纯窗口/非 carry 点
  （win16 1.62811、win17 1.624104、win18 1.619953、win24 1.599856、carry16 1.624979、
  c16c256 1.628940）⇒ 「同字节最强基线」成立 ✅；`carrypl16`（393,216 B，1.527595）**在包络上**
  （它的 bpc 低于所有 ≤ 其字节的臂）✅，且把 `win32` 挤出包络（`dominated_by.win32_s0 =
  [carrypl16_s0, carrypl16d_s0]` `:1735–1738`）。
- **未覆盖的两类竞争者**（如实）：量化点（int8/int4 KV 未同图，`gaps` 第 3 条）与**等参数量**
  纯窗口臂（`gaps` 第 4 条）⇒ 「最强基线」只在**本实验 13 臂集合内**成立（`README.md` §4 ⑥ 禁外推已限）。

### ③ `c16c256_s0`(1.628940) 比 `carry16_s0`(1.624979) 差 0.003961 ⇒ 削减 0.068228 吗？
**结论：不削减（量化：最坏情况仍剩 0.0643 bpc = 39×σ_cell）。**
- 两个臂**同配置同 seed**（单共享槽 delta 窗口 16 seed 0），只差 `carry_chunk` 64↔256
  （`results.json.rows` 的 `carry_chunk` 字段 `:1621` vs `:1229`；`checkpoints` 同）。
  ⇒ chunk（只改浮点累加顺序的实现细节）在本格能移动 **0.003961 bpc**，量级 > σ_cell(0.00166)
  ——**这条确实说明 σ_cell 低估了「实现细节」类方差**（这是本项的诚实加分：它自己报了这条）。
- **但不削弱主判据**：0.003961/0.068228 = **5.8%**；即便把整段 chunk 效应按最坏方向加到
  carrypl16 上，剩余 0.068228−0.003961 = **0.064267 = 38.7×σ_cell**；且主判据的对照臂 `win32`
  **不含 carry 路径**（chunk 与它无关），所以 chunk 无法系统性地伪造这 0.0682。
- 另：writer 已用**同 chunk** 的退化对照给下界 —— `pl_vs_shared_c256 = 0.101345` 与
  `pl_vs_carry16_s0 = 0.097384` 两个口径同量级（差 0.004，恰好 = chunk 敏感量级）⇒ 完备度结论在两个
  chunk 口径下都成立 ✅。

---

## §末 判词

**判词：FAIL（窄范围）。** 范围说明：**科学主张与数值证据 = PASS**（我在无执行工具条件下能做的
独立复算全部一致、未找到可推翻 +0.068228 的证据、口径未被软化）；**唯一阻断项是「记录 ≠ 文件」的
一致性缺陷**。

**Blocking（1 条）**

| # | 位置 | 违反的判据 | 复现（证据） | 最小修 |
|---|---|---|---|---|
| **B1** | `axes/1-time/1.1.4.1-explicit-memory/project.yaml` 的 `experiments:`（第 19–46 行，末两条 = 2638 的 `@69d431b`/`@443cb44`）；对应 `change_log` 第 128–130 行（`2026-10-14T09:10:00`，字面「**experiments 新增 2639**（analysis.md/results.json @ccdc91a）」）；同一断言也在 `c77cc0b` 的提交信息里（`.git/COMMIT_EDITMSG` 第 7 行：「experiments 新增 2639（analysis.md/results.json @ccdc91a）」） | `AGENTS.md` §8「提交信息只描述**已验证落地**的内容；一旦发现不符，**立刻补一个更正提交**（不要 amend 掩盖）」；`docs/research-schema.md` §11.2 兄弟字段一致性（新实验应在 `experiments` 登记）；`SCHEMA §9`「`experiments` 路径必须存在」隐含「列出的即真实证据根」 | `grep -n 2639 project.yaml` 只命中 `gap:11`、`WP1.evidence:81`、`WP2.evidence:89`、`roadmap:102,103`、`change_log:130`；**第 19–46 行的 `experiments` 列表里没有 2639**（`.scratch/63`/`.scratch/62` 的日志与 `git log` 均确认 `ccdc91a` 先于 `c77cc0b` 落盘 ⇒ 路径当时已存在，不构成「未来路径」豁免） | 在 `experiments:` 补两行 `…/experiments/2639-carry-perlayer-frontier-13m/analysis.md @ccdc91a`、`…/results.json @ccdc91a`；**或**把 `change_log` 与提交信息里「experiments 新增 2639」改为「evidence 新增 2639」并补一个更正提交（`AGENTS.md` §8 禁 amend） |

**Non-blocking（P2，report-only 或随手修）**
- **P2-1** `analysis.md` §8 成本表：`pl16d 30.0 min` ↔ 落盘 `train/pl16d/results.json:871 = 30.7`（时间戳差
  30 min 47 s）；`评测 0.49 min` ↔ `results.json.cost.wall_minutes_total_eval = 0.5`。
- **P2-2** `results.json:1429` 裸 `NaN`（`carrypl16d_s0` 遥测）⇒ 严格 JSON 不可解析（`train/pl16d/results.json:679` 同源）。
- **P2-3** `project.yaml:14,16`「同**算力**」↔ 判据体「同**字节**」（`method:12`/`README.md` §4③）术语不齐（历史文本，非本轮引入）。

**若父级判定 B1 属「收口后随手可补的记账」** ⇒ 本项降为 **PASS with notes**（科学结论不受影响）；
但按本轴 REVIEW-1…4 的既有尺度（同形态缺陷均按 blocking 处理）与 `AGENTS.md` §8 的明文要求，
我默认给 **FAIL（窄范围）**。

---

## 未复核 / 未覆盖（如实列出，别当我验过）

1. **未跑任何命令**（本复核体无 shell/执行工具，`watchdog_diff` 是唯一与 Git 相关的工具，且不含已提交区间）：
   以下 6 条**需父级代跑**，我不声明已验：
   - `sha256sum` 3 个新臂 checkpoint + `config.json`，与 `MODEL.md:5–19`/`results.json.checkpoints` 对拍；
   - `python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/test_2639_frontier.py -q`（6 条，我逐条读了源码，逻辑与 `results.json` 自洽）；
   - `python3 tools/summary.py --check` 与 `python3 tools/check_tree.py`；
   - `git diff --cached --name-only | wc -l`（应为 0）；
   - `git diff d3d33eb..HEAD --stat`（**越界/卫生的唯一硬证据**：本轮是否只动 `axes/1-time/**` + `.scratch/62-axis1-time/**` + 两份 E1 报告）；
   - 评测重跑（§20 锁）：先 `cp results.json /tmp/`，再 `python3 axes/.../2639-.../run.py`，比 `key_readings`（我未取锁、未跑）。
2. **未做因果探针的独立执行与变异**（§3 给了脚本；变异「读到 t+1」未做）。
3. **未核验「判据线在任何训练启动之前写死」的文件级证据**（无 mtime/git；只核到 ticket §9.2/§9.4 的时间线
   + `README.md` §4 的「写下后不得改」声明 + `run.py` 判定线与文字同向）。
4. **未核验 `train/{pl16,c16,pl16d}/*.pt` 与 `config.json` 的磁盘 sha256**（只能核到
   `config.json`/`results.json`/`MODEL.md` 三处字符串互相一致）。
5. **未核验训练过程本身**（`train_log.jsonl`/`gate_log.jsonl` 我只看键与末值；未独立复算训练 CE、
   未核对 12k 步/token 预算/余弦调度的实际执行）。
6. **未覆盖**量化点同图、召回轴、等参数对照、多 scale/多 seed、长上下文（这些都还在 `gaps` 里，
   我同意 writer 的「milestone 字面未达标」记录）。
7. **watchdog_diff 只覆盖工作树** ⇒ 已提交区间（`ccdc91a`、`c77cc0b`）的文件清单与其提交信息的一致性
   **未核**（B1 是我从 `project.yaml` + `.git/COMMIT_EDITMSG` + `.git/logs/HEAD` 读出来的，不是从提交 diff 读出来的）。