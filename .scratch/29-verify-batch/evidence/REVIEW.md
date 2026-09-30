# 独立复核（契约 D，fresh context，只读）

## Review — 契约 D 独立复核（只读，fresh context）

复核对象：`.scratch/29-verify-batch/issues/01-verify-batch.md` 的契约 A/B/C 三条线全部交付物。我只读检查（无 shell），已通过 `contact_supervisor` 回传完整报告供主 Agent 落盘到 `.scratch/29-verify-batch/evidence/REVIEW.md`。

### 逐条 pass/fail（SCHEMA §14）

| # | 检查项 | A（2611-jumphead-ar-control） | B（2611-equalparams-linear-control） | C（2612-gated-readout-24k） |
|---|---|---|---|---|
| ① | 判据线未被改（对照契约原文） | **pass**（README §4 逐字一致） | **pass**（README §4 逐字一致） | **pass**（阈值一致；§4 第 3 条对「中间情况」补了数值刻画，边界 −0.05/−0.014 未动） |
| ② | 三道对账（算术/任务/口径） | **pass**（analysis §2.1–2.4） | **pass**（§6 + `reconciliation`） | **pass**（§5 + `reconciliation`） |
| ③ | 对照臂三要素（输入/预测/部署）+ 同一任务定义 | **pass**（§5 表） | **pass** | **pass** |
| ④ | 干预活性量非零且与日志一致 | **pass** | **pass** | **pass** |
| ⑤ | analysis↔results 数字对应（抽查） | **pass**（≥12 个，1 处聚合口径混用） | **pass**（≥20 个） | **pass**（≥15 个） |
| ⑥ | 口径标注（快照/渐近/墙钟/FLOPs） | **pass** | **pass** | **pass** |
| ⑦ | pytest 全绿 / 既有实验未破坏 | **not-run**（我无 shell）；静态证据支持 | 同 | 同 |

### Correct（已核实良好，附证据）
- **对照臂诚实性**：A 的 `ar1_second_step_readout` 只替换紧邻位置 i+1（`train_jumphead.py` diff），配 `test_ar1_matches_bruteforce_rowloop` 逐行对照 + `test_ar1_prefix_keeps_ground_truth_not_whole_context` 复现旧 bug。**无「假上下文 + 真目标」**；oracle 仅 `oracle_is_upper_bound_only`。
- **算术自洽**：0.6934×0.6953=0.4822 ≤ 0.5489 ≤ min(0.6934,0.6953)；隐含条件命中 0.5489/0.6934=0.7916 ≥ 0.6953；0.4667/0.5489=0.8502。B：1.7694−1.8582=−0.0888、1.7770−1.8582=−0.0812、gap 0.0076、7,093,760−6,962,688=131,072=2·256²。C：−0.0463=3.3×方差界、Δ 缩小量 0.0349 = A 降幅 0.1644 − gated 降幅 0.1295。
- **原始日志复算**：C 的末 1000 步 loss 斜率我按日志 6 点最小二乘复算得 **+0.0777 / +0.0589**，与 `results.json` 逐位一致；24k 窗 loss 均值复算 1.4912 / 1.3730 亦逐位一致。A/B 的复用臂重测值（trunk 0.6934、oracle 0.6953；A 1.8582/0.9694、gated 1.7770/0.9680）与 23c / 2610 历史文件逐位相同。
- **干预是活的**：A jump head w_fro 21.39/21.38 nonzero 1.0；B P 12.2578 / Q 19.7416（smoke 20 步 P 0.2392，起步 P=0 单边零）；C W1 37.51 / W2 98.07（smoke 20 步 5.1476 / 0.21）。均为单边零 init，非鞍点。

### Finding（均为 P2，report-only，不改变任何判决）
1. **P2 · A · `analysis.md` §1/§4 聚合口径混用**：`ratio 0.8501`（两 seed 比值均值）与 `0.8502`（均值之比）并用未标注；两数都真实存在于 `results.json`（`arms_mean` 与 `criteria`）。最小修：加一行注明两个聚合口径。
2. **P2 · B · 排序语言与自定方差界冲突**：`analysis.md` §0/§2.1 写「不但追平、还略微超过」「线性更强 −0.0076」，而 §7.1 同时声明「0.0076 < 0.014 ⇒ 不可排序」。最小修：改述为「无差异（不可排序）」，判据① 仍成立。
3. **P2 · B · 未预登记阈值**：§2.1 用「>0.003 劣化线」描述 exit-4 ECE（实测 linear 0.0082 vs A 0.0053 = +0.0029）。最小修：只说相对增量 +0.0029。
4. **P2 · C · README §4 第 3 条为执行者补写**的中间带数值刻画（阈值未动，建议在 §4 注明「对『中间情况』的操作化，非改判据」）。
5. **P2 · C · `background_slope_audit`** 的 2610 斜率自报，未独立复算。
6. **P2 · B · `run.py::reload_model` 未断言复用 checkpoint 的 sha256**（A 线 `run.py` 有断言）；哈希仅记于 `MODEL.md`。
7. **P2 · B/C · README front-matter `status: planned`** 陈旧（A 为 `judged`），与既有 2610 写法一致。

### 可疑数字清单结论
无「虚假或不可靠数字」：全部抽查数字均能在 `results.json` 或原始 `train_log*.jsonl` 中找到来源并可复算；上面 7 条为口径/措辞/可追溯性 note。特别关注项 a) 假上下文+真目标 —— 未发现；b) 死干预 —— 未发现（三处活性量非零且可见于冒烟日志）；c) oracle 当判据 —— 未发现；d) 算术不自洽命中率 —— 未发现（A 的 ar1 命中夹在 [0.4822, 0.6934] 之间且条件命中方向一致）。

### Merge verdict
**OK with notes**（前提：主 Agent 跑 `python3 -m pytest tests/ -q` 得 **150 passed**，即 140 基线 + A 线 4 + B 线 6；tests/ 下静态计数正好 150，与 analysis 声明一致）。A=中性（主判据 0.8502∈[0.80,0.90)，草稿判据 0.5998<0.90）；B=判据① 触发（gap 0.0076<0.02 ⇒ gated 降级）；C=中间带不裁决（Δ=−0.0463）。三条线结论均被证据支持。