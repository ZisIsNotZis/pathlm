# ticket 61 轴 2 Phase A · round-2 收口（REPAIR-round2）

> actor：fresh 收口 writer（单一写入者，未参与本线任何先前写作）
> 输入：`.scratch/61-axis2-depth/REVIEW-2.md`（fresh 独立复核判 **BLOCK**：科学结论 PASS / 处置落盘自洽性 4 项文本级账实不符）
> 被修 revision：`483b181`｜本轮提交：`3ef14e6`（2026-10-13T14:14:41+08:00）
> 边界：**只改文本/数字一致性**；未动任何测量数字、未动实现（`pathlm/**` 零改动）。

---

## ① 人话摘要

round-2 独立复核把这条线拆成两层看：**科学内容**（逐 token 早退的序列级实测、数字、变异抓红、生产语义逐位不变）
全过；但**「处置记录/SSOT 与仓库现状是否对得上」**不过 —— 4 处文本级账实不符：单测数量说 16 实际 19、
臂数说 11 实际 14、F2 的区间把「省 20% 那一档」混进了「省 ≥25% 档」、生成一致率区间写 28–55% 实际到 71%。
这四处都不改变任何科学结论，是「自己写下的账没对平」。本轮把 B1–B4 逐条改到与 `results.json`/`pytest` 实测一致，
顺带把复核的 4 项 non-blocking 一并处理，并把 `_missing_mask` 的 head 维疑点**如实登记为待测**（不擅改实现）。
round-2 的复核报告与重跑证据已入库。**未改动任何测量数字与实现代码。**

---

## ② B1–B4 逐条落点（改前 → 改后）

### B1 · 单测计数 16 → 19（6 处 SSOT/文档 + TICKET §7.2 订正）
- `axes/2-depth/2.1.1-early-exit/project.yaml`：`conclusion`「16 条单测」→ 19；`conclusion_review`「/ 16 条单测 /」→ 19；`change_log`「（16 条，全绿）」→ 19。
- `.../2631-pertoken-exit-seqlevel/analysis.md`：§0 第 2 点「配套 16 条单测」→ 19；§10「（16 条，全绿）」→ 19。
- `.../2631-pertoken-exit-seqlevel/README.md`：§1「（16 条）」→ 19。
- `.scratch/61-axis2-depth/TICKET.md`：§4.5「16 条 … ⇒ 16 passed」→「19 条 … ⇒ 19 passed」；§7.2 F11 行「新增 2 条对拍单测 + 2 条活性断言」→「新增 3 条单测（16→19，含对拍与活性断言）」；§7.2 自测行「含 F11 新增 2 条」→「新增 3 条」。
- 依据：`grep -c "^def test_" tests/test_axis2_exit_decision.py` = **19**（`233e978` 为 16，本轮修订净增 3：`test_projfill_nonidentity_decoder_matches_forward` / `test_nofill_decoder_matches_forward` / `test_fill_path_is_unchanged_by_missing_mask`）。

### B2 · TICKET §4.5 臂数 11 → 14
- `.scratch/61-axis2-depth/TICKET.md` §4.5「11 条臂 + 4 类归因诊断」→「14 条臂 + 4 类归因诊断」，与 `README.md §2`（共 14 条臂）、`results.json::arms`（14）、`analysis.md §2` 对齐。

### B3 · F2 残留：区间与限定词（4 处）
真实值：省 ≥25% 档 = τ=0.3/0.6/0.9，`frontier_excess` = **0.2593 / 0.1168 / 0.0117**（0.014 噪声带的 18.5× / 8.3× / 0.84×）；0.0035 属 τ=0.95（省 20% 档）。
- `analysis.md` §7 ②「0.0035–0.2593 bpc（0.25–18.5 倍）」→「**0.0117–0.2593 bpc（0.84–18.5 倍）**；τ=0.95 属省 20% 档、超出 0.0035」。
- `analysis.md` §8 末行「主结论 Δ 是 8–18 倍噪声带」→「**超出量是 8.3–18.5 倍噪声带**」。
- `project.yaml` `argument`③「超出固定前缀 0.0035–0.2593 bpc（0.25–18.5 倍）」→「**0.0117–0.2593 bpc（0.84–18.5 倍；τ=0.95 属省 20% 档、超出 0.0035）**」。
- `project.yaml` `failure_analysis`④「主结论 Δ 是 8–18 倍噪声带」→「**超出量是 8.3–18.5 倍噪声带**」。
- `conclusion` 原本已写对（0.0117–0.2593 / 0.84–18.5×），未动。

### B4 · 生成一致率 28–55% → 28–71%（2 处）
- `analysis.md` §8 U4「发出流一致率 28–55%」→「**28–71%**」。
- `project.yaml` `gaps`「逐 token 一致率 28–55%」→「**28–71%**」（同款错值，自检 grep 覆盖范围内一并改）。
- 依据：`results.json::generation_agreement` 的区间 = 0.28125（τ=0.3）… **0.710938（τ=0.999-min4，91/128）**。

### 同批 non-blocking（REVIEW-2 §⑥）
- N1 `analysis.md` §3 末：「该列**不**用于对齐。。（…重叠到 1e-05）」→ 删重复句号、改「**数值上重合到最大 4.6e-05**」（实测 `|Δ|` 最大 4.6e-05）。
- N2 `project.yaml`：`argument`② 与 `failure_analysis`④ 的「Δ=±0.0000」统一为「**浮点噪声内重合（Δbpc 7e-06…4.6e-05）**」；`.scratch/.../TICKET.md` §4.5 锚点行同款措辞亦统一（同一缺陷的三处写法对齐，避免同口径两种表述）。
- N3 `analysis.md` §0 第 5 点「损伤**全部**来自…」→「损伤**主要**来自…」，与 §6 的「损伤**集中在**…（浅层桶两项未分离）」对齐。
- N4 `project.yaml` `conclusion_review` 的「**处置（全部已落盘）**」→「**处置（round-2 独立复核判 BLOCK；属 blocking 的 B1–B4 均为文本级账实不符，本收口轮已逐条修）**」；并把其尾句从「round 2 已派发、判词到达前效力只覆盖 round 1」改为「round 2 已返回：科学结论与运行证据 PASS / 处置落盘自洽性 BLOCK（B1–B4 已修）」。
- N5 `pathlm/depth_ar.py::_missing_mask` 未 expand 到 head 维（同仓 `_ragged_mask` 显式 expand；CUDA SDPA 不广播 bool 掩码 head 维；本实验走 CPU、`fill_kv=True` 未触发）：**不擅改实现**，登记为待测 —— `analysis.md` §8 新增 U7 行、`TICKET.md` §7.4 记为 N5 待测。
- 附带：`analysis.md` §10 复核状态补 round-2 报告指针；`TICKET.md` 新增 §7.4（真实时间戳 `date -Iseconds` = 2026-10-13T14:12:36+08:00）记录本轮处置，并更新 §7.3 尾句与 §8 残余缺口①。

---

## ③ 命令与结果

| 命令 | 结果 |
|---|---|
| `grep -c "^def test_" tests/test_axis2_exit_decision.py` | **19**（233e978 为 16） |
| `git diff 233e978 483b181 -- tests/test_axis2_exit_decision.py`（新增测试名） | 3 条：`test_projfill_nonidentity_decoder_matches_forward` / `test_nofill_decoder_matches_forward` / `test_fill_path_is_unchanged_by_missing_mask` |
| `python3 -m pytest tests/test_axis2_exit_decision.py -q` | **19 passed** in 2.84s |
| `python3 tools/summary.py --check` | **OK：46 个 project.yaml 全部符合 docs/research-schema.md** |
| `grep -rn "16 条\|11 条臂\|0.0035–0.2593\|28–55" axes/2-depth .scratch/61-axis2-depth` | 命中仅在 `REVIEW-1.md`/`REVIEW-2.md`（历史复核报告，按要求保留）；`axes/2-depth` 与非复核文件已清空 |
| `git diff --check` | 干净（无空白/冲突标记） |
| `git diff --cached --name-only \| wc -l` | 提交前 **7**，逐条核对为下述 7 个显式路径 |
| `git commit -F - <<'EOF'`（heredoc，非 `-m`） | commit **3ef14e6**；`git status --short` 提交后为空；staged 计数 **0** |

入库（显式 `git add`，未用 `-A/-u/.`）：
`axes/2-depth/2.1.1-early-exit/project.yaml`、`.../2631-pertoken-exit-seqlevel/analysis.md`、`.../README.md`、
`.scratch/61-axis2-depth/TICKET.md`、`.scratch/61-axis2-depth/REVIEW-2.md`、
`.scratch/61-axis2-depth/repro-review2/results.json`、`.scratch/61-axis2-depth/repro-review2/run_log.txt`。

---

## ④ 自查说明：保留的 grep 命中（逐条）

自检 grep 的**唯一剩余命中**全部位于历史复核报告：`REVIEW-1.md`（引用 `233e978` 时刻的「16 条」「11 条臂」「0.0035–0.2593」「28–55」）与 `REVIEW-2.md`（引用被复核时点的错值）。二者均为**历史记录**，按「日志文档允许日期流水、不得作为事实依据」的性质保留，不构成当前 SSOT 的账实不符。
当前 SSOT（`axes/**/project.yaml`、实验目录 `analysis.md`/`README.md`）与 `TICKET.md` 处置记录中已无这些错值。

---

## ⑤ 未决 / 残余风险

1. **N5（实现层，未改）**：`_missing_mask` 在 CUDA SDPA 下是否因 head 维未 expand 而报错**未验证**（本实验为 CPU、`fill_kv=True`）。已登记为 `analysis.md` §8 U7 与 `TICKET` §7.4 N5，需一次 GPU 冒烟或与 `_ragged_mask` 同款 `expand` 修复。
2. **本收口轮不再是独立验收**：本轮是「按 REVIEW-2 判词修文本」，其自身不构成独立复核；被修后的 revision `3ef14e6` 若需正式放行，应按 SCHEMA §14/§14b 再派**新的 fresh reviewer**（复核对象 = `3ef14e6`）。
3. **`failure_analysis`④ / `TICKET §4.5` 的「±0.0000」超出 REVIEW-2 §⑤ 逐字清单**：这是 N2 同一缺陷的同款措辞（同一事实的三处写法），本轮一并对齐口径；如认为应严格限定在 `.project.yaml` `argument`②，可回退这两处（不影响任何科学结论）。
4. **`TICKET §7.4` 中保留了「Δ=±0.0000」字样**用于说明「改前 → 改后」的溯源（日志文档允许），不作为当前事实依据。

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "按 REVIEW-2 §⑤ B1–B4 逐条修文本/数字一致性：单测计数 16→19（6 处 SSOT/文档 + TICKET §7.2 订正）、TICKET §4.5 臂数 11→14、F2 区间改 0.0117–0.2593（0.84–18.5×）并单列 τ=0.95（analysis.md §7②/§8、project.yaml argument③/failure_analysis④）、B4 区间 28–55%→28–71%（analysis.md §8 U4、project.yaml gaps）；另处理 REVIEW-2 §⑥ non-blocking（1e-05→4.6e-05 并删重复标点、Δ=±0.0000→浮点噪声内重合、§0「损伤全部」→「主要」、conclusion_review 措辞改实事求是、_missing_mask 登记为待测不擅改实现）。未改任何测量数字与实现（pathlm/** 零改动）。",
      "notes": ""
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "证据充分：pytest 19 passed；summary.py --check OK（46 份）；自检 grep 仅剩 REVIEW-1/2 历史引用并逐条说明；git diff --check 干净；提交 3ef14e6 含 7 个显式路径、提交后工作区 clean、staged 计数 0。",
      "notes": ""
    }
  ],
  "changedFiles": [
    "axes/2-depth/2.1.1-early-exit/project.yaml",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/README.md",
    ".scratch/61-axis2-depth/TICKET.md",
    ".scratch/61-axis2-depth/REVIEW-2.md",
    ".scratch/61-axis2-depth/repro-review2/results.json",
    ".scratch/61-axis2-depth/repro-review2/run_log.txt"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 -m pytest tests/test_axis2_exit_decision.py -q",
      "result": "passed",
      "summary": "19 passed in 2.84s"
    },
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md"
    },
    {
      "command": "grep -rn \"16 条\\|11 条臂\\|0.0035–0.2593\\|28–55\" axes/2-depth .scratch/61-axis2-depth",
      "result": "passed",
      "summary": "仅剩 REVIEW-1.md/REVIEW-2.md 历史引用；axes/2-depth 与非复核文件已清空"
    },
    {
      "command": "git diff --check",
      "result": "passed",
      "summary": "无空白/冲突标记"
    },
    {
      "command": "git add <7 显式路径> && git diff --cached --name-only | wc -l && git commit -F - <<'EOF'",
      "result": "passed",
      "summary": "提交前 staged=7 逐条核对；commit 3ef14e6；提交后 git status 空、staged=0"
    }
  ],
  "validationOutput": [
    "单测文件 19 个 def test_（233e978 为 16，本轮净增 3）；pytest = 19 passed。",
    "summary.py --check = OK（46 份 project.yaml）。",
    "results.json 实测：≥25% 档 frontier_excess=0.259346/0.116819/0.011661（0.014 带的 18.5×/8.3×/0.84×），τ=0.95=0.003493；generation_agreement 区间 0.28125…0.710938。",
    "自检 grep 命中仅 REVIEW-1.md/REVIEW-2.md（历史复核报告），已逐条说明。",
    "git diff --check 干净；提交 3ef14e6，7 files changed, 7900 insertions(+), 23 deletions(-)（含入库的 REVIEW-2.md 与 repro-review2 证据）。"
  ],
  "residualRisks": [
    "N5：_missing_mask 未 expand 到 head 维，CUDA SDPA 下是否可用未测（本实验 CPU、fill_kv=True 未触发）；已登记为 analysis.md §8 U7 / TICKET §7.4 N5。",
    "本收口轮为「按 REVIEW-2 判词修文本」，不构成独立验收；被修 revision 3ef14e6 若需放行应另派 fresh reviewer。",
    "failure_analysis④ 与 TICKET §4.5 的「Δ=±0.0000」超出 REVIEW-2 §⑤ 逐字清单，属 N2 同款缺陷的一并对齐，可按需回退（不影响结论）。"
  ],
  "noStagedFiles": true,
  "diffSummary": "4 个 SSOT/文档 + 3 个入库证据：project.yaml（16→19、区间 0.0117–0.2593、τ=0.95 单列、28–71%、Δ=±0.0000→浮点噪声、conclusion_review 措辞）、analysis.md（16→19、§0 损伤主要、§3 末 4.6e-05 并删重复标点、§7②/§8 区间、§8 U4 28–71%、§8 末行倍数、新增 U7、§10 补 round-2 指针）、README.md（16→19）、TICKET.md（§4.5 19 条/14 臂、§7.2 新增 3 条、§7.4 新增本轮处置+真实时间戳、§7.3/§8 更新、锚点 Δ 口径）；入库 REVIEW-2.md 与 repro-review2/{results.json,run_log.txt}。未改 pathlm/** 实现与任何测量数字。",
  "reviewFindings": [
    "no blockers: B1–B4 全部按 REVIEW-2 §⑤ 修到与 results.json/pytest 一致；自检 grep 仅剩历史复核报告引用。"
  ],
  "manualNotes": "科学结论未动：本轮只改文本/数字一致性。原始被复核 revision 483b181 的科学判词（round-2 = PASS）不变；本轮把其处置自洽性 BLOCK（B1–B4）修平，提交 3ef14e6。若父级需要对外放行，建议对 3ef14e6 再派一个 fresh reviewer 做文本级闭合复核。"
}
```
