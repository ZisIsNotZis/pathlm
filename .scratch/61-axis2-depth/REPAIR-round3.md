# REPAIR-round3 · 轴 2 Phase A 终局收口（fresh 单一写入者）

> 输入：round-3 复核 `.scratch/61-axis2-depth/REVIEW-3.md`（判 **BLOCK**，仅 1 类文本残留 = B3 的 2 处）。
> 工作区 `cd /home/z/vibe/pathlm`，HEAD 进入时 = `3ef14e6`；本轮只改文本、不动实现与测量数字。
> **提交：`e970990`**（5 文件；提交前 `git diff --cached --name-only | wc -l` = 5 并逐条核对）。
> 时间戳（本轮处置）：`date -Iseconds` = **2026-10-13T14:24:02+08:00**。

---

## 0 人话摘要（先读这段）

这轮是给轴 2「逐 token 早退」那条线的**最后一场文字对账**。前一轮（round-2）已经确认它
的**科学结论与运行证据全过**——19 条单测、全量 834 通过、4 处变异被抓红、重跑数字逐位复现——
只剩 4 条「文字写错数」的问题；上一条提交 `3ef14e6` 修掉了 3 条半，**还剩半条**：两份文档里
写「省得多时门控比同样算力的固定截断差 **8.3–18.5 倍噪声带**」，下界 8.3 取错了。

**错在哪、改成了什么**：8.3 是「省 46.6% 那档（τ=0.6）」的倍数；而「省 ≥25% 算力」这一档里
**省得最少**的那档（τ=0.9，省 25.4%）的超出量是 0.0117，除以 0.014 的噪声带 = **0.84 倍**。
所以「省 ≥25% 档」的倍数区间应是 **0.84–18.5**，不是 8.3–18.5——原先的下界把 τ=0.9 那档漏掉了，
而且和同一批文档里其它地方写的「0.84–18.5」「0.8–18」自相矛盾。本轮把这 2 处改成
**「省 ≥25% 档超出量 0.84–18.5 倍（其中 τ=0.3/0.6 为 8.3–18.5×）」**，这样那个「8.3–18.5」
还留着，但被正确限定为只是其中两档。

**顺带防再犯**：按任务要求，把 `axes/2-depth/**` 与 `.scratch/61-axis2-depth/**` 里所有相关数字
（`16 条`/`19`/`11 条臂`/`14 条臂`/`0.0035`/`0.0117`/`0.2593`/`0.84`/`8.3`/`18.5`/`0.25`/`28–55`/
`28–71`/`1e-05`/`4.6e-05`）逐处对了一遍（清单见 §3）。除 B3 的 2 处外，发现 TICKET §4.5 还有 3 处
陈旧数字（REVIEW-3 也点了名）：「0.25–18.5 倍」「深度≥3 时 ≤0.005」「831 passed」——一并订正为
「省 ≥25% 档 0.84–18.5、τ=0.95 属省 20% 档」「最大 0.0051」「834 passed」。

**入库**：只存在于 subagent artifact 目录的 `.scratch/61-axis2-depth/REPAIR-round2.md` 与
`REVIEW-3.md` 已**原样**复制进仓库（sha256 与源逐位相同）。

**没动什么**：实现代码 `pathlm/**` **零改动**；实验 `results.json` 与 round-2 重跑证据
**逐字节未改**；任何测量/物理数字未改。历史复核件 `REVIEW-1/2/3.md` 与 `REPAIR-round2.md` 里的
旧数字（含「8.3–18.5」）**保持原样**（它们是历史件，不是当前事实）。

**未决**：`_missing_mask` 的 head 维广播仍未在 GPU 验证（REVIEW-2 N5 / `analysis.md` §8 U7，
本实验走 CPU 路径未触发）；`F5`（milestone 验收口径是否收紧）仍待 Main 裁。

---

## 1 改动点位

| # | 文件 | 位置 | 改动 |
|---|---|---|---|
| A | `axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md` | §8 末行（L215） | 「统计假象（配对 SE ≤ 0.008，**超出量是 8.3–18.5 倍噪声带**）」→「…**省 ≥25% 档超出量 0.84–18.5 倍噪声带，其中 τ=0.3/0.6 为 8.3–18.5×**」 |
| B | `axes/2-depth/2.1.1-early-exit/project.yaml` | `failure_analysis` ④（L16 末句） | 同上（同措辞）⇒ 与 `conclusion`（0.84–18.5）自洽 |
| C | `.scratch/61-axis2-depth/TICKET.md` | §4.5 主判据行（L167） | 「（**0.25–18.5 倍噪声带**）」→「（**省 ≥25% 档 0.84–18.5 倍噪声带；τ=0.95 属省 20% 档、超出 0.0035**）」（O1） |
| D | `.scratch/61-axis2-depth/TICKET.md` | §4.5 校准行（L173） | 「深度 ≥3 时 **≤ 0.005**」→「深度 ≥3 时**最大 0.0051**」（O2，实测 `ece_true[depth=3]=0.0051`） |
| E | `.scratch/61-axis2-depth/TICKET.md` | §4.5 全量回归行（L157） | 「**831 passed**」→「**834 passed**（F11 新增 3 条测试后的全量）」（O3） |
| F | `.scratch/61-axis2-depth/TICKET.md` | §7.4 B3 行（L268–269） | 原称 §8/`failure_analysis`④ 已落「0.84–18.5」，与实际落盘（8.3–18.5）不符 ⇒ 改述为「只落成 8.3–18.5、下界 8.3 错，round-3 订正为 0.84（见 §7.5）」 |
| G | `.scratch/61-axis2-depth/TICKET.md` | 新增 §7.5（L282+） | round-3 复核返回与处置记录（含真实时间戳 `2026-10-13T14:24:02+08:00`） |
| H | `.scratch/61-axis2-depth/TICKET.md` | §8 残余缺口①（L306+） | 更新为「round 2/3 …B1–B4 与 round-3 B3 残留（8.3→0.84）均已订正」 |
| I | `.scratch/61-axis2-depth/REPAIR-round2.md`（新） | — | artifact → 仓，**原样**（sha256 `6169e5e7…`） |
| J | `.scratch/61-axis2-depth/REVIEW-3.md`（新） | — | artifact → 仓，**原样**（sha256 `f342ca6e…`） |

**证据（`results.json::frontier_comparison`，倍数 = excess/0.014）**：τ=0.3 lp=2.332（省 70.8%）
excess=0.259346 ⇒ **18.52×**；τ=0.6 lp=4.268（省 46.6%）excess=0.116819 ⇒ **8.34×**；
τ=0.9 lp=5.967（省 25.4%）excess=0.011661 ⇒ **0.83≈0.84×**；τ=0.95 lp=6.398（省 20.0%，**不属 ≥25% 档**）
excess=0.003493 ⇒ 0.25×。⇒「省 ≥25%」= {τ=0.3, 0.6, 0.9}，区间 **0.84–18.5**。

---

## 2 命令结果

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | **OK：46 个 project.yaml 全部符合 docs/research-schema.md。**（EXIT=0） |
| `python3 -m pytest tests/ -q` | **834 passed in 347.70s**（CPU；未用 GPU） |
| `python3 -m pytest tests/test_axis2_exit_decision.py -q`（复核者上轮） | 19 passed（本轮未再单跑，全量已含） |
| `grep -rn "8.3–18.5\|8.3-18.5" axes/2-depth` | 2 处命中，**均带正确限定**「省 ≥25% 档…其中 τ=0.3/0.6 为 8.3–18.5×」（analysis.md:215、project.yaml:16） |
| `git diff --check` | 干净（EXIT=0） |
| `git status --short` / 提交后 | 工作区干净，`staged=0` |
| `git show --name-only HEAD \| grep -c '^pathlm/'` | **0**（实现代码零改动） |
| `sha256sum` 入库两件 vs artifact 源 | REPAIR-round2.md `6169e5e7…` 一致；REVIEW-3.md `f342ca6e…` 一致 |

---

## 3 全面对平清单（命中 → 判定）

> 判定取值：**正确** / **历史复核报告中的引用（不改）** / **已改**。原始数字证据 `results.json`
> 与历史复核件 `REVIEW-1/2/3.md`、`REPAIR-round2.md` 不列为「当前事实」检查对象。

| 扫描词 | 命中（当前事实文件） | 判定 |
|---|---|---|
| `16 条` | 无命中（当前事实已无；`TICKET:229` 的「16→19」是变更记录） | 正确（B1 已清） |
| `19`（单测） | `README.md:20`、「`analysis.md:20/235`」、「`project.yaml:17/115`」、「`TICKET:152/153/229/236/261/265/288`」 | 正确（= 19 条单测/19 passed） |
| `11 条臂` | 无命中（`TICKET §4.5` 已是 14；`REVIEW-2` 里的「11 条臂」为历史引用） | 正确（B2 已清） |
| `14 条臂` | `README.md:49`、`TICKET:162` | 正确（`results.json::arms` = 14） |
| `0.0035` | `analysis.md:126/194`、`project.yaml:17`、`TICKET:166` | 正确（均归属 τ=0.95「省 20% 档」，限定无误） |
| `0.0117` | `analysis.md:32/125/194`、`project.yaml:17/31`、`TICKET:166/220/268` | 正确 |
| `0.2593` | `analysis.md:123/194`、`project.yaml:17/31`、`TICKET:166/220/268` | 正确 |
| `0.84` | `analysis.md:125/194`、`project.yaml:17/31`、`TICKET:220/268` | 正确（`2.2.3`/`run.py` 的 `0.84…` 为哈希/无关数，非本项） |
| `8.3` | `analysis.md:124`（表：τ=0.6=8.3×，正确）；`analysis.md:215`、`project.yaml:16` **← B3 残留**；`TICKET:269/270`（= 订正说明，引用旧值）；`98.3M`/`8.375`/哈希为无关数 | **已改**（215/16）+ 正确 |
| `18.5` | `analysis.md:123/194`、`project.yaml:17/31`（正确）；`analysis.md:215`、`project.yaml:16` **← B3 残留**；`TICKET:167` **← O1 残留**；`TICKET:220/268`（正确） | **已改**（215/16/167）+ 正确 |
| `0.25` | `analysis.md:126`（表：τ=0.95=0.25×，正确）；`TICKET:167` **← O1「0.25–18.5」** | **已改**（167）+ 正确 |
| `28–55` | 无命中 | 正确（B4 已清） |
| `28–71` | `analysis.md:54/208`、`project.yaml:36`、`TICKET:228/270` | 正确（`generation_agreement` 0.2812–0.7109） |
| `1e-05` | `TICKET:271`（= 处置变更记录） | 正确 |
| `4.6e-05` | `README.md:38`、`MODEL.md:34`、`analysis.md:112/114`、`project.yaml:16/31`、`TICKET:164/224/271/273` | 正确（锚点 Δbpc 上限） |
| `≤0.005`/ECE（O2） | `TICKET:173` **← 陈旧** | **已改**（→ 最大 0.0051） |
| `831`（O3） | `TICKET:157` **← 陈旧**（= F11 前的旧计数） | **已改**（→ 834 passed） |
| `8.3–18.5`（历史件） | `REVIEW-1/2/3.md`、`REPAIR-round2.md` | 历史复核报告中的引用（**不改**） |

**遗留观察**：`REPAIR-round2.md:36/38` 仍含「超出量是 8.3–18.5 倍噪声带」——这是 round-2 处置报告
（artifact 原样导入的历史件），按任务「历史件不改」保留；其作为「当时指令」的措辞正是 B3 产出的
来源，REVIEW-3 §④ 已记录该指令本身自相矛盾。

---

## 4 未决 / 残余

1. **`_missing_mask` 的 head 维广播**（`pathlm/depth_ar.py::step_auto` + `fill_kv=False`）仍未在
   GPU 验证（REVIEW-2 N5、`analysis.md` §8 U7）；本实验走 CPU、生产路径 `fill_kv=True` 未触发。
   **未擅改实现**（非本轮授权范围）。
2. **`F5`（milestone 验收口径）**：字面验收（Δ ≤ 0.014）在 τ≥0.99 成立但实质否证，是否收紧为
   「相对同算力固定深度前沿超出量 ≤ 0」仍待 Main 裁（属框架性取舍）。
3. 本轮把 TICKET §4.5 的 3 处陈旧数字（O1/O2/O3）一并订正——这三处不在任务点名的 B3 两处内，
   但属 REVIEW-3 §④ 明确登记的观察项、且同为「账实不符」，按任务「全面对平（防再犯）」处理；
   已在其后新增的 §7.5 逐条记录，保留可回溯性。
4. 未重跑实验（纯 CPU ~343s）：以「仓内 `results.json` 未改 + REVIEW-2 重跑 sha256 逐位相同 +
   本轮 `frontier_comparison` 字段级复算」代替独立重跑（复核者 REVIEW-3 §⑤ 亦采用同策略）。

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "按 REVIEW-3 BLOCK 修 B3 2 处（analysis.md §8 末、project.yaml failure_analysis④ 的 8.3→0.84 并加限定），并依任务「全面对平」订正 TICKET §4.5 的 O1/O2/O3、§7.4 表述，入库 REPAIR-round2.md/REVIEW-3.md（原样），追加 TICKET §7.5。未动实现代码与测量数字；对平范围限于任务点名的 .scratch/61-axis2-depth/** 与 axes/2-depth/**。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "本报告含 人话摘要/改动点位/命令结果/全面对平清单（逐命中→判定）/未决；提交 e970990 含 5 文件、staged=0、pathlm 改动=0；summary --check OK；pytest 834 passed；grep 8.3–18.5 两处均带正确限定。"
    }
  ],
  "changedFiles": [
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md",
    "axes/2-depth/2.1.1-early-exit/project.yaml",
    ".scratch/61-axis2-depth/TICKET.md",
    ".scratch/61-axis2-depth/REPAIR-round2.md",
    ".scratch/61-axis2-depth/REVIEW-3.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md（EXIT=0）"
    },
    {
      "command": "python3 -m pytest tests/ -q",
      "result": "passed",
      "summary": "834 passed in 347.70s（CPU，未用 GPU）"
    },
    {
      "command": "grep -rn \"8.3–18.5\\|8.3-18.5\" axes/2-depth",
      "result": "passed",
      "summary": "2 处命中（analysis.md:215、project.yaml:16），均带限定「省 ≥25% 档…其中 τ=0.3/0.6 为 8.3–18.5×」"
    },
    {
      "command": "python3 (读 results.json::frontier_comparison 复算 excess/0.014)",
      "result": "passed",
      "summary": "τ=0.3 18.5×/τ=0.6 8.3×/τ=0.9 0.84×/τ=0.95 0.25× ⇒ ≥25% 档区间 0.84–18.5"
    },
    {
      "command": "sha256sum 入库两件 vs artifact 源",
      "result": "passed",
      "summary": "REPAIR-round2.md 6169e5e7… 一致；REVIEW-3.md f342ca6e… 一致（原样导入）"
    },
    {
      "command": "git add <5 显式路径> + git diff --cached --name-only + git commit -F - <<EOF",
      "result": "passed",
      "summary": "staged=5 逐条核对；提交 e970990；git show --name-only | grep -c ^pathlm/ = 0"
    }
  ],
  "validationOutput": [
    "summary.py --check = OK（46 份）；pytest tests/ = 834 passed。",
    "8.3–18.5 在 axes/2-depth 仅 2 处且均带正确限定；16 条/11 条臂/0.25–18.5/28–55 在 TICKET 当前事实区无残留。",
    "pathlm/** 零改动；实验 results.json 与 repro-review2 逐字节未改；git diff --check 干净；提交后 working tree 干净、staged=0。"
  ],
  "residualRisks": [
    "_missing_mask 的 head 维广播仍未在 GPU 验证（REVIEW-2 N5 / analysis.md §8 U7）；未擅改实现。",
    "F5（milestone 验收口径 0.014 vs 前沿超出量 ≤0）仍待 Main 裁。",
    "REPAIR-round2.md:36/38 仍含历史指令措辞「8.3–18.5」（历史件，按任务不改）。"
  ],
  "noStagedFiles": true,
  "diffSummary": "5 文件：analysis.md/project.yaml 各 1 行（B3 0.84–18.5 修正）；TICKET.md 新增 §7.5 并订正 §4.5 O1/O2/O3 与 §7.4/§8 表述；新增 REPAIR-round2.md、REVIEW-3.md（原样入库）。无实现代码与测量数字改动。",
  "reviewFindings": [
    "no blockers: B3 两处 8.3→0.84 并加 τ=0.3/0.6 限定 ⇒ 与 conclusion/argument③/§7② 的 0.84–18.5 自洽；O1/O2/O3 已订正；入库件 sha256 一致；summary --check OK、pytest 834 passed、pathlm 零改动。",
    "信息: 顺带订正 TICKET §4.5 的 O1/O2/O3（超出 B3 点名 2 处的范围，但属 REVIEW-3 §④ 登记的同类账实不符，已按「全面对平」处理并在 §7.5 留痕）。"
  ],
  "manualNotes": "本轮为终局收口 writer（fresh、单一写入者）。提交 e970990。B3 的原始错误措辞源自 REPAIR-round2/REVIEW-2 自相矛盾的修法指令（REVIEW-3 §④ 已记录），本轮按验收值 0.84 落盘。未合并/未 push（仅本地 commit）。建议新增 fresh 复核（round 4）核 B3 是否放行。"
}
```
