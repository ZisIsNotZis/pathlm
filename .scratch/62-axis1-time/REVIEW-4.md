# REVIEW-4 · 轴 1 时间轴 Phase A 终审复核（研究项 `1.1.4.1` / 实验 `2638-carry-frontier-13m`）

## 0. 人话摘要（先读这一段）

**一句话：round-3 那条唯一的 blocking 的「核心矛盾」已经消解了——`project.yaml` 里「REVIEW-2 待派」和 `change_log` 里「已按 REVIEW-2 修」打架的那句话没了，`conclusion_review` 现在也把 REVIEW-1/2/3 三轮判词都写上了；科学数字一个没动、`summary.py --check` 全绿、`pytest` 834 条全过、工作树干净。但我仍判 FAIL，而且是窄范围 FAIL：因为 round-3 给 R3-B1 开了三步修法，提交只做了第 1 步，剩下两步没做、也没登记——**

1. **`project.yaml:20` 那段「重复的、过时的」独立复核句没删**：它和 940cb23 时**逐字相同**（`独立复核（REVIEW-1.md）：见该文件判词。复核前未改本 project.yaml 的判据或数字；若有缺陷按 SCHEMA §14/§14b 返工。`）。它和上面 `:19` 已经写全 R1/R2/R3 的那句**重复**（同一事实在一份事实文档里出现两次，违反 `AGENTS.md` §2.1「一条事实一处」），而且它只认 REVIEW-1、还留着「若有缺陷…返工」这种「复核还没结论」的口吻，和 `:19`「三轮复核均未推翻任何数字或判据」的语气不一致。
2. **`WP1.review`（`project.yaml:81`）仍只列 `REVIEW-1.md`**：实际有三轮复核件（REVIEW-1/2/3），字段里只写了一个 ⇒ 任务书里「`WP1.review` 指向完整」这条**没达成**。

另外 round-3 的 6 条 non-blocking（N1–N6）里，N2/N3/N4/N5/N6 都真修了，只有 **N1 修了一半**：`README.md:122` 的「单测 5 条 → 6 条」改了、主题也补了，但**同一个单测文件自己的 module docstring（`test_2638_frontier.py:6`）还写着「钉住三件事」**（实际 6 个 `test_*`）——又是「改了一处、兄弟处没同步」的老形态。

**所以我给 BLOCK（窄范围）**：科学结论 PASS、数值零回归、主矛盾已消，但 R3-B1 的三步修法只完成 1 步（`WP1.review` 不完整、`:20` 重复句未删），加 N1 的兄弟处未同步。**全部是纯文字/指针，改 3 行左右，不需要重跑 GPU、不需要重训、不需要重评。**

> 框架性取舍（留给父级裁决）：若父级认为「`review` 字段写单指针」+「`:20` 重复句」属可在收口后随手补记的**非关键记账**（而不是必须在收口前清掉的缺陷），则本条降为 P2，总判词转「OK with notes，轴 1 Phase A 可收口」。但按本轮任务书明写的验收项「`WP1.review` 指向完整」，我默认判 BLOCK。

---

## 1. 复核对象 / 复核者身份 / 能力（与 round-1..3 不同：本轮有 shell）

- **复核对象 revision**：**`HEAD = 54afd31`**（= 修复提交 `69d431b` + 锚点前移 `54afd31`）。`git status --porcelain` 为空 ⇒ 工作树对该 HEAD **干净**。
- **复核者身份**：fresh 上下文（未参与本线任何实验/文档/前几轮复核，非本项 writer）。本报告为唯一写入物（见 §5 说明）；未改仓库任何**既有**文件。
- **能力（本轮具备）**：可跑 shell。已实跑 `python3 tools/summary.py --check`、`python3 -m pytest tests/ -q`、2638 自包含单测、`sha256sum`、`git log/show/status`。故 round-1..3 里「代跑」的那些项**本轮已独立复算**。

---

## 2. 逐条判词

### 2.1 项 1 · R3-B1 是否已消解 —— ❌ **只消解了核心矛盾，未消解三步修法**

| 子项 | 判 | 证据 |
|---|---|---|
| (a) `conclusion_review` 不再自相矛盾、且含 REVIEW-1/2/3 判词 | **✅** | `project.yaml:19` 现为完整的「独立复核（REVIEW-1.md，fresh 上下文）：…；REVIEW-2（fresh）：…；REVIEW-3（fresh，对象 940cb23）：…三轮复核均未推翻任何数字或判据。」**「REVIEW-2 待派」已删**（对比 `940cb23` 该行逐字）。与 `change_log` 的「已按 REVIEW-2/3 修」不再冲突。 |
| (b) `project.yaml:20` 与 `change_log` 不再打架 | **❌（未动）** | `:20` 与 `940cb23` 版**逐字相同**：`独立复核（...REVIEW-1.md）：见该文件判词。复核前未改本 project.yaml 的判据或数字；若有缺陷按 SCHEMA §14/§14b 返工。` R3 的 minimal fix 第 2 步明写「删除 `:20` 整段重复的 REVIEW-1 段」，**未执行**。该行是 `:19` 的重复陈述（§2.1），且口气像「复核未结」。 |
| (c) `WP1.review` 指向完整 | **❌** | `project.yaml:81` 仍为 `…复核件见 .scratch/62-axis1-time/REVIEW-1.md`，**只列 REVIEW-1**，漏 REVIEW-2/3。R3 minimal fix 第 3 步明写要列三个，**未执行**。 |

- **R3-B1 三步修法的执行情况**：第 1 步（回填 / 删「待派」）✅；第 2 步（删 `:20` 重复段）❌；第 3 步（补 `WP1.review`）❌。
- **口径澄清（对 R3 的一处修正）**：R3 把 `:20` 的「复核前未改本 project.yaml 的判据或数字」读成与 `change_log` 直接冲突；严格说该句带「**复核前**」时限词，且 `change_log` 记的是**复核后**对 `conclusion`/`failure_analysis④`/`gaps`/`WP1.acceptance` 的改动（并非 `method`/`validation` 等「判据」字段），所以「硬矛盾」一说偏强。但 **`:20` 作为重复、过时段落本身**仍在，R3 要求删它是对的。
- **诚实性核对**：`69d431b` 的 commit message 与 `TICKET.md` §8.6 对 R3-B1 **只声称**「回填 REVIEW-2/REVIEW-3 判词，删去『REVIEW-2 待派』」——**没有**谎称删了 `:20` 或改了 `WP1.review`；但 `change_log:122` 的小结写成「**按 REVIEW-3 修 R3-B1 + N1–N6**」，字面上把整个 R3-B1 记成已修，属**轻微以偏概全**（§24 第 13 类形态）。

### 2.2 项 2 · round-3 non-blocking N1–N6 逐条落地

| 项 | 判 | 证据（当前 HEAD） |
|---|---|---|
| **N1** README 单测条数 5→6 + 补主题 | **⚠ 半修** | `README.md:122` 已是「`test_2638_frontier.py`（**6 条**，钉住支配定义、插值、**符号约定**、掩码可见对数公式，以及**落盘 `results.json` 的支配关系与判定线数值一致**）」✅；但**兄弟处** `test_2638_frontier.py:6` module docstring 仍写「**钉住三件事**」，`grep -c "^def test_"` = **6** ⇒ 同交付内两处对「钉了几件事」不一致。 |
| **N2** 悬空指针改指结果 | **✅** | `README.md:96`：`（见 results.json 的 4 个复用臂 bpc 与 2634/analysis.md）`（旧「见 analysis §3」已删）。`analysis.md` §3 确有「状态精度脚注」；`:74` 另有一处「见 analysis §3 脚注」指向该脚注，非悬空。 |
| **N3** §4a 纯窗口曲线定义补 win17/18 | **✅** | `README.md:65`：`{(k_eff, bpc)} for win16/17/18/24/32/128（seed 0；win17/18 为执行期追加，见 §5）`。 |
| **N4** 单测 docstring 收窄钉子范围 | **✅** | `test_2638_frontier.py:73-75` 新增「范围说明（复核 R3-N4）：本条抓的是 rel 公式 ↔ 判定线的**数值**一致性…抓不住『只改文档里那行判定线文字』这类纯文本回归」——言过其实已消除。 |
| **N5** analysis 表头重复行 | **✅** | `analysis.md:3-5` 三行不再有重复的「权重清单 = MODEL.md」（旧 `:4/:5` 皆同句）。 |
| **N6** `change_log` 时间倒序 | **✅** | `project.yaml` change_log 现按 10:15:48 → 15:35:09 → 15:48:43 → 15:58:00 → 16:10:00 **正序**。 |

### 2.3 项 3 · 科学结论未动 —— **✅（逐数字独立复算，与 `analysis.md`/`results.json` 一致）**

| 项 | 文档值 | 本轮复算（`results.json`） | 判 |
|---|---|---|---|
| `gap` = 窗口16 − 全量 | **0.125421** | `1.62811 − 1.502689 = 0.125421` | ✅ |
| `win17_s0` 严格支配 `carry16_s0` | 208,896 B < 212,992 B 且 1.624104 < 1.624979 | `win17_s0 total=208896 bpc=1.624104`；`carry16_s0 total=212992 bpc=1.624979` | ✅ |
| 前沿下包络 | win16/17/18/24/32/128/full（全纯窗口点） | `lower_envelope=['win16_s0','win17_s0','win18_s0','win24_s0','win32_s0','win128_s0','full']` | ✅ |
| `carry_minus_interp_window` | −0.002259 / −0.003920 | `location`: `carry16_s0=-0.002259`、`carry16_s1=-0.00392` | ✅ |
| recovery 2.5% / 1.2% | 0.025 / 0.0117 | `recovery_ratio_r`: 0.025 / 0.0117 | ✅ |
| `dominated_by` | `carry16_s0:[win17_s0]`、`carry16_s1:[win17_s0,carry16_s0]` | `{'carry16_s0': ['win17_s0'], 'carry16_s1': ['win17_s0', 'carry16_s0'], …}` | ✅ |

`analysis.md` §0 的人话表、§1 判据对照、§2 支配/插值、`run.py` 落盘数字**四方自洽**，与本轮读取的 `results.json` 逐位相同 ⇒ **实现/测量数字未被文字修复触及**。

### 2.4 项 4 · 无回归 —— **✅**

| 命令 | 结构 | 结果 |
|---|---|---|
| `python3 tools/summary.py --check` | — | **`OK：46 个 project.yaml 全部符合 docs/research-schema.md`**（exit 0） |
| `python3 -m pytest tests/ -q` | — | **`834 passed in 212.79s`** |
| `python3 -m pytest …/test_2638_frontier.py -q` | — | **`6 passed in 2.06s`** |
| `git status --porcelain` | — | **空（干净）**；`git status -sb` = `## main...origin/main [ahead 230]` |
| `git diff --cached --name-only \| wc -l` | — | **0（无暂存）** |
| `git diff --check HEAD~2..HEAD` | — | **无输出（无空白错）** |
| `git show --stat 69d431b` | 6 文件 | `REVIEW-3.md` / `TICKET.md` / 2638 `README.md` / `analysis.md` / `test_2638_frontier.py` / `project.yaml` ——**与 commit message 声称一致，且不含任何共享文件**（无 `pathlm/**`、`tests/**`）✅ |
| `git show --stat 54afd31` | 1 文件 | 仅 `project.yaml`（锚点前移）✅ |
| `sha256sum $D/{run.py,config.json}` | — | `run.py=2023f36d…`、`config.json=eb97327d…`，与 `MODEL.md:41-42`、`results.json.code_hashes` **逐字相同** ✅ |
| `sha256sum $D/train/*/model.pt`（5 个新训练臂） | — | win17/18/24/32/128 与 `MODEL.md:5-9` **逐字相同** ✅ |
| 锚点可达 | `analysis.md @69d431b`、`results.json @443cb44` | `git log -1 -- analysis.md` = `69d431b` ✅；`results.json` 最后变更为 `12e5fb7`，但 `443cb44` 是**可达提交**且该文件自 `12e5fb7` 起内容未变 ⇒ 按 `check_tree.py` 的「锚点须可达」纪律**合规**，非缺陷（仅非「最后变更 commit」）。 |

### 2.5 项 6 · 一处轻微的记账不一致（登记，非本轮任务项）

- `project.yaml` `change_log` 新条目 `:122` 记时 **`2026-10-13T16:10:00+08:00`**，但引入它的提交 `69d431b` 的 author/committer 日期均为 **`2026-10-13T16:07:47+08:00`** ⇒ **记录的事件时间晚于承载它的提交 2 分 13 秒**（`54afd31` = 16:07:53，仍早于 16:10）。`AGENTS.md` §6 要求按实际时钟记录，此处时间戳与提交不符。纯记账噪声，不影响结论；最小修法：把该条改成实际提交时刻（如 `16:07`）。

---

## 3. 复现（本轮全部为纯读，不需要 GPU）

```bash
# R3-B1 现状（核心矛盾已消 / 但 :20 与 :81 未修）
awk 'NR>=18 && NR<=20 {printf "%d: %s\n", NR, $0}' axes/1-time/1.1.4.1-explicit-memory/project.yaml | cut -c1-120
awk 'NR==81 {print NR": "$0}' axes/1-time/1.1.4.1-explicit-memory/project.yaml          # WP1.review 仍只列 REVIEW-1
git show 940cb23:axes/1-time/1.1.4.1-explicit-memory/project.yaml | sed -n '20p'          # 与当前 :20 逐字相同
git show 69d431b -- axes/1-time/1.1.4.1-explicit-memory/project.yaml | grep -E '^\+\-|^-独立复核'  # 只改了 :19

# N1 兄弟处未同步
grep -c '^def test_' axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py   # 6
sed -n '6p' axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py            # 仍「钉住三件事」

# 科学数字 / 无回归
python3 tools/summary.py --check
python3 -m pytest tests/ -q
python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py -q
git status --porcelain ; git diff --cached --name-only | wc -l
git show --stat 69d431b ; git show --stat 54afd31
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/{run.py,config.json}
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/train/*/model.pt
git log -1 -- axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/analysis.md
```

---

## 4. 未覆盖 / 残差（如实登记）

1. **round-1/2 的其它历史项未重审**：本报告只复核「R3-B1 收口 + round-3 N1–N6 + 科学数字未动 + 无回归」，未重新逐条审计 REVIEW-1/2 里已判 PASS 的 B1–B4/R2-* 之外的项。
2. **未重跑 GPU / 训练 / 评测**：数字一致性是用落盘 `results.json` + `run_log` 复算，非重跑 `run.py` 或重训（`AGENTS.md` §8 / `SCHEMA.md` §20 的串行锁未涉及）。
3. **5 个复用臂（full / win16_s0/s1 / carry16_s0/s1）的 `model.pt` 未做磁盘 sha**：它们不在 `2638/train/` 下（复用 2610 / 2634），本轮只核了 `run.py`/`config.json` 与 5 个新训练臂（已逐字一致）；复用臂哈希为三处（`MODEL.md` ↔ `config.json` ↔ `results.json.checkpoints`）一致性核验，非磁盘复算。
4. **`git show --stat` 的「字节层」完整性**：已核 `69d431b`/`54afd31` 的文件清单与 commit message、`:122` change_log 声称一致；未逐字节 diff 全部内容。
5. **R3 对 `:20`「硬冲突」的判读**：本报告已按 §2.1 澄清为偏强；若父级采信 R3 的「硬冲突」读法，则 `:20` 问题比本报告所述更严重（同样指向「该删」）。
6. 本报告自身的落盘让工作树多出一个**未跟踪文件**（`.scratch/62-axis1-time/REVIEW-4.md`，与 REVIEW-1/2/3 由 runtime 落盘、后续提交的模式一致）——这是本复核的产物，不是对既有文件的改动。

---

## 5. 一句话总判词

**BLOCK（窄范围）**：round-3 唯一 blocking 的**核心矛盾已消解**——`project.yaml:19` 的 `conclusion_review` 已回填 REVIEW-1/2/3 判词、「REVIEW-2 待派」已删、与 `change_log` 不再打架；**科学结论 PASS**（`gap=0.125421`、`win17` 严格支配 `carry16_s0`：208,896 B ≤ 212,992 B 且 1.624104 < 1.624979、下包络、`carry_minus_interp_window`、recovery 2.5%/1.2% 逐项复算一致，实现/测量数字未改）；**无回归**（`summary.py --check` OK、`pytest tests/` 834 passed、自包含单测 6 passed、工作树干净、无暂存、两提交 `--stat` 与 commit message 一致）。**但 R3-B1 的三步修法只做了 1 步**：`project.yaml:20` 那段重复/过时的「独立复核（REVIEW-1.md）」未删（与 `940cb23` 逐字相同）、`WP1.review`（`:81`）仍只列 `REVIEW-1.md`（漏 REVIEW-2/3）⇒ 任务书验收项「`WP1.review` 指向完整」未达成；另 round-3 N1 的兄弟处 `test_2638_frontier.py:6` 仍写「钉住三件事」（实际 6 条）。**全部为纯文字/指针，改约 3 行，不需要重跑 GPU。** 若父级把这两处「review 字段单指针 + 重复 stale 行」判为可在收口后随手补记的非关键记账，则本条降 P2、可转「轴 1 Phase A 可收口」；否则按本轮验收口径判 **BLOCK**。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "以终审者身份完成 R3-B1 收口核对 + N1–N6 逐条 + 科学数字复算 + 无回归实跑，仅落盘本复核报告，未改任何既有文件、未扩大范围。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "给出逐条 ✅/❌、精确行号引用、可复现命令、实跑结果（summary.py --check OK / pytest 834 passed / 自包含单测 6 passed / sha256sum / git show --stat）与未覆盖范围。"
    }
  ],
  "changedFiles": [
    ".scratch/62-axis1-time/REVIEW-4.md（本复核报告，runtime 落盘；旧文件未改）"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "python3 tools/summary.py --check", "result": "passed", "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md"},
    {"command": "python3 -m pytest tests/ -q", "result": "passed", "summary": "834 passed in 212.79s"},
    {"command": "python3 -m pytest …/test_2638_frontier.py -q", "result": "passed", "summary": "6 passed in 2.06s"},
    {"command": "git status --porcelain / git diff --cached --name-only | wc -l", "result": "passed", "summary": "工作树干净；无暂存文件"},
    {"command": "git show --stat 69d431b / 54afd31", "result": "passed", "summary": "文件清单与 commit message 一致，无共享文件改动"},
    {"command": "sha256sum run.py config.json train/*/model.pt", "result": "passed", "summary": "run.py=2023f36d…/config.json=eb97327d… 与 MODEL.md、results.json.code_hashes 逐字一致；5 个新训练臂哈希与 MODEL.md 逐字一致"},
    {"command": "python3 - <<PY results.json 复算 PY", "result": "passed", "summary": "gap=0.125421；win17_s0(208896,1.624104) 严格支配 carry16_s0(212992,1.624979)；下包络/插值/recovery 一致"},
    {"command": "git log -1 -- 2638/analysis.md 与 results.json", "result": "passed", "summary": "analysis.md 最后变更=69d431b（锚点正确）；results.json 最后变更=12e5fb7（443cb44 可达且内容未变，按 schema 合规）"}
  ],
  "validationOutput": [
    "summary.py --check: OK：46 个 project.yaml 全部符合 docs/research-schema.md",
    "pytest tests/ -q: 834 passed",
    "2638 自包含单测: 6 passed；grep -c '^def test_' = 6（与 README「6 条」一致；module docstring 仍『钉住三件事』）",
    "git status 干净；git show --stat 69d431b=6 文件（.scratch/62-axis1-time/{REVIEW-3.md,TICKET.md}+2638/{README.md,analysis.md,test_2638_frontier.py}+project.yaml），无共享文件",
    "project.yaml:19 已回填 R1/R2/R3 判词、『REVIEW-2 待派』已删；project.yaml:20 与 940cb23 逐字相同（重复 stale 段未删）；project.yaml:81 WP1.review 仍只列 REVIEW-1.md"
  ],
  "residualRisks": [
    "blocker（窄）：project.yaml:81 的 WP1.review 仍只列 REVIEW-1.md（漏 REVIEW-2/3）——R3-B1 minimal fix 第 3 步未执行，任务书验收项『WP1.review 指向完整』未达成",
    "blocker（窄）：project.yaml:20 重复/过时的『独立复核（REVIEW-1.md）…』段未删（与 940cb23 逐字相同）——R3-B1 minimal fix 第 2 步未执行，违反 AGENTS.md §2.1「一条事实一处」",
    "non-blocking：test_2638_frontier.py:6 module docstring 仍『钉住三件事』（实际 6 条）——R3-N1 兄弟处未同步",
    "non-blocking：project.yaml change_log 条目 :122 记时 16:10:00 晚于承载它的提交 69d431b（16:07:47），与 AGENTS.md §6『实际时钟』不符",
    "未覆盖：round-1/2 其它历史项未重审；未重跑 GPU/训练/评测；5 个复用臂 checkpoint 未做磁盘 sha（仅三处一致性核验）"
  ],
  "noStagedFiles": true,
  "diffSummary": "本复核未改动仓库任何既有文件；唯一新增物为 runtime 落盘的复核报告 .scratch/62-axis1-time/REVIEW-4.md（未跟踪，与 REVIEW-1/2/3 落盘模式一致）。",
  "reviewFindings": [
    "blocker: axes/1-time/1.1.4.1-explicit-memory/project.yaml:81 - WP1.review 仍只列 REVIEW-1.md，未列 REVIEW-2/3（R3-B1 minimal fix 第 3 步未执行）",
    "blocker: axes/1-time/1.1.4.1-explicit-memory/project.yaml:20 - 重复且过时的『独立复核（REVIEW-1.md）：见该文件判词…』段未删（与 940cb23 逐字相同；R3-B1 minimal fix 第 2 步未执行）",
    "non-blocking: axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py:6 - module docstring 仍『钉住三件事』，实际 6 个 test_*（R3-N1 兄弟处未同步）",
    "non-blocking: axes/1-time/1.1.4.1-explicit-memory/project.yaml:122 - change_log 记时 16:10:00 晚于其提交 69d431b（16:07:47）",
    "已消解: project.yaml:19 conclusion_review 已回填 REVIEW-1/2/3 判词、『REVIEW-2 待派』已删，核心 self-contradiction 消除",
    "科学结论未动、数值零回归：gap=0.125421、win17 严格支配 carry16、下包络/插值/recovery 逐项一致"
  ],
  "manualNotes": "默认判词 = BLOCK（窄范围），因任务书验收项明确要求『WP1.review 指向完整』而 :81 只列 REVIEW-1。若父级认定『review 字段单指针 + :20 重复 stale 行』属收口后随手补记的非关键记账，可降 P2 并转『轴 1 Phase A 可收口』——此属框架性取舍，留待父级/用户裁决。报告已同时写入仓库 .scratch/62-axis1-time/REVIEW-4.md 与 runtime artifact 路径。"
}
```
