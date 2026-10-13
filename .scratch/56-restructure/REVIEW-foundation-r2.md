# ticket 56 · WP-M1.0（规范冻结）第二轮独立复核报告

> 复核者：fresh 上下文、批判性（以「上一轮修复可能没修对 / 引入回归」为前提去证伪）；**只读**，未改动仓库工作树。
> 复核对象：基线 `ce2610a`，修复提交 `e63bcdc`（+ 更正 `fb81a71`、`a2bcffd`）；被复核范围 = `ab1d8d8`/`dff555f` 的修复增量。
> 复核时间锚：2026-10-13T10:09:36+08:00（`date -Iseconds`）。
> 证据来源：`git show`、`git reflog`、实跑 4 条门命令、`/tmp/rev56-r2/` 注入坏 YAML 的对照实验。
> 上一轮报告 = `.scratch/56-restructure/REVIEW-foundation.md`（FAIL，B1 + N1–N8）；修复报告 = `.scratch/56-restructure/REPAIR-foundation.md`。

---

## 0. 人话摘要

先说结论：**PASS**。上一轮判 FAIL 的那个阻塞项——出资人要求的「做不出来时要说清是设计问题 / 结构问题 /
还是 innovation 本身失败」——**这一轮真的补上了，而且是硬门**：schema 里新增了 `failure_analysis` 字段并写清
语义，校验器把它列进必填键、还加了一道「只要状态是『被否证』而归因栏为空就判失败」的门。我在 `/tmp` 里
造了个假研究项，亲测：状态改「被否证」+ 归因留空 ⇒ 校验器 **拒**；归因删键 ⇒ 拒；归因填上内容 ⇒ 过；
状态不是「被否证」时归因留空 ⇒ 过；连只填空格也算空 ⇒ 拒。四类边界都对。

8 条非阻塞项也逐条核过：文档与实现的口径统一了（`evidence` 路径那条）、`SCHEMA.md` 不再把已退役的
`build_wiki`/`catalog`/`check_catalog_strict` 当现行机制（每处都加了退役指针、没删掉仍生效的纪律）、
`validation`「判据」门真的会拒只写对照+口径的写法、关键字段空串会被拒、`CONTRIBUTING.md` 编号连续、
2434 被注明是「工作树文件计数」、§9 与 §12 的空格统一。四条门命令（`summary.py --check`、`check_tree.py`、
`check_schema.py`、`pytest`）**全绿**，`docs/structure.md` §1–§6 的 exact wording 一个字符没动。
提交卫生也没问题：三个提交的路径清单与提交信息逐条对得上、既有的 4 个未提交改动没被卷进去、没有 amend 历史。

只有三处**非阻塞**的边角还在（都不是上轮判据的重复，不影响 PASS）：①§2 表里除被点名的 5 个字段外，
`expected_effect`/`sota_baseline` 等「必填」字符串仍允许空串；②`validation`/判据门是关键词匹配，天生浅；
③`SCHEMA.md` §0 表格那一行仍写 `catalog.yaml`（但整节上方已有退役说明）。详见 §2。

---

## 1. 逐条判据

### 判据 1 · B1 已真修（schema 字段 + 校验器 REQUIRED + `被否证`硬门）— ✅

**文档侧**：`docs/research-schema.md`
- §2 字段表第 50 行新增 `| failure_analysis | str | ✅ | **失败归因**：…先区分是 **设计问题 / 结构问题 /
  innovation 本身失败** 中的哪一类（可并列）并给证据…触发与非空要求见 §7；未判失败时写 `""` |`。
- §5（milestone 后）、§6（work_packages 后）各加一句触发说明；§7 明写「`status = 被否证` … ⇒
  `failure_analysis` **必填且非空**；校验器对 `status = 被否证` **硬性强制**（空 ⇒ fail）」；
  §9 校验项清单同步加入「字段齐全（含 `failure_analysis`）… 被否证 ⇒ `failure_analysis` 非空」。

**实现侧**：`tools/check_tree.py`
- `REQUIRED` 元组含 `"failure_analysis"`（第 40 行）。
- 硬门（`validate_file` 内）：`if st == "被否证" and not _nonempty(doc.get("failure_analysis")): errors.append(...)`。
- `_nonempty` 对字符串做 `.strip()`，故纯空格视同空。

**注入复现**（`/tmp/rev56-r2/`，`tools/` + 一份无锚点最小 `project.yaml`；`git init` 后基线 `OK` exit 0）：

| 注入 | 期望 | 实测 | 结论 |
|---|---|---|---|
| `status=被否证` + `failure_analysis: ""` | 拒 | `status=被否证 但 failure_analysis 为空（必须区分 设计问题 / 结构问题 / innovation 本身失败 并给证据）` exit 1 | ✅ |
| `status=被否证` + 删 `failure_analysis` 键 | 拒 | `缺字段 failure_analysis` **且** 被否证门，共 2 条 exit 1 | ✅ |
| `status=被否证` + `failure_analysis: "结构问题…"` | 过 | `OK` exit 0 | ✅ |
| `status=被否证` + `failure_analysis: "   "`（纯空格） | 拒 | 被否证门 exit 1 | ✅ |
| `status=未开始` + `failure_analysis: ""` | 过 | `OK` exit 0 | ✅ |

复现命令（节选）：
```
mkdir -p /tmp/rev56-r2/axes/test/1.1.1-hard-window/experiments && cd /tmp/rev56-r2
# 写一份最小 project.yaml（无 @锚点），git init 后 python3 tools/check_tree.py ⇒ OK
# 把 status 改 "被否证"、conclusion 填非空、experiments 指向已建目录、failure_analysis 留空 ⇒ FAIL（1 项）
# failure_analysis 填 "结构问题：…" ⇒ OK
```

**两份 exemplar 也补齐**：`axes/1-time/1.1.1-hard-window/project.yaml`（非空，含「设计问题」初判）、
`axes/9-backlog/9.1-selector-scheduling/project.yaml`（`""`，因状态 `未开始`）——当前仓内
`check_tree.py` 对 2 份 YAML 报 `OK`，即新必填键未破坏既有样本。

### 判据 2 · N1–N8 逐条真修 — ✅（含 3 处非阻塞边角，见 §2）

| 项 | 判据 | 复核 | 复现 |
|---|---|---|---|
| **N1** | 文档与校验器口径一致（方案 a） | ✅ 选 (a) 并落地：`docs/research-schema.md` §9 改为「`experiments` 路径必须存在；`work_packages[].evidence` 是『证据将落在哪』，允许**未来路径**（只收锚点、不强制已存在）」，与 `check_tree.py` docstring 第 11 条、代码注释完全一致 | `/tmp`：顶层 `experiments` 指向不存在路径 ⇒ `FAIL：experiments 路径不存在`；WP `evidence` 指向不存在未来路径 ⇒ `OK`（与文档一致） |
| **N2** | `SCHEMA.md` 不再把退役工具当现行；未误删生效纪律 | ✅ 开篇(3–6)改为「`check_schema.py`（已转调 `check_tree.py`）…历史生成式 wiki…**已退役**」；§0/§3/§4/§12/§17.13/§21/§22 逐处加退役指针；`git diff` 显示**纯新增指针 + 少量在原文后追加括注**，无一条分析纪律被删 | `grep -n "build_wiki\|catalog\.yaml\|check_catalog_strict" SCHEMA.md` ⇒ 命中处均带退役指针；`git show e63bcdc -- SCHEMA.md` 无删除生效条款 |
| **N3** | `validation` 判据门会拒「只含对照+口径」 | ✅ 新增第 9 条 `validation 缺「客观判据」（该涨/该跌/必须不变）` | `/tmp`：`validation: "对照 = stock 基线；口径 = matched-step bpc"` ⇒ `FAIL：validation 缺「客观判据」` exit 1 |
| **N4** | 关键字段空串会被拒 | ✅ `NONEMPTY = (purpose, innovation, validation, objective_criteria, falsification)`，`_nonempty` 拒绝空串/空列表 | `/tmp`：单独把 `purpose`/`falsification`/`objective_criteria` 置 `""` ⇒ 各 `FAIL：<字段> 不得为空` exit 1 |
| **N5** | `CONTRIBUTING.md` 编号连续 | ✅「硬性规则」由 `1.2.4.5.6.7.` 重排为 `1.2.3.4.5.6.` 连续 | 读 `CONTRIBUTING.md` 第 19–30 行：`1.`…`6.` 无跳号 |
| **N7** | 2434 注明为工作树计数口径 | ✅ `MAPPING.md` §0 加「**工作树文件计数**（`find axes -type f`），含约 777 个被 gitignore 的产物…非 git 跟踪文件计数…清 cache 会漂移，须并记 `git ls-files axes \| wc -l`」；`57-program/TICKET.md` M2 行同注 | 实测 `find axes -type f \| wc -l`=**2436**；`git ls-files axes \| wc -l`=**1659**；2436−2（ab1d8d8 新增 2 文件）=2434，且 1659+777=2436，口径自洽 |
| **N8** | 空格统一 | ✅ `docs/structure.md` §9 改为「调度 / 选择器」，与 §12 第 282 行逐字一致 | `grep -n 调度 docs/structure.md` ⇒ 119、282 两行均为「调度 / 选择器」 |

### 判据 3 · 无回归（4 条门全绿 + §1–§6 exact wording 未坏）— ✅

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | `OK：2 个 project.yaml 全部符合 docs/research-schema.md。` exit 0 |
| `python3 tools/check_tree.py` | 同上，exit 0 |
| `python3 tools/check_schema.py` | 同上（转调 `check_tree.main()`），exit 0 |
| `python3 -m pytest tests/ -q` | `827 passed in 99.12s`，exit 0 |

- **§1–§6 exact wording**：`git diff ab1d8d8 HEAD -- docs/structure.md` **只有 §9 那一行时空格改动**，
  §1–§6 与 §12 均未被触碰。脚本比对 §12 的 57 行标题 vs §1–§6 条目：唯一「无 §1–§6 来源」的仍是
  `9.1`/`9.2`（二者来自 §9 Backlog，编号属 §12 职责，非本轮回归）。
- 上一轮曾报的「WP evidence 路径存在性未强制」现**已被文档收窄为一致口径**（N1），不再是文档/实现冲突。

### 判据 4 · 提交卫生 — ✅

- `git show --stat` 与 `git show --name-only` 三提交路径集合逐条一致：
  - `e63bcdc` = **11 个文件**（`.scratch/56-restructure/{MAPPING,REPAIR-foundation,REVIEW-foundation}.md`、
    `.scratch/57-program/TICKET.md`、`CONTRIBUTING.md`、`SCHEMA.md`、两份 exemplar `project.yaml`、
    `docs/research-schema.md`、`docs/structure.md`、`tools/check_tree.py`）——与提交信息列举的改动**逐条对得上**，无未声明文件、无声明却缺的文件。
  - `fb81a71` = 仅 `.scratch/56-restructure/TICKET.md`（+54）；提交信息只声称该文件。
  - `a2bcffd` = 仅 `tools/check_tree.py`（+1/−1，删行尾空白）；提交信息只声称该文件。
- **未卷入工作树既有 4 个未提交改动**：`git show --name-only e63bcdc | grep -E "INDEX.md|catalog.yaml|C-vertical"`
  ⇒ 无命中；`git status` 仍显示这 4 个文件为 ` M`（未提交），与上一轮复核（09:50）所见一致 ⇒ 系既有改动、非本轮引入。
- **未 amend 历史**：`git reflog` 依次为 `commit:`（`e63bcdc`→`fb81a71`→`a2bcffd`），无 `commit (amend)`；
  三条更正走的是**追加新提交**而非改写。
- **无 staged 残留**：`git diff --cached --name-only | wc -l` = `0`。
- **无行尾空白**：`git diff --check ce2610a HEAD` 干净。
- `.scratch/56-restructure/TICKET.md` §6 记录了 round 2 的执行日志（触发 / 改动 / 命令 / 未决），与 `REPAIR-foundation.md` 一致。

### 判据 5 · 未复核范围如实列出 — ✅

修复报告 `REPAIR-foundation.md` §4 与工单 §6.4 均如实声明「L1 逐字真实性、真实时钟、`docs/structure.md`
§1–§11 是否未改」不可核，并**未新增**对这些不可核事实的主张（N6 反而把它记为不可核）。见本报告 §3。

---

## 2. Material defects

### 阻塞（blocking）

**无。** 上一轮的 B1 已被真修并机器可证伪（见判据 1）。

### 非阻塞（non-blocking）——均非上轮判据的重复，不影响 PASS

**R1 · §2「必填 = 校验器会 fail」对点名 5 字段之外仍偏强**
- 位置：`docs/research-schema.md` §2 表头与各行；`tools/check_tree.py:NONEMPTY`。
- 事实：`expected_effect` / `sota_baseline` / `gap` / `method` / `argument` / `conclusion_review` 在 §2 标 ✅，
  但只做**键存在**校验，空串仍可通过。
- 复现：`/tmp` 中把 `expected_effect: ""` 与 `sota_baseline: ""` ⇒ `OK` exit 0。
- 判定：N4 的原文建议只点名 `purpose/innovation/validation/objective_criteria/falsification` 五字段，
  修复**按该范围**落地，属一致实现；此条为剩余边角，非「没修对」。

**R2 · 判据门是关键词匹配，天生浅**
- 位置：`tools/check_tree.py` `validation` 三要素门。
- 事实：`validation` 只要含「必须不变」等关键词即过，不保证真是可证伪判据；`NONEMPTY` 亦只拒空串。
- 判定：N3 的判据是「会拒只含对照+口径的写法」，实测**会拒**（见判据 2）；深语义判据非本轮范围。

**R3 · `SCHEMA.md` §0 表格行文本仍写 `catalog.yaml`**
- 位置：`SCHEMA.md` 第 27 行（§0 表「目录完备 / 谁来保证 = `catalog.yaml` + `check_schema.py`」）。
- 事实：该行字面仍把退役工具写成「谁来保证」，但整节**上方**已有明确退役说明（21–23 行）覆盖此列。
- 判定：N2 要求「每处带退役指针或无『现行』断言」，该处由节级说明覆盖，非「当现行机制」。

---

## 3. 未复核到的范围（如实列出）

- **用户 L1 逐字的真实性**：无法回溯验证 `TICKET.md` §5.1 引文与出资人原始输入逐字相同；只能确认其存在且标为 L1。
- **真实时钟**：仅验证时间戳格式为 `date -Iseconds` 兼容（`+08:00` ISO8601）且先后自洽，无法独立核验 09:40–10:09 的真实时刻。
- **`docs/structure.md` §1–§11 是否为「未改动的既有内容」**：git 无该文件先前版本（`ce2610a` 不存在），「§1–§11 未改」**不可独立核验**；本轮只验证**修复增量**未触碰 §1–§6。
- **M2 的实际迁移**（逐轴建 37 份 `project.yaml`、搬 `axes/**` 证据、清残留）：不在本 WP 范围，未复核。
- **`/tmp` 注入为最小复现**，非对全仓 37 格未来 YAML 的穷举；当前仓内只有 2 份 YAML，校验器在大样本下的行为未测。
- **`check_catalog_strict.py` / `build_wiki.py` 是否仍可运行**：已退役，不属验收目标，未复核。
- **提交命令的调用形式**（是否用 `git commit -F -` heredoc）：无法从 git 对象反推，未核。

---

## 4. 结论

**PASS**。阻塞 B1 已真修且经 `/tmp` 四类边界注入证伪通过；N1–N8 逐条落地；4 条门命令全绿；
`docs/structure.md` §1–§6 exact wording 未坏；提交卫生合格。仅存 3 处非阻塞边角（R1–R3）。
建议：进入 M2（逐轴迁移）前，可选地顺手把 R1（扩展 `NONEMPTY` 覆盖面）与 R3（§0 该行加内联指针）清掉，非强制。
