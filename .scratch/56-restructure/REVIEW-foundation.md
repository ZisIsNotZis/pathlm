# ticket 56 · WP-M1.0（规范冻结）独立复核报告

> 复核者：fresh 上下文、批判性（以「上一份工作做砸了」为前提去证伪）；**只读**，未改动仓库工作树。
> 复核对象：基线 `ce2610a`，复核提交 `ab1d8d8`（+ 更正 `dff555f`）。
> 复核时间锚：2026-10-13T09:50（`date -Iseconds`）。
> 证据来源：`git show/diff`、实际运行 5 条门命令、`/tmp/rev56/` 注入坏 YAML 的对照实验。

---

## 0. 人话摘要

这份 foundation 交付**主体是好的，但没到「可以照它开工」的程度**：研究树规范（`docs/structure.md` §12–§15）、
新 schema（`docs/research-schema.md`）、校验/汇总工具（`check_tree.py` / `summary.py`）、旧→新映射（`MAPPING.md`）
四件都真实存在、内容完整、命令真能跑（`summary.py --check`、`check_tree.py`、`check_schema.py`、`pytest` 全绿），
37 个旧格在 `MAPPING.md` 里一一对应、一条不漏。

但我用「注入坏 YAML 看校验器拒不拒」的方式证伪时发现**两处账实不符**，其中一处是**阻塞**：
用户原话明确要求「做不出来时要区分是设计问题 / 结构问题 / innovation 失败」，而冻结后的 schema
**根本没有「失败归因」（failure_analysis）这个字段**，校验器也不查——这意味着 M2 各大类照抄 schema 时，
这条用户要求会整批漏掉。另一处（非阻塞）：`docs/research-schema.md` §9 白纸黑字写「`experiments` / `evidence`
里的路径必须存在」，但校验器对 **work_package 的 `evidence`** 指向不存在路径**不报错**，文档的承诺大于实现。

另外发现若干文档不一致（`SCHEMA.md` 开篇仍把 `build_wiki`/`catalog` 当作现行前提、§21/§22 仍要求跑已退役工具；
`CONTRIBUTING.md` 规则编号跳号；提交信息把 `docs/structure.md` 说成「追加」而 git 记录是「新增」），
都属非阻塞。

**结论：FAIL**（阻塞 1 项：schema 缺失败归因；非阻塞若干）。

---

## 1. 逐条判据

### 判据 1 · exact wording（§1–§6 用词 == §12 节点清单）— ✅

**方法**：脚本抽取 `docs/structure.md` §1–§6（含 `###` 标题 / 加粗条目 / 缩进子项）与 §12 表格（57 行），
归一化去空白与 `**` 后做逐条包含比对。

```
§12 rows: 57 | §1-6 items: 56
exact-wording failures: 9.1 / 9.2（无 §1–§6 来源行，来自 §9）
```

- §12 的 57 行标题**全部**在 §1–§6 对应 id 的行文本中逐字出现（含 `2.1.3`/`2.2.5` 这两种写在括号里的
  Backlog 行）：`（2.1.3 repeating 循环 transformer：多算力换少参数 —— Backlog，不进编号）`、
  `（2.2.5 权重共享 —— Backlog）`。
- 唯一「无 §1–§6 来源」的是 `9.1`/`9.2`：§9 从不带编号地列出 `调度/选择器（原 G 轴）`、`迭代精化的其余（原 F 轴剩余）`，
  §12 给它们补了 `9.1`/`9.2` 编号（编号是 §12 的职责，非改名）。
- **无改名、无自造用词**。
- 细枝（非阻塞）：§9 写「调度/选择器」（无空格），§12 写「调度 / 选择器」（有空格）——纯排版差异。

### 判据 2 · schema 完整且被强制 — ❌（阻塞）

用 `/tmp/rev56/`（`tools/` + 一份基线合法 YAML）逐项注入，观察 `check_tree.py` 退出码：

| 注入 | 期望 | 实测 | 结论 |
|---|---|---|---|
| 缺 `role_map` | 拒 | `FAIL：2 项 … 缺字段 role_map / role_map 必须是 map` exit 1 | ✅ |
| 缺 `milestone` | 拒 | `缺字段 milestone / milestone 必须是 map … 实际 NoneType` exit 1 | ✅ |
| 两个 milestone（list） | 拒 | `milestone 必须是 map … 实际 list` exit 1 | ✅ |
| 两个 milestone（重复键） | 拒 | `YAML 解析失败：重复键 'milestone'` exit 1 | ✅ |
| milestone 缺 `output` | 拒 | `milestone 缺 output` exit 1 | ✅ |
| `cost_model` 写墙钟/日期 | 拒 | `cost_model 含墙钟/日期指标 '墙钟'` exit 1 | ✅ |
| `cost_model` 写 `tok/s` | 拒 | `cost_model 含墙钟/日期指标 'tok/s'` exit 1 | ✅ |
| `status=已完成` 无 `conclusion` | 拒 | `status=已完成 但无 conclusion` exit 1 | ✅ |
| `status=已完成` 无 `experiments` | 拒 | `status=已完成 但无 experiments（结论必须有证据）` exit 1 | ✅ |
| `status=被否证` 无 `conclusion`/`experiments` | 拒 | 两条都报 exit 1 | ✅ |
| `experiments` 指向不存在路径 | 拒 | `experiments 路径不存在：…/NOPE` exit 1 | ✅ |
| **WP `evidence` 指向不存在路径** | 拒 | **`OK …` exit 0（不拒）** | ❌ |
| `validation` 缺「对照」 | 拒 | `validation 缺「与 SOTA/stock 基线对照」` exit 1 | ✅ |
| `validation` 缺「口径」 | 拒 | `validation 缺口径` exit 1 | ✅ |
| **`validation` 有对照+口径、无「判据」** | 拒 | **`OK` exit 0（不拒）** | ❌ |
| `role_map` 缺 `integrator` | 拒 | `role_map 缺角色 integrator` exit 1 | ✅ |
| `work_packages[0]` 缺 `means` | 拒 | `work_packages[0] 缺 means` exit 1 | ✅ |
| `status` 非法值 | 拒 | `非法 status='完成'` exit 1 | ✅ |
| `id` 跨文件重复 | 拒 | `id='1.1.1' 与 … 重复` exit 1 | ✅ |
| `dir` 与实际目录不符 | 拒 | `dir=… 与实际目录 … 不一致` exit 1 | ✅ |
| `falsification: ""` | — | `OK` exit 0（空串放行） | ⚠️ |
| `objective_criteria: ""` / `expected_effect: ""` / `change_log: []` | — | `OK` exit 0 | ⚠️ |

**未被满足的要点**：

1. **缺 `failure_analysis` / 失败归因要素（阻塞）**。全仓 `grep -rn "失败归因\|failure_analysis"`
   （`docs/research-schema.md`、`docs/structure.md`、`SCHEMA.md`、`AGENTS.md`）**零命中**；
   `check_tree.py` 的 `REQUIRED` 里也没有。而 `.scratch/56-restructure/TICKET.md` §5.1 逐字记录的用户 L1 是：
   「如果做不出来……到底是哪里出问题？……是设计的问题？还是这个结构的问题？还是你的 Innovation 本身就是失败的？
   它的代价会是些啥？」——冻结后的 schema **没有任何字段承接这条要求**（`falsification` 是**预先登记**的否证条件，
   不是**事后**归因；`gaps` 是「未测项」，`conclusion_review` 是「结论对不对」）。M2 各大类照 schema 写 YAML 必整批漏掉它。
2. `validation` 三要素只强制了 2 条：`check_tree.py` 只查「对照关键词」与「口径关键词」，**不查「判据」**；
   `objective_criteria` 仅要求**键存在**、不查内容（空串放行）。schema §2 声称 `validation`「必须同时含 ①对照 ②客观判据 ③口径」，
   实现只落 ①③。
3. **WP `evidence` 路径存在性未强制（非阻塞，但同时是文档/实现冲突）**：`docs/research-schema.md` §9 写
   「`experiments` / `evidence` 里的路径必须存在」，而 `check_tree.py` 的注释明说「WP 的 `evidence` 是『证据将落在哪』，
   可能是**未来路径** ⇒ 只收集锚点，不强制路径已存在」。两者对**同一事实**说法相反。
4. 多数「必填」字符串字段是**存在性**校验，空串可过（`falsification: ""` 也 OK），与 §2「必填 = 校验器会 fail」读起来偏强。

> 说明：`role_map.reviewer != writer`（§4 明文纪律）未被机器强制——这属人类纪律，无法可靠机检，**不计缺陷**。

### 判据 3 · MAPPING 无证据丢失 — ✅

- `MAPPING.md` §1 数据行 **37 行**，与磁盘上 37 个旧格目录（`axes/<旧轴>/<旧格>`，减去本轮新增的
  `1-time/1.1.1-hard-window`、`9-backlog/9.1-selector-scheduling`）**双射**：脚本比对「旧格目录 → MAPPING 旧路径」
  与「MAPPING 旧路径 → 磁盘」两侧，均为 **0 missing / 0 extra**。
- 处置列齐全：`迁移`×7、`迁移（核心）`×3、`Backlog`×6、`工程纪律`×5、`降参照`×4、`并（手段…` 等共 12 种；
  **拆**仅 A3、A6 两行。判据要求的 A9/A10/M1/M2/C2/C5/C6/H2–H5/W2/W3/W4/F2/F4/G1/R1/R4/R7/P1/P3/P4/P5/P6
  全部出现。
- **拆并一格一主家**：§3 明确 A3→`1.1.1`（主家，全部 experiments 落此；`1.1.3` 只指针不接文件）、
  A6→`1.2.2`/`1.2.3`/`1.1.1`（`2614-topk`→1.2.3，`2616`/`2637`→1.2.2，长T窗口→1.1.1），
  并显式声明「不复制任何 A6 文件」；H1/H3/H4/H5 的 experiments 全落 `6.2/experiments`，作为 `6.2` 的 WP；
  M1 的 22 个实验目录落 `1.1.4.1`。
- **2434 口径可核**：现行 `find axes -type f | wc -l` = **2436**；`ab1d8d8` 在 `axes/` 下**只新增 2 个文件**
  （`git diff --name-status ce2610a..dff555f -- axes/` = 仅 2 个 `A`）⇒ 迁移前工作树 = 2436−2 = **2434**，与
  `MAPPING.md` §0 声明一致。
- **细枝（非阻塞）**：2434 是**工作树计数**，含 **777 个 gitignore 文件**（`runs/*results.json`、`*.pt`、`__pycache__`；
  跟踪文件在 `ce2610a` 仅 1657）。`57-program/TICKET.md` 的 M2 验收「`find axes -type f｜wc -l ≥ 2434`」因此
  依赖同一工作树/缓存状态，不是「跟踪文件数」口径；清一次 cache 就会漂移。建议 MAPPING/57 注明该口径是工作树计数。

### 判据 4 · 工具可运行 — ✅

| 命令 | 退出码 | 输出要点 |
|---|---|---|
| `python3 tools/summary.py` | 0 | 打印 11 列全表（id/轴/起/止/完成?/状态/结论/结论对不对/实验/论证/遗漏），2 行数据 + 「共 2 个研究项 ｜ ✅ 已完成 0 ｜ ❌ 被否证 0」 |
| `python3 tools/summary.py --check` | 0 | `OK：2 个 project.yaml 全部符合 docs/research-schema.md。` |
| `python3 tools/check_tree.py` | 0 | 同上 |
| `python3 tools/check_schema.py` | 0 | 同上（转调 `check_tree.main()`，实测同一输出） |
| `python3 -m pytest tests/ -q` | 0 | `827 passed in 89.49s`（与提交信息「827 passed」一致） |

- `tests/test_check_schema_anchor.py` 实测钉住了锚点工具转调（`check_schema.check_anchors is check_tree.check_anchors`）、
  假 sha 必逐条报错、git 不可用显式报错、非十六进制锚点（`@FINAL`/`@HEAD~1`/裸 `@`）必报错——这些用例真实存在且通过。

### 判据 5 · 文档无矛盾 — ❌（非阻塞）

- **一致的部分**：`AGENTS.md`（§2.1 文档地图改用 `structure.md`/`research-schema.md`/`project.yaml`；§2.3 标注
  `catalog/INDEX/build_wiki/check_catalog_strict/backfill_legacy` 退役；§8 命令换 `summary.py`/`check_tree.py`）、
  `README.md`（是什么/怎么读/轴记号）、`CONTRIBUTING.md`（流程与硬性规则）、`docs/design.md`（目录树）、
  `docs/methodology.md`、`docs/playbook.md` 均已指向新树与新工具，本仓 grep 未再发现被当作**现行命令**的旧流程
  （`CONTRIBUTING.md` 已删掉 `build_wiki.py`/`backfill_legacy.py` 步骤）。
- **仍存在的事实文档自相矛盾**（`SCHEMA.md`）：
  - **开篇（第 3–4 行）**仍写「它成立的前提是格式可机检：`tools/build_wiki.py` 由 `catalog.yaml` 生成所有索引与课题页，
    `tools/check_schema.py` 在 CI 里让格式错误直接失败」——与**紧接其下第 7–16 行的迁移通告**（声明 build_wiki/catalog 已退役、
    `check_schema.py` 已转调）**直接冲突**；通告只回收了 §1–§3/§0/§5，未回收这句开篇。
  - §3 第 69 行 `重跑 build_wiki.py 不丢手写内容`、§4 第 96 行、§12 第 197 行、§17.13 第 298–300 行、
    §21 第 455 行、§22 第 538–539 行与第 580–592 行（`三类表面（实验文档 / catalog.yaml verdict / 生成页）必须逐字一致 …
    tools/check_catalog_strict.py 会机检 … 改数字后必须 python3 tools/build_wiki.py 重生成页面`）仍把已退役的
    `catalog.yaml`/`build_wiki.py`/`check_catalog_strict.py` 写成**现行机制**。通告称「§6–§24 仍然全部生效」，
    于是这些条款与 `structure.md` §15 / `AGENTS.md` §2.3 的退役裁决**在同一事实上两处相反**。
- 细枝（非阻塞）：`CONTRIBUTING.md`「硬性规则（CI 会拦）」列表编号为 `1. 2. 4. 5. 6. 7.`（删掉原第 3 条后未重排）。

### 判据 6 · 工单与提交卫生 — ✅（含 1 处表述不精确）

- `.scratch/56-restructure/TICKET.md` §5 齐全：§5.1 出资人 L1 逐字；§5.2 九条决策与理由；§5.3 新增/修改清单；
  §5.4 命令与结果；§5.5 剩余未决。
- **时间戳**：TICKET §5 与 `FOUNDATION-REPORT.md` = `2026-10-13T09:48:38+08:00`，提交时间 09:49:21，exemplar
  `started` = 09:40:25，全为 `+08:00` ISO8601（与 `date -Iseconds` 同格式）。无法回溯验证当时真实时钟，但格式与先后自洽。
- **staged 路径逐条对齐**：`git show --name-only ab1d8d8` 与 `git show --stat ab1d8d8` 的路径集合**完全一致**（22 个文件，`diff` 无差异）。
- **未误用 `-A`/`-u`**：提交时刻工作树已有 4 个未提交改动（`INDEX.md`、`catalog.yaml`、`axes/C-vertical/README.md`、
  `axes/C-vertical/C1-layers-per-token/README.md`），而它们**未被纳入** `ab1d8d8`（`MAPPING.md` §6 明说要保留、不回滚、
  不提交）——这正是「显式路径 add」的有力证据；提交内 22 个文件全部落在声明范围内，无越界文件。
- **表述不精确（非阻塞）**：`ab1d8d8` 提交信息写「`docs/structure.md` **追加** §12 节点清单 …… 原 §1–§11 用词未改，仅加指针行」，
  `TICKET.md` §5.3 也把 `docs/structure.md` 列在「**修改**」栏；但 git 记录该文件在 `ab1d8d8` 是**新增**
  （`git cat-file -e ce2610a:docs/structure.md` → 不存在；`git log --all --diff-filter=A` / `--follow` 均只有 `ab1d8d8`；
  全仓无其它 ref/其它文档含 §1 独有语句「一次前向里，一个 token」）。若该文件此前是**未跟踪**文件，「未改」尚可自洽，
  但**从 git 无法核实**，且「追加」对一个 git 新增文件是误导性描述。
- 变更后未做替代：`FOUNDATION-REPORT.md` 初版未写死哈希，已由更正提交 `dff555f` 补上（`git show --stat dff555f` = 1 file, +2/−1），
  符合「发现问题立刻补更正提交（不 amend）」的纪律。

---

## 2. Material defects

### 阻塞（blocking）

**B1 · 冻结 schema 缺「失败归因」要素，校验器也不查**
- 位置：`docs/research-schema.md` §2 字段表（第 21–48 行）与 §6 附近；`tools/check_tree.py:REQUIRED`（第 38–43 行）。
- 违反：判据 2（「…`falsification`、`failure_analysis`/失败归因要素」）与 `TICKET.md` §5.1 的 L1
  （「做不出来时是设计问题 / 结构问题 / innovation 失败」）。
- 复现：`grep -rn "失败归因\|failure_analysis" docs/research-schema.md docs/structure.md SCHEMA.md AGENTS.md` → 无命中；
  在 `/tmp` 副本里即便删除任何「归因类」内容，`check_tree.py` 也不会因缺失败归因而报错（该字段不在 `REQUIRED`）。
- 影响：M2 各大类照 schema 建 YAML 会整批漏掉用户明确要求的「失败后归因设计/结构/innovation」。

**建议修法（给 writer）**：在 `docs/research-schema.md` §2 增加字段 `failure_analysis`（失败归因：设计问题 / 结构问题 /
innovation 失败，三选一或并列 + 证据），在 §6/§7 说明其触发时机（`status ∈ {被否证}` 或 `conclusion` 判失败时必填）；
在 `tools/check_tree.py` 的 `REQUIRED` 加入该键，并加一条「`status=被否证` ⇒ `failure_analysis` 非空」的门；
同步给两份 exemplar（`1.1.1`、`9.1`）补该键；更新 `docs/research-schema.md` §9 的校验项清单。

### 非阻塞（non-blocking）

**N1 · 文档/实现对「evidence 路径必须存在」说法相反**
- `docs/research-schema.md` §9「证据锚点：`experiments` / `evidence` 里的路径必须存在」；
  但 `check_tree.py` 对 WP `evidence` 只收锚点、不查路径存在（WP evidence 为「未来路径」）。
- 复现：/tmp 注入 WP `evidence: axes/1-time/1.1.1-hard-window/experiments/NOPE` → `OK` exit 0。
- 建议：二选一——(a) 把 §9 措辞收窄为「`experiments` 路径必须存在；WP `evidence` 允许未来路径」，
  或 (b) 对非 `未开始` 的 WP 也强制 `evidence` 路径存在。

**N2 · `SCHEMA.md` 开篇与 §3/§4/§12/§17.13/§21/§22 仍把退役工具当现行机制**
- 位置：`SCHEMA.md:3–4`、`:69`、`:96`、`:197`、`:298–300`、`:455`、`:538–539`、`:580–592`。
- 违反：判据 5（「退役条款清晰，无事实文档两处冲突」）。
- 复现：`grep -n "build_wiki\|check_catalog_strict\|catalog.yaml" SCHEMA.md`；对照 `docs/structure.md` §15、
  `AGENTS.md` §2.3（声明其已退役）。开篇句与其下迁移通告**同一文件内自相矛盾**。
- 建议：删/改开篇那句；在 §3/§4/§12/§17/§21/§22 各自加一行「（迁移窗口内适用旧页；工具已退役，见顶部通告 / §15）」的指针。

**N3 · `validation`「三要素」只强制 2 条**
- `check_tree.py` 只查「对照」与「口径」关键词，`判据` 不查；`objective_criteria` 仅查键存在、内容（空串）放行。
- 复现：`validation: "对照 = stock 基线；口径 = matched-step bpc"` → `OK` exit 0。
- 建议：补一条对 `objective_criteria`（或 `validation` 里的「判据」）的非空/关键词门，或把 schema §2 的措辞与实现对齐。

**N4 · 多数「必填」字段是存在性校验，空串可过**
- `falsification: ""`、`objective_criteria: ""`、`expected_effect: ""`、`change_log: []` 均 `OK`。
- 与 `docs/research-schema.md` §2「必填 = 校验器会 fail」读感偏强。建议对关键字段（`purpose`/`innovation`/
  `validation`/`objective_criteria`/`falsification`）改用 `_nonempty`。

**N5 · `CONTRIBUTING.md` 规则编号跳号**：`1. 2. 4. 5. 6. 7.`（缺 3）。

**N6 · 提交信息把「新增」写成「追加」**：`ab1d8d8` 提交信息与 `TICKET.md` §5.3 将 `docs/structure.md` 描述为「追加 §12–§15 /
原 §1–§11 未改」，git 记录为**新增文件**，「未改」不可核（见判据 6）。建议后续工单如实记为「新建（含既有 §1–§11 内容）」。

**N7 · 2434 是工作树计数（含 777 个 gitignore 文件），非跟踪文件计数**（见判据 3）。建议 MAPPING §0 / 57 M2 验收注明口径。

**N8 · §9「调度/选择器」vs §12「调度 / 选择器」空格不一致**（纯排版）。

---

## 3. 未复核到的范围（如实列出）

- **用户 L1 逐字的真实性**：无法回溯验证 `TICKET.md` §5.1 引文与用户原始输入逐字相同；只能确认其被标为 L1 并完整存在。
- **当时的真实时钟**：仅验证时间戳格式为 `date -Iseconds` 兼容（`+08:00` ISO8601）且先后自洽，未（也无法）独立核验
  09:40–09:49 这一真实时刻。
- **`docs/structure.md` §1–§11 是否为「未改动的既有内容」**：git 无该文件的任何先前版本（ce2610a 不存在、其它 ref 与
  `--follow` 均无、无其它文档含其独有语句），故「§1–§11 未改」**不可独立核验**（见 N6）。
- **M2 的实际迁移**（逐轴建 37 份 `project.yaml`、搬 `axes/**` 证据、清残留）：不在本 WP 范围，未复核；`MAPPING.md` 只冻结了映射。
- **`/tmp` 注入为最小复现**，非对全仓 37 格未来 YAML 的穷举；校验器在真实大样本下的行为未测（当前仓内只有 2 份 YAML）。
- **`check_catalog_strict.py` / `build_wiki.py` 本身是否仍可运行**：本 WP 只加退役 docstring，未复核其功能；因已退役，不属验收目标。

---

## 4. 需要哪个 writer 修什么（阻塞项）

**给 WP-M1.0 的 writer（或 M2 派发前的修订 worker），修 B1（阻塞）后本 WP 才可判 PASS：**
1. `docs/research-schema.md`：新增 `failure_analysis` 字段（失败归因：设计 / 结构 / innovation）并入 §2 必填表，
   §5/§6/§7 说明何时必填（判失败/被否证时）。
2. `tools/check_tree.py`：把 `failure_analysis` 加入 `REQUIRED`，并加门「`status=被否证` ⇒ `failure_analysis` 非空」；
   同步更新 §9 校验项清单。
3. 两份 exemplar（`axes/1-time/1.1.1-hard-window/project.yaml`、`axes/9-backlog/9.1-selector-scheduling/project.yaml`）补该键。
4. （可选、同批）顺带修 N1/N2/N3 与 N5，使文档与实现齐平。

修完须重跑 `python3 tools/summary.py --check && python3 tools/check_tree.py && python3 -m pytest tests/ -q`，并由**新的 fresh 复核**验收。
