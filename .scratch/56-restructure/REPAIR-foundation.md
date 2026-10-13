# ticket 56 · WP-M1.0 规范冻结 · 独立复核 FAIL → 修复报告

> writer：fresh 上下文、单一写入者。修复基线 = 复核提交 `ab1d8d8`（+ 更正 `dff555f`）。
> 复核依据 = `.scratch/56-restructure/REVIEW-foundation.md`（已**原样复制入库**，可审计）。
> 修复时间锚：2026-10-13T10:00:28+08:00（`date -Iseconds`）。

## 0. 人话摘要

上一轮 foundation 被独立复核判 **FAIL**：主干四件（研究树规范 `docs/structure.md`、新 schema
`docs/research-schema.md`、校验/汇总工具 `check_tree.py`/`summary.py`、旧→新映射 `MAPPING.md`）
都在、命令真能跑，**但冻结的 schema 漏掉了出资人明确要求的一条**——「做不出来时要区分是
**设计问题 / 结构问题 / innovation 本身失败**」；既没有字段承载、校验器也不查，M2 各大类照 schema
建 YAML 时会把这条要求**整批漏掉**。

本轮把这条补成 **必填字段 `failure_analysis`**，并加了一道**硬门**：只要 `status=被否证` 而
`failure_analysis` 为空，校验器直接 fail（已在 `/tmp` 副本注入复现，见 §3）。同时把复核确认的
8 处非阻塞项（文档/实现口径相反、退役工具仍被当现行、规则编号跳号、计数口径、空格不一致、
提交信息措辞）一并修齐。改完 `summary.py --check`、`check_tree.py`、`pytest` **全绿**（827 passed）。

## 1. 逐项修复对照

| 项 | 严重度 | 改了什么 | 落点 | 自证 |
|---|---|---|---|---|
| **B1** | 阻塞 | 新增必填字段 `failure_analysis`（失败归因：设计问题 / 结构问题 / innovation 本身失败，可并列 + 证据）；§2 字段表加行、§5/§6/§7 说明触发、§9 校验项同步；校验器 `REQUIRED` 加键 + 「`status=被否证` ⇒ 非空」硬门；两份 exemplar 补键 | `docs/research-schema.md` §2/§5/§6/§7/§9、`tools/check_tree.py`、`axes/1-time/1.1.1-hard-window/project.yaml`、`axes/9-backlog/9.1-selector-scheduling/project.yaml` | `/tmp` 注入 `status=被否证`+空 `failure_analysis` ⇒ `FAIL`；补非空 ⇒ `OK` |
| **N1** | 非阻塞 | **选方案 (a)**：`docs/research-schema.md` §9 收窄为「`experiments` 路径必须存在；`work_packages[].evidence` 允许未来路径（只收锚点）」——与校验器现状一致 | `docs/research-schema.md` §9 | 文档与 `check_tree.py` 注释/行为一致 |
| **N2** | 非阻塞 | `SCHEMA.md` 开篇(3–5 行)不再把 `build_wiki.py`/`catalog.yaml` 当现行；§0/§3(69)/§4(96)/§12(197)/§17.13(298–300)/§21(455)/§22(538–539、580–592) 逐处加「已退役，见顶部迁移通告 / `docs/structure.md` §15」指针；**未删任何仍生效的分析纪律** | `SCHEMA.md` | `grep -n "build_wiki\|catalog.yaml\|check_catalog_strict" SCHEMA.md` 每处均带退役指针或无「现行」断言 |
| **N3** | 非阻塞 | `validation` 三要素补齐②判据：新增「判据/该涨/该跌/必须不变/criteria/objective」关键词门 | `tools/check_tree.py` | 注入 `validation` 只含对照+口径 ⇒ `FAIL` |
| **N4** | 非阻塞 | 关键字段改非空校验：`purpose`/`innovation`/`validation`/`objective_criteria`/`falsification` 不允许空串/空列表 | `tools/check_tree.py` | 注入 `purpose: ""` ⇒ `FAIL` |
| **N5** | 非阻塞 | `CONTRIBUTING.md`「硬性规则」编号 `1.2.4.5.6.7.` → 重排连续 `1.2.3.4.5.6.` | `CONTRIBUTING.md` | 目视：1–6 连续 |
| **N6** | 非阻塞 | **不改写历史**；工单 §5.3 追加更正说明（`docs/structure.md` 本轮为**新建**文件，含既有 §1–§11 内容），更正提交信息如实说明「上一条信息超出实际改动」 | `.scratch/56-restructure/TICKET.md` §5.3、更正提交 | `git show <更正提交> --stat` |
| **N7** | 非阻塞 | 注明 **2434 是工作树文件计数**（含约 777 个 gitignore 产物），非跟踪文件计数；清 cache 会漂移，须并记 `git ls-files axes` | `.scratch/56-restructure/MAPPING.md` §0、`.scratch/57-program/TICKET.md` M2 行 | 目视 |
| **N8** | 非阻塞 | `docs/structure.md` §9「调度/选择器」→「调度 / 选择器」，与 §12 及 exemplar 标题一致 | `docs/structure.md` §9 | §9 与 §12 逐字一致 |

### 关于 N1 的方案选择与理由

**选 (a)（收窄文档措辞，而非加严校验器）**：
`work_packages[].evidence` 的语义是「**证据将落在哪**」，在预登记阶段天生是**未来路径**——M2 各轴 worker
建 YAML 时 WP 尚未执行，evidence 目录还不存在。若选 (b)（对非 `未开始` 的 WP 强制 evidence 路径存在），
会与「预登记契约应先于证据」的既有纪律冲突，并让合法的前瞻性登记无法入库。
故把**同一事实**统一到实现现口径，风险最低、且不削弱 `experiments` 的硬路径存在性检查。

## 2. 变更文件清单

- 修改：`docs/research-schema.md`、`tools/check_tree.py`、`SCHEMA.md`、`CONTRIBUTING.md`、
  `docs/structure.md`、`.scratch/56-restructure/MAPPING.md`、`.scratch/57-program/TICKET.md`、
  `.scratch/56-restructure/TICKET.md`、`axes/1-time/1.1.1-hard-window/project.yaml`、
  `axes/9-backlog/9.1-selector-scheduling/project.yaml`。
- 新增：`.scratch/56-restructure/REVIEW-foundation.md`（复核报告原样入库）、
  `.scratch/56-restructure/REPAIR-foundation.md`（本文件）。
- **保留未提交、未纳入提交**的工作树既有改动：`INDEX.md`、`catalog.yaml`、
  `axes/C-vertical/README.md`、`axes/C-vertical/C1-layers-per-token/README.md`（见 `MAPPING.md` §6）。

## 3. 自证命令与结果

| 命令 | 结果 |
|---|---|
| `python3 tools/check_tree.py` | `OK：2 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| `python3 tools/summary.py --check` | 同上（exit 0） |
| `python3 tools/check_schema.py` | 转调，同上（exit 0） |
| `python3 -m pytest tests/ -q` | **827 passed** in 99.79s（0 失败） |

### `/tmp` 注入对照（证明新门能拒）

在 `/tmp/repair-foundation-inject/` 放一份最小合法 `project.yaml` + `tools/check_tree.py`：

1. `status=被否证` + `failure_analysis: ""` ⇒ `FAIL：1 项 … status=被否证 但 failure_analysis 为空…`（exit 1）。✅ 新门生效。
2. `failure_analysis` 非空 ⇒ `OK`（exit 0）。
3. 删掉 `failure_analysis` 键 ⇒ `FAIL：缺字段 failure_analysis` + 被否证门（exit 1）。
4. `purpose: ""` ⇒ `FAIL：purpose 不得为空（必填字段）`（exit 1）——N4 生效。
5. `validation` 只含对照 + 口径、无判据 ⇒ `FAIL：validation 缺「客观判据」…`（exit 1）——N3 生效。

## 4. 未决 / 边界

- **需新的 fresh 复核**：本轮修复后，本 WP 的 PASS 判定须由**新的独立复核**给出（复核者不得为本 writer）。
- 复核报告 §3 已列「未复核范围」（L1 逐字真实性、真实时钟、`docs/structure.md` §1–§11 是否未改）——
  本轮**不新增**对这些不可核事实的主张；N6 反而把其中一条（「追加」vs「新建」）如实记为不可核。
- `check_catalog_strict.py` / `build_wiki.py` 是否仍可运行未复核（已退役，不属验收目标）。
