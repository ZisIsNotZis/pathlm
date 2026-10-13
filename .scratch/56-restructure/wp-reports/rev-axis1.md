# 轴 1（时间轴）迁移复核 · 独立复核报告

> 复核者：fresh 上下文，只读（仓库零改动；仅 /tmp 注入测试与临时报告落盘）。
> 复核范围：轴 1 = `axes/1-time/**`，对应 **WP-M2.1（§1.1，8 研究项 + 证据迁移）** 与
> **WP-M2.2（§1.2 四研究项 + §1.3 参照登记 + A6 拆分迁移）**。
> 假设：**以「迁移做砸了」为前提去证伪**；只报告，不修复。
> 契约：`docs/structure.md` §12–§15、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`、
> `.scratch/55-review/*`。

---

## 0. 复核对象（revision 快照）

- 复核时间：2026-10-13（本机时钟）。
- `git rev-parse HEAD` = **bacaaeba6464f75b24fd28e30bca621e18527462**（branch `main`）。
- `git status --short`：

  ```
   M INDEX.md
   M axes/C-vertical/C1-layers-per-token/README.md
   M axes/C-vertical/README.md
   M catalog.yaml
  ?? .scratch/54-subproject-schema/
  ?? .scratch/55-review/
  ```

  前 4 个 `M` 与 MAPPING §6「工作树既有未提交改动」一致（C1 判据修订 + 退役中的
  `catalog.yaml`/`INDEX.md`），**不是本次轴 1 WP 的产物**；两个 `??` 是 55-review 台账。
  轴 1 的两个 WP 已落为提交 `18a3e6b`（§1.1）与 `65adaec`（§1.2+§1.3），均在本 HEAD 历史内。
- 工作树内容 = HEAD 内容 + 上述 4 个 `M`（与轴 1 无关）。

---

## 1. 总判定

**结论：PASS（轴 1 两个 WP 达到迁移契约；无 blocking defect）。**

12 个研究项 `project.yaml` 全部字段齐全、用词逐字对齐 `docs/structure.md` §12、证据按 MAPPING
迁入且经内容级核验无测量数字丢失；三项校验命令全绿；四类坏 YAML 注入全部被校验器正确拒绝。
剩余问题均为 **non-blocking**（日志目录 `*.md.md` 命名残留、role_map 实名待指派、个别 55-review
条目的显式引用缺位、§12 与 §13.4 关于 `1.3` 目录的规范内歧义）。

---

## 2. 逐条判据（✅/❌ + 复现）

### 判据 A · 节点齐全且 exact wording ✅

复现：

```bash
python3 - <<'PY'   # 解析 docs/structure.md §12 轴 1 表 vs project.yaml
import re,yaml,glob
sec=open("docs/structure.md").read().split("### 轴 1 · 时间轴")[1].split("### 轴 2")[0]
rows=re.findall(r'^\| (1[\d.]*) \| (.+?) \| `([^`]+)` \| (\S+) \|$', sec, re.M)
yt={yaml.safe_load(open(f))["id"]: yaml.safe_load(open(f))["title"]
    for f in glob.glob("axes/1-time/**/project.yaml",recursive=True)}
for i,t,d,g in rows:
    if g in ("研究项","Backlog"): print(i, "OK" if yt.get(i)==t else "MISMATCH", repr(t), repr(yt.get(i)))
PY
```

结果：§12 轴 1 共 **16 个节点**（3 分组 + 12 研究项 + 1 参照）。

- 分组 `1.1` / `1.1.4` / `1.2` → 各有聚合 `README.md`（`axes/1-time/1.1-visibility/`、
  `1.1.4-right-shift/`、`1.2-visibility-structure/`）✅
- 12 个研究项 `1.1.1/1.1.2/1.1.3/1.1.4.1/1.1.4.2/1.1.4.3/1.1.4.4/1.1.4.5/1.2.1/1.2.2/1.2.3/1.2.4`
  → 12 份 `project.yaml`，**标题逐字等于 §12**（含 `implicit…vs **explicit**…` 的加粗与
  全角括号、`delta 规则族：DeltaNet / GLA / Mamba-2` 等），`axis` 均为 `"1 时间轴"` ✅
- 参照 `1.3 寻址` → 登记在 `axes/7-references/README.md`（注明「本轴不写 project.yaml、不建目录」），
  未新建 `axes/1-time/1.3-addressing/`。与 §13.4（参照进 `7-references`）一致；与 §12 表中给
  `1.3` 列出的 `目录` 单元格存在**规范内歧义**（见 §4 N4），但取 §13.4 的读法合理。

### 判据 B · 每个 project.yaml 字段齐全 ✅

复现：`python3 tools/check_tree.py`（下 §2 判据 E）对 46 份 YAML 全绿；另以脚本逐项核对 12 份：

- 必填键（含 `expected_effect/sota_baseline/gap/method/validation/objective_criteria/falsification/
  cost_model/role_map/milestone/work_packages/roadmap/failure_analysis/conclusion_review/experiments/
  argument/gaps/started/ended/change_log`）**无缺**；
- 关键字段 `purpose/innovation/validation/objective_criteria/falsification` **无空**；
- `role_map` 四角色齐（owner/writer/reviewer/integrator），且 **reviewer ≠ writer**（schema §4）；
- `milestone` 是 map 且**仅一个**、含 `output`+`acceptance`；
- 每个 `work_packages[*]` 含 id/what/goal/means/**acceptance**/**review**/evidence/status 八键，
  `review` 均为「fresh 复核（SCHEMA.md §14/§14b）」，即**独立复核**（非 writer 自评）✅
- `status` 只出现 `未开始`(4)/`进行中`(8)，均在合法枚举内；无 `已完成/被否证` 悬空。
- **`cost_model` 无墙钟/日期**：12 份全部通过 `WALLCLOCK_RE`；`grep -nE '墙钟|分钟|小时|tok/s'`
  在 `cost_model` 命中 **0**（墙钟字样只出现在**迁移来的实验证据** `experiments/**` 内，且这些
  文件都明确写「公式值不得当判据/分别报理论 FLOPs 与实测墙钟」，符合 schema §3 允许的「过程指标」）。

### 判据 C · 证据按 MAPPING 迁移、无丢失 ✅

复现（内容级，非只看计数）：

```bash
git ls-tree -r ce2610a --name-only | grep '^axes/' | wc -l     # 迁前 git 跟踪数 = 1657
git ls-files axes | wc -l                                       # 现跟踪数       = 1733
find axes -type f | wc -l                                       # 现工作树文件数 = 2510
git diff -M --raw ce2610a HEAD                                  # 重命名探测
```

- **工作树计数**：M1 基线 2434 → 现 **2510**（+76）；**git 跟踪数** 1657 → **1733**（+76）。
  工作树 - 跟踪 = 777，正等于 MAPPING §0.1 声明的「约 777 个 gitignore 产物」，口径自洽。
- **重命名探测**（`git diff -M ce2610a HEAD` 全仓）：`R`=1567（内容全同），`D`=5，`A`=91，
  `M`=48。**唯一 5 个真删除** = 旧格自动生成的实验索引
  `A10/A3/A4/A9/M2 的 experiments/README.md`（`<!-- AUTO -->` 旧 wiki 索引）；其对应新家
  （`1.1.4.3/1.1.1/1.1.2/1.1.4.5/1.1.4.2`）各**新增**了一份 `experiments/README.md`，
  旧索引内容被新索引取代，非研究证据 → **不计证据丢失**（见 §4 N5）。
- **迁入轴 1 的重命名计数**（dest 前缀 `axes/1-time/`）：
  `1.1.1`=226（A3:1 + A6 长T系列:225）、`1.1.2`=1（A4）、`1.1.4.1`=694、`1.1.4.2`=26（M2）、
  `1.1.4.3`=15（A10）、`1.1.4.5`=62（A9）、`1.2.2`=53（A6）、`1.2.3`=31（A6）。
  其中少量跨源条目（如 `C3/F4/M1 → 1.1.4.1`、`M1 → 1.1.4.5/1.2.2/1.2.3`）经查是**内容完全相同的
  小 `gate_log.jsonl`** 被重命名探测误配，非真实跨轴搬运 → 各新家实际来源与 MAPPING §1/§3 一致。
- **内容级无丢失**：对 182 个「重命名 + 内容有变」的文件逐个做「去路径归一」后比对，只有 1 个
  残差：`M1-carrier-type/experiments/README.md`（索引，38→34 行）。其余 181 个的变化**全部只是
  路径指针改写**（`axes/A-mixing/A6-mixing-operator/…` → `axes/1-time/1.1.1-hard-window/…` 等，
  含 `os.path.join(ROOT,"axes","…")` 形式的分解字符串），**测量数字零改动**。抽样 diff 复现：

  ```bash
  diff <(git show ce2610a:axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512/results.json) \
       axes/1-time/1.1.1-hard-window/experiments/2622-longT-512/results.json
  # 仅 "contract" 路径行变化，数值不变
  ```
- 旧格目录仍保留（`A3/A4/A6/A9/A10/M1/M2` 各只剩 `README.md` 和空 `experiments/`），
  `.scratch/**/evidence/**` 原地未动，`archive/` 未动 → 符合 MAPPING §0.4/§0.5。

### 判据 D · 55-review 相关问题逐条处置（见 §3 表）✅（少量 non-blocking 缺位）

轴 1 相关的 A 轴（P1–P23 / R1–R6）与 M 轴（M-01–M-25）：**根因 R1–R8 与绝大多数条目已处置**，
且以显式 `55-review <id>` 引用落进 YAML（1.1.4.1 引 12 个 M 条目；1.1.1/1.1.2/… 引 A 轴 P 条目）。
未显式引用/未清理的条目见 §4（均 non-blocking）。

### 判据 E · 三项命令真实输出 ✅

```bash
python3 tools/summary.py --check   # OK：46 个 project.yaml 全部符合 docs/research-schema.md。 exit=0
python3 tools/check_tree.py        # OK：46 个 project.yaml 全部符合 docs/research-schema.md。 exit=0
python3 tools/check_schema.py      # OK：46 个 project.yaml 全部符合 docs/research-schema.md。 exit=0（转调）
python3 tools/summary.py           # 打印 46 行全表（含 12 行轴 1），exit=0
python3 -m pytest tests/ -q        # 827 passed in 94.88s (0:01:34) exit=0
```

注：pytest 首轮在**同机 4 份并发 pytest 争用**下极慢、被中途终止（未出汇总）；待机器空闲后
以 `timeout 600 python3 -m pytest tests/ -q` 单跑，得 **827 passed in 94.88s**（真实结果，
`/tmp/pytest-axis1-final.log`）。

### 判据 F · /tmp 注入坏 YAML 证伪校验器 ✅

方法：把 `tools/check_tree.py`/`summary.py` 复制到 `/tmp/inject-axis1/tools/`，以合法
`1.1.3`（无 `@sha` 锚点）为模板注入四类坏 YAML，运行 `/tmp/inject-axis1/tools/check_tree.py`。

| 注入 | 校验器输出 | 判定 |
|---|---|---|
| 对照（合法模板） | `OK：1 个 project.yaml 全部符合…` exit 0 | ✅ |
| 缺 `role_map` | `FAIL：2 项 - 缺字段 role_map / role_map 必须是 map` exit 1 | ✅ |
| 缺 `failure_analysis` | `FAIL：1 项 - 缺字段 failure_analysis` exit 1 | ✅ |
| 墙钟 `cost_model`（"快 3×，墙钟 12 秒/step，5000 tok/s"） | `FAIL：1 项 - cost_model 含墙钟/日期指标 '墙钟'` exit 1 | ✅ |
| `status=被否证` 且 `failure_analysis=""`（conclusion/experiments 非空） | `FAIL：1 项 - status=被否证 但 failure_analysis 为空（必须区分 设计问题 / 结构问题 / innovation 本身失败…）` exit 1 | ✅ |
| 恢复对照 | `OK` exit 0 | ✅ |

四类坏 YAML **全部被拒**，校验门真实生效。

---

## 3. 55-review 逐条处置（轴 1 相关）

| 条目 | 处置 | 证据 |
|---|---|---|
| A-P1/P2 目的丢失、explicit 右移未立项 | ✅ 新 schema 强制 `purpose`；`1.1.4.1/.2/.4/.5` 立 explicit 右移设计集 | `1.1.4.*` 的 purpose/innovation |
| A-P3 clean bpc 测不到搬运 | ✅ 判据改用 KV 消融保留曲线 + 长程召回 + 毁源敏感度 | `1.1.1`/`1.1.4.1` validation/argument |
| A-P4 无 pareto 口径 | ✅ `validation`/`expected_effect` 写入「stock 全因果归一 100% + 质量-代价前沿」 | 全部 12 份 `validation` |
| A-P5 判「是否无损」 | ✅ 改为「窗口是手段、代价进前沿」 | `1.1.1` gap |
| A-P6 0.014 带自否 | ⚠️ 仅 `1.2.3` 显式重校准（两臂界≈0.0195）；其余条目仍以 0.014 作操作阈值 | `1.2.3` gap；§4 N3 |
| A-P7 A3/A4 无仓内可复核实验 | ✅ 记为 `gaps` + WP1「自包含复算」 | `1.1.1` gaps/WP1、`1.1.2` gaps |
| A-P8 A2 基线仍立项 | ✅（跨轴）降为 `7-references` 台账（轴 7 WP） | `axes/7-references/README.md` |
| A-P9 A3 标题即手段 | ⚠️ 标题按冻结 §12 仍为 `hard 窗口`（不可改）；目的转入 `purpose` | `1.1.1` purpose |
| A-P10 A4 ALiBi 口径错 | ✅ 目的改写，补 per-head slope + 长度外推判据，旧读数降为「口径不全旧观测」 | `1.1.2` gap/conclusion_review |
| A-P11/P12 A6 一格里两假设、长T 命名 | ✅ 拆 `1.2.2`+`1.2.3`，长T 系列归 `1.1.1`，命名改「窗口代价随 T」 | MAPPING §3；`1.1.1` change_log |
| A-P13 A8 手段/零实验 | ✅（跨轴）降 `7-references` | `7-references/README.md` |
| A-P14 A9≈A3 重复 | ✅ `1.1.4.5` 聚焦 implicit-vs-explicit 对撞；重复部分归 `1.1.1` | `1.1.4.5` gap/change_log |
| A-P15/P16 A10 判据≠实测、门槛不可达 | ✅ 判据改「固定 ρ + ρ-质量曲线 + 同可见性对数预算」，注明旧门槛算术不可达 | `1.1.4.3` gap/conclusion |
| A-P17 造轮子 | ✅ `sota_baseline` 各列业界巨人（Longformer/DeltaNet/ToMe/MLA…） | 各 YAML `sota_baseline` |
| A-P18 A3 自相矛盾 | ✅ 结论统一为「丢前缀不便宜（+0.212）」，去相反叙述 | `1.1.1` conclusion |
| A-P19 top-k 显著性口径 | ✅ `1.2.3` 注明配对 t 不含训练重跑方差 | `1.2.3` gap |
| A-P20/P23 status/notes 矛盾 | ✅ 随 `catalog.yaml` 退役（旧 SSOT 失效） | 不在新 YAML 复现 |
| A-P21 证据锚旧 `.scratch` | ✅ 锚点改指真实存在的仓内实验 + 可达 `@sha`（`check_tree` 锚点门通过） | 各 `experiments` 列表 |
| A-P22 跨轴泄漏 | ✅ explicit 落 `1.1.4.1`/`1.1.4.4` | — |
| A-R1 注释掉的 prior_art_note | ✅ 随 `catalog.yaml` 退役 | — |
| A-R2 A6 options 引已删 A5/A7 | ✅ `1.2.1` 把 A5 作**来源指针**（指 archive），非旧「同预算选项」 | `1.2.1` sota_baseline/change_log |
| A-R3 A6 verdict 历史叙事 | ✅ 新 YAML 陈述式、历史进 `change_log` | 各 YAML |
| A-R4 `*.md.md` 双扩展名 | ❌ **未清理**（83 个仍在 `.scratch/doc-log/**`，其中 A-mixing 20 + M-carrier 33 属轴 1 源） | §4 N1 |
| A-R5 A2 stub 实验目录 | ✅（跨轴）随 A2 迁 `7-references` | — |
| A-R6 A3 旧代号 X2/X3 | ⚠️ 新 YAML/README 无 X 代号；旧代号仍在**未改动的迁移证据** `2505/2622` README 内（R100 迁移，允许） | — |
| M-01/02/03/12 目的/前沿缺失 | ✅ `1.1.4.1` 目的改「信息右移 + 质量-代价前沿」，补 WP1 前沿图 | `1.1.4.1` purpose/WP1 |
| M-04/05/06 机制复刻+退化+设计不全 | ✅ 列 `gaps` + WP2「逐层状态+门控+写入门，与 DeltaNet/GLA/Mamba-2 对照」 | `1.1.4.1` WP2 |
| M-08/09 错误归因 | ✅ 列 WP3「撤销零样本 OOD『能力缺席』与『+0.2% 说明有用』」 | `1.1.4.1` WP3 |
| M-14 组合实验越界 | ✅ 列 `gaps`「应移出本项 scope」 | `1.1.4.1` gaps |
| M-16 实验编号不唯一 | ✅ 列 `gaps`（M-16） | `1.1.4.1` gaps |
| M-18/19/20 M2 名不副实/陈旧/无目的 | ✅ `1.1.4.2` 目的改「每位置字节预算下的压缩前沿」，指明旧证据只是 retention 仪器 | `1.1.4.2` |
| M-22 M1 误标 judged | ✅ `1.1.4.1` status `judged→进行中` | `1.1.4.1` conclusion_review |

---

## 4. Material defects

**blocking：无。**

**non-blocking：**

- **N1（残留未清，R4）** — 位置：`.scratch/doc-log/axes/A-mixing/**`（20）、`.scratch/doc-log/axes/M-carrier/**`（33），
  全仓共 83 个 `*.md.md`。违反：ticket 56 §2.4/验收 5「`*.md.md` 双扩展名…逐条消失」。
  复现：`find .scratch/doc-log -name '*.md.md' | wc -l` → 83（`ce2610a` 时也是 83，零清理）。
  影响：仅日志目录命名缺陷，**不动 axes 事实文档、不丢证据**；可能落在专门的「去残留」WP
  （跨轴），故不阻塞轴 1。若把 M2 验收 5 严格归到轴 1 WP，则此项未闭合。
- **N2（role_map 未实名）** — 位置：全部 12 份 YAML 的 `role_map`。`owner="轴 1 时间轴 lead（待指派）"`、
  `writer="1.x worker（单一写入者）"`、`reviewer="fresh 复核者（reviewer 池…）"`、
  `integrator="研究树 integrator"` 均为**角色占位**，非 schema §4 所称的「轴 lead 名」实名。
  复现：`grep -A4 role_map axes/1-time/*/project.yaml`。ticket 56 §5.5 已列此为「M2 派发时由主
  Agent 填实名」的待决项 → non-blocking。
- **N3（P6 重校准不一致）** — `1.2.3` 按 55-review P6 把 0.014 带重校准为两臂界 ≈0.0195，
  但 `1.1.1/1.1.3/1.2.1/1.2.2/1.2.4` 仍在 `objective_criteria` 直接用「0.014 噪声带」作可排序阈值。
  复现：`grep -n '0.014' axes/1-time/*/project.yaml`。属**条目间口径不一致**（P6 是跨轴根因 R8），
  未造成结论错误（相关项多为 `未开始/进行中`），non-blocking。
- **N4（规范内歧义）** — `docs/structure.md` §12 给 `1.3 寻址` 列了 `目录=axes/1-time/1.3-addressing/`
  且 `粒度=参照`；§13.4 则规定参照「不写 project.yaml、进 `axes/7-references/README.md`」。
  轴 1 WP 取 §13.4（登记于 7-references，未建目录）。二者**规范自冲突**，非执行方过错；
  建议后续统一 §12 表格。non-blocking。
- **N5（迁移口径小偏差）** — MAPPING §0.2 要求旧格「整目录搬到新家（保留子路径名）」，实际 5 个旧
  `experiments/README.md` 是**删除后被新索引取代**（非 rename）。内容为旧 wiki 自动索引，非研究
  证据，轴 1 新家各有对应新 `experiments/README.md`。non-blocking。
- **N6（个别 55-review 条目无显式引用）** — 轴 1 新 YAML 未显式引用
  `M-07/M-10/M-11/M-13/M-15/M-17/M-21/M-23`：其中 M-13/M-15/M-17（M1 单格 23 实验目录仍平铺、
  引擎子项目 `2503-engine-m1-extensions`/`2622-eng1-composable`/`2628-eng2-carry-ragged` 仍挂在
  `1.1.4.1/experiments/`、两份 `PRIOR_ART.md` 未合并）**尚未在轴 1 WP 内结构性处置**，可能落在
  后续 M2 去残留/结构收尾 WP。复现：`ls axes/1-time/1.1.4.1-explicit-memory/experiments/ |
  grep -E '2503|eng1|eng2'`；`grep -rl PRIOR_ART axes/M-carrier/`。non-blocking。

---

## 5. 未复核范围

- **轴 2–9 的结构与内容**：本次只核轴 1；`axes/2-depth/**` 等在 `summary.py --check` 中一并通过，
  但未逐份审其字段/证据/55-review 处置。
- **旧 wiki 物理退役**：`catalog.yaml`/`INDEX.md`/`tools/build_wiki.py` 等移入 `archive/wiki-legacy/`
  尚未发生（M2 收尾），本次未核其归档正确性；仅确认其不再是新 YAML 的判据来源。
- **跨轴一致性/integration（M3）**：`integrator` 的合并、用词唯一、证据不重复等未核。
- **迁移证据内部的语义正确性**：只核「数字未被改写、路径指针被正确改写」；**未**重跑任何实验、
  未核实验结论对错、未核迁移来的 `analysis.md` 科学质量（那是各研究项后续 `work_packages` 的任务）。
- **`.scratch/**/evidence` 与 `archive/`**：只确认原地未动，未逐文件核。
- **`role_map` 实名/轴 lead 指派**：属主 Agent M2 派发动作，未核。

---

## 6. 复核方法（可复现）

只读；所有改动仅落 `/tmp`（`/tmp/inject-axis1/`、`/tmp/pytest-axis1-*.log`、`/tmp/*.txt`）。
命令见 §2 各条；未 `git add`/`commit`，未改仓库任何文件。关键命令：`git rev-parse HEAD`、
`git status --short`、`git diff -M --raw ce2610a HEAD`、`git ls-files axes`、
`find axes -type f`、`python3 tools/{summary.py --check,check_tree.py,check_schema.py}`、
`python3 -m pytest tests/ -q`、`/tmp/inject-axis1/tools/check_tree.py`。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "交付了范围被钉死的轴 1（WP-M2.1+WP-M2.2）只读复核报告，未越界修改仓库（git status 与复核前一致，仅新增 log 期）；报告覆盖节点完整性/exact wording/字段/证据迁移/55-review/工具/坏 YAML 注入七项契约。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "每条判据均附可复现命令与真实输出：summary/check_tree/check_schema 全绿(exit0)、pytest 827 passed in 94.88s、四类坏 YAML 注入全部 FAIL、内容级证据比对(182 个改名文件仅 1 个索引残差)、rev= bacaaeba。"
    }
  ],
  "changedFiles": [
    "/home/z/.pi/agent/sessions/--home-z-vibe-pathlm--/subagent-artifacts/outputs/89c48c00-453d-4535-8c0d-2d2d779d1bda/.scratch/56-restructure/wp-reports/rev-axis1.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    { "command": "git rev-parse HEAD && git status --short", "result": "passed", "summary": "HEAD=bacaaeba…；4 个 pre-existing M + 2 个 ?? .scratch 目录" },
    { "command": "python3 tools/summary.py --check", "result": "passed", "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md。exit=0" },
    { "command": "python3 tools/check_tree.py", "result": "passed", "summary": "OK：46 个 project.yaml 全部符合。exit=0" },
    { "command": "python3 tools/check_schema.py", "result": "passed", "summary": "转调 check_tree，OK。exit=0" },
    { "command": "python3 -m pytest tests/ -q", "result": "passed", "summary": "827 passed in 94.88s (0:01:34) exit=0（首轮受 4 份并发 pytest 争用中断，机器空闲后单跑完成）" },
    { "command": "/tmp/inject-axis1/tools/check_tree.py（缺 role_map / 缺 failure_analysis / 墙钟 cost_model / status=被否证但归因为空 四类注入）", "result": "passed", "summary": "四类坏 YAML 全部 FAIL(exit1) + 对照 OK(exit0)" },
    { "command": "git diff -M --raw ce2610a HEAD + 内容级归一比对", "result": "passed", "summary": "R1567/D5/A91/M48；182 个改名有变文件仅 1 个索引 README 行数变化，其余仅路径指针改写，测量数字零改动" },
    { "command": "find axes -type f | wc -l / git ls-files axes | wc -l", "result": "passed", "summary": "现 2510 / 1733；基线 2434 / 1657（均 +76，口径自洽）" }
  ],
  "validationOutput": [
    "summary.py --check / check_tree.py / check_schema.py → OK：46 个 project.yaml 全部符合（exit 0）",
    "pytest → 827 passed in 94.88s (0:01:34)",
    "坏 YAML 注入：缺 role_map→FAIL、缺 failure_analysis→FAIL、墙钟 cost_model→FAIL、被否证空归因→FAIL；对照→OK",
    "证据：ce2610a→HEAD 无实验数据文件删除（仅 5 个旧自动索引 README 被新索引取代）；182 个改名有变文件经归一后仅路径差异"
  ],
  "residualRisks": [
    "N1 non-blocking：83 个 *.md.md 仍在 .scratch/doc-log（A-mixing 20 + M-carrier 33 属轴 1 源），ticket 56 验收 5 未闭合，疑属专门去残留 WP",
    "N2 non-blocking：12 份 YAML 的 role_map owner/writer/reviewer/integrator 全为占位，未实名（待主 Agent M2 指派）",
    "N3 non-blocking：仅 1.2.3 按 55-review P6 重校准 0.014 带，其余条目仍直接用 0.014，条目间口径不一致",
    "N4 non-blocking：docs/structure.md §12（1.3 有目录）与 §13.4（参照不建目录）规范自冲突；WP 取 §13.4",
    "N6 non-blocking：M-07/10/11/13/15/17/21/23 未被轴 1 YAML 显式引用；M-13/15/17/21 所示结构（1.1.4.1 仍挂 23 个实验目录与 3 个引擎子项目、两份 PRIOR_ART）尚未在轴 1 内收尾",
    "未复核：轴 2–9 内容、旧 wiki 物理归档、M3 跨轴 integration、迁移证据的科学结论正误",
    "pytest 曾因同机 4 份并发争用被中断；已在机器空闲后单跑取得真实通过结果（827 passed）"
  ],
  "noStagedFiles": true,
  "diffSummary": "只读复核：仓库零改动；仅在 /tmp 注入坏 YAML 与写报告输出。",
  "reviewFindings": [
    "blocker: 无",
    "non-blocking: .scratch/doc-log（A-mixing 20 + M-carrier 33）共 83 个 *.md.md 残留未清（ticket56 §2.4/R4）",
    "non-blocking: 全部 12 份 project.yaml 的 role_map 四角色均为占位未实名",
    "non-blocking: 0.014 噪声带的 P6 重校准只在 1.2.3 落实，其余条目仍引用旧带",
    "non-blocking: docs/structure.md §12 与 §13.4 对 1.3 参照的目录要求自冲突",
    "non-blocking: 5 个旧 experiments/README.md 为「删除+新索引」而非整目录 rename（非证据）",
    "non-blocking: M-13/15/17/21 所示 M1 扁平结构/引擎子项目/双 PRIOR_ART 未在轴 1 内结构收尾"
  ],
  "manualNotes": "总判定 PASS。轴 1 两 WP（18a3e6b、65adaec）在节点/exact wording/字段/证据迁移/55-review/校验/坏 YAML 七项上均达标，无 blocking defect。报告按要求写入 session artifact 路径（复核为只读、不改仓库，故未写 repo 内 .scratch 副本）。首轮 pytest 因 4 份并发争用被中断，机器空闲后单跑得真实结果 827 passed in 94.88s。"
}
```
