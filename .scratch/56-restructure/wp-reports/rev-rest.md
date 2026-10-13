# 独立复核报告 · ticket 56 M2 轴 3/4/5（WP-M2.5 + WP-M2.6）与轴 7/8/9（WP-M2.7）

> 角色：fresh 上下文的独立复核者，**只读**（未改仓库；仅用 `/tmp/acht/` 做注入证伪）。
> 复核对象：**工作树当前 revision** = `git rev-parse HEAD` = `bacaaeba6464f75b24fd28e30bca621e18527462`（分支 `main`）。
> 复核时刻 `date -Iseconds` = `2026-10-13T11:37:24+08:00`。
> 复核前工作树状态（`git status --short`，原样记录，未因本次复核改变）：
> ```
>  M INDEX.md
>  M axes/C-vertical/C1-layers-per-token/README.md
>  M axes/C-vertical/README.md
>  M catalog.yaml
> ?? .scratch/54-subproject-schema/
> ?? .scratch/55-review/
> ```
> 基线文档（先读后看）：`docs/structure.md` §12、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`；
> 复核的 WP 报告：`wp5.md`（轴 3）、`wp6.md`（轴 4/5）、`wp7.md`（轴 7/8/9）。

---

## 0. 人话摘要（先讲结论）

**结论：PASS（可在有非阻断遗留项的前提下接受）。** 没有发现「证据丢失、节点错、字段缺、口径混」这类致命问题；
三批迁移（`1dcb5ce` 轴 3、`d76d997` 轴 4/5、`bacaaeb` 轴 7/8/9）都用 `git mv` **整目录搬移**，我用「搬前/搬后
目录 object-tree hash 逐一对齐」证明 27 个实验目录**逐字节未变**，旧目录只剩 README（无重复复制），
`.scratch/**/evidence` 原地未动；三批提交**零删除**，工作树文件数 2434（基线）→ 2510、`git ls-files axes`
1659（foundation）→ 1733（HEAD）单调上升，证据只增不减。

字段与用词：19 份在范围内的 `project.yaml` **逐条**对 `docs/structure.md` §12 做 exact-wording 与目录/轴名比对，
**全部一致**；字段齐全、单 milestone、`role_map` 四角色、每条 WP 带 acceptance + 独立复核、`cost_model` 无墙钟，
校验器 `summary.py --check` / `check_tree.py` 实跑 **OK 46 项（exit 0）**。

证伪：在 `/tmp/acht/`（复制校验器 + 造 `axes/case/project.yaml`）注入 4 类坏 YAML（缺 `role_map`、缺
`failure_analysis`、墙钟 `cost_model`、`status=被否证` 但归因为空），**校验器全部判 FAIL**——校验门是活的，不是摆设。

**唯一没能拿到终值的是 `pytest`**：整套跑在后台推进到约 **80%（全为 `.`，未见 F/E）** 仍未结束，定向重跑 6 个
被迁移改过路径的测试文件在 280 s 内也未跑完（都是 torch 训练/推理单测，且当时有并发占用）。按主编排指示停止长时间轮询，
此条记为「已启动、观测到进度、**未完成**」，不作为 PASS 的正面证据，也不因此判 FAIL。

未复核范围见 §6。

---

## 1. 逐条判据（✅/❌ + 复现）

### 判据 1 · 节点齐全且 exact wording —— ✅

- 轴 3：`3.1`（分组）+ `3.1.1`–`3.1.5` + `3.2`（分组）+ `3.2.1`–`3.2.3`，目录全在，9 份 `project.yaml`。
- 轴 4：`4.1`–`4.3`，3 份 `project.yaml`。
- 轴 5：`5.1`–`5.6`，6 份 `project.yaml`。
- 轴 7：仅 `axes/7-references/README.md`（参照台账，**无** `project.yaml`），`experiments/` 收 A2/A8/W3/R7。
- 轴 8：仅 `axes/8-engineering/README.md`（工程纪律台账，**无** `project.yaml`），`experiments/2523b-shallow-weight-sweep`。
- 轴 9：`axes/9-backlog/README.md` + `9.1`/`9.2` 两份 Backlog `project.yaml`。
- **复现**：解析 `docs/structure.md` §12 表格得 57 节点，对全部 `axes/**/project.yaml` 比对 `title`/`dir`/`axis`：
  `grep` 脚本输出「0 处 MISMATCH」；轴名与 §12 小标题一致（如轴 4 = `4 专家轴（MoE）`、轴 9 = `9 Backlog（不删，移出主线）`）。
  分组节点（`3.1`/`3.2`）只有聚合 `README.md`、无 `project.yaml`，符合 §13 粒度规则。

### 判据 2 · 每个 `project.yaml` 字段齐全（role_map / 单 milestone / WP 含 acceptance + 独立 review / 无墙钟）—— ✅

- **复现（结构脚本，19 份在范围内 yaml）**：`ALL STRUCTURAL CHECKS PASS (in-scope)`，覆盖
  `id/axis/title/dir/purpose/innovation/expected_effect/sota_baseline/gap/method/validation/
  objective_criteria/falsification/cost_model/role_map/milestone/work_packages/roadmap/status/
  failure_analysis/conclusion_review/experiments/argument/gaps/started/ended/change_log` 全在，关键字段非空；
  `role_map` 四角色齐且 `reviewer != writer`；`milestone` 为单 map 且含 `output`+`acceptance`；
  每条 WP 的 `acceptance` 与 `review` 非空且 `review` 均指向独立复核（fresh/复核）；`status` 仅四枚举；
  `已完成` ⇒ 有 conclusion + 存在的 experiments；`status=被否证` ⇒ `failure_analysis` 非空；`cost_model` 无墙钟/日期。
- **范围外的全树**：`python3 tools/check_tree.py` → `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0）；`python3 tools/summary.py --check` 同上。
- 9 个「已完成/进行中」项的 `conclusion`/`conclusion_review`（如 `5.1`、`3.2.1`、`5.6`）均按 55-review 修正口径，
  且 `5.1`/`3.2.1` 的 `experiments` 路径经校验存在。

### 判据 3 · 证据按 MAPPING 迁移、无丢失（迁前/迁后计数对比，含 git 跟踪数）—— ✅

- **逐目录 object-tree hash 对齐**（`git rev-parse <pre>:<old>` vs `<post>:<new>`）——**27/27 全部 OK（逐字节相等）**：
  - 轴 3（`61e346b`→`1dcb5ce`）：`W1/2525-tax-attribution` → `3.1.1/`；`W4/{2610,2611,2612,2613,2621}` → `3.2.1/`。
  - 轴 4/5（`1dcb5ce`→`d76d997`）：`R1/{2614,2619,2620,2621,2631,2632}` → `5.1/`；`R4/2617` → `5.6/`。
  - 轴 7/8/9（`d76d997`→`bacaaeb`）：`A2`/`A8`/`W3`/`R7` 的 `experiments` → `7-references/experiments/`；
    `P5/2523b` → `8-engineering/`；`G1/{2508,2514,2515,2517}`、`G3/2522` → `9.1/`；`F2/2511`、`F4/2619` → `9.2/`。
- **零删除**：`git show --name-status -M` 对三批提交统计 `D`（删除）= **0 / 0 / 0**（轴3: 109 rename+13 add；轴4/5: 92 rename+13 add+6 modify；轴7/8/9: 98 rename+6 create+6 modify）。
- **计数**：
  - 工作树 `find axes -type f | wc -l` = **2510**（≥ MAPPING §0 基线 2434，只增不减）。
  - git 跟踪 `git ls-tree -r --name-only <c> axes | wc -l`：`ab1d8d8`=1659 → `61e346b`=1701 → `1dcb5ce`=1714 → `d76d997`=1727 → `bacaaeb`=1733（单调上升，与「只增文档、证据 rename 不改变计数」一致）。
- **一格一份家、不复制**：旧轴目录 `axes/{W-width,R-references,A-mixing,C-vertical,H-horizon,G-policy,F-iteration,P-modifiers,M-carrier}/**` 现仅剩 `README.md`/`PRIOR_ART*.md`，实验子目录已不在旧家（无重复实例）。
- **`.scratch` 原地不动**：三批迁移提交的变更列表里 **0 条 `.scratch/**` 路径**；`.scratch/**/evidence` 仅作 `@<sha>` 锚点。

### 判据 4 · 55-review 相关问题逐条处置 —— ✅（含一条「文档线索」非阻断备注）

- **W 轴（→轴 3/4/7）**：轴 3 README 边界表处置 W-01/W-02/W-05/W-03/W-18；`3.1.1` 处置 W-06/W-07/W-14/W-17；
  `3.2.1` 处置 W-10/W-11（判据改为「等参数对照归因 + 可迁移性」）；轴 4 README/`4.x` 处置 W-03/W-04/W-12/W-13。
- **W-08 / R8（作废的 int4 跨家族数字）**：`5.1` 与轴 5 README 明确「逐值标尺口径已撤回、改用块标尺」，并**禁止引用**
  「同 −75% 档 GQA 便宜 3.2×」；`5.6` 量化腿一律指向 `5.1` 块标尺口径。
- **A/W/R7 → 轴 7**：A-P08/A-P13/A-P21（A2/A8 降参照、stub 随迁）与 W-03、55-R07（R7 降参照）兑现为
  `axes/7-references/README.md` 的四条「必须打败的跑分线」+ `experiments/{A2,A8,W3,R7}` 整搬。
- **G/F → 轴 9**：`9.1` 的 `conclusion_review`/`falsification` 明确处置 G-P1–P4/P9（去目的 = 工具 KPI、方差带内不可排序、
  100M 假设失效升格）；`9.2` 的 `purpose`/`falsification`/`conclusion_review` 处置 F-01/F-02/F-06/F-07/F-12/F-15/F-19 等。
- **备注（非阻断）**：`wp5.md` 的 55-review 处置表显式列了 W-01…W-17 中大部分，但未单列 W-04/W-08/W-12
  （W-08 属 W3→轴 7；W-04/W-12 在 `gap`/`expected_effect` 内容中已落）；`wp7.md` 未给逐条处置表，仅按根因分组叙述。
  这是**文档追踪颗粒度**问题，产物本身已按处置落地。

### 判据 5 · 跑校验/测试并记录真实输出 —— ⚠️（2/3 完成，pytest 未完成）

- `python3 tools/summary.py --check` → `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`，`EXIT=0`。
- `python3 tools/check_tree.py` → `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`，`EXIT=0`。
- `python3 -m pytest tests/ -q` → **后台启动（PID 4055848），观测进度约 80%（输出全为 `.`，未见 F/E），超预算仍未结束**；
  另定向重跑 6 个被改路径的测试文件（`test_kv_{quant,budget_gaps,budget_rest,budget_causal,block_scale}.py`、
  `test_r7_exit_policies.py`）在 `timeout 280` 内**无输出、exit 143（超时被杀）**。
  ⇒ 记为「已启动、观测到进度、**未跑完**」，不作为正面证据。
- 三批提交 `git diff --check HEAD~1 HEAD / HEAD~2 HEAD~1 / HEAD~3 HEAD~2` 均 `exit 0`（无行尾空白）。

### 判据 6 · `/tmp` 注入坏 YAML 证伪 —— ✅（4/4 命中）

在 `/tmp/acht/`（复制 `tools/check_tree.py` + 造 `axes/case/project.yaml`，工作树零改动）逐例运行：

| 注入缺陷 | 校验器输出 | exit |
|---|---|---|
| 缺 `role_map` | `缺字段 role_map` + `role_map 必须是 map` | 1 |
| 缺 `failure_analysis` | `缺字段 failure_analysis` | 1 |
| 墙钟 `cost_model`（`"墙钟 3.1 秒"`） | `cost_model 含墙钟/日期指标 '墙钟'（只许 FLOPs / layer-passes / tokens-per-forward 等等效理论量）` | 1 |
| `status=被否证` 且 `failure_analysis` 为空 | `status=被否证 但 failure_analysis 为空（必须区分 设计问题 / 结构问题 / innovation 本身失败 并给证据）` | 1 |

---

## 2. Material defects

**Blocking：无。**

**Non-blocking（不影响本轮 PASS，供 integrator 收尾时处置）：**

- **N1（证据/指针一致性）· 新 SSOT 仍指向「待退役」旧文件**：
  `3.1.1`/`3.2.1` 的 `sota_baseline` 指 `axes/W-width/PRIOR_ART.md`（§W1/§W4）；`4.1`/`4.2`/`4.3` 指同文件 §W2；
  `9.1` 指 `axes/G-policy/PRIOR_ART.md`；`9.2` 指 `axes/F-iteration/PRIOR_ART.md`。
  这些旧目录按 §15 待退役，退役后指针将悬空。WP 报告已自认「待 integrator 退役」。
  复现：`grep -rn "PRIOR_ART.md" axes/{3-width,4-moe,5-quant,9-backlog} --include=project.yaml`。
- **N2（状态粒度，schema §7「status 只回答做完没有」）· 顶层 `已完成` 与未完 WP 并存**：
  `5.1.status=已完成` 而 `WP2=未开始`；`3.2.1.status=已完成` 而 `WP4=未开始`；且 `3.2.1.ended == started`。
  结论本身明确（`5.1` 块标尺判决完整、`3.2.1` 判「不采纳为通用质量机制」），属可辩护取值；WP 报告已列为未决项。
- **N3（W-06 归属张力）· `2525-tax-attribution` 落点**：物理上落 `3.1.1/experiments/`（遵 MAPPING §1），
  但 55-review W-06 判其属读出域（`3.2.1`）。当前处理：`3.1.1` 标 `未开始` 并把它登记为**背景证据**，`3.2.1` 指针引用。
  复现：`axes/3-width/3.1.1-uniform-narrowing/project.yaml` 的 `gap`/`argument` 与 `3.2.1` 的 `experiments`。
- **N4（文档追踪颗粒度）**：见判据 4 备注（`wp5` 未单列 W-04/W-08/W-12；`wp7` 无逐条表）。
- **N5（残留路径，已自认）**：迁入的实验目录内脚本仍硬编码旧轴路径（WP6 §4.1 记约 225 处），
  不致 pytest 失败，但会让手动重跑失效；`5.1` 的 `config.json`/`results.json` 里 `source/root` 仍写 `axes/W-width/...`。
- **N6（pytest 未完成）**：判据 5 第 3 条，属**证据缺口**（非缺陷），独立验收若要全绿证据需补跑。

---

## 3. 未复核范围（明确划界）

- 轴 1 / 轴 2 / 轴 6 的迁移（其 `project.yaml`、README、证据落点）——不在本次范围，仅当它们作为本范围指针目标时被间接看到。
- `tools/summary.py` / `check_tree.py` 的**实现质量**（仅把其输出当门用，未做代码级审计）。
- `catalog.yaml` / `INDEX.md` 的退役与 `archive/wiki-legacy/` 迁移（未发生）。
- 工作树既有未提交改动（`INDEX.md`、`catalog.yaml`、`axes/C-vertical/{README,C1.../README}.md`）的**内容正确性**——仅确认三批迁移提交未触碰它们（`git show --name-only` 过滤为空）。
- 迁入实验目录内部脚本的路径重写（N5）——只记录了现象，未逐文件核对。
- `pytest` 的完整终值（判据 5）。
- 参照台账/工程纪律台账内的**研究数字真实性**（如 A2 的 1.5074 bpc、W3 的 GQA +0.0143）——只核到「已按 55-review 口径登记、未用已作废数字」，未回溯原始 `results.json` 重算。

---

## 4. 复核方法（可复现）

1. `git rev-parse HEAD` + `git status --short` + `date -Iseconds` 记录 revision/脏状态/时刻。
2. 读 `docs/structure.md`（§12）、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`、`wp5/6/7.md`、`tools/check_tree.py`。
3. 解析 §12 得节点表，脚本比对 19 份在范围 `project.yaml` 的 `title/axis/dir`（exact wording）——0 不一致。
4. 结构脚本逐字段审计 19 份 yaml + `role_map`/`milestone`/WP/`cost_model`/`status` 语义。
5. `git rev-parse <pre>:<old>` vs `<post>:<new>` 逐目录树 hash 对齐 27 个实验目录；`git show --name-status -M` 统计 rename/delete。
6. 计数：`find axes -type f`、`git ls-files axes`、`git ls-tree -r <c> axes` 跨 5 个提交。
7. `summary.py --check` / `check_tree.py` 实跑；`git diff --check` 三批提交。
8. `/tmp/acht/` 注入 4 类坏 YAML 证伪校验门。
9. `pytest` 后台启动并观测（未完成）。

**工作树零改动**：全部写操作只落在 `/tmp/acht/` 与 `/tmp/pytest_out.txt`；仓库内无新增/修改（承 §0 的 `git status` 未被复核改变）。
