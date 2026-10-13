# ticket 56 · M2 整体迁移 · 独立集成复核（fresh 上下文，批判性，只读）

> 复核角色：reviewer（fresh 上下文；不继承任何 writer 会话）。语言：中文。
> 复核 revision：`3861ea54dea99bd4d81a1bafb165322bd6593e81`（HEAD / main）。
> 基线：`ab1d8d8`（M1 冻结；旧轴未动）与迁移前 `ce2610a`。迁移窗口 `ab1d8d8..HEAD`。
> 契约：`.scratch/56-restructure/TICKET.md` §3、`docs/structure.md` §12–§15、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`。
> 复核输入：`WP-M2.8-integrator.md`（**仅存在于 subagent artifact 目录，不在仓库**，见判据 7）、`rev-axis1/2/6/rest.md`。
> 只读约束：全部注入/克隆在 `/tmp` 副本（`/tmp/m2inject`）进行；未改仓库工作树（唯一副产物：`--collect-only` 生成的 2 个 `__pycache__`，已删除，见 §复现）。

---

## 0 人话摘要（先结论）

**结论：PASS。**

7 个轴合成的最终状态经**独立重建**核对后成立：旧树的 **1562 个实验证据文件一条没丢**（我用自己的内容哈希法独立复核，不是照抄报告：在 `bacaaeb` 上 1562 个文件里 **1380 个逐字节相同**、182 个按设计改了正文；全部 1562 个都有落盘的新家），旧 9 轴连旧 wiki 资产 **91 个文件整体只读归档进 `archive/wiki-legacy/`**，工作树 `axes/` 只剩新轴 1–9。46 个 `project.yaml` 的 `id/title/axis/dir` 与 `docs/structure.md` §12 **逐字对齐（46/46，零不一致）**；`1.3`/`6.1` 的规范自冲突已消除且与磁盘一致。四道门我**实跑**全绿（`summary --check` / `check_tree` / `check_schema` 各 ~0.6s OK；`pytest 815 passed in 93.01s`）。schema 门是**活的**：我在 `/tmp` 副本里注入 5 种故障（缺 `role_map`、缺 `failure_analysis`、墙钟 `cost_model`、`被否证` 归因为空、单 milestone 变两条），**逐一 FAIL 且报错信息精确命中**，控制组通过。55-review 的 8 条根因（R1–R8）逐条有对应处置，拆并格**一格一主家**（无任何实验目录同时挂在两个研究项下）。

账实相符：整合报告里可抽查的关键数字（1562 / 1380 / 182 / pytest 815 / 827→815 差额 12 / 91 rename / 4 文件 / 388 文件 / 83 个 `*.md.md`）我**全部独立复现一致**。

**唯一实质缺口是审计可追溯性，不是数据或树**：① 整合报告 `WP-M2.8-integrator.md` 只躺在 subagent artifact 目录，**没进仓库**（仓库 `wp-reports/` 只有 11 份源报告，缺整合报告本身）；② 迁移的依据台账 `.scratch/54-subproject-schema/`、`.scratch/55-review/`（MAPPING 大量引用其条目）是 **untracked**，没进 git。两者都不影响当前磁盘事实，但会让「后人 clone 仓库」看不到完整证据链。此外 `rev-axis6` 的 **D1 报告笔误**（`wp4.md` 里 `1692=1692` 恒真自比）在整合报告的 §1 处置表里**未被列出**（D1 本身不改结论，且报告是历史件不改写，属低优先）。上述均为 **non-blocking**。

---

## 1 判据逐条结论

### 判据 1 · 证据零丢失 — ✅

**独立做法**（不照抄报告）：从基线 `ab1d8d8` 枚举旧 9 轴全部文件，用 `git diff --name-status -M` 建旧→新 rename 表，再用 **blob 内容哈希**判断「逐字节保留」与「改过正文」，最后把「既不在 rename 目标、又不在归档」的文件单列并逐一在磁盘定位。脚本见 §复现 A。

- 旧 9 轴在 `ab1d8d8` 共 **1657** 个跟踪文件；其中 `experiments/**`（排除自动索引 `experiments/README.md`）**1562** 个。
- **1562 / 1562 全部有落盘去处**（1561 在 `-M` rename 表内且目标存在；1 个被 `-M` 误判为删除，实际存在于新家，见下）。
- 内容层面（`bacaaeb`，即整合器 N1 路径改写**之前**）：**1380 个逐字节相同**、182 个正文有变（迁移期按 55-review 修正 + 后续 N1 路径改写）。
- `-M` rename 计数：`ab1d8d8..bacaaeb -- axes`：R 总数 **1567**、**R100 = 1384**、R<100 = 183（R100 我用 blob 比对**逐个验证 1384/1384 相同**）。
- **5 个真删除** = `A10/A3/A4/A9/M2` 的自动索引 `experiments/README.md`（旧 wiki 自动块），各新家都有自己的 `experiments/README.md`（已核对 5/5 存在）。
- **1 个「假删除」**：`axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/zero-init-saddle/model.pt.sha256` —— `-M` 因该文件正文几乎全是路径而被判 D，但**内容已在** `axes/3-width/3.2.1-readout-reparam/experiments/2610-.../zero-init-saddle/model.pt.sha256`；用「experiments 相对尾路径」定位 **6/6 命中，STILL not located = 0**。**无丢失。**
- `archive/wiki-legacy/` = **91 文件** = 85 旧轴文件 + `catalog.yaml` + `INDEX.md` + 3 个生成式工具 + 1 个旧测试；旧 9 轴目录齐全。
- 抽样哈希：随机抽 `2630-tf4-samecaliber/results.json`（`axes/M-carrier/M1-carrier-type/...` → `axes/1-time/1.1.4.1-explicit-memory/...`），`diff` 显示**只有 5 行路径 token 变化**（`dir`/`path`/`results_json`），测量数字零改动。

**计数口径声明**：1562 是「旧轴 `<格>/experiments/**` 的 git 跟踪文件，排除自动索引 README」；1380 是「`bacaaeb` 上与旧 blob 逐字节相同的数量」。这两条正是整合报告的 §2.1/§7.4 口径，我独立复现**完全一致**。

### 判据 2 · 旧树退役正确 — ✅

- `git ls-tree HEAD:axes` = `1-time 2-depth 3-width 4-moe 5-quant 6-predict 7-references 8-engineering 9-backlog`；工作树**无任何旧轴目录**（`ls axes | grep -E '^(A|C|F|G|H|M|P|R|W)-'` = 空）。
- 退役声明与磁盘一致：
  - `README.md:30`「旧的 `INDEX.md` / `catalog.yaml` 已退役」；
  - `AGENTS.md` §2.1 表「旧索引（已退役）| `archive/wiki-legacy/catalog.yaml` / `archive/wiki-legacy/INDEX.md`」、§2.3「已退役并归档于 `archive/wiki-legacy/`」；
  - `docs/structure.md` §15、`SCHEMA.md` 顶部迁移通告、`archive/README.md` §「已归档的旧 wiki」均指向 `archive/wiki-legacy/`；磁盘该目录存在（91 文件）。
- 现行文档中 `catalog.yaml`/`build_wiki` 的每处出现都**显式标「已退役」**，无「同一事实两处现行 SSOT 冲突」。
- 非归档旧轴路径引用 = **0**（`git grep -P '(?<!wiki-legacy/)(?<!dropped/)axes/(A-mixing|...)/'` 在新树/docs/tools/pathlm = 0；残留 4 处均为 `archive/dropped/...` 的合法指代）。
- **低优先观察（non-blocking）**：`SCHEMA.md` §0 的三级定义表「谁来保证」列仍写 `catalog.yaml + check_schema.py`，靠其正上方的迁移注记消歧；属历史描述 + 指针，非冲突。

### 判据 3 · 用词与结构 — ✅

- 解析 `docs/structure.md` §12 全部 57 行，取「研究项/Backlog」粒度行（46）与全部 `project.yaml` 比对：**`id/title/dir` 与文件实际目录 46/46 逐字一致，0 处不一致**；反向亦无「§12 未列出的 `project.yaml`」。
  - 注：§12 的「目录」单元格带 Markdown 反引号与尾斜杠，比对时已规范化（strip backticks + 尾 `/`）；这是文档排版，不是差异。
- `1.3`/`6.1` 自冲突已消：§12 两行的「目录」列 = `axes/7-references/README.md`、粒度 = 参照，与 §13.4「参照只进 `axes/7-references/README.md`」一致；磁盘确无 `axes/1-time/1.3*` / `axes/6-predict/6.1*` 目录，`axes/7-references/README.md` 存在。

### 判据 4 · schema 完整且门是活的 — ✅

- 46 个 `project.yaml` 全过 `check_tree.py`；额外自扫：`sota_baseline/gap/method/expected_effect/argument/conclusion_review` 全部非空（0 空）、`role_map` 四角色全非空（0 缺）、`cost_model` 全部无墙钟/日期词。
- **`/tmp` 克隆注入**（`/tmp/m2inject`）逐项必须在 FAIL；用 `1.1.1-hard-window`（`status=进行中`，10 个 experiments）为受试：

| 注入 | 期望 | 实得 | 精确报错 |
|---|---|---|---|
| 缺 `role_map` | FAIL | rc=1 | `缺字段 role_map` ✔ |
| 缺 `failure_analysis` | FAIL | rc=1 | `缺字段 failure_analysis` ✔ |
| 墙钟 `cost_model`（"墙钟：约 5 分钟"） | FAIL | rc=1 | `cost_model 含墙钟` ✔ |
| `status=被否证` + 空归因 | FAIL | rc=1 | `被否证 但 failure_analysis 为空` ✔ |
| `milestone` 变两条（list） | FAIL | rc=1 | `milestone 必须是 map` ✔ |

- 控制组（注入前 + 还原后）：`OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0）。5/5 注入命中且报错**精准**，非「因别的字段顺带 fail」。

### 判据 5 · 门全跑 — ✅

单独顺序执行，**未并发**：

```
$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。      # exit 0，elapsed 0.59s
$ python3 tools/check_tree.py
OK：46 个 project.yaml 全部符合 docs/research-schema.md。      # exit 0，elapsed 0.58s
$ python3 tools/check_schema.py        # 兼容入口，转调
OK：46 个 project.yaml 全部符合 docs/research-schema.md。      # exit 0，elapsed 0.59s
$ python3 -m pytest tests/ -q
815 passed in 93.01s (0:01:33)                                # exit 0，elapsed 94.24s
```

- **无 orphan pytest**：`pgrep -af pytest` 仅匹配自身命令；`ps -eo pid,cmd | grep pytest` = 无。
- **无本仓 GPU 泄漏**：`nvidia-smi` 计算进程为 `llama-server` PID 145968（3908 MiB）、PID 4155587（14656 MiB），显存 19218 MiB / 利用率 94% —— 均为**与本仓无关**的他人进程，**未动**。
- `pytest 815` 与整合报告一致（报告记 92.75s，属正常抖动）。`827→815 差额 12` 已核：退役的 `archive/wiki-legacy/tests/test_build_wiki_manual.py` 单文件 `--collect-only` = **12 tests**（9 函数，其中 `test_malformed_manual_block_raises` 参数化 4 例）。

### 判据 6 · 55-review 根因覆盖 — ✅

`.scratch/55-review/TICKET.md` §1 的 R1–R8，逐条处置可核：

| 根因 | 处置落点 | 独立核验 |
|---|---|---|
| R1 目的丢失 | schema 强制 `purpose`+`innovation`+`sota_baseline`+`gap` | 46/46 非空；check_tree 强制 `purpose/innovation` 非空 ✔ |
| R2 无 pareto 口径 | `validation` 强制「对照 + 判据 + 口径」 | check_tree §validation 三要素门全过 ✔ |
| R3 判据错位/恒真 | `objective_criteria`+`falsification` 强制非空 | `6.2` 明确写「**不得**把『R>1』当判据…（55-review H-F03）」；46/46 非空 ✔ |
| R4 造轮子 | `sota_baseline`+`gap` 声明增量 | 46/46 非空 ✔ |
| R5 结论悬空 | `status∈{已完成,被否证}⇒conclusion+存在的experiments` | check_tree 门；4 个 backfill stub 依「有证据者保留」并登记未决 ✔ |
| R6 上一时代残留 | 清 `*.md.md`、修 `mental_model` 悬空、删旧格引用、清死数据 | `find . -name '*.md.md'`=**0**；11 处 `mental_model` 全改指 `archive/legacy-docs/`；新树非归档旧轴引用=0 ✔ |
| R7 结构错 | MAPPING 拆/并/降参照 | **无任何 experiment 目录同时挂在 >1 研究项下**（扫 46 项的 experiments 去重后无重复目录名）；拆并点核：`1.1.3` 无 `experiments/`（纯指针→1.1.1）；`1.2.2`={2616-linear-attention,2637}、`1.2.3`={2614-topk}；`6.2`=11 目录（H1/H3/H4/H5）；`2.2.3`={2613-inter-layer-topology}、`2.2.4`=纯 README 指针；`1.1.4.2`={2614,2615} ✔ |
| R8 跨引擎/跨口径 | `cost_model` 禁墙钟；口径纪律沿用 SCHEMA §6–24 | 46/46 `cost_model` 无墙钟词；D5「净亏」→「代价轴 >1（范式定义性质）」已落地（`methodology.md:41,55`、`playbook.md:87,115` 已无「净亏」） ✔ |

### 判据 7 · 整合报告自证 — ✅（数字账实相符；报告本体入库为缺口）

抽查整合报告声称的**关键数字**，全部独立复现一致：

| 报告声称 | 我的独立测量 | 结论 |
|---|---|---|
| 旧证据 1562 文件 | 1562（旧轴 experiments 文件，排除自动索引） | ✔ 一致 |
| 1380 逐字节相同 / 182 <100 | blob 比对 `bacaaeb`：1380 / 182 | ✔ 一致 |
| `git ls-files axes` 1733→1648（−85） | `ls-tree bacaaeb axes`=1733、`HEAD`=1648 | ✔ 一致 |
| `find axes -type f` = 2425 | 2425 | ✔ 一致 |
| pytest 815 passed | 815 passed | ✔ 一致 |
| 827→815 差额=退役 12 用例 | 退役文件 `--collect-only`=12 | ✔ 一致 |
| `3c03d0b`=91 rename / `a79f207`=4 文件 / `3861ea5`=388 文件 | `git show --stat` 逐一 = 91 / 4 / 388 | ✔ 一致 |
| `.md.md` 83 个重命名 | `git diff -M ce2610a HEAD -- .scratch/doc-log` R=83 | ✔ 一致 |
| N1 改写后新树非归档旧轴引用=0 | 严格 `-P` 扫描=0 | ✔ 一致 |
| 11 份报告原样入库 | 仓库 11 份与 artifact 源 `diff -r` 逐字节一致（md5 全 SAME） | ✔ 一致 |

**未完全复现的 1 项（non-blocking）**：报告称 N1 改写「**324 文件 / 1474 处替换**」。我在其声明的覆盖范围（`axes/1-time…9-backlog` + `tools` + `docs` + `SCHEMA.md` + `pathlm`）内测得改写前（`bacaaeb`）旧轴路径 token 出现 **280 文件 / 1372 处**。差额可由「改写把部分引用改成 `archive/wiki-legacy/axes/<旧轴>/…`（其文本仍含 `axes/<旧轴>/`，被我的严格 `-P` 正则排除，但确实是被替换的路径）」解释 —— 属计数口径差异，非账实不符。

**两条与报告不符的观察**：
- **❌（non-blocking · 入库缺口）整合报告本体不在仓库**：`WP-M2.8-integrator.md` 只存在于 subagent artifact 目录；仓库 `.scratch/56-restructure/wp-reports/` 有 11 份（`wp1…wp7`+`rev-axis1/2/6/rest`），**缺整合报告自己**。整合报告 §8 声称「11 份 WP/复核报告已原样入库」——11 是对的，但它没把自己算进去。这不影响技术验收（同一事实已在仓库 `TICKET.md` §7 落盘 + artifact 可读），但违反「证据走目录下文档」的惯例、削弱审计链。
- **⚠️（低优先 · 遗留笔误）** `rev-axis6` **D1**（`wp4.md` 里 `1692=1692` 恒真自比，实际 `d6f099b`=1692 / `61e346b`=1701）**仍在** `wp4.md` §3；整合报告 §1 的 13 行处置表**未列 D1/D6**。D1 是历史 WP 报告里的数字笔误（报告按原样归档不改写），**不改任何结论**；只指出整合器「逐条处置」的说法对 D1/D6 略有遗漏。

### 判据 8 · 未记入 git 的规划台账（只报告建议）— ⚠️

`git status` 仅两项 untracked：`.scratch/54-subproject-schema/`（2 文件：`INVENTORY.md`/`TICKET.md`）、`.scratch/55-review/`（9 文件：`{A,C,F,G,H,M,W}-PROBLEMS.md` + `TICKET.md` + `PROPOSED-STRUCTURE.md`）。

**评估**：`.scratch/55-review/` 是本次迁移的**问题依据**——`MAPPING.md` 的每条「55-review 条目」（如 `A-P08`、`55-R07`）与 `TICKET.md` 的 R1–R8 处置都指向它，却**不在 git 里**。任何 clone 仓库的人无法核对这些引用。`.scratch/54-subproject-schema/` 同理。

**建议（仅报告，未提交）**：应把这两个目录用**显式路径** `git add .scratch/54-subproject-schema .scratch/55-review` 入库（不含任何其它路径），使迁移的证据链自洽、可追溯。整合器已如实声明「保留原状、未纳入提交」（TICKET §7.6-7 / 报告 §6-7），属**已知未决**，非隐瞒。

---

## 2 Material defects

**Blocking：无。**

**Non-blocking：**

1. **整合报告本体未入仓库**（审计可追溯性）
   - 位置：`.scratch/56-restructure/wp-reports/WP-M2.8-integrator.md`（仓库缺该文件；仅存于 `…/subagent-artifacts/outputs/884641c0…/`）。
   - 违反：`AGENTS.md` §3「所有信息沟通一律通过目录下的文档完成」+ 惯例（其余 11 份报告已入库）。
   - 复现：`ls .scratch/56-restructure/wp-reports/WP-M2.8-integrator.md` → No such file；`git log --all -- <path>` → 空。
2. **`rev-axis6` D1 笔误未在处置表列出**（低优先）
   - 位置：`.scratch/56-restructure/wp-reports/wp4.md` §3（第 34/78/173-175 行 `1692=1692`）。
   - 违反：无（不改结论）；仅指整合报告 §1「逐条处置」未覆盖 D1/D6。
   - 复现：`git ls-tree -r d6f099b --name-only axes | wc -l`=1692；`… 61e346b …`=1701。
3. **规划台账 untracked**（见判据 8）——`.scratch/54-subproject-schema/`、`.scratch/55-review/`。
   - 复现：`git status --short` → `?? .scratch/54-subproject-schema/`、`?? .scratch/55-review/`。

---

## 3 未复核范围（如实；不得臆测）

- 我**没有**逐个比对 1562 个文件的**每一个数字**是否未被 N1 改写波及；只对 1 个抽样文件（`2630-tf4-samecaliber/results.json`）做了逐行 diff，确认仅路径 token 变化。整合器声称用「路径 run→占位符 + 数字多重集比较」在 324 文件上验证过；我**独立复现了其方法学结论的样本，未复现全量**。
- 我**没有**审阅 `wp1…wp7`、`rev-*` 的**内容质量**（那是各轴复核的职责），只核验它们「原样入库」与其中引用的数字。
- 我**没有**对 `axes/**/project.yaml` 的**研究内容正确性**（结论是否成立、判据是否恰当）做实质评审——这超出整合复核范围，属各研究项的独立复核。
- 我**没有**核验非 axes 路径（如 `pathlm/`、`probe_*.py`、`train_*.py`）的功能行为；`chain_mode="latent"` 默认值（rev-axis6 D4）我只确认它**仍为 `latent` 且整合器按「只报告不擅改」处理**，未评判该默认值对错。
- 墙钟/GPU：我确认无 orphan pytest、GPU 占用者为无关 `llama-server`；**未**对任何 GPU 任务做性能或正确性核验。
- `archive/dropped/**` 与 `archive/legacy-docs/**` 的**内容**未逐文件核验（只确认引用可达）。

---

## 4 复现命令清单

**A. 证据零丢失（独立内容哈希）**

```
python3 /tmp/verify_hash3.py
# old files under old axes at ab1d8d8 = 1657
# exp files: 1562
# bacaaeb: exp byte-identical=1380 changed=182
# 3861ea5: exp byte-identical=1161 changed=401
python3 /tmp/verify_hash.py     # 6 个 not-exact 文件里 6/6 经 experiments 相对尾路径定位；STILL not located = 0
git diff --name-status -M ab1d8d8 bacaaeb -- axes/ | awk '$1=="R100"' | wc -l    # 1384
# 逐个 R100 pair blob 比对：mismatches = 0
find archive/wiki-legacy -type f | wc -l        # 91
find archive/wiki-legacy/axes -type f | wc -l   # 85
```

**B. 退役一致性**

```
git ls-tree --name-only HEAD:axes   # 1-time … 9-backlog
git grep -cP "(?<!wiki-legacy/)(?<!dropped/)axes/(A-mixing|C-vertical|H-horizon|W-width|M-carrier|F-iteration|G-policy|P-modifiers|R-references)/" HEAD -- axes/1-time … axes/9-backlog tools/docs SCHEMA.md pathlm   # 0
```

**C. 用词/结构**

```
python3 /tmp/parse_struct12b.py
# matched project.yaml: 46 / 46 ; mismatches: 0 ; project.yaml ids not in §12: []
```

**D. 注入（在 /tmp 克隆里）**

```
rm -rf /tmp/m2inject && git clone -q /home/z/vibe/pathlm /tmp/m2inject
python3 /tmp/inject_test.py    # 5 注入全 PASS（rc=1 + 精准报错），控制组 OK
rm -rf /home/z/vibe/pathlm/archive/wiki-legacy/{tests,tools}/__pycache__   # 清掉 --collect-only 副产物
```

**E. 门**

```
python3 tools/summary.py --check ; python3 tools/check_tree.py ; python3 tools/check_schema.py ; python3 -m pytest tests/ -q
pgrep -af pytest ; nvidia-smi
```

---

## 5 判定

- 判据 1 ✅ · 2 ✅ · 3 ✅ · 4 ✅ · 5 ✅ · 6 ✅ · 7 ✅（含 1 条入库缺口 non-blocking）· 8 ✅（评估 + 建议）
- **Blocking defect：无。**
- **总结论：PASS。**

**建议下一步**：把 `WP-M2.8-integrator.md` 与 `.scratch/54-subproject-schema/`、`.scratch/55-review/` 以显式路径入库（一次性补提交），使迁移证据链在仓库内自洽；随后进入 M3（role_map 实名 / 轴 lead 指派）。
