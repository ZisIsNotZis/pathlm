# ticket 56 · M2 · 轴 6（预测轴，WP-M2.4）独立复核

> 复核者：fresh 上下文，未参与本 WP（writer 为轴 6 §6.2/§6.4 worker）｜只读，未改仓库
> 复核对象：`axes/6-predict/**` 工作树 + 提交 `61e346b`
> 基线 revision：`git rev-parse HEAD` = `bacaaeba6464f75b24fd28e30bca621e18527462`
> `git status --short`（复核时）：
> ```
>  M INDEX.md
>  M axes/C-vertical/C1-layers-per-token/README.md
>  M axes/C-vertical/README.md
>  M catalog.yaml
> ?? .scratch/54-subproject-schema/
> ?? .scratch/55-review/
> ```
> （这 4 个 modified + 2 个 untracked 均为**迁移前既有**、与轴 6 无关；轴 6 的改动已全部落在 `61e346b`。）
> 被复核 WP 报告：`.scratch/56-restructure/wp-reports/wp4.md`（轴 6，WP4）
> 依赖文档：`docs/structure.md` §6/§12/§13/§14、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`、`.scratch/55-review/H-PROBLEMS.md`
> 记录时间：2026-10-13T12:2x+08:00（`date -Iseconds`）

---

## 0. 人话摘要（先结论）

**结论：PASS（无 blocking defect；5 条非阻塞发现，其中 4 条 writer 已自报、1 条为报告数字笔误）。**

轴 6 这次迁移的实质是：把旧「视界轴（H）」那 5 个**手段型**格子，按 `docs/structure.md` §12 冻结布局
重挂成预测轴的 6 个节点——核心研究项 `6.2 多位置预测 = 投机解码`（H1 本体 + H3/H4/H5 三个工作包）、
`6.4 训练时未来位置的填充语义`（H2）、以及旧仓完全没有的三条新路线 `6.3.1/6.3.2/6.3.3`，
`6.1 自回归` 降为参照登记进轴 7 台账。

我按「这条迁移做砸了」为前提去证伪，逐条查了：节点是否齐全且用词逐字一致；每份 `project.yaml`
字段是否齐全（role_map / 单 milestone / 每条 WP 的 acceptance + 独立复核 / 八个字段无墙钟）；
证据是否整目录搬齐、有没有丢或重复；55-review 的 H 轴问题是否逐条处置；并跑了校验器与
「坏 YAML 注入」证伪。**结果是：证据一条没丢（旧 H 轴 124 个跟踪文件 = 12 个留下的 README + 112 个
100% 相似度改名），校验器全绿，四类坏 YAML（缺 role_map / 缺 failure_analysis / 墙钟 cost_model /
被否证但归因为空）全被拦下，55-review 的 5 个致命项（F01–F05）在正文里都有对应修正。**

唯一的独立证据缺口：`pytest tests/ -q` 本轮**没能跑完**（机器 load ≈ 28 / 12 核，并存多份 agent 任务，
跑到约 72% 仍在前行、**过程中未见任何 F/E 失败**，但未拿到最终计数）。WP 自己报的是
`827 passed in 93.36s`，我**无法在本机独立复现该数字**，如实标为「未复核完成」。

---

## 1. 判据逐条结论

| # | 判据（契约） | 结论 | 复现 / 证据 |
|---|---|---|---|
| 1 | 节点齐全且 exact wording（`docs/structure.md` §12） | ✅ | `6.2`/`6.3`（分组 README）/`6.3.1`/`6.3.2`/`6.3.3`/`6.4` 全部落位；5 份 `project.yaml` 的 `title`/`dir` 与 §12 逐字一致（脚本比对见 §2.1）；`6.1 自回归（一次一个位置）` 按 §13.4 登记在 `axes/7-references/README.md:27`（与轴 1 的 `1.3` 同做法，不建目录） |
| 2 | 每份 `project.yaml` 字段齐全（`docs/research-schema.md` §2） | ✅ | 5 份均含全部必填字段；`role_map` 四角色齐且 `reviewer != writer`；`milestone` 为**单个 map** 且含 `output`+`acceptance`；`failure_analysis` 键均在（4 份 `""`）。见 §2.2 |
| 2a | `work_packages` 每条含 `acceptance` + **独立** `review` | ✅ | 5 份共 9 条 WP，`missing=[]`（id/what/goal/means/acceptance/evidence/review/status 全齐）；每条 `review` 都写明「跑完另派 fresh 复核（SCHEMA.md §14/§14b）」，与 writer 分离 |
| 2b | `sota_baseline`/`gap`/`method`/`validation`/`objective_criteria`/`falsification`/`cost_model` 无墙钟 | ✅ | 5 份 `cost_model` 全为等效理论量（FLOPs / layer-passes / tokens-per-forward + pipelining 说明），无墙钟/无日期；`check_tree.py` 的 `WALLCLOCK_RE` 未命中（见 §3） |
| 3 | 证据按 MAPPING 迁移、无丢失（迁前/迁后计数，含 git 跟踪数） | ✅ | 旧 `H-horizon` 跟踪文件 **124 → 12**；提交 `61e346b` 中 **112 个 100% 相似度 rename、0 删除**，124 = 12 + 112 精确闭合；`axes` 跟踪数 **1692 → 1701**（+9 新文件）；工作树 `find axes -type f` = **2510 ≥ 2434**（MAPPING §0.1 基线）。MAPPING §3「一格一份家、不复制」成立（旧格目录已不残留副本）。见 §2.3 |
| 4 | `.scratch/55-review/H-PROBLEMS.md` 逐条处置 | ✅（5 个致命项全处置；次要项见 §4） | F01→purpose 改回目的；F03→`objective_criteria` 明文「**不得**把 R>1 当判据」；F04→`method`/WP6.2.1 冻结主干只训头；F02/F05→重立 WP `6.2.2 概率多候选/树`（含出资人路径概率加权式）+ 预登记 cross-K 判据；I1/I2/I3/I8→`gaps`+`validation` 墙钟不进判据；I7→`6.4` 单独立项写明目的；S2/S4/S5→`gaps`/`conclusion` 显式记录。见 §2.4 |
| 5 | `python3 tools/summary.py --check` | ✅ | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） |
| 5 | `python3 tools/check_tree.py` | ✅ | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） |
| 5 | `python3 -m pytest tests/ -q` | ⚠️ 未跑完（非 FAIL） | 见图，跑到约 72% 未见 F/E；因机器高负载（load ≈ 28）耗时异常，按编排指令停止轮询、如实标注 |
| 6 | `/tmp` 注入坏 YAML 证伪（4 类） | ✅ 4/4 被拦 | 见 §3：缺 role_map / 缺 failure_analysis / 墙钟 cost_model / `status=被否证` 且归因为空 —— 全部 `FAIL`（exit 1） |

---

## 2. 详细证据

### 2.1 节点齐全 + exact wording

脚本从 `docs/structure.md` §12 轴 6 表抽出 7 行，与实现逐字比对：

```
6.1   自回归（一次一个位置）              → 参照（登记 7-references/README.md:27，不建目录）
6.2   多位置预测 = 投机解码               → axes/6-predict/6.2-speculative-decoding/project.yaml  title 逐字一致
6.3   整段并行生成（不止 AR）             → axes/6-predict/6.3-parallel-generation/README.md（分组）
6.3.1 diffusion（离散扩散 / 掩码扩散）    → title 逐字一致
6.3.2 一次性批量 pad 填充（填空式）        → title 逐字一致
6.3.3 latent regression 批量生成（估计得贪心）→ title 逐字一致
6.4   训练时未来位置的填充语义：pad / self-fill / 错误 token → title 逐字一致
```

- 轴名 `axis: "6 预测轴"` 与 §6/§12 一致；目录名与 §12/§14 一致（§12 表内末尾带 `/`，`dir` 字段不带，属排版差异，非不符）。
- `6.2` 的 work_packages 用词与 §12 注一致：`6.2.1 草稿头（应免费、不动主干）`、`6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合）`、`6.2.3 验证`（`what` 字段逐字）。
- 复现：`python3 tools/summary.py` 输出 6.x 五行（6.2 进行中 / 6.3.1–6.3.3 未开始 / 6.4 已完成）。

### 2.2 字段齐全（脚本抽取）

```
6.2   role_map 4 角色 ✅  reviewer≠writer ✅  milestone=dict{output,acceptance} ✅
      WP 6.2.1 / 6.2.2 / 6.2.3  missing=[]  review="跑完另派 fresh 复核…"
6.3.1 role_map 4 ✅ reviewer≠writer ✅ milestone=dict ✅  WP1 missing=[] review=fresh ✅
6.3.2 同上（单 WP1）
6.3.3 同上（单 WP1）
6.4   role_map 4 ✅ reviewer≠writer ✅ milestone=dict ✅  WP1/WP2 missing=[] review=fresh ✅
```

`experiments` 列表：`6.2` 列 10 个仓内实验目录 + 7 条 `.scratch` 证据锚；`6.4` 列 1 个目录 + 3 条锚；
`6.3.1/6.3.2/6.3.3` 为空列表（无旧格，合法）。所有 `experiments` 路径经 `check_tree.py` 校「存在」、
所有 `@<sha>` 锚点经校「可达真实提交」——均通过（exit 0）。

### 2.3 证据迁移与计数

| 计数 | 值 | 命令 |
|---|---|---|
| 轴 6 迁移前 `axes` 跟踪文件 | 1692 | `git ls-tree -r d6f099b --name-only axes \| wc -l` |
| 轴 6 迁移后 `axes` 跟踪文件 | 1701 | `git ls-tree -r 61e346b --name-only axes \| wc -l` |
| 提交 `61e346b` 变更明细 | 112 R100 + 9 A + 5 M + **0 D** | `git show --name-status --find-renames=90% 61e346b` |
| 旧 `axes/H-horizon` 跟踪文件（前/后） | 124 / 12 | `git ls-tree -r d6f099b --name-only axes/H-horizon \| wc -l` 等 |
| 124 = 12（留旧格 README/PRIOR_ART）+ 112（整目录改名） | 精确闭合 | 上两行 |
| 工作树 `find axes -type f` | 2510（≥ MAPPING 基线 2434） | `find axes -type f \| wc -l` |
| 6.2/6.4 experiments 目录内文件 | 153（112 跟踪 + 41 gitignore 产物如 `*.pt`/`*.log`） | `find … -type f \| wc -l` |

- 11 个旧格实验目录（H1×2、H3×4、H4×3、H5×1 → `6.2`；H2×1 → `6.4`）**整目录改名**，与 MAPPING §1 逐条对上。
- gitignore 产物也随目录搬到位（新家可见 `trunk.pt`/`run.log` 等），旧家已空 → 无孤儿、无重复复制。
- 修改的 5 个非改名文件均为**指针同步**（`tests/test_spec_compute_ledger.py`、`tests/test_h3_spec_three_arms_ledger.py`
  的路径常量；`train_jumphead.py` 的 `contract` 字符串；`docs/methodology.md` 的证据锚行；
  `axes/7-references/README.md` 登记 `6.1`）。**证据内容一字未改**（112 条全 R100）。

### 2.4 55-review（H 轴）逐条处置

| 条目 | 处置落点 | 结论 |
|---|---|---|
| F01 目的丢失（手段当 topic） | `6.2.purpose` 写痛点为「每生成 1 token 跑 1 次完整前向」；README「topic 不是一次前向输出几个位置」 | ✅ |
| F03 主判据 R>1 恒真 | `6.2.objective_criteria` 明文「**不得**把 R>1 当判据——τ≤γ ⇒ R≥1 按构造恒成立、不可证伪」 | ✅ |
| F04「MTP 免费」被付费基线顶掉 | `6.2.method` 臂含冻结主干只训头；WP 6.2.1 写死；`conclusion(a)` 用 `Δbpc=0.0000` 坐实 | ✅ |
| F02/F05 H3 简化成单 argmax、树整条没做 | 重立 WP `6.2.2`（含出资人路径概率加权式 + 树/DP lattice）+ `acceptance` 要求完整树验证给 cross-K 定论 | ✅ |
| I1 τ 无稳健性 | `6.2.gaps` 列（温度>0 / batch>1 / 低局部先验） | ✅ |
| I2 无 pareto frontier | `6.2.gaps` + `roadmap` 末条（沿 γ/头规模/树宽度扫） | ✅ |
| I3/I8 墙钟不可复现却进头条 | `validation`「墙钟不进判据」；`conclusion(f)` 剔除「税效比最好」 | ✅ |
| I4 combined≈direct 是结构性预期 | `6.2.argument` 指出固定 λ=0.5 等权混合注定退化；WP 6.2.2 要求真概率加权 | ✅ |
| I5 把已知错方案列成选项 | 新树未再列「朴素概率乘积」；旧 `catalog.yaml` 退役 | ✅（内容层面） |
| I6 H4/H5 是实现选择/正确性不变量 | H4→WP 6.2.1、H5→WP 6.2.3，**不再各立研究项** | ✅ |
| I7 H2 目的不清、归属错 | `6.4` 单独立项，purpose 写明服务变深度/重入 AR 与多位置预测 | ✅ |
| S2 2510 权重已丢 | `6.2.gaps` 记「B2 权重已不可得…只能引用原件」 | ✅ |
| S4 @100M 张冠李戴 | `6.2.gaps`「jump head 的 @100M 零税未测（H-S4 的张冠李戴已拆开）」 | ✅ |
| S5「整段扩散」越界 | `6.2.gaps`「已移出本项、归 6.3」 | ✅ |
| S1 `chain_mode="latent"` 默认 | 属 `pathlm/config.py`（共享引擎），WP **未改**，报告 §4 自报为残留 | ⚠️ 残留（见 §4-D4） |
| S3/R-4 旧 schema stub 目录 | 4 个 stub 随迁入新家，front-matter 仍旧 schema，WP 自报为残留 | ⚠️ 残留（见 §4-D2） |
| S7/R-7 playbook 定律 8 变更史 | 未改；WP 自报为残留（交 integrator） | ⚠️ 残留（见 §4-D5） |
| S6 跨口径阈值 0.014 / S8 metrics 稀释 | 未在轴 6 正文显式回应；`catalog.yaml` 退役、`0.014` 作为 bpc 噪声带沿用 | ⚠️ 低（见 §4-D6） |

---

## 3. 校验器真实输出 + 坏 YAML 证伪

**工作树（`bacaaeb`，只读）**

```
$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。          # exit 0
$ python3 tools/check_tree.py
OK：46 个 project.yaml 全部符合 docs/research-schema.md。          # exit 0
```

**`/tmp` 注入证伪**（`git clone /home/z/vibe/pathlm /tmp/rev-axis6-repo`，在克隆内改 `6.3.1/project.yaml`，仓库本身未动）

| # | 注入 | `check_tree.py` 输出 | 期望 | 结论 |
|---|---|---|---|---|
| A | 删 `role_map` | `FAIL：2 项 / 缺字段 role_map / role_map 必须是 map` exit 1 | FAIL | ✅ 拦下 |
| B | 删 `failure_analysis` | `FAIL：1 项 / 缺字段 failure_analysis` exit 1 | FAIL | ✅ 拦下 |
| C | `cost_model` 写成「每次去噪 3.2 秒（wall clock），比 AR 快 1.8×」 | `FAIL：1 项 / cost_model 含墙钟/日期指标 '秒'` exit 1 | FAIL | ✅ 拦下 |
| D | `status: "被否证"` 且 `failure_analysis: ""` | `FAIL：3 项 / 被否证但无 conclusion / 无 experiments / failure_analysis 为空` exit 1 | FAIL | ✅ 拦下 |

恢复后克隆内 `check_tree.py` = `OK：46 个 project.yaml…`（exit 0）——证明是注入导致 FAIL，非环境噪声。
（校验器**未**强制 `reviewer != writer`，属校验器覆盖缺口；轴 6 的 5 份 `role_map` 本身合规。）

**`pytest`（未跑完）**

```
$ python3 -m pytest tests/ -q
……（progress dots）…… [ 69%] ……（42+ dots）……   # 跑到约 72% 仍在前进
# 未出现任何 F/E（失败/错误）标记；因机器 load≈28/12 核、并发多任务，耗时远超 WP 报告的 93s，
# 按编排指令停止长轮询。WP 报告数字：827 passed in 93.36s（未能本机复现）。
```

---

## 4. Material defects（位置 + 违反判据 + 复现；**只报告不修复**）

> 未发现 **blocking** defect。以下 6 条全部为 **non-blocking**；D1 为报告笔误，D2–D5 writer 已自报，
> D6 为低优先一致性。

**D1（non-blocking · 报告数字笔误）** — `wp4.md` §3 表末行声称
`git ls-tree -r HEAD~ --name-only axes | wc -l` = `git ls-files axes | wc -l` → **1692 = 1692**。
实际：`HEAD~`（`d6f099b`）= 1692，`HEAD`（`61e346b`）= **1701**；两者相等是拿同一 revision 自比，属恒真比较。
- 违反判据：判据 3 的「含 git 跟踪数」证据链要求可核。
- 复现：`git ls-tree -r d6f099b --name-only axes | wc -l` = 1692；`git ls-tree -r 61e346b --name-only axes | wc -l` = 1701。
- 影响：**结论不变**（1692 → 1701，未减少、0 删除），仅报告数字写错。

**D2（non-blocking · 残留未清）** — 4 个 backfill 生成的旧 schema stub 随迁入 `axes/6-predict/**/experiments/`，
front-matter 仍是旧 schema（`name/ticket/date/grid/engine/budget`），并含已废旧轴 id（`H3-composition`/`H5-verification`/
`F2-return-channel`/`A3-window-anchor`）：
`axes/6-predict/6.2-speculative-decoding/experiments/{2512-overnight-causal-chain,2513-four-experiments,2521-exit-scale-confirm}/README.md`、
`axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants/README.md`。
- 违反判据：ticket 56 §2.4 + §3.5「残留清单（含『旧 schema stub 实验目录』）逐条消失」；55-review R-4/S3。
- 复现：`head -8 axes/6-predict/6.2-speculative-decoding/experiments/2512-overnight-causal-chain/README.md`（末行「本条目由 tools/backfill_legacy.py 生成」）。
- writer 已在 `wp4.md` §4.3 自报为残留（交文档 WP / integrator）。

**D3（non-blocking · 状态自洽）** — `axes/6-predict/6.4-training-fill/project.yaml` 顶层 `status: "已完成"`，
但其 `work_packages` WP1 = `进行中`、WP2 = `未开始`，且 `milestone.output` 要求「13M matched-step（**n≥2**）」，
而 `gaps` 自述「填充语义对照**无 n≥2 复本**」。
- 违反判据：`docs/research-schema.md` §7（`status` 只回答「做完没有」；「做到多少」进 `conclusion`/`gaps`）——
  顶层「已完成」与其自有 WP 状态、milestone 输出不一致，属可能的高报。
- 复现：`python3 -c "import yaml;d=yaml.safe_load(open('axes/6-predict/6.4-training-fill/project.yaml'));print(d['status'],[w['status'] for w in d['work_packages']],d['gaps'][0])"`
- writer 已在 `wp4.md` 的 `residualRisks` 自报；建议 integrator 改判 `进行中` 或补 n≥2。

**D4（non-blocking · 共享引擎残留）** — `pathlm/config.py` 的 `chain_mode` 默认值仍为已被否证的 `"latent"`（旧 `T∘T`）。
- 违反判据：55-review S1/R-1；不属轴 6 `project.yaml` 字段判据。
- 复现：`grep -n "chain_mode" pathlm/config.py`。
- writer 自报（`wp4.md` §4.3 R-1）——共享引擎代码，本迁移 WP 明确不动。

**D5（non-blocking · 事实文档口径不一致）** — `docs/methodology.md`（本 WP 曾改其证据锚行）仍写
spec FLOPs 口径「**两规模都 >1，净亏**」，而同一迁移把 `6.2.objective_criteria` 改成「R>1 是范式定义性质、
**不得**当判据」；`docs/playbook.md` 定律 8/11 亦写「两规模都净亏」。
- 违反判据：`AGENTS.md` §2.1 一事实一家（同一事实两处表述冲突须收敛）。
- 复现：`grep -n "净亏" docs/methodology.md docs/playbook.md`（methodology:40-55、playbook:84-85/115）。
- 说明：playbook 项已被 writer 列为残留 R-7；**methodology 项 writer 未列入其残留清单**（低优先措辞问题：
  「净亏」作货币陈述未必算「当判据」，但两处措辞已分叉）。

**D6（non-blocking · 低优先）** — S6（`0.014` 阈值的跨口径身份）与 S8（`catalog.yaml` H1 metrics 稀释）
在轴 6 正文无显式处置记录；`catalog.yaml` 退役、`0.014` 作 bpc 噪声带沿用。
- 违反判据：判据 4「逐条处置」的记录完整性（非内容错误）。
- 复现：`grep -rn "S6\|S8" axes/6-predict/` 无命中。

---

## 5. 未复核范围（明确边界）

1. **未独立复现底层实验数字**：只抽检了三处与结论强相关的读数——
   `2610/results.json`（`zero_tax.delta_bpc=0.0`、`frozen_assertion_ok=true` ⇒ 6.2 结论(a) 一致）、
   `2520/results.json`（`per_token_compute_ratio__analytic` 1.09/1.2675 ⇒ 6.2 结论(c) 一致）、
   `2631/results.json`（`combined_lam=0.5`、`chain_mode=token` ⇒ WP 6.2.2 立论一致）。
   其余实验读数（τ 1.58/1.54/1.20、219-bpc 等）**未逐一复算**，也未重跑任何训练/评测。
2. **`pytest tests/ -q` 未跑完**（见 §3），未取得最终 pass/fail 计数；因此「全套测试无回归」这一条
   **未获独立确认**（仅观察到 ~72% 无 F/E）。这是本次复核最大的证据缺口。
3. 未复核 `docs/structure.md` §12 其它轴、`SCHEMA.md`/`docs/playbook.md` 全文口径、`catalog.yaml` 内容。
4. 未验证 `experiments/**` 内**证据文件的旧路径内部引用**（42 个文本文件仍含 `H-horizon` 等旧轴名，
   如 `2631/results.json` 的 `checkpoints.13m_s0.path`）——这是「证据不改」的必然残留，姊妹报告 wp6 已提，
   **不影响 pytest**（测试按路径常量读取、已更新），但会让旧脚本手动重跑失效；是否重写由 integrator 定。
5. 未评价 6.3.x 三条新路线设计的学术价值（只校字段/判据/口径是否合规），未做外部 SOTA 核实
   （证据内一律标 `记忆·未核`，符合 PRIOR_ART 纪律）。

---

## 6. 一句话

轴 6 的迁移**结构、用词、字段、证据归属、55-review 致命项处置**都站得住，校验器与坏 YAML 证伪都过关；
未发现的只是报告数字笔误、4 个 stub 旧 schema 残留、`6.4` 顶层状态自洽、`methodology` 措辞分叉与
共享引擎 `chain_mode` 等**非阻塞**项，且多数 writer 已自报——**判 PASS**；
唯一保留意见是 `pytest` 全套未能在本机独立跑完，建议 integrator 在收口时于空载窗口补跑一次。
