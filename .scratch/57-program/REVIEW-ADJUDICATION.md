# ticket 57 · 框架性取舍落地（ADJUDICATION-LANDING）· 独立复核报告

> 复核者：`review-adjudication`（fresh 独立上下文、只读；仅用 `/tmp` 副本做注入实验）
> 复核时间：`2026-10-14T06:49:22+08:00`（`date -Iseconds`）
> 对象修订：**HEAD = `ea8b5c2fa60ccbbfefd7b2ff3584563e60952440`**（`git status --porcelain` 空、工作树干净）
> 对象提交：`e98a26c`（落地 21 条裁决，18 文件）+ `ea8b5c2`（C3 引擎通道，1 文件）
> 落地报告：`.scratch/57-program/ADJUDICATION-LANDING.md` —— **仓库内不存在**；实际存在于 subagent 产物目录
> `…/subagent-artifacts/outputs/b87ec038-…/.scratch/57-program/ADJUDICATION-LANDING.md`（本轮据该副本复核，见「未覆盖」）。
> 台账：`.scratch/57-program/FRAMEWORK-DECISIONS.md`（§1 清单 + §6 裁决落地）
> 契约：`docs/research-schema.md`（§5 / §11.1 / §12.1–§12.3）、`docs/methodology.md`（§2）、`docs/structure.md`（§12/§13.6）
> 结论：**PASS**（无 blocker；3 条非阻断发现 + 1 条留痕观察，见 §4）

---

## 0 人话摘要

出资人把六轴报上来的 21 条框架性取舍一次性裁了（「按建议/默认处置推进」）。这一轮的工作是**把裁决写进
共同规矩（SSOT）和受影响的档案，并把台账标成「已裁决」**。我独立地逐条查了 21 条的落地位置、亲自跑了
三道门、并在 `/tmp` 副本里故意注入坏 YAML 证明门是真的会拒。

**结果：21 条全部有可核实的落地落点，门全绿且真拦坏输入，测量数字一个没改、实现代码零改动、证据零删除，
MoE 的处置（降为 Backlog + 业界跑分线）自洽，C3 引擎通道三项均标「未授权」。**

只有三处**不阻断落地**的小账差，最要紧的一条是：新写入的 C1 纪律说「**每个**研究项的 `falsification`
都必须声明用哪个带，未声明者判据不可判」——但全树 46 项里只有 **8 项**补了声明，另有 **26 项**在
`falsification` 里引用了噪声带却没声明（其中 8 项是「进行中/已完成」，含**已完成**的 `3.2.1`）。
这属于**新规矩盖过了旧文件**的一致性问题，落地报告已把「未逐一补」列为残留，但把它描述成「只有未开始的
没补」并不准确。修起来是机械回填，不影响已落地的裁决方向。

---

## 1 逐条落地核查（独立枚举，不照抄报告）

枚举口径：`FRAMEWORK-DECISIONS.md` §1 速览表 = 跨轴共性 **5 条（C1–C5）** + 各轴独有 **16 条**
（轴 1×3、轴 2×3、轴 3×2、轴 4×2、轴 5×4、轴 6×2）= **21 条**（与报告一致，独立数过）。

### 1.1 跨轴共性（C1–C5）

| # | 裁决内容 | 落点（文件 : 字段） | ✅/❌ | 复现 grep |
|---|---|---|---|---|
| **C1** | 0.014 仅限 bpc、仅「配对 Δ、≥5 窗口」；概率/接受率类单独定门（prompt 聚类 bootstrap / Wilson 95% CI 不含 0 为主判）；**本格带优先**；`falsification` 声明「用哪个带」 | `docs/research-schema.md` §12.1；`docs/methodology.md` §2；`project.yaml × 8` 的 `falsification`+`validation` | ✅ | `grep -n "12.1 判据带与可排序门" docs/research-schema.md`；`grep -n "0.014 的适用纪律（跨轴共性裁决 C1）" docs/methodology.md`；`grep -rln "用哪个带（C1" axes/` → 恰好 8 个文件 |
| **C2** | 实质验收 =「不被同算力朴素方案支配」（超出量 ≤ 0）；保留预登记文字；`conclusion` 并列写「字面成立 / 实质成立或否证」 | `docs/research-schema.md` §5；`project.yaml × 8` 的 `milestone.acceptance` | ✅（见 §4-F1 字段落点差） | `grep -n "验收口径（跨轴共性裁决 C2）" docs/research-schema.md`；`grep -rln "实质验收（C2 裁决）" axes/` → 8 个文件 |
| **C3** | 授权建受控引擎通道（本轮不改代码） | `.scratch/67-engine-changes/TICKET.md`（新建）；`docs/research-schema.md` §12.3 | ✅ | `ls -l .scratch/67-engine-changes/TICKET.md`；`grep -n "12.3 共享引擎改动通道" docs/research-schema.md` |
| **C4** | 旧 wiki/catalog 退役「追认」 | `FRAMEWORK-DECISIONS.md` §6 标记；`docs/structure.md` §15 / `AGENTS.md` §2.3 保持现状 | ✅ | `sed -n '/## 6 裁决落地/,$p' .scratch/57-program/FRAMEWORK-DECISIONS.md \| grep -n "C4"` |
| **C5** | `conclusion_review` 4 条历轮判词保持原样 | `docs/research-schema.md` §11.1（含「既有已累积多轮的条目保持原样、不回填」）；4 份 `project.yaml` 的 `conclusion_review` 未被两提交触碰 | ✅ | `sed -n '/### 11.1/,/### 11.2/p' docs/research-schema.md`；`git diff 39f6051..HEAD -- <4 文件> \| grep -E '^[+-]conclusion_review:'`（应为空） |

**C1 特别核（4 个要点逐项）**：
- 「只用 bpc」+「配对 + ≥5 窗口」：`docs/research-schema.md:210-219` 明文「0.014 仅限 bpc 口径，且仅在「配对 Δ、≥5 窗口」条件下作参考门槛」✅
- 「概率/接受率用 CI」：同节「按 prompt 聚类的 bootstrap / Wilson 95% CI 不含 0 为主判」+ `docs/methodology.md` §2 同义 ✅
- 「本格带优先」：同节「单格自测带优先于跨格 0.014；两带不得混用」；实例：`1.1.4.1`（σ_cell＝0.00166）、`2.1.1`（逐窗口 SE ≤0.008）、`5.1`（在力带 0.036324）、`3.1.4`（0.014 限配对 Δ、≥5 窗口）✅
- 「falsification 声明用哪个带」：8 个受影响项的 `falsification` 均含「用哪个带（C1 裁决）…」✅（但**全树覆盖不全**，见 §4-F2）

**C2 特别核**：「实质验收 = 不被同算力朴素方案支配」逐项措辞落地——`1.1.4.1`「相对同算力纯窗口前沿超出量 ≤ 0」、
`2.1.1`「相对同算力固定截断前沿 ≤ 0」、`3.1.4`「相对同算力位点前沿 ≤ 0」、`4.x`「相对同激活算力粗粒度 MoE / dense ≤ 0」、
`5.1`「相对同字节头共享臂 ≤ 0」、`6.2`「相对同算力最强单臂 ≤ 0」✅。
「conclusion 并列字面/实质」：`2.1.1` / `3.1.4` / `5.1` / `6.2` 落在 `conclusion` ✅；`1.1.4.1` 落在
**`failure_analysis`** 而非 `conclusion`（⚠️ §4-F1）；`4.1/4.2/4.3` 无判词（`status=未开始`，无可并列）—。

**C5 特别核**：`1.1.4.1` / `2.1.1` / `5.1` / `6.2` 的 `conclusion_review` 在两提交前后**逐字节未变**
（`git diff … | grep '^[+-]conclusion_review:'` 为空）✅。

### 1.2 各轴独有（16 条）

| # | 裁决内容 | 落点（文件 : 字段） | ✅/❌ | 复现 grep |
|---|---|---|---|---|
| 62-F2 | 代价主轴 = 每位置等效 KV 字节，FLOPs 副轴 | `axes/1-time/1.1.4.1-explicit-memory/project.yaml` : `method` | ✅ | `grep -n "62-F2" …/1.1.4.1-explicit-memory/project.yaml`（:12） |
| 62-F3 | 只报前沿位置与支配关系，不把「形态税」写成结论 | 同上 : `gaps` | ✅ | `grep -n "62-F3" …/1.1.4.1-explicit-memory/project.yaml`（:61） |
| 62-F4 | 保留 milestone 文字；`gaps` 记「业界实现对齐未做」 | 同上 : `gaps`（现状已具备，仅标记） | ✅ | `grep -n "业界实现.*对齐未做" …/1.1.4.1-explicit-memory/project.yaml`（:56） |
| 61-F1 | 「序列级 perplexity = teacher-forcing 在线决策整段 bpc」 | `axes/2-depth/2.1.1-early-exit/project.yaml` : `validation` | ✅ | `grep -n "61-F1" …/2.1.1-early-exit/project.yaml`（:13） |
| 61-F3 | 知悉排期（无字段改） | `FRAMEWORK-DECISIONS.md` §6 标记 | ✅ | `sed -n '/## 6 裁决落地/,$p' …/FRAMEWORK-DECISIONS.md \| grep -n "61-F3"` |
| 61-F4 | 两格维持 Backlog、`gaps` 互指、**不合并** | `axes/2-depth/2.1.3-repeating/project.yaml` : `gaps`；`axes/2-depth/2.2.5-weight-sharing/project.yaml` : `gaps` | ✅ | `grep -n "61-F4" axes/2-depth/2.1.3-repeating/project.yaml axes/2-depth/2.2.5-weight-sharing/project.yaml`（:25 各一处） |
| 64-F1 | 标准对照 = 等 FFN 预算 / 等算力位点 | `docs/research-schema.md` §12.2；`3.1.1` : `method`；`3.1.4` : `validation` | ✅ | `grep -n "12.2 剪枝类实验的标准对照口径" docs/research-schema.md`；`grep -n "64-F1" axes/3-width/3.1.1-uniform-narrowing/project.yaml axes/3-width/3.1.4-ffn-ratio/project.yaml` |
| 64-F3 | 接受部分臂 n=1，按既定递减 | `3.1.1` : `validation`；`3.1.4` : `validation` | ✅ | `grep -n "64-F3" axes/3-width/3.1.1-uniform-narrowing/project.yaml axes/3-width/3.1.4-ffn-ratio/project.yaml` |
| **65-F1** | MoE 判决降级为 Backlog/参照（按主 Agent 裁决以 Backlog 落地 + 业界跑分线） | `docs/structure.md` §12 轴 4 表（粒度→Backlog）+ §13.6 注；`axes/7-references/README.md` 新增行；`4.1/4.2/4.3` 的 `status`→未开始 + `gaps` | ✅ | 见 §2 |
| 65-F2 | 记录 GPU 协调事项 | `axes/4-moe/4.1-sparse-experts/project.yaml` : `gaps` | ✅ | `grep -n "65-F2" axes/4-moe/4.1-sparse-experts/project.yaml`（:28） |
| 66-F1 | 用本格在力带 0.036324，两带关系写清、不得混用 | `axes/5-quant/5.1-kv-quant/project.yaml` : `validation`/`falsification` | ✅ | `grep -n "66-F1" axes/5-quant/5.1-kv-quant/project.yaml`（:15） |
| 66-F2 | 头族优势记为「参数混杂的可见后果」，不算机制发现 | 同上 : `gaps` | ✅ | `grep -n "66-F2" axes/5-quant/5.1-kv-quant/project.yaml`（:39） |
| 66-F3 | `5.1` 保持 `已完成`（无改，仅标记） | 同上 : `status` | ✅ | `grep -n '^status: "已完成"' axes/5-quant/5.1-kv-quant/project.yaml`（:52）；`git diff` 显示该行未变 |
| 66-F4 | 等参数臂 = `5.1` 与 `5.4` 共用 WP，登记归属 | `5.1` : `work_packages`（新增 `WP3`）+ `gaps`/`roadmap`；`5.4` : `gaps`/`roadmap` | ✅ | `grep -n "66-F4" axes/5-quant/5.1-kv-quant/project.yaml axes/5-quant/5.4-low-rank/project.yaml`；`grep -n 'id: "WP3"' axes/5-quant/5.1-kv-quant/project.yaml` |
| 63-F2 | 多候选结论附等 FLOPs 对照要求，不可得则注明 | `axes/6-predict/6.2-speculative-decoding/project.yaml` : `validation`/`falsification` | ✅ | `grep -n "63-F2" axes/6-predict/6.2-speculative-decoding/project.yaml`（:13、:15） |
| 63-F4 | `6.3.3` 采用 (i) 直接发 / (ii) 当草稿再验证两条各作一臂分报代价 | `axes/6-predict/6.3.3-latent-regression/project.yaml` : `method`/`milestone`/`validation`/`objective_criteria`/`falsification`/`cost_model` | ✅ | `grep -n "63-F4" axes/6-predict/6.3.3-latent-regression/project.yaml`（:12/13/14/15/25/35 六处） |

**21/21 都有可核实的落地落点。** 独立验证方法（不依赖报告）：先按 §1 速览表枚举出 21 条，再对每条用
`git diff 39f6051..HEAD -- <文件>` 与上述 grep 定位；用 YAML 递归比对脚本确认所有 `project.yaml` 改动
**只做「追加」或「有记录的改词」**，无测量数字被删改。

---

## 2 MoE 处置自洽（65-F1）✅

- `docs/structure.md` §12 轴 4 表：`4.1`/`4.2`/`4.3` 粒度列 = **`Backlog`**（原为「研究项」）✅
  - 复现：`grep -n '^| 4\.[123] |' docs/structure.md` → 三行末尾均为 `Backlog`。
- 表下注 `docs/structure.md:228`：「**65-F1 裁决（已裁决，2026-10-13）**…降为 Backlog…保留 `project.yaml`
  与本仓等效代价口径记录（`status: 未开始`）；「参照」那层含义由 `axes/7-references/README.md` 新增的业界
  标准 MoE（Switch / GShard / DeepSeekMoE）跑分线条目承接（见 §13.6）」✅
- `docs/structure.md` §13.6：Backlog 保留研究项形式（`status: 未开始`），并把 `4.1/4.2/4.3` 与 `2.1.3/2.2.5`
  并列为挂本轴；「轴 4 的三项按 65-F1 裁决降为 Backlog」✅
- 三份 `project.yaml` **保留**、`status` 全 = `未开始` ✅
  - 复现：`for d in 4.1-sparse-experts 4.2-routing-balance 4.3-expert-granularity; do grep -H '^status:' axes/4-moe/$d/project.yaml; done`
  - `4.1`/`4.2` 由 `进行中` → `未开始`（`change_log` 已记）；`4.3` 原即 `未开始`。
- `axes/7-references/README.md` 新增业界 MoE 跑台行（:17）：列 Switch（`2101.03961`）/ GShard（`2006.16668`）/
  DeepSeekMoE（`2401.06066`）/ Mixtral（`2401.04088`）/ DeepSeek-V3（`2412.19437`），并声明「轴 4 按 65-F1 裁决
  降为 Backlog…其「参照」含义由本行承接」✅
- **用词 exact**：`project.yaml` 的 `id`/`axis`/`title` 与 `docs/structure.md` §12 轴 4 表**逐字一致**
  - `4.1` id=`4.1` / axis=`4 专家轴（MoE）` / title=`稀疏激活专家（top-k 路由）`
  - `4.2` title=`路由与负载均衡`；`4.3` title=`专家粒度 / 异构（细粒度、共享专家）` ✅
  - `axis` 字段写法（`"<号> <轴名>"`）与其它轴（`1 时间轴`、`5 量化轴`）一致 ✅

---

## 3 无越界核查 ✅

| 项 | 结论 | 复现 |
|---|---|---|
| 测量数字未改 | ✅ 关键数字改动前后出现次数不变（追加者除外） | `for n in 0.036324 1.624979 1.5709 1.5314 +0.000101 +0.022522 6528 46.875 1.483571; do git grep -c -F "$n" 39f6051 -- axes/ \| …; done` —— 除 `0.036324` 因新增 C1 声明由 42→44（**追加**而非改数），其余全部 `old==new`；另用 YAML 递归比对（旧值必为新值子串）证实其它字段改动均为纯追加或有记录改词 |
| 实现 `pathlm/**`/`tests/**` 零改动 | ✅ 两提交未触碰 | `git diff --name-only 39f6051..HEAD -- pathlm tests tools` → **空** |
| 证据未删 | ✅ 无删除文件 | `git diff --diff-filter=D --name-only 39f6051..HEAD` → **空**；两提交的 `--name-status` 全为 `M`（18）/`A`（1） |
| 引擎通道存在 + 三项「未授权」 | ✅ | `.scratch/67-engine-changes/TICKET.md`（72 行，7013 B）；`grep -c "授权状态 \| \*\*未授权\*\*"` = **3**（E1/E2/E3 各一）；另 `:29` 声明「本通道不代表授权」 |

---

## 4 非阻断发现（位置 + 判据 + 复现）

**F1（非阻断）· C2 的「conclusion 并列读法」在 `1.1.4.1` 落到了 `failure_analysis`，不在 `conclusion`。**
- 位置：`axes/1-time/1.1.4.1-explicit-memory/project.yaml:16`（`failure_analysis` 末尾含「验收并列读法（C2 裁决）：
  …**字面未达成**…**否证**」），而同项 `:17` 的 `conclusion` 无「字面/实质」字样。
- 判据：`docs/research-schema.md` §5（本轮新写）与 `FRAMEWORK-DECISIONS.md` §6 均写明「**`conclusion` 须同时写明**
  「字面成立 / 实质成立或否证」」，点名了字段；`ADJUDICATION-LANDING.md` §3 C2 行也声称 8 项覆盖 `conclusion`。
- 复现：`python3 -c "import yaml;d=yaml.safe_load(open('axes/1-time/1.1.4.1-explicit-memory/project.yaml'));print('字面' in d['conclusion'], 'C2' in d['conclusion'])"` → `False False`。
- 影响：信息量未丢（同一文件的兄弟字段给出并列读法，且该格 `status=被否证` 时 `failure_analysis` 正是判词主字段），
  但**字段落点与自写契约的字面要求不一致**；`4.1/4.2/4.3` 因 `status=未开始` 无判词，属 N/A（报告未区分 N/A 与已落地）。

**F2（非阻断，最值得关注）· C1 的「falsification 声明用哪个带」全树覆盖不全（8/34）。**
- 位置：`.scratch/57-program/FRAMEWORK-DECISIONS.md` §6 C1（及 `docs/research-schema.md` §12.1 / `docs/methodology.md` §2）
  把声明义务写成「**每个**研究项的 `falsification` 必须显式声明用哪个带；未声明者判据不可判」，但实际只回填了 **8 项**。
- 复现（列出「引用噪声带却未声明」的全部项）：
  ```bash
  python3 - <<'PY'
  import yaml, glob
  for f in sorted(glob.glob('axes/**/project.yaml', recursive=True)):
      d=yaml.safe_load(open(f)); fa=d.get('falsification') or ''
      if ('噪声带' in fa or '0.014' in fa) and not (('用哪个带' in fa) or ('C1' in fa)):
          print(d.get('id'), d.get('status'), f)
  PY
  ```
  → **26 项**：`1.1.2/1.1.4.3/1.1.4.4/1.1.4.5/1.2.1/1.2.2/1.2.3/2.2.1/2.2.2/2.2.3/2.2.4/2.3.1/2.3.2/3.1.1/3.1.2/3.1.3/3.2.1/3.2.2/3.2.3/5.2/5.3/5.4/5.5/6.3.1/6.3.2/6.3.3`，
  其中 **8 项为 `进行中`/`已完成`**（`1.1.2`、`1.1.4.3`、`1.1.4.5`、`1.2.2`、`1.2.3`、`2.2.3`、`2.2.4`、**`3.2.1`（已完成）**）。
- 判据冲突点：`3.2.1`（`status=已完成`）的判词直接以「差 < 0.014」为否证线
  （`grep -n "差 < 0.014" axes/3-width/3.2.1-readout-reparam/project.yaml`），按新 §12.1「未声明者判据不可判」
  该已完成项的判据在新规下变成「不可判」——**新写入的 SSOT 与既有事实文档不自洽**。
- 报告表述问题：`ADJUDICATION-LANDING.md` §4 把未补范围描述为「**未开始**的研究项未逐一补」，
  但未补的 26 项里有 8 项是进行中/已完成（含已完成项），描述**低估**了缺口。
- 影响：不推翻任何已落地裁决，但违反 `AGENTS.md`「保持受影响事实一致」的要求；回填是机械动作
  （给引用噪声带的 `falsification` 逐项补「用哪个带」）。

**F3（非阻断观察）· `4.1`/`4.2` 的 `status=未开始` 与「已有本地工作」并存，属契约张力而非落地错误。**
- 位置：`axes/4-moe/4.1-sparse-experts/project.yaml`（`status=未开始` 但 `experiments` 非空、`conclusion`
  含 WP0 记账）、`axes/4-moe/4.2-routing-balance/project.yaml`（同理，含坍缩探针）。
- 判据：`docs/research-schema.md` §7「`未开始` = 尚无本地工作」与 `docs/structure.md` §12/§13.6
  「Backlog = 研究项形式、`status: 未开始`」相互张力；65-F1 裁决明文要求 `status: 未开始`。
- 复现：`grep -H '^status:' axes/4-moe/4.[12]-*/project.yaml` + `grep -n '^experiments' -A2 …`。
- 影响：落地**遵循了**裁决与 §12/§13.6，不判落地错误；但新规下「有本地工作却记未开始」的口径冲突未在 SSOT 消解，
  建议在 §7 或 §12 明确 Backlog 的 `status` 语义例外。

**F4（留痕观察，非判据项）· 落地报告未入库。**
- `ADJUDICATION-LANDING.md` 只存在于 subagent 产物目录，**不在** `HEAD` 的 `.scratch/57-program/` 下
  （`git log --all -- `.scratch/57-program/ADJUDICATION-LANDING.md`` 为空；`find . -name "ADJUDICATION-LANDING*"` 无命中）。
- 对照：本项目此前的审计留痕纪律（ticket 56 §8.2；M3 曾把「复核件只在 artifact」当作缺口并把六轴报告/复核件**入库**）。
- 说明：这是 subagent 运行时输出路径约定，未必属于本轮 writer 的责任范围；本轮 6 条判据未包含此项，故仅留痕。

---

## 4 未覆盖范围（如实声明）

- **科学正确性**：本轮只核「裁决是否落地」，**不核** 21 条裁决背后的科学论证是否成立（那属各轴 ticket 的既有复核）。
- **未验对象**：未复核 ticket 56 / 61–66 的源报告与各轴 `TICKET.md` 全文；未复核实验目录内部证据（`results.json`/`analysis.md`）内容。
- **引擎通道可行性**：未对照 `pathlm/**` 实代码核验 E1/E2/E3 的「最小 patch 描述」是否真如所述（只核通道存在与三项未授权标记）。
- **其余 project.yaml**：除本提交触碰的 13 份 + MoE 3 份 + 全 46 份的 `falsification` 带声明扫描外，
  未逐字复核其余研究项正文。
- **未跑 GPU / 未重跑任何实验**；未验证墙钟类数字（本轮未改成本口径）。
- **落地报告的权威副本**：仓库内不存在该文件，本轮据 subagent 产物副本
  `…/outputs/b87ec038-…/.scratch/57-program/ADJUDICATION-LANDING.md` 复核；若主 Agent 另有版本，本报告的
  「报告表述」类判据（F1/F2 的表述偏差）需按权威副本重估。
- **门只跑一遍**（按要求单跑）；`pytest tests/ -q` = **834 passed in 95.63s**，与提交信息「未改共享文件」一致。

---

## 5 门与注入实验（单跑 + `/tmp` 拒收证明）

工作树修订 `ea8b5c2`，命令与真实输出：

| 命令 | 结果 | 输出 |
|---|---|---|
| `python3 tools/summary.py --check` | **passed** | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） |
| `python3 tools/check_tree.py` | **passed** | 同上（exit 0） |
| `python3 -m pytest tests/ -q` | **passed** | `834 passed in 95.63s (0:01:35)`（exit 0） |
| `git diff --check 39f6051..HEAD` | **passed** | 无输出（exit 0） |

**`/tmp` 注入（副本 = `tools/`+`axes/`+`docs/` 实拷贝，`.scratch`/`.git` 软链到权威仓库）**：

| 注入 | 期望 | 实际 |
|---|---|---|
| 清洁基线 | 通过 | `OK：46 个 project.yaml…`（exit 0） |
| A：把 `6.2` 的 `status` 改为非法 `完成` | 拒 | `FAIL：1 项 … 非法 status='完成'（合法值：(未开始/进行中/已完成/被否证)）`（**exit 1**） |
| B：`cost_model` 混入 `快 3×（墙钟）` | 拒 | `FAIL：1 项 … cost_model 含墙钟/日期指标 '墙钟'`（**exit 1**） |
| C：坏 YAML 语法（`\tbad: [1,`） | 拒 | `FAIL：1 项 … YAML 解析失败`（**exit 1**） |
| D：删掉必填 `falsification` | 拒 | `FAIL：1 项 … 缺字段 falsification`（**exit 1**） |
| 恢复后基线 | 通过 | `OK…`（exit 0） |

结论：**门是活的且真跑，且会拒绝坏输入**（非恒真）。

---

## 6 台账标记核查（21/21）✅

- `FRAMEWORK-DECISIONS.md` §6「裁决落地（21 条全部已裁决）」现状存在，逐条格式 = **「已裁决」+ 裁决内容 + 落地指针**。
- 复现：`awk '/^## 6 裁决落地/,0' .scratch/57-program/FRAMEWORK-DECISIONS.md | grep -nE '^- \*\*[0-9A-Z]'`
  → **21 条目**：C1–C5、62-F2/F3/F4、61-F1/F3/F4、64-F1/F3、65-F1/F2、66-F1/F2/F3/F4、63-F2/F4。
- 每条均含落地指针（文档节号 / `project.yaml` 字段 / 通道文件）；`61-F3`、`66-F3` 等「无字段改」的条目也标明
  「落地：本表标记（无字段改）/ `status`（无改）」，未伪装成有落地。

---

## 7 结论

**PASS。** 21 条裁决全部有独立可复核的落地落点；MoE 处置自洽且用词 exact；本轮无越界（测量数字未改、
`pathlm/**`/`tests/**` 零改动、证据未删）；C3 通道三项标记「未授权」；三道门真跑且注入实验证明其会拒；
台账 21/21 标注「已裁决 + 内容 + 落地指针」。

**非阻断发现 3 条 + 留痕 1 条**（详见 §4）：F1（C2 在 `1.1.4.1` 落到 `failure_analysis` 而非 `conclusion`）、
F2（C1 的 `falsification` 带声明只覆盖 8/34，26 项缺口含已完成项 `3.2.1`，且报告低估缺口）、
F3（`4.1/4.2` 的 `未开始` 与「已有本地工作」契约张力）、F4（落地报告未入库）。均为可机械修复的账差/一致性问题，
不改变任何已落地裁决方向。

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "独立枚举 HEAD=ea8b5c2 的 21 条裁决（C1–C5 + 16 条轴内），逐条给出落点文件:字段与复现 grep，全部命中；MoE 处置/无越界/门/台账 4 组判据逐项核过，未发现阻断项。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "三条门单跑全绿（summary.py --check / check_tree.py / pytest 834 passed）+ /tmp 副本四种坏输入注入全部 exit 1；关键测量数字改动前后计数一致；diff 无 pathlm/tests/tools、无删除文件。"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    { "command": "python3 tools/summary.py --check", "result": "passed", "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md。（exit 0）" },
    { "command": "python3 tools/check_tree.py", "result": "passed", "summary": "同上（exit 0）" },
    { "command": "python3 -m pytest tests/ -q", "result": "passed", "summary": "834 passed in 95.63s" },
    { "command": "git diff --check 39f6051..HEAD", "result": "passed", "summary": "无输出（exit 0）" },
    { "command": "/tmp 副本注入 非法 status / 墙钟 cost_model / 坏 YAML / 缺 falsification", "result": "passed", "summary": "四次注入全部 FAIL 且 exit 1，恢复后基线 OK —— 门非恒真、会拒坏输入" },
    { "command": "git diff --name-only 39f6051..HEAD -- pathlm tests tools", "result": "passed", "summary": "空 —— 实现代码零改动" },
    { "command": "git diff --diff-filter=D --name-only 39f6051..HEAD", "result": "passed", "summary": "空 —— 证据零删除" }
  ],
  "validationOutput": [
    "21/21 裁决均有落点；C1 4 要点、C2 实质验收、C5 4 条判词原样 均逐项核过",
    "MoE：structure.md §12 轴4 表 Backlog + §12 表下注 + §13.6；7-references README 新增业界 MoE 跑分线行；4.1/4.2/4.3 status=未开始、id/axis/title 与 §12 逐字一致",
    "非阻断 F1/F2/F3/F4 已定位（文件:行 + 复现命令）",
    "F2 复现：26 项 falsification 引用噪声带却未声明用哪个带（8 项进行中/已完成，含已完成 3.2.1）"
  ],
  "residualRisks": [
    "F1：C2 的 conclusion 并列读法在 1.1.4.1 落到 failure_analysis，与自写契约 §5 点名字段不一致（1.1.4.1:16 vs :17）",
    "F2：C1 的 falsification 带声明义务（SSOT 写作『每个研究项』）实际只覆盖 8/34；26 项缺口含已完成 3.2.1，构成新 SSOT 与既有事实文档不一致；落地报告 §4 低估该缺口",
    "F3：4.1/4.2 status=未开始 与 experiments/conclusion 非空并存，是 §7 与 §12/§13.6 的契约张力（落地遵循了裁决，未消解口径冲突）",
    "F4：落地报告未入库，仅存于 subagent 产物目录；若项目要求审计留痕入库，需另补"
  ],
  "noStagedFiles": true,
  "diffSummary": "本轮为只读复核：未修改仓库任何文件（changedFiles 为空）；仅在 /tmp 副本做注入实验。复核对象为 HEAD=ea8b5c2 的 e98a26c+ea8b5c2 两提交（19 文件：SSOT 3 份 + 台账 1 份 + 引擎通道 1 份 + project.yaml 14 份）。",
  "reviewFindings": [
    "no blockers",
    "non-blocking: axes/1-time/1.1.4.1-explicit-memory/project.yaml:16 - C2 并列读法落在 failure_analysis 而非 conclusion（违反 docs/research-schema.md §5 字面要求）",
    "non-blocking: docs/research-schema.md §12.1 声明义务 vs 26 项未声明（含 axes/3-width/3.2.1-readout-reparam/project.yaml，已完成）",
    "non-blocking: axes/4-moe/4.1-sparse-experts, 4.2-routing-balance - status=未开始 与已有本地工作并存（§7 与 §12 张力）",
    "observation: .scratch/57-program/ADJUDICATION-LANDING.md 未入库，仅存于 subagent 产物目录"
  ],
  "manualNotes": "复核对象 HEAD=ea8b5c2fa60ccbbfefd7b2ff3584563e60952440，工作树干净。落地报告的权威副本不在仓库内，本轮据 subagent 产物副本 b87ec038-…/ADJUDICATION-LANDING.md 复核；F1/F2 中『报告表述偏差』部分在权威副本不同时需重估。所有注入实验均在 /tmp 副本进行，未触碰权威仓库；/tmp 副本的 6.2/5.1 样板曾一度被上一条 `set -e` 中断污染，已从权威仓库重取后重跑，最终基线 OK。"
}
```
