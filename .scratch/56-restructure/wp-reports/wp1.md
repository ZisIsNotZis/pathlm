# ticket 56 · M2 轴 1 §1.1（时间轴 · 可见范围 / 信息右移）迁移 · WP 交付报告

> writer：worker（fresh 上下文，单一写入者）｜ 提交：`18a3e6b`
> 依据：`docs/structure.md` §12–§15、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`、`.scratch/55-review/{A,M}-PROBLEMS.md`
> 范围：`axes/1-time/` 的 §1.1 子树（8 个研究项 + 2 个分组 README + 轴 README）。

## 0. 人话摘要

这次把「时间轴」下**可见范围**那一支按冻结的新结构搬完了：原来散在旧格子里的实验，整目录搬进了新的
`axes/1-time/`，并按新契约（`docs/research-schema.md`）给每个研究项写了一份合格的 `project.yaml`。
搬的时候**一条证据都没删**（旧格目录和旧 README 原样留着，交后续 integrator 统一退役；`.scratch` 证据原地不动）。
顺带按上一轮独立复核（55-review）指出的毛病把结论改正了：不再把「手段」当研究目的、判据与目的对齐、
写清该涨/该跌/必须不变、给出否证条件与失败归因，pareto 口径统一成「stock 全因果 = 100%」。
最要紧的两条是：**M 轴那格以前从没测过它自己的目的（质量-代价前沿）**，还被引擎开销逼成了「单共享槽」的
退化实现却当成机制结论——这两条都如实改成了「进行中 + 缺口」而不是照抄「已判」；
**A 轴「窗口是目的」的定位错误**也纠正成「窗口是手段，代价进前沿」。
因为搬了实验目录，仓库里约 30 个测试文件/证据 JSON 里写死的旧路径必须跟着改（只改路径，不动任何测量数字），
改完 `pytest`、校验器、汇总脚本全绿。

## 1. 逐项验收自证

### 1.1 用词 = docs/structure.md §12 exact wording（硬约束 ①）

逐条比对（脚本按 §12 表格取 id/title/目录，与 project.yaml 比）：

- 8 个研究项标题**逐字相等**：`hard 窗口`、`soft 衰减`、`可见量上限`、
  `显式记忆槽 / 线性状态 / 递归状态（delta 规则族：DeltaNet / GLA / Mamba-2）`、
  `压缩：摘要槽 / 低秩投影 / pooling`、`压缩触发与优先级`、`外部检索`、
  `implicit（掩码/窗口，已证退化）vs **explicit**（有希望的方向）`（含 §12 原样的粗体标记，不改写）。
- 目录与 §12 的 `axes/1-time/<id>-<slug>/` **逐字相等**（含结尾斜杠归一后）。
- 分组 README 目录 `1.1-visibility/`、`1.1.4-right-shift/` 与 §12 相等。
- `axis: "1 时间轴"`，与 exemplar 一致。

### 1.2 证据按 MAPPING §1 迁移（硬约束 ②）

| 旧格 | 新家 | 迁入内容 |
|---|---|---|
| A3-window-anchor | 1.1.1 | `experiments/**`（`2505-m1-followups` + README） |
| A6-mixing-operator（长 T 系列） | 1.1.1 | `2622` `2623` `2624` `2625` `2626` `2628` `2629` `2636`（8 目录） |
| A4-soft-distance-penalty | 1.1.2 | `experiments/**`（`2507-x1-x3-distance-penalty` + README） |
| A9-old-info-retention | 1.1.4.5 | `experiments/**`（`2615-window-trained-retention-13m`、`2623-window-on-longrange-13m` + README） |
| A10-compression-trigger | 1.1.4.3 | `experiments/**`（`2615-priority-shootout-13m` + README） |
| M1-carrier-type | 1.1.4.1 | 全部 23 个实验目录 + README |
| M2-carrier-compression | 1.1.4.2 | 全部 2 个实验目录 + README |

- 一律 `git mv` 整目录搬家；**拆并格一格一主家**：1.1.3 只指针 see 1.1.1、**不接收任何文件**（`experiments: []`）。
- **一格一份家、不复制**：A6 的 `2614-topk` / `2616-linear` / `2637-linear` 未迁（属 1.2.2 / 1.2.3，留旧格待本轴后续 WP）。
- `.scratch/**/evidence` **原地不动**；`project.yaml` 指向 `.scratch` 的指针保留（1.1.1、1.1.2、1.1.4.5）。
- 旧格顶层 README **未删**（`git diff --cached --name-status` 里无 `D axes/A-mixing/A3-window-anchor/README.md` 等）——
  仅旧格 `experiments/README.md` 随实验目录搬走。
- 证据不丢：`find axes -type f` 迁移前 **2436** → 迁移后 **2446**（未减少，含新增 YAML/README）。

### 1.3 按 55-review 修正结论（硬约束 ③）

- **目的 vs 手段**：每个 `purpose` 写痛点（如 1.1.1 = 前缀可丢、成本随 T 线性；1.1.4.1 = 把远处信息搬进固定状态使账单有界），
  不再以手段当目的。分组 README / 轴 README 也据 `docs/structure.md` 陈述「信息右移」为目的。
- **pareto 口径**：`validation` 统一写「对照 = stock 全因果基线（归一为 100%）+ 同预算窗口/量化点」。
- **判据与目的对齐**：`objective_criteria` 写「该涨 / 该跌 / 必须不变」；`falsification` 给否证条件。
- **证据锚点指向真实存在的实验**：`experiments` 全部指向仓内新路径（校验器逐条验存在）；不再用不可复核的旧 `.scratch` markdown 当唯一锚。
- **A 轴问题**：A10 判据-实测错配 + 门槛算术不可达（P15/P16）、A4 口径不全（P10，固定 w/全头共用/无长度外推）、
  A9 与 A3 重复（P14）、A6「长 T」命名歧义改为「窗口代价随 T」（P11/P12）、A4 的正确口径（per-head slope + 长度外推）列为 WP。
- **M 轴问题**：M1 **从未测自身目的**（M-01/M-03）→ 补 WP1「把状态放进质量-代价前沿」；
  被引擎开销逼成单共享槽（M-05）→ 记入 gaps 并补 WP2「逐层状态 + 门控衰减 + 内容相关写入门 + 与 DeltaNet/GLA 基线对照」（M-04/M-06）；
  `status` 由 `judged` 改为 `进行中`（M-22，verdict 是「弱证据/未证」堆叠、非明确结论）；
  错误的 OOD「能力缺席」归因与「+0.2% 说明有用」误读列为 WP3（M-08/M-09/M-10）；
  M2 无自有压缩实验、旧 verdict 是过期待办 → `进行中` 并如实记（M-18/M-19）。
  **代价口径**：`cost_model` 一律等效理论量（FLOPs / layer-passes / tokens-per-forward + pipelining），**无墙钟/日期**。

### 1.4 字段齐全 + 单 milestone + 独立复核（硬约束 ④）

8 个研究项均含全部必填字段：`role_map`（owner/writer/reviewer/integrator 四角色）、**有且仅有一个** `milestone{output,acceptance}`、
`work_packages`（每条 `id/what/goal/means/acceptance/evidence/review/status`，`review` 均写「跑完另派 fresh 复核（SCHEMA §14/§14b）」）、
`sota_baseline`/`gap`/`method`/`validation`/`objective_criteria`/`falsification`/`failure_analysis`/`cost_model`/`conclusion_review`/`argument`/`gaps`/`started`/`ended`/`change_log`。
`status` 仅用四值：1.1.3、1.1.4.4 = 未开始；其余 6 项 = 进行中（任务边界要求「不照抄错误结论」，故不轻置「已完成」）。

### 1.5 边界

- **只写我负责的路径**：新增/改写全部落在 `axes/1-time/`；未改 `catalog.yaml` / `INDEX.md` / `axes/C-vertical/**`（工作树既有未提交改动**保留不回滚**，提交后 `git status` 仍显示它们为 ` M`）。
- 未碰其他轴的**新树**内容；未删除任何旧格目录/旧 README；未删任何证据。
- **唯一跨目录改动**（必要且最小）：随证据搬迁同步路径常量 ——
  `axis/1-time/**` 内的 `.py`/`.sh`/`.json` 路径指针、`tests/**` 内约 23 个文件的硬编码路径、
  `tests/evidence/longT_step0_anchors.json`、以及 **A6 内未迁的 `2637-linear-attn-quality/linear_driver.py`**
  对已迁 `2625-longT-stability` 的跨目录引用（不改则 `test_linear_attn_quality.py` 必失败）。
  **只改路径字符串，未改任何测量数字/判据**；`.log`/`.txt`/`.md` 历史记录未改写。

## 2. 命令与结果（全部在提交前于工作树基线 18a3e6b 的父提交上跑）

| 命令 | 结果 |
|---|---|
| `python3 tools/check_tree.py` | `OK：9 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| `python3 tools/summary.py --check` | 同上（exit 0） |
| `python3 -m pytest tests/ -q` | **827 passed**（0 失败） |
| `git diff --cached --check` | 无输出（exit 0，无空白错误） |
| `find axes -type f \| wc -l` | 2446（迁移前 2436；≥ M2 门限 2434） |
| `git ls-files axes \| wc -l` | 1669（迁移前 1659；+10 = 7 新 project.yaml + 3 新 README） |
| `git commit -F - <<'EOF' …` | `18a3e6b`，1069 files changed（含 rename） |

标题/目录与 §12 逐字比对、JSON 合法性（`json.load` 全过）另用一次性脚本核过（见 §1.1、§1.2）。

## 3. 未决项 / 风险（交 integrator 与复核）

1. **旧格（A3/A4/A9/A10、M1/M2、A6 长 T）原地保留**：旧目录与旧 README 仍在（按任务边界不删），
   `catalog.yaml` 仍存旧 `local:` 块——待 M2 全轴迁完后由 integrator 移入 `archive/wiki-legacy/` 统一退役。
   当前 `axes/1-time` 与旧格**并存**，读者须以新树 `project.yaml` 为准（已在旧格与新 README 中无冲突表述）。
2. **实验内部 `.log`/`.txt`/`.md` 历史记录内仍含旧路径字样**（未改写，属历史记录）；迁移证据 JSON 的路径指针已统一。
3. **实验编号不唯一**（同格 `2617×3` / `2624×2` / `2630×2`，跨轴 `2615/2623` 撞号，55-review M-16）**未重命名**
   （重命名会破坏既有复现锚点）；作为 1.1.4.1 的 gap 记录。
4. **`role_map` 用占位实名**（「轴 1 时间轴 lead（待指派）」等），与 exemplar 一致；按 `57-program/TICKET.md` §3 由主 Agent 在 M2 派发时填。
5. `2637-linear-attn-quality/linear_driver.py` 的一行路径由我（本 WP）改；该目录属后续 §1.2 WP，迁走时需保留此指向。

## 4. 复核建议（顺序屏障）

本 WP 交付后需**另派 fresh 独立复核**（`SCHEMA.md` §14/§14b，复核者不得为本 writer）：核对
(a) 用词/目录与 §12 逐字一致；(b) 证据搬迁与 MAPPING 一致、无复制、无丢失；(c) 55-review 修正是否落实、是否照抄了错误结论；
(d) `summary.py --check` / `check_tree.py` / `pytest` 三绿；(e) 路径同步改动未篡改测量数字。
