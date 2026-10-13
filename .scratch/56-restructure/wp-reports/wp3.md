# ticket 56 · M2 · 轴 2 深度轴（§2.1–§2.3）迁移报告（wp3）

时间：2026-10-13T10:37:02+08:00 起 ｜ actor：worker（fresh 上下文，单一写入者）
commit：`d6f099b`（67 files changed, 813 insertions(+), 5 deletions(-)）

## 人话摘要

这一轮把「深度轴」整条从旧 C 轴迁成了新树：3 个分组 + 10 个研究项，每个研究项一份合格的
`project.yaml`。最核心的 `2.1.1 early exit` 承接了旧 C1（每个 token 跑几层），并把旧 C5
（精化算子）和 C6（训练期深度监督）作为它的两个**手段**（精化门 / 深度监督配方）并进
`work_packages` 与 `experiments`——因为按新规范「同一个研究问题的手段不另立研究项」。
证据用 `git mv` **整目录**搬进新家，一格一份家、不复制；`.scratch/**/evidence` 原地不动，
只改 `project.yaml` 指针；旧 `axes/C-vertical/**` 目录与旧 README 一个没删，交后续 integrator 统一退役。

搬的时候**按独立复核（55-review C-PROBLEMS）改正了错误结论，没有照抄**。关键三条：
① 旧格把 `tf@4`（浅层 argmax 与深层 argmax 是否相同）当「质量不掉」的凭据——它只是单 token
自洽代理、近乎同义反复；新判据改成**序列级** perplexity/一致率 vs 全深度，算力记
layer-passes/token，**墙钟不进判据**（旧格的「exit-4 3.1× 墙钟」卖点已剔除）。
② 旧格声称「引擎已具备逐 token 退出门控」——实际能跑本线 checkpoint 的 depth-ar 解码器
`DepthARDecoder.step(inp, forced_depth)` **只支持固定退出深度**，逐 token 门控只长在旧引擎与
参照实验里；这条误述删除，改成 WP1 待补。
③ 旧格「出口资产只能由深度监督造出」的对照证据不成立（0.12 是未训练 tied 读出 proxy，正确对照
`23d-H1-12k` 的 tf@4 从未跑），降级成一个待办 WP。
另外，出资人在 C3 批注里点名要做的「把之前各层结果 stack 起来做 causal attention」**从未测**却被判
`judged`——新建 `2.2.2` 专门承接它；旧 C3 的 `judged` 降为 `进行中`（判据 `tf@4≥0.95` 对 stock dense
零区分度、n=1）。C4（权重共享）按出资人先验降为 `2.2.5` Backlog，并消掉旧格 todo/(a)/不判(a) 三处矛盾。

自检全绿：`check_tree.py`（23 个 project.yaml）、`summary.py --check`、`pytest` 827 passed；
证据一个没丢（`find axes -type f` 2469 ≥ 2434 门限；`git ls-files axes` 1675 → 1692 = 新增 17 个跟踪文件）。

---

## 一、逐项验收自证

### 硬约束① 用词 = `docs/structure.md` §12 exact wording

用脚本从 §12 轴 2 表取 id/title/目录，与各 `project.yaml` 比：

| id | title（与 §12 逐字相等） | 目录（= §12） | 粒度 |
|---|---|---|---|
| 2.1 | 按需深度 | `axes/2-depth/2.1-ondemand-depth/` | 分组 README |
| 2.1.1 | early exit（最重要） | `axes/2-depth/2.1.1-early-exit/` | 研究项 |
| 2.1.2 | skipping（跳层） | `axes/2-depth/2.1.2-skipping/` | 研究项 |
| 2.1.3 | repeating 循环 transformer：多算力换少参数 | `axes/2-depth/2.1.3-repeating/` | Backlog |
| 2.2 | 层间怎么聚合（残差族） | `axes/2-depth/2.2-layer-aggregation/` | 分组 README |
| 2.2.1 | 残差的形式 | `axes/2-depth/2.2.1-residual-form/` | 研究项 |
| 2.2.2 | 用 causal attention 代替残差 | `axes/2-depth/2.2.2-attention-residual/` | 研究项 |
| 2.2.3 | 层间学习加权和（DenseFormer 式） | `axes/2-depth/2.2.3-learned-layer-sum/` | 研究项 |
| 2.2.4 | 并行子层 | `axes/2-depth/2.2.4-parallel-sublayer/` | 研究项 |
| 2.2.5 | 权重共享 | `axes/2-depth/2.2.5-weight-sharing/` | Backlog |
| 2.3 | 深度的形状 | `axes/2-depth/2.3-depth-shape/` | 分组 README |
| 2.3.1 | 逐层容量变化：前大后小 / 逐渐放大 | `axes/2-depth/2.3.1-layer-capacity/` | 研究项 |
| 2.3.2 | 逐层分辨率：先粗后细、层级式（层级合并） | `axes/2-depth/2.3.2-layer-resolution/` | 研究项 |

- 13 个标题**逐字相等**（脚本比对全 OK）；`axis: "2 深度轴"` 与 §2 标题一致；分组节点只写
  `README.md`、不写 `project.yaml`（§13.2）。
- `1.1.1` exemplar 同款：`dir` 不带结尾斜杠（`docs/research-schema.md` §2 与校验器一致）。

### 硬约束② 按 MAPPING 迁证据（整目录 git mv、一格一份家、只指针不复制）

| 旧格 | 新家 | 迁入内容 | 处置 |
|---|---|---|---|
| C1-layers-per-token | 2.1.1 | `experiments/**`（`2523-depth-ar-first`、`2523d-residual-tax-decomposition`、`2621-depth-draft-e2e`） | 迁移（核心） |
| C5-refinement-operator | 2.1.1 | `experiments/2523c-three-ablations` → WP4 精化门 | 并（手段） |
| C6-depth-supervision | 2.1.1 | `experiments/{2518-exit-retry-law,2524-depth-ar-100m}` → WP3 深度监督配方 | 并（手段） |
| C2-layer-selection-order | 2.1.2 | `experiments/{2506-m2-element-tax-law,2509-m3-composition-law}` | 并 |
| C3-inter-layer-topology | 2.2.3（主家） | `experiments/2613-inter-layer-topology-13m` | 拆（并行子层读数归 2.2.4，只指针） |
| C4-weight-sharing | 2.2.5 | 无自有 experiments（本地零经验） | Backlog |

- 一律 `git mv` 整目录（`git diff --cached` 显示 `R100`/`R093` 纯重命名）；**一格一份家、不复制**：
  C3 的 `2613` 只落 `2.2.3`，`2.2.4` 的 `experiments: []`、用文字指针 `see 2.2.3`（与 1.1.3 指针-only 同款）；
  C5/C6 的实验因是本项手段而按 MAPPING 落 `2.1.1/experiments/`。
- `.scratch/**/evidence` **原地未动**：`2.1.1` 的指针指 `.scratch/23-depth-ar/evidence @1586745`、
  `.scratch/24-depth-ar-100m/evidence @7725891`、`.scratch/25-tax-attribution/evidence @2e86924`；
  `2.1.2` 指 `.scratch/{06-m2-sweep,09-m3-interactions,08-tts-probe}/evidence @a03a579/@39b1838/@aa97bcf`——全部可达、真实存在。
- **旧格顶层目录与 README 未删**（`git status` 无 `D axes/C-vertical/**/README.md`）；旧格的
  `experiments/README.md`（旧 AUTO 索引）也留在旧格，待 integrator 统一退役。
- 证据不丢：`find axes -type f` 2469（迁移前 2452，未减少）；`git ls-files axes` 1675 → 1692（+17）。

### 硬约束③ 按 55-review 修正结论（不得照抄错误结论）

- **tf@4 单 token 代理（C-P01/P02）**：`2.1.1` 的 `purpose/validation/objective_criteria` 全改
  **序列级** perplexity/一致率 vs 全深度；`argument` 写明单 token 代理不足（p^N 累积：depth-4 p=0.973、
  N=100 ⇒ ≈6.5%），并注明 tf@4 = depth-k argmax == depth-L argmax 是自洽不是正确性。
- **逐 token 门控实际缺失（C-P03）**：`2.1.1` 明确写 depth-ar 生产解码器只支持固定退出深度，
  逐 token 门控在旧引擎与参照实验里；WP1 就是补实装 + 校准。
- **序列级 e2e 未测（C-P01/C-P04）**：`2.1.1` 的主判据与 milestone 都钉在「逐 token 门控的自回归
  序列级结果」，`gaps` 第一条即「主判据未测」；墙钟卖点剔除。
- **判据与目的对齐、该涨/该跌/必须不变 + 否证条件 + 失败归因**：每个研究项 `objective_criteria`
  三态齐全、`falsification` 非空、`failure_analysis` 键齐（未判失败写 `""`）。
- **pareto 口径以 stock 全深度为 100%**：每个 `validation` 均写「对照 = stock 全深度（归一为 100%）」。
- **证据锚点指向真实存在的实验**：`experiments` 全部指向仓内新路径（校验器逐条验存在）+ 可达锚点
  （`@1586745`/`@7725891`/`@2e86924`/`@a03a579`/`@39b1838`/`@aa97bcf`/`@d77c6b4`/`@b3c2484` 均
  `git cat-file -t` = commit 且可达）。
- **未照抄的错误结论**：删去「引擎已具备逐 token 门控」；「出口资产只能由深度监督造出」降为 WP3 待办
  （C-P13）；C3 判据 `tf@4≥0.95` 零区分度、n=1 → `2.2.3` status 由 `judged` 降 `进行中`；
  出资人跨层注意力（C-P10）→ 新建 `2.2.2`；C4 三处矛盾口径（C-P11）→ `2.2.5` Backlog 单一收口。
- **工作树 C1 未提交修订照准迁入**：判据 = 序列级 perplexity/一致率 vs 全深度、算力 = layer-passes/token、
  墙钟不进判据已落 `2.1.1`（grep 命中 `layer-passes/token`×4、`序列级`×23、`墙钟不进判据`×2）；
  旧文件（`axes/C-vertical/C1-layers-per-token/README.md`、`axes/C-vertical/README.md`）
  **保持工作树原状、不回滚、不提交**（`git status` 仍 ` M`）。

### 硬约束④ 字段齐全 + 单 milestone + 独立复核

10 个研究项均含全部必填字段：`role_map`（owner/writer/reviewer/integrator 四角色，reviewer ≠ writer）、
**有且仅有一个** `milestone{output,acceptance}`、`work_packages`（每条 `id/what/goal/means/acceptance/evidence/review/status`，
`review` 均写「另派 fresh 复核（SCHEMA §14/§14b）」= 独立复核）、
`sota_baseline/gap/method/validation/objective_criteria/falsification/failure_analysis/cost_model/conclusion_review/argument/gaps/started/ended/change_log`。
`cost_model` **只算等效理论量**（FLOPs / layer-passes / tokens-per-forward，均含 pipelining 说明，
无墙钟/日期——校验器正则门通过）。

`2.1.1` 的 `work_packages` 按 §13.3 把手段写全：**WP1 退出决策头 / WP2 异构深度回看 /
WP3 深度监督配方 / WP4 精化门**，每条 what/goal/means/acceptance/evidence/review/status 齐。

### 硬约束⑤ status 枚举

仅用四值：`2.1.1`/`2.1.2`/`2.2.3`/`2.2.4` = 进行中；`2.1.3`/`2.2.1`/`2.2.2`/`2.2.5`/`2.3.1`/`2.3.2` = 未开始。无第五值。

### 边界

- 只写负责路径（`axes/2-depth/**`）；未碰其他轴目录；未改 `catalog.yaml` / `INDEX.md`
  （`git status` 仍显示二者为 ` M`，工作树既有改动**保留未回滚、未提交**）。
- 旧格目录与旧 README 未删、未改（旧 C1/C-vertical README 的未提交改动仍在工作树）。
- 未删任何证据。
- **唯一跨目录改动（必要且最小）**：随证据搬迁同步路径常量——`tests/test_depth_draft_e2e.py`
  （docstring + `sys.path`）、`tests/test_reconcile_harness.py`（`EXP`）、
  `axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/gpu_run.sh`（`OUT`/`EXP`）。
  **只改路径字符串，未改任何测量数字**（不改则 `pytest` 必失败）。

---

## 二、命令与结果

| 命令 | 结果 |
|---|---|
| `python3 tools/check_tree.py` | `OK：23 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| `python3 tools/summary.py --check` | 同上（exit 0） |
| `python3 -m pytest tests/ -q` | **827 passed in 89.89s**（0 失败） |
| `git diff --check` | 无输出（exit 0，无空白错误） |
| `git diff --cached --name-only \| wc -l` | 67（17 add + 2 modify + 48 rename） |
| `git diff --cached --name-only \| grep C-vertical\|INDEX\|catalog` | 空（未误 stage 旧文件） |
| `find axes -type f \| wc -l` | 2469（迁移前 2452；≥ M2 门限 2434） |
| `git ls-files axes \| wc -l` | 1675 → 1692（+17 = 10 project.yaml + 7 README） |
| commit | `d6f099b`；提交后 `git diff --cached --name-only \| wc -l` = 0 |

---

## 三、未决项 / 风险（交 integrator 与复核）

1. **`2.2.3` 为 C3 证据主家的选择属 MAPPING 歧义裁决**：MAPPING §1 只写「C3 → 2.2.3 + 2.2.4」、
   未指定哪一个收 `2613` 文件；我按「2.2 层间聚合主族 + DenseFormer 是 headline 中性结果」选 `2.2.3`
   为主家，`2.2.4` 只指针。若复核认为应反过来（并行子层为主家），需 integrator 裁决后换家。
2. **`2.2.4` 的 `experiments: []` + `status: 进行中`**：它确有一个质量面读数（parallel +0.0458），
   但证据在主家 `2.2.3`，按「一格一份家」只指针、不复制。是否接受「指针-only 的进行中项」由复核判定。
3. **status 未照抄 judged**：`2.1.1`/`2.1.2`/`2.2.3`/`2.2.4` 由旧 `judged`/`todo` 改判为 `进行中`
   （55-review 判其判据缺陷/证据不成立/未测对的东西），`conclusion` 保持空、读数记入 `conclusion_review`/`argument`。
   复核须确认这没有隐藏既有观测。
4. **机制重叠边界（未整合，已记 gaps）**：`2.3.1` 与 `3.1.2` 逐层宽度变化、`2.3.2` 与 `1.2.4` 动态合并、
   `2.1.3` 与 `2.2.5` 记账视角——均已在各自 `gap` 写边界，需 integrator 跨轴核一遍不重复、不矛盾。
5. **`2613` 内 `smoke/smoke.json` 仍含旧绝对路径**（`.../C-vertical/C3/.../smoke`）：这是当时**实际
   运行记录的原始命令**（历史证据），按轴 1 先例未改写；`project.yaml`/`gpu_run.sh` 的路径指针已更新。
6. **旧 `axes/C-vertical/**` 与 `catalog.yaml`/`INDEX.md` 并存**：待 M2 全轴迁完后由 integrator 移入
   `archive/wiki-legacy/` 统一退役；读者须以新树 `project.yaml` 为准。
7. **未做独立复核**：本 WP 的代码/文档复核须由 fresh reviewer（≠ 本 writer）按 `SCHEMA.md` §14/§14b 给出。

---

## 四、建议下一步

- 由 fresh reviewer 复核本提交（`d6f099b`）的 10 个 `project.yaml` + 2 分组/1 轴 README + 迁移完整性 +
  55-review 修正落实 + 路径同步改动未篡改数字。
- integrator 裁决 §三.1（C3 主家）与 §三.4（跨研究项边界），并把旧 C 轴目录统一退役。
