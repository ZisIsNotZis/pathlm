# ticket 56 · M2 · 轴 1 §1.2 + §1.3 迁移报告（wp2）

时间：2026-10-13T10:29:52+08:00 起 ｜ actor：worker（fresh 上下文，单一写入者）
commit：`65adaec`（92 files changed, 330 insertions(+), 2 deletions(-)）

## 人话摘要

这一轮把「时间轴」里**可见结构**那一片（§1.2）从旧格迁成了 4 个新研究项，并把 §1.3 寻址
登记为「参照」：dilated（空洞）、块稀疏/条带/局部+全局混合、内容选择 top-k、动态合并。
旧格 A6「混合算子」被拆开——它原来把「按内容选 top-k」和「线性注意力」两个独立假设塞进一格、
且用一堆选项罗列当研究问题（55-review 判为「定义里只有手段、没有目的」），现在
top-k 的实验（`2614`）去了 `1.2.3`，线性注意力的两个实验（`2616`、`2637`）去了 `1.2.2`，
用整目录 git mv 搬过去（一格一份家、不复制）。全新研究项 `1.2.1`、`1.2.4` 本仓没有旧格，
按规范写了目的、假设、判据与成本口径，状态「未开始」。

写每个研究项时按 55-review 的问题台账**改了错**，没有照抄：目的写成「要解决的痛点」而不是手段
（如 1.2.3 的目的是「省算力并保住需要内容匹配的远距离信息」，不是「用 top-k」）；pareto 口径以
**stock 全因果注意力归一为 100%**；判据写清「什么该涨、什么该跌、什么必须不变」；补了**否证条件**
与**失败归因**框架；证据锚点指向真实存在的实验（`@76a07a8`、`@3597039` 均可达）。旧结论里
「越 0.014 就有害」的强表述被**降级**——因为该 0.014 全局带的来源自己就写着「排序声明需要逐格复本」，
两臂差界应约 0.0195（55-review P6），所以只作边缘判定；配对 t 也只量评测批内差、不含训练重跑方差
（P19）。

自检全绿：`python3 tools/summary.py --check`（13 个 project.yaml）、`python3 tools/check_tree.py`、
`python3 -m pytest tests/ -q`（827 passed）。证据一个没丢：`find axes -type f` 计数未减少、
`git ls-files axes` 只增（新文件）。

---

## 一、逐项验收自证

### 硬约束① 用词 = docs/structure.md exact wording

新建/迁移的研究项 id 与 title **逐字**取自 `docs/structure.md` §12 轴 1 表：

| id | title（与 §12 逐字一致） | 目录（= §12 目录） |
|---|---|---|
| 1.2 | 可见结构（非均匀的可见模式） | `axes/1-time/1.2-visibility-structure/`（分组 README） |
| 1.2.1 | dilated / 空洞 | `axes/1-time/1.2.1-dilated/` |
| 1.2.2 | 块稀疏 / 条带 / 局部+全局混合（Longformer 式） | `axes/1-time/1.2.2-block-sparse/` |
| 1.2.3 | 内容选择 top-k | `axes/1-time/1.2.3-content-topk/` |
| 1.2.4 | 动态合并：越往下层 token 越少 | `axes/1-time/1.2.4-dynamic-merge/` |
| 1.3 | 寻址（前提，不是手段选择） | 参照（无 project.yaml，登记 `axes/7-references/README.md`） |

### 硬约束② 按 MAPPING 迁证据

- 整目录 `git mv`（保留子路径名，`R100` 纯重命名）：`2614-topk-sparse-attn-13m` → `1.2.3/experiments/`；
  `2616-linear-attention-13m`、`2637-linear-attn-quality` → `1.2.2/experiments/`。
- 旧 A6 目录**未删**：`axes/A-mixing/A6-mixing-operator/` 仍保留 `README.md` + `experiments/README.md`，
  待 integrator 统一退役。
- 一格一份家、不复制：`1.2.2` 与 `1.2.3` 各自的 experiments 无交叉复制（见 `git diff --cached` 的 rename 列表）。
- `.scratch/**/evidence` 原地未动（本次未改任何 `.scratch` 证据文件）；`project.yaml` 的 `experiments`
  指针改指新家。
- `catalog.yaml` / `INDEX.md` 未改（工作树既有未提交改动保留、未回滚）。

### 硬约束③ 按 55-review 修正结论

- **purpose 目的非手段**：四项 purpose 均为痛点陈述（如 1.2.2「每 token 可见量远小于 T 的前提下覆盖全序列、
  保住对全局位点的可达」），手段在 `innovation` / `method`。
- **pareto 口径以 stock 全因果为 100%**：每项 `validation` 均写明「对照 = stock 全因果基线（归一为 100%）」。
- **判据与目的对齐、该涨/该跌/必须不变**：每项 `objective_criteria` 三态齐全；
  `validation` 含对照 + 客观判据 + 口径（check_tree 三要素门通过）。
- **否证条件与失败归因**：每项 `falsification` 非空；顶层 `failure_analysis` 键齐（未判失败写 `""`）。
- **证据锚点指向真实存在实验**：`1.2.2` 指 `2616/2637/analysis.md @76a07a8`、`1.2.3` 指
  `2614/analysis.md @3597039`，两锚点 `git cat-file -t` 均为 commit 且可达。
- **未照抄错误结论**：`1.2.2` 把旧「线性更差」降为「带伤/已改核下的负观测」（2616 停摆、2637
  数值带伤且稳定化会改核）；`1.2.3` 把「k=32 越 0.014 即有害」降为按来源自否的带重校准的边缘判定。

### 硬约束④ 字段齐全

每份 `project.yaml` 含：`role_map`（owner/writer/reviewer/integrator 四角色，reviewer ≠ writer）、
**有且仅有一个** `milestone`（map：output + acceptance）、`work_packages`（每条 id/what/goal/means/
acceptance/evidence/review/status，`review` 均写「跑完另派 fresh 复核」= 独立复核）、
`sota_baseline`/`gap`/`method`/`validation`/`objective_criteria`/`falsification`/`failure_analysis`/
`cost_model`（**只算等效理论计算量** FLOPs / layer-passes / tokens-per-forward，均含 pipelining 说明，
无墙钟/日期）。

### 硬约束⑤ status 枚举

仅用四值：`1.2.1`=未开始、`1.2.2`=进行中、`1.2.3`=进行中、`1.2.4`=未开始。无第五值。

### 边界

- 只写负责路径（`axes/1-time/1.2*`、`axes/1-time/README.md` 追加、`axes/7-references/README.md` 新建、
  `tests/test_linear_attn_quality.py` 路径随迁更新）；未碰其他轴目录（`2-depth`/`3-width`/…未建未动）。
- `axes/1-time/README.md` **只追加**（§1.2 子树表 + §1.3 参照行），未覆盖 wp1 已有内容。
- 旧格目录与旧 README（`axes/A-mixing/**`）未删除/未改。
- 未删任何证据（见「证据不丢失」）。

---

## 二、命令与结果

| 命令 | 结果 |
|---|---|
| `git mv` 三处实验目录 | 成功；`find axes -type f` 计数不变（2446） |
| `python3 tools/check_tree.py` | `OK：13 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| `python3 tools/summary.py --check` | 同上（exit 0） |
| `python3 -m pytest tests/ -q` | **827 passed in 92.05s**（0 失败；修 `test_linear_attn_quality.py` 的 EXP 路径后） |
| `git diff --cached --check` | exit 0（无空白错误） |
| `git diff --cached --name-only \| wc -l` | 92（6 add + 2 modify + 84 rename） |
| commit | `65adaec`，staged 列表核对后为 0 残留 |
| `find axes -type f \| wc -l` | 2452（≥ 2434 基线；多出 6 为 pytest 生成的 `__pycache__`，gitignore） |
| `git ls-files axes \| wc -l` | 1669 → 1675（+6 = 4 project.yaml + 2 README，均为新增跟踪文件） |

**证据不丢失**：迁移只做整树移动（`R100`），末位文件计数未减少；`.scratch/**` 与 `archive/` 未动。

---

## 三、未决项 / 风险

1. **`axes/7-references/README.md` 与其他轴 worker 的潜在写冲突**：本 WP 按指令登记了 §1.3 寻址
   （并列出 A2/RoPE/GQA-MQA/R7 的落点由各轴 worker 执行）。该文件为新建、极简；若轴 7 worker
   另建/覆盖，请 integrator 合并（1.3 一行必须保留）。
2. **`1.2.2` 的机制归属**：MAPPING §1/§3 与派发指令都把 A6 的线性注意力实验（2616/2637）归到
   `1.2.2`（块稀疏/局部+全局）。线性核是 O(T) dense-kernel 算子、并非「块稀疏掩码」族，二者被合并在
   同一研究项下；已在 `gap`/`argument` 明确写出「真块稀疏掩码未实装、线性核无伤质量点未取得」两个 WP，
   未自造结构。若出资人认为算子与掩码应分家，需上抛决策。
3. **旧 A6 README 的 `evidence` 仍指旧路径**：按边界不动旧 README，交由 integrator 退役处理；
   新 `1.2.x` 的 `experiments` 指针已全部指向新家。
4. **`git` 的 rename 报告有跨臂误配**（如 `2614/smoke/smoke-k0/gate_log.jsonl ⇄ 2616/.../smoke-full/...`）：
   这是内容相同的文件被启发式匹配所致，**磁盘内容正确**（已抽查三个 `analysis.md` 标题与内容对应无误）。
5. **`1.2.2`/`1.2.3` 只判「本尺度本实装下」**：长 T 回本、真块稀疏、无伤线性核、位置对照均未做，
   status 保持「进行中」，未宣称整项否证。
6. **未做独立复核**：本 WP 的代码/文档复核须由 fresh reviewer（≠ 本 writer）按 `REVIEW` 程序给出。

---

## 四、建议下一步

- 由 fresh reviewer 复核本提交（`65adaec`）的四个 `project.yaml` + 迁移完整性 + 1.3 登记。
- integrator 合并时确认 `axes/7-references/README.md` 的 1.3 行未被覆盖，并把旧 A6 目录统一退役。
