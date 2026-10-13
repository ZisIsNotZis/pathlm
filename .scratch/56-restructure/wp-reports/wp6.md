# WP6 · ticket 56 M2 轴 4（专家轴）+ 轴 5（量化轴）迁移报告

## 0. 人话摘要

这次干的事：在一片「研究树」里，把原来散在旧格子里的「专家（MoE）」和「量化」两块，按新规范
（`docs/structure.md` §12 的用词、每格一份 `project.yaml`）重新建成两个轴——专家轴 3 个研究项
（稀疏激活专家 / 路由与负载均衡 / 专家粒度）、量化轴 6 个研究项（KV 量化 / 权重量化 / 激活量化 /
低秩压缩 / fp8·大 head_dim / 窗口+量化+检索组合）。旧的「量化」格（R1）里 6 个实验目录、
「窗口+量化+检索」格（R4）里 1 个情报目录，用 `git mv` **整目录**搬到量化轴新家，一个字节都没丢、没复制。
同时按上一轮复核（ticket 55）的意见把旧结论修对：**不再把「手段」当研究题目**、**每个研究项写清「要解决什么痛点」**、
**判据改成「质量-代价前沿」（以 stock/fp16 全因果为 100%）而不是单点掉不掉门槛**、**把「造轮子」风险显式写进 gap**、
**把 R1 已作废的 int4 数字（逐值标尺口径）换成正确的块标尺口径**。搬完实验后，5 个引用旧路径的测试文件路径同步更新，
全套自检（`summary.py --check` / `check_tree.py` / `pytest`）全绿。

一句话结论：**轴 4/轴 5 的树建好了、证据搬到位了、旧错口径改对了；KV 量化（5.1）已有明确本地判决，
其余 8 项是预登记的待做研究项。** 旧格目录与旧 README 一律保留、未删，等 integrator 统一退役。

---

## 1. 本 WP 交付物

### 1.1 新建（9 个研究项 + 2 个轴 README + 2 个 experiments README）

| 编号 | 标题（exact wording） | 目录 | status |
|---|---|---|---|
| 4.1 | 稀疏激活专家（top-k 路由） | `axes/4-moe/4.1-sparse-experts/` | 未开始 |
| 4.2 | 路由与负载均衡 | `axes/4-moe/4.2-routing-balance/` | 未开始 |
| 4.3 | 专家粒度 / 异构（细粒度、共享专家） | `axes/4-moe/4.3-expert-granularity/` | 未开始 |
| 5.1 | KV 量化 | `axes/5-quant/5.1-kv-quant/` | 已完成（有本地判决） |
| 5.2 | 权重量化 | `axes/5-quant/5.2-weight-quant/` | 未开始 |
| 5.3 | 激活量化 | `axes/5-quant/5.3-activation-quant/` | 未开始 |
| 5.4 | 低秩压缩（MLA 式） | `axes/5-quant/5.4-low-rank/` | 未开始 |
| 5.5 | fp8 / 大 head_dim | `axes/5-quant/5.5-fp8-head-dim/` | 未开始 |
| 5.6 | 组合：窗口 + 量化 + 检索 | `axes/5-quant/5.6-window-quant-retrieval/` | 进行中（本地仅覆盖窗口/锚点腿） |

轴 README：`axes/4-moe/README.md`、`axes/5-quant/README.md`
（含本目录子树表、前沿口径、跨轴边界、迁移说明）。
experiments README：`axes/5-quant/5.1-kv-quant/experiments/README.md`、
`axes/5-quant/5.6-window-quant-retrieval/experiments/README.md`。

### 1.2 证据迁移（整目录 `git mv`，一格一份家、不复制）

- R1 → 5.1：`2614-kv-quant-eval-13m`、`2619-kv-budget-chart`、`2620-kv-budget-gaps`、
  `2621-kv-budget-rest`、`2631-kv-budget-causal`、`2632-kv-block-scale`
  （92 个 git 重命名条目，全部 100% 相似度 = 无内容改动）。
- R4 → 5.6：`2617-prior-art-hybrid-kv`。
- `.scratch/**/evidence` 原地未动；5.6 的 project.yaml 用指针引用
  `.scratch/07-x1-x3/evidence @7b47555` 与 `axes/1-time/1.1.4.2-compression/.../2614-...analysis.md @c39dc88`。
- 旧格目录与旧 README 全部保留（`axes/R-references/R1-quantization/README.md` +
  `experiments/README.md`、`axes/R-references/R4-window-kv-quant-retrieval/**`、
  `axes/W-width/W2-expertization/**`），未删、未改，等 integrator 统一退役。

### 1.3 按 55-review 的修正（不照抄旧结论）

- **W-01/W-02（目的丢失 / 手段当轴）**：轴 4 拆成 4.1/4.2/4.3 三个真问题；每题 `purpose` 写「要解决的痛点」
  （容量与计算解耦、路由坍缩、固定激活算力下的容量分配），不写手段名。
- **W-03/W-04（造轮子）**：4.1–4.3 与 5.1–5.6 的 `sota_baseline` + `gap` 强制声明「相对 SOTA 的增量」，并把
  「造轮子」列为最高风险写进 `gap`；4.1 明确「若给不出相对 SOTA 的实质增量，须降为参照、不立项」。
- **W-08 / R8（作废的 int4 数字 / 跨口径混用）**：5.1 一律用**块标尺（含标尺/staging 的部署字节）**口径；
  明写旧逐值标尺口径（int8=3 B/值 ×1.50）已撤回；明写**不引用** W3 的「同 −75% 档 GQA 便宜 3.2×」
  （其 int4 数字作废）；轴 5 README 单列口径纪律。
- **W-09（无 pareto 口径）**：5.1 的判据改为「每位置部署字节 → 序列级质量（+ 长程召回第二腿）」前沿；
  5.6 改为「每位置字节 → 两腿质量」。
- **W-13（状态自相矛盾）**：4.1 不再保留「todo 待补做」与「prior_art 检索不可用」并存的口径。
- **W-17（墙钟口径）**：所有 `cost_model` 只用等效理论量（FLOPs / layer-passes / tokens-per-forward），
  写清 pipelining，禁墙钟/日期；校验器已硬查通过。
- **R4 的「窗口几乎零代价」**：与其自家 M2 ERRATA 冲突（keep32=+0.212、keep64=+0.081），5.6 改记为
  「bpc 损伤随保留预算单调（keep32→192：+0.212→+0.004）」。

### 1.4 共享面同步（迁移的连带必要改动）

- `tests/test_kv_quant.py`、`tests/test_kv_budget_gaps.py`、`tests/test_kv_budget_rest.py`、
  `tests/test_kv_budget_causal.py`、`tests/test_kv_block_scale.py`：证据路径常量随目录迁移更新
  （否则 importlib 找不到语义层、pytest 必红）。与轴 1/2/6 worker 更新 `tests/*.py` 的先例一致。
- `tools/evalkit/kv_quant.py`：契约注释路径同步（单行 docstring）。

---

## 2. 逐项验收自证

| 硬约束 | 结果 | 证据 |
|---|---|---|
| ① 用词 = `docs/structure.md` exact wording | 满足 | 9 个 `title` + `dir` 与 §12/§14 逐字一致；轴名用 §4/§5 的「4 专家轴（MoE）」「5 量化轴」 |
| ② 按 MAPPING 迁证据（整目录 git mv、一格一主家） | 满足 | 92 个 100%-similarity rename；R1 六目录 → 5.1、2617 → 5.6；`.scratch` 原地未动，仅更新指针；未复制 |
| ③ 按 55-review 修正结论 | 满足 | 见 §1.3（W-01/02/03/04/08/09/13/17 + R8 + R4 窗口口径） |
| ④ 每个研究项写全 schema 字段 | 满足 | `summary.py --check` OK（45 个 project.yaml）；`role_map` 四角色、单 `milestone`（output+acceptance）、`work_packages` 每条 what/goal/means/acceptance/evidence/review(独立复核)/status、`sota_baseline`/`gap`/`method`/`validation`/`objective_criteria`/`falsification`/`failure_analysis`/`cost_model` 齐 |
| ⑤ status 仅四值 | 满足 | 4.1/4.2/4.3/5.2/5.3/5.4/5.5 = 未开始；5.1 = 已完成；5.6 = 进行中 |
| 边界：不改 catalog.yaml/INDEX.md、不回滚 | 满足 | `git status` 只余迁移前既有改动（INDEX.md、catalog.yaml、C-vertical README）未提交、未回滚 |
| 边界：不碰其他轴目录 | 满足 | 只写 `axes/4-moe/`、`axes/5-quant/`、5 个测试文件、`tools/evalkit/kv_quant.py` |
| 边界：不删旧格目录/README/证据 | 满足 | `axes` 跟踪文件数 1714（HEAD）→ 1714（index），不变；旧 README 全在 |
| 自检全绿 | 满足 | 见 §3 |

---

## 3. 命令与结果

```
python3 tools/summary.py --check   → OK：45 个 project.yaml 全部符合 docs/research-schema.md。（exit 0）
python3 tools/check_tree.py        → OK：45 个 project.yaml 全部符合 docs/research-schema.md。（exit 0）
python3 -m pytest tests/ -q        → 827 passed in 122.53s
（迁移后聚焦复跑）pytest tests/test_kv_*.py -q → 87 passed
git diff --cached --check          → 无空白错误（提交前）
git diff --check HEAD~1 HEAD       → clean
git ls-tree -r HEAD~1 --name-only axes | wc -l → 1714（迁移前跟踪基线）
git ls-files axes | wc -l（提交后）        → 1727 = 1714 + 13（重命名不改变计数，新增 13 个 axes 文件）
find axes -type f | wc -l           → 2504（≥ 迁移前工作树基线 2434，未减少）
```

> 说明：`git show --stat HEAD` = 111 files changed（92 rename + 19 created/modified），721 insertions / 12 deletions。
> 12 deletions 全部来自 5 个测试文件与 `tools/evalkit/kv_quant.py` 的单行路径替换。

提交：
```
ticket 56 M2 轴 4/轴 5：建专家轴 3 研究项 + 量化轴 6 研究项 + 迁 R 轴证据 + 按 55-review 修正
111 files changed, 721 insertions(+), 12 deletions(-)
commit d76d997（显式路径 git add；未用 -A/-u/.；提交后 git diff --cached --name-only = 0）
```

---

## 4. 未决项 / 残余风险

1. **迁移后的实验目录内部旧路径未重写**（沿用其他轴 worker 的先例）：如
   `5.1/experiments/2619-kv-budget-chart/run.py` 仍硬编码 `axes/W-width/W3-kv-head-sharing/...`、
   `axes/M-carrier/M2-carrier-compression/...` 等旧路径；这些**不阻塞 pytest**（语义层模块无路径依赖），
   但会让 R1 脚本的手动重跑失效。建议 integrator 在退役旧目录时统一改写（或标注为「历史产物」）。
   轴上 1/2/6 迁入的实验同样留了 225 处旧轴名引用，属同一批残留。
2. **5.1 的 `status: 已完成`**：旧格 R1 原为 `todo`；本项按「KV 量化块标尺口径已有明确本地判决（int8 近无损省 46.875%、
   int4 有损）」判 `已完成`，`conclusion` 非空、`experiments` 非空、且与迁移型 3.2.1 的先例一致。若复核认为应在
   `5.5` 的 fp8/大 head_dim 缺口也补完后才收口，可降为 `进行中`（一句话改动）。
3. **5.6 的本地证据多为他线指针**（`.scratch/07-x1-x3/evidence` 与 `1.1.4.2` 的 M2 保留曲线），
   本项自跑的组合臂与检索腿仍缺 —— 已在 `conclusion`/`gaps` 如实标注。
4. **`tests/evidence/longT_step0_anchors.json` 等共享证据未涉及 kv**（已确认无引用），无需改动。
5. **外部检索不可用**：所有外部 SOTA 条目仍标 `记忆·未核`，只作设计方向、不作判据（沿用 PRIOR_ART 纪律）。

---

## 5. 建议下一步

- 交**独立复核**（fresh 上下文）核对本 WP 的 9 个 `project.yaml` 字段与结论是否照 55-review 修正到位、
  证据归属是否「一格一份家」。
- integrator 择机统一：(a) 退役旧 `axes/R-references/**`、`axes/W-width/**` 与旧 README；
  (b) 决定是否批量重写迁移实验目录内的旧轴路径引用。
