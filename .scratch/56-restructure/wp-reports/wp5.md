# WP5 交付报告 · ticket 56 M2 · 轴 3 宽度轴

> 角色：worker（fresh 上下文，单一写入者）｜时间：2026-10-13T10:57:34+08:00
> 负责范围：`axes/3-width/` 全部 9 个研究项 + 分组 README 3.1/3.2 + 轴 README；
> 迁移证据 W1→3.1.1、W4→3.2.1。
> 提交：`1dcb5ce`（122 files changed, 656 insertions，含 109 个 100% rename）。

## 0. 人话摘要（先讲做了什么、值不值）

原来的「宽度轴（W）」被 55 号复核判成**四件事硬塞进一个轴**：减宽（省算力但掉容量）、MoE（保容量
减计算）、GQA/MQA（省显存）、读出几何（读出怎么映射），四者代价口径互斥、且全是业界做透的既有
结构旋钮，没有一个「目的」。这一轮把它按冻结结构拆开重挂：

- **本轴只收两件事**：「每个 token 用多少宽度跑完」（3.1）与「读出怎么把状态变成词」（3.2）。
- **专家化（MoE）归专家轴 4、KV 头共享归参照台账轴 7**：本轴不接收、不复制，只在轴 README 里
  登记边界（它们的证据由轴 4 / 轴 7 的 worker 迁）。
- **9 个研究项**：3.1.1 uniform 减宽/width dropout（旧 W1 迁入）、3.1.2 逐层宽度变化、3.1.3 同层内
  位置相关宽度、3.1.4 前馈隐藏比、3.1.5 通道极化/稀疏激活；3.2.1 读出的重参数化（旧 W4 迁入）、
  3.2.2 逐出口读出、3.2.3 tied vs untied。每一项都写全了目的、相对 stock 改了什么、SOTA 基线、
  端到端验证（该涨/该跌/必须不变）、否证条件、失败归因字段、单一 milestone、带独立复核的
  work_packages，成本只用等效理论量（FLOPs / layer-passes / tokens-per-forward）。
- **最关键的修正**：旧 W1 说「减宽=减容量」——但复核查出它**唯一**的本地实验其实测的是「读出只读
  前 360 维」的读出切片（既不减算力也不减参数，属读出域），是拿别的变量的读数当减宽的判决。这一轮
  没有照抄，3.1.1 如实标成**未开始**（命题根本没本地测过），把那条证据登记为背景并让 3.2.1 指针引用。
- **值不值**：这一轮的产出是**结构正确 + 判据正确**，不是新实验结论。9 项里 8 项仍是「未开始」，
  1 项（3.2.1）是旧 W4 的既有结论（读出重参数化的增益来自容量/重参数化、不是非线性、且只补本项目
  depth-AR 形态的缺陷，不是通用改进）。

## 1. 交付物（逐项验收自证）

### 1.1 目录与用词（`docs/structure.md` §12/§14）

| 节点 | 目录 | 粒度 | 用词核对 |
|---|---|---|---|
| 3.1 每 token 的激活宽度 | `axes/3-width/3.1-active-width/` | 分组 README | ✅ 与 §12 逐字一致 |
| 3.1.1 uniform 减宽 / width dropout | `3.1.1-uniform-narrowing/` | 研究项 | ✅ |
| 3.1.2 逐层宽度变化：金字塔 / 沙漏 | `3.1.2-layer-width/` | 研究项 | ✅ |
| 3.1.3 同层内位置相关宽度：前 1024 与后 1024 意义不同 | `3.1.3-position-width/` | 研究项 | ✅ |
| 3.1.4 前馈隐藏比：FF/MLP 的 hidden ratio，及其逐层变化 | `3.1.4-ffn-ratio/` | 研究项 | ✅ |
| 3.1.5 通道极化 / 稀疏激活：只激活部分通道 | `3.1.5-channel-sparsity/` | 研究项 | ✅ |
| 3.2 读出 | `axes/3-width/3.2-readout/` | 分组 README | ✅ |
| 3.2.1 读出的重参数化：加非线性 / GLU 式 / 线性 | `3.2.1-readout-reparam/` | 研究项 | ✅ |
| 3.2.2 逐出口读出：early exit 或多出口时，每个出口怎么映射到词表 | `3.2.2-per-exit-readout/` | 研究项 | ✅ |
| 3.2.3 tied vs untied embedding | `3.2.3-tied-untied/` | 研究项 | ✅ |

标题用词已用脚本逐条与 `docs/structure.md` §12 轴 3 表比对，8/8 exact match；`dir` 与实际路径一致
（`check_tree.py` 门）。轴名一律 `"3 宽度轴"`。

### 1.2 证据迁移（MAPPING §0/§1，一格一份家，只指针不复制）

- `git mv axes/W-width/W1-active-width/experiments/2525-tax-attribution → axes/3-width/3.1.1-uniform-narrowing/experiments/`
- `git mv axes/W-width/W4-readout-geometry/experiments/{2610-gated-readout-13m-depthar, 2611-equalparams-linear-control-13m, 2612-gated-readout-24k-13m, 2613-readout-reparam-dense-13m, 2621-per-depth-readout-13m} → axes/3-width/3.2.1-readout-reparam/experiments/`
- 迁移前后 `find axes -type f | wc -l` = 2478 → 2478（git mv 保号），提交后 = 2491
  （+13 = 1 轴 README + 2 分组 README + 8 project.yaml + 2 experiments/README.md）。
- 工作树文件计数口径：`find axes -type f | wc -l` = **2491**；跟踪文件计数：`git ls-files axes | wc -l`
  = **1714**（均 ≥ MAPPING §0 基线；两口径并记，避免清 cache 漂移误判）。
- `.scratch/**/evidence/**` 原地不动，只更新 `project.yaml` 指针（`.scratch/25-tax-attribution/evidence @2e86924`、
  `.scratch/23-depth-ar/evidence @1586745`，二锚点均 `git rev-parse` 可达）。
- 旧格目录与其 README（`axes/W-width/**`，含 `W2`/`W3`/`PRIOR_ART.md`）**保留未删**，待 integrator 统一退役；
  W2/W3 证据本轴不接收、不复制。

### 1.3 每项 `project.yaml` 字段齐全（`docs/research-schema.md` §2）

9 份均含 `id/axis/title/dir/purpose/innovation/expected_effect/sota_baseline/gap/method/validation/
objective_criteria/falsification/cost_model/role_map/milestone/work_packages/roadmap/status/
conclusion/failure_analysis/conclusion_review/experiments/argument/gaps/started/ended/change_log`：
- `role_map` 四角色齐（owner/writer/reviewer/integrator；reviewer 均为 fresh、必非 writer）。
- `milestone` 有且仅有一个（map，含 `output` + `acceptance`）。
- `work_packages` 每条含 `what/goal/means/acceptance/evidence/review/status`；`review` 一律写「另派
  fresh 复核（`SCHEMA.md` §14/§14b），复核排在执行之后」。
- `cost_model` 只用等效理论量并写清 pipelining；无墙钟/日期（校验器 regex 门）。
- `status` 仅四值：3.1.1–3.1.5、3.2.2、3.2.3 = 未开始；3.2.1 = 已完成（有 conclusion + 存在的 experiments）。
- 3.2.1 `failure_analysis` 非空（区分：创新本身失败——非线性无额外贡献；结构问题——只补本 depth-AR
  形态读出缺陷、不可迁移 dense）。

### 1.4 按 55-review 修正（`W-PROBLEMS.md` 逐条）

| 条 | 处置 |
|---|---|
| W-01/W-02/W-05 整轴无目的 | 轴 README + 两组 README + 每项 `purpose` 明写痛点；前沿口径 = 以 stock 全宽为 100% |
| W-03/W-18 W2/W3 参照混入、未 migrate | 轴 README「边界」表声明 W2→轴 4、W3→轴 7；本轴不接收不复制 |
| W-06 W1 命题-证据错配 | 3.1.1 不照抄旧结论：`conclusion_review`/`gap`/`argument`/`experiments/README` 明写迁入的
`2525` 测的是读出切片、属 3.2.1 域、不测减宽 ⇒ `status: 未开始`、`conclusion` 空 |
| W-07 减容量已判死 | 3.1.1 不再断言「减宽=减容量」；把它列为 SOTA 共识与造轮子风险，判据改为前沿 + 等参数对照 |
| W-09 无 pareto 口径 | 全部 9 项 `validation`/`objective_criteria` 明写「以 stock 为 100% 的质量-代价前沿 + 该涨/该跌/必须不变」 |
| W-10 W4 自认无价值 | 3.2.1 结论明写「不采纳为通用质量机制、只补本 depth-AR 形态读出缺陷」，并保留 WP4（更长训练/更大尺寸） |
| W-11 W4 造轮子/平凡结论 | 3.2.1 `gap`/`argument` 声明与 GLU 同族、「加 2d² 参数使 bpc 降」平凡；判据改为等参数对照归因 + 可迁移性 |
| W-13/W-16 状态自相矛盾 | 3.1.1 状态统一为未开始（catalog 旧 todo/51-tree 旧 judged 不再继承） |
| W-14 不可核实证据 | 剔除「用户本地独立实验（减宽）」引用；`gaps` 注明若确有须补落盘证据 |
| W-15 PRIOR_ART 残留 | 旧文件保留（待退役），本轴新文档不引用其 W5 段 |
| W-17 墙钟口径 | `cost_model` 全改等效理论量，并写明 pipelining |

## 2. 命令与结果

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | `OK：36 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| `python3 tools/check_tree.py` | 同上（exit 0） |
| `python3 -m pytest tests/ -q` | `827 passed in 98.10s`（0 失败） |
| 标题用词比对脚本（vs `docs/structure.md` §12 轴 3 表） | 8/8 exact match；`dir` 一致 |
| `find axes -type f \| wc -l` | 2478（迁移前）→ 2478（git mv 后）→ 2491（含新增文档） |
| `git ls-files axes \| wc -l` | 1714 |
| `git diff --cached --name-only \| wc -l`（提交前核对） | 122（109 rename + 13 add，全部在 `axes/3-width/`、`axes/W-width/`） |
| 提交后 `git diff --cached --name-only \| wc -l` | 0（无遗留 staged） |
| `git diff --check HEAD~1 HEAD` | 无空白错误（exit 0） |
| `git show --name-only 1dcb5ce` 过滤非轴 3/轴 W 路径 | 空（未误伤 `INDEX.md`/`catalog.yaml`/`axes/C-vertical`） |

提交：`git commit -F - <<heredoc`（未用 `-m`/反引号；显式路径 `git add axes/3-width`，未用 `-A/-u/.`；
未对 `git add` 用 `2>/dev/null`）。

## 3. 未决项 / 需复核裁定的选择

1. **3.2.1 的状态枚举**：旧 W4 为 `judged`；本轮判 `已完成`（本地已有明确结论，含否定性结论）。
   备选是 `被否证`（假设「读出非线性解耦带来通用质量增益」已被等参数对照否证）。本报告选 `已完成`
   以贴合「已有明确结论」，未选更强断言。若复核认为「被否证」更准，可改（会连带要求
   `failure_analysis`，本项已非空，改后仍过校验）。**属我按 schema 语义的取值范围选择，非产品决策。**
2. **2525 的物理落点（W-06 与 MAPPING 的张力）**：55-review W-06 说 `2525-tax-attribution` 属读出域
   （3.2.1），但 MAPPING §1 明确把 W1 的证据去向写成 `3.1.1`。本轮**按 MAPPING 迁移**（物理落
   `3.1.1/experiments/`），同时按 55-review 修正结论（不当作减宽判决，登记为背景、3.2.1 只指针引用）。
   若复核认为应改落 `3.2.1/experiments/`，是一次涉及两条 `project.yaml` 的指针/路径小改，回滚成本低。
3. **3.1.1 是否应保留**：55-review D 段建议 W1 与已删的 W5/R3 同判（降/删）。但 `docs/structure.md`
   §12 冻结 3.1.1 为研究项，故本轮保留为研究项并把「造轮子风险最高」写进 `gap`；是否降格属框架性
   取舍，留给轴 lead/出资人。
4. **轴 4 / 轴 7 的接收**：W2、W3 的证据（`W2/experiments` 空、`W3/experiments/2616-gqa-mqa-13m`）仍
   留在 `axes/W-width/**`，等轴 4 / 轴 7 的 worker 迁走；本轮未动（不在本 WP 范围）。
5. **group README 未含「怎么读」小节**：与 6.3 分组 README 保持一致（轴 README 已给入口）。

## 4. 验收自评（对应 criterion）

- criterion-1（不越界）：改动仅 `axes/3-width/**`（新建）与 `axes/W-width/**`（git mv 源）；
  未改 `catalog.yaml`/`INDEX.md`/`axes/C-vertical`（工作树既有未提交改动保留、未回滚）；
  未碰其他轴目录；未删任何证据或旧 README。
- criterion-2（证据充分）：命令与结果见 §2；提交 `1dcb5ce`；逐条自证见 §1。
