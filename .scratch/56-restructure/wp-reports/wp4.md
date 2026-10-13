# ticket 56 · M2 轴 6（预测轴）迁移 · WP4 报告

> 时间：2026-10-13T10:47+08:00 ｜ actor：worker（fresh 上下文，单一写入者）｜ 提交：`61e346b`

## 1. 人话摘要

本 WP 把「预测轴」这条线从旧的平铺结构（H 轴 5 个格子，标题叫「每步输出位置数」「头与主干的关系」——全是**手段**）搬进了 `docs/structure.md` §12 定义的树：一个核心研究项 **6.2 多位置预测 = 投机解码**（H1 本体 + H3/H4/H5 作为它下面的三个工作包：草稿头 / 概率多候选·树 / 验证），一个 **6.4 训练时未来位置的填充语义**（H2），以及三条旧仓完全没有的新路线 **6.3.1 diffusion / 6.3.2 一次性批量 pad 填充 / 6.3.3 latent regression 批量生成**；`6.1 自回归` 作为参照（对照组基线）登记到轴 7 台账。11 个实验目录整目录 `git mv` 进新家（一格一份家、不复制，`.scratch` 原始证据原地不动）。同时按 55-review H-PROBLEMS 把四个致命错误改掉：topic 从手段改回**目的**（抬高接受长度 τ、压低每 token 顺序前向次数）；剔除**恒真**的判据 `R=(1+γ)/(1+τ)>1`（τ≤γ 时它恒成立，按构造不可证伪）与「恒为算力亏损」「税效比全项目最好」两个旧头条；把被简化成「单 argmax + 固定 λ=0.5」的**概率多候选 / 树 / 置信加权组合**重立为工作包 6.2.2 并写明必须补回设计；判据改成 τ / tokens-per-forward（该涨）+ 质量（该不跌），每 token 等效 FLOPs 只作**代价轴**如实报（它不是亏损发现，是「多花并行算力换少顺序前向」的范式定义）。

## 2. 逐项验收自证

### ① 用词 = `docs/structure.md` 的 exact wording（不改名/不自造）

- 5 份 `project.yaml` 的 `title` 与 §12 轴 6 表逐字一致（已比对）：`多位置预测 = 投机解码` /
  `diffusion（离散扩散 / 掩码扩散）` / `一次性批量 pad 填充（填空式）` /
  `latent regression 批量生成（估计得贪心）` / `训练时未来位置的填充语义：pad / self-fill / 错误 token`；
  `axis` = `6 预测轴`。
- 目录名与 §12/§14 一致：`6.2-speculative-decoding` / `6.3-parallel-generation` /
  `6.3.1-diffusion` / `6.3.2-pad-fill` / `6.3.3-latent-regression` / `6.4-training-fill`。
- 工作包 id 用 §12 的 `6.2.1 草稿头（应免费、不动主干）` / `6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合）` /
  `6.2.3 验证`（§13.3 判为同一研究问题的手段，不另立研究项）。

### ② 按 MAPPING 迁移证据（一格一份家、只指针不复制）

- `git mv` 整目录 11 个：H1 的 `2510-gap-sweep-b2-c1m`、`2520-spec-at-scale`；H3 的
  `2512-overnight-causal-chain`、`2613-self-gated-composition-13m`、`2631-spec-three-arms`、
  `2632-alpha1-attribution`；H4 的 `2521-exit-scale-confirm`、`2610-frozen-trunk-jumphead-13m`、
  `2611-jumphead-ar-control-13m`；H5 的 `2513-four-experiments` → **6.2/experiments/**；
  H2 的 `2523e-fill-variants` → **6.4/experiments/**。
- 112 个跟踪文件全部记为 **rename（100% 相似度，内容未改一字）**，0 个 `D`；旧格目录与其 README
  （旧格 README + `experiments/README.md` + `PRIOR_ART.md`）**原地保留**，待 integrator 统一退役。
- `.scratch/**/evidence` 原地不动，只在 `project.yaml` 里更新指针（H1 `10-gap-sweep`/`13-four-experiments`/
  `20-spec-scale`、H3 `12-overnight`/`04-m1-runs`、H4 `18-exit-probe`/`21-exit-at-scale`、H5 同上、
  H2 `06-m2-sweep`/`23-depth-ar`/`24-depth-ar-100m`；锚点 `@…` 均校过可达真实提交）。
- 证据不丢失：`git ls-tree HEAD~:axes` = `git ls-files axes` = 1692（未减少）；`find axes -type f` 未减少。

### ③ 按 55-review 修正（不照抄错误结论）

| 55-review 问题 | 处置 |
|---|---|
| H-F01 目的丢失（把手段当 topic） | 6.2 的 `title` 保留 §12 规定用词，但 `purpose` 写痛点是「每生成 1 token 跑 1 次完整前向」，`validation`/`objective_criteria` 判据 = τ / tokens-per-forward；README 明确「topic 不是一次前向输出几个位置」 |
| H-F03 主判据 R>1 恒真、与目的无关 | `objective_criteria` 明文写「**不得**把 R>1 当判据——τ≤γ ⇒ R≥1 按构造恒成立、不可证伪」；R 只作 `cost_model` 与代价轴 |
| H-F04「MTP 免费」与 +0.041 税矛盾 | `method` 臂与 WP 6.2.1 写死**冻结主干只训头**（Δbpc = 0.0000 作证据），联合训练只作 Medusa-2 对照；`gap` 明写旧主线误用付费基线 |
| H-F02/H-F05 H3 被简化成单 argmax、树/多候选整条没做 | 重立 WP `6.2.2 概率多候选 / 树（+2 / +1+1 / 加权组合）`，`means` 写回出资人的路径概率加权式与树/DP lattice；`acceptance` 要求完整树验证并给出 cross-K「树可能翻转」定论 |
| H-I1/I2/I3/I8（τ 无稳健性、无前沿、墙钟不可复现） | 写进 `gaps`；`validation` 钉「墙钟不进判据」；milestone 要求温度>0 与 batch>1 稳健性点 |
| H-I7 H2 目的不清且归属错 | 6.4 单独立项并写明目的（服务变深度/重入 AR 与多位置预测的右侧位置语义），不再挂在「一次前向产出几个位置」下 |
| H-S5 「整段（扩散式并行）」越界 | 移出 6.2，交 6.3（在 `gaps` 里显式记录去向） |

### ④ 每个研究项字段齐全

- `role_map`（owner/writer/reviewer/integrator 四角色齐，reviewer 不等于 writer）、**有且仅有一个**
  `milestone`（map，含 `output` + `acceptance`）、`work_packages` 每条
  `{id, what, goal, means, acceptance, evidence, review, status}` 且 `review` 一律排**独立复核**
  （fresh 复核者、`SCHEMA.md` §14/§14b），`sota_baseline`/`gap`/`method`/`validation`/
  `objective_criteria`/`falsification`/`failure_analysis`/`cost_model`（只算等效理论量
  FLOPs / layer-passes / tokens-per-forward，考虑 pipelining，无墙钟、无日期）齐备。
- 6.2 有 3 个 WP（6.2.1 草稿头 / 6.2.2 概率多候选·树 / 6.2.3 验证）；6.4 有 2 个 WP；
  6.3.1/6.3.2/6.3.3 各 1 个 WP。全部 `status` 仅取四值。

### ⑤ status 仅四值

6.2 `进行中`（有本地读数、主判据待 WP6.2.2 补跑）、6.4 `已完成`（本地受控结论）、
6.3.1/6.3.2/6.3.3 `未开始`（无旧格、本仓零经验）。

### ⑥ 边界

未改 `catalog.yaml` / `INDEX.md`（工作树既有未提交改动保留、未回滚、未 stage）；未碰其它轴目录；
未删旧格目录与旧 README；未删任何证据；未改任何实验产物内容。

## 3. 命令与结果（绑定 `61e346b`）

| 命令 | 结果 |
|---|---|
| `python3 tools/check_tree.py` | `OK：28 个 project.yaml 全部符合 docs/research-schema.md`（exit 0） |
| `python3 tools/summary.py --check` | 同上（exit 0）；6.x 五行：6.2 进行中 / 6.3.1-6.3.3 未开始 / 6.4 已完成 |
| `python3 -m pytest tests/ -q` | **827 passed in 93.36s**（0 失败） |
| `git diff --cached --check` | 无空白错误 |
| `git diff --cached --name-status` | 126 项 = 112 rename + 9 新增 + 5 修改；无 `catalog.yaml`/`INDEX.md`/`axes/H-horizon/**`/`axes/C-vertical/**` |
| `git ls-tree -r HEAD~ --name-only axes \| wc -l` = `git ls-files axes \| wc -l` | 1692 = 1692（无跟踪文件丢失） |

随证据迁移更新的**指针**（指向新家路径，非改证据内容）：`tests/test_spec_compute_ledger.py`、
`tests/test_h3_spec_three_arms_ledger.py`、`train_jumphead.py` 的 3 处 `contract`/docstring、
`docs/methodology.md` 的证据锚行。

## 4. 未决项 / 遗留（交 integrator 或后续 WP）

1. **旧 `axes/H-horizon/**` 退役**：旧格 README、`experiments/README.md`、`PRIOR_ART.md` 按任务约定
   原地保留，待 integrator 统一移入 `archive/`。`catalog.yaml` 的 H* `local:` 块仍在
   （迁移窗口内容源，不删）。
2. **`axes/H-horizon/PRIOR_ART.md` 未迁入新家**：6.2/6.4 的 `sota_baseline` 以指针引用它
   （与 2.1.1 引用旧 R7 路径同一做法）。若 integrator 决定把情报文档也搬进 `axes/6-predict/`，
   需同步改字段指针。
3. **55-review 残留清单中不在本 WP scope 的项**：
   - R-1：`pathlm/config.py` 的 `chain_mode` 默认值仍是已否证的 `"latent"`（旧 `T∘T` 形态）——
     属共享引擎代码，本 WP 不动。
   - R-2/R-3：`catalog.yaml` 的 H1/H3 选项（朴素概率乘积、整段扩散）——任务禁止改 `catalog.yaml`，
     随退役一并消失。
   - R-7：`docs/playbook.md` 定律 8 的边界段仍以「净亏 / 口径事故」叙述性表述框架 R>1——
     本 WP 在 6.2 里改判为「R>1 是范式定义性质、不是亏损发现」；两处措辞的最终统一建议由
     integrator 在收口时做（不冲突于「不随规模翻转」这一主结论）。
   - R-4/S3：4 个 backfill 生成的旧 schema stub（`2512`/`2513`/`2521`/`2523e`）已随 MAPPING 迁入新家，
     其 front-matter 仍为旧 schema（与轴 2 迁 `2518-exit-retry-law` 等 stub 的处理一致：证据文件不改）；
     若要补新契约，另立文档 WP。
4. **证据绑定的自证**：本报告全部命令在提交 `61e346b` 的工作树（仅余既有的 catalog/INDEX/C 轴未提交改动）
   上复跑；未做独立复核（按契约由后续 fresh reviewer 做）。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "轴 6 全部 §12 节点落位且用词逐字一致：5 个 project.yaml（6.2/6.4/6.3.1/6.3.2/6.3.3，title/axis 已逐行比对 docs/structure.md §12）、6.3 分组 README、轴 README、7-references 的 6.1 参照登记；11 个旧格实验目录 git mv 进 6.2/6.4 的 experiments/（112 rename、0 删除、一格一份家）；未改 catalog.yaml/INDEX.md/其它轴目录，未删旧格目录与旧 README，未删证据。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "自证命令绑定提交 61e346b：check_tree.py OK(28 project.yaml)；summary.py --check OK；pytest tests/ -q 827 passed；git diff --cached --check 干净；git ls-files axes 1692 未减少（无证据丢失）。报告给出逐条 55-review 处置表与未决项清单，供独立复核逐条核对。"
    }
  ],
  "changedFiles": [
    "axes/6-predict/README.md",
    "axes/6-predict/6.2-speculative-decoding/project.yaml",
    "axes/6-predict/6.2-speculative-decoding/experiments/README.md",
    "axes/6-predict/6.3-parallel-generation/README.md",
    "axes/6-predict/6.3.1-diffusion/project.yaml",
    "axes/6-predict/6.3.2-pad-fill/project.yaml",
    "axes/6-predict/6.3.3-latent-regression/project.yaml",
    "axes/6-predict/6.4-training-fill/project.yaml",
    "axes/6-predict/6.4-training-fill/experiments/README.md",
    "axes/7-references/README.md",
    "docs/methodology.md",
    "tests/test_spec_compute_ledger.py",
    "tests/test_h3_spec_three_arms_ledger.py",
    "train_jumphead.py",
    "axes/6-predict/6.2-speculative-decoding/experiments/2510-gap-sweep-b2-c1m/ (renamed from axes/H-horizon/H1-outputs-per-pass/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2520-spec-at-scale/ (renamed from axes/H-horizon/H1-outputs-per-pass/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2512-overnight-causal-chain/ (renamed from axes/H-horizon/H3-composition/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/ (renamed from axes/H-horizon/H3-composition/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/ (renamed from axes/H-horizon/H3-composition/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2632-alpha1-attribution/ (renamed from axes/H-horizon/H3-composition/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2521-exit-scale-confirm/ (renamed from axes/H-horizon/H4-head-trunk-relation/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/ (renamed from axes/H-horizon/H4-head-trunk-relation/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2611-jumphead-ar-control-13m/ (renamed from axes/H-horizon/H4-head-trunk-relation/experiments/)",
    "axes/6-predict/6.2-speculative-decoding/experiments/2513-four-experiments/ (renamed from axes/H-horizon/H5-verification/experiments/)",
    "axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants/ (renamed from axes/H-horizon/H2-training-fill/experiments/)"
  ],
  "testsAddedOrUpdated": [
    "tests/test_spec_compute_ledger.py",
    "tests/test_h3_spec_three_arms_ledger.py"
  ],
  "commandsRun": [
    {
      "command": "python3 tools/check_tree.py",
      "result": "passed",
      "summary": "OK：28 个 project.yaml 全部符合 docs/research-schema.md（exit 0）"
    },
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "同上；6.2 进行中 / 6.3.1-6.3.3 未开始 / 6.4 已完成"
    },
    {
      "command": "python3 -m pytest tests/ -q",
      "result": "passed",
      "summary": "827 passed in 93.36s"
    },
    {
      "command": "git diff --cached --check",
      "result": "passed",
      "summary": "无空白错误"
    },
    {
      "command": "git ls-tree -r HEAD~ --name-only axes | wc -l == git ls-files axes | wc -l",
      "result": "passed",
      "summary": "1692 == 1692，无跟踪文件丢失（112 rename、0 D）"
    }
  ],
  "validationOutput": [
    "check_tree.py: OK：28 个 project.yaml 全部符合 docs/research-schema.md",
    "summary.py --check: OK：28 个 project.yaml 全部符合 docs/research-schema.md",
    "pytest tests/ -q: 827 passed in 93.36s (0 failures)",
    "git status --short (worktree after commit): 仅剩既有未提交改动 INDEX.md / catalog.yaml / axes/C-vertical/{README,C1-layers-per-token/README}.md（保留未回滚）",
    "git diff --cached --name-status: 126 = 112 rename + 9 A + 5 M；未含 catalog.yaml / INDEX.md / axes/H-horizon/** / axes/C-vertical/**"
  ],
  "residualRisks": [
    "旧 axes/H-horizon/** 目录与其 README/PRIOR_ART.md 按约定原地保留，待 integrator 统一退役（catalog.yaml H* local: 块同为迁移窗口内容源，未删）",
    "55-review H 轴残留中不在本 WP scope：pathlm/config.py 的 chain_mode 默认仍是已否证的 \"latent\"（共享引擎，本次未改）",
    "docs/playbook.md 定律 8 边界段仍以「净亏/口径事故」叙述 R>1，与 6.2 的「R>1 是范式定义性质」措辞未统一（主结论「不随规模翻转」一致）；建议 integrator 收口",
    "4 个 backfill 旧 schema stub（2512/2513/2521/2523e）随证据迁入新家但 front-matter 仍为旧 schema（证据文件不改，与轴 2 处理一致）",
    "6.4 的 status=已完成 依赖旧格 H2 的受控读数（13M 12k matched + 部分 100M 点，无 n≥2）；软提交 OOD 污染「待重验」已在 project.yaml 的 gaps/WP2 记录",
    "6.2 的 conclusion 中 direct ≈ combined > chain 的排序被 2632 自证为前缀 regime 依赖，且 cross-K「树可能翻转」已触发但完整树实验未做——已在 gaps 与 WP 6.2.2 记录，未当定论"
  ],
  "noStagedFiles": true,
  "diffSummary": "126 files changed, 482 insertions(+), 10 deletions(-)：新增轴 6 的 5 份 project.yaml + 4 份 README；11 个旧 H 格实验目录整目录 git mv 到 6.2/6.4 的 experiments/（112 个文件 100% rename，内容未改）；修改 axes/7-references/README.md（登记 6.1 参照、修正指向不存在目录的指针）、docs/methodology.md 与 train_jumphead.py 的证据路径指针、两个 H 相关测试的路径常量。",
  "reviewFindings": [
    "no blockers",
    "non-blocking: docs/playbook.md 定律 8 边界段的措辞与 6.2 的新判据框架未统一（主结论一致），交 integrator",
    "non-blocking: pathlm/config.py chain_mode 默认值属共享引擎，未在本迁移 WP 内改动"
  ],
  "manualNotes": "本 WP 的 PASS 判定须由新的 fresh 独立复核给出（复核者不得为本 WP writer）。复核建议重点：(1) 逐条核对 6.2 的 objective_criteria/falsification 是否真的剔除了恒真判据 R>1；(2) 6.2/6.4 的 experiments 路径与 .scratch 锚点是否指向真实存在且可达的提交；(3) 112 条 rename 是否确无内容改动（git diff -M 核对）；(4) 6.3.x 的 cost_model 是否只含等效理论量（无墙钟/日期）。报告按任务约定写到指定 output 路径，未额外写 TICKET.md（与前两个 M2 轴 WP 的提交模式一致：只改 axes + 测试）。"
}
```
