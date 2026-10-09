# 50-doc-ssot · 项目日志（文档 SSOT 化）

> **性质**：日志文档。允许日期流水，记录「我们一步一步怎么做的」。
> 这里是过程记录，**不是权威**——事实以 `catalog.yaml`、`SCHEMA.md`、`AGENTS.md`、
> `README.md`、`docs/**` 与各实验目录为准。若与事实文档冲突，**以事实文档为准**，
> 并在本文件相应条目注明「此处结论已由事实文档取代」。

## 2026-10-02 · 文档 SSOT 化（R1：主目录文档）

用户要求：全部文档 SSOT——事实性文档必须优先存在、陈述式、无日期流水、无变更叠加；
日志类文档可以带日期。据此把散落在**事实文档**里的过程叙事搬到本文件，事实文档只保留
**规则本身**。

- 删除 `docs/OPEN_PROBLEMS.md`、`docs/REMEDIATION.md`、`docs/REMEDIATION-L2-CLOSURE.md`
  （问题台账 / 执行台账 / 留痕类）：其**规则/纪律**归入 `SCHEMA.md` / `AGENTS.md`，
  **仍有效的待办**归入对应课题目录（原 `axes/**/PROBLEMS.md` 已是其投影），
  **过程史**即本文件。冻结清单的去向见下。
- `AGENTS.md` 重写为一体化制度文档（文档地图 + 编排制度 + 语言/汇报纪律 + 提交纪律）；
  `SCHEMA.md` 去流水账、保留全部条款编号；`README.md` 收敛为「项目是什么」+ 指针。

### 过程叙事（自事实文档迁出）

**文档/提交纪律的事故史**（规则本身现载于 `AGENTS.md` §8 与 `SCHEMA.md`）：

- 脚本化改文本：主 Agent 用脚本批量改文，`oldText` 不匹配时脚本只打印警告继续执行，
  而提交信息却按「全部成功」撰写 ⇒ 提交信息与文件内容不符。事后规则：脚本未命中必须
  非零退出或显式汇总未命中清单，提交信息只写已验证落地项，发现不符立刻补更正提交。
- `git commit -m "…`x`…"`：反引号被 shell 当命令替换执行 ⇒ 提交信息被静默改写
  （历史上出现过 ERRATA 措辞、`l1@K8`、`partial` 三例）。事后规则：一律 `git commit -F -`。
- `git add -A`：把在途子代理正在编辑的文件扫进自己的提交，造成「提交里含他人半成品」。
  事后规则：提交用显式路径，禁用 `-A` / `-u` / `.`。
- `git add <dir> <不存在的文件> 2>/dev/null`：路径不存在时整条失败、什么都不 stage，
  错误被 `2>/dev/null` 吞掉 ⇒ 命令「看起来成功」，随后按「已落地」写了提交信息与 STATUS，
  两个目录对版本控制完全不可见（被复核判 P0）。事后规则：禁用 `2>/dev/null`，
  提交前核对 `git diff --cached --name-only`。

**结论冻结清单的来源**：2026-10-02 一次遮蔽 context 的独立全仓复核（全文
`.scratch/49-adversarial-review/REPORT.md`）判定「本仓库当时不能作为已独立验证的结论库
直接采信」，列出 7 条 P0，主 Agent 独立复现其中 4 条（A1 算术判据错误 / A2 量化侧信道 /
A3 距离表与原始数据相反 / A7 假 commit 锚点）。据此冻结了一批结论，随后由修复波（L1）
与重跑波（R1/R2/R4/R6）逐条替换。**冻结清单是当时的过程状态**；其对应格子
（量化部署价签、搬运 delta 写、T=2048、Spec 组合）的**当前结论**以 `catalog.yaml` 为准。

### L2 结论级清账留痕（原 `docs/REMEDIATION-L2-CLOSURE.md`）

L2 是「结论级清账线」，把复核发现的问题逐条落成文字/来源更正（**不动任何已入账数字**）。
下述更正**均已落地**在对应事实文档中，此处仅留过程记录：

| # | 更正内容 | 落点（现状） |
|---|---|---|
| C1 | 「AR 算术下界 = 两边际相乘」撤回，改为 Fréchet–Hoeffding 交集界 | `SCHEMA.md` §12 |
| C2 | 「距离腿六档全部 > chance」改为事实（d=32=0、d=192=0.00098 均低于 chance 0.015625） | `catalog.yaml` M1 格 |
| C3/C4 | 「delta 写稳定有效」降为单次观测；首验主腿标训练分布外 | `catalog.yaml` M1 格 |
| C5 | 「T=2048 同收敛」改为「同 step 的第二快照」 | `catalog.yaml` A6 格 |
| C6 | H3 命题边界限定为「冻结表征上的重复潜在映射与门控」 | `catalog.yaml` H3 格 |
| C7 | 「−0.0068 唯一改善」→「未观察到超带损伤」（落 Δ<0.014 不可排序带内） | `catalog.yaml` A4 格、`docs/methodology.md` §1、`docs/playbook.md` 定律 1 |
| C8 | 「任何结构都有税 / 只能走旁路」降为倾向 | `docs/design.md` §1.4、`README.md`、`docs/playbook.md` 定律 7 |
| C9 | beam search 否证撤回（确定性重复推理 ≠ beam） | `docs/playbook.md` 定律 2 |
| C10 | bpc「上界 = log₂(206)」改称均匀预测参考值 | C3 实验 `analysis.md` |
| C11 | 「抖动比 3pp 大」失败归因撤回（实为 1.367pp） | F4 实验 `analysis.md` |
| C12 | 「给路径优于给信号」降为假设（无等参数对照） | F4 实验 `analysis.md` |
| C13 | bpc 的 0.014 不得当准确率 1.4pp 界 | `docs/methodology.md` §2 |
| C14 | `reference` 格按分诊规则改判 `(c)`（检索不可用） | `catalog.yaml` 5 处 |
| C15 | 根 scope「不做组合」改为「默认留白 + 例外登记」 | `README.md`、`SCHEMA.md` §4、`catalog.yaml` |
| C16 | `STATUS.md` 刷新到当前 —— 此处结论已由事实文档取代（现以 `catalog.yaml` 为准） | `STATUS.md` |
| C17 | 自动生成页自相矛盾（改 catalog 源文本后重生成） | `catalog.yaml` 轴 `local_summary` |
| F_SCHEMA | `catalog.yaml` 50 处 `@ticketNN` 假锚点 → 真实提交 sha | `catalog.yaml` |
| F_YAML | `catalog.yaml` 重复 YAML 键合并 | `catalog.yaml` |
| F_B12 | `partial` 格状态语义与内容对齐 | `catalog.yaml` |

**L2 残留（当初交回时未做、需后续处理）**：`axes/H-horizon/H4-head-trunk-relation/PROBLEMS.md`
仍引用更正前的 H4 verdict 原文；`2611` 的实验产物内 `reconciliation_lower_bound = 0.4822`
是原始落盘；`tools/check_schema.py` 的 `@<sha>` 真伪校验属工具线。这些随各自课题目录/工具线的
后续统一处理收敛。

## 2026-10-02 · 修复与重跑台账与问题台账（原 `docs/REMEDIATION.md` / `docs/OPEN_PROBLEMS.md`）

这两份是用户授权「把问题全记下来 → 修所有代码 → 重跑该重跑的 → 报最新结果」期间的
**执行台账 / 问题台账**，性质是过程记录（允许日期流水）。其中：

- **仍生效的规则/纪律**已归入事实文档：
  - 「n=1 的**正**结论默认不可采信」→ `docs/methodology.md` §2；
  - 「合成任务结论外推力弱、须标适用域」→ `SCHEMA.md` §19；
  - 「口径选择/判据归因/形态税是否付属框架性取舍，须请用户裁决」→ `AGENTS.md` §3。
- **仍有效的待办**已在各课题目录（原 `axes/**/PROBLEMS.md` 是其投影），留待统一一轮收敛。
- **过程史**（下述）留在本文件：

### 主 Agent 自认的错误（原 REMEDIATION §0）

1. 把「不稳的东西」当好消息汇报（T=2048「同收敛」、量化「int8 近无损」、`delta` 写
   「稳定有效」）；2. 在方法学文档里写错了一条判据（「算术下界 = 两边际相乘」）而它被下游
   用来宣称「实现正确」；3. 把「测了另一个命题」当成「回答了用户的命题」（`T∘T` ≠ 概率路径组合）；
   4. 把 `git log` 当证据（提交信息多次超出实际改动）；5. 让「格式校验通过」被读成「研究验收
   通过」；6. 用单点/单种子支撑「定律」级措辞。以上对应的**规则**已分别落在 `SCHEMA.md`
   与 `AGENTS.md`，**故事**即本段。

### 修复与重跑的范围（原 REMEDIATION §1–§5）

- **修复清单（F1–F_B12）**：量化量化的因果化、前缀不变性测试、变异框架抓红判定、
  对账期望值推导、缺失证据显式失败、`@<sha>` 假锚点换真 sha、重复 YAML 键检测、`partial`
  状态语义对齐、`F_ENG` 引擎修复等（落在 `tools/**`、`tests/**`、`pathlm/**`、`catalog.yaml`）。
- **结论级清账（C1–C17）**：同 L2 表（见上）。
- **重跑（R1–R8）**：量化因果口径 / 搬运三臂多次重训 / 方差量化 / 长 T 四点同预算 /
  读出更长训练点 / Spec 三臂 / 5-6 位与旋转量化 / 递归-线性载体。过程与结果见各实验
  `analysis.md`。
- **Spec 三臂合同（原 REMEDIATION §1）**：与用户要的「N+1 再加一 vs 直接 N+2 vs 概率组合」
  一致，其落地即 `axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/`
  （命题 / 单因素 / 指标 / 判据均在该目录 README）。**反例警示**：H3 的 `T∘T`（潜在复用）
  **不是**这三者中的任何一个，其结论**不得**用于回答概率路径组合问题。

### 冻结清单的去向

2026-10-02 对抗式复核（`.scratch/49-adversarial-review/REPORT.md`）列出 7 条 P0，
据此冻结了一批结论（量化全部部署价签、「delta 写稳定有效」与「距离腿六档」、
「T=2048 同收敛」、「自门控组合 ⇒ Spec 有部分答案」、「AR 算术对账 ⇒ 实现正确」、
「33 judged = 33 条完整验收」、「−0.0068 是唯一改善」）。冻结是**当时的过程状态**；
各对应格子（量化价签、搬运、A6 长 T、H3 组合、H4 算术）的**当前结论**以 `catalog.yaml` 为准，
过程见 `.scratch/49-adversarial-review/`。

## 2026-10-02 · 重跑波（Spec / 量化 / 搬运 / 长 T）

- **F_ENG 引擎缺陷**：`pathlm/model.py::PathLM._mtp_nodes` 在 `chain_mode="token"` 的
  单位置解码下崩溃（条件张量无条件右垫一行 ⇒ 维度失配）⇒ `chain` 模式此前的读数来自
  `probe_chain.py` 的**批量 teacher-forcing 式**前向、**不是解码口径**。已按最小改动修复，
  三条证据（T≥2 逐位不变 / 可失败单测先红后绿 / 变异验证）落在
  `axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/evidence/`。
- Spec 三臂（chain / direct / combined）、量化因果口径、搬运三臂多次重训、长 T 四点
  同预算重跑的过程记录见各实验目录的 `analysis.md` 与 `.scratch/` 对应工单。

## 术语与入口

- `.scratch/NN-slug/` = 历史工单与原始证据（只增不改）；本目录是 50 号文档特化的工单目录。
- 旧项目（回环时代）的数字口径陷阱见 `archive/README.md`。
