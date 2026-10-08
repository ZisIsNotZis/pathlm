# L2 结论级清账 · 落地留痕（2026-10-02）

> 本文件是 **L2（结论级清账线）** 的逐条留痕：`docs/REMEDIATION.md §3` 的 **C1–C17** 与
> `§2` 里属于文档的 **F_SCHEMA / F_YAML / F_B12**，每条记录「改了哪个文件的哪一处、改成了什么」。
> **原则**：只做文字/来源更正，**不动任何已入账数字**（唯一例外是 C1 之外的纯口径措辞）；
> 每处更正都带 `2026-10-02 更正（Cn）` 标记，可 grep 回查。
> 覆盖范围：`docs/**`、`catalog.yaml`、`axes/**/experiments/*/analysis.md`、`STATUS.md`、
> 重生成的 `axes/**/README.md`（含所需的最小实验 `README.md` 注释）。

---

## C1 · 「AR 算术下界 = 两边际相乘」撤回（最重要）

- `SCHEMA.md` §12「算术对账」实例：改为 Fréchet–Hoeffding 交集界
  `max(0, a+b−1) ≤ P(A∩B) ≤ min(a,b)`（取 a=0.6934、b=0.6953 ⇒ 下界 **0.3887**、上界 0.6934），
  并注明原「下界 = 0.6934×0.6953 = 0.4822」推导作废。
- `axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/analysis.md`：
  §1 判据表「对账（硬性）」加注；§2.1 撤回「对账通过 ⇒ 实现正确」；§2.2 表行删「≥ 下界」；
  §2.4 总表算术行标「判据错误，已撤回」；§3.1 撤回「0.3128 < 0.4822 是铁证」。
  **实测 0.5489 保留**；实现正确性改由任务对账（输入构造差异）+ 代码层单测支撑。
- `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/analysis.md`：
  「对账（决定性）」一节的「下界 = 0.4822」撤回并给出正确界。
- `axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/README.md`：
  3 处（正文／判据线／失败决策树）加 `C1` 注释（预登记判据线保留为历史）。
- `catalog.yaml`（H4 格）`verdict` 与两条 `failure`：删「≥ 算术下界 0.4822 ✓ 对账过」等表述。
- **未改**（属原始证据或 L1 所有）：`2611/run.py`、`2611/results.json`、`2611/run.log`
  （其中 `reconciliation_lower_bound: 0.4822` 是原始落盘，属 F4/对账脚本线）。

## C2 · 「距离腿六档全部 > chance」改为事实

- `catalog.yaml`（M1 格）`local.benefit`：改为「**并非全部 > chance——d=32 = 0.0000、
  d=192 = 0.00098 均低于 chance 0.015625**（最远 d=255 为 0.5947）；曲线非单调」。
- `axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m/analysis.md`：
  §1 判据⑤ 行、§6.3 读法段同样更正。

## C3 / C4 · 「delta 写稳定有效」降为单次观测；首验改标分布外

- `catalog.yaml`（M1 格）`verdict`：① 首验主腿（距离 241–255）明标**训练分布外**、降级为
  「**分布外单次观测**」；② 「delta 写是稳定有效的那一支」改为「本快照中处于较优分支（**单次观测**）」；
  ③ 新增**入口结论改用分布内证据**一段（44-X2 的 ×24、41-Z2 两点保持，但 47-W2 seed=1 为 −3.26pp
  ⇒ 跨种子不稳健、未收口）。

## C5 · 「T=2048 同收敛」改为「同 step 的第二快照」

- `catalog.yaml`（A6 格）第四点条目：撤「欠训练混杂被直接消掉」，改称「同 step（6000）的第二快照」，
  并列出末窗斜率 −0.16/1k、token 预算 49.15M vs 98.30M。
- `axes/A-mixing/A6-mixing-operator/experiments/2629-longT-2048-matched/analysis.md`：
  标题、「人话摘要」、§1.5/§1.6、§2.2、§5、§10 表行、附录判据行逐条更正，并在文件顶部加 **ERRATA** 块。

## C6 · H3 命题边界限定

- `catalog.yaml`（H3 格）`verdict`：限定为「**冻结表征上的重复潜在映射与门控**」，明确**不回答**
  用户要的概率路径组合问题（`p(x_{N+2}|h)=Σ_z p(z|h)p(x_{N+2}|h,z)` vs `p_direct`）；`failure` 加边界条。

## C7 · 「−0.0068 唯一改善」→「未观察到超带损伤」

- `catalog.yaml`：A 轴 `local_summary`、A4 `verdict`、A4 `local.tax` 均改为「落在 Δ<0.014 不可排序带内，
  不构成『改善/唯一负税』」。
- `docs/methodology.md` §1「裁决」示例：`距离惩罚：采纳（免费正则）` → `未观察到超带损伤（不称改善）`。
- `docs/playbook.md` 定律 1：距离惩罚行加「不可排序带内，只能称『未观察到超带损伤』」。

## C8 · 「任何结构都有税」「只能走旁路」降为倾向

- `docs/design.md` §1.4：加 `C8` 更正（原「任何侵入主干本体的结构都会带来结构性税」降为倾向）。
- `README.md` §「三条已测的总规律」第 1 条：「速度只能走成熟旁路」→「**倾向**走成熟旁路」。
- `docs/playbook.md` 定律 7：正文与「边界」两处降为倾向。

## C9 · beam search 否证撤回

- `docs/playbook.md` 定律 2：撤「beam search 的多线正落在死区里」（确定性重复推理 ≠ beam）。
- `catalog.yaml`（G5 格）`failure`：同处更正。

## C10 · bpc「上界」改称均匀预测参考值

- `axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/analysis.md`：
  `log₂(206)=7.69` 由「理论上界」改称「**均匀预测参考值**（不是分布数据的理论上界）」。

## C11 · 撤回「抖动 > 3pp」失败归因

- `axes/F-iteration/F4-training-signal/experiments/2619-reconstruction-loss-13m/analysis.md`：
  「人话摘要」、§1 c3 行、§7 标题与结语、§9 第 3 条逐条更正——实读数为 **1.367pp < 3pp**，
  现象保留为「两次落在不同退化模式」，主判据本身 = B−A = −0.244pp（不依赖该量级）。

## C12 · 「给路径优于给信号」降为假设

- 同 `2619/analysis.md`：「人话摘要」与 §10 加 `C12` 更正（参数 `+264,296` vs `+52,736`，
  **无等参数对照** ⇒ 只能作假设，SCHEMA §15）。

## C13 · bpc 的 0.014 不得当准确率方差

- `docs/methodology.md` §2：新增 `C13` 段（0.014 是 bpc 口径；准确率应用二项/Wilson CI；
  「1.4pp」是**预登记实用阈值**，不是已测方差）。
- `axes/H-horizon/H3-composition/experiments/2613-self-gated-composition-13m/README.md`：
  §4 判决口径、§5 seed 行加注。
- `catalog.yaml`（M1 格）3 处 `1.4pp`：改为「预登记阈值」。
- M-carrier 7 个 `analysis.md`：`方差带` → `预登记阈值带`（并在 2624/2625 的定义行解释）。

## C14 · `reference` 格按分诊规则改判 `(c)`

- `catalog.yaml` 5 处 `prior_art_verdict: "(a)"` → `"(c)（检索不可用 ⇒ 按 SCHEMA §21 降为待核方向，不判 (a)）"`
  （A1 / A5 / A7 / C4 / W2）；对应的「来源类型：业界结论」标签改为「待核方向（记忆·未核，检索不可用）」。

## C15 · 根 scope 例外条款（任务枚举未列，按台账 §3「C1–C17」补做）

- `README.md`「不是组合实验库」、`SCHEMA.md` §4、`catalog.yaml` `project.scope`：
  「不做组合实验」→「**默认留白**」+ 已发生组合实验按**例外**登记 + 是否继续留白属待用户裁决。

## C16 · `STATUS.md` 刷新到当前

- 表头刷新时间/波次/测试基线（595）；§1 换成 ticket 49 复核 + 修复波状态；§3 结论表加
  `⚠️` 标注本轮降级项；§4 SCHEMA 节数；§5 下一波；§6 工程债（595 例）。

## C17 · 自动生成页自相矛盾（改源文本后重生成）

- `catalog.yaml` 轴 `local_summary`：A（M 轴「全零经验」→ 显式记忆槽已有首验）、
  C（「层间拓扑零经验」→ C3 已首探）、W（「KV 头共享零经验」→ W3 已有本地实测）、
  M（「显式记忆槽零经验」→ 已有旁路通道首验）、R（「本地零经验」→ R1 有本地量化实测）。
- 已执行 `python3 tools/build_wiki.py` 重新生成 `axes/**/README.md` 与 `INDEX.md`。

---

## F_SCHEMA · 50 处 `@ticketNN` 假锚点 → 真实提交 sha

用 `git log --diff-filter=A -- <证据文件>` 查出每份证据的**落地提交**，映射如下（全部用 `git cat-file -t` 验证为真实 commit）：

| 原锚点 | 新锚点 | 依据 |
|---|---|---|
| `@ticket31`（2 处） | `@3597039` | 31 第四批收官（2614 稀疏 / KV 量化分析落地） |
| `@ticket32`（2 处） | `@4d4ecb0` | 32 第五批收官（2615 优先级 / 2615 窗口训练分析落地） |
| `@ticket33`（2 处） | `@409f4f1` | 33 第六批收官（2616 线性注意力 / GQA-MQA 分析落地） |
| `@ticket34`（4 处） | `@88dcc53` | 34 情报与设计批（2617 长程电池 / prior-art 内存落地） |
| `@ticket35`（2 处） | `@cc64112` | 35 第八批（2618 通道分析落地） |
| `@ticket36`（36 处） | 按文件：A-mixing/C-vertical/F-iteration/W-width → `@b44c809`；H-horizon → `@575344f`；G-policy `PRIOR_ART_ENGINE.md` → `@0b590e1` | 各 `PRIOR_ART*.md` 的落地提交 |
| `@ticket37`（1 处） | `@ccbeb53` | 37 第十批收官（2620 去相关分析落地） |
| `@ticket39`（1 处） | `@88c0e92` | SCHEMA §17.17 提交（2622 longT-512 分析落地） |

合计 **50 处**（= 36+4+2+2+2+2+1+1），替换后 `grep -c '@ticket' catalog.yaml == 0`。
**注**：`@<sha>` 是否为真实提交的**机器校验**属 `tools/check_schema.py`（L1 所有），不在本线。

## F_YAML · `catalog.yaml` 重复 YAML 键

- A1 格的重复 `local:` 键（`local: {}` 静默覆盖上方 `local`）**已删**，其注释并入 `notes`。
- 用严格加载器（重复键即报错）复核：**无剩余重复键**。

## F_B12 · `partial` 状态语义与内容对齐

- **M2-carrier-compression**：`verdict` 原称「量化/摘要/低秩零经验」——但 KV 量化已有本地实测
  （R1/2614、2618/2621）⇒ 改为「KV 量化已有本地实测（旧价签已入冻结清单，待因果口径重测）；
  摘要压缩 / 低秩投影零经验」。其余 5 个 `partial` 格（A8/A9/F1/P2/P4）逐格核对：
  均符合「只覆盖部分选项」的定义，未改。

---

## 残留（不在本线范围或需后续处理）

1. `axes/H-horizon/H4-head-trunk-relation/PROBLEMS.md` 仍**引用**更正前的 H4 `verdict` 原文（该文件性质是
   「问题台账 · 原文引用」的快照，不在 L2 声明范围内）——建议由编排方决定是否刷新。
2. `2611/run.py` / `results.json` / `run.log` 内的 `reconciliation_lower_bound = 0.4822` 保留（原始证据；
   对账脚本期望值的修正属 `REMEDIATION.md §2 F4`，L1）。
3. `tools/check_schema.py` 的 `@<sha>` 真伪校验、变异框架抓红判定、重复键严格加载属 **L1**。
