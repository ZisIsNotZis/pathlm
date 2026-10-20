<!-- 头注（本工作包加入，正文逐字保留）：本文件是「陌生阅读测试」第一轮的原始报告，由一名没读过本仓的
     fresh reviewer 在只读条件下产出（只给仓库与 8 个问题，工具调用上限 20 次，无 shell）。正文 md5
     （本头注 4 行 + 空行；`tail -n +5 round1-stranger-reading.md | md5sum`）18fe310582834a2b2b182d47797587af。判定、题面与两轮的对应关系见同目录 README.md。 -->

## A. 逐题答案

**1. 这个仓库是什么 / 给谁 / 不做什么**
一本「LM 算力路径设计空间的实测代价账本」：把设计空间按轴（时间/深度/宽度/MoE/量化/预测/对照/工程/backlog）切开，每格一份 `axes/**/project.yaml` 记「谁试过、测出多少、边界在哪、为什么失败」。给药改架构、想知道「这个改动掉多少质量」的人看。明确不做：不训练不发布模型、不做排行榜、不做组合实验库（组合默认留白）。
出处：`README.md`（开头一句话 + 「这是什么 / 不是什么」）· 佐证 `STATUS.md:8`。

**2. 最硬的 3 条实测结论**
- **旁路判断头零税**：clean bpc **1.4836 → 1.4836，Δ = 0.0000**；口径 = 同一条**冻结主干**挂头前后的配对读数；参照系 = 同一主干不挂草稿头；边界 = 只读主干、不回传梯度的头，改主干的不在结论内。`docs/cost-table.md` 定律 7 第 3 行 · 源 `axes/6-predict/6.2-speculative-decoding/project.yaml`。
- **改主干的形态税随规模上升**：13M **+0.31 bpc** → 100M **+0.44 bpc**（matched / 同引擎口径）；参照系 = 同引擎同配方 dense 全栈；边界 = 100M / 1.09 epoch / char 级 enwik8 **单点**，且同表回环族方向相反，斜率不可外推。`docs/cost-table.md` 定律 8 第 1 行 · 源 `axes/2-depth/2.1.1-early-exit/experiments/2524-depth-ar-100m/README.md`。
- **纸面省的算力可以不兑现**：线性注意力把 T=256 推理缓存砍 93.6%，实测**一步慢 2.58×**（23.2 → 59.8 ms）、训练峰值显存 **1.41×**（0.81 → 1.14 GiB）；口径 = 实测墙钟与显存**不是 FLOPs**；参照系 = 同批同配方全因果 softmax；边界 = 13M / 未换融合核。`docs/cost-table.md` 定律 11 第 1 行。
（README 的第 4 条 KV int8 我也核了：+0.000101 bpc、12288 → 6528 B/位置 = ×0.53，与表内 46.875% 一致。）

**3. 被否证的研究项（共 5 项）**
`^status:` 全树扫描出 5 项 `被否证`：`2.1.1-early-exit`、`2.2.2-attention-residual`、`2.2.3-learned-layer-sum`、`2.3.1-layer-capacity`、`6-predict/6.3.1-diffusion`。举例：
- **2.2.2 层间 causal attention**：qkv 臂 **−0.0177**（越 0.014 带、方向更差，还多付 +196 616 参数与 +167.3% 注意力 MACs）⇒ 那份额外算力**不值**；scalar 臂 −0.0082 落带内，且层 2–3 退化成 one-hot（活性门未过）⇒ **本尺度无信号**，不可读成「机制无效」。
- **2.2.3 层间学习加权和（DenseFormer 式）**：Δ −0.0008（s0）/ −0.0015（s1）两 seed 同号但都在带内，且小于基线自身换 seed 散度 0.0034 ⇒ **本尺度无信号**；边界写明 L=8 太浅、它要缓解的层间瓶颈不存在（结构条件未满足）。
- **6.3.1 掩码扩散**：+2.6815 teacher-forced bpc（对照全因果 AR 2.2704），K=16 也拉不回 ⇒ **机制在本尺度极贵**。
出处：`docs/cost-table.md` 定律 10/11 各行 · `axes/2-depth/2.2.2-attention-residual/project.yaml:68` · `…/2.2.3-learned-layer-sum/project.yaml:78`。

**4. 代价表在哪 / 手写还是生成 / 怎么验没被手改**
`docs/cost-table.md`。**生成物**，首行横幅即声明「由 `tools/build_cost_table.py` 从 `axes/**/project.yaml` 生成，禁止手改」，生成器确实存在（`tools/` 下有 `build_cost_table.py`）。验证方式：`python3 tools/build_cost_table.py --check` 与手改副本**逐字节比对**，不一致即非零退出（`README.md`「全树自检」第 4 条 · `docs/plan.md` 中「代价表生成」工作包的验收也这么写）。
**我做的旁证（无需执行）**：我逐节数了表内的行数 = 3+2+1+1+1+2+3+2+3+13+10 = **41**，与表尾「代价事实 41 条 / 27 个研究项 / 46 份 project.yaml」及 README 的 41 一致；46 份 project.yaml 我用 `^status:` grep 也正好数到 46。→ **`--check` 本身仍需执行核实**。

**5. 核实 −0.0177 的最少步骤（我实际走的路径，3 次调用）**
1. `grep 0.0177` 定位 → 命中 `axes/2-depth/2.2.2-attention-residual/project.yaml:25`（`delta: "-0.0177"`）。
2. `docs/cost-table.md` → **「定律 10 · 先问变化空间」节**，行「层间 causal attention（可学习共享 W_q/W_k/W_v + 温度）」，观测代价 `-0.0177`，Δ 方向「正 = 改善」，出处 `…/2653-interlayer-attn-13m/analysis.md`。
3. `…/2653-interlayer-attn-13m/analysis.md:6`（判据定义 `Δ = 对照 − 本臂，正 = 本臂更好`）+ `:25`（`attn-qkv` 1.5431，−0.0177 越带）+ **`:34` 的机器可读字段 `results.json` → `"delta_vs_stock_sequential_2613_s0": -0.0177`**。
落点：`results.json` 该字段 + 同目录 `analysis.md:25`。参照系 = stock 单残差 1.5254，规模 13M / L=8 / 12k 步 / 98.3M token / **seed 0**。
**读法陷阱（仓库自己已警示）**：这一行 Δ 方向是「正 = 改善」，所以 −0.0177 = 更差；而定律 8 那行是「正 = 更差」。同一张表两种符号约定并存、**禁止跨行比符号**（`docs/cost-table.md` 表头声明 + `docs/research-schema.md` §2.3）。

**6. 树有多大 / 状态分布 / 最重的三格**
**46 项**，状态 13 已完成 / 5 被否证 / 14 进行中 / 14 未开始。我没有跑 `summary.py`，而是 grep `axes/**/project.yaml` 的 `^status:` 逐文件数出来，**与 README 声明的数字逐项相符**（按轴：1-time 12、2-depth 10、3-width 8、4-moe 3、5-quant 6、6-predict 5、9-backlog 2）。
最重三格（依据 = `axes/<轴>/<格>/experiments/*/README.md` 目录计数，排除 `archive/` 与 `.scratch/doc-log/` 的副本）：
1. **`1.1.4.1-explicit-memory` ≈ 23 个实验**（2503…2639），状态「进行中」。
2. **`6.2-speculative-decoding` = 14 个**。
3. **`1.1.1-hard-window` = 10 个**。
限制：无 shell，我只能数「目录里有 README.md」的实验；缺 README 的实验目录会被漏计（「按证据量」我没法量，只给了实验数）。

**7. 已知失败与偏离**
- **测试怎么个绿法**：`1 failed, 905 passed`（README），唯一失败是**预存**的 `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`（4 个 `axes/6-predict/**` 的 `mutate.py` 未接入统一变异框架），不属于任何在办工作包。**注意：STATUS.md 同一事实写的是 866 passed / 1 failed（见 C.1）。**
- **显存偏离**：实验 `2653` 训练峰值 **14961.7 / 15141.1 MB**，超 `SCHEMA.md` §20.1 的 5 GB 礼貌线（该线是共享卡礼貌线、不是科学约束），已在实验 `analysis.md` 申报。两份文档此处一致。
- **未测 / 未核实**：`2.2.2` 的 **n≥2 seed 复核未做**（`project.yaml:63` 的 roadmap 第一句自称「唯一会改判词的事」）；scalar 臂 P2 活性门未过；`2651` 残余缺口（更大规模 ~55M、α 单纯形对照臂、与 2.1.1/2.2.1 联合检验、层数外推）；**仓库可见性未核实**、**GitHub issue 是否开启未核实**、GitHub 全量 clone 未跑完、`runs/`（120 个 `results.json` ~1.1 MB）在干净克隆上补不回来、enwik8 与 `*.pt` 不入库。
- **工程事故（plan.md §0 自陈）**：4 个无界 `while :` supervisor 脚本导致 `2653` 写结果阶段崩 781 次（约 124 GPU 小时作废），并占着 GPU 锁把 `2651` 饿死 21580 次。

**8. 许可与报问题**
**代码许可 AGPL-3.0**（`LICENSE.md` 我读了前 14 行，是 GNU AGPL v3 全文）；**文档/论文许可另议、尚未裁定**（`README.md`「语言与许可」）。报问题：向 `origin`（`git@github.com:ZisIsNotZis/pathlm.git`）的 GitHub 提 issue 或 PR——**issue 是否开启未核实，PR 一定可用**；贡献流程见 `CONTRIBUTING.md`。

## B. 卡点

| 题 | 用了多少次 | 卡在哪 |
|---|---|---|
| Q1/Q2（读 README） | 2 | 不卡。README 是我见过信息密度最高的前线，直接给了口径与边界 |
| Q5 | 3 | 不卡，但**符号方向要读两次**：表里两种 Δ 方向并存，第一眼容易把 −0.0177 读成「省了 0.0177」 |
| Q6 状态分布 | 1（grep） | **需要执行才能严格核实**：README 的数字来自 `python3 tools/summary.py`，我无 shell，只能自己数 `^status:` 复核（结果相符） |
| Q6 最重的三格 | 1（find） | **部分不可核实**：只能数含 README.md 的实验目录，「证据量」无法在只读条件下度量 |
| Q4 `--check` 是否真能抓出手改 | 0 | **需要执行**：`build_cost_table.py --check` 的逐字节比对无法只读验证；我用「手数 41 行 + 46 份 project.yaml」做了弱旁证 |
| Q7 测试计数 | 0 | **需要执行**：README 说 905 passed、STATUS 说 866 passed，只读无法判定哪个是当前事实 |
| Q4/Q6 的 `check_docs.py --check` 0 违规 | 0 | **需要执行**，且见 C.3：我用只读方式已经找到至少 3 处行号级错引用，与该「0 违规」直接冲突 |

**未能回答的**：无（8 题都有证据）。但 Q4 的「怎么验证没被手改」与 Q7 的「测试怎么个绿法」我给的只是**弱证据 + 需要执行核实**，不是实测。

## C. 矛盾清单（本次最重要的产出）

1. **P1 · 同一当前事实，两个数**：`README.md`「当前状态」写 `1 failed, 905 passed`；`STATUS.md:17`「已知预存失败」写 `866 passed / 1 failed`。两份都是现在时陈述、都自称实测，失败**名字**一致而**通过数**差 39。两份文档不可能同时对，读者无法判断谁陈旧；且 `docs/plan.md:11` 被 STATUS 引为出处，同样只承载旧数。
2. **P1 · 行号引用指到相反状态**：`STATUS.md:16` 与 `FINDINGS.md` §二首行都引 `axes/2-depth/2.2.2-attention-residual/project.yaml:56` 作为「`status: 被否证`」的出处；但 `^status:` 实际在 **`:68`**，而 `:56` 落在 `work_packages:` 块里（WP1 自己的 `status` 是 **`已完成`**）。同类错位还有：STATUS 引 `2.2.3/…:66`（实际 `:78`）、`2.2.4/…:51`（实际 `:63`）——**三处系统性偏移 ~12 行**，说明「行号绑定提交」的机制没起作用，而 `projects.yaml` 侧一改就漂。
3. **P1 · 「引用可解析 0 违规」与已知错引用并存**：`README.md`「当前状态」称 `tools/check_docs.py --check`「0 处违规」；我按 C.2 的路径打开即见错引用。README 把该命令描述成「文档引用可解析」，绿了并不等于行号对——**绿检查的含义被高估**（`docs/plan.md` 的「引用即检」验收恰恰要求「校验目标存在且**行在范围内**」，说明设计意图是能抓，实际输入没抓到）。
4. **P1 · 计划层（AGENTS.md 指定的唯一权威）与路由层互相打架**：`STATUS.md:14-15` 称「里程碑 M1/M2 **已完成**」「工作包 **WP-R1…R6 已完成**」，出处写 `docs/plan.md:10-11` / `:49-53`；但 `docs/plan.md` 的里程碑只有 **自证 / 可引用 / 推进** 三个且状态**全是「未开始」**，工作包表里**没有任何 WP-R1…R6**（那批 id 只存在于 `.scratch/62-axis1-time/TICKET.md`）。路由文件引用了计划层里不存在的 id，且把「未开始」引成「已完成」。
5. **P1 · plan.md 引用 README 的一句已经不存在的话**：`docs/plan.md:12` 写「`README.md:59` 写着『本仓库是**本地工作副本**：提交均在本地，未 push 到远端』」并据此下诊断。我 grep 全仓 `*.md`：这句只出现在 `docs/plan.md` 自己（与一份 `.scratch` 工单）里，**当前 README 通篇没有这句**（README 现在写的是 origin URL +「仓库可见性未核实」）。一份事实文档断言另一份事实文档的当前内容，而该内容已不存在。
6. **P1 · plan.md 自身的交付物状态已被现实推翻**：`docs/plan.md` 的 10 个工作包**全部标「未开始」**，但其中「前线如实」（`README.md` 已重写）、「代价表生成」（`docs/cost-table.md` 存在且带生成横幅）、「引用即检」（`tools/check_docs.py` 存在）、「文档地图接线」、「Δ 方向机器可读」（代价表已有「Δ 方向」列）都已有实物。`tools/` 我 ls 过：`build_cost_table.py`、`check_docs.py`、`supervise.sh` 都在。⇒ **计划层是整仓最陈旧的文件**，而它恰被 AGENTS.md §2.1 定为计划事实的唯一权威。
7. **P2 · 「一格一份 project.yaml」被两个轴破坏**：`axes/7-references/experiments/`（4 个实验）与 `axes/8-engineering/experiments/`（1 个）把实验直接挂在**轴下**，没有格子目录，`^status:` 全树扫描的 46 份 project.yaml 里也没有它们。⇒ 这 5 个实验不被 `summary.py` 的「全表」看见；README「一格一个课题」的声明与树实际形状不符（可解释为「7/8 不算研究项」，但它们仍有实验产物而无研究项，属结构缺口）。
8. **P2 · 同一趋势两套数字，未合并**：`FINDINGS.md` 定律 8 行写回环族「`+0.122 → +0.082`（渐近口径）」，`docs/cost-table.md` 定律 8 第 2 行写「`+0.364@1k → +0.122@24k`」。同一「税随规模下降」的论断挂着两组不同端点，且都标「成立」，读者无法知道 0.082 与 0.364 的关系。
9. **P2 · 未来/过程时态混进当前事实文档**：`STATUS.md:12` 写「截至 **2026-10-19**」，`docs/plan.md:8` 写「一个外部 agent 读完本仓，写下了这句判断」「同一时期内，两个全新上下文的复核者……」——按本仓 `AGENTS.md` §2.1「事实文档一律陈述式、不带变更记录/事故叙事」，这两处是**过程史写进了事实文档**，正是该仓自己禁止的形态。
10. **P2 · 撤回/边界提示与「最硬结论」并列时易读反**：README「五条最硬的结论」第 2 条与第 9 条（cost-table 定律 10 若干行）的「适用边界」都在说「这可能是设计无效而非机制无效」「这不是通用规律」，但 README 的五条用**加粗祈使句**呈现，边界在括号里。读者若只抄前五条，会得到比仓库实际立场更强的断言——这与该仓「不能只看结论」的自我要求方向相反。

## D. 总体判断

**PASS（有条件）**——但缺陷集中在一个特定层，必须说清是哪一层。

理由：一个陌生读者能核实的**每一条可数事实都对得上**，而且是**独立复算**而非相信声明：46 份 project.yaml（我自己 grep 数出来）、13/5/14/14 的状态分布（逐文件数出来）、41 条代价事实（我逐节数表行数得到 41）、AGPL-3.0（读了 LICENSE 正文）、以及 Q5 那个数字能在 **3 次调用内**从 README 走到 `results.json` 的机器可读字段并读对符号方向。这是很高的自证水平——README 甚至主动申报「仓库可见性未核实」「issue 是否开启未核实」「逐 run 产物补不回来」，这类「把不知道标成不知道」的做法显著提高了我对它的信任。

失分处是**账目层，不是科学层**：`docs/plan.md`（计划唯一权威）与 `STATUS.md`（路由）落后于实物至少一轮，出现「引用了不存在的 WP-R1…R6」「把未开始引成已完成」「引用的行号落在相反状态上」「引用 README 里已不存在的一句话」；再加 README 与 STATUS 对同一测试数给出 905 / 866 两个当前值。这些**不影响任何一条实测结论的可核性**，但会让一个按文档干活的接续者去错地方——尤其 C.4/C.6 会让接手者以为已完成的工作没做、或去做已经做完的工作。

若给维护者一句话：把 `docs/plan.md` 的里程碑/工作包状态和 `STATUS.md` 的全部 `文件:行` 引用重刷一遍，并让 `check_docs.py` 真正校验**行号落在目标字段上**（它现在绿了，而 C.2 的三处错引用依然在），这份仓库的「陌生读者自证」就是干净的 PASS。需执行才能完成的核实项（留给我没有 shell 跑不了的部分）：`python3 tools/summary.py --check`、`python3 tools/check_docs.py --check`、`python3 tools/build_cost_table.py --check`、`python3 -m pytest tests/ -q`（判定 905 还是 866）。