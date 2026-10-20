<!-- 头注（本工作包加入，正文逐字保留）：本文件是「陌生阅读测试」第二轮的原始报告，由另一名没读过本仓的
     fresh reviewer 在只读条件下产出（只给仓库与同一份 8 个问题，工具调用上限 20 次，无 shell）。正文 md5
     （本头注 4 行 + 空行；`tail -n +5 round2-stranger-reading.md | md5sum`）ffabd8ecbfc9a28ab43ca9359bd8b9e0。判定、题面与两轮的对应关系见同目录 README.md。 -->

## Review

### A. 逐题答案（含依据）

**Q1 — 这是什么／给谁看／不做什么**
- 是什么：一本 **LM 算力路径设计空间的「代价账本」**，一格一个研究项（`axes/**/project.yaml`），记「谁试过、测出多少、边界在哪、为什么失败」。依据：`README.md`「一句话」「是什么 / 不是什么」两节。
- 给谁看：要改模型架构、想知道「这个改动掉多少质量、还是白赚」的人（`README.md`「给谁看」一行）。
- 明确不做：不训练也不发布模型；不做排行榜；**不做组合实验库**（单轴单格，组合默认留白，`SCHEMA.md` §4）；不是每个目录都算研究项（轴 7 参照线、轴 8 工程纪律**不立项**，`README.md`「不是」第 2 条 + `docs/structure.md:335-352`）；不能只看结论跨口径比较。
- 置信度：高，一次调用即答全。

**Q2 — 最硬的 3 条结论（数字／口径／参照系／独立可摘的边界）**
1. **不动主干的旁路判断头是零税**：clean bpc 1.4836 → 1.4836，Δ=0.0000；口径＝同一条冻结主干**挂头前后**的配对读数；参照系＝自身不挂头的同一条主干。
 边界（独立成条，`README.md` 第 1 条「边界：」行）：只说「只读主干、不回传梯度」的判断头；任何改主干的改动不在这条结论里。表内同行：`docs/cost-table.md` 定律 7 的 jumphead 行（适用边界＝「旁路判断头零税：不动主干就不付主干税；冻结断言（grad is None）钉住这条读数」）。
2. **改主干的税随规模上升**：13M `+0.31 bpc` → 100M `+0.44 bpc`；口径＝同引擎同配方 dense 全栈对照、两端 matched-step；100M 那档 `+0.4563` 总差里 96% 归「改形态」，引擎只占 `+0.02`。
 边界：100M 端只是 char 级 enwik8 的**一个点**（1.09 epoch），斜率不得当通用规律；同表回环 retry 族方向**相反**（`docs/cost-table.md` 定律 8 两行）。
3. **纸面省的算力可以不兑现**：线性注意力纸面把 T=256 推理缓存砍 93.6%，实测同批同配方**一步慢 2.58×**（23.2→59.8 ms），训练峰值显存 **1.41×**（0.81→1.14 GiB）；口径＝实测墙钟与显存（非 FLOPs）；参照系＝同批同配方全因果 softmax。
 边界（`README.md` 第 3 条「边界：」行）：13M / 未换融合核的实装，换融合核可能改写，本仓没测。
（另两条同样带独立边界：KV int8 `+0.000101 bpc`、12288→6528 B/位置；判据带 `0.011` 跨度 vs 种子散布 `0.0138`。）
- 关于「边界不能只活在括号里」：**PASS**，三条都是独立 `**边界：**` 行，且 `docs/cost-table.md` 有专门的「适用边界」列。
- 依据：`README.md`「五条最硬的结论」；`docs/cost-table.md:17-112`。定律正文的家 `docs/playbook.md` 我未打开（未读，非卡点，README 已标为唯一家）。

**Q3 — 被否证的研究项（`status: "被否证"` 全树 5 个：2.1.1、2.2.2、2.2.3、2.3.1、6.3.1；依据 `grep '^status:' axes/**/project.yaml`）**
- `2.2.2` attention-residual：层间 causal attention 在 13M/L=8 的 matched-step 快照上**无净增量**——qkv 臂 `−0.0177` 越带更差却多付 +167.3% MACs，scalar 臂落带内但层 2–3 训练后退化成 one-hot（活性门未过）⇒ 本尺度无信号/不值。
- `2.2.3` learned-layer-sum：`dense` 相对 stock 单残差 Δ 只有 s0 `−0.0008`、s1 `−0.0015`，既不越 `0.014` 带、也小于该臂换 seed 散度 `0.0034` ⇒ **L=8 太浅，机制要缓解的层间瓶颈不存在**（结构条件未满足，属「本尺度无信号」）。
- `6.3.1` diffusion：掩码扩散 teacher-forced bpc `+2.6815`（对照全因果 AR 2.2704），K=16 也只到 `+2.6873` ⇒ **机制在该形态下无效**，代价面的顺序收益换不回质量。
（其余两个：`2.3.1` 逐层容量前大后小 `−0.0140`（按 4dp 恰在带界、判「可排序更差（边界）」）；`2.1.1` early exit。）
- 依据：`docs/cost-table.md:96,103-108`、`FINDINGS.md:38-52`、`axes/2-depth/2.2.2-attention-residual/project.yaml:68`。

**Q4 — 代价表**
- 位置：`docs/cost-table.md`。**生成物，禁止手改**（该文件第 1 行 banner + 第 5 行说明）；由 `tools/build_cost_table.py` 从 `axes/**/project.yaml` 的 `laws` / `cost_rows` 生成。
- 怎么验证没被手改：`python3 tools/build_cost_table.py --check` 与手改副本**逐字节**比对（README + `docs/plan.md` §1/§3 都把它列为门）。当前是否通过 ⇒ **需执行**（README 与 plan 声称通过）。
- 每行是否自带 Δ 方向：**是**，每行有「Δ 方向」列（`正 = 改善` / `正 = 更差`），来源是 `project.yaml` 的必填字段 `delta_direction`（我在 `axes/2-depth/2.2.2-attention-residual/project.yaml:26` 看到 `delta_direction: comparator_minus_arm`）；同表两种方向并存，明确禁止跨行比较符号（`docs/cost-table.md` 表头说明 + `docs/research-schema.md` §2.3）。
- 规模：**41 条代价事实、来自 27 个研究项、扫描 46 个 `project.yaml`**（表尾「覆盖情况」节），与 `README.md:18` 的「41 条」一致。

**Q5 — 核实「层间因果注意力的税 = −0.0177」的最少路径（3 步，2 次读取）**
1. `docs/cost-table.md:110`（定律 10 分节里的「层间 causal attention（可学习共享 W_q/W_k/W_v + 温度）」行）——观测代价列写 `-0.0177` clean bpc Δ，Δ 方向列 `正 = 改善`，说明写「对照 − 实验；越带、负 = 更差」。
2. 事实源：`axes/2-depth/2.2.2-attention-residual/project.yaml` 的 **`cost_rows[0].delta`**（第 25 行 `delta: "-0.0177"`）+ **`cost_rows[0].delta_direction`**（第 26 行 `comparator_minus_arm`）。⇒ Δ = 对照 − 实验，负号 = 实验（层间 causal attention）比 stock 单残差 **差 0.0177 bpc**，越 `0.014` 不可排序带。
3. 原始读数（可选）：`axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/results.json` 的 **`delta_vs_stock_sequential_2613_s0`**（第 34 行 `-0.0177`）；判读在 `…/2653-interlayer-attn-13m/analysis.md:25、97、115`。
- 实测我做的是 1 次 `grep 0.0177` 就落到 `docs/cost-table.md:110`，再 1 次 `grep` 落到 `project.yaml:25/26` 与 `results.json:34` ⇒ **2 次调用可核实**，无需跑任何命令。
- 读歪风险（写在此备查）：`−0.0177` 是「更差」，若读者按 `Δ = 实验 − 对照` 顺读会读成「省 0.0177」，这与 README 第 2 条把税写成 `+0.44` 的写法不同向；本仓用逐行 `Δ 方向` 列解决，机制是对的，但需要读者先看列。

**Q6 — 树的大小与状态分布**
- **46 个研究项**；✅ 已完成 **13** / ❌ 被否证 **5** / 🔄 进行中 **14** / ⬜ 未开始 **14** —— 我逐个数了 `axes/**/project.yaml` 的 `^status:`（46 个文件：已完成 13、被否证 5、进行中 14、未开始 14），与 `README.md:111` **完全一致**。
- 不在研究项表内的实验：**5 个目录**（轴 7：`A2-full-attention`、`A8-positional-semantics`、`R7-existing-early-exit`、`W3-kv-head-sharing`；轴 8：`2523b-shallow-weight-sweep`），我 `ls` 了两个 `experiments/` 目录，与 `docs/structure.md:341-347` 的表**逐条对上、无遗漏**。原因：轴 7 是参照台账、轴 8 是工程纪律台账，**不立项**、不写 `project.yaml`、不进 `summary.py` 研究项表（`docs/structure.md:335-338`，「表里没有不是漏了」）。
- 表尾「另有 N 个辅助实验」的确切 N ⇒ **需执行**（`python3 tools/summary.py` 末行）。

**Q7 — 已知的失败与偏离**
- 测试的绿法：`python3 -m pytest tests/ -q` ⇒ **`1 failed, 913 passed`**（约 2 分钟、单机 CPU、不占 GPU）。唯一失败是**预存失败**：`tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`，它断言每个 `mutate.py` 都 `from tools.evalkit import mutation`，4 个实验尚未接入（`2674-mdiffusion-13m`、`2671-padfill-13m`、`2660-latentreg-vs-disc-13m`、`2670-freerun-bpc-13m`；我确认 `2670-freerun-bpc-13m` 确实在 `axes/6-predict/6.3.3-latent-regression/experiments/` 下，与 plan 的「都在 `axes/6-predict/` 下」一致）。红线是「除这一条外 0 failed」。依据：`docs/plan.md` §3。
- 已知成本/显存偏离：有 1 处——某实验训练峰值显存超出 `SCHEMA.md` §20.1 的「共享卡礼貌线」（该线不是科学约束），数字与理由落在 `axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/analysis.md`（README 与 `STATUS.md:29-31` 只放指针，不转述）。
- 未测判据：`FINDINGS.md` 五「敞开的缺口」5 条（更大规模点 ~55M 未跑；`dense` 的 α 单纯形约束对照臂未做；中间深度可读性与 2.2.1/2.1.1 的联合检验未做；s0 等值门无结构化落盘；长度/层数外推未测），加上 README 各条边界里的未测项（fp8 与更大 `head_dim`、4 位并越带、qkv 臂 n≥2 未做）。另有**未核实**类（不是未测）：仓库可见性、GitHub issue 是否开启。

**Q8 — 许可与报问题**
- 代码许可 **AGPL-3.0**，全文 `LICENSE.md`；**文档/论文许可另议**（`docs/plan.md` §5「需要用户裁决」：AGPL 覆盖代码，论文目标可引用，文档通常 CC BY，未裁定；属对外暴露需同意）。
- 报问题：向 `origin`（`git@github.com:ZisIsNotZis/pathlm.git`）的 GitHub 仓库提 issue 或 PR（README「怎么报问题 / 怎么贡献」；issue 是否开启**未核实**，PR 一定可用）。贡献流程 `CONTRIBUTING.md`。

### B. 卡点（调用次数／卡在哪／需执行项）

共 **19 次**工具调用（上限 20），无题目因此未答。
- Q2 1 次卡点：`docs/playbook.md`（定律正文与边界的家）未打开，仅靠 README + 生成表交叉验证；不打开也能答题，但「边界的最终措辞」我没在原家核过。属有意节省预算。
- Q4／Q6／Q7 的**需执行**项（只读工具无法证明，父级若要 PASS 必须跑）：
 1. `python3 tools/build_cost_table.py --check`（证明表未被手改）；
 2. `python3 tools/summary.py`（末行 46／13／5／14／14 与「另有 N 个辅助实验」）；
 3. `python3 tools/summary.py --check`、`python3 tools/check_docs.py --check`（README 声称 0 违规、即所有 `#heading` 锚点可解析）；
 4. `python3 -m pytest tests/ -q`（确认 `1 failed, 913 passed` 与失败名）；
 5. `git remote -v` 与仓库可见性、issue 开关（仓内自述「未核实」）。
- Q6 的辅助实验条数 N：仓内只有「表尾打印」这一条路，因此**必须执行**才能给出确切 N；但「哪 5 个目录」已由 `docs/structure.md:341-347` 与我 `ls` 的结果对上。

### C. 矛盾清单（含 A–E 逐条判定）

- **A. 计划层与路由层 vs 实物：PASS**
 证据：`docs/plan.md` §2 工作包表里「未完成」的交付物确实都**不存在**（`docs/repro.md`、`docs/triage.md`、`tools/inventory_runs.py` 均不在 `docs/`、`tools/` 列表里），§0 声称的「树内唯一重试驱动」`tools/supervise.sh` **存在**；`STATUS.md:24` 引用的 `project.yaml::status` / `::conclusion` 键存在；`STATUS.md` 与 README 引用的 `docs/structure.md#16-辅助实验`（`docs/structure.md:335`）、`docs/plan.md#3-验收证据全树机检与测试计数`（该节标题）都能对上；`STATUS.md` 指向的 `…/2651-inter-layer-topology-s1-13m/analysis.md` 的 §9/§10 **确实存在**（第 286、329 行）；`2670-freerun-bpc-13m` 确认在 `axes/6-predict/`。我未发现指向已不存在实物的引用。
 唯一时间性滞后的正常现象：`docs/plan.md` §1 写「尚未再跑一轮陌生阅读测试」、工作包「陌生阅读复测＝未开始」，而本轮测试正在发生——落地后需按 §2 更新，不算引用悬空。

- **B. 「测试通过数」等派生事实是否只有一家：FAIL（P2，1 处）**
 主读数确实只有一家：`docs/plan.md` §3（`1 failed, 913 passed` + 失败名）；`README.md`「当前状态·测试」与 `STATUS.md`（预存测试失败条）都只放指针、不写数字 ✅。研究项数也只在 `README.md:111` 写了一次，其余指向 `python3 tools/summary.py` ✅。
 但仓库级测试计数在事实文档里还有**第二个家且已过期**：`axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/MODEL.md:79-80` 写「提交 `4e445a5` ⇒ 当前 `tests/…` 单文件 23 passed；全量 `pytest tests/ -q` = **826 passed / 0 failed** … `CUDA_VISIBLE_DEVICES=""` = **825 passed**」。这是 913 之前的旧读数，且带提交号与「当前」措辞（同时踩 C 的日期/事件叙事）。
 最小修法：把这 3 行改成「仓库级测试计数见 `docs/plan.md` §3」的指针，只保留该实验自己的单文件计数。

- **C. 当前事实文档里的日期流水／过程叙事／「某个 agent 说过什么」：FAIL（P1，跨 SSOT 文件）**
 证据（`axes/**/project.yaml` 是 AGENTS.md §2.1 认定的逐项事实权威，仍在正文里带工单号与变更叙事）：
 - `axes/2-depth/2.2.3-learned-layer-sum/project.yaml:86-95`：`actor: "c1-backfill-writer（ticket 57 · C1 声明回填）"`、`change: "C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）…"`、`actor: "WP-2651 复核处置 writer（ticket 61 §9.2；fresh 上下文）"`；
 - `axes/2-depth/2.3.1-layer-capacity/project.yaml:74-89`、`axes/2-depth/2.3.2-layer-resolution/project.yaml:64-70`、`axes/5-quant/5.6-window-quant-retrieval/project.yaml:74-83` 同类；
 - `axes/2-depth/2.2.3-learned-layer-sum/project.yaml:11`：`gap: "旧格 C3 是「手段枚举、无目的」（55-review C-P10）…"`；`axes/5-quant/5.1-kv-quant/project.yaml:18`：`conclusion_review: "本项按 55-review W-08 / R8 修正：…"`。
 这些还**渗进生成物**：`docs/cost-table.md` 的规模列/边界列出现「（ticket 07）」「（ticket 06）」「该读数按 55-review P10 判口径不全（丢 per-head slope 与长度外推）」。本仓自己的规矩明令禁止：`AGENTS.md` §2.1「不带变更记录 / ERRATA / 事故编号叙事」「不带『原本是 A、后来改成 B』」，§8 与根策略 `@current_fact` 更规定内部 ticket 引用「forbidden outright」（`SCHEMA.md` §3 只豁免**溯源元数据的日期**，不豁免工单号与 `change/actor` 叙事）。
 另有两处较轻的同类：`FINDINGS.md:8`「绑定引入本条修订的提交：`git log -1 --format=%h -- STATUS.md FINDINGS.md` 指出该提交」（把 change-log 写进事实文档），以及上面 B 里 `MODEL.md` 的「提交 `4e445a5` ⇒ 当前」。
 最小修法：把 `change` / `actor` / 工单号叙事整块迁到 `.scratch/doc-log/`（AGENTS.md §2.2 明确的家），`project.yaml` 只留当前事实 + 溯源日期；重跑 `tools/build_cost_table.py`。

- **D. `.yaml` / `.json` 引用是否都用键锚点：PASS（抽样 3 处全部命中）**
 抽样并打开确认：① `FINDINGS.md:38、52` 的 `project.yaml::status` → `axes/2-depth/2.2.2-attention-residual/project.yaml:68` = `status: "被否证"` ✅；② `FINDINGS.md:38` 的 `project.yaml::conclusion` → 同文件 `:17` = `conclusion:` ✅；③ `results.json::delta_directions` / `::delta_vs_stock_sequential_2613_s0` → `…/2653-interlayer-attn-13m/results.json:2`（`"delta_directions": {`）与 `:34`（`-0.0177`）✅。全树 `grep` 到的 yaml/json 引用形态清一色是 `文件::键`（`config.json::criterion`、`results.json::cost.theory`、`results.json::anchor_single_point_f5` 等），未见 `foo.yaml:行号` 的写法；md→md 的 `playbook.md:9-21` 行号引用不受此约束。剩余靠 `tools/check_docs.py --check` 才可全量证明（需执行）。

- **E. 同一事实两处不同数字／不同状态：FAIL（P2，2 处）**
 1. **仓库级测试计数**：`docs/plan.md` §3 = `1 failed, 913 passed` vs `axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/MODEL.md:80` = `826 passed / 0 failed`（同一条命令的同一事实，一个说 0 failed）。即 B 的同一处，此处按「两处不同数字」重复记。
 2. **`FINDINGS.md` 是否拥有数字**：`AGENTS.md` §2.1 把 `FINDINGS.md` 定为「**路由器**，不拥有事实…不得在此新写结论或重算数字」，而 `FINDINGS.md:6`「已确立的结论…结论 + 关键数字 + 出处」与第 9–70 行三张表的「关键数字」列**确实在持有数字**。这处冲突已被本仓登记为待办（`docs/plan.md` §2 工作包「FINDINGS 数字改指针」＝未开始），所以是**已声明的债**，但两文档今天是互相矛盾的。
 交叉核对**没有**发现冲突的（说明主数字是单家的）：46 个研究项与 13/5/14/14（`README.md:111` vs 我数的 `project.yaml`）；`41 条代价事实`（`README.md:18` vs `docs/cost-table.md` 表尾）；`+0.31 → +0.44`（README 第 2 条 vs cost-table 定律 8 两行 vs `FINDINGS.md` 定律 8 行）；`0.011` / `0.0138`（README 第 5 条 vs cost-table 定律 9 行 vs `docs/methodology.md:17-20`）；`−0.0177`、`+167.3%`（cost-table vs `FINDINGS.md:38` vs `analysis.md:25`）。
 一条阅读摩擦（不作 finding，仅提示）：`FINDINGS.md:38` 把 2.2.2 写成「在本尺度**无净增量**」，而 `docs/cost-table.md:110` 同一研究项写「**越带变差**、无信号不等于机制无效」——两者都在讲两臂，不矛盾，但一句话版本会让人以为「只是没测出来」，比 qkv 臂的实测更差要软。

### D. 总体判断

**PASS**（边缘通过）——8 个问题全部在 20 次只读调用内答出，主数字与研究树实物逐项吻合（46／13／5／14／14、41 条代价事实、`−0.0177` 两跳可追溯到 `project.yaml::cost_rows[].delta`），A 与 D 干净；扣分点集中在「当前事实文档里混入工单号与变更叙事」（C）和「仓库级测试计数的第二个陈旧家」（B/E）。

**如果只能修三处：**
1. 清掉 `axes/**/project.yaml` 里的 `change` / `actor` / `ticket NN` / `55-review XX` 叙事（迁到 `.scratch/doc-log/`），再重跑 `tools/build_cost_table.py` —— 现在这些工单号已经被生成进 `docs/cost-table.md` 的「规模与配方 / 适用边界」列，是读者第一眼看到的事实文档。
2. 把 `axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/MODEL.md:79-81` 的仓库级测试计数改成指向 `docs/plan.md` §3 的指针（连带删掉「提交 `4e445a5` ⇒ 当前」这种事件措辞），消除 `913 passed` 与 `826 passed` 的两家。
3. 让 `AGENTS.md` §2.1 与 `FINDINGS.md` 对「FINDINGS 是否持有数字」说同一句话：要么把工作包「FINDINGS 数字改指针」做完，要么在 AGENTS.md 里把「过渡期内 FINDINGS 的数字列按 `docs/plan.md` 该工作包逐步换指针」写成当前事实，别让两份文档互相否证。