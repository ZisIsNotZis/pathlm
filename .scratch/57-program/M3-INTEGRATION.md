# ticket 57 · M3 集成收口报告（integrator）

> 角色：**M3 集成收口 integrator**（worker，fresh 上下文，**单一写入者**，其它 writer 已停）。
> 时间：`2026-10-13T20:05:02 → 20:30+08:00`（`date -Iseconds` 实取）。语言：中文。
> 契约：`docs/structure.md`、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`、
> `.scratch/57-program/TICKET.md`（M3 = 集成验收）。
> 仓库：`/home/z/vibe/pathlm`，进入时 HEAD = `285eb49`（`git status` 仅一个 pre-existing untracked
> `.scratch/66-axis5-quant/REVIEW-2.md`）。GPU：本任务**未跑任何 GPU**、未取 §20 锁。

---

## 0 人话摘要（先读这段）

**这一轮做了什么。** 把「迁移（M1/M2）」和「六个研究大类的第一轮（Phase A）」这两批成果，合成为
一份**一致的、可审计的当前状态**，交给出资人验收。具体四件事：①把只留在 subagent 临时目录里的
**复核/处置报告**扫回仓库（补齐审计留痕）；②对全树做**跨轴一致性**机检（编号唯一、用词与结构表
逐字一致、证据指针不悬空、旧轴引用清零、角色齐、判词不串轮）；③把六个轴上报出资人的**框架性
取舍**汇总成一张待裁决清单；④把本轮验证有效的**流程规则**写进 SSOT 文档。

**现在仓库处于什么状态。** 全树 **46 个研究项**，`project.yaml` 是唯一 SSOT；三道门全绿：
`check_tree.py` OK（46 份）、`summary.py --check` OK、`pytest tests/ -q` **834 passed**（112 秒）。
跨轴一致性**无阻塞级矛盾**。全树状态：**判过了 5 项**（已完成 3 + 被否证 2）、**在跑 15 项**、
**还没碰 26 项**（其中 Backlog 2 项）。

**哪些结论已经定了。**
- **判过了（已完成 3）**：`3.1.4` 前馈隐藏比——「总量是旋钮、分配不是」（等 FFN 预算下逐层分配
  不可排序）；`3.2.1` 读出重参数化——增益归容量、非线性本身无额外贡献；`5.1` KV 量化——8 位近无损、
  4 位两腿都掉（块标尺口径）。
- **被否证（2）**：`1.1.4.1` 显式记忆槽——本轮 13M 上「小本子买不到一个窗口位置的等效上下文」，
  被只宽 1 格的纯窗口（点估计）严格支配；`2.1.1` 逐 token 早退——「按需少跑」从未显著赢过最简单的
  「整段跑前 k 层」，最刺眼的是 τ=0.9 档多花两层反而更差。两项都写了失败归因（属**实现/结构**问题，
  都不是 innovation 本身失败），且都留了「可能反转」的下一步。

**哪些还没做。** 26 项未开始（本轮只做了每轴的 1 个最高价值 WP，其余是「认人 + 想清楚」）；
15 项在跑（多半是旧证据已迁入、结论待补）。**没有任何测量数字或实现被本轮改过。**

**需要出资人拍板的**：一张 **21 条**框架性取舍清单（见 `.scratch/57-program/FRAMEWORK-DECISIONS.md`），
最要紧三条——**0.014 噪声带能不能当判决阈值**、**milestone 验收口径要不要收紧成「相对同算力前沿
不差」**、**是否授权改共享引擎**。**没有一条由集成方自选。**

---

## 1 逐项交付

### 1.1 审计留痕补齐（任务 1）

源根 `/home/z/.pi/agent/sessions/--home-z-vibe-pathlm--/subagent-artifacts/outputs/`。逐文件 **原样复制 +
`cmp` 逐字节核对**；已存在于仓库的**不覆盖**（只 `cmp` 核对一致）。新增 11 个文件 + 1 个 pre-existing
untracked 纳入本次提交：

| # | 落盘路径 | 来源（artifact run 目录） | 核对 |
|---|---|---|---|
| 1 | `.scratch/61-axis2-depth/PHASE-A-REPORT.md` | `d3a1e187-…/61-axis2-depth/` | `cmp` OK |
| 2 | `.scratch/61-axis2-depth/REVIEW-4.md` | `4e891517-…/61-axis2-depth/` | `cmp` OK |
| 3 | `.scratch/61-axis2-depth/REPAIR-round3.md` | `713b2b7a-…/61-axis2-depth/` | `cmp` OK |
| 4 | `.scratch/62-axis1-time/PHASE-A-REPORT.md` | `f9a2d978-…/62-axis1-time/` | `cmp` OK |
| 5 | `.scratch/62-axis1-time/CLEANUP-final.md` | `54265412-…/62-axis1-time/` | `cmp` OK |
| 6 | `.scratch/63-axis6-predict/PHASE-A-REPORT.md` | `da654863-…/63-axis6-predict/` | `cmp` OK |
| 7 | `.scratch/64-axis3-width/PHASE-A-REPORT.md` | `96fa0b3d-…/64-axis3-width/` | `cmp` OK |
| 8 | `.scratch/65-axis4-moe/PHASE-A-REPORT.md` | `5ee0acc7-…/65-axis4-moe/` | `cmp` OK |
| 9 | `.scratch/66-axis5-quant/PHASE-A-REPORT.md` | `a82a77cb-…/66-axis5-quant/` | `cmp` OK |
| 10 | `.scratch/56-restructure/REVIEW-foundation-r2.md` | `1904e146-…/56-restructure/` | `cmp` OK |
| 11 | `.scratch/56-restructure/wp-reports/WP-M2.9-audit-commit.md` | `c378a278-…/56-restructure/wp-reports/` | `cmp` OK |
| 12 | `.scratch/66-axis5-quant/REVIEW-2.md` | 本轮进入即已 untracked（非 artifact 源） | 纳入提交 |

- **已存在、`cmp` 核对一致（未覆盖）**：`61` 的 `REPAIR-round2.md`/`REVIEW-3.md`、`62` 的 `REVIEW-4.md`、
  `56` 的 `FOUNDATION-REPORT.md`/`REPAIR-foundation.md`/`REVIEW-foundation.md` 及 `wp-reports/` 全部 13 件
  （`REVIEW-M2-integration.md`、`WP-M2.8-integrator.md`、`rev-*`、`wp1..wp7`）——逐件 `cmp` **IDENTICAL**。
- **各轴复核件盘点结论**：轴 2 补 `REVIEW-4`/`REPAIR-round3`/`PHASE-A-REPORT`；轴 1 补 `REVIEW-4`（已存在）
  /`CLEANUP-final`/`PHASE-A-REPORT`；轴 3 无第 2 轮复核（其报告自述「第 2 轮按时间盒未派」）⇒ **无缺件**；
  轴 4/5/6 只缺 `PHASE-A-REPORT`。ticket 56 补 `REVIEW-foundation-r2`/`WP-M2.9-audit-commit`。
- **未纳入**：`M2-workflow-board.md`（`89c48c00-…` 的 runtime workflow **返回转储**，非复核/处置报告，
  且内容含被截断的运行元数据；不宜当审计文档入库）。已在此登记，供出资人知悉。

### 1.2 跨轴一致性机检（任务 2）

全部机检，结论见 §3。**无阻塞级矛盾**（无 id 重复、无悬空证据、无两处冲突事实）。

### 1.3 框架性取舍汇总（任务 3）

落盘 `.scratch/57-program/FRAMEWORK-DECISIONS.md`（新文件）。**只汇总、不裁决**，共收 21 条：
跨轴共性 C1–C5 + 各轴独有 16 条。摘要见 §4。

### 1.4 流程规则入 SSOT（任务 4）

写入 `docs/research-schema.md` **新增 §11「复核与收口纪律（字段级）」**（本轮验证有效的 4 条规则），
并更新该文件 §2 的 `conclusion_review` 字段说明；`docs/structure.md` §10 加**一行指针**指向 §11
（一处事实一处写，不重复）。改后 `check_tree.py` / `summary.py --check` **仍全绿**（见 §2）。
规则内容：①`conclusion_review` 只写当前判词、历轮写 ticket；②返回前做兄弟字段一致性扫描；
③复核累计 ≤2 轮；④按被改对象定验证门（改共享代码必跑全量 pytest，纯文本/指针改动不必）。

### 1.5 全量门（任务 5）

见 §2（单独跑，记录真实输出/退出码/耗时；无 orphan pytest、无 GPU 泄漏）。

### 1.6 状态总表（任务 6）

见 §5（`python3 tools/summary.py` 全树状态表 + 人话统计）。

### 1.7 提交（任务 7）

分 2 个主题提交，显式路径 `git add` + heredoc commit（禁 `-A`/`-u`/`.`），提交后核对 staged 列表。
提交哈希与文件清单见 §6。

### 1.8 报告落盘

本报告写至 run 指定 output 路径，并**同内容复制入仓库** `.scratch/57-program/M3-INTEGRATION.md`（补齐
仓库审计留痕，与 M2 把整合/复核报告入库同例）。

---

## 2 门命令真实输出（任务 5）

全部工作目录 `/home/z/vibe/pathlm`，每条单独跑。

| 命令 | 退出码 | 真实输出 / 耗时 |
|---|---|---|
| `python3 tools/check_tree.py` | **0** | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（`real 0m0.696s`） |
| `python3 tools/summary.py --check` | **0** | `OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（`real 0m0.721s`） |
| `python3 -m pytest tests/ -q` | **0** | `834 passed in 112.06s (0:01:52)`（`real 1m53.419s`；`2026-10-13T20:27:39 → 20:29:33+08:00`） |
| `git diff --check` | **0** | 无输出（无行尾空白/冲突标记） |

**孤儿进程 / GPU 泄漏核验**：
- 运行前 `pgrep -af pytest`：有一个**外部**进程（`/tmp/helpagent-gate-report-…/gate.ini`，跑的是
  helpagent 的 `test_restock_picker.py` + gate canary），**非本任务**、且在我开始前已在跑。
- 运行后 `pgrep -af pytest`：**无任何 pytest 残留**（只有本条检测命令自身）；`pgrep -af "python3 -m pytest tests"`
  无 pathlm pytest 残留。
- `nvidia-smi`：前后均为 `12825→12795 MiB / 24564 MiB`、利用率 `98%→95%`——**同一外部进程**占用，
  本任务**未跑 GPU**、无泄漏、未取锁。

---

## 3 跨轴一致性结论（任务 2）

机检脚本（一次性、未入库）逐项核，全部通过：

| 检查项 | 方法 | 结论 |
|---|---|---|
| `id` 全局唯一 | `check_tree.py`（seen_ids） | **OK**（46/46 无重复） |
| `id/title/dir` 与 `docs/structure.md` §12 **逐字一致** | 解析 §12 表 57 行 → 与 46 份 `project.yaml` 对比 | **0 处不符** |
| `axis` 与 §12 轴标题逐字一致 | 解析 `### 轴 N · 名（目录 …）` → 对比 | **0 处不符** |
| §12「研究项/Backlog」节点 ↔ `project.yaml` 全覆盖 | 集合对比 | **46 ↔ 46 完全对应**（无缺项、无多项；9 个「分组」+ 2 个「参照」按 §13 不写 `project.yaml`） |
| `experiments` 指针无悬空 | `check_tree.py`（路径存在性） | **OK**（全部存在；含 `@<sha>` 锚点可达） |
| 旧轴引用清零 | 正则扫 `axes/**/project.yaml` + `README.md`，排除 `archive/` | **project.yaml 0 命中**（旧轴仅以 `archive/wiki-legacy/axes/<旧轴>` / `archive/dropped/…` **归档指针**形式出现，合规）；仅 1 处历史实验件残留（见下） |
| `role_map` 四角色齐全 | `check_tree.py` | **OK**（46/46 四角色非空） |
| `reviewer ≠ writer`（文本层） | 逐份比较字符串 | **0 处相等**（44 个研究项实名；`9.1`/`9.2` Backlog 为角色占位） |
| `conclusion_review` 不串轮 | 扫多轮标记 | **4 条累积**（见「已上报」） |

**已上报（不擅自改）的非阻塞发现**：

1. **`conclusion_review` 累积历轮判词（4 条）**：`1.1.4.1`（R1/R2/R3）、`2.1.1`（round 1/2）、
   `5.1`（旧口径 + WP2 + REVIEW-1）、`6.2`（复核 #1/#2）。其中 `1.1.4.1` 一条已由 Main 在本轮裁定
   「保留、不再改」。**本任务未改任何判词**，只记录 + 上报（已列入 `FRAMEWORK-DECISIONS.md` C5 待裁）。
2. **1 处历史实验件残留旧轴名**：`axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/README.md:13`
   有 `> \`R-references\`｜引擎…` 一行。这是**迁移进来的历史实验 README**（证据上下文），
   按「不删证据、不擅改实验产物」**未改动**；非事实文档、非阻塞。

---

## 4 框架性取舍清单（任务 3）

落盘：`.scratch/57-program/FRAMEWORK-DECISIONS.md`（新文件，**待出资人裁决**）。

**跨轴共性 5 条（表 A）**：
- **C1 · 0.014 噪声带能否当判决阈值**（来源 56 §8.6.4、轴 1/3/4/5/6）——`docs/methodology.md` 只定义
  它是「13M 跨 run **bpc** 漂移带」，却被跨量纲（准确率/接受率）与跨格（单臂跨窗口极差 0.036324）
  借用；各轴一律双列并报、不自选。**待裁**：是否限定只用于 bpc + 配对 Δ + ≥5 窗口；概率类是否单独
  用 Wilson/bootstrap 门。
- **C2 · milestone 验收口径是否收紧为「相对同算力前沿不差」**（来源轴 2 F5）——`2.1.1` 按预登记文字
  （Δ ≤ 0.014）**过关**、按实质（相对同算力固定深度**不差**）**不过关**。**待裁**：是否收紧口径并写进
  `milestone.acceptance` 纪律。
- **C3 · 是否授权改共享引擎**（来源轴 1 F5、轴 3 F2、轴 6 F3）——三轴都撞到「不扩建共享引擎就测不了
  完整机制」，均不擅改、交最小 patch。**待裁**：是否授权、谁改、谁复核、是否本轮做。
- **C4 · 旧 wiki / `catalog.yaml` 退役**（来源 56 §5.3/§7.3）——已退役并归档 `archive/wiki-legacy/`
  （不删）。**待裁**：出资人知悉/追认。
- **C5 · 既有 `conclusion_review` 累积是否回填**（来源本轮扫描）——默认保持原样。**待裁**：是否回填。

**各轴独有 16 条（表 B）**：轴 1 ×3（代价轴定义、形态税、DeltaNet 对齐不可得）、轴 2 ×3（序列级
perplexity 操作化、WP 顺序、`2.1.3`/`2.2.5` 是否合并）、轴 3 ×2（等预算位点、部分臂 n=1）、
轴 4 ×2（13M 是否值得跑 MoE、GPU 争用）、轴 5 ×4（可排序带、头族支配是否算发现、`5.1` status、
等参数臂归属）、轴 6 ×2（树代价口径、`6.3.3` 判据归因）。细则与「各轴默认处置 / 影响面 / 待裁决点」
逐条见 `FRAMEWORK-DECISIONS.md` §3。

> 已由 M3 处理/知悉：`role_map` 实名（44 项研究项已实名；`9.1`/`9.2` Backlog 占位）；审计留痕缺口
> （本报告 §1.1 已补）。

---

## 5 状态总表（任务 6）

`python3 tools/summary.py`（SSOT 命令）输出的人话版（`python3 tools/summary.py` 原始表同源；
末行为其汇总行）：

| id | 状态 | 研究项 |
|---|---|---|
| 1.1.1 | 进行中 | hard 窗口 |
| 1.1.2 | 进行中 | soft 衰减 |
| 1.1.3 | 未开始 | 可见量上限 |
| 1.1.4.1 | **被否证** | 显式记忆槽 / 线性状态 / 递归状态 |
| 1.1.4.2 | 进行中 | 压缩：摘要槽 / 低秩投影 / pooling |
| 1.1.4.3 | 进行中 | 压缩触发与优先级 |
| 1.1.4.4 | 未开始 | 外部检索 |
| 1.1.4.5 | 进行中 | implicit vs explicit |
| 1.2.1 | 未开始 | dilated / 空洞 |
| 1.2.2 | 进行中 | 块稀疏 / 条带 / 局部+全局混合 |
| 1.2.3 | 进行中 | 内容选择 top-k |
| 1.2.4 | 未开始 | 动态合并 |
| 2.1.1 | **被否证** | early exit（最重要） |
| 2.1.2 | 进行中 | skipping（跳层） |
| 2.1.3 | 未开始 | repeating 循环 transformer |
| 2.2.1 | 未开始 | 残差的形式 |
| 2.2.2 | 未开始 | 用 causal attention 代替残差 |
| 2.2.3 | 进行中 | 层间学习加权和 |
| 2.2.4 | 进行中 | 并行子层 |
| 2.2.5 | 未开始 | 权重共享 |
| 2.3.1 | 未开始 | 逐层容量变化 |
| 2.3.2 | 未开始 | 逐层分辨率 |
| 3.1.1 | 未开始 | uniform 减宽 / width dropout |
| 3.1.2 | 未开始 | 逐层宽度变化 |
| 3.1.3 | 未开始 | 同层内位置相关宽度 |
| 3.1.4 | **已完成** | 前馈隐藏比 |
| 3.1.5 | 未开始 | 通道极化 / 稀疏激活 |
| 3.2.1 | **已完成** | 读出的重参数化 |
| 3.2.2 | 未开始 | 逐出口读出 |
| 3.2.3 | 未开始 | tied vs untied embedding |
| 4.1 | 进行中 | 稀疏激活专家（top-k 路由） |
| 4.2 | 进行中 | 路由与负载均衡 |
| 4.3 | 未开始 | 专家粒度 / 异构 |
| 5.1 | **已完成** | KV 量化 |
| 5.2 | 未开始 | 权重量化 |
| 5.3 | 未开始 | 激活量化 |
| 5.4 | 未开始 | 低秩压缩（MLA 式） |
| 5.5 | 未开始 | fp8 / 大 head_dim |
| 5.6 | 进行中 | 组合：窗口 + 量化 + 检索 |
| 6.2 | 进行中 | 多位置预测 = 投机解码 |
| 6.3.1 | 未开始 | diffusion |
| 6.3.2 | 未开始 | 一次性批量 pad 填充 |
| 6.3.3 | 未开始 | latent regression 批量生成 |
| 6.4 | 进行中 | 训练时未来位置的填充语义 |
| 9.1 | 未开始 | 调度 / 选择器（原 G 轴，Backlog） |
| 9.2 | 未开始 | 迭代精化的其余（原 F 轴剩余，Backlog） |

**共 46 个研究项 ｜ ✅ 已完成 3 ｜ ❌ 被否证 2 ｜ 🔄 进行中 15 ｜ ⬜ 未开始 26**（含 Backlog 2）。

**人话统计**：
- **判过了 = 5 项（11%）**：已完成 3（`3.1.4`、`3.2.1`、`5.1`）+ 被否证 2（`1.1.4.1`、`2.1.1`）。
- **在跑 = 15 项（33%）**：多是「旧证据已迁入、结论待补」，尚未有终局判读。
- **还没碰 = 26 项（57%）**：本轮每轴只做了 1 个最高价值 WP，其余仅「认人 + 想清楚」；
  6 项无旧格（如 `1.2.1`、`2.2.1`、`2.3.1`、`3.1.2`、`3.1.3`、`5.2`…）属**全空**。
- 按轴看：轴 1（12 项）7 在跑 / 4 未开始 / 1 被否证；轴 2（11）1 被否证 / 3 在跑 / 6 未开始；
  轴 3（9）2 已完成 / 6 未开始；轴 4（3）2 在跑 / 1 未开始；轴 5（6）1 已完成 / 1 在跑 / 4 未开始；
  轴 6（5）2 在跑 / 3 未开始；轴 9（2）全 Backlog。

---

## 6 提交（任务 7）

分 2 个主题提交（均为**显式路径** `git add`，`git commit -F -` heredoc）：

### 6.1 提交 A · 审计留痕补齐
`959a186da299412721cdc91159a8b589ef28a8d4`（`959a186`）· **12 files changed, 2175 insertions(+)**
（12 个 create，见 §1.1 表 1–12；提交后 `git diff --cached --name-only | wc -l` = 0）。

### 6.2 提交 B · M3 集成产出
`1c755f3b271dc8e8e8910a4a9aa4bc9f4216c26c`（`1c755f3`）· **4 files changed, 528 insertions(+), 2 deletions(-)**
（`FRAMEWORK-DECISIONS.md`、`M3-INTEGRATION.md`、`docs/research-schema.md`、`docs/structure.md`）。

> 另有一个后续小提交 C · 报告补记上述提交哈希（本 §6 的补记），不涉内容变更。
> 两次主题提交后：`git status --short` 空、`git diff --cached --name-only | wc -l` = 0；
> 提交后复跑 `check_tree.py` / `summary.py --check` 仍全绿（见 §2）。

---

## 7 未决项（交付给 Main / 出资人）

1. **框架性取舍 21 条**（`FRAMEWORK-DECISIONS.md`）——尤其 C1/C2/C3，**未自选**，待裁。
2. **C5 `conclusion_review` 累积 4 条**——是否回填为「只写当前判词」，待裁（本轮未擅改）。
3. **`pathlm/config.py` `chain_mode` 默认 `"latent"`**（56 §8.6.1 遗留）——改默认 = 行为变更，
   需产品/引擎 owner 定夺，本任务不列、不擅改。
4. **轴 3 复核门未闭合**：其第 1 轮复核判 FAIL（窄范围），第 2 轮按时间盒未派——属各轴 Phase A
   的收尾责任，非集成阻塞；已在轴 3 `PHASE-A-REPORT.md` 如实登记。
5. **`M2-workflow-board.md`**（runtime 转储）未入库，见 §1.1。

---

## 8 边界遵守自述

- **只做集成与留痕**：未新增研究内容、未改任何测量数字或实现（`pathlm/**` 零改动）。
- **不删证据**：`.scratch/**/evidence` 原地未动；`archive/**` 未动。
- **未擅改研究结论**：`conclusion_review` 累积条目保持原样（记录 + 上报）。
- **单一写入者**：本轮只有本 worker 写仓库。
