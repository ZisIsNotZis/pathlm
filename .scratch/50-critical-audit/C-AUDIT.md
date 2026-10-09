# C 轴（垂直机制 / 深度）批判性审计（2026-10-02）

- 审计者：独立上下文 subagent（fork=false），**只读**；未改实验代码 / 数字 / catalog。
- 声明：本次会话**无写入工具**（只有 read/grep/find/ls/watchdog_diff），因此本文件由主 Agent 落盘；
  `watchdog_diff` 返回「reviewer-launch HEAD 不可用」⇒ 未审工作树增量，只审磁盘内容。
- 方法：逐格读 README / analysis / results.json / 原始 jsonl，**自己复算**关键数；
  对 `.scratch/**` 原始证据做交叉核对；对上一轮 `.scratch/49-adversarial-review/REPORT.md` 只作对照，不复述。

---

## 一句话总判

**C 轴是 C 轴全仓最有决策价值的一条线，其核心负结论（形态税 +0.33/+0.44 bpc、出口资产换延迟不换算力）
经我独立复核在证据层站得住；但它的三个「收益数字」（exit-4 3.1× / uniform-exit 1.8× / 退火配方 +9%）
是同一统计量在不同 run 间漂移 1.5–5.5× 之下的**单次抽样**，不能采信；C4 的 `reference` 状态与
C5/C6 的机制归因分别违反仓库自己的分诊规则与「单因素」纪律。**
⇒ **可以用**的是：机制存在性、税的量级与干净归属、FLOPs 净账、失败路径；
**不能直接用**的是：任何 100M 级别的墙钟倍率、C4 的「业界结论」标注、C5「门是税承载者」、
C6「出口资产只能由深度监督造出（0.12 vs 0.97）」的证据对。

---

## 1. 建议删除清单（供主 Agent 树管理）

| 项 | 位置 | 理由（含证据） |
|---|---|---|
| **C2 的「层序 shuffle」子项** | `catalog.yaml:515-524`、`.scratch/06-m2-sweep/evidence/L3/results.json` | 没有任何部署会打乱层序；它在自家组合实验里还是「毒药」（`IX2/IX3/INT` 全落 2.2+ bpc）。结论无论正负都不改变「该算力路径值不值」的判断 ⇒ 按用户判据应删；数字留 playbook 一条 note 即可。 |
| **C3 的「并行子层有害」作为质量结论** | `catalog.yaml:544,549`、`2613/analysis.md §2` | 业界采用并行子层的理由是**吞吐**（8% 秒/步），不是质量。在「13M + 短 T + bs32」这个并行最不划算的 regime 上宣布「明确有害」，等于把业界权衡的另一半留着不测。要么补吞吐腿（大 batch / 长 T），要么删子项。 |
| **C6 的「只监督最终深度 ⇒ 无出口资产」作为研究项** | `catalog.yaml:611`、`axes/C-vertical/README.md:39` | 这正是 LayerSkip / CALM 的设计前提（`PRIOR_ART.md:127-135` 已列）。改为 `reference` 结论 + 本地税，不再投实验。 |
| （条件性）**C4 整格** | `catalog.yaml:558-575` | 我**不建议删**（见 §3 开倒车清单第 4 条）——但若主 Agent 决定本轮不跑，则必须把 `status: reference` 改成 `stub`，不能留着「业界结论」的名义。 |

---

## 2. 建议重做 / 补做清单（按优先级）

**P1-A（≈15 min，判决整条 C 线的价值）：隔离「门」的税。**
- 证据：`train_depth_ar.py:282-285,453` 唯一的 `--dense` 一次关四件；`catalog.yaml:584` 的「门是税承载者」
  来自 `23c-decomposition.json` 的**余项**（nofill 2.2496 − H1 2.0216 = 0.228），而被当常数的「退出+填充 +0.200」
  在 `23e` matched 12k 下**反号**（fill = −0.0856，`23-depth-ar/evidence/SUMMARY.md §12`）。
- 做法：① 给 `23d-H1-12k`（dense trunk + 全深度 aux，无门/退出/填充，d8 = 1.6362 vs dense-B-12k 1.5287）
  补一次 `tf@4`/`tf@6`（纯推理 ≈5 min）——它的 `results.json` **没有** `tf_acceptance` 字段；
  ② 新增一臂 `gate-off`（g≡1，保留随机退出 + 填充 + aux）12k ≈10 min。
- 判据：若 `tf@4(H1-12k)` 高（≳0.9）⇒ 出口资产不需要门/退出/填充堆栈，形态税里 ≥0.22 的部分是可拆的
  ⇒「逐 token 变深度必须付 +0.44」这个总判断要改写。

**P1-B（≈30 min）：把墙钟收益重测到可判定。**
- 证据：同协议 100M 三次 exit-4 倍率 **1.57/3.08/5.53×**（`25c-gates-wall.json:81-160`、`100m-s0/decode_wall.json`、
  `25b-gates-wall.json:81-160`）；单臂内逐 prompt 55% 散布；唯一做漂移控制的是 `2621`（交错轮次 ⇒ 1.86× vs 同机械）。
- 做法：100M 用 `2621` 的交错协议重测 full / exit-4 / uniform-exit / 深度草稿四臂（3 轮 × ≥4 prompt）。
- 判据：报**比值的中位与极差**；若极差 > 效应 ⇒ 结论写「不可分辨」。

**P2-A（≈10 min）：100M 端到端深度草稿。** `2621/analysis.md §7` 已列（缺 100M checkpoint 的 sha256）。
这是唯一能回答「资产是否随规模消失」的实验（C1 自述 launch-bound ⇒ 预期消失）。

**P2-B（≈50 min）：C4 权重共享三臂。** 不共享 / 全共享循环 / 每 k 层一组；同参数量与同 FLOPs 两条腿；
循环=1 时与不共享臂逐位等价（单测）。这是本轴唯一的真空白，且直接改写「参数 ↔ 循环」的账。

**P2-C（≈16 min）：C3 dense 用法�ontology 对。** 加「均匀起步（α≡1）± 单纯形约束」臂，与 identity 起步臂比；
只有这样才能谈与 DenseFormer 的关系。

**P3（≈20 min）：C2 的 skip 回到 depth-ar 引擎。** 按「逐 token 退出深度比例」画 Δbpc–Δcompute 曲线，
替代跨引擎的 +0.175（`p_skip` 是逐 batch 全局丢层，`pathlm/config.py:150-165`）。

---

## 3. 开倒车清单（业界标准是什么 / 我们错在哪）

1. **逐 token 变深度 / 早退 / 自适应算力**：业界 = CALM、LayerSkip、MoD、Universal Transformer（`PRIOR_ART.md:20-21` 自认机制 (a)）。
   我们错在：把「机制成立 + 出口资产」当本地发现写进 headline（`catalog.yaml:479`），且**没有**按业界口径
   （正常训练 + 轻量早退项）测税——我们的主目标被重新归一为 2/14（`pathlm/depth_ar.py:2267`）。
2. **自投机解码（浅出口当草稿）**：业界 = LayerSkip（同模型早退 + 全栈验证），且其报道口径也是 forwards/延迟。
   我们错在：把「tf@4 = 0.97 是可部署草稿资产」当结论，而自家 `2621` 已测出 FLOPs **+5.6%（负）**、墙钟只在 launch-bound 成立。
3. **训练期随机丢层**：业界 = LayerDrop / stochastic depth（并把「丢层后评估」与「带元素部署」分开报）。
   我们错在：`p_skip` 在 eval 时仍开着（`pathlm/eval.py:41-42` 不清 path 旋钮），量的是部署质量，
   却与业界的训练期正则化数字并列，并升格为「元素税定律」。仓库自己已在 `OPEN_PROBLEMS.md` P-13 承认措辞过度。
4. **跨层权重共享**：业界 = Universal Transformer / ALBERT（已系统研究）。
   我们错在：本地零经验，却把状态标成 `reference`（= 业界已透），并写下「业界结论」——而同一 catalog 块的
   `prior_art_verdict` 写 (c)、`PRIOR_ART.md:117` 又写 (a)，三处不一致，且违反 SCHEMA §21「(c) 不得据以判 (a)」。
5. **并行子层 / 层间加权和**：业界 = GPT-J/PaLM（并行，为吞吐）、DenseFormer（DWA）。
   我们错在：用质量面宣布「明确有害」而不测吞吐；用**语义不同的变体**（identity 起步、α 无约束、浅层放大 2.36×）
   说「与业界不一致」。
6. **深度监督 / 早退损失**：业界 = LayerSkip 的 early-exit loss、GoogLeNet 辅助头、CALM 置信损失。
   我们错在：把「出口资产只能由深度监督造出」当本项目最有迁移价值的结论，证据对却是「训练过的读出 vs 没训练的读出」。

---

## 4. 本轮最严重的三件事

**① 墙钟收益类数字全线不可采信（C1 的 3.1×/1.8×、C6 的 +9%）。**
- 同协议、同规模的 exit-4 倍率：`25c` **1.57×** / `100m-s0` **3.08×** / `25b` **5.53×**；uniform-exit 1.51×/1.80×/3.12×。
- `25c` 的固定退 4 臂在同一个臂内逐 prompt 是 168.4 / 113.6 / 109.3 tok/s（55% 散布）；
  full-depth 基线跨臂是 49.5 / 76.3 / 83.2 tok/s（1.7×）。
- C6 的「+9% 等效验证吞吐」是**纯计算副产品**：加速因子本身 C(1.4750) < A(1.4794)，9% 全来自 full-depth 墙钟差。
- 唯一带漂移控制的测量（`2621`，交错轮次）给 1.86×，且它同时给出 FLOPs **负账**。
⇒ 最小修复：catalog / 轴页 / mental_model / findings / retrospective 里所有 100M 墙钟倍率改成交错口径并附极差；
verdict 行加「FLOPs +5.6%（不省算力）」。

**② 「出口资产」在 100M 的对照是未训练读出，而真正的判决实验（H1-12k 的 tf@4）从未跑。**
- `24b-decode-wall-dense.json::tf_note` 自述 0.1199 是「无直接监督的 tied 读出 proxy，非训练出口资产」；
  同款 proxy 在 13M 给 0.444 ⇒ 这个数随规模漂移，不能当「trunk 没有浅层截断能力」。
- 缺的那一次 eval 是 `23d-H1-12k` 的 `tf@4`（dense trunk + 全深度 aux，12k，d8 = 1.6362）——`results.json` 无该字段。
- 同一件事决定 C5 的「门是税承载者」是否成立（门的份额只是链式分解的余项，且其推导所依赖的
  「退出+填充 +0.200」已被 `23e` 反号）。两格共用一个 ≈5 min 的 eval ⇒ 这是本轮**性价比最高**的补做。

**③ 状态与归因的标注违反仓库自己的规则（C4 的 reference、(c)/(a) 三处不一致；C2 无独立复核；
C6/C2 的 benefit 与来源结论相反；同一证据路径挂着不同 commit 锚点）。**
- `catalog.yaml:568` 写 (c)，同一块 `:563` 是 `status: reference`，`axes/C-vertical/PRIOR_ART.md:117` 写 (a)。
- `C2` 的源工单明文「**User approved self-review in lieu**」（`.scratch/06-m2-sweep/issues/01-m2-breadth-sweep.md`）。
- `catalog.yaml:512`（C2 benefit）与 `catalog.yaml:609`（C6 benefit）的来源文件自己的判读是「no composed number beats B0」
  与「噪声内」。
- 锚点不一致（可核、不需要 git）：`.scratch/24-depth-ar-100m/evidence` 在 `catalog.yaml:592`、`:614` 挂 `@7725891`，
  在 `catalog.yaml:232`、`:491`、`:672` 与来源实验 README（`2524-depth-ar-100m/README.md:36`）挂 `@7cdcdaa`；
  `.scratch/23-depth-ar/evidence` 同时挂 `@1586745`（`:490`、`:592`、`:865`）、`@6b27b5e`（`:672`、`:1356`）、`@b8c9f86`（`:1387`）。

---

## 5. 我独立核到了什么（与 49 号报告的关系）

**确认（独立复核后成立）**：
- 形态税 +0.4362 @100M 的干净归属（形态 96% / 引擎 4% / 配方 ≈0）：`100m-s0/results.json`（1.9175）、
  `24b-dense-s0/results.json`（1.4813）与 `24b-eq-s0`（1.9200）我逐一读了原始 json，复算相符。
- `tf@4 = 0.9726` / `tf@6 = 0.9946` / `d6−d12 = 0.0003`：`100m-s0/results.json::tf_acceptance/depth_curve` 逐值相符。
- `2621` 的接受率分解与 FLOPs 账：`results.json` 的 `accept_by_pos_pooled`（0.9704/0.9527/0.9349/0.9112）、
  `eq_layerpos_per_token`（8.45 vs 8.00）、`quality_guard.total_mismatch=0` 全部相符；`spec_decode.py` 里
  `cost=(γ+1)·dd+(γ+1)·(L−dd)=40/cycle` 与 6760/169 = 40 相符。
- `C3` 的全部算术与等参数对照相符（见该格 ③-2）。
- C2 的 +0.175 / +0.545 / locality sweep 与原始 json 相符。

**推翻/补充（49 号报告没抓到的 C 轴问题）**：
- 49 号报告关注算术下界、量化侧信道、距离表、锚点真伪、收敛口径；**没有**覆盖「C 轴墙钟收益是噪声」
  （3.1× 在三 run 间 1.57–5.53×）、也没有覆盖「+9% 是计算副产品」、也没有覆盖「门的税无隔离对照」。
  这三条是本轮新增。
- 49 号报告的 A7 结论（「不能把 33 judged 读作 33 条经完整验收」）在 C 轴上我给出更具体的形式：
  C2 明面上就是 self-review；C4 是 (c) 被标成 (a)；C5/C6 的机制归因缺单因素臂。

**我无法核实的**：
- `@7cdcdaa` / `@7725891` / `@1586745` … 哪些是真实 commit（无 shell、不能跑 git）⇒ 只能说「同一路径挂了不同锚点，至少一个是错的」。
- LayerSkip / CALM / DenseFormer / LayerDrop 的**具体数字**（本环境无检索；`PRIOR_ART.md` 亦自述联网不可用）。
  因此我判定「开倒车」用的是**机制与口径层面**的比对（我引用的是仓库自己写下的业界做法），不是引用外部数字。
- 上游 100M checkpoint 是否可复现（`*.pt` 不入库；我按仓库策略只审落盘证据）。

---

## 6. 需要 supervisor 代跑（无 shell）

- `python3 -m pytest tests/ -q`
- `python3 tools/check_schema.py`、`python3 tools/build_wiki.py --check`
- 若要落盘本报告与 6 份 CRITIQUE.md：请用 `git add <显式路径>`（**不要** `-A`），并在提交前
  `git diff --cached --name-only | wc -l` 人工核对（AGENTS.md 的 2026-10-02 第 3/5 次事故纪律）。
