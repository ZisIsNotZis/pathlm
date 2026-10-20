---
name: 2623-standard-early-exit-13m
ticket: WI-16 / R7
date: "2026-10-09"
grid: R7-existing-early-exit
engine: "depth-ar 13M（d=256 / L=8 / h=8 / seq=256 / bs=32，12k 步 = 98.3M tokens，seed 0；**seed-1 重复未跑**，见 analysis.md §7 #2）"
budget: "GPU 墙钟 ≤120 min（含冒烟 + 全部训练臂 + 推断电池 + 对账）"
---

# 2623-standard-early-exit-13m（R7 · WI-16）

> 主格：`R7-existing-early-exit`｜关联格：`C1-layers-per-token`、`C6-depth-supervision`、
> `R-references`｜引擎：depth-ar 13M｜预算：见 §6

## 1. 命题（一句话、可证伪）

**把「提前退出/自适应算力」的既有标准实现（CALM 式置信门控 / LayerSkip 式逐层损失 +
自草稿 / MoD 式逐 token 路由）在本仓引擎上本地实现成可测量的诚实对手后：C1 形态
（逐通道 carry 门 + 随机退出深度 + 混合填充）相对它们的增量，是否在「同 token 预算、
同解析算力口径」下仍然成立；以及 C1 的 13M matched-step 形态税 +0.3295 里，有多大一块
其实来自它那份**把主目标归一化降权**的训练配方（最终深度只占 1/9），而非形态本身。**

可证伪点：① 若「公平配方（主目标不降权 + 轻量早退项）」下的税与 legacy 配方的税相差
< 本格自测噪声 σ_cell ⇒ 「配方固有」这一解释被否证，C1 的税就是形态税；② 若某个标准
实现在同 token 预算下同 FLOPs 质量不劣且出口资产不劣 ⇒ 「C1 形态是独有资产」被否证。

## 2. 单因素声明（本实验的三条对照轴，各自只改一个因素）

| 对照 | 对照物 | 唯一变量 |
|---|---|---|
| 配方轴 | `c1-legacy` vs `c1-fair` | **损失归一**（加权平均 vs 主目标保权 + 轻量加法项）；trunk/退出/填充/调度逐字段相同 |
| 形态轴 | `c1-fair` vs `ls-fair` | **carry 门的有无**（`ls-fair` 参数少 L·(d²+d)=526k，块输出原样写入）；配方/退出/填充相同 |
| 基线轴 | `c1-fair` / `ls-fair` vs `stock` | **深度机制的有无**（stock = 普通 transformer，全深度、仅最终 CE） |
| 标准实现轴 | `mod` vs `stock` | MoD 路由的有无（**isoFLOP**：ratio·L = 8 = stock 的 8 层；代价是参数 ×2，见 §3） |

- **基底 = 主流标准 decoder-only transformer**（stock 臂就是它）。本实验不引入任何
  「主流 SOTA 的非标准因素」之外的结构；`c1-*` 的 carry 门是本仓既有形态（被试对象，
  不是新引入的混杂）。
- 「CALM 式」在本实验里是**推断期门控**（置信度阈值 + 标定），训练侧由 `ls-fair` 的
  轻量早退项提供可用的浅出口——CALM 的原始设定同样只用标准 LM 目标训练主干，
  出口分类器另训；本实验实现的是其**置信度门控**变体（其「学习式分类器/对比路由」变体
  需要第二个模型，列为未测项，见 §7）。

## 3. 指标表（多维、逐项）

| 维度 | 指标 | 它解决什么问题 |
|---|---|---|
| 质量（主） | 最终深度 bpc（13M，enwik8 val，teacher forcing） | 「税」的分子：形态/配方对主干质量的代价 |
| 质量（出口） | 逐深度 bpc 曲线 + `tf@k`（depth-k argmax == depth-L argmax） | 出口资产是否还在（早退可用性的前提） |
| 质量（CALM） | 标定后门控的**平均退出深度** + 与全深度流的**逐 token 一致率** | CALM 式门控的兑现度（该涨：一致率；该跌：平均深度） |
| 质量（LayerSkip） | 真解码路径的**草稿接受率**（γ=1/2/4/8，按位置） | 自草稿的接受率≠代理口径；必须在真路径上量 |
| 质量（MoD） | 最终 bpc + 逐层放行比例 | 路由是否真的在跑（活性量）+ 是否值 |
| 算力（等价） | **解析 layer-position / token**：CALM = 实测平均退出深度；LayerSkip = L(1+γ)/(1+τ)；MoD = ratio·L（实测复核） | **不得用墙钟当算力**（`docs/methodology.md` §3） |
| 墙钟 | 交错轮次（3 轮 × 4 prompt）单流解码 tok/s 的**比值中位 + 极差** | 部署口径；与算力口径必须并报（`SCHEMA` §17.9） |
| 内存 | 训练 peak / 解码 peak（`nvidia-smi` 前后 + torch 峰值） | 共享卡礼貌线；MoD 的近等算力不等于近等显存 |
| 活性 | 门均值（`c1-*`）、放行比例与路由器概率分离度（`mod`）、接受率与轮内位置曲线（spec） | `SCHEMA` §10：不许交付「死开关」 |
| 副产品 | 无门臂的参数账（−L·(d²+d)）、CALM 阈值曲线（ε → 平均深度） | 拆「容量」与「机制」 |

## 4. 判据线（预登记，事后不得调整）

**符号约定**：`Δ = bpc_对照 − bpc_实验`（**正 = 实验臂更好**，`SCHEMA` §17.3）。

**该涨**：
1. `Z1`：`Δ_recipe = bpc(c1-legacy) − bpc(c1-fair) > σ_cell`（本格自测噪声带，见 §5）。
   预期量级 ≥ 0.014（仓内 bpc 方差带）⇒ 主判据：**C1 的 matched 税里有一部分属于
   「目标降权」**，其点估计 = `Δ_recipe`，并给出「形态（c1-fair − stock）」的余项。
2. `Z2`：`tf@4(c1-fair) − tf@4(stock) > 0` 且量级与 `c1-legacy` 同阶（≥ 0.3）⇒
   出口资产不是降权配方的产物。
3. `Z3`：CALM 门控在该臂上「一致率 ≥ 0.95 时的平均退出深度」显著小于 L ⇒ 门控可用；
   否则（平均深度 ≈ L）如实报「不省」。

**该跌**：
4. `D1`（否证「C1 形态是独有资产」）：若 `bpc(ls-fair) ≤ bpc(c1-fair) + σ_cell` **且**
   `tf@4(ls-fair) ≥ tf@4(c1-fair) − σ_tf`（σ_tf = 接受率的二项 SE×2；`tf@4` 的 n = 81600
   个位置 ⇒ 2×SE ≈ 0.0033），则 C1 的门必须降级为「不必要」。反向（门更好）则写「门有
   增量」并给差值。
   **实测（预登记判据的合取未满足，如实登记、不事后改判据）**：第一条成立
   （bpc 好 0.2110），**第二条不成立**（0.8402 < 0.9102 − 0.0033）⇒
   **不得据 D1 判「门必须降级」**；「门买的出口资产可否由配方替代」降级为**待测**
   （判读见 `analysis.md` §3 D1 与 §7 未测项 #5）。
5. `D2`：若 `bpc(mod) − bpc(stock) > σ_cell`（MoD isoFLOP 更差）⇒ 「逐 token 路由在
   13M/enwik8 无头寸」；落在 ±σ_cell 内 ⇒ 写「不可分辨」，不得写「有效」。
6. `D3`：墙钟一律报**比值中位与极差**；极差 > 效应 ⇒ 写「不可分辨」。单次抽样倍率
   不得进结论（`C-AUDIT.md` §4①的教训）。

**必须不变**：
7. `I1`：`stock` 臂复现仓内锚点 `23d-dense-B-12k` 的 d8 bpc **1.5287**；差 > σ_cell ⇒
   先报测量/实现问题，结论不进账。
8. `I2`：LayerSkip 自草稿解码发出的 token 流与同 arm 的全深度贪心流**逐 token 相同**
   （质量零损失护栏，可失败断言）；`c1-fair`/`ls-fair` 的 teacher-forced 全深度前向
   在加/不加速率关闭时逐位相同。
9. `I3`：默认代码路径逐位不变：`aux_mode="renorm"` 的损失 == v1 加权平均式（单测
   `test_renorm_mode_matches_v1_formula_bitwise`）；`carry_gate=True` 且 `mod_ratio=0`
   时前向与历史路径一致（无门臂的 trunk 初始化随机数流必须与有门臂同 seed 逐位相同，
   单测钉住）。
10. `I4`：门活性量（`c1-*`）非零、MoD 放行比例 == 容量比例、路由器参数拿到梯度。

**主判据（primary）**：`Z1`（配方拆分的点估计 + 余项）+ `D1`（门是否有增量）。
其余为并列判据。

**复核后追加的对照臂（不改判据、不改口径）**：复核指出配方轴（`renorm` 除分母 vs
`additive` 不除）同时改了「主目标份额」与「全局损失尺度」⇒ 追加
`r7-c1-fair-share-s0`（`--aux-mode additive --aux-weight 8`：主目标份额 = 1/9，与 C1 现配方
同份额）作为**同份额控制臂**。它只回答「方向是否由份额造成」，**不替换**任何预登记判据；
「损失尺度」这一残余混淆如实记为未隔离（`analysis.md` §2 末段、§7 未测项 #4）。

## 5. 对照与方差（本格自测噪声门限）

- 对照臂见 §2；**同一数据**（enwik8_full，`data/enwik8_full.npz`，train 段训练、val 段
  10M bytes 评测）、同一 token 预算（12k×32×256 = 98.3M tokens）、同引擎、同优化器配方
  （AdamW lr 1e-3 / wd 0.01 / warmup-cosine / bf16 autocast / clip 1.0）。
- **σ_cell 本格自测**：`stock`、`c1-fair`、`ls-fair` 三臂各跑 seed 0 与 seed 1（其余两臂
  单 seed，如实标注），取三对 |Δbpc| 的最大值作为 σ_cell（保守），并逐对报出。
  只有 |Δ| > σ_cell 的 bpc 差异才进入排序结论；`tf@k` 用二项 SE 报。
  > **执行现状（指针，不改判据）**：**seed-1 重复被预算砍掉** ⇒ `σ_cell` **未直接测**，
  > 全部训练臂为 **n=1**，改用「两条臂各自复现历史同配置值」的漂移（0.0003/0.0007）与
  > 仓内种子带 0.0138 作界；负向结论按「方向反转、待复现」陈述。
  > 详见 `analysis.md` §7 #2 与 `config.json::noise`（`sigma_cell_measured: null`）。
- 对照复用：**不复用历史 run**（`c1-legacy` 臂重跑，避免跨 run 混杂；`SCHEMA` §9）。
- 墙钟：**同一进程内交错**（同一轮内依次测各臂）⇒ 外部占用对相对比值的影响被抵消；
  报中位与极差（`C-AUDIT.md` §4① 的修复口径）。

## 6. 成本（唯一预算数字）

- `gpu_wall_budget: 120 min`（含冒烟 + 全部训练臂 + 推断电池 + 对账；无 100M）。
- 先测步时再定预算：`SCHEMA` §17.8 要求成本 = 臂数 × **本实验实测步时**；冒烟结果写
  `run_log.txt` 的 `[smoke]` 行与 `config.json`，`analysis.md` 附表引用同一批数字。
- 峰值显存：**实测训练 935–1340 MB、解码 ≤190 MB**（两次实测 154.0 / 190.1 MB，取上界；
  出处见 `config.json::peak_gpu_mem_mb` 与 `analysis.md` §4 内存段；13M 级别；`SCHEMA` §20.2 的 5 GB
  礼貌线不构成约束）；
  实测（训练 peak + 解码 peak + `nvidia-smi` 前后）逐臂落盘。
- 可中断性：高（每臂独立 run 目录、独立落盘；臂间释放 GPU 锁 ≥60 s，`SCHEMA` §20.1）。
- **预算不够时砍什么**：先砍 seed-1 重复（先 `ls-fair` 后 `c1-fair`），再砍推断电池的
  重复轮次（3 → 2 轮）与 `mod` 臂；**不砍判据线、不改科学设定**。

## 7. 数据与脚本清单

| 项 | 值 |
|---|---|
| 数据 | `data/enwik8_full.npz`（字符级，V=206；train 段训练，val 段 10M 评测）；prompt = val 的 4 个固定 offset × 16 字符 |
| 训练入口 | `train_depth_ar.py`（既有 runner；本波新增 `--no-carry-gate` / `--aux-mode` / `--mod-ratio` / `--mod-router-weight`，默认全为历史行为） |
| 臂驱动 | `gpu_run.sh`（按臂取/放 GPU 锁，臂间 ≥60 s） |
| 推断入口 | `run.py`（深度曲线 / `tf@k` / CALM 标定与门控 / 真草稿链接受率 / 交错墙钟 / 内存） |
| 共用解码机械 | `exit_policies.py`（与 `DepthARDecoder` 同 fill 语义；**所有臂共用同一套**，唯一差别 = 策略） |
| 引擎新代码 | `pathlm/mod_router.py`（MoD 式路由）+ `DepthARConfig.carry_gate/aux_mode/mod_ratio` |
| 单测 | `tests/test_r7_early_exit_arms.py`（19 条） |
| 落盘 | `results.json`（含 `intervention` 活性块与 `accept_rate_window`/`n_accepts` 派生字段）/ `run_log.txt` / `cycle_log.jsonl`（spec 每轮一行；由 `run.py --stage cyclelog` 生成）/ `analysis.md` / `MODEL.md` / `config.json` / `controls/` / `enrich_results.py`（派生可复算） |
| 对账 | `reconcile_numbers.py` |

## 8. 失败决策树

1. **测量**：先查 `stock` 复现（`I1`）；若不复现，查 seed/数据切分/步数/精度四项配置，
   再查 `run_log.txt` 的步时与 `nvidia-smi` 争用；不复现即停，不进结论。
2. **实现**：跑 `tests/test_r7_early_exit_arms.py` 与 `tests/test_depth_ar.py`（全绿是前置）；
   无门臂/加法配方臂都要先过「全深度 == stock 逐位」与「解码 == 并行前向」两条单测。
3. **规模**：若差异落在 σ_cell 内，如实报「不可分辨」，并把尺度列为未测项；
   **不得**外推到 100M（`SCHEMA` §19/§17.11）。
4. **设计**：若标准实现全面不劣于 C1 形态，按设计层判读（「门/形态不是资产」），
   不推翻已有机制结论；并把 C1 的增量重新写成「+门在有/无**监督配方**下的差异」。

## 9. 产出

- `results.json`（机读：各臂训练结果 + 推断电池 + 交错墙钟原始行 + 内存）
- `run_log.txt`（`[smoke]` 步时 / 逐臂一行）、`cycle_log.jsonl`（spec 每轮一行活性量；
  `run.py --stage cyclelog` 生成，`SCHEMA` §17.4 的周期活性量要求）
- `analysis.md`（人话摘要 + 四腿表格 + 判据逐条判读 + 未测项）
- `MODEL.md`（各臂权重 sha256、体积、生成命令；权重不入库）
- `config.json`（resolved config：数据/参数/seed/步时/预算）
- `controls/`（对照臂与**复现漂移**的原始数字；`σ_cell` 未直接测，见 `analysis.md` §7 #2）

---

## §既有做法（prior art，`SCHEMA` §18）

> **检索能力声明**：本环境**无联网**（`C-AUDIT.md` §5 自述）。
> 下列条目的**具体数字**一律标 `记忆·未核`，只用于设计方向；本实验的判据与结论
> **只依赖本地实测**。机制层面的比对用的是本仓自己写下的业界做法（`C-AUDIT.md` §3）。

| 谁做过 | 它做了什么（机制） | 为什么这么设计 | 结果（口径） | 我们的增量 |
|---|---|---|---|---|
| **CALM**（Schuster et al. 2022，`记忆·未核`） | 推理期按**置信度阈值**（或学习式出口分类器）在浅层退出，阈值按「与全深度预测的一致率」标定 | 不牺牲任务质量的前提下省算力 | 报道口径多为**前向次数/加速比**（`记忆·未核`） | 本地首次把它实现成可测对照：**本站尺度 + 同一套解析算力口径 + 与自家形态同配方对比** |
| **LayerSkip**（Elhoushi et al. 2024，`记忆·未核`） | 同模型浅出口当**草稿**、全深度验证（self-speculative decoding）；训练加**逐层早退损失** | 不引入第二个模型，草稿与主干同分布 | 报道口径为吞吐/延迟（`记忆·未核`） | C1 的 2621 已测出「深度草稿省延迟不省 FLOPs（+5.6%）」；本实验补**公平配方**版本与 `stock`/无门形态的同刻度对照 |
| **MoD**（Raposo et al. 2024，`记忆·未核`） | 每块逐 token 路由，容量内 top-k 进块、其余残差旁路；路由器用二值辅助损失训练 | 每 FLOP 换更多**深度**（同 FLOPs 下可以更深） | 同 FLOPs 质量不劣（`记忆·未核`） | 本地首次实现；本站给出 isoFLOP 设计点（L=16/ratio 0.5 vs dense L=8）的**实测墙钟/显存/质量**，并声明参数 ×2 的固有非公平点 |
| **LayerDrop / stochastic depth**（`记忆·未核`） | 训练期随机丢层（本仓即「随机退出深度」） | 正则化 + 训练可退出的模型 | 训练正则化口径 | 本仓已把「随机退出」当成形态的一部分；本实验用它做**公平配方下的形态对照** |

**未找到直接前例**的部分：把「主目标是否被归一化降权」当成**单因素**、在同引擎上拆
「配方固有 vs 目标降权」——检索词（本地无联网，故只列本仓内检索）：`aux_weight`、
`final_weight`、`renorm`、`norm` 于 `pathlm/`、`.scratch/23-depth-ar`、`docs/methodology.md`。

## §22.8 能力面分析（派发前逐条）

| 契约要求 | 引擎支持？ | 处置 |
|---|---|---|
| 逐深度读出 + 置信度（CALM 门控） | ✅ `readout(h, depth=k)` 对 k=0..L 都在（`depth_curve`/`tf_acceptance` 已用） | 直接用；门控/标定在 `run.py` 实现 |
| CALM 的「学习式出口分类器 / 对比路由」变体 | ❌ 需第二个模型或独立分类器 + 另训 | **未测项**（§未测项第 2 条），不硬凑 |
| 逐层损失 + 随机退出（LayerSkip 训练侧） | ✅ `exit_mode=uniform` + 逐深度 CE | 直接用；「主目标不被降权」需新归一模式 |
| 损失归一单因素（公平配方） | ⚠️ 原引擎只有加权平均（`pathlm/depth_ar.py` 的 `loss/(w(n−1)+f)`） | **加 `aux_mode="additive"`**（默认 `renorm` 逐位不变，单测钉住） |
| 无门 trunk（LayerSkip 的「普通 transformer」形态） | ⚠️ 非 dense 路径恒有 carry 门 | **加 `carry_gate=False`**（不建门参数、块输出原样写入；初始化随机数流与有门臂逐位相同） |
| 自草稿解码（真 draft-verify 循环） | ❌ 引擎无 spec 解码循环（2621 的实现属该实验目录） | 在**本实验目录**实现 `exit_policies.py`，并单测「无门混合深度解码 == 并行前向」；所有臂共用同一套机械 |
| MoD 逐 token 路由 + 路由器损失 | ❌ 引擎无路由 | **新模块 `pathlm/mod_router.py`**（`mod_ratio>0` 才激活；默认路径逐位不变） |
| 组合校验 | ✅ `mod_ratio>0` 只与 `dense=True` 同开；`carry_gate=False` 只在非 dense 有意义（都在构造时显式报错） | 单测钉住（`test_mod_rejects_non_dense_and_bad_ratio` 等） |

## §23 七问自查

| # | 问题 | 本实验的答案 |
|---|---|---|
| 1 | 既有做法查过吗？ | 见「既有做法」；分诊 = **(b) 做实验**（我们设定：13M、单因素、同引擎、解析算力口径、主目标降权拆分——都不一样） |
| 2 | 判据是最严的那个吗？ | 诚实对手 = 同 token 预算的 stock 全深度 + 同 FLOPs 的标准实现；毁源/因果护栏 = 解码流逐 token 相同 + 无门臂「全深度 == stock 逐位」；活性 = 门均值/放行比例/接受率 |
| 3 | 三类「错的对照」试过吗？ | 同参数量：无门臂参数 −L·(d²+d)，MoD 参数 ×2（**显式声明**，并把它列为未测对照）；同算力：MoD 走 isoFLOP；同字节预算：不适用 |
| 4 | 规模/预算边界探过吗？ | **只有 13M 一档**（明确声明禁外推）；退出深度/门控阈值/spec γ 有多档 |
| 5 | 反方向/否证路径走完吗？ | 是：`D1`/`D2` 就是否证 C1 增量的路径；失败决策树四层 |
| 6 | 未测项列成「最小实验 + 成本」了吗？ | 见 `analysis.md` §未测项 |
| 7 | 复核通过了吗？ | 待独立复核（`SCHEMA` §14/§14b：本实验落盘后再派 fresh-context 只读复核者） |
