# 2629 · carry（delta 搬运通道）× 随机退出 × proj-fill（13M · 非 dense RAP 形态 · 合成 assoc-mqar）

> 契约来源：ticket 45 · W1 的**任务下发文本**（§4 逐字抄录，一字未改）。
> 本文件是该线的**执行契约**；§4b 记录派发前的**能力面分析**（SCHEMA §22.8）与仍存的
> 硬互斥（未测因子）。
> 入口：`train_arms.sh`（训练两臂）+ `run.py`（评测电池）+ `smoke_steptime.sh`（先测步时）；
> 语义层 `ragged_projfill_arms.py`（两臂单因素定义 + 能力面自检）；单测
> `tests/test_carry_ragged_projfill_13m.py`；变异 `mutate.py`；对账
> `reconcile_numbers.py`；原始结果 `results.json`；判读 `analysis.md`；
> 权重清单 `MODEL.md`；对照说明 `controls/README.md`。

## 1. 命题

**「delta 搬运通道 × 随机退出深度 × proj-fill」= 产出 100M `tf@4=0.9726` 出口资产的那条
配方，此前因引擎硬互斥从未可测（ENG2 刚解锁）。** 把它装起来：加上搬运通道之后，
① 训练分布内的长程召回得多少？② 出口资产 `tf@4` 在 carry 存在时**是否保持**？

一句话可证伪的形式：*把 `projfill`（随机退出 + proj-fill）与 `projfill_carry`
（同上 + delta 写）排成单因素对照，两臂只差 `carry` 一个因子；若目标臂的训练分布内多查询
召回显著高于基准臂（差值 CI 不重叠），则「搬运通道在该配方里带来召回增益」成立；若
`tf@4` 的差值落在 ±1.4pp 方差带内，则「出口资产在 carry 存在时保持」。*

## 2. 单因素声明

- **基底**：本仓 **非 dense RAP 形态**（`DepthARModel(dense=False, topology="sequential")`
  = 逐通道携带门 + 随机退出深度 + 混合填充），13.9M（d256 × 12 层 × 8 头 × mlp6 × T256）。
  这是产出 100M `tf@4=0.9726` 出口资产的**形态件**（100M 配方：随机退出 U{0..L} +
  fill + proj-fill + 全深度并行监督、最终深度 2×）。
- **唯一变量 = `carry_rule` 的开关**（两臂恰好是 (关) / (delta 写)）：

  | 臂 | `carry_rule` | `--exit-mode` | `proj_fill` | `fill_kv` | `dense` | 块类 |
  |---|---|---|---|---|---|---|
  | `projfill`（基准） | 关 | uniform（随机退出） | 开 | 开 | False | `Block` |
  | `projfill_carry`（目标） | **delta** | uniform | 开 | 开 | False | `CarryChannelBlock` |

- **必要形态件（两臂同款、不是变量）**：随机退出 + proj-fill + fill（100M 配方的形态件）；
  `--final-weight 2.0`（100M 配方的损失配方：权重均匀 + 最终深度 2×）。
- **训练/评测可见性一致**：两臂**都没有**训练期可见窗口（见 §4b：`train_window > 0`
  与 `dense=False` 硬互斥）⇒ 评测口径 = 全因果；长程压力来自**随机退出 + 填充**造成的
  逐位置可见性差异，不是「源 key 被推窗外」。**这是本实验与 2618/2627（dense + window16）
  在任务构造上的关键区别**，写在这里以免混淆。
- **无「必要性例外」**：两臂的任务、token 预算、评测批、退出模式逐字段相同（含 seed）。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

两臂**不改任何引擎代码**（本线不写 `pathlm/`、不写 `train_depth_ar.py`），只把既有开关
组合起来；新增的只有本实验自己的语义层 `ragged_projfill_arms.py`。

- **carry（`--carry-write delta`）**：单共享记忆槽，
  `S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；每层读出 `a ← a + γ ⊙ mem_t`，γ 单边零初始化
  （起步恒等于无通道臂、非鞍点）。**写入发生在任何层与任何退出之前、对全部位置（含后来
  会退出/填充的位置）各写一次 h₀** ⇒ 通道状态与退出模式无关（ENG2 的写入时机语义，
  `test_eng2_carry_write_timing_ignores_exit_depths` 钉住）。
- **随机退出（`--exit-mode uniform`）**：逐位置采样 `d_i ∈ U{0..L}`（两臂同款）。
- **proj-fill（`--proj-fill`）**：`fill_kv` 的混合 KV 源在已退出位置上再经恒等初始化的
  逐层 adapter remap（`kv_src = adapter(frozen)`）——它是**注意力 KV 源**的 remap，不经通道。

## 4. 判据线（**逐字**抄录任务下发文本，未改一字）

> 【W1 · GPU】carry × 随机退出 × proj-fill 质量实验（ENG2 刚解锁，≈45–60 min）
> 目录：axes/M-carrier/M1-carrier-type/experiments/2629-carry-ragged-projfill/
> 动机：**产品上真正有意义的那对**——「delta 搬运通道 × 随机退出深度 × proj-fill」= 产出 100M `tf@4=0.9726` 出口资产的那条配方——此前因引擎硬互斥**从未可测**；ticket 45-E1（ENG2）刚解锁。
> 干预（同 token 预算、逐字段一致）：`proj-fill 基线`（无 carry，随机退出）vs `proj-fill + carry(delta)`。任务 = 合成 assoc-mqar。
> 判据（预登记）：① 主腿 = **训练分布内**（§17.18）多查询关联回忆召回（chance + 二项 CI）+ **毁源对照臂**；② **出口资产 tf@4**（这是本条线的产品卖点）；③ 每步增量（§17.17 分量分解）；④ clean bpc 作辅助（跨分布，注明合成训练）；⑤ 若能复现 100M 量级的 tf@4≈0.97，明确记录「出口资产在 carry 存在时是否保持」；⑥ 禁外推（13M/单 seed）。

**执行时的操作性定义（不改判据，只把口径落定）**：

- **(1) 的「训练分布内多查询召回」**：主腿 = **`in_distribution_random_exit`** —— 用
  训练**同款**的随机退出模式（`d_i ∈ U{0..L}`，两臂**成对**：同一批 + 同一退出模式，
  seed 决定）在训练型多查询 `assoc-mqar`（K=8、Q=8、gap=14；查询-键距离 ≤240，
  §17.18）上评测。并列口径 = **`full_depth_deployment`**（退出深度恒 = L，部署全栈模式；
  训练时从未见过全深度 ⇒ 严格说是 OOD，仅作并列参考）。**毁源对照**：把源段 `[0, 16)`
  换成同分布新随机符号（`corrupt_span`），任一臂毁源后 **CI 下界 > 4×chance** ⇒ 该臂读数作废。
  两臂比较用 **Wilson 95% CI + `two_proportion` 差值 CI**。
- **(2) 的「出口资产 tf@4」**：`tf@4` = depth-4 出口 argmax 与全栈 argmax 的一致率。
  两个口径并报：**enwik8**（对齐 100M 的 0.9726）与 **synth 训练分布内**（多查询 assoc-mqar，
  **真实非 dense 前向**的 `aux["states"]`）。
- **(3) 的「每步增量」**：§17.17 的**分量分解**——两臂设计下净和 == 单因素增量，仍显式给出
  两列；对**召回**（越高越好）、**tf@4**（越高越好）、**clean bpc**（越低越好）分别给。
  判据线用 **pp**，方差带 = 1.4pp。
- **(4) 的「clean bpc 作辅助」**：训练在**合成**任务上 ⇒ clean bpc（enwik8 全因果）是
  **跨分布**读数，只作非退化锚 + 质量代价项，**不作召回判据**。
- **(5) 的「出口资产在 carry 存在时是否保持」**：两臂 `tf@4(enwik8)` 都 ≥0.90 ⇒ 100M 量级
  （0.9726）在本尺度/步数下复现；`|Δcarry| ≤ 1.4pp` ⇒ 「保持」。两件事分开记。
- **(6) 的「禁外推」**：13M / 单 seed，结论只在本尺度成立。

## 4b. 能力面分析（SCHEMA §22.8，派发前逐条做）

契约要求的每个 flag/组合，引擎是否支持？`grep -n "ValueError\|互斥" pathlm/depth_ar.py`
列出的校验逐条核对：

| 组合 | 引擎现状 | 本线 |
|---|---|---|
| `carry_rule` ⊕ `dense=False`（随机退出） | **ENG2 后允许**（`depth_ar.py` 解耦了 carry 的 `dense=True` 硬校验） | `projfill_carry` 臂 |
| `proj_fill` ⊕ 非 dense | **允许**（`fill_kv` 缺省 True；非 dense 前向有 `kv_from = where(active, h, adapter(frozen))`） | 两臂 |
| `carry × 非 dense × 随机退出 × proj-fill`（四者齐开） | **可建可前向**（本线 `capability_probe()` + 单测钉住；读取量非零） | 目标臂 |
| `train_window > 0` ⊕ `dense=False` | **仍硬互斥**（`raise ValueError`，`depth_ar.py:1834`） | 未用（见下方未测因子） |
| `proj_fill` ⊕ `fill_kv=False` | **报错**（`proj_fill requires fill_kv=True`） | 两臂均 `fill_kv=True` |

**关键实测（CPU，2026-10-06，`capability_probe()`）**：两臂都能建；块类 =
`{projfill: Block, projfill_carry: CarryChannelBlock}`；**主干参数逐位相同**（同 seed，
carry 参数只在目标臂 ⇒ 单因素对照成立，§9/§15）；目标臂多出的参数全部含 `carry`
（`carry.*` + 块上的 `carry_scale`），**没有偷改主干**。

**未测因子（显式声明，禁止外推）**：本线**没有**训练期可见窗口（非 dense 路径不允许）
⇒「源 key 被推到窗外、不搬就答不出」的那种**强长程压力**在本线**不存在**；本线的长程
压力只来自随机退出 + 填充的逐位置可见性差异。因此本线结论**不得**外推到
「window × carry × 非 dense」这一格（它在引擎里仍不存在）。

> 本实验**不改引擎绕开互斥**（SCHEMA §22.8 禁止在未获授权时改引擎）。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向、**不得当判据**；本地条目给文件路径。可检索时应查的关键词
> 见本块末尾。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果（记忆·未核 / 本地） | 我们的增量 |
|---|---|---|---|---|
| 1 | **显式写入通道 / delta rule（fast weight）**：Schmidhuber 1992；Schlag 等 2021（delta rule）；Yang 等 2024（Gated DeltaNet） | 用固定状态把窗口外信息搬到可达范围 | 本地 2618：dense + window16 下无通道 0.0137≈chance、delta 写 0.0950（6.08×） | **本线**把该通道装到**非 dense 随机退出 + proj-fill** 的 100M 形态件上，问「在无窗口、注意力可见全部源的条件下，通道还带来什么」 |
| 2 | **深度自适应 / 早退（RAP 形态件）**：逐通道门 + 随机退出 + KV 填充（本地 ticket 23/24） | 用浅层出口当草稿、全栈验证加速解码；填充让已退出位置仍贡献 KV | 本地 ticket 23：13M 12k proj-fill `tf@4=0.9788`；ticket 24 100M `tf@4=0.9726` | **本线**在该形态件上叠加搬运通道，量它对出口资产的**保持性**（这是产品的核心担心：加了机制会不会把出口资产打坏） |
| 3 | **机制组合的干扰**（记忆·未核）：多任务 / 多监督竞争同一表征容量 | 叠加的两个机制可能互相干扰 | 本地 2627：carry × dense-aux 的交互项 +0.38pp（可加）；远端残差 carry 读数见 §备注 | **本线**是**首次**把 (carry × 随机退出 × proj-fill) 排成单因素对照（此前引擎硬互斥、从未可测） |

**若可检索应查**：`delta rule fast weight MQAR`、`ragged batch variable depth KV fill`、
`early exit draft acceptance small transformer`、`memory channel + variable-depth inference interference`。

## 5. 对照与方差

- **基准 = `projfill` 臂**（随机退出 + proj-fill，**无 carry**；本仓「基准 = 同配方但关掉
  被检验的那一个因子」口径）。判据是**单因素对照**（目标 − 基准），**不与裸 stock 比**
  （那会把形态件的代价算进 carry 的账）。
- **方差带 = 0.014 bpc / 1.4pp 召回**（`docs/methodology.md`）；召回判据用 **Wilson 95% CI**
  与 `two_proportion` 的差值 CI、`ci_overlap`。
- **局部先验地板** = `local_prior_accuracy(k=8)`（§19：合成任务上局部先验必须 ≈ chance）；
  **chance** = `1/sym_vocab` = 1/64 = 0.015625。
- **逐臂毁源对照**（§4 (1)）。
- **纯推理、无随机采样**：同 seed 评测应逐位相同；两臂**成对**（同一批 + 同一退出模式）。
- **逐字段对照表**（唯一变量 = carry 的开关）：

  | 字段 | `projfill` | `projfill_carry` |
  |---|---|---|
  | seed | 0 | 0 |
  | 训练数据 | assoc-mqar（K8/Q8/gap14） | 同 |
  | 步数×批×T | 6000×32×256 | 同 |
  | lr / 优化器 / schedule | 1e-3 / AdamW+cosine | 同 |
  | `exit_mode` | uniform（随机退出） | 同 |
  | `fill_kv` / `proj_fill` | True / True | True / True |
  | `final_weight` | 2.0 | 2.0 |
  | `dense` / `topology` | False / sequential | 同 |
  | **`carry_rule`** | **None** | **delta** |
  | 参数量 | 见 `results.json::param_declaration` | 见同处（+carry 参数） |

## 6. 成本（§8 的唯一预算数字 + 砍什么）

```
gpu_wall_budget: 45 min   # 含两臂训练 + 评测（纯推理）
```

- **实测步时**（先测后定，§17.8/§20.3；`smoke/steps300/`，300 步稳态）：
  `projfill` **102.3** / `projfill_carry` **144.5** ms/step（step0→299；
  末段 step200→299 为 109.1 / 139.4）⇒ 6k×2 推算 ≈ **24.7 min**。
- **步数预算的依据（pilot，`smoke/pilot1500/`）**：预算定案前用 1500 步基线臂探过一条
  学习曲线：末步 `synth_answer_ce=4.13` vs chance `4.16` ⇒ **1500 步远不够**（任务尚未
  学起来）。故 6k 是必要下限，取 6k（与 13M MQAR 谱系同刻度）。
- **实测训练墙钟**（`train_<arm>.log` / `train_arms.runlog`）：见 `analysis.md` §成本；
  两臂间留 65s 窗口（§20.1）。评测电池纯推理，峰值 ≪5GB net。
- **预算不够时砍什么**（§17.7）：砍**步数（两臂同步砍，不砍单臂）** > 砍评测批数；
  **不砍判据线**（主腿召回 + 毁源、tf@4、单因素增量、参数量/活性量都必须出）。
- **§20.1 锁记账**：`train_arms.sh` **按臂**取/放锁、臂间 65s 窗口（不整程独占）；
  本波另有 GPU 线（A6 长 T 线等）在同一锁上串行，等待时长记入 `train_arms.runlog`。
- **§20.3**：若共享卡被外部占用降速，则**缩步数保「臂间同刻度」**（不砍臂、不换配方），
  并在报告首段声明「Nk 快照口径、与既往 Mk 不可比」，附答案行 CE 末窗斜率。

## 7. 数据与脚本清单

- **训练数据**（§19：**不用 enwik8 当长程主任务**）：
  `tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`（K=8 对、Q=8 查询、gap=14）。
- **评测**（合成长程任务）：`assoc_recall_multi_batch`（**主判据腿** + 毁源对照）、
  `assoc_recall_batch`（训练分布外外推参照）。
- **clean bpc 腿**：`data/enwik8_full.npz` 的评测尾段（vocab 206，与模型词表一致）——
  跨分布、合成训练，只作非退化锚 + 质量代价项。
- **出口资产**：`pathlm/depth_ar.py::tf_acceptance`（enwik8）+ `run.py::per_depth_agreement`
  （synth 分布内，真实非 dense 前向）。
- **入口**：`train_arms.sh`（训练）、`python3 run.py`（评测，`config.json`）、
  `smoke_steptime.sh`（先测步时）。
- **落盘**：`results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}`、
  `train_<arm>.log`、`train_arms.runlog`、`analysis.md`、`MODEL.md`、`controls/README.md`、
  `reconcile_numbers.py → reconcile_report.json`、`mutate.py → mutation_report.json`、
  `smoke/`（`steps300/` 步时 + `pilot1500/` 步数依据 + `results.json` 冒烟）、
  `pytest_full.txt` / `pytest_line.txt`。
- **测试**：`tests/test_carry_ragged_projfill_13m.py`（12 条可失败单测 + 变异验证 `mutate.py`）。

## 8. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：主腿（随机退出）与并列腿（全深度）是否分列？chance/局部先验是否同值？
   毁源是否同一批？两臂退出模式是否成对（同 seed）？→ 先修 `run.py` 的接线，**不得**动模型。
2. **实现**：目标臂 `carry read_share` 恒 0（通道没被用）或 `attn_carry` 在非 dense 前向
   遥测里缺档 → 跑变异验证排除死开关；`read_scale` 仍单边零 = **步数**问题，如实记录、
   不外推。
3. **规模**：两臂都 ≈ chance ⇒ 该任务/该尺度上（无窗口）注意力本身没学会 ⇒ 只能读
   「臂间增量」，**不得**表述为「搬运不可能」（SCHEMA §19）。
4. **设计**：某臂毁源后仍高 ⇒ 读数混杂、该臂作废；`tf@4` 被 carry 打坏且远超方差带 ⇒
   用分量分解（召回 vs bpc vs tf）分开读，并在结论里写清是设计还是口径。

## 9. 产出与复核

- `analysis.md` 给**人话摘要 + 七问**（§23）+ 单因素表格 + 判读；`MODEL.md` 给权重哈希与
  代码指纹；`controls/README.md` 给对照臂与变异说明。
- 本实验完成后按 SCHEMA §14（含 §14b 顺序屏障）另派 fresh-context **只读复核者**独立检查。

## scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；本实验是 M1 载体类型格内的**单因素对照**
（delta 搬运通道 × 随机退出 × proj-fill 的组合，唯一变量 = `carry_rule` 的开关）。
`attn_sparse_k` / `attn_linear` / `train_window` 与上述因子的交互仍受引擎互斥限制，
是显式留白、不是遗漏；「window × carry × 非 dense」是**显式未测因子**（§4b）。
`deps` 只记录已知耦合方向（M1 载体类型 ← 2618 搬运通道线 / ticket 24 出口资产线 /
ENG2 引擎解耦）。
