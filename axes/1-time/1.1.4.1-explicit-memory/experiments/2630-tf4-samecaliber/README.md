# 2630 · 出口资产**同口径**判决（carry 是否弄坏 tf@4）· 13M · enwik8 原生预训练口径

> 契约来源：ticket 46 · W1 的**任务下发文本**（§4 逐字抄录，一字未改）。
> 本文件是该线的**执行契约**；§4b 记录派发前的**能力面分析**（SCHEMA §22.8）与仍存的
> 硬互斥（未测因子）。
> 入口：`train_arms.sh`（训练两臂）+ `run.py`（评测电池）+ `smoke_steptime.sh`（先测步时）；
> 语义层 `samecaliber_arms.py`（两臂单因素定义 + 能力面自检）；单测
> `tests/test_tf4_samecaliber_13m.py`；变异 `mutate.py`；对账 `reconcile_numbers.py`；
> 原始结果 `results.json`；判读 `analysis.md`；权重清单 `MODEL.md`；对照说明
> `controls/README.md`。

## 1. 命题

**46-W1（2629）的出口资产两个口径结论相反，原因疑为「模型训在合成任务上 ⇒ enwik8 对它
是跨分布」。本线把两臂直接训在 enwik8 上，得到「训练分布 == 评测分布」的**同口径**
tf@4 判决。**

一句话可证伪的形式：*把 `base`（随机退出 + proj-fill，无 carry）与 `carry`（同上 + delta
写）排成单因素对照，**两者都在 enwik8 上训练、都在 enwik8 上评 tf@4**；若 |Δtf@4| < 1.4pp，
则「carry **不弄坏**出口资产」在该同口径下成立；若 |Δ| ≥ 1.4pp，则不成立。*

## 2. 单因素声明

- **基底**：本仓**非 dense RAP 形态**（`DepthARModel(dense=False, topology="sequential")`
  = 逐通道携带门 + 随机退出深度 + 混合填充），13.9M（d256 × 12 层 × 8 头 × mlp6 × T256）。
  这是产出 100M `tf@4=0.9726` 出口资产的**形态件**（100M 配方：随机退出 U{0..L} + fill +
  proj-fill + 全深度并行监督、最终深度 2×）。
- **唯一变量 = `carry_rule` 的开关**（两臂恰好是 (关) / (delta 写)）：

  | 臂 | `carry_rule` | `--exit-mode` | `proj_fill` | `fill_kv` | `dense` | 训练数据 | 块类 |
  |---|---|---|---|---|---|---|---|
  | `base`（基准） | 关 | uniform（随机退出） | 开 | 开 | False | **enwik8** | `Block` |
  | `carry`（目标） | **delta** | uniform | 开 | 开 | False | **enwik8** | `CarryChannelBlock` |

- **必要形态件（两臂同款、不是变量）**：随机退出 + proj-fill + fill（100M 配方的形态件）；
  `--final-weight 2.0`（100M 配方的损失配方：权重均匀 + 最终深度 2×）。
- **训练/评测可见性一致**：两臂**都没有**训练期可见窗口（`train_window > 0` 与
  `dense=False` 硬互斥）⇒ 评测口径 = 全因果。长程压力来自随机退出 + 填充的逐位置可见性
  差异，不是「源 key 被推窗外」。
- **口径不混用（关键）**：本线 = **enwik8 训练 + enwik8 评测的同口径**；2629 = **合成
  assoc-mqar 训练 + enwik8 跨分布代理**。这是**两个不同分布的口径，数字不得混用**、
  不得拼成一对「臂」。
- **无「必要性例外」**：两臂的任务、token 预算、评测批、退出模式逐字段相同（含 seed）。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

两臂**不改任何引擎代码**（本线不写 `pathlm/`、不写 `train_depth_ar.py`），只把既有开关
组合起来；新增的只有本实验自己的语义层 `samecaliber_arms.py`。

- **carry（`--carry-write delta`）**：单共享记忆槽，
  `S_t = S_{t−1}(I − β k̂ k̂ᵀ) + β v k̂ᵀ`；每层读出 `a ← a + γ ⊙ mem_t`，γ 单边零初始化
  （起步恒等于无通道臂、非鞍点）。写入发生在任何层与任何退出之前、对全部位置各写一次 h₀
  ⇒ 通道状态与退出模式无关（ENG2 的写入时机语义）。
- **随机退出（`--exit-mode uniform`）**：逐位置采样 `d_i ∈ U{0..L}`（两臂同款）。
- **proj-fill（`--proj-fill`）**：`fill_kv` 的混合 KV 源在已退出位置上再经恒等初始化的
  逐层 adapter remap（`kv_src = adapter(frozen)`）——它是**注意力 KV 源**的 remap，不经通道。

## 4. 判据线（**逐字**抄录任务下发文本，未改一字）

> 【W1 · GPU】出口资产同口径判决：carry 是否弄坏 tf@4（≈40 min）
> 目录：axes/M-carrier/M1-carrier-type/experiments/2630-tf4-samecaliber/
> 动机：46-W1 的出口资产两个口径**结论相反**（分布内 synth：99.66%→98.68% 带内保持；enwik8 跨分布代理：93.69%→88.98% 超带恶化）。原因可能是「模型训在合成任务上 ⇒ enwik8 对它是跨分布」。**必须同口径判决。**
> 干预：两臂 = `无 carry` vs `carry(delta)`，**在 enwik8 上训练**（原生预训练口径、非 dense ragged + proj-fill + 随机退出），其余逐字段一致。
> 判据（预登记）：① 主腿 = **同口径 tf@4**（enwik8 训练、enwik8 评测）；若 |Δ| < 1.4pp ⇒ 记「carry 不弄坏出口资产」；② clean bpc（同口径）；③ 参数量；④ 活性量；⑤ 注明与 46-W1 是两个不同分布的口径、**不得混用**。

**执行时的操作性定义（不改判据，只把口径落定）**：

- **(1) 的「同口径 tf@4」**：`tf@4` = depth-4 出口 argmax 与全栈 argmax 的一致率（TF 代理），
  用引擎 `pathlm.depth_ar.tf_acceptance(model, eval_arr, n_batches=20, batch_size=16)` 在
  **enwik8 的评测段**上算——`eval_arr` 与训练时 `load_enwik8_full` 的同一 eval 切分；
  两臂用同一引擎函数、同一批（`generator` seed 3）、同一批数（20×16=320 个 256 长序列）。
  主判据 = `tf4_enwik8_samecaliber`；判定函数 = `samecaliber_arms.tf4_holds(Δ)`：
  `|Δ(carry−base)| < 1.4pp ⇒ 保持`。副口径 `tf@6` 并列报。
  - **基准唯一（§17.14）**：基准 = `base`（同配方但关掉 carry）；代价项（clean bpc）也相对
    `base` 算；主判据 = 同口径 tf@4，其余（tf@6 / clean bpc / 参数 / 活性）为并列项。
- **(2) 的「clean bpc（同口径）」**：引擎 `depth_curve(model, eval_arr, n_batches=20,
  batch_size=16)` 的**全深度末点**（depth=L=12）bpc；同分布、非退化锚 + 质量代价项。
- **(3) 的「参数量」**：逐臂 `n_params` 声明 + `Δ = carry − base`（+1.845%）。
- **(4) 的「活性量」**：`train/<arm>/train_log.jsonl` 里每 200 步的 `attn_carry` 遥测
  （`read_share` / `read_scale_absmean` / `state_fro` / `beta_mean`）；无 carry 臂为 null。
  `run.py` 另落 SCHEMA §10 的 `intervention` 块（`params_added` / `w_fro` / `w_absmax` /
  `nonzero_frac`）。
- **(5) 的「口径不混用」**：`analysis.md` 与 `results.json` 显式标注本线是 enwik8 同口径，
  2629 是合成训练 + 跨分布代理，**分开写**；`run.py` 另报一条**反向跨分布诊断**
  （enwik8 训好的模型在 synth 上评 tf@4），只用于解释「口径为何会打架」，**不作判据**。
- **(6) 的「禁外推」**：13M / 单 seed，结论只在本尺度/本口径成立。

## 4b. 能力面分析（SCHEMA §22.8，派发前逐条做）

契约要求的每个 flag/组合，引擎是否支持？`grep -n "ValueError\|互斥" pathlm/depth_ar.py`
列出的校验逐条核对：

| 组合 | 引擎现状 | 本线 |
|---|---|---|
| `carry_rule` ⊕ `dense=False`（随机退出） | **ENG2 后允许**（`depth_ar.py` 解耦了 carry 的 `dense=True` 硬校验） | `carry` 臂 |
| `proj_fill` ⊕ 非 dense | **允许**（`fill_kv` 缺省 True；非 dense 前向有 `kv_from = where(active, h, adapter(frozen))`） | 两臂 |
| `carry × 非 dense × 随机退出 × proj-fill`（四者齐开） | **可建可前向**（`capability_probe()` + 单测钉住；读取量非零） | 目标臂 |
| **训练数据 = enwik8**（`--synth-task` 缺省 `none`） | **允许**（`train_depth_ar.py` 在 `synth is None and not dense` 分支跑 enwik8 电池：`depth_curve` + `tf_acceptance`） | 两臂 |
| `train_window > 0` ⊕ `dense=False` | **仍硬互斥**（`raise ValueError`，`depth_ar.py:1834`） | 未用（见未测因子） |
| `proj_fill` ⊕ `fill_kv=False` | **报错**（`proj_fill requires fill_kv=True`） | 两臂均 `fill_kv=True` |

**关键实测（CPU，`capability_probe()`）**：两臂都能建；块类 =
`{base: Block, carry: CarryChannelBlock}`；**主干参数逐位相同**（同 seed，carry 参数只在
目标臂 ⇒ 单因素对照成立，§9/§15）；目标臂多出的参数全部含 `carry`
（`carry.*` + 块上的 `carry_scale`），**没有偷改主干**。

**关键实测（GPU，`smoke/steps300/`）**：两臂都能在 **enwik8** 上跑
`非 dense + 随机退出 + proj-fill`（本线是这条数据口径的**首跑**）——`base` 300 步
loss 5.27→2.79、`carry` 300 步正常（`attn_carry` 遥测非空）。

**未测因子（显式声明，禁止外推）**：本线**没有**训练期可见窗口（非 dense 路径不允许）
⇒「源 key 被推到窗外、不搬就答不出」的**强长程压力**在本线**不存在**。因此本线结论
**不得**外推到「window × carry × 非 dense」这一格（它在引擎里仍不存在）。

> 本实验**不改引擎绕开互斥**（SCHEMA §22.8 禁止在未获授权时改引擎）。

## 既有做法（prior art，SCHEMA §18）

> **来源标注纪律**：本环境**联网检索不可用**（未注册 web / 检索工具）。外部条目一律标
> `记忆·未核`，只作设计方向、**不得当判据**；本地条目给文件路径。可检索时应查的关键词
> 见本块末尾。

| # | 谁做过（记忆·未核） | 为什么这么设计 | 实测结果（记忆·未核 / 本地） | 我们的增量 |
|---|---|---|---|---|
| 1 | **深度自适应 / 早退草稿（RAP 形态件）**：本地 ticket 23/24（逐通道门 + 随机退出 + KV 填充） | 用浅层出口当草稿、全栈验证加速解码；填充让已退出位置仍贡献 KV | 本地 ticket 23：13M 12k proj-fill `tf@4=0.9788`；ticket 24 100M `tf@4=0.9726` | **本线**首次把「出口资产是否被搬运通道弄坏」放到**训练分布内**（enwik8）判，而不是跨分布代理判 |
| 2 | **显式写入通道 / delta rule（fast weight）**：Schmidhuber 1992；Schlag 等 2021；Yang 等 2024 Gated DeltaNet（`记忆·未核`） | 用固定状态把窗口外信息搬到可达范围 | 本地 2618：dense + window16 下无通道 0.0137≈chance、delta 写 0.0950（6.08×）；2629：合成训练下 tf@4 分布内 −0.98pp（保持）/ enwik8 跨分布 −4.71pp（恶化） | **本线**把 2629 的「两口径相反」用**同口径**（enwik8 训 + enwik8 评）判决，拆掉跨分布混杂 |
| 3 | **早退 / 辅助出口的分布外退化**（记忆·未核）：浅监督出口在分布漂移下更脆的普遍观察 | 浅层出口依赖的特征更局部，分布漂移时先坏 | 记忆·未核：多篇 early-exit 工作报告 OOD 下浅出口准确率下降更快 | **本线**给出一例可核对的**反证/正证**：把训练分布换成 enwik8 后，carry 的 tf@4 是否仍在带内——这是可失败的判决，不是复述 |

**若可检索应查**：`delta rule fast weight MQAR`、`ragged batch variable depth KV fill`、
`early exit draft acceptance distribution shift`、`memory channel + variable-depth inference interference`。

## 5. 对照与方差

- **基准 = `base` 臂**（随机退出 + proj-fill，**无 carry**；本仓「基准 = 同配方但关掉被
  检验的那一个因子」口径）。判据是**单因素对照**（目标 − 基准），**不与裸 stock 比**
  （那会把形态件的代价算进 carry 的账）。
- **方差带 = 1.4pp**（`docs/methodology.md` 的 Δ < 0.014 不可排序）。
- **两臂唯一变量 = carry**：见 §2 表；逐字段对照表另见 `controls/README.md`。
- **纯推理、无随机采样**：同 seed 评测应逐位相同；两臂用同一引擎函数、同一批、同一批数。
- **tf@4 是快照口径**：6k 步 / 49M token；末窗全深度 CE 与 depth-4 CE 斜率并列报（§17.15）。
- **逐字段对照表**（唯一变量 = carry 的开关）：

  | 字段 | `base` | `carry` |
  |---|---|---|
  | seed | 0 | 0 |
  | **训练数据** | **enwik8（90M/10M 切分）** | **同** |
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
gpu_wall_budget: 40 min   # 含两臂训练 + 评测（纯推理）
```

- **实测步时**（先测后定，§17.8/§20.3；`smoke/steps300/`，300 步稳态，**在 enwik8 上**）：
  `base` **106.4** ms/step / `carry` **150.5** ms/step（step 0→299）⇒ 6k×2 推算
  ≈ **25.7 min** 训练 + 两臂末段 enwik8 电池（eval-batches 20）≈ 数分钟 + 臂间 65s。
- **预算不够时砍什么**（§17.7）：砍**步数（两臂同步砍，不砍单臂）** > 砍评测批数；
  **不砍判据线**（同口径 tf@4 + clean bpc + 单因素增量 + 参数量/活性量都必须出）。
- **§20.1 锁记账**：`train_arms.sh` **按臂**取/放锁、臂间 65s 窗口（不整程独占）；
  等待时长记入 `train_arms.runlog`。
- **§20.3**：若共享卡被外部占用降速，则**缩步数保「臂间同刻度」**（不砍臂、不换配方），
  并在报告首段声明「Nk 快照口径、与既往 Mk 不可比」，附全深度 CE 末窗斜率。

## 7. 数据与脚本清单

- **训练数据**：`data/enwik8_full.npz`（+ `.tmp/enwik8` 原始字节），引擎
  `pathlm.data.load_enwik8_full`（train 90M / eval 10M，vocab 206）——**两臂同款**。
- **主判据 tf@4**：`pathlm/depth_ar.py::tf_acceptance`（enwik8 = **同口径**）。
- **clean bpc**：`pathlm/depth_ar.py::depth_curve`（enwik8 全深度末点）。
- **反向跨分布诊断**：`tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`
  + `run.py::per_depth_agreement`（synth，**不作判据**）。
- **入口**：`train_arms.sh`（训练）、`python3 run.py`（评测，`config.json`）、
  `smoke_steptime.sh`（先测步时）。
- **落盘**：`results.json`、`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json,model.pt}`、
  `train_<arm>.log`、`train_arms.runlog`、`analysis.md`、`MODEL.md`、`controls/README.md`、
  `reconcile_numbers.py → reconcile_report.json`、`mutate.py → mutation_report.json`、
  `smoke/steps300/`（步时）、`pytest_full.txt` / `pytest_line.txt`。
- **测试**：`tests/test_tf4_samecaliber_13m.py`（13 条可失败单测 + 变异验证 `mutate.py`）。

## 8. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：tf@4 的 `by_depth` 是否单调、`tf4` 是否 == `by_depth[4]`？两臂是否同一批
   （n_batches/batch_size 相等）？run.py 与训练入口的 tf@4 是否逐位一致？→ 先修接线，
   **不得**动模型。
2. **实现**：`carry` 臂 `read_share` 恒 0（通道没被用）或 `attn_carry` 遥测缺档 → 跑变异
   验证排除死开关；`read_scale` 仍单边零 = **步数**问题，如实记录、不外推。
3. **规模**：两臂 tf@4 都 ≈ 0.5 以下（出口资产没成形）⇒ 该步数不足以形成出口资产 ⇒
   如实报「快照未成形」并按 §20.3 缩步数保同刻度，注明与 2629 不可比。
4. **设计**：若同口径下 |Δ| ≥ 1.4pp（carry 确实弄坏）⇒ 明确记为「carry 在与返回分布
   一致的出口资产上有代价」，给出分量分解（clean bpc 税 vs tf@4 降），**不**合并成
   「机制无用」。

## 9. 复现手段

```bash
bash axes/M-carrier/M1-carrier-type/experiments/2630-tf4-samecaliber/train_arms.sh   # 训练两臂
python3 axes/M-carrier/M1-carrier-type/experiments/2630-tf4-samecaliber/run.py        # 评测电池
python3 axes/M-carrier/M1-carrier-type/experiments/2630-tf4-samecaliber/reconcile_numbers.py
python3 axes/M-carrier/M1-carrier-type/experiments/2630-tf4-samecaliber/mutate.py
python3 -m pytest tests/test_tf4_samecaliber_13m.py -q
```

权重哈希见 `MODEL.md`（权重不入库）。
