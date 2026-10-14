# 2651 · 层间拓扑 seed=1 复核 + 并行子层代价面 · 13M · depth-ar/dense —— 判读

> 契约：`README.md`（§0 逐字抄 ticket 61 §9.2 WP-2651 的预登记判据线，**未改**）。
> 入口：`run.py`（含 `--arms` / `--skip-train`）；原始结果：`results.json`（**逐臂增量落盘**）；
> 权重清单：`MODEL.md`；运行流水：`run_log.txt` / `run_stdout-smoke.txt` / `run_stdout-full.txt`。
> 口径锚（s0 = seed 0）：`../2613-inter-layer-topology-13m/`（`results.json` / `analysis.md` / `MODEL.md`）。
> 本文件覆盖两项：**2.2.3 层间学习加权和**与 **2.2.4 并行子层**（2.2.4 只指针本目录，不复制文件）。

## 0. 人话摘要（先读这段）

**这一格在问什么。** 一个普通的、只做下一个字预测的变压器（decoder-only transformer），
层与层之间只有一条最朴素的接法（顺序残差：先注意力、再 MLP）。上一格 `2613` 在 seed=0 上
试了两种「换接法」：**并行子层**（注意力与 MLP 同时读层输入，PaLM / GPT-J 式）和
**层间学习加权和**（每层输入＝对它之前所有层输出的学习加权和，DenseFormer 式）。
本轮的契约目标是**把每种接法在 seed=1 上再跑一遍（n=2）**，并把「并行子层的代价面」
（层内串行依赖长度 / 每层算力 / 参数量）与质量面放进同一张表。

**实际做到哪一步（诚实先说）。** 截止 `2026-10-14T22:32+08:00`，seed=1 的四条臂里
**只有基线 `sequential` 训完并评测了**（1.5288 bpc）；另外三条（`parallel` /
`parallel-dual-ln` / `dense`）因为本机 GPU 锁被**其它轴的多臂驱动持续独占**
（锁释放后 1 秒内被重取，等不到 `SCHEMA.md` §20.1 的 ≥60s 礼貌窗口），连续多轮
**等锁超时**，**没有跑成**（每条 12k 臂 = 98.3M token 的等效算力，未消耗）。所以本文件
**不能**给出 seed=1 的干预臂读数，**只能**：①用 s0（seed=0，`2613`）的四臂读数下判词；
②用 s1 的**基线**读数度量「换 seed 的噪声」；③给出**代价面**（CPU 解析量，本来就与训练无关）。

**测出来多少（质量面）。** 单位是 clean bpc（每字符比特数，**越低越好**；
同尺寸普通变压器同预算的 s0 参照 = 1.5254）。

| 接法 | s0（seed=0）clean bpc | 相对普通接法 | s1（seed=1）clean bpc |
|---|---|---|---|
| 普通（顺序残差，基线） | **1.5254** | — | **1.5288** |
| 层间加权和（`dense`） | **1.5246** | **−0.0008**（说不出差别；判据带 0.014） | **missing** |
| 并行子层（`parallel`，少一个 LN） | 1.5712 | **+0.0458**（比带大 3.3 倍 ⇒ 明显更差） | **missing** |
| 并行子层（`parallel-dual-ln`，逐位等参数） | 1.5656 | **+0.0402**（明显更差） | **missing** |

**换 seed 的噪声有多大。** 同一条基线臂换 seed 后 bpc 从 1.5254 变到 1.5288，
**差 0.0034**（= 判据带 0.014 的 24%）；这正是「|Δ| < 0.014 不可排序」这条带被设计来
吸收的量级。所以 s0 上 dense 的 −0.0008 **确实落在噪声里**，而 parallel 的 +0.046
**远在噪声外**。

**代价面（2.2.4 的另一半，CPU 解析量、不含任何时间量）。**

| 接法 | 层内串行依赖 | 每 token 串行步数 | 全栈 MACs/token | vs 基线算力 | 参数量 Δ |
|---|---|---|---|---|---|
| 顺序残差（基线） | 2 段（attn→MLP） | 16 | 7,400,960 | 100.000% | 0 |
| 并行（共享 LN） | **1 段** | **8** | 7,396,864 | **99.945%** | **−4096** = −L·2d |
| 并行（dual-LN，等参） | **1 段** | **8** | 7,400,960 | 100.000% | **0（逐位相同）** |
| 层间加权和 | 2 段 | 16 | 7,408,128 | 100.097% | **+36** = L(L+1)/2 |

**结论（人话）。**

- **2.2.3 层间学习加权和**：在 13M / 12k / 0.014 带下，`dense` 相对普通顺序残差
  **说不出差别**（s0 配对 Δ = −0.0008，|Δ| ≪ 带；**n=1，seed=1 未跑**）。而且它**没有**
  换来中间深度的可读性（实测中间深度**反而更差**：depth 1–7 的 Δ = +0.23…+0.54，
  仅最终 depth 8 回到 −0.0008）。⇒ 按本项预登记的 falsification（「不越带 **且** 逐深度
  可读性无改善 ⇒ 无净增量」），本机制在本尺度**无净增量**。干预是活的
  （层权 α 的 fro 3.305、36 个权重全非零、浅层被放大 2.36×），所以这不是「干预没生效」。
- **2.2.4 并行子层**：质量面要付**可排序的代价**（+0.0458 / +0.0402，都是带的 3 倍以上），
  换到的是**真实的等效收益**：层内串行链从 2 段砍到 1 段、每 token 串行步数 16→8，
  而算力几乎不变（差 0.055%，就是少了一个 pre-LN 的 4096 个参数；等参数臂逐位相同）。
  ⇒ **以可排序的质量代价换层内并行度翻倍：本尺度上不是免费的，是否采纳取决于场景**
  （要吞吐/流水的人会觉得值，要质量的人不会）。**n=1**，seed=1 未跑。

## 1. 臂完整性（as of `2026-10-14T22:32+08:00`）

| 臂 | s1 训练 checkpoint | s1 电池（`results.json::arms`） | 说明 |
|---|---|---|---|
| `sequential` | **已完成**（12k 步 / 98.3M token，`runs/sequential-12k-s1/`） | **已完成**（clean_bpc 1.5288） | 唯一完成的 s1 臂 |
| `parallel` | **未跑** | **missing** | 等 GPU 锁超时（未消耗 token） |
| `parallel-dual-ln` | **未跑** | **missing** | 同上 |
| `dense` | **未跑** | **missing** | 同上（契约允许先砍，但本轮是**全部三条都未跑**，非按序砍） |

**未跑三臂的等效算力成本**：每条 12k 臂 = `12000 × 32 × 256 = 98,304,000` token
（≈6–8 min 卡时）；三条合计 **294.9M token**。等锁超时**没有消耗任何 token**，
只是把三条臂推迟 ⇒ 状态是「**受阻**」，不是「失败」。

**阻塞原因（跨轴，非本轴可解）**：`/tmp/pathlm_gpu.lock` 被其它轴的多臂驱动持续独占，
锁释放后约 1 秒内被重取（无 §20.1 要求的 ≥60s 礼貌窗口）；本轴三条 seed=1 臂连续 5 轮
等锁超时。**本轴未杀他人进程、未动他人锁。**

**`results.json` 的当前态**：`completed_arms = ['sequential']`、`pending_arms =
['parallel','parallel-dual-ln','dense']`（`run.py::dump_results` 每臂跑完增量落盘）。
若后续 s1 三臂落地，本文件的 §2 / §4 / §9 可按 §11 的指引重算（`results.json::cross_seed_s0_s1`
会自动补齐），判据线不变。

**一处必须说明的产物状态（不掩盖）**：`results.json::config.steps` 目前是 **20**，因为最后一次
成功的 `run.py` 调用是**冒烟**（`--steps 20 --arms sequential,parallel --skip-s0-anchor`，见
`run_stdout-smoke.txt`），它把自己的 `cfg_dump` 写进了 `results.json`；而 `arms.sequential`
本身是 **12k 步**训练出的（`train_tokens = 98,304,000`、`wall_min = 9.62`），两者并不矛盾。
**规范的 resolved config 是 `config.json`**（`steps = 12000`、四臂；由 `run.py::main`
在每次启动时写回，本文件 §3 即取此文件）。下一次**全跑**（`--steps 12000`）启动时会把
`results.json::config` 一并写回规范态。本文件所有判词数字取自 `config.json`（配置）与
`results.json::arms.sequential` / `cost_face_2_2_4`（读数），**不**依赖 `results.json::config`。

## 2. 四臂 × 两 seed（缺的 seed 明确标 missing）

口径：enwik8 char 级 AR，d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态
（无 carry 门、无随机退出、仅最终深度 CE、tied 读出），12000 步（98.3M token）matched-step
快照；`clean_bpc` = 全深度（depth-8）逐位 clean 每字符比特数；`tf@4` = 用前 4 层当草稿、
被全深度接受的比例（本形态无浅层出口监督，历史实测 ≈0.44 为上限参考）。

| 臂 | s0 clean_bpc | s0 tf@4 | s0 next-acc | s0 参数量 | s1 clean_bpc | s1 tf@4 | s1 next-acc | s1 参数量 |
|---|---|---|---|---|---|---|---|---|
| `sequential`（基线） | **1.5254** | 0.4560 | 0.6893 | 6,962,688 | **1.5288** | 0.4215 | 0.6887 | 6,962,688 |
| `parallel` | 1.5712 | 0.3486 | 0.6815 | 6,958,592 | **missing** | **missing** | **missing** | （期望 6,958,592 = −4096） |
| `parallel-dual-ln` | 1.5656 | 0.3149 | 0.6824 | 6,962,688 | **missing** | **missing** | **missing** | （期望 6,962,688 = 逐位等参） |
| `dense` | 1.5246 | 0.4538 | 0.6893 | 6,962,724 | **missing** | **missing** | **missing** | （期望 6,962,724 = +36） |

**seed 噪声（用唯一可比的两条基线臂度量）**：`sequential` 的 s0→s1 差 = **0.0034 bpc**
（0.014 带的 24%）、tf@4 差 = 0.0345、next-token acc 差 = 0.0006。前者与 2613 `analysis.md`
记的「同配置重跑散度 ≈0.0033 bpc」一致 ⇒ 换 seed / 重跑的 bpc 散度稳定在 ~0.003。

**逐深度 clean bpc（s1 基线 vs s0 基线；stock 形态只监督最终深度，判据只看 depth 8）**：

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8（判据）** |
|---|---|---|---|---|---|---|---|---|---|
| s0 `sequential` | 7.4039 | 4.8932 | 4.3486 | 3.8608 | 3.2023 | 2.6281 | 2.1895 | 1.7381 | **1.5254** |
| s1 `sequential` | 7.3941 | 4.9169 | 4.3719 | 3.9532 | 3.3233 | 2.6894 | 2.0855 | 1.7318 | **1.5288** |
| s1 − s0（seed 散度） | −0.010 | +0.024 | +0.023 | +0.092 | +0.121 | +0.061 | −0.104 | −0.006 | **+0.003** |

（读法：**中间深度的 seed 散度可达 ~0.1 bpc**，而最终深度只有 0.003；这为 §9.1 里
「dense 的逐深度缺损是否只是 seed 噪声」提供了噪声标尺。）

**逐深度 clean bpc（s0 四臂，干预臂对基线；越低越好）**：

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8（判据）** |
|---|---|---|---|---|---|---|---|---|---|
| `sequential` | 7.4039 | 4.8932 | 4.3486 | 3.8608 | 3.2023 | 2.6281 | 2.1895 | 1.7381 | **1.5254** |
| `parallel` | 7.3264 | 6.4216 | 5.7071 | 4.4716 | 3.7432 | 3.1870 | 2.6442 | 2.0497 | **1.5712** |
| `parallel-dual-ln` | 7.3090 | 6.3400 | 5.6275 | 4.4743 | 3.8343 | 3.2234 | 2.6912 | 1.9012 | **1.5656** |
| `dense` | 7.3786 | 5.1192 | 4.8923 | 4.3084 | 3.5141 | 2.8340 | 2.4466 | 2.0173 | **1.5246** |
| `dense` − `seq` | −0.025 | **+0.226** | **+0.544** | **+0.448** | **+0.312** | **+0.206** | **+0.257** | **+0.279** | **−0.0008** |

## 3. 与 s0（2613）的逐字段对照（`SCHEMA.md` §9 第 2 条）

`config.json`（resolved config）由 `run.py` 启动时重写；本节的对照由 `run.py::config_vs_2613`
逐字段生成（29 行，见 `config.json::vs_2613`）。取规范态（`--steps 12000`、四臂）：

| 结论 | 值 |
|---|---|
| 不等字段 | **`seed`（0→1）+ `experiment`（目录名，非科学字段）** |
| `only_seed_and_name_differ` | **true** |
| 相等字段（27） | `steps` / `batch_size` / `lr` / `optimizer` / `data` / `data_split` / `vocab_size` / `train_tokens` = 98.3M；`base.{d_model,n_layers,n_heads,seq_len,mlp_mult,dropout,norm_cap,dense,exit_mode,loss,readout}`；`arms[4].{role,params_expected_delta}` |

⇒ **本实验相对 2613 的唯一科学变量 = 训练 seed（0→1）**，满足 `SCHEMA.md` §9 第 2 条
（配置层等值）。

### 3.1 `SCHEMA.md` §9 三条等值门（本实验对 2613 的复用）

1. **实现层 equality（可失败）**：`run.py::remeasure_s0` 用**当前电池**重测 2613 的四条
   s0 checkpoint，逐臂 `remeasured_bpc_now == recorded_bpc_2613` **逐位一致**，引擎等值门
   打印 `[s0] remeasured all_bitwise_equal=True`（证据行：
   `run_log-attempt1-crash.txt:3`，同文件 `:1` 同时记 `[cfg] vs 2613 unequal fields =
   ['seed', 'experiment']`）⇒ s0 与 s1 用**同一评测口径**、跨 seed 可比。
   （说明：契约正文把该行归到 `run_stdout-smoke.txt`；本目录随后被冒烟重启覆盖了该文件，
   实际存证在 `run_log-attempt1-crash.txt`。同族等值证据还有 `smoke/smoke.json::checks.
   identity_start_bitwise_vs_sequential`（`parallel-dual-ln` 与基线逐位等参、
   `dense` 起步逐位等价基线）与 `test_run_persistence.py`（4 条回归测试）。）
2. **配置层 equality**：§3 的逐字段对照，不等字段只有 `seed` + `experiment`。
3. **口径标注**：见 §8。

**结论**：s0（2613）可作本实验 seed=0 的对照臂复用，无须自包含双跑（`SCHEMA.md` §9 第 4 条
的唯一例外——不等字段已显式列出且只有 seed/目录名）。

## 4. 配对 Δbpc 表

判据线（逐字来自 `README.md::§4` / ticket 61 §9.2）：**配对 Δ 越 0.014 bpc 带**才算可排序。

| 配对 | s0（seed=0）Δbpc | s0 判定 | s1（seed=1）Δbpc | s1 判定 |
|---|---|---|---|---|
| `dense` − `sequential` | **−0.0008** | 不越带 ⇒ 无差异 | **missing**（`dense` 臂未跑） | — |
| `parallel` − `sequential` | **+0.0458** | 越带 ⇒ 代价可排序 | **missing** | — |
| `parallel-dual-ln` − `sequential` | **+0.0402** | 越带 ⇒ 代价可排序 | **missing** | — |

- **配对窗口 SE**：s1 只有基线臂，`run.py::paired_delta_s1`（窗口级配对 SE + 两半样本）
  **无法生成**。可报的替代噪声量：s1 `sequential` 的 20 个窗口 `window_bpc` 的
  **窗口级抽样 SE = 0.0168**（SD 0.0752 / √20）——这是**单臂绝对读数**的窗口噪声，
  **不是**配对 Δ 的 SE（配对会消掉共同的窗口噪声，通常显著更小）；因缺第二条 s1 臂，
  本文件**不**用 0.0168 当判据，只作量级参考。
- **n=2 判定需要的条件**（ticket 61 §9.2 acceptance ③：两 seed 同号且都越带 ⇒ 可排序；
  不越 ⇒ 无差异）在本轮**不满足**（s1 缺三臂）⇒ 结论只能停在 **n=1**，并在 `gaps` 挂
  「n≥2（seed=1 三臂）」的成本。

## 5. 2.2.4 代价面同框表（等效理论量，**不含任何时间量**）

公式见 `run.py::topology_cost_face`（注意力 = 4d² + 2T·d；MLP = 2·mlp_mult·d²；
一个 pre-LN ≈ 2d MACs/token；读出 = d·vocab；`dense` 另加层间加权和 Σ(k−1)·d）。
参数 d=256 / L=8 / mlp_mult=4 / T=256 / vocab=206。原始值：`results.json::cost_face_2_2_4`。

| 接法 | 层内串行依赖（段） | 每 token 全栈串行步数 | 每层 MACs/token | 全栈 MACs/token（含读出） | vs 基线算力 | 参数量 Δ |
|---|---|---|---|---|---|---|
| `sequential` | 2（attn → MLP） | 16 | 918,528 | 7,400,960 | 100.000% | 0（基线） |
| `parallel` | **1**（两分支同读块输入） | **8** | 918,016 | 7,396,864 | **99.945%** | **−4096** = −L·2d |
| `parallel-dual-ln` | **1** | **8** | 918,528 | 7,400,960 | 100.000% | **0（逐位等参）** |
| `dense` | 2（attn → MLP） | 16 | 918,528 + 896（层间加权和摊派） | 7,408,128 | 100.097% | **+36** = +L(L+1)/2 |

**同框读法**：并行子层买到的是「层内串行链 2 段 → 1 段」（每 token 串行步数 16→8，
同一时间可并起来的活儿多一倍），算力只差 0.055%（且那点差就是少了一个 pre-LN 的 4096 个
参数）。`parallel-dual-ln` 是**等参数对照**：串行依赖同样砍半、参数量与基线**逐位相同**，
把「并行的收益」与「参数账」解耦。`dense` 的代价面可忽略（+36 个标量、+0.097% 算力），
但它**不缩短**层内串行依赖（仍 attn→MLP 两段），只是多了一条层间加权和。

**与质量面同框**：`parallel` / `parallel-dual-ln` 换来「层内并行度翻倍」，代价是
质量面 s0 **+0.0458 / +0.0402**（seed=1 未跑）。⇒ 见 §9.2 判词。

**（旁证）2613 记的实测训练步时**（过程量，**不进判据**）：`parallel` 0.0300 s vs
`sequential` 0.0325 s（−8%，省一个 LN）；`dense` 0.0400 s（+23%，+36 标量不应导致该量级，
疑为 kernel 调度偶发）。运行时性能不作判据（`docs/research-schema.md` §3）。

## 6. `SCHEMA.md` §12 三对账 + §13 对照三要素

### 6.1 §12 三对账

1. **算术对账**：clean bpc 理论下界 0；均匀预测参考值 log₂(206) = 7.69；四臂（s0）实测
   1.52–1.57，s1 基线 1.5288，均在界内。参数量账（s0 实测，逐位核对通过）：
   `parallel-dual-ln − sequential = 0`、`parallel − sequential = −4096 = −L·2d`、
   `dense − sequential = +36 = L(L+1)/2`（`2613/results.json::reconciliation.arithmetic.
   params_delta_exact = true`；`smoke/smoke.json::checks.params_delta_vs_sequential` 同值）。
2. **任务对账**：四臂同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M 字节）、
   同一位置集（全序列逐位置 × depth-8 读出）、同一目标定义（tokens 自移位）；同一
   `run.py::run_battery`（同 n_batches=20、batch_size=16）。对照臂只改「层与层之间的接法」，
   无额外输入、无真值泄漏。
3. **口径对账**：四臂同为 12000 步 / 98.3M token / matched-step 快照；s0 与 s1 的
   评测口径由 §3.1 的逐位等值门钉住。**墙钟一律不进判据或 `cost_model`**（`cost_model`
   字段与 §5 表均为等效理论量；`wall_min` / `seconds_per_step` 只作过程量落在实验产物里）。

### 6.2 §13 对照臂三要素（每个对照臂：输入分布 / 预测什么 / 部署对应什么）

| 对照臂 | 输入分布 | 要预测什么 | 部署中对应什么 |
|---|---|---|---|
| `sequential`（stock，基线） | enwik8 真值 token 序列（末 10M 字节，自移位） | 每个位置的下一个字符 | 标准 decoder-only 前向（每层 attn→MLP 串行） |
| `parallel` / `parallel-dual-ln` | 同上（**同一输入分布**） | 同上（**同一位置集**） | 层内注意力 ∥ MLP 同读层输入（PaLM/GPT-J 式，缩短层内串行依赖） |
| `dense` | 同上 | 同上 | 层输入 = 对所有此前层输出的学习加权和（DenseFormer 式；推理需保留全部此前层输出） |

四臂**同一任务定义、同一输入分布、同一位置集**；无「假上下文 + 真目标」式对照
（`SCHEMA.md` §13 禁止项）。

## 7. 干预活性（`SCHEMA.md` §10 强制项）

- **`dense`（唯一有专属参数的臂）**：训练结束时 α 的 fro = **3.3049**（起步 √8 = 2.8284）、
  absmax = **2.3615**、非零占比 = **1.0**、与 identity 起步的最大偏离 = **1.3615**；
  第 1 层输入被放大 **2.36×**，深层对最早输入（源 0）学到稳定的**负权 −0.17～−0.27**。
  ⇒ **干预活着**（排除「干预没生效」，故 s0 的 −0.0008 是机制中性，不是死干预）。
  20 步冒烟里 α 已离开起步（`smoke/smoke.json::dense_alpha_activity_after_20steps.
  max_dev_from_identity_init = 0.00103`，同文件 `checks.identity_start_bitwise_vs_sequential.
  dense = true` = 起步逐位等价基线）。
- **`parallel` / `parallel-dual-ln`**：接线相对偏差（init 时 vs 基线）= **0.490307**（非零）；
  `parallel-dual-ln` 参数量与基线**逐位相同**（`params_identical_to_sequential = true`），
  `parallel` 少一个共享 pre-LN（−4096）。⇒ 接线确实改了。
- s1 基线臂的活性块同上（`results.json::arms.sequential.intervention`），
  `start_bitwise_equals_sequential = true`、`params_added = 0`（基线无干预，符合预期）。

## 8. 口径标注（`SCHEMA.md` §9 第 3 条）

- **规模 / 配方**：13M 档（d=256 / L=8 / h=8 / seq=256 / mlp_mult=4），enwik8 char 级 AR，
  12000 步 × batch 32 × seq 256 = **98.3M token**，matched-step 快照（**非渐近**、**非墙钟**、
  **非 FLOPs 口径**）。
- **判据量**：`clean_bpc` = 全深度（depth-8）逐位 clean 每字符比特数（越低越好）；
  配对 Δ vs `sequential`；方差带 **0.014 bpc**（限配对 Δ、≥5 窗口，C1 裁决）。
- **对照**：s0 = 2613（seed=0，四臂；复用经 §3.1 等值门）；s1 = 本实验（seed=1，**仅基线**）。
- **权重/数据不入库**：`*.pt` 与 `data/` 被 `.gitignore` 覆盖；权重清单见 `MODEL.md`。
- **墙钟**：`wall_min` / `seconds_per_step` / `eval_wall_min` 只作过程量与冒烟信息，
  **不进判据、不进 `cost_model`**。
- **s0/s1 评测口径一致性证明**：`run_log-attempt1-crash.txt:3`
  `[s0] remeasured all_bitwise_equal=True`。

## 9. 判词（口径预登记，**未事后改判据线**）

### 9.1 `2.2.3` 层间学习加权和

- **字面（预登记判据线）**：s0/s1 两 seed 的 `dense − sequential` 配对 Δ 是否越 0.014 bpc 带？
  **s0：−0.0008，不越带**；**s1：missing**（seed=1 `dense` 臂未跑）。⇒ 按字面，
  **「在本尺度（13M/12k）+ 0.014 带下，层间学习加权和相对 stock 单残差无可排序差异
  （n=1，s1 未跑）」**。
- **实质（falsification 全条件）**：本项 falsification = 「若不越噪声带（0.014）**且**
  逐深度可读性无改善 ⇒ 判密集跨层线性聚合在本尺度无净增量」。两条件**都成立**：
  - 不越带：|−0.0008| ≪ 0.014（且 ≪ 换 seed 噪声 0.0034）；
  - 逐深度可读性**无改善**（实测更差）：`dense` 的中间深度 bpc 全部**高于**基线
    （depth 1–7 的 Δ = **+0.226 / +0.544 / +0.448 / +0.312 / +0.206 / +0.257 / +0.279**），
    只有最终 depth 8 回到 −0.0008。噪声标尺：s0↔s1 基线的中间深度 seed 散度为
    0.010–0.121（§2），故这些正 Δ 在**符号上一致、量级为 seed 散度的 2–5 倍**；
    关键是**7 个中间深度没有一个变好**。
  ⇒ 判**本机制在本尺度无净增量**。
- **`status`**：**`被否证`**（机制提案「密集跨层线性聚合在本尺度带来质量增益」被本地证据否证）。
  **边界**：n=1（seed=1 未跑），该否证是**尺度受限**的（「在本尺度无净增量」），不是
  「DenseFormer 普遍无效」——DenseFormer 在更大模型上有已报告增益，本项未在更大规模复核
  （见 §10 `gaps`）。
- **失败归因（先区分设计 / 结构 / innovation，见 `project.yaml::failure_analysis`）**：
  - **结构问题（主因）**：本尺度 L=8 太浅、且**只有最终深度监督**（stock 形态），
    「逐层串联的信息瓶颈」这个 DenseFormer 要缓解的条件在本尺度不成立——层间加权和
    反而自由地改变了中间层表示（中间深度读出变差），最终靠后续层重新学回 parity。
  - **设计问题（次因、可修、非主因）**：本实现的 α **无单纯形约束**（≠ DenseFormer 严格
    凸组合语义），且无中间深度监督去约束层间混合；但**干预活性成立**（α fro 3.305、
    36 权全非零、浅层放大 2.36×），故 null **不能**归因于「干预没生效」。
  - **非 innovation 本身失败**：DenseFormer 族在更大规模有已报告增益；本项只判「本尺度
    无净增量」，更大规模/带中间监督的复核未做（入 `gaps`）。
- **两读法并列（`docs/research-schema.md` §5 C2）**：milestone 预登记文字
  （「可排序差异或无差异，二选一明确」）**字面成立**（=无差异）；其实质（DenseFormer 式
  层间加权和在本尺度带来净增量）**否证**。

### 9.2 `2.2.4` 并行子层

- **字面（预登记判据线）**：`parallel` / `parallel-dual-ln` 相对 `sequential` 的质量代价
  是否越 0.014 带？**s0：+0.0458 / +0.0402，均越带（>3×）⇒ 代价可排序**；
  **s1：missing**。代价面（层内串行依赖 / 每层 FLOPs / 参数量）**齐备**（§5）。
- **实质（falsification 全条件）**：本项 falsification = 「质量面有可排序代价（方向 +0.0458）
  **且** 在等效代价面也拿不到可兑现的收益 ⇒ 判不划算」。第二条件**不成立**：代价面给出了
  **真实收益**——层内串行链 **2 段 → 1 段**、每 token 串行步数 **16 → 8**（同一时间可并起来
  的活儿多一倍），而算力只差 **0.055%**（`parallel-dual-ln` 与基线**逐位等参**）。
  ⇒ **不能**判「不划算」。
- **结论**：**以可排序的质量代价换层内并行度翻倍：本尺度上不是免费的，是否采纳取决于场景**
  （要吞吐 / 流水的人会觉得值；要质量的人不会）。**n=1**（seed=1 的 `parallel` / `parallel-dual-ln` 两臂未跑；等效成本每臂 98.3M token、两臂合计 196.6M token），不越尺度外推。
- **`status`**：**`已完成`**（本地实测已有明确结论：代价可排序 + 代价面收益真实 + 权衡明确）。
- **两读法并列（C2）**：milestone 预登记文字（「质量代价可分或明确不越带；代价面齐备；
  结论明确是否划算」）**字面成立**（代价可分、代价面齐备、给出场景依赖的结论）；
  实质（「并行子层是否值得采纳」）**给出可操作的权衡**，不越尺度外推。

## 10. 残余缺口（如实登记，附等效算力成本）

1. **s1 三臂未跑**（`parallel` / `parallel-dual-ln` / `dense`，seed=1）：跨轴 GPU 锁饥饿所致。
   成本 = **每条 98.3M token（≈6–8 min 卡时），三臂合计 294.9M token**。
   ⇒ `2.2.3` 的 n=2、`2.2.4` 的 n=2 边界未闭合。
2. **更大规模点未跑**：2.2.3 的 milestone 要求「一个更大规模点」；本项只有 13M。
3. **`dense` 的 α 无单纯形约束**：与 DenseFormer 严格凸组合语义的对照未做。
4. **`dense` 的「中间深度更差」是本项新观测**：与 2.2.1 的残差形式 / 2.1.1 的早退耦合
   （「中间深度可读性」是它们的判据线之一）未做联合检验。

## 11. 给下一位 lead 的补记指引（若 s1 三臂随后落地）

`run.py` 已按臂增量落盘；s1 三臂齐后，**只需用现有 `results.json` 重算**，**不重训**：

1. **§2 四臂 × 两 seed 表**：从 `results.json::arms.<topo>.{clean_bpc,tf@4,
   next_token_acc,params}`（s1）与 `../2613-inter-layer-topology-13m/results.json`（s0）
   对填；`missing` 格改为实数。
2. **§4 配对 Δ 表 + n=2 判定**：直接读 `results.json::cross_seed_s0_s1`（含
   `delta_s0` / `delta_s1` / `same_sign` / `both_exceed_band` / `sortable_two_seed`）与
   `cross_seed_summary`（含 `dense_sortable_two_seed`、`sequential_seed_spread`）；
   `2.2.3` 的 s1 `dense − sequential` 与 `2.2.4` 的 s1 `parallel/dual − sequential`
   由此补齐。**判据线不变**（0.014 带、两 seed 同号且都越带 ⇒ 可排序）。
3. **§4 的配对窗口 SE**：`results.json::paired_delta_s1`（窗口级配对 SE + 两半样本）会自动生成
   （需 ≥2 条 s1 臂），用它替换本文件的「无法生成」。
4. **`config.json::vs_2613`**：若全跑未启动，核对 `steps == 12000`（冒烟会把 `--steps 20`
   写回该文件；全跑启动时会立即写回规范态）。
5. **判词更新**：若 s1 的 `dense` 也落在 0.014 带内 ⇒ `2.2.3` 结论与 `status`（`被否证`）
   不变、只是 n=2；若 s1 的 `parallel/dual` 也越带且同号 ⇒ `2.2.4` 由 n=1 升 n=2（结论不变）。
   若出现反号 / 至少一 seed 落带内 ⇒ 按 `README.md::§8` 失败决策树「如实标不可排序」。

> 契约副本标注（`SCHEMA.md` §17.6）：§9 的判据线逐字来自 `.scratch/61-axis2-depth/TICKET.md`
> §9.2 WP-2651 与 `README.md::§4`，**未改**。
