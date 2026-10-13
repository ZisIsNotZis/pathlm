---
name: 2622-longT-512
ticket: 39-X3
date: "2026-10-02"
grid: A6-mixing-operator（长 T 首探：T=512）
engine: depth-ar `--dense`（stock decoder-only transformer，6.96M / T=512 时 7.03M）
budget: "gpu_wall_budget: 45 min（含冒烟 + 成本基准 + 4 训练臂 + 电池 + 合成腿）；峰值显存 ≤5GB net"
status: judged
---

# 2622 · 长 T 首探：T=512（A6 混合算子的长上下文端）· 6.96M · depth-ar/dense

> 本文件即本实验的本地契约。§0 把 `.scratch/39-close-gaps/issues/01-close-gaps.md` 的
> 「X3」段与裁决（含 SCHEMA §17.16）**逐字/摘要**
> 抄下；§1 起是实现与执行细节。判据线一字未改。
> 原始结果：`results.json`；判读：`analysis.md`；权重清单：`MODEL.md`。

## 0. 契约原文与裁决

**【X3 原文（逐字）】**

> 【X3 · GPU】长 T 首探：T=512（≈45–60 min）
> 目录：axes/1-time/1.1.1-hard-window/experiments/2622-longT-512/
> 动机：A6 的 T=256 净亏结论与注意力占比外推（14.3%@256→72.7%@4096）**从未在 T>256 实测**。
> 干预：seq_len 256→**512**（最小改动：扩绝对位置表，K10 估 +2.8% 参数），比较 full attention
> vs 窗口 256（同 token 预算、同步数），**必须双口径**（等效 FLOPs + 墙钟）。
> 判据：① 质量 Δbpc 与 T=256 同判据**同号/异号**（**两点只描述、禁止外推**）；② 注意力占比随
> T 的实测值（与 K10 公式对照）；③ 显存实测确认未越 5GB net；④ 若窗口在 T=512 首次「不亏」
> （Δ ≥ −0.014），记「交叉点首个观测点」。

**【主 Agent 裁决（摘要 + 逐条）】**——能力面分析发现「窗口 256 在 T=256 上是恒等
算子」⇒ 契约的单点退化（SCHEMA §17.16）：

1. **(A) 认下恒等事实**：`window_anchor_keep_mask(256,256,0)` ≡ 精确因果掩码、两臂参数逐位相等
   ⇒ `window256@T=256` 的 Δbpc ≡ 0，**不是真观测点**。用**可失败单测**钉住「该配置在此 T 下退化」。
2. **(B) 补真实同族第二点**：在 **T=256 上跑窗口 = 128**（与 T=512 的窗口 256 **同一相对刻度 50%**）。
   同族两点 = **(128@256)** 与 **(256@512)**，判据 = 两者相对各自 full 的 Δbpc **同号/异号**。
   **禁止**用 window256@T=256 或任何非等相对刻度配置充当第二点。
3. **(C) 禁外推声明**：两点只描述形状，**不得外推**（含「交叉点 T*」）。
4. **合成腿在 scope 内**：`var_recall_batch(seq_len=512, distance∈{128,255,300,400,480})`，**不改工具**；
   必须报 chance 与局部先验基线，并**显式声明**「距离档 >255 是本次新配置，与既往 255 上限的距离
   曲线**不可直接比较**」。
5. **成本口径必须诚实**：窗口臂走 SDPA 的稠密 bool 掩码（O(T²)）⇒ **墙钟收益只是理论值**，
   报告分开写「理论 FLOPs 节省」与「实测墙钟（无节省/可能更慢）」。
6. **注意力占比对照值**：按公式 `512T/(512T+786432)`，T=512 = **25.0%**
   （`512·512 = 262144`，`262144+786432 = 1048576`），**按 25.0% 对照**。T=512 的参数增量按
   **实测 +65,536 = +0.94%**。

源契约行（供回溯，逐字）：`archive/wiki-legacy/axes/A-mixing/PRIOR_ART_LENGTH.md`（K10）；
`SCHEMA.md §17.16`、`.scratch/39-close-gaps/issues/01-close-gaps.md`。

## 1. 命题（一句话、可证伪）

在 stock decoder-only transformer（`train_depth_ar.py --dense`）上，把训练序列长度从 **256 提到 512**
（唯一其他变化 = 位置表随 `seq_len` 线性增大），比较 **全因果注意力** vs **位置式窗口注意力**
（窗口 = T/2：`window128@T=256` 与 `window256@T=512`，同相对刻度），12k/6k 步 matched-token 快照下：

1. 窗口相对 full 的 **Δbpc 在两个 T 点上的符号**（同号/异号）；
2. **注意力子层算力占比**随 T 的**实测**值（与公式 `512T/(512T+786432)` 对照，T=512 = 25.0%）；
3. 训练**峰值显存**是否越过 5GB net；
4. `window256@T=512` 是否**首次**相对 full 不亏（`Δ ≥ −0.014`）。

**判据线逐字见 §4（预先写下，不得修改）。**

## 2. 单因素声明

- **唯一变量 = 训练期可见窗口**（`train_window` ∈ {0 关闭, T/2}），T 在两臂间相同；两个 T 点之间只
  改 `seq_len`（位置表随之线性变大，是 `seq_len` 的**必然后果**，不是第二个开关）。
- 基底是**主流标准 decoder-only transformer**（dense 形态：无 carry 门、无随机退出、仅最终深度 CE、
  tied 读出、`topology=sequential`）。三条 RAP 共享底座偏离（tied 读出 + 无 final LN、变深度填充
  语义、全深度重打分）在**四臂间完全相同**，不作变量。
- **实现说明（诚实记账）**：窗口掩码是**位置式**（不是内容选择式 top-k）——由
  `pathlm.depth_ar.window_anchor_keep_mask` 生成 `keep[i,j] = (j≤i) ∧ ((i−j) < w)`，训练侧
  (`DepthARModel.forward`) 与评测侧 (`retention.masked_logits`) **同一函数**（单测钉住）。
- **非必要性例外**：无新增开关、未改 `pathlm/` 与 `train_depth_ar.py`（见 §9 文件所有权）。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | A6 的「T=256 净亏」结论在 T>256 是否仍成立；稀疏/窗口算子的**长 T 端质量与算力** | A6 的 T=256 下界 |
| 代价（质量） | Δ clean bpc（native 工作点：窗口臂 = 自身训练掩码；full 臂 = 全因果） | 各点自己的 full |
| 代价（质量·并列） | release 口径 Δ（窗口臂在全可见下评测） | 同上 |
| 有效性 | Δ + 方差界（Δ < 0.014 不可排序，`docs/methodology.md §2`） | SCHEMA §12 判据线 |
| **收益 1（等效 FLOPs）** | 理论：窗口保留的因果对占比（`visible_frac`）、注意力子层 MACs 下降、整层 MACs 下降 | full 注意力 |
| **收益 2（墙钟）** | 实测：整模型步时比 `r_time = ms(窗口)/ms(full)`、峰值显存比（**如实报可能 >1**） | full 注意力（同机同时刻） |
| **算力占比（判据②）** | 实测：注意力孤岛 fwd+bwd 步时 / 整模型步时，随 T；与公式 `512T/(512T+786432)` 对照 | 公式外推值（对照，非判据） |
| **显存（判据③）** | 训练峰值（`torch.cuda.max_memory_allocated` + `nvidia-smi` 子进程峰值） | 5GB net 硬预算 |
| 可控性 | 窗口旋钮是否可预测：`visible_frac` 实测 == 理论闭式（单测） | 理论式 |
| 副产品 | T=512 相对 T=256 的**位置表参数增量**（实测 +65,536 = +0.94%） | K10 表 |
| **干预活性（必报）** | `window_mask_liveness`（每 200 步随日志）+ `window_mask_probe`（掩码开/关 logit 差 > 0） | 0（死干预） |
| **稳定性（必报）** | gnorm max/中位/尖峰、NaN、loss 突跳 | 各臂日志 |
| **长程腿（§19）** | 合成长程任务 `var_recall`（距离档 × 臂）的 top-1 准确率 + chance + 局部先验 | 同任务 chance / 局部先验 |

**判读规则**：先过「实现/算子」门（单测 + 冒烟 + 变异验证 + 活性量），再判质量。

## 4. 判据线（预先写下，逐字来自契约 X3 + 裁决，不得修改）

- **① 质量（两点，只描述、禁外推）**：`Δbpc(T) = bpc_full(T) − bpc_window(T)`（SCHEMA §17.3：正 =
  窗口更好）。两点 = `T=256·window128` 与 `T=512·window256`。判读：两者符号**同号**（同一方向）
  或**异号**；|Δ| < 0.014 ⇒ 该点「不亏」（同质量带）。**只描述两点，不作任何外推/内插。**
- **② 注意力占比**：报 `T ∈ {256,512}` 的注意力孤岛/整模型**实测**步时比，与公式
  `attn_share(T) = 512T/(512T+786432)` 对照（T=256 = 14.3%，**T=512 = 25.0%**，后者来自
  「K10 公式」）。**公式值只是对照，不是判据。**
- **③ 显存**：报每臂训练峰值（进程 + `nvidia-smi`）。**超过 5GB net ⇒ 如实记「越界」**（本实验
  为 T=512 首探，若越界则需降批，并显式记账「降批与长度形成混杂」）。
- **④ 交叉点首观测点**：若 `window256@T=512` 相对 full **不亏**（`Δ ≥ −0.014`），记「T=512 为
  窗口算子交叉点出现的首个观测点」；否则记「T=512 仍亏」。**单点结论，不外推。**

Δ 符号（SCHEMA §17.3，全篇统一）：`Δ = bpc_对照 − bpc_实验 = bpc_full − bpc_window`（正 = 窗口更好）。
方差界：Δ < 0.014 不可排序（`docs/methodology.md §2`）。

## 5. 既有做法（prior art，SCHEMA §18）

> 本环境**联网不可用**（与 `archive/wiki-legacy/axes/A-mixing/PRIOR_ART_LENGTH.md §0`、`archive/wiki-legacy/axes/A-mixing/PRIOR_ART.md §0`
> 一致）。外部内容一律 **`记忆·未核`**，只作设计方向、**不得当判据**。

1. **谁做过**：滑动窗口注意力（Longformer / Mistral 的 SWA）是长上下文标准件；稀疏注意力
   （BigBird、Longformer）在长序列上以「子二次」注意力换取上下文长度。
2. **为什么这么设计**：注意力 FLOPs 随 T² 增长⇒在长 T 上按位置丢弃远端键可把注意力算力降到 O(T·w)，
   在长上下文吞吐上占优。
3. **它的实测**：公开报告多为**长上下文卖点**（吞吐/可及长度），**不报短上下文净亏的下界**，也不报
   6.96M/单因素受控下的交叉点。`记忆·未核`，不可当判据。
4. **我们的增量**：A6 已在 **T=256 实测**稀疏（内容 top-k）**净亏**（k=32 Δ=−0.0166，成本 2.2×），
   并只有**公式外推**的注意力占比（14.3%@256→72.7%@4096）。本实验**首次在 T>256 实测**窗口算子的
   质量与算力占比，并（按 §17.16）用**同相对刻度的两点**（128@256 / 256@512）作描述。

**检索词**（供将来可检索的 run 核证）：`sliding window attention long context quality`、
`sparse attention short context perplexity net loss`、`attention share of FLOPs by sequence length`。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **45 min**（含冒烟 + 成本基准 + 4 训练臂 + 电池 + 合成腿）；主 Agent 批准上限 60 min |
| **实测总墙钟（GPU 阶段）** | **≈33 min** = preflight ≈1.5 + 四臂训练 **27.41**（`results.json::wall_total_min`）+ eval ≈4；**未用满 45 min 预算**，砍项（合成腿距离档 → release 口径 → cost-iters）**未触发** |
| 预算过程（§8） | 开跑前用 `--stage preflight` **实测步时/显存**再锁定（禁按公式估）——实测值见 `results.json::cost` 与 `analysis.md` |
| 训练臂（4） | `full-256`(12k 步) / `win128-256`(12k) / `full-512`(6k) / `win256-512`(6k)，均 bs32、seed0 |
| **token 口径** | **同 token 预算 98.3M**（T=256: 12000×32×256；T=512: 6000×32×512）。**同步数**在 T=512 对内成立（两臂均 6000 步）；跨点用同 token 口径（K10 §3.3 主口径）。另报步数/epoch 数 |
| 最小可判定版本 | 4 臂匹配训练 + native/release bpc + 成本实测 + 合成腿 |
| 峰值显存 | ≤5GB net（共享卡；先测再定） |
| 可中断性 | 是（每 200 步落日志；`--skip-train` 可用已有 checkpoint 只跑电池） |
| 冒烟计划 | `--stage preflight`：四臂各 20 步（实测步时/显存/活性，**不含任何 20 步外推时间字段**，SCHEMA §17.5） |
| 预算不够时砍什么 | 砍项顺序：合成腿距离档 → release 口径 → `--cost-iters`；**不砍判据线、不砍第二点（win128）** |

规模纪律（SCHEMA §7）：单卡 RTX 4090 24GB 多任务共享，GPU 串行；只做 6.96M 原理验证，
**不外推**规模/渐近/交叉点。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，前 90M train / 末 10M eval / vocab 206） |
| 合成腿任务 | `tools/evalkit/longrange_tasks.py::var_recall_batch`（**未改工具**）；距离档 T=512: {128,255,300,400,480}、T=256: {64,128,192,255}；chance = 1/64；局部先验 k=8 |
| 入口脚本 | `run.py`（preflight / train / eval / report）；`smoke.py`（CPU 不变量）；`mutate.py`（单测变异） |
| 代码改动 | **无**——本实验不改 `pathlm/`、`train_depth_ar.py`、`tests/test_depth_ar.py`（X2 是本轮唯一共享代码写入者）。只**新增** `tests/test_longT_512.py` 与实验目录文件 |
| 可失败单测 | 见 `tests/test_longT_512.py`：恒等退化点、visible_frac 闭式、同相对刻度、聚合几何、可变距离合成任务合法性 |
| 变异验证 | `mutate.py`：**monkeypatch** 方式破坏掩码/占用比例 → 必须抓红 → 撤销（**不写共享文件**，见 §9） |
| 落盘 | `results.json` / `config.json` / `<arm>-train_log.jsonl` / `<arm>-gate_log.jsonl` / `analysis.md` / `MODEL.md` / `smoke/smoke.json` / `mutation_report.json` / `reconcile_report.json` / `run_log.txt` |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 单测红 / 变异不抓红 | ① 实现 | 先修实现；未排除前不判机制 |
| 干预死（`window_mask_probe` diff = 0 / `alive=False`） | ① 实现 | 活性门不过不判机制 |
| 两臂 `\|Δ\|` < 0.014（两点都） | ② 该 T 无头寸 | 如实报「窗口不亏」；交叉点记「首观测点」；**不加第二个机制** |
| 窗口劣化 ≥0.014（某点） | ④ 设计在该设定下净亏 | 记为否证/下界，记录窗口档与幅度 |
| 显存 >5GB net | ③ 硬件 | 降批重跑并**显式记账**「降批与长度混杂」；不擅自改判据 |
| 结果不可重复 | ⑤ 方差 | 按 Δ<0.014 不可排序解读；n=1 负结果复现后入账 |

## 9. 文件所有权与共享面声明（SCHEMA §22）

- 本实验**只写**本实验目录 + **新增** `tests/test_longT_512.py`。
- **不写** `pathlm/depth_ar.py`、`train_depth_ar.py`、`tests/test_depth_ar.py`（本轮 X2 是唯一
  共享代码写入者）。因此变异验证**不用文件改写**，改用进程内 monkeypatch（§7），且运行前后
  记录共享文件 sha256 断言未变。
- GPU 取锁遵守 §20/§20.1（多臂**按臂释放**，留 ≥60s 窗口）；峰值 ≤5GB net。

## 10. 产出

本目录下 `README.md`(本文件) / `run.py` / `smoke.py` / `mutate.py` / `config.json` /
`results.json` / `analysis.md` / `MODEL.md` / `reconcile_numbers.py` / `reconcile_report.json` /
`mutation_report.json` / `smoke/smoke.json` / `run_log.txt` / `<arm>-train_log.jsonl`；
`tests/test_longT_512.py`。

> 契约副本标注（SCHEMA §17.6）：§0 的 X3 原文段为**逐字**抄录；裁决段与 §1–§10 为执行者
> 实现/执行细节（裁决 6 条按逐条摘要记，未改判据线）。
