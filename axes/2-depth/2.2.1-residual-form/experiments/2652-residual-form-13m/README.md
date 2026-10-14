---
name: 2652-residual-form-13m
ticket: 61-axis2-depth（§9.2 WP-2652）
date: "2026-10-14"
grid: 2.2.1-residual-form
engine: depth-ar（13M 档 · dense 形态）
budget: "gpu_wall_budget: 45 min（含 CPU 等值门 + 20 步冒烟 + 4 臂 × 12k 步 + 全部评测）；峰值显存 ≤5GB net"
status: planned
---

# 2652 · 残差形式最小判决 · 13M · depth-ar/dense

> 本文件即本实验的本地契约。判据线 P1–P6 **预登记、不可改**（`SCHEMA.md` §8 / §17）。
> 原始结果：`results.json`（汇总）+ `results_<arm>_s0.json`（逐臂）；判读：`analysis.md`；
> 权重清单：`MODEL.md`；`analysis.md` 未写全前的任何数字都以 `results.json` 为准。

## 1. 命题（一句话，可证伪）

在**训练/数据/主干逐位相同**、且**只改残差形式**的前提下，把普通 decoder-only transformer 的
pre-LN 单残差换成（a）归一化后置（post-LN）、（b）固定深度缩放残差（`1/√(l+1)`）、
（c）可学习逐层标量门（ReZero 式，init=1.0），能否把 clean bpc 或**逐深度 bpc 曲线**
改善超过项目方差带（0.014）？

- **H1（备择）**：至少一种形式在 clean bpc 或中间深度可读性上相对 stock pre-LN 单残差
  **可排序**更优（按 0.014 带判）；
- **H0（零假设）**：所有形式相对 base **不可排序**（|Δbpc| < 0.014，且逐深度曲线不越带）
  ⇒ 在本尺度，**残差形式不是一条独立有增益的维度**（残差的「连接语法」自由度只有
  「有条残差」这一个有效量）。

本条直接对准轴 2 的核心问题：残差形式决定**中间层状态是否已经接近可用**（可截断性），
若各形式在 13M/12k 上连质量都不越带，则「靠换残差形式改善深度可截断性」这条路径
在本尺度没有头寸。

## 2. 单因素声明

**唯一变量 = 残差形式**（子层输出如何接回残差流）。基底 = 主流标准 decoder-only transformer
（`train_depth_ar.py --dense --topology sequential`：无 carry 门、无退出采样、仅最终深度 CE；
= stock 形态 + 共享底座三条必要偏离，见 `docs/design.md` §2）。

- d=256 / L=8 / h=8 / seq=256 / mlp_mult=4 / dropout=0 / norm_cap=1 / bs=32 / lr=1e-3 /
  12k 步 / seed=0 / 98.3M token / enwik8 前 90M train、末 10M eval / vocab 206 —— **四臂逐字段相同**；
- attention 算子（SDPA causal）、embedding、位置嵌入、LN 模块、FFN、tied 读出**逐位相同**；
- **非必要性例外：无**。

### 2.1 干预实现（为什么它不是「换了个模型」）

四臂都先用**同一个**（未改动的）引擎构造 `DepthARConfig(dense=True, topology="sequential")`
的完整模型 —— 即所有臂的 embedding / attention / LN / FFN 权重**来自同一次随机初始化抽样**
（同 seed、同构造顺序）。之后本实验**只重新接线 `model.blocks`**：
`residual_block.ResidualFormBlock` **接管** stock `Block` 的同一批子模块（同一 Parameter
对象），只改子层增量如何接回残差流。于是臂与臂之间**唯一**的差别是残差形式 ——
既不是「不同初始化的两个模型」，也不是「不同参数量的两个模型」。

- **`base` 臂 = 引擎 stock `Block` 原样**（不包裹）⇒ 它就是引擎 stock 对照；
- 残差形式实现在**实验目录内**（`residual_block.py`，工厂替换 `train_depth_ar.DepthARModel`），
  **不改 `pathlm/**`、不改 `train_depth_ar.py`**（模式沿用 `2641-ffn-ratio-frontier-13m/run.py`）。
  训练全程走共享 `train_depth_ar.main()`（参数解析 / 配置装配 / `train()` / checkpoint /
  `results.json` 都是共享实现）。

## 3. 臂定义（残差形式）

`A(x) = proj(SDPA(qkv(x)))` = 注意力子层输出，`M(x) = mlp(x)`，`x` = 进块的残差流。
`l` = 0-based 层号。

| 臂 | 公式 | 相对 base 参数 Δ |
|---|---|---|
| **`base`** | `x₁ = x + A(ln1 x)`；`x' = x₁ + M(ln2 x₁)`（stock pre-LN 单残差） | 0（基线） |
| **`postln`** | `x₁ = ln1( x + A(x) )`；`x' = ln2( x₁ + M(x₁) )`（**归一化后置**：残差相加之后再 LN，子层输入不再 pre-LN） | 0 |
| **`scaled`** | 同 base，但两个子层增量各乘**固定** `α_l = 1/√(l+1)` | 0 |
| **`gated`** | 同 base，但两个子层增量各乘**可学习**逐层标量门 `attn_gate` / `mlp_gate`（init = **1.0**） | **+2L = +16** |

- **`gated` 的等值起步**：门值恒 1.0、浮点乘 1.0 精确 ⇒ init 处 `gated` 前向与 `base`
  **逐位一致**（`torch.equal`，CPU fp32 与 CUDA bf16 autocast 两条路径都由 `smoke.py` 钉住）。
  门在乘之前 `to(x.dtype)`（bf16 autocast 下降成 bf16，1.0 可精确表示）——**不**让残差流被
  数值提升成 fp32，否则「只改残差形式」这条单因素声明就不成立。
- **`scaled` 的 `l=0` 层 `α=1.0`**（该层与 base 逐位一致），`l≥1` 层 `α<1` ⇒ 干预在
  `l≥1` 层生效（活性量逐层落盘）。
- **`postln` 的已知结构差异（写死在契约里，防事后解释）**：归一化后置 ⇒ 残差流在每子层后被
  LN 归一到 RMS≈1，而 base 的残差流是**未归一温流**；tied 读出（共享底座偏离：无 final LN）
  因此看到的输入量级不同。这是 post-LN 的**固有**性质（本实验要测的就是它），但它同时是
  「质量差可能来自读出输入量级的改变」这一替代解释的来源 —— 报 `stream_rms_by_layer`
  逐层曲线把它显式量化。

## 4. 判据线（**预登记，不可改**）

| # | 判据 | 该涨 / 该跌 / 必须不变 | 判定规则 |
|---|---|---|---|
| **P1（主判据）** | 各形式相对 base 的 clean bpc：`Δ_form = bpc_base − bpc_form`（**正 = 该形式更好**） | 该跌（各形式 bpc 相对 base） | `Δ ≥ +0.014` ⇒ 该形式**可排序更优**；`\|Δ\| < 0.014` ⇒ **不可排序（H0）**；`Δ ≤ −0.014` ⇒ 可排序**更差** |
| **P2** | **逐深度 bpc 曲线**（engine `depth_curve`，全深度前向的逐深度读出）：`Δ_by_depth[k] = bpc_base[k] − bpc_form[k]`，重点看**中间深度** k=1..7 | 该跌（中间深度可读性） | 中间深度（k=1..7）任一处 `\|Δ_by_depth\| ≥ 0.014` ⇒ 「中间深度可读性被改变」；否则报「深度形状未变」 |
| **P3** | 训练稳定性：末 25% 窗 loss 斜率（同一最小二乘、同一窗口）+ gnorm 末窗均值/全程 max | 该不变 | 出现非有限 loss 或 `gnorm > 1e3` ⇒ **如实报**崩步与 gnorm，不得隐藏；不因不稳定改判据 |
| **P4** | 干预活性（`SCHEMA.md` §10） | 必须报 | `gated`：逐层门终值 + 门控梯度非零（两条腿）；`scaled`：逐层 α 落盘；`postln`：逐层残差流 RMS 落盘；`base`：对照臂本身无干预 |
| **P5** | 引擎等值门 | 必须不变 | `base-s0` 的 clean bpc 与历史 dense 13M 锚（`.scratch/23-depth-ar/evidence/23d-dense-B-12k` = **1.5287**；2613 重训 baseline-s0 = **1.5257**）之差 < 0.005 ⇒ 本实验与既有 dense 主干同引擎、同配方 |
| **P6** | 等参数对照口径 + 单因素机械核对 | 必须不变 | ①`gated` 的额外参数 **= 2L = 16**（实测审计），占总参数 **0.00023% ≤ 0.1%** ⇒ **直接与 base 比**，**不另加等参数对照臂**（本契约选定此口径并保持一致）；②四臂共享 CLI 模板逐字相同（`results.json::single_factor_check`） |

- 质量口径 = **clean bpc**（dense 形态只有最终深度被监督 ⇒ `depth_curve()[-1]["bpc"]`，
  序列级、越低越好），eval 批固定 generator seed 2（臂间同一批）。
- 方差带 = **0.014**（`docs/methodology.md` §2：13M 跨 run bpc 漂移界；限配对 Δ、≥5 窗口）。
  本实验另报**同配置跨 seed 实测散布**（2613 的 baseline s0/s1 = 1.5257/1.5298 ⇒ 0.0041）
  作参照，但**不以它替代 0.014 预登记门槛**。
- **不得事后调判据**（`SCHEMA.md` §8）：预算不够时**砍臂**（先砍 `scaled`）、**不砍判据线**。

## 5. 对照与方差

- 自包含：**四臂本实验从头训练**，不复用任何历史 run 作手臂（历史只作 §4-P5 的等值性锚）。
- n = 1 seed（0）：本 WP 预算内四臂各一次；**跨 seed 复现留给后续**（写进 `analysis.md` 残余缺口）。
- 臂间不得比较不同 token 预算：四臂 `steps=12000, batch=32, seq=256` = **98.3M token**（enwik8
  前 90M 的 1.09 epoch）**matched-step / 同 token 预算**。
- 收敛对账（`SCHEMA.md` §16）：报每臂**同方法**（末 25% 同一窗口、同一最小二乘）训练 loss 斜率，
  防「某臂收敛更慢」被读成「质量更差」；快照增益只读臂间增量（`SCHEMA.md` §20.3）。

## 6. 规模、预算与显存

- 规模：13M 档（d=256 / L=8 / h=8 / T=256 / B=32）＝本仓全部 26xx 实验的同一档（实测总参数
  6,962,688；`base`/`postln`/`scaled` 逐位相同，`gated` +16）。
- **`gpu_wall_budget: 45 min`（唯一数字，含 CPU 等值门 + 20 步冒烟 + 4 臂 × 12k 步 + 全部评测）**；
  GPU 按 `SCHEMA.md` §20 互斥锁**按臂取/放**（臂间 ≥60 s 礼貌窗口）；峰值显存 ≤5GB net。
- 预算不够时砍什么（`SCHEMA.md` §8 / §9.2）：**先砍 `scaled`**（执行顺序 `base, postln, gated, scaled`
  把它排最后 ⇒ 直接不跑即符合「先砍 scaled」），并把「缩放残差未测」写进 `gaps`；
  **不砍判据线、不砍步数**（步数一改臂间刻度就不可比）。

## 7. 指标表（逐项分析，不合成单一分数）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 残差形式是否独立于「有条残差」的设计维度（质量 + 深度可读性） | base（stock pre-LN） |
| 代价 | 参数量（`gated` +16 = 2L；其余 0）；每 token 的 layer-passes 与 per-layer FLOPs 主项**不变**（残差形式不改计算图形状） | base / 臂间逐位 |
| 有效性 | clean bpc（越低越好，须越 0.014 才可排序） | base |
| 收益与用法 | 逐深度 bpc 曲线（中间深度可读性 → 可截断性代理）；`gated`/`scaled` 的逐层缩放量 | 臂间 |
| 可控性 | 门/缩放量相对 init 的偏离（`gated` 学到了什么） | 逐层落盘 |
| 副产品 | 逐层残差流 RMS（post-LN 的归一化效应、base 的温流形状）；末窗 loss 斜率与 gnorm | 臂间 |
| 活性 | `gated` 门终值 + 门控梯度；`scaled` 逐层 α；`postln` 逐层 RMS | 契约定义 |

## 8. 等值门与变异验证（`SCHEMA.md` §9 / §24 / `docs/methodology.md` §5）

`smoke.py`（CPU 等值门）+ `mutation_check.py`（可达性自证）+ `run.py --smoke-steps 20`：

- **G1 等值起步（等值门）**：同 seed 构造的 `base` 与 `gated`，**共用子模块参数逐位相同**，
  且同输入下 `logits` 与 `loss` **逐位相同**（`torch.equal`）；CPU fp32 与 CUDA bf16 autocast
  两条路径都测（后者是训练真正走的路径）。
- **G2 参数审计**：`gated` 相对 `base` 的增量**恰好 = 2L = 16**；`postln` / `scaled` = 0。
- **G3 非无操作**：`postln` / `scaled` 在 init 处与 `base` 的前向**必须不同**。
- **M1/M3/M4 变异（必须抓红）**：`mutation_check.py` 在同一进程里注入三处变异——
  M1 门初值改 0.99、M3 `postln` 偷偷退回 base 公式、M4 残差形式根本没接线——等值门**必须**
  相应失败；三处都不红 ⇒ 说明对应的门是恒真的摆设（`SCHEMA.md` §24）。
- **冒烟**：20 步 × 4 臂（一次持锁，整程 <10 min），确认四臂都能跑通并量实测步时。

## 9. 否证条件（做不出来时怎么区分）

- 若 `postln` / `scaled` / `gated` 相对 base 的 `Δ` **全部落在 0.014 带内**（P1），且 P2 的
  中间深度曲线也不越带 ⇒ **H0 成立**：判「残差形式在 13M/12k 不是独立有增益的维度」，
  `project.yaml.status = 被否证`；`failure_analysis` **必须先区分** ①**设计问题**
  （残差形式本身在此尺度无头寸）②**结构问题**（引擎/接线逼出的退化，如 post-LN 的读出
  输入量级改变、tied 读出无 final LN）③**innovation 本身失败**（「换残差形式换质量/可截断性」
  这条路线不成立）并给证据。
- 若 P5（等值门）失败 ⇒ 视为**实现/口径问题**，先修实现再判机制（不得据此判机制失败）。
- 若某臂发散/崩 ⇒ **如实报**崩步与 gnorm，并把该臂从机制判决里剔除（只报不判），
  另按失败决策树（测量 / 实现 / 规模 / 设计）逐层排除。
- 若只有 `gated` 越带而其余不越 ⇒ 先过 P6 的等参数口径（额外参数 ≤0.1% 已满足），
  再判「增益来自可学习门」；`gated` 的增益**不得**归因于参数量。

## 10. 既有做法（`SCHEMA.md` §18，先站巨人肩膀）

本环境检索不可用 ⇒ 以下均为 `记忆（未核）`，只作设计方向，**不作判据**：

- **post-LN vs pre-LN**：原始 Transformer（Vaswani 2017）是 post-LN；GPT-2 起 pre-LN 成为
  decoder-only 默认，公认更易训（无需精调 warmup）；DeepNorm（Wang 2022）用「残差放大 +
  定制初始化」把 post-LN 推深。**我们的增量**：在**同一初始化 + 同一 12k 预算 + 只改归一化
  位置**下给出受控单因素读数，并把「读出输入量级改变」这个混杂量化（`stream_rms_by_layer`）。
- **缩放残差**：ReZero（Bachlechner 2020）用可学习标量 α（init 0）做残差门；「Depth-µP /
  残差 1/√l 缩放」一族（Zhang/Shazeer 2024 前后）主张随深度衰减的残差以稳定大深度训练。
  **我们的增量**：在 13M/8 层这个**远不到「需要深层稳定化」**的尺度上测它有没有**质量**增益；
  若没有，那是对「缩放残差的收益随深度才出现」这条说法的**下界读数**（不外推）。
- **门控残差 / Highway**：Highway Networks（Srivastava 2015）、gated residual（ReZero 的
  逐通道变体）一族的共识是**层次越深越有用**。**我们的增量**：同尺度、受控、且给出
  「额外参数 2L=16 可忽略」的等参数口径，把「增益是否来自参数量」这件事排除。

**没有直接前例的部分**：在本仓自己的质量-代价前沿（13M / enwik8 / 0.014 带 / matched-step）
上，把这四种形式放在**同一次初始化 + 同一条共享训练代码**里做单因素对照，并**同时**读
clean bpc 与逐深度 bpc —— 这正是本实验的增量。

## 11. scope 与混杂声明（`SCHEMA.md` §4）

- 单轴单格：只改 `2.2.1 残差的形式`；与 `2.1.1`（按需深度）的耦合（截断可用性）本实验只读
  **逐深度 bpc** 这个代理，真截断由 2.1.1 的口径承担（见 `gaps`）；与 `8-engineering` 的分工：
  本项测残差形式的**质量/深度效果**，纯实现细节（eps、放置惯例）归 8。
- 组合默认留白：残差形式 × 其它干预（carry 门、退出填充、拓扑、稀疏注意力、窗口）**本实验
  全部未做**（`ResidualFormBlock` 对非 dense/无掩码路径直接报错，不静默降级）。
- 已知混杂/局限（写死在契约里，防事后解释）：
  1. `postln` 的残差流被 LN 归一 ⇒ tied 读出（无 final LN）输入量级与 base 不同（§3）；
  2. `scaled` 的 `l=0` 层与 base 逐位一致 ⇒ 它的「干预量」逐层递减（首层无干预）；
  3. 13M / 12k 快照：只读臂间增量，绝对水平是快照值（`SCHEMA.md` §20.3）；
  4. 单任务（enwik8 字符级 next-token）+ 单 seed：结论不外推到其他任务/规模，跨 seed 复现
     写进残余缺口；
  5. GPU 与其它 worker 共享（`SCHEMA.md` §20 锁排队）；**墙钟不进判据**，仅作过程量留痕。

## 12. 复现手段

```bash
# CPU 等值门（init 逐位一致）+ 可达性自证（三处变异必须抓红）
python3 smoke.py
python3 mutation_check.py
# 20 步冒烟（四臂，一次持锁）
python3 run.py --smoke-steps 20 --smoke-only
# 全臂（一条命令：等值门 → 冒烟 → 按臂取/放 GPU 锁 → 汇总）
python3 run.py 2>&1 | tee run_log.txt
# 单臂重跑 / 仅评测（复用 runs/<arm>-s0-12k/model.pt）
python3 run.py --arm postln --seed 0
python3 run.py --arm postln --seed 0 --no-train
# 仅汇总
python3 run.py --aggregate
```

- 权重**不入库**（`runs/**/model.pt`，`.gitignore` 覆盖）；`MODEL.md` 落哈希清单。
- 原始数字 = `results_<arm>_s<seed>.json`（逐臂）+ `results.json`（汇总判据）；
  训练日志 `runs/<arm>-s0-12k/train_log.jsonl`（每 200 步一行，含 loss / gnorm / depth_ce）。
- 入口在任何 CWD 下都可运行（`run.py` 启动时 `chdir` 到仓库根，共享 `train_depth_ar.main()`
  用仓库根相对路径读数据）。
