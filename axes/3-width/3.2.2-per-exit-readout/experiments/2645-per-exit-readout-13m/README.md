# 2645 · 逐出口读出（per-exit readout）· 13M · depth-AR

> 本文件是**预登记契约**（臂 / 判据线 P1–P4 / 否证线 / 带声明）：起跑后不改（口径差异
> 一律写进 `analysis.md` 声明）。研究项 = `axes/3-width/3.2.2-per-exit-readout`。

## 1. 人话摘要

**它是干什么的**：普通 transformer 只有一个出口（末层），一张单词表映射（tied 线性读出）
就够。早退 / 多出口模型每一步都要把「该深度的隐藏状态」映射到整张词表，于是出现一个
选择——**九个出口共用一张表，还是每个出口自己一张**？本实验就是判这一条。

**改了什么**：只改「出口 → 词表」的映射，主干（层数、宽度、注意力、FFN、门、退出采样）
逐位不动。四个形态：共用 tied 表（`T`）、共用表 + 每出口 d×d 变换（`PE`）、每出口独立表
（`IND`）、每出口低秩校正（`LR`）。

**为什么值得测**：共享 tied 表把所有出口绑在「概率 = 状态在 embedding 基的坐标」这条
约束上；浅出口质量（早退真正用到的那一层）可能因此被压低。代价一侧 `IND` 是词表 × 维度
（真词表 100k+ 时不可 scale），`PE`/`LR` 是每出口 d² 或 d·r（可 scale）。

**怎么测**：13M、d=256/L=8/变深度 + 随机退出 + `fill_kv`、12000 步、enwik8 字符级
（vocab=206）、同 seed；对照 = 同批重训的共享 tied 臂 `T`（若未跑则复用历史锚
`.scratch/23-depth-ar/evidence/23c-H2-w0-12k`，clean bpc 1.8582 / tf@4 0.9694 / tf@6 0.9938）。
主判据是**各出口 tf@k**（深度 1..7 的草稿被全深度出口接受的比例）+ 末层 clean bpc + 读出
参数量。

**测出来多少**：见 `analysis.md`（本次是否跑成，如实登记，不编读数）。

## 2. 基底与不变项（逐字段）

| 字段 | 值 | 来源 |
|---|---|---|
| 形态 | depth-AR（`dense=false`、变深度 + 逐层读出 + 随机退出 + `fill_kv=true`） | 派单 W-B2 契约 |
| d_model / layers / heads / seq_len / mlp_mult | 256 / 8 / 8 / 256 / 4 | 派单共同不变量 |
| exit_mode / aux_weight / final_weight / aux_anneal_to | `uniform` / 1.0 / 1.0 / None | 2610 契约 + 锚逐字段对照 |
| steps / batch_size / lr / seed | 12000 / 32 / 1e-3 / 0 | 派单共同不变量 |
| 数据 | `data/enwik8_full.npz`，字符级，vocab=206 | 同全轴 |
| 出口数 | 9（深度 0..8；深度 8 = 全深度，`tf_acceptance` 的验证者） | 引擎 `states` |
| 评测 | `depth_curve`（n_batches=20、batch_size=16、eval seed 2）取最终深度 clean bpc；`tf_acceptance`（同口径）；prob0 ECE（n_batches=8） | 2610 同款 |

**唯一变量** = 每个出口到词表的映射。所有臂的 embedding / attention / LN / FFN / 门权重
来自**同一次随机抽样**（同 seed、同构造顺序；每出口读出子模块建在 stock 构造之后 ⇒
不消耗主干随机数），由单测 `T3` 逐位钉住。

## 3. 臂（实验目录内实现，不改引擎）

| 臂 | 定义 | 每出口自有参数 | 9 出口合计 | tied 桥 |
|---|---|---|---|---|
| `T` | `logits_k = h_k·Eᵀ`（stock，共享 tied 表） | 0 | 0 | 有（= 100% 参照） |
| `PE` | `logits_k = (M_k h_k)·Eᵀ`，`M_k` [d,d]，`M_k = I` 起步 | d² = 65,536 | 589,824 | 有（`F_k = E·M_k`） |
| `IND` | `logits_k = tab_k(h_k)`，`tab_k` [V,d] ~ N(0,0.02) | V·d = 52,736 | 474,624 | **无** |
| `LR` | `F_k = E + A_k B_k`（`A_k` [V,r]=0、`B_k` [r,d]~N(0,0.02)，r=16） | V·r + r·d = 7,392 | 66,528 | 有 |

实现要点（全部写在 `run.py` 内，`pathlm/**` 不动）：

* `PerExitModel(DepthARModel)` 只覆写 `readout(h, depth)`；`T` 臂直接回落
  `super().readout(...)`（= stock 路径），单测 `T1` 钉住逐位相同。
* 起步恒等：`PE`（`M_k = I`）与 `LR`（`A_k = 0`）的起步读出**逐位等于** `T`
  （单测 `T1`），且非鞍点（`∂L/∂M_k ≠ 0`、`∂L/∂A_k ≠ 0`，单测 `T4`；`LR` 的
  另一侧 `∂L/∂B_k = 0` 是该结构的恒等式，一并钉住）。
* **表达力定理（本尺度）**：V=206 < d=256 ⇒ 对任意目标头 `Y ∈ R^{V×d}` 存在 `M`
  使 `E·M = Y`（`E` 满行秩，`M = E⁺Y` 即可）⇒ **`PE` 的有效出口头族 = 任意线性头族，
  ⊇ `IND` 的表族**。所以本实验的 `PE` vs `IND` 比的是**参数化 + 初始化**（`PE` 恒等起步、
  每出口 d² > V·d 参数），不是表达力；`IND` 的「不可 scale」是**参数量/真词表**问题
  （V·d·出口数），不是「只有它能表达」。数值证据见单测 `T5`（`lstsq` 解 `E M = Y`，残差 < 1e-5）。

## 4. 判据线（预登记，跑后不改）

* **P1（主判）**：各出口 `tf@k`（深度 1..7，全深度出口 8 为验证者）。
  逐臂相对参考臂（同批 `T`；无 `T` 则历史锚）做**同批配对差**，用按批 cluster
  bootstrap 95% CI 判「可排序 / 不可分」，同时报各臂水平的 Wilson 95% CI 与
  n（每批 16×255 = 4080 token、20 批 = 81,600）。**结论以「浅出口（1..4）是否出现
  可分且为正的 Δ」为核心**。
* **P2**：末层（全深度）clean bpc **不劣于**参考臂（不越 **0.014 bpc 带**，限配对 Δ），
  并单列读出参数量（`IND` = 出口数 × V·d = 9 × 52,736 = 474,624，判可否 scale）。
* **P3**：prob0 / 软提交语义仍可用——prob0 ECE 相对参考臂的差 ≤ **0.003**（承 2610 判据线，
  本实验声明为容差），并报「出口 logits 的 argmax 与 tied 表基坐标 argmax 的一致率」
  （`T`/`PE`/`LR` 构造性 1.0，`IND` 无 tied 桥、须实测）。
* **P4（否证线）**：
  - 若 `PE` 不能让浅出口 `tf@k` 可排序提升 ⇒ 判「**共享表够用**（本尺度）」；
  - 若只有不可 scale 的 `IND` 能提升、而 `PE`/`LR` 不能 ⇒ 判「该手段不实用」；
  - 若任何臂损末层质量（`Δ clean bpc > +0.014`）⇒ 判「**不成立**」。
* **带声明**：clean bpc 差用 **0.014 bpc 带**（限配对 Δ）；`tf@k` / 一致率等**率类**指标走
  配对 cluster bootstrap / Wilson 95% CI，**不跨量纲借用** 0.014。
* **必须不变**：因果性（单测 `T6` + 评测里的 `causal` 列）、评测确定性（同 checkpoint
  两次 `depth_curve` 逐位一致，`reeval_bitwise_identical`）。

## 5. 口径与锚复用（SCHEMA §9 三道门）

1. **开关关闭逐位同前向单测**：`T` 臂 = stock `readout`（单测 `T1`）；
2. **逐字段 config 对照**：`--stage anchor` 输出 `anchor_recheck.json::config_equivalence`
   （17 个字段，含 `final_weight` 的代码谱系标注），并含一个**能变红**的变异用例（单测 `T7`）；
3. **口径标注**：用**本文件自己的**电池重测 23c 锚 checkpoint，与 2610 用当前代码重测的
   `results.json::A_arm`（clean bpc 1.8582、tf@4 0.9694、tf@6 0.9938、ECE 0.0066）
   **逐位对照**（`anchor_recheck.json::tf_matches_2610_bitwise`）。
   ⇒ 本实验的 `tf@k` / bpc / ECE 与 2610、与锚同口径可比。

**与 3.2.1 的关系（须在复核里核）**：`3.2.1` 的 `2621-per-depth-readout-13m` 已有
「逐深度独立读出」臂 `PD`（`h_k → h_k + P_k(Q_k h_k)` 再走共享 tied 表），其有效出口头
`F_k = E(I + P_kQ_k)` **落在 `PE` 的同一函数族内**；`3.2.1` 的结论是 `PD` 被等参数
「共享宽变换」对照否证（Δ −0.0759 vs −0.1313）。本实验的 `PE` 是**另一种参数化**
（直接 `M_k`、恒等起步、主判据是逐出口 `tf@k` 而非整体 bpc），二者不是同一个读数；
但若 `PE` 出读数，必须与该结论同框解读（`analysis.md` 声明），不得当成独立证据。
指针（只引用，不复制）：`axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/`。

## 6. 每出口读出的等效理论量（MACs/token，不含时间量）

主干 = L·(4d² + 2Td) + L·2·mlp_mult·d²；出口读出每出口 = d·V（tied 表）+ 校正项
（`PE` +d²、`IND` +0 但参数 V·d、`LR` +d·r+r·V）。训练期每 token 在 9 个出口算 logits、
推理期只 1 个出口 —— 两列分开报（`results.json::arm_rows[*].cost`）。出口读出**不重跑主干**，
layer-passes 每 token 不增加；tokens-per-forward = 1。

## 7. 单测门（`test_2645_per_exit_readout.py`，17 项）

`T1` 等值门（`T` = stock、`PE`/`LR` 恒等起步 = `T`、`IND` 表拷成 `E` = `T`）；
`T3` 参数量逐臂对齐解析值 + 四臂主干逐位相同；`T4` 变异（改一个出口的头必须让该出口变红、
其余出口逐位不变；`PE`/`LR` 起步梯度非零）；`T5` 表达力（`E M = Y` 可解）；
`T6` 因果性；`T7` 锚逐字段对照 + 变红用例；`T8` 自持接受率电池与引擎 `tf_acceptance` 逐位一致。

## 8. 运行计划、优先级与现状

```
python3 run.py --stage anchor                        # 锚重测（不需训练）
python3 run.py --arm PE --seed 0 --steps 12000       # 首选臂
STEPS=60 HOLD_ALL=1 ARMS="T:0 PE:0 IND:0 LR:0" bash train_arms.sh   # 冒烟（rc=0 前置）
python3 run.py --aggregate                            # 汇总 P1–P4 → results.json
```

* 优先级（派单）：冒烟 rc=0 → `PE:0`（价值最高）→ `T:0`（同批对照）→ `IND:0` → `LR:0`；
* 时间盒硬停 **23:30+08:00**；GPU 按 `SCHEMA.md` §20 逐臂取锁（`/tmp/pathlm_gpu.lock`，
  前台、单次等待 ≤ 15 min、**不杀他人进程、不回收他人锁**）；
* 单条命令 ≤ 180 s ⇒ 12k 臂（2610 同规模实测墙钟 ≈ 12.7 min）**无法**在前台单次命令内跑完，
  且引擎无 `--resume` ⇒ 分块续训不可行。故本次实测只承诺**冒烟 + 锚重测**；
  12k 臂若未跑，`results.json` / `analysis.md` 一律写「**未跑**」，不推断。

## 9. 本次状态（只写事实）

* 预登记与实现已入库：本文件（`README.md`）+ `run.py` + `test_2645_per_exit_readout.py`
  + `train_arms.sh` + `drive.sh`；`pytest -q` 17 项全绿（CPU 门，见 §7）。
* **实测臂：一条也没跑**（`T` / `PE` / `IND` / `LR` 全部「未跑」）。两条独立原因：
  ①本单硬约束「单条命令 ≤180 s」vs 同规模实测 12k 墙钟 12.69 min（`2610/run_log.txt`），
  且引擎无 `--resume` ⇒ 前台无法承载也无法分块续训；②GPU 锁被其它轴长任务连续占用，
  两次共约 4 min 前台等锁（110 s + 140 s）均未取到（不杀他人进程、不回收他人锁）。
  ③冒烟（60 步）与 `--stage anchor` 同样需锁 ⇒ 未执行。
* `results.json` = `--aggregate` 的**纯 CPU 聚合**（每个臂 `status: "未跑"`，无任何读数）；
  `analysis.md` 首段人话摘要 + §4 受阻原因 + §5 与 `3.2.1` 的边界 + §6 下一步最小实验。

## 10. 边界与未测项（预登记）

* 单 seed、单尺寸（13M / 8L / V=206）：量级与规模外推未测；**真词表（100k+）下 `PE` 的
  表达力优势消失**（V > d 时 `E·M` 只有 d 维自由度）⇒ 本项结论**不外推到真词表**。
* 「出口在哪一层」与「每 token 跑几层」归深度轴 `2.1.1`，本项只判「出口 → 词表」。
* 只测**训练期逐层监督**（aux 等权 1.0）下的出口质量；推理期早退调度不在本项。
