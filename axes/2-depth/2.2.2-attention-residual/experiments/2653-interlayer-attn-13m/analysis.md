# 2653 · 层间 causal attention（2.2.2）· 判读

状态：**两臂（`attn-scalar` / `attn-qkv`，seed 0、12000 步、各 98.3M token）已训练并评测，
主判据 B 表已落盘**；判据线（`README.md` §4 P1–P6）**未改**。原始读数：`results.json` /
`results_<arm>_s0.json`。

## 人话摘要（先读这段）

**这一格在问什么。** 普通 transformers 的每一层只读**上一层**的输出（单残差串联）。本实验把这件事
换成「每一层对**之前所有层的输出**做一次 causal attention 聚合」——后面层可以直接回头看早层，
权重由内容决定（`q·kᵀ`）。问题：这份「内容相关权重」值不值它多付的参数与算力。

**怎么测的。** 两个参数化共用同一套聚合：
`attn-scalar` 的 q/k/v 恒等、每层只学一个温度 `τ_k`（多 **8** 个参数，每 token 多 18 432 MACs
= 注意力的 **+0.59%**）；`attn-qkv` 换成可学习共享 `W_q/W_k/W_v`（多 **196 616** 个参数，
多 5 261 312 MACs = **+167.3%**）。对照是两个同样跑过 12k 步、同 seed、同配置、同 98.3M token
的历史臂：stock 单残差 `sequential`（clean bpc **1.5254**）与 2.2.3 的层间**线性**学习加权和
`dense`（**1.5246**）。判据：`Δ = 对照 − 本臂`（正 = 本臂更好），|Δ| < 0.014 一律判「不可排序」。

**测出来多少。**

| 臂 | clean bpc（越低越好） | 相对 stock 单残差 1.5254 | 相对线性聚合 1.5246 | 训练后活性 |
|---|---|---|---|---|
| `attn-scalar` | **1.5336** | **−0.0082**（比带 0.014 小 ⇒ 不可排序） | −0.0090（同样不可排序） | ⚠️ 层 2–3 退化成「只取紧邻上一层」 |
| `attn-qkv` | **1.5431** | **−0.0177**（越带 ⇒ 明显更差） | −0.0185（越带 ⇒ 明显更差） | ✅ 所有层都在混合 |

读法：scalar 臂的 1.5336 与 stock 的 1.5254 差 0.0082 —— 小于 0.014 的噪声带，也就是说
**这一个 seed 上说不出谁更好**（而且差的方向是不利于本臂）。qkv 臂差 0.0177 —— 超过噪声带
26%，是**可以说出口的「更差」**，同时还多付了 167% 的每层注意力算力与 19.7 万参数。

**结论（人话）。**

- **本尺度（13M / 12k 步 / 0.014 带）没有净增量，而且 qkv 反被否证**：把层间接法换成层间
  causal attention，少的那个参数化（+8）只到「说不出差别」，多的那个（+19.7 万参数、
  +167% 注意力算力）**明显更差**（−0.0177）。所以「内容相关权重（q·kᵀ）值那份额外算力」
  在本尺度的答案是**不值**；层间连接用**线性**加权和（2.2.3 的 `dense`，+36 参数）就够，
  这也正是 `README.md` §1 预登记的可证伪方向。
- **预登记的可证伪条件只在一臂上成立**（原文：两臂都落在 ±0.014 内 ⇒ 判「在本尺度无净增量，
  应退化为线性聚合」）。实测 scalar 落带内、qkv 越带且方向为「更差」⇒ 结论（无净增量）不变，
  但证据形态是「一臂中性 + 一臂显著更差」，不是「两臂都不可排序」。
- **scalar 的「中性」要打折看**：训练后它的层 2 与层 3 的层维权重已经退化成 one-hot
  （只取紧邻上一层，即退回到 stock 残差），层 4–8 才仍在混合（熵 0.59–0.99）。也就是说
  这个臂在浅层**自己选择了不做跨层聚合**——1.5336 是「部分退化 + 部分混合」的读数，
  不能读成「混合没收益」的干净证据（`README.md` §4 的 P2 活性门未过）。
- **不是「没收敛」造成的假象**：两臂末 25% 窗的 loss 斜率只有 0.00039（scalar）/ 0.00296
  （qkv）nats/千步，是接近收敛的 matched-step 快照。

## A 表：训练前的门（冒烟，`smoke_report.json`）

| 项 | `attn-scalar` | `attn-qkv` | 出处 |
|---|---|---|---|
| stock 参数量（同配置 sequential） | 6 962 688 | 6 962 688 | `smoke_report.json:C_param_accounting` |
| 本臂参数量 | 6 962 696 | 7 159 304 | 同上（训练后实测同值，`results_<arm>_s0.json::params`） |
| `params_added`（= 解析值） | +8 | +196 616 | 同上 |
| 额外 MACs/token（层间聚合，解析） | 18 432 | 5 261 312 | `interlayer_attn.aggregation_cost()` |
| 额外 / stock 每层注意力（3 145 728 MACs/token） | 0.005859 | 1.672526 | 同上 |
| 等值门（one-hot ⇒ 与 stock 逐位一致，前向 + loss） | ✅ `true` | ✅ 同一聚合路径 | `smoke_report.json:A_equivalence_gate` |
| trunk 参数逐位相同 | ✅ | ✅ | 同上 |
| 活性（训练**起点**）：层维权重非退化 | ✅ | ✅ | `smoke_report.json:B_liveness_gate` |
| 单测 | 11 passed | — | `test_2653_interlayer_attn.py` |

等值门探针（证明这条门会红）：不强制时前向**必须**与 stock 不同 —— `probe_free_weights_forward_differs=true`；
`MODE`/`FORCE_LAST` 在建实例时冻结（前向时读全局会让这条门静默失效，已修并有回归单测）。

## B0 表：训练后的活性与代价（P2 / P3，实测落盘）

| 项 | `attn-scalar` | `attn-qkv` | 出处 |
|---|---|---|---|
| 层维权重熵（k ≥ 2，两次前向的全部读数） | 0.00 – 0.99 | 0.68 – 1.58 | `results_<arm>_s0.json::liveness.per_layer` |
| `w_immediate_prev`（只取紧邻上一层 = 1） | 0.56 – **1.00** | 0.06 – 0.46 | 同上 |
| **P2 活性门**（k ≥ 2 每层 `w_immediate_prev < 1` 且熵 > 0） | ❌ **未过**：层 2 = (熵 0.00, w 1.00)、层 3 = (熵 0.03, w 0.994)；层 4–8 正常混合 | ✅ 过：每层都混合 | 同上 |
| `τ` 终值（层 1→8） | 0.937 / 0.954 / 0.841 / 0.095 / 0.166 / 0.336 / 0.243 / 0.424 | 0.937 / 0.093 / 0.078 / 0.047 / 0.073 / 0.074 / 0.118 / 0.129 | `results_<arm>_s0.json::liveness.tau_by_layer` |
| 额外参数 / 额外 MACs（P3，解析） | +8 / 18 432（+0.59%） | +196 616 / 5 261 312（+167.3%） | `results_<arm>_s0.json::params` |
| 训练墙钟（过程量，**不进判据**） | 9.73 min | 10.78 min | `results_<arm>_s0.json::shared_train_results.wall_minutes` |
| 峰值显存（净分配，`max_memory_allocated`） | 14 961.7 MB | 15 141.1 MB | `results_<arm>_s0.json::shared_train_results.peak_gpu_mem_mb` |

峰值显存只是资源事实登记，**不进判据**：`README.md` 的预登记目标为 ≤5GB net，本实验两臂实测
≈14.6 / 14.8 GiB（同一进程内 32×256 批训练峰值），**超出该自定目标**；这笔超支不改两臂共享同一
训练配方与同一批尺寸 ⇒ 不影响跨臂对比，也不改变任何判词。

两个 `alive` 同名不同义，读数前先分清：`topology_liveness.alive`（`runs/attn-scalar-s0-12k/results.json::topology_liveness`，引擎 `topology_liveness` 的输出）判的是**温度 τ 是否离开初值 1.0**——scalar
八层 τ 全部动过（`tau_frac_layers_moved = 1.0`）⇒ `true`；`liveness.alive`
（本目录 `results_attn-scalar_s0.json::liveness.alive`，即下表 **P2 层维权重非退化门**）判的是
**层维权重有没有退化**（k ≥ 2 每层 `w_immediate_prev < 1` 且熵 > 0）——scalar 的层 2–3 塌成
one-hot ⇒ `false`。所以「干预是否活」在两条门上结论相反不是矛盾：τ 一直在学，只是它学到的层维
权重在浅层退化成「只取紧邻上一层」。

scalar 退化的机制是该参数化固有的：`q = h_{k−1}` 而 key 里恰好也有 `h_{k−1}`（自相似），
训练把 `τ` 压小 ⇒ 注意力变尖 ⇒ 浅层（源只有 2–3 个）直接塌成 one-hot，等价于 stock 残差。
这也解释了为什么它的读数更接近基线：**一部分层已经不做跨层聚合了**。

## B 表：主判据（P1，实测）

| 臂 | clean bpc | Δ vs stock（2613 s0 = 1.5254） | Δ vs dense 线性聚合（= 1.5246） | 判词 |
|---|---|---|---|---|
| `attn-scalar` | **1.5336** | **−0.0082** | **−0.0090** | 不可排序（H0）：\|Δ\| < 0.014 |
| `attn-qkv` | **1.5431** | **−0.0177** | **−0.0185** | 可分（反向）：劣于两个对照 |

- **窗口读数**（5 窗口 × 4 批，同一 generator seed 的批序，判据取窗口均值）：
  `attn-scalar` = 1.520219 / 1.572351 / 1.541370 / 1.522854 / 1.511347（均值 1.5336）；
  `attn-qkv` = 1.528960 / 1.587777 / 1.543955 / 1.528530 / 1.526134（均值 1.5431）。
  `results_<arm>_s0.json::eval.windows.window_bpc`。
- **窗口间 SE**（相对 stock 的窗口均值差）：scalar −0.008228 ± 0.010844；qkv −0.017671 ± 0.011612
  （5 窗口，`results.json::P3_window_se`）。这是**同一 checkpoint 内的抽样噪声**；历史基线臂的
  逐窗数据不可得，故**不是真配对 SE**，不用于判据，只作量级参考（0.011 与 0.014 同量级 ⇒
  两臂的 Δ 都只有「一两个窗口噪声」的强度，scalar 落在带内、qkv 略越带，结论按预登记带判）。
- **两条评测路径一致**：`clean_bpc`（窗口法）与 `clean_bpc_from_depth_curve` 逐臂相同
  （`clean_bpc_two_paths_agree = true`）。
- **逐深度 clean bpc**（stock 形态只监督最终深度，判据只看 depth 8）：
  `attn-scalar` = 7.3964 / 4.4522 / 3.9651 / 3.5883 / 3.4111 / 2.9819 / 2.5294 / 2.1583 / **1.5336**；
  `attn-qkv` = 7.4028 / 5.3628 / 4.6907 / 5.9630 / 5.3782 / 4.4102 / 3.4455 / 3.0442 / **1.5431**
  （depth 0→8）。对照 `sequential`（2613 s0）= 7.4039 / 4.8932 / 4.3486 / 3.8608 / 3.2023 /
  2.6281 / 2.1895 / 1.7381 / **1.5254**。⇒ scalar 的中间深度整体略优于基线（depth 1–5 更低），
  但最终深度落后 0.0082（带内）；qkv 的中间深度**明显更差**且非单调（depth 3 反弹到 5.963），
  最终深度也落后 0.0177。
- 对照口径（`harness/SCHEMA.md` §9 三条）：①**实现层等值** = 上面的 one-hot 逐位单测（`smoke_report.json`）
  **且**两臂与 stock 走同一构造顺序（`tau`/`W=init` 不消耗随机数流）⇒ trunk 权重逐位相同；
  ②**配置层等值** = `run.py baseline_equivalence()` 与 2613 s0 的 resolved config 逐字段对照
  （d/layers/heads/seq/mlp_mult/dense/exit_mode/dropout/norm_cap/steps/batch/lr/seed/optimizer/
  data/train_tokens，`unequal_fields = {}`、`pass = true`）；③**口径标注** = 2613 s0 与 2653 均为
  「全深度、无退出、teacher-forcing、序列级 bpc、批 16 × 20 批」，同口径。
- 噪声带 0.014（跨 run 训练漂移）；同一 checkpoint 内的比较另报**窗口间 SE**（见上）。
- **对账（`harness/SCHEMA.md` §12）**：算术对账——额外参数与额外 MACs 都是解析式并可实测核对（已一致）；
  任务对账——两臂与基线在同一任务定义、同一输入分布、同一位置集打分（同一评测批序）；
  口径对账——全深度 bpc，无墙钟量进判据。

## 实现要点（为什么这样接线）

- 复用引擎 `topology="dense"` 的前向路径（`h, _ = blk(self._layer_mix_input(states, k))`），
  只覆写 `_layer_mix_input` = 层间注意力聚合；`pathlm/**` 未改（实验内工厂替换
  `train_depth_ar.DepthARModel`，训练整段走共享 `main()`）。
- **层索引因果**：聚合只取 `states[0..k−1]`（`j ≤ k−1`），看不到当前层之后的信息。
- 初始化：`τ = 1`、`W_q = W_k = W_v = I`。一个**如实观测**：由于 `q = h_{k-1}` 而 key 里
  恰好也有 `h_{k-1}`（自相似），训练起点处层维权重**已经**偏离「只取上一层」
  （k ≥ 5 的层熵 ≈ 1.6–2.1、`w_immediate_prev` ≈ 0.15–0.24；层 2–4 更低，熵 0.69–1.38；
  全 k ≥ 2 区间为熵 0.688–2.075、`w_immediate_prev` 0.152–0.551，与 `smoke_report.json`
  的 `B_liveness_gate.per_layer` 一致；最大权重落在紧邻上一层上）。
  也就是说两臂的起点并不等于 stock 起点（不等值起步），这是该参数化固有的、已落盘的性质；
  训练后的去向（scalar 浅层塌回 one-hot、qkv 全层混合）见 B0 表。
- `results_<arm>_s0.json::liveness.max_dev_from_identity_prev` 这个键名沿用旧称，实际值是
  「k ≥ 2 各层 `w_immediate_prev` 的最大值」（scalar 1.0、qkv 0.4636），**不是**「偏离 1 的量」；
  本文按 `per_layer` 逐层读数说话。

## 残余缺口 / 下一步最小实验

1. **n ≥ 2（唯一会改判词的事）**：本尺度只跑 s0。两臂的 Δ 都在 0.014 带附近（scalar −0.0082、
   qkv −0.0177，后者的窗口 SE 0.0116 与带同量级），补 seed 1 才能把 qkv 的「更差」压成
   「可排序地更差」的稳定结论。成本 = 每臂 12k × 32 × 256 = 98.3M token（≈10 min 卡时/臂）。
2. **浅层塌缩的机理未做干预**：把 `q` 改成对紧邻上一层单独加一个投影（拆开自相似），或给 `τ`
   加上下界，可以检验「scalar 的中性是不是被浅层退化吃掉了」——这是新机制，需新预登记，不在本项判据内。
3. **外推**：长度/层数外推未测（层数越多、可读的早层越多，本尺度 L=8 只有 8 个源），`gaps` 已记。
4. **基线自包含双跑（未触发）**：P5 的配置逐字段对照不等字段为空、实现层等值门通过 ⇒ 按
   `harness/SCHEMA.md` §9 复用历史对照臂，不追加 2 × 98.3M token 的双跑。

## 独立复核

**第 1 轮独立复核（fresh 上下文）判 PASS**（判词由 lead 落盘于 `.scratch/61-axis2-depth/**`
与 `TICKET.md`；`harness/SCHEMA.md` §14/§14b，`harness/research-schema.md` §11.3 累计 ≤2 轮）。
其 non-blocking 编号清单 N1–N3 已处置：N1（参数倍率 2.7 万 → **2.5 万倍**，`project.yaml::argument`）、
N2（活性区间限定词 → 「k ≥ 5（层 2–4 更低，熵 0.69–1.38）」，本文件与 `project.yaml` 同步）、
N3（`README.md §7/§9` 落盘清单删去并不存在的 `run_log.txt`）。两臂训练完成后的这一版 B 表
**尚待新一轮独立复核**（判据线未改，只填读数）。

## 回归与可复算的校验命令

| 命令 | 结果 |
|---|---|
| `python3 smoke.py` | 三道门（等值 / 活性 / 参数账）全过，报告落 `smoke_report.json` |
| `python3 -m pytest …/test_2653_interlayer_attn.py -q` | 11 passed |
| `python3 run.py --arms attn-scalar,attn-qkv --seed 0 --steps 12000` | 两臂 12k 步训练 + 评测 + 汇总（`--no-train --force` 可用已存权重只重算 B 表） |
| `python3 tools/summary.py --check` | OK（46 份 `project.yaml`） |
| `python3 -m pytest tests/ -q` | 866 passed / 1 failed —— 失败为 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`，涉 `axes/6-predict/**` 的三个 `mutate.py`（**非本 WP 引入、与本 WP 无关**，预存失败分开登记） |
