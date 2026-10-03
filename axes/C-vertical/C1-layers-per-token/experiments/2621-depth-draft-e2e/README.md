---
name: 2621-depth-draft-e2e
ticket: 38-W2
date: "2026-10-03"
grid: C1-layers-per-token
engine: "depth-ar（d=256 / L=8 / 13M，23c-H2-w0-12k）"
budget: "GPU 纯推理 ≤25 min（不含训练；不训练）"
---

# 2621-depth-draft-e2e（ticket 38 · W2）

> 主格：`C1-layers-per-token`｜关联格：`H5-verification`、`C6-depth-supervision`、
> `C5-refinement-operator`｜引擎：depth-ar 13M 复用 checkpoint｜预算：纯推理 ≤25 min

## 1. 命题（一句话、可证伪）

**在同一 checkpoint 上，把「depth-4 出口」当真草稿、把全栈当真验证，端到端实测的
接受率与 tok/s 是否支持「tf@4 = 0.97 是一条可部署的草稿资产」——具体地：实测接受率与
TF 代理口径的偏差有多大；自 spec 的墙钟收益是否只在 launch-bound 小模型上成立；
「分级验证」（d4 → d6 → 全栈，逐级放行）是否比「一次全栈验证」更省。**

可证伪点：若实测接受率 ≈ TF 代理（偏差 < 1pp）且 spec 墙钟增益在等效 FLOPs 上也是正的，
则「深度出口 = 免费草稿」成立；反之则不成立。

## 2. 单因素声明

- 本实验**不训练**，只做推理调度。基底 = 已入账的 stock depth-ar 形态
  （`23c-H2-w0-12k`，sha256 见 `MODEL.md`）。
- **唯一干预** = 草稿用哪个深度的出口 + 验证分不分级；其余一切（checkpoint、prompt、
  批大小=1、提交方式=`hard`、生成长度、硬件）逐字段相同。
- **对照臂**：① 普通全栈贪心（同 checkpoint）；② 固定 exit-4 解码（同 checkpoint）；
  ③ TF 代理口径（`tf_acceptance` on enwik8）。三者的对照物都是**同一模型的自身输出**，
  不是 oracle（SCHEMA §11）。
- 无「主流 SOTA 的非标准因素」引入（不新增任何算子/参数；`spec_decode.py` 是新模块，
  不改 `pathlm/`——ticket 38 期间 `pathlm/depth_ar.py` 归 W4 线独占，SCHEMA §22）。

## 3. 指标表（多维、逐项）

| 维度 | 指标 | 它解决什么问题 |
|---|---|---|
| 草稿准确性 | 实测接受率（真解码路径、逐位置） | **用户直接关心的量**：tf 代理 0.9694 能否兑现 |
| 代理可靠性 | `tf_enwik8` / `tf_selfstream` / `real_pos1` / `real_pooled` 四口径差 | 量化 TF 代理口径的偏差来源（分布 vs 提交态） |
| 吞吐（墙钟） | tok/s（3 次重测中位数，批大小 1） | 部署口径 |
| 吞吐（算力） | 等效 FLOPs / 等效 layer-position | **与墙钟必须并报**（SCHEMA §17.9） |
| 验证分级 | `single` vs `graded` 的 tok/s 与等效 FLOPs | 「逐级放行」是否真省 |
| 成本 | 每生成 token 的等效 layer-position、草稿接受数/全栈前向数 | 成本交叉点 |
| 质量 | 发出流 vs 普通贪心流的逐 token 一致 | 自 spec 不改质量的正确性护栏 |
| 副产品 | 轮内第 i 个草稿的接受率曲线 | 「加深草稿链是否还赚」 |

## 4. 判据线（预登记，事后不得改）

1. **接受率偏差**：报 `Δ_tf = real_pooled − tf_enwik8@4`，并**同时报** `real_pos1`
   （与代理最可比的那个位置）与 `tf_selfstream@4`，用它把偏差拆成
   「分布位移」和「提交态 vs 新鲜 embedding」两部分。判据：若 `real_pos1` 与
   `tf_enwik8@4` 相差 > 3pp，则**代理口径不可直接当部署接受率用**（必须在结论里写明）。
2. **端到端 tok/s**：主臂（`single, γ=4, dd=4`）的 tok/s 与普通全栈、固定 exit-4
   三分；3 次重测取中位数。判据：`single` 的 tok/s 是否 > 普通全栈。
3. **分级 vs 单级**：`graded`（d4→d6→全栈）与 `single`（一次全栈）的 tok/s 与等效
   FLOPs。判据：`graded` 的等效 FLOPs 是否 < `single`（省算力），以及 tok/s 是否
   随之改善或持平（launch-bound 下可能不改善）；两臂发出流必须**逐 token 相同**。
4. **双口径**：每个臂都报「等效 FLOPs」与「墙钟」两列，并给出「wall 增益 / FLOPs 增益」
   的比值；若一正一负，结论句必须同时写两个方向，不得只报有利的那个。

**方差界**：接受率以 2 项分布 SE 报（n = 总草稿数）；tok/s 用 3 次中位数并以
极差（max−min）/中位数 报抖动。Δ < 0.014 的接受率差不排序（SCHEMA 方差界借用）。

## 5. 对照与方差

- 对照臂见 §2；**同一 prompt 集合**（enwik8 val 的 4 个固定 offset，prompt_len=16）
  喂给所有臂——对照臂在同一任务定义、同一输入分布、同一位置集下打分（SCHEMA §13）。
- 每个臂 × 每个 prompt **重测 3 次**，取中位数；同一 prompt 的多次重测必须给出
  **逐 token 相同**的贪心流（`run.py` 内断言），否则报错。
- seed：推理无采样（贪心），因此 seed 不构成方差源；prompt offset 是唯一的
  「数据 seed」，固定并列出。
- **对照复用**：复用 `23c-H2-w0-12k` checkpoint（SCHEMA §9）。等值性证明 =
  （a）实现层：`tests/test_depth_draft_e2e.py::test_committed_cache_matches_depthar_decoder`
  证明本实验的 KV 语义与 `pathlm/depth_ar.DepthARDecoder` 在容差内一致；（b）配置层：
  checkpoint 的 `results.json` 记录了 d256/L8/h8/seq256/m4/seed0/exit_mode=uniform，
  与本实验 `CKPT_CFG` 逐字段一致（`run.py` 断言加载）；（c）口径标注见 `analysis.md`。

## 6. 成本

- `gpu_wall_budget: 25 min`（含冒烟 + 全跑 + 对账；**无训练**）。
- 峰值显存：冒烟实测 **35.7 MB**（远低于 §20.1 的 1 GB 例外线）。
- 可中断性：高（纯推理，每臂独立落盘）。
- 冒烟计划：先跑 32 token 测步时再定预算（SCHEMA §17.8）；已执行，结果写
  `run_log.txt` 的 `[smoke]` 行。
- **预算不够时砍什么**：砍臂（先砍 `dd=6`、再砍 `γ=2`），砍 prompt 数（4→2），
  砍 proxies 的 batch 数；**不砍判据线**。

## 7. 数据与脚本清单

| 项 | 值 |
|---|---|
| 数据 | `data/enwik8_full.npz`（字符级，V=206，val 段 10M）；prompt = val 的 4 个固定 offset 各 16 字符 |
| 入口 | `run.py --ckpt … --out …`；核心实现 `spec_decode.py` |
| resolved config | `CKPT_CFG = {d_model:256, n_layers:8, n_heads:8, seq_len:256, mlp_mult:4}` |
| 落盘 | `results.json` / `run_log.txt` / `cycle_log.jsonl` / `analysis.md` / `MODEL.md` |
| 单测 | `tests/test_depth_draft_e2e.py`（10 条） |
| 对账 | `reconcile_numbers.py` |

## 8. 失败决策树（无效时按此顺序排除）

1. **测量**：先查 `cycle_log.jsonl` 的 `emitted == accepted+1` 与 `committed` 单调性；
   查 `tok_s` 的三次极差是否被邻居线污染（若 >20% ⇒ 重测）。
2. **实现**：跑 `test_single_spec_matches_plain_greedy` 与
   `test_spec_cache_kv_matches_reference_batch_forward`；任一红 ⇒ 实现错，先修。
3. **规模**：若墙钟增益为正但很小，用 13M 与 ticket 20 的 100M 宽度-spec 结果
   （增益随规模消失）说明这是 launch-bound 现象——**不得**外推到 100M（SCHEMA §17.11）。
4. **设计**：若接受率 ≈ TF 代理而墙钟仍不赚，则结论是「深度出口的部署价值在
   *forward 次数*口径，不在单流墙钟」——按设计层判读，不推翻机制。

## 9. 产出

- `results.json`（机读，含全部臂与代理口径）
- `run_log.txt`（[ckpt] 哈希 / [smoke] 步时 / 逐臂一行）
- `cycle_log.jsonl`（每 cycle 一行的活性量）
- `analysis.md`（判读 + 人话摘要）
- `MODEL.md`（checkpoint 哈希与复现命令）
- `reconcile_numbers.py`（报告数字逐一对应的对账脚本）

---

## §既有做法（prior art，SCHEMA §18）

- **谁做过**：① `LayerSkip`（Elhoushi et al. 2024）——同模型早退层当草稿 + 全栈验证的
  **自推测解码**（self-speculative decoding），并用 early-exit loss 训练出可用的浅出口；
  ② 本仓 **H5-verification** 格已判 Medusa 式批量验证可用且逐位一致（`.scratch/13-four-experiments`、
  `.scratch/20-spec-scale`）；③ 经典 spec decoding（Leviathan 2023 / Chen 2023）。
  （①②中「LayerSkip 的具体数字」为 `记忆·未核`，仅用于设计方向；③ 为常识。）
- **为什么这么设计**：草稿要便宜、验证要保真。同模型浅出口的好处是**不引入第二个模型**、
  与主干权重/输入分布天然一致；代价是浅出口本身要训练。
- **它的结果**：`记忆·未核`：LayerSkip 报告早退+自推测有吞吐收益，但报道口径多为
  **forward 次数/延迟**而非 FLOPs。**`本地实测`（ticket 20，宽度轴）**：批量验证的墙钟
  收益是 **launch-bound 小模型现象**——13M 宽度-3 前向成本 1.28×、k=2 稳态 **1.68×** 墙钟；
  100M 变 compute-bound（3.4–5.3×），**k=1 墙钟 0.80–0.96×（净亏）**；接受级联在规模上稳定
  （tok/forward ≈2.4）。
- **我们的增量**：
  1. **换轴**：ticket 20 沿「宽度（一次前向多位置）」spec；本实验沿「**深度（浅出口）」spec**，
     用的是 C1 自己的出口资产，且**不训练**（复用 23c checkpoint）。
  2. **把 tf 代理口径拉回真实解码路径**：C1 的出口资产一直是 `tf_acceptance`（enwik8 真值流、
     新鲜 embedding）；本实验给出**同一 checkpoint 的四口径分解**（enwik8 代理 / 自生成流代理 /
     真解码路径位置-1 / 真解码路径汇总）。
  3. **算力价签**：给出「等效 FLOPs」与「墙钟」两口径，并显式回答「同模型浅草稿到底省不省 FLOPs」
     ——按机制推导预期是 **≈ 中性（全接受时）/ 略负（有拒绝时）**，本实验直接测出来。
  4. **分级验证**：d4 → d6 → 全栈的「逐级放行」与一次全栈的对照（本地无前例）。

## §23 七问自查（每格预算内极致）

| # | 问题 | 本实验的答案 |
|---|---|---|
| 1 | 既有做法查过吗？业界做透的写成 reference 了吗？ | 见上「既有做法」；`LayerSkip`=同机制、ticket 20=同仓宽度轴先例。分诊：**机制 (a)（不重跑），但我们的设定（同引擎形态出口资产 + 双口径 + 分级验证）实质不同 ⇒ (b) 做实验**，增量四条已列。 |
| 2 | 判据是最严的那个吗？有诚实对手/毁源/因果/活性护栏吗？ | 诚实对手 = **同一模型的全栈贪心**（不是 oracle）；质量护栏 = 发出流逐 token == 普通贪心（可失败单测）；活性量 = 逐 cycle 账本落盘 + `cycle_log.jsonl`；因果 = 草稿深度开关的 liveness 单测。 |
| 3 | 三类「错的对照」都试过吗（同参数量/同算力/同字节）？ | 同参数量：无新参数（不适用）；**同算力**：这是本实验的核心（等效 FLOPs 与墙钟两口径同列）；同字节预算：不适用（无压缩）。 |
| 4 | 规模/预算的边界探过吗（≥两档）？ | γ ∈ {2,4,8} 三档 + 草稿深度 {4,6} 两档 + 固定 exit-4/exit-6 两个部署参照点；**规模只有 13M**（100M 无 sha256 记录，列为下一步 — 显式不外推）。 |
| 5 | 反方向/否证路径走完了吗？ | 计划内：若墙钟不赚或接受率与代理偏差大，按 §8 决策树逐层排除并把「agent 口径不可用于部署」写成结论。 |
| 6 | 未测项显式列成「下一步最小实验 + 成本」了吗？ | `analysis.md` §6 列出（100M sha256 化 + 100M 重测；「信任 d6」的省一级变体；γ 更大档）。 |
| 7 | 复核通过了吗？ | 待独立复核（§14，新 run、在本实验落盘后启动）。 |
