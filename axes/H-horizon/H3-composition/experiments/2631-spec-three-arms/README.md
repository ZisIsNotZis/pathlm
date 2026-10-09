---
name: 2631-spec-three-arms
ticket: 48-R6
date: "2026-10-02"
grid: H3-composition
engine: "width-MTP（pathlm/model.py + pathlm/decode.py；d=256/L=12/13.4M，n_mtp=3，chain_mode=token）"
budget: "GPU ≤120 min（步时冒烟 + 4 臂训练 + 3 臂 × 2 尺度的评测）；先测步时再定稿"
---

# 2631-spec-three-arms（R6 · Spec 三臂：`chain` / `direct` / `combined`）

> **格归属（主 Agent 2026-10-02 裁决）**：本实验放在**已有格 `H3-composition`**，不新开 H6。
> 理由：H3 的命题原文就是「**多位置预测的组合方式**」，而 `chain / direct / combined`
> 正是这个命题。此前该格的 `2613-self-gated-composition-13m` 测的是 `T∘T` **潜在复用**
> （direct `T(h)` / composed `T(T(h))` / gated 通道混合），**回答的不是这个命题**
> （独立复核 A6 已确认）。
> 因此：**本实验取代 2613 作为该格命题的正式回答**；2613 降级为「冻结表征上的重复潜在
> 映射与门控」的背景实验（不改 2613 的任何文件）。
>
> `TODO(主Agent): 结果出后更新 H3 的 catalog verdict`
>
> 入口：`smoke_steptime.sh`（先测步时）→ `train_arms.sh`（按 (scale, seed) 训练，**按臂释放锁**）→
> `eval.sh`（取锁整程、驱动 `run.py`：三臂 × native 对照评测 + 聚类 bootstrap + 算力账；
> 本波实测 `--repeats 6`）；臂定义与指标
> `spec_arms.py`；本地单测 `test_spec_arms.py`；变异 `mutate.py`；对账
> `reconcile_numbers.py`；原始结果 `results.json`；判读 `analysis.md`；权重清单 `MODEL.md`；
> 对照与方差说明 `controls/README.md`。

## 1. 命题（一句话、可证伪）

**在同一 checkppoint、同一算力预算下，把「第 N+2 个 token 的草稿」按三种不同的草稿路径
构造——`chain`（先出 N+1，**以 N+1 为条件**出 N+2）/ `direct`（从同一状态**一次直接**出
N+2）/ `combined`（两条分布**按概率混合**）——哪一种的墙钟加速比（vs 原生逐 token 自回归，
同硬件同批量）最高？**

一句话可证伪的形式：*三臂共享同一模型、同一次前向、同一个验证器、同一批 prompt，唯一
差别是第 N+2 草稿位的分布来源；若某一臂的**配对逐次重复墙钟加速比**的 95% CI 下界
> 1.0 且显著高于其余两臂，则它胜出；若三臂之差落在重复间抖动带内，则记为**不可排序**
并如实报「三者等价」。*

**反例警示（写进契约）**：
`T∘T`（潜在复用：把 `T1` 再套一次在**同一个** latent 上）**不是**这三者中的任何一个——
它没有概率、第一遍不受 N+1 的 token 监督。**H3 的 2613 结论不得用于回答本命题。**
本实验刻意**不设** latent-chain 臂，以免把两种语义混在一起。

## 2. 单因素声明

- **基底**：本仓 stock `PathLM`（tied E=U、无 final LayerNorm 的宽度-MTP 解码器，
  `pathlm/model.py`），13.4M 档 = `d_model 256 / n_layers 12 / n_heads 8 / seq_len 512 /
  mlp_mult 6`，`n_mtp=3`、`chain_mode="token"`、`transport="none"`，其余开关全关。
- **唯一变量（臂）= 第 N+2 草稿位的分布来源**：
  - `direct`：`argmax node2(h_N)`（T2 变换头，一次直接出 N+2）；
  - `chain`：`argmax node1.chain2(h_N | d1)`，其中 `d1 = argmax node1(h_N)`，`chain2` =
    `T1(chain_cat([h_N, embed(d1)]))`（DeepSeek-MTP 式**以 N+1 为条件**的链头）；
  - `combined`：`argmax (λ·softmax(chain2) + (1−λ)·softmax(node2))`，**λ = 0.5 预登记**。
- **第二个草稿位（N+3）在三臂中固定**为 `argmax node3(h_N)`（direct 头）——它是**共享背景**，
  不参与此变量；把它固定住才能让「τ / tok/fwd」非退化（γ=1 时 τ ≡ α₁）。
  **能力面分析（SCHEMA §22.8）**：引擎的 token 条件链头只有**一级**条件步
  （`model.py::_mtp_nodes` 只对 node-1 产出 `chain2`），无法对 N+3 再条件一次；
  强行二次条件是**新结构**、须另立工单（记入 `analysis.md` 的下一步）。
- **第一个发出位（N+1）不设接受测试**：本引擎里 `v1 = argmax node1(h_N)` 就是**验证器自己
  的输出**，无信息可验（换任何臂都一样）。故「投机草稿位」从 N+2 起算，γ = 2（N+2, N+3）。
  这是**引擎性质**，如实写进判读，不当作「α₁ ≡ 1 的免费午餐」。
- **必要性例外（显式记账）**：`n_mtp=3` 相对 `n_mtp=2` 有 +0.0166 bpc 的税
  （`ticket 13 / mtp3`），但**四臂共享同一主干与同一 `n_mtp`** ⇒ 该税在臂间抵消，
  不构成本实验的自变量。
- 其余逐字段一致：模型规格、训练配方、数据、步数、批大小、学习率、优化器、评测协议、
  prompt 集合、生成长度、批量（B=1）。机器可核的字段落在 `results.json` 的 `resolved`。

## 3. 干预定义（写清是「实现」还是「标签」）

**本线不写共享文件**：不改 `pathlm/**`、`train_m1.py`、`tools/**`、`tests/**`、
`catalog.yaml`、`INDEX.md`、`axes/**/README.md`（本波并行写入者分工：`catalog.yaml` 属 L2，
`tools/**`+`tests/**` 属 L1）。本线只新增本实验目录内的代码。

三臂**复用已入账的生产解码器** `pathlm/decode.py::decode_spec`（Medusa 式批量验证），
只把它的 `draft_fn` 钩子接上本实验的臂定义（`spec_arms.py::make_draft_fn`）：

```
draft_fn(nodes, max_drafts) -> [d_2, d_3]
  d_2 = argmax( <arm 分布>(nodes) )        # 唯一变量
  d_3 = argmax( node3["logits"] )          # 三臂固定
```

- 验证：`decode_spec` 的批量验证（一次 width-3 前向，各位置 node-1 argmax 对立项），
  **接受最长前缀**、拒绝后缀截断重锚。发出流 == 同 checkpoint 的原生贪心流（门禁）。
- `chain` 的条件项在**训练**时用 teacher forcing（真 t_{i+1}），在**推理**时用 node-1 的
  argmax（`deploy` 口径）——这是本引擎的既有语义（`model.py::_mtp_nodes`），本线未改。
- **不做拒绝采样**：全实验为 `temperature=0` 的 argmax 路线 ⇒ 明确声明
  **非分布保持**（见 §4 门禁）。

## 4. 判据线（预登记，事后不得改）

1. **门禁（全部臂，缺一即作废）**：`temperature=0` 的发出流与**同 checkpoint 的原生贪心流
   逐位相同**（`decode` vs `decode_spec`）。若出现分歧，记录首个分歧位置与 top-2 gap，
   该臂的**流等价**判为未过（fp 平局性质，非语义错），并把该臂从主判据中剔除。
2. **主判据（墙钟）**：三臂各自的**墙钟加速比** `wall = spec_tok_s / native_tok_s`，
   同进程、同硬件、同批量（B=1）、**交替重复**测量（native/spec 在同一循环里交错，
   同一 thermal/clock 状态）。
   - 判定线 A：`wall` 的 **95% CI 下界 > 1.0** ⇒ 该臂「测得墙钟收益」。
   - 判定线 B：三臂**排序**须由**配对**（同一 prompt 集合、同一重复次数）的
     CI 支持；若最强臂与次强臂的配对差 CI 含 0 ⇒ 记「**不可排序**」，不得宣称胜者。
3. **次判据**：`α₁`（第 N+2 位接受率）、`α₂|α₁`（第 N+3 位条件接受率）、`τ`（每轮平均被接受
   草稿数）、`tok/fwd`（= 1 + τ）。臂间差 **< 0.014 不得排序**（SCHEMA 方差界）。
4. **等效算力净账（缺一不可）**：把**草稿自身成本**算进去。口径 = 以**实测**的
   width-3 / width-1 前向成本比 `c₃` 为单位：
   `净账 = c₃ / (tok/fwd)`（相对 native 的 1.0）；并另报 **MTP 头解析成本**
   （`T1/T2/T3` 各 d×d 线性 + `chain_cat` 2d×d + 4 个读出 d×V，按 MAC 计）占一次
   width-1 前向的百分比。两列并报，不得只报有利的一列（SCHEMA §17.9）。
5. **规模**：**≥2 个点**（13.4M + 一个更大档）。**≥2 次独立训练**（13.4M 档 seed {0,1}）。
   大档若预算只够 1 个 seed，**如实声明「该点只 1 个 seed」**，不得当作跨种子证据。
6. **禁外推**：绝对水平只适用于「该 checkpoint / 该 prompt 集合 / 该步数快照
   （6000 步、98.3M token）/ enwik8 风格文本」；不得读成「业界做不到/做得到」，
   不得外推到别的任务、别的规模或别的温度。
7. **报告口径**：先说「人话结论」，再给数字；代号只作括号定位。

**方差界**：接受率类以**按 prompt 聚类的 bootstrap**（B=2000，seed=2631，percentile 95%）
报 CI——**不得**把每轮草稿当独立二项样本（复核 B2 的同类错误）。墙钟以**逐次重复的极差 +
配对 bootstrap** 报，单次 tok/s 不作结论。Δ < 0.014 的接受率差不排序（SCHEMA 方差界借用）。

## 5. 对照与方差

- **对照臂 = 原生逐 token 自回归**（同一 checkpoint、同一 prompt、同一批量、同一进程、
  交替重复）。对照物是**模型自身**，不是 oracle（SCHEMA §11）。
- **数据 seed 与模型 seed 无关**：prompt 由 enwik8 val 的固定 offset 决定，四臂
  **在同一批位置上被问同一批问题**（配对可比）。推理为贪心 ⇒ seed 不构成方差源。
- **跨种子**：13.4M 档 seed {0,1} 各训一次，逐臂报两次读数的**极差与方向一致性**；
  方向不一致即如实推翻，不修图。
- **对照复用**：本实验**不复用历史 checkpoint 作主臂**（历史 checkpoint 全是
  `chain_mode="latent"`，没有 token 条件链头）⇒ 全部自包含双跑（SCHEMA §9 第 4 条）。

## 6. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 120 min`（含步时冒烟 + 4 臂训练 + 全部评测；含共享卡争用余量）。
**步时实测写在 `smoke/steptime.md`（先测后定，SCHEMA §17.8/§20.3，不借用别的实验的步时）。**

**预算不够时砍什么**（SCHEMA §17.7，优先级从高到低）：砍大档 seed 数（2→1，显式声明）>
砍评测 prompt 数（16→8）> 砍重复次数（墙上界）> 砍大档步数（保**臂间同刻度** §20.3，
并声明「Nk 快照口径、与 13.4M 档不可比绝对水平」）。**不砍判据线、不砍三臂、不砍 native 对照。**

## 7. 数据与脚本清单

| 项 | 值 |
|---|---|
| 数据 | `data/enwik8_full.npz`（字符级，V=206，val 段）；prompt = val 的固定 offset |
| 训练 | `python3 train_m1.py <arm> --config config_13m.json --steps 6000 --batch-size 32 --seed {0,1} --out-root train`（驱动 `train_arms.sh`） |
| 评测 | `python3 run.py --ckpts train/* --out .`（`spec_arms.py` 定义臂与指标） |
| resolved config | `results.json::resolved`（每个 checkpoint 一份） |
| 落盘 | `results.json` / `cycle_log.jsonl` / `run_log.txt` / `smoke/steptime.md` / `analysis.md` / `MODEL.md` / `controls/README.md` |
| 单测 | `test_spec_arms.py`（可失败单测；**放在本目录**，不写 `tests/`） |
| 变异 | `mutate.py`（复用 `tools/evalkit/mutation.py`，未抓到 ⇒ 非零退出） |
| 对账 | `reconcile_numbers.py`（复用 `tools/evalkit/reconcile.py`，期望值重算，禁 golden 兜底） |

## 8. 既有做法（prior art，SCHEMA §18）

1. **谁做过**：① **Medusa**（Cai et al. 2024）——在冻结主干上挂多个解码头，用**同一个
   forward 的批量验证**接受最长前缀；② **DeepSeek-V3 的 MTP**（2024）——「node-2 以
   node-1 的 token 为条件、re-embed 后再预测」（即本实验的 `chain` 臂形态）；
   ③ **EAGLE / EAGLE-2**、**Hydra**——草稿头与树状候选；④ **LayerSkip**——同模型浅出口草稿；
   ⑤ 经典无损 spec decoding（Leviathan 2023 / Chen 2023）——拒绝采样保分布。
   （数字类为 `记忆·未核`——本环境联网不可用；仅用作设计方向，不当判据。）
2. **为什么这么设计**：草稿要**便宜**、验证要**保真**。多 token 预测头是「用一次前向的
   宽度换多个 token」，条件链头是「用冻结表征 + 已出 token 换更准的第 2 个 token」。
   Medusa 反复强调「头本身近零成本、真正贵的是验证」，但**没有**把
   「直连头 vs 条件链头」当作**唯一变量**在同一次前向上对照过。
3. **它的实测结果**：**本仓已测的相邻结论**（`本地实测`）——
   - `ticket 12 P1`（6000 步 × 2 seed，`chain_mode=token` vs `latent`）：t+2 准确率
     direct 0.560、chain-**oracle**（真 t+1 条件）0.682、chain-**deploy**（node-1 argmax
     条件）0.547；spec 接受率 chain-deploy 0.7525/0.7506 vs direct 0.7729
     ⇒ **部署路径上条件链不赚**（差 1.3pp），oracle 与 deploy 差 0.135 = 「草稿质量」的标价。
     **但那一次没有 `combined` 臂、没有墙钟 CI、没有算力净账、也没有把 `T∘T` 排除写清。**
   - `ticket 13 / 20`（宽度-3 spec）：13M 墙钟 k=2 **1.68×**、100M 塌到 **1.15–1.29×**
     （前向 compute-bound）；接受级联不随规模提升。
4. **我们的增量**：
   a. **把「草稿路径构造」做成唯一变量的三臂对照**（chain / direct / **combined**），
      三臂共享同一 forward、同一验证器、同一 K；
   b. 用**生产解码器**（`decode_spec` 的 `draft_fn` 钩子）+ **逐位一致门禁**，把结论钉在
      可部署口径上；
   c. **墙钟 CI + 等效算力净账**双口径并报（把草稿头成本算进去）；
   d. 明确把 `T∘T` 潜在复用**排除**在本命题之外（复核 A6 的更正落地）；
   e. **两个规模点**（13.4M / 更大档），检验「臂序是否随规模变化」。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：先查 `cycle_log.jsonl` 的 `emitted == 1 + accepted`、`committed` 单调、
   每次重复的极差是否被邻居线污染（>20% ⇒ 重测）；墙钟必须在**同进程交替**下测。
2. **实现**：跑 `test_spec_arms.py`（臂分布映射、混合分布、逐位一致、账目恒等）；
   任一红 ⇒ 实现错，先修再判。再跑 `mutate.py` 确认单测真的能抓红。
3. **规模**：若 13.4M 有收益、大档没有，按 `ticket 20` 的 launch-bound/compute-bound
   解释记录——**不得**外推，**不得**用大档单 seed 下结论。
4. **设计**：若 `chain` 与 `direct` 接受率差 < 0.014 且墙钟不可排序，结论是「在本引擎的
   **部署（argmax 条件）口径**下，条件链相对直连头无差异」——这是设计层判读；
   若 `chain` 明显更差，与 ticket 12 的 deploy 结论一致，作为**独立复现**记录。
   **不得**写成「条件链不可能」或「概率组合无价值」——采样/树状/多级条件的组合未测。

## 10. scope 声明

本研究**只做单轴单格实验**（H3-composition 一格内的「多位置预测的组合方式」），
基底恒为 stock 对照；**组合默认留白**（组合留白是显式声明，不是遗漏）。
本线的「三臂」是**同一命题下的三档干预**，不是跨轴组合。`deps` 只记录已知耦合方向
（H3 ← H1 每步输出位置数、H5 验证机制、H4 头与主干关系）。组合取舍属具体产品方 scope。

## 11. 产出与复核

见 §7 的落盘清单。本实验完成后按 SCHEMA §14/§14b **另派 fresh-context 只读复核者**
（复核者须在本线全部产物落盘之后再启动，不得同波）。

---

## §23 七问自查（每格预算内极致）

| # | 问题 | 本实验的答案 |
|---|---|---|
| 1 | 既有做法查过吗？业界做透的写成 reference 了吗？ | 见 §8：Medusa / DeepSeek-MTP / EAGLE 已把**机制**做透（(a) 类，不复现）；但「直连 vs 条件链作为唯一变量的同 forward 对照 + 墙钟 CI + 算力净账」**没有**先例 ⇒ (b) 做实验。 |
| 2 | 判据是最严的那个吗？有诚实对手/毁源/因果/活性护栏吗？ | 诚实对手 = **同 checkpoint 原生贪心**（非 oracle，SCHEMA §11）；门禁 = 逐位一致；活性量 = 每轮接受账本落盘（`cycle_log.jsonl`）；因果 = 臂开关的 liveness 单测（三臂在同一 forward 上读到**不同**的草稿分布）。 |
| 3 | 三类「错的对照」都试过吗？ | 同参数量：三臂**零新增参数**（读的是同一模型的既有头）；**同算力**：这正是等效算力净账（`c₃/(tok/fwd)`）要回答的；同字节预算：不适用。 |
| 4 | 规模/预算的边界探过吗（≥两档）？ | 13.4M + 更大档两个点（见 §4.5）；**禁外推**。 |
| 5 | 反方向/否证路径走完了吗？ | 计划内：若 chain 不赚，按 §9.4 作「部署口径无差异/更差」的设计层判读，并列出采样/树状未测项。 |
| 6 | 未测项显式列成「下一步最小实验 + 成本」了吗？ | `analysis.md` §7 列出（多级条件链；树状候选；温度>0 的拒绝采样保分布；100M 档）；给成本。 |
| 7 | 复核通过了吗？ | 待独立复核（§14，新 run、落盘后启动）——**本文件不声称已复核**。 |
