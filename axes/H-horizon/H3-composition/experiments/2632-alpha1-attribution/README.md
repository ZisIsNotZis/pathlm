---
name: 2632-alpha1-attribution
ticket: 51-H3-alpha1-attribution
date: "2026-10-09"
grid: H3-composition
engine: "复用 2631 的 3 个 checkpoint（13m_s0 / 13m_s1 / 53m_s0）；纯推理诊断（CPU，逐 token Decoder.step + 生产 decode_spec）"
budget: "CPU 纯推理诊断；不取 GPU 锁（接受率是 argmax 确定量，与墙钟/占用无关）。冒烟后定 positions 数。"
---

# 2632-alpha1-attribution（Q1：第 N+2 草稿位接受率差的归因）

> **格归属**：`H3-composition`。这是对 `2631-spec-three-arms` 结论的一次**归因诊断**，
> 不引入新机制：复用 2631 已落盘的 3 个 checkpoint，只在**同一份权重**上改变读哪个头
> 去构造第 N+2 草稿位。目的是回答出资人的 Q1：`chain` 的 α₁（0.657）为何远低于
> `direct`（0.844）——是训练漂移，还是目标/条件的问题。

## 0. 术语与位置口径（本实验所有数字都用这套坐标）

设一次投机轮从「已提交前缀到位置 N（含）」出发：

```
v1 = argmax node1(h_N)          位置 N+1；**它就是验证器自己的输出** ⇒ 无接受测试（引擎性质）
d2 = 草稿                     位置 N+2；**α₁ 量的是它**（"第一个草稿位"）
d3 = 草稿                     位置 N+3；2631 里三臂固定为 node3
验证器目标 T2* = argmax node1(前缀 + v1 之后的隐状态)  ← 交付流里 N+2 的"真值"是模型自己
```

- `node1 = T1(h)`、`node2 = T2(h)`、`node3 = T3(h)`、`chain2 = T1(chain_cat([h, embed(cond)]))`
  （`pathlm/model.py::_mtp_nodes`）。
- **关键架构事实**：`chain2` 复用 `transforms[0]`（= node1 的 T1）；`node2` 用**独立的** T2。
  `chain2` 的条件 `cond` 在训练时 = 真 `t_{N+1}`（teacher forcing），部署/推理时 = 模型自己的
  `v1`（`model.py:339-345`，`deploy_chain` 或 `cond is None`）。

## 1. 命题（一句话、可证伪）

**2631 三臂的 α₁ 差（`direct` 0.844 > `chain` 0.657）不是三臂各训一个模型造成的训练漂移，
而是「同一个 checkpoint 上，第 N+2 草稿位由不同头/不同条件产出」的可分解效应；
其主因是 `chain2` 在部署时用了模型自己的 `v1` 当条件（而非训练时的真 `t_{N+1}`），
而不是「chain 目标把第一个头带坏了」。**

可证伪形式：
- 若三臂并非共用同一 checkpoint（存在臂级训练差异），或发出流不逐位相同 ⇒ **命题作废**（门禁 G1）。
- 若 `chain2` 在真条件下（oracle）仍显著劣于 `node2` ⇒ 主因是**头架构**（共享 T1 的 concat 头），
  而非部署条件；命题的归因部分被推翻。

## 2. 单因素声明

- **基底**：`pathlm` stock 宽度-MTP 解码器，`n_mtp=3`、`chain_mode="token"`、`transport="none"`，
  配置与 2631 逐字段一致（checkpoint 直接复用 2631 的 `train/*/model.pt`）。
- **唯一变量 = 从同一 `h_N`/同一 checkpoint 读出的「第 N+2 草稿」的来源 + 其条件**：
  `direct`=`T2(h)`；`chain_deploy`=`chain2(h, cond=v1)`；`chain_oracle`=`chain2(h, cond=真 t_{N+1})`；
  `combined`=`0.5·softmax(chain2_deploy)+0.5·softmax(node2)`。
  **零新增参数、零训练**（这本身就是「无训练漂移」的构造性证据）。
- **必要性例外**：无。本实验不引入非标准因素。
- **不用 oracle 当判据**：`chain_oracle` 只用于**归因**（拆出「条件错」的份额），
  不作部署优势结论（SCHEMA §11）。

## 3. 干预定义（实现 vs 标签）

- 复用 `2631-spec-three-arms/spec_arms.py` 的臂定义与 `pathlm/decode.py::decode_spec`（生产解码器）。
- 新增 `diag.py`：在同一 checkpoint 上，
  1. **free-gen 门禁**：native 走 `pathlm.decode.decode`（`Decoder.step()` 逐 token），三臂走
     `decode_spec`，断言发出流逐位相同（argmax 路线）；
  2. **teacher-anchored 诊断**：在 enwik8 val 的真实文本前缀上（逐 token `Decoder.step`），
     逐位置读出 `node1/node2/chain2(deploy)/chain2(oracle)`
     与两个验证器目标（`T2*_deploy` = 前缀含 `v1`；`T2*_teacher` = 前缀含真 `t_{N+1}`），
     并用 `combined` 的 λ=0.5 概率混合读 `d2`。
- **不写共享文件**：不改 `pathlm/**`、`tools/**`、`tests/**`、`catalog.yaml`、`axes/**/README.md`；
  只新增本目录内文件。

## 4. 判据线（预登记，事后不得改）

1. **门禁 G1（前提核实，缺一即命题作废）**
   - G1a：三臂命令都从**同一 `model.pt` 路径**加载（脚本落盘 ckpt 路径 + sha256）；
   - G1b：同 checkpoint 上，`v1` 与发出流与 native 贪心**逐位相同**（`decode` vs `decode_spec`）。
2. **主归因判据 D1（部署条件 vs 头架构）**——teacher-anchored 口径、同一前缀、同一 checkpoint：
   - `q_direct` = P(argmax node2(h_N) == 真 t_{N+2})
   - `q_chain_deploy` = P(argmax chain2(h_N, cond=v1) == 真 t_{N+2})
   - `q_chain_oracle` = P(argmax chain2(h_N, cond=真 t_{N+1}) == 真 t_{N+2})
   - 预登记判决（Δ 一律用 `q_chain_oracle − q_chain_deploy`，正 = oracle 更好）：
     - **(i) 部署条件为主因**：`q_chain_oracle − q_chain_deploy ≥ 0.05`（≥ 0.014 噪声带的 ~3.5×）
       且 `q_chain_oracle ≥ q_direct − 0.014`；
     - **(ii) 头架构为主因**：`q_chain_oracle ≤ q_direct − 0.014` 且该差 > 0.014；
     - **(iii) 两者都贡献**：不落 (i)/(ii) 时，两列并报，不下单一归因。
3. **分层判据 S1**：按 `e = 1[v1 == 真 t_{N+1}]` 分层报 `a_chain`、`q_chain_deploy`。
   - 若 `chain` 的亏空在 `e=0` 上明显更大（相对 direct 的亏空集中在 e=0）⇒ 部署条件是主因的旁证。
4. **接受率口径 A1**：free-gen（与 2631 同口径）复现 `direct ≈ combined > chain`，
   α₁ 用**按 prompt 聚类的 bootstrap**（B=2000, seed=2632）报 95% CI；臂差 < 0.014 不排序。
5. **Q2 树候选判据 T1（top-k recall 探针）**：`direct` 与 `chain_deploy` 各自对**部署目标 T2***
   的 top-k（k=1,2,4,8）召回，teacher-anchored 口径。
   - 预登记：若 `recall@k(chain) < recall@1(direct)` 对 k=2,4 均成立 ⇒ **同 K 的树候选不能翻转排序**；
     若存在 k 使 `recall@k(chain) ≥ recall@1(direct)` ⇒ **树可能翻转**，须做完整树实验才能定论。
6. **禁外推**：绝对水平只适用于「该 checkpoint / enwik8 val 的真实文本前缀或自生成前缀 / 该步数快照」；
   不读成「业界做不到」，不外推到温度>0、别的规模、别的任务。
7. **报告口径**：先人话，后数字；代号只作括号定位。

**方差界**：接受率类以**按 segment 聚类的 bootstrap**（B=2000, seed=2632）报 CI；Δ < 0.014 不排序（借 SCHEMA 方差界）。

## 5. 对照与方差

- **对照物 = 模型自身**（同 checkpoint 的 `node2`/native 贪心），不是 oracle（SCHEMA §11）。
- **同一前缀**：teacher-anchored 下四臂被问**同一批位置、同一前缀**（配对可比）。
- **跨点**：13m_s0 / 13m_s1 / 53m_s0 三个 checkpoint；方向不一致即如实报，不修图。
- **方差**：按 segment 聚类 bootstrap；确定性 argmax ⇒ 无采样方差。

## 6. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 0 min`（**纯 CPU 推理诊断**，不取 GPU 锁）。
**砍什么**（不砍判据线）：先砍每 checkpoint 的位置数（冒烟实测步时后再定），再砍 free-gen 的 prompt 数。

## 7. 数据与脚本清单

| 项 | 值 |
|---|---|
| 数据 | `data/enwik8_full.npz` 的 val 段（真实文本前缀 + free-gen prompt） |
| checkpoint | `.../2631-spec-three-arms/train/{13m_s0,13m_s1,53m_s0}/model.pt`（复用，不改） |
| 入口 | `python3 diag.py --ckpts <dir...> --out . [--smoke]` |
| resolved config | `results.json::resolved` |
| 落盘 | `results.json` / `run_log.txt` / `diag_records.jsonl` / `analysis.md` / `MODEL.md` / `test_diag.py` |

## 8. 既有做法（prior art，SCHEMA §18）

1. **谁做过**：Medusa（Cai 2024，并行头 + 树验证）；DeepSeek-V3 MTP（token 条件链头）；
   Leviathan 2023 / Chen 2023（spec 无损定理）。本环境联网不可用 ⇒ 均为 `记忆·未核`。
   **本地近邻**：`2631-spec-three-arms`（三臂，唯一变量 = N+2 来源）；ticket 12
   `2512-overnight-causal-chain`（**批量 teacher-forcing** 口径报过 chain-oracle 0.682 /
   chain-deploy 0.547 / direct 0.560）。
2. **为什么这么设计**：草稿头「便宜 + 准」；条件链头声称用已出 token 换更准的第 2 个 token。
3. **实测结果**：ticket 12 的 oracle-vs-deploy 差已提示「条件错是瓶颈」，但那是**批量前向**口径，
   且没分层、没在**生产解码器**上复现。
4. **我们的增量**：在**同一 checkpoint + 生产解码器口径**上，把 α₁ 差**分解**为
   「部署条件错」与「头架构」两列，并给出**树候选是否会翻转排序**的 top-k 召回探针。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：先查 free-gen 门禁（`decode` vs `decode_spec` 逐位）；不一致 ⇒ 记首个分歧位置，先修测量。
2. **实现**：`test_diag.py`（条件接线、混合、top-k 召回、T2* 取位）；任一红 ⇒ 先修。
3. **规模**：两个 13M + 一个 53M；若方向随规模变，如实报（禁外推）。
4. **设计**：若 `q_chain_oracle` 仍显著低于 `q_direct` ⇒ 记「共享 T1 的 concat 头本身更差」，
   并列出「无 chain 损失的消融训练」为下一步最小实验。

## 10. scope 声明

本研究**只做单轴单格**（H3-composition 内的「多位置预测的组合方式」），基底恒为 stock 对照；
组合默认留白。`deps` 只记录耦合方向。**不引入新机制、不训练、不写共享文件。**

## 11. 产出与复核

见 §7。完成后按 SCHEMA §14/§14b **另派 fresh-context 只读复核者**（本线全部产物落盘后再启动）。

---

## §23 七问自查

| # | 问题 | 本实验的答案 |
|---|---|---|
| 1 | 既有做法查过吗 | §8：Medusa/DeepSeek-MTP/ticket12；本诊断是 ticket12 的**生产口径 + 分层 + 分解**升级 |
| 2 | 判据最严吗 | 诚实对手 = 同 checkpoint 的 node2/native；门禁 = 逐位等价；归因 = oracle 只作分解，不当判据 |
| 3 | 三类错对照试过吗 | 同参数量：四臂零新增参数；同算力：同一 forward；同字节：不适用 |
| 4 | 规模边界探过吗 | 两个 13M seed + 一个 53M（§5）；禁外推 |
| 5 | 否定路径走完吗 | §9.4：oracle 仍差 ⇒ 归因头架构，并列下一步消融 |
| 6 | 未测项列成本吗 | `analysis.md` §未测项：完整树验证、温度>0、无 chain 损失消融、100M |
| 7 | 复核通过吗 | 待独立复核（§14b）——本文件不声称已复核 |
