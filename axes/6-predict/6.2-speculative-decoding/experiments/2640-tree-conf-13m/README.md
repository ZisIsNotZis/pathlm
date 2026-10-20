---
name: 2640-tree-conf-13m
ticket: 63-axis6-predict / WP 6.2.2
date: "2026-10-13"
grid: 6.2-speculative-decoding
engine: "width-MTP（pathlm/model.py + pathlm/decode.py；13.4M d=256/L=12 + 53.9M d=512/L=12，n_mtp=3，chain_mode=token，window=0，anchors=0，transport=none）"
budget: "GPU ≤60 min（硬时间盒）；先跑通 13m_s0 全部臂并落盘 results.json，再扩 13m_s1 / 53m_s0"
---

# 2640-tree-conf-13m · 概率多候选 / 树 + 路径概率加权组合（WP 6.2.2）

> **本文件是预登记契约，写在跑之前**：判据、成瘾口径、成本口径、否证条件在此写死，
> 事后**不得**调参。实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它补的是什么**:`axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/`
> 把出资人设计的「**每个头自带概率的多候选 / 树 + 路径概率加权组合**」简化成了
> 「单 argmax + 固定 λ=0.5 概率混合」,树/多候选**整条没做**(复核问题清单
> `.scratch/55-review/H-PROBLEMS.md` F2 / F5；`2632-alpha1-attribution` §6.1 的
> `recall@k(chain) ≥ recall@1(direct)` 预登记判据实测**已触发**)。本实验把这部分**补回来并端到端测**。
> 臂定义与指标 `spec_tree.py`；评测驱动 `run.py`；本地单测 `test_spec_tree.py`；
> 变异验证 `mutate.py` / `mutation_report.json`；原始读数 `results.json` +
> 逐轮活性量 `cycle_log.jsonl`；判读 `analysis.md`；对照物清单 `controls/README.md`。

## 1. 命题（一句话、可证伪）

**在同一 checkpoint、同一 τ 计法下，把「第 N+2 个 token」的草稿从「单个 argmax」换成
「自带概率的多候选 / 树（验证时全部候选送进同一次前向）」能否把平均接受长度 τ 抬起来、
把每 token 顺序前向次数压下去，且质量（发出流）不变？**

可证伪形式：*树臂的 α₁ 相对**最强的单 argmax 臂**提高 > 0.014（SCHEMA 噪声带）且按 prompt
聚类的配对 bootstrap 95% CI 不含 0 ⇒ 树有效；否则判「本尺度下无效」并如实否证。*

## 2. 位置口径与单因素声明

设已提交前缀到位置 N（含）。一次投机轮：

```
v1 = argmax node1(h_N)    位置 N+1；**验证器自己的输出** ⇒ 不设接受测试（引擎性质）
d2 = 草稿                 位置 N+2；**本实验唯一变量 = 它的来源**
d3 = argmax node3(h_N)    位置 N+3；**全部臂固定**（与 2631 共享背景，保证单因素）
验证：inputs = [v1, d2, d3] 一次前向；接受最长前缀；发出 1 + a 个 token
```

头：`node1=T1(h)`、`node2=T2(h)`、`node3=T3(h)`、`chain2=T1(chain_cat([h, embed(v1)]))`
（`pathlm/model.py::_mtp_nodes`，`chain_mode="token"`，推理走 deploy 分支）。

**基底与训练**：复用 `2631-spec-three-arms/train/{13m_s0,13m_s1,53m_s0}` 的**同一份**已训练
checkpoint（`*.pt` 不入库，只在主工作树；本实验**不训练**）。三个「臂」只在**解码时读哪个头 /
送几个候选**上不同，权重完全相同。

**唯一变量**：位置 N+2 的草稿来源（单候选时的**分布**、树臂的**候选集**）。
其余逐字段一致：prompt 集、生成长度、批量（B=1）、温度 0、γ=2、验证宽度、缓存语义。

## 3. 臂定义（预登记，事后不得改）

| 臂 | 位置 N+2 的草稿 |
|---|---|
| `native` | 无（对照 = `pathlm.decode.decode` 逐 token 贪心，走 `Decoder.step`） |
| `direct1` | `argmax node2`（直连头，旧路线） |
| `chain1` | `argmax chain2`（+1+1 条件链，deploy 分支） |
| `combined_fixed` | `argmax (0.5·softmax(chain2) + 0.5·softmax(node2))`（**旧 2631 臂**，固定 λ=0.5） |
| `combo_pw` | **路径概率加权**：`w_A = softmax(node1)[v1]·softmax(chain2)[a]`（a = argmax chain2）、`w_B = softmax(node2)[b]`（b = argmax node2）、`q = (w_A·softmax(chain2) + w_B·softmax(node2))/(w_A+w_B)`，草稿 = `argmax q` |
| `tree_union2` | 候选集 = `{argmax chain2, argmax node2}`（去重，1–2 个），**全部候选送进同一次前向验证** |
| `tree_direct_k2` | 候选集 = `top-2 softmax(node2)`（axis-lead 追加；与 `2632` 探针 `direct@2` 交叉印证用） |
| `tree_qpw_k2` | 候选集 = `top-2 softmax(q)`（与 `tree_direct_k2` **同 K、同 FLOPs**，唯一差别 = 候选排序分布） |
| `tree_topk4` | 候选集 = `top-4 softmax(q)`（自带概率的多候选，K=4） |
| `oracle_accept` | 候选 = **目标真值 argmax**（从 native 贪心流取，第 L+1 位；L = 已提交 token 数）⇒ **只作 τ 上界附报，不作判据**（SCHEMA §11） |

`combo_pw` 的权重按**原式**（含 `p1[v1]` 因子）计算；`p1[v1]` 是 `w_A` 独有因子（`w_B` 没有），
因此**不能**约掉。逐轮落盘 `w_A`/`w_B`/`w_A`（去掉 `p1[v1]` 的对照口径）/两份草稿分歧率，
使「两臂到底是不是同一个臂」可核。

**多候选/树验证（实现要求，必须精确无损）**：候选 K 个 ⇒ 一次 **B=K 行的批量前向**
（共享前缀 KV 广播到 K 行；每行 = `[v1, c_i, d3]`；内部因果掩码；`window=0` ⇒ 行内因果 +
全部前缀 key）。接受判据 = **存在候选 == 目标模型在该位置（同一真实前缀下）的 argmax**；
随后**只把接受行的 K/V 与位置写回缓存**（`dec.kv` / `dec.layer_pos` / `dec.n` / `dec.last_nodes`，
位置切片复用 `pathlm.decode._slice_nodes`）。每轮**顺序前向次数恒为 1**（不得退化成
「每个候选各跑一次」）。该前向在**本实验目录内**实现，复用 `model.blocks`/`model.embed`/
`model.pos_embed`，**不改 `pathlm/**`**。
单候选臂走**生产解码器** `pathlm.decode.decode_spec`（其 `draft_fn` 钩子接入本实验分布）——
与 2631 同一条代码路径。

## 4. 判据（预登记；逐条给结论）

- **P1（该涨）**：`tree_union2` / `tree_topk4` 的 α₁ 相对**最强的单 argmax 臂**
  （`max over direct1/chain1/combined_fixed/combo_pw`，由同一批 prompt 实测决定）提高
  **> 0.014**，且按 prompt 聚类的**配对** bootstrap 95% CI 不含 0。
- **P2（该涨）**：τ 与 tok/fwd（=1+τ）相对同一最强单臂上升，配对 95% CI 不含 0；报 CI 与样本量。
- **P3（出资人的加权假设）**：`combo_pw` 的 α₁ 比 `combined_fixed` 高 **> 0.014** 且配对 CI
  不含 0 ⇒ 加权成立；否则**如实否证**「固定 λ 等权 vs 路径概率加权在本尺度有差」，
  并用**逐轮草稿分歧率**作为「两臂是否真不同」的活性量。
- **P4（必须不变）**：全部臂 × checkpoint × prompt 的**发出流逐位 == `native` 贪心流**。
  13M 必须全过；53.9M 若出现 fp 平局翻转，**显式记录**首个分歧位置与规模，并记
  「逐位相等不是跨规模判据」（不得隐瞒，不得据此改判 P1）。
- **C1（实现有效性交叉印证，axis-lead 追加；先过 C1 才谈 P1）**：`tree_direct_k2` 的 α₁
  必须与该 checkpoint 上**独立测量**的探针 recall@2 **同量级**
  （`2632-alpha1-attribution/probes_output.json::P3_free_gen_topk_vs_T2["2"]["direct"]`：
  13m_s0 0.8605 / 13m_s1 0.8922 / 53m_s0 0.9260；同一部署口径：模型自生成前缀、`step_multi` 验证）。
  探针用 4 prompt（n≈311–337 轮），本臂用 12 prompt（n≈900+ 轮），且逐轮位置采样不同 ⇒
  只要求**落在同一量级且 CI 覆盖探针点**（用本臂 α₁ 的 95% CI 与探针点比较）。
  若实测**显著低于**探针 recall@2（CI 不含探针点且低于）⇒ 判**树实现（多行验证 / 接受判据 /
  缓存回写）有 bug**，必须先修再报结论，**不得**把该读数当科学结论。
- **P5（代价轴，如实报，不作判据）**：每 token 等效 FLOPs `R = B_verified·(1+γ)/(1+τ)`
  （树臂 `B_verified` = 每轮实际候选行数 K（1–4），单候选臂 = 1），并**明确声明 `R>1` 是
  投机解码范式的定义性质（用 FLOPs 换顺序前向次数）、不可证伪、不是「亏损发现」**；
  附解析的 MTP 头 MAC 占比（照抄 `2631-spec-three-arms/spec_arms.py::mtp_head_macs`）。另并报树臂 vs 单臂的
  `R` 对比（树用 K 倍 FLOPs 换更高 τ 是否值），**不得**写成硬判据。

**否证条件（预登记）**：

1. 若**没有任何**配置把 α₁ 抬到最强单臂 + 0.014 以上（CI 不重叠）⇒ 判
   「概率多候选 / 树 + 置信加权组合在**本尺度**（13M 冻结主干、该 checkpoint）不成立」。
2. 若**只有** `oracle_accept` 抬得动 ⇒ 判「约束在**草稿头的候选集覆盖**（结构问题，
   需要新的多候选条件头）」，并把**最小 patch 描述**写进日志、上报 axis-lead，
   **不自行改 `pathlm/**`**。
3. 若 `combo_pw` 与 `combined_fixed` 逐轮分歧率 ≈ 0 且 α₁ 差 ≤ 0.014 ⇒ 判
   「该尺度下『加权 vs 等权』无信息」，不宣称任何一方更优。
4. 若 C1 未过（`tree_direct_k2` 显著低于探针 recall@2）⇒ 判**实现 bug**，
   修好重跑后再谈 P1–P3；修不好则如实报「树读数不可信、未定论」。

## 5. 口径纪律

- **禁墙钟判据**：本实验**不测墙钟**。τ、α、tok/fwd、R 全为解析量。
  `results.json` 里**不得**出现墙钟字段，`analysis.md` 判定表里不得出现墙钟数字。
- **方差**：按 **prompt 聚类**的 bootstrap（B=2000，seed=2640，percentile 95%）；
  **不把同一序列的多轮草稿当独立二项样本**。臂间差用**配对**（同一 prompt 重采样索引）
  bootstrap。接受率可排序带 **Δ ≥ 0.014**。
- **样本量**：每 checkpoint 12 prompts（enwik8 val，stride 5000，`PROMPT_LEN=32`）×
  `n_new=200` token；主判 `13m_s0` + `13m_s1`（n=2）；`53m_s0` 时间允许才算，
  未测则如实记「未测」，不得外推。
- **时间盒内的臂优先级**：`13m_s0` 必须先跑满 `{native, direct1, chain1, combined_fixed,
  combo_pw, tree_union2, tree_direct_k2, tree_qpw_k2, oracle_accept}` 这 9 臂并落盘；
  再扩 `tree_topk4` / `13m_s1`；`53m_s0` 可只跑 `{native, direct1, tree_direct_k2}` 作规模点。
- **规模与成本**：`R` 为解析量；MTP 头 MAC 占比照抄 2631（头成本与臂无关 ⇒ 臂间相消）。
- **禁外推**：绝对水平只适用于「该 checkpoint / 该 prompt 集合 / 该训练步数快照 /
  enwik8 风格文本 / 温度 0 / γ=2 / 未取锁外的外部占用水位」。不得读成
  「业界做不到 / 做得到」，不得外推到别的任务、规模或温度。
- **GPU**：按 `harness/SCHEMA.md` §20 取锁（`/tmp/pathlm_gpu.lock`，mkdir 法 + holder_pid/heartbeat
  + trap 释放，最多等 10 分钟；等不到就上报，不杀别人的进程）。**锁粒度 = 每 checkpoint**：
  每个 checkpoint 取锁 → 跑完 → 释放 → 留 ≥60 秒窗口再取下一把（整程 <10 min 才允许整程持锁；
  本程 ≥3 个 checkpoint、按臂粒度释放会耗掉 ~27 min 纯 sleep 窗口，超出 60 min 硬盒 ⇒
  粒度选 checkpoint 并在此显式记账）。纯 CPU 的写码/单测**不取锁**。
- **单元测试与变异**：`python3 -m pytest test_spec_tree.py -q`（CPU，无需锁）；
  变异用 `tools/evalkit/mutation.py`，**只有 `test_failure` 算抓红**，且每条变异必须
  **逐字节还原**（sha256 前后相等）。

## 6. 必要自检（写进单测）

1. **缓存正确性**：「接受行写回的缓存」与「从头重算同一前缀」的下一步 logits 一致
   （报 float32 实际最大差）。
2. **树的前向次数**：每轮 1 次（`n_forward == n_rounds`）。
3. **单候选退化**：候选集只剩 1 个 ⇒ 与对应单臂**同值**（同 prompt、同 token 流、同 accepts）。
4. **无损性门禁**：`test_failure` 级 —— 接受判据改成「无条件接受」必须被抓红。
5. **臂接线**：`direct1`/`chain1`/`combined_fixed` 的首草稿必须与 2631
   `spec_arms.second_slot_logits` **一致**（复用口径而不是重造）。
6. **K 等价性**：`tree_direct_k2` 的首候选 == `direct1` 的草稿、`tree_qpw_k2`/`tree_topk4` 的
   首候选 == `combo_pw` 的草稿（top-k 的首项 = argmax）。

## 7. 跑法

```bash
# 单元测试（CPU，不取锁）
python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/test_spec_tree.py -q
# 冒烟（GPU，取锁；结果落 /tmp，不入库）
python3 run.py --smoke --out /tmp/wp622-smoke
# 正式（GPU，按 checkpoint 取放锁；写出 results.json + cycle_log.jsonl）
python3 run.py --out . \
  --ckpts ../2631-spec-three-arms/train/13m_s0 ../2631-spec-three-arms/train/13m_s1 \
          ../2631-spec-three-arms/train/53m_s0
# 变异验证（CPU）
python3 mutate.py
```
