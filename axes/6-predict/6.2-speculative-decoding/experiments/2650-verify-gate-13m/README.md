---
name: 2650-verify-gate-13m
ticket: 63-axis6-predict / Phase B · WP 6.2.3
date: "2026-10-14"
grid: 6.2-speculative-decoding
engine: "width-MTP（pathlm/model.py + pathlm/decode.py；13.4M d=256/L=12，n_mtp=3，chain_mode=token，window=0/anchors=0，transport=none）"
budget: "GPU ≤15 min（硬时间盒）；只跑 13m_s0 全臂 + 13m_s1 子集；53m_s0 只在时间允许时跑 1 个臂"
---

# 2650-verify-gate-13m · 批量验证的正确性门禁、失败模式与验证开销（WP 6.2.3）

> **本文件是预登记契约，写在跑之前**：判据、臂定义、成本口径、否证条件在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它补的是什么**：`2640-tree-conf-13m`（WP 6.2.2）已证「多候选 / 树抬 τ 成立、发出流逐位无损」，
> 但只在 **合法候选集**（K≤4）上验过，且 `spec_tree.tree_forward` 对 `window>0` **直接抛
> `NotImplementedError`**（带驱逐的树验证未测）。本实验回答 WP 6.2.3 的两个问题：
> ① **更激进 / 对抗性的候选集**下，批量验证的**正确性门禁**是否仍然无损、失败模式有哪些；
> ② **验证开销是否吞掉 τ 收益**（等效理论量口径）。

## 1 命题（可证伪）

**在同一 checkpoint、同一 τ 计法下，把送进同一次批量验证前向的候选集换成「更激进（K=8 / 跨头联合）」
或「对抗性（随机 / 分布尾部 / 重复）」的集合时，接受门禁是否仍然只接受「等于目标模型该位置 argmax」
的候选（⇒ 发出流仍逐位等于 native 贪心），以及 K 从 4 加到 8 时 τ 的边际收益是否已耗尽。**

可证伪形式：
*（i）对抗臂若拿到显著高于先验假接受率的 α₁，或发出流偏离 native ⇒ 判门禁有漏洞；
（ii）K 从 4 加到 8 的 Δτ/ΔK 若 ≥ 0.5 ×（K=1→2 的 Δτ/ΔK）⇒ 判「K>4 仍有价值」，否则判饱和。*

## 2 位置口径与单因素声明（与 2640 逐字段一致）

已提交前缀到位置 N（含）。一次投机轮：

```
v1 = argmax node1(h_N)    位置 N+1；**验证器自己的输出** ⇒ 不设接受测试（引擎性质）
d2 = 候选集 C            位置 N+2；**本实验唯一变量 = C**
d3 = argmax node3(h_N)    位置 N+3；**全部臂固定**
验证：rows = [[v1, c_i, d3] for c_i in C] 一次前向（B=|C| 行）；接受判据 = 存在 c_i == 该行索引 0 的 argmax
```

- **唯一变量** = 位置 N+2 的**候选集 C**（本实验含对抗性 C）。其余逐字段固定：同一批 checkpoint、
  同一 prompt 集（enwik8 val、stride 5000、`PROMPT_LEN=32`、12 段 × `n_new=200`）、B=1、温度 0、γ=2。
- **不重跑 2640 的臂**（省 GPU）：`native`/`direct1`/`chain1`/`combined_fixed`/`combo_pw`/`tree_*`/
  `oracle_accept` 的读数**直接引用** `2640-tree-conf-13m/results.json`（指针，不复制数字）。
  本实验**只新增**下面 §3 的臂，并对基线臂做**同 harness 复现**（V0）。

## 3 臂定义（预登记，事后不得改）

| 臂 | 位置 N+2 的候选集 C | 预期 |
|---|---|---|
| `repro_topk4` | = 2640 `tree_topk4` 的定义（`pw_mix.q` top-4） | **harness 复现门**：逐轮 accepts 必须与 2640 的 `tree_topk4` 逐位相同 |
| `repro_combo_pw` | = 2640 `combo_pw` 的定义（单候选，走生产 `decode_spec`） | 同上（对 2640 `combo_pw`） |
| `single_k1_ctl` | = `combo_pw` 的草稿，1 个候选，**走本实验的 K 行树机制** | **退化等价门**：逐轮 accepts 必须与 `combo_pw` 完全一致（K=1 时树机制 == 单候选） |
| `dup_k4` | = `combo_pw` 的草稿复制 4 份（**去重前**） | **去重/重复门**：与 `combo_pw` 逐轮一致（重复候选不得改变接受语义） |
| `aggr_topk8` | `pw_mix.q` top-8 | **该涨（饱和判据）**：α₁ ≥ `tree_topk4`；Δτ/ΔK 用于饱和判定 |
| `aggr_union_heads` | `dedup{argmax pc, argmax p2, top-2 pc, top-2 p2}`（跨头联合，K=3–6） | 该涨（弱于 topk8 也可）；测跨头候选共存 |
| `adv_random_k4` | 4 个**均匀随机** token（固定 seed，与任何分布无关） | **负对照（该跌）**：α₁ ≈ 4/206 ≈ 0.019，Wilson 95% 上界 < 0.06 |
| `adv_bottomk4` | `pw_mix.q` 概率**最低**的 4 个 token | **负对照（该跌）**：α₁ ≤ 0.02 |
| `win64_tree_k2`（可选，仅时间允许） | = `tree_direct_k2` 候选集，但 `window=64`（带驱逐） | **失败模式**：与 `native(window=64)` 逐位一致才有结论；否则判「带驱逐的树验证未实装」并登记 |

**对抗臂的语义**：接受判据不变（候选必须等于目标模型该位置 argmax）。因此**理论上对抗臂仍逐位无损**，
只是 α₁ 极低——这正是门禁是否「真验证」的判别量：若对抗臂 α₁ 明显高于假接受率，或发出流偏离
native，则门禁有漏洞（真缺陷，先修再报）。

## 4 判据（预登记；逐条给结论）

- **V0（harness 保真，硬门）**：`repro_topk4` / `repro_combo_pw` 的逐轮 `accepts` 与
  2640 `results.json` 对应臂**逐位相同**（同一 ckpt × prompt × 轮序）。不同 ⇒ 本实验 harness
  与 2640 不等价，**新读数不可信**，先修再报。
- **V1（必须不变，硬门）**：**全部臂 × 全部已跑 ckpt × 全部 prompt** 的发出流**逐位 == native 贪心流**
  （含对抗臂）；`n_forward == n_rounds` 逐臂成立（每轮恰 1 次顺序前向）。
  13M 是逐位相等硬门禁（2640 的 P4 口径）。若出现分歧，显式记录首个分歧位置 + 规模。
- **V2（负对照该跌 / 门禁灵敏度）**：`adv_random_k4` 的 α₁ 落在 [0, 0.06)（Wilson 95% 上界 < 0.06，
  与均匀随机假接受率 4/206≈0.019 同量级）；`adv_bottomk4` 的 α₁ ≤ 0.02。
  **任一对抗臂 α₁ ≥ 0.05 或发出流偏离 native ⇒ 判「接受判据不是真验证」**（blocker 级，先修）。
- **V3（退化等价）**：`single_k1_ctl` 与 `dup_k4` 的逐轮 `accepts` 与 `combo_pw`（2640）**逐个相同**，
  α₁/τ 逐位相等。不同 ⇒ K 行机制的 bookkeeping（去重 / 行↔候选映射 / 缓存回写）有 bug。
- **V4（饱和判据）**：`aggr_topk8` 的 α₁ 相对 `tree_topk4`（同 ckpt）的差与按 prompt 聚类的配对
  bootstrap 95% CI；若 CI 含 0 或 |Δα₁| ≤ 0.014 ⇒ 判「K=4→8 饱和」（τ 收益已被候选集覆盖率限制，
  不是验证行数限制）。`aggr_union_heads` 同判。
- **V5（代价轴台账，如实报、不作判据）**：按 K 列出台账——
  每轮主干位置数 `3K`（本实现「K 行平铺」）/ `2K+1`（节点共享理想树，未实装）；
  `R_K = K·(1+γ)/(1+τ_K) = 3K/(1+τ_K)`（等效 FLOPs/每 token = 每轮 3K 个主干位置 ÷ 每轮发出
  `1+τ_K` 个 token，与 2640 已落盘口径 `R = B_verified·(1+γ)/(1+τ)` 逐字一致）；`τ_K / K`（每行验证换来的 token）；
  `τ_K / τ_oracle`（对 τ 上界 2640 `oracle_accept` 的饱和比例）；
  边际 `Δτ/ΔK` 在区间 K=1→2 / 2→4 / 4→8 上逐段给值。
  **「验证开销是否吞掉 τ 收益」的预登记答案形式**：在**硬指标**（每 token 顺序前向次数 = 1/(1+τ)）上，
  验证只增加**并行行**、不增加顺序前向 ⇒ 不吞；在 **FLOPs 轴**上，若 K=4→8 的 `Δτ/ΔK` <
  0.5 ×（K=1→2 的 `Δτ/ΔK`）⇒ 判「K>4 的验证开销不划算（边际收益减半）」，否则判「K>4 仍值，按代价如实报」。
- **V6（失败模式目录，机器可核）**：逐条列出并各给一条可核证据：
  (a) 候选重复/去重（`dup_k4` + 单测）；(b) 对抗候选不得假接受（`adv_*` + V2）；
  (c) 行↔候选映射错位（变异抓红）；(d) 接受行缓存回写错行（变异抓红）；
  (e) 接受判据放宽为「无条件接受」（变异抓红）；
  (f) K 与 `n_mtp=3` 的上限（位置 N+3 的候选需二级条件头 ⇒ **本实验不做**，登记为引擎改动需求）；
  (g) `window>0` 的带驱逐树验证（`spec_tree.tree_forward` 现抛 `NotImplementedError`）——
  跑了 `win64_tree_k2` 则给读数，没跑则**显式记「未测」+ 下一步最小实验与成本**。

**否证条件（预登记）**：

1. 对抗臂 α₁ ≥ 0.05 或发出流偏离 native ⇒ 判门禁有漏洞，读数不可信，先修再报。
2. `single_k1_ctl` / `dup_k4` 与 `combo_pw` 不逐位一致 ⇒ 判 K 行机制有 bookkeeping bug。
3. `repro_*` 与 2640 不逐位一致 ⇒ 判新 harness 不等价（新读数作废）。
4. `aggr_topk8` 的 α₁ 显著**低于** `tree_topk4`（配对 CI 不含 0 且为负）⇒ 判「更激进候选集伤害验证路径」，
   登记 bug 线索，不得当「饱和」结论。

## 5 口径纪律

- **禁墙钟判据**：`results.json` **不得**出现墙钟字段；τ / α / tok-fwd / R / 边际量全为解析量。
  GPU 实耗只允许出现在 `run_log.txt`（过程日志）。`results.json::criteria.V5_cost_ledger` 是
  **离线派生字段**（由 `recompute_v5.py` 从落盘的 τ/α₁ 重算回写，见 `results.json::v5_recomputed_note`），
  不是 run 直接产出；τ/α₁ 的逐 prompt 原始读数未经回写。
- **方差**：按 **prompt 聚类**的 bootstrap（B=2000、seed=2650、percentile 95%）；臂间差用**配对**
  （同一 prompt 重采样索引）；率类指标的区间另给 **Wilson 95% CI**（负对照用）。
  **不**跨量纲借用 0.014（C1 裁决）；0.014 只在 V4 里作参考量级。
- **样本量**：13m_s0 全臂（12 prompt × 200 token）；13m_s1 子集 =
  `{repro_topk4, repro_combo_pw, single_k1_ctl, dup_k4, aggr_topk8, adv_random_k4}`；
  53m_s0 只在时间允许时跑 `adv_random_k4` 一个臂（规模点，未跑则如实记「未测」）。
- **实现边界**：本实验**不修改** `2640-tree-conf-13m/**` 的任何文件（只读 import；
  把被 import 文件的 sha256 记进 `results.json`）；**不修改** `pathlm/**`；
  新臂在本目录内实现（薄扩展 + 对抗候选构造函数）。
- **禁外推**：绝对水平只适用于该 checkpoint / 该 prompt 集 / 温度 0 / γ=2 / enwik8 char 文本。
- **GPU**：按 `harness/SCHEMA.md` §20 取锁（`/tmp/pathlm_gpu.lock`，mkdir 法 + holder_pid/heartbeat + `trap` 释放，
  最多等 10 分钟；等不到就上报，不杀别人的进程）。**锁粒度 = 每 checkpoint**：取锁 → 跑完 → 释放 →
  留 ≥60 s 窗口再取下一把。纯 CPU 的写码/单测**不取锁**。

## 6 必要自检（写进单测，须能失败）

1. **V0 复现门**：`repro_topk4` / `repro_combo_pw` 的 accepts 与 2640 落盘逐位相同（离线断言，CPU 可跑）。
2. **对抗候选构造**：`adv_random_k4` 的候选在固定 seed 下可复现、且与 `pw_mix.q` 的 top-4 不相同；
   `adv_bottomk4` 的候选 ∈ 概率最低 4 个（机器可核）。
3. **去重语义**：`dup_k4` 的候选列表 = 同一 token × 4；接受判据在该输入下与单候选同值。
4. **变异（须抓红且逐字节还原）**：① 行↔候选映射错位；② 接受行缓存回写错行；
   ③ 接受判据放宽为无条件接受。三条都须 `test_failure` 级抓红 + sha256 前后相等。
5. **禁墙钟**：`results.json` 无墙钟字段的机器可核断言。

## 7 跑法

```bash
# 单元测试（CPU，不取锁）
python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/test_verify_gate.py -q
# 冒烟（GPU，取锁；结果落 /tmp，不入库）
python3 run.py --smoke --out /tmp/wp623-smoke
# 正式（GPU，按 checkpoint 取放锁；写 results.json + cycle_log.jsonl）
python3 run.py --out . --ckpts ../2631-spec-three-arms/train/13m_s0 ../2631-spec-three-arms/train/13m_s1
# 变异验证（CPU）
python3 mutate.py
```

## 8 契约更正记录

V5 的 `R_K` 口径曾有一处笔误，经过与处置（含 worker 在开跑前的并列上报与定稿口径）见
`.scratch/63-axis6-predict/TICKET.md` §B4.1；本文件 §4 即定稿口径。
