# 2676 · 同可见键数下「按内容选」对「按位置选」（1.2.3 的 WP2 最小判决）· 13M · depth-ar/dense

> 本文件是**预登记契约**：判据线在下节写死，跑之前不得修改；读数落 `results.json`，
> 人话判读落 `analysis.md`。归属 `axes/1-time/1.2.3-content-topk` 的 `work_packages[WP2]`。
> 判据带的来源是 `harness/methodology.md` §2（0.014 bpc 带 **仅限配对 Δ、≥5 窗口**）。

## 1 要回答的判决问题

旧读数（`2614-topk-sparse-attn-13m`）只把「内容 top-k」对「全因果」比，得到的是**「少看键」的净代价**，
它把两件事混在一起：少看键本身掉多少质量，与「按内容挑」相对「按最近的位置挑」多赚多少。
本实验只拆后者：**每行可见键数逐行相等**的两臂，唯一差别 = 这些键**按什么规则挑**。

一句话可证伪的形式：*在 13M / T=256 / 12k 步 / matched-step 快照下，`topk32` 的 clean bpc 相对
`win32`（同 k、按最近位置）**越 0.014 配对带更差**（Δ ≤ −0.014）⇒ 内容选择这条省算力路线在本尺度
不成立；|Δ| < 0.014 ⇒ 内容信号相对位置窗口无质量增量；Δ ≥ +0.014 ⇒ 内容选择占优。*

## 2 单因素声明

**唯一变量 = 可见键的选择规则**（按内容相似度 vs 按位置最近）。七臂都是同一 stock dense
decoder-only 骨架（`train_depth_ar.py --dense --topology sequential`，无 carry 门、无退出、
仅最终深度 CE、tied 读出），同一配方逐字段一致：d=256 / L=8 / h=8 / T=256 / mlp_mult=4 /
dropout=0 / norm_cap=1 / bs=32 / lr=1e-3 / 12k 步 / 98.3M token（enwik8 前 90M 训练、
末 10M 评测、vocab 206）；`seed` 是本实验唯一按臂变化的配方字段（主判对 seed 0，升格对 seed 1）。

| 臂 | 来源 | 选择规则 | 每行可见键数 | seed | 参数量 |
|---|---|---|---|---|---|
| `full` | 复用 `2614-topk-sparse-attn-13m/runs/full-12k-s0` | 全部因果（基线） | `i+1`（≤256） | 0 | 6,962,688 |
| `topk32` | 复用同上 `runs/topk32-12k-s0` | 因果内按 `q·kᵀ` 取前 32（**内容**） | `min(32, i+1)` | 0 | 6,962,688 |
| `topk64` | 复用同上 `runs/topk64-12k-s0` | 同上，k=64 | `min(64, i+1)` | 0 | 6,962,688 |
| `win32` | **本实验训练** | 最近的 32 个位置（**位置**） | `min(32, i+1)` | 0 | 6,962,688 |
| `win64` | **本实验训练** | 最近的 64 个位置（**位置**） | `min(64, i+1)` | 0 | 6,962,688 |
| `topk32_s1` | **本实验训练** | 同 `topk32`，seed 1 | `min(32, i+1)` | 1 | 6,962,688 |
| `win32_s1` | **本实验训练** | 同 `win32`，seed 1 | `min(32, i+1)` | 1 | 6,962,688 |

后两臂是§3 预登记的**升格对**（主判 k=32 的第二 seed），配方与 seed 0 的同名臂逐字段相同、只换 `--seed`。

- **可见键数对齐（硬约束，落盘核）**：`win` 臂用 `pathlm.depth_ar.window_anchor_keep_mask(256, k, 0)`
  （`keep[i,j] = (j ≤ i) ∧ ((i − j) < k)`），`topk` 臂用 `causal_topk_keep_mask`
  （因果 ∩ 分数前 k）。两式的每行计数同为 `min(k, i+1)`（都含自身 `j = i`，同一约定），
  本实验逐行核这个等式并落盘（`visible_keys_per_row`），不靠文字声明。
- **参数量相等自动满足**：两条掩码路径都不新增参数；`TopKBlock` 与 `Block` 逐位同参
  （`2614` 的单测②⑦钉住），`train_window` 路径在 `Block` 上加一个 `persistent=False` 的 mask buffer（不入 state_dict）。
- **非必要性例外**：无。基底是主流标准 decoder-only transformer。
- **本实验不改共享文件**：`pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**` 全部只读调用；
  新代码只在本目录（`train_arms.sh` 取锁驱动 + `run.py` 评测与对账）。

## 3 判据线（预登记，写死后不得改）

Δ 方向统一（`harness/SCHEMA.md` §17.3）：**Δ = bpc_对照 − bpc_实验**，本实验的「实验」= 内容选择（`topk`），
「对照」= 同 k 的位置窗口（`win`）⇒ **Δ = bpc_win − bpc_topk，正 = 内容选择更好**。

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① **主判（质量）**：同 k 下「内容 vs 位置」的 clean bpc 配对差 | 20 批配对 Δ 对 **0.014 带**（`harness/methodology.md` §2，仅限配对 Δ、≥5 窗口）；三态：`Δ ≥ +0.014` = 内容选择占优、`|Δ| < 0.014` = 无质量增量、`Δ ≤ −0.014` = 内容选择更差 | 该涨（内容该更好）或如实报无增量 / 更差 |
| ② **质量参照线**：各臂相对 `full` 的配对 Δ | 同 0.014 带；只作「少看键的代价」参照，**不与 ① 混用** | 该跌（少看必掉质量） |
| ③ **成本（该跌）**：每位置 KV 缓存字节 | 解析量 `12288 B × 每位置可见键数`（fp16、MHA、d=256/L=8）：`topk32/topk64` 结构性 **−0%**（按内容选必须先算完 T×T 分数 ⇒ 全部 K/V 必须在场），`win32` = −87.5%、`win64` = −75%（相对 `full`）。**本实装是稠密 bool 掩码，省幅是结构上界，未兑现为实测**，如实标注 | 该跌（只有位置窗口能真降） |
| ④ **必须不变**：因果性 | 两条掩码都不得放出未来键：`keep ⊆ causal` 逐位为真，且 `mask_offending_future_pairs == 0` | 必须不变 |
| ⑤ **必须不变**：可见键数逐行相等 | 同 k 两臂的 `kept_per_row[i]` 逐行相等（`min(k, i+1)`），`visible_keys_per_row.mismatch_rows == 0` | 必须不变 |
| ⑥ **必须不变**：参数量与配方 | 五臂参数逐位相等（6,962,688）；配方逐字段一致（见 `results.json::arms`） | 必须不变 |
| ⑦ **申报不进判据**：墙钟、峰值显存、步时 | 逐臂申报；`harness/methodology.md` §3 禁把墙钟当判据 | — |

**n 与带的关系（预登记）**：本实验对 `full`/`topk32`/`topk64` 是 **seed 0 单 run 复用**，对新臂 `win32`/`win64`
是 **seed 0 单 run**。按 `harness/methodology.md` §2「n=1 的负结果必须复现后入账」，主判若越带，本轮只报
**「临时结果（n=1）」**，`status` 停在「进行中」并在 `conclusion` 注明缺第二 seed；不把 n=1 越带写成定论。

**升格路径（预登记，唯一）**：主判 k=32 的第二 seed = 两臂各补一支 seed 1（`topk32_s1` / `win32_s1`，
配方逐字段只换 `--seed`）。判词规则：两 seed 的配对 Δ **同向且各自越 0.014 带** ⇒ 判词从
「临时结果（n=1）」升到 **n=2**，Δ 报成「seed 0 / seed 1 两值 + 两 seed 差」；两 seed **不同向**（或任一
落在带内）⇒ 判「本 n 下不可排序」，`status` 停「进行中」，**不得**把不一致解释成噪声或取平均掩盖。
k=64 的升格对不在本轮预登记内（n 仍为 1）。单格 n=2 只给一致性，不给跨 seed 方差的合并显著性检验
（同 `axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/results.json::cross_seed` 的口径）。

## 4 指标口径（操作性定义）

- **clean bpc**：逐批因果移位 CE（token 平均，bits/char，越低越好），20 批 × batch_size 16，
  `window_seed=2`（与 `2614` 的 `paired_eval.py` 同一批位置、同一 batch 顺序）⇒ 臂间差是**配对差**。
- **每臂在自身可见性下评测**：`full` 走 `is_causal=True`；`topk*` 走 `TopKBlock` 自带的因果 top-k；
  `win*` 走 `masked_logits(model, x, window_anchor_keep_mask(256,k,0))`（=`retention.masked_logits`
  的显式掩码路径，与训练期同一 `window_anchor_keep_mask` 语义）。
- **配对统计**：逐批 Δ 的均值 / 标准差 / 标准误 / t 值 / 同向批数（同 `2614::paired_eval` 口径）。
- **可排序门**：bpc 类只用 0.014 带（配对）；**不跨量纲**借用给任何率类指标（本实验不报率类）。

## 5 预算与成本

| 项 | 值 |
|---|---|
| GPU 预算 | 2 支新臂 × ≈7 min + 冒烟 ≈1 min + 评测 ≈3 min ≈ **18 min**（本实验自报上限 25 min） |
| 升格对预算（seed 1） | `topk32_s1` ≈10 min + `win32_s1` ≈7 min + 评测 ≈3 min ≈ **20 min**（两臂共用同一把 GPU 锁，臂间 ≥61 s） |
| 单臂命令超时 | ≤20 min（`train_arms.sh` 的 `timeout`） |
| 显存 | 逐臂实测 `peak_gpu_mem_mb`；礼貌线 5GB net（`harness/SCHEMA.md` §20.2） |
| 可中断性 | 每 200 步落 `train_log.jsonl`；断点即重取锁重跑该臂（`run_s 是 append 语义` ⇒ 驱动先清日志） |

## 6 失败决策树

| 症状 | 处置 |
|---|---|
| `win*` 臂训练 NaN / 发散（gnorm 尖峰、跳步） | 先修数值腿再判机制；该臂不进判据 |
| 掩码不满足 `keep ⊆ causal` | 实现/构造缺陷，中止判决 |
| 可见键数逐行不等 | 对齐失败，判决作废（报「不可判」而不是硬判） |
| 主判 `|Δ| < 0.014` | 如实报「内容选择相对位置窗口无质量增量」，并指出成本侧仍有结构差（③） |
| 主判越带但只 n=1 | 报「临时结果（n=1）」，`status` 停「进行中」，`gaps` 写第二 seed |
| 升格对两 seed 不同向（或任一落带内） | 报「本 n 下不可排序」，`status` 停「进行中」，不得解释成噪声、不得对两 seed 取平均掩盖 |

## 7 产出

本目录下 `README.md`（本文件）/ `train_arms.sh` / `run.py` / `results.json` / `analysis.md` /
`prereg_binding.json`（契约绑定记录，训练前由 `tools/prereg_bind.py` 写）/ `runs/<arm>/{model.pt,train_log.jsonl,gate_log.jsonl,results.json,stdout.log}`
（`<arm>` = `win32` / `win64` / `topk32_s1` / `win32_s1`）。

## 8 复现 / 重跑

```bash
# 两臂训练（逐臂取/放 SCHEMA §20 GPU 锁，臂间 61 s；单臂 timeout 20 min）
bash train_arms.sh
# 升格对（主判 k=32 的第二 seed）
ARMS="topk32_s1 win32_s1" bash train_arms.sh
# 评测 + 对齐核查 + 等值门 + 申报量（写 results.json，<3 min GPU）
python3 run.py
```

- **跑之前先绑定契约**：`python3 tools/prereg_bind.py axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m`
  （缺绑定 `train_arms.sh` / `run.py` 拒绝启动；判据「先于读数」的机检见 `harness/methodology.md` §8）。
- 冒烟（40 步、量步时、不进判据）：`MODE=smoke bash train_arms.sh`。
- 墙钟与峰值显存见 `results.json::train_declaration`（`win32` / `win64` 是本实验新训的两臂；
  三条复用臂的墙钟取自 `2614-topk-sparse-attn-13m` 的落盘读数）。
