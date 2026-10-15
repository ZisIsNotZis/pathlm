# 2656 · 低秩压缩（MLA 式）vs 头共享 · 同名同字节判决 · 13M 从零训练

研究项：`axes/5-quant/5.4-low-rank`（WP1）。事实文档（研究项 SSOT）是
`../../project.yaml`，本文件是**本实验的运行前预登记契约**（判据线逐字抄自 SSOT，
未改）。契约格式见 `docs/research-schema.md`；布局见 `docs/structure.md`。

---

## 0. 人话版本（先说结论与做法，不先给代号）

**它想解决什么**：模型生成文本时要把它读过的每个历史位置的 key/value 存起来（KV cache），
存的东西随上下文长度线性膨胀，长上下文时显存/带宽被它吃掉。头共享（GQA/MQA）的办法是
「几个查询头合用一套 key/value」，MLA（DeepSeek-V2 的低秩压缩）的办法是
「把每个位置的 key/value 先压成一条很短的向量存起来，用的时候再展开回去」。

**本实验要判什么**：在**同样大小的每位置存储**下（每个位置都只存 3072 字节），
「低秩再展开」这条路和「几个头合用一套 KV」这条路，谁的语言建模质量更好；以及
低秩那条路的代价（展开的计算、必须从零训练）值不值。

**怎么做的**：同一条 13M 的 decoder-only 主干（12 层 × 256 宽 × 8 头 × mlp6，序列长 256），
只改 KV 的存法：MHA（基线）/ GQA2（2 个 KV 头）/ MLA（潜维 128，与 GQA2 同字节）/
「GQA2 + 把省下的参数还给前馈层」的等参数臂。全部从零训 12000 步、同配方同数据。

**现在到哪一步**：Phase 1 —— 契约、MLA 实现、`--mla-latent 0` 与引擎的逐位等价冒烟、
以及 5 条训练已经挂起排队。**数字还没出来**，本文件里凡未跑的一律写「未跑」。

---

## 1. 本实验是什么

| 项 | 值 |
|---|---|
| 命题 | 在同一「每位置 KV 部署字节」轴上，MLA 式低秩压缩（潜维 128）在 13M 从零训练下能否优于头共享（GQA2），且其独有代价（上投影计算 + 须从零训练）可接受 |
| 单因素 | 只改 **KV 的压缩方式**（低秩 vs 头共享 vs 等价无压缩的 MHA）；主干形状、数据、步数、batch、lr、seed 逐字段不变 |
| 干预实现 | `mla_arms.MLABlock`（低秩 KV：`c=wkv_down(ln1(x))` → `k=wk_up(c)`、`v=wv_up(c)`）；训练入口 `train_mla.py` = 引擎 `train_depth_ar.py` 的逐字副本 + 3 处最小 patch（§5） |
| 不做（非目标） | 低秩 × 量化的叠加（`5.6` 组合，默认留白）、decoupled RoPE、更长上下文（512+）、更大尺度（>13M）的外推、MLA 之外的变体（吸收式 `W_UV` 融合等） |

## 2. 机制归属：相对 DeepSeek-V2 MLA 的增量

- 业界 MLA（DeepSeek-V2，`arXiv:2405.04434`：KV 联合低秩压缩 + decoupled RoPE，
  `head_dim=128`）**只作设计方向**（`5.4::sota_baseline`），**不作判据**；本项不重造 MLA。
- **本仓基底模型没有 RoPE**：`pathlm/model.py::DepthARModel` 用学习式绝对位置嵌入
  `pos_embed`（也无 QK-Norm）⇒ 「RoPE 会破坏低秩结构」这一问题在本尺度不存在，
  故本实现**不需要** decoupled RoPE 分量，K/V 联合低秩投影 + 缓存潜向量就是该尺度下
  MLA 的**忠实降级**（不是遗漏；字节账里没有 RoPE 项，见 §6）。
  这是与 DeepSeek-V2 的**实质结构差异**，必须如实写明。
- 本项相对 SOTA 的**增量**（不是重造 MLA）：
  1. 在 13M **受控同尺寸、同训练预算**下，把「每位置部署字节 → 序列级两腿质量（clean bpc +
     合成长程召回）」的前沿与头共享族放在**同一字节轴**上比（`2617` §3 记业界缺统一字节轴）；
  2. 把 MLA 的**独有代价**入账：上投影的每 token 等效解码 FLOPs + 「不能像 GQA 那样轻量
     uptrain、须从零训练」（业界多在已训好的大模型上做 uptrain 或推理包装）；
  3. 给等参数对照（`5.1::WP3` 共用臂），把「头共享少参数」的参数混杂与机制分开。

## 3. 臂表（最小判决，按优先级）

| 臂 | 干预 | 每位置 KV 值数 | fp16 字节/位置 | 参数量 | 来源 |
|---|---|---|---|---|---|
| `mha` | 基线（8 KV 头 = stock `Block`） | 6144 | 12288 | 13,536,768 | **复用** `2616/runs/mha-12k-s0`（`557453eb…`） |
| `mha_rep` | 基线重跑（同 seed 同配方；run-to-run 散度 + 副本全长旁证） | 6144 | 12288 | 13,536,768 | 本实验训练（`mha-rep-s0`，seed 0） |
| `gqa2` | 头共享对照（2 KV 头） | 1536 | 3072 | 12,352,512 | **复用** `2616/runs/gqa-12k-s0`（`bd682447…`） |
| `gqa2_rep` | 头共享第二次复制（n=2 的第二点） | 1536 | 3072 | 12,352,512 | **复用** `2616/runs/gqa-12k-s0-rep`（`7fa95a7b…`） |
| `mla_c128` | **本项主臂**：低秩潜维 `d_c=128` | 1536 | 3072 | 13,137,408 | 本实验训练（`mla-c128-s0`/`s1`） |
| `gqa2_eq` | **等参数臂**（与 `5.1::WP3` 共用）：GQA2 + Δparams 还给 FFN | 1536 | 3072 | 13,536,516 | 本实验训练（`gqa2-eq-s0`/`s1`） |
| `mla_c256` | 潜维扫描点：`d_c=256`（= MHA 同字节） | 3072 | 6144 | 14,317,056 | 预算允许才跑（排最后，可选） |

- **同一「每位置字节」轴**：`L·d_c = 12·128 = 1536 = L·2·n_kv·dh = 12·2·2·32`
  ⇒ `mla_c128` 与 `gqa2` 在 fp16 下都是 **3072 B/位置**（`config.json::band_axis`）。
- **等参数臂的算式与不可达性**（`mla_arms.eqparams_solution()`；`checks.json` 逐位断言）：
  Δparams = `L·2·(d − n_kv·dh)·(d+1)` = `12·2·192·257` = **1,184,256**；逐层 FFN 隐藏宽
  对参数量的斜率 = `2d+1` = **513**（`mlp[0].weight` d·h + `mlp[0].bias` h + `mlp[1].weight` h·d）；
  `1,184,256 mod 513 = 252 ≠ 0` ⇒ **逐位相等不可能**。本实验取最近可达
  `[1729×4, 1728×8]`（ΣΔh = 2308 ⇒ 返还 1,184,004）⇒ 等参数臂 = 13,536,516
  （比 MHA 少 **252** 参数 = −1.86e-5，远小于被隔离的 8.75% 混杂）。
  > 派单给的「9 层 +193 / 3 层 +192（Σ 精确相等）」按 `512Δh` 估算，漏了 `mlp[0].bias`
  > 的 +Δh 项；见 `config.json::corrections`。
- **残余参数混杂（如实登记）**：同字节下 `mla_c128` 比 `gqa2` 多 **784,896** 参数（+6.35%），
  而 `gqa2_eq` 又比 `mla_c128` 多 **399,108** 参数 ⇒ 「纯机制」读数由 `gqa2`（等字节、少参数）
  与 `gqa2_eq`（等参数于 MHA、等字节）两个对照**夹住**，不许只报其中一个。

## 4. 判据线（逐字抄自 `../../project.yaml`，未改）

> **objective_criteria**：该跌：每位置 KV 部署字节（潜维 + RoPE 分量）；该不跌/可排序：
> clean bpc（相对同字节头共享臂与 stock 基线，须越 0.014 噪声带）；该保持：合成长程召回
> （第二腿不塌）；必须不变：因果性与可见性对数；必须报：上投影的每 token 等效解码计算、
> 等参数对照。

> **falsification**：若 MLA 式低秩压缩在等参数对照下相对同字节头共享臂的 clean bpc 不占优
> （越噪声带变差或持平）⇒ 判低秩压缩在本尺度不产生新知识；若「从零训练」的收敛代价吃掉
> 全部 KV 收益（等字节下不占优）⇒ 判该路线在本尺度不划算；若上投影额外计算使每 token
> 等效 FLOPs 显著上升且未被 KV 收益补偿 ⇒ 判代价记账不利。 用哪个带（C1 裁决）：0.014 bpc
> 带（限配对 Δ、≥5 窗口）为主判；长程召回等率类指标走配对 bootstrap / Wilson 95% CI、
> 不跨量纲借用 0.014。

**C1 判据带声明（本实验的落地口径）**：主判 = **0.014 bpc 带**，只用于**配对 Δ**、
要求 **≥5 窗口**（本实验 24 批同窗口序逐臂独立评 ⇒ 满足）；长程召回等**率类**指标走
Wilson 95% CI / 配对 bootstrap，**不**跨量纲借用 0.014；墙钟**不进判据**（仅记训练时长）。

| 指标 | 该涨/该跌 | 口径（怎么算） | 判据 |
|---|---|---|---|
| 每位置 KV 部署字节 | **该跌** | 解析：值数 × 每值字节；值数由**实测张量形状**核对（`mla_liveness` / `kv_head_liveness`） | 报比值（臂/对照） |
| clean bpc | **该不跌/可排序** | `depth_curve(n_batches=24, batch_size=16, window_seed=2)` 最终深度 bpc；第二条路径 `retention.masked_bpc` 必须逐位一致 | 配对 Δ vs 同字节头共享臂与 MHA；越 0.014 才算可排序 |
| 合成长程召回（第二腿） | **该保持** | `longrange_tasks.var_recall_batch` 距离档 16/32/64/128/192/255，附 `1/64` chance 与 `local_prior_accuracy` | Wilson 95% CI；与 gqa2 的配对差走 bootstrap |
| 因果性/可见性对数 | **必须不变** | 改未来 token ⇒ 过去位置输出逐位不变（`mla_arms.check_causality_invariance`，MLA 与 stock `Block` 各一遍） | 逐位相等（已过，见 §5.2） |
| 上投影每 token 等效解码计算 | **必须报** | 解析 MACs/FLOPs（§6） | 与 FFN/注意力同框 |
| 等参数对照 | **必须报** | `gqa2_eq`（§3） | 与同字节头共享臂参数逐位对齐（差 252，已证明不可达 0） |
| 严格前缀不变（本项非判据，结构性质） | 不变 | MLA 的 K/V 是逐位置投影（无跨位置混合）⇒ 上投影只依赖本位置的潜向量 | 结构性成立（不设门禁，只记） |

## 5. 实现、副本纪律与冒烟证据

### 5.1 副本 + 最小 patch（只动本目录）

| 文件 | sha256 | 说明 |
|---|---|---|
| `train_depth_ar.py`（引擎，**未改**） | `cddbbe0d3a758016308ecd9e501a7a0a946f0844b45d3f06a54aaeec9722b6a3` | 副本的来源 |
| `train_mla.py`（本目录） | `940c20d128310210c9c51882cddb63718fc969b49f117be338b41af2b6546237` | **逐字副本 + 3 处 patch**（P1 sys.path/ROOT；P2 新增 `--mla-latent`；P3 建块后换 MLA 块 + 活性块改用 `mla_liveness`）。`--mla-latent 0`（缺省）⇒ 三处 patch 全不生效 |
| `mla_arms.py`（本目录，新写） | 见 `smoke/summary.json` | `MLABlock` + 解析账 + 5 项自检 |
| `gpu_lock.py`（本目录） | 见 `wp1_status.json` | **照抄** `2632-kv-block-scale/run.py::GpuLock`（类体逐字一致；只拆成单独模块） |

`pathlm/**`、`tools/**`、`tests/**`、`train_depth_ar.py` **一字未动**。

### 5.2 MLA 结构与自检（CPU，`python3 mla_arms.py` → `checks.json`）

- 结构：`c = wkv_down(ln1(x))`（d→`d_c`，无 bias）→ `k = wk_up(c)`、`v = wv_up(c)`
  （`d_c`→`H·dh`，无 bias）；`q` 与 stock `Block` 同款（d→d，带 bias）；注意力
  `scaled_dot_product_attention(..., is_causal=True, enable_gqa=False)`；
  残差接线 `x = x + proj(a); x = x + mlp(ln2(x))` 与 `Block` 逐字段相同。
- **缓存语义 = 每层每位置只存 `c`**（`[B,T,d_c]`）：这是字节账的机制本身，不是标签
  （`check_up_projection_reference` 钉住 K/V 的唯一输入是 `c`）。
- 已过的自检（`checks.json`）：
  ① 参数量解析式 == 实测逐位（MHA/GQA2/MLA_c128/MLA_c256/GQA2_eq 五臂全过）；
  ② 等参数算式与「逐位相等不可能」的证明；
  ③ MLA_c128 与 GQA2 同字节轴（断言相等）；
  ④ **因果性/可见性对数不变**：改未来 token 后过去位置输出 `max|Δ| = 0.0`
     （MLA 与 stock `Block` 对照组都是 0.0）；
  ⑤ 上投影参考实现（显式 c→k,v + 显式因果 softmax）与 SDPA 路径 `max|Δ| = 3.58e-7`；
  ⑥ 潜向量语义（`latent_equals_down_projection = true`）。

### 5.3 `--mla-latent 0` 与引擎的等价性（GPU 冒烟，已跑）

数据（`smoke/` 各 run 的 `train_log.jsonl`）：

| 对账 | step 0 | 末点 | 逐位相同？ |
|---|---|---|---|
| 引擎（50 步）vs 副本 `--mla-latent 0`（50 步），同 seed | 5.4226 = 5.4226 | step 49：2.7002 vs 2.7000（Δ −0.0002） | 否 |
| 引擎（201 步）vs 副本 `--mla-latent 0`（201 步），同 seed | 5.4226 = 5.4226 | step 200：2.5591 vs 2.5571（Δ −0.002） | 否 |

**为什么不是逐位相同（原因已核实，非猜测）**：本仓的 CUDA 训练**同 seed 重训本身就非逐位
可复现**，已有独立先验证据 —— `2616` 的 `gqa-12k-s0` 与 `gqa-12k-s0-rep`（同 seed、同配方、
同数据序，**两次都由未改动的引擎训练**）首行 loss 逐位相同（5.4463），step 200 就已差
**+0.0032**，全程 `max|Δ| = 0.0112`，末值差 0.0014（`2616/stability.md` §2 记末值 bpc 差
0.0004，并写明「初始权重逐位相同 ⇒ 差异只能来自 CUDA 归约顺序」）。

- 本实验的副本/引擎差（step 200：0.002）**小于**引擎自身重训的差（step 200：0.0032）；
- 两者的 **step 0 loss 都逐位相同** ⇒ 初始化与数据序一致 ⇒ 差异不来自 patch 的语义；
- patch 的三处改动在 `--mla-latent 0` 下分别是「加 `sys.path` 项」「注册一个 argparse 参数」
  「一个 `if > 0` 不进入的分支」，无一处触及前向或 RNG 调用序（`diff` 全文见 §5.1）。

结论：`--mla-latent 0` 与引擎**机制等价**（同初始化、同数据序、同更新规则），逐位复现受
GPU 归约序限制；该限制对**所有**臂一视同仁（复用臂也由同一训练器/同一张卡产出）。

| 对账 | 结果 |
|---|---|
| `train_mla.py --mla-latent 128` 的 50 步训练（全路径，含 `results.json` 的 MLA 活性块） | **首轮失败**：`install_mla_blocks` 新建的块留在 CPU，而基底模型已 `.cuda()` ⇒ `ln1` device 不匹配。**已修**（install 后按基底参数的 device/dtype 搬一次并断言一致；CPU 复测通过），修复后的全路径冒烟在下一锁会话自动重跑 ⇒ 未跑完，**不写成已过** |
| 非确定性对照（引擎 vs 引擎） | 见上（2616 的两条日志；`smoke/nondeterminism_control`） |

## 6. 成本（等效理论量，**不含时间量**）

| 项 | MHA | GQA2 | MLA_c128 | MLA_c256 |
|---|---|---|---|---|
| 每位置 KV 值数 | 6144 | 1536 | 1536 | 3072 |
| 每位置字节（fp16） | 12288 | 3072 | **3072** | 6144 |
| 解码期算出新位置 q/k/v 的 MACs/层/token | 196,608 | 98,304 | 163,840 | 262,144 |
| 其中**上投影** MACs/层/token | 0 | 0 | 65,536 | 131,072 |
| 上投影 MACs / FLOPs 每 token（12 层） | 0 / 0 | 0 / 0 | **786,432 / 1,572,864** | 1,572,864 / 3,145,728 |
| 上投影占单层 FFN 的比 | 0 | 0 | **8.33%** | 16.67% |
| 投影 MACs 相对 GQA2 / 层 | −98,304 | 0 | **+65,536** | +163,840 |

- `layer-passes` 每 token 仍为 `L`；`tokens-per-forward` = 1；低秩不额外加层。
- 严格前缀不变（pipelining）：上投影只依赖当前已生成位置的潜向量 ⇒ 不引入额外顺序依赖、
  不需要 staging（与 token 轴分块量化不同）。
- 「须从零训练」的代价：本实验所有臂同为 12k 步从零训练（同等预算记账），
  **不**折算成小时数（业界 MLA 多在已训大模型上 uptrain/推理包装，本项如实记录这一差异）。

## 7. 数据、脚本与复现命令

| 文件 | 作用 |
|---|---|
| `config.json` | 运行前预登记：判据线、臂表、checkpoint 定义、评测流、成本、修正与残余混杂 |
| `mla_arms.py` | `MLABlock` / `install_mla_blocks` / `load_mla_model` / `mla_liveness` / 解析账 / 自检 |
| `train_mla.py` | 引擎逐字副本 + 最小 patch（`--mla-latent`、`--mlp-hidden-sched` 透传） |
| `gpu_lock.py` | 照抄 `2632` 的 `GpuLock`（SCHEMA §20 互斥锁；等锁上限 600 s，**不杀他人进程**） |
| `run_arms.py` | 串行训练驱动（取锁 → 冒烟 → 逐臂训练，失败重试、可中断续跑）→ `train_driver.log` / `train_driver.json` |
| `smoke_mla.py` | 等价性 + 前向冒烟 → `smoke/summary.json` |
| `checks.json` | 解析账 / 因果性 / 上投影参考 / 字节轴的自检输出 |
| `runs/<run>/` | `model.pt` / `train_log.jsonl` / `results.json` / `cmd.txt` |
| 数据 | `.tmp/enwik8` + `data/enwik8_full.npz`（`load_enwik8_full`；**cwd 必须是仓库根**） |

```bash
# 训练（detach；可中断续跑：已有 model.pt 的臂自动跳过）
cd <repo> && nohup setsid python3 axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/run_arms.py \
  >> axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/train_driver.log 2>&1 &
# 加潜维扫描点（排最后）
python3 .../run_arms.py --include-optional
# 自检（纯 CPU）
python3 .../mla_arms.py
```

## 8. 失败决策树

| 观察 | 判定 |
|---|---|
| `mla_c128` 比 `gqa2`（同字节）干净 bpc 差 > 0.014 | 判低秩压缩在本尺度不产生新知识；若在 `gqa2_eq` 上占优 ⇒ 说明是参数混杂而非机制，结论收窄到「等参数下占优」 |
| `mla_c128` 与 `gqa2` 的 Δ 落在 ±0.014 内 | 判「不可分」（该尺度该字节值不值无判决）；按 milestone 明确写「不可分」，不许外推 |
| `mla_c128` 优于 `gqa2` 但劣于 `gqa2_eq`（等参数） | 归因 = 「GQA 省下的参数还给 FFN 更划算」，结论停在参数分配层，不宣称低秩机制优势 |
| 上投影 FLOPs 显著上升且未被字节收益补偿 | 判代价记账不利；报每 token 等效 FLOPs 与 FFN 的比值（`mla_c128` = 8.33%） |
| 长程召回第二腿塌（与 `gqa2` 的 Wilson 95% CI 不重叠且更低） | 记「低秩压缩伤长程召回」，不采纳该潜维 |
| 训练未收敛（loss 斜率未走平、与 `mha_rep` 斜率差可排序） | 先记「未收敛」（`SCHEMA.md` §19），不得在未收敛上判机制 |
| 引擎-副本等价性不成立 | 停在实现层：先修 patch 并重跑冒烟，**不得**用不等价的训练器出的数字作判据 |

## 9. 产出

- 判决包：`{mha, gqa2, mla_c128(±c256)}` 的每位置字节 + clean bpc + 合成长程召回 +
  上投影等效 FLOPs，含 `gqa2_eq` 等参数对照与 n≥2；
- 结论停在「该尺度该字节值不值」，不外推；
- 跑完另派 fresh 复核（`SCHEMA.md` §14/§14b：核对同字节轴、等参数对照、两腿齐备）。

## 10. Phase 1 状态（本文件的运行前结论，数字以 `runs/` 与 `smoke/` 落盘为准）

- 已落盘：本契约、`config.json`、`mla_arms.py`（含 5 项自检，全过）、`train_mla.py`
  （副本 + 3 处 patch）、`gpu_lock.py`、`run_arms.py`、`smoke_mla.py`。
- 训练：**已挂起**（detach，`run_arms.py`，pid 见 `train_driver.pid`），顺序
  `mha_rep → mla_c128(s0) → mla_c128(s1) → gqa2_eq(s0) → gqa2_eq(s1)`（可选 `mla_c256`）；
  每臂 ≈9 min，共享卡按 SCHEMA §20 取 `/tmp/pathlm_gpu.lock`。**训练数字 = 未跑**（写完时仍在跑）。
- 冒烟等价性：见 `smoke/summary.json`（同一次锁会话里跑完）。
- 每位置字节/参数量/FLOPs：**解析值已定**（§3/§6），CPU 自检逐位断言（`checks.json`）。
- 两腿质量与召回：**未跑**（等训练完成后的评测腿，由 finisher 接手）。
