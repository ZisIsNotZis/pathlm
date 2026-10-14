---
name: 2646-layer-width-13m
ticket: 64-axis3-width
date: "2026-10-14"
grid: 3.1.2-layer-width
engine: depth-ar（13M 档 · dense 形态 · 前缀子空间切片）
budget: "gpu_wall_budget: 60 min（含冒烟 ≤2 min + 四臂 × 12k 步 ≈6 min/臂 + 评测/重评）；峰值显存 <5 GB"
status: planned
---

# 2646 · 逐层宽度 schedule（金字塔 / 沙漏 / 倒金字塔）· 13M · dense · 前缀子空间切片

## 1. 命题（一句话，可证伪）

在**总参数逐位对齐**（Σ_l w_l² 相等）、训练/数据/主干配方逐位相同、唯一变量 = **宽度随深度的分配**
的前提下：把逐层宽度按非均匀 schedule 分配（金字塔 / 沙漏 / 倒金字塔）相对**等宽 192**，
clean bpc 是否**越 0.014 噪声带**（即可排序）。

- **H1（备择）**：非均匀分配相对等宽在固定总预算下有可排序的质量增益（「把宽度放到最需要的深度」成立）；
- **H0（零假设）**：Δbpc 落在 ±0.014 带内 ⇒ 逐层宽度变化在本尺度不是独立有增益的维度。

## 2. 单因素声明与干预实现

**唯一变量 = 宽度随深度的分配**（哪一层宽多少），总预算（Σ_l w_l²）逐臂逐位相等。

- 基底 = 主流标准 decoder-only transformer（`train_depth_ar.py --dense --topology sequential
  --exit-mode uniform`，仅最终深度 CE）；**残差流恒 256 宽**（读出恒在 256 宽上做）。
- 数据、步数（12000）、批大小（32）、seq_len（256）、lr（1e-3）、seed（0）、mlp_mult（4）、
  dropout（0）、norm_cap（1.0）、评测批（`depth_curve` 固定 generator seed 2、n_batches=20、
  batch_size=16）全同；attention 接线、LN 位置、位置嵌入、tied 读出、残差形式**逐字段相同**；
  head_dim 恒 32（`w_l` 恒为 32 的整数倍 ⇒ 整头对齐）；layer-passes 每 token 恒为 L。
- **所有臂的 CLI argv 逐位相同**（见 §2.1）：干预不在配置里，在块的前向与切片里。

### 2.1 干预 = 前缀子空间切片（实验目录内自造）

引擎**没有逐层 `d_model`**（`pathlm/depth_ar.py` 的 `Block` 宽度由 `cfg.d_model` 全局给定），
因此干预实现在本实验目录内，`pathlm/**`、`train_*.py` **零改动**：

```
h ∈ R^256                          （残差流恒 256 宽）
第 l 层：xp = x[..., :w_l]
  ln1/ln2      在活跃前缀上做 LayerNorm（ln.weight[:w_l] / ln.bias[:w_l]）
  qkv          q/k/v 各自取「输出行前缀」、输入取「列前缀」，拼成 [3·w_l, w_l]
  attention    头数 = w_l/32（head_dim 恒 32），SDPA 因果，输出 [B,T,w_l]
  proj         proj.weight[:w_l, :w_l]
  mlp          mlp[0].weight[:4w_l, :w_l]、mlp[2].weight[:w_l, :4w_l]（FFN 隐藏 = 4·w_l）
  写回         out = cat(y, x[..., w_l:])   （未触及通道原样通过）
读出           logits = h.float() @ E.float().T     （h 全 256 宽，tied 读出）
```

- `w_l = 256` 时该层是**恒等切片**（`x[..., :256]` 与 `x` 同形状同连续布局）⇒ 前向逐位等于 stock。
- 实现 = `PrefixSubspaceBlock(Block)`：**复用共享 `Block.forward`**（把前缀交给它，块内部
  `d = w_l`），只在本层外把未触及的通道拼回 ⇒ 前缀路径与 stock 路径是**同一段数值代码**，
  不是另写一份前向。权重是 w 级前缀子矩阵 ⇒ **真算少**（每 token MACs 按 w 计），不是掩码。
- **同一次抽样**：所有臂先按同一 seed 建同一份全宽 stock 模型，再按各自 schedule 取**同一份
  抽样的前缀子矩阵**（建块时的 `nn.Linear` 初始化不消耗模型随机数：前后存/还原 RNG 状态）
  ⇒ 臂间不是「两个不同初始化的模型」在比。
- **QKV 前缀语义**：`qkv` 是 `Linear(w, 3w)`，输出按 `chunk(3)` 分成 q|k|v 三块 ⇒ 前缀 =
  **每块各自取前 w 行 + 全体取前 w 列**（拼成 `[3w, w]`），**不是**输出行的连续前缀 `[:3w]`
  （后者是掩码式错切，单测 T2/T3 钉住必须区分开）。

### 2.2 共享探针的替换（不是判据，是让共享入口不崩）

`train_depth_ar.main()` 训练后无条件调用共享 `_kv_head_liveness`，它断言「每层 `qkv` 输出宽
== 3·`cfg.d_model`（单一全局 256）并按该单一 d 推 K/V 头数」。**逐层宽度下该断言必然失败**，
故本实验在训练期把该全局名换成 `_kv_head_liveness_prefix_safe`：在**一次真实前向**上抓每层
`qkv` 的**实际输出宽度**并与目标 `3·w_l` 逐位对照（形状类活性量的硬证据），不符即抛错。
探针只读形状、不参与训练/评测数值路径、**不入判据**。

### 2.3 等值门（`SCHEMA.md` §9 / `docs/methodology.md` §5，落成 pytest 且能变红）

| 门 | 内容 | 判法 |
|---|---|---|
| **T1** | 全层 `w_l = d = 256` 时的**前缀子空间模型** == stock 引擎前向 | state_dict 逐位相同 **且** logits/loss 逐位相同（`torch.equal`），fp32 与 bf16 autocast × train/eval 各测 |
| **T2** | 逐层**实测**活跃维 == 目标 schedule；实测参数量 == 解析值；活跃维来自 `LayerNorm` 的**实际输入宽度**（形状约束）与 qkv 实际输出宽（真实前向 hook） | 逐臂逐层逐位相等 |
| **T3** | 变异必须抓红 | ① 切错**行**（形状相同 ⇒ 形状审计抓不住，逐位比对必须抓）② 切错**列** ③ 换层宽 ④ `w` 不整除 head_dim ⇒ 拒绝 ⑤ 只手改宽度标签不重切 ⇒ 前向必须报错 ⑥ EQ 相对全宽必须不同 |
| **P3-②** | 因果性 | 改末位 token ⇒ 之前位置 logits 逐位不变、末位必须变 |

## 3. 臂（总参数逐位对齐：Σ_l w_l² 相等）

等宽臂 `w = 192` ⇒ Σw² = 8·192² = **294,912**。非均匀 schedule 由 `search_schedule.py`
在「32 的整数倍、32 ≤ w ≤ 256」的格点上**搜索**得到（结果落盘 `schedule_search.json`）：

| 臂 | 定义 | schedule（层 1→8） | Σw² | Σw |
|---|---|---|---|---|
| **EQ** | 等宽（对照） | `[192,192,192,192,192,192,192,192]` | 294,912 | 1,536 |
| **PYR** | 金字塔（前宽后窄，非增） | `[256,224,224,192,192,160,160,64]` | 294,912 | 1,472 |
| **HG** | 沙漏（中段宽两端窄，单峰） | `[64,160,192,224,256,224,192,160]` | 294,912 | 1,472 |
| **APYR** | 倒金字塔（= PYR 镜像，非减） | `[64,160,160,192,192,224,224,256]` | 294,912 | 1,472 |
| **U4**（锚，复用） | d=256/heads=8 全宽 stock | `[256]×8` | 524,288 | 2,048 |

- **搜索规则（写死，可复算）**：
  - **PYR**：全部**非增** 8 元组（共 13 个，见 `schedule_search.json`）中，与契约示例
    `[256,224,224,192,192,160,160,128]` 的 **L1 距离最小**者（并列取字典序最小）。示例本身
    Σw² = 307,200 ≠ 294,912 **不可行**；最近可行解与示例**仅第 8 层不同**（128 → 64），
    前 7 层逐位相同。
  - **APYR** = PYR 的逐层倒序（镜像）。
  - **HG**：**非均匀的严格左右对称沙漏在该格点上无解** —— 唯一的对称单峰解就是等宽本身
    （`schedule_search.json` 的 `symmetric_unimodal_candidates`，`n_symmetric_unimodal_nonuniform = 0`）。
    故 HG 取 **PYR 的宽度多重集按升序交替发到两端（外→内）**的单峰重排：**多重集不变**
    ⇒ Σw² 与 Σw 与 PYR **逐位相同** ⇒ PYR/HG/APYR 三臂的参数与 MACs **完全相等**，
    臂间唯一差别就是「哪个宽度放在哪一层」。HG 满足契约的「中段宽、两端窄」（峰在第 5 层，
    两端 64/160 都窄于中段 256）。
- n：**seed 0 全新增臂**（EQ/PYR/HG/APYR 各 1 条）。
- 复用纪律（`SCHEMA.md` §9，三道具）：**U4/U2 只作阈值/前沿位点引用，不重训** ——
  ① 开关关闭逐位等值单测 = 本实验 T1（全层 `w_l = d` 时前缀模型 == stock）；
  ② 逐字段配置对照 = `anchor_config_parity`（本实验所有臂的轴配置与 2641 U4-s0 逐字段相同）；
  ③ 口径标注 = `cost`/`eval` 块内注明与 2641 同口径（`depth_curve` 最终深度、generator seed 2）。
  U4 = clean bpc **1.5314**、7,392,768 MACs/token、6,962,688 参数（不替代 EQ 这个同预算对照）。

## 4. 判据线（**预登记，不可改**）

| # | 判据 | 该涨/该跌/该不变 | 判定规则 |
|---|---|---|---|
| **P1** | 逐臂 `Σ_l w_l²`、实测参数量（总 / 活 / FFN / 死门分列）、每 token 等效 MACs（attention `Σ(4w²+2Tw)`、FFN `Σ8w²`、读出 `d·V` with d=256）、clean bpc，与 **EQ** 及锚 **U4** **并列** | 该报 | `|Δbpc| < 0.014` ⇒ **不可排序**；`Δ ≥ +0.014` ⇒ 可分（更差）；`Δ ≤ −0.014` ⇒ 可分（更好）。参数/MACs 的小偏差必须**逐位报告**，不得悄悄比不同预算 |
| **P2** | 非均匀臂（PYR/HG/APYR）相对**等宽 EQ** 的 Δbpc（配对 Δ） | 该不跌（H1 要求可排序下降） | 按 **0.014 bpc 带**（`docs/research-schema.md` §12.1 / C1：限配对 Δ、≥5 窗口）判「可排序 / 不可排序」 |
| **P3** | 必须不变 | 必须不变 | ① 评测确定性：`--no-train` 重评与首评 clean bpc 及逐深度 bpc **逐位一致**；② 因果性（改末位 token）；③ 活性：每层真实前向的 `‖block_out‖/‖block_in‖ > 0`；④ 逐层实测活跃维 == 目标 schedule（`LayerNorm` 实际输入宽度 + qkv 实际输出宽）；⑤ 实测参数量 == 解析值；⑥ 等值门 T1 全绿 |
| **P4** | 否证线（`docs/research-schema.md` §12.2） | — | 若非均匀臂在**固定总参数**下相对等宽的 Δbpc **不越 ±0.014 带** ⇒ 判「逐层宽度变化在本尺度不是独立有增益的维度」，本读数并入 `2.3.1` 的深度形状结论；若越带**更差** ⇒ 判该分配在本尺度**反向有害**；若越带**更好** ⇒ 判有可排序增益（并说明分配形态） |

- 质量口径 = **clean bpc**（dense 形态只有最终深度被监督 ⇒ `depth_curve()[-1]["bpc"]`，序列级、
  越低越好），eval 批固定 `depth_curve` 的 generator seed 2、n_batches=20、batch_size=16
  —— 与 2641 逐字段同口径（paired）。
- 方差带 = **0.014 bpc**（C1 裁决）；本实验无率类判据，故不借用量纲。
- **口径**：墙钟 / 吞吐**禁入判据与任何代价表述**（只进 `run_log.txt` 过程日志）；一切代价 =
  **等效理论量**（每 token MACs：attention `4w²+2Tw`、FFN `8w²`、tied 读出 `256·V`，V=206，
  不含任何时间量）。

## 5. 预算与显存

- `gpu_wall_budget: 60 min`（含冒烟 60 步 ≤2 min；四臂 × 12k 步 ≈6 min/臂（MACs 为 U4 的
  0.59 倍）+ 逐臂评测 ≈40 s + 重评 ≈40 s）；峰值显存 <5 GB（13M 档实测 ≈1 GB 量级）。
- GPU 按 `SCHEMA.md` §20 互斥锁**按臂取/放**（`/tmp/pathlm_gpu.lock`），臂间 sleep ≥61 s；
  12k 长跑用 detached supervisor-retry（`.scratch/64-axis3-width/supervise_arms.sh`），
  **不杀他人进程、不回收他人锁**。
- 预算不够时砍什么（`SCHEMA.md` §8）：先砍 **seed 数**（seed 1 一律不跑），
  **不砍判据线、不砍步数**（步数一改臂间刻度就不可比）。

## 6. 指标表（逐项分析，不合成单一分数）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 逐层宽度分配是否是独立于等宽的有增益维度 | — |
| 代价 | 每 token 等效 MACs（attention/FFN/读出分列）+ 参数量（总/活/FFN/死门分列） | EQ（同 Σw²）/ U4（全宽锚） |
| 有效性 | clean bpc（越低越好，越 0.014 才可排序） | EQ（同预算对照）、U4（阈值锚） |
| 分配形态 | 逐层宽度分布（对照列）+ Σw²/Σw 逐位 | 臂间 |
| 可控性 | 逐层实测活跃维 / qkv 输出宽 / FFN 隐藏维 | 目标 schedule |
| 活性 | 等值门 T1–T3、因果性、评测确定性、逐层 `‖out‖/‖in‖` | 契约 §2.3 |
| 副产品 | 未触及通道上的「旁路」（前缀外通道只经 embedding 直连读出） | 与 2642 的 D192 对照时说明（不同干预） |

## 7. 否证条件与归因（做不出来时怎么区分）

- 若 P2 全部落带内 ⇒ H0：判「本尺度逐层宽度变化无可排序收益」，并入 `2.3.1`（深度形状结论）
  与本格 `project.yaml` 的 `falsification`。
- 若 T1/T2 失败 ⇒ 视为**实现问题**（先修实现），不得据此判机制失败。
- 归因三分（`docs/research-schema.md` §7）：①**设计问题**（逐层宽度在此尺度无头寸）
  ②**结构问题**（残差流恒 256 宽 ⇒ 前缀外通道恒为「只经 embedding 的直连旁路」，读出在 256 宽上
  做，故 EQ 相对 2642 的原生 d=192 并非同一模型 —— 不跨干预比较结论）③**innovation 本身失败**
  （宽度分配路线不成立）。
- **不外推**：只判 13M / 12k / seq 256 这一点。

## 8. 复现手段

```bash
# 搜索 Σw² = 294,912 的 schedule（写 schedule_search.json）
python3 search_schedule.py
# 冒烟（60 步，确认干预真的到了模型 + 量步时/显存，不入判据）
STEPS=60 ARMS="EQ:0 PYR:0 HG:0 APYR:0" HOLD_ALL=1 bash smoke.sh
# 全部臂（按臂取/放 GPU 锁，臂间 ≥61 s；长跑由 detached supervisor 逐轮重排）
ARMS="EQ:0 PYR:0 HG:0 APYR:0" bash train_arms.sh
# 单臂重跑 / 仅重评（复用 runs/<arm>-s<seed>-12k/model.pt，写 reeval_<arm>_s<seed>.json）
python3 run.py --arm PYR --seed 0 --steps 12000
python3 run.py --arm PYR --seed 0 --steps 12000 --no-train
# 等值门与变异验证（自包含，不改共享 tests/）
python3 -m pytest test_2646_layer_width.py -q
# 汇总 P1–P4 → results.json
python3 run.py --aggregate --arms EQ,PYR,HG,APYR --seeds 0
```

- 权重**不入库**（`runs/**/model.pt`，`.gitignore` 覆盖）；原始数字 =
  `results_<arm>_s<seed>.json`（逐臂）+ `results.json`（汇总判据）+ `schedule_search.json`（搜索）；
  训练日志 `runs/<arm>-s<seed>-12k/train_log.jsonl`、共享 `results.json`。
