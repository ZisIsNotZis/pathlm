---
name: 2647-position-width-13m
ticket: 64-axis3-width
date: "2026-10-14"
grid: 3.1.3-position-width
engine: depth-ar（13M 档 · dense 形态）
budget: "gpu_wall_budget: ≤40 min（2 臂 × 12k 步 ≈ 8 min/臂 + 评测 + 合成探针）；峰值显存 ≤5GB net；锁被占时排队（单次 ≤25 min）"
status: planned
---

# 2647 · 同层内位置相关宽度（前 128 / 后 128 位置走不同宽度子网络）· 13M · dense

本文件是**预登记契约**：§1–§10 在起跑前写定，起跑后**不改**；口径差异只在 `analysis.md` 里声明。
派单契约 = `.scratch/64-axis3-width/DISPATCH-B.md` §W-C。

## 1. 命题（一句话，可证伪）

在**训练/数据/主干/总参数逐位相同**（只改「同一层内不同位置用多少通道」）、
**可见集合逐位不变**（attention mask 与 stock 相同）的前提下，把前 128 个位置与后 128 个位置
分别接到全宽（256）与窄子网络（128）上：

- **H1（备择）**：位置分段宽度在**每 token 平均等效 MACs 下降约 35%** 的同时，clean bpc 相对
  stock 全宽锚（`2641` 的 `U4-s0` = 1.5314）的 |Δ| < 0.014 bpc，且长程召回不跌 ⇒
  「同一层内按位置分配宽度」是独立的算力-质量前沿维度；
- **H0（零假设）**：Δbpc 越 0.014 带、或长程召回越配对 CI 退化 ⇒ 在本尺度不划算。

对准的痛点：条件计算 / 位置相关容量（local-global 分段模型的位置版）许诺「同层内按位置省算力」。
本条的机制**没有路由参数、不改层数、不改可见集合**，只问一件事：**同一层里给不同位置不同宽度值不值**。

## 2. 单因素声明

**唯一变量 = 同一层内每个位置的激活宽度（参与前向的残差流通道前缀长度）**。

- 基底 = 主流标准 decoder-only transformer（`train_depth_ar.py --dense --topology sequential
  --exit-mode uniform`；三条共享底座偏离见 `docs/design.md` §2）；
- embedding / 位置嵌入 / 层数（8）/ 头数（8）/ `mlp_mult=4` / 优化器 / 步数 / 批大小 / lr / seed /
  数据切分 / 评测批（`depth_curve`，`n_batches=20, bs=16, generator seed 2`）**逐字段相同**；
- **总参数逐臂/逐位相同且全部有梯度**：不新增、不裁剪任何参数；窄段只是**用**全宽权重的前缀
  子矩阵，尾部通道的参数在窄段位置不参与前向，但**被宽段位置使用** ⇒ 没有恒零梯度区、
  没有「参数在账上但不训练」的退化（§7 门 T6）。臂间唯一差别是**每个位置**前向里
  「哪些通道参与」；
- **可见集合不变**：attention 调用与 stock 逐位相同（`attn_mask=None, is_causal=True`，
  同形状 q/k/v），无任何位置相关掩码（§7 门 T2 用 SDPA 调用记录钉住）；
- **必要性例外：无**。

### 2.1 干预实现（为什么它不是「换了个模型」）

- 挂点 = `pathlm.model.Block.forward`（pre-LN：`ln1 → qkv → SDPA → proj` 残差、
  `ln2 → mlp[0] → GELU → mlp[1] → Dropout` 残差）。本实验在**实验目录内**把每个
  `blk.forward` 换成 `position_width_forward`（只改前向实现、不进 `pathlm/**`）；
  训练时把 `train_depth_ar` 模块里的全局名 `DepthARModel` 换成工厂（与 `2641-ffn-ratio-frontier-13m/run.py` 同款接线），
  参数解析 / 配置装配 / `train()` 循环 / checkpoint / `results.json` 全走**共享实现**。
- **位置分段**：段由**绝对位置**决定（前 128 / 后 128），与内容无关、与可见集合无关。
  每段宽度 `w(p) ∈ {256, 128}`。
- **前向语义（「窄子网络」的精确定义）**：设 `k(p) = {0..w(p)−1}`（通道前缀），`c(p)` 为
  `k(p)` 的补集。该块对位置 p 的计算是
  1. LN1：`h1 = LN1_k(x)`（在 `k(p)` 维上做 LayerNorm，用 `ln1.weight/bias` 的前缀），
     `h1[c(p)] = 0`；
  2. qkv：`q,k,v = (W_qkv · h1)[k(p)]`（= `W_qkv[:w, :w]` 前缀子矩阵），`q,k,v[c(p)] = 0`
     —— 尾部**头**（head 4..7）的 q/k/v 恒 0，即窄段位置只用前 4 个注意力头；
  3. 注意力：与 stock **同一个调用**（`SDPA(qs, ks, vs, attn_mask=None, is_causal=True)`），
     窄段位置的 q/k/v 尾部为 0 ⇒ 其对尾部头的贡献恒 0（`v=0` 且 softmax 权重不改变其余头的输出）；
  4. 输出投影：`Δ_attn = (W_proj · a)[k(p)]`（前缀子矩阵），残差只加在 `k(p)` 上；
  5. LN2 / FFN：LN2 同 1 用前缀参数；`mlp[0]` 用 `[:4w, :w]` 前缀（隐藏维 = 4w），GELU（`GELU(0)=0`），
     `mlp[1]` 用 `[:w, :4w]` 前缀，残差只加在 `k(p)` 上。
  因此位置 p 的该层计算**逐位等于**「把该位置接到一个宽 `w(p)` 的子网络、其余维度不参与」；
  残差流的尾部通道对窄段位置**只被读出（readout）看见、不再被任何层更新**（这是本干预的
  语义代价，登记在 §5 P2）。
- `w(p) ≡ 256`（不缩）时前向与 stock **逐位相同**：实现里该情形直接委托
  `Block.forward` 原实现（§7 门 T1）。注意这不是「掩码恒 1 的近似相等」，是**同一条代码路径**。
- **代价口径**：墙钟/吞吐**禁入判据**；一切代价用等效理论量（每 token 平均激活通道数、
  每 token 等效 MACs、参数量）。

## 3. 臂定义与代价

配置（逐字段沿用共同不变量）：`d=256, layers=8, heads=8, seq=256, mlp_mult=4, steps=12000,
bs=32, lr=1e-3, seed=0`，enwik8 字符级（vocab 206）、`data/enwik8_full.npz`。

| 臂 | 位置→宽度 | 每 token 平均激活通道 | 每 token 等效 MACs | 参数 |
|---|---|---|---|---|
| `PN` | 前 128 位置 256、后 128 位置 128 | 192 | 4,771,328 | 6,962,688（逐位同锚） |
| `NP` | 前 128 位置 128、后 128 位置 256 | 192 | 4,771,328 | 6,962,688（逐位同锚） |
| `FULL`（**只做等值门与锚探针，不训练**） | 全部位置 256 = stock | 256 | 7,392,768 | 6,962,688 |

- 解析口径（每层每 token，宽 `w`）：attention `4w² + 2Tw`（qkv 3w² + proj w² + QKᵀ `Tw` + AV `Tw`）、
  FFN `2·(4w)·w = 8w²`；按位置段加权，再加 tied 读出 `d·V`。`T=256, d=256, L=8, V=206`。
  `FULL` 的解析值与 `2641` 的 `U4-s0` 实测 `macs_per_token.total = 7,392,768` **逐位一致**（口径对账）。
- `PN` / `NP` 的 MACs 相等（分段只换位置顺序）⇒ 两臂互相构成「同一代价、相反位置分配」的对照；
  与 `FULL` 的代价比 = 0.645405（省 35.46%）。
- **`GATE` 臂（位置 gate 选宽度）：本轮不跑**（时间盒；见 §9 未做项登记）。契约定义：
  `w(p) = 256` 若 `σ(g_p) > 0.5` 否则 `128`，`g_p` 为每位置一个标量（+128 参数 ⇒ 总参数不再逐位相同，
  必须单列）；登记为未做，不当作结论。

## 4. 锚复用（`2641` 的 `U4-s0` = 1.5314，按 `SCHEMA.md` §9 三道门）

1. **等值门（本仓可跑）**：`FULL` 臂前向与 stock 逐位相同（单测 T1），且 `FULL` 的解析 MACs /
   参数量与 `U4-s0` 实测值逐位一致（§3 表）；
2. **逐字段 config 对照**：本实验的 `shared_argv` 与 `2641` 的逐字段相同（同 seed / 步数 /
   批大小 / lr / eval-batches / dense / sequential / uniform）；`2641` 的 `U4` 臂不做任何切片
   （`narrow_ffn_prefix` 在 `schedule == [4]*8` 时是无操作）⇒ `U4-s0` 就是 stock 全宽 13M；
3. **口径标注**：锚值 1.5314 = `depth_curve(...)` 最终深度 clean bpc（`n_batches=20, bs=16,
   eval generator seed 2`），与本实验**同一函数、同一参数**，逐位同源。

锚的 `model.pt`（`2641/runs/U4-s0-12k/model.pt`）**只读**用于合成召回探针（纯推理，§20.1.2 允许的插入作业）。

## 5. 判据（预登记，起跑后不改）

- **P1（主判）clean bpc**：`Δbpc = bpc(臂) − 1.5314`（锚 `U4-s0`），按 **0.014 bpc 带**判
  「可排序 / 不可排序」。限配对 Δ（同一评测批、同一 `depth_curve` 口径、≥5 窗口 = 9 个深度点）。
  `PN` 与 `NP` 之间同样按 0.014 带判（同代价、相反位置分配）。
  - 判「划算」：两臂 |Δ| < 0.014 **且** 平均 MACs 显著下降（P2）**且** 长程召回不退化（P3）；
  - 判「不划算」：任一臂 Δ ≥ +0.014（bpc 退化越带）。
- **P2（该跌）每 token 平均激活通道数与等效 MACs**：194→…即 256 → 192（−25%）通道、
  7,392,768 → 4,771,328 MACs（−35.5%）。必须报**实测**值（前向里非零通道计数）与解析值一致。
- **P3（该不跌 / 可分）长程召回列（必须报）**：同一个已训 checkpoint 上，用引擎自带合成任务
  零样本探针 `assoc-mqar`（`assoc_recall_multi_batch`：`n_pairs=8, n_queries=8, gap=14`，
  即 `--synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14`；seq 256，
  `sym_vocab=64`，chance = 1/64 = 0.015625）。`PN` / `NP` / 锚 `U4-s0` 三者在**同一条探针、
  同一批任务**上评测：报 top-1 准确率 + Wilson 95% CI + `PN`/`NP` 相对锚的**配对 bootstrap**
  ΔCI（率类指标走 CI，**不跨量纲借用 0.014**）。配对单元 = (序列, 查询) 命中位；默认
  4 批 × 16 序列 × 8 查询 = 512 个配对单元。
  - 该不跌：Δ 的 95% CI 含 0（或下界 > −0.05）；可分：CI 不含 0 且 |Δ| 大。
  - **若探针无法在时间盒内接通，如实记「未跑」并写明原因**（不得省略该列而不声明）。
- **P4（否证线）**：
  1. 省了算力但 clean bpc 越 +0.014 或长程召回配对 CI 显示显著退化 ⇒ 判**本尺度不划算**；
  2. **窄段恰是长程关键位置**（`PN`：后 128 位置 = 靠近查询的远端/近端信息；`NP`：前 128 位置 =
     (key,value) 表所在处）且召回退化吃掉全部算力收益 ⇒ **否证**。两个臂在位置上互为对照，
     故可分辨「窄段位置是否关键」：若只有一侧退化，则退化归因到**位置**而非宽度。
- **必须不变**：因果性（改后一 token 不改变前面位置的 logits）、参数量逐位相同、层数 8、
  **attention mask 逐位相同**（P2/§7 门 T2）、评测确定性（`--no-train` 重评逐位一致）。

## 6. 口径与禁入项

- 一切质量读数 = clean bpc（越低越好）：最终深度（无退出）、序列级、`n_batches=20, bs=16`、
  eval generator seed 2、窗口采样顺序与 `depth_curve` 同源；
- 一切代价 = 等效理论量（每 token MACs / 平均激活通道数 / 参数量）；**墙钟与吞吐只作过程日志，
  禁入判据与代价表述**；
- 率类指标（召回准确率）用 Wilson / 配对 bootstrap 95% CI；bpc 差用 0.014 带；**不跨量纲借用**；
- 结论不越 13M / 12k 步 / seq 256 / 单 seed 外推（`PN`/`NP` 只有 seed 0；跨 seed 散布不作推断）。

## 7. 等值门与变异（落成 pytest，能变红）

- **T1**：`FULL` 臂（`w ≡ 256`）前向与 stock 逐位相同（`torch.equal` logits）；
- **T2**：attention 调用与 stock 逐位相同 —— 记录两边的 `SDPA` 调用参数
  （`attn_mask is None`、`is_causal is True`、q/k/v 形状一致），逐层比对；**只改宽度的证据**；
- **T3**：平均激活通道数 / 等效 MACs 与解析值一致（解析公式在单测里**独立重推**：384/192/256 通道
  ↔ 4.77M/7.39M MACs；并实测每位置进入 qkv 的非零通道数 = 段宽）；
- **T4（变异）**：把窄段掩码取错维度（掩掉前缀而非尾部，即保留 `[d−w:]` 而非 `[:w]`）
  ⇒ 与正确参考**不相等**（断言 `not torch.equal`），且与 stock 不等 ⇒ 门 T1/T3 不是恒真；
- **T5（活性）**：窄段位置的**块内输出增量**对其自身的尾部通道扰动**逐位不变**，
  而对前缀通道扰动**必然改变**（正对照，证明 T5 不是恒真）——「窄段真的是窄的前向」；
- **T6（训练路径）**：训练模式下前向 + 反向的 loss 有限；`blocks.*` 参数**全部**有非零梯度，
  且零梯度参数集合与 stock **逐位相同**（两臂的宽段位置覆盖了块内全部参数 ⇒ 本干预不引入
  死参数；`gates.*` 在 dense 形态下 stock 本来就没有梯度——预先存在，不是本干预引入的）。

## 8. 复现

```bash
python3 run.py --arm PN --seed 0 --steps 12000        # 训练 + 评测（需 GPU 锁）
python3 run.py --arm PN --seed 0 --steps 12000 --no-train   # 仅评测（复用 model.pt）
python3 run.py --synth-probe --arm PN --seed 0        # 合成召回探针（纯推理）
python3 run.py --synth-probe --arm FULL --ckpt <2641 U4 model.pt> --tag anchor2641U4-s0
python3 run.py --aggregate                            # 汇总 P1–P4 → results.json
python3 -m pytest -q test_2647_position_width.py      # 等值门 + 变异
bash train_arms.sh                                    # 逐臂取/放 /tmp/pathlm_gpu.lock
bash ../../../../.scratch/64-axis3-width/supervise_arms.sh  # detached supervisor-retry
```

## 9. 未做项登记

- `GATE` 臂（位置 gate 选宽度）**未做**：时间盒内只跑 `PN`/`NP`；契约已写定（§3），
  下一步最小实验 = 每位置 1 个标量 gate（+128 参数，参数不再逐位对齐，须单列）+ 同配方 12k，
  成本 ≈ 8 min GPU/臂 + 评测；
- **单 seed（s0）**：`PN`/`NP` 各 1 个 seed，跨 seed 散布未测（另加 seed1 ≈ 16 min GPU）；
- 位置「分段」只测了**绝对位置前半/后半**这一个默认（`3.1.3` `gaps` 里登记的「位置定义未定」：
  「距离近端/远端」与「内容分区」两种定义未测）；
- 探针只测 `assoc-mqar`（gap 14）一条长程召回任务；窗口/锚点等其它可达性口径未测。

## 10. 落盘清单

`README.md`（本契约）· `run.py` · `test_2647_position_width.py` · `train_arms.sh` ·
`runs/<臂>-s0-12k/{model.pt,train_log.jsonl,results.json}` · `results_<臂>_s0.json` ·
`synth_<tag>_s0.json` · `results.json`（汇总）· `analysis.md` · `run_log.txt` / `supervise.log`
（过程日志）· `.scratch/64-axis3-width/2647-position-width-process.md`（过程记录）。
