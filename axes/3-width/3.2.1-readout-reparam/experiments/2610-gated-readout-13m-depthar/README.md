---
name: 2610-gated-readout-13m-depthar
ticket: 28-pilot-batch1
date: "2026-09-30"
grid: W4-readout-geometry
engine: depth-ar（13M 档）
budget: "13M 单臂：冒烟 5 min + 全跑 ~25 min（共享卡按 2× 折扣估 ~50 min）；峰值显存 ≤6GB（net）"
status: planned
---

# 2610 · gated 残差读出（保 tied 桥）· 13M · depth-ar

## 1. 命题

在 **tied 线性读出**上加一个**非线性状态变换**（而非替换成 vocab 尺寸的 MLP 读出），
可以在同尺寸、同 token 预算下收回一部分形态税，同时**不破坏** prob0 / 软提交 /
latent 传输所依赖的 tied 几何。

具体干预（唯一变量 = 读出几何）：

```
基线 A ：logits = h @ Eᵀ
本实验 ：logits = (h + SiLU(W1 h) ⊙ (W2 h)) @ Eᵀ          ← 主臂 gated-residual
       （init：W1 ~ N(0, 0.02)、**W2 = 0** ⇒ 起步 transform = 0，严格恒等起步；
        旧契约「零初始化或极小值」的**「零」选项已删除**，标注为陷阱——
        W1=W2=0 两端梯度同时为 0 ⇒ 永不可学的鞍点：∂L/∂W2 ∝ SiLU(W1h)⊗h = 0、
        ∂L/∂W1 ∝ diag(W2h)·SiLU'(W1h)·(h⊗I) = 0，实测 12k 步后 |W1|=|W2|=0）
       （可选对照臂 logits = h @ Eᵀ + MLP(h) @ W3ᵀ，vocab×hidden，仅作规模陷阱对照）
```

为什么是这个形式（设计理由，写进契约以免被误改成 MLP 读出）：

- 保持 **E 表为唯一读出基** ⇒ 概率仍是「状态 × token 方向」的读出，
  prob0 的语义、`Σp·E` 的凸包安全、linear/soft 传输的投影语义**全部不变**。
- 参数量 = 2·d²（d=256 时 ~131k，可忽略）⇒ **可 scale**；而 vocab×hidden 的 MLP
  在真词表（100k+）上是瓶颈 ⇒ 后者只作对照，不采纳。
- 非线性发生在**读出前的状态变换**上，因此 workspace 维不再被读出秩「饿死」——
  这正是 W1 结构切分失败（能量迁出读出子空间）想解决的问题。

## 2. 单因素声明

只改读出几何。其余与 A 臂（ticket 25 的 depth-ar 配置：全深度并行监督 + carry 门 +
随机退出 + fill_kv）**逐位一致**（同 seed、同超参、同 token 预算）。
基底 = RAP 共享底座（三条必要偏离，见 `docs/design.md` §2），不引入第二个非标准因素。
**非必要性例外**：无。

## 3. 指标表（逐项分析）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | 计算空间与读出可见空间的耦合（宽度约束来源） | — |
| 代价 | Δ clean bpc（13M，同 token 预算） | A 臂 |
| 有效性 | 有效 / 无效 / 反向 | A 臂（tied 线性）+ 同引擎 dense |
| 收益·用法 1（质量） | 税收回幅度 | 已知形态税锚（13M matched ≈ +0.33） |
| 收益·用法 2（自由度） | null 能量占比、functional_dirs、有效读出秩 | A 臂 workspace 探针（null 62 维 / 26 维功能读取 @d=256） |
| 可控性 | prob0 ECE、软提交 rescore（是否仍 OOD 污染） | A 臂（ECE 0.002–0.011；soft 8.4 vs hard 0.8） |
| 副产品 | 出口资产 tf@4 / tf@6 是否保持 | A 臂（0.9726 / 0.9946） |

**判读规则**：任何单一指标改善都不算成功；必须同时满足
(a) clean bpc 不劣于 A 臂 + 方差界内、且 (b) 出口资产 tf@4 ≥ 0.95。
若 (a) 改善但 (b) 崩，判为「又一次整形 vs 自适应」（定律 6），不算采纳。

## 4. 判据线（预先写下）

- **成功**：clean bpc 相对 A 臂 **≤ −0.05（改善）**，且 tf@4 ≥ 0.95，且 ECE 不劣化 >0.003。
- **中性**：|Δ| < 0.014（方差界内）⇒ 判「无头寸」，记录不采纳。
- **失败**：Δ > +0.014 ⇒ 否证，按 §8 决策树归因。
- **自由度判据（探索性）**：null 能量占比相对 A 臂变化 ≥ 5pp，作为「耦合是否被解开」的证据。

## 5. 对照与方差

- A 臂（tied 线性，同配置同 seed）= 主对照；
- 同引擎 dense-AR = 税的分母（已知 13M 级 dense 参照）；
- **seed 数**：冒烟 1 seed；若落在判据线 ±0.02 内，补第 2 个 seed。
- 方差界：Δ < 0.014 不可排序（`docs/methodology.md` §2）。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| 最小可判定版本 | 13M、12k 步、1 seed、只测 clean bpc + tf@4 + workspace 探针 |
| 峰值显存 | ≤6GB net（当前共享卡已占 10GB/24GB） |
| 预计墙钟 | 冒烟 5 min；全跑 25 min（共享卡 ×2 折扣 ≈ 50 min） |
| 可中断性 | 是（每 500 步落 log；中断后可按 checkpoint 续） |
| 冒烟计划 | 20 步 + 全电池（含梯队：gated 读出的恒等起步必须能还原 A 臂：W2=0 ⇒ transform=0 ⇒ logits == h@Eᵀ；且 W1≠0 保证 ∂L/∂W2≠0 非鞍点） |

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`，200MB，本地已有；下游自行切分 12MB 级子集） |
| 入口脚本 | `train_depth_ar.py`（新增 `--readout-gated` 开关）+ `pathlm/depth_ar.py` |
| 可失败单测 | ① `--readout-gated` 关闭时逐位还原 A 臂；② W1=W2=0 时 gated 读出 == tied 线性（恒等极限）；③ gated 起步恒等（W2=0 且 W1 随机）；④ 干预活性门：一步 backward 后 ∂L/∂W2 ≠ 0（非鞍点）；⑤ 变异验证：去掉非线性 / 忽略开关 / 退回零初始鞍点 都必须抓红 |
| 落盘 | `results.json` / `train_log.jsonl` / workspace 探针 json / `analysis.md` / `MODEL.md`（权重哈希） |

## 8. 失败决策树

| 症状 | 诊断顺序 | 处置 |
|---|---|---|
| 税 ≈ 0 | 无头寸 | 记录「13M 无头寸」，升级到 29M 档次再判（不叠加第二机制） |
| 税反向且实现可疑 | ① 实现 bug | 先跑变异验证（去掉非线性 / 关开关逐位还原）；bug 未排除不判机制 |
| 税反向、实现正确 | ② 设计无效 | 否证并写清「在什么条件下才会有头寸」 |
| tf@4 崩 | ③ 定律 6（整形 vs 自适应） | 判为 profile 冲突，记录为反例 |
| 结果不可重复 | ④ 方差 | 补 seed；仍不可排序则撤回 |

## 9. 产出

`axes/W-width/W4-readout-geometry/experiments/2610-gated-readout-13m-depthar/`
下的 `README.md`(本文件) / `run.py` / `config.json` / `results.json` / `analysis.md` / `MODEL.md`。
