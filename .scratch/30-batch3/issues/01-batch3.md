# 01 — 第三批（ticket 30）：两个纯空白方向 + 一个结论收尾 + 独立复核

用户离场前的授权：主 Agent 编排、子 Agent 执行、只做小规模单因素实验、判据预登记、
每个研究完成后另派一人复核。本批全部基于**主流标准 decoder-only transformer**
（`train_depth_ar.py --dense`），除契约指定因素外不动其他部件。

统一纪律：单因素；对照逐字段等值（含 seed）；干预活性（SCHEMA §10）；
三道对账（§12）；对照臂三要素（§13）；判据取向 = 与自循环/诚实对手比、oracle 只作上界（§11）；
增益归因须过等参数对照（§15）；快照增益须做收敛对账（§16）。

---

## 契约 E：层间拓扑（纯空白格 C3，首次本地探索）

- 目录：`axes/C-vertical/C3-inter-layer-topology/experiments/2613-inter-layer-topology-13m/`
- 文件所有权：可改 `pathlm/depth_ar.py`（新增 block 拓扑变体 + 开关）、`train_depth_ar.py`、`tests/test_depth_ar.py`、本实验目录。
- **干预（唯一变量 = 层间连接方式）**，三个臂：
  - `sequential`（基线）：`h = h + attn(ln1 h)`，然后 `h = h + mlp(ln2 h)`（现有 Block）；
  - `parallel`：`h = h + attn(ln h) + mlp(ln h)`（PaLM/GPT-J 式并行子层）；
  - `dense`：每层输入 = 对**所有此前层输出**的学习加权和（DenseFormer 式），再进 attn/mlp；
    权重初始化为「只取当前层」（等价于基线起步）、且必须报活性。
- 配置：d=256、n_layers=8、mlp_mult=4、12k 步、bs=32、seq=256、seed=0、enwik8 前 90M/末 10M。
- **判据（预登记）**：
  - 主判据：任一变体相对 `sequential` 的 clean bpc 改善 **≥ 0.014**（方差界外）；
  - 并列：`tf@4 ≥ 0.95`（不破坏出口资产）；
  - 中性：|Δ| < 0.014；失败：Δ ≥ +0.014（更差）。
  - **必须附**：三臂同方法（末 25% 窗）loss 斜率，作为「是否只是收敛差异」的收敛对账。
- 成本：3 臂 × ≈13 min = ≈40 min；峰值显存 ≤5GB net。
- 交付：契约 README / run.py / config.json / results.json / analysis.md / stability.md / MODEL.md / 单测。

## 契约 F：读出重参数化在普通 dense 模型上的可迁移性（结论收尾）

- 目录：`axes/W-width/W4-readout-geometry/experiments/2613-readout-reparam-dense-13m/`
- 文件所有权：只写本实验目录（复用既有 `--readout-linear-resid` 与 `--readout-gated` 开关）。
- **干预**：在普通 dense 模型（无门、无退出、仅最终 CE）上，比较
  `baseline`（tied 线性读出）vs `linear-resid`（h + P·(Q h)，等参数 2d²）。
  （gated 臂可选，若预算允许则加第三臂。）
- 动机：12k 快照下「读出重参数化」在 depth-ar 形态上给了 ≈0.08 的改善，但已被归因于
  容量而非非线性。**若它在普通 dense 上也给 ≈0.08，则这是一个与我们的架构无关的通用
  发现（可迁移）；若它只在 depth-ar 上有效，则说明它是「补足某种形态缺陷」。**
- **判据（预登记）**：Δbpc 相对 baseline ≥ 0.014 ⇒ 可迁移（记为通用发现）；
  |Δ| < 0.014 ⇒ 不可迁移（只在形态内有效）；Δ ≥ +0.014 ⇒ 负迁移。
  同样必附两臂同方法 loss 斜率。
- 成本：2 臂 × ≈13 min = ≈26 min。
- 交付：同统一清单。

## 契约 G：自门控多跳组合（纯空白格 H3，用户设计）

- 目录：`axes/H-horizon/H3-composition/experiments/2613-self-gated-composition-13m/`
- 文件所有权：可改 `train_jumphead.py`、`tests/test_jumphead.py`、本实验目录。
- **冻结主干**：复用 `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
  （13.5M dense，12k 步，bpc 1.4836，t+1 acc 0.6979；核对 MODEL.md 的 sha256）。**只训头**。
- **干预（唯一变量 = 组合结构）**，三个臂，**参数量对齐**：
  - `direct`：一个 d→d 变换 T 作用一次 → 读出得 t+2；
  - `composed`：**同一个 T 作用两次**（T∘T）→ 读出得 t+2（即「跳 1 走两次」，与 direct 参数完全相同）；
  - `gated`：`h_mix = g ⊙ (T∘T)(h) + (1−g) ⊙ T(h)`，`g = sigmoid(G(h))` 逐通道；
    G 单边零初始化（起步等价 direct 或 composed，须活性判据）。
- **判据（预登记）**：
  - 主判据：`gated` 的 t+2 命中（vs 真值）**≥ max(direct, composed) + 0.014** ⇒ 自门控组合有效；
  - 并列（用户取向）：**草稿接受率**（= 头 argmax == 主干自身 argmax）≥ 0.90 ⇒ 可作草稿；
  - 同时报 `oracle`（上界，不作判据）；
  - 附加洞见：报 g 的分布（模型偏好一跳还是两跳）与逐桶命中率。
- 成本：头训练 3 臂 × ≈2–5 min + 评测 ≈3 min ⇒ ≈20 min。
- 交付：同统一清单。

## 契约 H：独立复核（fresh context，只读）

- agent：`reviewer`；对象：E/F/G 三条线。
- 清单：`SCHEMA.md` §14 七项 + 本轮特别关注：① **等参数对照是否真的等参数**（逐位核对参数量）；
  ② 是否用 oracle/真值当判据；③ 快照增益是否给了同方法斜率；④ 对照臂是否同一任务定义；
  ⑤ 干预活性（尤其 dense 拓扑的层权与自门控的 g）。
- 输出：逐条 pass/fail + 可疑数字清单；通过 `contact_supervisor` 回传（复核者只读，由主 Agent 落盘）。
