# 轴 3 Phase B · 后续派单契约（lead 写契，worker 先落预登记再跑）

> 本文件是 `.scratch/64-axis3-width/TICKET.md` §10.2 派单表的展开版：把 W-B1 / W-A2 / W-B2 / W-C 的
> 契约写死，供 fresh worker 直接执行、fresh 复核者直接对照。**契约内容一旦起跑不得改**（口径差异改在
> `analysis.md` 声明）。判据带按 `docs/research-schema.md` §12.1 声明，剪枝类对照按 §12.2。

## 共同不变量（所有单适用）

- 配置（除非该单显式另说）：`--d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --steps 12000 --batch-size 32 --lr 1e-3 --seed 0 --eval-batches 20`，enwik8 字符级（vocab 206）、`data/enwik8_full.npz`，评测口径 = `pathlm.depth_ar.depth_curve` 最终深度 clean bpc（n_batches=20、batch_size=16、eval seed 2）。
- harness 模式：实验目录内替换 `train_depth_ar.DepthARModel` 为工厂再调 `train_depth_ar.main()`（见 `2641/run.py`）。
- 边界：worker 只写自己的实验目录；`project.yaml`/README 由 lead 写；不改 `pathlm/**`、`docs/**`、`tools/**`、别轴。
- 提交：`git add <显式路径>` → `git diff --cached --name-only` 人工核对（只许出现自己的目录）→ `git commit -F - <<'EOF' … EOF`；索引锁冲突 sleep 3 重试 ≤5 次。
- GPU：逐臂取/放 `/tmp/pathlm_gpu.lock`，臂间 ≥60 s，锁被占最多等 20 min，**不杀他人进程**。
- 墙钟/吞吐只作过程日志，禁入判据与代价表述；一切代价用等效理论量（每 token MACs / 参数量 / layer-passes）。

## W-B1 · `3.2.3` tied vs untied embedding（fresh worker，实验目录 `2643-tied-untied-13m`）

- **唯一变量 = 输入/输出是否绑定**；其余逐字段沿用共同不变量（dense 形态，单出口）。
- 臂（dense 13M；锚复用 `2641` 的 `U4-s0` = 1.5314，按 `SCHEMA.md` §9 三道门：①「解绑开关关闭」逐位同前向单测 ②逐字段 config 对照 ③口径标注）：
  | 臂 | 定义 | 参数增量 |
  |---|---|---|
  | `T`（锚） | tied 读出 `logits = h·Eᵀ`（= 2641 `U4-s0`） | 0 |
  | `U` | 独立 unembed `logits = h·Uᵀ`，`U` 形状 `[206,256]`，独立随机初始化（同 seed 流） | +52,736 |
  | `PR` | 部分绑定：`logits = h·(E + A·B)ᵀ`，`A [256,32]`、`B [32,206]`；init `B ~ N(0,0.02)`、`A = 0`（`A` 有梯度、非鞍点，须写进契约） | +14,784 |
- **实现挂点**：`pathlm/depth_ar.py::DepthARModel.readout(h, depth=None)`（tied、fp32 头）——实验目录内 monkeypatch/subclass 换成上面的读出；不改引擎。
- 判据（预登记）：
  - P1：Δbpc(`U` − `T`)、Δbpc(`PR` − `T`)，按 **0.014 bpc 带**（限配对 Δ、≥5 窗口）判「可排序 / 不可排序」。
  - P2：必须报参数量（总/读出各自单列）+ **本质语义代价**：tied 下「概率 = 状态在 embedding 基的坐标」在解绑后不再成立 ⇒ 报 `argmax(E h) == argmax(U h)` 的位置占比（prob0 一致性），作为不可逆的语义损失列。
  - P3：必须不变：因果性、评测确定性（`--no-train` 重评逐位一致）；活性（幅度非 0）。
  - P4（否证线）：若 Δbpc 不越 0.014 ⇒ 判「本尺度绑定/解绑质量中性（tied 省参数即占优）」；若越带提升但语义代价 + 参数 `vocab×d` 不可 scale ⇒ 判不采纳。**大词表档不可判**（本仓 vocab=206）须显式写。
  - 带声明：clean bpc 差用 0.014 bpc 带；prob0 一致性等率类指标走配对 bootstrap / Wilson 95% CI，不跨量纲借用。

## W-A2 · `3.1.5` 通道极化 / 稀疏激活（fresh worker，实验目录 `2644-channel-sparsity-13m`）

- **唯一变量 = 每 token 激活的通道子集**；总参数逐臂相同（全部臂都保留完整 FFN 权重，只改前向里哪些通道参与）。
- 臂（配置沿用共同不变量；锚 = `U4-s0` = 1.5314 复用；另并列 `2641` 的 `U2-s0` = 1.5709 作「等算力位点但更少参数」的参照）：
  | 臂 | 定义 | 每 token 激活 FFN 通道 |
  |---|---|---|
  | `K50` | 逐 token 按 `|pre-activation|` 取 top-k，k = 50% 隐藏通道（选择器无参数，用既有激活） | 50% |
  | `K25` | 同上，k = 25% | 25% |
  | `F50` | **对照臂**：固定前缀 50% 通道常开（无「条件性」，等于「保参数、砍计算」的退化解） | 50% |
- 实现：实验目录内 patch 各块 FFN 前向（`blk.mlp` = [Linear, GELU, Linear]），把 `gelu(pre)` 乘上选择掩码再走 `lin2`；须给**等值门**：掩码 = 全 1 时前向与 stock 逐位相同（变异用例能变红）。
- 判据（预登记）：
  - P1：clean bpc（各臂 vs 锚 U4）按 0.014 bpc 带判可排序/不可排序；并列报每 token 等效 MACs（按激活比例缩放 FFN 项）与参数量。
  - P2：**活性列**：逐通道激活频率分布（跨若干评测批）+ 使用熵 ⇒ 判是否坍缩（`F50` 是坍缩上界；`K50` 若熵 ≈ 0 即等于 `F50` ⇒ 机制退化）。
  - P3：与 `2641` 的 `U2-s0`（同算力位点、参数减半）同框：**保参数换计算**相对**减参数换计算**谁占前沿。
  - P4（否证线）：若稀疏相对 dense 的质量代价吃掉全部算力收益 ⇒ 判本尺度不成立；若通道使用率坍缩到少数通道 ⇒ 判机制失效。
  - 带声明：clean bpc 差用 0.014 bpc 带；使用率/熵等率类指标走配对 CI。
  - 必须不变：总参数量（逐臂相同）、因果性。

## W-B2 · `3.2.2` 逐出口读出（fresh worker，实验目录 `2645-per-exit-readout-13m`）

- 基底 = depth-AR 形态（`2610` 的 base：`dense=false`、变深度 + 逐层读出 + 随机退出 + `fill_kv`、`aux_weight=1.0`、`exit_mode=uniform`，其余同共同不变量）；锚 = `.scratch/23-depth-ar/evidence/23c-H2-w0-12k`（clean bpc **1.8582**、`tf@4=0.9694`、`tf@6=0.9938`），按 §9 三道门复用；同时**同批重训一条共享 tied 臂**（`T`）作同批对照。
- 臂：`T`（共享 tied）/ `PE`（共享 tied + 每出口 `d×d` 变换）/ `IND`（每出口独立线性表 `[206,256]`，参数量单列且判可否 scale）/ `LR`（每出口低秩校正 `E + A_k B_k`，rank 16）。
- 判据（预登记）：
  - P1：**各出口 tf@k**（深度 1..8 的接受率）为主要判据，率类指标走配对 bootstrap / Wilson 95% CI；clean bpc 差用 0.014 bpc 带、不跨量纲借用。
  - P2：末层（全深度）质量**不劣于**共享 tied 锚（不越带）；读出参数量单列（`IND` = 出口数 × 52,736）。
  - P3：prob0 / 软提交语义仍可用。
  - P4（否证线）：若逐出口读出不能让浅出口 tf@k 可排序提升 ⇒ 判「共享表够用」；若只有不可 scale 的 `IND` 能提升 ⇒ 判不实用；若损末层质量 ⇒ 判不成立。

## W-C · `3.1.3` 同层内位置相关宽度（**本轮只在时间允许时派**，实验目录 `2646-position-width-13m`）

- 臂：`PN`（前宽后窄：前半序列位置用全宽、后半用窄子网络）/ `NP`（前窄后宽）/ `GATE`（位置 gate 选宽度）；**必须报**长程召回列（率类 CI）与每 token 平均激活通道数；只改宽度、不改可见集合（attention mask 不变）⇒ 与时间轴正交可核（mask 逐位对照）。
- 判据：clean bpc（0.014 bpc 带）+ 平均激活通道数（该跌）+ 长程召回（该不变/该跌）+ 层数与因果性不变。
- 否证线：省了算力但 bpc 或长程召回越带退化 ⇒ 判不划算；窄段恰是长程关键位置 ⇒ 否证。

## W-D · `3.1.2` 逐层 d_model schedule（**本轮大概率不派**：实现前置未满足）

- 事实：引擎无逐层 `d_model`（`Block` 的 `d_model` 由 `cfg.d_model` 全局给定），E2（逐层 FFN 宽）**也不覆盖本项**（且尚未落树）。
- 若派：实验目录内用「**前缀子空间切片**」自造——第 l 层只在残差流前 `w_l` 维上运算（该层 attention/FFN 权重取前缀行/列子矩阵、LN 在该子空间上算），
  令 `Σ_l w_l²` 逐位对齐等宽臂（`docs/research-schema.md` §12.2 / 本项 `cost_model`）；须给等值门（`w_l = d` 全层时与 stock 逐位相同）与「切片 == 掩码」逐位门。
- 判据：clean bpc（0.014 bpc 带）、总参数逐位相等、每层宽度分布（对照列）、layer-passes 不变。
