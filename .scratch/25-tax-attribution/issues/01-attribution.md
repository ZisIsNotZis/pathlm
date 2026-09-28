# 25 — 税收归因：H-短视（宽度切分） vs H-时长（48k+冷却）@100M

- **Status:** in_progress（委派子代理）
- **Related:** 24/24b（税 +0.4563 干净归属形态 96%）；用户提出两个候选机制

## 假设与判别

| 假设 | 机制 | 臂 |
|---|---|---|
| H-短视 | 每深度出口损失把各层拉向"随时可退出"的绝对贪心，牺牲全局最优 | **B 结构切分** |
| H-时长 | 全深度监督使收敛更难，训练不足 | **C 48k+冷却** |

判别：哪条臂全深度 bpc 相对 A 恢复最多 → 归因定案；两臂独立有效 → 组合臂 D。

## 臂（全部 d=720/L=12/heads 12/seq 512/mlp_mult 6/batch 8/bf16/seed 0）

| 臂 | 配置 | 步 | 新实现 |
|---|---|---|---|
| A | =ticket 24 主 run（出口读全空间，final 2×） | 24000 | 已有：1.9175，税 +0.4563 |
| **B** | **结构切分**：读出 360 / workspace 360——最终 unembed 与所有出口头只读 h[:360]（tied 表切片 E[:360,:]），trunk 层照常全维度 | 24000 | depth_ar.py 加 `readout_dims` 配置 + 切片实现 + 单测 |
| **C** | 48k 步 + **冷却退火**：出口/浅层损失权重均匀到 75%，末 25% 线性降到 0.3（final 2× 同步降） | 48000 | train_depth_ar.py 加 `--exit-anneal` 调度 + 单测 |
| dense-48k | 同引擎 dense 对照，48k | 48000 | C 的公平对照（等训练量 vs B0_100M 1.4612@24k） |

- B 细节：exit 头与最终头共用同一读出切片（保持机制一致性）；embedding 端
  不动（tied 切片只作用 unembed 侧）；workspace 维仍经残差参与全层计算。
- C 细节：退火对象 = 各深度 aux CE 权重（final 头除外？——不，final 头权重
  2× 也按同调度降回 1.0），纯 trunk CE 恒为 1。切换点/终点可配。

## 测量（每臂统一电池）

- 主：全深度 bpc（税 vs 1.4612 / vs dense-48k）；
- depth 曲线、深度 4/6 出口 bpc + tf@；
- **等效验证吞吐**（自spec 模拟）：接受率(tf@4) + 草稿成本模型 → 等效验证
  加速估计（不报纯跳层 wall 作为主指标——用户定口径：必须全量验证等效）；
- 门分布（B 臂额外：读出/workspace 切片维度范数比——workspace 是否真被用）。

## 判读表

- B 恢复 ≥50% 税 → H-短视主因，切分进配方；
- C 恢复 ≥50% → H-时长主因，渐近/冷却进配方；
- 两臂各恢复部分 → D 组合臂（切分+48k）下一票；
- 均不恢复 → 税结构性存在，"等智商免费加速"正式证伪收口。

## 产出

`.scratch/25-tax-attribution/`（契约+证据+SUMMARY 判读表逐格填）；
findings 追加；WORKSPACE；commit 不 push。

## 全局约束

- 仓库 /home/z/vibe/pathlm；策略 /home/z/vibe/AGENTS.md；中文/代码英文。
- 先 20 步冒烟每臂；GPU 带 timeout；PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
- GPU ≤4h；pytest 全绿（122 基线）+ 新开关可失败单测（变异验证）。
