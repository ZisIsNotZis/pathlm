# 6.2 多位置预测 = 投机解码 · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2510-gap-sweep-b2-c1m`](2510-gap-sweep-b2-c1m/README.md) | 目录内 README | 旧格 H1 迁入 → 草稿头的质量税首次读数（B2 权重不可复位） |
| [`2520-spec-at-scale`](2520-spec-at-scale/README.md) | 目录内 README | 旧格 H1 迁入 → 每 token 等效 FLOPs 的解析账（R=(1+γ)/(1+τ)）与 fp 平局 |
| [`2610-frozen-trunk-jumphead-13m`](2610-frozen-trunk-jumphead-13m/README.md) | 目录内 README | 旧格 H4 迁入 → WP 6.2.1：冻结主干挂头零税（Δbpc = 0.0000） |
| [`2611-jumphead-ar-control-13m`](2611-jumphead-ar-control-13m/README.md) | 目录内 README | 旧格 H4 迁入 → WP 6.2.1：诚实对手 1-step AR 两次前向 |
| [`2521-exit-scale-confirm`](2521-exit-scale-confirm/README.md) | 目录内 README | 旧格 H4 迁入 → WP 6.2.1：@100M probe 式头零税（jump head 的 100M 零税未测） |
| [`2631-spec-three-arms`](2631-spec-three-arms/README.md) | 目录内 README | 旧格 H3 迁入 → WP 6.2.2：direct / chain / 固定 λ=0.5 combined 三臂 |
| [`2632-alpha1-attribution`](2632-alpha1-attribution/README.md) | 目录内 README | 旧格 H3 迁入 → WP 6.2.2：同一 checkpoint 换 N+2 头/条件的归因与 cross-K 探针 |
| [`2640-tree-conf-13m`](2640-tree-conf-13m/README.md) | 目录内 README | WP 6.2.2 主判：概率多候选 / 树 + 路径概率加权组合（臂 = native/direct1/chain1/combined_fixed/combo_pw/tree_union2/tree_direct_k2/tree_qpw_k2/tree_topk4/oracle 上界） |
| [`2650-verify-gate-13m`](2650-verify-gate-13m/README.md) | 目录内 README | WP 6.2.3：批量验证的正确性门禁 / 失败模式 / 验证开销（对抗与退化候选集、K=1/2/4/8 代价台账、边际收益阈值） |
| [`2613-self-gated-composition-13m`](2613-self-gated-composition-13m/README.md) | 目录内 README | 旧格 H3 迁入 → WP 6.2.2 背景实验（冻结表征上的重复潜在映射与自门控，非本命题） |
| [`2512-overnight-causal-chain`](2512-overnight-causal-chain/README.md) | 目录内 README | 旧格 H3 迁入 → WP 6.2.2：条件链 deploy 首验 |
| [`2513-four-experiments`](2513-four-experiments/README.md) | 目录内 README | 旧格 H5 迁入 → WP 6.2.3：批量验证可用性（含 Medusa 批量验证） |

> 旧格 `H1-outputs-per-pass`（本体）、`H3-composition`（→ WP `6.2.2 概率多候选 / 树`）、
> `H4-head-trunk-relation`（→ WP `6.2.1 草稿头`）、`H5-verification`（→ WP `6.2.3 验证`）
> 按 `.scratch/56-restructure/MAPPING.md` §1 并入本项：四格的 `experiments/**` 整目录迁入本目录
> （一格一份家，不复制）。旧格目录与其 README 仍保留在 `archive/wiki-legacy/axes/H-horizon/**`，旧树退役已由本轮 integrator 完成（旧目录归档只读保留）。
> 旧轴的既有做法情报 `archive/wiki-legacy/axes/H-horizon/PRIOR_ART.md` 仍留在原位，本项字段以指针引用它（外部条目
> 来源标 `记忆·未核`，只作设计方向、不作判据）。
