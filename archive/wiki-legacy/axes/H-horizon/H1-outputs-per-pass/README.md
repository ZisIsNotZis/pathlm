---
id: H1-outputs-per-pass
axis: H
grid_dir: H-horizon/H1-outputs-per-pass
title: "每步输出位置数"
status: judged
judge: "税 + 草稿接受率 + 端到端 tok/s（forwards/FLOPs/墙钟三口径）+ 逐位一致（验证正确性）"
cost: "13M 单臂 ~25–30 min；100M ~32 min"
tax: "n_mtp 1→2 +0.041（B2 1.5481）；2→3 +0.0166（B3 1.5604 vs B2fresh 1.5438，递减）"
benefit: "forwards 口径 2.37–2.40 tok/fwd（两规模）；墙钟 13M 1.68–1.90×、100M 约 1.0–1.35×；@100M 接受级联稳定（a₂ 0.80、a₃|a₂ 0.767↑）"
verdict: "**已有（本地实测）**：n_mtp 1→2 税 +0.041、2→3 +0.017（递减）；13M/100M 接受级联稳定（2.37–2.40 tok/fwd）。**算力账**：一次前向送 `1+γ` 个位置、产出 `1+τ` 个 token ⇒ 每 token 算力 `R=(1+γ)/(1+τ)`（代码级核实：`pathlm/decode.py::_run_stack_multi`/`_mtp_nodes` 的 trunk、注意力、MLP、MTP 头都随位置线性，无非线性固定分量），实测 **R = 1.065–1.268×（13M 1.065–1.253、100M 1.090–1.268）**，**两个规模都 >1 ⇒ 恒为算力亏损**。**符号纠正**：「@100M 只剩 FLOPs 口径」不成立——那个「FLOPs 口径」把**墙钟前向比** `c_wall`（13M 1.283 launch-bound → 100M 5.276 微基准，与宽度配对）当成了算力比，造出「随规模翻转」的假象；spec 真正买的是**顺序前向次数**（tokens/forward≈2.4），不是算力；墙钟收益是 13M/launch-bound 现象（13M 1.68–1.90× → 100M 约 1.0–1.35×，本机外部占用下不可稳定复现）。**正确性门禁**：100M enwik8 k=2 有一次首错位 169、级联 68 token、翻转处 gap 0.0079 的平局翻转（两轮运行复现，超出预登记 1e-3 容差）⇒ **逐位相等不是跨规模判据**。**尚缺**：采样/T>0 与树状草稿、更大规模（待补做）。"
deps: ["H3-composition", "H5-verification", "C1-layers-per-token"]
evidence: [".scratch/10-gap-sweep/evidence/B2 @57d64a7", ".scratch/13-four-experiments/evidence/mtp3 @9abc181", ".scratch/20-spec-scale/evidence @a864bcf", "axes/H-horizon/H1-outputs-per-pass/experiments/2510-gap-sweep-b2-c1m/analysis.md @c3cf43c", "axes/H-horizon/H1-outputs-per-pass/experiments/2520-spec-at-scale/analysis.md @b133872", "axes/H-horizon/H1-outputs-per-pass/experiments/2520-spec-at-scale/ledger.json @b133872"]
---

<!-- AUTO:begin -->

# H1-outputs-per-pass · 每步输出位置数

> 轴：**H 视界（一次前向产生几个位置）** ｜ 状态：**✅ judged**

> **判读**：**已有（本地实测）**：n_mtp 1→2 税 +0.041、2→3 +0.017（递减）；13M/100M 接受级联稳定（2.37–2.40 tok/fwd）。**算力账**：一次前向送 `1+γ` 个位置、产出 `1+τ` 个 token ⇒ 每 token 算力 `R=(1+γ)/(1+τ)`（代码级核实：`pathlm/decode.py::_run_stack_multi`/`_mtp_nodes` 的 trunk、注意力、MLP、MTP 头都随位置线性，无非线性固定分量），实测 **R = 1.065–1.268×（13M 1.065–1.253、100M 1.090–1.268）**，**两个规模都 >1 ⇒ 恒为算力亏损**。**符号纠正**：「@100M 只剩 FLOPs 口径」不成立——那个「FLOPs 口径」把**墙钟前向比** `c_wall`（13M 1.283 launch-bound → 100M 5.276 微基准，与宽度配对）当成了算力比，造出「随规模翻转」的假象；spec 真正买的是**顺序前向次数**（tokens/forward≈2.4），不是算力；墙钟收益是 13M/launch-bound 现象（13M 1.68–1.90× → 100M 约 1.0–1.35×，本机外部占用下不可稳定复现）。**正确性门禁**：100M enwik8 k=2 有一次首错位 169、级联 68 token、翻转处 gap 0.0079 的平局翻转（两轮运行复现，超出预登记 1e-3 容差）⇒ **逐位相等不是跨规模判据**。**尚缺**：采样/T>0 与树状草稿、更大规模（待补做）。

## 1. 命题

一次前向为几个未来位置产出预测——决定草稿的宽度。

## 2. 选项枚举（本格全部可能取值）

1. `1（纯 AR）`
2. `k（MTP，本项目 n_mtp 1→3）`
3. `整段（扩散式并行）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：forwards 口径 2.37–2.40 tok/fwd（两规模）；墙钟 13M 1.68–1.90×、100M 约 1.0–1.35×；@100M 接受级联稳定（a₂ 0.80、a₃|a₂ 0.767↑）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：n_mtp 1→2 +0.041（B2 1.5481）；2→3 +0.0166（B3 1.5604 vs B2fresh 1.5438，递减）
- **收益**：forwards 口径 2.37–2.40 tok/fwd（两规模）；墙钟 13M 1.68–1.90×、100M 约 1.0–1.35×；@100M 接受级联稳定（a₂ 0.80、a₃|a₂ 0.767↑）
- **判读**：**已有（本地实测）**：n_mtp 1→2 税 +0.041、2→3 +0.017（递减）；13M/100M 接受级联稳定（2.37–2.40 tok/fwd）。**算力账**：一次前向送 `1+γ` 个位置、产出 `1+τ` 个 token ⇒ 每 token 算力 `R=(1+γ)/(1+τ)`（代码级核实：`pathlm/decode.py::_run_stack_multi`/`_mtp_nodes` 的 trunk、注意力、MLP、MTP 头都随位置线性，无非线性固定分量），实测 **R = 1.065–1.268×（13M 1.065–1.253、100M 1.090–1.268）**，**两个规模都 >1 ⇒ 恒为算力亏损**。**符号纠正**：「@100M 只剩 FLOPs 口径」不成立——那个「FLOPs 口径」把**墙钟前向比** `c_wall`（13M 1.283 launch-bound → 100M 5.276 微基准，与宽度配对）当成了算力比，造出「随规模翻转」的假象；spec 真正买的是**顺序前向次数**（tokens/forward≈2.4），不是算力；墙钟收益是 13M/launch-bound 现象（13M 1.68–1.90× → 100M 约 1.0–1.35×，本机外部占用下不可稳定复现）。**正确性门禁**：100M enwik8 k=2 有一次首错位 169、级联 68 token、翻转处 gap 0.0079 的平局翻转（两轮运行复现，超出预登记 1e-3 容差）⇒ **逐位相等不是跨规模判据**。**尚缺**：采样/T>0 与树状草稿、更大规模（待补做）。

## 5. 失败路径与原因

1. @100M 墙钟与算力**都不是收益**：k=2 墙钟仅约 1.0–1.35×（外部占用下不可稳定复现）、算力 R=(1+γ)/(1+τ)=1.25（亏损）——「只剩 FLOPs 口径」是把墙钟前向比当算力比的符号错误（`c_wall` 非算力）
2. fp 平局翻转在 100M enwik8 k=2 两轮运行都出现（首错位 169、级联 68 token、gap 0.007912，超出预登记 1e-3 容差）→ 逐位相等不能作为跨规模的正确性判据
3. node-k 概率朴素复合是数学错误（CE 3.96 vs 1.58，见 H3）

## 6. 判据与判决实验设计

- **判据**：税 + 草稿接受率 + 端到端 tok/s（forwards/FLOPs/墙钟三口径）+ 逐位一致（验证正确性）

## 7. 成本估计

- 13M 单臂 ~25–30 min；100M ~32 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `H3-composition`
2. `H5-verification`
3. `C1-layers-per-token`

## 9. 复现手段

1. .scratch/10-gap-sweep/evidence/B2 @57d64a7
2. .scratch/13-four-experiments/evidence/mtp3 @9abc181
3. .scratch/20-spec-scale/evidence @a864bcf
4. axes/H-horizon/H1-outputs-per-pass/experiments/2510-gap-sweep-b2-c1m/analysis.md @c3cf43c
5. axes/H-horizon/H1-outputs-per-pass/experiments/2520-spec-at-scale/analysis.md @b133872
6. axes/H-horizon/H1-outputs-per-pass/experiments/2520-spec-at-scale/ledger.json @b133872

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
