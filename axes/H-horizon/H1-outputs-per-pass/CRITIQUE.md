## ① 它是什么（我的理解）
本格问「一次前向该吐出几个未来位置的预测」，实测的是：在本仓库的宽度-MTP 引擎上把 `n_mtp` 从 1 加到 3 要付多少质量代价，以及「草稿 + 批量验证」的投机解码能带来多少吞吐收益。

## ② 对着什么 sota / 我们偏离在哪
- **sota（据本仓 `axes/H-horizon/PRIOR_ART.md` §H1 自列，我无联网核证）**：Medusa（Cai 2024）在冻结主干挂多解码头 + 树/批量验证；Gloeckle 2024 报 MTP「同时改善质量并自带自投机」；DeepSeek-V3 把 MTP 当训练目标；Leviathan 2023 / Chen 2023 的投机解码无损定理；以及「加速比随规模由 launch-bound 转 compute-bound」的硬件 regime 认识。
- **偏离 1（自造口径，见③-3）**：把一次宽度-K 前向的**墙钟比值**当「等效算力」。
- **偏离 2（有价值的）**：在 13M 字符级 enwik8 上从头测税曲线（业界多在大模型上报告）。

## ③ 五类问题逐条
**1. 开倒车：有。** 本格三条主结论——「MTP 税小、spec 加速」「墙钟收益是 13M/launch-bound 现象」「@100M 前向 compute-bound 后塌」——在本仓 `axes/H-horizon/PRIOR_ART.md` 的机器可读分诊表里被自己判为 `(a)+b`（H1 行）。但 `catalog.yaml:620-649` 仍以 `status: judged` 记账，`axes/H-horizon/README.md` 的本地总体结论把它写成「全项目税效比最好的机制」。按 SCHEMA §21，(a) 类应写 `reference`（业界结论、非本地实测），不跑实验。本地唯一真增量是「@100M fp 平局翻转（1/400 token）」。

**2. 过程错误：有（口径混用 + 证据不可核）。**
- `.scratch/20-spec-scale/evidence/SUMMARY.md`（结果表）**自相矛盾**：同表既写「宽度-3/宽度-1 前向成本 **3.4–5.3×**」，又写「k=2 wall speedup **1.15–1.29×**」，而 tok/forward = 2.37–2.40。由 `speedup = (tok/fwd) / c₃` 反解 ⇒ `c₃ ≈ 2.37/1.29 … 2.40/1.15 = **1.86–2.09**`，与 3.4–5.3 差约 1.8×；二者不能同真。（同表 13M 行自洽：`c₃ = 2.4/1.68…2.4/1.85 = 1.30–1.43 ≈ 表内 1.28`。）**无法核实哪个为真**——无原始 `results.json`。
- 本格两个实验目录只有 backfill 生成的 10 行摘要（`2510-gap-sweep-b2-c1m/README.md`、`2520-spec-at-scale/README.md`），**没有 `results.json` / `analysis.md` / `controls/`**；证据是 `.scratch/*` 的 `@a864bcf` 等锚点，而 `tools/check_schema.py:44-84` 只查证据**路径存在**、把 `@commit` 直接丢弃（`ev.split(" @")[0]`）⇒ 这类锚点无法被门禁识别。SCHEMA §14 的复核清单在本格不可执行。
- 共享卡外部占用使墙钟数字不可复现（H3 的 `2631/analysis.md` §5 自己记录 `llama-server` 常驻、util 77–100%，且在它自己的实验里 1.30–1.58× vs 本格 1.68–1.90×）。

**3. 结论错误：有（严重，符号级别）。**
`catalog.yaml:620-649` 的 H1 failure 行写「@100M 墙钟收益消失…**只剩 FLOPs 口径**」，`docs/methodology.md:25` 的口径表也把「spec 0.55–0.70 fwd/tok」放进名为 **FLOPs（等价 width-1）** 的行。这是把**墙钟比值**当**算力比值**：
- `pathlm/decode.py::Decoder._run_stack_multi / step_multi` 对 **K 个位置**都跑 qkv/proj/mlp，并各跑一遍 MTP 头 ⇒ 一次宽度-(1+γ) 前向的层-位置/线性 FLOPs ≈ **(1+γ)×** 一次宽度-1 前向（与 13M 实测的 `c₃=1.28` 无关；1.28 是 launch-bound 的墙钟）。
- ⇒ 每 token 算力 = `(1+γ)/(1+τ)`：13M `3/2.40 = **1.25**`；100M `3/2.37–3/2.40 = **1.25–1.27**`。
- **正确结论应是**：在本引擎里 γ=2 的投机解码**在算力/FLOPs 口径恒为 1.25× 左右亏损（与规模无关）**；墙钟口径只在 launch-bound（小模型、单流）下赢，「@100M 只剩 FLOPs 口径」不成立（100M 的算力账是 1.25× 亏损，不是收益）。

**4. 无意义：无。** 命题直接对准用户 P-02（`docs/OPEN_PROBLEMS.md` §P-02）。

**5. 值得做：是（中优先）** —— 但只做两件：把口径修对；补一次**干净的 c₃ 实测**（`spec_arms.py::micro_forward_cost` 的 min-法在共享卡上摆动 ±15%，H3 §5 自报 `c₃∈[1.05,1.44]`）。

## ④ 建议动作
**保留 + 重做口径**（不删）。`catalog.yaml` 的 H1 结论句必须改写为「算力亏损 / 墙钟 launch-bound 收益」两列；`status` 建议从 `judged` 降为 `partial`（本格无可核 artifact，证据全在 `.scratch`）。

## ⑤ 重做后的验证方式
- **该涨**：13M 的墙钟 tok/s（仅在该硬件/单流/空卡条件下）。
- **该跌**：任何「省算力 / 省 FLOPs」的主张，必须写成 ≥1.25× 亏损。
- **必须不变**：接受级联 `a₂`、`a₃|a₂`、`tok/fwd ≈ 2.4`；且「层-位置口径」与「墙钟口径」必须**分列并报**，不得把后者叫 FLOPs。
