### ① 这个课题是什么（我自己的话）
让每个 token 只跑到「够用」的那一层就出答案，把省下来的层换成吞吐；本项目把它落成「每层一个逐通道携带门 + 全深度并行监督 + 训练期逐位置随机退出深度 + 混合深度 KV 填充」，并用「浅层出口当草稿、全栈做验证」把省下来的东西兑现。

### ② 对着什么 sota，我们偏离在哪
- 机制对着的是自适应算力一整条线（CALM / Universal Transformer / LayerSkip / Mixture-of-Depths 的 self-speculative early exit）。仓库自己在 `axes/C-vertical/PRIOR_ART.md:20-21` 承认机制层面是 (a)。本地增量是「税」与「FLOPs 净账」，这部分是真空缺。
- **偏离 1（口径，重要）**：业界叙事的判据是「匹配质量下省算力」，我们用 clean bpc 的税。而我们的训练配方把**主目标（最终深度 CE）重新归一**：`pathlm/depth_ar.py:2267` `loss = loss / (w*(len(readouts)-1) + final_weight)`。100M 配方（浅层 w=1、最终 2×）⇒ 最终深度只占 2/14 的权重。业界（LayerSkip / CALM）是「正常训练 + 一个轻量早退项」。所以 +0.44 bpc 里「形态固有」与「主目标降权」没拆开——这是本格最大的解释缺口，且仓库把它当成了已知。
- **偏离 2（墙钟口径）**：`2621` 自己证明仓库 `DepthARDecoder` 比同机械缓存实现慢 **2.02×**（`2621/analysis.md §3`），而 100M 的墙钟倍率仍是用这条慢路径量的。

### ③ 五类问题逐条判定
**1 开倒车：有（部分）。** 机制、以及「浅出口能当草稿」的结论都是业内既有（LayerSkip 的 self-speculative decoding 同机制同边界）。本地真增量只有税与净账。把「逐 token 变深度」写成「本项目最厚的资产」（`catalog.yaml:479`、`catalog.yaml:61`、`INDEX.md:32`）是把业内机制读成本地发现。**业界标准：自适应算力在匹配质量/匹配训练配方下省算力；我们偏离在：把主目标降权后测税，且没有业界的「正常训练 + 轻量早退项」对照臂。**

**2 过程错误：有，3 处。**
- **(a) 墙钟收益在噪声里，却进了结论行。** 同一协议（greedy hard commit、prefill 64 + 96 new、增量逐层 KV、同一 100M 形态）三次运行给出 exit-4 倍率 **1.57× / 3.08× / 5.53×**：
  - `.scratch/24-depth-ar-100m/evidence/100m-s0/decode_wall.json` = 76.32 → 235.41（**3.08×**）
  - `.scratch/25-tax-attribution/evidence/25b-gates-wall.json:81-160` = 49.48 → 273.72（**5.53×**）、uniform 154.62（3.12×）
  - `.scratch/25-tax-attribution/evidence/25c-gates-wall.json:81-160` = 83.24 → 130.43（**1.57×**），且同一臂内逐 prompt 为 168.38/113.62/109.30 tok/s（**15 分钟内互差 55%**）
  `catalog.yaml:482` / `axes/C-vertical/README.md:22` / `catalog.yaml:63` / `archive/legacy-docs/mental_model.md:33` 引的 **3.1× / 1.8×** 是其中一次抽样。唯一做了漂移控制的测量是 `2621`（交错轮次、报比值），它给的是 **1.86×**（同机械全栈 ↑）。
- **(b) FLOPs 负账没有进结论行。** `2621/analysis.md §5` / `results.json::arms.single_g4.eq_layerpos_per_token=8.45 vs plain_cache_full 8.00`（**+5.6%**）：深度草稿不省算力。这是本格最重要的口径反转，但 `catalog.yaml:479-482` 的 verdict/tax/benefit 三行都没有它。
- **(c) n=1 seed。** 13M 与 100M 各一次训练；100M 的「等权臂 ≈ final-2× 臂」只覆盖了配方维度（`24b`），**形态税本身没有跨种子**。
- 未被过程错误覆盖的部分（我独立核过、**成立**）：判据线未事后改（`2621/README.md §4` 预登记与 `analysis.md §4` 一致）；质检护栏真实（`results.json::quality_guard.total_mismatch=0`，800/800 token 与贪心逐位相同）；活性账本可核（`cycle_log.jsonl` 867 行、`emitted == accepted+1`）；对账落盘（`reconcile_report.json` 存在）；该实验有独立复核（`.scratch/39-close-gaps/evidence/REVIEW_X4.md` 的 W2 = OK with notes，重点①通过）——但 `2621/README.md §23 第 7 问` 仍写「待独立复核」，是过期自述。

**3 结论错误：有。** 正确结论应是：**「出口资产（tf@4 0.97）是真的，但它换的是延迟不是算力（FLOPs −5.6%）；换延迟只在单流 launch-bound 解码里成立（13M 实测 1.86×，同机械基线）；代价 +0.3295 bpc（13M matched-step）/ +0.4362 bpc（100M 同引擎）；100M 的 3.1× 不可采信（同协议三 run 极差 3.5×）。」** 「最厚的资产」这个提法把负账藏起来了。

**4 无意义：无。** 这条线直接决定「Pro 当 Flash 用」值不值，是仓库里最具决策相关性的一条。

**5 值得做：是（补做）。** `2621` 已把 100M 端到端列为「下一步 ≈10 min」（`2621/analysis.md §7`），而 `catalog.yaml:490-492` 的证据锚点里**没有 2621**（只在 README §5 出现）；`C1-layers-per-token/PROBLEMS.md §3` 的 TODO 又重复了 `OPEN_PROBLEMS.md` P-02 的「缺净账」——而净账（8.45 vs 8.00）其实已经有了。

### ④ 建议动作
**保留，但重做口径（不是重跑实验）。**
1. 把 `catalog.yaml:482`、`catalog.yaml:63`、`axes/C-vertical/README.md:22`、`archive/legacy-docs/mental_model.md:33`、`axis`/`STATUS.md` 相关行的 **exit-4 3.1× / uniform-exit 1.8×** 换成 `2621` 的交错口径数（1.86× vs 同机械全栈），并附同协议三 run 的极差（1.57–5.53×）说明 100M 墙钟不可测；把「FLOPs +5.6%」写进同一行。
2. 把 verdict 改成「省延迟不省算力 + 形态税 +0.33/+0.44」，删掉「最厚的资产」这种无口径的定性话（`STATUS.md:53` 已经写对了，`catalog.yaml:479` 没跟上）。
3. 补做：100M 端到端深度草稿（10 min，含 sha256）；把 `2621` 挂进 catalog 证据锚点。
4. 若要争「形态固有税」，必须先补一条「正常训练 + 轻量早退项」的对照臂（不降权主目标），否则 `depth_ar.py:2267` 的归一化就是未排除混杂。

### ⑤ 正确的验证方式
- **该涨**：深度草稿相对「同缓存机械的全栈解码」的**交错 tok/s 比值**（3 轮 × ≥4 prompt，报中位与极差）；`tf@4` / `tf@6`。
- **该跌**：等效 layer-position/token（相对全栈必须 ≤1 才算省算力）；γ>1 时草稿链后段的接受率。
- **必须不变**：发出流与同 checkpoint 贪心流逐 token 相同（质量护栏，可失败单测）；判据线不得事后改；**不得**把仓库 `DepthARDecoder` 当墙钟基线而不并列同机械基线；`deps` 不得当组合授权。
