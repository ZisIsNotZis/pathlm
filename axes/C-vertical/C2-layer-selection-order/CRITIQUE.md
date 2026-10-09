### ① 这个课题是什么（我自己的话）
把「跳层、重复层、打乱层序、替换/屏蔽 token」这些**路径扰动**各自单独打开一次训练，量它们对 clean bpc 的代价，排出一张「元素有多贵」的表。

### ② 对着什么 sota，我们偏离在哪
- 对着 LayerDrop / stochastic depth（训练期随机丢层）、输入腐蚀（MLM 式 mask / wrong-token）、test-time ensembling。
- **偏离（口径）**：我们的元素在**评估时也开着**。`pathlm/eval.py:41-42` 的 `eval_pc` 只清 `corrupt_wrong/corrupt_mask/p_retry/p_token_retry/w_dense_exit/perturb_noise/pure_noise`，**不清** `p_skip` / `p_redo` / `shuffle_locality`。所以 +0.175 是「带着 10% 层被丢掉的部署质量」，不是 LayerDrop 惯常报的「训练期丢层、回全深度评估」的质量。**业界标准：LayerDrop 报训练期正则化后全深度评估的退化；我们偏离在：报的是带元素的部署质量。两者不能并列。**
- 另一个偏离：把「省算力的元素（skip）」「费算力的元素（redo）」「无收益的腐蚀元素（noise/mask）」放进**同一张排序**（`.scratch/06-m2-sweep/evidence/SUMMARY.md` 的 "Element-tax ranking"）。

### ③ 五类问题逐条判定
**1 开倒车：有。** `p_skip` / `p_redo` 就是 LayerDrop / stochastic depth；「层序是强先验」也是常识（`catalog.yaml:523` 的「连层的执行顺序都是先验」当新发现写）。`axes/C-vertical/PRIOR_ART.md:142` 说「层序重排是最贵元素（+0.545）业界不报」——这是**未核信息**（该文件自己声明联网不可用、外部一律 `记忆·未核`），却被写进了「我们的增量」。

**2 过程错误：有，4 处。**
- **(a) 无独立复核**：`axes/C-vertical/C2-layer-selection-order/experiments/2506-m2-element-tax-law/README.md` 的源工单 `.scratch/06-m2-sweep/issues/01-m2-breadth-sweep.md`（Comments 段）明文：「review: independent reviewer unavailable（… rate limit …，then reviewer agent tool-contract failure）… **User approved self-review in lieu**」。SCHEMA §14 要求 fresh-context 独立复核，这里没有。
- **(b) n=1 seed**：7 个 run 各一次，无 seed 复现、无 CI；`catalog.yaml:511` 直接用单点数字下排序结论。
- **(c) 数字自相矛盾（小）**：`L3/results.json::bpc.bpc = 2.0527`，而同文件 `locality_sweep` 的 0.5 点是 2.0403；差 0.0124（<0.014，噪声带内，但两者应是同一个量）。
- **(d) benefit 行的来源与结论相反**：`catalog.yaml:512` 写「skip 是最好的 TTS 基底（K=8 −0.047，DIVL1 再 +0.023）」，而其来源 `.scratch/08-tts-probe/evidence/SUMMARY.md` 自己的判读是「**no composed number beats B0**」「ensembling cannot outrun the tax」（L1@K8 = 1.6454，仍高于 B0 1.5074 达 +0.138，代价是 **8× 评估算力**）。把「输得最少的那条」写成「收益」。
- **可核为真的部分**：数字与原始 json 一致（`L1/results.json::bpc.bpc = 1.6822` ⇒ +0.1748；`L3` = 2.0527 ⇒ +0.5453；locality sweep 2.0098/2.0133/2.0403/3.8438 与 SUMMARY 一致）；配置确为单因素（`configs/L1.json`、`L3.json`、`L4.json` 各自只比 `PathConfig` 默认多一个键，`pathlm/config.py:49-51` 的默认是 0）。

**3 结论错误：有。** 「元素税定律」把三类不同性质的东西混排，且在**没有任何部署 regime 下能赢过基线**（最好点 L4@K8 = 1.5562 @8× 评估）。仓库自己已经在 `docs/OPEN_PROBLEMS.md` P-13 承认「『保留型近乎免费』来自 A4 的 −0.007 单点」「『元素税定律』由少数元素支撑 ⇒ 降级」，但 `docs/playbook.md` 定律 1 与 `catalog.yaml:510` 仍按「定律」写。**正确结论应该是：「在带元素部署的口径下，元素代价排序 = shuffle ≫ skip > wrong-token > mask > redo ≈ noise；其中只有 skip 换来省算力，其代价必须按 Δquality/Δcompute 报；TTS 集成在任何 K 都不足以追平元素税。」**

**4 无意义：有（子项）。** **层序 shuffle** 这一子项建议删除：没有任何部署会打乱层序（它在 `IX2/IX3/INT` 里还是「组合毒药」），结论无论正负都不改变「该算力路径值不值」的判断（`catalog.yaml:520-524`）。数字作为一条 playbook note 留存即可。

**5 值得做：有（把 skip 放回同族引擎重测）。** `p_skip` 是**逐 batch 全局丢层**（`pathlm/config.py:150-165`：per-layer skip，按 batch 采样一次），与 C1 的**逐 token 退出深度**不是同一个机制。C2 给的 +0.175 因此不能直接用来给 C1 的退出定价。要定价，得在 depth-ar 引擎里按「逐 token 退出深度比例」画 Δbpc–Δcompute 曲线。

### ④ 建议动作
- **保留课题**（成本表本身是 C1/C5/C6 的定价输入），但：**降级措辞**（「定律」→「本口径下的代价表」）、**补口径声明**（元素在 eval 时开着；报 Δquality 时并列该元素省的/费的算力）、**删除 shuffle 子项**（理由：无部署 regime、不改变任何价值判断）、**benefit 行修正**（把「TTS 基底」改成「TTS 在任何 K 都追不平税，K=8 仍 +0.138 @8× 评估」）。
- 补做（优先 P3）：在 depth-ar 引擎里用「逐 token 退出」重测 skip 的代价（2 臂 ≈ 20 min）。

### ⑤ 正确的验证方式
- **该涨**：带元素部署的 clean bpc（同一元素、同配方、对照不带元素）。
- **该跌**：该元素实际省下的算力（skip：被跳层数占比的实测值；redo：增加的层数），以及墙钟。
- **必须不变**：与对照臂逐字段一致（除该元素）；元素的活性量非零并与训练日志一致（被跳/被重复的层数占比，每 200 步落盘）；`Δ = bpc_对照 − bpc_实验` 的符号全篇一致；Δ<0.014 不排序。
