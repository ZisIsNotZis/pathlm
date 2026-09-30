# 独立复核（契约 L，fresh context，只读）

## Review

对象：契约 I/J/K（`.scratch/31-batch4/issues/01-batch4.md`）交付物，只读复核。

**Correct（已核实为好的部分）**
- **判据线未改**：I §4 与 §2、J §0、K §3/§4 均逐字抄录，比对契约原文一字不差；「逐字/摘要」标注如实。
- **三道对账 + 同一任务定义**：I §6 / J §4 / K §3 齐备，我独立复算算术对账通过（三段 band 求和闭合、保留键数 `mean_i min(k,i+1)`、量化往返上界）。
- **干预活性**：J 每 200 步同行落 `attn_sparse`（61 行，`kept_mean` 恒等于理论值）；I 按配置落 `n_dropped_pairs/dropped_frac`；K 落 `rel_err/changed_frac≈1` 且有 fp32 阴性对照。
- **本轮 a–d 四点全部 PASS**：单调性/复现性有数据（`criteria.a_detail` 无逆序、重跑差 4.94e-07）；删除位置集合逐点正确且变异 M1–M4 全抓红；top-k 确按内容且 k 生效（`sel_in_lastk_frac` 0.77/0.86≪1）；量化确降精度（`changed_frac≈1`、触达 0.5·scale 上界）。
- **守纪**：J 仅改授权文件、默认关闭逐位还原 v1；等参数对照自动满足；变异还原后 sha256 逐字节一致。

**Finding**
- **P1，契约 I `README.md` §7/§11** —— 写「实测墙钟 24.7 s、136 次评测、重跑 25 s」，与 analysis/stability/results/run_log 一致的 **43.4 s / 137 次 / 重跑 43 s** 矛盾。最小修：README §7 改用 43.4 s、137 次。
- **P1，契约 K `README.md` §7 与 `analysis.md` §7** —— 写「全电池 55.7 s（0.93 min）」，与 `results.json::cost`（53.1 s / 0.885 min）及 run_log（0.89 min）不符。最小修：统一到 53.1 s。
- **P2，契约 I `analysis.md` §6.3** —— 出现「37 s 的 137 次评测」，与本文件 §0/§11 的 43.4 s 不一致。
- **P2，契约 J `analysis.md` §5.1** —— 「61 行 × 8 层 = 96,000 个层样本」等式错误（61×8=488）；正确来源是 **12,000 步 × 8 层**（`stability.md` §3 已写对）。数字 96,000 本身正确。
- **P2，契约 J `README.md` §6/`analysis.md` §6** —— 逐项成本相加 33.65 min ≠ 所报 `wall_total_min` 33.21（差额≈成本基准 0.5 min，疑未计入总量）。
- **P2，契约 I `analysis.md` §3/`stability.md` §1** —— 「三次独立进程逐位相同」依据为 `/tmp/rep1`、`/tmp/rep2`，不在仓库内、现不可复核（仓库内可复核的只有同进程重跑 4.94e-07）。
- **P2，流程归属** —— 工作树含 `axes/A-mixing/A6-mixing-operator/experiments/README.md`（自动生成索引，属主 Agent）的改动；无法判定作者，提请主 Agent 统一重跑 `build_wiki.py`（R1 的 `experiments/README.md` 仍为陈旧态）。
- **P2，轻微** —— J README front-matter 仍为 `status: planned`；三目录均无 `controls/`（对照/方差界落在 `results.json::noise_floor`）。
- 我据 `full-12k-s0-train_log.jsonl` 逐点复算 J 的 `_slope` 得 +0.00251，与报告一致（非假数字）；但该斜率是 loss-vs-step 原始最小二乘（正值=回归意义上略升），与端点 1.0501→1.0421 并存，建议注明符号口径。

**未验证（留给主 Agent）**：`python3 -m pytest tests/ -q`（我无 shell，基线 165）；trunk.pt 的 sha256 实算。

**Merge verdict: OK with notes** —— 核心 bpc 结论与判据判读成立；上述均为口径/表述一致性问题，不改变科学结论，P1 两处建议一行修后再入账。