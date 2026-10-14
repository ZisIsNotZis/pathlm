# 待派 WP（Phase D · 排队中）：把 4 个 `mutate.py` 接入统一变异框架

**来源**：仓库测试 `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness` 当前**红**，
列出的未接入脚本**全部属于本轴**：
```
axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m/mutate.py
axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/mutate.py
axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/mutate.py
axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/mutate.py
```
（`.scratch/67-engine-changes/TICKET.md` 的 engine-writer 记录与本轴自查一致；该红项由**本轴**引入，须由本轴修。）

**测试要求**（`tests/test_mutation_framework.py:145`）：
1. 源码含 `from tools.evalkit import mutation`；
2. 源码**不得**出现 `subprocess.run(`（改用 `mutation.pytest_run(...)`，它同形返回 `returncode/stdout/stderr`，
   并把 collection/environment error 直接抛 `ClassifiedError` ⇒「rc != 0」恰好等价于「抓红」）；
3. 收尾须有「未抓到 ⇒ 非零退出」路径（`mutation.exit_if_uncaught(ok, report, report_path)` 或既有 `raise SystemExit` / `sys.exit(0 if ...)`）。

**已有合规样例**（只读参考）：`axes/1-time/1.1.1-hard-window/experiments/2622-longT-512/mutate.py`
（`from tools.evalkit import mutation` + `mutation.inprocess_outcome` / `mutation.exit_if_uncaught` 的写法）。

**约束**：只改这 4 个 `mutate.py`（**不改被测仪器** `freerun.py` / `padfill.py` / `latent_reg.py` / `mdiff.py`，
不改契约 `README.md`，不碰 `pathlm/**`、`tools/**`）；改后每个脚本**必须重跑一遍**并保持原语义：
≥3 条变异全部**抓红** + **逐字节还原** + `mutation_report.json` **确定性**（连跑两次哈希相同）且**不含逐次运行耗时**。

**验收**：① `python3 axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m/mutate.py` 等 4 个脚本逐一跑通
（输出 `all_caught=true` / `all_restored=true`）；② `python3 -m pytest tests/test_mutation_framework.py -q`
**全绿**（该红项归零）；③ `git diff --check` 干净；④ 4 个报告连跑两次哈希相同。

**边界**：只写这 4 个 `mutate.py`（+ 各自 `mutation_report.json`）；不得顺手改别的文件。
