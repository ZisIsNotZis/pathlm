# 独立复核 Y4（ticket 40）：X1/X2/X3

我已完成对 X1/X2/X3 三条线的只读复核（基于磁盘=已提交修订：`watchdog_diff` 仅显示 ticket 40 的 issue 文件 untracked，三条线文件无工作树改动）。以下是复核报告。

## Review

范围：X1（`axes/G-policy/G5-parallel-lines/experiments/2622-decorrelation-v2`）、X2（`axes/M-carrier/M1-carrier-type/experiments/2622-eng1-composable`）、X3（`axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512`）。只读；无 shell；结论基于已提交修订（`watchdog_diff` 对三条线为空）。

### Correct（已核实、正确）

- **① X1 的两个混杂都真的修掉了。**
  - gate：`train_m1.py:27` `return not bool(torch.isfinite(gnorm))` —— 只在非有限时跳；旧 `gnorm>1e4` 阈值已删（docstring 记 42.7% 混杂）。`test_gate_skip_only_on_non_finite` 钉住 `1e9 → False`，变异 M4 抓红（`mutation_report.json`）。四臂 `gate_skips` 实测全 0（`results_13m.json::train_products`、`results_50m.json`）。
  - 罚则形式：`pathlm/model.py:31-53` `repr_decorr_penalty` 对**trunk 隐状态**（`forward` 里 `h = _run_layers(...)` 后的残差流，`model.py:483-488`）做逐维 Pearson 相关的**平方均值**，`return ((cov/scale)**2).mean()`——**无 cap**（旧 capped-−JS 已不作为 v2 罚则；`w_diversity` 与 `w_repr_decorr` 由 `validate_pcap` 禁止同开，单因素）。
- **② 「13M 机制成立 ⇒ 放行 50M」忠实执行了预登记规则。** README §5 预登记规则 = `l1@K8 − arm@K8 ≥ 0.014 ⇒ release_50m`；`results_13m.json::lambda_sweep.repr01` 记 `l1_K8_minus_arm_K8 = 0.0246`（= 1.6454 − 1.6208）`≥ 0.014`（方差界），配对 CI [−0.0281,−0.0208] 不含 0 ⇒ `rule_status: release_50m`；λ=1.0 臂 `close_unorderable`（+0.0059，<0.014）。判据 threshold 与 >方差界均满足，规则字面被忠实执行。
- **③ X2 缺省逐位还原 v1 有证据。** `results.json::default_bitwise_v1`（`weights_bitwise_equal=true`、`forward_bitwise_equal=true`、`12704=12704`）+ `run.py:108-127::default_bitwise_v1` 独立断言 `list(sd_ref)==list(sd_comp)`；`results.json::code.pathlm/depth_ar.py` 记 sha256，`MODEL.md` 记代码指纹。两条旧「互斥」断言确认**只翻成正向断言、判据未动**：`test_kv_heads_guards…`（GQA⊕train_window）、`test_carry_config_rejects_combinations`（carry⊕GQA）改为「现在能建」，引擎侧对应放行（`depth_ar.py` 的 `_gqa × carry_rule → ComposableBlock` 路由 + 两处互斥注释改为「可组合」），其余护栏（`attn_sparse_k`/`attn_linear`/`topology`/`dense`）原样保留并被断言。
- **④ X3「两点同号但带宽分类相反」两种读法都报、n=1 不外推。** `analysis.md §2` 同时给「带宽分类（主口径）：T=256 亏 / T=512 不亏 ⇒ 异号」与「raw Δ 符号：两点都负 ⇒ 同号」；`results.json::criteria.criterion_1_quality.note` 亦声明两读法。`analysis.md §2/§6` 明写「n=1 单 seed」「单点禁外推」，`criterion_4_crossover` 只记「首个观测点」。退化点识别有单测（`test_window256_at_T256_is_identity_and_thus_degenerate`、`test_two_points_have_same_relative_scale`，满足 §17.16）。
- **⑤ 锁台账与 §20.2 记账在位。** X1 `analysis.md §10`：全程 `/tmp/pathlm_gpu.lock`、多臂按臂释放 + `sleep 65`（`train_all.sh`）、并如实登记一次锁目录被外部回收；§20.2 三件套齐全（逐臂 `peak_gpu_mem_mb` 落盘；跑前后 `nvidia-smi` free **8228 / 8203 MiB（≥8GB）**；显式列为超礼貌线偏离，未改 bs/步数）。X3 `train_all_log.txt` 逐臂「释放锁，留 60s 窗口」；X2 仅一次 <1min 冒烟取锁，无违反 §20.1 的整程持锁。
- **⑥ 数字与产物逐一对应（抽查 >10 全部命中，含成本/次数/等式/哈希）**：
  - X1：13M 主判据 `0.1357 = repr@K8 1.6513 − stock@K1 1.5156`、CI[0.1290,0.1432]；并列 `0.0059`；`repr01` `l1@K8−repr01@K8=0.0246`；税 `0.04 / 0.1671`；battery `1.6714 vs 1.6699 = −0.0015`；50M 主判据 `0.0322`、并列 `0.01`、税 `0.0195`、峰值 `5633.4MB`、`gate_skips 0`、`n_repr_observations 12000`；哈希 `e5aea26a…`（13m/repr01）、`7b892d8f…`（50m/repr01）、`3649f076…/e9304612…`（复用 2620）与 `run_log_50m.jsonl` 逐一一致；`train_tokens 98304000`；参数 `13076483/51843075`。
  - X2：qkv `48/96`、`read_share 0.003198/0.004629/0.002974`、`12704=12704`、冒烟 `3.7849→0.7768`（`smoke/master/train_log.jsonl`）、峰值 `735.6MB`、参数 `12616808`、回归 `367 passed`、ENG1 `6 passed`、变异 4/4 抓红且 sha256 还原、reconcile `52/52`。
  - X3：native `Δ256 −0.014836 / Δ512 −0.008730`、release `−0.111682 / −0.233451`、`0.0541/0.1078` vs 公式 `0.1429/0.25`、`r_time 1.0852/0.9113`、峰值 `2.697GiB`、参数 `6962688/7028224(+65536)`、`tokens 98304000` 逐臂、变异 4/4 抓红逐字节还原、reconcile `79/79`、模型 sha256 与 `results.json::checkpoints` 一致。

### Finding

- **P2 — X1 报告引用的对账次数与自身产物矛盾。** `analysis.md:308` 写「`reconcile_numbers.py` → **71/71** 与 **43/43** pass」，但产物 `reconcile_report_13m.json:n_checks=81`、`reconcile_report_50m.json:n_checks=50`（均 `n_fail=0`），且 `progress.md` 记「13m 81/81、50m 50/50」。三处不自洽。**影响**：不改变结论（两次均 0 fail），但违反「报告每个数字与 results.json/日志逐一对应（⑥）」。**最小修法**：把 `analysis.md:308` 的 `71/71`、`43/43` 改为 `81/81`、`50/50`。
- **P2 — X2 新增单测里有一条断言被并进注释而成死断言。** `tests/test_eng1_composable.py:147`：`comp = ComposableBlock(mcfg)                # n_kv_heads=None, carry_on=False    assert list(ref.state_dict().keys()) == list(comp.state_dict().keys())` —— 句末 `assert …` 在 `#` 之后，**不再执行**（疑似脚本改文本时漏了换行）。**影响**：`test_eng1_composable_default_matches_stock_block_bitwise` 仍校验权重逐位与前向逐位，且 `run.py::default_bitwise_v1` 独立断言了键集相等，故「缺省逐位还原 v1」这一性质仍有覆盖、结论不受影响；但该测试比声称的弱。**最小修法**：在 `assert list(ref.state_dict()...)` 前插入换行，使其脱离注释。
- **说明（非正式 finding，仅备忘）**：X3 `reconcile_report.json` 的 check 标签 `"sign_T512 == sign(Δ512)"` 对 `Δ512=−0.008730` 给出 `got/ want = "0"`——其「sign」实为带宽分类码而非 Δ 的符号；`analysis.md` 文本对 raw 符号与带宽分类均正确表述，故仅内部命名易混，无科学影响。

### Merge verdict: **OK with notes**

无 P0/P1。两条 P2 均为文档/测试卫生问题，不影响三条线的科学结论（X1 主判据两档判负、机制在 13M 净成立/50M 不可排序；X2 组合可测且缺省逐位还原；X3 单点、两种读法都报、禁外推）。

### 未能执行的检查（留给主 Agent）
- **pytest**：本复核无 shell 工具，`python3 -m pytest tests/ -q` 未运行。需主 Agent 跑并与 ticket 40 基线 **394** 对齐（X2 的 `pytest_full.txt` 记 ticket-39 当时 `367 passed`）。
- **git 提交 SHA**：无 git 访问，未能枚举 X1/X2/X3 的具体 commit；已通过 `watchdog_diff`（三条线工作树无改动）+ 产物内 sha256 交叉验证「磁盘内容即已提交修订」。