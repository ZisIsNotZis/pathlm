# 独立复核 V4（ticket 37）：上一波 U1/U2/U3（基于已提交 HEAD 68483da）

我已完成对 HEAD `68483da` 的只读复核。先说明**证据基准**：`.git/refs/heads/main = 68483da56cfe…`（commit "36 第九批收官…"），且 `watchdog_diff` 显示工作树对 tracked 文件**无改动**（仅一个 untracked `.scratch/37-gaps/issues/01-gaps.md`）⇒ 我读到的三份证据文件即 HEAD 版本，V1/V2/V3 的编辑未进入我的读数。（注：`watchdog_diff` 不能检查已提交区间，我是直接读文件 + 用 git ref 确认修订。）

## Review

### A. `SCHEMA §14` 七项

| # | 项 | 判定 | 证据 |
|---|---|---|---|
| 1 | 判据线是否被改过 | **PASS** | 逐条对照源契约 `.scratch/36-parallel/issues/01-parallel.md:32-47`：U1「单调或明确拐点 + 交换率 ≥3 实测点，非单调/CI 不可分则如实报不可分辨」、U2「≥+3pp 且 CI 不重叠，否则记 13M/12k 不成立」、U3「图+表、解析/实测不得混列」。三份 `README.md §4`（U1:…/U2:…/U3:…）逐字保真，操作化（c0/c2/c3、C1–C7）均落在契约允许范围内。 |
| 2 | §12 三道对账 | **PASS（实质），U1 无显式小节** | U2 `analysis.md:226` §11.2、U3 `analysis.md:139` §3 显式给出算术/任务/口径三道且通过；U1 无标题小节，但实质三项俱在（§3 毁源+局部先验=任务对账、§4 的 ×chance/CI=算术、§12 口径声明=口径）。**但** U1 的算术对账漏掉了 `dh*(K32)` 错误（见 P2-4）。 |
| 3 | 对照臂可比性（§13 三要素） | **PASS（U1 格式弱）** | U3 `controls/README.md:8-14` 三要素表；U2 `controls/README.md:8-11,21-25`；U1 `controls/README.md` 说明任务/预测/用途，但未按「输入分布/要预测什么/部署对应」逐臂列表（实质可比，格式缺）。 |
| 4 | 干预活性（§10） | **PASS** | U1：`read_share 0.667–0.753`、`β≈0.492`、`state_fro` 4 臂各值、step0 read_share=0→逃逸（`analysis.md:219-241`，`train/*/train_log.jsonl` 51 行/200 步）；U2：`acc_symbol` step0 0.0049→末 0.9897、`head_gnorm_mean 0.007064`、λ=0 臂 `grad_flowed=null/alive=false`（`train/B/train_log.jsonl:61`）；U3：`rel_err` 1.8e-4/4.8e-3/8.8e-2、`dropped_frac`、GQA `k_batch_shape [4,2,256,32]`。 |
| 5 | 证据可回溯 | **PASS（有例外）** | 绝大多数数字可回 `results.json`；例外见 P2-1…P2-4（成本/等式四处对不上）。 |
| 6 | 口径标注 | **PASS** | U1「同一预算下的横向比较，非收敛值」；U2「快照 matched-step 12k」；U3「解析/实测分列、bpc 非主判据」。 |
| 7 | 回归 pytest | **未运行（我无 shell）** | 自报 294（U2 `analysis.md:283`）。需 supervisor 执行 `python3 -m pytest tests/ -q`（期望 294 绿）。 |

### B. 本轮特别关注 ①–⑧

| # | 项 | 判定 | 依据 |
|---|---|---|---|
| ① | 曲线直接测点、未外推 | **PASS** | 难度曲线 4 个直接测点（dh=32，K=4/8/16/32）；容量曲线 4×4 个直接测点；交换率只有 1 个实测点，明确写「不拟合/不外推」（U1 `analysis.md:171-186`、`results.json::exchange_rate.n_distinct_measured_points=1`）。 |
| ② | 容量「不成立」措辞 | **PASS** | U1 `analysis.md:44-46`、`analysis.md:326`、`catalog.yaml:1010` 全部用「该任务/该尺度上加大状态宽度不划算」并显式写「不得外推为容量机制不可能」，另立「训练预算不足」假说为当前最有解释力且**未经判决**。 |
| ③ | U2 λ=0 / fp32 / pre-step grad-gate | **λ=0 PASS、gate PASS、fp32 部分** | λ=0：`add_recon_term` 原样返回同一 loss 对象（`train_depth_ar.py:41-42`）+ `test_recon_lambda_zero_returns_same_loss_object` + `test_recon_head_dead_at_lambda_zero…` + 单进程自检 `trunk_bitwise_equal=true / logits_bitwise_equal=true / added_tensors=[recon_head.weight]`（`results.json:136-145`）。gate：`train_depth_ar.py:290-299` clip 前取范数、`>1e4`/非有限则 skip，recon 项在 backward 之前已加 ⇒ 同一门覆盖。fp32：仅**头输出 logits** 被钉（`recon_signal.py:130`、`test_recon_logits_is_fp32_under_bf16_autocast`），`F.cross_entropy`（`recon_signal.py:151`）在外层 bf16 autocast 区内、无测试钉其 dtype ⇒ 见 P2-7。 |
| ④ | U3 毁源 ERRATA 是否真剔除虚高读数 | **PASS** | `run.py:540-545` 改为按各档源区间毁源；`diagnose_copy_confound.json::span_sensitivity` 复现存档（96 档 0.15625→0.013672、191 档 0.132812→0.013672，与 `analysis.md:186-195` 逐位一致）；`table.md` 表 2 毁源列全为修正值 0.0104–0.0163；`catalog.yaml:1172` 记录 ERRATA 并写「未进任何结论」。 |
| ⑤ | U3 横轴口径统一 | **PASS** | 单一解析式 `2·L·kv_heads·head_dim·value_bytes·(kept/T)`，14 点逐点验算（`analysis.md:146-152`），解析列与实测显存 430.5 MiB 分节（`results.json:5819-5823`）；14 点不含任何 sink/锚点，锚点单列进缺失组合 ⇒ 口径自洽。 |
| ⑥ | 是否把 bpc 当主判据 | **PASS** | U1/U2 主判据 = 准确率（U2 `README.md §4` 明写「禁止把 bpc 当主判据」）；U3 `README.md §3` 明写「bpc 不是主判据」，窗口档结论由召回腿定案（192 档 bpc −0.0042 但召回 −25%）。 |
| ⑦ | 成本/次数/等式与 `results.json` 逐一对应 | **FAIL** | 4 处对不上，见 P2-1…P2-4，且三份 `reconcile_numbers.py` 的 `num_in()` 是**子串匹配**，用弱舍入位（`"32"`、`"0.8"`）假通过。 |
| ⑧ | GPU 锁违规是否影响结论 | **PASS（不影响），事件已自述** | 已知两条协议 bug 不重复报。影响判定：U3 与 U1 训练**确曾时间片重叠**（U3 `analysis.md:224` 自述 `pgrep -af train_depth_ar` 显示 U1 在跑），但 U3 为纯推理、无采样，追加复现跑 `max|Δbpc|=0、max|Δcopy|=0`（`results.json:5933-5936`）⇒ 未观察到污染；U1 训练 19:44:21 结束、U2 训练 19:49 起，**训练之间无重叠**；峰值 2.38/2.02/0.43 GB 均 ≤5GB ⇒ 无结论影响。 |

### Finding

**P2-1 · U3 墙钟与 `results.json` 不符（⑦）**
`axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/analysis.md:8` 与 `:205` 写「GPU 墙钟 **32.7 s**（`results.json::cost.wall_seconds`）」，但 `results.json:5939` 为 `32.11`（`rerun/results.json:5832 = 32.33`）。对账检查 `reconcile_numbers.py:165` 只用了 `.1f` 档的 `"32.1"`，最终以子串 `"32"` 通过（`reconcile_report.json:1255-1258`）⇒ 机检未抓住。
最小修法：把 analysis 两处的 32.7 改为 32.11，并把 `num_in` 的候选收紧（禁止落入无小数点整数档）。

**P2-2 · U2 评测墙钟与 `results.json` 不符（⑦）**
`axes/F-iteration/…/2619-reconstruction-loss-13m/analysis.md:220` 与 `:292` 写「评测 **1.1 min**」，但 `results.json:2040` 为 `wall_minutes_total_eval = 0.82`，`run_log.txt:40` 亦为 0.82（连带「总 ≈31 min」偏高）。对账 `reconcile_numbers.py:181` 以子串 `"0.8"` 通过——它命中的其实是 §3 里 smoke 的 loss`0.8147`（`analysis.md:139`）。
最小修法：把 1.1 改为 0.82（或注明 1.1 = 两次 eval 的挂钟含等待），并把该检查改成必须匹配 `.2f`。

**P2-3 · U1 前向批数等式写错（⑦）**
`axes/M-carrier/…/2619-carry-difficulty-capacity/analysis.md:262`：「1,280（= **64 bs ×** 40 批 × 4 档 × 4 臂 × 2 主/毁源）」，左式 = 81,920 ≠ 1,280。正确为 `40×4×4×2 = 1,280`（`results.json:2352 n_forward_batches=1280`，`config.json:32-33` bs=64/n_batches=40）。
最小修法：删掉 `64 bs ×`（bs 是每批规模，不是批数）。

**P2-4 · U1 交换率「反向口径」与 `results.json` 不符（⑦）**
`analysis.md:187` 写「`dh*(K)`… K=4/8/16/32 **全部 = 16**」，但 `results.json:1684` 为 `K32: dh_min_passing = 0, arms_passing = []`（K=32 上四臂 CI 下界 0.0542–0.0547 均 < 0.0625，**无臂通过**）。这不只是数字错：按实测反向口径有 {16,16,16,无} 两个取值，因此「反向也只有 1 个测点」这句推论不成立。
最小修法：改为「K=4/8/16 的 dh*(K)=16；K=32 无臂通过（未定义）」，并删去「只有 1 个点」的反向论断。

**P2-5 · U1 字号（×chance 口径混用）**
`analysis.md:28`「4 对时远超门槛（**8.6× chance**）」，而同一句其余三值（23.4×/6.9×/3.6×）与 §4 详表（34.2×）都是 ×chance 口径；8.6 实为「× 门槛(=4×chance)」。属标题数字口径混用，数值本身（0.535）无误。
最小修法：改为「34.2× chance（= 门槛的 8.6 倍）」。

**P2-6 · U3 `results.json` 缺契约承诺的机读块**
`README.md:85` 的 C7 要求三道对账「落进 `results.json::reconciliation`」，且 C1–C7 应有 `criteria` 旗标；实际 `results.json` 顶层只有 `config/instrument/controls/cross_check_vs_source/points/missing_combinations/memory_convention/code/reproducibility/cost/chart_png`（无 `criteria`/`reconciliation`），对账仅存在于 `analysis.md` §3 散文。
最小修法：把 §3 三道的判定与 C1–C7 旗标补写进 `results.json`（不改结论）。

**P2-7 · U2「fp32」证据只覆盖头输出（未验证项）**
契约（`README.md §3`）要求重建项在 `autocast(enabled=False)` 内算（fp32）。实测仅钉住 `recon_logits` 的输出 dtype（`recon_signal.py:130`、`tests/test_recon_signal.py:170-186,178`），而 `F.cross_entropy`（`recon_signal.py:151`）位于**外层 bf16 autocast 区内**。本环境无网络、torch 源码不可读，我**无法离线确认** PyTorch autocast 是否会把 `cross_entropy` 降精度 ⇒ 记为**未验证**（若会降，则契约的 fp32 只兑现了一半；对稳定性目标的实际影响小：`grad_skips=0`、`gnorm_max 92.91/93.69/138.07` 未发散）。
最小修法：加一条可失败单测（在 bf16 autocast 下断言 `stats["loss"].dtype == torch.float32`），或在 `recon_loss` 内把 CE 也包进 autocast-off。

**P2-8 · U2 smoke「完全一致」略过头**
`analysis.md:138-140` 称 `smoke_A` 与 `smoke_A_off` 逐步日志「完全一致」。`train_log.jsonl` 的数值字段确实一致（step0 loss 4.7499 / step19 0.8147 / gnorm 66.07→1.76），但两文件 step19 的 `min`（挂钟）不同，且 stdout 的 kv 遥测在第 6 位小数量级不同（`1.769341` vs `1.769344`，`kv_head_pair_max_abs_weight_diff 0.131423 vs 0.131427`）——跨进程 CUDA 归约的非确定性。λ=0 等价性的**强证据**是单进程自检（`trunk_bitwise_equal`/`logits_bitwise_equal`），不依赖此句。
最小修法：把「完全一致」限定为「损失/梯度/答案 CE 逐步一致（挂钟与权重范数遥测除外）」。

### 可疑数字清单（汇总）

1. `2619-kv-budget-chart/analysis.md:8,205` = 32.7 s ↔ `results.json:5939` = **32.11 s**
2. `2619-reconstruction-loss-13m/analysis.md:220,292` = 1.1 min ↔ `results.json:2040` = **0.82 min**
3. `2619-carry-difficulty-capacity/analysis.md:262` 等式 64×40×4×4×2 = **81,920** ≠ 1,280
4. `…/analysis.md:187` 「K=4/8/16/32 全部 = 16」↔ `results.json:1684` K32 = **0**
5. `…/analysis.md:28` 「8.6× chance」↔ 同句×chance 口径应为 **34.2×**
6. 机检弱点：三份 `reconcile_numbers.py` 的 `num_in()` 子串匹配，使第 1、2 项以 `"32"`/`"0.8"` 假通过（`reconcile_report.json:1255-1258`；U2 `reconcile_report.json:648-651`）

### Merge verdict: **OK with notes**

- **结论层（U1/U2/U3 的机制结论）全部 pass**：难度曲线成立、容量曲线/交换率在本任务·本尺度不成立（措辞合规）、U2 重建信号不成立但显式通道成立、U3 统一预算图与 ERRATA 可信；无 P0/P1。
- **报告/对账层 ⑦ = fail**（4 处成本/等式与 `results.json` 不符，2 处被弱子串放行）。按本批「fail 的结论不计入状态灯」规则：这 4 处**不影响**机制结论，但账本条目若要翻成「已确认」，建议先把 P2-1…P2-4 落盘修掉（都是改字，不需重跑 GPU）。
- 需要 supervisor 执行的命令：`python3 -m pytest tests/ -q`（期望 294 绿）；`python3 axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/reconcile_numbers.py`、`python3 axes/F-iteration/F4-training-signal/experiments/2619-reconstruction-loss-13m/reconcile_numbers.py`、`python3 axes/M-carrier/M1-carrier-type/experiments/2619-carry-difficulty-capacity/reconcile_numbers.py`（三份均报 0 failed，但**正是它们漏掉了上表 1–4**，修 P2-6/P2-1/P2-2 的匹配器后才算有效门禁）。