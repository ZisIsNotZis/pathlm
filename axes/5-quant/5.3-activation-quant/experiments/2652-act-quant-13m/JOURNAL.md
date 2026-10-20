# 2652 · 过程流水（JOURNAL，允许日期）

> 纪律：本文件是**过程记录**，不是事实文档；与事实文档冲突时以事实文档为准。
> 事实（当前结论与证据）在 `analysis.md` / `results.json` / 本项 `project.yaml`。

## 0. 角色与边界

- writer = `axis5-w5.3`（5.3 激活量化 WP1，fresh 上下文，**单一写入者**）。
- 只写 `axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/**` 与
  `.scratch/66-axis5-quant/evidence/W-5.3-*.md`。
- 只读复用 `axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/run.py`（内含 2619 的评估管线）、
  `pathlm/**`、`tools/evalkit/**`。**未改任何共享文件**，未动 git 索引。

## 1. 时间线（真实时钟 +08:00）

- `2026-10-14T20:54+08:00` 开工：读 `axes/5-quant/5.3-activation-quant/project.yaml`（本项契约）、
  5.1 的 harness 模板（`2632-kv-block-scale/run.py` / `2621-kv-budget-rest/run.py` + `config.json` / `2619-kv-budget-chart/run.py`）、
  `pathlm/depth_ar.py`（`Block` / `DepthARModel._input_states` / `readout`）、
  `tools/evalkit/kv_quant.py`（量化与活性量口径）、`tools/evalkit/retention.py`
  （`masked_logits` / `masked_bpc`）、`.scratch/66-axis5-quant/TICKET.md`。
- `20:55–20:59` 先写**运行前**产物：`config.json`（含 `criterion`，mtime 早于 `results.json`）、
  新语义层 `aq_arms.py`、可失败单测 + 变异 `selftest_aq.py`。
  设计选择：标尺粒度 = **per-token 动态**（每 `(batch, token)` 一行共享一把 fp16 标尺），
  理由 = 激活难点是「同一 token 内通道量级差数量级」，且 per-token 只在当前 token 内归约 ⇒ 不引入
  跨位置依赖（因果性可证）。
- `20:56` `README.md`（预登记契约：命题 / 单因素 / 指标表 / 判据线逐字 / prior art /
  对照与方差 / 成本 / 数据与脚本清单 / 失败决策树 / 产出）。
- `20:57–21:00` 自纠 3 处实现缺陷（都在跑之前）：
  1. `summarize_stats` 的逐层聚合误用 `v["tag"]`（`per_tensor` 里没有该字段）⇒ 改为从 key 拆 tag；
  2. 权重字节账的手算预期值把 bias 也算进参数量（正确值 12582912，不是 12616704）⇒ 修正单测预期；
  3. `AqForward._act` 原先在 fp16 档直接 `return x`（绕过量化函数）⇒ 改为**所有档都过
     `fake_quant_activation`**，让「fp16 = 恒等」只有一个定义处，变异可覆盖。
- `21:00` `python3 -m pytest -q selftest_aq.py` → **17 passed**；
  `python3 selftest_aq.py` → **7/7 变异全部抓红 + 逐字节还原**（`mutation_report.json`）。
  其中 M7（fp16 档偷偷走一次 fp16 往返）最初**未被抓到**（因为它只破坏 `fake_quant_activation`，
  而当时前向的 fp16 分支绕过了它）⇒ 按纪律「未抓住先修单测/实现」，做了上面第 3 条修改后才抓红。
- `21:02` `python3 run.py --smoke`：等待 GPU 锁 405 s 后取到；工作本身 ≈3 s
  （4 臂 + 仪器自检），峰值 405 MiB。冒烟读数（**小批量，不作结论**）：
  fp16 bpc 1.470954 / A8 1.474095 / A4 5.086626 / W8A8 1.472828 ⇒ A4 在 4 位下崩坏已可见。
- `21:09` 自纠 2 处（跑全量前）：
  1. 墙钟计时把**等锁时间**算进去了（2632 的口径是「wall 从取锁成功后起算」）⇒ 改为取锁后起算，
     另记 `等锁 Ns`；
  2. 额外 4 个窗口 seed 原计划 `n_batches=8`（小批量只为省算力），但工作本身只要 ~20 s ⇒
     改为**与主窗同口径的 24×16**（消除批量混杂，5 个窗口的配对 Δ 统计量同口径）。
- `21:17` `python3 run.py`（全量第一次）：**等锁 600 s 超时**（`/tmp/pathlm_gpu.lock` 被
  `2662-hard-window-anchor-13m` 的两臂训练持有）⇒ 按 §20 与任务书 §5，**不杀任何进程**、如实落
  `lock_blocked.json` 并返回码 3。
- `21:19` 起：改为**有界重试**（后台循环，最多 6 次尝试、每次等锁上限 600 s，臂间 20 s 退让），
  每次把读数追加进 `run_log.txt`；同时把 CPU 可做的产物（`MODEL.md` / 本文件 / `analysis.md` 骨架）
  写完。这也是「锁拿不到时先做完所有 CPU 可做的」的落地。

## 2. 全量读数（已完成 ⇒ 见 §4）

- 4 臂 bpc × 5 窗口、三条召回腿、离群通道判定 ⇒ `results.json` / `table.md` /
  `analysis.md` 已落盘；
- `results.json::gpu_lock`（CPU 口径 `lock_mode=cpu_no_lock`）与 `code` 指纹。

## 3. 转 CPU 口径（2026-10-14T21:32–21:46+08:00）

等锁重试循环的 `attempt 1` 在 21:29:58 等锁 600s **超时**，落 `lock_blocked.json`（证据保留）。
此后 GPU 锁转由轴 2 的 `wp2652-2.2.1-residual-form` 作业持用（name-协议），同时
2642 / 2651×2 / 2652-residual / 2662 / 2671 共 6 个作业在抢锁 ⇒ 释放窗口只剩臂间 ~61s 的缝
（本实验 GPU 全量只要 ~20–30s，但抢不到）。**未杀任何别人的进程、未动 `/tmp/pathlm_gpu.lock`**，
只停本实验自己起的等锁循环（pid 2409135/2409138 等）。

按主 Agent 裁令改用 **CPU 口径**（纯 CPU 工作，`SCHEMA.md` §20 不适用 ⇒ 不取 GPU 锁）：
`python3 run.py --device cpu`。实测 CPU **6.0 s/批**（`cpu_eval_timing.py`，负载均值 27–30 / 12 核；
主 Agent 先前测得 3.4 s/批）⇒ 原预登记 4 臂全量 ≈ 28 min，超出 20 min 预算。
依 README §7 的**预算裁剪条款**（「预算不够时先砍额外 seed 的 n_batches，再砍副腿，
不砍判据线」）与主 Agent 授权，缩批量如下（**判据线不动**：仍 5 窗口 = seed [2,3,4,5,6]、
0.014 带、配对 Δ、Wilson CI）：

| 项 | 预登记 | 本 run | 说明 |
|---|---|---|---|
| `eval.n_batches`（主窗 seed=2） | 24 | 3 | 预算裁剪；bpc 噪声随之变大，仅作描述 |
| `eval.extra_seed_n_batches` | 24 | 2 | 5 个窗口的**窗口数**不变 ⇒ 0.014 带适用条件「≥5 窗口」仍满足 |
| `tasks.copy.n_batches` | 2 | 1 | 召回主腿保留（3 个 src_pos 位置） |
| `tasks.induction.n_batches` | 16 | 1 | 副腿（预登记允许先砍） |
| `tasks.var_recall.n_batches` | 16 | 1 | 副腿（预登记允许先砍） |

`run.py` 的最小改动（只在本实验目录内）：
1. 新增 `--device {cpu,cuda}`（CPU 口径覆盖 `config.json::eval.device`）；
2. 新增 `NullLock`（CPU 口径不取 GPU 锁；`gpu_lock.lock_mode = "cpu_no_lock"`）；
3. 新增 `rec["caliber"]`（`device` / `gpu_lock_held` / `caliber_note`：跨设备不构成判据）；
4. 新增**逐臂持久化**（每臂写完即写 `results.json`，带 `partial` / `arms_completed` /
   `arms_pending`）——防止被切断时丢结果。

**未改**：判据线、臂集合、`config.json::criterion`、任何共享文件、任何别人的进程/锁。

## 4. 全量结果（2026-10-14T21:55+08:00，CPU 口径）

4 臂全跑完，墙钟 **8.54 min**（`results.json::cost.wall_seconds=512.2`；等锁 0 s）。

| 臂 | 激活字节/token | clean bpc（seed2, 3×16） | max\|配对Δ\|（5 窗） | 越 0.014 带? | copy 召回 | 离群回退吃省幅 |
|---|---|---|---|---|---|---|
| `fp16` | 55296 | 1.474621 | — | — | 0.1413 | — |
| `A8` | 27744（省 49.83%） | 1.478151 | **0.003530** | **不越带** | 0.1387 | 0.03% |
| `A4` | 13920（省 74.83%） | 5.040267 | **3.598298** | **越带** | 0.0202（地板） | 0.07% |
| `W8A8`（组合） | 27744 | 1.477052 | **0.002955** | **不越带** | 0.1445 | 0.03% |

- 硬护栏：`bpc_of == masked_bpc`（1.474620661）**逐位 True**；复刻前向 ↔ 库 **逐位 True**；
  变异 **7/7 抓红**；因果性 `past_logits_equal` **全臂 True**；字节账 `n_tensors 48/48`、
  `n_scales 196608` 对账 ✓。
- **与历史锚 1.483569 的差 = 8.95e-03 已定性为「批数口径不同」**（主窗 3×16 vs 5.1 的 24×16），
  非设备问题、非实现缺陷（独立路径 `masked_bpc` 在同批数下与 `bpc_of` 逐位相同）——见 `analysis.md` §2②。
- 离群通道：A8 仅 8/27648 通道 >6×中位数、回退吃标称省幅 **0.03%**；A4 18 通道、**0.07%**
  ⇒ **均远低于「迫使回退」的 10% 阈值 ⇒ 未迫使回退**
  （更正：10% 是 `run.py` 自设的活性阈值，预登记只写「远小于标称」，见 §5 的 N2 与
  `analysis.md` §0/§4）。
- 校验：`python3 tools/check_tree.py` 与 `python3 tools/summary.py --check` 均 **EXIT=0**（46 个
  `project.yaml` 合规）。

## 5. 第 1 轮独立复核 FAIL + 本轮逐条处置（2026-10-14T22:40+08:00）

- 复核报告指针：`.scratch/66-axis5-quant/REVIEW-52-53.md` 的「包 B（5.3）」一节
  （VERDICT = **FAIL**，blocker 3 / normal 4 / info 2；其复核基线 HEAD = `e8f0d4233e45`）。
  本文件不复制判词，只记处置与证据。
- **B1（预登记硬护栏 ② 未达、对照点没跑）⇒ 已修（实测）**：新增
  `python3 run.py --anchor-single-point --device cpu` —— 同一 `bpc_of` 路径，24×16@seed2、
  `fp16` 臂，壁钟 **36.2 s**（纯 CPU，未取 GPU 锁）。实测 **bpc = 1.483568730**，与 GPU 历史锚
  1.483569 的 |Δ| = **2.70e-07**（< 1e-6），同口径独立路径 `masked_bpc` **逐位相同**。
  ⇒ 「主表 3×16 与锚的 |Δ| = 8.95e-03 来自批数口径差」从**推断**变为**实测确认**（设备不是原因）。
  结果写为**新增字段** `results.json::anchor_single_point`，**主表任何已有数字未动**。
- **B2（`eval.note` 的三句口径叙述不实）⇒ 已修**：`config.json::eval.note`、
  `results.json::config.eval.note` 副本、`README.md` §6（并连带 §2 两处、§7 裁剪条款段、
  §9 决策树第 1 条）改为**与实跑一致**：「主窗 3×16 / 附加 seed 2×16；与 5.1 的 24×16@seed2 锚
  **不同批数口径、不可直接比**；判据只用同缩口径、同窗口序的配对 Δ」。**判据线未动**。
- **B3（`anchor_check_applicable` 与自身定义相反）⇒ 已修**：`run.py` 改为按实跑 `n_batches`
  门控（`(not --smoke) and n_batches == 24 and window_seed == 2`）⇒ 本 run 主表为 **false**，
  并新增 `anchor_check_applicable_rule` / `anchor_check_carried_by` 指向 24×16 单点；
  `results.json::instrument.bpc_path` 同步更新（门禁不再只看 `--smoke`）。
- **N1（§2② 把推断写成「原因已定」）⇒ 已修**：改为实测并列表（3×16 / 24×16 / GPU 锚）+
  同口径逐位相同 + 独立旁证（5.2 `2651` 在 CPU 24×16 亦得 1.483569）。
- **N2（10% 阈值被说成「判据里」）⇒ 已修**：`analysis.md` §0/§4 标注该阈值为 `run.py` 自设的
  活性阈值，预登记只写「远小于标称」；并声明该读数对 1%/10%/50% 任何取值都不变。
- **N3（裁剪超出预登记条款、被误述为「依 README §7」）⇒ 已修**：`analysis.md` §5.3 改为如实登记
  「**超出** README §7 预登记裁剪条款的授权范围」（该条款只授权额外 seed 24→8 与砍副腿），
  并列对结论的影响（配对 Δ 有效性不受影响 / 5 窗批数不等 ⇒ 不得称同口径 / 跨 run 锚可比性失效）
  与补齐成本。**本文件 §3 表格上方「依 README §7 的预算裁剪条款」一句是错的，以此处更正为准。**
- **N4（`table.md` 表 1–4 在 Markdown 里不渲染）⇒ 已修**：`run.py::render_table` 的表 1/2/3
  改为整表一次 join（去掉行间空行），并由 `--anchor-single-point` 顺带重画 `table.md`；
  **数字未改**（仅渲染结构）。
- **info I1（横轴张量集表述）/ I2（`results.json::caliber_note` 手工追加）**：本轮**未动**
  （不在返工契约内）。I2 的既有 `caliber_note` 在本次重写 `results.json` 时**被原样保留**；
  新增的 `anchor_single_point` 由 `run.py` 原生生成，不是手改生成物。
- 新增文件：`anchor_single_point.log`（该单点的 stdout 证据）；`run_log.txt` 追加了
  `[单点]` 行（trunk sha256 断言、bpc_of、masked_bpc、完成行）。
- `2026-10-14T22:59+08:00` 订正：第 2 轮独立复核（`.scratch/66-axis5-quant/REVIEW-ROUND2.md`（B）new-normal-2）指出的 normal #2 已订正——`results.json::caliber_note.open_defect` 原写「`config.json::eval.note` 仍写 24×16、本次**未改**、交 5.3 owner 处置」，该说法已过时（`config.json::eval.note` / `results.json::config.eval.note` / `README.md` §6 已由返工改为实跑 `3×16`+`2×16` 口径并声明与 24×16@seed2 锚不可直接比）；现标记为 `closed`，写明已修复并注明第 2 轮复核已核。仅订正该字符串，未动任何数字/判据线/臂集合。
