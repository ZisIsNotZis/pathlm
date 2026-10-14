# WP 6.2.3 执行日志（2650-verify-gate-13m）

只增不改。actor=worker（fresh 上下文，单一写入者）。时区 +08:00。

## 契约与口径

- 唯一权威契约：`axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/README.md`（不得修改）。
- 只读复用：2640 `spec_tree.py` / `run.py`（sys.path import）；checkpoint `../2631-spec-three-arms/train/{13m_s0,13m_s1,53m_s0}`。
- 硬边界：只写本实验目录 + 本日志；不改 `pathlm/**`、`tools/**`、`docs/**`、2640 目录。

## 时间戳流水

- 2026-10-14T09:20:13+08:00 开工。读完 2650 README、2640 spec_tree.py/run.py/test_spec_tree.py/mutate.py；确认 2640 arms 结构（native/direct1/chain1/combined_fixed/combo_pw/tree_* /oracle_accept/tree_topk4，per_prompt.accepts 可逐位比对）。

- 2026-10-14T09:2x:xx+08:00 【口径观察，待报主管】README §4 V5 写的 `R_K = 3K·(1+γ)/(1+τ_K)` 与 2640 已落盘口径 `R = B_verified·(1+γ)/(1+τ) = K·(1+γ)/(1+τ) = 3K/(1+τ)`（B_verified=每轮行数=K，3K=每轮主干位置数）相差一个因子 3。**未擅自改口径**：本实验同时落 `R_K` 两种算法（`R_K_2640consistent = 3K/(1+τ_K)` 与 `R_K_readme_literal = 3K(1+γ)/(1+τ_K)`），台账如实两种并列，并请主管裁决。

- 2026-10-14T09:3x+08:00 落盘 `verify_gate.py`（通用 K 行树验证主循环 `decode_gate` + 对抗/激进候选构造函数；只读 import 2640 `spec_tree.py`/`run.py`，后者按显式路径加载为 `run2640` 避免与本目录 `run.py` 撞名）、`run.py`（薄入口）、`test_verify_gate.py`、`mutate.py`。
- 2026-10-14T09:3x+08:00 CPU 单测：`9 passed, 2 skipped`（V0 与禁墙钟两条待 results.json 生成）。变异：M1 行↔候选映射错位 / M2 接受行回写错行 / M3 无条件接受，`all_caught_red=true, all_restored_bitwise=true`。
  - 注：随机初始化微型 `PathLM` 贪心流会塌缩成常量（无判别力）；单测里把 embedding 行归一化（`tiny_model`）以得到非退化可判别 fixture。M2 的判别用手写真值置末行 + `tree_forward`/`write_back_accepted_row` 双 spy 再生接受行索引。

- 2026-10-14T09:27:28+08:00–09:34:09+08:00 GPU 正式跑（`run.py --lock ckpt`，与并发 agent wp6.3.3-p1 按 §20 逐 checkpoint 交替取放锁)：13m_s0 全 9 臂 + 13m_s1 子集 6 臂；整程 401s（含 60s 让出窗口 + 45s 等锁）。
  - V0 ✅（repro_topk4/repro_combo_pw 与 2640 逐位同值）；V1 ✅（全臂发出流逐位==native，n_forward==n_rounds）；V2 ✅（adv_random α₁≈0.0185，adv_bottomk4=0）；V3 ✅（single_k1_ctl==dup_k4==combo_pw）；V4 未饱和（K=4→8 Δα₁=+0.047/CI[0.030,0.063]）；V5 K=4→8 Δτ/ΔK=0.019 < 阈值 0.058 ⇒ 不划算；V6(g) win64 抛 NotImplementedError（未实装）。
  - 发现并修 `v5_ledger` 一处 int/str 键 bug（`k4_to_k8_threshold` 曾为 None）；因该量纯由已落盘 τ 派生、无重新测量，用 `recompute_v5.py` 离线回写（CPU）。
- 2026-10-14T09:3x+08:00 smoke 为 §20.1 并发例外登记：GPU 空闲 22.5GB / 自身峰值 <1GB，--lock none 跑 2 prompt×32 token（13s），不计入正式锁序列。
- 2026-10-14T09:3x+08:00 落盘 `analysis.md`；CPU 单测再次 `11 passed`。

- 2026-10-14T10:22:06+08:00 【修缺·REVIEW-B1 N1/I2】`results.json::config.budget_min` 原为 **runner 缺省值 25.0**，与契约 README 前沿
  「GPU ≤15 min 硬时间盒」冲突；已更正为**契约值 15.0**（**不改任何读数/判据**）。`run_log.txt` 末尾追加同样说明
  （GPU 实耗 401 s 只在 run_log.txt；`results.json` 无墙钟字段）。
  I2：`analysis.md`「口径与边界」注明 `criteria.V5_cost_ledger` 是**离线派生字段**（非 run 直接产出），指向 `results.json::v5_recomputed_note`。
  修后：`python3 -m pytest test_verify_gate.py -q` → **11 passed**；`python3 mutate.py` → **M1/M2/M3 全抓红 + 逐字节还原**；
  `python3 tools/summary.py --check` → 全绿。**未改** `2650/README.md`（axis6-lead 拥有）。`git add` 未执行。
