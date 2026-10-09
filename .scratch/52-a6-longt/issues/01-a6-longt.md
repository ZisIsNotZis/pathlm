# ticket 52 · A6 长 T 统一口径（52-A6U）+ 线性注意力质量腿（52-A6L）+ 复核修复

> 本单由执行者于本批创建（复核 P1-10 指出：实验 front-matter 的 `ticket: 52-A6U`/`52-A6L` 此前无对应单）。
> 契约与判据见两个实验目录的 `README.md`（预登记，未事后改）。

## 背景与动机

`.scratch/50-critical-audit/A-AUDIT.md` 对 A6 的长 T 系列给出三重缺口：
① 四点口径不齐（步数 24000/12000/6000/3000，lr 以 steps 为分母 ⇒ 退火刻度不同）；
② 「省不省」没有同会话四腿（质量/解析 FLOPs/实测墙钟/峰值显存）；
③ 线性注意力**质量腿空白**（`2616` 三次停摆，只有成本数据）。

## V1 · 52-A6U 长 T 四点统一口径（目录 `experiments/2636-longT-unified/`）

- 干预：窗口（`train_window ∈ {0, T/2}`）× T ∈ {256,512,1024,2048}，**同 step S=6000**，
  全臂套 `qk-normfrom512`。
- 新跑：T=256/512 两点的 full/window 各一臂 × seed {0,1}（8 臂）；T=1024/2048 只读复用
  `2626`/`2629` 的同 step(6000) 点（逐字段等价见 `reconcile_report.json::cross_run_field_compare`）。
- 预登记判据：C1 质量形状（同 step 主口径 + 同 token 副口径）；C2 稳定性前置；C3 四腿；
  C4 噪声门限本格自测；C5 口径对照（不得混用）。
- 结果（见 `analysis.md`）：同 step 四点 Δ 全带内；同 token 副口径短 T 越界；两口径之差
  （T=256 **0.012210**、T=512 **0.010573**）**超过**本格判据量噪声（**0.009459 / 0.006231**）
  但 **≤ 工程带 0.014** ⇒ 两口径在短 T **可测地不同**、不得择优/合并；旧结论「窗口代价随 T 消失」
  在本 n 下**不可判定**。
  四腿：**解析 FLOPs 省；实测墙钟无稳定收益（T≥1024 两轮一致更慢，T≤512 测不出方向）；峰值显存更费（1.169→3.305×）**。
- 成本：训练 ≈ 4 臂 seed0（≈16 min）+ 4 臂 seed1（≈16 min）+ 冒烟/成本探针/评测（≈5 min）；
  峰值 ≤0.77 GiB（训练臂）；T=2048 窗口复用点 5.58 GiB（§20.2 记账，非本实验新跑）。

## V2 · 52-A6L 线性注意力质量腿（目录 `experiments/2637-linear-attn-quality/`）

- 干预：注意力算子 ∈ {full, linear}；另加一档「稳定化分母」（全局 QK-Norm，零参数）做 2×2；
  T=256/bs8/seed0/6000 步。
- 预登记判据：C1 质量 Δ（带宽 0.014）；C2 成本四腿；C3 停摆诊断（触发步/栈/诊断矩阵）。
- 结果：**原实装·带伤** Δ=−0.214107（den 自 step 800 贴 clamp、gnorm_max 899385、25 跳步）；
  **稳定化·核已改** Δ=−0.624194（gnorm_max 7.51、跳步 0、den_min 5.97）。
  **停摆在本配置（bs8/6000 步/f2e72361）下不复现**，但 `2616` 的 bs32×12k 配置**未复跑**（触发条件未分离）。
  成本 2.78× 慢、峰值显存 1.27×。
- 未产出：`smoke/smoke.json`（preflight 未单独跑）、`diag_matrix.json`（停摆未复现，边际信息低）。

## V3 · 复核与修复（本单的第三段）

- 复核（fresh context，只读）结论：**BLOCK — 1 个 P0 + 8 个 P1**，核心新结论方向成立、数字可复算。
- 修复（本批落地）：
  - **P0-1**：成本腿墙钟列此前用了**不存在**的轮2 值（1.234/1.005/0.883/1.799）；改为
    `eval.json` 实测（0.992/0.915/1.069/1.784），结论改为「无稳定收益；T≥1024 一致更慢」。
  - **P1-1**：欠训练论证改用**同 T** 证据（`full-256-s6k` 末态 loss 1.4157 vs `2626 full-256-clean` 1.218），
    删去跨 T 绝对 loss 论证（`2629` 已撤回过同一论证）。
  - **P1-2**：噪声门限改用**判据量 Δ 的跨 seed 极差** 0.009459（单臂 0.0158 降为参考）；
    并标 T=1024/2048 无 seed 复本。
  - **P1-3**：「口径敏感」降级为**未排除解释**；两口径之差（0.0122/0.0106）**超过**本格判据量噪声
    （0.0095/0.0062）但 ≤ 工程带 0.014 ⇒ 短 T 可测地不同、不得择优/合并。
  - **P1-4**：合成腿更正为「**用的正是** `var_recall_batch`（末位布局，已判伪影）⇒ 读数不可解释」。
  - **P1-5**：运行期指纹与报告期指纹分列（`code_fingerprints_at_run` / `_at_report`；新增 `run_fingerprints.json`）。
  - **P1-6**：跨 run 逐字段对照落成机器可读（`reconcile_report.json::cross_run_field_compare`）。
  - **P1-7**：2637 两档改名（「原实装·带伤」/「稳定化·核已改」），删「≥0.21」「下界/上界」。
  - **P1-8**：`2616` 的改判加范围限定（本配置不复现；bs32×12k 未复跑）。
  - **P1-9**：catalog A6 三处 SSOT 自相矛盾一并修（verdict / notes / 历史 failure 条）。
  - **P1-10**：本单（此前缺失）。
  - **P2**：八臂 gnorm 表补全（6.55–7.70）；FLOPs 公式更正为 `512w/(512T+786432)`；2637 §3 补 step0 7.53；
    产物清单同步；逐臂 `train_log.jsonl` 入库；仪器单位与轮次口径更正；证据锚点更新。

## 产物清单

- `axes/A-mixing/A6-mixing-operator/experiments/2636-longT-unified/`（README/run/reconcile/mutate/analysis/
  MODEL/results/eval/config/smoke + 八臂 train_summary + train_log + run_fingerprints）
- `axes/A-mixing/A6-mixing-operator/experiments/2637-linear-attn-quality/`（README/run/linear_driver/reconcile/
  mutate/analysis/MODEL/results/eval/config + 四臂 train_summary + train_log）
- `tests/test_longT_unified.py`、`tests/test_linear_attn_quality.py`
- `catalog.yaml` 的 A6 段 + 生成页

## 2 红 pytest 的归属（登记）

本批结束时 `pytest tests/ -q` = 808 passed / **2 failed**：`tests/test_carry_seed2_13m.py::
test_reuse_equality_report_verdict_ok` 与 `::test_engine_files_behaviorally_unchanged_since_2631_run`。
根因 = **他线（R7）在 2026-10-10 01:56 的提交 `b8b78fc`** 改了 `pathlm/depth_ar.py`（+127），
使该测试「引擎自 2631 未变」的前提失效；**与本批改动无关**（本批提交未触及任何引擎文件或他人测试）。
开工基线（HEAD `e4fc8dd`）= **692 passed 全绿**。
