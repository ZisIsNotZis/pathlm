# 派单补充 · 5.6「组合：窗口 + 量化 + 检索」（Phase C，CPU 口径）

> 与 `.scratch/66-axis5-quant/DISPATCH-5.6.md`（Phase B 写的 WP1 契约）**联用**；冲突时以本文件的补充为准。
> 背景：Phase B 未推进 5.6；本轮按 Main 要求用 §20 **queue-behind** 推进 WP1/WP2。

## 覆盖项（相对 Phase B 契约）
1. **口径 = CPU**（F5 已批，含 4 条约束）：全臂同设备、同窗口序、**只报配对 Δ**；跨设备绝对水平**不得**作判据；
   如实落盘 `fp16`/`full` 臂在本机复现 GPU 历史锚（`1.483569`，5.2/5.3 已实测 `1.4835687~1.483569`，`|Δ|≈2.7e-07`）作为等价证据；
   `cost_model` 只用等效理论量。已实测成本：`load_model≈1.2 s`、`16×256` 一批 **≈3.4 s** ⇒ 主表 24 批 ≈83 s/臂、配对窗 4×8 批 ≈110 s/臂。
2. **不取 GPU 锁**（纯推理；`SCHEMA.md` §20 不适用）：`results.json` 记 `device:"cpu"` + `caliber_note`（含 `gpu_lock_held:false` 与 `gpu_lock:null` 的声明式说明）。
3. **逐臂持久化 + 能救多少救多少**：`run.py` 每跑完一臂即写 `results.json`（`partial`/`arms_completed`/`arms_pending`）；
   子 Agent 有 **30 分钟硬上限**（Phase B 已有 3 个 writer 被切断丢结果）⇒ **先保 `results.json` 与 `table.md`，再写 `analysis.md`**；
   被切断时**另起 detach monitor** 保证产物落盘。单条 bash 命令总时长 **≤180 s**；轮询 `sleep` ≤60 s。
4. **判据带**：主判 = **0.014 bpc 带，限「配对 Δ、≥5 窗口」**；主表 `window_seed=2`（与历史可比）+ 另跑 `[3,4,5,6]`（可缩批）⇒ 5 窗口。
   做到 ≥5 窗才算标准口径；做不到就显式声明「本格自测在力带 = 单臂跨窗口 bpc 极差（给数值）」并声明两带不混用。
   长程检索召回等**率类**指标走配对 bootstrap / Wilson 95% CI（对照 chance = 0.0156 与同掩码地板），**不跨量纲借用 0.014**。
5. **量化腿口径**：一律 `5.1` 的**块标尺**（`tools/evalkit/kv_quant.py::quantize_kv(scheme="block", G=32)` ⇒ int8 1.0625 B/值 = 6528 B/位置）；
   **禁止**引用已作废的逐值标尺数字。
6. **同字节水平**：组合臂应落在 `5.1::2642` 已给出的 7 个同字节水平（432/768/816/864/1536/1632/3072 B/位置）之一，
   或**明确另立水平并说明理由**；**禁插值**。
7. **WP1 的取数策略（省时间且不造假）**：现有落盘点在 `2614`/`2615`/`2619`/`2620`/`2621`（用 `b21.SOURCE` 找来源）——
   **允许只读复用**，但必须①在 `results.json::points[*].source` 注明来源；②对**复用的点**在同一口径下**自己重跑一遍**（或在 `results.json` 里显式标 `reused:true` + 说明「未重跑」）；
   ③把**缺失的保留预算点**（keep32→192 里没有的）自己跑出来。**不得**把「兄弟线的单点重绘」当新发现。
8. **两腿并报**：clean bpc + 合成长程召回（copy/induction/var_recall），`at_floor` 只读作「该任务上不可分辨」，不得写成「机制无效」。
9. **必须回答**（契约 `falsification`）：① 窗口 bpc 损伤是否随保留预算单调（keep32 ≈ +0.212 → keep192 ≈ +0.004，与旧「窗口几乎零代价」判读冲突 ⇒ 旧判读被否证）；② 窗口+量化组合在**等字节**下是否优于单用其一；③ 与 M2 ERRATA（keep32=+0.212、keep64=+0.081）逐数字一致。

## 边界（不变）
只写 `axes/5-quant/5.6-window-quant-retrieval/experiments/**`（可另写 `.scratch/66-axis5-quant/evidence/W-5.6-*.md`）。
禁止写/改 `project.yaml`（事实文档由 axis5-lead 写）、其它轴、`docs/**`、`pathlm/**`、`train_*.py`、`tools/**`、`tests/**`；
禁止 `git add/commit`；禁止 GPU 与 `/tmp/pathlm_gpu.lock`。
自测 ≠ 独立验收；复核由 axis5-lead 另派 fresh reviewer。
