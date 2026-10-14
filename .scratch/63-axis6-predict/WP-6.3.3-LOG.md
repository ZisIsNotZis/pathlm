# WP 6.3.3 阶段-1 LOG（只增不改）

契约（预登记，不得改）：`axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/README.md`
写入范围：`axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/**` + 本文件。

## 2026-10-14T09:20:40+08:00 · 开工
- actor: WP6.3.3 阶段-1 worker（单一写入者，fresh 上下文）
- 读契约、`pathlm/model.py`、`2631` 主干与 config。关键口径确认：
  - `h` = 最终层残差流；单位置解码 `Decoder.step` = `cap_norm(embed+pos_embed)` → 12 层 → `_mtp_nodes(h)`，
    故 `nodes[0]["latent"] == h`（`pathlm/decode.py:85-105`）。
  - 主干 node 语义：node k 在行 i 预测 `t_{i+k}`（`model.py` docstring）；原生逐 token 解码用 **node 1**（`decode.py:399`）。
  - 契约 §2 的 `z_j → t_{i+1+j}`，j∈1..k ⇒ k=2 头预测 **t+2、t+3**（与 6.2 的 d2/d3 草稿位对齐）。
    `ar_t1` = 冻结主干 node 1 的 t+1 命中率（原生 AR 质量参照，非变量）。
- GPU 空闲（659MiB/24.5GB，util 0%），无锁在持。

## 2026-10-14T09:2x · 代码落盘
（见下方逐条追加）

## 2026-10-14T09:23:00+08:00 · 代码落盘 + 单测全绿
- 落盘：`latent_reg.py`（特征/头/训练/评测/统计）、`run.py`（阶段 1 驱动）、`gpu_lock.py`、`test_latent_reg.py`。
- 口径决策（按契约，不自选）：
  - head j（0-based jj）输入 `h_i`，输出 `z`，读出 `z @ embed.weight.T`；目标 `t_{i+jj+2}` ⇒ j=1 → t+2（契约 §2 自查 2）。
  - `disc_k2` 与 `lat_k2_lam0` 同为 `ARMS` 里 `lam=0.0` 的同代码路径（λ=0 完全跳过 MSE）⇒ L0 逐位等价。
  - `ar_t1` = 冻结主干 node 1（`decode.py:399` 原生下一 token 头）对 t+1 的命中率；非变量。
- `python3 -m pytest test_latent_reg.py -q` → **7 passed**（含包装不变、位移钉死、读出共享、冻结断言、λ=0 等价+确定性、禁墙钟）。

## 2026-10-14T09:24:50+08:00 · 冒烟 + 变异全绿
- 冒烟（`--smoke`，写 /tmp/wp633_smoke，8 训练块/4 评测块/20 步）跑通端到端：L0 all_pass=True、frozen pass、禁墙钟自检 `[]`。
- `mutate.py` → 6 条变异**全部抓红** + 逐字节还原一致 + 还原后全绿：
  `wrap_changes_backbone / target_off_by_one / readout_not_shared / backbone_not_frozen / lambda0_arm_not_equivalent / wall_clock_field_undetected`
- 开始正式 P1：`13m_s0`（seeds {0,1}）。

## 2026-10-14T09:26:26+08:00 · P1 完成（13m_s0，seeds {0,1}，wall 86.1s）
- L0 **pass**（`lat_k2_lam0` == `disc_k2` 逐位相同：hits/CE/头权重全等；冻结断言 pass）。
- 命中率（teacher-forced，路径 (i)）：`ar_t1`(t+1)=0.6714；`disc_k2` t+2 0.4572/t+3 0.335；
  `lat_k2_lam01` t+2 0.4688/t+3 0.3453；`lat_k2_lam1` t+2 0.4878/t+3 0.3663。
- **L1 采纳**：t+2 配对聚类 bootstrap（B=2000/seed2660/512 块）Δ(lam1) = **+0.03125 CI[+0.02955,+0.03302]**（两个 seed 单独也都 CI 不含 0）；
  Δ(lam01) = **+0.0114 CI[+0.01006,+0.01274]**。t+3 同向（Δ(lam1)=+0.02935）。
  ⇒ 契约 §2 的“λ>0 可能无增益”预言在本尺度被**否证**（λ 越大增益越大，且两 seed 一致）。
- `results.json` 禁墙钟自检 `[]`；L5 轨迹落盘。

## 2026-10-14T09:31:24+08:00 · P2 完成（13m_s1，seeds {0,1}，wall 212.6s，独立目录 second-backbone-13m_s1/）
- `ar_t1`(t+1)=0.6715；`disc_k2` t+2=0.4598/0.4614；`lat_k2_lam1` t+2=0.4935/0.4934；L0 pass。
- L1：Δ t+2 = **+0.03286 CI[+0.03122,+0.03450]**（不含 0）；t+3 = +0.02795[+0.02648,+0.02947]。与 13m_s0 同向同量级。

## 2026-10-14T09:38:52+08:00 · P3 完成（k=4，13m_s0，seeds {0,1}，独立目录 second-k4-13m_s0/）
- 首次 run 的 L0 比较硬编码 `disc_k2/lat_k2_lam0` ⇒ k=4 目录 L0 空（误读为 fail）。已把 L0 泛化为 `disc_k{k}/lat_k{k}_lam0` 并**重跑** P3（wall 102.0s）。
- 重跑后：L0 pass（k=4 逐位相同）；全部头 sha256 与首跑**逐一相同**（确定性复核，见 MODEL.md §4）。
- P3 数字：`disc_k4` t+2 0.4579/0.4548；`lat_k4_lam1` t+2 0.4846/0.4850。
  L1 Δ t+2 **+0.02844 CI[+0.02679,+0.03016]**、t+3 +0.02525、t+4 +0.01051、t+5 +0.01342，全部不含 0。
  `disc_k4` 与 `disc_k2` 的 t+2/t+3 基本持平 ⇒ 增益来自 λ 而非头数。

## 2026-10-14T09:39:00+08:00 · 阶段 1 收尾
- 最终 revision 复核：`mutate.py` 6 条变异全抓红 + 逐字节还原 + 还原后 7 单测全绿。
- 落盘产物：根目录 `results.json`（P1 + `supplementary_runs` 指针）、`analysis.md`（人话摘要 + P1 表 + P2/P3 补充节 + 对契约 §2 预言的结论）、
  `MODEL.md`（§2 P1 头表 + §4 P2/P3 头表，含 sha256 与复现命令）、`run_log.txt`、`train_log.jsonl`、
  `heads/*.pt`（8 个）、`second-backbone-13m_s1/**`、`second-k4-13m_s0/**`、`mutation_report.json`。
- **结论**：L0 pass；**L1 采纳**（λ>0 在 t+2 相对同参数量离散臂有独立增益，CI 不含 0，两主干点/两种子/k∈{2,4} 均同向）；
  契约 §2「λ>0 无增益」预登记预言在本尺度**被否证**。
- **未做**：路径 (ii)（当草稿再验证 α₁/τ，L2）——留给阶段-2 worker；头权重已落盘可直接加载。
- `git add` 未执行（0 staged）；`.pt` 由 `.gitignore` 覆盖；GPU 锁已释放。

## 2026-10-14T09:58:17+08:00 · 阶段 2（路径 ii，L2）完成
- actor: WP6.3.3 **阶段-2** worker（单一写入者，fresh 上下文，不与阶段-1 共享）
- 新增：`draftverify.py`（路径 (ii) 库）、`run.py --stage draftverify`、`test_latent_reg.py` +4 单测、
  `mutate.py` +2 变异。**未改**阶段 1 已落盘的数字/结论（比对 stage-1 备份逐节相同）。
- **保真硬门 pass**：`repro_direct1`（2640/2631 `direct1` 草稿定义 = 引擎默认）逐 prompt 逐轮 `accepts`
  与 `2640/results.json::checkpoints[13m_s0].arms.direct1` **逐位相同**（12/12），α₁=0.802243、τ=1.453619
  精确复现 ⇒ harness 与 2640/2631 等价。
- 接线：N+2 草稿 = `argmax(head0(nodes[0]["latent"]) @ embed.weight.T)`；N+3 固定 = 引擎 `nodes[3]`。
  协议同 2640：enwik8 val、stride 5000、PROMPT_LEN=32、12 prompt × n_new=200、B=1、温度 0、γ=2。
- 读数（α₁ / τ，配对 prompt 聚类 bootstrap B=2000/seed2660）：
  `disc_k2` s0/s1 α₁ 0.6421/0.6682、τ 1.1976/1.2388；`lat_k2_lam1` s0/s1 α₁ 0.7337/0.7537、
  τ 1.3440/1.3764；可选 `lat_k2_lam01_s0` α₁ 0.7066、`lat_k4_lam1_s0`（k=4 头0）α₁ 0.7354。
- **L2 判词 = 采纳**：Δα₁(`lat_k2_lam1_s0 − disc_k2_s0`)=**+0.091646 CI[+0.052530,+0.131047]**、
  Δτ=**+0.146391 CI[+0.097061,+0.197260]**；s1 同向（Δα₁ +0.085479、Δτ +0.137507）；lam01/k4 亦同向。CI 不含 0。
- 参考（不作判据，训练 regime 不同）：`lat_k2_lam1_s0 − 2640 direct1` Δα₁=−0.068521、`disc_k2_s0 − direct1`
  Δα₁=−0.160166（冻结特征训头整体弱于主干内联合训练头）。
- 测试/变异：`pytest test_latent_reg.py` → 11 passed；`mutate.py` → 8 条全抓红 + 逐字节还原 + 还原后全绿
  （新增 `draftverify_head_wrong_position`（草稿头接错位置）、`draftverify_alpha1_loosened`（接受判据放宽））。
- 落盘：`draftverify.py`、`results.json::draftverify`（新节；禁墙钟 `_wall_clock_fields=[]`）、
  `analysis.md` 追加「阶段 2」节、`mutation_report.json`、`run_log.txt`（GPU 实耗 165.1s 只在此）、
  `test_latent_reg.py`、`mutate.py`。
- `git add` 未执行（0 staged）；GPU 锁已释放。

## 2026-10-14T10:22:06+08:00 · 修缺（fresh worker，REVIEW-B2 F1–F6 修补；未改任何读数/结论数字/判据）

- actor: ticket 63 Phase B 修缺 worker（fresh 上下文，单一写入者）；写入范围 = 本实验目录 + 本日志。
- **A1（F1 账实不符）**：主 `results.json::supplementary_runs.k4_13m_s0` 的 `L0_all_pass` false→true、
  `verdicts` 覆盖为重跑（`second-k4-13m_s0/results.json::L0.all_pass=true`）的 t+2 采纳判词 + Δ
  （+0.02844 CI[+0.02679,+0.03016]）。该两条原为**首跑手工汇总未随重跑回写**（首跑 L0 硬编码 k2 ⇒ k=4 目录 L0 空 ⇒ 误判 false）；**未重跑 P1/P2/P3**。
- **A2（F1 附带 硬编码臂名）**：`run.py` 的 `verdicts["L0"]` 与 `_write_analysis` 的 L0 文案改为按 k 参数化
  （`lat_k{k}_lam0 == disc_k{k}`）；并手工更正 `second-k4-13m_s0/{results.json::verdicts.L0,analysis.md}`
  中误写成 k=2 的臂名为 k=4（标签文案，**不改读数**）。
- **A3（F2 证据链闭环）**：新增 `latent_reg.head_files_all_tensors_equal`（逐头逐键 `torch.equal`，覆盖全部张量）；
  `run.py` L0 权重门改用它并落盘新字段 `L0.*.state_dict_all_tensors_equal`。主 `results.json`、
  `second-k4-13m_s0/results.json`、`second-backbone-13m_s1/results.json` 三处同步回填，实测**六对全部为 true**
  （k2_s0/s1、k4_s0/s1、13m_s1_s0/s1）。三处 `MODEL.md` 加注「`.pt` 的 sha256 只标识**文件**、**不构成内容同一性证明**；内容同一性以逐张量 `torch.equal` 为准」。
- **A3(c) 更正**：原 LOG 断言「全部头 sha256 与首跑逐一相同」**只对同一 arm 的重跑成立**，**不能**用「sha256 不同」反推内容不同。
  轴 lead 已用 torch 逐张量复算：`heads/disc_k2_s0.pt` vs `lat_k2_lam0_s0.pt` **8 个张量全部 `torch.equal` 为真**（s1 同）、
  zip entry 时间同为 1980-01-01、文件 sha256 不同 ⇒ 判定「**内容相同、文件字节不同**」。
  **字节差异成因已定**：zip 各 entry 的 CRC 两两相同，但 entry 名前缀取自被保存文件的 basename，
  `disc_k2_s0`（10 字符）vs `lat_k2_lam0_s0`（14 字符）差 4 字符 × 14 个 entry = 56 字节，恰为两文件大小之差
  （2107127 vs 2107183）⇒ **纯文件名/字节层差异，无内容差异，不影响结论**。
- **A4（F3）**：`analysis.md` P3 中「`disc_k4` ≈ `disc_k2` ⇒ 增益非头数」降级为「**描述性读数，未做显著性检验**（无跨 run 配对 CI）」，只作参考。
- **A5（F4）**：`test_latent_reg.py` 新增「MSE 目标位移」单测（构造 H 使仅正确位置 `h_{i+2}` 可降 loss）；
  `mutate.py` 新增变异 `mse_target_off_by_one`（须抓红 + 逐字节还原）。
- **A6（F5）**：主 `analysis.md`（P3 节）/ `MODEL.md`（P3 节）注明 `lat_k4_lam0` 是 L0 等价自检**补充臂**（不在契约 §3 预登记臂表内；不改任何已登记臂定义）。**未改 README**。
- **A7（F6）**：`analysis.md`「对契约 §2 预登记预言的结论」节补归因边界：增益可归因于**同一参数化上的辅助 MSE 监督**；
  本实验**未**把「回归产出方式」与「其它辅助监督」区分开 ⇒ **不得**读成「回归机制优于分类机制」。
- **验证**：`python3 -m pytest test_latent_reg.py -q` → **13 passed**；
  `python3 mutate.py` → **9 条全抓红 + 逐字节还原 + 还原后全绿**；
  `python3 tools/summary.py --check` → 全绿。`git add` 未执行（0 staged）；`.pt` 未入库。
