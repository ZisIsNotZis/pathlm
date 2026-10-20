# controls · 2619 的对照臂与方差界（索引）

本实验的对照**不另建独立目录**（对照臂本身就是被训练的臂 / 评测期的一次前向），
落盘位置如下（SCHEMA §3 的 `controls/` 要求以本文件满足：三要素 + 方差界在这里写清）：

## 1. 对照臂 ①：无损失基线 `A`（λ=0）

- **输入分布**：与 `B` 完全相同的合成任务批（同一生成器、同一 seed、同一 token 约定）。
- **要预测什么**：同一位置集上的同一个 token（关联回忆的答案）。
- **部署中对应什么**：stock decoder-only transformer（13M）+ 窗口 16 训练，
  **没有任何搬运结构、也没有搬运信号**——即「只丢不搬」的那条基线。
- **权重**：`train/A/model.pt`（sha256 见 `MODEL.md`）；读数 `results.json::assoc_recall.A_*`。
- **实现级等价**：`A` 与「无解码头」的 v1 路径**逐位一致**，两处证据：
  (i) 单测 `tests/test_recon_signal.py::test_recon_lambda_zero_returns_same_loss_object`
  （λ=0 时 `add_recon_term` 原样返回同一 loss 对象）；(ii) 20 步冒烟实测
  `smoke/smoke_A` 与 `smoke/smoke_A_off`（`--recon-window 0`）逐步 loss 完全相同
  （见 `smoke/` 与 `analysis.md` §0）。

## 2. 对照臂 ②：毁源对照（每臂强制，评测期）

- **输入分布**：把源片段（关联回忆的 `[0, 2K)` 键值表 / 变距离回忆的
  `[src_pos, src_pos+2)`）换成**同分布新随机符号**，其余逐位不变（`corrupt_span`）。
- **要预测什么**：同一个答案（问题不变，只是「被搬的东西」被毁掉）。
- **部署中对应什么**：「读数是不是真的来自那段远处内容」的判据；若毁源后仍高，
  说明读的是位置/局部规律 ⇒ 该臂判读作废（README §4 的 c2）。
- **读数**：`results.json::assoc_recall.*.corrupt_source`、`::distance_leg.*.*.corrupt_source`；
  判定见 `results.json::criteria.c2_corrupt_source_control`
  （操作化：毁源后 CI 下界 > max(4×chance, 0.03) ⇒ 作废）。

## 3. 对照臂 ③：同配置重跑 `B_rep`（c3 的方差界臂）

- **输入分布 / 预测什么 / 部署对应**：与 `B` **逐字段相同**（同 seed、同代码、同数据流、
  同 λ=1.0）。
- **用途**：训练侧散度。依据：本批 `2618` 实测同一配置两次（plain 写臂）相差
  **+3.271pp** 且 CI 不重叠 ⇒ 该训练配方在本尺度上存在多 pp 级训练侧散度，
  单跑一次不足以支持 +3pp 结论（README §4 的 c3）。
- **权重**：`train/B_rep/model.pt`；读数 `results.json::criteria.c3_*` 与
  `::comparisons.B_vs_Brep`。

## 4. 方差界与不可排序线

| 量 | 值 | 来源 |
|---|---|---|
| 评测侧同 seed 整批重跑 | 见 `results.json::reproducibility`（期望逐位相同） | `run.py` 复现性块 |
| 训练侧同类散度（本批已实测，另一条线） | **+3.271pp**（同配置两次 plain 写臂） | `.scratch/35-carry-channel` → `2618-carry-channel-13m/divergence.json` |
| 本实验训练侧散度 | `B` vs `B_rep` 的答案行 CE 与主腿准确率之差 | `results.json::comparisons.B_vs_Brep` |
| 方法学的 Δ < 0.014（bpc 口径）不可排序线 | 本实验**不用 bpc** 作判据（合成任务的 CE 是混合量），主判据用准确率 + 二项 CI | `docs/methodology.md` §2 |

⇒ 判据 c1/c3 都按**二项 95% CI 不重叠**判定（比 Δ<0.014 的 bpc 门槛更严：
n=4096 时 CI 半宽 ≈0.9pp）；且**同配置重跑先于主判据**（README §4 的 c3 解读规则）。
