# controls · 2618 的对照臂与方差界（索引）

本实验的对照**不另建独立目录**（对照臂本身就是被训练的臂 / 评测期的一次前向），
落盘位置如下（SCHEMA §3 的 `controls/` 要求以本文件满足：三要素 + 方差界在这里写清）：

## 1. 对照臂 ①：无通道基线 `U0`

- **输入分布**：与 U1/U2 **完全相同**的合成任务批（同一生成器、同一 seed、同一 token 约定）。
- **要预测什么**：同一位置集上的同一个 token（关联回忆的答案）。
- **部署中对应什么**：不做任何显式搬运的 stock decoder-only transformer（本仓 13M 基线）。
- **权重**：`train/U0/model.pt`（sha256 见 `MODEL.md`）；读数 `results.json::assoc_recall.U0_*`。

## 2. 对照臂 ②：毁源对照（每臂强制，评测期）

- **输入分布**：把源片段（关联回忆的 `[0, 2K)` 键值表 / 变距离回忆的 `[src_pos, src_pos+2)`）
  换成**同分布新随机符号**，其余逐位不变（`longrange_tasks.corrupt_span`）。
- **要预测什么**：同一个答案（问题不变，只是「被搬的东西」被毁掉）。
- **部署中对应什么**：「读数是不是真的来自那段远处内容」的判据；若毁源后仍高，说明读的是
  位置/局部规律 ⇒ 该臂判读作废（契约判据 4）。
- **读数**：`results.json::assoc_recall.*.corrupt_source`、`::distance_leg.*.*.corrupt_source`；
  判定见 `results.json::criteria.c4_corrupt_source_control`（操作化：毁源后 CI 下界 > 4×chance ⇒ 作废）。

## 3. 对照臂 ③：同配置重跑 `U1rep`（判据 ⑥ 的方差界臂）

- **输入分布 / 预测什么 / 部署对应**：与 `U1` **逐字段相同**（同 seed、同代码、同数据流）。
- **用途**：训练侧散度；本实验实测到 plain 臂的**双稳态**（见 `stability.md` §3）。
- **权重**：`train/U1rep/model.pt`；读数 `divergence.json`。

## 4. 方差界与不可排序线

| 量 | 值 | 来源 |
|---|---|---|
| 评测侧同 seed 整批重跑（1024 查询） | **逐位相同**（12/12 命中，差 0.0） | `results.json::reproducibility` |
| 训练侧同代码同 seed 逐步散度 | ~3e-4/步（600 步：0.5109 vs 0.5106） | `smoke/detA|detB`、`stability.md` §2 |
| **训练侧「学不学得动」的散度（本实验主方差源）** | **+3.271pp**（同配置两次 plain：0.0732% vs 3.3447%，CI 不重叠） | `divergence.json` |
| 方法学的 Δ < 0.014（bpc 口径）不可排序线 | 本实验**不用 bpc** 作判据（合成任务的 CE 是混合量），主判据用准确率 + 二项 CI | `harness/methodology.md` §2 |

⇒ 判据 ②③ 都按**二项 95% CI 不重叠**判定（比 Δ<0.014 的 bpc 门槛更严：CI 半宽 ≈0.9pp）；
判据 ③ 额外用重跑臂做了稳健性检查（见 `analysis.md` §7.2）。
