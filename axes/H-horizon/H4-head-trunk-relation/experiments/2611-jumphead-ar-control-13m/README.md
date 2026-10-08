---
name: 2611-jumphead-ar-control-13m
ticket: 29-verify-batch（契约 A）
date: "2026-09-30"
grid: H4-head-trunk-relation
engine: 复用 2610 冻结主干 + 跳跃头（纯推理，无训练）
budget: "纯推理 ~15 min（冒烟 + 全跑，共享卡 ×2 折扣）；峰值显存 ≤5GB net"
status: judged
---

# 2611 · 跳跃头对照臂修正 + 草稿接受率 · 13M

本目录把 ticket 29「契约 A」抄成本地契约（判据线原样复制，未改一字）。
上游契约：`.scratch/29-verify-batch/issues/01-verify-batch.md` 的「契约 A」。

## 0. 背景（必须理解，否则会重犯 ticket 28 的错）

ticket 28 的所谓「1-step AR 对照」实现为 `xp[:, 1:] = L[:, :-1].argmax(-1)`——
把**整段上下文**（位置 1..T−1）都换成模型自预测，再用**真值** t+2 打分。
因果注意力下，读位置 i+1 时会连带吃到被污染的前缀 1..i，量的是「在一段约 30%
字符被替换过的假文本上猜原文下一个字」——按构造不可完成，**不是自回归两次**。
实测 0.3128 与该新构造下的 0.5489 相差极大，正是这一点的把柄（见 `2610/analysis.md` ERRATA）。
**【2026-10-02 更正（C1）】原文以「低于算术下界 0.4822」为证，该算术下界已撤回；证据改用输入构造差异（整段替换 vs 只替换紧邻位置）+ 代码层单测。**

正确做法（本契约）：**只替换紧邻的前一个位置 i+1** 为 pass-1 自身 argmax，
位置 0..i 保持真值，再前向一次读 t+2。工程上用 KV-cache 语义一次算完所有行
（真值前缀的逐层 K/V 由第一次前向的逐层输入唯一确定），与「逐行各跑一次标准
前向」逐位等价（`test_ar1_matches_bruteforce_rowloop` 钉住）。

## 1. 命题

在 2610 的冻结 13M dense 主干 + 已训跳跃头上，修正对照臂后判断：
**「1 次前向 + 跳跃头」（成本 1 单位）能否达到诚实「1-step AR 两次前向」（成本
2 单位）t+2 质量的 ≥90%？** 若不能，跳跃头作为 draft 是否可用（接受率 ≥90%）？

## 2. 单因素声明

本实验**不训练、不加新参数**。干预 = **把对照臂从「整段替换」修正为「只替换
紧邻位置」**（一个测量定义的修正），主臂（跳跃头）、主干、数据、位置集、
口径全部不变。基底 = 主流标准 decoder-only transformer（dense AR）。
非必要性例外：无。

## 3. 指标表（多维，逐项分析）

| 维度 | 指标 | 定义 |
|---|---|---|
| 解决什么问题 | 视界轴多跳的成本（headroom 在成本而非质量） | — |
| 代价 | 不加任何参数（纯推理）；跳跃头开销 = 头 262,912 参数 ≈ 主干每 token 算力的 +2.5% | `equiv_speedup_detail` |
| 有效性（主口径） | `jump_t2_hit` vs `ar1_deploy_t2_hit`（同位置集、同为 vs 真值命中） | `ratio_jump_over_ar1` |
| 收益·用法 1（草稿） | 草稿接受率 `accept_jump_t2` = P(跳跃头 argmax == 主干自身 1-step AR argmax) | 无作弊（对照物 = 主干自身输出） |
| 收益·用法 2（上界参考） | `oracle_t2_hit` = 喂真 t+1 的第二次前向（**不作判据**） | 上界 |
| 收益·速度 | `equiv_speedup` = 产 1 token 的等效成本比（跳跃头 1 前向 vs AR 2 前向） | FLOPs 口径 |
| 可控性 | 跳跃头置信度 ECE / 分层单调性 | 承 2610（本实验不重复报） |
| 副产品 | 确认旧 deploy 臂的构造错误（可复现的 bug 证据） | `test_ar1_prefix_keeps_ground_truth...` |

## 4. 判据线（预登记，原样复制契约，不得修改）

**要产出的量（同一位置集、同一目标定义）**

| 量 | 定义 |
|---|---|
| `ar1_deploy_t2_hit` | 1-step AR：**只把位置 i+1 替换**为模型自己 pass-1 的 argmax，其余保持真值，再前向一次读 t+2（主对手，诚实、同成本） |
| `jump_t2_hit` | 一次前向 + 跳跃头（主角） |
| `accept_jump_t2` | `P(jump argmax == 主干自身 argmax 同位置)`（草稿接受率，严谨指标） |
| `oracle_t2_hit` | 喂真 t+1（仅上界参考，**不作判据**） |
| `equiv_speedup` | 产出 1 token 的等效成本比（跳跃头 1 前向 vs AR 2 前向） |

- **对账（硬性）**：`ar1_deploy_t2_hit ≥ 0.4822`（= 0.6934 × 0.6953 算术下界）。
  不满足 → 实现仍有问题，**先修再判**。
  **【2026-10-02 更正（C1）】该门槛的推导（两边际相乘）已撤回；此预登记判据线保留为历史记录，但对账不再构成「实现正确」的证明。**
- **主判据（成功）**：`jump_t2_hit ≥ 0.90 × ar1_deploy_t2_hit`。
- **并列判据（草稿可用）**：`accept_jump_t2 ≥ 0.90`。
- 中性：主判据比值 ∈ [0.80, 0.90)。失败：< 0.80。
- **禁止**用 oracle 作判据；oracle 只作上界附报。
- 失败决策树：若对账不过 → 修实现；若实现正确但比值 < 0.80 → 结论「冻结主干
  跳跃头不如自回归两次」，并说明信息可用性差异。

## 5. 对照与方差

- **主臂（成本 1）**：主干 1 次前向 + 跳跃头（t+2）。
- **对手（成本 2）**：诚实 1-step AR 两次前向（第二次只吃自己生成的 t+1）。同成本口径。
- **上界（仅参考）**：ORACLE 2 次前向（喂真 t+1）。
- **seed**：跳跃头 2 seeds（0/1），主干固定（无种子方差）；位置集固定 seed 7。
- **方差界**：Δ < 0.014 不可排序。两 seed 的 `ratio` 极差 0.0064 < 0.014 ⇒ 头种子
  差异不可排序；结论以均值与两 seed 并列为准。

## 6. 成本与 scope

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **15 min**（含冒烟 + 全跑，共享卡 2× 折扣） |
| 最小可判定版本 | 复用 2610 权重 + 24×16 窗口评测（本目录 `run.py --smoke` 可缩到 2×4） |
| 峰值显存 | ≤5 GB net（实测 298.7 MiB，预算内） |
| 可中断性 | 是（纯推理，秒级可重跑） |
| 冒烟计划 | `--smoke`：2 批 × bs4，打印步时与显存 |
| 预算不够时砍什么 | 砍 `n_batches`（24→12），**不砍判据线** |

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | char 级 enwik8（`data/enwik8_full.npz`；90M train / 10M eval），不下载 |
| 复用权重 | `../2610-frozen-trunk-jumphead-13m/trunk.pt`、`head_s{0,1}.pt`（哈希见 `MODEL.md`，`run.py` 启动时断言核对） |
| 入口脚本 | `run.py`（调用 `train_jumphead.py` 的 `evaluate_ar1` / `ar1_second_step_readout`） |
| resolved config | `config.json`（含 seed；reuse 块含权重路径与 sha256） |
| 可失败单测 | `tests/test_jumphead.py`：`test_ar1_matches_bruteforce_rowloop` / `test_ar1_prefix_keeps_ground_truth_not_whole_context` / `test_ar1_is_live_to_fed_token` / `test_trunk_layer_inputs_final_matches_trunk_hidden` |
| 落盘 | `results.json` / `run.log` / `analysis.md` / `MODEL.md` |

## 8. 失败决策树

| 症状 | 诊断 | 处置 |
|---|---|---|
| `ar1_deploy_t2_hit < 0.4822` | 实现仍有问题（如前缀被污染） | 先修实现（对照逐行 brute-force 单测），再判（**C1：该门槛推导已撤回，此行为历史预登记**） |
| 实现正确但比值 < 0.80 | 设计在该设定下无效 | 判「冻结主干跳跃头不如自回归两次」，说明信息可用性差异 |
| 两个口径互相矛盾 | 口径/判据问题 | 双报口径，检查是否假失败（泄漏/构造） |
| 结果不可重复 | 方差 | 按 Δ < 0.014；n=1 负结果必须复现后入账 |

## 9. 产出

`axes/H-horizon/H4-head-trunk-relation/experiments/2611-jumphead-ar-control-13m/`
下的 `README.md`（本文件）/ `run.py` / `config.json` / `results.json` /
`run.log` / `analysis.md` / `MODEL.md`。
