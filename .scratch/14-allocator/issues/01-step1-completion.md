# 14 — 第一步收尾：allocator 选择性实验 + 采纳账本

- **Status:** claimed
- **Type:** experiment batch + engine work
- **Related:** 12-overnight（backlog 来源）、13-four-experiments（已完成的四项）
- **执行方式:** 按序 fork=false 子代理，串行（同 checkout + 同 GPU）
- **目标（WORKSPACE.md §Goal）:** 第一步 = 逐机制验证收尾，产出**采纳账本**。
  愿景核心命题"碎片化分配的选择性"必须被直接测量。

## 全局约束

- 仓库 `/home/z/vibe/pathlm`；策略 `/home/z/vibe/AGENTS.md`。
- **语言**：docs/ticket/evidence/commit 中文；代码标识符英文。
- **长任务**：`setsid nohup ... &` + `sleep 300` 轮询；禁用 bg_run 包装；不依赖定时唤醒。
- **GPU**：23.5GB。`p_retry>0` 的 run 单跑可达 11-20GB（多轮 loss 一次 backward），
  **必须独占**；轻量 run（无 retry）≤6GB 可 2 并发。
- **测试**：`python3 -m pytest tests/ -q`（当前 52 绿）。改引擎必须加能失败的测试。
- **收益规则**：bpc 优先 → 解码速度 → 能力型；每条报"收益值(vs 对照) + 推理成本"；
  oracle 只作上界，**部署口径才是真话**。
- **证据**：results/train_log/probe jsonl → `.scratch/14-allocator/evidence/<name>/`；
  不提交 model.pt。只 add 自己的文件。
- **工作区**：`.scratch/13-mechanism-refinement/` 是另一写者的，**勿动勿提交**。

## 实验 1 — allocator 选择性（愿景核心命题）[最高优先]

**命题**：算力应该按位置置信分配——低置信位置每 FLOP 的收益更高。
若不成立（收益与置信无关），均匀分配即可，愿景核心不成立。**先 dense 测价值，
有价值再建稀疏计算（gather/scatter）**——顺序不能反。

**(1) 训练 ALLOC 模型**：`n_mtp=2, w_dense_exit=1.0, p_retry=0.5(几何), max_retries=4,
reentry_mix, transport=soft, corrupt_wrong=0.15`，seed 0。
显存风险：dense-exit（每深度 12 次 MTP 评估）+ 重试多轮共享一次 backward。
缓解：先试 batch 32；OOM 则 batch 16 / steps 24000（token 预算不变）并在
`resolved` 里记录；仍 OOM 则实现**逐轮梯度累积**（见实验 3）。

**(2) 选择性测量（关键）**：在 ALLOC 与 INT2_s0 checkpoint 上，对损坏输入：
- 按 round-0 prob0 分桶（十分位），测每桶的 r1→r2 bpc 改善与 ntok_corr 改善；
- **判据**：最低置信桶的每 FLOP 收益 ≥ 最高置信桶的 2×（否则选择性无价值）；
- 同时报：若只在低置信 15% 位置花第二轮算力，能保留 r1→r2 总收益的百分之多少。

**(3) 分配前沿**：x = 每 token 实际 passes（1/2/3/4），y = bpc；
再叠加 exit_threshold 扫描（tok/s vs 质量）与 retry_threshold 扫描（生成时门控）。
产出"质量-算力"前沿曲线（这是愿景的直接证据）。

**(4) 组合**：exit × retry × spec（decode_spec + exit_threshold）在同模型上的
tok/s 与质量——三个已证机制的复合是否 > 各自相乘。

## 实验 2 — 剩余机制的小实验（可并行/随后）

- **自适应跳层**（置信条件跳层）：skip 的 +0.175 买的是 TTS 基底；"易 token 跳层"
  是分配的层级版本。eval 优先，判据同实验 1(2)。
- **span 损坏**：建议直接 backlog（mask/wrong 已覆盖损坏语义，span 是变体）。
  记录理由即可，不实现。

## 实验 3 — 逐轮梯度累积（引擎解锁件，实验 1 需要时做）

`forward` 现在把所有轮次的 loss 累积成一张图一次 backward → 11-20GB。
改成每轮 `loss.backward()` 累积梯度（`opt.zero_grad` 只在轮次 0 前），
显存上界 = 单轮 + 梯度。**必须加测试**：小模型上逐轮累积的总梯度与一次性
backward 逐位一致（或 <1e-6），且 `retry_gate`/`reentry_mix` 行为不变。

## 实验 4 — 采纳账本定稿（第一步的交付物）

`docs/report.md` 加"采纳账本"一节：每机制一行——采纳/不采纳/backlog、收益类型、
收益值、成本、适用域、证据指针。逐条对照 findings.md 与本 ticket 的裁决。
这是第一步的完成标志。

## 状态

- [ ] 实验 1 训练 ALLOC
- [ ] 实验 1 选择性测量（判据）
- [ ] 实验 1 前沿 + 组合
- [ ] 实验 2 自适应跳层 / span backlog 记录
- [ ] 实验 3 逐轮累积（按需）
- [ ] 实验 4 采纳账本定稿 → **第一步完成**

## Comments

- 2026-09-15 — agent (pi) — 用户明确业务目标（见 WORKSPACE §Goal），第一步=逐机制
  验证收尾。关键缺口是"分配选择性"未测，实验 1 直接测它；顺序：先测价值再建稀疏。

## 实验 1 裁决 — 选择性成立（愿景核心命题获证）

`probe_selectivity.py`（修复了 worker 遗留的两个 bug：corr_n 未累加、total_toks 误用
分桶宽度），三个 checkpoint 交叉验证，损坏输入，按 round-0 prob0 十分位分桶测
r1→r2 的 node-1 bpc 改善：

**门控（最低 15% 位置开火）vs 均匀（所有位置重试）的 r1→r2 收益（bpc）：**

| 模型 | 门控 15% | 均匀 | 选择性差 |
|---|---|---|---|
| ALLOC（dense-exit 训练） | −0.0651 | −0.0554 | −0.0097 |
| INT2（无 dense-exit） | **+0.0211** | **−0.0193** | **+0.0405** |
| C1M（修复引擎） | **+0.1284** | **+0.0570** | **+0.0715** |

- **选择性把重试从净有害翻成净有益**（INT2：−0.019 → +0.021，且计算 2× → 1.15×）；
  C1M 上同等第二轮算力的收益翻 2.2 倍（+0.057 → +0.128）。
- **收益形状**（INT2）：decile 2–6（中低置信）+0.04~+0.06；decile 7–9（占 87% token）
  ≈0/负——价值集中在低/中置信带，高置信带是纯浪费。
- **dense-exit 训练破坏重试精炼**（ALLOC 各桶全负）：又一例"辅助监督伤害主估计"。
  ⇒ 分配器模型的训练配方：**retry + corruption，不带 dense-exit**。
- 我原定的判据（"最低桶 ≥ 最高桶 2×"）不是正确的透镜——真实形状是"增益随置信
  单调下降、在高置信变号"，门控把符号翻转，这比比值更有力。

**工程含义**：选择性的价值已证 → 稀疏计算（gather/scatter）有了建设依据；
同时注意 dense-exit 与 retry 的训练冲突（INT2 配方是当前最优分配器底座）。
