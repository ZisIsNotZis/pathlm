# 13 — 四项独立实验（按序执行）

- **Status:** claimed
- **Type:** experiment batch + engine work
- **Related:** 11-retry-matrix-validity（裁决树）、12-overnight（终极形态）
- **执行方式:** 每项一个 fork=false 子代理（fresh context），**严格串行**（同一 checkout +
  同一 GPU；worktree 隔离会丢 `.tmp/enwik8`/`data/`，训练无法运行）

## 全局约束（每个子代理必须遵守）

- 仓库 `/home/z/vibe/pathlm`；策略 `/home/z/vibe/AGENTS.md` 适用。
- **语言**：docs/、ticket、evidence README、commit message 用中文；代码标识符、
  日志、config key 用英文。
- **长任务**：`setsid nohup ... &` 脱离会话，然后 `sleep 300-600` + 轮询日志。
  **禁用** harness 的 bg_run 包装（会被会话重载杀掉）。不依赖定时任务唤醒。
- **GPU**：23.5GB 卡，单 run 5.75–11.5GB。**最多 2 并发**；几何重试的 run
  （p_retry>0）单跑可达 11-20GB（多轮 loss 共享一次 backward），**与任何东西
  并发会 OOM** → 串行。
- **测试**：`python3 -m pytest tests/ -q`（当前 40 绿）。改引擎必须加能失败的测试；
  每次提交前跑全绿。
- **收益规则**：bpc 优先 → 解码速度 → 能力型；每条都要“收益值(vs B0=0) + 推理成本”。
  oracle(teacher forcing) 只作上界，**部署版才是真话**。
- **证据**：`results.json` + `train_log.jsonl` + probe `*.jsonl` 复制进
  `.scratch/13-four-experiments/evidence/<name>/`。**不要提交 `model.pt`**
  （`.git` 已 1.6GB）。权重留在 `.tmp/`。
- **提交**：只 `git add` 自己动过的文件，commit message 中文、说明证据。

## 实验 1 — prob0 逐位置门控（收益最确定）

**动机**：现在重试是批级（整批一起重入，几何深度），但损坏只打在 15% 位置；
INT2 修复收益集中在损坏位置（+3.1pp），却在所有位置付 2–5 遍计算。逐位置门控
按 prob0 只在低置信位置重入 → 预期 10–15% 开火、1.1–1.15× 计算、保留大部分收益。

**设计**：
- 新 config 旋钮 `retry_gate: float = 0.0`（0 = 关，保持现状；>0 = 阈值 τ）。
- `model.forward` 中 r>0 的重入改成逐位置：
  `h = torch.where(gate.unsqueeze(-1), reentry, h)`，其中
  `gate = (sigmoid(nodes[0]["conf"]) < τ)`，detached。
  注意：混合累加器权重本来就是逐位置 `[B,T,1]`，语义一致。
- 训练与推理用同一个门控定义（都读模型自己的 detached conf）。

**必测**：
1. 单测：`retry_gate=0` 与改动前**逐位一致**（bit-identical，同一 seed）；
   `retry_gate=1.0` 时所有位置都重入（退化为现状）；门控后高置信位置的 h 确实不变。
2. **eval-only 阈值扫描**（无需重训）：在已有修复引擎 checkpoint
   `.tmp/ckpts4/C1M_s0`（混合，12000 步）与 `.tmp/ckpts4/C1_s1` 上，对
   τ ∈ {0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} 用 `probe_e2e.py`（或扩展它）测：
   开火率、损坏输入 r2 bpc、ntok_corr。判据：**~15% 开火保留 ≥80% 的 r1→r2 收益**。
3. 若 (2) 通过：再跑一个 6000 步训练 run（`p_retry=0.5` + 选定 τ）测“税是否缩小”
   （对照同配置无门控的 6000 步点）。

**产出**：`evidence/gating/` + findings.md 的 retry 节补一条 + report 账本行。

## 实验 2 — 锚区豁免损坏（双通道 needle）

**动机**：隔离实验已证 needle×损坏拮抗（0.849 → 0.005，X2C）。与驱逐设计同构的
解法：**锚 = 免疫驱逐的可靠通道** 延伸为 **锚 = 免疫损坏的可靠通道**。

**设计**：新旋钮 `corrupt_exclude_anchors: bool = False`。开启时，`_input_latents`
里 `mask_pos`/`wrong_pos` 在 `position < anchors` 处清零（仅当 `anchors > 0`）。

**必测**：
1. 单测：旋钮开启 + anchors=N 时，前 N 个位置在 `corrupt_mask` 中恒为 False；
   旋钮关闭时行为与现状逐位一致。
2. 训练 run：`X2C-v2` = `configs/X2.json` 的 path + `corrupt_mask 0.075` +
   `corrupt_wrong 0.075` + `corrupt_exclude_anchors: true`，6000 步。
   对照：X2C（0.0047/0.0129）、X2v2（anchor 76.8%）。
   判据：**锚区 needle 召回 ≥50%** 且干净 bpc 与 X2C（1.6797）不劣化超种子方差。

**产出**：`evidence/anchor_exempt/` + findings.md X2 节 + report。

## 实验 3 — n_mtp=3 的税与 a₃（吞吐的前提）

**动机**：k=1 草稿对逐位置 KV 解码无吞吐收益（验证轮=下一生成轮）；k≥2 需要
Medusa 式**批量验证**。先测前提，再决定是否写解码路径。

**设计（先做 a，再按判据决定 b）**：
- (a) 训练 `B3` = `configs/B2.json` + `n_mtp: 3`，6000 步。测：税增量
  （对照 B2 的 +0.041 与 B0 的 1.5074）与 **a₃** = P(node-3 草稿 == node-1@n+2)，
  扩展 `probe_mtp.py` 支持 n_mtp=3。
  **判据**：a₃ ≥ 0.6 **且** 税增量 < +0.05 → 才做 (b)。
- (b) 仅在 (a) 通过时：`pathlm/decode.py` 加批量验证路径（草稿 k 令牌 → 一次
  forward 覆盖 k 位置 → 读各位置 node-1 argmax → 接受最长前缀），
  **必须加测试**：接受全部草稿时的输出与逐位置解码逐位一致（M1 review 的 P0
  就是 decode 位置 bug）。测实测 tok/s；判据 ≥1.3×。

**产出**：`evidence/mtp3/` + findings.md MTP 节 + report §3.6 更新。

## 实验 4 — 文档重构（无 GPU）

- `docs/findings.md` 变**纯账本**：每机制一行（受益域 | 收益类型 | 收益值 vs B0 |
  推理成本/税 | 证据 | 状态），叙事移到 `docs/report.md`。目标 ≤200 行（现 274）。
- 不丢事实：逐条对照，任何被删的细节必须已存在于 report.md 或 ticket。
- 同步 `WORKSPACE.md` 与 `VERDICTS.md` 的交叉引用。

**产出**：findings.md ≤200 行且与 report.md 无重复；提交。

## Comments

- 2026-09-14 — agent (pi) — 用户批准四项全做，按序、fork=false 子代理执行。
