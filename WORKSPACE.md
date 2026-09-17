# Workspace

## Goal (业务目标, 2026-09-15 用户明确)

做一个**在同等算力下比常规单路径模型更强**的模型。机制:放弃"每 token 一次前向"
的均匀推理,改为**概率驱动的碎片化精细算力分配**——探索多条独立路径、每条按置信
加权、是否继续探索由其探索代价决定。

- **主度量**: 同等推理 FLOPs 下的质量(次要: 同等训练算力)。oracle 与部署口径分开报。
- **阶段划分**: 第一步 = 逐机制验证(单独有效性 / 组合方向 / 采纳与否),产出
  **采纳账本**;完成后第二步 = 组合形态 + 外部基线 + 规模。
- **愿景的三个已证支柱**: prob0 校准(ECE 0.002–0.011,信号存在)、重试精炼
  (r1→r3 单调,花费侧)、早退+spec decode(1.45× / 1.68–1.90×,节省侧)。
- **最大缺口**: "碎片化分配"的**选择性**未被直接测量——按位置置信分桶后,低置信
  位置的每 FLOP 收益是否更高?若收益与置信无关,均匀分配就够了,愿景的核心命题
  不成立。这是下一步的关键实验(先 dense 测价值,有价值再建稀疏计算)。
- **口径假设**(用户可推翻): 探索代价 v1 = 每多一轮 ≈ 一次等价 forward;
  "更强"的对照 = 同等训练算力下的常规单路径模型(外部基线校准在第二步)。

## 新会话入口(2026-09-16 重整)

1. 目标:本文件 §Goal。2. 愿景与用法:`docs/mental_model.md`。3. 结果:`docs/report.md`
(账本在 §3/§3.8/§3.9,细节在 `docs/report_details.md`)。4. 逐机制账本:`docs/findings.md`。
5. 裁决树:`.scratch/11-retry-matrix-validity/VERDICTS.md`。
**状态**: 第一步(逐机制验证+采纳账本)已完成(ticket 14 关闭)。**Rung 3 已完成**
(ticket 15):Slider 端到端建成并验证——校准→双货币成本模型(forwards/flop)→
预算/质量求解器→验证解码→在线代理;严格占优点复现(corrupt: k=1 τ=0.98 比 plain
少 37% forwards 且 bpc 更好);prob0 代理 TF 窗口 ECE ≤ 0.008;a2 部署口径校准后成本预测全线 ≤±13%;
负结果:自由生成段 prob0 检测反向(吸引子混淆)、单轮重试 mixture==overwrite(设计保证)。
**Rung 4 已完成**(ticket 16):100M × B0/C1 × 2 seeds——税首次不随规模增长
(+0.126;13M +0.147→29M +0.174),能力溢价持续(E2E 差距 ≥1.54 bpc,B0 脆性
加深 4.18@r2),修复 r3 饱和,prob0 校准存活(ECE ≤0.009)。底座在 100M 成立。
**下一步(无阻塞,按 backlog 优先级)**:INT profile @100M 的 Slider 前沿复测、
probe 式深度头(exit 校准,解锁 flop 口径 cost<1.0)、自由生成段质量信号、
4.4-epoch 渐近口径。**待决策**: 无阻塞项(backlog: a2 部署口径校准、
自由生成段质量信号、probe 式深度头/exit 校准)。
**工程债**: 确定性开关、resume/长日程、FLOPs 记账、逐轮梯度累积、探针 CLI 整合。

## Status (2026-09-14)

- **Design:** frozen in `docs/design.md` (stage ring 0–4, cycles, aggregation principle, training recipe). One post-freeze amendment: shuffle-locality formula in §L (v1 was mathematically a no-op).
- **M0 probe: DONE, merged.** Micro-model (d=64, 2L, ~1M params, enwik8-12MB). Verdicts: (a) confidence calibration PASS (ECE 0.004–0.011); (b) ensemble ≥ best single PASS (+0.0008…+0.0015, 3 runs/2 seeds); (c) consistency loss tightens estimates PASS (cosine 0.9525→0.9882); (d) direct-transport retry FAIL (structural no-op). Evidence + report: `.scratch/02-engine-m0/evidence/REPORT.md`.
- **M1 engine extensions: DONE, merged** (ticket 03, review APPROVE round 2 after fixing a P0 decode-retry position bug + 4 P1s). 27 tests green. Transports linear/soft, dense early-exit supervision, eviction+anchors, stage-4 token retry, KV-cache decode (per-layer caches, exit threshold, prob0 retry), full eval battery, configs/*.json. Base scale: d=256, 12 layers, mlp_mult 6 = 13.08M params (deviation recorded in experiments.md).
- **M1 runs: DONE** (ticket 04) — B0 1.507 bpc (published anchor holds), I1 repair 52.8%, C1 retry +2.1pp, C3 ≈ C1, R1 token retry FAIL (ungated), L2 plateau at depth 6, X2 needle FAIL (signal starvation). Summary: `.scratch/04-m1-runs/evidence/SUMMARY.md`.
- **M1 follow-ups: DONE** (ticket 05) — mixture re-entry implemented (design §3 amendment); C4 PASS-equal (mixture = robustness at depth, monotone to 4 rounds), C5 PASS-with-caveat (token-round catastrophe removed, −1.3pp remains; profit needs prob0-gated per-position re-embed — recorded, unscheduled), X2v2 PASS (in-window 84.9% / anchor 76.8% / beyond 0.4%), retry curves saturate ~round 3. Evidence: `.scratch/05-m1-followup/evidence/`. 33 tests green.
- **M2 sweep: DONE** (ticket 06) — element-tax law: identity/order-destroying elements expensive (shuffle +0.545, skip +0.175, wrong-token +0.127), identity-preserving ones nearly free (noise ±0, redo +0.044, mask +0.059). L3's locality boundary: trained 0.5 collapses at eval 1.0 (3.84).
- **X1/X3: DONE** (ticket 07) — distance penalty FREE (−0.007, mild regularizer), composes with eviction (+0.004), anchor channel improves 76.8→93.3%.
- **TTS probes: DONE** (ticket 08) — composition gain ∝ path diversity (L1 −0.047); deterministic-path checkpoints gain 0; depth/round ensembles degrade (auxiliaries are fallbacks, not voters); nothing beats B0 — taxes dominate.
- **M3 + diversity: DONE** (ticket 09) — IX1 sub-additive win (+0.087 < 0.277 sum); composition law: shuffle is the poison (every shuffle combo lands 2.2+; order-freeness antagonizes repair); INT full grammar does not compose at 13M. DIVL1: diversity pressure (capped −JS, float32 + grad-norm guard) improves both baseline AND ensemble slope (−0.062 > −0.047, ceiling +0.023) — mechanism proven, tax still dominates at this scale.
- **Overnight research (ticket 12):** conditional chain built+measured (oracle 0.682 ≈ node1, deploy 0.547 < direct 0.560 — draft-quality bottleneck), D1 causally confirmed (fixed-engine mixture r2 1.936 vs pre-fix 95.2), D6 fixed (true vocab projector), INT2 integration (sub-additive +0.309, corrupted-input −1.79 bpc vs B0, needle×corruption antagonism 85%→0.5%), scale probe (2.2×: tax flat, E2E gap narrows). Final report: `docs/report.md`. Verdict tree: `.scratch/11-retry-matrix-validity/VERDICTS.md`.
- **Four experiments (ticket 13, fork=false subagents):** ① per-position retry gating — FALSIFIED (direct re-entry is a fixed point ⇒ vacuous; linear ~15% fire makes r2 worse; state masking saves no compute — needs sparse gather/scatter); ② anchor-exempt corruption — FALSIFIED three ways (anchor recall 0.764 → 0.016 / 0.033 / 0.035 with no exemption / positional / task-level; task-level also worsened bpc 1.680→1.997) ⇒ repair-vs-copy needs an explicit mode/channel signal; ③ n_mtp=3 + Medusa batched-verify spec decode — **POSITIVE: 1.68–1.90× measured tok/s for +0.0166 tax** (independently reproduced); ④ findings.md restructured 325→163 lines (ledger), detail in report §3.7. Two review catches: a flaky gate test (unseeded numpy) and a real `needle_acc` eval leak (`corrupt_mask` not zeroed — verdict-neutral, proven by positive control X2v2 reproducing 0.8658/0.7637).
- **Step 1 (done, ticket 14):** 采纳账本收尾——allocator 选择性实验 + 剩余机制裁决。
- **Rung 3 (done, ticket 15):** Slider 端到端(INT2_r3 重训底座;原 INT2 权重未入库):
  校准(fire 曲线 split gap ≤0.004)、双货币成本模型(k=0 误差 ±8%;k=1 a2 口径差
  −13~−20%,保守)、求解器+验证解码、在线代理(TF 窗口 ECE 0.0011–0.0079;
  自由生成段失效=负结果);严格占优复现;单轮 mixture==overwrite(引擎无损失)。
  证据 `.scratch/15-slider-rung3/evidence/`;代码 `pathlm/slider.py` + 根探针 `slider.py`。
- **Next (Rung 4):** 规模研究——100M+ 训练(1–2h/6000 步),2 seeds + 更大跨度,
  定案税/增益缩放;顺带 backlog: 自由生成段质量信号、
  probe 式深度头(exit 校准)。

## Key lessons (do not re-derive)

- **Retry is information-free unless the return channel transforms the state** — direct re-entry re-derives the same fixed point (measured). Retry experiments ARE transport experiments; repair value lives in soft (repair-in-context) and discrete (token retry) channels.
- **Node semantics:** node k at row i predicts clean token t_{i+k}; node 0 = self/repair of the CURRENT token (source of prob₀, the universal control signal). Targets = the clean token stream itself — never a pre-shifted y (this exact off-by-one was a critical review find in M0).
- Ensemble voters sharing one trunk are too correlated for real gains — decorrelation must come from transports/depths.
- Forward signature: `model(tokens, paths, targets)` — `paths` is one sampled layer path PER round (fresh shuffle/skips each pass, via `sample_rounds`); corruption happens inside the model; eval corruption is seeded via `torch.manual_seed(1234)`.

## Conventions

- Char-level = byte-level enwik8 (205-ish vocab incl. [mask] slot; vocab logged in results.json).
- Tickets: Matt Pocock layout in `.scratch/NN-slug/`; evidence under `.scratch/NN-slug/evidence/`; data caches under `data/` (gitignored), raw enwik8 in `.tmp/enwik8`.
- Tests: `python3 -m pytest tests/ -q` (10 tests, all fail-capable). Review gate: fresh-context subagent reads diff + spec only; two rounds were needed for M0.
- Training runs go to background nohup with log polling; ~2.5 min per M0-size run on the RTX 4090, M1 runs est. 30–60 min.

## Open decisions (for the user)

- **Parallel writer — resolved (2026-09-16).** The untracked `.scratch/13-mechanism-refinement/` + stashed patch (`.tmp/stale_mechanism_refinement.patch`) contained: a `gate`/`gate_tau` variant (superseded by the committed `retry_gate`, and the mechanism itself was falsified), a `corrupt_spare_anchors` implementation (independently reimplemented and tested in ticket 13 exp 2), and a cosmetic type-checker refactor. User confirmed no second agent; work is reasonable in intent but fully superseded — keep the stash archived, directory stays untracked.
- **Docs size:** `docs/report.md` is 289 lines (the deliverable report). If the
  200-line budget applies to it too, split into report.md + report_details.md.
- **`.git` 1.6 GB** of committed `model.pt` blobs — removal is a history rewrite.
- ~~复制+修复共存的设计决策~~ **已关闭(2026-09-16 用户裁决)**: **无需模式信号**。
  设计原则——模型从不"保留错误",**目标恒为正确的 token**(损坏只是输入侧增广,
  训练 targets 一直是干净流,实现已满足)。实测: 13M 下损坏训练会压低精确复制
  行为(X2Cv3 连干净 needle 批次也未恢复,0.05;且 bpc 1.680→1.997),属容量/训练
  规模限制而非目标冲突 → needle 保持 backlog;若未来需要精确复制,优先更大规模
  /更长训练,或专门检索头,而非模式信号。
- **Per-position retry** would need sparse (gather/scatter) compute to pay off —
  a real engine change, not a knob.
