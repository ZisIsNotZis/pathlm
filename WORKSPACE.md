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
(账本在 §3/§3.8,细节在 `docs/report_details.md`)。4. 逐机制账本:`docs/findings.md`。
5. 裁决树:`.scratch/11-retry-matrix-validity/VERDICTS.md`。
**状态**: 第一步(逐机制验证+采纳账本)已完成(ticket 14 关闭)。**下一步 = Rung 3**:
Slider 端到端(预算 b → 阈值求解 → 分配解码 → 在线质量代理),13M 上 4090 可行;
其后 Rung 4 = 规模。**待决策**: 见 Open decisions(复制+修复的通道设计最优先)。
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
- **Step 1 (进行中, ticket 14):** 采纳账本收尾——allocator 选择性实验 + 剩余机制裁决。
- **Next:** prob0-gated per-position re-embed; node-3+ chain expansion (spec-decode throughput); dual-channel (anchor exempt from corruption) for copy+robust; conditional chain × gated-retry (draft quality); scale study with 2 seeds & wider span.

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
- **复制+修复共存(需要你决策设计方向)**: 训练"容忍任意 token 错误"(容错)与
  "精确抄录锚点对"(复制)目标冲突,三种豁免(无/锚区位置/任务级)全部失败
  (anchor 召回 0.764→0.016/0.033/0.035,任务级豁免还使 bpc 1.680→1.997)。
  要共存必须给模型**显式模式信号**,候选:(a) 模式 token(词表加一槽,输入前缀
  标注任务模式,成本最低,推荐);(b) 复制旁路通道(独立小头读未损坏输入);
  (c) 部署期按画像分离(复制能力放独立 profile)。**前提问题:你的部署是否需要
  "从上下文精确复制"(引用/抽取/检索增强)?** 若不需要 → needle 永久 backlog,
  该决策即关闭。
- **Per-position retry** would need sparse (gather/scatter) compute to pay off —
  a real engine change, not a knob.
