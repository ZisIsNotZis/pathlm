# Workspace

## 方向修订（2026-09-20，用户明确）

用户反思后重聚焦：**等参数、等智商、decode 选择性更快**。回环（重试/修复）
方向经实验证伪为解码加速手段（三次失败 + 目标竞争实测），正式归档（证据保留
于 tickets 14-22 与 git 历史）。新方向 = **深度自适应 AR**（渐进精化 + 早退 +
自spec，参考 MoD/LayerSkip/CALM）：单一 AR 目标、全深度并行监督、逐通道携带门、
时钟沿式提交。哲学演化全文见 `docs/mental_model.md` §0。回环时代的机制账本
（§1-§7 与 findings）仍为有效证据，但架构引用以 §0 为准。

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
**Slider 上规模已完成**(ticket 17):INT_100M_s0(104.3M,n_mtp=2+损坏+soft retry,
p_needle=0 偏离记录)——分配器全链自洽(成本预测 ≤11.1%、prob0 ECE ≤0.0096)、
spec 严格占优复现(0.566 fwd/tok 且 bpc 更优)、**门控重试质量杠杆在 100M 消失**
(τ 轴平坦;轮次价值仍在训练侧)、自由生成段劣化检测恢复正向(13M 反向系容量现象);
堆叠溢价 +0.256(13M +0.309 收窄)。
**exit 杠杆已定案**(ticket 18):probe 式头(trunk stop-grad)保 retry(−0.0098)但
exit 质量贵(+0.93@depth6);dense 0.2 温和整形仍杀 retry(+0.0146)但 exit 轴好
(+0.199)且 clean/corrupt 双升(1.6097/1.9740,aux 正则)——整形 vs 自适应是同一
梯度两面,**双边前沿以两 profile 成立**(质量=EX2 / 自适应=probe)。
**渐近口径已定案**(ticket 19):税 +0.126 → **+0.082**(4.4 epoch,2 seeds,曲线
单调降);**等训练口径 13M +0.122 → 100M +0.082——税真正随规模下降**(快照"不缩"
系 1.1-epoch 上界,D5 模式跨尺度重现);能力溢价随训练**扩大**(corrupt gap
1.54→1.89)。B0_asym 1.341≈13M 渐近 1.346:此数据规模下参数增益饱和。
**自由生成段质量信号已测**(ticket 22,纯推理,INT2_r3):路径分歧度(同窗多条独立
层路径 TF 重评分成对 JS)=**真实但不完备的互补信号**——判据 A 经互补路线成立
(prob0 高置信半区内 js 分层 0.12→0.48,AUC 0.66–0.69>prob0 0.58–0.62)、B 成立
(生成段窗口 js real<corrupt<random,prob0 压天花板分不开 real/random)、C 成立
(注入 0/7.5/15% wrong 剂量单调);**盲区:吸引子塌缩窗 js→0**(不能单独监测塌缩);
**ticket-15“13M prob0 反向”未复现**(同权重同协议 n=6 方向转正,原反向系 n=1
假象,与 100M 一致,findings/report 已修正)。证据 `.scratch/22-gen-quality/evidence/`;
代码 `pathlm/gen_quality.py` + 根探针 `probe_gen_quality.py`(14 新单测,全绿 88)。
**塌缩域监测已定案**(ticket 22b,纯推理,INT2_r3):文本重复率(distinct-2)区分塌缩/健康窗
AUC 0.9815(人工语义标注)且与 js 正交(rank_corr 0.14)——js 在塌缩域双向失灵(假阴 js=0
与假阳 js=0.37 并存);监测栈定形:prob0 管 token 对错+js 管分歧+distinct-2 管塌缩;注意
greedy 生成全局重复偏置(全部低于自然 enwik8 水平),监测参考水平需相对化。证据同目录
(`SUMMARY_22b.md`+`collapse_monitor_INT2r3.json`+人工标注文件);代码 `probe_collapse_monitor.py`
+`collapse_summary` 等纯函数(5 新单测,全绿 93)。
**下一步(无阻塞,按优先级)**:13M 渐近全账本重校(可选)。**待决策**: 无阻塞项(backlog: probe 式深度头/exit 校准)。
**工程债**: 确定性开关、resume/长日程、FLOPs 记账、逐轮梯度累积、探针 CLI 整合。
**深度自适应 AR 首验已完成**(ticket 23,方向修订后第一个实验,2026-09-28):
新形态 `pathlm/depth_ar.py`(纯 AR+全深度并行监督+携带门+随机退出+混合深度 KV
填充+逐深度 tied 读出+soft/hard/latent 提交,14 新单测全绿 107)。小模型
四性质:深度精化成立(曲线全单调)、宽度-workspace 假设成立(d=256 null 62 维
~10% 能量且 26 条功能性读取;d=128 结构性 0)、出口提交可用(半栈 TF 接受率
0.9667,18/18 生成合法)、门行为健康(浅层开深层微开);**形态税判据证伪**
(+0.6786 ≫ 0.05;归因 ablation B-nofill:监督稀释 +0.512 主因、混合填充扰动
+0.167 次因,代码 bug 已排除)。意外:soft 提交 OOD 污染 context,默认待重验。
证据 `.scratch/23-depth-ar/evidence/`;账本 findings「Depth-adaptive AR」章;
**stage-2 入口** = 监督深度加权 sweep(先定案税归零配方,再谈规模)。
**形态税 @100M 重大节点已完成**(ticket 24,2026-09-28,同 token 预算
98.3M=1.09ep,全量 32 min,无 OOM):**税 +0.4563**(clean 1.9175 − B0_100M
1.4612)——高于 13M matched +0.31、未过 +0.5 恶化线,**形态税未随规模收缩、
方向反向,契约「缩放定律修正」分支触发**;ticket-19 经验不外推(快照口径)。
判读混杂(必读):对照 B0 系 train_m1 跨引擎 + 本 run 按契约用 final 2× 配方
(13M matched 系等权),规模/配方/引擎归属本轮不可分,stage-2 需等权臂与同引擎
dense-100M 拆账。机制侧存活且出口资产变厚:depth 曲线 13 格单调(极端半栈平台
d6−d12=0.0003)、tf@4 0.9726 / tf@6 0.9946、墙钟实测 exit-4 3.1× /
uniform-exit 1.8×;soft OOD 污染跨规模复现;**门 100M 新形态**=全层从 0.119
收小(精化压缩为 layer0–1 小门×大 delta);workspace null 能量 52% 但
functional_dirs=0。代码 additive(`--mlp-mult`/`--final-weight`+gnorm 聚合;
`final_weight` 缺省 1.0 逐位还原 v1),122 单测全绿。证据
`.scratch/24-depth-ar-100m/evidence/`;账本 findings 24 行(只追加)。
**24b 拆混杂已完成**(ticket 24b,2026-09-29,两臂各 24000×8,EQ 31.2 min /
dense 20.2 min,无 OOM):ticket 24 的 +0.4563 **归属拆清 ≈ 形态 +0.44(96%)+
引擎 +0.02(4%)+ 配方 ≈0**——等权臂 1.9200 ≈ final-2× 1.9175(Q7:配方项非
税源)、同引擎 dense-AR 1.4813 ≈ B0 1.4612(引擎基本无罪);**同引擎形态税
+0.4362**,13M +0.31 → 100M +0.44 规模斜率为正坐实,主判读维持且升级干净
归属。副发现:出口资产系全深度监督产物(纯 dense tf@4 0.12 vs 形态 0.97);
等权臂门收敛与 final-2× 同款(门收小非配方现象)。证据同目录 24b-*;
findings 24b 行(只追加)。
**文档缺口(待主会话/用户)**: docs/mental_model.md 磁盘上无 §0 章节(方向修订
全文只在 WORKSPACE 本节;commit 4280dec 仅改 WORKSPACE)。

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
