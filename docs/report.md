# PathLM — 终极形态研究报告(草稿,持续填充)

> 状态:夜间自主研究(ticket 12)进行中。本文件是最终交付物;每填一个结果就更新
> 一次。所有结论带证据路径。收益规则(用户定的):bpc 优先 → 解码速度 →
> 能力型(无法用 bpc/速度表达的部署收益);只有税没有收益的条目无意义。

## 0. 阅读地图

- 架构(冻结): `docs/design.md`
- 测量程序: `docs/experiments.md`
- 逐机制结论(SSOT): `docs/findings.md`(ticket 11 修正版)
- 裁决树 + 计划: `.scratch/11-retry-matrix-validity/VERDICTS.md`
- 本夜状态: `.scratch/12-overnight/issues/01-overnight-ultimate.md`
- 证据: `.scratch/12-overnight/evidence/`

## 1. 一句话结论

PathLM 的弹性语法里, **税不是设计的语言, 能力才是。** 机制价值分三类:

1. **基线根本没有的能力** — 损坏鲁棒(INT2 在损坏输入上比 B0 好 1.79 bpc)、
   环形缓冲部署(不重 prefill)、prob0 校准(ECE 0.002–0.011);
2. **计算前沿上的点** — 早退 1.45×;**MTP + Medusa 批量验证 spec-decode
   实测 1.68–1.85×、税仅 +0.017**(本程序唯一"税小收益大"的机制);
3. **控制/草稿能力** — prob0 是全部弹性机制的控制信号(它不自己产生收益,
   但没有它就没有门控)。

干净 enwik8 上的税(+0.01~+0.18)都是"保费",在受益域里换成 1.8 bpc 级或 1.7× 级
收益。组合(INT2)次可加(+0.31 < 各项和 ~0.58),但暴露**目标冲突**:容错训练与
精确复制不可兼得——三路尝试(无豁免 / 锚区位置豁免 / 任务级豁免)把 anchor 召回
从 0.764 打到 0.016 / 0.033 / 0.035,且任务级豁免还把 clean bpc 从 1.680 恶化到
1.997。**终极形态不是单模型堆叠,而是按部署画像组装;冲突目标需要显式模式/通道
信号**,不能靠位置或批次的损坏豁免。

另修五个引擎/评测缺陷(D1 范数钳位 / D3 重试深度 / D6 假投影 / resolved-config
记录 / `needle_acc` 损坏泄漏),撤回旧 retry 矩阵的全部排序(它们不可分辨),
并证伪逐位置 latent 重入门控(需稀疏计算,状态掩码此路不通)。

## 2. 测量方法(先于一切的原则)

1. **受益域先于测量**:每个机制先写清"对什么输入/部署才有价值",再在该域测;
   干净 enwik8 只当税列。
2. **三列账本**:收益类型 | 收益值(相对 B0=0)| 税(纯推理)。B0 是 1.1 epoch 快照
   (1.5074 bpc),渐近值用 24000 步(4.4 epoch)报。
3. **两套方差**:同种子 nondeterminism 0.0019 bpc;种子方差 0.0001–0.0138/格。
   结论 <0.014 bpc 不可排序 → 关键结论 2 种子。
4. **E2E 优先于 tax**:`probe_e2e.py` 每 checkpoint 报每深度 bpc/repair/吞吐;
   retry 的税 +0.12 与损坏输入上的 +1.98 bpc 收益对比是范式案例。
5. **诚实护栏**:oracle(teacher forcing)只作上界,部署版(argmax 条件)才是真话;
   n=1 标注;失败如实记录。

## 3. 逐机制账本(最终数字落地中)

| 机制 | 受益域 | 收益类型 | 收益值 vs B0 | 税(推理) | 证据 |
|---|---|---|---|---|---|
| mask 损坏 | 低质量/部分损坏输入 | 能力型(可修复) | 修复 59.4% | +0.059 | findings §I |
| wrong-token 损坏 | spec-decode/未知损坏位置 | 能力型(可进 KV 可验证) | 修复 52.8% | +0.127 | findings §I |
| 噪声/pure-noise | 连续损坏 | 能力型 | (范数吸收) | ~0/+0.063 | findings §I |
| MTP+prob0 | 一切弹性机制的控制 | 能力型(校准信号) | ECE 0.002-0.011 | ~0 | findings §MTP |
| **条件链(新)** | 预测 t+2 | **bpc** | oracle 0.682≈node1 0.693; **deploy 0.547 ≈
  direct 0.560**;接受率 0.75<direct 0.77 | ~0 | P1(双种子,证据 e2e/CH_*) |
| 早退 | 低延迟部署 | 速度型 | 1.4× tok/s | +0.067 | findings §Early |
| retry(修复引擎) | 损坏输入 | **E2E bpc** | r1 1.99 → r2 1.94 → r3 1.93(饱和);
  混合版 r1 1.99→r2 1.94,双种子一致(s0 1.943/s1 1.932) | +0.12 | P2 |
| retry 逐位置门控 | 损坏输入 | 能力型(重入状态) | direct 上 r2 不变(空洞);linear ~15% 开火 r2 保留 −0.15…−0.48 | 0(不省算力/不缩税) | ticket 13 gating |
| skip | 测试时扩展 | 速度×质量斜率 | K=8 −0.047(DIVL1 −0.062) | +0.175 | findings §Skip |
| 驱逐+锚 | 长上下文/有界内存 | 能力型(不重 prefill) | needle 76.8→93.3 | +0.058 | findings §X2 |
| 锚区免疫损坏 | needle+损坏组合 | 能力型(救复制) | anchor 0.016;**判据 ≥50% 失败** | +0.0044 | ticket 13 anchor_exempt |
| needle 批次整体免损坏 | 同上 | 同上 | anchor 0.035;**仍失败且 bpc 1.680→1.997** | +0.32 | ticket 13 needle_antagonism |
| MTP + Medusa spec-decode | 低延迟部署 | **速度型** | **1.68–1.85× tok/s**(n_mtp=3 税 +0.0166) | +0.041(n_mtp=2)/+0.017(n_mtp=3 增量) | ticket 13 mtp3 |
| 距离惩罚 | 长程注意 | 免费正则 | needle 通道 93.3 | **−0.007** | findings §X1 |
| 多样性压力 | TTS 扩展 | 斜率 | −0.062(DIVL1) | +0.023 | findings §TTS |

## 3.5 条件链裁决(P1,双种子,高度可重复)

**机制成立,部署不赚。** 给定真 t+1,t+2 准确率 0.682 几乎追平 node-1 对 t+1 的
0.693——信息在,DeepSeek 式 concat+proj 能利用。但部署(argmax 条件)反而比无条件
的 direct 头差 1.3pp:node-1 约 31% 的草稿错误注入的噪声吃掉了全部条件收益。
接受率 0.75 ≈ direct 0.77,速度侧也不赚。oracle-deploy 差 0.135 是"草稿质量"的标价。

**工程课(同一条判决的一部分):** 第一版把状态整个换成裸 embedding,丢掉上下文,
A/B 立刻证伪(oracle 0.24 ≪ direct 0.56);第二版 concat+proj 修复(主模型 h 拼
embed(t),投影回 d)。这验证用户规则——先想清有效性再跑,A/B 当场指出错误。

**对 INT2 的影响:** 不用 token 链,用 n_mtp=2(node-2 direct 头,接受率 0.77,供
spec-decode 草稿)。

## 3.6 Spec-decode: 批量验证已实现并实测吞吐(实验 3)

**旧结论(未实现前的分析):** 接受率 0.77 是必要不充分条件。贪心解码里单令牌
草稿的"验证轮"就是下一步的生成轮,若把验证当作独立 forward,则 k=1 草稿产出
2 令牌仍用 2 次 forward,吞吐 = 基线;真正的增益需要 k≥2 并做 Medusa 式批量
验证(一次 forward 覆盖多个候选位置)。

**现已建成并实测(ticket 13 实验 3)。** `pathlm/decode.py::decode_spec`:
在当前位取 node-2..n_mtp 草稿 + node-1 自身下一 token,一次 forward 全部喂入,
读各位置 node-1 argmax 验证,接受最长前缀,拒绝后缀从 KV cache 截断。输出与
逐位置贪心**逐位一致**(有强制全接受/强制拒绝/window 驱逐三个专门测试)。

前提测量(B3 = `configs/B2.json` + `n_mtp:3`,6000 步 seed 0;B2fresh 配对基线):

| | B2fresh | B3 | 差 |
|---|---|---|---|
| clean bpc | 1.5438 | 1.5604 | **+0.0166** |
| node-3 t+3 acc | — | 0.4511 | — |

- teacher-forcing 级联:B3 a₂=0.660 / a₃=0.514(条件 a₂=0.774 / a₃=0.621)。
- **部署口径**(验证器条件在模型自己生成的 token 上)才是真话:a₂=0.844 /
  a₃|a₂=0.653 → **2.40 tokens/forward**;teacher forcing 低估了接受率。
- 微基准:宽度-3 前向成本 = 宽度-1 的 **1.28×**(K 个位置远低于 K 倍成本)。

实测稳态生成吞吐(B=1,greedy,短 prefill + 生成 400):

| 配置 | plain tok/s | spec tok/s | 加速 | tok/forward |
|---|---|---|---|---|
| enwik8, k=2 | 101.9 | 170.9 | **1.68×** | 2.40 |
| enwik8, k=1 | 100.8 | 143.3 | 1.42× | 1.84 |
| random, k=2 | 103.0 | 190.6 | **1.85×** | 2.68 |

判据 ≥1.3× **通过**。注意:若把 256-token 的未批量 prefill 也算进去
(`eval.decode_speed` 口径),加速被稀释到 ~1.0–1.1×;部署中 prefill 应一次
批量完成,稳态生成吞吐才是解码的真实指标。

## 3.7 逐机制细节(findings 账本删减部分的归档)

> `docs/findings.md` 是压缩账本;本节归档被删减的逐 run 值、中间量与完整裁决,
> 保证任何事实都有一处落点。findings 的每节结论仍以那份为准。

### Redo / Skip / Early exit / Shuffle

- **Redo**(`p_redo`,税 +0.044):10% 层重复 ≈ +3% 相对 PPL;IX2 失败归因 shuffle
  而非 redo。with/without 未做,两个食谱:eval-only ablation(`p_redo`→0);层内
  probe(同输入 1× vs 2× 的 frozen-head CE)。理论收益(层内自校正、层级自适应
  算力)从未按层身份/该层置信度条件化;该旋钮最便宜,收益列开放。
- **Skip**(`p_skip`,税 +0.175):最贵深度元素,也是最好的 TTS 基底。K=8
  1.6926→1.6454(−0.047);DIVL1 baseline 1.6842、增益 −0.062、比 plain-skip K=8
  上限 +0.023。未测:按层置信度自适应 skip;FLOP-matched 比较(应按省下 FLOP 记账)。
- **Early exit**(`w_dense_exit`,税 +0.067):~iso-PPL 1.4×;深度曲线 ~6/12 平台
  (半栈 +~0.003 bpc)。深度**集成失败**:mixed 1.602 vs single 1.5744;depth CE
  1.06→0.74(15× PPL 悬崖);置信头排序深度而非逐 token 可靠性。未测:退出阈值
  扫描(1.4× 是一个工作点);退出上叠 spec-decoding。
- **Shuffle**(`shuffle_locality`,税 +0.545):层序携带 ~0.5 bpc。locality 0.5
  训练容忍 eval ≤0.5,全随机崩 3.84。含 shuffle 的组合(IX2/IX3/INT)均落 2.2+
  bpc;与修复拮抗(repair 51%→38%)。唯一测得收益是失败鲁棒性(层 dropout/置换
  部署)。未测:2–4× 下 partial locality(<0.25);eval order-canonicalization
  (排序执行当免费"修复到规范序")。

### MTP / 条件链

- B2(`n_mtp=2`,M1 scale)clean 1.5481 / Δ+0.041;1→2 步 +0.0407。node-2 t+2 acc
  55.97%(node-1 t+1 68.8%,chance 0.5%,M0 仅 0.26)。与 node-1 验证一致 65.8%;
  给定 node-1 正确则接受 77.3%。朴素概率复合:P1·P2chain 的 CE 3.96 vs node-2
  单独 1.58——正确链需 node-2 条件在 node-1 采样 token 上(re-embed+re-predict)。
  consistency loss 中性(IX4 1.6589 ≈ C1 1.6549)。
- 条件链(ticket 12 P1,双 seed):DeepSeek 式 concat+proj;oracle 0.682≈node-1
  0.693;deploy 0.547<direct 0.560;~31% 草稿错误;接受 0.75≈0.77。k=1 naive
  verify-next-round 零吞吐收益。
- n_mtp=3(ticket 13 实验 3):税 +0.0166(B3 1.5604 vs B2fresh 1.5438);node-3
  t+3 acc 45.1%、CE 2.0292;node-2 t+2 0.5594(与 B2fresh 0.5611 持平);TF 级联
  a₂=0.660/a₃=0.514(条件 0.774/0.621);部署 a₂=0.844/a₃|a₂=0.653 → 2.40
  tok/forward;宽度-3 前向 1.28×。`decode_spec` 逐位 == 逐位置贪心(5 单测)。
  稳态 enwik8 1.68×(k=2)/1.42×(k=1)、random 1.85×(k=2)。

### Latent retry 矩阵(cell 排序已撤回,ticket 11)

4 transport × {ungated overwrite, mixture vote};全 C run 带 corrupt_wrong 0.15。

| Run | transport | re-entry | bpc | seeds | repair base→last | forced 4-round curve |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | direct | overwrite | 1.6549 | 1 | 0.507→0.528 | — |
| C1M_n1 | direct | mixture | 1.6645/1.6595 | 2 | 0.493→0.518 | 0.493/0.518/0.521/0.522 |
| C2 | linear | overwrite | 1.6621 | 1 | 0.499→0.519 | 0.499/0.519/0.523/0.518 |
| C2M_n1 | linear | mixture | 1.6548/1.6686 | 2 | 0.503→0.522 | 0.503/0.522/0.524/0.524 |
| C3 | soft | overwrite | 1.6584 | 4 | 0.496→0.521 | — |
| C4 | soft | mixture | 1.6535 | 1 | 0.501→0.526 | 0.501/0.526/0.529/**0.530** |
| R1 | token | overwrite | 1.6775 | 1 | 0.513→0.467 | — |
| C5 | token | mixture | 1.6856 | 1 | 0.500→0.431 | 0.500/0.431/0.361/0.303/**0.427** |

**四缺陷(ticket 11):**
- **D2 config 混杂**:C1M/C2M 漏 `n_mtp`(默认 2),而对照钉 1;单旋钮 +0.041
  (B2 1.5481 vs B0 1.5074)。published C1M +0.194/C2M +0.164 虚高;n_mtp:1 对照
  C1M_n1 1.6645/1.6595、C2M_n1 1.6548/1.6686。
- **D1 re-entry gauge**:`_mixture_reentry` 对 stage-L 状态套 stage-E `norm_cap`
  (1.0),而 `_transport(direct)` 返回未截断 h。测得 direct/linear mixture 重入
  范数 1.00 vs overwrite 28.7;soft 0.90(cap no-op)。direct/linear mixture cell
  测到的是 ~30× rescale 而非 latent mean,故 soft>linear>direct 排序只是 cap
  破坏程度的单调。
- **D3 未训练 accumulator**:`p_retry` 为 Bernoulli,训练从不超过 2 遍;
  accumulator 只在 r≥2(推理 forced rounds=k)区别于 overwrite。r=1 时 soft
  mixture 与 soft overwrite 逐位相同(`max|d|=23.3726, cos=0.92109`)。
- **D4 方差**:同种子 nondeterminism 0.0019(三个同 seed 6000 步 run);种子方差
  C3 0.0001 / C1M_n1 0.0050 / C2M_n1 0.0138——cell 依赖,C2M_n1 单格即超
  6-cell 全部跨度(0.011)。所有 cell `n_mtp:1`;B0 单 seed,故 cell-vs-cell 无 B0
  依赖,但绝对 Δ 带 B0 未知种子方差。

**修订读法(保留原四条中仍成立者):** 无排序;overwrite 内 transport 不可分辨
(跨度 0.007),支持"多一遍"的价值而非回传几何;税可靠(+0.146…+0.178,10–100×
任何底);门控价值在修复侧;**direct/linear mixture 测到的是 rescale,不构成对
latent-mean mixing 的反证,只反这个 gauge**;retry 是修复非集成;L-loop×depth 协同。

**D5**(ticket 11,24000 step≈4.4ep,测于 pre-D1/D3 引擎):retry 税 +0.3644→+0.1217,
|Δlast|/|Δfirst|=0.33,~3k 起平台 +0.12–0.13,比已发布 +0.147 小 ~20%。故所有已发布
绝对税为 1.1-epoch 上界(B0 1.5074@1.1ep vs 1.3459@4.4ep)。fixed engine 改变 retry
深度,需重测。

**D1 因果确认**(ticket 12 P2,12000 step×2 seed):修复 gauge 后 mixture run 的
round-2 损坏输入 bpc 单调 2.71→2.33→2.19→2.09→2.03→1.98→1.95→1.936(seed
1.943/1.932),pre-fix 发散 3.7→8.5→32.9→95.2;round 序 r1 1.993→r2 1.936→r3
1.934 饱和。(probe_e2e 指标为 node-1 下一 token acc on 损坏位置,非 node-0 自修复。)

**逐位置 latent 门控**(ticket 13 实验 1,`retry_gate=τ`):fixed-engine direct ckpt
上 τ 从 0→100% 开火但 C1M_s0 r2 恒 1.9353(r1 1.9928)/ C1_s1 1.9287(r1 1.9861),
ntok 不变(direct 重入状态 == h,r=1 mixture 单累加项,`torch.where` 无切换)。linear
ckpt 上 ~15% 开火使 r2 劣于 r1:bpc 保留 C2M −0.15/C2 −0.48,ntok 0.40/0.07。6000
步 gated(direct+mixture, τ=0.9)vs 同引擎无门控:1.6884 vs 1.6860(Δ+0.0024)。
判据不成立;状态掩码不省算力,需稀疏计算。

### Token retry / Input corruption / X1 / X2 / TTS 细节

- **Token retry**:R1 1.6775(−4.6pp,ECE 0.001→0.015);C5 1.6856(−1.3pp,ECE
  0.007)。修复效应 3–4.5pp vs 0.16pp 采样误差(n≈98k corrupt positions),过
  ticket 11。bpc 侧:仅高于 worst latent cell 0.009–0.017,在该 cell 0.0138 种子
  散布内(各单 seed),故"token round 在干净数据上 landing below base"仅 suggestive。
- **Input corruption**:I 族 [税 | 15% 损坏修复 acc]:mask +0.059|59.4%;wrong-token
  +0.127|52.8%;embedding noise ±0.000|n/a;pure-noise latent +0.063|n/a。噪声由
  norm-cap 吸收,纯噪声只 ~4% 相对 PPL。IX1 mask×soft-retry +0.087 < 0.277 之和
  (次可加,唯一已证组合增益)。
- **X1**(税 −0.007):与驱逐共 +0.004;anchor 76.8→93.3%;per-head 平均注意距离
  26–43,无头纯 local;beyond-window 命中是 local-LM 强度非抗驱逐召回。
- **X2**(税 +0.121 原 / +0.058 v2 needle 重设计——改善是测量非机制):ring-buffer
  免重 prefill;anchor 76.8%(+penalty 93.3%);per-layer cache+早退+prob0 retry
  1.4×;retry 轮间 KV 位置严格递增(P0 类 bug)。三路尝试表与 `needle_acc` bug
  见 §1 与 `.scratch/13-four-experiments/evidence/needle_antagonism/`:位置豁免
  <4%,任务级豁免 bpc 1.680→1.997。
- **TTS/DIVL1**:capped −JS(bf16 下 float32 + pre-step grad-norm gate)同时改善
  baseline 与斜率(−0.062 > −0.047,上限 +0.023);税 +0.023;确定性路径集成恰 0。

## 4. 集成实验(INT2,24000 步)与"终极形态"结论

配置:mask+wrong 损坏、MTP(n_mtp=2)、retry(soft×mixture,几何深度)、skip、
驱逐+锚+needle;**无 shuffle/redo/距离惩罚/早退/多样性**(显存与隔离理由见下)。
对照 B0long(24000 步,1.3459 bpc)。

| 指标 | INT2 | 对照 | 判定 |
|---|---|---|---|
| clean bpc(24000 步) | 1.6548 | B0long 1.3459 | 总溢价 +0.309 |
| 单独税之和(混合视野,上界) | ~0.58 | — | **次可加**(0.31 < 0.58) |
| 损坏输入 r1→r2 bpc | 2.080 → 2.006 | B0 损坏 3.80 | **−1.79 bpc vs B0** |
| 修复(node-1,损坏位置)r1→r3 | 0.428 → 0.459 | B0 ≈0 | +3.1pp |
| 解码 exit | 42.7 tok/s | 29.5 常规 | **1.45×** |
| needle 召回 | 0.005 | X2 单独 84.9% | **崩溃(组合拮抗)** |

**组合拮抗确认(隔离实验 X2C):** needle+损坏 → 0.005。损坏训练(容忍任意 token
错误)与复制任务(精确抄锚点)目标冲突。skip/retry 不是元凶。

**"终极形态"结论(诚实版):**
1. **单模型堆叠到 0.31 溢价,换来:损坏鲁棒 +1.79 bpc、修复 +3.1pp、1.45× 解码、
   spec-decode 草稿能力 —— 但丢了 needle。** 次可加成立(共享修复机制),但
   "全部都要"不是最优形态。
2. **按部署画像组装**(配置即形态): 有界内存+长上下文 → X2 profile(0.058);
   低质输入 → I+retry profile(+0.15 换 +1.8 bpc 鲁棒); 低延迟 → 早退 profile
   (0.067 换 1.45×); spec-decode → MTP profile(0.041 换 0.77 接受率)。
3. **需要精确复制 + 容错的场景**: 分开通道(锚区不做损坏训练)——设计建议,
   未实现。
4. **INT2 的 needle 失败与旧 INT 的 shuffle 失败是同族教训**: 组合前先查
   机制间的目标冲突;共享机制(修复)次可加,冲突目标(容错 vs 精确)拮抗。

## 4.5 规模探针(2.2× 参数,29.2M,6000 步,单种子)

| | 1×(13M) | 2.2×(29M) | 读法 |
|---|---|---|---|
| B0 clean bpc | 1.5074 | 1.4527 | 参数在 1.1ep 下值 +0.055 |
| retry 税 | +0.1475 | **+0.1741** | **不缩小**(1.18×,恰在种子方差边缘) |
| B0 损坏 r2 | 3.756 | **3.138** | 大模型没训练也更抗造 |
| 机制损坏 r2 | 2.094 | **2.008** | 机制质量也改善 |
| E2E 差距 | +1.66 | +1.13 | 收窄——因为基线不那么脆,不是机制变差 |

结论: 税框架在规模下依旧误导——**能力溢价大致守恒,基线脆弱性随规模下降**。
"税随规模缩小"的希望(2.2×)不成立;真正要问的是规模下受益域是否扩大
(更大模型部署在更脏的输入上?),那是部署问题不是缩放定律。2 种子 + 更大跨度
才能定案 → backlog。

## 5. 尚存问题 / 已验证边界

- 已验证边界: 13M/4.4epoch 下, 税与收益的账本在本文件 §3 与 findings.md。
- 未实现: prob0 门控逐位置重入(状态掩码已证伪)、按画像自动组装工具。
  node-3+ 草稿与 Medusa 批量验证已在实验 3 建成(§3.6)。
- 未测: 规模(2-4× 参数下税/增益缩放)、条件链 × prob0 门控重试(草稿质量组合)。
- 方法债: .git 1.6GB 已提交权重(history rewrite,user-gated)。findings.md 已重构为
  ≤200 行账本(现 163 行),细节归入本文件 §3.7。
