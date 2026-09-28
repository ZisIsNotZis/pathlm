# PathLM 研究报告 — 逐实验细节(审计层)

> 本文件是 `docs/report.md` 的**审计层**:逐实验的完整数字、矩阵与过程记录。
> 结论与采纳账本以 report.md 为准;数字与事实以本文件为准(自 findings.md
> 账本归档,原始证据在各 ticket 的 `evidence/`)。
> 边界:report.md = 结论与账本(决策层);本文件 = 逐实验证据(审计层)。

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


## D. INT2 组合拮抗细节(X2C 隔离实验)

**组合拮抗确认(隔离实验 X2C):** needle+损坏 → 0.005。损坏训练(容忍任意 token
错误)与复制任务(精确抄锚点)目标冲突。skip/retry 不是元凶。


## E. Depth-adaptive AR 逐臂过程（findings 账本压缩部分的归档）

> 本节归档 findings「Depth-adaptive AR」章压缩前的逐臂过程与中间量（ticket 23 系/
> 24/24b/25）。结论以 findings 为准；逐数字的原始出处是各 SUMMARY 与 run 目录
> `results.json`（索引 `docs/artifacts.md`）。证据目录：`.scratch/23-depth-ar/`、
> `.scratch/24-depth-ar-100m/`、`.scratch/25-tax-attribution/`。

### E.1 ticket 23 首验（形态 v1，13M 级）

实验矩阵：sanity-A（d=128/L=4/0.92M）、sanity-B（d=256/L=8/6.96M）、dense-B（对照）
等，29.5M tokens，总 GPU 11.4 min。四性质数字：depth-bpc B: 6.250→2.673→…→2.4161
（单调）；workspace：B null 62 维（~10% 能量）、26 条 (batch,方向,±) 超阈响应，
C（d=128）结构性 0（探针口径 = null 方向 ±5% 扰动 → depth-L logits 响应，对照
3.4e-05 通过）；出口：B@depth4 TF 接受率 0.9667、@depth1 0.8016、18/18 生成合法；
门：起步 0.12 → 浅层 g1 0.18–0.43 → 深层 0.04–0.11。形态税 B 2.4161 − dense-B
1.7375 = **+0.6786**；归因 ablation（sanity-B-nofill）→ 监督稀释 +0.512 主因、
混合填充扰动 +0.167 次因。soft 提交 OOD：生成文本 rescore 6.9 vs hard 1.2。

### E.2 23b 监督加权 sweep（4×sanity-B，10.8 min）

w∈{1, 0.1, 0.02, 1→0.1 退火}：depth-L 2.45→2.57→2.73（税 +0.71→+0.99，单调反向）；
退火臂 2.4444 保底座但税仍 +0.71；推荐配方 = v1 等权。**归因改判**：浅层全深度
监督是表示正则而非稀释主因；carry 门乘性瓶颈/欠拟合升级主嫌。soft OOD 复现
（d4 7.4 vs hard 1.2），幅度随浅头质量耦合。（基线 dense-B 1.7375；契约基线
1.1719 系笔误已勘误。）

### E.3 23c 三消融因果分解（3600 步链式，15.2 min）

H1 dense+aux 2.0216（aux 在 dense trunk 上即 +0.284——非无罪）；H3 stop-grad 浅头
2.7129（恶化：浅层门不再打开 g1 0.088 vs 0.173、tf@4 0.973→0.791——**排除**）；
H2 12k 步 1.8582（税 +0.713 → +0.121，收缩 83%；深层门开度随训练增大 g2 0.150 vs
0.095）——**欠拟合主因**。门×退出×填充堆栈 +0.429（门本身 +0.228、退出+填充
+0.200 vs ticket-23 nofill 臂）。aux 在门控臂穿 trunk 的梯度为净收益 −0.263。

### E.4 23d 残税 +0.121 分解（dense-B-12k 1.5287 / H1-12k 1.6362，13.7 min）

恒等闭合：+0.121 = {dense 欠拟合 −0.209 + aux 残余 +0.108 + 门/填充固有 +0.222}。
**「残税主要是堆栈固有」证伪**：matched-step 真税 @12k = +0.3295（vs @3.6k
0.7126，收缩 54%），aux/堆栈两分量 3.6k→12k 仍收缩 48%/62% 未平台化；23c
「aux+门堆栈合计 <0.12」系欠拟合锚点误推（锚点 dense 自身还有 −0.209 训练收益）。
100M 外推（ticket-19 渐近系数）：matched 税 ≈ +0.21 bpc（区间 +0.15…0.23）——
训练加长过不了 |税|<0.05，需配方级干预或改写交易目标为出口资产。侧观察：12k 下
dense+aux 逐深度支配门堆栈（d4 1.669 vs 1.863、d8 1.636 vs 1.858）。

### E.5 23e fill 变体对决（12k matched 步，~29 min）

三分支同配置唯一变量 = KV 填充：fill 1.8582 vs ragged nofill 1.9438（**+0.0856**
≫ ±0.03 平局带）——「填充必须」成立，ticket 23 3.6k「填充是负分量 +0.200」反转
不复现（欠拟合伪影 + 对照混淆：当时 nofill 臂同时无随机退出）；**proj-fill
（fill + 逐层恒等初始化 adapter）= 1.8356，全深度逐深度支配 fill，族新最优**（残税
vs dense-B-12k +0.307）。副判据：三臂曲线全单调；tf@4 0.969–0.980 全存活且 nofill
最高（0.980）——底座质量与出口自洽轴向分离。

### E.6 ticket 24 @100M 主 run

骨架 d=720/L=12/mlp_mult 6，**112.64M**（形态件 carry 门 6.23M + proj-fill adapter
6.23M，如实记录）；24000 步 × bs8 = 98.30M tokens（1.09 epoch，与 B0_100M 同预算）；
wall 32.0 min；冒烟峰值 9.1GB；gnorm skips 0。判读表：税 +0.4563（1.9175 − 1.4612）
vs 13M matched 同预算 +0.31（12k vs 24k 步口径）——**未收缩、方向反向**；契约
「缩放定律修正」分支触发。混杂声明：对照 B0_100M 系 train_m1 跨引擎 + 本 run
final 2× 配方。机制侧：depth 曲线 13 格单调（4.7659→…→1.9175，d6−d12=0.0003）、
tf@4 0.9726 / tf@6 0.9946、墙钟 exit-4 3.1×（235 vs 76 tok/s）/ uniform-exit 1.8×、
生成 18/18 合法、soft OOD 复现（d12 soft 8.42 vs hard 0.79）。门：全层从 0.119
收小（p90≤0.13、frac>0.5≈0）——「渐进开门」未出现。workspace：null 能量 52% 但
functional_dirs=0（δ 口径未冻结）。

### E.7 ticket 24b 拆混杂（等权臂 + 同引擎 dense 对照，各 24000×8，~70 min）

两臂先 20 步冒烟全电池再全跑（冒烟抓出探针墙钟 gen 段 fd 选择写反并修复）。等权臂
clean 1.9200 vs final-2× 1.9175（Δ 0.0025 ≪ 种子散布 0.0103，**Q7：配方项非税源**）；
同引擎 dense-AR 1.4813 vs B0_100M 1.4612（+0.0201，引擎基本无罪）；**同引擎形态税
+0.4362**。副发现：出口资产系全深度监督产物（dense trunk tf@4 0.120 / tf@6 0.160
vs 形态臂 0.970/0.994；截断资产 proxy 口径）；EQ 出口资产配方鲁棒（等权 tf@4
0.9699）；uniform-exit 1.67× / exit-4 2.82× 复现；形态 full-depth 解码 ~1.5× 慢于
dense（143 vs 94 tok/s，方向性口径）；等权臂门收敛与 final-2× 同款（g0 0.069 vs
0.070）——「门收小」非配方现象（Q8 剩规模/时长）。dense 增量解码器与并行 forward
逐位一致（max|Δlogit| 9.2e-5）。

### E.8 ticket 25 三臂归因（B 35.3 / C 68.5 / dense-48k 47.0 min）

臂定义：B = `--readout-dims 360`（unembed 与全部出口头只读 h[:360]，tied 表列切片
E[:, :360]；trunk/embedding 全维）；C = 48000 步 + `--exit-anneal`（末 25% 线性
1.0→0.3、final 2× 同窗 2.0→1.0）；dense-48k = C 的公平同引擎对照。B：clean 2.1475、
税 +0.6863（vs B0）/ 同引擎 +0.6662，恢复 −50.4%/−52.7%；切片能量占比 depth0
0.85 → depth2 0.026 → ≈0，null 100% 但 functional_dirs=176；门反转 layer3–11
p90=1.0（vs A 全层 ≤0.07）；depth 曲线尾部倒退 d7→d12 +0.012；tied 表不对称：输入
态能量比 0.85（随机 0.50）。C：clean 1.8429，名义恢复 +16.4%；48k 锚拆解：形态税
+0.4372 ≈ 24k 锚 +0.4362（Δ 0.001）；C 对 A 改善 0.0746 vs dense 侧训练收益
0.0756（1.4813→1.4057）逐位镜像。dense-48k 1.4057 < B0 1.4612（纯 dense 2× 收益
−0.0555，对照健康）。副产出：C 等效验证吞吐 122.8 tok/s（tf@4 0.9666 → 加速
1.4750）vs A 112.9，+9%（spec_cost_model FLOPs 口径主指标，纯跳层 wall 副记录）；
门收小再排除时长候选（C 48k 仍全层收小）；出口资产三形态臂 tf@4 全 ≥0.95；
soft OOD 三度复现（C d12 soft 3.36 vs hard 0.74）。冒烟：B 验证切片生效（rank 206/
结构性 null 514/能量比 ≈0.5 随机基线）；C 验证调度接线（step 19 → aux_w 0.44 /
final_w 1.2 与公式精确一致）。7 新可失败单测（变异 M1 切片忽略 / M2 调度退化均
抓红），全套 129 绿（基线 122 保持）。
