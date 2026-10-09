# F/M/G 三轴批判审计（F1–F4 · M1–M3 · G1–G5，共 12 格）

- 审查者：独立 reviewer（fork=false，独立上下文，只读）
- 基准：本 checkout 的工作树（HEAD 见 watchdog 报告）；**未运行任何 GPU、未改任何数字/代码**
- 范围：`axes/F-iteration/**`(4) + `axes/M-carrier/**`(3) + `axes/G-policy/**`(5)
- 方法：逐格读 `README.md` / `catalog.yaml` 条目 / 实验 `analysis.md` / `results.json` /
  引擎代码（`pathlm/model.py`、`pathlm/depth_ar.py`、`pathlm/gen_quality.py`、
  `tools/evalkit/kv_quant.py`），并回溯 `.scratch/` 原始证据；不复用 `.scratch/49` 的结论。
- 起点材料：`.scratch/49-adversarial-review/REPORT.md`（本文**独立**核对，明确标注确认/推翻）

---

## 一、一句话总判：**不能直接用**

这 12 格里，**4 格（M1 / M2 / M3 / F4）的主结论按本仓自己的 SCHEMA §21 分诊属于 (a)
「业界已做透」**，却仍挂在 `judged`/`partial` 状态灯上（`SCHEMA.md:374` 明写
「(a) 类条目**不得**进入状态灯的 `judged`」）；**F1/F2 的税与通道结论建立在已作废的引擎修订上**
（ticket 11 自己写「no retry number survives as-is」）；**G 轴五格自认全部 (a)**；
**G5 是为一个用户已判无意义的自设命题烧掉两个规模 + 24k 收敛点的钱**。
同时，判据层面存在多处「带内差被写成结论」（Δ = 0.0004 / 0.0043 / 0.0028 对 0.014 或对
0.0035 的重跑噪声）。

**可以作为材料继续调查**，但**12 格里没有一格达到「可对用户复述为已定论」的标准**。

---

## 二、建议删除清单（供主 Agent 做树管理）

| 格 | 理由（含证据） |
|---|---|
| **G5-parallel-lines**（整格） | 用户已判命题无意义（`docs/OPEN_PROBLEMS.md` P-01；`catalog.yaml:1168` 已写「自设命题…无意义且多花算力」）；部署判据两档皆负（+0.1052 / +0.0322，8× 推理）；机制结论贴门槛（+0.0144 vs 门槛 0.014）且稳健性被自己撤回为 `on_threshold`（`2628-jitter-l1/analysis.md` §0）。资产迁移见该格 CRITIQUE §④。 |
| **F1 的选项 3–6**（整段去噪 / 候选集 / 隐式推理步 / 权重迭代） | 本仓 `axes/F-iteration/PRIOR_ART.md:143` 自判 (a)；且在本仓唯一基底（单卡 13M char-enwik8）上不构成可判定问题。 |
| **M2 的「摘要压缩 / 低秩（MLA 式）」两支** | 同一份 PRIOR_ART 判 (a)。 |
| **F4 的「无载体 + 重建损失」子设计** | 业界该损失从不单独使用（总配压缩瓶颈）；本实验的负结果不能外推，属 off-design。 |
| **G2-currency 的「研究格」身份** | 是记账纪律不是可证伪命题；无自有实验目录（`G2/experiments/README.md:11`「本格暂无实验目录」）。建议并入 `docs/methodology.md`。 |

（**不建议删**：M1、M3、G1、G3、G4、F2、F3 —— 它们含真问题或真资产，见下。）

---

## 三、建议重做清单（按优先级）

### P1 —— M1-carrier-type：把「显式通道」做成可判定的问题（≈100 min GPU）
- **为什么**：本格是整条「信息右移」主线的落点，但当前入口结论 = 「跨种子尚未稳健，
  未收口为已证结论」（`catalog.yaml:1015` 段），而状态灯仍是 `judged`；
  首验主腿（距离 241–255）**在训练分布外**且已永久降级（同文件 `:1020` 段）；
  「delta 优于 plain」是 **n=1 vs 双稳态臂**（`2618/analysis.md:191,303-305`）。
- **怎么做**：`U0/U1/U2` × seed 0/1/2，预登记「学习成功判据 = 答案行 CE < ln64 − 0.5」，
  判据落在**分布内**多查询腿，每臂报状态范数轨迹。
- **判据**：该涨 = 三 seed 的**最差**分布内召回显著高于 chance；该跌 = 无；
  必须不变 = `U0 ≈ chance`、毁源掉回 chance、U1/U2 参数量逐位相同。

### P2 —— F2-return-channel：修 D6/D3 后重扫通道（≈40 min GPU）
- **为什么**：D1（`_mixture_reentry` 被 embedding cap 压成 ~30× rescale）已证明
  「soft > linear > direct」的排序是**cap 的产物**；D6（linear 实为 `EᵀE`，放大 71.7×）
  在验收单里**未勾选**。
- **判据**：该涨 = 能注入新信息通道的逐轮 bpc 边际；该跌 = direct 的边际停 0 且
  `alive=true`；必须不变 = `transport=none` 的 clean bpc 逐位一致。

### P3 —— M2：用量化标尺复测（分钟级，纯推理）
- **为什么**：当前「量化在任何拓扑上都是内存负收益」是**逐值 fp16 scale** 这一自造粒度
  的产物（`tools/evalkit/kv_quant.py` docstring + `cummax(dim=-2)`），与 KIVI 的
  per-channel/per-token 标尺不是一回事。
- **判据**：该涨 = 含标尺总字节下降；该跌 = Δbpc 超 0.014；必须不变 = 因果前缀两条测试。

### P4 —— F3/G1 合并：把「严格占优」改成带内判据（分钟级）
- **为什么**：现读数 bpc 2.0502 vs 2.0545（Δ=0.0043 < 0.014），不可排序；
  且两格 evidence 完全重合。
- **判据**：该涨 = 低置信分桶每 FLOP 收益（用**原预登记**的分桶 2× 判据）；
  该跌 = 高置信带收益 ≈0/负；必须不变 = 单轮 mixture≡overwrite 逐位一致。

### P5 —— G3/G4 的口径补字（分钟级）
- js 的 **2× 前向**成本；prob0 的校准结论限定在 TF 窗口；「自由生成段失效」写进 `verdict`。

### P6 —— M3：只在「窗口+sink vs 窗口」的可分离对照下继续（≈16 min）
- 否则整格转 `reference`。理由：两点、无头寸（不重训 192 只 +0.0029）、
  重跑噪声 0.003486 > 效应 0.002813。

---

## 四、开倒车清单（业界标准是什么、我们错在哪）

| # | 我们做的 | 业界标准 | 偏离的后果 |
|---|---|---|---|
| 1 | M1「载体类型首验」：`S ← S(I−βk̂k̂ᵀ)+βvk̂ᵀ` + WY 分块三角求解，在 MQAR 式任务上测 | **DeltaNet / Gated DeltaNet**（2021–2025）的 delta 规则与 chunked 实装；**MQAR** 是线性注意力标准基准（Zoology 2023） | 本仓等于在 13M 上复现；且绝对水平 **9.5%** 远低于业界同机制（近满分），「6.08× chance」被当成成功读数 |
| 2 | M1 把「direct 重入 = 结构性 no-op」写成自创铁律 | looped/Universal Transformer 系公论：**循环需输入注入，否则退化** | 把业界教训包装成本地发现；且「结构性」用词超过证据（见 五-3） |
| 3 | M3「大窗口训练税近零」当结论 | 滑动窗口注意力 / StreamingLLM 的既有共识 | 本仓数字只是 13M 复核；且本地「192 免训也 +0.0029」⇒ 无头寸 |
| 4 | F4「重建/充分性损失」当研究格 | Compressive Transformer（2019）compression loss；AutoCompressors / ICAE / RECOMP / DMC | 本仓自认非空白格，却仍保留为待做实验 |
| 5 | M2 量化 / 驱逐 / 摘要 / 低秩 | KIVI / KVQuant / H2O / SnapKV / StreamingLLM / Compressive Transformer / MLA | 全部 (a)；且本仓为「因果量化」自造**逐值 fp16 scale**（2 B/值），使「量化永远负收益」成为粒度产物 |
| 6 | F3/G1 置信门控早退、G4 校准、G3 三信号 | ACT / CALM / PABEE / MoD；Guo 2017 温度缩放；P(True) / Semantic Uncertainty / SelfCheckGPT / distinct-n | 结论层全是 (a)；本仓的增量只在「求解器闭环」与「三信号组合」这两个工程产物上 |
| 7 | G5 跳层随机路径集成 | stochastic depth（训练正则）；deep ensembles（推理集成） | 把训练期正则手段当推理期集成源，且命题非用户目标 |

**另一条自造口径（方法学层）**：`docs/methodology.md` 的「1.4pp 准确率方差带」被当作已测方差
使用（`docs/methodology.md:24-25` 自己已更正为**预登记实用阈值**），但 M1 的多个表
（如 `catalog.yaml:1030` 段「< 1.4pp 预登记阈值（C13）」）仍以它作为可排序线。

---

## 五、本轮最严重的**三件事**

### 1）状态灯与自己的分诊结论相反（契约级冲突）
`SCHEMA.md:374`：「(a) 类条目**不得**进入状态灯的 `judged`；用 `reference`」。
但：`axes/M-carrier/PRIOR_ART.md:96` 判 M1/M2/M3 主结论 (a)；
`axes/F-iteration/PRIOR_ART.md:143` 判 F1/F2/F3/F4 (a)+b；
`axes/G-policy/PRIOR_ART.md:52` 判 G1–G5 **五格主结论全部 (a)**。
而实际状态灯：M1/M3/G1–G5 全 `judged`、F2/F3/F4 `judged`。
⇒ 「33 judged = 33 条本地实测」这个读法在本 12 格里**至少有 9 格不成立**。
（这条**确认**了 `.scratch/49` 的 B11/A7，并给出更精确的计数与文件行号。）

### 2）「信息右移」这条主线目前**没有一条站得住的正面结论**
M1 的正面首验：主腿在训练分布外（已撤）→ 跨种子符号翻转（seed0 +6.05pp / seed1 −2.44pp）
→ 分布内腿跨种子不稳健（+0.36pp / −3.26pp）
（均在 `catalog.yaml:1015-1026` 段）。
留下来的**唯一跨条件稳定**的结论是**负面的**：「13M dense 主干本来就没有检索能力」
（关联回忆 0.0061 ≈ chance，`catalog.yaml:1016` 段末）与
「掩码逼不出搬运」（`M3-.../README.md:5` 第 1 条，整段窗外 494 根差 **恰好 0.000**）。
⇒ 本线的真实产出是**一台合格的天平 + 两条负结果**，不是「通道能搬」。

### 3）多个结论建立在噪声带内或已作废修订上
- G5：24k 机制净效应 **+0.0144** vs 门槛 **0.014**（只高 **0.0004**）；随后 `l1` 臂自身
  重跑抖动 **0.0074 = 效应 51%**（`2628-jitter-l1/analysis.md` §0/§3）。
- F3/G1：「严格占优」= bpc **2.0502 vs 2.0545**（Δ **0.0043** < 0.014）。
- M3：训练税 **−0.002813**，而同配方整条重跑差 **0.003486**（`2616/analysis.md` §3）。
- F1/F2：税值取自 **pre-D1/D3 引擎**；ticket 11 自认「all prior retry numbers must be
  re-measured」（`issues/01-retry-matrix-validity.md`「IMPORTANT scoping note」），
  D6 未勾选。
- F4：单臂 n=1，重跑落 **0/4096** 另一退化模式（`2619/analysis.md` §7）。

---

## 六、对 `.scratch/49-adversarial-review/REPORT.md` 的独立核验

**确认**（我独立读到同样证据）：
- **A2（量化 scale 非因果）**：旧实现确实沿 batch+token 取 amax —— 但**代码已被修复**：
  `tools/evalkit/kv_quant.py` docstring「因果/流式口径（2026-10-02 修复 A2；旧实现是缺陷）」
  + `running_amax = x.abs().cummax(dim=x.dim()-2)`。**49 报告的这一条已过时**；
  它**衍生的新问题**（逐值标尺 ⇒ 量化永远负收益）反而是当前更严重的问题（见 四-5）。
- **A3（距离腿六档并非全部 > chance）**：确认，`2618-carry-channel-13m/analysis.md:71,241-243`
  已写明 d=32 = 0.000000、d=192 = 0.000977 低于 chance。
- **A4（越界腿支撑入口结论）**：确认，`catalog.yaml:1013-1026` 已加更正。
- **B11（用「记忆·未核」判 (a)）**：确认，且我与 49 的差别是给出了**具体行号与格数**。
- **B8（2628 分量公式未更正）**：**推翻**。`2628-jitter-l1/run.py:43-48` 的
  `COMPONENT_NOTE` 与 `results_13m.json:3511` 的 note 都已是
  `(0.0332 − 0.0300) − (−0.0112) = 0.0144`。49 报告的这条已被修掉。

**新增（49 未指出）**：
- `catalog.yaml` 的 `prior_art_note` 注释块**整块错位一格**：`:901` 块表头写 `F1`、
  内容却是 W5-capacity 的「减容量付质量、不买灵活性」、`prior_art_source` 指向
  `axes/W-width/PRIOR_ART.md`；而 F1 真正的参照线句（「循环式重复若无输入注入会退化为固定点」）
  出现在 `:929` 表头为 `F2` 的块里（与 `axes/F-iteration/PRIOR_ART.md:143` 的 F1 行逐字相同）。
  依次 `:951`(F3←F2)、`:972`(F4←F3)、`:1003`(M1←F4) 同错。且 M2/M3/G1–G5 **完全没有**
  `prior_art_note` 块。⇒ 任何人按 `catalog.yaml` 做树管理都会读错「这格对标什么业界标准」。
- `axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md`
  的**首屏与判据表仍给出已撤回的「安全丢弃半径 d\* = 32」**（`:45,56,162,175,192`），
  而 ERRATA 只写在 `M2/README.md:52`。
- `pathlm/model.py:398-401` 显示 `transport="direct"` 的重入是**把上一轮输出再喂回同一栈**
  （`_run_layers(h, path)`），并非恒等 ⇒ F2 的「**结构性** no-op」用词超过证据。

---

## 七、给主 Agent 的一句话建议

**先做两件不花钱的事**（改措辞/状态灯，不碰数字）：
① 把 M1/M2/M3/F4 以及 G1–G4 中属 (a) 的主结论改成 `reference` 口径（或明确标注
「业界结论 + 本地复核」），并把 G5 整格删除；
② 把 `catalog.yaml` 的 `prior_art_note` 错位修正、把 M2 的 d\*=32 加上 ERRATA 指针。
**再做一件花钱的事**：M1 的三臂×三 seed 复测（≈100 min）——它决定「信息右移」这条主线
到底有没有正面结论。
