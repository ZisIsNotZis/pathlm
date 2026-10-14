# WP-2652 收口报告 · 2.2.1 残差的形式

- ticket：`61-axis2-depth` · §9.2 WP-2652；落点 `axes/2-depth/2.2.1-residual-form/experiments/2652-residual-form-13m/`
- 本轮工作：四臂 12k 已训完并评测（CPU 收口，不取 GPU 锁）；聚合折齐、判读、权重清单、SSOT 判词
- 时间：`2026-10-14T22:46+08:00`；actor：WP-2652 worker
- 独立复核：**未做**（fresh 复核者排在 writer 停止后，`SCHEMA.md` §14 / §14b）

## 人话摘要

把普通 transformer 的「一条残差 + 层前归一化（pre-LN）」换成三种残差写法——归一化**后置**（post-LN）、
子层输出乘**固定** `1/√(l+1)` 缩放、子层输出乘**可学习**逐层标量门（init=1.0）——在同一批初始化、
同一条训练代码、同一个 12k 步预算下各训一遍。问的是：换哪一种残差写法，能把**终质（clean bpc）**
或**「跑到一半就截断可用的程度」（逐深度可读性）**改善到超过噪声带（0.014 bpc）。

结论两半：

1. **终质这一半，三种写法全都没有可排序差别。** 四臂 clean bpc = base 1.5310、post-LN 1.5284、
   门控 1.5326、固定缩放 1.5413；相对 base 配对差最大 0.0103，全在 0.014 带内。⇒
   **靠换残差写法改终质，在 13M/12k 这个尺度没有头寸**（正面假设 H1 不成立）。
2. **截断这一半，逐深度曲线确实被改动、且越带——但方向要看清。** `post-LN` 让中间深度读出
   **大幅变差**（depth 1 的 bpc 从 base 的 4.95 升到 **10.13**，≈2 倍退化；depth 1–7 的
   `Δbase−form` = **−5.18…−0.48**）⇒ 更**难**截断；反过来**固定缩放 `1/√(l+1)`** 让中间深度
   **小幅变好**（`Δbase−form` = +0.40…+0.06），代价是终质略降 −0.0103（仍在带内）；门控臂只有
   `|Δ| ≤ 0.0864` 的方向混杂微扰。

所以判词 = **「终质（clean bpc）维不可排序；可截断性（逐深度可读性）维可分」**——残差写法不是一条
「提高终质」的自由度，但它是一条能移动「中间层状态离可用差多远」的自由度，只是 **pre-LN 单残差在
这条线上已经是较好的那个**（post-LN 明显更差）。预登记的否证条件（「clean bpc 与逐深度可读性**都不越带**
或优势全由参数解释 ⇒ 判不是独立维度」）**因逐深度可读性确实越带而未完全成立**，故**不判「残差形式没用」**；
反过来也**不得**把「中间深度被改动」当成终质收益。`status = 已完成`（有明确结论、非「手段不成立」，
故不判 `被否证`、`failure_analysis` 保持空）。

必须如实说明的混杂：post-LN 的残差流被每子层后的归一化压到 RMS≈0.87（base 是 1.5–2.1 的未归一温流），
读出（tied `h·Eᵀ`、无 final LayerNorm）看到的输入量级因此不同——这是契约 §3 预先写死的已知结构差异，
post-LN 中间深度变差**可能部分来自读出尺度**而非表征本身，本项只量化，不单独归因。

## 数字表

### A. 终质（P1，主判据）

| 臂 | 参数 Δ | clean bpc | Δ = base − form | 判词 |
|---|---|---|---|---|
| `base` | 0 | 1.5310 | — | 基线（stock pre-LN） |
| `postln` | 0 | 1.5284 | +0.0026 | 不可排序 |
| `gated` | +16（0.00023%） | 1.5326 | −0.0016 | 不可排序 |
| `scaled` | 0 | 1.5413 | −0.0103 | 不可排序 |

### B. 逐深度 bpc（P2；depth k = 跑完 k 块的读出，越低越可读）

| depth | base | postln | gated | scaled |
|---|---|---|---|---|
| 0 | 7.4079 | 7.3369 | 7.3697 | 7.4291 |
| 1 | 4.9516 | **10.1311** | 4.8652 | 4.5489 |
| 2 | 4.4377 | **8.5542** | 4.4838 | 4.0560 |
| 3 | 3.8648 | **7.9159** | 3.9324 | 3.3971 |
| 4 | 3.2078 | **6.1513** | 3.2271 | 2.7430 |
| 5 | 2.6131 | **4.7384** | 2.6402 | 2.2757 |
| 6 | 2.1649 | 3.2029 | 2.1522 | 2.0046 |
| 7 | 1.7200 | 2.2043 | 1.6999 | 1.6628 |
| 8 | 1.5310 | 1.5284 | 1.5326 | 1.5413 |

中间深度（k=1..7）`|Δbase−form|` 最大值：**postln 5.1795**（depth 1，负 = 更不可读）、
**scaled 0.4677**（depth 3，正 = 更可读）、**gated 0.0864**（depth 1）。三形式都 ≥ 0.014 ⇒
按 P2 字面规则都「改变了中间深度可读性」，否证条件第一条不成立。

### C. 活性 / 稳定性

| 臂 | 活性量 | 末窗 loss 斜率 | gnorm 末窗 | gnorm max | 发散 |
|---|---|---|---|---|---|
| `base` | 对照臂无干预 | +0.00204 | 0.4263 | 7.86 | 否 |
| `postln` | stream_rms≈0.86–1.0（base 1.5–2.1） | +0.00291 | 0.3925 | **19.81** | 否 |
| `gated` | 门终值偏离 1.0 absmax 0.8117；门梯度 absmax 0.0218 | +0.00220 | 0.3856 | 7.86 | 否 |
| `scaled` | α_l = 1.0…0.3536 逐层落盘 | +0.00295 | 0.4612 | 3.20 | 否 |

`post-LN` 的 `gnorm_max 19.81`（base 7.86）如实报：未发散、无 grad skip、末窗 gnorm 均值反而更低
（0.3925 vs 0.4263）⇒ 早期尖峰，不改判据。

### D. 等值门 / 变异自证 / P5

- `gated@init` 与 `base` 的 `logits`/`loss` **逐位相同**（`torch.equal`，CPU fp32）；`postln`/`scaled`
  在 init 处与 `base` 前向**必须不同**，二者均 true。
- P5 引擎等值门：base 1.5310 vs 历史 dense 锚 23d-dense-B-12k 1.5287 = +0.0023 **< 0.005 ⇒ PASS**。
- P6 等参数口径：`gated` +16 = 0.00023% ≤ 0.1% ⇒ 直接与 base 比；四臂共享 CLI 模板逐字相同。

## 命令与真实输出（CPU，不取 GPU 锁）

```text
$ python3 run.py --aggregate
（输出末尾）
  "verdict_by_form": {
    "postln": {"0": "不可排序（H0）：|Δ| < 0.014 ⇒ 在本尺度不是独立有增益的维度"},
    "scaled": {"0": "不可排序（H0）：|Δ| < 0.014 ⇒ 在本尺度不是独立有增益的维度"},
    "gated":  {"0": "不可排序（H0）：|Δ| < 0.014 ⇒ 在本尺度不是独立有增益的维度"}
  },
  "P5_engine_equivalence_gate": {"base_s0_clean_bpc": 1.531, "vs_hist_23d_dense_B_12k": 0.0023,
                                 "vs_2613_baseline_s0": 0.0053, "pass": true},
  "single_factor_check": {"all_arms_same_shared_cli": true, "n_arms": 4}
（results.json 的 source_files 已含 results_scaled_s0.json；四臂折齐）

$ python3 smoke.py
[smoke] device=cpu autocast=none params_added={'postln': 0, 'scaled': 0, 'gated': 16} pass=True
        gated@init == base (bitwise): True
[smoke] PASS (1 条路径)

$ python3 mutation_check.py
[mutation] M1 gate init 0.99: 红 ✓
[mutation] M3 postln→base: 红 ✓
[mutation] M4 forms not applied: 红 ✓
[mutation] 恢复后等值门 pass=True
[mutation] PASS: {'M1_gate_init_0.99': 'red', 'M3_postln_falls_back_to_base': 'red', 'M4_forms_not_applied': 'red'}

$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。

$ sha256sum runs/*/model.pt   # 见 MODEL.md
0df68815…  base / 63df6a58…  postln / 8f62e2cc…  gated / c8bbb8aa…  scaled
```

## 需裁决 / 纠正（方向与派单文字相反）

派单文字与本轮 ticket §9.8 记「post-LN 让中间深度大幅**更可读**」（/「postln 逐深度 bpc 在中深度
显著**更低**」），但落盘证据方向相反：post-LN 的中间深度 bpc **大幅更高**（depth 1 10.13 vs base 4.95）
= **更不可读**；「中间深度更可读」的正面读数落在 **`scaled`**（+0.40…+0.06）。我按落盘证据写
（analysis.md / project.yaml 两处标注），**未改任何测量数字**。

两次 `contact_supervisor(need_decision)` 均**超时无回复**，无法确认这是否只是措辞笔误。请复核者 /
主 Agent 裁决：
- **拟（已按此落盘）**：结论结构保持派单原意（终质维不可排序 / 可截断性维可分 / 否证未完全成立 / `status=已完成`），
  只把方向改回与证据一致：**「post-LN 让中间深度大幅更不可读，scaled 让中间深度小幅更可读但终质更差」**。
- 若主 Agent 认为应保留原方向措辞，需给出与证据相容的读法（例如把「可读性」另定义成别的量），
  否则该措辞与 `results_*_s0.json` 冲突、过不了 `SCHEMA.md` §12 口径对账。

## 残余缺口（已写进 `project.yaml::gaps`）

1. 单 seed（s0）：n≥2 跨 seed 未做；逐深度形状（尤其 post-LN）稳健性待 s1。
2. 无独立等参数对照臂：契约 §4-P6 选「额外参数 ≤0.1% ⇒ 直接与 base 比」，`gated` +16 的未剥离项保留。
3. post-LN 读出尺度混杂未剥离（stream_rms≈0.87 vs base 1.5–2.1；未做加 final LN 的对照臂）。
4. 与 2.1.1 早退的联合（可截断性端到端）未测，本项只读逐深度 bpc 代理。
5. 「残差随深度衰减」形式（如 α^l）未单列；本项 `scaled` 是固定 `1/√(l+1)`。
6. 不越尺度外推：单任务（enwik8 字符级）+ 13M/12k + dense/sequential 单轴单格。
