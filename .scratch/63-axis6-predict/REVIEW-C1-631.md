# REVIEW-C1-631

- 时间戳（`date -Iseconds`）：`2026-10-14T23:06:10+08:00`（报告撰写期间复核命令继续执行至约 23:16）
- 被审 revision（`git rev-parse HEAD`）：`6f8758a1344a4c153b7a531577f774fe530a8286`
- 复核对象：`axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m/**`（WP 6.3.1-WP1）
- 只读依赖：`axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/{results.json,analysis.md,padfill.py}`
- 复核者：独立 fresh 上下文，未参与写作；除本报告外**未新增/修改任何文件**、**未 commit**
- 纪律：不开 GPU；命令直跑，以下所有数字均为**本人亲自执行**所得，不引用他人自报值

## VERDICT: PASS

科学结论（§4 否证条件触发 ⇒「掩码扩散并行生成在本尺度 / 本任务上不成立」）可**独立复算、逐值吻合**，
判据方向无误，无 blocker。残留问题均为**未收尾 / 已声明 gap 的完整性缺陷**，不推翻主判。

---

## blocker

无。

## normal（完整性 / 未收尾，不影响主判）

1. **N1 · `RESUME.md` 未收尾（writer 超时残留）** — 位置 `RESUME.md`；违 `AGENTS.md` §4「不得停在部分完成」，
   以及 WP 契约「可回溯、可 resume」。证据：`RESUME.md` 仍写「执行中，GPU 锁被他人长占」「mdiff 训练 + K 扫描（未完成）」，
   而 `run_log.txt` 已记录 `23:01:52/23:05:21 train DONE`、`23:05:24 finalize DONE`、`results.json` 已落 verdicts。
   `RESUME.md` mtime `22:43`，早于 `results.json` mtime `23:05` ⇒ resume 文档与实际状态不一致。
   （按复核纪律不代其补写，如实列为 finding。）

2. **N2 · 契约 L3（外部锚 rescore）未交付，以 gap 替代** — 位置 `analysis.md` L3 行 / `results.json::L3`；
   违 WP 契约 §3 L3「报告齐全 + 声明方向性偏差」。实际 `L3.status="gap"`，理由「2671 AR checkpoint 未落盘」。
   契约 §2 允许 checkpoint 不可得时的复用 fallback，且本项已**显式声明**复用 2671 读数与信息集不对称；
   但 L3 本身（用冻结 AR 对生成序列 rescore + 配对差 + CI）确为未做，属已声明 gap，非科学缺陷。

3. **N3 · K=1 连接点未显式裁决** — 位置 `analysis.md`「K=1 连接点」段；违契约 §4 第 3 条的可诉性要求
   （应明确「在统计上是否可区分」）。本项已**并列** mdiff K=1 与 2671 `fill_bidir_r25` 的同 M Δ
   （+2.68152 CI[+2.57617,+2.79239] vs +1.07264 CI[+1.00339,+1.14548]），数字与两侧落盘**完全一致**（见验证 V3），
   二者 CI **不相交 ⇒ 统计上可区分**，故「连接点成立（不可区分）」这一条件**不满足**；但 `analysis.md`
   仅写「应落在附近」而未给出「可区分/不可区分」的明确结论，也未解释差异（变 r 训练 vs 固定 r=0.25 训练）。

4. **N4 · `results.json` 未落盘逐 block 原始配对数组（部分不可复算依赖 2671）** — 位置 `results.json::mdiff.*.per_K.*`。
   本项落盘 `s_mdiff_blocks` / `n_m_blocks`（mdiff 侧逐 block 原始量），但 **AR 侧逐 block `s_ar_M` 未复制进本项**，
   复算 Δ 的配对基准必须回读 `2671/results.json::ar.*.blocks`。契约 L0 称「同一位置集」，且 n_m 逐 block 对齐已核（V4），
   跨文件配对可复算（见 V1/V2）；但「本项自足可复算」依赖外部文件存活，属可复现性上的**轻微脆弱点**。

## info

1. **I1 · `supervisor.out` / `nohup.run.out` / `train_log.jsonl` / `__pycache__`** 为超时 worker 的运行时残留；
   契约 §6 交付物清单未列它们（不构成违规，仅目录整洁度）。
2. **I2 · `padfill.py` 的第二个 64-hex** 出现在 2671 记录中（`28e4efd5…`），非本项依赖，非问题。
3. **I3 · K 形状的统计可分性** — Δ 跨 K 极差仅 0.006（2.6815→2.6873），远小于 CI 半宽（≈0.11）；
   「提高 K 不能拉回带内」在**点估计**上成立（K16 ≥ K1），但 K 之间的差异本身在噪声内。
   主判不依赖于该形状：即使只看 K=1，Δ 也已越带 190 倍，故主判稳健。仅提示读者勿把 K 形状当作精细可分结论。

---

## 亲自执行的命令与真实输出（节选）

### V1 · 逐 K 独立复算 Δ 与 block 聚类配对 bootstrap CI（脚本 `/tmp/recompute.py`，**未 import `mdiff.py` 判据函数**）

复算口径：mdiff 逐 block bpc = `s_mdiff_blocks/n_m_blocks/ln2`；AR 逐 block bpc = `2671 ar.*.blocks` 的 `s_ar_M/n_m/ln2`；
逐 block Δ 后按 seed 对齐取均值，再对 64 个 block 做有放回配对 bootstrap（B=2000，seed=2673，percentile 95%）。输出：

```
K  seed  mdiff_meanblk  mdiff_pooled  stored   ar_meanblk  delta_meanblk  stored_delta  dmean-delta
 1 0  4.909650   4.905062  4.905063  2.204170  2.705480   2.705480   +0.00e+00
 1 1  5.031385   5.026631  5.026632  2.373820  2.657565   2.657565   -4.44e-16
 4 0  4.916565   4.911994  4.911993  2.204170  2.712395   2.712395   +0.00e+00
 4 1  5.031689   5.026944  5.026944  2.373820  2.657869   2.657869   +0.00e+00
 8 0  ...        stored_delta 一致 (差值 0)
16 1  5.032051   5.027267  5.027267  2.373820  2.658231   2.658231   -4.44e-16
COMBINED K= 1 delta=2.68152245 stored=2.68152245 CI=[2.576172,2.792386] stored=[2.576172,2.792386] excl0=True band=True
COMBINED K= 4 delta=2.68513205 stored=2.68513205 CI=[2.578978,2.795912] stored=[2.578978,2.795912] excl0=True band=True
COMBINED K= 8 delta=2.68660123 stored=2.68660123 CI=[2.580350,2.797737] stored=[2.580350,2.797737] excl0=True band=True
COMBINED K=16 delta=2.68730429 stored=2.68730429 CI=[2.581355,2.798831] stored=[2.581355,2.798831] excl0=True band=True
```

**结论**：4 个 K 的点估计与落盘**逐值一致（≤4.4e-16）**，bootstrap CI 与落盘**逐值一致（小数点后 6 位）**，
`ci_excludes_zero=True`、`|Δ|>0.014` 全部成立。Δ 走向 K=1:2.6815 < K=4:2.6851 < K=8:2.6866 < K=16:2.6873，
与 `analysis.md`「提高 K 不能拉回带内」一致。

### V2 · 否证条件方向核对（`run.py` 逻辑 + 原始读数）

```
run.py:418  c1 = comb["ci_excludes_zero"] and abs(comb["delta_mean"]) > REFERENCE_BAND
run.py:443  "delta_K16_less_than_K1": (d16 < d1)
```
落盘：`falsification.per_K[*].cond_single = true`（4/4）；`delta_K16_less_than_K1=false`（2.68730 > 2.68152，方向正确）；
`conditions.mask_diffusion_parallel_generation_fails.triggered=true`。
⇒ §4「任一 K CI 不含 0 且越带，且 Δ16 不见小于 Δ1」**确实被满足**，主判 `不成立` 由原始读数正确推出。
替代条件 `indistinguishable_at_this_scale` 正确 `triggered=false`（并非全部 CI 含 0）。
**未发现取键 / 比较方向导致的假触发或假未触发。**

### V3 · K=1 连接点两侧数字（从 2671 原始 `delta_blocks` 复算）

```
ar seed 0 ar_at_M pooled bpc= 2.2023510519263785 ar_full= 2.184217728309825
ar seed 1 ar_at_M pooled bpc= 2.371697209393906 ar_full= 2.3565269550426473
fill_bidir_r25 combined delta= 1.072639429223154
fill_bidir_r25 CI= [1.00339414 1.14547716]
2671 stored L1 delta/ci95: 1.072639429223154 [1.0033941361791663, 1.145477161598173]
```
`analysis.md` 的 mdiff K=1 Δ=+2.68152 CI[+2.57617,+2.79239] 与 `fill_bidir_r25` Δ=+1.07264 **均与两侧落盘一致**；
二者 CI 不相交 ⇒ 统计上**可区分**（见 normal N3，analysis 未显式给该裁决）。

### V4 · L0 实现保真 / M 对齐

`results.json::L0.m_align_2671_diff_list == []`（空）；本人复算 `n_m_blocks` 与 2671 `ar.*.blocks.n_m`
逐 block 相等（脚本内 `assert [ar[s][o][1] for o in offs] == nm` 通过，64/64）⇒ 同一位置集。

### V5 · 采样器守恒（契约 §3/§2）

`mdiff.py::sample_denoise_k`：`per_step_fix = torch.ceil(n_m/K)`（每 block 每步 ceil(|M|/K)），
`k==K` 时 `n_fix = remaining.sum()`（末步全定格），步末 `assert fixed==M`（守恒硬断言）。与契约 §2 逐字一致。

### V6 · gate probe 可失败性（在 `/tmp/probe/` 的**副本**上做变异，未改仓库任何文件）

```
== pristine (leak+conserve) ==  2 passed, 7 deselected in 1.62s
== mutate leak: x = torch.where(sel, true_tokens, x) ==
FAILED test_mdiff.py::test_no_true_leak_into_context - assert False
== mutate floor: torch.ceil(n_m/K) -> torch.floor(n_m/K) ==
FAILED test_mdiff.py::test_step_counts_conserve - assert 4 == 5
```
两个独立变异各自被对应测试**抓红** ⇒ `test_mdiff.py` 的 gate probe 确实可失败（非恒绿）。
仓库内 `python3 -m pytest test_mdiff.py -q` = **9 passed in 2.14s**。

### V7 · 守纪核对

```
live padfill.py sha256 = 67be5cc63536ffdfc87e35ab6276f5de844810efdf218cbb4b00fda85e7f2982
  与 2674 MODEL.md / 2674 results.json::config.padfill_sha256 / 2671 记录     完全一致 ⇒ 只读依赖未被改动
mdiff.py sha256 = 0c1227d9458744ff85a8d0009df52fb43b9f183ba266e1d2d77a47e323281d94  == MODEL.md 记录
run.py   sha256 = cc38fcab565e5c583ddd24b8da075d35ba56425d6017640e123d8a8ccabcb3e0  == MODEL.md 记录
results.json 墙钟语义字段：L5.wall_clock_fields=[]、_wall_clock_fields=[]；全文 "wall/clock/elapsed/duration"
  仅命中 2 处中文注释串（"禁墙钟"）与 2 个空字段名，**无墙钟数值字段**
git status --short -- pathlm docs tools  → 空（pathlm/** 无未提交改动）
git log --oneline -3 -- pathlm → fa6a1ad/4638882/e894f20，均为 ticket 67 引擎通道（E1/E2/E3）已提交
git diff --cached --name-only | wc -l → 0（无 staged 文件）
README.md mtime 2026-10-14T21:06:30  <  results.json mtime 2026-10-14T23:05:24  ⇒ 预登记早于结果
```

---

## not-run 清单

- **GPU 复跑 / 同 seed 逐值确定性重跑**：复核纪律「不开 GPU」，`determinism_rerun_K4_equal=True` 未由复核者独立重跑。
- **仓库内 `python3 mutate.py` 全量执行**：只读纪律**禁止**改动 `mdiff.py`，仓库版未跑；改以 `/tmp/probe` 副本做等价的 2 条独立变异（V6）。
- **L3 冻结 AR rescore**：AR checkpoint 未落盘（`analysis.md`/`results.json::L3` 已声明 gap），无法执行。
- **2671 侧现况复核**：本次仅**回读** 2671 的 `results.json`/`padfill.py` sha，未复核 2671 的其它交付物（超出本 WP 契约范围）。

## 收尾未收尾项清单

- `RESUME.md` 过期（未更新为已完成态）—— N1
- 契约 L3 未做、以 gap 计 —— N2
- K=1 连接点未显式裁决（可区分 vs 不可区分）—— N3
- 本项未自足复制 AR 逐 block `s_ar_M`，复算需回读 2671 —— N4

## 关键复现数字（逐 K 的 Δ 与 CI，相对 ar_at_M；两 seed 合并）

| K | Δ (bpc) | 95% CI | CI 含 0 | 越 0.014 带 |
|---|---|---|---|---|
| 1 | +2.68152245 | [+2.576172, +2.792386] | 否 | 是 |
| 4 | +2.68513205 | [+2.578978, +2.795912] | 否 | 是 |
| 8 | +2.68660123 | [+2.580350, +2.797737] | 否 | 是 |
| 16 | +2.68730429 | [+2.581355, +2.798831] | 否 | 是 |

`Δ(K=16)=2.68730 > Δ(K=1)=2.68152` ⇒ 提高 K **不见**拉回 ⇒ §4 触发，主判「不成立」成立。
