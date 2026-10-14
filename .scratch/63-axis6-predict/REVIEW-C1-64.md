# REVIEW-C1-64

- **复核者角色**：轴 6 独立复核者（fresh 上下文、未参与写作、只读；仅新增本报告文件）
- **时间戳**：2026-10-14T22:46:23+08:00（`date -Iseconds`）
- **被审 revision**：`555d4abf0ced59c68e7c60eeaca69dc6b76245d4`（`git rev-parse HEAD`）
- **复核对象**：WP 6.4-WP1b + WP2 交付物，目录
  `axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/**`（预登记 `README.md`；聚合
  `results.json`；`analysis.md`；`MODEL.md`；`RESUME.md`；`run_log.txt`；`rescore-seed0.json` /
  `rescore-snap2k.json`；`rescore_eval.py` / `aggregate.py` / `config_match.py`；6 个 `snap2k-*/` run）；
  只读对照 `.scratch/23-depth-ar/evidence/{23c-H2-w0-12k,23e-nofill-12k,23e-projfill-12k}/`
- **边界遵守**：未改除本文件外任何文件；未 `git commit`；未开 GPU；未仓库遍历。

## VERDICT: PASS

无 blocker。2 条 normal（纪律面、不改变结论），3 条 info。所有关键读数经独立复算与落盘逐值一致；
快照口径与 n≥2 的声明如实。

---

## 1 亲自执行的命令与关键输出

### 1.1 环境与边界

```
$ date -Iseconds ; git rev-parse HEAD
2026-10-14T22:46:23+08:00
555d4abf0ced59c68e7c60eeaca69dc6b76245d4

$ git status --short -- pathlm docs tools        # 空 ⇒ 本 WP 未动 pathlm/docs/tools
（exit 0，无输出）

$ git diff --cached --name-only | wc -l
0                                                # 无 staged

$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。   # exit 0
```

被审目录全部为 untracked（`??`，33 条目），落在
`axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/**` 之内，未见该 WP 写越界。

### 1.2 独立复算臂级读数（自写 `/tmp/review_recompute.py`，不 import 本目录 aggregate.py）

从每个 `snap2k-*/results.json::depth_curve` 直取 depth-8/depth-4 bpc，与聚合
`results.json::point_values` 逐值比对：

```
snap2k-fill-s0     recon depth8=2.7995 vs agg=2.7995  depth4=2.8016 vs 2.8016  OK
snap2k-nofill-s0   recon depth8=2.8770 vs agg=2.877   depth4=2.8794 vs 2.8794  OK
snap2k-projfill-s0 recon depth8=2.8292 vs agg=2.8292  depth4=2.8309 vs 2.8309  OK
snap2k-fill-s1     recon depth8=2.7580 vs agg=2.758   depth4=2.7608 vs 2.7608  OK
snap2k-nofill-s1   recon depth8=2.8221 vs agg=2.8221  depth4=2.8252 vs 2.8252  OK
snap2k-projfill-s1 recon depth8=2.7798 vs agg=2.7798  depth4=2.7821 vs 2.7821  OK
```

Δ 复算（点值相减）：

```
seed0: d8 fill-nofill=-0.0775 agg=-0.0775 | projfill-fill=0.0297 agg=0.0297 | d4 fill-nofill=-0.0778 agg=-0.0778
seed1: d8 fill-nofill=-0.0641 agg=-0.0641 | projfill-fill=0.0218 agg=0.0218 | d4 fill-nofill=-0.0644 agg=-0.0644
```

全部与 `analysis.md` §2 / `results.json::deltas` 逐值一致。

### 1.3 config 对平（独立 python 逐字段 diff，未调用本目录 `config_match.py`）

```
=== same-scale seed0 vs seed1, per arm (config.* + train.*) ===
fill      config diffs: NONE | train seed 0 -> 1
nofill    config diffs: NONE | train seed 0 -> 1
projfill  config diffs: NONE | train seed 0 -> 1

=== snap2k-<arm>-s0 vs 12k evidence, per arm ===
fill     vs 23c-H2-w0-12k  config diffs: 仅有「snap2k 显式键 vs 12k 记录缺席(None)」19 个键 + train.steps(2000 vs 12000)
nofill   vs 23e-nofill-12k config diffs: 同上 17 键 + train.steps
projfill vs 23e-projfill-12k config diffs: 同上 17 键 + train.steps

=== train_tokens / params across all ===
snap2k-fill-s0     16384000  6962688
snap2k-nofill-s0   16384000  6962688
snap2k-projfill-s0 16384000  7489024
snap2k-fill-s1     16384000  6962688
snap2k-nofill-s1   16384000  6962688
snap2k-projfill-s1 16384000  7489024
23c-H2-w0-12k      98304000  6962688
23e-nofill-12k     98304000  6962688
23e-projfill-12k   98304000  7489024
```

对缺席键是否「= 引擎缺省」，独立导入引擎核对：

```
$ python3 -c "from dataclasses import asdict; from pathlm.depth_ar import DepthARConfig; ..."
attn_linear=False  attn_sparse_k=0  carry_chunk=64  carry_state_dim=None  aux_mode='renorm'
carry_decay=1.0  topology='sequential'  train_window=0  train_anchors=0
readout_gated=False  readout_perdepth=False  readout_wide_shared=0
readout_linear_resid=False  recon_window=0  linear_chunk=64  mod_ratio=0.0
n_kv_heads=None  carry_gate=True  carry_per_layer=False
$ grep -n '"--carry-write"' train_depth_ar.py        # default="none"
$ grep -n '"--no-fill-kv"' train_depth_ar.py         # store_false, 缺省 True
```

3 臂 × 2 seed 齐备（`snap2k-{fill,nofill,projfill}-s{0,1}` 6 个目录，各含
`model.pt`/`train_log.jsonl`/`gate_log.jsonl`/`results.json`）；`missing_runs=[]`；
`train_tokens` 臂间/ seed 间均 = 16384000（逐位相同）；`fill`/`nofill` params = 6962688 相等，
`projfill` 增量 = 7489024−6962688 = 526336 = 8 层 ×(256×256+256)（逐层恒等初始化 adapter，符合声明）。

### 1.4 rescore 表复算（`/tmp/review_rescore.py`）

"峰值" = 各序列 max-over-depths（本数据下 soft/hard 峰值都落在 depth 0，故 Δpeak = depth-0 Δ）：

```
23c-H2-w0-12k     peak soft=25.4544 hard=2.5519 latent=3.1674 | dpeak(soft-hard)=+22.9025
23e-nofill-12k    peak soft=16.6289 hard=2.2125 latent=4.8612 | dpeak(soft-hard)=+14.4164
23e-projfill-12k  peak soft=18.3808 hard=3.8551 latent=5.5188 | dpeak(soft-hard)=+14.5257
snap2k-fill-s0    dpeak(soft-hard)=+3.8325  （d0/d4/d8 = +3.8325/+1.2396/+1.5686）
snap2k-fill-s1    dpeak(soft-hard)=+6.2138  （d0/d4/d8 = +6.2138/+3.8994/+2.3330）
snap2k-nofill-s0  dpeak(soft-hard)=+5.8908  （d0/d4/d8 = +5.8908/+4.7713/+4.6671）
snap2k-nofill-s1  dpeak(soft-hard)=+6.5625  （d0/d4/d8 = +6.5625/+2.9540/+3.1157）
snap2k-projfill-s0 dpeak(soft-hard)=+6.2739 （d0/d4/d8 = +6.2739/+0.8138/+1.2723）
snap2k-projfill-s1 dpeak(soft-hard)=+4.7332 （d0/d4/d8 = +4.7332/+0.8289/+1.2646）
```

全部与 `analysis.md` §4 各表逐值一致；9 个 checkpoint 的 `stored_match=true`、`stored_diffs=[]`。

「同一 checkpoint ⇒ 参数逐位相同」的依据独立核实：
`rescore_eval.py::load_checkpoint` 每个 `run_dir` 只 `torch.load` 一个 `model.pt`，
`generation_quality()`（`pathlm/depth_ar.py:3097`）先 `model.eval()`，仅按 `commits` 参数循环
三种提交规则、**无梯度/无参数更新** ⇒ 三种提交共享同一权重；`MODEL.md` §1 给出各 `model.pt` sha256，
抽验一致：

```
$ sha256sum snap2k-fill-s0/model.pt      # ee544a55…b179fa6  与 MODEL.md 一致
$ sha256sum snap2k-projfill-s1/model.pt  # 03bbc75c…ea61c82  与 MODEL.md 一致
```

### 1.5 预登记未放宽 / 禁墙钟

```
$ stat -c '%y %n' README.md results.json analysis.md
2026-10-14 20:54:42 README.md
2026-10-14 22:39:49 results.json
2026-10-14 22:40:37 analysis.md
```

`README.md`（预登记契约）mtime 20:54:42 早于 `results.json` 22:39:49，也早于首 run（`run1-fill-s1.*`
≈21:54）⇒ 预登记写在跑之前，事后未放宽。

禁墙钟键扫描（聚合 `results.json` 判据区）：

```
results.json wall-like hits: ['/l5_wall_clock_fields=[]']   # 仅自检字段本身
l5_pass True  l5_wall_clock_fields []
analysis.md 内 wall/elapsed/墙钟/秒 仅命中第 199 行 L5 结论文字（无墙钟数字）
```

---

## 2 分级清单

### blocker —— 无

### normal-1 · L5「墙钟只允许进 run_log.txt」在**非判据区**的辅助文件里未被字面满足

- **位置**：`rescore-snap2k.json` / `rescore-seed0.json` 每个 checkpoint 的 `elapsed_s`
  （8.5/8.8/7.4/11.5/9.5/10.7；19.7/21.9/16.9）；`snap2k-*/results.json` 的 `wall_minutes`
  （1.56/1.54/1.82/1.58/1.42/1.87）。
- **违的判据**：契约 §3 L5「`results.json` / `analysis.md` 的判据区不得含墙钟字段……墙钟只允许进
  `run_log.txt`」的最后一句。
- **复现证据**：见 §1.5 键扫描 + `python3 -c` 打印 `elapsed_s`；`results.json::l5_pass=True`、
  `l5_wall_clock_fields=[]`。
- **判读**：L5 的**判据区自检**（聚合 `results.json`）通过，`analysis.md` 判据区无墙钟数字 ⇒
  核心 L5 结论不为假。`snap2k-*/results.json::wall_minutes` 是**引擎自带的产物字段**，在只读对照的
  23e 三个 `results.json` 里也存在（`grep` 命中），非本 WP 自造；真正由本 WP 自己写入的越界项只有
  `rescore_*.json` 的 `elapsed_s`。属纪律瑕疵，不改变任何结论，故列 normal 而非 blocker。

### normal-2 · 交付物目录名与契约 §6 不一致（`run1-<arm>-s1/` → `snap2k-<arm>-s{0,1}/`）

- **位置**：契约 §2/§6 写 `run1-<arm>-s1/`；实际目录为 `snap2k-fill-s0` 等 6 个，另有遗留
  `run1-fill-s1.nohup.log` / `run1-fill-s1.train.log`（首次派发的残迹）。
- **违的判据**：契约 §6 交付物命名（形式面）。
- **复现证据**：`ls -d */` 只列 6 个 `snap2k-*/`；`ls run1-*` 只有两个 `.log`。
- **判读**：因 §2 授权缩步数、且必须「臂间同刻度」，seed-0 需在 2k 重跑（不能沿用 12k 的 seed-0），
  故命名随之变为 `snap2k-<arm>-s{0,1}`，`RESUME.md` 已说明。功能与判据不受影响，属形式瑕疵。

### info-1 · 「峰值」定义为各序列 max-over-depths，本数据下等于 depth-0 Δ

`analysis.md` §4 的 `Δpeak(soft−hard)` = max(soft) − max(hard)，在本 9 个 checkpoint 上与 depth-0 的
Δ 数值相同（soft/hard 峰值均落 depth 0，见 §1.4）。定义一致且逐深度表同时给出，不构成账实不符；
仅提示读者该「峰值」非逐深度配对量。

### info-2 · seed-0 臂是**新跑的 2k run**，不是契约 §2 预告的「既有 seed 0 读数」

契约 §2 原意「seed 0 已有 12k 读数 ⇒ 本轮只跑 seed 1」；实际 6 run 全部 2k（seed 0 重跑）。
这是「臂间同刻度」的必然要求（seed1 缩到 2k 后，若 seed0 仍取 12k 则 n≥2 不再同刻度），
12k 的 seed-0 读数作方向参照另表列出。已在 `README.md` §2 预算声明 / `scale.scale_note` /
`RESUME.md` 显式登记，未砍臂/砍 seed/改配方。

### info-3 · WP2 的机制措辞（「注入分布外内容」）强于直接测量

直接测量量是 `rescore_bpc`（无独立 OOD 指标）。可被证据支持的强命题是「同一 checkpoint、唯一变量 =
提交规则 ⇒ 差异非容量效应」，这正是契约 §3 L2 的判据；「OOD 污染」为其机制命名，与契约措辞一致。

---

## 3 需判断的证伪项逐条结论

- **A 快照口径**：**如实**。`analysis.md` 首段显式声明「6 run 全部 2k 快照口径（2000 步、16.38M
  train_tokens），与既有 12k 档（98.3M）不可比，只读臂间增量与排序，不读绝对值」；`results.json
  ::scale.scale_note` 与 `reference_12k.note` 同口径；§3 12k 参照表标注「**不可与上面直接比**」。
  未见把快照读数当作 12k 可比结论的写法；反而显式指出 `Δ(projfill−fill)` 在 2k 为正、12k 为负 ⇒
  标注「该排序依赖训练步数，不能从 2k 快照外推」。契约 §2 的「臂间同刻度 + 显式声明不可比」两条
  均已满足。
- **B n≥2 是否成立**：**成立**。3 臂 × 2 seed = 6 run 全部落盘且成功，`missing_runs=[]`；
  config 逐字段对平（同刻度种子对差异为空），`train_tokens` 臂间逐位相同；无补零、无缺臂。
- **C 等参数剥离**：**成立**。同一 `run_dir` 只加载一个 `model.pt`、三提交规则共用权重且评测无梯度；
  `MODEL.md` 提供 sha256（抽验一致）；`stored_match=true`（重放与 checkpoint 自带 generation 逐位一致）
  ⇒ soft−hard 差**不是**容量效应。措辞与证据相称（见 info-3）。
- **D 账实相符 / 事实文档纪律**：**相符**。≥2 臂 depth8/depth4、≥1 Δ、rescore soft−hard 峰值全部
  独立复算与落盘逐值一致；`analysis.md` 无日期流水 / 变更记录 / TODO 混入判据区（grep 无命中）；
  未验证项（逐 prompt rescore 原始数据、本目录变异抓红、matched-step 12k 复本）如实列入 §8。
- **契约 §4 否证条件**：§4.1 未触发（`Δ(fill−nofill)`、`Δ(projfill−fill)` 两种子同号）；
  §4.2 未触发（3 臂 Δpeak 两种子同号、CI 不含 0，并如实声明 n=2 区间退化）；§4.3 未触发（无单点显著的臂）。
  `analysis.md` §7 的声明与落盘证据一致。

---

## 4 not-run 清单（我未做 / 对方未做，均如实）

- 我**未**重跑训练或 rescore（无需 GPU；`SCHEMA.md` §20 取锁协议未触发，未杀他人进程）。
- 我**未** import 本目录 `aggregate.py` 复算（按要求自写 `/tmp/review_recompute.py`、
  `/tmp/review_rescore.py`）。
- 交付物未做项（对方已在 `analysis.md` §8 如实登记，我未代做）：
  - `mutate.py` + `mutation_report.json`（契约 §6 的 ≥3 条变异抓红）——**未做**，声明沿用引擎既有单测。
  - 逐 prompt 的 rescore 原始数据——**未落盘**，故 checkpoint 内抽样不确定度**未测量**。
  - matched-step（12000 步）的 n≥2 复本——**未跑**（本轮为 2k 快照口径）。

---

## 5 一句话结论

**快照口径如实声明、n≥2 真实成立**：6 个 run（3 臂 × 2 seed）齐备且同刻度，config 除 `seed`
（同刻度）/`steps`（跨刻度）外逐字段对平，所有臂级 depth-8/4 bpc、Δ、rescore soft−hard 峰值
经独立复算与落盘逐值一致；未见把 2k 快照当 12k 可比结论、未见补零或账实不符。**VERDICT: PASS**
（仅剩 2 条 normal 级形式/纪律瑕疵，不影响结论）。
