# WP-2651 报告 · 轴 2 深度轴 · 2.2.3 + 2.2.4（seed=1 复核 + 并行子层代价面）

- 实验目录：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/`
- 契约：同目录 `README.md`（预登记，判据线逐字来自 `.scratch/61-axis2-depth/TICKET.md` §9.2 WP-2651）
- 口径锚（s0 = seed 0）：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2613-inter-layer-topology-13m/`
- 本轮 writer：WP-2651 收口 writer（fresh 上下文，**只做收口，不训练**）
- 硬边界：只写 `axes/2-depth/2.2.3-learned-layer-sum/**`、`axes/2-depth/2.2.4-parallel-sublayer/**`
  与本文件；**未改写 `results.json`**；未碰 `2652/2653/2654/2655`、其它轴、`TICKET.md`、`pathlm/**`、
  `train_depth_ar.py`、`tools/**`、`docs/**`。

## 0. 人话摘要（先读这一段）

这一格问的是**「变压器层与层之间的接法换个花样，质量会不会变、代价是什么」**。上一格
`2613` 在 seed=0 上试了两种换法：**并行子层**（注意力与 MLP 同时读层输入，PaLM/GPT-J 式）
和**层间学习加权和**（每层输入＝对之前所有层输出的学习加权和，DenseFormer 式）。本轮契约
是**把每种接法在 seed=1 上再跑一遍（n=2）**，并把并行子层的**代价面**与质量面同框。

**实际做到哪一步（诚实先说）。** 截止 `2026-10-14T22:34+08:00`，seed=1 的四条臂里**只有
基线 `sequential` 训完并评测了**（1.5288 bpc）；另外三条（`parallel` / `parallel-dual-ln` /
`dense`）因为本机 `/tmp/pathlm_gpu.lock` 被**其它轴的多臂驱动持续独占**（锁释放后约 1 秒内
被重取，等不到 `SCHEMA.md` §20.1 的 ≥60s 礼貌窗口），连续多轮**等锁超时**，**没有跑成**。
所以本报告**不能**给出 seed=1 的干预臂读数；它给出的是：①用 s0（2613）四臂下判词；
②用 s1 的**基线**读数度量「换 seed 的噪声」；③**代价面**（CPU 解析量，与训练无关）；
④两份 `project.yaml` 的判词与本实验的 `analysis.md` / `MODEL.md`。**判据线未改。**

**三句话结论。**

1. **2.2.3 层间学习加权和 = 本尺度无净增量（`被否证`）**：s0 上 `dense − sequential` 配对
   Δ = **−0.0008 bpc**，远小于 0.014 判据带（也小于换 seed 噪声 0.0034）；而且它**没有**换来
   中间深度可读性（s0 中间深度**反而更差**：depth 1–7 的 Δ = +0.23…+0.54，只有最终 depth 8
   回到 −0.0008）。⇒ 按本项预登记的 falsification（「不越带 **且** 逐深度可读性无改善 ⇒
   无净增量」），密集跨层线性聚合在本尺度被否证。干预是活的（层权 α fro 3.305、36 个权重
   全非零、浅层输入被放大 2.36×），所以不是「干预没生效」。
2. **2.2.4 并行子层 = 场景依赖的取舍（`已完成`）**：质量面要付**可排序的代价**
   （+0.0458 / +0.0402，都是 0.014 带的 3 倍以上）；代价面给出**真实收益**——层内串行链
   从 **2 段砍到 1 段**、每 token 串行步数 **16 → 8**（同一时间能并起来的活儿多一倍），
   而算力只差 **0.055%**（等参数臂与基线**逐位相同**）。⇒ 以可排序的质量代价换层内并行度
   翻倍：**本尺度上不是免费的，是否采纳取决于场景**。
3. **边界**：两条都是 **n=1**（seed=1 三臂未跑）；不越尺度外推。

## 1. 本轮改动（只 4 类文件 + 本报告）

| 文件 | 改动 |
|---|---|
| `.../2651-inter-layer-topology-s1-13m/analysis.md` | 新建：臂完整性 + 四臂×两 seed 表 + config 逐字段对照（SCHEMA §9 三条等值门）+ 配对 Δ 表 + 2.2.4 代价面同框表 + §12 三对账 + §13 对照三要素 + 干预活性 + 口径标注 + 2.2.3/2.2.4 判词（含 C2 字面/实质两读法）+ 残余缺口 + 补记指引 |
| `.../2651-inter-layer-topology-s1-13m/MODEL.md` | 新建：s1 权重清单（sha256 + 体积 + 生成命令；未生成臂如实标注）+ s0 只指针 |
| `axes/2-depth/2.2.3-learned-layer-sum/project.yaml` | `status` 进行中→**被否证**；补 `conclusion`/`failure_analysis`/`conclusion_review`；改 `argument`；`gaps` 重写（记 n≥2 与成本）；`experiments` 加 `2651/analysis.md`；WP1 进行中→**已完成**；`change_log` 追加 |
| `axes/2-depth/2.2.4-parallel-sublayer/project.yaml` | `status` 进行中→**已完成**；补 `conclusion`/`conclusion_review`；改 `argument`；`gaps` 重写；`experiments` 加 `2651/analysis.md`；WP1 进行中→**已完成**；`change_log` 追加 |
| `.scratch/61-axis2-depth/WP-2651-REPORT.md`（本文件） | 覆盖上一版：人话摘要 + 数字表 + 命令与真实输出 + 残余缺口 + 补记指引 |

**未做的事**：没有改写 `results.json`（按契约只读；它由后台 launcher 逐臂增量写）；
没有改 `run.py` / `smoke.py` / `test_run_persistence.py` / `README.md` / `config.json` 的内容
（`config.json` 只被**重建回规范态** `steps=12000`：§6 命令 2）。

## 2. 臂完整性（as of `2026-10-14T22:34+08:00`）

| 臂 | s1 训练 checkpoint | s1 读数 | 说明 |
|---|---|---|---|
| `sequential` | **已完成**（12k 步 / 98.3M token） | clean_bpc **1.5288**、tf@4 0.4215 | 唯一完成的 s1 臂 |
| `parallel` | **未跑** | **missing** | 等 GPU 锁超时 |
| `parallel-dual-ln` | **未跑** | **missing** | 同上 |
| `dense` | **未跑** | **missing** | 同上 |

- **未跑三臂的等效算力成本**：每条 12k 臂 = `12000 × 32 × 256 = 98,304,000` token
  （≈6–8 min 卡时）；三条合计 **294.9M token**。等锁超时**没有消耗任何 token**，只是把
  三条臂推迟 ⇒ 状态是「**受阻**」，不是「失败」。
- **阻塞原因**：`/tmp/pathlm_gpu.lock` 被其它轴的多臂驱动持续独占（释放后 ~1s 重取）。
  本轴未杀他人进程、未动他人锁。

## 3. 数字表

口径：enwik8 char 级 AR，d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态，
12000 步（98.3M token）matched-step 快照；`clean_bpc` = 全深度（depth-8）逐位 clean bpc
（越低越好）；`tf@4` = 用前 4 层当草稿被全深度接受的比例。

| 臂 | s0 clean_bpc | s0 tf@4 | s0 参数量 | s1 clean_bpc | s1 tf@4 | s1 参数量 |
|---|---|---|---|---|---|---|
| `sequential`（基线） | **1.5254** | 0.4560 | 6,962,688 | **1.5288** | 0.4215 | 6,962,688 |
| `parallel` | 1.5712 | 0.3486 | 6,958,592 | **missing** | **missing** | （期望 −4096） |
| `parallel-dual-ln` | 1.5656 | 0.3149 | 6,962,688 | **missing** | **missing** | （期望 逐位等参） |
| `dense` | 1.5246 | 0.4538 | 6,962,724 | **missing** | **missing** | （期望 +36） |

**配对 Δbpc（判据带 0.014）**：s0 `dense−seq` = **−0.0008**（不越带）；
s0 `parallel−seq` = **+0.0458**、`parallel-dual-ln−seq` = **+0.0402**（均越带）；
s1 全部 **missing**。**换 seed 噪声**（`sequential` s0→s1）= **0.0034 bpc**（=带的 24%）。

**s0 逐深度 clean bpc（`dense` 对基线；中间深度更差，只有最终深度回到 parity）**：

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8（判据）** |
|---|---|---|---|---|---|---|---|---|---|
| `sequential` | 7.4039 | 4.8932 | 4.3486 | 3.8608 | 3.2023 | 2.6281 | 2.1895 | 1.7381 | **1.5254** |
| `dense` | 7.3786 | 5.1192 | 4.8923 | 4.3084 | 3.5141 | 2.8340 | 2.4466 | 2.0173 | **1.5246** |
| `dense − seq` | −0.025 | **+0.226** | **+0.544** | **+0.448** | **+0.312** | **+0.206** | **+0.257** | **+0.279** | **−0.0008** |

**2.2.4 代价面（等效理论量，不含任何时间量；`run.py::topology_cost_face`）**：

| 接法 | 层内串行依赖 | 每 token 串行步数 | 全栈 MACs/token | vs 基线算力 | 参数量 Δ |
|---|---|---|---|---|---|
| `sequential` | 2 段 | 16 | 7,400,960 | 100.000% | 0 |
| `parallel` | **1 段** | **8** | 7,396,864 | **99.945%** | **−4096** = −L·2d |
| `parallel-dual-ln` | **1 段** | **8** | 7,400,960 | 100.000% | **0（逐位相同）** |
| `dense` | 2 段 | 16 | 7,408,128 | 100.097% | **+36** = L(L+1)/2 |

**config 等值（`SCHEMA.md` §9）**：`config.json::vs_2613.unequal_fields = ['seed', 'experiment']`、
`only_seed_and_name_differ = true`（29 行逐字段对照）——本实验相对 2613 的唯一科学变量 = seed 0→1。
引擎等值门：`run.py::remeasure_s0` 用当前电池重测 2613 四条 checkpoint **逐位一致**
（`run_log-attempt1-crash.txt:3` `[s0] remeasured all_bitwise_equal=True`）。

## 4. 判词

- **2.2.3 层间学习加权和 = `被否证`**（机制「密集跨层线性聚合在本尺度有净增量」被否证）。
  归因（先区分 设计 / 结构 / innovation）：**结构问题（主因）** L=8 太浅 + 仅最终深度监督，
  DenseFormer 要缓解的串联瓶颈在本尺度不成立（中间深度读出变差、最终深度靠后续层学回）；
  **设计问题（次因、可修、非主因）** α 无单纯形约束、无中间监督，但干预活性成立 ⇒ null 不能
  归因于「干预没生效」；**非 innovation 本身失败**（更大规模未复核）；**非实现/口径假象**
  （等值门 + 参数量账逐位核对通过）。**边界 n=1**。
- **2.2.4 并行子层 = `已完成`**：质量代价可排序（+0.0458/+0.0402，>3× 带）+ 代价面真实收益
  （串行链 2→1 段、行步数 16→8、算力差 0.055%）⇒ 不满足 falsification 的「代价面无收益」条件
  ⇒ 结论 = **以可排序的质量代价换层内并行度翻倍，本尺度上不是免费的，是否采纳取决于场景**。
  **边界 n=1**。
- 两条都按 `docs/research-schema.md` §5 C2 给出「**字面成立 / 实质否证（或实质给出权衡）**」
  两读法（见 `analysis.md` §9）。

## 5. 命令与真实输出

```bash
EXP=axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m

# 1) 语法核对 + 回归测试（CPU，不取 GPU 锁）
python3 -m py_compile "$EXP/run.py"                       # -> COMPILE OK
python3 -m pytest "$EXP/test_run_persistence.py" -q       # -> 4 passed in 2.06s

# 2) 重建规范 config.json（冒烟把 --steps 20/2 臂写回了它；用 run.py 自己的 config_vs_2613 重建）
python3 - <<'EOF'
# 以 steps=12000 / seed=1 / 四臂重建 cfg_dump，调 run.py::config_vs_2613 后写回 config.json
EOF
# -> unequal_fields = ['seed', 'experiment']; only_seed_and_name_differ = True; n_rows = 29
#    （重建后 `config.json` 的 steps = 12000、arms = 四臂）

# 3) 产物/契约校验（只校验，不改）
python3 tools/check_tree.py                               # -> OK：46 个 project.yaml 全部符合 docs/research-schema.md。
python3 tools/summary.py --check                          # -> OK：46 个 project.yaml 全部符合 docs/research-schema.md。

# 4) 代价面（run.py 自带函数，非手抄；原始值同时落在 results.json::cost_face_2_2_4）
#    sequential segs=2 serial=16 total=7400960 frac=1.0
#    parallel   segs=1 serial=8  total=7396864 frac=0.999447
#    parallel-dual-ln segs=1 serial=8 total=7400960 frac=1.0
#    dense      segs=2 serial=16 total=7408128 frac=1.000969

# 5) 机会主义补臂（见 §6 的处置说明）：锁被轴 6 独占，本轮窗口内未取到
cat /tmp/pathlm_gpu.lock/name        # -> wp2672-snap2k-nofill-s0 / -projfill-s1（轴 6，持续重取）
tail -4 "$EXP/run_stdout-smoke.txt"  # -> 多轮 "[sh] smoke attempt N smoke_ok=no"（等锁 10 min 超时）
```

**真实输出（关键行）**：

```
[cfg] vs 2613 unequal fields = ['seed', 'experiment']          # run_log-attempt1-crash.txt:1
[lock] acquired pid=2283632                                     # :2
[s0] remeasured all_bitwise_equal=True                          # :3   ← 引擎等值门证据
```
```
[arm] sequential: clean_bpc=1.5288 tf@4=0.4215 params=6962688 wall=9.62 min   # run_stdout-smoke.txt
```

## 6. 机会主义补臂的处置（偏离登记）

- **契约给的字面命令** `python3 run.py --arms parallel,parallel-dual-ln,dense --skip-s0-anchor`
  **会被 `run.py` 直接拒绝**：`main()` 要求 `--arms` 必含 `sequential`
  （`if unknown or "sequential" not in want: raise SystemExit(...)`）。故字面命令无法执行。
- **实际情况**：本实验目录下**已有一个后台 launcher `run_remaining.sh` 在跑**（前一位 writer
  留下的，`setsid` 脱离，最多 12 轮、每轮最多等锁 10 min），它先做 20 步冒烟、冒烟成功后才跑
  全跑四臂。**为不破坏「`results.json` 单一写入者」**（契约：该文件由后台 launcher 增量写，
  我只读），本轮**没有另起第二个 `run.py`**（两个 run.py 会争同一把锁、并在臂边界并发写
  `results.json`，有损坏风险）。
- **本轮做的一件事**：`touch "$EXP/.smoke_ok"`，让该 launcher **跳过 20 步冒烟、直接跑全跑
  四臂**。理由：①冒烟用 `--steps 20 --arms sequential,parallel`，一旦取得锁会把**20 步的
  `parallel`** 写进 `results.json`、并把 `config.json` 写成 `steps=20`（本轮 `config.json`
  就已被这样写脏过一次，已按 §5 命令 2 重建）；②全跑命令是
  `python3 run.py --arms sequential,parallel,parallel-dual-ln,dense --caps-min 14 --budget-min 40`，
  正是契约要的「机会主义补臂」，且启动即把 `config.json` 写回规范态；③20 步冒烟要验证的
  `run.py` 崩溃已由 `test_run_persistence.py`（4 passed）与 2026-10-14T22:31 的一次真实
  电池运行（`[arm] sequential: clean_bpc=1.5288 ...`）钉住。
- **结果**：launcher 仍在重试；截至本报告，三条臂仍未落地（锁被轴 6 独占）。这**不改变**
  任何判词（判词只依赖 s0 四臂 + s1 基线 + 代价面）。

## 7. 残余缺口（如实登记，附等效算力成本）

1. **s1 三臂未跑**（`parallel` / `parallel-dual-ln` / `dense`，seed=1）：跨轴 GPU 锁饥饿所致。
   成本 = **每条 98.3M token（≈6–8 min 卡时），三臂合计 294.9M token**。⇒ 2.2.3 / 2.2.4
   的 n=2 边界未闭合。
2. **更大规模点未跑**（2.2.3 milestone 要求）。
3. **`dense` 的 α 无单纯形约束**（≠ DenseFormer 严格凸组合语义），严格对照未做。
4. **`dense` 中间深度更差**是新读数，与 2.2.1 残差形式 / 2.1.1 早退的耦合未做。
5. **`results.json::config.steps` 目前是 20**（最后一次成功调用是冒烟）；规范的 resolved
   config 是 `config.json`（`steps=12000`，已重建）。全跑启动会把 `results.json::config`
   一并写回。**本报告判词不依赖 `results.json::config`。**
6. **独立复核未做**（本轮 writer 自测；复核须由 fresh 上下文在 writer 停止后另派）。

## 8. 给下一位 lead 的补记指引（若 s1 三臂随后落地）

`run.py` 已按臂增量落盘；**只需用现有 `results.json` 重算，不重训**。逐表对应：

| 要重算的表 | 来源（现成字段） |
|---|---|
| §2 与 §3 的「四臂 × 两 seed」表 | `results.json::arms.<topo>.{clean_bpc,tf@4,next_token_acc,params}`（s1）+ `2613/…/results.json`（s0）；`missing` 格填实数 |
| §3 配对 Δbpc 表 + n=2 判定 | `results.json::cross_seed_s0_s1`（`delta_s0`/`delta_s1`/`same_sign`/`both_exceed_band`/`sortable_two_seed`）与 `cross_seed_summary`（`dense_sortable_two_seed`、`sequential_seed_spread`） |
| §4「配对窗口 SE」 | `results.json::paired_delta_s1`（窗口级配对 SE + 两半样本；需 ≥2 条 s1 臂） |
| 代价面表（§3 末） | `results.json::cost_face_2_2_4`（不随训练变化） |
| 判词 | 判据线不变（0.014 带；两 seed 同号且都越带 ⇒ 可排序）。若 s1 的 `dense` 也落带内 ⇒ 2.2.3 结论/`status` 不变、只是 n=2；若 s1 的 `parallel/dual` 也同号越带 ⇒ 2.2.4 升 n=2（结论不变）；若反号/落带内 ⇒ 按 `README.md::§8` 如实标不可排序 |

**核对清单**：①`config.json::steps == 12000`（冒烟会写回 20；全跑会写回 12000）；
②`results.json::completed_arms == 四臂`；③每臂 `train_tokens == 98,304,000`（否则是冒烟残臂，
当 missing 处理）；④重算后同步 `analysis.md` §2/§4/§9 与两份 `project.yaml` 的 `conclusion`
（`gaps` 里对应条目可删）。
