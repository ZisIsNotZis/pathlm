# 2655 · 逐层容量 schedule（2.3.1）· 判读

状态：**三臂全部训完并落盘**（13M / 12k / s0）。判据线 P1–P6 为 `README.md` 预登记原文、未改。
主判据读数：`front-big` **可排序更差（边界）**、`growing` **不可排序（H0）** ⇒ 预登记否证条件成立：
**H1（非均匀容量可排序更优）被本地证据否证**；逐层容量变化在本尺度不是独立有增益的维度。

> 臂完整性 as of `2026-10-14T22:53:04+08:00`（`results.json` 的 `generated_at`；三臂
> `results_<arm>_s0.json` 与 `runs/<arm>-s0-12k/model.pt` 均已落盘）。

## 人话摘要

普通 decoder-only transformer 每层 FFN 一样宽（本例 `d_model=256`、`mlp_mult=4` ⇒ 每层隐藏宽 1024）。
本实验问：在**总参数、每 token 算力都逐位不变**的前提下，把 FFN 隐藏宽沿深度重新分配，
能不能把质量（clean bpc，每字符比特数，越低越好）拉好。为此先把「怎么比较」钉死——三臂的
`Σ_层 隐藏宽 = 8192`、FFN 参数、每 token FFN 乘加数**逐位相等**（实测三臂都是总参数 6 962 688 /
FFN 参数 4 204 544 / 每 token FFN 4 194 304 MACs），所以任何差异只能来自「分配方式」自身。

结果：

- **前大后小**（浅层宽 1792→深层窄 512）bpc **1.5431**，比 uniform 的 **1.5291 更差 0.0140**——
  取值正好压在噪声带边界上（见 B 表）；**逐窗 5/5 都比 uniform 差**，方向一致。
- **逐渐放大**（front-big 的逐位镜像，512→1792）bpc **1.5309**，比 uniform 差 0.0018，
  远在噪声带内 ⇒ 与 uniform **不可排序**。

也就是说：**两种非均匀分配都没有比「每层一样宽」更好**；一种略差、一种分不出。
「把有限容量按深度更聪明地分配能免费换来质量」这个假设，在本尺度（8 层 / d256 / 12k 步、
仅最终深度 CE 监督）**没有被本地证据支持**。

## A 表：实测（已落盘，可独立复算）

| 臂 | 逐层隐藏宽 | Σh | 总参数 | FFN 参数 | 每 token FFN MACs | 每 token 总 MACs | 隐藏维 == schedule | 活性（逐层 ‖FFN_out‖/‖FFN_in‖，全 >0） | clean bpc |
|---|---|---|---|---|---|---|---|---|---|
| `uniform` | 1024×8 | 8192 | 6 962 688 | 4 204 544 | 4 194 304 | 7 392 768 | ✅ | ✅（0.484 – 2.014） | **1.5291** |
| `front-big` | 1792,1536,1280,1024,768,768,512,512 | 8192 | 6 962 688 | 4 204 544 | 4 194 304 | 7 392 768 | ✅ | ✅（0.375 – 2.410） | **1.5431** |
| `growing` | 512,512,768,768,1024,1280,1536,1792 | 8192 | 6 962 688 | 4 204 544 | 4 194 304 | 7 392 768 | ✅ | ✅（0.442 – 1.875） | **1.5309** |

- 每 token 总 MACs = attention 3 145 728 + FFN 4 194 304 + tied 读出 52 736 = 7 392 768
  （解析，**不含任何时间量**）；FFN 占总 MACs 的 0.567352，三臂逐位相同。
- 三臂逐层隐藏维实测**逐位等于** schedule（`params.hidden_matches_schedule = true`）。
- 逐深度曲线两路径一致：`clean_bpc`（窗口法）== `clean_bpc_from_depth_curve`（末深度），
  三臂 `clean_bpc_two_paths_agree = true`。
- 末 25% 窗 loss 斜率：`uniform` +0.00219、`front-big` +0.00263、`growing` +0.00314 /1k 步
  （均仍非零 ⇒ 结论只作 matched-step **快照**口径，不外推到收敛）。

## B 表：主判据（P1–P3）

符号约定（`SCHEMA.md` §17.3）：`Δ = bpc_uniform − bpc_该臂`，**正 = 该臂更好**；噪声带 0.014（限配对 Δ、≥5 窗口）。

| 臂 | clean bpc（全精度） | Δ vs uniform（4dp 报数 / 全精度） | 逐窗配对 Δ 均值 | 窗间 SE | t = 均值/SE | 判词（0.014 带） |
|---|---|---|---|---|---|---|
| `front-big` | 1.5431000 | −0.0140 / **−0.0140016** | −0.014002 | 0.003062 | −4.57 | **可排序更差（边界）** |
| `growing` | 1.5309272 | −0.0018 / −0.0018288 | −0.001829 | 0.001794 | −1.02 | **不可排序（H0）** |

- `uniform` clean bpc 全精度 = 1.5290984（4dp 报数 1.5291）。
- **front-big 边界判读（全精度）**：全精度配对窗均 Δ = **−0.0140016**，`|Δ|` 名义超带
  **1.6×10⁻⁶**（4dp 报数恰为 −0.0140，即正好落在带界）。按预登记规则 `Δ ≤ −0.014 ⇒ 可分更差`
  ⇒ 判**可排序更差**；5/5 窗口的配对 Δ 全为负（见 P3），方向无歧义。**注意**：即便按更保守的
  `|Δ| ≤ 0.014 ⇒ 不可排序` 读法，读数也只是「不可排序」而已，**两种读法都得不到正面增益**——
  H1（可排序更优）在任一读法下都已被否定。
- **growing**：Δ = −0.001829 远在带内，且窗间符号不定 ⇒ 与 uniform **不可排序（H0）**。
- **P3 逐窗配对 Δ（`uniform − 该臂`，同一 generator seed 的同一批序、5 窗口 × 4 批）**：
  - `front-big`：[−0.011519, −0.015335, −0.021190, −0.018388, −0.003576]
  - `growing`：[+0.000331, +0.001921, −0.005843, −0.006500, +0.000947]
  - 窗间 SE 是**同 checkpoint 内的抽样噪声**，与 0.014 的**跨 run 训练漂移带**是两回事，分开报。

## 三道门（P4–P6）

### P4 等 FFN 预算门（逐位相等 + 解析式与实测互核）

| 量 | `uniform` | `front-big` | `growing` | 是否逐位相等 |
|---|---|---|---|---|
| `Σ_l h_l` | 8192 | 8192 | 8192 | ✅ |
| FFN 参数（解析 `Σ_l (2·h_l·d + h_l + d)`） | 4 204 544 | 4 204 544 | 4 204 544 | ✅ |
| FFN 参数（实测 `params_ffn`） | 4 204 544 | 4 204 544 | 4 204 544 | ✅（解析 == 实测） |
| 每 token FFN MACs（解析 `Σ_l 2·h_l·d`） | 4 194 304 | 4 194 304 | 4 194 304 | ✅ |
| 总参数 | 6 962 688 | 6 962 688 | 6 962 688 | ✅ |

`results.json:P4_equal_ffn_budget_check.pass = true`（三臂 `(ffn_macs, params_ffn, sum_hidden)` 三元组同一）。
attention / embedding / LN / 读出参数逐位相同（非 FFN 参数 2 758 144）。解析值由 `smoke.py` 与
`run.py analytic_cost()` 两条独立路径各算一遍，结果相同。

### P5 引擎等值门

- **模型层**：显式 `--mlp-hidden-sched 1024,…,1024` 与引擎缺省（`--mlp-mult 4`）的
  `state_dict` 与前向输出**逐位一致**；同 `smoke.py` 带必红探针（改一位 schedule ⇒ 必不等）。
  `smoke_report.json:A_engine_equivalence.pass = true`。
- **训练层旁证**：`uniform` 臂 clean bpc 1.5291 与同配方历史锚之差 < 0.005
  （2613 sequential-s0 1.5254，差 +0.0037；2641 U4-s0 1.5314，差 −0.0023）⇒ 同引擎同配方。
  `results.json:P5_engine_equivalence_gate.pass = true`。

### P6 活性（`SCHEMA.md` §10）

- 逐层隐藏维 == schedule：三臂全 true。
- 逐层 FFN 相对幅度（真实前向 `‖mlp_out‖/‖mlp_in‖`）全 > 0：`uniform` 0.484–2.014、
  `front-big` 0.375–2.410、`growing` 0.442–1.875 ⇒ 窄层不是死路径、宽层不是摆设。

## 对账（`SCHEMA.md` §12 三对账）

- **算术对账**：FFN 参数解析式 `Σ_l (2·h_l·d + h_l + d)` 与实测 `params_ffn` 逐位相等
  （4 204 544 == 4 204 544）；FFN MACs 解析 `Σ_l 2·h_l·d` == 4 194 304。三臂的解析值与实测值
  逐臂核对一致；`Σh=8192` ⇒ FFN 参数与 FFN MACs 在臂间**逐位相等**（P4）。
- **任务对账**：三臂同一任务定义（enwik8 前 90M train / 后 10M eval，vocab=206）、同一输入分布、
  同一位置集（同 generator seed 的同一批序，同 20 批 × 16 × 256）——三臂测的是同一件事。
- **口径对账**：全深度、无退出（`exit-mode uniform`）、teacher-forcing、序列级 clean bpc；
  三臂同为 matched-step 快照（12k 步）。**无任何墙钟量进判据或 `cost_model`**（训练的墙钟 /
  显存只作过程量留在 `shared_train_results`）。

## 对照三要素（`SCHEMA.md` §13）

对照臂 = 同实验 `uniform` 臂（= stock uniform 全深度、引擎缺省形态）：

- **输入是什么分布**：`data/enwik8_full.npz` 后 10M eval 段，206 词表，与主臂同一切分；
- **要预测什么**：下一 token（teacher-forcing，序列级 clean bpc，最终深度输出）；
- **部署中对应什么**：普通自回归语言模型的下一 token 预测；`uniform` 是「每层容量相同」的
  stock 默认，正是本机制要挑战的 baseline。

三臂与对照在**同一任务定义、同一输入分布、同一位置集**下打分；无「假上下文 + 真目标」式对照。

## 跨轴旁证（**标注：非本项自有证据**）

轴 3 `axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/`（同引擎同配方、
等 FFN 预算下 PYR/APYR vs uniform，n=2）得 H0「不可排序」：

- 本项 `uniform`（1.5291）与 2641 的 `U4-s0`（1.5314）锚差 **−0.0023 < 0.005**（引擎等值门）
  ⇒ 两实验同引擎同配方可比。
- 该旁证方向与本项一致（同尺度下逐层/逐维 FFN 分配均无净增量），但**不作为本项终局判据**。
- 「跨轴干预证据能否当本项终局判据 / 2.3.1 与 3.1.4 是否同题」属**框架性取舍，已上报 Main**，
  本目录不自行合并或改名。

## 字面 / 实质两读法（`docs/research-schema.md` §5 C2）

- **字面读法**（milestone 预登记文字）：验收要求「各臂总参数逐位相等」+「非均匀相对 uniform
  有/无越噪声带差异，二选一明确」。两条均满足——等预算门逐位相等（P4 pass）；差异二分明确
  （`front-big` 可分更差[边界]、`growing` 不可排序）⇒ **字面成立**。
- **实质读法**（C2：不被同算力的朴素方案支配 / 相对同算力最强基线超出量 ≤ 0）：非均匀臂均**无正面增益**
  （`front-big` 更差、`growing` 不可分），未提供可兑现的净增量 ⇒ **实质否证**
  （逐层容量变化在本尺度不是独立有增益的维度）。

## 缺口的处理

- **hourglass 臂（中段大）**：本 WP 只做前大后小 / 逐渐放大两种非均匀形态，hourglass 未列（见 `gaps`）。
- **不动的维度**：本实现只重分配 FFN 隐藏宽；`d_model` 逐层不变、注意力头数不变（后者是
  `2.3.2` 的轴，未测）。
- **seed 数 n=1（seed=0）** ⇒ 判词只到「可排序 / 不可排序」二分，不做 trend 外推；seed=1 未跑。
- **外推**：只在本尺度（8 层 / d256 / 12k / 仅最终深度 CE）内判读，不外推。

## 独立复核

**独立复核未做**（fresh 复核者排在 writer 停止之后，`SCHEMA.md` §14/§14b；本轮窗口内未启动）。

## 回归与可复算的校验命令

| 命令 | 结果 |
|---|---|
| `python3 smoke.py` | 四道门（等预算 / 引擎等值 / 活性 / CLI 接线）全过，报告落 `smoke_report.json` |
| `python3 -m pytest …/test_2655_layer_capacity.py -q` | 8 passed |
| `python3 run.py --aggregate` | 汇总 P1–P6 落 `results.json`（三臂，`P4.pass=true`、`P5.pass=true`） |
| `python3 tools/summary.py --check` | OK（46 份 `project.yaml`） |
| `python3 -m pytest tests/ -q` | 865 passed / 1 failed —— 失败为 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`，涉 `axes/6-predict/**` 的 4 个 `mutate.py`（`2674-mdiffusion-13m` / `2671-padfill-13m` / `2660-latentreg-vs-disc-13m` / `2670-freerun-bpc-13m`）（**非本 WP 引入、与本 WP 无关**，作为预存失败分开登记） |
