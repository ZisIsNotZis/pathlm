# ticket 64 · 轴 3 宽度轴 lead（Phase A）

> 性质：**过程工单**（轴 3 的编排 + 执行 + 复核轨迹）。事实文档 = `axes/3-width/**/project.yaml`
> 与 `axes/3-width/README.md`、分组 README（`3.1-active-width/`、`3.2-readout/`）；本文件只记
> 「我们一步一步怎么走的」。用词以 `docs/structure.md` §12 为准。
> 记录纪律：显式时间戳 + actor + 命令与结果 + 证据锚点；框架性取舍上报 Main（§4）。

## 1 范围与契约（轴 3 lead）

| 项 | 值 |
|---|---|
| 轴 | 3 宽度轴（`axes/3-width/`） |
| owner | axis3-lead（轴 3 宽度轴 lead） |
| writer | 本轴单一写入者 = axis3-lead（`project.yaml` / 轴与分组 README / 本工单）；WP 执行 writer = 本 WP 的单一写入者（本轮 = axis3-lead，就近实现在实验目录内） |
| reviewer | axis3-reviewer（fresh 独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后），复核件落 `.scratch/64-axis3-width/REVIEW-*.md` |
| integrator | tree-integrator（研究树 integrator；本轮本轴的合并与跨轴一致性核对由 axis3-lead 代行） |
| 规范输入 | `docs/structure.md` §12/§13（节点/用词/粒度）、`docs/research-schema.md`（字段契约）、`SCHEMA.md` §7/§8/§14/§14b/§15/§16/§20、`docs/methodology.md` §2/§3/§5 |
| 本轴节点 | 分组 `3.1` / `3.2`；研究项 `3.1.1 3.1.2 3.1.3 3.1.4 3.1.5 3.2.1 3.2.2 3.2.3`（共 8） |
| 前情输入 | `.scratch/55-review/W-PROBLEMS.md`（W-01…W-18）、`.scratch/56-restructure/MAPPING.md`（§1 W1→3.1.1、W4→3.2.1；§2 其余无旧格）、`.scratch/57-program/TICKET.md`（M2/M3） |
| 边界 | 只动 `axes/3-width/**` + `.scratch/64-axis3-width/**`；不碰其它轴、`docs/**`、共享 `pathlm/**`/`train_*.py`/`tools/**`（需改则上报） |
| 并发/锁 | 并发子 Agent ≤3；GPU 训练按 `SCHEMA.md` §20 互斥锁**按臂**串行；纯 CPU/文档不取锁 |
| 预算（Phase A） | 硬时间盒 ≤2h：①8 项 `role_map` 实名 + 逐项「想清楚」复核（≤30 min）；②**只执行 1 个最高价值 WP**（≤60 min GPU 盒）；③fresh 独立复核（累计 ≤2 轮）；④兄弟字段一致性扫描 + `summary.py --check` 全绿；⑤更新 `status/conclusion/experiments/change_log` |

## 2 轴级共同口径（沿用已冻结项，不自选）

- 目标函数：**用更少的等效算力拿到序列级类似质量**；代价口径**只**用等效理论量（FLOPs / MACs /
  layer-passes / tokens-per-forward，考虑 pipelining），**墙钟/吞吐/日期不进判据**
  （`docs/research-schema.md` §3）。
- 对照 = 同尺寸 **stock 全宽全通道**（13M dense 锚：`.scratch/23-depth-ar/evidence/23d-dense-B-12k`
  bpc = 1.5287 @12k；2613 全新重训 baseline-s0 = 1.5257 / s1 = 1.5298）归一为 100%。
- 判据 = 序列级 clean bpc（该不跌 / 可排序，须越 **0.014** 噪声带）+ 每 token 等效算力 / 参数量（该跌）
  + 必要活性/损伤对照列。**前沿口径**：横轴 = 每 token 等效算力或参数量，纵轴 = clean bpc，以 stock 为 100%。
- 「0.014 带」的来源 = `docs/methodology.md` §2 的 13M 跨 run **bpc** 漂移界；本轴比较的正是 bpc
  同量纲量 ⇒ **不构成跨量纲借用**（对比轴 6 的 α/τ 情形）；另报同配置跨 seed 实测散布作参照。
- 分格归属（防同一读数两处记账）：`3.1.1` = **d_model / 头维**整体缩小；`3.1.4` = **FFN 隐藏比 r**
  （d_model 固定）及其逐层变化；`3.1.2` = **整体 d_model** 的逐层 schedule；`3.1.3` = 同层内**位置**相关宽度；
  `3.1.5` = 通道粒度稀疏激活（对专家轴 `4` 的 MoE 划界）。

## 3 逐项「想清楚」复核（8/8，actor = axis3-lead，2026-10-13T18:0x+08:00）

复核维度（对齐 `docs/research-schema.md` §2 与 `SCHEMA.md` §23 七问）：①痛点 ②业界 SOTA 与出处
③相对 SOTA 的增量 ④方法（单因素）⑤实现 ⑥**客观判据（该涨/该跌/必须不变）**⑦做不出来怎么归因
（设计/结构/innovation）⑧等效理论代价 ⑨单一 milestone ⑩WP 与推进顺序。

| 研究项 | 复核结论 | 处置（本轮已落盘） |
|---|---|---|
| **3.1.1 uniform 减宽 / width dropout** | 十项齐全；55-review W-06/W-07/W-14 的纠正已在 `conclusion_review`/`gap` 写死（旧 W1 唯一实验测的是**读出切片**、属 `3.2.1` 域，不构成本项结论；「减宽=减容量」是照抄业界共识）。**两处需补**：(a) 减宽是**剪枝类**（参数与算力同时降）⇒ 按 `SCHEMA.md` §15 例外，对照应是**同 FLOPs 的稠密前沿点**而非「等参数量」；(b) 与 `3.1.4` 会重复表达「uniform 减宽」（d_model 缩 vs FFN r 缩） | 已改 `method`（写死 §15 例外的对照位点）与 `gap`（边界：d_model 归本项、FFN r 归 `3.1.4`）；`gaps` 追加「同 FLOPs 稠密前沿点尚未实测，可借 `3.1.4` 的 2641 当量纲参照」 |
| **3.1.2 逐层宽度变化（金字塔 / 沙漏）** | 十项齐全；判据钉在「总参数逐位对齐」。**两处需补**：(a) 与 `3.1.4` 的「逐层变化」半边必须划界（本项 = 逐层 d_model；FFN r 归 `3.1.4`）；(b) **实现前置**：引擎 `mlp_mult` 是全局标量、`Block` 的 `d_model` 逐层不可配 ⇒ 本项在引擎收编前**不可跑** | 已改 `gap`（边界）+ 追加 `gaps`（实现前置，指针到本工单 §5 的引擎改动申请） |
| 3.1.3 同层内位置相关宽度 | 十项齐全；判据把「每 token 平均激活通道数/等效算力」该跌与「clean bpc 该不跌」分开，并把**长程召回**列为必须报的损伤列（防窄段丢信息被 bpc 掩盖）；与时间轴 `1.1`/`1.2` 正交已声明。**位置的定义（绝对位置 / 近远端 / 内容分区）未定**（已记 gap），属最小判决里要先定的默认，不阻塞 | 仅补 role_map 实名 |
| **3.1.4 前馈隐藏比** | 十项齐全，且**本轴核心问题的最佳落点**（FFN 是每 token 参数与算力主项、d_model 固定 ⇒ 单因素干净）。**一处实质缺口**：`method`/`objective_criteria`/`validation` 只说「层间变化 vs 常数比」，**没把「等 FFN 预算」位点写死** ⇒ 会拿 Σr 不同的两臂比「容量 + 分配」两件事 | 已改 `method`/`validation`/`objective_criteria`/`falsification`（写死 Σ_l r_l 对齐 = FFN 参数与每 token FFN 算力逐位相等）；`experiments` 登记 2641；WP1 置 `进行中` 并在本轮执行 |
| 3.1.5 通道极化 / 稀疏激活 | 十项齐全；「保容量减计算」的通道粒度版，与专家轴 `4` 的 MoE（专家粒度 token 级路由）划界已写进 `gap`；活性（通道使用率分布）与「实装是否真省 FLOPs」两列必须报。**kernel 真省 FLOPs 未验**（已记 gap） | 仅补 role_map 实名 |
| 3.2.1 读出的重参数化 | `status: 已完成`（旧 W4 迁入 + 55-review W-10/W-11 口径修正）；`failure_analysis` 已按「创新本身失败（主因）+ 结构问题（次因）」分列并给证据（等参数纯线性对照 Δ −0.0888 vs gated −0.0812 不可排序；dense 上 Δ −0.0055 不可迁移；12k→24k 缩 43%）。判据已从「bpc 降了」改为「等参数对照归因 + 可迁移性」= 55-review W-11 的处置。**遗留**：WP4（更长训练/更大尺寸）未做、单 seed 单配置 | 仅补 role_map 实名（WP4 留 `未开始`，不并进本轮时间盒） |
| 3.2.2 逐出口读出 | 十项齐全；把「末层质量不劣于 stock」与「独立表 vocab×d 参数单列 + 可否 scale」钉死；与 `2.1.1`（跑不跑这层）划界清楚 | 仅补 role_map 实名 |
| 3.2.3 tied vs untied | 十项齐全。**一处事实错误**：写「本仓 vocab=205 / +52480」，实际模型词表 = **206**（205 个不同字节 + 1 个 `[mask]` 槽，`data/enwik8_full.npz` 的 `vocab_size=206`）⇒ 参数量 +52736，且同一错误在分组 README `3.2-readout` 的「共同口径」里重了一遍 | 已改 `innovation`/`argument`/`gaps`/`objective_criteria` + `3.2-readout/README.md`（206 / 52736）；边界（本项 vocab×d vs `3.2.1` 的 2·d²）保持 |

- 全 8 项 `cost_model` 复核 = 纯等效理论量、无墙钟/日期/吞吐 ✓。
- 全 8 项 `milestone` = 单数 map、含 `output` + `acceptance` ✓；`work_packages[].review` 非空 ✓。
- 脚本化改文本：`.scratch/64-axis3-width/edits_phase_a.py`（命中 16/26，未命中 10 处已定位）+
  `edits_phase_a2.py`（命中 10/10）；两个脚本都**未命中即非零退出**并列清单（`AGENTS.md` §8）。
- `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合 docs/research-schema.md`（改后复跑）。

## 4 框架性取舍与上报 Main（不自选）

| # | 取舍 | 我的处置（暂定） | 待 Main 裁决 |
|---|---|---|---|
| **F1** | **等预算位点选哪个**：减宽 / 减 FFN 宽是**剪枝类**（参数与算力同时降）⇒ `SCHEMA.md` §15 的「等参数对照」在字面上不可得（等参数就没有减宽了） | 取**等 FFN 预算位点**（同 Σ_l r_l ⇒ FFN 参数与每 token FFN 算力**逐位相等**）+ 以 stock r=4 为 100% 代价锚；并在 `3.1.1`/`3.1.4` 的 `method` 里显式引用 §15 的剪枝例外 | 是否确认「等 FFN 预算 / 等算力位点」是剪枝类实验的标准对照口径；已上报 |
| **F2** | **引擎缺「逐层 FFN 宽」杠杆**（`mlp_mult` 全局标量）——是上报等授权，还是就近实现？ | **不擅改共享 `pathlm/**`**：把逐层 schedule 实现在**实验目录内**（构造后对 FFN 做**前缀切片**，等值性门 T1 证明与引擎原生窄 FFN 逐位同前向）；同时把「收编该开关」的最小 patch 描述记在本工单 §5 并上报 | 是否授权把 `mlp_mult` 收编为逐层 schedule（谁改/谁复核/何时），以及是否把本实验的切片实现改判为「应等引擎」；已上报 |
| **F3** | **n≥2 与 60 min GPU 盒的冲突**：4 臂 × 2 seed × 12k 步 ≈ 7–8 min/臂 ⇒ 8 臂 ≈ 60 min | 优先保证**核心对照**的 n=2（U2 / PYR）与全臂的 seed0；若 GPU 竞争吃掉预算，按 §6 的「先砍 U4-s1、再砍 APYR-s1」递减，并在结论里如实记 n | 是否接受「部分臂 n=1」；已上报 |

## 5 引擎改动申请（**本轴不执行**；`pathlm/**` 不在本轴边界内）

- **需要什么**：`DepthARConfig` / `ModelConfig` 的 `mlp_mult` 由全局标量扩展为
  **逐层 schedule**（如 `mlp_mult_sched: list[int] | None`）。
- **为什么**：宽度轴的 `3.1.2`（逐层 d_model）与 `3.1.4`（逐层 FFN 隐藏比）都以「逐层宽度」为核心
  干预；引擎没有该杠杆时，各实验只能在**实验目录内**用「构造后切片」自造（本轮 `3.1.4` 的 2641 就是
  这么做的），代价是每个实验都要自证实现等值、且不同实验的切片实现可能不一致。
- **最小 patch 描述**：`pathlm/depth_ar.py::DepthARModel.__init__` 里构造 `self.blocks` 处，
  把 `ModelConfig(mlp_mult=cfg.mlp_mult, ...)` 换成逐层传入（`Block.__init__` 增可选
  `mlp_mult: int | None = None`，缺省回落 `cfg.mlp_mult`）；`train_depth_ar.py` 加
  `--mlp-mult-sched "4,3,2,2,2,1,1,1"`（缺省 None = 逐位还原现状）；配套单测 2 条
  （① 缺省 None 与现状逐位同前向；② schedule 真的到达每层 `mlp[0].weight.shape[0]`）。
  **改动量约 30–50 行，只动 `pathlm/depth_ar.py` + `train_depth_ar.py`**；风险：`Block` 有 4 个
  变体（`ParallelBlock`/`GQA`/`Composable`/`LinearAttn`）都各自内联 `mlp_mult`，需一并接线，否则
  会得到「开关开着、部分臂没接」的死干预（`SCHEMA.md` §10）。

## 6 WP 排序（按价值）与预算（actor = axis3-lead，2026-10-13T18:0x+08:00）

排序原则（`AGENTS.md` §5）：**它是否可能改变我们对「某种算力路径值不值」的判断**。

| 序 | WP | 所在项 | 判决力 | 成本（等效） | 本 Phase |
|---|---|---|---|---|---|
| 1 | **WP1：FFN 隐藏比前沿 + 等 FFN 预算下的逐层非均匀宽度**（U4/U2/PYR/APYR × seed 0/1，12k） | `3.1.4` | **最高**：直接回答轴的核心问题（非均匀宽度在等预算下有没有增益）+ 给出「FFN 预算 → 质量」的前沿斜率 | GPU 7 臂 × ~7.5 min ≈ 55 min（**本轮执行**） | 执行 |
| 2 | WP1：uniform 减宽 + width dropout 各档单调性（含同 FLOPs 稠密前沿点） | `3.1.1` | 高（轴的基础读数：减宽值不值 + 弹性宽度） | GPU ~4 臂 × 12k | 下一轮 |
| 3 | WP1：逐层 **d_model** schedule（金字塔/沙漏/倒金字塔） | `3.1.2` | 高（与 #1 互补：整体宽度 vs FFN 宽度） | GPU 4 臂 × 12k；**需引擎杠杆（§5）** | 下一轮（被 §5 阻塞） |
| 4 | WP1：通道稀疏激活（含活性列） | `3.1.5` | 中高（保容量减计算的通道版） | GPU 4 臂 × 12k + kernel 真省 FLOPs 需验 | 下一轮 |
| 5 | WP1：同层内位置相关宽度（含长程召回列） | `3.1.3` | 中（与时间轴正交的宽度读法） | GPU 4 臂 × 12k；需位置分段的掩码实现 | 后置 |
| 6 | WP1：逐出口读出（共享 vs 每出口） | `3.2.2` | 中（早退可用性） | GPU 4 臂 × 12k | 后置 |
| 7 | WP1：tied vs untied | `3.2.3` | 中低（小词表上大概率中性） | GPU 3 臂 × 12k | 后置 |
| 8 | WP4：更长训练 / 更大尺寸确认 | `3.2.1` | 低（边界钉实） | 高（长训练） | 后置 |

- 本轮只执行 #1；预算递减规则见 `2641/README.md` §6（先砍 U4-s1 → 再砍 APYR-s1，不砍步数与判据线）。

## 7 执行记录（显式时间戳 + actor + 命令/结果）

### 7.1 Phase A-1（2026-10-13T17:53–18:0x，actor = axis3-lead）

1. `2026-10-13T17:53:35+08:00` 读权威输入：`docs/structure.md`（§3/§12/§13）、`docs/research-schema.md`、
   8 份 `axes/3-width/**/project.yaml` + 轴与两个分组 README、`.scratch/55-review/W-PROBLEMS.md`
   （W-01…W-18）、`.scratch/56-restructure/MAPPING.md`、`.scratch/57-program/TICKET.md`、
   `SCHEMA.md`（§7/§8/§14/§14b/§15/§16/§20）、`docs/methodology.md`（§2/§3/§5）、
   `.scratch/62-axis1-time/TICKET.md` 与 `.scratch/63-axis6-predict/TICKET.md`（同层格式参照）。
2. `2026-10-13T17:56:0x+08:00` `git status --short` = 干净；`HEAD=cfee001`、分支 `main`；
   `nvidia-smi` = used 5626 MiB / free 18483 MiB / util 25%；`/tmp/pathlm_gpu.lock` **不存在**
   （其它锁属别的线）⇒ 本轴 GPU 工作按 §20 取 `pathlm_gpu.lock` 串行。
3. `2026-10-13T17:56:26–17:56:41+08:00` stock 路径 300 步冒烟（`smoke_stock.sh`，整程持锁 <10 min，
   §20.1.1 允许）⇒ rc=0，`wall=15s` ⇒ ≈ 33–50 ms/step（12k 步 ≈ 7–10 min/臂），峰值显存落在礼貌线内；
   步时**不进判据**（只用于预算排程）。
4. `2026-10-13T17:5x` 写 `2641/README.md`（**先写契约再跑**）+ `2641/run.py` + `test_2641_ffn_ratio.py`
   + `train_arms.sh`；`pytest test_2641_ffn_ratio.py -q` ⇒ **13 passed**（含 T1 等值性逐位门与 4 条变异）。
5. `2026-10-13T18:00:32+08:00` 60 步端到端探针（U2-s0）⇒ rc=0，`hidden=[512]*8`、`params_total=4861440`
   ⇒ 切片真的到了模型；探针产物移入 `2641/smoke/probe-60steps-U2-s0/`，其 `results_U2_s0.json` 删除
   （防被聚合器误读）。
6. `2026-10-13T18:00:37+08:00` 启动 `train_arms.sh`（`ARMS="U4:0 U2:0 PYR:0 APYR:0 U2:1 PYR:1 APYR:1"`,
   `STEPS=12000`，后台 + 逐臂取/放锁，臂间 61 s）；日志 `2641/train_arms_driver.log`。
7. `2026-10-13T18:0x` 逐项复核 8/8 落盘（§3）；脚本化改文本 26 处全命中（16+10，两个脚本）；
   `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合 docs/research-schema.md`。

### 7.2 Phase A-2（WP1 执行：`3.1.4` / 2641）

| 项 | 值 |
|---|---|
| WP | `3.1.4` / WP1「前馈隐藏比前沿 + 等 FFN 预算下的逐层非均匀宽度」 |
| 实验 | `axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/` |
| 预登记 | `2641/README.md`（先写契约再跑；臂定义 / P1–P6 / 否证条件 / 禁墙钟） |
| writer | axis3-lead（本 WP 的单一写入者；实现落在实验目录内，不碰共享 `pathlm/**`） |
| 臂 | `U4` r=4（100% 代价锚）/ `U2` r=2 等宽 / `PYR` [4,3,2,2,2,1,1,1] / `APYR` 镜像；**U2/PYR/APYR 同 Σr=16** |
| seed | 0 / 1（GPU 竞争时按 §6 递减） |
| GPU | §20 锁按臂取/放，臂间 ≥60 s；驱动器日志 = `2641/train_arms_driver.log` |
| 独立复核 | fresh 复核者，报告落 `.scratch/64-axis3-width/REVIEW-*.md` |

#### 7.2.1 执行中的两次工程事件（显式登记，均**不影响**训练路径与判据）

1. `2026-10-13T18:09:33+08:00` · axis3-lead · **修正一个派生审计字段的单位**：
   `run.py::evaluate()` 里 `params.hidden_matches_schedule = (hidden_by_layer == SCHEDULES[arm])`
   把「隐藏维（1024）」与「r（4）」直接比较 ⇒ 恒 `False`。改为
   `hidden_by_layer == [r * D_MODEL for r in SCHEDULES[arm]]`。
   - 该字段只在**评测阶段**由已记录的原始量派生（`hidden_by_layer` 与 `schedule_r_by_layer`
     都原样落盘），**不参与训练、不参与前向、不参与 bpc**；
   - 时序：`U4-s0`（18:00:37 起）与 `U2-s0`（18:09:30 起）两臂的评测进程早于修正
     ⇒ 它们的 `results_*.json` 里该字段是错的；`PYR-s0` 起（18:17:21）用修正后版本
     （自证：`PYR-s0.hidden_matches_schedule = true`，`U4-s0/U2-s0 = false`）；
   - 处置：训练**不重跑**（修正只影响派生字段，重跑会浪费 GPU 预算）；训练后对 `U4:0 U2:0`
     两臂做 **`--no-train` 重评**（`reeval_check.sh`）⇒ 8 个结果文件统一由最终版代码生成，
     并顺带核验**评测确定性**（重评 bpc 与首评逐位一致）。修订前版本可由
     `git show 49f697f:<实验目录>/run.py` 复得（`9e68bfa2…`）。
   - **口径诚实记账**：`code_hash_run_py` 是进程**结束时**从磁盘读到的哈希 ⇒ 对上述两臂，
     它记的是修正后的哈希而非它们实际加载的代码；实际加载版本 = `49f697f` 版（差该 1 个派生字段）。
2. `2026-10-13T18:0x` · axis3-lead · **预登记时序证据**（回应「判据线是否被事后改过」）：
   `2641/README.md` mtime = `17:58:24`、`train_arms.sh` = `17:59:41`、
   `test_2641_ffn_ratio.py` = `18:00:10`，**全部早于** 起跑 `18:00:37`；三者的内容哈希与
   `49f697f` 里的版本逐位相同（`sha256sum` 对照）。提交 `49f697f` 的 `%cI` = `18:02:28`
   **晚于**起跑（我在起跑后才提交）⇒ 「先写契约再跑」的依据是**文件 mtime + 内容哈希**，
   不是提交时间；此处如实登记，不用提交时间作证据。

#### 7.2.2 逐臂读数（读数时间见各臂 `runs/*/results.json` 的完成时刻）

> 口径提醒：本节表内 `FFN 参数` 列的 2,099,200 是 `18:2x` 的**中途快照**（= 2,097,152 + 2,048 的 `lin2` 偏置），
> 当时的审计字段还在旧口径；**最终读数见 §7.3 表**（2,103,296 = MACs 2,097,152 + lin1 偏置 4,096 + lin2 偏置 2,048）。

| 臂 | clean bpc | Σr | FFN 参数 | 总参数 | 逐层隐藏维（实测） |
|---|---|---|---|---|---|
| U4-s0 | 1.5314 | 32 | 4,204,544 | 6,962,688 | [1024]×8 |
| U2-s0 | 1.5709 | 16 | 2,099,200 | 4,861,440 | [512]×8 |
| PYR-s0 | 1.5730 | 16 | 2,099,200 | 4,861,440 | [1024,768,512,512,512,256,256,256] |
| APYR-s0 | 进行中（`18:25:39` 起） | 16 | 2,099,200 | 4,861,440 | [256,256,256,512,512,512,768,1024] |

- 关键初读（seed 0）：**U2（等宽、Σr=16）vs PYR（金字塔、同 Σr）Δ_up = bpc_U2 − bpc_PYR = 1.5709 − 1.5730 = −0.0021**
  ⇒ |Δ| = 0.0021 ≪ 0.014 噪声带 ⇒ **H0 方向**（逐层 FFN 分配在等 FFN 预算下不可排序）。
  同时 **U4→U2 的代价 = +0.0395 bpc**（FFN 减半 = 每 token 总 MACs −28.4%）⇒ FFN **总量**是真旋钮。
- **初始化控制已独立核验**（CPU，`axis3-lead`，`18:2x`）：seed 0/1 下 `U2/PYR/APYR` 的
  **trunk 权重与 U4 逐位相同**、且三个臂的 FFN 权重是 U4 初始化的**逐位前缀**
  （⇒ 臂间唯一差别 = 每层保留多少中间通道，不是「两次不同初始化」）。

#### 7.3 全部 7 臂读数与判词（`2026-10-13T19:00:26+08:00` 最后一条臂收尾；完整判读见 `2641/analysis.md`）

| 臂 | Σr | clean bpc | 每 token MACs | FFN MACs | FFN 参数 | 总参数 | loss 斜率/1k |
|---|---|---|---|---|---|---|---|
| U4-s0 | 32 | **1.5314** | 7,392,768 | 4,194,304 | 4,204,544 | 6,962,688 | 0.00401 |
| U2-s0 | 16 | 1.5709 | 5,295,616 | 2,097,152 | 2,103,296 | 4,861,440 | 0.00418 |
| U2-s1 | 16 | 1.5611 | 5,295,616 | 2,097,152 | 2,103,296 | 4,861,440 | 0.00332 |
| PYR-s0 | 16 | 1.5730 | 5,295,616 | 2,097,152 | 2,103,296 | 4,861,440 | 0.00270 |
| PYR-s1 | 16 | 1.5750 | 5,295,616 | 2,097,152 | 2,103,296 | 4,861,440 | 0.00354 |
| APYR-s0 | 16 | 1.5644 | 5,295,616 | 2,097,152 | 2,103,296 | 4,861,440 | 0.00580 |
| APYR-s1 | 16 | 1.5601 | 5,295,616 | 2,097,152 | 2,103,296 | 4,861,440 | 0.00178 |

- **P1（PYR vs 等宽）**：Δ_up = **−0.0021**（s0）/ **−0.0139**（s1）⇒ 两 seed 都 |Δ| < 0.014
  ⇒ 判 **不可排序（H0）**；如实加注：两 seed **同号（负）**、s1 压在带边（0.0139）、
  **没有任何 seed 显示 PYR 更优**。
- **P2（APYR vs 等宽）**：Δ_down = **+0.0065**（s0）/ **+0.0010**（s1）⇒ 判 **不可排序（H0）**；
  两 seed 同号（正）⇒ 弱倾向「后宽前窄略好」，但幅度小于同配置跨 seed 散布（U2 = 0.0098）。
- **P4**：两个方向都跨 seed 同号 = True（方向可复现），但幅度不够 ⇒ 按预登记只报「不可排序」。
- **P3 前沿**：U4→U2 = **+0.0395 bpc（2.8× 带）换 −28.4% 每 token 算力 / −30.2% 参数**；
  等代价三点（U2/PYR/APYR）互为平局（差异全在带内）⇒「非均匀」没把前沿往外推。
- **P5**：U4-s0 与 `23d-dense-B-12k` 差 +0.0027 ⇒ 过（四个同配置 dense 13M/12k run 的
  极差 0.0057，本臂落在既有散布内）。
- **P6**：7/7 臂 `hidden_matches_schedule=True`、FFN 参数解析式 == 实测；逐层 FFN 输出相对幅度
  全正（窄层 0.30–0.35 / 宽层 0.49–2.08）⇒ 干预活、窄层不是死路径。
- **副产品（不作判据）**：PYR 的中间深度 bpc 明显更低（第 6 层 1.9727 vs U2 2.3338、
  第 7 层 1.7492 vs 1.7887），未传导到最终深度 ⇒ 对早退（`2.1.1`）可能有用，本轮未测。
- **工程复核**：`collect_logs.py` 的 `data/enwik8_full.npz` 路径少了一级（`19:07` 修 4→5 级
  `..`，修后 7 臂归集成功、`code_frozen=True`）；`report_table.py` 复算出的表格与 `analysis.md` 一致。

## 7.4 预算递减的落地记录（契约 §6）

- 契约 §6 定的递减顺序是「先砍 U4-s1」⇒ **U4 只有 seed 0**（100% 代价锚单 seed）；
  它的跨 run 散布改用 4 个历史 dense 13M/12k run（1.5257 / 1.5287 / 1.5298 / 1.5314，极差 0.0057）。
- 实际 GPU 占用：7 臂 × 6.7–7.8 min ≈ 52 min（含锁等待），落在 60 min 盒内，因此**不需再砍**
  剩下任何臂（U2/PYR/APYR 全部有两个 seed，支撑 P4 的判据）。

（判词与复核记录在 §9 与 `2641/analysis.md` 续写。）

## 8 复核计划（`SCHEMA.md` §14/§14b）

- 排程：writer（axis3-lead）停止写入 → 派 **fresh** 复核者（独立上下文、只读）复核
  ① `project.yaml` 的 Phase A 改动与用词/边界；② 2641 的契约-实现-数字一致性（判据线是否被改过、
  对照臂可比性、等 FFN 预算是否逐位成立、活性与等值性门、证据可回溯、口径标注）；
  ③ 否证路径与归因分类。复核件由复核者**直接写进仓库** `.scratch/64-axis3-width/REVIEW-*.md`。
- 有缺陷 ⇒ 派 writer 修 ⇒ **新的** fresh 复核；**累计 ≤2 轮**；2 轮未 PASS 即如实登记 blocker 返回。
- 复核者须自跑与独立复算（不得引用 axis3-lead 自报值）：`pytest test_2641_ffn_ratio.py -q`、
  `python3 run.py --aggregate ...`、`python3 tools/summary.py --check`、`git diff --stat`（越界扫描）。

## 9 复核记录

### 第 1 轮（`e6d2db1`，fresh 独立上下文、只读）

- `2026-10-13T19:1x+08:00` · axis3-lead · 派 fresh reviewer（subagent run `e27c6dbf-25a6-44c7-919e-5f8fc780748c`），
  复核目标 revision = `e6d2db1`（writer 已停止写入），交付 = `.scratch/64-axis3-width/REVIEW-1.md`。
- 派单里要求**自跑/自算**（不得引用本工单与 `analysis.md` 的自报值）：
  `pytest test_2641_ffn_ratio.py -q`；自写脚本从 `results_<arm>_s<seed>.json` 重算 P1/P2/P4；
  等 FFN 预算逐位相等核验；臂间 `config` 逐字段对照；`build_model()` 复算初始化控制（trunk 逐位 + FFN 前缀）；
  契约未被事后改过（`README.md`/`train_arms.sh`/`test_*.py` 与 `49f697f` 的内容哈希 + mtime 早于起跑）；
  `analysis.md` 数字逐条保真；口径扫描（墙钟禁入判据）；越界扫描（`git diff --stat 49f697f..e6d2db1`）；
  `python3 tools/summary.py --check`；以及两条最强反假设的**主动证伪**：
  (a) 共享初始化反而压住了分配效应（窄臂只能用前 r·d 个通道）；
  (b) s1 的 Δ_up=−0.0139 贴带边、n=2 ⇒ 不能排除幅度≈0.014 的真效应。
- blocker 定义：数字与落盘证据不符 / 判据线被改 / 等预算声称不成立 / 结论越证据 / 证据缺失。
- **第 1 轮判词 = FAIL（窄范围）**（`2026-10-13T19:15+08:00` 复核结束）。复核者声明工具限制：**无 shell / 无写文件工具**
  ⇒ T1（pytest 13 passed）/T2（自写脚本重算）/T5（build_model 逐位初始化控制）/T6（三份契约文件与
  `49f697f` 的内容哈希 + mtime）/T9（越界扫描、`summary.py --check`）的**执行部分未跑**，
  它以**读文件 + 手算 + 静态论证**替代（逐字段对账），并在报告 §3 给出待跑命令 —— 未跑项不计通过。
  - 复核者确认成立的部分：等 FFN 预算逐位（Σr 16/16/16、MACs_ffn 2,097,152×3、U4 4,194,304 恰两倍；
    params_ffn 2,103,296×3）、7/7 臂 `config` 26 字段逐字段相同（仅 seed 不同）、训练/评测批与 vocab=206、
    逐层 FFN MACs == 2·r·256²、`ffn_params_analytic == params_ffn` 7/7、run_log 锁取放成对且臂间恒 61 s、
    墙众 6.73–7.76 min、峰值 771–935 MB、H0 按预登记阈值规则成立、口径无墙钟。
  - **3 处 blocker（已改）**：**B1** `analysis.md:32`（并复制到 `project.yaml::conclusion`/`failure_analysis`、
    `analysis.md:153/:94`）把「四个 Δ 全小于同臂换 seed 散布 0.0098」用作「分配效应被种子噪声吃掉」的论据——
    PYR-s1 的 |Δ| = **0.0139 > 0.0098**，且 ≈ 项目自认种子散布上界 0.0138（`docs/methodology.md` §2）；
    **B2** `analysis.md:99` APYR-s1 第 6 层写作 2.4699，落盘为 **2.4684**；**B3** `analysis.md:140`
    参数差余项写作 1,024，应为 **4,096**（(4−2)·256·8 的 `lin1` 偏置；`lin2` 偏置 2,048 不随 r 缩放）。
  - **N1–N7（措辞/口径，已改）**：活性幅度括注讲反（实际 0.299–2.08、由层号主导）；
    「U4 恰为其两倍」对 FFN **参数**不成立（4,204,544 ≠ 2×2,103,296）；「nvidia-smi 前后读数」实为仅 before；
    中间深度读数未标 seed（s1 同向但小 3.4×）；把「实现等价」归功于 T1（前缀语义实由 T2 承载）；
    头部 conclusion/falsification 缺「仅两条单调阶梯」限定；`MODEL.md` 的 `code_frozen_during_training` 无条件断言。
  - **两条反假设不可由现有证据排除（写进边界，并据此弱化结论）**：(a) 全部臂共享一次初始化且窄臂
    只保留前 r·d 个通道 ⇒ 「通道子集 × 深度位置」交互不可识别，不能排除 0.01–0.014 量级的效应；
    (b) PYR-s1 |Δ| = 0.0139 贴种子散布上界 ⇒ 不能排除幅度约 0.014 的真实（负向）效应
    （决策不变：没有任何 seed 显示非均匀分配更优）。
  - **处置（全部落盘）**：``analysis.md`` §0/§1/§3-P4/§3-P6/§4/§5.1/§5.2/§5.4/§6.1/§7/§8 与 `MODEL.md`
    按复核者的最小修逐条改正（纯文字/数字，**未重训未重评**）；`project.yaml` 的 `conclusion`/`failure_analysis`/
    `falsification`/`gaps`/`conclusion_review` 同步弱化并加限定；**预登记契约 `2641/README.md` 不改**
    （其价值正是「未被事后编辑」），契约表与落盘 `ffn_params_analytic` 的同名不同值口径差异改在
    `analysis.md` §5.2 声明。
  - **本轮未采纳的修（带理由）**：N2 的 README 部分与 N5 的 README §8 部分均属改写**预登记契约**；
    改它会直接破掉「契约冻结」的证据链，故改为在 `analysis.md`/本工单里声明口径差异。
  - **复核全文** = `.scratch/64-axis3-width/REVIEW-1.md`（我原样落盘复核者最终回复全文，含其 L1–L5
    工具限制声明与交 supervisor 的待跑命令清单）。
  - **第 2 轮：未派**（本 task 明示的 2 轮上限 + 时间盒；且第 1 轮的补跑项是官能力面而非研究设计面）。
    ⇒ **复核门未闭合（第 1 轮 FAIL 已修，无 PASS 判词）**：下一轮只需核 3 处改动与结论限定，
    并补跑 T1/T5/T6/T9（无需 GPU）。主 Agent 可从本行接手。
- **复核期间的工作树变动鲁实登记**：我在派复核后提交了工单补记 `0ac0f22`（纯日志文档）⇒ 若复核
  以此为不一致点（其 L5 以 `watchdog_diff` 核工作树 == `e6d2db1`），责任在我；被审的
  `analysis.md`/`results*.json`/`project.yaml` 在复核期间**未改动**（见最终提交的 `git diff --name-only e6d2db1..HEAD`）。

（§9 由本工单在复核后补写：历轮判词只留本工单；`project.yaml` 的 `conclusion_review` 只写**当前
最后一轮**判词 —— 沿用 ticket 62 §8.8 的纪律。）

---

# 10 Phase B（自主推进 · 本轮）· 2026-10-14 20:45+08:00 起

> 本轮 = 「把本轴所有非 Backlog 研究项推到终局」。actor = axis3-lead；执行与复核**全部派 fresh 子 Agent**，
> lead 只做编排、事实文档（`project.yaml`/README）与工单。硬上限 3h（≈23:45），并发子 Agent ≤2，GPU 按 §20 串行。
> 边界不变：只写 `axes/3-width/**` 与 `.scratch/64-axis3-width/**`。

## 10.1 起点盘点（2026-10-14T20:49+08:00）

| 项 | 状态 | 已有证据 | milestone 差距 |
|---|---|---|---|
| 3.1.4 | 已完成 | 2641（7 臂 × 12k） | 复核门未闭合（第 1 轮 FAIL 已修、第 2 轮未派）⇒ 本轮派第 2 轮 |
| 3.2.1 | 已完成 | 2610/2611/2612/2613/2621 | 已终局（WP4「更长训练/更大尺寸」留白） |
| 3.1.1 | 未开始 | 无（旧 2525 属读出域，不算） | WP1（减宽前沿 + 同算力位点对照）+ WP2（width dropout 弹性宽度） |
| 3.1.2 | 未开始 | 无 | WP1（逐层 d_model schedule，固定总参数）——**实现前置**：引擎无逐层 d_model（E2 只覆盖 FFN 宽） |
| 3.1.3 | 未开始 | 无 | WP1（同层内位置相关宽度 + 长程召回对照列） |
| 3.1.5 | 未开始 | 无 | WP1（通道稀疏激活 + 活性对照列） |
| 3.2.2 | 未开始 | 无 | WP1（逐出口读出 + 各出口 tf@k + 末层质量） |
| 3.2.3 | 未开始 | 无 | WP1（tied vs untied + 参数量/语义可用性列） |

**共同代价锚（本轴复用，不再重训）**：`2641` 的 dense 13M/8L/d=256/heads=8/mlp_mult=4/12k/seed0 全宽臂
`U4-s0 = 1.5314`；同配置历史 dense 13M/12k run 散布极差 0.0057（1.5257 / 1.5287 / 1.5298 / 1.5314）；
`2641` 的 `U2-s0 = 1.5709`（每 token MACs 5.2956M）为「等算力位点」参照。复用按 `SCHEMA.md` §9 三道门
（开关关闭逐位等值 / 逐字段配置对照 / 口径标注）。

## 10.2 本轮派单契约（lead 写契，worker 先落预登记再跑）

| 单 | 项 | fresh worker | WP | 臂（新增；stock 锚复用 U4-s0） | GPU 等效预算 |
|---|---|---|---|---|---|
| W-A1 | 3.1.1 | axis3-w-3.1.1 | WP1+WP2 合并判决包 | D192 / D128（uniform 减宽）+ WD256（嵌套宽度 dropout，一 checkpoint 多档推理） | 3 臂 × 12k ≈ 25 min |
| W-A2 | 3.1.5 | axis3-w-3.1.5 | WP1 | K50 / K25（逐 token top-k 通道激活，k=50%/25%）+ F50（固定前缀半通道对照，测「条件性」本身） | 3 臂 × 12k ≈ 25 min |
| W-B1 | 3.2.3 | axis3-w-3.2.3 | WP1 | U（独立 unembed [206,256]）+ PR（tied + rank-32 低秩校正） | 2 臂 × 12k ≈ 17 min |
| W-B2 | 3.2.2 | axis3-w-3.2.2 | WP1 | PE（共享 tied + 每出口 d×d）+ IND（每出口独立表）+ LR（每出口低秩）+ T（共享 tied 同配置重训，作同批锚） | 4 臂 × 12k ≈ 33 min |

- **干净边界**：worker 只写自己的实验目录；`project.yaml` / 轴与分组 README / 本工单由 lead 写（唯一写入者纪律）。
- **预登记纪律**：worker 必须**先写 `README.md` 契约（判据线 P1..Pn + 否证线 + 带声明）再起跑**，
  跑后不改契约（口径差异改在 `analysis.md` 声明）；`README.md` 里写明「用哪个带」（`docs/research-schema.md` §12.1）。
- **口径**：判据只用等效理论算力（每 token MACs / 参数量 / layer-passes），**墙钟/吞吐只在过程日志里、禁入判据与 `cost_model`**；
  剪枝类对照按 §12.2（等 FFN 预算 / 等算力位点，并列报参数量与每 token 等效算力）。
- **实现纪律**：引擎 E2（逐层 FFN 宽）**尚未落入工作树**（`git log pathlm/depth_ar.py` 最新为 E1）⇒ 本轴按 C3
  在实验目录内自造（切片 / monkeypatch），**须自证实现等值门**（与引擎原生路径逐位同前向）；不得改 `pathlm/**`。
- **GPU 纪律**：逐臂取/放 `/tmp/pathlm_gpu.lock`（§20），臂间 ≥60 s，锁被占则等待或登记稍后重试，
  **绝不杀他人进程**；两单并发时按臂互斥交错，日志各自落盘。

## 10.3 复核计划（每项一份 fresh 复核，writer 停止后派）

- 复核目标 = 该项 `README.md` 契约 + `run.py` + 落盘 `results*.json` 与 `analysis.md` 的**一致性**：
  ①判据线未被事后编辑（契约文件哈希 + mtime 早于起跑）②数字与落盘 JSON 逐条对账（复核者自算，不引用 worker 自报值）
  ③单因素/等预算位点是否逐位成立 ④活性与等值性门非恒真 ⑤口径无墙钟 ⑥越界扫描。
- 判词落 `.scratch/64-axis3-width/REVIEW-<项>.md`；缺陷 ⇒ 派 writer 修 ⇒ **新的** fresh 复核；累计 ≤2 轮。
- 复核者无 shell 时其未跑项记为「未跑 = 不计通过」（`SCHEMA.md` §14b / AGENTS「不跑不算过」）。

## 10.4 引擎（E2）状态核对（2026-10-14T20:50+08:00，actor=axis3-lead）

- `git log --oneline -1 -- pathlm/depth_ar.py` = `e894f20`（ticket 67 · E1）；`grep mlp_mult_sched pathlm/depth_ar.py` 无命中；
  `.scratch/67-engine-changes/TICKET.md` 里 E2 的「授权状态」仍写 **未授权**。⇒ 本轮按 C3 在实验目录内自造，并再次记录该事实。

## 10.5 C1 声明回填核对（2026-10-14T20:52+08:00，actor=axis3-lead）

- 脚本枚举本轴 8 份 `project.yaml` 的 `validation` / `objective_criteria` / `falsification` / `conclusion`，
  检查「引用噪声带/阈值」与「声明用哪个带」两件事；ticket 57 的 `c1-backfill-writer` 已于 `2026-10-14T06:59` 回填 7 项。
- **唯一缺口 = `3.1.5` 的 `falsification`**（原文无带声明）⇒ 本轮补「用哪个带（C1 裁决）：0.014 bpc 带（限配对 Δ、≥5 窗口）
  为主判；通道使用率 / 激活比例等率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用 0.014」。
- `3.2.1`（已完成项）的 `falsification` 已带声明（「用哪个带（C1 裁决）」）⇒ 无需回填。

## 10.6 执行记录（本轮，追加）

- `2026-10-14T20:52:28+08:00` · axis3-lead · 读权威输入（ticket 64 §3/§6/§9、`docs/structure.md` §12 轴 3 表 + §13、
  `docs/research-schema.md` 全文、`SCHEMA.md`、本轴 8 份 `project.yaml`、`.scratch/67-engine-changes/TICKET.md`）；
  `git status --short` 干净、`HEAD=fb8a576`、`/tmp/pathlm_gpu.lock` 不存在、GPU 余量 23.6 GiB。
- `2026-10-14T20:53+08:00` · axis3-lead · 派 **第 2 轮 fresh 复核者**（subagent run `d0fe4f22-42ce-4d01-86ff-dabfafad9bfa`）
  闭合 3.1.4 复核门，交付 `.scratch/64-axis3-width/REVIEW-2.md`。
- `2026-10-14T20:5x+08:00` · axis3-lead · 补 `3.1.5` 的 `falsification` 带声明（§10.5）。

## 10.7 本轮派单用的「不变量」事实（lead 核过，worker 直接用，别自己重推）

| 事实 | 值 / 指针 |
|---|---|
| dense 13M 训练 CLI（2641 逐字） | `train_depth_ar.py <run_name> --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --dense --topology sequential --exit-mode uniform --steps 12000 --batch-size 32 --lr 1e-3 --seed 0 --eval-batches 20 --out-root <dir>` |
| harness 复用模式 | 实验目录内替换 `train_depth_ar.DepthARModel` 为工厂 + 调 `train_depth_ar.main()`（见 `2641/run.py` §训练）⇒ 训练循环/checkpoint/`results.json` 全走共享实现 |
| clean bpc 口径 | `pathlm.depth_ar.depth_curve(model, eval_arr, n_batches=20, batch_size=16)` 的**最终深度** bpc（`2641/run.py::evaluate`） |
| dense 锚 | `2641` `U4-s0` = 1.5314（每 token MACs 7,392,768、参数 6,962,688）；`U2-s0` = 1.5709（MACs 5,295,616） |
| depth-AR（非 dense，变深度 + 逐层读出）锚 | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k`：clean bpc **1.8582**、`tf@4 = 0.9694`、`tf@6 = 0.9938`、ECE 0.0066；逐深度 `tf_acceptance` 曲线见 `2610` 的 `results.json::A_arm` |
| 读出挂点（3.2.3 解绑用） | `pathlm/depth_ar.py::DepthARModel.readout(h, depth=None)`（tied `h·Eᵀ`，fp32 头）——实验目录内 monkeypatch 成独立 `U` 即可，不改引擎 |
| 逐 token top-k 通道（3.1.5） | 无引擎开关，实验目录内 patch 各块 FFN 前向（`blk.mlp` = [Linear, GELU, Linear...]） |
| 逐层 FFN 宽（E2） | `grep mlp_mult_sched pathlm/depth_ar.py` = 无命中 ⇒ **未落树**；本轴一律实验目录内自造 |

## 10.8 执行记录（续）

- `2026-10-14T21:0x+08:00` · axis3-lead · 第 2 轮复核结束（subagent run `d0fe4f22…`，builtin `reviewer`）：
  **判词 = FAIL（窄范围）**，原因 = 该 agent 环境**只有 read/grep/find/ls/watchdog_diff：无 shell、无写文件工具**，
  派单要求的 6 项执行类检查一项未跑（按其自述「未跑 = 不计通过，命令全跑过才 PASS」）。其静态复核结论：
  第 1 轮 B1/B2/B3 三处**已真改**、7 臂数字保真、Δ 与判词自洽、结论未越证据（H0 明确为阈值判词 + 两条反假设限定在位）；
  另有两条非阻塞 P2：① `analysis.md` §0「一个真信号」条目未标 seed（已补「seed 0」）；② 本工单 §7.2.2 表的
  `FFN 参数` 2,099,200 是 18:2x 中途快照（已加口径提醒，指向 §7.3 的 2,103,296）。报告全文原样落盘
  `.scratch/64-axis3-width/REVIEW-2.md`（复核者无写工具，由 lead 落盘，未改一字）。
- `2026-10-14T21:1x+08:00` · axis3-lead · **工具面教训（后续派单必须遵守）**：builtin `reviewer` agent **没有 shell**，
  凡要求「自跑/自算」的复核必须派 **`worker` agent**（有 bash），派单里写死「只读 + 只写一份报告、不改被审产物、不提交」。
- `2026-10-14T21:1x+08:00` · axis3-lead · 据此派 **第 2B 轮 fresh 复核**（`worker`，subagent run `3269e36c-838b-4789-9019-e9a016168789`）：
  实跑 T1（pytest 13）/T2（自算 Δ 与等预算）/T3（B1–B3 事实）/T5（build_model 逐位初始化控制）/T6（契约三件与 `49f697f` 逐位 + mtime）
  /T9（`git diff 49f697f..9e9b291` 本轴越界扫描 + `summary.py --check`，并明确不用 `..HEAD` 以免混入并发其它轴的提交），
  交付 `.scratch/64-axis3-width/REVIEW-2B.md`。
- `2026-10-14T21:0x+08:00` · axis3-lead · 派 **W-A1**（`3.1.1` WP1+WP2，fresh worker，run `71318038-19d5-4aac-b15c-7fa15cd8dade`），
  实验目录 `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/`（新建），过程文件
  `.scratch/64-axis3-width/2642-uniform-narrowing-process.md` 已落盘（worker 自述见该文件）。

## 10.9 引擎 E2 已落树（2026-10-14T21:24+08:00 发现，actor=axis3-lead）

- `git log --oneline -3` 出现 `4638882`「E2（与 E4 统一）：逐层 FFN 隐藏宽 schedule（引擎 + CLI + 单测 + 能力面）」（engine-writer，同 ticket 67）。
  落点：`DepthARConfig.mlp_hidden_sched: list[int] | None`（逐层 FFN 隐藏宽，任意正整数；缺省 None 逐位回落 `mlp_mult·d`）+ CLI `--mlp-hidden-sched`；
  `Block` 基类与 4 变体 + `TopKBlock`/`CarryChannelBlock` 全部接线；E4（轴 5 的等参数宽度）由同一实现承接。
- 对本轴的影响：① `3.1.4` 的逐层 FFN 宽**今后不必再自造切片**（2641 用切片做的结论不受影响，其等值门 T1/T2 已验证同前向）；
  ② **`3.1.2`（逐层 d_model）仍未被覆盖**——E2 只动 FFN 隐藏宽，`Block` 的 `d_model` 仍由 `cfg.d_model` 全局给定 ⇒ 本项「实现前置」照旧未满足，
  替代路径 = 实验目录内「前缀子空间切片」（见 `DISPATCH-B.md` §W-D）；③ 后续新增实验若需要逐层 FFN 宽，优先用 `--mlp-hidden-sched`，不再自造切片。
- 该事实按纪律回填到 `3.1.2` 的 `gaps`（措辞见下一轮 SSOT 更新）。

## 10.10 执行事故与处置（2026-10-14T21:27+08:00，actor=axis3-lead）

- **事故 1（子 Agent 时间上限）**：W-A1 的 fresh worker 在 **30 min 处被框架杀掉**（`Subagent timed out after 1800000ms`）。
  它的产物（`README.md` 契约 / `run.py` / 单测 / `smoke.sh` / `train_arms.sh`）已落盘，但 `run_batch.sh`（后台驱动）
  变成**孤儿进程**继续排队等 GPU 锁。⇒ 后续派单一律显式给 `timeoutMs`（W-A1b 给 70 min），并在契约里写死「硬停时刻」。
- **事故 2（冒烟失败 + 孤儿占队）**：孤儿驱动排到锁后会跑一个**已知失败**的冒烟（`smoke_D192_s99.log`：
  `run.py:408 AttributeError: 'tuple' object has no attribute 'detach'`，活性 forward hook 假设输出是 tensor；
  训练本体 60 步已跑完，失败在训练后的活性测量）。⇒ 我已杀掉**自己的**孤儿驱动（`kill`/`pkill -f 2642-…`；
  只杀本轴进程，未碰他人）。**踩坑记录**：`pkill -f "2642-uniform-narrowing-13m"` 同时匹配了我自己那条命令行 ⇒
  把自己的 shell 一起杀了（`AGENTS.md` §20 早就警告过的 `pgrep` 自匹配坑，这次以 `pkill` 形式复发）；之后改用
  `grep -F` 前缀截断写法核验。
- **事故 3（GPU 竞争）**：运行期间 `/tmp/pathlm_gpu.lock` 由**轴 1**的 12 层训练持有（21:17 起，pid 2319397），
  轴 2 / 轴 5 的臂也在排队。⇒ 本轴的臂按 §20 逐臂排队；契约里写死「单次等锁 ≤25 min、超时先落 CPU 部分」。
- **处置**：派 **W-A1b**（fresh worker，`timeoutMs` 70 min，run `84284b58-8e14-478d-8197-044103e054c4`）接手：
  修 hook（tensor/tuple 兼容）+ 补能变红的单测 + 双冒烟 rc=0 + 3 臂 12k + WD256 三档切片评测 + `analysis.md` + 提交；
  硬停 `22:55+08:00`。
