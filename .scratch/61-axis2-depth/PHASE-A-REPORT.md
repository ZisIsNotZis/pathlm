# 轴 2 深度轴 · Phase A 报告

> 角色：**轴 2 深度轴 lead**（`axes/2-depth/**` 唯一 owner，10 个研究项的 `role_map.owner`）
> 阶段：Phase A（有限预算）｜产出 revision：`233e978`（初版）+ `483b181`（按独立复核处置）
> 工单：`.scratch/61-axis2-depth/TICKET.md`｜复核：`.scratch/61-axis2-depth/REVIEW-1.md`（round 1 已回）
> 口径纪律：全部代价只用等效理论量（layer-passes/token、FLOPs、tokens-per-forward），
> **墙钟/吞吐不进任何判据**（`docs/research-schema.md` §3、`SCHEMA.md` §7/§8）。

---

## 0 人话摘要（先读这段）

**我做了什么。** 四件：①把轴 2 的 **10 个研究项的「谁负责」全部落实**（owner = 我「轴 2 深度轴 lead」，
writer / reviewer / integrator 各自写清，复核者独立于写作者）；②把 10 项逐个对照
`docs/research-schema.md` 复核了一遍「想清楚了没有」（痛点 / 业界 SOTA 与出处 / 相对 SOTA 的增量 /
方法 / 客观判据「该涨该跌必须不变」/ 做不出来怎么归因 / 等效理论代价 / 单一 milestone / WP），
并把 10 项的 WP **按价值与风险排了序**写进轴工单；③**把轴 2 的旗舰「逐 token 早退」真的跑了一次判据**；
④派 **fresh 独立复核者**复核，并按它提的问题逐条修完。

**拿到了什么（本轴最要紧的一件事）。** 上一轮独立复核（`55-review/C-PROBLEMS.md` C-P01/C-P02/C-P03）
判定：「逐 token 早退降算力又保质量」这个主张**从来没端到端验过**，而且真正能跑本线权重的解码器
**根本不会逐 token 退出**。我把这个承载体补上了——**逐 token 退出判断真的装进生产解码器**
（`DepthARDecoder.step_auto`），配套 19 条单测；然后在一个 13M 模型、**8.16 万个真实字符位置**上
逐位置**在线**决定退出深度，量**整段 bpc**与**每字符平均层数**。

**结果值不值（结论）。** 测出来的是**干净、有归因的否证**：
- 「逐字符按需深度」**从未显著赢过**最简单的「整段只跑前 k 层」：省算力多的时候（25%–71%）它比
  同算力的固定截断差 **0.0117–0.2593 bpc（0.84–18.5 倍噪声带）**；最刺眼的一条是 τ=0.9 档——
  它平均跑 5.97 层、质量 Δ+0.0120，**连「固定只跑 4 层」（Δ+0.0046）都不如**：**多花两层反而更差**，
  而且这条比较**不依赖任何插值**。只有「省 4.5–6.5%」（几乎还是全深度）时才打平。
- **不是实现错、不是口径错、不是统计假象**：这次的一个口径锚点是——「所有字符都退到第 k 层」
  这种退化情形**逐步复现了历史记录的 depth 曲线**（7 个深度全部在 Δbpc ≤ 4.6e-05 内重合），
  说明新口径是既有判据的严格扩展；四个替代解释又被逐个对照排掉（改后面的字符不影响前面、
  打散深度分布反而更差、关掉填充反而更差、填充开关在全同深度下 Δ 严格为 0）。
- **最可疑的那条机制被测出来很小**：大家担心的「前面字符早退了、后面深字符看到陈旧填充被毒害」——
  同位置配对显示**跑到满深度的字符其实没被毒害**（隔离出来的上下文效应只有 +0.0033 bpc，
  逐窗口 SE 0.0012、两半样本 +0.0020/+0.0045）；损伤**集中在早退字符自己的读出**上
  （第 1 层就交答案的字符：bpc 0.53 → 0.79）。原因是两条叠加：①门控挑走的正是本来就便宜的字符，
  浅层读出在**它们**身上加的那点税相对来看很大；②这个模型的深度曲线**在第 3 层以后极平**
  （3 层 1.8863 → 8 层 1.8582，只差 0.028），「按需少跑几层」本来就没有多少油水。
- 顺带量到一个与既有认知相反的事实：**置信度（top-1 概率）对「答案对不对」校准良好**（ECE ≤ 0.030），
  但**对「浅层答案会不会和深层不同」严重失准**（ECE 0.20–0.37）⇒ 仓库过去拿 `tf@4 = 0.97`
  反推「可以放心早退」，在量上站不住（把 `55-review` C-P02 的定性批评量化了）。

**所以：这条路本轮不值得采纳**（13M / 这套配方上），但**这件事值得做**——它把 2.1.1 从「没验过的
口号」变成「有锚点、有归因、可复算的判决」，并明确指出两个**可能反转结论**的下一步。
`2.1.1` 的 `status` 由「进行中」改为**「被否证」**，并写了失败归因（**结构问题为主因、设计问题为次因、
不是 innovation 本身失败**）——不谎报 PASS、也不含糊地停在「部分完成」。

**代价。** 本轮**没训练任何模型**（复用已入账、带 sha256 的 13M checkpoint），全部 CPU 纯推理，
**没有占用共享 GPU、没有取锁**（GPU 当时被外部任务占用 94%）。

**复核。** round 1 fresh 独立复核判「技术结论成立、双向诚实」，提 1 条 blocking（正文一处数字写紧了：
`≤0.003` vs 实测 `+0.003296`）+ 13 条 non-blocking + 1 个此前无人知道的代码缺陷（解码器的冻结填充
**无视 `proj_fill`**，对非恒等 adapter 的模型会静默给错语义）。**14 项全部已修并复验**
（含自测 6 处变异全部被抓红、实验重跑数字逐位一致、`pytest tests/` 834 passed）。round 2（有 shell 的
fresh 复核）已派发但按父级指示**不等待**，其判词到达前运行证据的效力仅限「writer 自测」。

---

## 1 逐项交付

### 1.1 role_map 实名（任务 1，10/10）`233e978`

| 角色 | 值（10 项一致） |
|---|---|
| owner | 轴 2 深度轴 lead |
| writer | 轴 2 深度轴 lead（本项单一写入者；若派发 WP worker，则 writer = 该 WP 的单一写入者，且 reviewer 换新的 fresh 上下文） |
| reviewer | fresh 复核者（独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后） |
| integrator | 研究树 integrator（M3 集成；本轴合并与跨轴一致性核对由轴 2 lead 代行） |

`reviewer ≠ writer` 在 10 项全部成立；本轮实际派发的复核者是独立上下文 subagent（见 §5）。

### 1.2 逐项「想清楚」复核（任务 2，10/10）`233e978` + `483b181`

复核维度 = ①痛点 ②业界 SOTA 与可核实出处 ③相对 SOTA 的增量 ④方法（单因素）⑤实现 ⑥客观判据
（该涨/该跌/必须不变）⑦失败归因（设计/结构/innovation）⑧等效理论代价 ⑨单一 milestone ⑩WP 与顺序。
**结论：10 项字段齐全，无「手段枚举 / 无目的 / 墙钟进判据 / 判据零区分度」类硬伤**（上一轮迁移已按
`55-review` 改过）。本轮实际改动：

1. 10 项 `role_map` 实名（§1.1）。
2. `2.1.1` 按本轮实测重写 `status/conclusion/failure_analysis/conclusion_review/gaps/work_packages/
   roadmap/experiments/argument/change_log`（§3）。
3. `2.1.3` / `2.2.5` 的 `gaps` 互相写入「同题边界」（两格判据都是「同参数量 vs 非共享 dense」且都
   不省 FLOPs；均维持 Backlog，是否合并上报 Main）。
4. 两条口径层面的**框架性取舍**上报 Main、未自选：F1「序列级 perplexity 的操作化」、
   F5「字面验收成立但实质否证时是否收紧验收口径」。

### 1.3 WP 拆分与排序（任务 3）`.scratch/61-axis2-depth/TICKET.md §3`

排序原则：先补**能承载主判据的承载体**，再做**低成本高判决力**（纯推理/协议），训练类后置。

| 序 | WP | 项 | 本 Phase |
|---|---|---|---|
| 1 | WP1-impl 退出决策头实装 + 校准 + 序列级协议 | 2.1.1 | ✅ 已执行 |
| 2 | WP1-e2e 多阈值序列级 bpc/一致率 vs 全深度 + layer-passes | 2.1.1 | ✅ 已执行 |
| 3 | WP2 异构深度 KV 语义（fill / proj-fill / no-fill） | 2.1.1 | 部分（fill vs no-fill 已测；proj-fill 待补） |
| 4 | WP5 省算力对手（同 token 预算更浅 dense） | 2.1.1 | 下一 Phase（需 GPU） |
| 5 | WP1-cal 校准（ECE / risk–coverage，两种标签） | 2.1.1 | ✅ 已执行 |
| 6 | WP6 策略感知训练（训练/推理一致的退出监督） | 2.1.1 | 下一 Phase（需 GPU） |
| 7 | WP3 深度监督配方 / WP4 精化门 | 2.1.1 | 后置 |
| 8 | `2.1.2`/`2.2.1`/`2.2.2`/`2.2.3`/`2.2.4`/`2.3.1`/`2.3.2` 各自 WP1 | 7 项 | 轴 2 下一 Phase |

## 2 执行：2.1.1 WP1（任务 4）

### 2.1 承载体（逐 token 退出进生产解码器）`233e978` + `483b181`

| 交付 | 位置 |
|---|---|
| 逐 token 退出判断 | `pathlm/depth_ar.py::DepthARDecoder.step_auto`（共享文件**只加这一处**接线；§22.7 最小接线） |
| 策略 / 序列级口径 / 校准 / 算力账 | `pathlm/exit_policy.py`（新模块）：`Top1ProbPolicy` / `PrescribedDepths` / `NeverExit`、`score_stream`、`stream_bpc`、`generate_with_exit`、`exit_calibration`、`risk_coverage`、`layerpass_ledger`、`depth_histogram` |
| 单测 19 条 | `tests/test_axis2_exit_decision.py`：三方同刻度（`step_auto` 链 / 批量逐层 / 并行 `forward`）、在线策略一致、阈值边界、**因果性**、no-fill / proj-fill 语义（含非恒等 adapter 对拍）、算力账、`NeverExit` 生成等价 |
| 复核驱动的修复 | `_layer_kv` 填充源改为经 `kv_adapters`（原为静默错语义）；新增 `_missing_mask` 让 `fill_kv=False` 的 no-fill 语义在解码器里成立；`fill_kv=True` 路径逐位不变（单测 + 变异验证） |

### 2.2 序列级实测（`2631-pertoken-exit-seqlevel`）

协议（预登记见该目录 `README.md`）：13M checkpoint（sha256 启动校验）、`batch(seed=2)×20×16`、
T=256 ⇒ **320 窗口 / 81,600 预测位置**、**CPU 纯推理、不训练**；**14 条臂**（1 基线 + 6 固定前缀 +
6 阈值扫描 + 1 地板臂）+ 4 类归因诊断；判据 = 序列级 bpc（越低越好）、代价 = layer-passes/token；
统计 = 各臂同窗口的**配对差 Δbpc**（另报绝对 bpc 的窗口间 SE）；校准 = 两种标签的 ECE + risk–coverage。

- **口径锚点（决定性）**：`prescribed-k`（k=2..7）与全深度臂**逐步复现历史 `depth_curve[k]`，
  Δbpc = 7e-06…4.6e-05（7/7，浮点噪声内重合）**⇒ 本口径是既有判据的严格扩展（加入异构深度填充代价）。
- **主判据（同算力固定前缀折线前沿的超出量）**：τ0.3 **+0.2593**、τ0.6 **+0.1168**、τ0.9 **+0.0117**、
  τ0.95 +0.0035、τ0.99 **+0.0000**、τ0.999 −0.0002、`τ0.999-min4` −0.0000（bpc）。
  无插值的严格支配：τ0.3 被 `prescribed-2` 支配（多付 0.33 层、差 0.22 bpc）、τ0.6/τ0.9 被
  `prescribed-4` 支配（多付 1.27 / 1.97 层）、τ0.95 被 `prescribed-5` 支配（多付 1.40 层）。
  唯一未被支配的两档 = τ0.999 与 `τ0.999-min4`（省 4.5–6.5%、Δ 与全深度不可分）。
- **归因（四个替代解释被逐个排掉）**：同位置配对（d8 桶 +0.0033、逐窗口 SE 0.0012、
  两半 +0.0020/+0.0045）；深度置换更差（+0.095/+0.080）；no-fill 更差（+0.051/+0.024）；
  结构零假设 `prescribed-4 × {fill, no-fill}` Δ = 0.000000。
- **校准**：`ECE_true ≤ 0.030`（深度 ≥3 时最大 0.0051）vs `ECE_tf` 0.20–0.37（方向为低估）⇒
  `tf@k` 不能当「可以放心退」的依据（量化 C-P02）；risk–coverage 显示门控**有区分力**
  （深度 1 / τ=0.6：保留 0.877 vs 筛掉 0.341），但区分力换不来前沿。
- **生成诊断列**：与全深度贪心流逐 token 一致率 28–71%（非判据；早退本就会改流）。

## 3 结论与状态变更（任务 6）`483b181`

`axes/2-depth/2.1.1-early-exit/project.yaml`：

- `status`: `进行中` → **`被否证`**（对照本项**预登记**的 falsification ②：门控被同算力固定截断支配）；
- `conclusion`: 同时写明「字面预登记验收在 τ ≥ 0.99 成立（省 6–12%、Δ ≤ 0.0002）」与「实质判决 = 否证」+
  两条新测事实（异构深度上下文代价 ≤ 0.005 bpc；`tf` / `true` 两种校准的落差）；
- `failure_analysis`: **结构问题（主因：深度曲线平坦 ⇒ 自适应可撬动空间 ≈ 0）/ 设计问题（次因、可修：
  训练配方为均匀随机退出、与推理门控不一致 ⇒ WP6）/ 非 innovation 本身失败（MoD 式容量约束路由未测）/
  非实现·口径·统计假象（四条对照）**；
- `gaps`: 6 条；`work_packages`: WP1 收口 `已完成`，新增 **WP5 / WP6**；`roadmap` / `experiments` /
  `argument` / `change_log` 同步为当前事实；`conclusion_review` = round 1 复核判词 + 处置 + 残余缺口。

其余 9 项 `status` 不变（`2.1.2`/`2.2.3`/`2.2.4` = 进行中；`2.1.3`/`2.2.5` = Backlog 未开始；
`2.2.1`/`2.2.2`/`2.3.1`/`2.3.2` = 未开始）——本轮**没有**为它们伪造进展。

## 4 命令与结果

```bash
python3 tools/summary.py --check                 # OK：46 个 project.yaml 全部符合 docs/research-schema.md
python3 -m pytest tests/test_axis2_exit_decision.py -q   # 19 passed
python3 -m pytest tests/ -q                      # 834 passed（默认 CPU，未用 GPU）
# 主线实验（纯 CPU；不取 GPU 锁；重跑两次数字逐位一致）
python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py \
        --out axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel
# 变异检查 6 处（A..F：want&~decided / frozen_self / min_depth / proj_fill adapter / _missing_mask /
#             score_stream depth-0 决策）⇒ 全部被抓红，改完立即恢复、git diff 干净
git add <显式 21/14 个路径> && git diff --cached --name-only | wc -l && git diff --cached --check && git commit -F -
```

环境事实（如实登记）：`2026-10-13T12:41+08:00` 实测 RTX 4090 被**外部任务**占用（used 19240 MiB /
24564 MiB、util 94%；`/tmp/llama_gpu.lock` 存在、`/tmp/pathlm_gpu.lock` 不存在）⇒ 本 Phase **未做任何
GPU 工作、未取锁、未杀他人进程**；训练类 WP 顺延（占位理由与排队策略见 `TICKET.md §3`）。

## 5 独立复核（任务 5）

| round | 复核者 | 对象 | 结果 |
|---|---|---|---|
| 1 | fresh 独立上下文 subagent（非 writer，只读工具集） | `233e978` | 四块判「成立」；`被否证` 判「双向诚实、无谎报」；**1 blocking（F1）+ 13 non-blocking（F2–F14）**；报告 `.scratch/61-axis2-depth/REVIEW-1.md`（原样落盘） |
| 2 | fresh 独立上下文、**有 shell** 的复核者（run `5957ebc2`） | `483b181`（处置提交） | **已派发，本轮按父级指示不等待**；范围 = round 1 未能运行的运行证据（pytest / 变异 / 重跑 / `summary.py --check`）+ 增量处置核查 |

**round 1 处置（14/14 已落盘，`483b181`）**：blocking F1（正文「≤0.003 bpc」vs 实测 +0.003296）
按实测改写为「≤0.005 bpc」并补逐窗口 SE / 两半样本；F2 区间、F3 算术恒等式不当佐证、F4 支配定义加横轴、
F5 历史 `tf_acceptance` 是另一样本（seed 3 vs 2）、F6「浮点噪声内重合」、F7「从未显著超过」、
F8 表列定义、F9 新增 SE/两半字段、F10 校准表补 depth 5 行与一致率区间、**F11 解码器填充语义缺陷
（proj_fill 静默错 + no-fill 未定义）**、F12 复核报告与 `conclusion_review`、F13 臂数 14 / 日志覆盖模式 /
去重复字段、F14 `generate_with_exit` 语义标注。

**writer 自测补跑（round 1 无 shell ⇒ 由其替代；自测 ≠ 独立验收）**：19 passed / 834 passed；
6 处变异全部被抓红；**自测还发现并修掉一个测试缺陷**——`tiny_model` 的默认初始化让「填充源取错状态」
与「adapter 被绕过」两类对照落到 1e-6 量级而**静默通过**（等于零测试），已新增 `excited_model`
与活性断言后 B/D/E 才被抓红；实验重跑与 `233e978` 数字逐位一致。

## 6 剩余 / 受阻项与下一步

| # | 项 | 类型 | 代价 |
|---|---|---|---|
| N1 | **round 2 独立复核**（运行证据 + 增量） | 复核 | 已派发（`5957ebc2`），未等待 ⇒ 运行证据当前效力限于 writer 自测 |
| N2 | **WP5 省算力对手**（同 token 预算 L=4/L=6 dense）——回答 C-P12：要省一半算力，是「8 层固定退 4」还是「直接训小模型」 | 训练 | GPU 2 臂 × 13M × 12k 步，按 §20 排队取锁 |
| N3 | **WP6 策略感知训练**（训练/推理一致的退出监督）——本结论唯一可修项，可能反转 | 训练 | 同 N2（+1 对照臂） |
| N4 | WP2 的 `proj-fill` 臂；自由生成质量 | 纯推理 | 0 GPU |
| N5 | 100M checkpoint 复跑；L=16（更深曲线是否仍平坦） | 纯推理 / 训练 | 0 GPU / GPU 1 臂 |
| N6 | `2.1.2` / `2.2.x` / `2.3.x` 各自 WP1 | 训练为主 | 逐项排期 |
| N7 | F4（`2.1.3` 与 `2.2.5` 是否合并）、F5（是否收紧 `2.1.1` 验收口径）待 Main 裁 | 决策 | — |

**本 Phase 明确未做（不得当成已完成）**：任何 GPU 训练；`2.1.2`/`2.2.1`/`2.2.2`/`2.2.3`/`2.2.4`/
`2.3.1`/`2.3.2` 的实验执行；`2.1.1` 的 WP2/WP3/WP4/WP5/WP6；round 2 的判词。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "只动 axes/2-depth/**、.scratch/61-axis2-depth/** 与实现承载体所需的 pathlm/exit_policy.py（新模块）、pathlm/depth_ar.py 的 step_auto/_layer_kv/_missing_mask 接线、tests/test_axis2_exit_decision.py。未碰其它轴、docs/structure.md、docs/research-schema.md、共享工具。10 份 project.yaml 只改 role_map（+2.1.1 的当前事实、2.1.3/2.2.5 的 gaps 边界）。共用 2 个提交（233e978 初版、483b181 复核处置）。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "证据齐备且可复跑：axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/{README.md(预登记契约),config.json,run.py,results.json,run_log.txt,analysis.md,MODEL.md}；.scratch/61-axis2-depth/{TICKET.md,REVIEW-1.md}（分半稳健性检查的原始输出在 /tmp，其数值已落在 results.json 的 same_position_paired[].delta_halves）；命令级证据见 report §4（summary.py --check OK / 19 passed / 834 passed / 6 处变异全部抓红 / 重跑数字逐位一致）。"
    }
  ],
  "changedFiles": [
    "pathlm/exit_policy.py",
    "pathlm/depth_ar.py",
    "tests/test_axis2_exit_decision.py",
    "axes/2-depth/2.1.1-early-exit/project.yaml",
    "axes/2-depth/2.1.2-skipping/project.yaml",
    "axes/2-depth/2.1.3-repeating/project.yaml",
    "axes/2-depth/2.2.1-residual-form/project.yaml",
    "axes/2-depth/2.2.2-attention-residual/project.yaml",
    "axes/2-depth/2.2.3-learned-layer-sum/project.yaml",
    "axes/2-depth/2.2.4-parallel-sublayer/project.yaml",
    "axes/2-depth/2.2.5-weight-sharing/project.yaml",
    "axes/2-depth/2.3.1-layer-capacity/project.yaml",
    "axes/2-depth/2.3.2-layer-resolution/project.yaml",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/README.md",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/config.json",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/results.json",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run_log.txt",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md",
    "axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/MODEL.md",
    ".scratch/61-axis2-depth/TICKET.md",
    ".scratch/61-axis2-depth/REVIEW-1.md"
  ],
  "testsAddedOrUpdated": [
    "tests/test_axis2_exit_decision.py (19 tests: 三方同刻度 / 在线策略一致 / 阈值边界 / 因果性 / no-fill 与 proj-fill 非恒等对拍 / 活性断言 / 算力账 / 校准表 / NeverExit 生成等价)"
  ],
  "commandsRun": [
    {"command": "python3 tools/summary.py --check", "result": "passed", "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md（改前/改后各跑一次）"},
    {"command": "python3 -m pytest tests/test_axis2_exit_decision.py -q", "result": "passed", "summary": "19 passed"},
    {"command": "python3 -m pytest tests/ -q", "result": "passed", "summary": "834 passed in 179s（含新增 19 条；默认 CPU，未使用 GPU）"},
    {"command": "python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py --out <同目录>", "result": "passed", "summary": "CPU 纯推理；产出 14 臂 + 锚点 + 前沿 + 校准 + 4 类诊断；prescribed-k 复现历史 depth_curve（Δbpc 7e-06..4.6e-05）；重跑与 233e978 数字逐位一致"},
    {"command": "mutation A-F（want&~decided / step_auto frozen_self / min_depth / proj_fill adapter / _missing_mask / depth-0 决策）", "result": "passed", "summary": "6 处变异全部被抓红（1..5 failed），每次改完立即恢复、git diff 干净"},
    {"command": "python3 - <<'PY' 分半稳健性检查（前/后 160 窗口的 d8 桶 Δ）", "result": "passed", "summary": "τ=0.6: +0.0020/+0.0045；τ=0.9: +0.0007/+0.0004 ⇒ 上下文代价稳健上界 ≤0.005 bpc"},
    {"command": "git add <显式路径> && git diff --cached --name-only | wc -l && git diff --cached --check && git commit -F -", "result": "passed", "summary": "两个提交：233e978（21 路径）、483b181（14 路径）；均逐条核对 staged 列表、无空白错误"}
  ],
  "validationOutput": [
    "口径锚点：prescribed-2..7 与全深度臂逐步复现历史 depth_curve[k]（1.9988/1.8863/1.8628/1.8593/1.8585/1.8584/1.8582），Δbpc = 7e-06..4.6e-05 ⇒ 新口径与既有判据同刻度（浮点噪声内）",
    "主判据（14 臂）：门控相对同算力固定前缀折线的超出量 = +0.2593 / +0.1168 / +0.0117 / +0.0035 / +0.0000 / −0.0002 / −0.0000（τ=0.3/0.6/0.9/0.95/0.99/0.999/min4）；τ=0.9（5.967 层）连 prescribed-4（4.000 层）都不如（不依赖插值）",
    "归因：同位置配对 d8 桶 +0.0033（窗口 SE 0.0012、两半 +0.0020/+0.0045）；置换臂更差 +0.095/+0.080；no-fill 更差 +0.051/+0.024；结构零假设 prescribed-4×{fill,no-fill} Δ=0.000000",
    "校准：ECE_true ≤0.030（深度≥3 最大 0.0051）vs ECE_tf 0.20–0.37（低估）；risk-coverage d1/τ=0.6 保留 0.877 vs 筛掉 0.341",
    "变异检查：A..F 六处实现改动均被测试抓红（writer 自测；round 2 将独立复做）",
    "独立复核 round 1：四块判成立；status=被否证 判「双向诚实、无谎报 PASS/FAIL」；提 1 blocking + 13 non-blocking，14/14 已落盘处置（483b181）"
  ],
  "residualRisks": [
    "round 2 独立复核（有 shell，运行证据 + 增量核查）已派发（run 5957ebc2）但按父级指示未等待 ⇒ 「单测能失败 / 数字可复现」目前由 writer 自测支撑，属自测而非新的独立验收",
    "结论外推边界：单 checkpoint、单 seed、13M、1 epoch 快照、训练配方为逐位置随机退出；不得外推为「逐 token 变深度机制本身不成立」（MoD 式容量约束路由未测）",
    "「异构深度上下文代价 ≤0.005 bpc」是在**被随机退出训练过的** checkpoint 上测的（它对混合深度鲁棒）；未在只训过全深度的模型上测",
    "本 Phase 未做任何 GPU 训练：WP5（省算力对手）与 WP6（策略感知训练，可能反转结论）均未执行",
    "2.1.1 的 milestone 验收文字（Δ ≤ 0.014）字面成立而实质判决为否证 ⇒ 是否收紧口径为「相对同算力固定深度前沿的超出量 ≤ 0」待 Main 裁（未自改判据线）",
    "F4：2.1.3 与 2.2.5 判据近乎同题，是否合并待 Main 裁（当前两格均 Backlog 并在 gaps 互指）"
  ],
  "noStagedFiles": true,
  "diffSummary": "两个提交：233e978 = role_map 实名（10 份 project.yaml）+ 逐 token 退出承载体（新模块 pathlm/exit_policy.py、DepthARDecoder.step_auto、16 条单测）+ 新实验 2631（14 臂序列级实测）+ 2.1.1 状态/结论/归因/gaps/WP/roadmap 重写；483b181 = 按独立复核 REVIEW-1 处置 14 项（1 blocking 数字修正 + 13 non-blocking，含解码器 _layer_kv 的 proj_fill 静默错语义修复与 _missing_mask no-fill 支持、单测增至 19 条、实验结果与文档同步更新、复核报告落盘）。",
  "reviewFindings": [
    "blocker（已修）: axes/2-depth/2.1.1-early-exit/project.yaml 与 2631/analysis.md 的「异构深度上下文的代价 ≤ 0.003 bpc」与 results.json::same_position_paired(τ=0.6,d8)=+0.003296 矛盾；同族副例「ECE_true ≤0.005」vs depth 3 实测 0.0051 ⇒ 已按实测改写为 ≤0.005 / 0.0051，并补逐窗口 SE 与两半样本",
    "blocker（round 1 未提、我自查发现，已修）: tests 原 16 条对「填充源取错状态」与「proj_fill adapter 被绕过」两类变异抓不住（tiny_model 默认初始化让差异落到 1e-6，等于零测试）⇒ 新增 excited_model + 活性断言，6 处变异现已全部抓红",
    "no blockers remain open from round 1: F2–F14 全部落盘（区间限定词、恒等式不当佐证、dominated_by_fixed 横轴条件、tf 样本 seed 差异、浮点噪声内重合、从未显著超过、表列定义、SE/两半字段、校准表补行、解码器填充语义代码修复、REVIEW 落盘、臂数 14/日志覆盖模式/去重复字段、generate_with_exit 语义标注）",
    "pending: round 2（run 5957ebc2）需独立复做运行证据与增量核查；其判词未到前不得声称「已独立验收运行证据」"
  ],
  "manualNotes": "框架性取舍已上报 Main 且未自选：F1「序列级 perplexity 的操作化」= teacher-forcing 在线决策整段 bpc（以 prescribed-k 逐步复现历史 depth_curve 作锚点证明同刻度）；F2 同 checkpoint 比较另报配对 SE（判据门槛仍用 0.014 跨 run 带）；F3 本轮只做纯推理、训练类 WP 后置（GPU 被外部任务占用）；F4/F5 待 Main 裁。本轮未做任何 GPU 工作、未取 §20 锁。round 2 复核（subagent run 5957ebc2）在返回后会自动投递，若其提 blocking，需要一轮 writer 修复 + 新的 fresh 复核。"
}
```
