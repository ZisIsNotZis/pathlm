# 轴 3 · 宽度轴 —— Phase A 报告（`.scratch/64-axis3-width`）

- 时间盒：`2026-10-13 17:53 → 19:2x`（2h）；收口时刻见文末。
- 角色：轴 3 宽度轴 lead（`docs/structure.md` §12 全部研究项：`3.1.1 3.1.2 3.1.3 3.1.4 3.1.5 3.2.1 3.2.2 3.2.3` + 分组 README `3.1`/`3.2`）。
- 落盘：本轴全部产物在 `axes/3-width/**`；过程/证据/取舍在 `.scratch/64-axis3-width/**`。
- 四个提交：`49f697f`（Phase A-1 逐项想清楚 + 契约）· `e6d2db1`（WP1 执行与判决）· `0ac0f22`（工单补记）· `9e9b291`（第 1 轮复核判词的处置）。
- 状态口径：`状态` 用人话，括号内是 `docs/research-schema.md` 的枚举。

---

## 0 人话摘要（结论）

**这一轴问的事**：普通 decoder-only transformer 的宽度「处处一样宽」。能不能**不处处一样宽**——把同样的算力/参数换个地方花，从而同样质量更省算力、或同样算力质量更好？

**本轮的关键认识**：宽度轴的「非均匀」必须拆成两个不同的意思，否则会拿两个变量互相顶替：
1. **总量**：整个模型有多宽（前馈隐层总量、`d_model`）；
2. **分配**：同样的总量怎么摊到 8 层上（金字塔 / 倒金字塔）。
本项目方差带 0.014（clean bpc，`docs/methodology.md` §2），**任何小于它的差别都不可判读**。

**做了什么**：只赌一个最高价值实验（`3.1.4` 前馈隐藏比 + 逐层 schedule；前馈是每 token 参数/算力的最大单项，`d_model` 固定时它是唯一单因素）。13M / 8 层 / 12k 步，四条臂**共享同一次初始化**：

| 臂 | 每层前馈宽度（/`d_model`） | 含义 |
|---|---|---|
| U4 | 4,4,4,4,4,4,4,4 | 普通 transformer（100% 代价锚） |
| U2 | 2,2,2,2,2,2,2,2 | 等宽，前馈预算砍半 |
| PYR | 4,3,2,2,2,1,1,1 | 金字塔（浅层宽） |
| APYR | 1,1,1,2,2,2,3,4 | 倒金字塔（PYR 镜像） |

后三条臂的**前馈参数量与每 token 前馈算力逐位相等**（2,103,296 参数 / 2,097,152 MACs·token⁻¹）⇒ 它们之间比的是「**怎么分**」，不是「分多少」。

**结果（数字 + 参照系 + 人话量级）**：

1. **分多少 → 可分（真旋钮）**：前馈宽度 4→2，clean bpc（每字符比特数，越低越好）从 **1.5314 掉到 1.5709**，差 **+0.0395** = 方差带的 **2.8 倍**，可判读；换来每 token 等效算力 **−28.4%**、参数 **−30.2%** ⇒ **每省 1% 算力 ≈ +0.0014 bpc**。
2. **怎么分 → 不可分**：同样一份前馈预算在 8 层间重排，相对等宽的差是 PYR **−0.0021 / −0.0139**（两个 seed 都更差）、APYR **+0.0065 / +0.0010**（两个 seed 都更好）——四个值**全在 0.014 带内** ⇒ 按预登记判 **H0：逐层前馈分配在本尺度不是独立于前馈总量的设计维度**。**边界（与结论一起读）**：PYR-s1 的 |Δ| = 0.0139 已贴在项目自认的种子散布上界（0.0138）⇒ **判 H0 是阈值规则的判词，不等于效应已被排除**；本实验同臂换 seed 的散列（U2 0.0098 / APYR 0.0043 / PYR 0.0020）**不是 |Δ| 的上界**；全部臂共享一次初始化且窄臂只保留前 r·d 个通道 ⇒ 「通道子集 × 深度位置」交互不可识别。**决策仍稳：没有任何 seed 显示非均匀分配更优**（PYR 两 seed 都差、APYR 两 seed 都好但最大只 +0.0065）。
3. **一个真信号（副产品，不作判据）**：非均匀分配**确实改变「质量沿深度怎么分布」**——金字塔臂的中间深度明显更好（seed 0：第 6 层 1.9727 vs 等宽 2.3338；seed 1 同向但小得多：1.9729 vs 2.0799），但没有传导到最终层。

**排除了哪些别的解释**（实验成立的关键）：① 不是「换了模型」——切片实现与引擎原生窄前馈在**同形同权下前向逐位相同**（T1），切片取前缀行的语义由 T2 钉住；② 不是「两次初始化在比」——全臂共享初始化、主干逐位相同、前馈为同一次抽样的前缀；③ 不是「窄层成死路径」——逐层前馈输出幅度全为正（最小 0.299，由层号主导）；④ 不是「读数不稳」——`--no-train` 重评与首评**逐位一致**；⑤ 不是「判据事后凑」——契约（README/判据/驱动/测试）的内容哈希与 mtime 均早于起跑且与 `49f697f` 一致（提交时间 `18:02:28` 晚于起跑，已如实说明不以提交时间作证据）。

---

## 1 逐项交付（8 个研究项 + 2 个分组 README）

| 研究项 | 当前状态 | 本轮交付 |
|---|---|---|
| `3.1.4` 前馈隐藏比 + 逐层 schedule | **判过了（已完成）** | WP1 判决包：7 臂 × 12k 的 bpc + 等效算力 + 参数 + 每层分布 + 判据 P1–P6 + 三方对账 + 第 1 轮独立复核处置（`experiments/2641-ffn-ratio-frontier-13m/`） |
| `3.1.1` 剪枝式宽度 | **还没碰（未开始）** | role_map 实名 + 判据位点写死（剪枝类按 `SCHEMA.md` §15 例外 = 同每 token 等效 FLOPs 的 dense 前沿点）+ 与前馈比例的边界 |
| `3.1.2` 逐层宽度变化 | **还没碰（未开始）· 卡点** | 边界写死（判整体 `d_model` 的逐层 schedule）；**卡点**：引擎 `mlp_mult` 是全局标量、无逐层宽度开关 ⇒ 最小 patch（≈30–50 行、4 种 Block 变体）已上报 Main，**未自改共享代码** |
| `3.1.3` 位置相关宽度 | **还没碰（未开始）** | role_map 实名 |
| `3.1.5` 通道极化 / 稀疏激活 | **还没碰（未开始）** | role_map 实名 |
| `3.2.1` 读出宽度 | **还没碰（未开始）** | role_map 实名 + 轴内一致性扫描 |
| `3.2.2` 嵌入宽度 | **还没碰（未开始）** | role_map 实名 |
| `3.2.3` 词表相关宽度 | **还没碰（未开始）** | role_map 实名 + **事实更正**：词表 = **206**（205 字节 + 1 `[mask]`），tied 读出 206×256 = **52,736**（原 205 / 52,480） |
| 分组 README `3.1` / `3.2` | 更新 | 组内 5 项分工写死 + 「总量 vs 分配」两分法 + 0.014 带归属（`docs/methodology.md` §2）；词表事实更正 |

---

## 2 WP 与独立复核

**WP1**（`3.1.4`）= 前馈隐藏比前沿 + 等前馈预算下的逐层非均匀宽度。
- 契约（预登记）：`2641/README.md`（判据 P1–P6、否证条件、可判读区间、臂与预算、引擎边界），mtime `17:58:24` **早于**起跑 `18:00:37`；跑后**未改**（复核者提出的两处契约改写已拒绝，改在分析里声明口径）。
- 执行：7 臂 × 12k 步（98.3M token，matched-step），按 `SCHEMA.md` §20 逐臂取/放 GPU 锁（release→acquire 实测 **恒 61 s**），峰值显存 771–935 MB，逐臂墙钟 6.73–7.76 min（只作排程，不进判据）。
- **独立复核**：fresh 只读 reviewer，目标 revision `e6d2db1`，交付 `.scratch/64-axis3-width/REVIEW-1.md`（复核者无写文件工具，全文由我原样落盘）。
  - **第 1 轮判词 = FAIL（窄范围）**：实验本体（等预算逐位、7 臂配置逐字段一致、初始化/前缀切片的构造性论证、门与变异非恒真、锁时序与口径纪律、按预登记阈值规则推出的 H0）经其独立静态核验**成立**；判 FAIL 只因 **3 处数字/等式与落盘证据不符**（B1 用错数承载「分配效应被噪声吃掉」的论证；B2 一个中间深度 bpc 与落盘不符；B3 一处算术余项写错）+ N1–N7 措辞口径类。
  - **处置**：全部按复核者给出的最小修逐条改正（纯文字/数字，**未重训未重评**）：`analysis.md` §0/§1/§3-P4/§3-P2/§3-P6/§4/§5.1/§5.2/§5.4/§6.1/§7/§8 + `MODEL.md` + `project.yaml` 的 `conclusion`/`failure_analysis`/`falsification`/`gaps`/`conclusion_review`；结论按两条**不可由现有证据排除**的反假设（共享前缀初始化的通道子集混杂；|Δ| 贴种子散布上界）弱化并加限定。
  - **复核门未闭合**：第 2 轮**未派**（本轮上限 2 轮 + 时间盒；第 1 轮判词本身写明「修完即可复审 PASS」，且其未跑项是官能力面/工具面而非研究设计面）。下一轮只需核这 3 处改动与结论限定，并补跑 T1/T5/T6/T9（无需 GPU）。主 Agent 可从 `project.yaml::conclusion_review` 与 `.scratch/64-axis3-width/TICKET.md` §9 接手。

---

## 3 命令与结果

```bash
cd /home/z/vibe/pathlm
python3 tools/summary.py --check                      # OK：46 个 project.yaml 全部符合 research-schema
cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m
python3 -m pytest test_2641_ffn_ratio.py -q           # 13 passed（等值性/前缀门 T1+T2、参数与隐藏维审计、4 条变异 T3）
python3 run.py --aggregate --steps 12000              # → results.json（P1–P6）
python3 collect_logs.py                               # → MODEL.md / config.json / run_log.txt / 逐臂日志副本
ARMS="U4:0 U2:0" bash reeval_check.sh                 # 评测确定性：重评与首评逐位一致
python3 -c "print(1.5611-1.5750, 1.5709-1.5611)"      # -0.0139 0.0098（复核者要求的 B1 复算）
python3 -c "print(2*2103296, 4204544, 2101248-2*(4-2)*256**2*8)"   # 4206592 4204544 4096（B3/N2 复算）
git diff --name-only 49f697f..e6d2db1 | grep -vE '^(axes/3-width/|\.scratch/64-axis3-width/)' | wc -l   # 0（越界扫描）
```

关键判据输出（全部来自 `results.json`，非人工录入）：

- `P1_P2_deltas_by_seed`：seed0 `Δ_up=-0.0021 / Δ_down=+0.0065`；seed1 `Δ_up=-0.0139 / Δ_down=+0.0010`。
- `P1_verdict_by_seed`：两 seed 均「不可排序（H0）：|Δ| < 0.014」。
- `P4_same_sign_across_seeds = True`（方向可复现；注意该字段只由 Δ_up 计算，见复核 I1）。
- `equal_ffn_budget_check.pass = True`（Σr 16/16/16、FFN 参数 2,103,296×3 逐位相等、每 token FFN MACs 2,097,152×3；U4 的 Σr 与 FFN MACs 恰两倍，FFN **参数**非整两倍 = 4,204,544，差 2,048 的 `lin2` 偏置）。
- `P5_engine_equivalence_gate.pass = True`（U4-s0=1.5314；vs `23d-dense-B-12k` +0.0027）。
- `P6_liveness`：7/7 臂 `hidden_matches_schedule=True`、解析参数 == 实测参数；逐层幅度 0.299–2.08 全为正。
- `code_frozen = True`（7 臂记录的 `run.py` 哈希一致；**口径**：U4-s0/U2-s0 训练时加载的是 `49f697f` 版，差一个只用于评测阶段的派生审计字段，其哈希由训练后重评写入，bpc 逐位一致）。
- 锁时序：7 臂 release→acquire 恒 **61 s**，无重叠；重评两点 77 s。

---

## 4 未完成 / 受阻 / 没做的事（如实登记）

1. **只有 1 个研究项走到判决**（`3.1.4`）；其余 7 项仍是「还没碰（未开始）」，本轮只做了 role_map 实名与逐项想清楚（痛点/SOTA/判据/否证/代价，见 `.scratch/64-axis3-width/TICKET.md` §3）。
2. **`3.1.2` 卡在引擎能力面**（无逐层宽度开关），只把最小 patch 交上去待 Main 裁决。
3. **框架性取舍已上报 Main（不自选）**：① 引擎是否加「逐层前馈宽」开关；② 实验目录内切片这一变通是否可接受；③ 剪枝类判据位点 = 同每 token 等效 FLOPs 的 dense 前沿点；④ 预算下 `n≥2` 与 U4 锚 `n=1` 的取舍（按契约「先砍 U4-s1」执行）。
4. **`3.1.4` 的未测项**（已写进 `project.yaml::gaps`，含下一步最小实验与 GPU 成本）：非单调 schedule 未搜索（同多重集 1120 种排序；4–6 臂 ≈ 40 min）· 中间深度更优未接早退口径（`2.1.1` 的 tf@k；0 训练）· **共享前缀初始化 × 通道子集交互不可识别**（通道置换对照臂 ≈ 15 min，或把 U2/PYR seed 提到 4 ≈ 30 min）· 常数隐藏比横扫与 GLU 等价臂未跑 · 12k 未收敛（斜率 0.0018–0.0058/1k 步仍在降）⇒ 只有快照结论 · U4 锚仅 seed 0（跨 run 散布用 4 个历史 dense 13M run 估计：1.5257/1.5287/1.5298/1.5314，极差 0.0057）。
5. **复核门未闭合**（第 1 轮 FAIL 已修、第 2 轮未派）——见 §2。
6. **跨轴整合未做**：与深度轴/时间轴的同预算点同图比较属 integrator 职责。

---

## 5 交付与文件

- `49f697f` — Phase A-1：8 份 `project.yaml` role_map 实名 + 逐项想清楚（脚本化改动 16+10 处，脚本未命中即非零退出）+ `2641` 预登记契约/驱动/测试 + 工单（24 文件）。
- `e6d2db1` — Phase A-2：WP1 执行与判决（36 文件）：`analysis.md`、`results*.json`、逐臂训练/门日志、`MODEL.md`、`run_log.txt`、`gates_report.txt`、`project.yaml` 的 conclusion/failure_analysis/status/WP2/gaps/change_log、两份分组 README 口径修正。
- `0ac0f22` — 工单补记：7 臂全读数与判词、执行中的两次工程事件、预算递减记录、复核派单契约（1 文件）。
- `9e9b291` — 第 1 轮复核判词的处置：`REVIEW-1.md` 落盘 + 3 处 blocker 与 N1–N7 改正 + 结论弱化（5 文件）。
- 提交纪律：全部用**显式路径** `git add`、`git commit -F - <<'EOF'`，提交前 `git diff --cached --name-only | wc -l` 人工核对；提交后 `git status --short` 无残留（staged=0）。

---

## 6 残留风险与诚实边界

- **复核未 PASS**：第 1 轮 FAIL（窄范围）已修，但**没有新一轮 PASS 判词**；第 1 轮的 T1/T2/T5/T6/T9 由复核者本人**未执行**（其无 shell），由我自跑/自算补足（`pytest` 13 passed、B1/B3 复算、`summary.py --check`、越界扫描 0 命中、契约哈希与 mtime 核对）——**自测不是独立复核**，这部分独立确认仍然缺失。
- **结论的适用边界**：单尺度（13M/8 层/12k 快照）、单任务（enwik8 字符级）、U4 锚 n=1、只覆盖两条单调阶梯、共享前缀初始化使「通道子集 × 深度位置」不可识别；判 H0 是阈值规则的判词而非「效应已被排除」。
- **过程瑕疵（如实登记）**：复核派发后我提交了工单补记 `0ac0f22`（纯日志文档），使工作树在复核期间相对 `e6d2db1` 有一个额外提交（被审的 `analysis.md`/`results*.json`/`project.yaml` 在复核期间未改）；另有一次 `pgrep -f reeval_check.sh` 自匹配浪费约 5 min（§20 已警告的坑）。
- 本报告为收口版本；`project.yaml` / `analysis.md` / `REVIEW-1.md` / 工单 §9 是事实与过程的权威处。

收口时刻：`2026-10-13T19:2x+08:00`

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "仅在授权范围内改动：git diff --name-only 49f697f..HEAD 非空项全部落在 axes/3-width/** 与 .scratch/64-axis3-width/**（越界扫描 grep -vE 命中 0）。交付 = 8 份 project.yaml 实名与逐项想清楚 + 2 份分组 README 口径/事实更正 + 唯一最高价值 WP（3.1.4 / 2641，7 臂 × 12k matched-step）+ 独立复核件与判词处置；未自改共享 pathlm/**、train_*.py、docs/**、其它轴。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "独立验收证据齐备：.scratch/64-axis3-width/REVIEW-1.md（fresh 只读 reviewer 全文原样落盘，含 T1–T11 逐项、B1–B3/N1–N7/I1–I4 带 file:line 与复现方式、其自身工具限制 L1–L5 与待跑命令）、project.yaml::conclusion_review（第 1 轮判词 + 处置 + 「复核门未闭合」如实登记）、TICKET.md §7.2/§7.3/§7.4/§9（锁时序证据、逐臂读数与判词、预算递减、复核记录）、以及可复跑的 commands（pytest 13 passed、run.py --aggregate、summary.py --check OK、B1/B3 复算、越界扫描 0 命中）。"
    }
  ],
  "changedFiles": [
    "axes/3-width/3.1.1-uniform-narrowing/project.yaml",
    "axes/3-width/3.1.2-layer-width/project.yaml",
    "axes/3-width/3.1.3-position-width/project.yaml",
    "axes/3-width/3.1.4-ffn-ratio/project.yaml",
    "axes/3-width/3.1.5-channel-sparsity/project.yaml",
    "axes/3-width/3.2.1-readout-reparam/project.yaml",
    "axes/3-width/3.2.2-per-exit-readout/project.yaml",
    "axes/3-width/3.2.3-tied-untied/project.yaml",
    "axes/3-width/3.1-active-width/README.md",
    "axes/3-width/3.2-readout/README.md",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/README.md",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/run.py",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/test_2641_ffn_ratio.py",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/train_arms.sh",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/reeval_check.sh",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/collect_logs.py",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/report_table.py",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/analysis.md",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_U4_s0.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_U2_s0.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_U2_s1.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_PYR_s0.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_PYR_s1.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_APYR_s0.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/results_APYR_s1.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/MODEL.md",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/config.json",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/run_log.txt",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/gates_report.txt",
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/aggregate_out.txt",
    ".scratch/64-axis3-width/TICKET.md",
    ".scratch/64-axis3-width/REVIEW-1.md",
    ".scratch/64-axis3-width/edits_phase_a.py",
    ".scratch/64-axis3-width/edits_phase_a2.py"
  ],
  "testsAddedOrUpdated": [
    "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/test_2641_ffn_ratio.py（13 项：T1 切片 vs 引擎原生窄 FFN 逐位同前向、T2 前缀切片语义 + 参数/隐藏维审计 + 等预算、T3 四条判别性变异）"
  ],
  "commandsRun": [
    {
      "command": "python3 -m pytest test_2641_ffn_ratio.py -q",
      "result": "passed",
      "summary": "13 passed in 4.33s（复核后复跑一次同样 13 passed）"
    },
    {
      "command": "ARMS=\"U4:0 U2:0 PYR:0 APYR:0 U2:1 PYR:1 APYR:1\" STEPS=12000 bash train_arms.sh",
      "result": "passed",
      "summary": "7 臂全部 rc=0（19:00:26 收尾），逐臂按 §20 取放锁、release→acquire 恒 61 s、峰值显存 771–935 MB"
    },
    {
      "command": "python3 run.py --aggregate --steps 12000",
      "result": "passed",
      "summary": "results.json：P1 两 seed 均判 H0、P4 同号、equal_ffn_budget_check.pass=True、P5 pass、P6 7/7 活性"
    },
    {
      "command": "ARMS=\"U4:0 U2:0\" bash reeval_check.sh",
      "result": "passed",
      "summary": "U4-s0 prev=1.5314 now=1.5314 bitwise_equal=True；U2-s0 prev=1.5709 now=1.5709 bitwise_equal=True"
    },
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md（复核修完后复跑同样 OK）"
    },
    {
      "command": "python3 -c \"print(1.5611-1.5750, 1.5709-1.5611)\" / python3 -c \"print(2*2103296, 4204544, 2101248-2*(4-2)*256**2*8)\"",
      "result": "passed",
      "summary": "-0.0139 0.0098（B1）；4206592 4204544 4096（B3/N2）——与复核者给出的期望值一致"
    },
    {
      "command": "git diff --name-only 49f697f..e6d2db1 | grep -vE '^(axes/3-width/|\\.scratch/64-axis3-width/)' | wc -l",
      "result": "passed",
      "summary": "0（越界扫描通过：改动只落在本轴与本 axis ticket 目录）"
    }
  ],
  "validationOutput": [
    "臂读数（clean bpc，各 results_<arm>_s<seed>.json）：U4-s0 1.5314 / U2-s0 1.5709 / U2-s1 1.5611 / PYR-s0 1.5730 / PYR-s1 1.5750 / APYR-s0 1.5644 / APYR-s1 1.5601",
    "Δ_up = −0.0021(s0) / −0.0139(s1)；Δ_down = +0.0065 / +0.0010；|Δ| 全 < 0.014 ⇒ H0；两向同号 ⇒ P4=True",
    "等预算逐位：sum_r 16/16/16、macs_ffn 2,097,152×3、params_ffn 2,103,296×3；U4 sum_r/macs 恰两倍、params 4,204,544（非整两倍，差 2,048 的 lin2 偏置）",
    "总量旋钮：U4→U2 每 token MACs 7,392,768→5,295,616（−28.37%）、参数 6,962,688→4,861,440（−30.18%）、bpc +0.0395（= 2.82× 噪声带）；每省 1% 算力 ≈ +0.001393 bpc",
    "评测确定性：U4-s0/U2-s0 --no-train 重评与首评逐位一致；§20 锁 release→acquire 恒 61 s、无重叠",
    "第 1 轮独立复核判词 = FAIL（窄范围）：3 处 blocker（B1 用错数承载论证 / B2 一个中间深度 bpc 与落盘不符 / B3 算术余项写错）+ N1–N7 措辞口径，已逐条改正；结论按两条不可排除的反假设弱化；第 2 轮未派 ⇒ 复核门未闭合"
  ],
  "residualRisks": [
    "复核门未闭合：第 1 轮为 FAIL（窄范围，已修），没有新一轮 PASS 判词；第 1 轮的 T1/T2/T5/T6/T9 执行部分由复核者本人未跑（其无 shell），现由我自跑/自算补足——自测不等于独立复核，这部分独立确认缺失",
    "结论适用边界：单尺度（13M/8 层/12k 快照）、单任务、U4 锚 n=1、只覆盖同多重集两条单调阶梯（1120 种排序未搜）、共享前缀初始化使「通道子集 × 深度位置」不可识别；判 H0 是阈值规则判词，不等于效应已被排除（PYR-s1 |Δ|=0.0139 ≈ 种子散布上界 0.0138）",
    "3.1.2 受阻于引擎能力面（无逐层宽度开关），最小 patch 待 Main 裁决；其余 6 个研究项本轮未执行（仅 role_map 实名 + 逐项想清楚）",
    "过程瑕疵：复核派发后提交了工单补记 0ac0f22（纯日志），工作树在复核期间相对 e6d2db1 多一个提交（被审产物未动）；一度因 pgrep 自匹配浪费约 5 min",
    "未跑项（已登记为下一步并给 GPU 成本）：非单调 schedule 搜索、通道置换对照臂、早退 tf@k 联动、常数隐藏比横扫与 GLU 臂、更训练点收敛读数"
  ],
  "noStagedFiles": true,
  "diffSummary": "4 个提交（49f697f / e6d2db1 / 0ac0f22 / 9e9b291）：本轴 8 份 project.yaml 实名与边界/口径重写、2 份分组 README 更新、新增 2641 实验（7 臂 × 12k 的前馈隐藏比前沿 + 等 FFN 预算逐层分配，含 analysis/results/日志/哈希/门与变异测试）、工单与独立复核件；阶段内共 60+ 文件，越界 0。",
  "reviewFindings": [
    "blocker: analysis.md:32（并复制到 project.yaml::conclusion/failure_analysis、analysis.md:153/:94）- 「四条等预算 Δ 全小于同臂换 seed 散布 0.0098」为伪（PYR-s1 |Δ|=0.0139>0.0098），该句承载「分配效应被噪声吃掉」的论证 ⇒ 已删并改述为「判 H0 是阈值规则的判词，不等于效应已被排除」",
    "blocker: analysis.md:99 - APYR-s1 第 6 层 bpc 写成 2.4699，落盘为 2.4684 ⇒ 已改，并补两 seed 读数",
    "blocker: analysis.md:140 - 参数差余项写作 1,024，等式不成立，应为 4,096 ⇒ 已改并注明 lin2 偏置不随 r 缩放",
    "normal: 活性幅度括注讲反（实测 0.299–2.08、由层号主导）/「U4 恰为其两倍」对 FFN 参数不成立 / 「nvidia-smi 前后读数」实为仅 before / 中间深度读数未标 seed / 把实现等价归功于 T1（前缀语义实由 T2 承载）/ 头部 conclusion 缺「仅两条单调阶梯」限定 / MODEL.md 的 code_frozen 断言无条件 ⇒ N1–N7 已逐条改正（对预登记契约 README.md 的两处改写被拒，理由=会破契约冻结证据链，改在 analysis.md §5.2 声明口径差异）",
    "info: run.py 的 P4_same_sign 只由 Δ_up 计算（本数据无错报，未改，已登记）/ analysis.md 引用的「核验脚本」未入库（已改引 T2/T3 测试）/ MODEL.md 哈希表字节列为空 / README §12 未列 pytest（未改）",
    "复核者指出两条不可由现有证据排除的反假设：(a) 共享前缀初始化的通道子集混杂；(b) PYR-s1 |Δ| 贴种子散布上界 ⇒ 已写进结论限定、failure_analysis、gaps 与 analysis.md §7/§8，并给出下一步最小实验与 GPU 成本"
  ],
  "manualNotes": "收口状态：3.1.4 的判决包与第 1 轮独立复核（含其 FAIL 判词的逐条处置）已落盘并提交 9e9b291；复核门未闭合（第 1 轮 FAIL 已修、第 2 轮按时间盒未派），下轮只需核 3 处 blocker 改动 + 结论限定并补跑 T1/T5/T6/T9（无需 GPU）。其余 7 个研究项仍是「还没碰（未开始）」，只做了 role_map 实名与逐项想清楚；3.1.2 卡在引擎缺「逐层宽度/逐层前馈宽」开关（最小 patch 已上报 Main 待裁决）。本轴未动任何共享代码/文档/其它轴（越界扫描 0 命中）。"
}
```
