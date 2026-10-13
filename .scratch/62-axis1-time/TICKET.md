# ticket 62 · 轴 1 时间轴 lead（Phase A）

> 性质：**过程工单**（轴 1 的编排 + 执行 + 复核轨迹）。事实文档 = `axes/1-time/**/project.yaml`
> 与 `axes/1-time/README.md`、各分组 `README.md`；本文件只记「我们一步一步怎么走的」。
> 用词以 `docs/structure.md` §12 为准（轴 1 的分组 1.1/1.1.4/1.2 与 12 个研究项；`1.3 寻址` = 参照）。
> 记录纪律：显式时间戳 + actor + 命令与结果 + 证据锚点；框架性取舍上报 Main（见 §4）。

## 1 范围与契约（轴 1 lead）

| 项 | 值 |
|---|---|
| 轴 | 1 时间轴（`axes/1-time/`） |
| owner | axis1-lead（轴 1 时间轴 lead） |
| writer | 本轴单一写入者 = axis1-lead；WP 若派 worker，则 writer = 该 WP 的单一写入者 |
| reviewer | axis1-reviewer（fresh 独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后） |
| integrator | tree-integrator（研究树 integrator；M3 集成，本轮由 axis1-lead 代行本轴合并与跨轴一致性核对） |
| 规范输入 | `docs/structure.md` §12（节点/用词）、`docs/research-schema.md`（字段契约）、`SCHEMA.md` §6–§24 |
| 本轴节点 | 分组 1.1 / 1.1.4 / 1.2；研究项 `1.1.1 1.1.2 1.1.3 1.1.4.1 1.1.4.2 1.1.4.3 1.1.4.4 1.1.4.5 1.2.1 1.2.2 1.2.3 1.2.4`（共 12） |
| 边界 | 只动 `axes/1-time/**` + `.scratch/62-axis1-time/**`；不碰其它轴、`docs/**`、共享 `pathlm/**`/`train_*.py`/`tools/**` |
| 并发/锁 | 并发子 Agent ≤3；GPU 训练按 `SCHEMA.md` §20 互斥锁**串行**；纯 CPU/文档不取锁 |
| 预算（Phase A） | 时间盒 ≤2.5h；目标 = ①12 项 role_map 实名；②逐项「想清楚」复核并修 `project.yaml`；③WP 拆分与按价值排序落盘；④只执行 1 个最高价值 WP（旗舰 = 信息右移 1.1.4.1/1.1.4.5 的质量-代价前沿）；⑤fresh 独立复核至无问题（返工 ≤2 轮） |

## 2 Phase A 计划与逐项复核

### 2.0 轴级共同口径（沿用已冻结项，不自选）

- 目标函数：**用更少算力拿到序列级类似效果**；代价口径**只**用等效理论量（FLOPs / layer-passes /
  tokens-per-forward，考虑 pipelining），**墙钟/吞吐不进判据**（`docs/research-schema.md` §3）。
- 对照 = 同尺寸 **stock 全因果注意力**（`axes/7-references/README.md`，13M B0 快照 bpc = 1.5074；
  2610 冻结主干 = 1.4836 @ 24×16 口径）归一为 100% + 同掩码无记忆地板 + 同预算窗口/量化点。
- 判据 = 长程召回 / KV 保留曲线（该涨）+ 每位置内存字节、每 token 额外 FLOPs（该跌）+ 质量-代价前沿。
- 噪声带 0.014（`docs/methodology.md:19` 的 13M 跨 run bpc 漂移界）。**该带是否可当阈值 = 框架性取舍**，
  见 §4 F1（上报 Main，不自选）。
- 「局部先验抗性」（`SCHEMA.md` §19）：长程类结论必须同时在合成任务与真实文本上复核；`enwik8`
  局部先验强，单独读数不得下机制结论。

### 2.1 逐项「想清楚」复核（12/12）

复核维度（对齐 `docs/research-schema.md` §2 与 `SCHEMA.md` §23 七问）：
①痛点 ②业界 SOTA 与出处 ③相对 SOTA 的增量 ④方法（单因素） ⑤实现 ⑥**客观判据（该涨/该跌/必须不变）**
⑦做不出来怎么归因（设计/结构/innovation） ⑧等效理论代价 ⑨单一 milestone ⑩WP 与推进顺序。

| 研究项 | 复核结论 | 处置 |
|---|---|---|
| 1.1.1 hard 窗口 | 十项齐全；主数字（丢前缀代价 +0.212、锚点 76.8%→93.3%、beyond 0.4%）仍在 `.scratch`/旧 stub，`experiments` 已迁入 8 个长 T 目录；WP1（仓内自包含复算）与 WP2（拆「评测期丢前缀」/「累计预算」两列）未做 | 补 role_map 实名；WP 排序见 §3 |
| 1.1.2 soft 衰减 | 十项齐全；旧读数（固定 w 全头共用、−0.0068 带内）已按 A-P10 降为「口径不全的旧观测」；per-head slope 与长度外推两栏未做 | 补 role_map 实名 |
| 1.1.3 可见量上限 | 十项齐全；本项按 MAPPING §3 = 1.1.1 的拆分位、只指针不接收文件 | 补 role_map 实名 |
| **1.1.4.1 显式记忆槽/线性状态** | 十项齐全；55-review M-01/M-03/M-12 = **从未测自身目的（质量-代价前沿）**；M-04/M-05/M-06 = 机制被降级单共享槽、无业界对照；M-08/M-09 两处错误归因。**本 Phase 执行 WP1（前沿）** | 补 role_map 实名；WP 排序见 §3；WP1 本轮执行 |
| 1.1.4.2 压缩 | 十项齐全；旧证据只是 retention 仪器、不构成压缩结论（M-18）；三类压缩无任何自有实验 | 补 role_map 实名 |
| 1.1.4.3 压缩触发 | 十项齐全；旧判据（ρ 曲线）与实测量（驱逐优先级）错配、门槛算术不可达（P15/P16） | 补 role_map 实名 |
| 1.1.4.4 外部检索 | 十项齐全；本仓零经验，判据/否证条件已预登记 | 补 role_map 实名 |
| 1.1.4.5 implicit vs explicit | 十项齐全；implicit 已证退化（窗口外 ≈ chance + 同掩码地板）；缺「同任务/同预算/同仪器」正面对撞与「前缀可丢」端到端协议 | 补 role_map 实名 |
| 1.2.1 dilated | 十项齐全；本仓零经验；判据钉在「同每 token 可见键数」 | 补 role_map 实名 |
| 1.2.2 块稀疏/线性核 | 十项齐全；旧「线性更差」降为带伤/已改核的负观测；真块稀疏掩码从未实装（R2） | 补 role_map 实名 |
| 1.2.3 内容选择 top-k | 十项齐全；旧读数（T=256 净亏、缓存结构性 −0%）可信但 0.014 阈值判读按来源自否（P6）判为边缘 | 补 role_map 实名 |
| 1.2.4 动态合并 | 十项齐全；本仓零经验；判据钉在「同每 token 等效总 FLOPs」 | 补 role_map 实名 |

- 全 12 项 `cost_model` 复核 = 纯等效理论量、无墙钟/日期 ✓。
- `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合 docs/research-schema.md`（见 §5）。
- 唯一实质改动 = role_map 由「（待指派）」占位改为实名（owner=axis1-lead，writer=每项 worker，
  reviewer=fresh 复核者，integrator=tree-integrator）；其余字段为**只读复核**，结论与 `55-review`
  的 M/A 台账一致，不越界改判据。

## 3 WP 拆分与推进顺序（按价值排序）

排序原则（`AGENTS.md` §5 判据）：**它是否可能改变我们对「某种算力路径值不值」的判断**。
先做能承载主判据、且低成本高判决力的 WP；训练类打包后置（昂贵、需 GPU 锁、需排队）。

| 序 | WP | 所在项 | 类型 | 判决力 | 成本（等效） | 本 Phase |
|---|---|---|---|---|---|---|
| 1 | **WP1：把显式状态放进质量-代价前沿** | 1.1.4.1 | 训练+纯推理 | 最高（补本项自身目的，M-01/M-03/M-12） | GPU：2–3 个 13M 窗口臂 × 12k 步（~10min/臂）；其余复用 | ✅ **本轮执行** |
| 2 | WP1：implicit vs explicit 同仪器对撞 | 1.1.4.5 | 复用+推理 | 高（核心命题：explicit 是否有希望） | 复用 1.1.4.1 臂 | 下一轮（依赖 #1） |
| 3 | WP1：自包含复算窗口/锚点臂（升级为仓内 results.json） | 1.1.1 | 训练+推理 | 高（价格上界，所有前沿的对照） | GPU：{full, win-k, win+anchor} × n≥2 | 下一轮 |
| 4 | WP1：外部检索最小判决 | 1.1.4.4 | 训练+推理 | 高（explicit 右移的另一形式，零经验） | GPU：{stock, stock+检索} 两臂 | 下一轮 |
| 5 | WP1/WP2：长 T 是否回本 + 内容 vs 位置同预算对照 | 1.2.3 | 训练 | 中高 | GPU：{full, top-k} × T∈{512,1024,2048} | 后置 |
| 6 | WP1/WP2：真块稀疏掩码 + 线性核无伤质量点 | 1.2.2 | 训练 | 中高 | GPU：{local+global, block+stripe} 对齐可见键数 | 后置 |
| 7 | WP1：摘要槽/低秩/pooling 同预算质量 | 1.1.4.2 | 训练 | 中 | GPU：六臂 | 后置 |
| 8 | WP1/WP2：per-head slope + 长度外推 | 1.1.2 | 训练 | 中 | GPU：三臂 × 2 seed | 后置 |
| 9 | WP1：动态合并最小判决 | 1.2.4 | 训练 | 中 | GPU：{stock, 金字塔 r=2/4/8} | 后置 |
| 10 | WP1：dilated 掩码最小判决 | 1.2.1 | 训练 | 中 | GPU：三臂对齐可见键数 | 后置 |
| 11 | WP1：按正确判据重判优先级 | 1.1.4.3 | 训练+推理 | 中低 | GPU：三规则 × 保留/重建 | 后置 |
| 12 | WP1：复用 1.1.1 排 k 剂量曲线 | 1.1.3 | 纯推理 | 低（并入 1.1.1 的判定） | 0 GPU | 后置 |

**本轮预算理由**：① 旗舰方向 = **信息右移**，其最致命缺口是「从未把自己放进前沿」；
② 该 WP 可用**已有 2634 四臂**（win16 / carry-win16，seed0/1）+ **已有全因果锚**（2610 trunk）
+ **少量新窗口臂**完成，判决力最高、成本可控；③ 纯 CPU 的文档/对账工作不取锁，与 GPU 线并行。

## 4 上报 Main 的框架性取舍（不自选）

| # | 取舍 | 我的处置（暂定） | 理由 | 状态 |
|---|---|---|---|---|
| F1 | **0.014 噪声带是否当阈值**：carry 相对 none 的臂间差 = +0.0031/+0.0025 bpc，**大于本格自测 σ_cell=0.00166、远小于项目 0.014 带**。按 0.014 判 → 不显著；按本格同配置换种子极差判 → 显著 | 双列并报；判决文字同时给两种读法，**不自选** | 口径归因属框架性取舍（`SCHEMA.md` §6 声明 `falsification` 写「越噪声带」，但未指明用哪个带） | 已上报 `progress_update`，待 Main 裁 |
| F2 | **质量-代价前沿的「代价」轴定义**：每位置等效 KV 字节（fp16，n_layers×2×d×2×k）还是每 token 额外 FLOPs | 采用「每位置等效 KV 字节」为主轴（与 R1 量化点同轴），FLOPs 作副轴 | M-12 明确要求状态字节换算成每位置字节并与窗口/量化同图 | 已上报，待 Main 确认 |
| F3 | **是否付「形态税」**：显式状态点相对同预算纯窗口点的质量差在 0.014 带内 ⇒ 是否判「不值得付这份实现复杂度」 | 只报前沿位置与支配关系，不把「形态税」写成结论 | 属框架性取舍 | 已上报 |
| F4 | **1.1.4.1 的 `milestone` 措辞**要求「与真实 DeltaNet/GLA 实现对齐」；本仓**无联网、无外部实现**，只能与自写 plain/delta + 文献条目对照 | 在 `gaps` 显式记「业界实现对齐未做（无检索/无外部代码）」；不擅自改 milestone 文字 | 不改预登记契约 | 已上报 |
| F5 | **需要改共享引擎才能测完整机制（WP2：逐层状态 + 门控衰减 + 内容相关写入门）** | **不擅自改**；只在 `gaps` 记「需要什么改动 + 为什么 + 最小 patch 描述」，上报 Main | 边界纪律（本轴不碰共享 `pathlm/**`） | 已上报 |

### F5 的最小 patch 描述（改共享引擎的申请，本轴不执行）

- **需要什么**：`pathlm/depth_ar.py::CarryChannel`（约 1490–1530 行）目前是**单共享槽**
  （`shared_slots=true`：一个 8192-float 状态覆盖全部 12 层、只从输入流 h0 写）。要测完整机制需：
  ① 每层一个独立状态（12 个槽）；② 写入门内容相关（β 之外再加一个从 h0 派生的门）；
  ③ 读出增益 γ 逐层可学（已存在）；④ 可选衰减 γS。
- **为什么**：55-review M-05/M-06 指出单共享槽是**被引擎/墙钟开销逼出的退化实现**
  （per-layer 12 状态实测 207–425 ms/step，超预算 2.4×），业界 Gated DeltaNet/GLA 都是**逐层**；
  不修就无法区分「机制不成立」与「本实现不成立」。
- **最小 patch**：把 `CarryChannel` 的状态从 `[1, n_heads, hd, hd]` 扩到 `[n_layers, n_heads, hd, hd]`，
  `CarryChannelBlock` 各层持层号并索引自己的槽；`write_read` 增加 `layer_idx` 参数；
  训练侧 `carry_scan_*` 按层循环（或一次性 batch）；**改动量约 60–120 行**，只动 `pathlm/depth_ar.py`，
  不新增公开开关（先内部实验）。风险：墙钟会再超（每层一次扫描），需按 §20.3 缩步数。

## 5 执行记录

### 5.1 Phase A-1（2026-10-13T14:38–14:4x，actor：axis1-lead）

1. `2026-10-13T14:38:08+08:00` 读权威输入：`docs/structure.md`（§12–§15）、`docs/research-schema.md`、
   `axes/1-time/**/project.yaml`（12 份）与轴/分组 README、`.scratch/55-review/{A,M}-PROBLEMS.md`、
   `.scratch/56-restructure/MAPPING.md`、`.scratch/57-program/TICKET.md`、`SCHEMA.md` §6–§24、
   `axes/2-depth` 的同类 Phase A（`.scratch/61-axis2-depth/TICKET.md`）作格式参照。
2. `2026-10-13T14:39:52+08:00` `git status` ⇒ 仅 12 份 `axes/1-time/**/project.yaml` 待改（工作树其余干净）；
   分支 `main`，HEAD = `e970990`。
3. `2026-10-13T14:40:00+08:00` `nvidia-smi` ⇒ RTX 4090，used ~5.6GB / free ~18.4GB，util 45%；
   外部进程 pid 145968（3.9GB）、pid 17714（1.1GB）；`/tmp/pathlm_gpu.lock` **不存在**（其它锁：
   `llama_gpu.lock`/`campaign_driver.lock` 等属其它线）⇒ 本轴 GPU 工作按 §20 取 `pathlm_gpu.lock` 串行。
4. `2026-10-13T14:40:10+08:00` 12 份 `project.yaml` 写入 role_map 实名（脚本 `/tmp/axis1_rolemap.py`，
   逐文件确认 12/12 命中）。
5. `2026-10-13T14:41:00+08:00` `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合
   docs/research-schema.md`（含 role_map 四角色齐、reviewer≠writer）。
6. `2026-10-13T14:4x` 提交本批（显式路径 `git add axes/1-time/*/project.yaml .scratch/62-axis1-time/`）。

### 5.2 Phase A-2（旗舰 WP1：1.1.4.1 质量-代价前沿）

见 §6 与实验目录 `axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/`。

- `2026-10-13T14:43:30+08:00` 取 `/tmp/pathlm_gpu.lock`，300 步冒烟（win24）⇒ 0.22 min / 300 步
  ≈ 44 ms/step ⇒ 12k 步 ≈ 8.6 min/臂；峰值显存 2015 MB。
- `2026-10-13T14:44:31 – 15:31:46` 训练 5 个纯窗口臂（win17/18/24/32/128，anchors=0，seed 0），
  按臂取/放锁、臂间 ≥61 s；逐臂 rc=0。
- `2026-10-13T15:32:18` 评测（整程持锁，0.32 min，峰值 627.7 MB）⇒ `results.json`。
- `2026-10-13T15:3x` 写 `analysis.md` / `MODEL.md`，提交本批。

### 5.3 WP1 判决（一句话）

**显式记忆状态（单共享槽 delta）在其自身目的（质量-代价前沿）上被否证**：
它被一个**只宽 1 个位置、内存更少**的纯窗口 `win17` **支配**
（208,896 B < 212,992 B，bpc 1.624104 < 1.624979；**点估计支配，σ_cell 读法下为边缘、0.014 带读法下不显著**）；
其全部收益（+0.0031 bpc = 窗口→全量差距的 2.5%）小于把窗口加宽 1 格（+0.0040 bpc = 3.2%）。
口径读法待 Main 裁（F1，见 §4）；**预登记点集下（win16→win24 弦插值到 k_eff=17.333 = 1.623401）同样判负**
⇒ 主结论不依赖事后追加臂。详见实验目录 `analysis.md` §0/§1。

## 6 WP1（1.1.4.1 质量-代价前沿）设计与执行

## 7 复核计划（SCHEMA.md §14/§14b）

- 排程：writer（axis1-lead）停止写入 → 派 fresh 复核者（独立上下文 subagent，只读）复核
  WP1 代码 + 协议 + 数字 + `project.yaml` 改动 → 有缺陷 ⇒ 派 writer 修 → **新的** fresh 复核 → 至无问题。
- 复核报告由**复核者直接写进仓库** `.scratch/62-axis1-time/REVIEW-*.md`（不留在 artifact 临时目录）。
- 返工 ≤2 轮；2 轮仍未 PASS ⇒ 如实登记 blocker 并返回。
- 复核记录见 §8。

## 8 复核记录

### 8.1 round 1（派发 `2026-10-13T15:33+08:00`；复核件落盘 `2026-10-13T15:41+08:00`）

- 复核者：`reviewer`（fresh 独立上下文，非本项 writer）；复核件 = `.scratch/62-axis1-time/REVIEW-1.md`（复核者直接写入仓库）。
- **能力限制（复核者自述，必须登记）**：该 run **无 shell** ⇒ `summary.py --check`/`pytest`/`sha256sum`/
  `git show` **全部未运行**；且 revision 前移后 `watchdog_diff` 拒绝服务 ⇒ `a679170 → ab41bb9` 的
  提交区间它**完全没看过**，对本报告的陈述依赖「逐行重读内容一致」。
- **判词**：判词 1（支配）/2（gap-closing）/3（必要条件）/4（被否证归因）的**事实与算术独立复算后
  成立**（10 臂逐数字、支配关系、下包络、gap-closing 全部对平）；并**额外独立证明主结论不依赖
  事后追加臂**（只用预登记点集 win16→win24 弦插值到 k_eff=17.333 得 1.623401 > 1.624979）。
- **总判词：FAIL / 科学结论 PASS**；blocking 4 条（B1–B4，均为文字与登记层面，**不需重跑 GPU**）：
  B1 判据③ 判定线方向文字与自身公式相反；B2 `analysis.md` 引数 0.006157 无出处且应为 0.005026；
  B3 契约臂集/预算/引用路径未同步、`win17/18` 追加未登记；B4 交付维度未达标却写成「齐备/已完成」。
  non-blocking：N1–N10。

### 8.2 round 1 处置（`2026-10-13T15:44–15:48`，actor：axis1-lead）

| 编号 | 处置 | 落点 |
|---|---|---|
| B1 | 判据③ 判定线改为「`>0` ⇒ 改善；`≈0` ⇒ 不可排序；`<0` ⇒ 被支配」，与公式及 SCHEMA §17.3 一致 | `README.md` §4、`analysis.md` §1、`test_2638_frontier.py`（新增符号钉子） |
| B2 | `0.006157` → **0.005026**（carry16_s0 对 win18_s0） | `analysis.md` §3 |
| B3 | 补 `win17/18` 行 + 追加时点/理由；预算改实测 ≈48 min 并记账；§7「8 臂」→「10 臂」 | `README.md` §5/§6/§7、`analysis.md` §2（预登记点集加固句） |
| B4 | `WP1.acceptance` 改陈述交付事实；`conclusion` 拆分「假设被否证」与「milestone 交付范围被裁剪」；`failure_analysis` 加 ④；`gaps` 首条记交付未达标 | `project.yaml` |
| N1 | `results.json` 补 `liveness` 块（逐臂掩码活性 + 训练日志 window_mask + carry liveness） | `run.py`（并重跑） |
| N2 | 非退化改**硬断言**；加评测期反事实探针 `window_mask_probe(win17)` | `run.py` |
| N3 | 状态精度假设（fp16 vs 引擎 fp32）显式记账 | `README.md` §4a、`analysis.md` §3 脚注 |
| N4 | `smoke/steps300/win24/…` → `smoke/steps300/smk24/…` | `README.md` §6 |
| N5 | 区间「0.0009–0.0025」→「0.000875 / 0.002259 / 0.003920」 | `analysis.md` §3 |
| N6 | 标题句加「点估计支配；σ_cell 边缘、0.014 带不显著」限定 | `analysis.md` §0、**`project.yaml` `conclusion_review`**（+ §5.3 本文件） |
| N7 | `conclusion` 补 `T=256`；`conclusion_review` 回填复核判词 | `project.yaml` |
| N8 | 新增实验目录内自包含单测 `test_2638_frontier.py`（**不改共享 `tests/`**，边界纪律） | 实验目录 |
| N9 | 明示复用臂范围 = **eval-only**，等值性由「4 复用臂 bpc 逐位复现」佐证 | `README.md` §5（§3 未加此句——R2-N6 指出，落点已订正） |
| N10 | 复核者请求 supervisor 跑的检查由 axis1-lead 代跑并登记 | §8.3 |

### 8.3 复核者请求的检查（由 axis1-lead 代跑，`2026-10-13T15:50`）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `git show --stat ab41bb9` | 只改 `axes/1-time/1.1.4.1-explicit-memory/project.yaml`（+23/−14）⇒ 消除了复核者「提交区间未看过」的残差风险 #1 的**上半** |
| 2 | `git diff a679170 ab41bb9 --stat` | 同：唯一文件 = `project.yaml`；实验产物未变 ✓ |
| 3 | `git status --short` | round-1 修复后的工作树（见提交）⇒ 解释了复核者残差 #1 的下半 |
| 4 | `sha256sum <10 checkpoints>` | 10/10 前缀与 `MODEL.md` / `config.json` / `results.json.checkpoints` **逐位一致**（c17a1cf1/70f0acb9/39316ac0/ebd4ef84/dccc58c0/d9211482/b3d3cdbc/c3f6f08a/2bfe167c/86953dcb）⇒ 消除残差 #3 |
| 5 | `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| 6 | `python3 -m pytest tests/ -q` | `834 passed`（本轮**未改共享文件**，包括 `tests/`） |
| 7 | `python3 -m pytest .../2638-…/test_2638_frontier.py -q` | `5 passed`（新增自包含单测，钉住符号约定 B1） |

### 8.4 round 2（派发 `2026-10-13T15:54+08:00`；复核件落盘 15:56）

- 复核者：**新的** fresh 上下文（非 round-1 复核者、非本项 writer）；复核件 = `.scratch/62-axis1-time/REVIEW-2.md`。
- **能力限制**：无 shell ⇒ `summary.py --check`/`pytest`/`sha256sum` 仍全部未跑；`ab41bb9 → 12e5fb7`
  提交区间亦未独立看过（只有内容层逐行重读）。
- **判词**：round-1 的 **B1–B4 逐条独立复算后判「真修、方向自洽」**；科学结论逐项独立复算**仍成立**
  （并再次独立复现「主结论不依赖追加臂」）。**总判词 = FAIL（窄范围）**：剩 1 blocking（R2-B1：
  `MODEL.md` 与 `results.json` 对同一个 `run.py` 给出不同 sha256）+ 8 non-blocking（R2-B2/N1–N8）。
- **处置（round-2 返工）**：
  - R2-B1：`sha256sum run.py` = `2023f36d…`（= `results.json.code_hashes.run.py`，即**落盘 results.json
    确由交付的 run.py 产生**）；`MODEL.md:41` 的 `eff58d515…` 是**陈旧值**（它写于 run.py 被 N1/N2 修改之前）
    ⇒ 已按实际值重生成 `MODEL.md` 的 `run.py`/`config.json` 指纹。
  - R2-B2：`README.md` §7 训练清单补为「默认 ARMS="24 32 128"；本实验另按 ARMS="17 18 24 32 128" 训练 5 臂」。
  - R2-N1：删掉 `README.md` §4 与 `analysis.md` 首段的「复核修正记录」（事实文档只写当前状态；
    历史归本 ticket 与 commit message）。
  - R2-N2：`project.yaml` 的 2638 证据锚点前移到本轮落盘 commit（见 §8.6）。
  - R2-N3：`WP1.means` 加半句「（本轮只交付质量轴 + 窗口点 + full 锚；召回轴与量化点见 gaps 首条）」。
  - R2-N4：`analysis.md` 判据④ 栏把 seed1 的 `+0.001470` 标注为**跨 seed** 口径，并列出同 seed 配对口径 `+0.002453`。
  - R2-N5：`test_2638_frontier.py` 的弱子串断言换成**实质断言**「`rel < 0 ⇔ 被支配` 与 `results.json`
    的 `frontier.dominated_by` 逐臂一致」（方向写反即红；6 条单测）；文件**仍放实验目录**（本轴边界
    不允许写共享 `tests/`——已在报告中登记为边界取舍）。
  - R2-N6：本文件 §8.2 的 N6/N9 落点已订正（见上两行）。
  - R2-N7：`README.md` 训练跨度改 8.4–8.53 min；评测墙钟改 0.30 min（`results.json.cost` 现为 0.3）。
  - R2-N8：复用等值性已在 `README.md` §5 明示限 **eval 口径**；`analysis.md` §3 未加此句，故本项
    以 README 为准（R2-N6 已订正落点）。

### 8.5 复核者请求的检查（axis1-lead 代跑，round 2）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `sha256sum .../2638/run.py` | `2023f36d1a9694c8579bfab76c76da5e5b57ab215cfdba46f52f992164615778` = `results.json.code_hashes.run.py` ⇒ R2-B1 属「MODEL.md 陈旧值」分支，无需重跑 |
| 2 | `python3 -m pytest .../test_2638_frontier.py -q` | `6 passed` |
| 3 | `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| 4 | `python3 -m pytest tests/ -q` | `834 passed`（共享底座未改） |

### 8.6 round 3（派发 `2026-10-13T16:00+08:00`；复核件落盘 16:02）

- 复核者：**新的** fresh 上下文（非 round-1/2 复核者、非本项 writer）；复核件 = `.scratch/62-axis1-time/REVIEW-3.md`；
  复核对象 = `940cb23`。
- **能力限制**：无 shell ⇒ `summary.py --check`/`pytest`/`sha256sum`/`git diff` 仍未跑（已由 axis1-lead 代跑，见 §8.7）。
- **判词**：R2-B1 **已消解**（`MODEL.md` 与 `results.json` 对同一 `run.py` 现同为 `2023f36d…`；`MODEL.md`
  全部 16 个哈希逐条扫过无第二处陈旧值）；R2-B2 与 N1/N2/N3/N4/N6/N7 逐条落地；**数值零回归**
  （`rows`/`frontier`/`location` 与 round-2 记录值、与 `run_log.txt` 逐位相同）；科学结论抽查通过。
  **总判词 = FAIL（窄范围）**：1 blocking **R3-B1**（`project.yaml` 的 `conclusion_review` 写「REVIEW-2 待派」
  而同一文件 `change_log` 写「已按 REVIEW-2 修」——同一 SSOT 内两条互相打架的陈述）+ 5 non-blocking
  （R3-N1 单测条数 5→6、R3-N2 悬空指针、R3-N3 §4a 曲线定义、R3-N4 单测钉子范围、R3-N5 重复行 + R3-N6
  `change_log` 时间倒序）。
- **处置（按 orchestrator 收口指令：修 blocking 即提交、不再开新一轮复核）**：
  - R3-B1：`conclusion_review` 回填 REVIEW-2/REVIEW-3 判词，删去「REVIEW-2 待派」。
  - R3-N1/N2/N3：`README.md` 单测条数改 6 并补主题；悬空指针改指 `results.json` + `2634/analysis.md`；
    §4a 纯窗口曲线补 `win17/18` 与预登记 4 点集的关系。
  - R3-N4：`test_2638_frontier.py` docstring 收窄钉子范围（只钉 rel 公式 ↔ 判定线的**数值**一致，
    明说抓不住文档文字型回归）。
  - R3-N5/N6：`analysis.md` 去重复行；`project.yaml` `change_log` 改时间正序。

### 8.7 round-3 代跑检查（axis1-lead，`2026-10-13T16:02–16:05`）

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| `python3 -m pytest .../test_2638_frontier.py -q` | `6 passed` |
| `python3 -m pytest tests/ -q` | `834 passed`（共享底座未改） |
| `sha256sum` 抽 2 个 checkpoint + `MODEL.md` 全部 16 哈希 vs 磁盘 | 全部一致（10 权重 + `run.py`/`config.json`） |
| `results.json.code_hashes` vs 磁盘 | 全部一致 |
