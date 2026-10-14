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

### 8.8 round 4（终审）与收口清理（`2026-10-13T16:17:46+08:00`，actor：axis1-lead）

- 复核者：**新的** fresh 上下文（本轮具备 shell）；复核件 = `.scratch/62-axis1-time/REVIEW-4.md`；复核对象 = `54afd31`（= `69d431b` + 锚点前移）。
- **判词**：**BLOCK（窄范围）**。科学结论 PASS、数值零回归（`gap=0.125421`、`win17` 严格支配 `carry16_s0`：208,896 B ≤ 212,992 B 且 1.624104 < 1.624979、下包络、`carry_minus_interp_window`、recovery 2.5%/1.2% 逐项独立复算一致）；无回归（`summary.py --check` OK、`pytest tests/` 834 passed、自包含单测 6 passed、工作树干净、无暂存）。窄范围 blocker 全属纯文字/指针：R3-B1 三步修法只做了第 1 步（`project.yaml` `conclusion_review` 里重复/过时的独立复核句未删、`WP1.review` 只列 `REVIEW-1.md` 漏 REVIEW-2/3）+ R3-N1 兄弟处（`test_2638_frontier.py` module docstring 仍写「钉住三件事」，实际 6 条）未同步。
- **Main 裁定（原文）**：「REVIEW-4 的 R3-B1 剩余 2 步 + N1 兄弟处，裁量为非关键记账，收口后清理，不做新一轮终审」。
- **本轮处置（收口后一次性清掉这 3 处文字/指针；不动实现代码、不动测量数字）**：
  - `project.yaml` `conclusion_review`：删除那段重复且过时的「独立复核（`REVIEW-1.md`）：见该文件判词。复核前未改本 `project.yaml` 的判据或数字；若有缺陷按 SCHEMA §14/§14b 返工。」（与 `940cb23` 逐字相同）；保留已写全 REVIEW-1/2/3 判词的那段（并补回其闭合引号）。
  - `project.yaml` `WP1.review`：由只列 `REVIEW-1.md` 改为列全本 WP 全部轮次 `REVIEW-1.md`/`REVIEW-2.md`/`REVIEW-3.md`（WP2/WP3 尚未开始、零轮复核件，维持 `fresh 复核` 原样）。
  - `test_2638_frontier.py` module docstring：`钉住三件事（复核 N8 / B1）` → `钉住以下事项（复核 N8 / B1）`（不写死数量；实际 6 个 `test_*`）。
  - `.scratch/62-axis1-time/REVIEW-4.md` 复制入仓（与 REVIEW-1/2/3 落盘模式一致）。
- **流程规则（本轮反复账实不符的根因，登记为纪律）**：`conclusion_review` **只写当前（最后一轮）判词**；**历轮判词写本 ticket**（§8.1/8.4/8.6/8.8）；**不得**在同一字段累积多轮判词。本条为**自本轮起**的纪律；本项 `conclusion_review` 现有那段（已写全 R1/R2/R3）按 Main 裁定保留、不再改，自下一次修订起该字段只留最后一轮判词。

---

## 9 Phase B（第二轮）：用**完整机制**重测 1.1.4.1（E1 通道已就绪）

> 触发：`.scratch/67-engine-changes/` 的 **E1**（`CarryChannel` 逐层独立状态 + 内容相关写入门 +
> 可选衰减 γ，附加式、默认关闭、默认逐位不变；独立复核 **PASS** `REVIEW-E1.md`）。
> 上一轮 2638 的否证归因是「单共享槽 = 实现退化，完整机制未测」；本轮**消掉这个借口**。

### 9.1 读入的权威输入（`2026-10-14T07:28:58+08:00`，actor：axis1-lead-2）

| 路径 | 读到的关键事实 |
|---|---|
| `.scratch/67-engine-changes/TICKET.md` / `E1-CARRYCHANNEL.md`（工件目录）/ `REVIEW-E1.md`（工件目录） | E1 新增 `DepthARConfig.carry_per_layer=False` / `carry_decay=1.0`；逐层模式下每层在**本层当前 hidden** 上写自己的 8192-float 状态（写门 β/键/值随之）；γ 走 `S←γS+write`（分块实现在 fp64 下与逐位置参考对到 <1e-12）；默认路径逐位不变；全量 `849 passed`；独立复核 PASS（含 4 处变异均 RED、默认逐位不变跨 3 树 × 多配置） |
| `axes/1-time/1.1.4.1-explicit-memory/project.yaml` | `status: 被否证`；否证点 = `carry16` 被 `win17` 严格支配（212,992 B vs 208,896 B，1.624979 vs 1.624104）；`failure_analysis` ①实现退化 ②结构 ③非 innovation 失败 ④交付裁剪 |
| `.scratch/62-axis1-time/TICKET.md` §4 F5 / §5–§8 | F5 = 「需要改共享引擎」的最小 patch 申请，**本轴不执行**；本轮 E1 已由引擎线完成 ⇒ F5 **关闭**（见 9.8） |
| `2638/results.json` | 复用臂基线与前沿：`win16_s0 1.628110` / `win17_s0 1.624104` / `win18_s0 1.619953` / `win24_s0 1.599856` / `win32_s0 1.595823` / `win128_s0 1.513325` / `full 1.502689` / `carry16_s0 1.624979`（k_eff 17.33） |

### 9.2 契约的**执行前**修订（`2026-10-14T07:35–07:39`，actor：axis1-lead-2）

预登记契约 = `axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/README.md`
（判据线 §4 与 §4a 在**任何训练启动之前**写死；下列修订同样发生在训练启动之前）：

1. **`carry_chunk` 记账**：E1 文档把 chunk 定义为「只改浮点累加顺序、不改语义」（复核在 fp64 下
   对拍到 <1e-12）。300/600 步冒烟实测（`07:30:40–07:39:26`）：`chunk=64` **350 ms/step**、
   `chunk=256` **153 ms/step**（2.3×）⇒ 新臂一律 `chunk=256`；并**另训一个 chunk=256 的单共享槽
   对照臂**（`c16c256_s0`），既给「完备 vs 退化」的 chunk 匹配对照、也充当 chunk 敏感性探针。
2. **臂集与砍序**：主臂 `pl16`（完备机制，γ=1）不可砍；其次 `c16`（9 min）；`pl16d`（γ=0.99）
   若不能在 `08:30` 前启动则**不跑**并列入 gaps；**种子臂本轮不跑**（臂内噪声由 σ_cell 覆盖，
   列入 gaps）。
3. **主判据的结构变化**：`carrypl16` 的每位置等效字节 = 196,608（窗口 16）+ 196,608
   （12 层 × 8192 floats fp16）= **393,216 B = `win32_s0` 的字节数** ⇒ 主判据是**等预算点的直接点比**
   （`pl_minus_win32 = bpc(win32) − bpc(carrypl)`），**不依赖任何插值**。

### 9.3 GPU 锁与冒烟（`2026-10-14T07:30–07:39`，actor：axis1-lead-2）

- `07:28:58` `nvidia-smi` ⇒ 676 MiB / 24564 MiB，util 0%；`/tmp/pathlm_gpu.lock` **不存在**
  （其它 `*.lock` 属别线）⇒ 本轴按 `SCHEMA.md` §20 取 `pathlm_gpu.lock`。
- 冒烟 4 次（每次都**取锁→跑→放锁**，逐条 rc=0）：`smk_pl16`（300 步 chunk64，118 s）、
  `smk_pl16_c256`（300 步 chunk256，48 s）、`smk600_c64`（600 步，223 s）、`smk600_c256`（94 s）。
  步时：**c64 350 ms/step、c256 153 ms/step**；峰值显存 4.87 GB（c64）/ 9.44 GB（c256）。
  冒烟的 `carry_liveness` 已证开关真的开：`per_layer=true`、`shared_slots=false`、
  `n_layers_sampled=12`、`alive=true`。

### 9.4 训练执行（`2026-10-14T07:40:07` 起，actor：axis1-lead-2）

`nohup env ARMS="pl16 c16 pl16d" bash train_arms.sh > train_driver.log &`，逐臂取/放锁、臂间 61 s。
见 §9.6 的实测落地记录。

### 9.5 执行前的两处补充登记（`2026-10-14T07:42–07:44`，均在**评测跑起来之前**）

1. **`run.py` 的 CPU dry-run 验证**（`07:42:00–07:43:25`）：用只含 6 个**复用臂**的临时 config
   （`/tmp/2639_cfg_cpu.json`，`device=cpu`，2×2 批）把评测链路端到端跑通
   （sha256 断言 → 掩码自检 → bpc → 前沿 → location → liveness → 复用等值块 → 落盘），
   确认分节与键名无 bug ⇒ 正式评测只需换 config。dry-run 产物落在 `/tmp/2639_eval_check/`
   （**不进仓库**，只作过程证据；其 bpc 是 2×2 批的粗口径，不可引用）。
2. **敏感性口径（`sensitivity`）执行前登记**：状态的固定成本与 T 无关、窗口 KV 随 T 线性增长，
   冻结口径（62-F2 / 2638）= 状态整块计作等效字节（同 T 同 footprint）。本轮另报**摊平口径**
   （`k_eff_amort = (kv + state/T)/12288`）作**敏感性**，**不构成判据**；两口径的读数在
   `results.json.sensitivity` 与 `analysis.md` 并列。真实长上下文（T≫256）未测 ⇒ 进 `gaps`。
   （契约文本落 `2639/README.md` §4a 末条，时间戳同上。）

### 9.6 训练与评测实测（`2026-10-14T08:52:27–08:58:55`，actor：axis1-lead-2）

| 臂 | 取锁 | 放锁 | 墙钟 | 结果 |
|---|---|---|---|---|
| `pl16`（完备机制 γ=1） | 07:40:07 | 08:08:45 | 28.56 min | rc=0；`carry_liveness.alive=true`、`per_layer=true`、`shared_slots=false`、`carry_chunk=256`；训练 CE 1.2389 |
| `c16`（单共享槽 c256） | 08:09:47 | 08:20:39 | 10.77 min | rc=0；`shared_slots=true`、`chunk=256`；训练 CE 1.2902 |
| `pl16d`（完备机制 γ=0.99） | 08:21:40 | 08:52:27 | 30.7 min | rc=0；训练 CE（末段）见 `train/pl16d/results.json` |

- `08:54:59–08:55:24` `make_config.py --require-new` ⇒ 13 臂 config（复用臂 sha256 逐个与 2638 config 核对通过）+ `run.py` 评测（0.49 min、峰值 1019 MB、rc=0）。
- `08:56:23–08:56:54` **给评测补两个反否证探针后重跑**（见 §9.7）：结构因果探针 + 换评测窗口序鲁棒性。
- `08:58:23–08:58:55` 因修正 `make_config.py` 的 `reused_from` 标签（把 2634 来源的 4 个臂标错成 2638）重生成 config 并**第三次**跑评测 ⇒ 结构化数值逐位复现（`+0.068228` 等全部相同）。

### 9.7 主结果与两读法（`2026-10-14T08:55`）

**完备机制在等字节点上翻转了上一轮的否证**：`carrypl16`（窗口 16 + 12×8192 floats 状态 =
393,216 B = `win32` 的字节数）clean bpc **1.527595** vs `win32` **1.595823** ⇒
`pl_minus_win32 = +0.068228`（41×σ_cell=0.00166、4.9×0.014 带）；下包络 =
`win16 → win17 → win18 → win24 → win32 → **carrypl16** → win128 → full`；补回
「窗口 16 → 全量」差距（0.125421）的 **80.1%**（单共享槽实现只有 2.5%）；完备度本身
= `carry16_s0 (1.624979) − carrypl16 (1.527595) = 0.0974 bpc`（chunk 匹配口径 0.1013）。
`pl16d`（γ=0.99）1.585531：仍优于等字节窗口（+0.010292）但比 γ=1 差 0.0579。

**看到第一轮评测后的两个补充探针（如实登记：判据/臂集/数字均未改，只加反否证检查）**：
① **结构因果探针**（`objective_criteria` 的「必须不变」项）：只改后半段 token ⇒ 前半段
logits **逐位不变**，12/12 掩码臂 `max_abs_logit_diff_prefix = 0.0`，且探针有抓力
（后半段差 21.6–26.7）⇒ 完备机制**无未来泄漏**；
② **换评测窗口序**（`window_seed=7`）：`pl_minus_win32 = +0.071912`（同号、同量级）
⇒ 不是窗口序运气。
两探针均**加强**而非软化判定，且落在 `results.json` 的 `liveness.*.causality` 与
`robustness_alt_window_seed`。

### 9.8 上报 Main 的框架性取舍（本轮新增）

| # | 取舍 | 我的处置 | 状态 |
|---|---|---|---|
| F6 | **状态的「每位置等效字节」口径**：冻结口径（62-F2/2638）= 状态整块计作等效字节（同 T 同 footprint）；另一口径 = 状态是固定成本、按 T 摊平（`k_eff_amort`）⇒ 完备臂 16.06 而非 32.00。两口径回答不同问题（「同上下文长度同内存」vs「长上下文下摊平」） | 判据**一律**用冻结口径；摊平口径在 `results.json.sensitivity` 与 `analysis.md` §3 **并列报数**（+0.100265，同号）⇒ 主结论不依赖该口径选择 | 记录待裁（不改判据） |
| F7 | **1.1.4.1 的 `status` 标法**：主问题已有明确结论（完备机制在前沿上占优），但 milestone 字面验收（每预算点两轴 + 与业界实现对齐）仍未交付 ⇒ 记 `进行中`（schema §7：「做到多少」写 conclusion/gaps，状态只回答「做完没有」），不记 `已完成`（避免对 milestone 过度声明）、更不记 `被否证` | `status: 进行中` + `failure_analysis: ""`（schema：未判失败时写空）+ `conclusion` 里两读法并列 | 待 Main 裁 |
| F8 | **E1 的两半绑在一个开关**：`carry_per_layer` 同时打开「逐层独立状态」与「写门/键/值取自本层当前 hidden」⇒ 0.0974 bpc 的增益不能归因到哪一半 | 不擅自改共享引擎加开关（边界纪律）；记入 `gaps` 首条 + `analysis.md` §5/§7 | 待补实验 |

### 9.9 复核计划（本轮）

writer（axis1-lead-2）停止写入 ⇒ 派 **fresh、独立上下文** 复核者（只读，具备 shell），
复核对象 = 提交 `ccdc91a`（2639 实验全套）+ 其后 `project.yaml`/本 ticket 的改动；
复核件由复核者**直接写进** `.scratch/62-axis1-time/REVIEW-5.md`。返工 ≤2 轮。

### 9.10 复核记录 round 1（派发 `2026-10-14T09:01:30`（`c77cc0b` 之后）；报告返回并入库 ≤`2026-10-14T09:07:00`）

- 复核者：`reviewer`（fresh、独立上下文、**非本项 writer**）；复核件 = `.scratch/62-axis1-time/REVIEW-5.md`
  （复核者无写文件工具，报告以最终回复交付，由 axis1-lead-2 **原样复制**入仓；sha256 =
  `f29a427e3313375e46c08ff8e7a1473e73efaa7f5633a07786fa00de43028d01`，与工件目录源逐位一致）。
- **能力限制（复核者自述，必须登记）**：该 run **无 shell/执行工具** ⇒ `sha256sum` / `pytest` /
  `summary.py --check` / `git diff` / GPU 重跑**全部未运行**；它只能读盘 + 手工复算 +
  `watchdog_diff`（且该工具不含已提交区间）。
- **判词**：**FAIL（窄范围）** —— **科学主张与数值证据 = PASS**（14 组数字独立复算逐位一致：主判据
  `+0.068228`、`r=0.80143`、完备度 `0.097384`、chunk 匹配 `0.101345`、chunk 敏感性 `0.003961`、
  换窗口序 `0.071912`、41.1×σ_cell、4.87×0.014、γ 残留 0.99²⁵⁶=7.63%、窗 111.4 等；
  状态字节记账 `98304 floats × 2 = 196,608` 与 `run.py` 从**加载后的模型**读数一致；
  10/10 复用臂与 2638 逐位相同；口径未被软化；12/12 掩码臂 `max_abs_logit_diff_prefix = 0.0`）；
  **唯一 blocking = B1（记录 ≠ 文件）**：`project.yaml` 的 `experiments:` 列表**没有** 2639，
  而同一文件的 `change_log`（`2026-10-14T09:10:00`）与该提交信息都写「experiments 新增 2639」。
- **复核者独立发现（本轮采纳）**：`carry16_s0` 与 `carrypl16_s0` 的**参数量完全相同**
  （均为 13,801,064）⇒「退化实现压掉 0.0974 bpc」是**等参数对照**，不能被「参数更多」解释。
- **三个要否证的关键点，复核者的结论**：① 等字节点比较**公平**（且按 fp32 记状态时仍占优）；
  ② `win32` 在 393,216 B 预算内确为最强纯窗口基线，`carrypl16` 确在包络上；
  ③ chunk 敏感性 0.003961 **不削减**主判据（最坏方向仍剩 0.064267 = 38.7×σ_cell，而对照臂
  `win32` 不含 carry 路径、chunk 与它无关）。
- **non-blocking**：P2-1（`analysis.md` §8 成本表两处记账）、P2-2（`results.json` 含裸 `NaN`、
  非严格 JSON）、P2-3（`project.yaml` 的「同算力」vs 判据体「同字节」术语不齐，属 C2 落地时的历史文本）。

### 9.11 round-1 处置（actor：axis1-lead-2，`2026-10-14T09:06–09:10`）

| 编号 | 处置 | 落点 |
|---|---|---|
| **B1** | `experiments:` 补两行 `2639-carry-perlayer-frontier-13m/analysis.md @ccdc91a`、`…/results.json @ccdc91a`（现 29 条） | `project.yaml` |
| P2-1 | `pl16d` 墙钟 `30.0 → 30.7` min（= 臂内 `wall_minutes`）；评测墙钟按最新落盘值 0.49 min；合计 ≈75.5 min | `analysis.md` §8、本文件 §9.6 |
| P2-2 | `run.py` 落盘边界加 `_strict()`（非有限浮点 → `null`）+ `allow_nan=False`；重跑评测 ⇒ `results.json` 现为**严格 JSON**（`pl16d` 的 `rel_dev_vs_other_rule = null`）；**共享代码未改**（源头 `train/pl16d/results.json` 的 `NaN` 是共享 runner 在 γ≠1 时的遥测产物，已在 `analysis.md` §8 记账） | `run.py`、`results.json`、`MODEL.md`（run.py 指纹 `62690cc2… → f04d59da…`）、`analysis.md` §8 |
 P2-3 | **不自行改**：该措辞来自跨轴裁决 C2 的落地文本（`project.yaml` 的 `milestone.acceptance`），本轴不改裁决用词；登记为框架项 **F9**（见 §9.14）待 Main 裁 | §9.14 |

### 9.12 round-1 复核者请求的代跑检查（axis1-lead-2 代跑，`2026-10-14T09:08–09:10`）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `sha256sum train/{pl16,c16,pl16d}/model.pt config.json` | 4/4 与 `config.json` / `results.json.checkpoints` / `MODEL.md` 逐位一致（`382a74aa…`/`3e935b33…`/`cb1940ee…`/`28ba9485…`） |
| 2 | `pytest …/test_2639_frontier.py -q` | `6 passed` |
| 3 | `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| 4 | `python3 tools/check_tree.py` | 同上 `OK` |
| 5 | `git diff --cached --name-only \| wc -l` | `0` |
| 6 | `git diff d3d33eb..HEAD --stat` | 38 文件；**越界文件数 = 0**（脚本过滤：只允许 2639 实验目录 / `1.1.4.1/project.yaml` / `.scratch/62-axis1-time/**` / 两份 E1 报告） |
| 7 | 评测重跑（取/放 §20 锁，`run.py`） | rc=0；`pl_minus_win32` 与 `pl_minus_win32_alt_seed` 逐位复现（`0.068228` / `0.071912`） |

### 9.13 round 2（派发与判词）

见本节之后的追加记录。

### 9.13 round 2（派发 ≈`2026-10-14T09:09:30`，在 `6f42ed5`（`09:08:50`）之后；报告返回 ≤`2026-10-14T09:12`）

- 复核者：**新的** fresh 上下文（非 round-1 复核者、非本项 writer）；复核件 = `.scratch/62-axis1-time/REVIEW-6.md`
  （复核者无写文件工具 ⇒ 报告以最终回复交付，由 axis1-lead-2 **原样**（逐字节）复制入仓；sha256 =
  `adbb088ed9caa74895fdd6918c46d05bb49f9bbe6066bcf9cfea7339aa9e7009`）。
- **判词：PASS（带 notes），blocking = 0**。复核者独立核到：B1 真修（`experiments` 29 条，**逐条**路径存在性全过）、
  P2-1/P2-2 真修（`results.json` 无 `NaN`/`Infinity` 字面量、`rel_dev_vs_other_rule = null`）、
  **返工零数值回归**（把现盘值与 round-1 在 `c77cc0b` 时读到的值逐位对拍）、主判据独立复算复现
  （`0.068228`、同字节、`state_floats=98304`、`r=0.801429`），并**枚举全部 13 臂**回答关键点：
  「**不存在任何 `total_bytes ≤ 393,216` 且 bpc < 1.527595 的臂**」⇒ 主结论未被支配。
- 能力限制（复核者自述）：**无 shell** ⇒ sha256/pytest/summary/严格解析/git 区间 diff/GPU 重跑未跑；
  §9.12 的代跑表它只读自洽性、**未据其放行**（正确态度）。
- **non-blocking（round 2，4 条）与处置**：

| # | 内容 | 处置 |
|---|---|---|
| P2-A | `analysis.md` §8 把峰值显存写成「9.44 GB（`pl16`/`pl16d`）」，实际 `pl16d = 10.28 GB` | **已修**：改「9.44 GB（`pl16`）、10.28 GB（`pl16d`）」（证据：`results.json.liveness.*.train_peak_gpu_mem_mb`） |
| P2-B | 本文件 §9.11 的 P2-3 行写「登记在 §9.8 的框架项中待裁」，而 §9.8 只有 F6/F7/F8（**声称 > 实际**，同形态第 6 次） | **已修**：§9.8 新增 **F9**（见下），并把 P2-3 行的落点改指 F9 |
| P2-C | 两处「**可能**在返工中被静默改写」：① `change_log` 末条日期 `09:10:00` → `09:00:00`；② `objective_criteria` 的「同算力」措辞 | ① **属实且我未在提交信息里披露** ⇒ 此处显式披露：我在写 `change_log` 时**先写了 09:10 再看时钟**（实际编辑发生在 `≤09:01`，由 `c77cc0b` 的提交时刻 `09:01:24` 与两次 `date -Iseconds` 佐证）⇒ 返工时把它**更正为 `09:00:00`**（时间戳属溯源元数据，写未来时刻是错的）；这是我的过程失误（应先取时钟再记），记录在此。② **round-1 复核的引用有误**（不是事实文档被改）：`git show c77cc0b:…/project.yaml` 与 `git show d3d33eb:…/project.yaml` 的 `objective_criteria`（`:14`）**都不含**「同算力」；该措辞只在 `milestone.acceptance`（现盘 `:75`，C2 裁决落地文本）。证据：`git show c77cc0b:axes/1-time/1.1.4.1-explicit-memory/project.yaml \| grep -n objective_criteria` ⇒ 「…在同一每位置等效字节上不被同预算纯窗口支配」（ticket 62 引入时即为此措辞） |
| P2-D | 源头 `train/pl16d/results.json` 仍是裸 `NaN`（共享 runner 在 γ≠1 时的遥测），严格 JSON 工具解析失败 | **不改**（跨轴共享代码，需授权）；已在 `analysis.md` §8 与本文件记账；若要收口 = 共享 runner 对该字段返回 `None`（另立小修项） |

### 9.14 F9（框架项，新增）：`milestone.acceptance` 的「同算力」措辞

| # | 取舍 | 我的处置 | 状态 |
|---|---|---|---|
| F9 | `project.yaml` 的 `milestone.acceptance` 写着「不被**同算力**最强基线支配（相对**同算力**纯窗口前沿超出量 ≤ 0）」，而判据体（`method` / `objective_criteria` / `README.md` §4③）一律是**每位置等效 KV 字节**（62-F2 裁决）。纯窗口与线性状态的每 token FLOPs 本来就不同 ⇒ 「同算力」是口径模糊的简写 | **不自行改**（该措辞来自跨轴裁决 C2 的落地文本，与本轴边界纪律一致）；只登记待 Main 裁：建议改为「同预算点（每位置等效字节）」 | 待裁（report-only） |

### 9.15 round-2 复核者请求的代跑检查（axis1-lead-2 代跑，`2026-10-14T09:12–09:13`，落在 `3ec06b0`（`09:13:22`）之前）

round-2 复核者的 §4「请父级代跑」清单 (a)–(d) 全部执行，命令与结果见下表。

| # | 命令（原文） | 结果 |
|---|---|---|
| a1 | `python3 -c 'json.load(results.json); print(key_readings.pl_minus_win32, liveness.carrypl16d_s0…rel_dev_vs_other_rule)'` | 严格解析 OK：`0.068228` / `None` |
| a2 | `jq -e . $EXP/results.json` | `jq-OK`（无裸 `NaN`，严格 JSON） |
| b | `sha256sum $EXP/run.py $EXP/config.json $EXP/train/{pl16,c16,pl16d}/model.pt` | `run.py f04d59da…` = `results.json.code_hashes.run.py` = `MODEL.md`；`config.json 28ba9485…` 同；3 个 checkpoint = `382a74aa…`/`3e935b33…`/`cb1940ee…`，全部与 `config.json`/`MODEL.md` 逐位一致 |
| c1 | `git diff --stat d3d33eb..HEAD` | 代跑时点 39 文件（= 36 实验 + 2 判决/返工 + `REVIEW-5.md`）；**越界文件数 = 0**（过滤脚本口径同 §9.12，另把 `.scratch/62-axis1-time/REVIEW-5.md`/`REVIEW-6.md` 计入允许集） |
| c2 | `git diff c77cc0b..HEAD -- project.yaml` | 只有两处：`experiments:` 补 2639 两锚点、`gaps` 里「同参数量对照」那一条澄清 —— **外加 P2-C① 的 `change_log` 日期更正**（当时未在提交信息披露，已在 §9.13 显式披露） |
| c3 | `git show c77cc0b:…project.yaml \| grep -n '同算力\|09:1\|09:0'` | 只命中 `change_log` 的 `09:10:00`（`:128`）——**`objective_criteria`/`milestone.acceptance` 在该版本均不含「同算力」**（现盘的「同算力」在 `milestone.acceptance`，C2 落地文本）⇒ round-1 的 P2-C② 引用有误 |
| c4 | `git show c77cc0b:…results.json \| jq .cost` | `wall_minutes_total_eval: 0.5`（返工前旧值，与 REVIEW-5 引数一致）⇒ 与 round-1 引数自洽 |
| d1 | `pytest …/test_2639_frontier.py -q` | `6 passed` |
| d2 | `python3 tools/summary.py --check` / `python3 tools/check_tree.py` | 均 `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |

---

## 10 Phase C（第三轮）：把本轴推到终局（`2026-10-14T20:51:51+08:00`，actor：axis1-lead-3）

> 触发：orchestrator 任务「轴 1 lead · 自主推进：把本轴推到终局」。目标 = 每个**非 Backlog** 研究项
> 达到 `status ∈ {已完成, 被否证}` + `conclusion` + 落盘 `experiments` + fresh 独立复核判词。
> 预算：本 run 硬上限 3h；并发子 Agent ≤2；GPU 按 `SCHEMA.md` §20 串行。

### 10.1 枚举与「milestone 差距 + 下一步最高价值 WP」（12 项，按价值排序）

| 序 | id | status | milestone 差距 | 下一步最高价值 WP | 类型/成本 |
|---|---|---|---|---|---|
| 1 | 1.1.4.1 | 进行中 | 质量轴 ✅（2639）；**召回轴未测**；量化点未叠图；归因不可分（F8，需引擎新开关）；n≥2 新臂未做 | **WP2：完备机制的召回轴**（复用 2639 checkpoint 纯推理） | 推理：6 臂 × 96 批 × T256（0 训练 token） |
| 2 | 1.1.4.5 | 进行中 | 缺「同任务/同预算/同仪器」正面对撞；缺「前缀可丢」端到端协议 | **WP1：implicit(win16) vs explicit(carrypl16) 正面对撞**（与 #1 同仪器、同批臂） | 与 #1 同一次会话（0 额外训练 token） |
| 3 | 1.1.1 | 进行中 | 主数字仍在 `.scratch`/旧 stub；**锚点臂无仓内自包含复算**；T 子线无 seed 复本 | **WP1：win16+anchors 臂自包含复算（2 seed）** + 复用 2634 needle 仪器复算「锚点通道召回」 | 训练：2 臂 × 98.3M token（+ 推理 5 臂） |
| 4 | 1.1.4.4 | 未开始 | 本仓零经验 | WP1：{stock, stock+检索} 最小判决（需新检索腿实现） | 训练：2 臂 × 98.3M token + 检索腿实现 |
| 5 | 1.1.4.2 | 进行中 | 三类压缩无自有实验（摘要槽/低秩/pooling 六臂） | WP1：同预算压缩质量 | 训练：6 臂 × 98.3M token |
| 6 | 1.2.1 | 未开始 | 零经验；需 dilated 掩码 | WP1：三臂对齐可见键数 | 训练：3 臂 × 98.3M token + 掩码实现 |
| 7 | 1.2.4 | 未开始 | 零经验；需合并实现 | WP1：{stock, 金字塔 r=2/4/8} | 训练：4 臂 × 98.3M token + 合并实现 |
| 8 | 1.2.3 | 进行中 | 长 T 是否回本未测；**与「按位置最近 k」同预算对照未做（只缺窗口臂，可复用 2614 的 topk 臂）** | WP1/WP2：在 2614 配置（8L/mlp4）上补 {win32, win64} 两臂 | 训练：2 臂 × 98.3M token（与 2614 同 token 预算） |
| 9 | 1.2.2 | 进行中 | 真块稀疏掩码未实装 | WP1 | 训练：≥4 臂 × 98.3M token + 稀疏核实现 |
| 10 | 1.1.2 | 进行中 | per-head slope + 长度外推未做 | WP1/WP2 | 训练：3 臂 × 2 seed × 98.3M token |
| 11 | 1.1.4.3 | 进行中 | 判据错配未修（固定 ρ 下重建质量） | WP1 | 训练：3 臂 × 98.3M token |
| 12 | 1.1.3 | **已完成**（见 §10.12） | 无自有实验（设计上只指针 1.1.1）；已按预登记走「若否」分支 | WP1（已完成） | 0 训练 token（纯复用 + 再分析） |

**本轮（Phase C）只执行 #1/#2（合一）与 #3**——理由是：①#1/#2 是本轴核心命题（explicit 是否值得）
且**纯推理、成本最低、判决力最高**；②#3 是全体前沿的**价格上界**且**不需要新引擎能力**（只用已有
`--train-anchors`）。其余项在 3h 预算内不足以完成「训练 + 复核」两轮，如实登记在 **§10.9** 的 blocker 表。

### 10.2 WP 契约（预登记，写在任何执行之前）

**WP-C1 · 2661**：`1.1.4.5` 目录下新实验 `2661-implicit-vs-explicit-recall-13m`（同时供 `1.1.4.1` 的召回轴）。

- what：用**已有 checkpoint**、**同一台仪器**（2634 的毁源敏感度探针 + needle v3 短语针 + clean bpc），
  在**同任务/同预算/同可见性**下正面比 {implicit = 纯窗口 16} vs {explicit = 完备机制逐层独立状态 + 内容写门}。
- goal：回答本轴核心命题「窗口丢掉的，能不能靠显式状态搬回来」——此前只在**退化实现**（单共享槽）上测过。
- means：fork `2634/run.py` + `realtext_probe.py` + `needle_stats.py` 到新目录，**只改臂集与签名**，
  探针/判据/掩码口径一字不改；臂 = {win16_s0, win16_s1, carry16_s0, carry16_s1, carrypl16_s0, carrypl16d_s0}。
- acceptance（判据，C1 带已声明）：①**该涨**：`carrypl16_s0` 的 beyond 桶 needle 召回相对**配对无针地板**
  在硬同索引子集上 CI（配对 bootstrap / Wilson 95%）**不含 0**；②**该跌**：`win16_*` 的 beyond 召回
  停在同掩码地板（负对照，须复现 2634 的读法）；③**该涨**：`carrypl16_s0` 的毁源敏感度
  `Δbpc_far > 0` 且 CI 不含 0，而 `win16_*` 的 `max_abs_logit_diff_on_eval_rows == 0`（前提①硬断言）；
  ④**必须不变**：只有后半段 token 改变时前半段 logits 逐位不变（因果性）；⑤质量轴与 2638/2639 记录值逐位复现。
- evidence：`axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/`。
- review：WP 停后派 **fresh** reviewer（非本 WP writer），独立复核至无问题，累计 ≤2 轮。

**WP-C2 · 2662**：`1.1.1` 目录下新实验 `2662-hard-window-anchor-13m`。

- what：`win16 + anchors`（锚点臂，2 seed）自包含复算 + 复用 2634 needle 仪器复算「锚点通道召回」。
- goal：把「锚点通道是真产物（needle 76.8%→93.3%）」从 `.scratch` 摘要升级为仓内可复核证据；
  同时给出 1.1.1 的**锚点价格**（同预算下 bpc 差）。
- means：`train_depth_ar.py --train-window 16 --train-anchors <s>`（已有 CLI，**不需新引擎能力**），
  其余 CLI 与 2634 逐字段相同（d256/12L/8H/mlp6/T256/12k 步/bs32/seed 0,1）；对照臂复用
  `2634/train/rt_none_s{0,1}`（纯窗 16）与 `2610` trunk（全因果锚）。
- acceptance：①锚点臂 relative to 同 seed 纯窗臂的 clean bpc 差可排序（越自测带 σ_cell=0.00166 才对上，
  0.014 带并列报）；②锚点臂的 beyond-window needle 召回相对同掩码地板有可排序提升（CI 不含 0）；
  ③必须不变：因果性 + 掩码非退化 + 复用臂 bpc 与 2634 逐位复现。
- evidence：`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/`。
- review：同 WP-C1 的复核纪律。

### 10.3 框架性取舍（本轮，按「轴建议默认处置」自行采用）

| # | 取舍 | 处置 | 记录 |
|---|---|---|---|
| F9 | `milestone.acceptance` 的「同算力」措辞 | **改为「同预算点（每位置等效字节）」**（`1.1.4.1` 的 `milestone.acceptance` 与 `conclusion` 同事实处一并改） | 本文件 + `change_log` |
| F10 | 2661 一个实验目录**同时**作为 `1.1.4.5`（正面对撞）与 `1.1.4.1`（召回轴）的证据 | 目录归 `1.1.4.5`（该项的 milestone 就是这场对撞），`1.1.4.1` 的 `experiments` 加**同一路径**作指针（先例：`1.1.1` 的 `experiments` 里已有 `.scratch/07-x1-x3/evidence` 这种跨目录指针） | 本文件 + 上报 |
| F12 | **`1.1.3` 的树级处置** | **已裁（Main，2026-10-14）：不并入、不删除**——保留为独立目录条目（`status: 已完成`），在 `conclusion`/`gaps` 记录「不是本仓独立旋钮 + 位置类子问题归属 `1.1.1`、内容类归属 `1.2.3`」并加交叉指针；**不改 `docs/**`、不移入 `archive/dropped/**`**。理由：`docs/structure.md` §12 是冻结 SSOT、目录完备性就是本项目的目的，且该发现（k 16→128 补回 91.5% 差距、窗外召回零变化）**改变判断**，按 `AGENTS.md` §5 不得删 | 落地于 `1.1.3/project.yaml` 的 `conclusion`/`argument`/`gaps`/`conclusion_review`/`change_log`（本 run 提交见 §10.21） | **已裁·已落地** |
| F11 | C1 判据带声明缺口（`1.1.1`/`1.1.3`/`1.1.4.2`/`1.2.4`） | 按 `docs/research-schema.md` §12.1 机械补齐声明（bpc 类走 0.014 配对带；召回等率类走配对 bootstrap / Wilson 95% CI，不跨量纲借用 0.014） | 本文件 §10.4 |

### 10.4 C1 回填与 F9 执行记录（`2026-10-14T20:51:51+08:00`，actor：axis1-lead-3）

- 机械动作：脚本 `/tmp/axis1_c1.py`（**精确子串匹配，未命中即非零退出**）。
  先枚举全部 12 项：`falsification` 缺「用哪个带」声明者 = **4**（`1.1.1` / `1.1.3` / `1.1.4.2` / `1.2.4`）；
  改后重扫 **12/12 全部声明**；命中清单 = 上述 4 个文件，未命中数 = **0**。
  （附带发现：`1.1.1` 在 `2026-10-14T06:59:49` 的上一轮 C1 回填的落点只有 `validation` 与 `conclusion` 两处（见 §10.14 的 `fb8a576` 复核），
  `falsification` 仍缺声明 ⇒ 本轮补足，已在 `change_log` 如实写明。）
- F9：`1.1.4.1` 的 `milestone.acceptance` 与 `conclusion` 里「同算力」→「同预算点（每位置等效字节）」（2 处，
  残留事实字段命中 0；仅 `change_log` 叙述里还提到旧措辞）。
- `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合 docs/research-schema.md`。
- 提交 `d4175cc`（6 文件，显式路径；`git diff --cached --name-only` 人工核对 6/6）。

### 10.5 已派发的 WP 与执行状态

| WP | 实验 | 派发时刻 | writer | 状态 |
|---|---|---|---|---|
| WP-C1 | `2661-implicit-vs-explicit-recall-13m`（1.1.4.5；兼供 1.1.4.1 召回轴） | `2026-10-14T20:52:00+08:00` | axis1-w-2661（fresh worker） | 执行中 |
| WP-C2 | `2662-hard-window-anchor-13m`（1.1.1） | `2026-10-14T20:52:45+08:00` | axis1-w-2662（fresh worker） | 执行中 |

- 两个 WP 串行共用 §20 GPU 锁：`mkdir /tmp/pathlm_gpu.lock` + `holder_pid` 协议、按臂/按跑取放锁、
  等锁 >20 min 退让并先做 CPU 工作。
- 复核：writer 全部停止后派 **fresh** reviewer（非任一 writer 上下文），一份复核件逐 WP 给判词
  （`SCHEMA.md` §14「一个复核者可覆盖多个包，但覆盖须显式、每包须有判词」）；返工 ≤2 轮。

### 10.6 复核契约（预登记；reviewer 只读 + 独立复算）

| 项 | 内容 |
|---|---|
| 复核对象 | 2661 全套（`axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/`）、2662 全套（`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/`）、两份 WP 报告、本 tick 的 §10 记录、以及 lead 据此改动的 `1.1.4.5`/`1.1.4.1`/`1.1.1` 的 `project.yaml` |
| 复核者 | **fresh 独立上下文**（非 2661/2662 任一 writer）；一份复核件**必须逐 WP 给判词**（`SCHEMA.md` §14：一个复核者可覆盖多个包，覆盖须显式） |
| 必须独立复算（不许只读叙述） | ① 主判据数字（bpc、needle 命中率与 CI、`Δbpc_far`）从 `results.json` 重算；② 前沿/支配关系（2661 引用的 2638/2639 数字与磁盘 `results.json` 逐位对拍）；③ sha256（权重、脚本、config）与 `MODEL.md`/`results.json` 逐位一致；④ 掩码/因果性断言真的存在于代码里且**有抓力**（能否找出一个应当让它变红的情形）；⑤ 「该涨/该跌/必须不变」三条是否与实测方向一致；⑥ 复核者自己列**反否证**：主结论有没有被某个更简单解释支配（如窗口序运气、局部先验、参数更多、chunk、评测口径） |
| 禁越界 | 只读；不改任何文件（判词以最终回复交付，由 lead 原样复制入仓）；不跑 GPU 重训（复核者若需代跑命令，列出来让 lead 代跑） |
| 判词格式 | 每条 blocking/non-blocking 必须给「位置 + 违反的判据 + 复现方式/证据」；**总判词 = PASS / FAIL（窄范围）/ BLOCK**，附独立复算过的数字 |
| 轮数 | 累计 ≤2 轮（`docs/research-schema.md` §11.3）；2 轮未 PASS ⇒ 如实登记 blocker 返回，不开第 3 轮 |
| 已知能力限制登记 | 复核者若无 shell，必须**自述**哪些检查未跑（历轮已多次出现此限制，须原样登记） |

### 10.7 更正记录（时间戳；`2026-10-14T20:54:44+08:00`（= 更正提交 `110f9e2` 的 committer 时刻），actor：axis1-lead-3）

- **我写错了时间戳，按 §8 立刻补更正提交（不改上一条提交历史）**：本文件 §10.4/§10.5 与 5 个
  `project.yaml` 的 `change_log` 首轮写的是 `21:05` / `21:08` / `21:12`，而**实际时刻**落在两个
  已观测时钟读数之间（`date -Iseconds` = `20:49:29` 与提交 `d4175cc` 的 `20:51:51`），
  `20:51:51` 之外的读数都是**未来时刻**。现更正为 `20:51:00`（C1/F9 的 `change_log`，落在上面两个读数之间）与
  `20:52:00` / `20:52:45`（两个 WP 的派发时刻，落在 `d4175cc`（`20:51:51`）之后、更正提交 `110f9e2`（`20:54:44`）之前）。根因同上一轮 P2-C①：**先写再看时钟**；纪律
  重申为「先取时钟、再写时间戳」。

### 10.8 复核排程（Phase C）

- 派发前置条件：**两个 WP 的 writer 全部停止写入**（2661/2662 的 worker 返回、且 lead 不再改这两项的
  `project.yaml`）之后才派 `reviewer`（fresh、独立上下文、非任一 writer）。
- 覆盖：一份复核件对 **WP-C1(2661) / WP-C2(2662) / 1.1.3 的终局判决**分别给判词（`SCHEMA.md` §14：
  覆盖显式、每包有判词），契约见本文件 §10.6。
- 返工：缺陷 ⇒ lead 修 ⇒ **新的** fresh 复核 ⇒ 累计 ≤2 轮（`docs/research-schema.md` §11.3）。
- 已知风险（提前登记，供复核者重点证伪）：① `1.1.3` 由「未开始」改为「已完成」的依据是
  「问题被数据回答 + 并入 1.1.1」，而不是本项自有实验——**这一条最需要复核者独立判断**；
  ② 2661 的 `carrypl16` 与 `win16` 之间除「有无通道」外还差 chunk（64→256）与参数量（+264,296，
  +1.95%）；③ 2662 的「锚点桶召回」与「信息右移」是两件事，必须按判据④分开读；④ 1.1.1 的
  证据跨越三种口径（T=256/12L/mlp6、24×16/12L/mlp6、T 子线 8L/mlp4/bs8），**不得合并成一条曲线**。

### 10.9 未到终局项：未满足判据 + 下一步（Phase C 收口时的 blocker 表）

> 判据 = 「milestone 终局」（`status ∈ {已完成, 被否证}` + `conclusion` + 落盘证据 + fresh 复核）。
> 未满足原因分三类：**算力**（本 run 3h 内跑不完「训练 + 两轮复核」）、**引擎能力**、**依赖前序项**。

| id | 未到终局 | 未满足的判据 | 下一步最小实验（等效成本） | 原因分类 |
|---|---|---|---|---|
| 1.1.2 | 进行中 | milestone 的 per-head slope 与长度外推两栏无仓内读数 | 13M/{per-head slope vs 固定 w} × 2 seed（6 臂 × 98.3M token） | 算力 |
| 1.1.4.2 | 进行中 | 三类压缩（摘要槽/低秩/pooling）零自有实验 | 同预算六臂（6 臂 × 98.3M token；先 3 臂最小判决） | 算力 |
| 1.1.4.3 | 进行中 | 判据错配未修（固定 ρ 下重建质量） | 固定 ρ∈{2,4,8} × 三规则（3 臂 × 98.3M token） | 算力 |
| 1.1.4.4 | 未开始 | 本仓零经验、无检索腿 | {stock, stock+检索}（2 臂 × 98.3M token + 检索腿实现） | 算力 + 实现 |
| 1.2.1 | 未开始 | 本仓零经验、无 dilated 掩码 | 三臂对齐可见键数（3 臂 × 98.3M token + 掩码实现） | 算力 + 实现 |
| 1.2.2 | 进行中 | 真块稀疏掩码未实装 | {local+global, block+stripe} 对齐可见键数（≥4 臂 × 98.3M token） | 算力 + 实现 |
| 1.2.3 | 进行中 | 与「按位置最近 k」的同预算正面对照未做 | **只缺窗口臂**：在 2614 配置（8L/mlp4/T256/12k）上补 {win32, win64}（2 臂 × 98.3M token，复用 2614 的 topk32/topk64/full 臂） | 算力 |
| 1.2.4 | 未开始 | 本仓零经验、无合并实现 | {stock, 金字塔 r=2/4/8}（4 臂 × 98.3M token + 合并实现） | 算力 + 实现 |
| （1.1.4.1 的剩余） | 部分 | milestone 字面「每个预算点两轴 + 与业界实现对齐」+ 归因不可分（F8） | ①召回轴由 2661 补；②量化点 0 训练 token（再分析）；③归因需引擎第二开关（E1 把「逐层状态」与「内容写门」绑在一起） | 引擎能力 + 算力 |

- **依赖方向登记**（`deps` 只记方向，不构成组合授权）：`1.1.4.5` 的 explicit 半边依赖 `1.1.4.1` 的
  机制完备度（本轮 E1 已补）；`1.1.3` 的判决依赖 `1.1.1` 的窗口曲线。

_(本节由 axis1-lead-3 于 2026-10-14T21:03:51+08:00 写入；数字与路径均可在仓内复核。)_

### 10.10 运行中发现（Phase C，2026-10-14T21:28:29+08:00）

1. **子 Agent 单次运行有 30 min 硬上限**（harness 报 `Subagent timed out after 1800000ms`）：两个 WP 的
   worker 都在**等 §20 GPU 锁**时被判超时（2661 在 `21:22` 左右、2662 随后）。⇒ 本轮把 WP 的
   「墙钟预算」改写为：**子 Agent 内 ≤25 min，且不得把等锁算进交付路径**；训练/评测一律
   `nohup` 后台 + completion banner 判活，子 Agent 只做「落盘 + 读结果 + 写报告」。
2. **两个 WP 的 detached 作业仍在跑**：2661 的 `/tmp/2661_drive.sh`（轮询 §20 锁，拿到就跑全电池）、
   2662 的 `run_all.sh`（冒烟→两臂训练→`make_config --require-new`→评测），两者都逐次取/放锁。
   **未杀任何进程**（含别线的训练与 `llama-server`）。
3. 2661 的 worker 在超时前已完成：`README.md`（预登记契约，含 8 条判据线）、`config.json`、
   `run.py`（含判据⑤的逐位复现硬断言与 §4a 口径）、`MODEL.md`、`analysis.md`（六问骨架）。
   其 `/tmp/2661_cpu3.log` 暴露过一个 `KeyError: 'carry16_s0'`（已在 `21:07` 的 `run.py` 里改为
   `for name in quality` 遍历，**已修**）——过程证据留在 `/tmp`，不进仓。
4. 2662 的 worker 在超时前已完成：`README.md`（预登记契约）、`train_arms.sh`、`run_all.sh`、
   `make_config.py`、`run.py`；冒烟 `smk16a` rc=0（300 步，21:08）；`win16a_s0` 训练中
   （21:17:27 起，21:28 时 9600/12000 步）。

### 10.11 队列与重派（Phase C，2026-10-14T21:33:48+08:00，actor：axis1-lead-3）

- 事实：`2661` 的 detached driver 在 **20 min 等锁上限**处自行退让（`/tmp/2661_drive.out` 末行
  `21:31:28 LOCK TIMEOUT (>20min) — 退让`），**本轮它一条读数都没拿到**。同时 2662 的
  `win16a_s0` 训练在 `21:30:39` rc=0 结束（二臂 `win16a_s1` 排队中）。
- 处置：**只有 lead 能重排 GPU 作业队列**（子 Agent 的 30 min 上限不允许「自己排队 + 自己跑完」），
  故由 lead 重写并重派 2661 的 driver（`/tmp/2661_drive_v2.sh`，与原脚本**同一套命令**，
  两处改动：① 等锁上限 20 → **60 min**；② **关闭 stale 回收**——本轮观测到有别的线用
  `/tmp/pathlm_gpu.lock` 时只写 `name` 不写 `holder_pid`，按原「无 holder_pid 且锁龄 >15 min」
  规则会把**正在跑的他线作业**误回收，破坏 `SCHEMA.md` §20 互斥）。
- 纪律说明：重派只做**排队与落盘**，不改任何判据线、不改 `run.py`。数据到了再派 fresh worker 做收尾
  （写 `analysis.md` 人话摘要 + 数字表 + 报告 + 建议的 `project.yaml` 字段）。
- 队列观测（同一把锁）：轴 3 的 `2642-uniform-narrowing-13m`（两臂）→ 2662 的 `win16a_s1` + 评测
  → 2661 的全电池。**未杀任何进程**。

### 10.12 `1.1.3` 终局判决记录（`2026-10-14T22:04:15+08:00`，actor：axis1-lead-3）

- 动作：`status: 未开始 → 已完成`；写 `conclusion`（质量腿 + 召回腿 + T 子线 + 两读法）、`argument`、
  `failure_analysis`（§7 三分归因）、`experiments`（5 条指针）、`gaps`（3 条）、`conclusion_review`（自评 + 风险点）、
  `WP1.status: 已完成`、`work_packages.WP1.evidence`。
- **理由（一句话）**：本项自己的 `milestone.acceptance` 事前就写了「若否，明写并入 1.1.1」的条件分支，而本仓
  限制「每 token 最多看多少」的机制只有两根（按位置的 `window_anchor_keep_mask`，属 1.1.1；按内容的
  `--attn-sparse-k`，属 1.2.3），且两者互斥 ⇒ 不存在「只限基数、不指定选法」的第三个旋钮，走「若否」分支成立。
- **替代处置（复核者给的建议，我不采纳的理由）**：改为「进行中 + 指针」会与「acceptance 的若否分支已被数据回答」不符，
  且会让一个**已被回答的问题**永远停在未判状态。
- 独立复核：REVIEW-7（fresh，见 §10.13）；其 P1/P2 已逐条修完（见 §10.13 处置表）。

### 10.13 复核记录（Phase C 第 1 轮，冻结块）

- 复核者：`reviewer`（fresh 独立上下文，**非本轮任一 writer**；自述**本环境无 shell**，故 `check_tree`/`summary`/
  `git show`/`sha256` 未跑，已在判词里显式登记）。复核件 = `.scratch/62-axis1-time/REVIEW-7.md`
  （复核者只读、未写文件；判词以最终回复交付，由 lead 原样复制入仓）。
- 范围：**不含** 2661/2662 的实验数据（那时还在 GPU 队列里），只覆盖 C1 回填 / F9 / 1.1.3 终局 / ticket §10。
- 判词：**FAIL（窄范围）** —— 数值、路径、字段、口径声明**全部 PASS**（16 个数字逐位复算对平、5 条证据路径全在、
  C1 12/12 声明且措辞合 §12.1、F9 事实字段已无「同算力」、§10 的时间戳与 `d4175cc`(20:51:51)/`110f9e2`(20:54:44)
  自洽且无未来时刻）；**唯一 blocking = P1**：1.1.3 的 `failure_analysis` 触发未处置。另有 7 条 P2（非阻塞）。
- 处置（本轮全部修完，不改数字、不改判据线）：

| # | 复核项 | 处置 | 落点 |
|---|---|---|---|
| P1 | `1.1.3` 的 §7 失败归因未填 | 补 `failure_analysis`（三分归因：**设计问题**为主因——本项被设计成 1.1.1 的拆分位、无自有判据；**结构问题**作用于「该涨」腿——两种选法都带「选谁」偏好；**innovation 不适用**） | `1.1.3/project.yaml` |
| P2-① | 「并入 1.1.1」是树级处置、未登记待裁 | 措辞软化为「**建议**并入 1.1.1（树级处置待 Main 裁）」，并在 §10.3 表加 **F12** | `1.1.3/project.yaml`、本文件 §10.3 |
| P2-② | T 子线的带未在 `falsification` 声明（借 1.1.1 的自测带 0.009459） | `falsification` 并列声明该借用带，并写明两种带下方向相同 | `1.1.3/project.yaml` |
| P2-③ | `change_log` 非时间正序 | 按时间重排（内容一字不改） | `1.1.3/project.yaml` |
| P2-④ | 指针悬空（§10 自行指针写错为 §10.5）/ §10.1 行 12 陈旧 / 缺终局记录 | 修正指针为 §10.9、行 12 标注「已完成（见 §10.12）」、新增 §10.12 | 本文件 |
| P2-⑤ | `20:5x` 非 ISO8601 | 两处改为 `2026-10-14T20:51:51+08:00` | 本文件 §10/§10.4 |
| P2-⑥ | §10.7 更正理由句只覆盖三条中的一条 | 改写为「`20:51:00` 落在 20:49:29–20:51:51 之间；`20:52:00`/`20:52:45` 落在 `d4175cc` 之后、`110f9e2` 之前」 | 本文件 §10.7 |
| P2-⑦ | 证据归属写成「1.1.1 迁入臂」 | 改为「2638/2639 的纯窗口 K 臂（目录家 = 1.1.4.1）」 | `1.1.3/project.yaml` |
- 复核者请求的代跑检查：见 §10.14。
- **第 2 轮复核**：第 2 轮已派并返回（判词落 **§10.18**）；按 `docs/research-schema.md` §11.3 累计 2 轮已用完。

### 10.14 复核者请求的代跑检查（axis1-lead-3 代跑，`2026-10-14T22:04:22+08:00`）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `python3 tools/check_tree.py` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| 2 | `python3 tools/summary.py --check` | 同上 `OK`（本轮每次改动后都跑，见各节） |
| 3 | `git show --format='%cI %h %s' --stat d4175cc` | 6 文件 / `2026-10-14T20:51:51+08:00`（与 §10.7 自述一致） |
| 4 | `git show --format='%cI %h %s' --stat 110f9e2` | 6 文件 / `2026-10-14T20:54:44+08:00`（更正提交） |
| 5 | `git show fb8a576:axes/1-time/1.1.1-hard-window/project.yaml \| grep -n '用哪个带\|判据带（C1 裁决）'` | 命中 `validation:`(13) 与 `conclusion:`(18)（及 `change_log`(101) 的叙述），**`falsification:`(15) 行不含任何 C1 声明** ⇒ 证实 §10.4 的「上一轮 C1 回填的落点只有 `validation`/`conclusion`，`falsification` 缺声明」 |
| 6 | `git show fb8a576:axes/1-time/1.1.3-visibility-cap/project.yaml \| grep -c 用哪个带` | `0`（全文件零命中，改前确实缺声明） |
| 7 | `git diff --check` | 无输出（无空白/行尾问题） |
| 8 | 未来时间戳专项扫描 | 见下（§10.15 的清单） |

### 10.15 未来时间戳专项扫描（`2026-10-14T22:04:22+08:00`）

- 扫描口径：`axes/1-time/*/project.yaml` 的全部 `change_log[].date` + `started` + `ended`，与本文件 §10 的显式时间戳，
  逐个与「当前时刻上界」（本行 `date -Iseconds`）比较；任何 > 当前时刻者判缺陷。
- 结果：**0 处未来时刻**（本文件此前两处通配符 `20:5x` 已按 §10.13 P2-⑤ 改为 `20:51:51`；§10.7 的两处更正已在提交里）。

### 10.16 GPU 队列观测与「未到终局」的等效算力成本（2026-10-14T22:08:44+08:00）

- **锁占用观测（同一把 `/tmp/pathlm_gpu.lock`，非本轴作业）**：`2283632`（轴 2 2651 训练）→ `wp2652-2.2.1-residual-form`
  （轴 2）→ `2642-uniform-narrowing`（轴 3，重试循环 5 s）与 `2643-tied-untied`（轴 3）→ `wp6.3.2-fill_bidir_r*`（轴 6）。
  本轴的两个作业**始终排在队尾**，两次 20 min 等锁上限都自行退让（`21:31:28` 2661、`21:51:57` 2662 第二臂）。
- **本轴为拿到读数还做了什么（均未违反 §20）**：① 不启用会误回收他线锁的 stale 回收；② 按 `SCHEMA.md` §20.1
  逐臂取/放锁；③ 依 Main 裁决**只缩批数不缩臂**（`2661`：bpc/probe 96→48、needle 64→32、retention 24→12；
  臂集仍为预登记 6 臂），并把缩批事实写进 driver 日志与 `analysis.md` 待办；④ 不杀任何进程（含他线训练与 `llama-server`）。
- **2662 的一个真缺陷（本轮新发现，已记录）**：`make_config.py --require-new` 在缺 `win16a_s1` 时**不把已训好的
  `win16a_s0` 写进 `config.json`**（其 `arms` 现只剩 `win16_s0/win16_s1/full`）⇒ 即便评测跑起来也**测不到锚点臂**。
  下一步必须先重生成 config（含 `win16a_s0`、把 `win16a_s1` 标为未训练）再跑 `run.py`。
- **未到终局的等效算力成本（给下一次派发用）**：
  - `2661`（1.1.4.5 正面对撞 + 1.1.4.1 召回轴）：**0 训练 token**；评测 = 6 臂 ×（bpc 96×16×T256 + 探针 2×96×16×T256
    + needle 64×16×行）≈ 6 × 3.9e5 token ≈ **2.3e6 token 的前向**（纯推理，约 4–8 min 独占 GPU）。
  - `2662`（1.1.1 锚点臂）：训练 = 2 臂 × 98.3M token（`win16a_s0` 已完成，剩 `win16a_s1` 1 臂 × 98.3M token）；
    评测 = 5 臂纯推理（同上一行量级）。

### 10.17 Main 指派的收口任务：轴 1 变异锚点更新与自检重跑（2026-10-14T22:15:50+08:00）

> 事由：Main 情报——E2/E3 已落地（`REVIEW-E2E3.md` PASS/0 blocker），E2 改了 `pathlm/depth_ar.py` 的接线行，
> 使本轴 `2614…/mutate.py` 的变异锚点 `old=` 失配；要求更新锚点、重跑自检确认仍会红，并顺带扫其它内联了
> E2/E3 引擎行的本轴脚本。**只写 `axes/1-time/**` 与 `.scratch/62-axis1-time/**`。**

1. **全量扫描（机械，32 个 `mutate*.py`）**：以每个模块**自己声明的文件目标**（`TARGET`/`TESTS`/
   `TFILE`/`TR` 等）为口径，逐个变异的 `old` 做「是否仍存在于目标文件」判定。结果：
   - **失配 1 处（E2 造成）**：`1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py` 的
     `k-ignored`（`DepthARModel.__init__` 的 `TopKBlock(...)` 接线行）。
   - **失配 1 处（非 E2/E3 造成，早前 `tools/evalkit/retention.py::needle_fate` 改名造成）**：
     `1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/mutate.py` 的 `M2`（`keep` → `ks`）。
     按同一纪律一并修（记录在案：**这一处不是本轮 E2/E3 引起的**）。
   - 其余 30 个脚本**零失配**。
2. **锚点更新（只改锚点文本，不改变异语义）**：
   - 2614 `k-ignored`：`old` → `TopKBlock(mcfg, cfg.attn_sparse_k, _mh(i)) … for i in range(cfg.n_layers))`；
     `new` → `TopKBlock(mcfg, mcfg.seq_len, _mh(i)) …`。
   - 2615 `M2`：`old`/`new` 的 `keep` → `ks`（仍是 `all` → `any`）。
3. **自检重跑（隔离运行，避免污染他线）**：变异是**改写共享引擎/工具文件**的破坏性操作，而共享卡上还有
   别的轴在起训练/评测子进程 ⇒ 本轮在**隔离副本** `/tmp/axis1mut/`（`pathlm/`/`tests/`/`tools/`/
   `train_depth_ar.py`/`configs/` 为拷贝、`data/` 为符号链接）里跑：
   - `2614/mutate.py` ⇒ **5/5 变异全部抓红**（rc=1 / `test_failure`）、还原**逐字节一致**、还原后基线 **6 passed**；
   - `2615…/mutate.py` ⇒ **M1–M4 全部抓红**（`returncode=1`）、`restored_sha256_equal=true`、还原后基线绿。
   - 副本与仓库的对应文件 **sha256 逐位相同**，且**跑前跑后仓库文件 sha 不变**（证明隔离有效）：
     `pathlm/depth_ar.py = 45b7d406…`、`tools/evalkit/retention.py = 9ef983b7…`。
4. **落盘**：两份 `mutation_report.json` 按重跑结果更新（旧版在 git 历史里），并在各自实验目录写
   `MUTATION-RERUN-NOTE.md`（说明事由、锚点改动、隔离运行与 sha 证据）。

### 10.18 复核记录（Phase C 第 2 轮）与收口后订正（2026-10-14T22:17:14+08:00，actor：axis1-lead-3）

- 复核者：**新的** fresh 独立上下文（非第 1 轮复核者、非任一 writer）；复核件 = `.scratch/62-axis1-time/REVIEW-8.md`
  （复核者无写文件工具 ⇒ 报告由 lead 原样复制入仓）。复核对象 = 提交 `e8f0d42`（`2026-10-14T22:05:04+08:00`，28 条路径）
  及其后的工作树。
- **判词：PASS（窄范围）** —— R1 的 **P1 与 7 条 P2 逐条复核为「真修」**；破坏性对拍通过（16 个数字与盘上原始值
  逐个对平、12/12 带声明仍在、F9 未回退、5 条证据路径全在、提交时刻自洽、**0 处未来时刻**）；**未发现 P0/P1**。
  其自述能力限制：本环境无 shell ⇒ `check_tree`/`summary`/`git show`/`sha256`/`pytest` 未跑（已按 §10.14/§10.19 由 lead 代跑）。
- 新增 5 条**非阻塞 P2**（Finding 1–5）与**收口后订正**（这些修发生在 2 轮复核之后，**不再开第 3 轮**——
  `docs/research-schema.md` §11.3 的累计上限已用完；故如实登记为「未复核的收尾订正」）：

| # | 复核项 | 处置 |
|---|---|---|
| F1 | `1.1.3.conclusion_review` 仍写「独立复核未做」，与同文件 `change_log` 打架（§11.1） | 已改为「REVIEW-8 = PASS（窄范围）+ 历轮见 §10.13/§10.18」 |
| F2 | F12 行引用的「Phase C 收口报告 §④」仓内不存在（悬空引用） | 已改为「本 run 的收口汇报；本节即其登记，未另落盘报告」 |
| F3 | §10.13/§10.14 两处自指针指错节（「见 §10.15」应为 §10.18；「§10.15 的表」应为「清单」） | 已改 |
| F4 | §10.7 头部 `20:55` 超出其自述的观测时钟 | 已改为 `2026-10-14T20:54:44+08:00`（= 提交 `110f9e2` 的 committer 时刻，可核验） |
| F5 | 承载修复的提交号 `e8f0d42` 未进 ticket | 已在本节与 §10.13 记录：`e8f0d42`（`22:05:04`，28 路径）、`676c858`（变异锚点更新，见 §10.17） |
- 复核者还记了 3 条 report-only（`conclusion:17` 同字段两种强度、§8 旧轮的非完整 ISO 写法属历史行、§10.3 行序 F9/F10/F12/F11），**均不改**。

### 10.19 第 2 轮复核者请求的代跑检查（axis1-lead-3 代跑）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `python3 tools/check_tree.py` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| 2 | `python3 tools/summary.py --check` | 同上 `OK` |
| 3 | `git diff --check` | 无输出、rc=0（无空白/行尾问题） |
| 4 | `git show --format='%cI %h %s' --stat e8f0d42` | `2026-10-14T22:05:04+08:00 e8f0d42 轴 1 Phase C：按 REVIEW-7 … 逐条修 + 落盘两个 WP 的契约与半成品`（28 条路径）——与复核者从 reflog 独立解出的时刻**逐秒一致** |
| 5 | `git show --name-only --format= e8f0d42 \| grep -vE '^(axes/1-time/\|\.scratch/62-axis1-time/)' \| wc -l` | **0** ⇒ 该提交**无越界路径** |
| 6 | `git ls-files axes/1-time \| grep -c '\.pt$'` | **0** ⇒ 权重未入库（与 `.gitignore:11` 的 `*.pt` 一致） |
| 7 | `date -Iseconds` | 执行时刻 ≥ `2026-10-14T22:17:17+08:00` > `e8f0d42` 的 `22:05:04` ⇒ 该提交**不是**未来时刻；§10.15 的扫描结论成立 |
| 8 | `git diff --cached --name-only \| wc -l` | `0`（提交前均为空，无残留暂存） |

### 10.20 收口状态：单臂并发插入的结论、仍在排队的作业、离开时的边界（2026-10-14T22:26:13+08:00，actor：axis1-lead-3）

1. **§20.1.2 单臂并发插入：试过两次，两次都被实验自身的自检挡回（0 读数）**。
   - 依据 Main 裁决「实测单臂是否满足 §20.1.2（纯推理 ≤2 min、≤1 GB），满足则可并发插入、不取锁」，
     新派 fresh worker 用单臂配置（`single_arm/config_carrypl16_only.json`、`single_arm/config_win16_only.json`）
     跑 `2661/run.py`，**未取锁、未动锁目录**（符合 §20.1.3）。
   - 结果：两臂都在自检阶段 `rc=1` 退出——`AssertionError: 无通道臂参数量不一致`（只留 `carrypl16_s0` 时）与
     `AssertionError: 通道臂参数量不一致`（只留 `win16_s0` 时）。⇒ 该实验的**等量对照**要求「有通道臂与无通道臂
     同时在场」，**单臂配置在结构上不可跑**（不是环境问题）。
   - 过程产物：`2661…/single_arm/{config_*.json,run_log_*.txt}`（未产生 `results.json`/`needle_raw.json`）。
2. **预登记跑法（6 臂、缩批版）仍在排队，离开时未取得锁**：driver = `/tmp/2661_drive_v4.sh`
   （等锁上限 90 min，逐次取/放锁；臂集 6 臂不变，仅 bpc/probe 96→48、needle 64→32、retention 24→12，
   已在 driver 日志与 §10.16 登记）。**未杀任何进程**；若它在我返回后拿到锁，产物直接可用于
   §10.1 第 1/2 行的下一步（`results.json` + `needle_raw.json` → 写 `analysis.md`）。
3. **本次 run 的边界与未做项（如实登记）**：
   - `1.1.1` / `1.1.4.1` / `1.1.4.5` 的终局**未达成**（判据缺读数），已把「作业已建、排队失败、
     下一步的等效成本」写进各自 `gaps` 与 `PHASE-C-REPORT.md` §3；
   - 全量 `pytest tests/` 未跑（本轮只改文本/指针 + 两个实验目录；按 §11.4 的门 = `check_tree` +
     `summary --check` + 命中目标单测；后者以 2614/2615 的隔离变异自检承担）；
   - 第 2 轮复核之后的 5 条收尾订正**未再复核**（§11.3 累计 2 轮已用完），已在 §10.18 显式登记。

### 10.21 新一轮预算（axis1-lead-4）：F12 落地 + 2662 config 修复与排队 + 1.1.4.3 配对 CI 重算（2026-10-14T22:32:41+08:00）

1. **F12 裁决落地**（详见 §10.3 表 F12 行）：`1.1.3` **不并入、不删除**，保留独立条目；`conclusion`/`argument` 改为
   「不是本仓独立旋钮 + 位置类子问题 → `1.1.1`、内容类 → `1.2.3`」并加交叉指针，`conclusion_review` 补裁决，
   `change_log` 记裁决与理由。**未改 `docs/**`、未移动目录**。
2. **`2662` 的 `make_config --require-new` 缺陷**：缺第二臂时不落盘已训好的第一臂（`win16a_s0` 白训）⇒ 派 fresh worker
   修语义（逐臂独立判定 + `missing_new_arms` 字段）并重生成 `config.json`、写 eval-only 的 §20 driver 后台排队
   （报告：`.scratch/62-axis1-time/WP-2662-EVAL-QUEUE-REPORT.md`）。
3. **`1.1.4.3` 的保留腿配对 CI 重算（0 训练 token，已完成）**：见 §10.1 第 11 行与 `1.1.4.3` 的 gaps 末条。
   - 结论：**配对判据在现盘证据上不可复原**（只有聚合计数、无逐针命中向量）⇒ 配对 SE 不可定为唯一值 ⇒
     **本项召回腿仍不可判**，`status` 维持 `进行中`（判据悬空不入账）。
   - 补齐动作：新增并落盘 `recompute_pair_ci.py`（可重跑）、`pair_ci_report.json`、`PAIR-CI-NOTE.md`；
     报告出**非预登记口径**的补充读法（硬子集 n=2322 上三规则相对 `distance` 的差在 N∈{8,32,64} 上**全部含 0**，
     最大正向 = N=8 `lru:k=64` +2.7563pp、CI [−0.1026, +5.6088]pp）——因非配对近似一般更宽，**不据此下否定结论**。
   - 下一步最小实验（0 训练 token）：重跑 2615 的评测腿一次并把**逐针命中落盘**（`needle_raw.json`），之后按 §12.1 做配对 bootstrap。
4. **GPU 队列**：`/tmp/2661_drive_v4.sh`（6 臂缩批版，等锁上限 90 min）**保留**；`2662` 的 eval-only driver 由 §2 的 worker 排队。
   两者都按 §20.1 逐次取/放锁、**不缩臂**。终态见 §10.22（离开时的作业台账）。

### 10.22 `2662` 的两处缺陷与结构性依赖（2026-10-14T22:48:10+08:00，actor：axis1-lead-4）

1. **`make_config --require-new` 的缺陷（已修）**：缺第二臂时不落盘**已训好**的第一臂 ⇒ `config.json` 只剩 3 个复用臂，
   已花的算力（`win16a_s0` 12k 步）等于白扔。修法：逐臂独立判定，把存在的臂全部写入并把缺失的新臂记进
   `missing_new_arms`（现盘：`arms = [win16a_s0, win16_s0, win16_s1, full]`、`missing_new_arms = [win16a_s1]`）。
2. **结构性依赖（本轮新发现）**：`run.py` 的跨臂统计会直接对 `win16a_s1` 取键 —— 用 4 臂配置实测
   `KeyError: 'win16a_s1'`（`run.py:394`，纯 CPU 57 s，rc=1，无 `results.json`）⇒ **本实验的评测要求预登记的
   两条锚点臂都在场**，不是「config 少一个臂也能跑」的问题。
3. 因此 `1.1.1` 的下一步 = **①先补训 `win16a_s1`（1 臂 × 98.3M token）→ ②`make_config --require-new` 回填 → ③跑 `run.py`**，
   已落成一个后台全流程 driver（`run_all_missing.sh`，逐段取/放 §20 锁、臂间 ≥61 s）。
4. 纪律：**不得**用缩臂（4 臂）替代预登记的 5 臂判据；临时为绕过缺臂而对 `run.py` 做的改动必须回退
   （worker 报告须给出 `git diff` 证明该文件无改动）。

### 10.23 GPU 窗口终于到手（部分）+ 一次 §20 并发事故的如实登记（2026-10-14T23:00:41+08:00，actor：axis1-lead-4）

#### A. 拿到两个窗口的实测

| 时刻 | 事件 | 结果 |
|---|---|---|
| `22:57:19` | `2662` 的 driver（仓内 `run_all_missing.sh`）取到锁，开始训 `win16a_s1`（seed 1、12k 步、gpu_free 21091 MiB） | 训练进行中（23:00 时 step 4000/12000） |
| `22:57:38 – 22:58:17` | `2661` 的 driver 取到锁，先 `--smoke`（rc=0）、再跑**缩批版**全电池 | **`full rc=1`：判据⑤（与 2638/2639 记录值逐位复现）按设计硬失败** |

- **缩批与预登记判据不兼容（本轮新增事实）**：缩批版（bpc/probe 96→48、needle 64→32）实测
  `win16_s0 1.627905 / carry16_s0 1.624146 / carrypl16_s0 1.527851 / carrypl16d_s0 1.583067`，
  与记录值差 `−0.000205 / −0.000833 / +0.000256 / −0.002464`（6 位小数不等）⇒ `run.py` 的
  逐位复现断言（预登记的「口径一致」判据）**必然失败**。⇒ **结论：`2661` 只能用预登记原批数跑**
  （默认 `config.json` 即 6 臂 / bpc 96×16 / probe 96×16 / needle 64×16）。已据此重写并重派
  `/tmp/2661_drive_v5.sh`（`23:00:28` 起排队）。
- 缩批版的**副作用**登记：它没有产出 `results.json`（断言在落盘前失败），但它证了流水线可跑通
  （`--smoke rc=0`）与「缩批会把 bpc 拉低 0.0002–0.0025」这一量级。

#### B. §20 并发事故（如实登记，按 proof.anomaly）

- **事实**：`22:56:57` 我停掉自己重复的 `/tmp/2662_drive.sh`（`kill`），该脚本的 `trap release EXIT`
  执行了 `rm -f $LOCK/holder_pid` —— **但它当时并未持锁**（它在等锁）。后果：约 40 s 内
  `2662` 的训练（`22:57:19` 取锁）与 `2661` 的评测（`22:57:38` 取锁）**并发**跑了 ~40 s，
  构成一次 §20 互斥违例窗口。
- **影响评估**：两边的**读数不受影响**（bpc 与训练都是「配置 × 数据 × seed」的确定函数，并发只改墙钟、
  不改算术；两边峰值显存合计远低于 24 GB）；受影响的只是**锁的互斥语义**。
- **根因与纪律（写进本轴纪律）**：① 不要临时起「退出时无条件 rm 锁内容」的驱动脚本；② 若必须停一个
  **正在等锁**的 driver，用 `kill -9`（跳过 trap）；③ 锁目录里 `holder_pid` 被误删时，等待者会按
  「无 holder_pid」处理，可能触发别的等待者的回收逻辑 —— 本次未观察到进一步的连锁。
- **环境侧同时存在的干扰**（不是本轴造成，供核对）：他线用 15 s 重试循环反复取锁、有的只写 `name`
  不写 `holder_pid`；本轴两个 driver 都**只**保留「holder_pid 已死」这一条自愈规则。

### 10.24 结果判读 WP 的预登记契约（数据到手即派；不再改）

**WP-R1 · `2661` 结果判读**（供 `1.1.4.5` 与 `1.1.4.1` 的召回轴）

- what：读 `2661` 的 `results.json` / `run_log.txt`，把**预登记 8 条判据线**逐条对成数字，写 `analysis.md`
  的人话摘要 + 六问 + 数字表，并给出一份**建议**的 `project.yaml` 字段（`status`/`conclusion` 供 `1.1.4.5`；
  `1.1.4.1` 的召回轴段落）——**不直接改任何 `project.yaml`**（那是 lead 的活）。
- acceptance：① 每条判据给「实测值 + 方向（该涨/该跌/必须不变）+ 是否过线」；② 主对撞 = `carrypl16_s0` vs `win16_s0`
  的 beyond 桶（整体与同索引硬子集）与 `Δbpc_far`；③ 必须原样报 判据⑤（与 2638/2639 记录值逐位复现）的结果；
  ④ 结论只覆盖「enwik8 / T=256 / 13M / 12k 步 / 该可见性快照」，禁外推。
- evidence：`2661…/analysis.md` + `.scratch/62-axis1-time/WP-2661-RESULTS-REPORT.md`。
- 纪律：**不改** `run.py`/`config.json`/`README.md`；不跑 GPU；不动别的目录。

**WP-R2 · `2662` 结果判读**（供 `1.1.1`）

- what：同上一套流程，读 `2662` 的 `results.json`，判「锚点价格（bpc 差）」与「锚点桶召回 vs 配对地板」，
  并**如实分开**报「锚点抬高召回」与「锚点算不算信息右移」（判据④毁源敏感度）。
- acceptance：① 质量腿给 `price_bpc = bpc(win16a) − bpc(win16)`（两个 seed 各一）+ 本格自测带/0.014 双带并列；
  ② 召回腿给锚点桶「植入 vs 配对地板」的两比例差 CI（含 Wilson 与配对 bootstrap）与 beyond 桶负对照；
  ③ 必须不变项（因果性、掩码非退化、复用臂逐位复现）逐条报；④ 建议的 `1.1.1` 字段（`status`/`conclusion`/`gaps`）。
- evidence：`2662…/analysis.md` + `.scratch/62-axis1-time/WP-2662-RESULTS-REPORT.md`。
- 纪律同上。

**复核**（两个 WP 各一次、fresh 独立上下文；新派发 ⇒ 新一轮 ≤2 轮预算）：复核者须独立复算
①两个实验的 `key_readings`/判据数值与 CI；②与 2638/2639/2634 记录值的逐位复现；③「该涨/该跌/必须不变」
三条与实测方向是否一致；④反否证（窗口序、种子方差、参数更多、缩批、并发争用是否影响读数）。

### 10.25 两个实验都跑出来了（rc=0）：头条数字与可信性说明（2026-10-14T23:09:52+08:00，actor：axis1-lead-4）

| 实验 | 起止 | 结果 | 产物 |
|---|---|---|---|
| `2661`（1.1.4.5/1.1.4.1 召回轴） | `23:05:42 – 23:08:53`（预登记原批数、6 臂同跑） | **`full rc=0`** | `results.json` + `needle_raw.json` + `run_log.jsonl` |
| `2662`（1.1.1 锚点臂） | 训练 `22:57:19 – 23:0x`（补 `win16a_s1`）→ 评测 | **`EVAL done rc=0`** | `results.json`（+ `train/win16a_s1/`） |

**`2661` 头条（预登记 8 条判据全绿）**：同索引硬子集（`intersection`，n=2699，六臂 control 全答不出）上
的 beyond 桶植入召回 = `win16_s0 0.704%` / `win16_s1 1.000%` / `carry16_s0 1.593%` / `carry16_s1 1.964%` /
**`carrypl16_s0 10.226%`** / `carrypl16d_s0 2.853%`；**explicit(完备) − implicit = +9.522pp（CI [8.336, 10.708]）**、
对 seed1 亦然 +9.226pp；beyond 桶整体（n=8178）`carrypl16_s0` 植入 0.520665 vs 配对地板 0.456835 = **+6.383pp（CI [4.854, 7.912]）**
（退化实现 `carry16_*` 仍是 +0.05pp）；毁源敏感度 `Δbpc_far` = **+0.032965（CI [0.026986, 0.038727]）**，
是 2634 退化实现的 +0.0158 的 **2.1×**；判据⑤**逐位复现**、判据⑥六臂全因果、判据④前提成立（implicit 逐位不敏感）。
⇒ **「完备机制（逐层独立状态 + 内容写门）既搬得动远处内容、又把窗外离散事实的可取回性抬起来了」**，
而**退化实现（单共享槽）与 implicit 都停在地板**。

**`2662` 头条**：锚点臂 bpc = `win16a_s0 1.622527`（对 `win16_s0 1.628110` ⇒ **−0.005583**）、
`win16a_s1 1.621799`（对 `win16_s1 1.629093` ⇒ **−0.007294**）⇒ **锚点是质量收益**（越本格自测带
σ_cell≈0.0007–0.00166、**不越** 0.014 跨格带，两带并列）；锚点桶召回：s0 **+2.592pp**（Newcombe CI 含 0、
**配对 bootstrap CI [1.196, 4.187] 不含 0**）、s1 +1.097pp（两套 CI 都含 0）⇒ 按预登记「**两个 seed 都过才算有区分力**」
**未达成**；窗外（beyond 桶）两个 seed 的差 CI 都含 0 ⇒ **锚点保住的是锚区，不是把远处搬近**。

**可信性说明（并发争用不改读数）**：本轮锁环境有他线 15 s 重试循环造成的频繁易手，两个实验都出现过
短时并发窗口（见 §10.23 B）。但 bpc / 召回 / 探针都是「权重 × 数据 × 掩码 × seed」的**确定函数**，
并发只改墙钟与显存占用、不改算术；且 `2661` 的**判据⑤逐位复现**（与 2638/2639 落盘值 6 位小数全等）
本身就是「流水线按预登记口径正确跑完」的强证据。⇒ 两套读数可用；并发事实仍按 §10.23 如实登记。

### 10.26 收尾复核（第 3 次派发，新预算）与落地记录（2026-10-14T23:18:07+08:00，actor：axis1-lead-4）

- **本轮两个实验的判读 WP 已交付**：`WP-R1`（2661 → `1.1.4.5` / `1.1.4.1` 的召回轴）与
  `WP-R2`（2662 → `1.1.1`）；产物 = 两个实验目录的 `analysis.md` +
  `.scratch/62-axis1-time/WP-{2661,2662}-RESULTS-REPORT.md`。
- **lead 的落地**（改的是事实文档；数字一律取自两个 `results.json`）：
  - `1.1.4.5`：`进行中 → 已完成`（两读法：字面大部分成立 / 实质成立；`falsification` 的两支都**未触发**）。
  - `1.1.1`：`进行中`（**不改状态**）——主正面主张（锚点召回）在 n=2 下**不可判**、WP1/WP3 未完成、
    KV 消融保留曲线未在本目录复算；改写 `conclusion`（用仓内自包含读数取代旧「锚点 needle 76.8→93.3」表述 ——
    后者是另一台仪器的绝对召回跳变、可归因于距离惩罚）、补 `failure_analysis`（§7 三分归因）、
    `experiments` 加 2662、`gaps` 换成本轮实测的三条下一步、WP1/WP3 标进行中、WP4 标已完成。
  - `1.1.4.1`：`进行中`（**不改状态**）——`conclusion` 追加召回轴段落、`experiments` 加 2661、
    `gaps` 换成本轮实测的两条下一步（量化点同图 + 完备臂种子重复）。
- **lead 的判据归因裁决（须复核者与 Main 看）**：2661 的 `results.json` 里并存两个召回聚合口径：
  预登记的**主对撞**（`needle_leg.criterion_rule`：`carrypl16_s0` vs `win16_s0` —— 本项据此判**过线**）与
  更严的 `judgement.intervention_arms_all_pass=false`（要求四条带通道臂**全过**；但其中 `carrypl16d_s0`（γ=0.99）
  与 `carry16_s0/s1`（退化实现）在预登记里就是**对照组**、本就期望停在地板）⇒ **以主对撞为准、把更严读法并列登记**，
  **不改判据线**；该裁决属「口径归因」，已写进 `1.1.4.5` 的 `conclusion`/`gaps` 并在此登记，**请复核者独立判断**。
- **判据带（C1）**：bpc 类 = 本格自测带为主判（2661 `σ_cell=0.001661`；2662 `σ_cell=0.000983`）、0.14 跨格带为参考，
  两带并列不混用；召回/接受率类只走配对 bootstrap / Wilson / Newcombe CI，不跨量纲借用 0.014。
- **复核排程（第 3 次派发，新预算）**：fresh 独立上下文，覆盖 ① 两个实验的判据数值与 CI 独立复算；
  ② 与 2634/2638/2639 记录值的逐位复现；③「该涨/该跌/必须不变」与实测方向的一致性；
  ④ 上面那条**口径归因裁决**；⑤ 三个 `project.yaml` 的改动与数字是否对平（§11.2 兄弟字段扫描）。
  契约 = §10.24 末段 + 本节。
