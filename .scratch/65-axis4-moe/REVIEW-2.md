# REVIEW-2 · 轴 4 专家轴 Phase A（第 2 轮，frozen 复核词）

> 复核者：fresh 独立上下文 subagent（builtin `reviewer`），非 writer；复核对象 = **修订后的工作树**（N1–N12 处置后）。
> 本文件是复核者**原样**返回的判词，由 axis4-lead 落盘（不改写内容）。

## 结论（PASS / FAIL-narrow）

**PASS-narrow（0 阻断项；5 条非阻断，其中 D1/D3 建议合并前各改一行）。**

复核对象修订：**工作树（未提交）**，基线 = reviewer-launch HEAD。`watchdog_diff --stat` 显示改动仅 3 份 `axes/4-moe/**/project.yaml`，未跟踪文件全部落在 `axes/4-moe/4.1-…/2610-moe-arm-ledger-13m/**` 与 `.scratch/65-axis4-moe/**`（无 `pathlm/**`、`tools/**`、`tests/**`），边界与第 1 轮一致。

判据支撑：`REVIEW-1.md` 的 N1–N12 我逐条在被复核文件里定位到修改点；三份 `project.yaml` 我静态套用 `tools/check_tree.py` 的全部规则（REQUIRED / NONEMPTY / milestone 单 map / role_map 四角色 / WP 8 键 / status 枚举 / `cost_model` `WALLCLOCK_RE` / `validation` 三要素 / `dir`==实际目录）**无一违规**；`ledger.json` 五臂数字 ↔ `analysis.md` 表逐项对上；`preflight.json` 与 `run.py` 的 `pass` 合取式一致。第 1 轮「不可核实」第 1 项（命令实跑）在本轮**仍未核实**——我没有执行工具（见「命令与结果」/「不可核实项」）。

---

## N1–N12 处置核对表

| # | 判定 | 修改点与证据 |
|---|---|---|
| N1 | **已修** | 实验 `README.md:19-20` 已改为「**8×（top-1）/ 4×（top-2）的 FFN 参数**（对应总参数 **5.9× / 3.1×**，因注意力/embedding/读出不变）」。复算：79,837,696/13,602,304 = 5.8697；42,015,232/13,602,304 = 3.0888；FFN 比 75,694,080/9,458,688 = 8.0026、37,871,616/9,458,688 = 4.0039 ⇒ 与 `ledger.json::ffn_params_x_dense` 一致。 |
| N2 | **已修** | `4.2-routing-balance/project.yaml:15` `falsification` 已由「实际不可触发」改为「**本轮探针口径下尚无先验可触发性**」，且紧接写明反向考虑（「探针是无均衡的单臂，capacity factor 截断 / expert-choice 指派**自身可制造偏斜**，该分支的前提（加均衡机制后）从未被测」）。过度声称已消除，分支本身未被删除。 |
| N3 | **已修** | `analysis.md:85` 终评行已拆为「bpc 6.537（**163,520 位置**）；熵/占比取自**最后 1 个 eval batch**（8,192 token×12 层）」。与 `probe_collapse_1200.json` 一致：`n_eval_positions=163520`，`eval.moe_liveness.usage` 合计 98,304 = 12×8,192。 |
| N4 | **已修** | 实验 `README.md` 判据线 4 加注「该阈值在随机初始化第 1 步即已满足（0.968/0.205），命中近乎必然 ⇒ 主证应为『是否下移（斜率）』」；`analysis.md:89-92` 主证改为「①熵不下移（斜率≈0）+ ②温和持久偏斜」；`4.2:17` 亦有同义括注。 |
| N5 | **已修（代码已改，非仅文档）** | `analysis.md:92-94` 新增池化口径限制段（含「1 层全坍缩、其余 11 层均匀 ⇒ 池化最大占比仍≈0.198、归一化熵≈0.99，落在观测区间内」）；`4.2:17` 写「使用率为跨 12 层池化口径（不分辨整层坍缩）」；**读码确认**：`run.py:356-358` `eval_bpc` 新增 `moe_liveness_per_layer`，`run.py:314-316` 训练日志新增逐层最小归一化熵（该新列有 1 处逻辑瑕疵，见 D1）；`analysis.md:94` 如实写「本探针未测」。 |
| N6 | **已修（我读码判定：非恒真）** | `run.py:210-233` 新增判别检验：E=2/k=1、router 权重只留 `[0,0]=+5 / [1,0]=−5`、专家 1 权重全零，比较 `x[:,0]=+1` 与 `−1` 两次前向。**判别力分析**：若路由退化（恒选专家 0）⇒ `yb≠0`；若恒选专家 1 ⇒ `ya=0`；两种退化都使 `routes_to_different_experts=false`，只有「选择取决于输入」时为 true ⇒ **不是恒真断言**。`preflight.json::routing_discrimination.routes_to_different_experts=true`（`expert0_out_absmax=0.13929687`、`expert1_out_absmax=0.0`），且已进 `run.py:251` 的 `pass` 合取式（`preflight.json::pass=true` 与该式一致）。范围限制：只覆盖 E=2/k=1 的单向判别，不等于证明 k>1 的加权组合正确；`analysis.md` 的措辞（「路由真的选到不同专家」）未夸大。 |
| N7 | **已修** | `run.py:76` `h2 = (target - d) / (2 * d + 1)`，docstring（:70-71）与注释一致。复算：target = 8×(2·256·1536+1536+256)+8·256 = 6,307,840 ⇒ h2 = 6,307,584/513 = 12,295.876 ⇒ `mlp_mult=48`，与 `ledger.json` 的 48/24 及实测总参 79,791,616 / 41,969,152 一致。off-by-one 已消。 |
| N8 | **已修** | `4.3-expert-granularity/project.yaml:26` 已改为「使 (k+1)·h_ff（每 token 激活算力）与**总参数 P** 同时保持」，并显式注明「E 增大时每专家参数 P/E 必然下降」⇒ 自相矛盾已消。 |
| N9 | **部分已修（残留）** | `README.md:7` frontmatter 已改为「本阶段 0 GPU 判决训练（无对照质量实验）；仅含一条单臂坍缩探针（§20 持锁）」；但同文件正文 `README.md:14` 仍写「预算：本阶段 **CPU-only**（见 §6）」，与 §6（探针 §20 持锁）矛盾 ⇒ 见 D3。 |
| N10 | **部分已修（残留于票）** | `analysis.md:100/109/114/116/120` 已全部改为 `4.1-WP1 / 4.2-WP1 / 4.3-WP1`；`.scratch/65-axis4-moe/TICKET.md:118` 仍写 `WP2`/`WP3` ⇒ 见 D4（票是只增日志，阻断性低）。 |
| N11 | **已修** | `analysis.md:104` 已注明观测来源「= TICKET §9 的 `nvidia-smi` 读数，未落盘为 JSON」。 |
| N12 | **已修** | `analysis.md:104` 已明写 20 步 `first_20_steps_s = 14.958` 「取自 `timing_20.json`（另一条命令，非本次探针）」，与探针自身 `first_20_steps_s=11.949` 区分。 |

---

## 新增/残留缺陷

**D1（N，报告级；修订引入）`run.py:314-316`** —— 新增的逐层列用 `(r.get("usage_entropy_norm") or 1.0)`：`liveness()`（`moe.py`）返回 `round(ent/ent_max, 6)`，某层完全单专家时该值**恰为 `0.0`**，而 `0.0` 在 Python 中为假 ⇒ 被替换成 `1.0`，正好把它要检测的「整层坍缩」抹平（`None` 与 `0.0` 被混为一谈）。影响面：该列只供下一阶段、未进任何已报数字/判据/探针（`probe_collapse_1200.json` 未重跑，`analysis.md:94` 已声明「本探针未测」）⇒ 非阻断。最小修：`v = r.get("usage_entropy_norm"); (v if v is not None else 1.0)`。

**D2（N）`analysis.md:71` 文本损坏** —— 「`4.2` 的整个前提是「路由会�collaps」」含 U+FFFD 替换字符 + 英文残片，句子不完整（应为「路由会坍缩」），违反 `AGENTS.md` §7「一个文档内不中英段落交替」与 SSOT 可读性。不改变任何数字。

**D3（N；N9 残留）`README.md:14`** —— 正文「预算：本阶段 CPU-only（见 §6）」与 frontmatter:7 及 §6（探针 §20 持锁、`total_s=1395.141`）冲突。最小修：正文改为「本阶段 0 GPU 判决训练；仅一条单臂坍缩探针（见 §6）」。

**D4（N；N10 残留）`TICKET.md:118`** —— 仍以 `WP2`/`WP3` 指 4.2/4.3（其它处已统一为 `4.x-WP1`）。票是过程日志，建议随下次追加修正。

**D5（N）`4.2-routing-balance/project.yaml:17`、`:63`** —— `**无声 loss**`（应「无 aux loss」）与 `**跱 12 层池化**`（应「跨 12 层池化」）为用词/字符错误，同属本轮改动引入；主题句可读性下降，不改变数字与结论。

**备注（不单列）**：`results.json` 的 `collapse_probe` 段落本轮被加入 `evidence_strength_note` / `pooling_caveat` / `entropy_norm_slope_per_step` / `caveats` 等派生字段，与 `AGENTS.md` 文档地图中「`results.json` = 原始数字」的定位略有出入（数字本身与 `probe_collapse_1200.json` 逐项一致：bpc 6.5369、0.975568、0.186991、usage 13,099/9,197/7,041/18,178/10,550/8,863/12,994/18,382）。

---

## 命令与结果

**我执行的（只读工具；无 shell）**

| 动作 | 结果 |
|---|---|
| `watchdog_diff --stat` | 改动仅 3 份 `axes/4-moe/**/project.yaml`；未跟踪文件全部在 `axes/4-moe/4.1-…/2610-…/**` 与 `.scratch/65-axis4-moe/**` ⇒ **边界 OK** |
| `ledger.json` ↔ `analysis.md` 表逐项对账 | 13,602,304 / 79,837,696 / 42,015,232 / 79,791,616 / 41,969,152；FFN 75,694,080 / 37,871,616 / 75,648,000 / 37,825,536；`tokens_per_param` 7.227 / 1.2313 / 2.3397 / 1.232 / 2.3423；`rel_gap` 0.00058 / 0.0011；`ffn_macs_ratio_vs_dense` 1.002604 ⇒ **全部对上** |
| 手算 | 5.8697 / 3.0888（总参数比）；8.0026 / 4.0039（FFN 比）；eqparam 解析式 6,307,840 → h2=12,295.876 → 48；探针 13 点归一化熵最小二乘斜率 **1.537e-5**（落盘 1.5368e-5）⇒ 一致 |
| `preflight.json` ↔ `run.py:240-252` | 字段集合与 `pass` 合取式逐项一致（含 `routing_discrimination`、`aux_loss_check`）⇒ JSON 未过期 |
| 静态套用 `tools/check_tree.py` 全部规则 | 三份 `project.yaml` **无违规**（字段齐 / 非空 / milestone 单 map / 四角色实名 / WP 8 键 / status 合法 / `cost_model` 无墙钟日期 / `validation` 三要素齐 / `dir` 匹配 / `4.1::experiments` 路径存在） |
| 口径纪律 | 三份 `cost_model` 均无墙钟/日期；墙钟只出现在 `timing_process_metric` 与 `analysis.md:103-104` 的过程指标引用块（明写「不入判据」）⇒ 合规 |

**须由主管运行（我无执行工具，未核实）**

```bash
cd /home/z/vibe/pathlm
python3 tools/summary.py --check                 # 期望全绿（46 个 project.yaml）
python3 tools/check_tree.py                      # 同上
cd axes/4-moe/4.1-sparse-experts/experiments/2610-moe-arm-ledger-13m
CUDA_VISIBLE_DEVICES="" python3 run.py preflight # 期望 pass=true、routes_to_different_experts=true
CUDA_VISIBLE_DEVICES="" python3 run.py ledger --out /tmp/l2.json && diff /tmp/l2.json ledger.json
git status --short                               # 期望仅 axes/4-moe/** + .scratch/65-axis4-moe/**
```

---

## 不可核实项

1. **上表 5 条命令**：我本轮**无执行工具**（只有 read/grep/find/ls/watchdog_diff），故 `summary.py --check`、`run.py preflight/ledger` 与 `git status --short` **未实跑**；替代做法为静态套用 `check_tree` 规则 + 手算 + `watchdog_diff`。TICKET §7/§8 自述的「全绿」「pass=true」不得视为已核实。
2. **committed range**：`watchdog_diff` 不含已提交区间，TICKET §8 无 `git add/commit` 记录；若 19:25–20:0x 间存在未记录提交，「工作树 = 交付全集」的推断与第 3 条边界就无法完全核对。
3. **arXiv 标题联网核实**：本轮无网络；编号↔论文映射我按既有知识核对（7 条）**一致**，但 HTTP 200 只见于 TICKET §4。
4. **GPU 实跑数字**（`probe_collapse_1200.json` / `timing_20.json` / `smoke_*.json`）未重跑；只做内部自洽与跨文件一致性核对。
5. **README §4 判据线的「先于结果写下」**：只有 TICKET §7 自述时间线（19:29–19:31 写 README，探针 19:29:03 启动、19:52:23 出结果）；README 未提交、无机器时间戳锚点。
6. **N6 检验的判别力**：我按代码静态判定其为非恒真，但**未实跑**；且其覆盖范围仅 E=2/k=1（不能证 k>1 的加权组合正确性）。
7. **D1**：`usage_entropy_norm_min_layer` 的触发条件要求某层归一化熵**恰为 0.0**；本轮无该情形的实跑数据，判定基于代码语义。