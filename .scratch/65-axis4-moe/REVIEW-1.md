# REVIEW-1 · 轴 4 专家轴 Phase A（frozen 复核词）

> 复核者：fresh 独立上下文 subagent（builtin `reviewer`，经 `subagent` 派发），非 writer。
> 复核对象修订：**工作树**（未提交），基线 = reviewer-launch HEAD。
> 本文件是复核者**原样**返回的判词，由 axis4-lead 落盘（不改写内容）。

## 结论（PASS / FAIL-narrow）

**PASS-narrow（0 阻断项；12 条非阻断建议，其中 N1/N2 建议在交付前改一行）。**

复核对象修订：**工作树（未提交）**。`watchdog_diff --stat` 显示本复核基线相对 reviewer-launch HEAD 的全部改动 = 3 份 `project.yaml`（`axes/4-moe/**`）+ 未跟踪文件全部落在 `axes/4-moe/4.1-…/2610-moe-arm-ledger-13m/**` 与 `.scratch/65-axis4-moe/**`；TICKET §8 未记录任何 `git add/commit`，故可认为工作树 = 本次交付全集（committed range 我无法查看，见「不可核实项」）。

判定依据：`docs/research-schema.md` 字段/口径纪律、`tools/check_tree.py` 的校验规则（我逐条静态套用）、`SCHEMA.md` §14/§15 复核与等参数对照、以及**五臂记账的独立算术核对**（见下）。所有「数字对账」项逐一对上，WP0 的四条验收（专家 MACs 比 1.0 / 等参数相对差 <0.2% / `bitwise_equal=true` / `router_alive=true`）**全部由落盘证据支持**。

---

## 逐条判词（1–9）

### 1. 字段齐全与实名 —— **PASS**
- 三份 `project.yaml` 均含 `REQUIRED` 全字段；`purpose/innovation/validation/objective_criteria/falsification` 非空；`milestone` 各为**单个 map**（含 `output`+`acceptance`）；`work_packages` 每条 8 键齐（4.1 有 WP0/WP1，4.2 WP1，4.3 WP1）；`status` 合法。
- `role_map` 四角色**实名**：`owner: axis4-lead` / `writer: axis4-w-4.1|4.2|4.3` / `reviewer: axis4-reviewer` / `integrator: tree-integrator`（4.1:60-64、4.2:41-45、4.3:45-49）。
- `cost_model` 静态套用 `check_tree.py` 的 `WALLCLOCK_RE`（`墙钟|秒|分钟|小时|tok/s|latency|elapsed|\d{4}-\d{2}-\d{2}`）：**三份都不匹配**（4.1:71、4.2:52、4.3:26），只有等效理论量（FLOPs / layer-passes / tokens-per-forward + pipelining 说明）。
- `sota_baseline` 有可核实出处（arXiv 编号），编号↔论文映射我以既有知识逐条核对**全部正确**（2101.03961 Switch / 2006.16668 GShard / 2202.09368 Expert-Choice / 2401.04088 Mixtral / 2401.06066 DeepSeekMoE / 2202.08906 ST-MoE / 2412.19437 DeepSeek-V3）；联网 HTTP 200 由 TICKET §4 记录（本轮我无网络工具，见「不可核实项」）。
- 名称与 `docs/structure.md` exact wording 一致：轴名 `4 专家轴（MoE）`（structure.md:80）、`4.1 稀疏激活专家（top-k 路由）`、`4.2 路由与负载均衡`、`4.3 专家粒度 / 异构（细粒度、共享专家）`（structure.md:81-83、222-224）；`dir` 与实际目录一致；`id` 全局唯一（grep `^id: "4` 仅 3 条）。

### 2. 数字对账 —— **PASS（逐项对上）**
分析/README 引用的每个数字都能在落盘 JSON 找到同名同值字段：

| 判词要求的数 | 文档 | 落盘 JSON | 结论 |
|---|---|---|---|
| dense-B0 total_params 13,602,304 | analysis.md:31 | `ledger.json::arms["dense-B0"].total_params` | ✅ |
| ffn_macs/token 9,437,184 | analysis.md:31 | 同上 `expert_macs_per_token`/`ffn_macs_per_token` | ✅ |
| moe-top1-E8 79,837,696 / token-per-param 1.2313 | analysis.md:32 | `arms["moe-top1-E8"]` | ✅ |
| moe-top2-E8 42,015,232 / 2.3397 | analysis.md:33 | `arms["moe-top2-E8"]` | ✅ |
| 等 FLOPs 专家 MACs 比 =1.000 | analysis.md:32-34、README:53 | `expert_macs_ratio_vs_dense = 1.0`（两臂） | ✅ |
| 路由税 +0.26% | README:54、analysis.md:32 | `ffn_macs_ratio_vs_dense = 1.002604` | ✅ |
| 等参数相对差 0.058% / 0.11% | analysis.md:40 | `eqparam_check[0/1].rel_gap = 0.00058 / 0.0011` | ✅ |
| 探针归一化熵 0.952–0.983 | analysis.md:85、4.2:7 | `probe…json` log 最小 0.952481、最大 0.982648 | ✅ |
| 最大占比 0.178–0.214 | analysis.md:86 | log 最小 0.178319、最大 0.214101 | ✅ |
| 斜率 +1.5e-5/步 | analysis.md:85、4.2:17 | 我按最小二乘复算 13 点 = 1.537e-5/步 | ✅ |
| 终评 bpc 6.537 / 熵 0.976 / 占比 0.187 | analysis.md:83 | `eval.bpc 6.5369 / 0.975568 / 0.186991` | ✅（覆盖口径问题见 N3） |
| 偏斜「18,178/18,382 vs 7,041」 | analysis.md:88 | `eval.moe_liveness.usage` | ✅ |
| 探针耗时 ≈23 min、19:29:03→19:52:23 | analysis.md:97、TICKET §7 | `timing_process_metric.total_s=1395.141` | ✅ |
| 20 步 14.958（≈0.75 s/步） | analysis.md:97 | `timing_20.json` 同名 | ✅（归属含糊，见 N12） |
| 冒烟 40 步末归一化熵 0.67 | analysis.md:64 | `smoke_moe-top2-E8.json` 0.66974 | ✅ |
| preflight 逐位一致/额外键/参数自洽/活性/梯度 | analysis.md:52-57 | `preflight.json` 逐字段一致；`pass=true` 恰等于代码 `run.py:227-229` 的合取式 | ✅ |

另有：`13M 基底 = configs/B0.json`（实测值核对：`configs/B0.json` d=256/L=12/H=8/mlp_mult=6，steps 6000×batch 32×seq 512 = 98,304,000 ✅）；B0 锚 1.5074 在仓内确有唯一家（`axes/7-references/README.md:13`，见 `.scratch/63-axis6-predict/TICKET.md:97`）✅；噪声带 0.014 / nondeterminism 0.0019 / 种子散布 ≤0.0138 与 `docs/methodology.md:19` 一致 ✅。

### 3. pre-flight 断言是否为真 —— **PASS（逻辑与算术我独立复核成立；实跑未做，见不可核实项）**
- **是否恒真**：**成立，且近乎由构造保证**。E=1、top_k=1 时 `softmax(logits[E=1]) ≡ 1.0`（`moe.py:102-103`），`w = 1.0`，`out = 1.0·expert(x)`；专家子模块与 stock `Block.mlp` 结构逐位相同（`pathlm/model.py:70-73` vs `moe.py:52-55`），`run.py:170` 把 stock mlp 权重灌入 `experts[0]` ⇒ 断言**不可能失败**（除非接线断掉）。因此它是「接口透明 + 权重搬运正确」的证据，**不是** top-k 路由正确性的证据（见 N6）。
- **`copy_shared_from` 是否掩盖差异**：不会掩盖 **MoE 侧多出的非 mlp 键**（`moe.py:188` 返回 `extra`，preflight 记录 `[]`，且 `pass` 含 `not extra`）；但它**不会报告 dense 侧有、MoE 侧缺的非 mlp 键**（`to_load` 取交集，`strict=False` 静默跳过）。此路径在本例中不可能触发（`MoEDepthARModel` 是 `DepthARModel` 子类、同 cfg 构造），但这是弱面；建议在 `copy_shared_from` 里同时返回「dense 有而 moe 缺」的键。
- **算术独立核对（我手算，不依赖 `preflight.json` 自述）**：dense 解析式 = 12×(`2·256·1536`+1536+256) = 12×788,224 = **9,458,688**（= 落盘 `measured`）；MoE(E=8,k=2,h=768) 解析式 = 12×[8×(393,216+768+256)+8×256] = 12×3,155,968 = **37,871,616**（= 落盘 `measured`）。两式与落盘值一致 ⇒ **记账自洽项可信**。
- **非 FFN 参数五臂同源（我独立的单因素核对）**：total − ffn：dense-B0 13,602,304−9,458,688 = **4,143,616**；moe-top1 79,837,696−75,694,080 = 4,143,616；moe-top2 42,015,232−37,871,616 = 4,143,616；eqparam-top1 79,791,616−75,648,000 = 4,143,616；eqparam-top2 41,969,152−37,825,536 = 4,143,616。**五臂完全相同** ⇒ 「只改 FFN」的单因素声明在实测参数量上成立。

### 4. 记账口径自洽 —— **PASS（两处小瑕疵为非阻断）**
- 等 FLOPs：`expert_macs = k·2·d·h_e`，构造上 `k·h_e = h`（1×1536 = 2×768）⇒ 与 dense 的 `2·d·h` **逐位相等**（ledger 两臂均 786,432 MACs/层）。路由税 `E·d = 2,048`/层**单列**，且 `ffn_macs_ratio_vs_dense = 1.002604` 明确把 >1 部分归因路由，**没有混进「等 FLOPs」声称**（`ledger.json::notes[1]`、analysis.md:37-39）。
- 等参数：先解 `mlp_mult`（`run.py:69-72`，解析式），再用**实测**总参数报 `rel_gap`（`run.py:141-149`），不是自证 ⇒ 对照臂 mlp_mult=48/24 的两臂总参数 79,791,616 / 41,969,152 与 MoE 臂相对差 0.058%/0.11% 成立。
- **参数量 vs 每 token 算力**：`ledger`/`analysis` 把两者分列（`ffn_params` vs `expert_macs_per_token`）并把 `tokens_per_param` 单列作数据饥饿列（analysis.md:42-46）——**未混淆**。唯一混淆出现在 **README.md:20**（把 FFN 参数比 8.00×/4.00× 说成「总参数」，实际总参数比 5.87×/3.09×）⇒ 见 N1。
- 小瑕疵：`run.py:72` 解等参数时多减一个 `d`（见 N7）。

### 5. 判据/否证 —— **PASS（附两条口径提醒）**
- README §4 判据线**是**在拿到探针结果前写下的：TICKET §7 记 `19:29–19:31` 写 README，探针 `19:29:03` 启动、结果 `19:52:23` 才产出 ⇒ 阈值先于结果（但**不早于**启动，且 README 未提交、无机器时间戳锚点，见「不可核实项」）。
- 可证伪性：判据线 1/2（记账，已可判定）、3（Δ 与 0.014 噪声带三分类）、5（因果性）均可证伪；判据线 4 为「熵>0.5 **或** 最大占比<0.6」的**析取式**，且在随机初始化第 1 步即满足 ⇒ 证据力弱（N4），但阈值本身与 `4.1` 的 `objective_criteria`（"该均衡：专家使用率分布"）、`validation`（三要素齐）一致；`README §1 可证伪` 段落与 `4.1::falsification` 一致。
- `4.2` 把 `falsification` 的「仍坍缩」分支标注为「实际不可触发」：**部分正确、有过度声称**（N2）。正确的一面是「朴素 top-k 在此口径下不自发坍缩」有预登记阈值 + 落盘数据支撑；过度的一面是把「加均衡机制后仍坍缩」这一分支持有事前一票否决，而该分支的前提（加了均衡机制）**从未被测**，且 capacity factor 截断 / expert-choice 指派本身可能制造偏斜。`4.2::objective_criteria` 与另外两条否证分支未被削弱 ⇒ 非阻断。

### 6. 坍缩探针的解释是否越界 —— **PASS（限制已如实标注），但有一条指标分辨率缺陷**
- 单臂 1200 步能否支持「13M 尺度坍缩不可观测」：**在它自己限定的口径内成立**（README §4 判据线 4、analysis §4 标题「单臂、非判决」），且限制**逐条写明**（未收敛快照、单 seed、单臂、无 aux 对照、禁外推；analysis.md:90-94、4.2:17-18）。
- 未把无对照探针写成判决：`4.2::conclusion_review`、`4.1::conclusion`（「结论属 4.2，不计入本项」）与 analysis §4 均明确「不是质量判决」 ✅。
- **缺陷（非阻断）**：全部活性指标是**跨 12 层池化**的，`run.py:290` 只记 `moe_liveness()["global"]`，因此**整层坍缩不可分辨**（N5，附算术），「无专家死亡」的措辞强于证据。

### 7. 口径纪律 —— **PASS**
- 墙钟只出现在过程指标：`probe…json::timing_process_metric`、`timing_20.json::timing_process_metric`、analysis.md:97 的引用块（明写「过程指标…不入判据」）、README §6（明写「墙钟仅作过程指标」）；**未进** `cost_model`、未进任何判据线/验收。
- 状态标签与证据相符：`4.1` = `进行中`（WP0 已完成 + 有落盘证据；WP1 未开始；`conclusion` 开头就写「本项命题的本地判决尚未做」）✅；`4.2` = `进行中`（前提探针已跑，自身四臂未做，明写「本地判决尚未做」）✅；`4.3` = `未开始`（`experiments: []`、无本地工作）✅。**无谎报**。唯一措辞不符：README.md:7「本阶段 0 GPU 训练」（N9）。

### 8. 边界 —— **PASS（工作树范围）**
- `watchdog_diff --stat`：改动仅 `axes/4-moe/{4.1,4.2,4.3}/project.yaml`；未跟踪文件全部在 `axes/4-moe/4.1-…/experiments/2610-moe-arm-ledger-13m/**` 与 `.scratch/65-axis4-moe/**`。**无** `pathlm/**`、`tools/**`、`tests/**` 改动。
- 交叉佐证：`grep -i "moe|expert" pathlm/` → 无匹配；`grep -i moe tools/` → 无匹配；`tests/` 仅 `test_check_schema_anchor.py` 出现过字符串 `ce2610a0d`（与本次无关）。`MODEL.md` 的「不动共享 `pathlm/model.py`、运行期替换 `Block.mlp`」与 `moe.py:52-55` + `MoEDepthARModel.__init__` 的做法一致 ✅。
- committed range 不可核实（见下）。

### 9. 兄弟字段一致性 —— **PASS（1 处编号命名不一致）**
- 臂名/边界互相引用正确：`4.1` 臂名 `moe-top1-E8`/`moe-top2-E8`/`dense-eqparam-top1|2` 与 ledger 键名一致；`4.2` 引用「`4.1` 的最小 MoE / 朴素 top-k（`4.1`）/ `4.1` 的 `2610` 探针」✅；`4.3` 引用「稀疏激活机制复用 `4.1`、均衡复用 `4.2`、对照 = 均匀粗粒度 MoE（`4.1`）」✅。
- 三格边界互不重叠且与 `4.1` 的 `aux_weight = 0` 一致（`moe.py:19-23`、analysis.md:21-23）✅；probe 归属明确（`4.1` README frontmatter「关联格 4.2（坍缩探针）」、`4.2 experience: []` 但指针指向 `4.1/…/probe_collapse_1200.json`，路径真实存在）✅。
- 不一致点：analysis.md:94/106/112 与 TICKET §5 用 `WP2`/`WP3` 指 4.2/4.3，而 4.2/4.3 的 `project.yaml` 各自只有 `WP1`（N10）。

---

## 阻断项（含文件:行 + 复现）

**无阻断项（blocker = 0）。** 没有发现违规判据线、数字不符、口径导致结论错误的错误、恒真断言被当作路由正确性证据的裁决、或谎报状态。

---

## 非阻断建议（N）

**N1（建议交付前改一行）README.md:20** —— 「用 8×（top-1）或 4×（top-2）的**总参数**」：8.0026×/4.0039× 是 **FFN 参数**比（`ledger.json::arms[*].ffn_params_x_dense`），**总参数**比是 5.87×/3.09×。
复现：`python3 -c "print(79837696/13602304, 42015232/13602304)"` → `5.869… 3.088…`。
最小修：改成「前馈（FFN）参数 8× / 4×（总参数 5.9× / 3.1×）」。

**N2（建议交付前改一行）`axes/4-moe/4.2-routing-balance/project.yaml:15`** —— 「本条的「仍坍缩」分支**实际不可触发**」是事前一票否决自己的否证分支：(a) 分支前提是「加均衡机制后」，而四臂全未跑；(b) capacity factor 截断 / expert-choice 指派**自身可制造偏斜**（探针是**无均衡**的单臂，不含此类机制）；(c) 依据为单臂/单 seed/1200 步/无对照探针（`:17` 自述）。
最小修：改为「在本轮探针口径下该分支尚无先验可触发性；均衡机制自身是否引入偏斜仍须测」。

**N3 analysis.md:83** —— 「终评（**163,520 位置**）」行把 bpc 的覆盖范围与活性量并列，但熵 0.976 / 最大占比 0.187 只来自**最后一个 eval batch**：`eval_bpc`（`run.py:290` 附近）在 20 个 batch 循环**结束后**才读一次 `moe_liveness()`。算术证据：`eval.moe_liveness.usage` 合计 98,304 = 8,192 token ×12 层 = (16×512)×12，而 163,520 是 bpc 的 `n_eval_positions`。
最小修：行标题写「bpc 覆盖 163,520 位置；活性量取自最后 1 个 batch（8,192 token）」或让 eval 累加全评测集的使用率。

**N4 analysis.md:85-86、`4.2::falsification`(:15)/`conclusion`(:17)** —— 「命中预登记阀值」的证据力被高估：阈值为「熵 > 0.5（或最大占比 < 0.6）」，而**随机初始化第 1 步**就已是 0.968129 / 0.204803（`probe…json::log[0]`）⇒ 命中近于必然。真正有信息量的是「1200 步内未下移（斜率 +1.5e-5/步）」与「温和持久偏斜」。
最小修：以斜率+偏斜为主证，阀值只作附带说明。

**N5 analysis.md:86（与 4.2:17 的「无专家死亡」）** —— 结论只由**跨 12 层池化**使用率支持，无法排除**整层坍缩**：若 1 层把 16,384 token 全给某专家、其余 11 层均匀，则池化后最大占比 = 0.198、归一化熵 = 0.990（我按 196,608 = 12×16,384 计），**都落在观测区间 0.178–0.214 / 0.952–0.983 内**。
最小修：`run.py:290` 增记 `per_layer`（`moe_liveness()` 已返回逐层熵/最大占比），或把措辞限定为「池化口径下无专家死亡」。

**N6 run.py:167-181 / moe.py:102-103** —— 透明性断言近乎恒真（`softmax([x]) ≡ 1.0` + 结构同构 + 灌权重 ⇒ 必然逐位相等），不能证伪 k≥1、E>1 的路由。
最小修：加一条差别检验（E=2、top-2 时对「显式参考 gather/加权和」比对，或两专家同权重不同缩放、核对输出随路由切换）。

**N7 run.py:72** —— `h2 = (target - d - d) / (2 * d + 1)` 与注释/`docstring`(:69-71) 差一个 `d`；正解为 `(target - d)/(2d+1)`。现解使 dense 每层 FFN 参数 = target − 256（共 3,072 ≈ 总参数 0.0038%），被 `mlp_mult` 取整（48/24）完全吸收，**不影响任何报告数字**。
最小修：`(target - d)` 或修 docstring。

**N8 `axes/4-moe/4.3-expert-granularity/project.yaml:26`** —— 「使 (k+1)·h_ff 与 **P/E** 同时保持」自相矛盾：总参数 P 固定且 E 增大 ⇒ P/E（每专家参数）**必然下降**。
最小修：把「P/E」改为「总参数 P」。

**N9 README.md:7** —— 「本阶段 **0 GPU 训练**」：坍缩探针是 1200 步带 `opt.step()` 的 GPU 训练（`probe…json::timing_process_metric.total_s=1395.141`；TICKET §7 19:29:03→19:52:23 且 §9 已披露取锁）。非谎报，仅措辞。
最小修：「0 GPU 判决训练（无对照质量实验）」。

**N10 analysis.md:94/106/112 + TICKET §5** —— `WP2`/`WP3` 指 4.2/4.3，与 4.2/4.3 各自 `project.yaml` 的 `WP1` 编号冲突。
最小修：统一用「4.2-WP1 / 4.3-WP1」式定位。

**N11 analysis.md:97** —— 「GPU util 持续 94–100%」无落盘观测（JSON 无 util 字段），仅 TICKET §9 文字。最小修：注明来源或删。

**N12 analysis.md:97** —— 20 步墙钟 14.958 写在探针段落内（探针自身 `first_20_steps_s=11.949`）；README §6/analysis §5 归属正确。最小修：注明「取自 `timing_20.json`」。

---

## 不可核实项

1. **命令类断言（本复核无执行工具，未能自跑）**：`python3 tools/summary.py --check` / `tools/check_tree.py`、`git status --short`、`CUDA_VISIBLE_DEVICES="" python3 run.py preflight`。替代做法见「命令与结果」：静态套用 check_tree 规则 + 手算 + watchdog_diff。**TICKET §7/§8 自述的「summary.py --check 全绿（46 个 project.yaml）」与 `pass=true` 我未复跑**，不得视为已核实。
2. **committed range**：`watchdog_diff` 不含已提交区间；TICKET §8 无任何 `git add/commit` 记录，故我据以推断「工作树 = 全集」，但若 19:25–19:53 间存在未记录的提交，边界（第 8 条）与改动范围就无法完全核对。需主管 `git log --since=… ` 确认。
3. **arXiv 标题的联网核实**：本轮无网络，编号↔论文映射我以既有知识核对正确；HTTP 200 记录只见于 TICKET §4。
4. **GPU 实跑复现**：`probe_collapse_1200.json` / `timing_20.json` / `smoke_*.json` 的数字未在 GPU 上重跑（按契约不跑 GPU）；我只能做**内部自洽**核对（见第 2、3 条判词），无法确认这些数字确由本仓库当前代码 + 声明 seed/配置产生。
5. **README 判据线的「事先」性**：只有 TICKET §7 的自述时间线（19:29–19:31 写 README，探针 19:29:03 启动、19:52:23 出结果）；README 未提交、无时间戳锚点，无法机器验证。
6. **「GPU util 94–100%」**：无落盘观测（见 N11）。

---

## 命令与结果

**须由主管运行（我无执行工具）**

```bash
cd /home/z/vibe/pathlm
python3 tools/summary.py --check          # 期望：OK（46 个 project.yaml）；我静态套用 check_tree 规则未发现违规
python3 tools/check_tree.py               # 同上（纯校验）
git status --short                        # 期望：仅 axes/4-moe/*/project.yaml + 新增实验文件 + .scratch/65-axis4-moe/**
cd axes/4-moe/4.1-sparse-experts/experiments/2610-moe-arm-ledger-13m
CUDA_VISIBLE_DEVICES="" python3 run.py preflight   # 期望：pass=true、bitwise_equal=true、non_mlp_extra_keys=[]
CUDA_VISIBLE_DEVICES="" python3 run.py ledger      # 期望：与 ledger.json 逐字段一致
python3 -c "print(79837696/13602304, 42015232/13602304)"   # N1：5.869… 3.088…
```

**我已执行的核对（只读工具）**

| 动作 | 命令/工具 | 结果 |
|---|---|---|
| 改动范围 | `watchdog_diff --stat` | 仅 3 份 `axes/4-moe/**/project.yaml` 改动；未跟踪文件全部在本次两个目录内 |
| 共享面未动 | `grep -i "moe\|expert" pathlm/`、`grep -i moe tools/` | 均无匹配 |
| 引擎接口 | `grep` on `pathlm/model.py:59-73`、`pathlm/depth_ar.py:1832/1995-2024` | `Block.mlp = Sequential(Linear,GELU,Linear,Dropout)`；`DepthARModel.blocks` 与 `aux["depth_logits"]` 存在 ⇒ MoE 透明替换接口成立；preflight 权重装载键名（`0.weight/0.bias/2.weight/2.bias`）与专家子模块同构 |
| 记账独立手算 | 解析式复算 | dense FFN 9,458,688 = 落盘；MoE(E=8,k=2) 37,871,616 = 落盘；五臂 non-FFN 参数均为 4,143,616 |
| 斜率复算 | 探针 13 点最小二乘 | 1.537e-5/步（文档写 +1.5e-5/步） |
| 池化不可分辨性 | 手算 | 1 层全坍缩 ⇒ 池化最大占比 0.198、归一化熵 0.990（落在观测区间内） |
| 名称/出处一致性 | `grep docs/structure.md`、`grep ^id: "4`、`grep 1.5074\|0.0138\|0.0019` | exact wording 一致；id 唯一 3 条；1.5074 有唯一家；0.0019/0.0138/0.014 与 methodology §2 一致 |

**结论重申：PASS-narrow（0 阻断）；WP0 四条验收在落盘证据上全部成立；N1/N2 建议在交付前各改一行，其余为报告级建议。上述 6 项「不可核实」中，第 1、2 项须在合并前补跑/确认。**