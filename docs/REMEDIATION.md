# 修复与重跑总清单（2026-10-02 用户授权）

> 用户指令：**把问题全记下来 → 修所有代码 → 重跑該重跑的 → 报最新结果**，并且**先把「要跑什么」想清楚**。
> 本文件是执行台账。冻结清单见 `docs/OPEN_PROBLEMS.md §3`，复核全文见 `.scratch/49-adversarial-review/REPORT.md`。

---

## 0. 我的错误是什么（用户问，答在此，避免再犯）

1. **我把「不稳的东西」当好消息汇报**：T=2048「同收敛」、量化「int8 近无损」、`delta` 写「稳定有效」——
   三条都在冻结清单上，而我当时都当成正面结论报给了用户。
2. **我在自己的方法学文档里写错了一条判据**（`SCHEMA.md` 的「算术下界」= 两个边际相乘）——
   而这条错误判据被下游用来宣称「实现正确」。**错误发生在标准本身，比错误发生在实验里严重。**
3. **我把「测了另一个命题」当成「回答了用户的命题」**：H3 的 `T∘T` 潜在复用 ≠ 概率路径组合，
   我却说「Spec 已有一半答案」。
4. **我把 `git log` 当证据**：提交信息多次超出实际改动 ⇒ 不能再把提交信息当可读汇报。
5. **我允许「格式校验通过」被读成「研究验收通过」**：`check_schema` 只查形状，`@ticketNN` 假锚点 50 处没人管。
6. **我用单点/单种子支撑「定律」级措辞**（「保留型近乎免费」= −0.0068 一个点，连自家噪声界都没过）。

---

## 1. Spec（投机解码）：要跑什么（用户要求先想清楚）

### 1.1 要回答的问题
> 在**同一算力预算**下，「用浅层出口/小模型做草稿，再让全深度验证」能否比**原生逐 token 自回归**更快？
> 输出是否**保持不变**？哪条**草稿路径构造**最划算？

### 1.2 三种草稿路径构造（**这是用户指出的核心，必须分开做臂**）
| 臂 | 定义 | 用户原话对应 |
|---|---|---|
| `chain` | 先出 N+1，**以 N+1 为条件**再出 N+2（自回归草稿，深度 γ） | 「用 N+1 再加一」 |
| `direct` | 从同一状态**一次直接**出 N+2（多 token 预测头） | 「直接用 N+2」 |
| `combined` | 上面两条的**概率分布按概率混合**（或保留为树的候选） | 「它用概率组合」 |

反例警示（写进契约）：**H3 的 `T∘T` 不是这三者中的任何一个**（无概率、第一遍不受 N+1 监督）⇒
该实验结论不得用于回答本问题。

### 1.3 指标（缺一不可；用户点名的量在前面）
1. **逐步草稿接受率 α_i**：第 i 个草稿位置被全深度接受的比例（i = 1..γ）。**必须给每个位置，不只给平均。**
2. **平均接受长度 τ** = 每次验证平均被接受的 token 数（用户点名的量）。
3. **每次前向产出 token 数**（tok/fwd）。
4. **墙钟加速比 vs native**（tok/s；同硬件同批量；**必须给 CI**）。
5. **等效算力净账**：草稿 + 验证的 layer-position 总成本（**必须把草稿自身成本算进去**）。
6. **分布保真**：用拒绝采样 ⇒ 声明分布保持并给逐位验证；用 argmax ⇒ 明确声明「非分布保持」。

### 1.4 判据（预登记）
- **主判据**：墙钟 tok/s 加速比（同硬件、同评测评测流、同批量）。
- **次判据**：平均接受长度 τ、tok/fwd。
- **门禁**：与 native 的输出**逐位相同**（argmax 路线）或**分布保真**（采样路线）。
- **规模**：**至少两个点**（13M + 100M 档）。理由：13M 的墙钟收益已被判定为「启动开销主导」，
  单点会重犯旧错。
- **复现**：**至少两次独立训练**（或明确固定 checkpoint 范围并声明只测评测期）。
- **禁外推**；快照口径须标注。

---

## 2. 修复清单（代码与工具；按文件）

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| F1 | **量化 scale 沿 batch+token 取 amax** ⇒ 非因果、跨请求耦合 | `tools/evalkit/kv_quant.py` | 改为**逐请求、因果分块/流式**量化；scale 只能用 ≤ 当前时刻的信息；明写缓存旧值是否重编码、scale 存储开销 |
| F2 | 缺**前缀不变性**测试 | `tests/test_kv_quant.py` | 加两条：① 改未来 token 不改过去 logits；② 改同批其他请求不改本请求 logits |
| F3 | 变异框架把**任意非零返回码**当抓红；未抓到不报错 | `*/mutate.py`（全仓统一形制） | 区分「测试失败 / 收集错误 / 环境错误」；未抓到 ⇒ **非零退出** |
| F4 | 对账脚本把期望值写成常量；`tol=1.0` 放大成目标值量级误差 | `*/reconcile_numbers.py`、`2621-depth-draft-e2e/reconcile_numbers.py` | 期望值从原始计数/日志**推导**；tol 语义写清并收紧 |
| F_KEY | 「缺失证据静默 skip」+ 证据在被 git 忽略目录 | `tests/test_longT_stability.py:233–238`、`tests/test_longT_clean.py:202–203`、`.gitignore` | 缺证据 ⇒ **显式失败**（或明确 xfail 并在 CI 汇总）；必要证据入 git |
| F_SCHEMA | 校验器只查文档形状，却被读成研究验收；`@commit` 不校验 | `tools/check_schema.py`、`catalog.yaml` **50 处 `@ticketNN`** | 改名「格式门禁」；**校验 `@<sha>` 是否为真实提交**；50 处锚点换成真实 sha |
| F_YAML | `catalog.yaml` **重复键 `local`**（后者静默覆盖前者） | `catalog.yaml:203,206` | 合并；CI 加**重复键检测**（严格加载） |
| F_JITTER | 分量公式更正未落全（run.py + results_13m.json 的旧 note） | `2628-jitter-l1/run.py:281`、`results_13m.json:3505–3511` | 同步为 `net = (repr01_gain − l1_gain) − clean_tax` |
| F_CLOSURE | 收口文称 ENG2 已解锁 `carry × GQA × 非稠密`，代码仍拒绝 | `G5-parallel-lines/CLOSURE.md:75` vs `pathlm/depth_ar.py:1904–1908,2568–2576` | 撤回「引擎就绪」表述（或实现它，另立工单） |
| F_B12 | 目录源含重复 YAML 键、`partial` 格状态与内容不符 | `catalog.yaml` | 逐格校正状态语义 |

## 3. 结论级清账（文字，不改数字）

| # | 条目 | 位置 | 处置 |
|---|---|---|---|
| C1 | 「AR 算术下界 = 两边际相乘」 | `SCHEMA.md:168–170`、`2611-jumphead-ar-control-13m/analysis.md:61–72` | 改为正确的关系（交集下界 = max(0,a+b−1)）；**撤回「对账通过 ⇒ 实现正确」**与「0.3128<0.4822 是铁证」 |
| C2 | 「距离腿六档全部 > chance」 | `catalog.yaml:1013`、`2618-carry-channel-13m/analysis.md:71,241–245` | 改为事实：**d=32=0.0000、d=192=0.00098 低于 chance 0.015625**；曲线非单调 |
| C3 | 「delta 写是稳定有效的那一支」 | `catalog.yaml:1009` | 降级为**单次观测**（训练侧种子散布未知，`2618/analysis.md:305–307`） |
| C4 | 已降级 OOD 腿仍作入口主判据 | `catalog.yaml:1009` | 首验降级为**分布外单次观测**；入口改用分布内证据 |
| C5 | 「T=2048 同收敛 / 欠训练混杂已排除」 | `catalog.yaml:339`、`2629-longT-2048-matched/analysis.md:36–37,103–105,184–188` | 改称「**同 step 的第二快照**」；撤「已排除收敛解释」（末窗斜率仍 −0.16/1k；token 预算 49.15M vs 98.30M） |
| C6 | 「自门控组合被否证 ⇒ Spec 有部分答案」 | `catalog.yaml:687–706` | 限定为「冻结表征上的重复潜在映射与门控」；**不回答用户概率组合问题** |
| C7 | 「−0.0068 唯一改善」 | `catalog.yaml:281–283`、`docs/methodology.md:13` | 改为「**未观察到超带损伤**」（<0.014 不可排序） |
| C8 | 「任何侵入主干的结构都有税」「速度只能走成熟旁路」 | `docs/design.md:23–24`、`README.md:29–31` | 降级为「当前证据支持的倾向」 |
| C9 | beam search 被否证 | `docs/playbook.md:27–33`、`catalog.yaml:1174–1176` | 撤回（确定性重复推理 ≠ beam） |
| C10 | 「bpc 上界 = log₂(206)」 | `2613-inter-layer-topology-13m/analysis.md:250–251` | 改为「均匀预测参考值」，不与理论边界混 |
| C11 | 「抖动比 3pp 大」（实为 1.367pp） | `2619-reconstruction-loss-13m/analysis.md:27–30` | 撤回该失败归因 |
| C12 | 「给路径优于给信号」无等参数对照 | `2619-reconstruction-loss-13m/analysis.md:40–43` | 降级为**假设**（参数 +264296 vs +52736） |
| C13 | bpc 的 0.014 当准确率 1.4pp 界 | `2613-self-gated-composition-13m/README.md:82–90` 等 | 改称「预登记的实用阈值」，不得称「已测方差」 |
| C14 | `reference` 格用「记忆·未核」判 `(a)` | `catalog.yaml:306–310,561–565,799–803` | 按自家分诊规则改判 `(c)`（待核） |
| C15 | 根 scope 称「不做组合」但已有组合实验 | `README.md:20–22`、`SCHEMA.md:76–80`、`catalog.yaml:22` | 同步例外条款与授权边界 |
| C16 | `STATUS.md` 停在 ticket 42 / 394 测试 | `STATUS.md` | 刷新到当前（47/595+） |
| C17 | 自动生成页自相矛盾 | `axes/**/README.md`（源 = catalog `local_summary`） | 修 catalog 源文本后重生成 |

## 4. 重跑清单

| # | 重跑什么 | 为什么 | 成本 |
|---|---|---|---|
| R1 | **KV 预算图全组合**（因果/流式量化后重测） | A2：旧价签非部署有效 | **评测为主**，便宜 |
| R2 | **搬运三臂（无通道/plain/delta）多次重训** | A3/A4/C3：现结论是单次观测且在分布外腿上 | 训练，13M×3 臂×≥2 seed |
| R3 | **越界腿/分布内腿方差量化**（同配置多跑） | 旧「极差」不是方差上界 | 训练+评测 |
| R4 | **长 T 四点同预算/同 token** | C5：现四点口径不齐（49.15M vs 98.30M） | 训练，4 点 |
| R5 | **读出改动的更长训练点** | B5：报增益未履行「须有 2× 点」 | 训练 |
| R6 | **Spec 三臂（chain/direct/combined）** | 用户主用例，见 §1 | 训练 + 评测，≥2 训练 + 2 规模 |
| R7 | **量化 5/6 位与旋转/投影类** | P-14（**必须在 R1 的天平修好之后**） | 评测，便宜 |
| R8 | **递归/线性载体（SSM/Mamba 族）质量** | P-15：最根本的信息右移路径，完全空白 | 需先解决线性注意力实现停摆 |

## 5. 验收标准（本轮）
1. 修复清单 F1–F_B12 全部落地，且**每条有可失败单测**（改前红、改后绿）。
2. 全文 `pytest` 绿；`check_schema`（格式门禁）绿；**假 `@commit` 锚点为 0**。
3. 结论级清账 C1–C17 全部落地，catalog/生成页/STATUS 一致。
4. R1（量化）出**因果口径**的新数字，并明确与旧价的差异方向；R6（Spec）出 §1.3 的**六项指标**。
5. 每项重跑遵循：预登记判据 + 诚实对照（含毁源/等参数）+ ≥2 独立训练或明确范围 + 独立复核（**排在执行之后**）。
6. **报告口径**：先说「人话结论」，再给数字；代号只作括号定位；失败讲清「以为会怎样/实际怎样/为什么」。


---

## L1（instr）交回：语义决定与待办（2026-10-02）

> 本节由 L1（工具/测试线）追加，**只增不改**；上面的 F 清单仍是主台账。

### 1. F1 的语义决定（R1 必须据此重算价签）

F1 采用的是**严格逐 token 因果**的实现：位置 `t` 的 scale = `max_{t'≤t}|x_{t'}|/qmax`，
**不重编码、无 fp16 残差窗口**。代价必须明写：

- **scale 存储开销 = 每值一个 fp16 scale**（`[B,H,T,head_dim]` 与 K/V 元素一一对应）
  ⇒ int8 净 **3 B/值**、int4 净 **2.5 B/值**（fp16 基线 2 B/值）。
  **旧的「int8 省一半」在严格因果口径下不成立**。
- 想要「真省内存」只能用**分块 scale + 有界重编码窗口**（每 G 个 token 摊一个 scale，
  块关闭时回填一次）——但那会让**块内**过去值随同块后续 token 变，牺牲严格前缀确定性。
  该口径**未获批准**，本线不实现；R1 若要它，请先立决定。
- 组合入口（窗口/锚点/头共享，`2620-kv-budget-gaps/gaps_arms.py`、`2621-kv-budget-rest/rest_arms.py`）
  全部复用 `quantize_kv` ⇒ 自动同源；`kv_cache_footprint` 新增
  `scale_bytes_streaming_per_position` / `*_streaming_bytes_by_seq_len` 给因果口径总量，
  旧的 `per_position_bytes` / `total_bytes_by_seq_len` 保持值字节口径（与已入账 results.json 兼容）。

### 2. 本次未做、需要别的线接手的（不要当成已完成）

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| L1-1 | `axes/.../2621-carry-under-compression/carry_quant.py:12`、`2624-.../quant_carry.py:11`、`axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/README.md:32` | 模块文档仍写「(head, channel) 一个 scale、沿 batch/token 取 amax」——F1 后实现已改 | 这三处属各自线，请按 F1 后的口径更新文字（本线不越界改） |
| L1-2 | 其余 37 个 `*/reconcile_numbers.py` | 仍以 golden 常数作部分期望（例：`2619-kv-budget-chart:57–63` 的 `1.483569/12288/6144/3072/0.045661/0.0310`；`W3/2616:141–142` 的 `6144/12288`） | 按 F4 的 `tools/evalkit/reconcile.Ledger` 逐条改成 `derived(...)`，确实只能核对历史值的用 `golden(..., why=...)` |
| L1-3 | 其余 35 个 `*/mutate.py` | 已全部接入统一框架（分类 + 未抓到非零退出），但**本轮只抽样重跑了 4 个**（W4/2613、A6/2622、M1/2619、R1/2614）⇒ 其它 `mutation_report.json` 仍是旧 runner 的产物 | 各线重跑一次刷新报告（纯 CPU，多数 <1 min） |
| L1-4 | `.scratch/49-adversarial-review/REPORT.md` B12 提到的 `.gitignore` | `runs/` 仍不入 git（本次用「最小证据入 git + 缺证据 xfail」替代，见 `tests/evidence/`） | 若要把整段训练日志入 git，需另立决定（体积） |

---

## L5（GPU / R6）交回：引擎缺陷与待办（2026-10-02）

> 本节由 L5（Spec 三臂线）追加，**只增不改**；上面的 F 清单仍是主台账。

### F_ENG（**P0 级，本波新发现**）

**问题**：`pathlm/model.py::PathLM._mtp_nodes` 在 `chain_mode="token"` 的**单位置解码**下崩溃。
修复前该分支**无条件**把条件张量右垫一行：

```python
t_next = torch.cat([cond, cond[:, -1:]], dim=1)   # 无条件右垫
```

`h.shape[1] == 1` 时 `n_cond = max(h.shape[1] - 1, 1) = 1` ⇒ `cond` 已是 1 行、右垫后变 2 行 ⇒
下一行 `torch.cat([h, self.embed(t_next)], dim=-1)` 在维度 1 上 1 vs 2 失配
⇒ `RuntimeError: Sizes of tensors must match except in dimension 2`。

**影响面 = 所有单 token 解码与 prefill**（`pathlm/decode.py::Decoder.step()` 是唯一入口）
⇒ `chain_mode="token"` 下**连 native 对照都跑不起来**。推论（已写进
`docs/OPEN_PROBLEMS.md §3` 冻结清单第 8 条）：**ticket 12 的 token 条件链结论
（`chain deploy 0.547`、`accept 0.7525`）不是解码口径**——它来自 `probe_chain.py` 的
**批量 teacher-forcing 式**前向（B=32/T=512 走 `deploy_chain`，那条路径 T≥2 正常）。
该结论**降级**并冻结，待 R6 的解码口径结果替换。

**修法**（已按主 Agent 裁决 ① 落地，1 行，最小改动）：

```python
t_next = (cond if cond.shape[1] >= h.shape[1]
          else torch.cat([cond, cond[:, -1:]], dim=1))   # 只在行数不足时右垫
```

**证据（三条，全部落盘）**：

| 证据 | 位置 |
|---|---|
| T≥2 逐位不变（`max|Δ| == 0`，4 个用例 × h_c / chain2 logits；修复前/后 live 张量 sha256 逐条相等） | `axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/evidence/fix_mtp_nodes_T1/invariance_{pre,post}.json` |
| 可失败单测**先红后绿**（改前 3 红 / 改后 15 绿，模型文件 sha256 逐字节还原） | 同目录 `red_before_fix.txt` / `green_after_fix.txt` |
| 变异验证（7 条全抓红 + 逐字节还原；其中 2 条直接打在修复那一行） | `.../2631-spec-three-arms/mutation_report.json` |

**回归**：`python3 -m pytest tests/ -q` = **641 passed**（与开工时基线相同）。

**我们不知道的**：修复只解决「能不能跑」；`chain` 相对 `direct` 到底赚不赚，正是 R6 要测的。

### 本线待办（**不要当成已完成**）

| # | 位置 | 待办 |
|---|---|---|
| L5-1 | `catalog.yaml` 的 `H3-composition` 格 | **`TODO(主Agent)`：R6 结果出后更新该格的 `verdict`/`local`**（本实验取代 2613 作为该格「多位置预测的组合方式」的正式回答；2613 降级为背景实验）。L5 **未**改 `catalog.yaml`/`INDEX.md`/`axes/**/README.md`（本波分工）。 |
| L5-2 | `axes/H-horizon/H3-composition/README.md`（自动生成页） | 上面条目落地后 `python3 tools/build_wiki.py` 重生成（L2 或主 Agent 执行）。 |
| L5-3 | `pathlm/decode.py::decode_spec` | 它只支持「第 1 个草稿位」由 `draft_fn` 注入；**多级条件链**（N+3 也条件化）需要引擎新结构 ⇒ 记入 R6 的 `analysis.md` 下一步（**未实现**）。 |
