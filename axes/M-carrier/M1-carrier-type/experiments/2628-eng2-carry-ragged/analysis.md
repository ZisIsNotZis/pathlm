# 2628-eng2-carry-ragged · 判读（analysis）

## 0. 人话摘要（先说结论，再给数据）

**这是干什么的**：这条线**不追求性能**，只解决一个工程问题——本仓库里那个「把信息
沿位置往右搬的写入通道」（`carry`），以前**只允许在普通 decoder-only 形态（`dense=True`）
下打开**；一旦你要用真正有产品意义的深度自适应形态（每个位置随机决定跑几层、早退位置
的 KV 用冻结表征填充），引擎会**直接报错**。于是「写入通道 × 随机退出 × 冻结填充」
这个组合**从来没被跑过一次**。本线把它从「不存在」变成「存在且可测」，并且给
「已退出位置到底写不写、写什么」定下了明确语义。

**相对普通的 decoder-only transformer 改了什么**：改的是**引擎的校验与块的调用约定**，
不改任何模型公式。具体三处：

1. 删掉一条硬校验（`carry_rule` 必须 `dense=True`）；参数量零变化。
2. 让带通道的块 `CarryChannelBlock` 支持非 dense 路径特有的三个入口：
   `kv_from`（早退位置的混合 KV 源）、`gate_vec`（逐通道门）、`attn_mask`（ragged 掩码）
   —— 与 stock `Block` 逐项对齐；参数量零变化。
3. 非 dense 前向把块的活性量收进 `aux`（以前只收 dense 的），否则通道在退出路径上
   「只有权重、没有活性读数」。

**为什么要改**：这是**可测性**问题，不是质量问题。组合不存在 ⇒ 任何想验证
「`carry` 能不能和早退/填充一起用」的实验都无从下手（ticket 44-X2 被迫把它降级成
`dense-aux` 的替代物）。

**怎么测的**：① CPU 组合自检（4 个组合各建模型，量通道是否真的被读/退出是否真的生效/
填充 adapter 是否真被消费/写入是否与退出模式无关）；② 7 条可失败单测 + 5 处变异验证
（故意破坏 ⇒ 抓红 ⇒ 逐字节还原）；③ 13M 引擎 20 步 GPU 冒烟（四者齐开）；
④ 全量 pytest 回归。

**测出来多少**（带参照系）：

| 量 | 读数 | 参照系 / 人话量级 |
|---|---|---|
| `carry × ragged / 随机退出 / fill / proj-fill` 可建 | **4/4 可测** | ENG2 前 4 个里 **4 个都不可测**（直接 `ValueError`） |
| 通道真的被读（`carry_read_share`，非 dense 路径） | **0.0034 ~ 0.0037**（四组） | γ 置零时**恒等**；>0 即「不是摆设」（量级参照：dense 路径 0.0034 同阶） |
| 退出即冻结（`frozen_invariant_ok`） | **4/4 True** | 若实现静默回落 fill 或掩码漏筛，会 False |
| 混合填充 adapter 被消费（`max|Δlogit|`） | **0.00287**（仅 proj-fill 组） | 非 proj-fill 组该量为 None；忽略 `kv_from` 时该量为 **0** |
| 写入时机与退出模式无关 | **3 种退出模式下槽逐位相同** | 若写入按退出跳过，会不同 |
| 20 步冒烟（13M 非 dense + carry + proj-fill） | 峰值显存 **861.3MB**、墙钟 **0.05 min**、读出增益 **0 → 0.000225** | 共享卡礼貌线 5GB；20 步太短，`alive` 仍 false（与 ENG1 同口径） |
| 回归 | `test_depth_ar.py` **134/134**；全量 `pytest tests/` **549 passed / 0 failed** | 变异 5/5 抓红 + 5/5 逐字节还原 |

**结论与下一步**：**采纳（可测性）**——「carry × 随机退出 × proj-fill」现在可以跑实验了，
写入时机有明确定义且被单测钉住。**但本线不给任何质量结论**：通道与早退一起用是否划算，
需要另开实验（本线只交付入场券）。未测项与后续项见 §9。

---

## 1. 命题与判据（一句话）

命题：**ENG2 前「carry × 随机退出 × proj-fill」在引擎里不存在（硬校验 + 块拒绝混合
深度调用）⇒ 从未被测过；本线让它可测，且缺省路径逐位还原 v1。**
判据线逐字抄录于 `README.md` §4，本节只做判读。

## 2. 四个组合的逐格读数（`results.json::combos`）

CPU 测试位点：`d=32 / 3L / 4H / dh=8 / seq=12 / dense=False / carry_rule=delta /
carry_scale=0.5`；退出模式见每行。**全部是活性量（布尔或占比），不是质量读数。**

| 组合 | `fill_kv` | `proj_fill` | 块类 | 通道进 aux 层数 | `carry_read_share` | 通道改 logits | 退出即冻结 | 退出 vs 全深度不同 | adapter 消费 |
|---|---|---|---|---|---|---|---|---|---|
| `carry_x_ragged` | False | False | `CarryChannelBlock` | 3/3 | 0.003395 | True | True | True | — |
| `carry_x_random_exit` | False | False | `CarryChannelBlock` | 3/3 | 0.003416 | True | True | True | — |
| `carry_x_random_exit_x_fill` | True | False | `CarryChannelBlock` | 3/3 | 0.003651 | True | True | True | — |
| `carry_x_proj_fill` | True | True | `CarryChannelBlock` | 3/3 | 0.003651 | True | True | True | **0.00287** |

- **通道真的活着**：把 `carry_scale`（单边零起步）置零后 logits **逐位不同**
  ⇒ 读出确实参与前向（`carry_changes_logits=True`，四组全成立）。
- **退出语义真的生效**：混合退出模式的 logits 与「全深度」**逐位不同**（非退化点，
  SCHEMA §17.16 的教训：dense 分支会完全忽略 `exit_depths`，ticket 44-X2 已实测那种
  「死开关」形态）。
- **填充 adapter 只在 proj-fill 组被消费**：非 proj-fill 组该量为 `None`；proj-fill 组
  把 adapter 从恒等换成**换方向**的矩阵后 `max|Δlogit| = 0.00287 > 0`。
  **口径提醒**：`kv` 源要过 `ln1`（LayerNorm），**恒定正缩放会被 LN 吃掉**（实测
  1.3× 缩放的 logit 差只有 1e-7 量级）⇒ 测试必须「换方向」而不是「缩放」，否则是
  退化测试。这条坑已写进单测注释与 `controls/README.md`。

## 3. 写入时机的语义选择（ENG2 的关键设计点）

**选择**：记忆槽在 `DepthARModel._input_states` 里对**输入流 h₀** 因果扫描**一次**，
发生在**任何层与任何退出之前** ⇒ **所有位置（含已退出位置）都写各自的 h₀（depth-0
表征）**，不按退出深度跳过、也不改写成冻结表征。

**证据**（`results.json::write_timing`）：三种退出模式（全深度 / 固定混合 / 全 0）
下 `_carry_box["mem"]` **逐位相同**（两条布尔均 True）。单测
`test_eng2_carry_write_timing_ignores_exit_depths` 钉住；变异 M5（把 depth-0 退出位置
的写入抹掉）**抓红**。

**为什么这样选**（三条）：

1. **可缓存/可部署**：槽内容与退出模式**无关** ⇒ 同一个 token 在任何退出模式下得到
   同一段记忆，可在 prefill 一次算好；若写入依赖退出模式，decode 时每换一次退出
   模式就要重扫。
2. **语义一致（不会让已退出位置继续跑）**：读取的生效范围由调用方的
   `where(active, out, frozen)` 决定——位置 i 读到的记忆只影响 `depth ≤ d_i` 的层，
   更深层的块输出被冻结表征替换 ⇒ 「已退出位置也写」**不会**让已退出位置继续参与
   更深层的自注意力。
3. **与 fill 对齐**：fill 让已退出位置的**冻结表征**进 KV；通道则让它们的 **h₀** 进
   记忆——两者都是「已退出位置也贡献」，语义自洽。`proj_fill` 的 adapter 是**注意力
   KV 源**的 remap，**不**经过通道（通道有自己的 w_k/w_v/w_q/β）。

**取舍（另一条路及其代价，如实记账）**：若改成「按退出深度写冻结表征」，则槽必须
**逐层**重算、且内容依赖退出模式 ⇒ 既不能 prefill 一次算好，也会让同一序列在两种
退出模式下得到不同记忆（破坏单因素对照）。本线选前者。

## 4. 缺省路径逐位还原 v1

`results.json::default_untouched`：缺省（`carry_rule=None`）非 dense 模型的块类 = `Block`、
`has_carry_module=False`、`has_carry_box=False`、`aux` 里无 `attn_carry`。
`pytest tests/test_depth_ar.py` **134/134 绿**（含既有的 v1 复现用例：`test_nofill_*`、
`test_proj_fill_identity_init_equals_fill`、`test_block_kv_from_is_noop_when_equal` 等）。

**（后于复核）证据类型标注**：本节的「逐位还原 v1」证据是**结构性**的——块类/参数字典/`aux`
断言（`test_eng2_default_path_bitwise_v1` 名含 `bitwise`，但断言只到结构层）+ 缺省前向**代码
路径未被本线改动**；**未**对缺省前向 logits 做逐位 golden 对照。故更严的说法是「缺省路径
**结构性还原 v1**」。见文末强断言证据标注。

## 5. 单测与变异（硬验收 ①）

- 新增单测（`tests/test_depth_ar.py` §ENG2 段，7 条）：
  1. `test_eng2_carry_x_ragged_layer_matches_manual_rule`（模型前向 == 手写 ragged+通道
     规则；掩码/逐通道门/读出/读出进 proj 四处逐项对照）；
  2. `test_eng2_carry_x_random_exit_composes`（退出即冻结 + 与手写规则一致 + 通道活着）；
  3. `test_eng2_carry_x_random_exit_x_fill_consumes_frozen_kv`（adapter 换方向 ⇒
     logits 必变）；
  4. `test_eng2_carry_x_proj_fill_identity_starts_equal_to_fill`（恒等起步 == fill 臂）；
  5. `test_eng2_carry_write_timing_ignores_exit_depths`（写入时机，见 §3）；
  6. `test_eng2_default_path_bitwise_v1`（缺省不含通道结构）；
  7. `test_eng2_decoder_rejects_carry_channel`（增量解码显式拒绝，见 §7）。
  另把 `test_carry_config_rejects_combinations` 里「`dense=False` 必须报错」的旧负向
  断言改为正向断言（与 ENG1 的处置一致；判据线未改）。
- 变异（`mutate.py` → `mutation_report.json`）：**M1–M5 全部抓红、全部逐字节还原**
  （`all_caught_red=true`、`all_restored_bitwise=true`、`sha256_restored ==
  sha256_original`）。对照表见 `controls/README.md`。
  **执行口径**：变异在**仓库副本**上跑（共享卡上另有一条线在各臂之间 import 同一份
  `pathlm/depth_ar.py`，按 SCHEMA §22 避免争写）；副本的 `depth_ar.py` 与仓库逐字节
  相同（sha256 均为 `45f94679…8c585`），还原也逐字节核对。

## 6. 20 步 GPU 冒烟（硬验收 ③）

配置：`d=256 / 12L / 8H / mlp 6 / seq 256`，**非 dense**、`--exit-mode uniform`、
`--carry-write delta --carry-chunk 64 --proj-fill`、合成关联回忆（`assoc-mqar`）、
20 步 × bs8 = **0.041M token**。

| 量 | 读数 | 口径 |
|---|---|---|
| 参数量 | **14,590,568**（形态件 carry 写门 + proj-fill adapter 计入） | 不是 13.08M：那是 dense 无形态件口径 |
| 峰值显存 | **861.3 MB** | ≪5GB 礼貌线（nvidia-smi/torch 两口径见 `smoke/master/results.json`） |
| 墙钟 | **0.05 min** | 20 步被启动开销主导，**不外推**（SCHEMA §17.5） |
| `attn_carry` 落盘 | step 0 与 step 19 各一行，`n_layer_samples` 12 → **240**（12 层/步 × 20 步） | 活性量每 200 步同行的口径 |
| 读出增益逃逸 | `read_scale_absmean` **0 → 0.000225**（step 0 → step 19） | 单边零起步是否已松动 |
| 读出占比 | `read_share` **0 → 1e-06** | >0 即「不是摆设」 |
| 写入真的改状态 | `write_rel` **0.473 → 0.462** | ‖ΔS‖/‖S‖（块 2..n 均值） |
| β | 0.499 | 起步 ≈0.5（σ(0)），未退化 |
| `carry_liveness.alive` | **false** | 20 步太短：`rel_dev_vs_channel_off` 四舍五入到 0.0；与 ENG1 的 20 步冒烟同口径 ⇒ **只证明共存可训练，不外推通道质量** |
| 训练 loss | 2.0071（step 19） | 20 步快照，无统计含义 |

**注**：`smoke/master/results.json::carry_liveness.write_rel = 0.0` 是**探针口径**问题——
`_kv_head_liveness` 的探针用 `min(seq_len,64)=64` 个 token = 1 个扫描块，
`write_rel` 只在块 2..n 上有定义（单块时取 0）；训练日志里的 `write_rel`（0.46）才是
该量的有效读数。这是**既有探针**的已知口径，不是本线引入的。

## 7. 明确的范围边界（不静默）

- **增量解码（KV cache）路径未接入通道读出**：`DepthARDecoder.step` 只算注意力输出，
  不会加 `γ ⊙ mem_t` ⇒ ENG2 前 carry 只允许 dense（解码器只用于非 dense）所以二者
  碰不上；ENG2 后可以碰上了，因此**显式拒绝构造**（`ValueError`，同 GQA 的处置），
  不交付「看起来能用、实际少了一路」的路径。单测
  `test_eng2_decoder_rejects_carry_channel` 钉住。**解码侧 carry = 后续项**。
- **`carry × GQA × 非 dense` 仍不可测**：本线只解耦 carry；GQA 仍要求 `dense=True`。
- **`train_depth_ar.main` 的 enwik8 评测电池**在「非 dense + 合成任务」下此前会崩
  （`depth_curve(eval_arr=None)`）——本线加了 `synth is None` 条件，使该组合的训练入口
  可用（非 dense + enwik8 的电池路径不变）。这是冒烟能跑起来的前提，不改任何数值语义。

## 8. 三道对账（SCHEMA §12）

`reconcile_numbers.py` → **75/75 pass**（`reconcile_report.json`）：

1. **算术对账**：每组合的 `carry_read_share > 0`、`carry_changes_logits`、
   `frozen_invariant_ok`、`n_carry_layers_in_aux == n_layers`；adapter 消费量只在
   proj-fill 组为正。
2. **任务对账**：四组合同一非 dense CPU 位点（d/layers/heads/seq/seed 一致）；
   `fill_kv`/`proj_fill`/退出模式与组合名一一对应；退出模式非退化（vs 全深度不同）。
3. **口径对账**：缺省不含通道结构；写入时机与退出模式无关；冒烟的 `carry_rule/fill_kv/
   proj_fill/块类/step 19/attn_carry` 全部对应；回归无 fail；变异全抓红且 sha 还原。

**禁外推声明（§17.11）**：本实验的所有数字都是**确定性 CPU 前向的活性量**或**20 步
冒烟快照**——不构成任何「通道在早退形态下有效/无效」的趋势或性能结论。

## 9. 未测项与下一步最小实验

| 未测项 | 为什么没测 | 下一步最小实验 | 成本（估） |
|---|---|---|---|
| **质量结论**：carry × 随机退出 × proj-fill 在合成关联回忆上的召回 | 本线只交付可测性（scope） | `--carry-write delta --proj-fill` vs `--proj-fill` 两臂，13M、同 token 预算、合成 `assoc-mqar`，报**训练分布内档**召回 + 毁源对照（§17.18/§19） | 2 臂 × ~12k 步 ≈ 2 × 25 min |
| **tf@4 出口资产**是否在 carry 开启下保持 | 同上 | 复用 `tf_acceptance`（全深度前向），与 100M 配方同口径 | 随上一条一并评测 |
| **增量解码侧 carry** | 解码器未接入读出（本线显式拒绝） | 在 `DepthARDecoder.step` 里维护槽状态 S 并加 `γ⊙S_t q_t`；用现成的 `test_decoder_matches_parallel_forward` 钉「step == 并行前向」 | 引擎级小改 + 单测，~1 波 |
| **carry × GQA × 非 dense** | GQA 仍 dense-only（单因素） | 把 `_gqa` 分支与 `ComposableBlock` 的 `kv_from/gate_vec` 一起对齐 | 引擎级小改 |
| **RNG 口径的回归面**：proj-fill adapter 不再消耗随机数后，是否有历史 proj-fill 组合的 seed 读数受影响 | 该组合（proj-fill + carry/readout 几何/recon）在已入库实验中未出现 | grep 全部已入库 `proj_fill=true` 配置核对（本线已核对：0 命中需重跑） | CPU，分钟级 |

## 10. 相关规范与教训

- **SCHEMA §22（共享文件）+ §22.8（能力面分析）**：本线先做了能力面分析，确认
  「carry ∧ dense=False」是**硬校验**而「carry × GQA」是**块类二选一**，二者性质不同
  ⇒ 本线只解耦前者（后者留后续项），不越权绕过引擎。
- **§10（活性判据）**：参数化干预必须报「是否真的被激活」——本线把非 dense 路径的
  `aux["attn_carry"]` 接上，并逐组合量 `read_share`/`carry_changes_logits`。
- **§17.16（两点同刻度、排除退化点）**：`kv` 源要过 LayerNorm，「恒定缩放」是退化扰动
  ⇒ 测试改用「换方向」扰动。
- **§9（对照复用与 seed 逐字段）**：发现 proj-fill adapter 构造消耗随机数会让
  `carry×fill` 与 `carry×proj-fill` 的 carry 权重不同 ⇒ 修复后同 seed 逐位一致。

---

## 强断言证据标注（后于复核：W3 强断言纪律）

> 依据（本轮 W3）：报告里的「逐位 / 唯一 / 首次 / 保持 / 稳健」等**强断言**必须附**可核对的
> 证据类型**（**数值 golden 对照** / **结构性断言** / **统计检验**），**用词不得超出证据**。
> 本节**只标注证据类型与措辞边界，不改任何数字、不改任何结论**；正文与判据线原样保留。
> 机检入口：`reconcile_numbers.py` → `reconcile_report.json`；单测：`tests/test_depth_ar.py` §ENG2；
> 变异：`mutation_report.json`。

| 断言（原文用词） | 证据类型 | 是否达标 | 若否则降级表述 |
|---|---|---|---|
| 「缺省路径**逐位还原 v1**」（§1/§4/README §2） | **结构性**：缺省（`carry_rule=None`）块类 = `Block`、`has_carry_module/_box=False`、`aux` 无 `attn_carry`（`results.json::default_untouched`）；+ 缺省前向**代码路径未被本线改动** | **未达「逐位」等级**——单测 `test_eng2_default_path_bitwise_v1` 名含 `bitwise`，但断言只到结构层，**未做缺省前向 logits 的 golden 逐位对照**（对照 `test_*_default_off_matches_v1_bitwise` 类用 `torch.equal`） | 「缺省路径**结构性还原 v1**（块类/参数字典/aux 无通道项；未做缺省前向 logits 的逐位 golden 对照）」 |
| 「通道真的被读/改前向（非摆设）」（§2） | **数值 golden 对照**：`carry_scale` 置零前后 logits **逐位不同**（`carry_changes_logits=True`） | **达标** | — |
| 「写入时机与退出模式**无关**」（§3） | **数值 golden 对照**：三种退出模式下 `_carry_box["mem"]` **逐位相同**（`write_timing` 两条布尔 True） | **达标** | — |
| 「变异 M1–M5 **全部抓红、逐字节还原**」（§5） | **数值 golden 对照**：`sha256_restored == sha256_original`（`mutation_report.json::all_restored_bitwise`） | **达标** | — |
| 「四组合 **4/4 可测**」（§2） | **结构性**：`combo_buildable` + 块类断言（ENG2 前为硬 `ValueError`） | **达标** | — |
| 「同 seed **逐位一致**（修 RNG 后）」（§5/§10） | **数值 golden 对照**：`test_eng2_carry_x_proj_fill_identity_starts_equal_to_fill` | **达标** | — |
| 「**唯一**被改动的引擎结构 = 写入通道在非 dense 路径上的可组合性」（README §2 单因素） | **结构性 + 回归**：缺省路径结构未变（`default_untouched`）、单特性块类不变（`Block` / `CarryChannelBlock` / `ComposableBlock`）、`test_depth_ar.py` 134/134 绿 | **达标** | — |
