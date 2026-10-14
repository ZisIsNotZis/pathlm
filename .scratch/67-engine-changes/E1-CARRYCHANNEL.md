# E1 执行报告 · `CarryChannel` 逐层独立状态 + 内容写门 + 可选衰减 γ

> 性质：**执行报告**（engine-writer 交付物，落 session 工件目录；直接上游事实 = 仓库内
> `.scratch/67-engine-changes/TICKET.md` §2 E1 与本文件）。
> 执行者：engine-writer（worker，fresh 上下文，`pathlm/**` 单一写入者，持 `/tmp/pathlm_code.lock`；
> 纯 CPU，未取 GPU 锁）。**独立复核未在本 worker 内执行**（见 §5）。
> 提交：`e894f20`（分支 `main`）+ 更正提交 `d3d33eb`（仅改变更记录里的单测计数 8→9，不动代码/结论）。

## 0. 人话摘要（先读这段）

**改了什么**：原来「显式记忆槽」只有一个共享状态——**全部 12 层共用一块**、内容只从
**输入层 h₀** 写一次、而且没有遗忘。本轮把它补成**可选开关**下的完整机制：

1. **逐层独立状态**：开关打开后，每一层**用自己的那一层当前表征**去写自己的状态
   （写门 β、键、值都取自当前 hidden）。深层看到的是已经精化过的表征，所以各层状态
   天然不同——这才叫「每层一个可写状态」，也正是 Gated DeltaNet / GLA 的形状。
2. **内容相关写入门**：β = σ(W_β h + b) 现在按**当前 hidden** 算（原来是按 h₀ 算），
   即「该不该写」由这一层此刻看到的内容决定。
3. **读出增益逐层可学**：本来就有（每层一个 `carry_scale`），未改。
4. **可选衰减 γ**：状态每步先乘 γ 再写（`S ← γS + write`），γ=1 就是关。打开后旧写入
   随距离被压小——GLA 式遗忘因子。

**默认行为变没变**：**逐位不变**。两个新字段 `carry_per_layer=False`、`carry_decay=1.0`，
默认走的是**原有那几行公式**（γ=1 时不加任何乘子；不逐层时保持 `_input_states` 里 h₀ 扫一次
的老路径）。有单测直接钉：默认模型与显式关掉新字段的模型**参数字典与前向逐位相等**，
`carry_scan(decay=1.0)` 与不传 `decay` 逐位相等。

**门是不是真的能开**：能。单测抓住四件事——① 各层状态两两不同、且与单共享槽不同；
② 写门随不同 hidden 变化、随位置变化、不是常数门；③ 端到端改一个 token 会改动下游
读出；④ γ<1 真的改变前向且分块实装与逐位置参考实现在 fp64 下对到 1e-12。

**测试抓力**：新增 9 条单测，含一条 fp64「分块 == 逐位置」对拍（覆盖 γ∈{1,0.999,0.9,0.5,0.2}
× 两规则 × 多 chunk），一条「默认逐位等价」，一条「逐层状态独立」用 monkeypatch 记录每层
`write_read(layer_idx, mem)` 并断言两两不等。全量 `pytest tests/ -q` = **849 passed**。

**还没做的**：独立复核（fresh 上下文）——本 worker 无派发权，留给 Main。E1 只交付**引擎能力 +
证明引擎能力可用的单测**；「完整机制到底涨不涨质量」（WP2 实验）是后续实验项，不是本 patch。

## 1. 契约、范围与验收判据

来源：`.scratch/67-engine-changes/TICKET.md` §2 E1（= 55-review M-05/M-06 的最小 patch 需求）；
上游规则：`SCHEMA.md` §20（GPU 锁）/§22（代码锁、锚点、能力面 §22.8）/§14b（复核顺序）/§10（活性）；
`AGENTS.md` 的边界与提交纪律。

| 硬性要求（ticket 原文） | 落点 | 证据 |
|---|---|---|
| 逐层独立状态（每层一个 slot） | `CarryChannelBlock`/`ComposableBlock` 逐层支路 + `_wire_carry` | `test_carry_per_layer_states_are_independent` |
| 内容相关写入门（由当前 hidden 算） | 逐层模式下 β/键/值取自块输入 `x` | `test_carry_write_gate_is_content_dependent` |
| 读出增益逐层可学（已存在） | `blocks[k].carry_scale`（未改） | 既有 `test_carry_...` |
| 可选衰减 γ | `carry_decay` + `carry_scan*` 的 `decay` | `test_carry_decay_chunked_matches_sequential` / `..._forgets...` |
| 附加式、默认关闭、默认逐位不变 | 新字段缺省 + 保留原公式分支 | `test_carry_decay_one_is_bitwise_identity` / `test_carry_default_flags_are_inert_bitwise` |
| 局部追加、先声明锚点、改完回读 | 见 §3 锚点与 `git diff`（只加不重写） | 本文件 §3、§4 |
| 能力面分析（§22.8）：不互斥、不换 block 类；不支持则显式报错 | 校验段 + C17–C22 事实行 | `test_carry_e1_validation_and_decay_plumbing` / `test_engine_capability.py` |
| 单测 ①默认不变 ②逐层独立 ③门内容相关 ④活性/梯度 | tests/test_depth_ar.py 新增 8 条 | §4 命令与结果 |

**显式实现决策（与 ticket 最小 patch 描述的差异，需复核者盯）**：ticket 建议「状态扩到
`[n_layers, ...]`、`write_read(layer_idx)` 按层索引、写仍来自 h₀」。但**同一套写投影 + 同一
个 h₀** 下逐层槽会**逐位相同**，测不出「独立」，等于换了个张量形状的退化实现。因此本 patch
把「写入源」换成**各层当前 hidden**（`carry_per_layer` 的定义行为），逐层差异来自各自表征；
`write_read` 仍接受 `layer_idx`（逐层模式下由块传入并写进该层遥测）。这与 ticket 面向的目标
（M-05/M-06：修掉「只从 h₀ 写」的退化）一致，且不新增参数。**若复核认为必须再加「逐层独立写
投影」（每层自己的 `w_k/w_v/w_q`），那是另一个（改参数量/等参对照的）决策，未在本 patch 内做。**

## 2. patch 落点（`pathlm/depth_ar.py`）

- `DepthARConfig`：新增 `carry_per_layer: bool = False`、`carry_decay: float = 1.0`（带长注释）。
- `carry_scan_sequential` / `carry_scan_chunked` / `carry_scan` / `_carry_chunk_plain` /
  `_carry_chunk_delta`：新增 `decay` 参数；`decay == 1.0` 走**原公式**（分支外置，逐位不变）。
- `CarryChannel.__init__`：`self.decay`；`write_read(h0, rule=None, decay=None, layer_idx=None)`。
- `CarryChannelBlock` / `ComposableBlock`：`__init__` 加接线属性；`forward` 逐层支路按本层 `x` 自算
  记忆（缺 mem 时显式 `RuntimeError`）。
- `DepthARModel.__init__`：新 flag 的校验（死组合显式报错）；`_wire_carry` 逐层接线；
  `_input_states` 逐层模式下不建共享槽；`forward` 聚合逐层扫描遥测到 `model._carry_tele`。
- `_carry_scan_tele_mean`（新函数）：逐层扫描遥测按层聚合成与旧口径同形的 dict。
- `carry_liveness`：报 `per_layer` / `carry_decay`，`shared_slots` 不再恒 True；
  `carry_read_mass_stats`：逐层模式下 h₀ 派生的两个口径显式置 `None`。

### 衰减的并行式（风险最高的代数，已对拍）
带 γ 的 delta WY 形式：写侧把写强度与转移解耦——`A_dec[i,j]=β_i γ^{C−1−i} T[i,j]`；读出侧
`Ω·γ^{i−j}`、`s_i·γ^{i+1}`；状态 `γ^C S`。plain：`Ω·γ^{i−j}`、`y_state·γ^{i+1}`、写权重 `β_j γ^{C−1−j}`。
逐位置参考实现与分块实装都对 `S_t = γS_{t−1} + write`，`test_carry_decay_chunked_matches_sequential`
在 fp64 下钉到 <1e-12（多 chunk）。

## 3. 锚点声明与回读校验（SCHEMA §22.3 / §22.4）

改前声明锚点（行号按改前 `pathlm/depth_ar.py`，sha256 `1f8f9eb4…`）：
配置字段 `carry_state_dim`（L223 后）；`carry_scan_sequential`（L1324）；`_carry_chunk_plain`（L1357）；
`_carry_chunk_delta`（L1371）；`carry_scan_chunked`（L1410）；`carry_scan`（L1457）；
`CarryChannel.__init__`/`write_read`（L1491/1514）；`CarryChannelBlock.__init__`/`forward`（L1573/1583）；
`ComposableBlock.__init__`/`forward`（L581/607）；`carry_liveness`（L1631）；`carry_read_mass_stats`（L1784）；
`DepthARModel` 校验（L1943）；`_wire_carry`（L2147）；`_input_states`（L2207）；`forward`（L2255）。

**只做局部追加**：全部改动是「新增字段 / 新增可选参数 / 新增分支 / 新增函数」，未重写任何块；
默认分支保留原语句顺序（`decay==1.0` 与「无逐层」路径命中原代码）。**回读校验**：改后用
`grep -n "CARRY_RULES\|carry_scan\|write_read\|_wire_carry\|_carry_box"` 复核其它线的标记
（`ENG1`/`ENG2`/`carry_state_dim`/`recon_window` 等）仍在位；全量测试过。

## 4. 命令与结果（复现）

- 锁定：`/tmp/pathlm_code.lock`（退出时 `rm -f holder_pid/heartbeat` + `rmdir`，已释放；未取 GPU 锁）。
- `python3 -m pytest tests/test_depth_ar.py -q` ⇒ **134**（含新增前）→ 加测后 **143 passed**。
- `python3 -m pytest tests/test_engine_capability.py -q` ⇒ **24 passed**（C1–C22）。
- `python3 -m pytest tests/ -q` ⇒ **849 passed**（91.9s）。
- `python3 tools/summary.py --check` ⇒ `OK：46 个 project.yaml 全部符合 docs/research-schema.md`。
- `python3 tools/check_tree.py` ⇒ 同上 `OK`。
- `git diff --check` ⇒ 无空白告警。`git diff --cached --name-only | wc -l` ⇒ 5（人工核对：见下）。

新增单测（`tests/test_depth_ar.py`）：`test_carry_decay_one_is_bitwise_identity`、
`test_carry_default_flags_are_inert_bitwise`、`test_carry_decay_chunked_matches_sequential`、
`test_carry_decay_forgets_and_matches_recurrence`、`test_carry_per_layer_states_are_independent`、
`test_carry_write_gate_is_content_dependent`、`test_carry_per_layer_changes_forward_and_gradients_flow`、
`test_carry_per_layer_composes_with_gqa_and_liveness`、`test_carry_e1_validation_and_decay_plumbing`。

**staged 列表**（提交 `e894f20`，显式路径，无 `-A/.`）：
`pathlm/depth_ar.py` · `tests/test_depth_ar.py` · `tests/test_engine_capability.py` ·
`docs/engine_capability.md` · `.scratch/67-engine-changes/TICKET.md`（5 条，与提交信息逐条对应）。

提交后文件 sha256：`depth_ar.py 2f4b8887…`、`test_depth_ar.py 7a7a6f34…`、
`test_engine_capability.py d8028d63…`、`engine_capability.md d0bb7e0c…`、`TICKET.md d83ef2b9…`。
提交 `e894f20acef4aaa3b2197a8ff89b4cbc8f82b6a3`；提交后 `git status --short` 干净。

## 5. 独立复核状态（SCHEMA §14b：复核排在执行之后）

**本 worker 未执行独立复核，也不能自行派发**（当前运行不授予 `subagent` 工具；派发权属 Main）。
因此：**E1 的独立复核是待办**，请 Main 派 **fresh、非继承上下文** 的复核者，复核以下清单并以
「复核者不得是本 patch 的 writer」为界。建议复核点：

1. **默认逐位不变**：确认 `decay==1.0` 与 `carry_per_layer=False` 命中原语句、未夹带新乘子；
   跑 `test_carry_decay_one_is_bitwise_identity` / `test_carry_default_flags_are_inert_bitwise`
   并**独立造**一条等价性对拍（不要只信本人单测）。
2. **门是不是真的**：确认逐层模式下 β/键/值取自当前 `x`（读 `CarryChannelBlock.forward` 支路），
   并确认「逐层状态独立」单测有抓力（例如把逐层支路临时改成读 `_carry_box` 应抓红）。
3. **衰减代数**：独立复核 delta 带 γ 的 WY 公式与 `test_carry_decay_chunked_matches_sequential`
   的覆盖（是否含多块、两规则、γ≠1），并检查 chunk 边界是否正确。
4. **能力面（§22.8）**：C17–C22 事实行与源码一致；确认没有「开关开着、某臂没接」的死干预。
5. **范围**：确认未改测量数字、未动其它轴目录、`pathlm/**` 只此一个 writer。

复核报告请写进 `.scratch/67-engine-changes/`（本 worker 未替复核者写报告，避免伪造独立结论）。
一旦复核发现问题 ⇒ 由 writer 修 ⇒ 再派**新的** fresh 复核（≤2 轮）。

## 6. 未决 / 下一步

- **独立复核**（见 §5）——Main 派发。
- **WP2 实验未做**（本 patch 只交付引擎能力）：用 `carry_per_layer=True`（±`carry_decay`）
  重跑轴 1 的质量-代价前沿，才能把「显式状态只 +0.2%」的旧结论从「退化实现产物」里分开。
- **未加 CLI**：按 ticket「不新增公开开关（先内部实验）」，只加 `DepthARConfig` 字段；实验侧需
  直接构造 config（或后续授权再加 `--carry-per-layer`/`--carry-decay`）。
- **逐层状态的成本**：12 层各扫一次，墙钟会再超（ticket 已预警）；实验需按 `SCHEMA.md` §20.3
  缩步数保臂间同刻度。
- **发现的一处既存潜在缺陷（本 patch 未改，避免扩大范围）**：`carry_liveness` 在
  `scan_tele=None` 且通道开启时会执行 `out["write_rel_chunks"] > 0`（此时该值为 `None`）
  ⇒ `TypeError`（`pathlm/depth_ar.py` 的 `write_rel_note` 行）。既有调用（训练脚本）总带 tele，
  故未暴露；建议作为独立小修跟进（一行：`(out["write_rel_chunks"] or 0) > 0`）。
- **逐层模式下 `carry_read_mass_stats` 的两个 h₀ 派生口径**显式返回 `None`（已记在函数文档）；
  若后续要用「槽分数质量 / 窗口外份额」在逐层模式做活性证据，需要写逐层版仪器。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "只在 pathlm/depth_ar.py 做局部追加：新增 carry_per_layer/carry_decay 两字段（默认 False/1.0）、carry_scan*/chunk 的 decay 参数、write_read 的 layer_idx/decay、两个 block 的逐层支路、_wire_carry 接线、校验、_carry_scan_tele_mean、carry_liveness/carry_read_mass_stats 口径对齐；未重写块、未改测量数字、未动其它轴目录（git diff 5 文件）。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "提交 e894f20（5 文件，491+/41-）；新增 9 条单测覆盖默认逐位不变/逐层独立/内容写门/γ 分块==逐位置 fp64 1e-12/活性梯度/能力面；pytest tests/ -q = 849 passed；summary.py --check 与 check_tree.py = OK；能力面事实表补 C17–C22。"
    }
  ],
  "changedFiles": [
    "pathlm/depth_ar.py",
    "tests/test_depth_ar.py",
    "tests/test_engine_capability.py",
    "docs/engine_capability.md",
    ".scratch/67-engine-changes/TICKET.md"
  ],
  "testsAddedOrUpdated": [
    "tests/test_depth_ar.py::test_carry_decay_one_is_bitwise_identity",
    "tests/test_depth_ar.py::test_carry_default_flags_are_inert_bitwise",
    "tests/test_depth_ar.py::test_carry_decay_chunked_matches_sequential",
    "tests/test_depth_ar.py::test_carry_decay_forgets_and_matches_recurrence",
    "tests/test_depth_ar.py::test_carry_per_layer_states_are_independent",
    "tests/test_depth_ar.py::test_carry_write_gate_is_content_dependent",
    "tests/test_depth_ar.py::test_carry_per_layer_changes_forward_and_gradients_flow",
    "tests/test_depth_ar.py::test_carry_per_layer_composes_with_gqa_and_liveness",
    "tests/test_depth_ar.py::test_carry_e1_validation_and_decay_plumbing",
    "tests/test_engine_capability.py::test_fact_table_row_matches_engine (C17-C22 rows)"
  ],
  "commandsRun": [
    {"command": "python3 -m pytest tests/test_depth_ar.py -q", "result": "passed", "summary": "143 passed"},
    {"command": "python3 -m pytest tests/test_engine_capability.py -q", "result": "passed", "summary": "24 passed (C1-C22)"},
    {"command": "python3 -m pytest tests/ -q", "result": "passed", "summary": "849 passed in 91.92s"},
    {"command": "python3 tools/summary.py --check", "result": "passed", "summary": "OK: 46 project.yaml conform"},
    {"command": "python3 tools/check_tree.py", "result": "passed", "summary": "OK: 46 project.yaml conform"},
    {"command": "git diff --check", "result": "passed", "summary": "no whitespace errors"},
    {"command": "git add <5 explicit paths> && git diff --cached --name-only | wc -l", "result": "passed", "summary": "5 staged, matches commit"},
    {"command": "git commit -F - <<EOF", "result": "passed", "summary": "e894f20, 5 files changed; follow-up correction d3d33eb (ticket changelog count 8 to 9)"}
  ],
  "validationOutput": [
    "pytest tests/ -q => 849 passed in 91.92s",
    "test_carry_decay_chunked_matches_sequential: fp64 |chunked - sequential| < 1e-12 for gamma in {1,0.999,0.9,0.5,0.2} x {plain,delta} x chunk in {1,3,5,8,21}",
    "carry_scan(decay=1.0) torch.equal to carry_scan(default) for all rules/chunks; default-config model state_dict and forward torch.equal to flags-explicitly-off model",
    "per-layer write_read spy: layer_idx keys 0..L-1, per-layer mem pairwise unequal and != shared-slot mem",
    "write gate: different hidden -> different beta; beta varies by position; std>1e-4",
    "liveness/alive=True for per_layer and GQA x carry x per_layer; model._carry_tele values all float()-castable",
    "lock /tmp/pathlm_code.lock acquired and released by owner (rm -f + rmdir); no GPU lock taken"
  ],
  "residualRisks": [
    "独立复核未执行（本 worker 无 subagent 派发工具）；e1 acceptance 依赖 Main 派 fresh 复核者，见报告 §5。",
    "显式实现决策：逐层状态写自各层当前 hidden（共享写投影），未做「逐层独立写投影」；若复核认为需另立该（改参数量的）决策，属本 patch 之外。",
    "逐层模式墙钟会显著增加（12 次扫描/步）；实验需按 SCHEMA 20.3 缩步数。",
    "同一提交内含既存但未修的潜在缺陷：carry_liveness 在 scan_tele=None 且通道开启时 TypeError（training 路径总带 tele，未暴露）；已在报告 §6 记录，未修以避免扩范围。",
    "未加 CLI 开关（按 ticket「不新增公开开关」）；实验侧需直接构造 DepthARConfig。"
  ],
  "noStagedFiles": true,
  "diffSummary": "pathlm/depth_ar.py: additive E1 flags (carry_per_layer/carry_decay) + per-layer write in blocks + decay in carry_scan* (gamma=1 branch unchanged) + telemetry/validation/aggregation; tests/test_depth_ar.py: 9 new carry tests; tests/test_engine_capability.py + docs/engine_capability.md: rejection line numbers shifted + new rows C17-C22; .scratch/67-engine-changes/TICKET.md: E1 executed status + changelog entry. Default path bit-identical.",
  "reviewFindings": [
    "no blockers found in self-review; independent review pending Main dispatch (worker cannot spawn reviewer)"
  ],
  "manualNotes": "本 worker 不可派发 subagent（运行不授予该工具），故 §5 的独立复核与复核报告由 Main 派 fresh 复核者完成；本文件不冒充复核结论。E1 只交付引擎能力 + 抓住力的单测，WP2 质量实验属后续。"
}
```
