# E2 + E3 执行报告 · 受控引擎通道（ticket 67）

> 性质：**执行报告**（engine-writer 交付物，落 `.scratch/67-engine-changes/`；直接上游事实 =
> `.scratch/67-engine-changes/TICKET.md` §2 E2/E3/E4）。本文件与 TICKET.md 变更记录同源。
> 执行者：engine-writer（worker，`pathlm/**` 单一写入者，持 `/tmp/pathlm_code.lock`；纯 CPU，
> 未取 GPU 锁）。**独立复核未在本 worker 内执行**（见 §5）。
> 提交：E2 = `4638882`（7 文件）；E3 = `fa6a1ad`（11 文件）。

## 0. 人话摘要（先读这段）

**E2（与 E4 统一成一个实现）**：原来每个 transformer 块的 FFN 隐层宽只有一个全局标量
`mlp_mult`（×d_model），**所有层一个值**。现在多了一个可选开关：给一条「**逐层隐层宽**」
列表（任意正整数），每层的 FFN 就用自己那个宽度。**不给就是原来那样**（逐位不变）。
这一个字段同时满足两件事：轴 3 想要的「逐层 mlp_mult」（值 = mult×d_model），和
轴 5（E4）想要的「等参数对照所需的任意整数宽」（例如 GQA 少了 1 184 256 个参数，就可以
按 E4 的算法把这 118 万参数逐层还给 FFN）。`Block` 的全部块类（默认 `Block`、`ParallelBlock`、
`GQABlock`、`ComposableBlock`、`LinearAttnBlock`、`TopKBlock`、`CarryChannelBlock`）都接上了。

**E3**：`PathLM` 的 MTP 头原来只有**一级** token 条件链（`chain2`：预测 t_{i+2}，条件只能取
t_{i+1}）。现在多了一个可选开关 `chain_second`：再加一层**二级条件头** `chain3`（预测
t_{i+3}，条件是 **t_{i+2} 的候选**，上下文取主干右移一位 = h_{N+1}，EAGLE 式复用，不需要
额外主干前向）。默认关；打开时验证/解码走新增的 `Decoder.step_tree_multi`（B=K 行 + 树注意力）。

**默认行为变没变**：E2 与 E3 **都是逐位不变**（E2 字段缺省 None；E3 开关缺省 False，且新头
建在参数初始化之后 ⇒ 连既有参数的随机数流都不动）。有单测把这两条钉住。

**测试抓力**：E2 新增 4 条、E3 新增 7 条单测；能力面事实表补 C23–C28。

## 1. 契约、范围与验收判据

来源：`.scratch/67-engine-changes/TICKET.md` §2 E2/E3/E4；上游规则：`SCHEMA.md` §20（GPU 锁）/
§22（代码锁、锚点、能力面 §22.8）/§14b（复核顺序）/§10（无死干预）。

| 要求（ticket 原文） | 落点 | 证据 |
|---|---|---|
| E2：`mlp_mult` 全局标量 → 逐层 schedule | `DepthARConfig.mlp_hidden_sched` | `test_mlp_hidden_sched_*`（4 条） |
| E2：`Block` 4 变体一并接线 | `Block`/`ParallelBlock`/`GQABlock`/`ComposableBlock`/`LinearAttnBlock`（+`TopKBlock`/`CarryChannelBlock`） | `test_mlp_hidden_sched_wires_all_block_variants` |
| E2：`train_depth_ar.py` 加开关（缺省 None） | `--mlp-hidden-sched` + `_parse_int_sched` | `test_mlp_hidden_sched_validation_and_cli_plumbing` |
| E4：任意整数隐藏宽（等参数对照） | 同一字段（值即隐藏宽） | 同上 + 能力面 C23 |
| E3：二级条件头（`transforms_chain2`/`chain_cat2`，输入 cat([h_{N+1}, embed(c)]） | `PathLM` + `_mtp_nodes` 的 `node["chain3"]` | `test_chain_second_adds_only_its_three_heads` |
| E3：`PathConfig` 开关（默认关） | `PathConfig.chain_second: bool = False` | `test_chain_second_default_off_is_bitwise_inert` |
| E3：损失 = t_{i+3} 的 CE + conf BCE | `forward` 里 `_node_loss(node["chain3"], targets[:, 3:])` | 同上（节点存在） |
| E3：`Decoder`/`decode_spec` 树验证（B=K 行 + 树注意力） | `Decoder.step_tree_multi` + `_run_stack_multi(tree_parents=)` + `decode_spec` 二级链 | `test_step_tree_multi_*` / `test_decode_spec_tree_path_matches_greedy_with_chain_second` |
| 能力面（§22.8） | 事实表 C23–C28 + 建时校验 | `tests/test_engine_capability.py` |

### 显式实现决策（与 ticket 最小 patch 描述的差异，需复核者盯）

1. **E2 的字段名与语义（与 E4 统一，Main 本 run 协调）**：ticket E2 写 `mlp_mult_sched: list[int] |
   None`（倍率），E4 写 `mlp_hidden_sched: list[int] | None`（任意整数隐藏宽）。Main 指示**统一为
   一个实现**。落点取 `mlp_hidden_sched`（**每层 FFN 隐藏宽，任意正整数**）；E2 的「逐层 mlp_mult」
   是它的特例（值 = mult·d_model）。CLI 名随之取 `--mlp-hidden-sched`。**未**同时保留 `mlp_mult_sched`
   （避免第二套开关）。
2. **E3 的头建在参数 init 循环之后**：若建在循环之前，新建 `nn.Linear` 会在循环前消耗随机数 ⇒
   既有主干参数整体错位。改为循环后创建 + 显式初始化，单测钉住「打开只新增 3 个头的参数、其余逐位不变」。
3. **E3 树路径的最小形态**：`Decoder.step_tree_multi` 是**验证原语**（行 j 只看祖先链，父指针
   `parents[j]`；B=K 行，位置仍按 p0+j 扁平铺开）。`decode_spec` 的二级链用**链父指针**走该路径
   （链 == 因果，单测钉住）。**同级多候选的分支树草稿/接受策略**属轴 6 实验的算法工作，不在本 patch。
4. **E3 的 context 取 h 右移一位**（ticket 的 `h_{N+1}`）：训练里 row i 用 h[i+1]、条件 t_{i+2}，
   对齐 `chain2`（row i 用 h[i]、条件 t_{i+1}）多推一位。`decode_spec` 在**同一位置**只能取到 h_N，
   故那里 chain3 的 context 是 h_N（草稿只是猜测，验证仍精确 —— 无损性由单测钉住）。

## 2. patch 落点

**E2**
- `pathlm/depth_ar.py`：`DepthARConfig.mlp_hidden_sched`；`DepthARModel.__init__` 校验（长度 ==
  n_layers、每项为正整数）+ 按层接线；7 个块类的 `mlp_hidden` 可选参数（缺省 `None` = `cfg.mlp_mult·d`）。
- `pathlm/model.py`：`Block.__init__(cfg, mlp_hidden=None)`。
- `train_depth_ar.py`：`--mlp-hidden-sched` + `_parse_int_sched` + `build_depthar_config` 用 `getattr` 回落。

**E3**
- `pathlm/config.py`：`PathConfig.chain_second: bool = False`。
- `pathlm/model.py`：`PathLM.__init__` 两处建时校验（需 `chain_mode="token"`、`n_mtp>=2`）+ 三个
  新模块（循环后创建）；`_mtp_nodes` 追加 `node["chain3"]`；`forward` 追加 chain3 的 CE + conf BCE。
- `pathlm/decode.py`：`_run_stack_multi(tree_parents=)`、`step_multi(tree_parents=)`、
  `step_tree_multi(tokens, parents)`；`decode_spec` 草稿取 chain3 + 二级链走树路径。

## 3. 锚点声明与回读校验（SCHEMA §22.3 / §22.4）

改前声明锚点（行号按改前 `pathlm/depth_ar.py` / `pathlm/model.py` / `pathlm/decode.py`）：
E2 = `DepthARConfig.mlp_mult`（L50）、各块 `__init__` 的 `self.mlp = nn.Sequential(...)`、
`DepthARModel` 的块构造段（L2166–2197）；E3 = `PathLM.__init__`（L145 起）、`_mtp_nodes`（L323）、
`forward` 的 node 损失环、`Decoder._run_stack_multi`/`step_multi`/`decode_spec` 的 `step_multi` 调用点。

**只做局部追加**：E2 全是「新增可选参数 / 新增字段 / 新增分支 / 新增函数」；E3 同为「新增字段 /
新增可选参数 / 新增分支 / 新增方法」。默认分支保留原语句（E2 的 `_mh(i)` 恒 `None`；E3 的
`tree_parents is None` 命中原因果掩码、`chain_second=False` 不产生 `chain3`）。

**回读校验**：改后 `grep -n "mlp_mult\|_mh(i)\|chain_cat\|chain_second\|tree_parents"` 复核既有接线
（`ENG1`/`ENG2`/`carry_*`/`recon_window` 等标记仍在位）；全量测试过。E2 的字段/构造改动使
`docs/engine_capability.md` 既有拒绝点行号整体后移，已逐行同步（C6–C22 由 2146/2137/… 改为
2172/2163/…），`tests/test_engine_capability.py` 逐行核对通过。

**跨线影响（未越界修，如实登记）**：`axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py`
的变异 `old` 字符串内联了被 E2 改动的 `nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k)\n ... for _ in range(cfg.n_layers))`
那一行；E2 后该字符串不再匹配 ⇒ 该变异脚本**重跑会报 unmatched**（无测试运行它，故全量测试不受影响）。
该文件在 `axes/**`（本 patch 边界外）⇒ 只登记，未改。**交下一步处置**。

## 4. 命令与结果（复现）

- 锁定：`/tmp/pathlm_code.lock`（退出时 `rm -f holder_pid/heartbeat` + `rmdir` 释放；未取 GPU 锁）。
- E2 定向：`python3 -m pytest tests/test_depth_ar.py tests/test_engine_capability.py -q` ⇒ **174 passed**。
- E2 全量：`python3 -m pytest tests/ -q` ⇒ **855 passed, 1 failed**（见下「既有失败」；
  口径见「N1 计数口径更正」——`xfailed` 不计入 `passed`）。
- E3 定向：`python3 -m pytest tests/test_m1.py tests/test_engine_capability.py tests/test_depth_ar.py -q` ⇒ **227 passed**。
- E3 全量：`python3 -m pytest tests/ -q` ⇒ **865 passed, 1 failed**（同一既有失败；带本地
  `runs/` 证据的开发工作树实测）。**干净检出**（无 `runs/`）下同命令 = **863 passed,
  2 xfailed, 1 failed**——差额的 2 条是证据锚测试，见「N1 计数口径更正」。
- `git diff --check` ⇒ 无空白告警。
- **提交哈希**：E2 = `4638882`；E3 = `fa6a1ad`。

**N1 计数口径更正**（E2/E3 复核后收尾补记）：pytest 汇总里的 `xfailed` **不计入** `passed`，
两者是两列。本仓库 2 条证据锚测试（`tests/test_longT_clean.py`、`tests/test_longT_stability.py`
经 `tests/evidence_anchor.py`）在本地 `runs/` 原始证据**缺失**（干净检出）时 `xfail`、**存在**
时正常通过。故同一命令的计数随环境而异：带证据的开发工作树 = `865 passed, 1 failed`；
干净检出 = `863 passed, 2 xfailed, 1 failed`。原记录 E3 行只写「865 passed」，未把 `xfail`
单列，此处更正（E2 行的分列口径相同，其干净检出分列值未复测，**不臆造**）。**未改任何测量数字**。

**既有失败（与本 patch 无关，须复核者独立确认）**：
`tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness` —— 报
`axes/6-predict/{6.3.3-latent-regression/experiments/{2660-…,2670-…},6.3.2-pad-fill/experiments/2671-…}/mutate.py`
未接入统一 harness。本 patch 未动 `axes/**`（`git diff --name-only | grep axes/6-predict` = 0），
该测试只读 `axes/**` 的 mutate 脚本 ⇒ 失败独立于本 diff（E2 提交前的全量也只见它一条）。
**跟踪态**下只有 `2660-…`（既有）；`2670-…`/`2671-…` 是并发轴 6 写者新建的**未跟踪**脚本
（`git ls-files` 不识别），故本 `pytest tests/ -q` 在当前工作树列出三条，与 E2/E3 diff 无关。

**E2 新增单测**（`tests/test_depth_ar.py`）：`test_mlp_hidden_sched_default_is_bitwise_identical`、
`test_mlp_hidden_sched_reaches_every_layer_and_changes_forward`、
`test_mlp_hidden_sched_wires_all_block_variants`、
`test_mlp_hidden_sched_validation_and_cli_plumbing`。

**E3 新增单测**（`tests/test_m1.py`）：`test_chain_second_default_off_is_bitwise_inert`、
`test_chain_second_adds_only_its_three_heads`、
`test_chain_second_conditions_on_t_i_plus_2_not_t_i_plus_1`、
`test_chain_second_requires_token_chain`、`test_step_tree_multi_chain_parents_matches_causal`、
`test_step_tree_multi_attention_follows_parents_not_causal`、
`test_decode_spec_tree_path_matches_greedy_with_chain_second`。

**归档阶梯护栏（2620/2622/2626）**：新增 `PathConfig` 字段使三份归档配置各缺一个未钉住键
（`chain_second`）⇒ 按仓库既有先例（同一护栏里已为 `w_repr_decorr` 开过「归档后新增字段吃默认值」的
例外）在护栏里**显式豁免**并写明理由；**未改 `archive/**`**。

## 5. 独立复核状态（SCHEMA §14b：复核排在执行之后）

**本 worker 未执行独立复核，也无派发权**（派发权属 Main）。E2/E3 的独立复核是**待办**。建议复核点：

1. **默认逐位不变**：E2 的 `mlp_hidden_sched=None` 与 E3 的 `chain_second=False` 是否命中原语句；
   **独立**对拍（不要只信本人单测）——E2 造 `mlp_hidden_sched=[m·d]*L` 与缺省对拍；E3 造
   `chain_second` 缺省与显式 False 对拍 state_dict 与前向。
2. **E3 条件对齐**：确认 `chain3` 的条件列比 `chain2` 右移一位（读 `_mtp_nodes`），并用独立造例
   复现（改 `condition[:, k]` 看哪一行变）。
3. **树注意力**：确认 `step_tree_multi` 的祖先掩码（读 `_run_stack_multi`），独立造一棵**分支树**
   验证兄弟不可见（本人单测用的是 `[-1,0,0]`，建议复核者另造 `[-1,0,1,1]` 之类）。
4. **能力面（§22.8）**：C23–C28 事实行与源码一致；确认没有「开关开着、某臂没接」的死干预。
5. **范围**：确认未改测量数字、未动 `axes/**`/`archive/**`、`pathlm/**` 只此一个 writer；确认
   2614 变异脚本的跨线断链已如实登记（§3）。

复核报告请写进 `.scratch/67-engine-changes/`（本 worker 不替复核者写结论）。发现问题 ⇒ writer 修 ⇒
再派**新的** fresh 复核（≤2 轮）。

## 6. 未决 / 下一步

- **独立复核**（§5）——Main 派发。
- **E3 训练/重训头**（ticket：~1 GPU 小时）属**轴 6**，不在本 patch；本 patch 只交付引擎能力 +
  可失败单测。分支树（同级多候选）的草稿/接受策略同样是轴 6 的实验工作。
- **E2 的轴 3/轴 5 实验**：轴 3 用 `mlp_hidden_sched=[mult·d, …]`；轴 5（E4）用任意整数宽做等参数臂
  （`--mlp-hidden-sched "1728,1729,…"`）。两者均需重训主干，属各轴实验。
- **2614 变异脚本断链**（§3）：需轴 1 侧决定是否更新其 `old` 字符串（本 patch 不改 `axes/**`）。
- **E2 墙钟/参数面**：逐层宽会改变参数量（这正是目的）；等参对照须按 E4 的 Σ 精确相等算法构造。
```