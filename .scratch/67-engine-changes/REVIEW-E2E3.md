# E2 + E3 独立复核（ticket 67 · 受控引擎通道）

> 性质：**复核报告**（fresh 上下文复核者，未参与 E2/E3 编写；只读仓库，唯一写入 = 本文件）。
> 复核对象：E2 = `4638882`、E3 = `fa6a1ad`、更正 = `8e80b7c`（HEAD）。
> 上游：`.scratch/67-engine-changes/TICKET.md` §2、`E2-E3-ENGINE-REPORT.md`；
> 契约：`SCHEMA.md` §22/§22.8/§14/§14b、`docs/engine_capability.md` C23–C28。
> 环境：RTX 4090 单卡共享；本次复核**全程未取 GPU 锁**（所有 pytest 以 `CUDA_VISIBLE_DEVICES=""` 跑 CPU）。
> 复现基线：`git archive` 导出 `46ba4cf`（E2 父）/`238ecab`（E3 父）/`8e80b7c`（HEAD）到 `/tmp/rev/`；
> 另 `git clone --local` 一份带 `.git` 的副本；变异在 `/tmp/rev/mut` 副本做。

## 0. 人话摘要（先读这段）

**判词：5 条判据全部 PASS，blocker = 0。** E2（逐层 FFN 隐藏宽）与 E3（二级条件头 + 树注意力）
两处引擎改动的**默认行为逐位不变**这一条我用**独立造例**证实了（不信本人单测）：E2 拿 8 种块类、
E3 拿 4 种条件链配置，在两提交的父树与本树上各建一遍、同 seed 同输入，`state_dict` 全量 sha256
与 `forward` loss 逐位相同。能力也是**真的**：任意正整数 schedule 真到达每层 `mlp[0].out_features`
（我用的 `[7,100,33,5]` 既非 `mult·d` 也非 d 的倍数）；`chain_second` 打开真改变前向、缺省关真不改变；
树注意力对**分支树**隐藏兄弟、保留祖先。测试**有抓力**：4 处变异（拆 schedule 接线 / 钉死二级条件 /
破坏树掩码 / 扰动缺省回落）每处都至少打红一条测试，还原后三文件 sha256 全部回原值。提交**没越界**
（只碰 `pathlm/**`、`train_depth_ar.py`、`tests/**`、`docs/engine_capability.md`、`.scratch/67-engine-changes/**`）。
唯一那条全量失败**不是** E2/E3 引起的：在父树两处与本树三处，失败的都只有同一条
`test_all_repo_mutate_scripts_use_the_harness`，且根因是**提交里就存在的**
`axes/6-predict/6.3.3-latent-regression/experiments/2660-*/mutate.py` 未接 harness——与引擎 diff 无关。

## 1. 逐条判据

### 判据 1 · 默认逐位不变 —— **PASS**

| 面 | 父树 | 本树 | 结果 |
|---|---|---|---|
| E2（`4638882`）| `46ba4cf` | `4638882` | 8 变体全等 |
| E3（`fa6a1ad`）| `238ecab` | `8e80b7c` | 4 配置全等 |

E2 覆盖块类（`state_dict` sha256 + forward loss 逐位）：stock `Block`、`ParallelBlock`（`parallel`）、
`ParallelBlock`（`parallel-dual-ln`）、`GQABlock`（`n_kv_heads=1`）、`LinearAttnBlock`、`TopKBlock`、
`CarryChannelBlock`、`ComposableBlock`（`carry_rule=delta × n_kv_heads=1`）——即 dense/非稠密/GQA/并行/
线性/稀疏/通道全类。E3 覆盖 `chain_mode=token`（n_mtp=3/2）、`latent`（n_mtp=3）、`n_mtp=1`。
两者父树↔本树的 `state_dict` 哈希与 loss **零处差异**。

补充独立门（不依赖本人单测）：E2 用 `mlp_hidden_sched=[64,64,64,64]`（= `mlp_mult·d`）建树，与缺省
`mlp_hidden_sched=None` 的参数逐位相同；E3 `chain_second=False` 与缺省 `PathConfig` 的参数键集与前向逐位相同。

复现：
```bash
cd /tmp/rev/e2base && PYTHONPATH=$PWD python3 /tmp/rev/probe_e2.py /tmp/rev/e2_base.json
cd /tmp/rev/head   && PYTHONPATH=$PWD python3 /tmp/rev/probe_e2.py /tmp/rev/e2_head.json
cd /tmp/rev/e3base && PYTHONPATH=$PWD python3 /tmp/rev/probe_e3.py /tmp/rev/e3_base.json
cd /tmp/rev/head   && PYTHONPATH=$PWD python3 /tmp/rev/probe_e3.py /tmp/rev/e3_head.json
# 比对四个 json：E2 base=head（9 字段）、E3 base=head（4 字段），mismatches=[]
```

### 判据 2 · 能力真实 —— **PASS**

- **E2 schedule 真到达每层**：`mlp_hidden_sched=[7,100,33,5]`（d_model=16、n_layers=4）下
  每层 `mlp[0].out_features == [7,100,33,5]`、`mlp[2].in_features == [7,100,33,5]`、
  `mlp[2].out_features == d`；前向 loss 与缺省不同（非死开关）。7 个块类（`test_..._wires_all_block_variants`）
  全部接线，无「开关开着、某臂没接」：源码里 `mlp = nn.Sequential` 站点共 4 处（`Block`/`ParallelBlock`/
  `GQABlock`/`ComposableBlock`），继承类 `TopKBlock`/`LinearAttnBlock`/`CarryChannelBlock` 走 `super().__init__`，
  8 条 `self.blocks = nn.ModuleList(...)` 分支逐条带 `_mh(i)`。建时校验长度/正整数（C24/C25）。
- **E3 二级条件真生效**：`chain_second=True` vs `False` 前向 loss 不同（同 seed 同路径：26.855 vs 22.250）；
  打开只新增 3 个头（`transforms_chain2`/`chain_cat2`/`conf_chain3` 各 weight+bias），**其余参数逐位不变**；
  条件对齐正确——只改 `condition[:,1]`：`chain2` 行 1 变、行 0 不变；`chain3` 行 0 变、行 1 不变（= 比 chain2 右推一位）。
  缺省关不建任何新参数、不产生 `chain3`。
- **树注意力语义**：`step_tree_multi` 链父指针 `[-1,0,1,2,3]` 与 `step_multi` **逐位相同**；
  分支树 `[-1,0,1,1]` 下行 3 不受**兄弟**行 2 改变影响、受**祖先**行 1 影响、行 2 受其父行 1 影响（均如预期）。
  `decode_spec`（`chain_second=True`）走树路径且发出流与逐 token 贪心**逐位相同**、`drafts_offered>0`。
  **契约声明的差异**：`decode_spec` 只走**链**父指针（非分支树），同级多候选的草稿/接受策略明确留给轴 6 ——
  符合判据「与单臂贪心一致**或**按契约声明差异」。行位置按 `p0+j` 扁平铺开（已在报告 §3 声明）。

复现：
```bash
cd /tmp/rev/head && PYTHONPATH=$PWD python3 /tmp/rev/probe_cap.py     # （分支树那段见 §3 记录）
cd /tmp/rev/head && python3 -m pytest tests/test_depth_ar.py -k mlp_hidden_sched -q          # 4 passed
cd /tmp/rev/head && python3 -m pytest tests/test_m1.py -k "chain_second or step_tree or decode_spec_tree" -q  # 7 passed
```

### 判据 3 · 测试有抓力（≥3 变异）—— **PASS**

在 `/tmp/rev/mut`（HEAD 副本）逐次单点变异，每次跑定向测试，跑完即还原并核对 sha256：

| # | 变异（文件:改动） | 变红测试 | 还原 sha256 |
|---|---|---|---|
| M1 | `pathlm/depth_ar.py`：`_mh` 恒返回 `None`（拆 schedule 接线） | `test_mlp_hidden_sched_reaches_every_layer_and_changes_forward`、`test_..._wires_all_block_variants`（2 red） | ✔ |
| M2 | `pathlm/model.py`：`_mtp_nodes` 的 `if self.pcap.chain_second:` → `if False:`（钉死二级条件） | `test_chain_second_conditions_on_t_i_plus_2_not_t_i_plus_1`（`KeyError: 'chain3'`） | ✔ |
| M3 | `pathlm/decode.py`：树分支掩码改回纯 causal `tril()`（破坏树注意力） | `test_step_tree_multi_attention_follows_parents_not_causal` | ✔ |
| M4 | `pathlm/model.py`：`Block` 缺省回落 `mlp_mult` → `mlp_mult+1`（扰动缺省不变门） | `test_mlp_hidden_sched_default_is_bitwise_identical` | ✔ |

还原后 `sha256sum -c`：`depth_ar.py=45b7d406…`、`model.py=2bc19db5…`、`decode.py=0616a0ff…`，三处 **OK**。

复现：见本报告末「附录 · 变异命令」。

### 判据 4 · 无越界 + 提交卫生 —— **PASS**（含 2 条 non-blocker）

`git diff-tree --name-only` 逐条判定（允许集 = `pathlm/**`、`tests/**`、`train_depth_ar.py`、
`docs/engine_capability.md`、`.scratch/67-engine-changes/**`）：

- `4638882`（7 文件）、`fa6a1ad`（11 文件）、`8e80b7c`（2 文件）：**越界项 = 0**；
  未碰 `axes/**`、`docs/structure.md`、`docs/research-schema.md`、`SCHEMA.md`、`README.md`；无杂散文件。
- 提交信息与 `--stat` 一致；E2 的 message 已声明「TICKET.md 同时携带 axis5-lead 先前写在工作树里的 E4 条目」。
- `docs/engine_capability.md` C23–C28 行号与源码一致（C24→`depth_ar.py:2028`、C25→`:2033`、
  C27→`model.py:151`、C28→`:157` 均落在 `raise ValueError(` 行；`tests/test_engine_capability.py` 30 passed 已机器校验）。
- 归档阶梯护栏（`test_decorrelation_v2/at_scale`、`test_mechanism_convergence`）对新增 `chain_second`
  的豁免是**窄口径**（仅减一个键，沿用既有 `w_repr_decorr` 先例），未削弱其它检查；25 passed。

**non-blocker（不影响判词）**：
- **N1**：报告 §4 全量计数「865 passed, 1 failed」与实测差 2。本复核在带 `.git` 的 `8e80b7c` 干净副本上实测
  = **1 failed, 863 passed, 2 xfailed**（`420.7s`）；总数 866 一致、失败数一致，差额正是那 2 条 **xfailed**
  （被报告的「passed」合并计入）。仅计数口径问题。
- **N2**：E2 顺带把 `tests/test_engine_capability.py` 的 `C17`/`C18` 两个 dict 条目并到了同一行（`…Block"},    "C18"…`），
  纯格式、无功能影响，但属**计划外编辑**。同文件 `SRC_REL`/`SRC` 在 E3 后变为**未使用常量**（死代码）。
- **N3（须轴 1 处置）**：E2 改了 `depth_ar.py` 的 `self.blocks = nn.ModuleList(TopKBlock(...))` 行，
  使 `axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py` 的锚点文本失配
  （该行已含 `, _mh(i)`，旧串 `attn_sparse_k)\n` 在源码中已为 0 匹配）。报告 §3 已如实登记；这是对该轴
  既有检查的**失效**，按 `AGENTS` invalidate 纪律须由轴 1 更新锚点或重跑其变异。**本 patch 边界外**，非 E2/E3 缺陷。

复现：
```bash
git diff-tree --no-commit-id --name-only -r 4638882
git diff-tree --no-commit-id --name-only -r fa6a1ad
git diff-tree --no-commit-id --name-only -r 8e80b7c
grep -n "TopKBlock(mcfg" pathlm/depth_ar.py            # 已含 , _mh(i)
grep -c "nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k)" pathlm/depth_ar.py   # 0（2614 旧锚失效）
sed -n '73p' tests/test_engine_capability.py           # C17/C18 同行
```

### 判据 5 · 全量唯一失败的归因 —— **PASS（既有 / 与 E2/E3 diff 无关）**

在三处**只含提交内容**的干净树（`git archive` 导出，排除并发 untracked）跑该测试，**三处同一条失败、同一根因**：

```
FAILED tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness
AssertionError: 未接入统一框架：['axes/6-predict/6.3.3-latent-regression/
                                 experiments/2660-latentreg-vs-disc-13m/mutate.py']
```

| 树 | 结果 | 根因脚本 |
|---|---|---|
| `46ba4cf`（E2 父）| 1 failed | `2660-…/mutate.py`（缺失 `from tools.evalkit import mutation`）|
| `238ecab`（E3 父）| 1 failed | 同上 |
| `8e80b7c`（HEAD，带 `.git` 副本）| 1 failed / 863 passed / 2 xfailed | 同上 |

`2660-…/mutate.py` 是**已跟踪**文件、三处都在 ⇒ 失败**早于** E2、与 E2/E3 diff 无关。

**「引擎改动致」vs「并发轴编辑致」的分离**：在**当前工作树**（含并发 untracked）该测试报 3 个脚本未接 harness
（`2660` 跟踪 + `2670-freerun-bpc-13m`、`2671-padfill-13m` 均 untracked、`git ls-files` 不识别）。即
**跟踪态只有 `2660`（既有）**，`2670`/`2671` 是并发轴 6 写者新建的未跟踪文件——两者都**与 E2/E3 无关**。
报告 §4 只列了 `{2660,2670}`、漏了 `2671`（见 N2），但归因结论正确。

复现：
```bash
cd /tmp/rev/e2base && python3 -m pytest tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness -q
cd /tmp/rev/e3base && python3 -m pytest tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness -q
cd /tmp/rev/clone  && git checkout 8e80b7c && CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/ -q   # 1 failed, 863 passed, 2 xfailed
cd /home/z/vibe/pathlm && git ls-files --error-unmatch axes/6-predict/6.3.3-latent-regression/experiments/2670-*/mutate.py  # 报未跟踪
```

## 2. blocker / non-blocker 汇总

- **blocker：0**。
- non-blocker：N1（报告计数口径差 2，xfail 被并入 passed）、N2（E2 顺带的格式并行 + E3 遗留未使用常量）、
  N3（2614 变异脚本锚点失配，须轴 1 处置，边界外）。其余见 §3 未覆盖面。

## 3. 未覆盖面 / 登记的差异（复核者发现的偏差点）

1. **E3 训练/重训头未做**（明确属轴 6，不在 patch）：本复核只验引擎原语与损失接线，未训二级头的收敛/质量。
2. **分支树（同级多候选）草稿/接受策略未验**：`decode_spec` 只走链父指针；`step_tree_multi` 的分支树
   语义已验，但实验侧的树构造/接受算法未做（轴 6）。
3. **墙钟/收敛未验**：E2 逐层宽改变参数量（这正是目的），未在 GPU 上跑实际墙钟与收敛（patch 宣称只交付构造能力）。
4. **`--mlp-mult` 与 `--mlp-hidden-sched` 同给的优先级**：实现为 schedule 逐层覆盖（schedule 优先），
   无专项单测；语义直观、无隐患，仅登记。
5. **报告 §4 计数（N1）与失败清单（N2 漏列 2671）与实测有出入**；判词不受影响。

## 附录 · 变异命令（每次跑完即还原）

```bash
cp -a /tmp/rev/head /tmp/rev/mut && cd /tmp/rev/mut
sha256sum pathlm/{depth_ar,model,decode}.py            # 记原值
# M1: _mh 恒 None → pytest tests/test_depth_ar.py -k mlp_hidden_sched（2 red）；还原
# M2: _mtp_nodes 的 if self.pcap.chain_second → if False → pytest tests/test_m1.py -k chain_second（1 red）；还原
# M3: decode.py 树掩码 → tril() → pytest tests/test_m1.py -k step_tree（1 red）；还原
# M4: model.py Block 缺省回落 +1 → pytest tests/test_depth_ar.py -k mlp_hidden_sched_default（1 red）；还原
sha256sum -c /tmp/rev/orig.sha256                       # 三处 OK
```

## 4. 结论

E2 + E3（含更正 `8e80b7c`）**通过独立复核**：默认逐位不变（独立造例证实）、能力真实、测试有抓力、
提交无越界、唯一全量失败归因为**既有** `2660` 变异脚本（与引擎 diff 无关）。**blocker = 0**，
**PASS**。3 条 non-blocker 中仅 N3 需跨线动作（轴 1 更新 2614 锚点）。
