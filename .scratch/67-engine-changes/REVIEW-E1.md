# E1 独立复核报告 · CarryChannel 逐层状态 + 内容写门 + 衰减 γ

> 复核者：fresh、只读；在 `/tmp` 做副本与变异，未改仓库。
> 被复核对象：补丁提交 `e894f20` + 更正提交 `d3d33eb`（HEAD）。
> 复核时点 HEAD：`d3d33ebe040fb42fe99cd00ef4b0f4640588a434`，`git status` 干净，
> `git diff --cached --name-only` = 0。
> 补丁前基线：`ea8b5c2`（与 `e894f20^` 的 `pathlm/depth_ar.py` 逐位相同，已核）。
> 交付报告：session 工件目录
> `.../outputs/87160bd7-c291-4be3-9cf6-d7b376846a14/.scratch/67-engine-changes/E1-CARRYCHANNEL.md`。

## 0. 人话摘要（先读这段）

**结论：PASS。** 这次改动给「显式记忆通道」加了三个能力——**逐层独立状态**、**内容相关
写门**、**可选衰减 γ**——并把它们放在默认关闭的开关后面。

最重要的那条（**默认行为不能变**）我**没有只信报告**，而是自己造了三份模型对拍：
补丁前的旧代码、补丁后的默认配置、补丁后显式把新开关写成关闭值，三者的 **参数表
（state_dict）、前向输出的 4 组 depth_logits、读出占比、损失、遥测键名**全部
`torch.equal` 逐位相同；又扩展到 plain/delta 两条规则、GQA 组合块、非稠密 ragged/fill 路径、
训练窗、状态宽度——**全部逐位相同**。`carry_scan` 的新参数取 `decay=1.0` 也和不传它逐位相同。

三个新开关**真的能开、不互相排斥**：逐层、衰减、两者同开都改变前向；GQA 与非稠密路径也能同开且前向有限；
非法组合（逐层/衰减但没有通道、衰减超出 `(0,1]`）都**显式报错**，与能力面事实表 C17–C22 一行行对得上。

测试有抓力：全量 `pytest tests/ -q` = **849 passed**；我做了 4 处**变异**（关掉逐层支路、把写门
钉成常数、去掉逐位置衰减、去掉 delta 分块的 γ 乘子），每处都**至少有一条测试变红**；还原后
文件 sha256 回到 `2f4b8887…`、定向 9 条测试重新全绿。

越界/卫生也过了：只动 5 个文件（引擎 + 两个测试 + 能力面事实表 + ticket），没碰其它轴目录、
没改任何测量数字、没删证据；两个提交 `--stat` 与提交信息一致、无 `-A`/`.` 带入的杂散文件。

**只有两条非阻塞观察**（下面 §6）：
1. ticket 变更记录里指的仓库内报告路径 `.scratch/67-engine-changes/E1-CARRYCHANNEL.md`
   **在仓库里不存在**（报告实际落在 session 工件目录，是运行时输出路径改写的正常结果）；
2. `carry_liveness(scan_tele=None)` 的既存 `TypeError` 在补丁前后**逐字相同**，是既存缺陷、
   非本次引入（报告已如实披露）。

---

## 1. 判据①：默认行为逐位不变 —— ✅ PASS

**方法（独立，不看报告的自测）**：把一个脚本分别在三个代码树上跑（都只用两版都存在的字段，
不传 E1 新字段），把 `state_dict`、前向 `depth_logits`、`attn_carry[*].read_share`、`loss`、
`_carry_tele` 键名序列化到 `.pt`，再用 `torch.equal` 逐位比对：

- `HEAD 默认配置` vs `HEAD 显式关闭`（`carry_per_layer=False, carry_decay=1.0`）；
- `HEAD 默认/显式关闭` vs `ea8b5c2 旧代码`。

**覆盖的配置**（全部 `torch.equal` 逐位相同）：

| 配置 | 规则 | 块类 | 结果 |
|---|---|---|---|
| dense 默认 | delta | CarryChannelBlock | IDENTICAL（sd + 4 logits + shares + loss + tele 键名） |
| dense 默认 | plain | CarryChannelBlock | IDENTICAL |
| dense + GQA(n_kv_heads=2) | delta | ComposableBlock | IDENTICAL |
| dense + GQA(n_kv_heads=2) | plain | ComposableBlock | IDENTICAL |
| 非稠密（ragged） | delta / plain | CarryChannelBlock | IDENTICAL（含 `exit_depths=None` 与固定深度两路） |
| 非稠密 + `fill_kv=True` | delta | CarryChannelBlock | IDENTICAL |
| dense + `train_window=8` | delta | CarryChannelBlock | IDENTICAL |
| dense + `carry_state_dim=8` | plain | CarryChannelBlock | IDENTICAL |

另有函数级：`carry_scan` / `carry_scan_sequential` / `carry_scan_chunked` 在
`rule∈{plain,delta} × chunk∈{None,4,11}` 下，`decay=1.0` 与不传 `decay` **逐位相同**（`torch.equal`）。

`git diff --check e894f20^ e894f20` 无空白告警。

**复现**：
```bash
git worktree add --detach /tmp/pathlm-pre ea8b5c2
python3 /tmp/e1/dump_default.py /home/z/vibe/pathlm /tmp/e1/head_default.pt
python3 /tmp/e1/dump_default.py /tmp/pathlm-pre     /tmp/e1/pre_default.pt
python3 /tmp/e1/dump_off.py     /home/z/vibe/pathlm /tmp/e1/head_off.pt
# 比对脚本见 /tmp/e1/dump_multi.py + /tmp/e1/dump_extra.py 与比对 heredoc
# 输出：PRE(ea8b5c2) vs HEAD(default) => BITWISE IDENTICAL
#       HEAD(default) vs HEAD(explicit-off) => BITWISE IDENTICAL
#       delta/plain × dense/GQA × nd/nd_fill/tw/csd => IDENTICAL
```

## 2. 判据②：能力面真实 —— ✅ PASS

**开关真的改前向**（d=32/4 层，`carry_scale=0.7`，dense delta）：

```
per_layer 改前向 vs 默认：True  (maxdiff 1.54)
decay     改前向 vs 默认：True  (maxdiff 1.94)
两者同开  改前向 vs 默认：True  (maxdiff 1.55)
per_layer != decay（互相独立）：True
GQA×per_layer 有限且 != 默认：True；GQA×per_layer×decay != 默认：True（块类 ComposableBlock）
非稠密 ragged per_layer / per_layer+decay：改前向且前向有限
```

**互不排斥**：`per_layer` 与 `decay` 可同开；`per_layer` 与 GQA、与非稠密路径均可同开。

**不支持组合显式报错（SCHEMA §22.8）**：

```
carry_per_layer=True 且 carry_rule=None   → ValueError（…只在开启写入通道…）
carry_decay=0.5       且 carry_rule=None   → ValueError（…只在开启写入通道…）
carry_decay ∈ {0.0,1.5,-0.1}               → ValueError（必须落在 (0,1]）
合法：per_layer / decay=0.99 / decay=1.0 → 正常构造
```

**能力面事实表一致**：`pytest tests/test_engine_capability.py -q` = **24 passed**（C1–C22，含新增
C17–C22 的接受/拒绝、消息片段、拒绝点行号与源码 4 行窗口核对）。我另核了块类选择分支：
`carry_rule` 开启时块类必为 `CarryChannelBlock`（无 GQA）或 `ComposableBlock`（有 GQA），
`_wire_carry` 在这两条分支都调用 ⇒ **不存在「开关开着、某臂没接」的死干预**（`_input_states`
在逐层模式下不建共享槽，且无互相矛盾的组合）。

## 3. 判据③：测试有抓力（真跑 + 变异）—— ✅ PASS

**真跑**：`python3 -m pytest tests/test_depth_ar.py tests/ -q` = **849 passed in 90.64s**
（`tests/test_depth_ar.py` 单跑 143 passed）。`tools/summary.py --check` 与
`tools/check_tree.py` 均 `OK：46 个 project.yaml 全部符合`。

**变异**（在 HEAD 的独立 worktree 副本 `/tmp/pathlm-mut` 上做，逐条还原）：

| 变异 | 改坏点 | 结果 | 抓红的测试 |
|---|---|---|---|
| MA | `CarryChannelBlock.forward` 的 `if self._carry_per_layer:` → `if False`（关掉逐层支路） | RED（3 failed） | `test_carry_per_layer_states_are_independent`、`test_carry_write_gate_is_content_dependent`、`test_carry_per_layer_changes_forward_and_gradients_flow` |
| MB | `CarryChannel.write_read` 里 `beta` 钉成常数 `0.5`（写门取值坏） | RED（2 failed） | `test_carry_channel_write_path_gets_gradients`、`test_carry_per_layer_changes_forward_and_gradients_flow` |
| MC | `carry_scan_sequential` 去掉 `S = S * decay`（逐位置 γ 乘子坏） | RED（2 failed） | `test_carry_decay_chunked_matches_sequential`、`test_carry_decay_forgets_and_matches_recurrence` |
| MD | `_carry_chunk_delta` 去掉写强度上的 `γ^{C−1−i}`（分块 delta γ 乘子坏） | RED（1 failed） | `test_carry_decay_chunked_matches_sequential` |

还原后：文件 sha256 = `2f4b88877028f2ac…`（与报告一致），定向 9 条 E1 测试 **9 passed**；
worktree `git status` 干净。

**复现**：`python3 /tmp/e1/mutate.py`（脚本逐条变异→跑 `tests/test_depth_ar.py`→还原）。

## 4. 判据④：无越界 —— ✅ PASS（含一条附注）

`git show --stat e894f20` 只有 5 个文件、`d3d33eb` 只有 1 个文件；无任何其它路径：

```
pathlm/depth_ar.py            (218+/27-)    引擎：局部追加
tests/test_depth_ar.py        (242+/0)      9 条 E1 单测
tests/test_engine_capability.py      (10+/0)  C17–C22 用例
docs/engine_capability.md     (19+/13-)     拒绝点行号同步 + C17–C22 事实行
.scratch/67-engine-changes/TICKET.md  (3+/2-)  授权状态 + 变更记录
```

- **未碰**其它轴目录（`axes/**` 全无改动）、`archive/**`、任何 `results.json` / 证据文件；无删除。
- 测量数字：`docs/engine_capability.md` 改的是**源码行号**与**能力面事实行**，不是任何测量/指标数字；
  `TICKET.md` 改的是授权状态与变更记录里的单测计数。
- **附注（非 blocking）**：`docs/engine_capability.md` 不在判据字面「pathlm/**+对应测试+ticket」里，
  但它是 `tests/test_engine_capability.py` **机器解析**的能力面事实表（拒绝点行号必须与源码一致），
  源码新增 218 行后若不改它，`test_fact_table_row_matches_engine` 会红；且 `SCHEMA.md` §22.8 要求
  派发前做能力面分析并落事实行。故它是**强耦合的配套产物**，不是无关越界。

## 5. 判据⑤：提交卫生 —— ✅ PASS

- `git show --stat` 与提交信息**逐条对应**：`e894f20` 信息列出的 5 个文件与实际 5 个文件完全一致；
  `d3d33eb` 只改 `TICKET.md` 一个数字（1+/1-），与信息「只改这一处数字」一致。
- **无 `-A`/`.` 带入的杂散文件**：提交内无 `__pycache__`、`*.pt`、锁文件等；当前
  `git diff --cached --name-only | wc -l` = 0。
- 提交信息里「新增单测 8 条」与实际的 **9 条** 不符——`d3d33eb` 是**更正提交**（非 amend，
  符合仓库「发现不符立刻补更正提交、不要 amend 掩盖」的纪律），改正了 `TICKET.md` 变更记录里的 8→9。
  我逐个数了 `tests/test_depth_ar.py` 新增的 E1 测试函数 = **9 条**，更正**正确**。
  （唯一残留：`e894f20` 的**提交信息正文**仍写「8 条」，无法在不改写历史的前提下修正，属可接受。）

## 6. 非阻塞观察（如实列出，不构成 BLOCK）

- **O1（traceability）**：`TICKET.md` 变更记录指向仓库内 `.scratch/67-engine-changes/E1-CARRYCHANNEL.md`，
  但该文件**不在仓库**（`git check-ignore` 未忽略、`git status` 干净、目录内只有 `TICKET.md`）。
  报告实际落在 session 工件目录——这是运行时「输出路径改写」的正常结果，**非内容缺失**；
  但仓库内引用是悬空的，若后续要凭该路径复读报告需走工件目录。
- **O2（既存缺陷，非本次引入）**：`carry_liveness(model, scan_tele=None, layer_stats=None)` 在通道开启时
  会在 `write_rel_note` 处 `None > 0` 抛 `TypeError`。我用**补丁前 `ea8b5c2`** 复现，**逐字相同**的报错 ⇒
  既存缺陷、本次未改、未扩大范围（报告 §6 已如实披露并给出单行修法）。
- **O3（测试抓力粒度）**：`test_carry_write_gate_is_content_dependent` 用 `ch.beta_w` **直接**重算门，
  测的是「门公式」而非 `write_read` 里的接线；所以把 `write_read` 的 `beta` 钉成常数（变异 MB）它**没红**，
  是**其它**测试（梯度类）抓到的。结论不变（总有测试红），但这条测试对「写门接线」本身的抓力偏弱。
- **O4（设计边界，已声明）**：逐层状态在层间**共用同一套写投影**（`w_k/w_v/w_q/β` 不逐层），
  逐层差异只来自各自的输入表征与逐层读出增益；换「逐层独立写投影」会改参数量（需等参对照），
  报告 §1 已明确列为本 patch 之外。属**已声明的范围内决策**，非缺陷。

## 7. 未复核 / 未覆盖范围（如实列出）

1. **WP2 质量实验**（用 `carry_per_layer=True`（±`carry_decay`）重跑轴 1 质量-代价前沿）**未做、未复核**——
   本 patch 只交付引擎能力与证明其可用的单测，属后续实验项。
2. **墙钟/显存成本未测**：逐层模式每层各扫一次（12 次扫描/步），GPU 训练成本未实测（`/tmp` 纯 CPU 复核，
   未取 GPU 锁、未跑训练）。
3. **`carry_read_mass_stats` 逐层支路无单测**：我手动跑过（返回 `slot_score_mass_per_query`/
   `far_read_norm_frac` 显式 `None`、`read_share` 正常），但仓库测试只覆盖默认共享路径（`test_depth_ar.py:2654/2658`），
   逐层支路**无回归测试钉住**。
4. **增量解码（`DepthARDecoder`）路径**：carry/ GQA 的 decode 组合本就被拒绝（C10/C11），未覆盖。
5. **`_carry_scan_tele_mean` 的逐层聚合**仅经 `test_carry_per_layer_composes_with_gqa_and_liveness` 与
   我手动构造覆盖；不同 `decay`/`chunk` 组合下聚合口径的数值一致性与训练日志落盘字段的端到端一致性未逐一验证。
6. **`torch` 版本**：本机 `2.14.1+cu130`（CPU 复核）；其它 torch/后端下的逐位不变性未验证
   （逐位结论只对同版本、同后端成立）。

## 8. 复现命令汇总

```bash
# 基线
cd /home/z/vibe/pathlm && git rev-parse HEAD        # d3d33eb…
git status --short                                   # 干净
git worktree add --detach /tmp/pathlm-pre ea8b5c2    # 补丁前

# ① 默认逐位不变（跨 3 树 × 多配置）
python3 /tmp/e1/dump_default.py /home/z/vibe/pathlm /tmp/e1/head_default.pt
python3 /tmp/e1/dump_default.py /tmp/pathlm-pre     /tmp/e1/pre_default.pt
python3 /tmp/e1/dump_off.py     /home/z/vibe/pathlm /tmp/e1/head_off.pt
python3 /tmp/e1/dump_multi.py   /home/z/vibe/pathlm /tmp/e1/head_multi.pt
python3 /tmp/e1/dump_multi.py   /tmp/pathlm-pre     /tmp/e1/pre_multi.pt
python3 /tmp/e1/dump_extra.py   /home/z/vibe/pathlm /tmp/e1/head_extra.pt
python3 /tmp/e1/dump_extra.py   /tmp/pathlm-pre     /tmp/e1/pre_extra.pt
# （比对见 §1；结果全部 BITWISE IDENTICAL）

# ② 能力面
python3 /tmp/e1/capability.py /home/z/vibe/pathlm
python3 -m pytest tests/test_engine_capability.py -q     # 24 passed

# ③ 测试 + 变异
python3 -m pytest tests/test_depth_ar.py tests/ -q       # 849 passed
git worktree add --detach /tmp/pathlm-mut d3d33eb
python3 /tmp/e1/mutate.py                                # 4 处变异均 RED，还原 sha256 一致

# 清理
git worktree remove --force /tmp/pathlm-pre
git worktree remove --force /tmp/pathlm-mut
```

## 9. 最终结论

**PASS** —— 六条判据全部满足；无 blocking。仅 4 条非阻塞观察（§6）与未覆盖范围（§7）需留意。
本复核只读、只报告，未做任何修复。
