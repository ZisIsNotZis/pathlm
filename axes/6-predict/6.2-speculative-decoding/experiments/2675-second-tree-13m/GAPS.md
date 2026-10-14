# GAPS.md · 2675 未测因子与缺失能力（供主 Agent 登记受控通道 E6，本 WP 不得改引擎）

## G1（能力缺口，待 E6）引擎缺「位置忠实的多兄弟树验证原语」

- **缺什么**：`Decoder.step_tree_multi` 能表达**链**（`parents=[-1,0,1,…]`，与因果逐位一致，引擎单测钉住），
  但不能表达**同深度多兄弟**的分支树：`pathlm/decode.py:224` 把扁平第 j 个节点放在绝对位置 `p0+j`，
  而 K 个 N+2 候选的正确位置都是 N+2。实测（本目录 `test_second_tree.py::test_flat_tree_siblings_are_not_position_faithful`
  + `QUESTIONS.md` §3）：兄弟节点 logits 与位置正确条件分布的 TV = 0.0116，vs fp 噪声 3e-8。
- **最小 patch 是什么**：给 `step_multi`/`_run_stack_multi`/`step_tree_multi` 增一个可选
  `positions: list[int] | None = None`（缺省 `None` ⇒ 现状 `p0+j`，逐位不变），把它用于
  `model.pos_embed.weight[positions]` 与 `self.layer_pos[i] = [positions...]`；树的每个节点用**真实生成位置**
  （根 N+1、候选 N+2、草稿 N+3）。配套 3 条单测：① 缺省 `None` 与现状逐位相同；
  ② `parents=[-1,0,0]` + `positions=[N+1,N+2,N+2]` 下每个节点的 logits == 对应根到叶路径（`step_multi`）的读数；
  ③ 兄弟不可见（改兄弟行不改本行 logits）。风险/显存：无新增参数，位置表长度 = K。
- **本 WP 的替代**：位置忠实的 K 行平铺（2640 `spec_tree.tree_forward`，只读复用，非引擎改动）。
  代价：每轮送进主干的**行数**与引擎原语路径相同（K 行），但每行 3 个位置（`[v1,c_i,d_i]`）。

## G2（未测因子）`chain_second` 草稿的上下文严格对齐 h_{N+1}

- 训练对齐是 `(h_{i+1}, t_{i+2}) → t_{i+3}`。起草 N+3 草稿时，**同一轮前向之前**只拿得到 h_N
  （`h_{N+1}` 要先把 v1 送进主干）——本 WP 与引擎 `decode_spec` 一致，用 `h_N` 作上下文、
  用候选 c 作条件（= 契约命题的那个变量）。
- 严格对齐版需要**每轮多一次主干前向**（+1 前向/轮，行数从 K 变 K+1 ⇒ 代价轴口径全变），
  本 WP **未测**，不外推。
- 下一步确切命令：`python3 run.py --stage eval --arms lvl2_tree_ctx1 --k 2 --seeds 0 --trunks 13m_s0`
  （`lvl2_tree_ctx1` 已预留为「两级前向：先 `step_multi([v1])` 取 h_{N+1}，再走 K 行平铺」的臂名；
  本 revision **未实现该臂**——实现前须先确认它是否值得，见 `RESUME.md`）。

## G3（未测组合）K 与主干/种子的扩展（时间盒）

- 已跑/未跑的 (主干 × 头种子 × K × 臂) 矩阵见 `results.json::coverage` 与 `RESUME.md`；
  未跑到的组合不等于「无差别」，禁止外推。
