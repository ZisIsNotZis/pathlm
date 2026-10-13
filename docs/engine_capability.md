# 引擎能力面事实表（F_CLOSURE，2026-10-02）

> **用途**：把 `pathlm/depth_ar.py` 的**真实能力边界**逐条列清楚——哪些组合真跑得起来、
> 哪些明确拒绝、拒绝在哪一行——供账本（L2）引用。本表**只陈述代码现状**，不主张任何
> 尚未实现的组合，也不改任何已入账数字。
>
> **机器校验**：`tests/test_engine_capability.py` 解析下表的受控块（`CAPABILITY:begin` …
> `CAPABILITY:end` 注释标记之间），逐行核对「结果 / 拒绝点行号 / 源码字面片段 / 运行时
> `str(e)` 片段」。**改代码不改表 ⇒ 该测试红**；改表不改代码同理。
>
> **结论（对应复核 B9）**：`archive/dropped/axes/G-policy/G5-parallel-lines/CLOSURE.md:75` 称「ENG2 已解锁
> `carry × GQA × 非稠密`」**与代码不符**。代码现状是：
> **ENG2 解锁的是「`carry_rule` × 非 dense 前向」**（单开通道 + 随机退出/ragged/填充，
> 走 `CarryChannelBlock` 的 `kv_from`/`attn_mask`/`gate_vec` 约定）；
> **`carry × GQA × 非 dense` 仍被明确拒绝**——`n_kv_heads < n_heads` 的校验要求
> `dense=True`（`pathlm/depth_ar.py:2146`），且增量解码路径两个组合都拒绝
> （`:2853` GQA、`:2862` carry）。**未实现的组合不得当作已就绪引用。**

## 1. 接受矩阵 / 拒绝点（受控块：机器核对）

列含义：**结果** = `DepthARModel(cfg)` 或 `DepthARDecoder(model)` 构造结果；
**拒绝点** = 抛出该 `ValueError` 的 `raise` 所在行（`file:line`）；
**源码字面片段** = 该行起 4 行内在源码里出现的字面量（f-string 模板按模板写）；
**运行时消息片段** = 实际抛出的 `str(e)` 里必须出现的子串。

<!-- CAPABILITY:begin -->
| id | 组合 | 结果 | 构建产物 / 拒绝点 | 源码字面片段 | 运行时消息片段 |
|---|---|---|---|---|---|
| C1 | `carry_rule="plain"`（dense=False，topology=sequential） | 接受 | CarryChannelBlock | — | — |
| C2 | `carry_rule="plain"` + `train_window=8` + dense=True | 接受 | CarryChannelBlock | — | — |
| C3 | `carry_rule="delta"` + `n_kv_heads=2` + dense=True | 接受 | ComposableBlock | — | — |
| C4 | `n_kv_heads=2` + `train_window=8` + dense=True | 接受 | GQABlock | — | — |
| C5 | `n_kv_heads=2` + dense=True（单开 GQA） | 接受 | GQABlock | — | — |
| C6 | `carry_rule="plain"` + `n_kv_heads=2` + dense=False（**B9 的核心**：carry × GQA × 非稠密） | 拒绝 | pathlm/depth_ar.py:2146 | KV 头共享（GQA/MQA）只在 dense=True | KV 头共享（GQA/MQA）只在 dense=True |
| C7 | `carry_rule="plain"` + `attn_sparse_k=4` | 拒绝 | pathlm/depth_ar.py:2137 | carry_rule 与 {other} 互斥 | carry_rule 与 attn_sparse_k 互斥 |
| C8 | `carry_rule="plain"` + `topology="parallel"` + dense=True | 拒绝 | pathlm/depth_ar.py:2128 | carry_rule 与层间拓扑互斥 | carry_rule 与层间拓扑互斥 |
| C9 | `n_kv_heads=2` + `attn_linear=True` + dense=True | 拒绝 | pathlm/depth_ar.py:2160 | n_kv_heads 与 {other} 互斥 | n_kv_heads 与 attn_linear 互斥 |
| C10 | `carry_rule="plain"` + `topology="sequential"`，再建 `DepthARDecoder` | 拒绝 | pathlm/depth_ar.py:2862 | 增量解码（KV cache）路径未定义写入通道读出 | 增量解码（KV cache）路径未定义写入通道读出 |
| C11 | `n_kv_heads=2` + dense=True，再建 `DepthARDecoder` | 拒绝 | pathlm/depth_ar.py:2853 | 增量解码（KV cache）路径未定义 KV 头共享 | 增量解码（KV cache）路径未定义 KV 头共享 |
| C12 | `attn_sparse_k=4` + dense=False | 拒绝 | pathlm/depth_ar.py:2045 | 内容 top-k 稀疏注意力只在 dense=True | 内容 top-k 稀疏注意力只在 dense=True |
| C13 | `carry_gate=False` + dense=True（R7 无门变体用在没有门的 dense 上） | 拒绝 | pathlm/depth_ar.py:2035 | carry_gate=False（R7 无门变体） | carry_gate=False（R7 无门变体）只对非 dense（变深度）路径有意义 |
| C14 | `mod_ratio=0.5` + dense=False（MoD 路由与变深度形态混开） | 拒绝 | pathlm/depth_ar.py:2027 | mod_ratio > 0（MoD 式路由） | mod_ratio > 0（MoD 式路由）只与 dense=True 同开 |
| C15 | `mod_ratio=0.5` + dense=True 但直接建 `DepthARModel`（不实装路由的类） | 拒绝 | pathlm/depth_ar.py:2031 | mod_ratio > 0 需要 `MoDModel` | mod_ratio > 0 需要 `MoDModel` |
| C16 | `aux_mode="mean"`（未知归一模式） | 拒绝 | pathlm/depth_ar.py:2022 | unknown aux_mode | unknown aux_mode |
| C17 | `carry_rule="delta"` + `carry_per_layer=True`（E1 逐层独立状态） | 接受 | CarryChannelBlock | — | — |
| C18 | `carry_rule="delta"` + `carry_per_layer=True` + `n_kv_heads=2` + dense=True（E1 × GQA） | 接受 | ComposableBlock | — | — |
| C19 | `carry_rule="delta"` + `carry_decay=0.5`（E1 可选衰减 γ） | 接受 | CarryChannelBlock | — | — |
| C20 | `carry_per_layer=True` 但 `carry_rule=None`（逐层状态没有通道） | 拒绝 | pathlm/depth_ar.py:2105 | carry_per_layer 只在开启写入通道 | carry_per_layer 只在开启写入通道（carry_rule）时有意义 |
| C21 | `carry_decay=0.5` 但 `carry_rule=None`（衰减没有通道） | 拒绝 | pathlm/depth_ar.py:2110 | carry_decay 只在开启写入通道 | carry_decay 只在开启写入通道（carry_rule）时有意义 |
| C22 | `carry_rule="delta"` + `carry_decay=0.0`（衰减超出 (0, 1]） | 拒绝 | pathlm/depth_ar.py:2114 | carry_decay 必须落在 (0, 1] | carry_decay 必须落在 (0, 1] |
<!-- CAPABILITY:end -->

## 2. 边界的三条要点

1. **「非 dense」是一组语义**（随机退出 + ragged/fill 混合填充 + 变深度），而不是一个开关。
   `carry_rule` 与**前向**的非 dense 路径同开（C1/C2）；但 `n_kv_heads < n_heads`
   （GQA/MQA）与**任何**非 dense 形态互斥（C6）——KV 头共享只在「无退出/无填充」的
   稠密形态下被验证与实现。
2. **增量解码（`DepthARDecoder`）是更窄的子集**：它拒绝 GQA（C11，理由：解码器只服务非
   dense 路径）与 carry（C12，理由：`step` 未接入 `γ ⊙ mem_t` 读出，静默丢路比报错更糟）。
   因此「引擎就绪」的说法必须限定到**前向**。
3. **GQA × 窗口**（C4）与 **carry × 窗口**（C2）是允许的：窗口/锚点是作用在 `attn_mask`
   上的装饰层，与 block 类正交；两者都不解锁增量解码。

## 3. 与收口文的关系（不越界：只记录，不改别线文件）

- `CLOSURE.md` §2 未测项 #3 的「引擎已由 ENG2 解锁该组合 ⇒ 引擎就绪」应改为：
  「ENG2 解锁的是 `carry × 非 dense 前向`；`carry × GQA × 非 dense` **仍被引擎拒绝**
  （见本表 C6/C11/C12）⇒ 该组合若要测，需先立引擎工单。」
- 该文件属 G5 线，本波由本表提供事实依据，**措辞更正由 L2/该线执行**（一文件一写入者）。
