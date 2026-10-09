# 2632 · 过程史（日志，非权威）

> 事实文档是 `analysis.md` / `table.md` / `results.json` / `README.md` / `MODEL.md`。
> 本文件只记「我们一步一步怎么做的」，冲突时以事实文档为准。

1. **起点（WI-08 + 独立批判性审计）**：`2631` 的价签用**逐值标尺**测出「int8 = 3 B/值 = fp16 的 1.5×」，
   据此判「量化在任何拓扑上都是内存负收益」。审计指出：业界标准是标尺**按块/通道摊薄**（一块 32 个值
   共用一把 ⇒ int8 ≈ 1.0625 B/值，省 46.9%），且沿非 token 轴摊薄**仍然前缀确定**；本仓自己早就算过
   通道标尺与序列长度无关。

2. **判据预登记（`config.json`，运行前写入）**：五条判据 + 门禁（主臂前缀不变性）+ 本格自测方差
   （不套用全局 0.014）+ 禁外推。见 `README.md` §4。

3. **实现**：在 `tools/evalkit/kv_quant.py` 增 `scheme` 分支（默认 `per_value`，向后兼容）：
   `block`（沿 head_dim，每 G 个相邻值一把标尺，严格前缀确定）、`block_tok`（沿 token，非前缀确定）、
   `kivi`（K per-channel 沿 token 分组 / V per-token）；新增 `scheme_scale_bytes_per_value` 与
   `scheme_footprint` 的字节账。新前向与字节账落在新模块 `block_arms.py` / `scale_caliber.py`。
   途中修过一个实现 bug：`_token_block_quant` 的 amax 归约轴写成了 `dim=-3`（应为 `dim=-2`），
   单测 `test_kivi_k_per_channel_is_not_prefix_invariant` 先红了（这也是为什么它现在是可失败单测）。

4. **冒烟（`--smoke`，0.38 min）**：确认前向能跑通、前缀不变性在 block 上为 True、block_tok/kivi 为 False；
   步时与显存实测（不外推）。

5. **全跑（1.43 min，峰值 1798.6 MiB）**：27 点 + 13 方案行 + 自测方差。非量化臂与 `2631` `max|Δbpc|=0`；
   `scheme=per_value` 前向与生产前向逐位一致。结果见 `results.json`。

6. **单测 + 变异**：`tests/test_kv_block_scale.py`（21 项）；`mutate.py` 7 个变异全抓红 + 逐字节还原。
   首轮 M4/M5 未抓到（测试对 KIVI 的扰动不够、且穿透了 `make_logits_fn`），改成「同块过去位置被改写」
   与「经 `make_logits_fn` 的 block_tok 非不变」两条后全抓红。

7. **文档**：`analysis.md`（当前结论，含正面对照表、三道对账、局限）、`table.md`（表 1–5）、`MODEL.md`。

8. **待办（交给主 Agent / 别的线）**：同步 `catalog.yaml` 与 R1 格页（本线不改）；`head_dim ≥ 64` 的
   点、fp8 标尺、权重/激活量化未测。
