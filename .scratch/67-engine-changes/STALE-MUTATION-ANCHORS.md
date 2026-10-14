# 陈旧变异锚点扫描（N3 · 只读）

> 性质：**只读扫描报告**（engine-writer 收尾，N3）。扫描对象 = `axes/**/mutate.py`
> 的 `old=` 锚点，判定「在 E2 父提交 `46ba4cf` 的引擎文件里能匹配、在当前引擎
> 源码里不匹配」者。**未修改任何 `axes/**`**。
> 扫描脚本：`.scratch/67-engine-changes/scan_stale_anchors.py`；引擎文件 = `pathlm/depth_ar.py`、`pathlm/model.py`、`pathlm/decode.py`、`pathlm/config.py`。
> E2 = `4638882`，E3 = `fa6a1ad`；引擎文件改动窗口内只有这两个提交（`git log 46ba4cf..HEAD -- <引擎文件>`）。

## 0. 人话摘要

- **按 committed HEAD 版本的变异脚本**：1 条锚点失配。
- **按当前工作树版本**：0 条锚点失配。
- 已知的 `2614-content-topk` 锚点失配**已被轴 1 在工作树里修好**（未提交）——
  该修复把 `TopKBlock(...)` 接线行同步到 E2 之后的 `_mh(i)` / `for i in range`。
- 未发现轴 1 修复之外的其它失配。

### 1. 提交态（HEAD 的 mutate.py 对当前引擎）

这些锚点在 **committed HEAD** 状态下会 unmatched。

- **[k-ignored]** `axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py:58` → 引擎文件 `pathlm/depth_ar.py`
  - `old=` 字面片段：
    ```python
                self.blocks = nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k)
                                            for _ in range(cfg.n_layers))
    ```
  - 当前 `pathlm/depth_ar.py` 中 **0 匹配**（E2 父提交 `46ba4cf` 能匹配）⇒ 重跑该变异会 unmatched。
  - 对应引擎行（当前）：
    - `pathlm/depth_ar.py:2201`  `self.blocks = nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k, _mh(i))`
    - `pathlm/depth_ar.py:2202`  `for i in range(cfg.n_layers))`

### 2. 工作树态（当前 mutate.py 对当前引擎）

这些锚点在**当前工作树**下会 unmatched。

- （无）

### 3. 提交态失配、工作树已修（并发轴 1 修复，未提交）

- `axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py`（id `k-ignored`，行 58）

### 4. 内联 E2/E3 触碰行的锚点（不论当前是否仍匹配）

判定 = 锚点 `old` 里有一行等于 E2/E3 改动（`-`/`+`）的引擎行。

- `axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/mutate.py:58` [id `k-ignored`] → `pathlm/depth_ar.py`
  - 触碰行：`for i in range(cfg.n_layers))`
  - 触碰行：`self.blocks = nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k, _mh(i))`

