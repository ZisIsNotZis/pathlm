# 5.1 KV 量化 · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2614-kv-quant-eval-13m`](2614-kv-quant-eval-13m/README.md) | 目录内 README | 旧格 R1 迁入 → 参照线首测（fp16 1.483571 / int8 +0.00015 / int4 +0.0457，**逐值标尺**口径，其字节账已被 `2632` 取代） |
| [`2619-kv-budget-chart`](2619-kv-budget-chart/README.md) | 目录内 README | 旧格 R1 迁入 → 统一 KV 预算总图（量化/头共享/窗口/不压缩同轴，**逐值标尺 + 非因果**，跨家族数字须用 `2632` 口径重算） |
| [`2620-kv-budget-gaps`](2620-kv-budget-gaps/README.md) | 目录内 README | 旧格 R1 迁入 → 缺失组合（GQA×int8 / MQA×int8 / 窗口×锚点） |
| [`2621-kv-budget-rest`](2621-kv-budget-rest/README.md) | 目录内 README | 旧格 R1 迁入 → 剩余缺失组合（GQA×int4 / MQA×int4 / 窗口×int8 / GQA×窗口 / 锚点×int4） |
| [`2631-kv-budget-causal`](2631-kv-budget-causal/README.md) | 目录内 README | 旧格 R1 迁入 → 因果/流式口径重测（27 点）；**逐值标尺**下 int8 = ×1.50、int4 ×1.25（内存负收益）——该字节口径**已撤回** |
| [`2632-kv-block-scale`](2632-kv-block-scale/README.md) | 目录内 README | 旧格 R1 迁入 → **块标尺**口径重测（本项 SSOT）：int8 = 1.0625 B/值（省 46.875%、Δbpc +0.000101）、int4 = 0.5625 B/值（Δbpc +0.022522）；严格前缀不变性成立 |

> 旧格 `R1-quantization` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本项：其 `experiments/**` 的
> 六个 kv 目录整目录迁入本目录（整目录 `git mv`，一格一份家，不复制）。旧格目录与其 README 仍保留在
> `axes/R-references/R1-quantization/**`，待 integrator 统一退役。
> 口径提示：**引用 KV 量化的字节数一律用 `2632` 的块标尺部署口径（含标尺）**；`2619`/`2620`/`2621` 是
> 逐值标尺/非因果口径的历史图，其跨家族数字已作废（55-review W-08），不得当现行；`2631` 的逐值标尺
> 字节账同样已撤回。
