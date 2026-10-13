# 3.1.1 uniform 减宽 / width dropout · 实验目录

本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / `analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。

命名：`YYMM-<干预>-<规模>-<引擎>`。

| 实验 | 来源 | 说明 |
|---|---|---|
| [`2525-tax-attribution`](2525-tax-attribution/README.md) | 目录内 README | 旧格 W1 迁入。测的是**读出可见子空间切片**（unembed 与出口头只读 h[:360]，trunk 仍全宽 720），按 55-review W-06 属**读出域**（`3.2.1`），**不是减宽**——既不减算力也不减参数。故本项命题（uniform 减宽 / width dropout）**仍无本地判决**，`status: 未开始` |

> 旧格 `W1-active-width` 按 `.scratch/56-restructure/MAPPING.md` §1 迁入本项：其 `experiments/**`
> 整目录迁入本目录（一格一份家，不复制）。按 55-review W-06，迁入的实验与命题**错配**（测的是读出
> 切片），故只作**背景证据**登记，不构成本项结论；`3.2.1` 以指针引用它。
> 旧格目录与其 README 仍保留在 `axes/W-width/W1-active-width/**`，待 integrator 统一退役。
