# 2621 · KV 预算图剩余缺失组合 —— 更新后的总表（含所有已测点）

横轴公式（判据 C1，与 2619/2620 **逐字同一式**）：`bytes_per_position = 2 · n_layers · n_kv_heads_eff · head_dim · value_bytes · (positions_kept / T)`，T = 256。**本列是解析口径**；实测显存单列在文末。`window_anchor` / `combo_mask` 档的 `positions_kept = window + anchors`（锚点占用位置预算）。

本表**含所有已测点**：2619 的 14 点 + 2620 的 5 点 + 本批 8 点，全部由本实验**同进程、同一路径**重测（复用 2619 的 `bpc_of` / `synth_accuracy`）。逐点来源实验路径见『来源实验路径』列；与来源实验 / 2619 / 2620 的逐位对照见 `results.json::cross_check_vs_source` / `reproducibility_vs_2619` / `reproducibility_vs_2620`。召回地板：chance = 0.0156（1/64）、局部先验 = 0.0146（copy）/ 0.0156（induction）/ 0.0195（var_recall）。『量化增量』= 组合臂 bpc − 同拓扑 fp16 臂 bpc；『窗口增量』= 头共享×窗口臂 bpc − 同头数全精度臂 bpc；『相对 stock』= bpc − 1.483569（不压缩 trunk 的释放版 bpc）。

| 点 | 类别 | 每位置 KV 字节（解析） | vs 本族基线 | bpc | Δbpc==damage | 量化增量 | 窗口增量 | 相对 stock | 合成召回 copy（汇总） | 95% CI | copy 毁源 | vs chance | 来源实验路径 | 口径 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fp16` | 不压缩基线 | 12288 (0.0%↓) | fp16 | 1.483571 | +0.000000 | — | — | +0.000002 | 0.1413 | [0.1294, 0.1540] | 0.0143 | 9.04× | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `int8` | 量化（R1） | 6144 (50.0%↓) | fp16 | 1.483721 | +0.000150 ⚠不可排序 | — | — | +0.000152 | 0.1426 | [0.1307, 0.1554] | 0.0143 | 9.12× | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `int4` | 量化（R1） | 3072 (75.0%↓) | fp16 | 1.529232 | +0.045661 | — | — | +0.045663 | 0.1247 | [0.1135, 0.1368] | 0.0127 | 7.98× | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `win16` | 窗口/驱逐（M2） | 768 (93.8%↓) | fp16 | 2.085526 | +0.601955 | — | — | +0.601957 | 0.0150 (地板) | [0.0112, 0.0199] | 0.0137 | 0.96× | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win32` | 窗口/驱逐（M2） | 1536 (87.5%↓) | fp16 | 1.695384 | +0.211813 | — | — | +0.211815 | 0.0140 (地板) | [0.0104, 0.0188] | 0.0133 | 0.90× | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win64` | 窗口/驱逐（M2） | 3072 (75.0%↓) | fp16 | 1.564077 | +0.080506 | — | — | +0.080508 | 0.0589 | [0.0511, 0.0678] | 0.0166 | 3.77× | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win128` | 窗口/驱逐（M2） | 6144 (50.0%↓) | fp16 | 1.503634 | +0.020063 | — | — | +0.020065 | 0.1100 | [0.0994, 0.1216] | 0.0143 | 7.04× | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 解析字节（非实测显存） |
| `win144` | 窗口/驱逐（M2） | 6912 (43.8%↓) | fp16 | 1.497328 | +0.013757 ⚠不可排序 | — | — | +0.013759 | 0.1100 | [0.0994, 0.1216] | 0.0150 | 7.04× | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 解析字节（非实测显存） |
| `win160` | 窗口/驱逐（M2） | 7680 (37.5%↓) | fp16 | 1.492435 | +0.008864 ⚠不可排序 | — | — | +0.008866 | 0.1087 | [0.0982, 0.1202] | 0.0140 | 6.96× | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 解析字节（非实测显存） |
| `win176` | 窗口/驱逐（M2） | 8448 (31.2%↓) | fp16 | 1.489695 | +0.006124 ⚠不可排序 | — | — | +0.006126 | 0.1074 | [0.0970, 0.1189] | 0.0133 | 6.88× | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 解析字节（非实测显存） |
| `win192` | 窗口/驱逐（M2） | 9216 (25.0%↓) | fp16 | 1.487744 | +0.004173 ⚠不可排序 | — | — | +0.004175 | 0.1064 | [0.0960, 0.1178] | 0.0133 | 6.81× | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 解析字节（非实测显存） |
| `win16_anchor16` | 组合：窗口 × 锚点 | 1536 (87.5%↓) | fp16 | 1.858406 | +0.374835 | — | — | +0.374837 | 0.0618 | [0.0539, 0.0709] | 0.0130 | 3.96× | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | 解析字节（非实测显存） |
| `win32_anchor32` | 组合：窗口 × 锚点 | 3072 (75.0%↓) | fp16 | 1.590539 | +0.106968 | — | — | +0.106970 | 0.0934 | [0.0836, 0.1042] | 0.0143 | 5.98× | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | 解析字节（非实测显存） |
| `win32_anchor8` | 组合：窗口 × 锚点 | 1920 (84.4%↓) | fp16 | 1.633097 | +0.149526 | — | — | +0.149528 | 0.0290 | [0.0236, 0.0355] | 0.0150 | 1.85× | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | 解析字节（非实测显存） |
| `mha` | 不压缩基线 | 12288 (0.0%↓) | mha | 1.482026 | +0.000000 | — | — | -0.001543 | 0.1530 | [0.1407, 0.1662] | 0.0107 | 9.79× | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `gqa` | 头共享（W3） | 3072 (75.0%↓) | mha | 1.496325 | +0.014299 | — | — | +0.012756 | 0.1689 | [0.1561, 0.1826] | 0.0137 | 10.81× | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `mqa` | 头共享（W3） | 1536 (87.5%↓) | mha | 1.493763 | +0.011737 ⚠不可排序 | — | — | +0.010194 | 0.2432 | [0.2283, 0.2586] | 0.0124 | 15.56× | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `gqa_int8` | 组合：头共享 × 量化 | 1536 (87.5%↓) | mha | 1.496489 | +0.014463 | +0.000164 | — | +0.012920 | 0.1686 | [0.1558, 0.1823] | 0.0130 | 10.79× | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | 解析字节（非实测显存） |
| `mqa_int8` | 组合：头共享 × 量化 | 768 (93.8%↓) | mha | 1.494105 | +0.012079 ⚠不可排序 | +0.000342 | — | +0.010536 | 0.2448 | [0.2299, 0.2603] | 0.0120 | 15.67× | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | 解析字节（非实测显存） |
| `gqa_int4` | 组合：头共享 × 量化 | 768 (93.8%↓) | mha | 1.592538 | +0.110512 | +0.096213 | — | +0.108969 | 0.1396 | [0.1278, 0.1524] | 0.0133 | 8.94× | `2616 gqa-12k-s0 × R1 int4（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `mqa_int4` | 组合：头共享 × 量化 | 384 (96.9%↓) | mha | 1.689346 | +0.207320 | +0.195583 | — | +0.205777 | 0.1709 | [0.1580, 0.1846] | 0.0117 | 10.94× | `2616 mqa-12k-s0 × R1 int4（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `win32_int8` | 组合：窗口/锚点 × 量化（本批） | 768 (93.8%↓) | fp16 | 1.695690 | +0.212119 | +0.000306 | — | +0.212121 | 0.0146 (地板) | [0.0110, 0.0195] | 0.0140 | 0.94× | `2615 win32 掩码 × R1 int8（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `win64_int8` | 组合：窗口/锚点 × 量化（本批） | 1536 (87.5%↓) | fp16 | 1.564170 | +0.080599 | +0.000093 | — | +0.080601 | 0.0602 | [0.0523, 0.0692] | 0.0173 | 3.85× | `2615 win64 掩码 × R1 int8（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `win16_anchor16_int4` | 组合：窗口/锚点 × 量化（本批） | 384 (96.9%↓) | fp16 | 1.943239 | +0.459668 | +0.084833 | — | +0.459670 | 0.0550 | [0.0475, 0.0636] | 0.0130 | 3.52× | `2620 win16_anchor16 掩码 × R1 int4（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `win32_anchor32_int4` | 组合：窗口/锚点 × 量化（本批） | 768 (93.8%↓) | fp16 | 1.643489 | +0.159918 | +0.052950 | — | +0.159920 | 0.0814 | [0.0722, 0.0916] | 0.0130 | 5.21× | `2620 win32_anchor32 掩码 × R1 int4（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `gqa_win64` | 组合：头共享 × 窗口（本批） | 768 (93.8%↓) | mha | 1.559100 | +0.077074 | — | +0.062775 | +0.075531 | 0.0658 | [0.0575, 0.0751] | 0.0117 | 4.21× | `2616 gqa-12k-s0 × 2615 win64 掩码（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `gqa_win128` | 组合：头共享 × 窗口（本批） | 1536 (87.5%↓) | mha | 1.512094 | +0.030068 | — | +0.015769 | +0.028525 | 0.1364 | [0.1247, 0.1490] | 0.0117 | 8.73× | `2616 gqa-12k-s0 × 2615 win128 掩码（本批组合，无单一来源）` | 解析字节（非实测显存） |

## 表 2 · 逐任务召回 + copy 分 src_pos（同一仪器，同一 seed）

| 点 | 每位置字节 | copy 汇总 | copy@src0（锚区） | copy@src96 | copy@src191 | induction | var_recall | 不可分辨的腿 |
|---|---|---|---|---|---|---|---|---|

| `fp16` | 12288 | 0.1413 | 0.1406 | 0.1572 | 0.1260 | 0.0645 | 0.0039 | var_recall |
| `int8` | 6144 | 0.1426 | 0.1406 | 0.1582 | 0.1270 | 0.0625 | 0.0039 | var_recall |
| `int4` | 3072 | 0.1247 | 0.1299 | 0.1318 | 0.1133 | 0.0488 | 0.0020 | var_recall |
| `win16` | 768 | 0.0150 | 0.0127 | 0.0176 | 0.0146 | 0.0137 | 0.0020 | copy, induction, var_recall |
| `win32` | 1536 | 0.0140 | 0.0078 | 0.0137 | 0.0205 | 0.0117 | 0.0020 | copy, induction, var_recall |
| `win64` | 3072 | 0.0589 | 0.0127 | 0.0205 | 0.1436 | 0.0137 | 0.0059 | induction, var_recall |
| `win128` | 6144 | 0.1100 | 0.0127 | 0.1709 | 0.1465 | 0.0742 | 0.0059 | var_recall |
| `win144` | 6912 | 0.1100 | 0.0107 | 0.1689 | 0.1504 | 0.0684 | 0.0059 | var_recall |
| `win160` | 7680 | 0.1087 | 0.0107 | 0.1660 | 0.1494 | 0.0664 | 0.0039 | var_recall |
| `win176` | 8448 | 0.1074 | 0.0088 | 0.1660 | 0.1475 | 0.0625 | 0.0039 | var_recall |
| `win192` | 9216 | 0.1064 | 0.0088 | 0.1621 | 0.1484 | 0.0605 | 0.0039 | var_recall |
| `win16_anchor16` | 1536 | 0.0618 | 0.1602 | 0.0107 | 0.0146 | 0.0117 | 0.0020 | induction, var_recall |
| `win32_anchor32` | 3072 | 0.0934 | 0.2471 | 0.0176 | 0.0156 | 0.0176 | 0.0039 | induction, var_recall |
| `win32_anchor8` | 1920 | 0.0290 | 0.0527 | 0.0156 | 0.0186 | 0.0156 | 0.0020 | induction, var_recall |
| `mha` | 12288 | 0.1530 | 0.1680 | 0.1416 | 0.1494 | 0.0488 | 0.0156 | var_recall |
| `gqa` | 3072 | 0.1689 | 0.1826 | 0.1807 | 0.1436 | 0.0273 | 0.0117 | induction, var_recall |
| `mqa` | 1536 | 0.2432 | 0.1875 | 0.2607 | 0.2812 | 0.0938 | 0.0039 | var_recall |
| `gqa_int8` | 1536 | 0.1686 | 0.1787 | 0.1777 | 0.1406 | 0.0293 | 0.0117 | induction, var_recall |
| `mqa_int8` | 768 | 0.2448 | 0.1865 | 0.2598 | 0.2832 | 0.0957 | 0.0039 | var_recall |
| `gqa_int4` | 768 | 0.1396 | 0.1611 | 0.1475 | 0.1279 | 0.0215 | 0.0117 | induction, var_recall |
| `mqa_int4` | 384 | 0.1709 | 0.1338 | 0.1875 | 0.2129 | 0.0625 | 0.0039 | var_recall |
| `win32_int8` | 768 | 0.0146 | 0.0088 | 0.0156 | 0.0205 | 0.0117 | 0.0020 | copy, induction, var_recall |
| `win64_int8` | 1536 | 0.0602 | 0.0127 | 0.0215 | 0.1465 | 0.0137 | 0.0059 | induction, var_recall |
| `win16_anchor16_int4` | 384 | 0.0550 | 0.1465 | 0.0107 | 0.0088 | 0.0137 | 0.0020 | induction, var_recall |
| `win32_anchor32_int4` | 768 | 0.0814 | 0.2266 | 0.0166 | 0.0107 | 0.0195 | 0.0020 | induction, var_recall |
| `gqa_win64` | 768 | 0.0658 | 0.0088 | 0.0117 | 0.1768 | 0.0137 | 0.0137 | induction, var_recall |
| `gqa_win128` | 1536 | 0.1364 | 0.0068 | 0.2246 | 0.1777 | 0.0352 | 0.0137 | var_recall |

> 判据 C4：某腿 ≤ `chance + 2·σ_wilson` ⇒ 在该任务上**不可分辨**（不是「机制无效」，SCHEMA §19）。copy@src0 是唯一落在锚区（绝对位置 [0,anchors)）的源档：只有带锚点的臂对它有效——这是「锚点是不是通用召回杠杆」的判读核心。

## 表 3 · 与来源实验逐一对照（参考臂应逐位一致）

| 点 | 本实验 bpc | 来源实验路径 | 来源值 | abs diff | 一致 |
|---|---|---|---|---|---|

| `fp16` | 1.483571 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 1.483571 | 0.0 | ✓ |

| `int8` | 1.483721 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 1.483721 | 0.0 | ✓ |

| `int4` | 1.529232 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 1.529232 | 0.0 | ✓ |

| `win16` | 2.085526 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 2.085526 | 0.0 | ✓ |

| `win32` | 1.695384 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.695384 | 0.0 | ✓ |

| `win64` | 1.564077 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.564077 | 0.0 | ✓ |

| `win128` | 1.503634 | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 1.503634 | 0.0 | ✓ |

| `win144` | 1.497328 | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 1.497328 | 0.0 | ✓ |

| `win160` | 1.492435 | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 1.492435 | 0.0 | ✓ |

| `win176` | 1.489695 | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 1.489695 | 0.0 | ✓ |

| `win192` | 1.487744 | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` | 1.487744 | 0.0 | ✓ |

| `win16_anchor16` | 1.858406 | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | None | — | — |

| `win32_anchor32` | 1.590539 | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | None | — | — |

| `win32_anchor8` | 1.633097 | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | None | — | — |

| `mha` | 1.482026 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 1.482026 | 0.0 | ✓ |

| `gqa` | 1.496325 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 1.496325 | 0.0 | ✓ |

| `mqa` | 1.493763 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 1.493763 | 0.0 | ✓ |

| `gqa_int8` | 1.496489 | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | None | — | — |

| `mqa_int8` | 1.494105 | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` | None | — | — |

| `gqa_int4` | 1.592538 | `2616 gqa-12k-s0 × R1 int4（本批组合，无单一来源）` | None | — | — |

| `mqa_int4` | 1.689346 | `2616 mqa-12k-s0 × R1 int4（本批组合，无单一来源）` | None | — | — |

| `win32_int8` | 1.695690 | `2615 win32 掩码 × R1 int8（本批组合，无单一来源）` | None | — | — |

| `win64_int8` | 1.564170 | `2615 win64 掩码 × R1 int8（本批组合，无单一来源）` | None | — | — |

| `win16_anchor16_int4` | 1.943239 | `2620 win16_anchor16 掩码 × R1 int4（本批组合，无单一来源）` | None | — | — |

| `win32_anchor32_int4` | 1.643489 | `2620 win32_anchor32 掩码 × R1 int4（本批组合，无单一来源）` | None | — | — |

| `gqa_win64` | 1.559100 | `2616 gqa-12k-s0 × 2615 win64 掩码（本批组合，无单一来源）` | None | — | — |

| `gqa_win128` | 1.512094 | `2616 gqa-12k-s0 × 2615 win128 掩码（本批组合，无单一来源）` | None | — | — |

## 表 4 · 与 2619 参考臂的复现性（跨 run 可控性）

| 点 | 本实验 bpc | 来源实验 bpc | abs diff | copy abs diff | 一致 |
|---|---|---|---|---|---|

| `fp16` | 1.483571 | 1.483571 | 0.00e+00 | 0.00e+00 | ✓ |

| `int8` | 1.483721 | 1.483721 | 0.00e+00 | 0.00e+00 | ✓ |

| `int4` | 1.529232 | 1.529232 | 0.00e+00 | 0.00e+00 | ✓ |

| `win16` | 2.085526 | 2.085526 | 0.00e+00 | 0.00e+00 | ✓ |

| `win32` | 1.695384 | 1.695384 | 0.00e+00 | 0.00e+00 | ✓ |

| `win64` | 1.564077 | 1.564077 | 0.00e+00 | 0.00e+00 | ✓ |

| `win128` | 1.503634 | 1.503634 | 0.00e+00 | 0.00e+00 | ✓ |

| `win144` | 1.497328 | 1.497328 | 0.00e+00 | 0.00e+00 | ✓ |

| `win160` | 1.492435 | 1.492435 | 0.00e+00 | 0.00e+00 | ✓ |

| `win176` | 1.489695 | 1.489695 | 0.00e+00 | 0.00e+00 | ✓ |

| `win192` | 1.487744 | 1.487744 | 0.00e+00 | 0.00e+00 | ✓ |

| `mha` | 1.482026 | 1.482026 | 0.00e+00 | 0.00e+00 | ✓ |

| `gqa` | 1.496325 | 1.496325 | 0.00e+00 | 0.00e+00 | ✓ |

| `mqa` | 1.493763 | 1.493763 | 0.00e+00 | 0.00e+00 | ✓ |

## 表 4 · 与 2620 组合臂的复现性（跨 run 可控性）

| 点 | 本实验 bpc | 来源实验 bpc | abs diff | copy abs diff | 一致 |
|---|---|---|---|---|---|

| `gqa_int8` | 1.496489 | 1.496489 | 0.00e+00 | 0.00e+00 | ✓ |

| `mqa_int8` | 1.494105 | 1.494105 | 0.00e+00 | 0.00e+00 | ✓ |

| `win16_anchor16` | 1.858406 | 1.858406 | 0.00e+00 | 0.00e+00 | ✓ |

| `win32_anchor32` | 1.590539 | 1.590539 | 0.00e+00 | 0.00e+00 | ✓ |

| `win32_anchor8` | 1.633097 | 1.633097 | 0.00e+00 | 0.00e+00 | ✓ |

## 实测显存（**与解析列分列**，判据 C2）

- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）：**462.5 MiB**（全臂共用一个进程，是上界，不是逐臂读数）。
- 为什么不逐臂报：评测实现把 K/V 立刻反量化、窗口/锚点按掩码取子集，SDPA 仍物化 `T×T` 分数矩阵 ⇒ **实测峰值不随档下降**，把它当横轴会得出「量化/窗口不省显存」的错结论。
- 真 cache 实现的实测峰值仍是 U3/2619 表 4 的缺失项（不在本批范围）。

## 仍缺的组合（显式列出）

- **GQA/MQA × int4 × 窗口（三因子）**：本批只叠两个因子；三因子组合未测（本项目 scope 禁组合，需单独授权）。 ⇒ 下一步：若产品侧需要，单列授权三因子臂并给逐字节对照；否则留白。
- **头共享 × 窗口 × 量化（GQA×win×int8）**：combo_mask 的 GQA 臂只测了 fp16（纯掩码）；GQA+窗口+量化未测。 ⇒ 下一步：在 combo_mask 上把 GQA 臂的 mode 换成 int8，2 档直接测点。
- **权重 / 激活量化**：R1 只测了 KV cache 量化；权重/激活量化本项目零经验。 ⇒ 下一步：先分诊（SCHEMA §21：业界已做透 ⇒ 可能直接写 reference 结论）。
- **MLA 低秩 KV（W3 选项之一）**：零经验；低秩投影的压缩率不是本图的单一乘式（需报秩 r）。 ⇒ 下一步：先把横轴公式扩展为 2·d·r 形式，再测 13M。
- **摘要压缩（M2 选项之一）**：零经验；压缩率依赖摘要器，不是单一乘式。 ⇒ 下一步：先定摘要器与预算口径，再测；注意 SCHEMA §17.9 的成本交叉点。
- **长 T（>256）下的窗口 / 驱逐 / 量化**：本图所有点都在 T=256 上评测；窗口的 bytes/pos = 12288·kept/T 随 T 下降。 ⇒ 下一步：在 T∈{512,1024,4096} 上重测窗口/量化档的 bpc 与合成召回。
- **各臂的**实测**显存峰值（真 cache 实现）**：评测实现把 K/V 立刻反量化、窗口按掩码取子集（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值不随档下降，本实验只报解析列（判据 C2）。 ⇒ 下一步：写一个真正的 cache 实现（分页/分组存储 + 只在需要时反量化），再测部署口径峰值。
- **召回腿的真实文本针（needle v3）跨族**：2615 只给 trunk 族的窗口档测了真实文本针召回；量化/头共享/组合族没有该腿（不同仪器不得混列，本图未并入）。 ⇒ 下一步：用同一 2615 仪器补跑新组合臂的针召回，作为**第二条召回腿**单列。
- **GQA/MQA 的等参数对照臂**：W3 的自述缺口：GQA/MQA 参数量比 MHA 少 8.75%/10.25%，代价混着容量差；本批所有含头共享的臂继承该混杂。 ⇒ 下一步：补等参数对照（加宽 MLP 或少层），再判头共享的净代价。

## ASCII 图

```
x 轴 = log10(每位置 KV 字节)（左小而省 → 右大而贵）；y 轴 = damage_bpc（上=更差）

+0.6020 |               w                                                            
+0.5779 |                                                                            
+0.5538 |                                                                            
+0.5297 |                                                                            
+0.5056 |                                                                            
+0.4816 |                                                                            
+0.4575 |t                                                                           
+0.4334 |                                                                            
+0.4093 |                                                                            
+0.3853 |                              a                                             
+0.3612 |                                                                            
+0.3371 |                                                                            
+0.3130 |                                                                            
+0.2889 |                                                                            
+0.2649 |                                                                            
+0.2408 |                                                                            
+0.2167 |e              s              w                                             
+0.1926 |                                                                            
+0.1685 |               T                                                            
+0.1445 |                                   n                                        
+0.1204 |               d                                                            
+0.0963 |                                             A                              
+0.0722 |               h              S              w                              
+0.0482 |                                             4                              
+0.0241 |               q              gH             G              W  W            
+0.0000 |                              Q                             8    W W W    Mb
         +----------------------------------------------------------------------------
               384      913     2172     5166    12288
          每位置 KV 字节（解析）

标记→点：b=fp16  8=int8  4=int4  w=win16,win32,win64  W=win128,win144,win160,win176,win192  a=win16_anchor16  A=win32_anchor32  n=win32_anchor8  M=mha  G=gqa  Q=mqa  g=gqa_int8  q=mqa_int8  d=gqa_int4  e=mqa_int4  s=win32_int8  S=win64_int8  t=win16_anchor16_int4  T=win32_anchor32_int4  h=gqa_win64  H=gqa_win128
标记→类别：baseline=(不压缩基线)  quant=(量化（R1）)  window=(窗口/驱逐（M2）)  head=(头共享（W3）)  quant_head=(组合：头共享 × 量化)  window_anchor=(组合：窗口 × 锚点)  combo_quant_window=(组合：窗口/锚点 × 量化（本批）)  combo_head_window=(组合：头共享 × 窗口（本批）)
```
