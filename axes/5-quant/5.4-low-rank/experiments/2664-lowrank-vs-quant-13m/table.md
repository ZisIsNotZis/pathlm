# 2664 · 同字节判决表（`results.json` 的渲染）

> 本文件由 `run.py::render_table` 从 `results.json` 生成；判读见 `analysis.md`。

## 1. 逐臂读数

| 臂 | 模式 | 运行 | bpc | 每位置字节 | ×fp16 | 参数 | 前缀不变 |
|---|---|---|---|---|---|---|---|
| `mha` | fp16 | `mha-12k-s0` | 1.482022 | 12288.0 | 1.0 | 13536768 | True |
| `mha` | int8 | `mha-12k-s0` | 1.482004 | 6528.0 | 0.53125 | 13536768 | True |
| `mha` | int4 | `mha-12k-s0` | 1.505358 | 3456.0 | 0.28125 | 13536768 | True |
| `mha` | fp16 | `mha-s1` | 1.482822 | 12288.0 | 1.0 | 13536768 | True |
| `mha` | int8 | `mha-s1` | 1.482874 | 6528.0 | 0.53125 | 13536768 | True |
| `mha` | int4 | `mha-s1` | 1.501846 | 3456.0 | 0.28125 | 13536768 | True |
| `mha_rep` | fp16 | `mha-rep-s0` | 1.479584 | 12288.0 | 1.0 | 13536768 | True |
| `mha_rep` | int8 | `mha-rep-s0` | 1.479597 | 6528.0 | 0.53125 | 13536768 | True |
| `mha_rep` | int4 | `mha-rep-s0` | 1.503618 | 3456.0 | 0.28125 | 13536768 | True |
| `mla_c272` | latent_fp16 | `mla-c272-s0` | 1.48488 | 6528.0 | 0.53125 | 14464512 | True |
| `mla_c272` | latent_fp16 | `mla-c272-s1` | 1.481292 | 6528.0 | 0.53125 | 14464512 | True |
| `mla_c272_eq` | latent_fp16 | `mla-c272-eq-s0` | 1.484399 | 6528.0 | 0.53125 | 13536495 | True |
| `mla_c272_eq` | latent_fp16 | `mla-c272-eq-s1` | 1.489149 | 6528.0 | 0.53125 | 13536495 | True |
| `mla_c144` | latent_fp16 | `mla-c144-s0` | 1.488682 | 3456.0 | 0.28125 | 13284864 | True |
| `mla_c144` | latent_fp16 | `mla-c144-s1` | 1.475687 | 3456.0 | 0.28125 | 13284864 | True |
| `mla_c128` | latent_fp16 | `mla-c128-s0` | 1.48073 | 3072.0 | 0.25 | 13137408 | True |
| `mla_c128` | latent_fp16 | `mla-c128-s1` | 1.477781 | 3072.0 | 0.25 | 13137408 | True |
| `gqa2` | fp16 | `gqa-12k-s0` | 1.496324 | 3072.0 | 0.25 | 12352512 | True |
| `gqa2` | int8 | `gqa-12k-s0` | 1.496441 | 3072.0 | 0.25 | 12352512 | True |
| `gqa2` | int4 | `gqa-12k-s0` | 1.553126 | 3072.0 | 0.25 | 12352512 | True |
| `gqa2` | fp16 | `gqa-12k-s0-rep` | 1.495938 | 3072.0 | 0.25 | 12352512 | True |
| `gqa2` | int8 | `gqa-12k-s0-rep` | 1.496061 | 3072.0 | 0.25 | 12352512 | True |
| `gqa2` | int4 | `gqa-12k-s0-rep` | 1.551551 | 3072.0 | 0.25 | 12352512 | True |
| `gqa2_eq` | fp16 | `gqa2-eq-s0` | 1.494677 | 3072.0 | 0.25 | 13536516 | True |
| `gqa2_eq` | int8 | `gqa2-eq-s0` | 1.494862 | 3072.0 | 0.25 | 13536516 | True |
| `gqa2_eq` | int4 | `gqa2-eq-s0` | 1.547532 | 3072.0 | 0.25 | 13536516 | True |
| `gqa2_eq` | fp16 | `gqa2-eq-s1` | 1.481265 | 3072.0 | 0.25 | 13536516 | True |
| `gqa2_eq` | int8 | `gqa2-eq-s1` | 1.481385 | 3072.0 | 0.25 | 13536516 | True |
| `gqa2_eq` | int4 | `gqa2-eq-s1` | 1.521089 | 3072.0 | 0.25 | 13536516 | True |

## 2. 同字节格（主判）

| 字节/位置 | 低秩 | 对照 | 配对 Δ（低秩−对照） | 带 | 裁决 |
|---|---|---|---|---|---|
| 6528 | `mla_c272` | `mha@int8` | 0.000647 | ±0.014 | indistinguishable_in_band |
| 3456 | `mla_c144` | `mha@int4` | -0.021417 | ±0.014 | low_rank_better_sortable |
| 3072 | `mla_c128` | `gqa2@fp16` | -0.016876 | ±0.014 | low_rank_better_sortable |

逐对明细：
- 6528 B：`mla-c272-s0` − `mha-12k-s0` = 0.002876（seed 0 / 0，对齐=True）
- 6528 B：`mla-c272-s1` − `mha-s1` = -0.001582（seed 1 / 1，对齐=True）
- 3456 B：`mla-c144-s0` − `mha-12k-s0` = -0.016676（seed 0 / 0，对齐=True）
- 3456 B：`mla-c144-s1` − `mha-s1` = -0.026159（seed 1 / 1，对齐=True）
- 3072 B：`mla-c128-s0` − `gqa-12k-s0` = -0.015594（seed 0 / 0，对齐=True）
- 3072 B：`mla-c128-s1` − `gqa-12k-s0-rep` = -0.018157（seed 1 / 0，对齐=False）

## 3. 族内失效模式

- `quant_vs_fp16` / `int8`：Δbpc 均值 1.7e-05（mha-12k-s0=-1.8e-05, mha-s1=5.2e-05）
- `quant_vs_fp16` / `int4`：Δbpc 均值 0.02118（mha-12k-s0=0.023336, mha-s1=0.019024）
- `lowrank_vs_mha_fp16` / `mla_c272`：Δbpc 均值 0.000664（mla-c272-s0=0.002858, mla-c272-s1=-0.00153）
- `lowrank_vs_mha_fp16` / `mla_c272_eq`：Δbpc 均值 0.004352（mla-c272-eq-s0=0.002377, mla-c272-eq-s1=0.006327）
- `lowrank_vs_mha_fp16` / `mla_c144`：Δbpc 均值 -0.000237（mla-c144-s0=0.00666, mla-c144-s1=-0.007135）
- `lowrank_vs_mha_fp16` / `mla_c128`：Δbpc 均值 -0.003167（mla-c128-s0=-0.001292, mla-c128-s1=-0.005041）

## 4. 召回（配对 bootstrap，a − b）

- 6528 B `mla_c272` − `mha@int8` @ copy_by_pos: Δ 0.03776，95% CI [0.022786, 0.053711]，n=3072
- 6528 B `mla_c272` − `mha@int8` @ induction_by_start: Δ 0.066406，95% CI [0.039062, 0.09375]，n=512
- 6528 B `mla_c272` − `mha@int8` @ var_recall_by_d: Δ 0.008464，95% CI [-0.001302, 0.01888]，n=1536
- 3456 B `mla_c144` − `mha@int4` @ copy_by_pos: Δ -0.005208，95% CI [-0.018888, 0.008797]，n=3072
- 3456 B `mla_c144` − `mha@int4` @ induction_by_start: Δ 0.0，95% CI [-0.021484, 0.021533]，n=512
- 3456 B `mla_c144` − `mha@int4` @ var_recall_by_d: Δ -0.010417，95% CI [-0.018229, -0.002604]，n=1536
- 3072 B `mla_c128` − `gqa2@fp16` @ copy_by_pos: Δ -0.031576，95% CI [-0.044596, -0.016927]，n=3072
- 3072 B `mla_c128` − `gqa2@fp16` @ induction_by_start: Δ 0.039062，95% CI [0.017578, 0.0625]，n=512
- 3072 B `mla_c128` − `gqa2@fp16` @ var_recall_by_d: Δ -0.005208，95% CI [-0.01237, 0.001302]，n=1536

## 5. 仪器自检与字节账

- 路径逐位：{'bpc_of': 1.482026, 'masked_bpc': 1.482026, 'bitwise_equal': True}
- 低秩前向 == MLABlock：{'arm': 'mla_c128', 'max_abs_delta': 0.0, 'bitwise_equal': True}
- 跨实验 5.1 口径复现：{'checkpoint': 'axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt', 'sha256': '86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2', 'expected': {'fp16': 1.483571, 'int8': 1.483672, 'int4': 1.506093}, 'got': {'fp16': 1.483571, 'int8': 1.483672, 'int4': 1.506093}, 'reproduces_2632': True}
- fp16 存储效应（MHA）：{'bpc_kv_path_fp16': 1.482022, 'bpc_pure_fp16_mha': 1.482026, 'delta': -4e-06, 'note': '量化路径对 K/V 做 fp16 存储往返（部署口径），与纯 fp32 前向的差是**存储精度**效应；低秩臂的潜向量往返同源（见各臂 bpc_roundtrip_effect）'}
