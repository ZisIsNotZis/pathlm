# 轴 5 Phase A · 变更清单与指纹

HEAD = 1ffce0b1c9f8a1f2037f7a0dbf63b350fc4590b8｜生成于 2026-10-13T20:20:41+08:00｜actor = axis5-lead

## 提交（df0601b..HEAD）
1ffce0b 轴 5 · ticket 66 §7.4：登记跨文档独立核的已知边界（静默漏行守门、5e-5 容差、独立性强度定位）
d9794b5 轴 5 · ticket 66：第 1 轮独立复核判词落盘 + 处置 N1/N2（含新增跨文档独立核）
98ffb41 轴 5 · ticket 66 §7.2：补记 axis5-lead 第一手手算（7 臂部署字节 vs 2632 逐臂相等）
9359530 轴 5 · 5.1-WP2（2642）复核期间 writer 自纠 + 机检记录（纯文档，未改数值）
f37d12e 轴 5 量化轴 Phase A（ticket 66）：六格 role_map 实名 + SOTA 出处核实 + 5.1-WP2 跨家族同字节前沿重画

## 改动文件（相对 df0601b）
A	.scratch/66-axis5-quant/REVIEW-1.md
A	.scratch/66-axis5-quant/TICKET.md
A	.scratch/66-axis5-quant/evidence/README.md
A	.scratch/66-axis5-quant/evidence/REVIEW-1-COMMANDS.md
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/MODEL.md
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/README.md
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/analysis.md
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/config.json
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/run_log.txt
A	axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/table.md
M	axes/5-quant/5.1-kv-quant/project.yaml
M	axes/5-quant/5.2-weight-quant/project.yaml
M	axes/5-quant/5.3-activation-quant/project.yaml
M	axes/5-quant/5.4-low-rank/project.yaml
M	axes/5-quant/5.5-fp8-head-dim/project.yaml
M	axes/5-quant/5.6-window-quant-retrieval/project.yaml

## 关键产物指纹（sha256）
5c5a2662d9048f9fb416bfa5022aa1cd7616adeebeb5e64cca957e7c9ee44259  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/MODEL.md
dba695d94f7da8e5899335297ee4423b7bf65fa8ef02f42ce309aeb5b3367344  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/README.md
9ff5fe6147017e02527d13a7676ca23bd86b80d7fd1546b7e03f813642ff2b0b  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/analysis.md
20f8a168a503c0249bb1cf94af823786cee3473bc1ac2797cc88c33e7688c1be  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/config.json
4a9297177dd6c3c54e6eced13fd3f3187bdb6f91f79fd154be7cc8fa87f8c91b  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py
ca166b3f5ff01a9ca66e8dd511e5a9f2f06be883053ce7944238717df719058b  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json
93a188a5e00924a6fded2a006749eea939effd4aa1c42260b723bffe7e600416  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/run_log.txt
b3f6120eb861fd4232d40a2199e5d4ebab7176c3ccdb7cab817eae80c3e3ef2b  axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/table.md
0273328e337662829a44ac95316c9bf2e4d34bd45f4771ca8ed660f0733fd209  axes/5-quant/5.1-kv-quant/project.yaml
c373f7873dd86d6d6b4bce56df8e005777f6ef8b4e03a881524679b69a15ee40  axes/5-quant/5.2-weight-quant/project.yaml
936b34a4e145721902833b78aab3caa48a6fdaa0e0d686b6b5e9f394141e0ca4  axes/5-quant/5.3-activation-quant/project.yaml
55ef3301684ee90f54e1379720f573c435bc8cf796ec3adc2a2cf6f1cfff2d02  axes/5-quant/5.4-low-rank/project.yaml
a032a4338171ae8bd2ffe666078bb6e281b8860764766bcf30da23b3c206f061  axes/5-quant/5.5-fp8-head-dim/project.yaml
86105da0101615726ce4423b77af45b0651e2814744dc6f8377a8f0f58c31e52  axes/5-quant/5.6-window-quant-retrieval/project.yaml
3697fc854b7b6a2589dc15c8827b552d4bc4c3e638134324884d7b79e39f2689  .scratch/66-axis5-quant/TICKET.md
44e7882254f8457bb6040a3e93dc73e2c3faa44d1b4b623e5e660ff79858be2f  .scratch/66-axis5-quant/REVIEW-1.md
