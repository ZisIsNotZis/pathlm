# REVIEW-1 · 复核者列为「无法执行」的命令，由 axis5-lead 代跑（原始输出）

时间：2026-10-13T20:19:20+08:00｜actor：axis5-lead｜原因：复核者会话无 exec/curl 工具（REVIEW-1 I1）

## ① python3 axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py
```
parent results.json sha256 = 0e64b05b38cfd6154e2f0b1c52241bc0b52c146e044f10ba43a6d884fe3c3d2d
parent config.json  sha256 = 63b47cbeb2aa702c4170eff42bd917a222901f0406394f43c458b0b1d34f4f05
input axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/results.json: OK
input axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/config.json: OK
input axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt: OK
input axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt: OK
input axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt: OK
input axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt: OK
必须不变 (a) 透传事实 = True（非独立检验）
必须不变 (b) 跨文档独立核（2632/table.md 表 1 vs 2642 rows）：解析 27 臂，max|Δ| = {'deployment_bytes_per_position': 0.0, 'bpc': 0.0, 'recall_copy': 4.900000000000043e-05, 'recall_copy_ci': 4.900000000002125e-05}，不一致 = 0 ⇒ PASS
上游独立检验：2632::reuse.nonquant_bitwise_preserved = True（max|Δbpc| = 0.0，相对 2631 的检验）
G1 独立重算 vs 2632 记录：max|Δ部署字节| = 0 ⇒ PASS
G2 量化臂标尺口径：非块标尺量化臂 = [] ⇒ PASS
仅值口径核对：points[*].bytes_per_position 与独立重算的仅值字节不一致条目 = 0
在力排序带 = 0.036324 bpc（2632::ranking_band）
参数混杂 gqa: 磁盘实测 MHA−GQA = 1184256（= 8.7484%），解析 = 1184256 ⇒ 一致
参数混杂 mqa: 磁盘实测 MHA−MQA = 1381632（= 10.2065%），解析 = 1381632 ⇒ 一致
MHA 总参数量 = 13536768，trunk 参数量 = 13536768（同架构，差 0）
Pareto（原始 bpc 序）：6 臂 = ['mqa_int4', 'gqa_win64', 'mqa_int8', 'mqa', 'int8', 'mha']
Pareto（越 0.036324 带才算更好）：16 臂 = ['mqa_int4', 'gqa_win64', 'mqa_int8', 'mqa', 'gqa_win128', 'gqa_int8', 'gqa', 'int4', 'win128', 'int8', 'win144', 'win160', 'win176', 'win192', 'mha', 'fp16']
输出：results.json（27 臂）、table.md（7 个同字节组）
结论摘要：G1 PASS；同字节跨家族组 = 7；其中 bpc 越带 = 7；参数量混杂已拆 = True
exit=0
```
## ② sha256sum 六个输入
```
0e64b05b38cfd6154e2f0b1c52241bc0b52c146e044f10ba43a6d884fe3c3d2d  axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/results.json
63b47cbeb2aa702c4170eff42bd917a222901f0406394f43c458b0b1d34f4f05  axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/config.json
86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2  axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt
557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528  axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt
bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00  axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt
24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97  axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt
```
## ③ python3 tools/summary.py --check
```
OK：46 个 project.yaml 全部符合 docs/research-schema.md。
exit=0
```
## ④ results.json::code 脚本指纹 vs 磁盘
```
recorded 4a9297177dd6c3c54e6eced13fd3f3187bdb6f91f79fd154be7cc8fa87f8c91b
on disk  4a9297177dd6c3c54e6eced13fd3f3187bdb6f91f79fd154be7cc8fa87f8c91b
match True
verdict PROCEED | basis: 只由 G1（独立重算字节账）与 G2（块标尺口径）决定；跨文档独立核另列（cross_artifact_table_md）
cross_artifact_table_md_ok True
```
