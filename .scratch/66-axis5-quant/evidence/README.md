# `.scratch/66-axis5-quant/evidence/` — 证据索引

本目录只放**指路**（证据本体在各自的家，不复制）。

| 证据 | 家 | 说明 |
|---|---|---|
| 5.1-WP2 判决包（跨家族同字节预算重画，块标尺口径） | `axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/` | 预登记契约 `README.md`+`config.json`；入口 `frontier_ledger.py`；结果 `results.json`；表 `table.md`；判读 `analysis.md`；权重清单 `MODEL.md`；日志 `run_log.txt` |
| 块标尺口径的上游读数（27 臂） | `axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/` | 本 WP 的**唯一数据源**（运行时以 sha256 校验） |
| 头族 checkpoint（参数量与 sha256） | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/{mha,gqa,mqa}-12k-s0/model.pt` | 只读参数量，权重不入库 |
| trunk checkpoint | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | 同上 |
| 独立复核报告（第 1 轮） | `.scratch/66-axis5-quant/REVIEW-1.md` | fresh 上下文、只读 |
| 过程流水（本票） | `.scratch/66-axis5-quant/TICKET.md` | 显式时间戳、判据、预算、复核派单与处置 |
