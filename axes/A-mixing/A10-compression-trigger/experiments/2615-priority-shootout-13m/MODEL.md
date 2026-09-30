# MODEL · 2615 三种压缩优先级对决（13M dense，纯推理）

**本实验不产生任何新权重**（纯推理：评测期只改「删除哪些位置」，模型权重逐位不动）。
下表列出**复用**的被测量权重与运行时强制断言。

## 被测量主干（复用，只读）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M，2610 记录） |
| 体积 | 54,193,675 B（54.2 MB，本实验 `os.path.getsize` 复测） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0（2610 记录） |
| 记录 bpc | 1.4836（clean，24×16 窗口）——本实验同一窗口序复算得 **1.483569**（= 契约 M/2614 记录值） |

`run.py::load_trunk` 在加载前断言 sha256，不匹配直接 `SystemExit`（不会静默使用别的权重）；
`results.json::checkpoint.sha256_verified = true`，且与契约 M 记录的 sha256 **逐字相同**
（`reuse_check_contract_M.checkpoint_same = true`）。

## 生成命令（2610 侧，本实验未重跑）

```bash
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 两条腿 × 24 判据/敏感臂 + 56 点预算曲线 + 12 个正文限制臂 + 注意力质量分布 + 重叠度 + 三道对账素材 + 噪声底 + 复现性 + 成本 |
| `retention_log.jsonl` | 39 行：依赖腿逐批（每 4 批一行，含逐臂被删位置数）、保留腿逐批（全保留/一条 LRU 召回）、逐配置汇总 |
| `run_log.txt` | 全电池 stdout（`python3 run.py > run_log.txt 2>&1`；墙钟 101.57 s（共享卡；四次全跑 100.01 / 124.99 / 103.56 / 101.57 s），4444 次前向） |
| `mutation_report.json` | 10 处变异验证（统计量因果/对角线/按头平均、排序方向、对角线保留、查询轴错位、尾部切片、候选池、重叠度、代价方向）+ 还原后 sha256 一致性 |
| `analysis.md` / `stability.md` | 判读（第一段人话摘要）/ 稳定性与噪声底专报 |
| `reconcile_numbers.py` | 对账工具：逐条断言报告里引用的 **278** 个数字 == `results.json`（`python3 reconcile_numbers.py`，纯 CPU <1 s） |
| `smoke/` | 冒烟产物（`python3 run.py --smoke`，墙钟 3.51 s；**不含**外推时间字段，SCHEMA §17.5） |

## 共享底座改动（只增不改；SCHEMA §3 迁移计划下的 evalkit）

| 文件 | 改动 | 回归 |
|---|---|---|
| `tools/evalkit/retention.py` | 新增「优先级规则」一节：`attention_stats` / `priority_drop_mask` / `_rank` / `drop_mask_keep` / `drop_mask_overlap` / `drop_mask_stats` / `attention_stats_summary` / `needle_score`；`masked_logits` 支持 `[B,T,T]` 逐样本掩码；`masked_bpc` / `row_bpc` 新增 `mask_fn`；`needle_fate` / `needle_recall` 支持逐样本掩码与逐行桶判定 | 既有 16 条单测全绿；新增 11 条全绿；变异验证 10/10 抓红；`python3 -m pytest tests/ -q` = **216 passed**；sha256 `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502`（= 全电池运行时的同一份文件） |
| `tests/test_retention.py` | 新增 11 条（统计量逐项对手写参考 / 因果与归一化闭式 / 规则预算与确定性 / 规则语义与平局次序 / 与 `prefix_keep_mask` 逐位一致 / 重叠度 / 规则活性 / 分布报告 / 逐样本掩码管路 / 候选池 / `drop_mask_stats` 与 `mask_stats` 同口径） | 同上 |

**跨契约说明（给复核者）**：契约 M 的 `results.json::code.retention_py_sha256` 记录的是它运行时的
`f1c6a8f3…`；本实验在其后**追加**了「优先级规则」一节，故当前文件是 `8320f03a…`（只增不改既有函数
语义）。判据 M 的数字不受影响的证据：本实验在同一进程内重算它的 `prefix:N∈{8,32,64}` 三点，
最大差 **2.59e-07**；M 的 16 条单测在本实验修订后仍全绿。

**对本仓库既有实验行为零改动**：`masked_bpc(prefix_keep_mask)` 在本进程内复算契约 M 记录的
`prefix:N∈{8,32,64}`，最大 |Δbpc| = **2.59e-07**（`results.json::reuse_check_contract_M`）。
