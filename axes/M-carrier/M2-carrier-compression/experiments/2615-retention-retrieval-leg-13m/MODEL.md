# MODEL · 2615 保留曲线补「检索腿」（13M dense，纯推理）

**本实验不产生任何新权重**（纯推理：评测期改注意力可见集 + 在窗口里植入答案来自窗口内
那次出现的真实文本短语）。下表列出**复用**的被测量权重与运行时强制断言。

## 被测量主干（复用，只读）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 体积 | 54,193,675 B（54.2 MB） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0（2610 记录） |
| 记录 bpc | 1.4836（clean，24×16 窗口）——本实验用同一窗口序复算得 **1.483569** |
| 关键属性（本实验实测） | **纯文本训练（`p_needle = 0`）⇒ 没有「任意内容复制」能力**（随机字节对针 1.5%、随机 token 复读 10%、任意答案复制 6%）；只有「续写真实文本」的能力（真实短语针 68.7%） |

实测 sha256 与 2614 复核记录的 `86953dcb…159d2` 一致；`run.py::load_trunk` 在加载前断言，
不匹配时直接 `SystemExit`，不会静默使用别的权重。

## 生成命令（2610 侧，本实验未重跑）

```bash
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 两条腿 × 39 配置 + v2 构造诊断 + 无针地板 + 判据块 + 配对/难度控制口径 + 噪声底 + 复现性 + 成本 |
| `retention_log.jsonl` | 82 行：v2 诊断逐批、针批次逐批（召回/地板/分桶）、逐配置汇总（活性量） |
| `run_log.txt` | 全电池 stdout（`python3 run.py > run_log.txt 2>&1`） |
| `mutation_report.json` | 4 处「针被删错/放错/半删当没删/最近 d 差一」变异验证 + 还原后 sha256 一致性 |
| `analysis.md` / `stability.md` | 判读（第一段人话摘要）/ 稳定性与噪声底专报 |
| `smoke/` | 冒烟产物（`python3 run.py --smoke`，约 4 s；**不含**外推时间字段） |

## 共享底座改动（只增不改，SCHEMA §3 迁移计划下的 evalkit）

| 文件 | 改动 | 回归 |
|---|---|---|
| `tools/evalkit/retention.py` | 新增 `recent_keep_mask` / `phrase_needle_batch` / `needle_fate` / `needle_recall` / `needle_summary`（追加在文件末尾 + 扩展模块 docstring） | 既有 11 条单测全绿；变异验证 4/4 抓红；sha256 `f1c6a8f377af2cec3bb0b4c8fc360745affb46a99f1c44ca1f14d28342e16978` |
| `tests/test_retention.py` | 新增 5 条（`recent_keep_mask` 闭式 / needle v3 布局与唯一性 / 桶互斥 / 半删判被删 / 召回计数与分桶） | `python3 -m pytest tests/ -q` 全绿 |

**本实验对 2614 的仪器行为零改动**：`prefix` 曲线在本进程内重算与 2614 记录**逐点相同**
（0.196528 / 0.344876 / 0.640980 / 1.248019 / 2.429482），见 `analysis.md` §5.2。
