# 复现清单三态判定的抽检（`rebuildable_only` 是否判宽了）

## 人话摘要

`docs/repro.md` 把 120 个实验分成三态：**克隆内可重算 0 / 只能重训 86 / 不可重建 34**。
零个「可重算」很可疑——是不是规则把「其实只要用仓内的 `results.json` 做纯 CPU 重算」的实验，
因为脚本里提了一句 `train/` 就判成了「要重训」？

我在**最可能被误判的那一类**里抽了 5 个逐个核：4 个是「README 声明的复现命令本身不触训练、但仍被判要重训」
的全部候选（`2652` / `2646` / `2650` / `2642-kv-same-byte-frontier`），另 1 个是批量样本（`2626`）。
结论：**5/5 的判定都成立**——它们的数字确实要么来自一次 GPU 训练，要么要读别的实验没入库的权重。
**规则未改**，三态分布不变。

## 抽检对象的选取方法（不是随机抽）

规则过宽只会在一种实验上暴露：**声明的复现命令本身不触训练，却被判 `rebuildable_only`**。
所以先机检整批 86 个：82 个的入口脚本自身就命中训练 token（`train_depth_ar` / `pathlm.data` /
`load_enwik8` / `torch.cuda` / `DEVICE = "cuda"`），或入口就是 `train_*.sh` / `run_all.sh`——
这一批的判定没有争议。剩下 4 个是全部候选，全部纳入抽检；再从 82 个里取 1 个（`2626`，
它的判定依据指向的不是入口脚本，用来核「依据归因」这一半）。

## 逐个核对

| 实验 | 判定 | 判定依据（生成物给的） | 我核过的事实 | 成立？ |
|---|---|---|---|---|
| `2626-longT-clean`（`axes/1-time/1.1.1-hard-window/experiments/2626-longT-clean`） | `rebuildable_only` | `clean_driver.py` 触训练（命中 `train_depth_ar`） | 入口 `run.py` 自身就 `from pathlm.data import load_enwik8_full`、`DEVICE = "cuda"`、有 `--stage train`，语料取 `.tmp/enwik8`（不入库）；`results.json` 的 clean bpc 是训练后评测该权重的读数 | 成立 |
| `2646-layer-width-13m`（`axes/3-width/3.1.2-layer-width/experiments/2646-layer-width-13m`） | `rebuildable_only` | `run.py` 触训练（命中 `train_depth_ar`） | README §8 的第一条复现命令是纯 CPU 的 `python3 search_schedule.py`（只写 `schedule_search.json`）；但实验的四个数字文件 `results_{EQ,PYR,HG,APYR}_s0.json` 来自 `bash train_arms.sh` / `python3 run.py --arm … --steps 12000`，而 `--no-train` 那条复用 `runs/…/model.pt`，`runs/` 不入库 | 成立 |
| `2652-residual-form-13m`（`axes/2-depth/2.2.1-residual-form/experiments/2652-residual-form-13m`） | `rebuildable_only` | `common.py` 触训练（命中 `train_depth_ar`） | 入口命令是纯 CPU 的 `python3 mutation_check.py`（变异检查），但四臂数字在 `results_{base,gated,postln,scaled}_s0.json`，由 `run.py` 逐臂训练 12k 步得出；`run.py --no-train` 复用 `runs/…/model.pt`（不入库） | 成立 |
| `2650-verify-gate-13m`（`axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m`） | `rebuildable_only` | `verify_gate.py` 触 CUDA（命中 `torch.cuda`）+ 引用 `../2631-spec-three-arms/train/13m_s0`、`…/13m_s1` | 入口 `run.py` 只是转发 `verify_gate.main()`；`2631/train/13m_s{0,1}/` 入库的只有 `results.json` 与 `train_log.jsonl`，权重 `model.pt`（52 MB）被 `.gitignore` 的 `*.pt` 排除 ⇒ 克隆里没有可加载的 checkpoint | 成立 |
| `2642-kv-same-byte-frontier`（`axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier`） | `rebuildable_only` | 引用不入库的产物：`2610-frozen-trunk-jumphead-13m/trunk.pt`、`W3-kv-head-sharing/2616-gqa-mqa-13m/runs/{mha,gqa}-12k-s0/model.pt` | `frontier_ledger.py` 直接把这些路径当 checkpoint 加载；`trunk.pt`（51.7 MB）本机存在但不在索引里，`runs/<arm>-12k-s0/model.pt` 同（`runs/` 与 `*.pt` 都被 `.gitignore` 排除） | 成立 |

## 结论与是否改规则

- **抽检 5/5 成立，规则未改**：`tools/inventory_runs.py` 的判定逻辑与 `docs/repro.md` 的生成结果都不动。
- 三态分布前后一致：`reproducible 0 / rebuildable_only 86 / not_reproducible 34`（改动前 = 改动后）。
- 上一轮执行者担心的「被 `train/` 引用误判」在这 5 个上不成立：真正把数字钉死的不是「代码里出现
  `train/` 这个词」，而是**数字来自一次训练**（`results_*.json` 是那份权重的评测输出）或**要读别人没入库的权重**。
  仓里唯一的纯 CPU 脚本（`search_schedule.py`、`mutation_check.py`）算的是**旁证**（schedule 搜索、
  变异自检），不是实验的读数，所以把它们当入口不改变实验本身要重训这一事实。
- `reproducible = 0` 是真的：全树没有任何实验的读数能只用克隆内已入库文件纯 CPU 重算出来。
  代价是这一态的判别力暂时为零（它现在只是「没有例外」），一旦有实验把读数落成纯派生的聚合 json，
  它就会亮。记在这里，不改规则。

## 已知的读法陷阱（不改判定，留给读者）

- `2650` 的依据写「`../2631-spec-three-arms/train/13m_s0` …（克隆里不存在）」。严格读**目录**在库
  （里面有 `results.json` 与 `train_log.jsonl`），缺的是目录内没入库的 `model.pt`。判 `rebuildable_only`
  是对的，但这句话要按「它要的**产物**（权重）不在库」读，不能按「这个路径不存在」读。
- `2626` / `2672` 的依据把触发点归到某个**非入口**脚本（`clean_driver.py` / `config_match.py`）。
  `_train_evidence` 扫的是实验目录下**所有** `.py` / `.sh`，不是只扫入口——这是有意的（宁可判「要重训」，
  也不把一次训练误报成可重算），代价是依据里的文件名不等于「你该跑的那个」。
