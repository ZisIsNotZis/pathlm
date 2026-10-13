# WP7 报告 · ticket 56 M2 · 轴 7 参照 + 轴 8 工程纪律 + 轴 9 Backlog

时间：2026-10-13T11:32:11+08:00 ｜ actor：worker（fresh 上下文，单一写入者）｜ commit：`bacaaeb`

## 人话摘要（第一段）

把旧仓库里「拿来当跑分线、不当研究做」的四条基线（全注意力/位置语义·RoPE/GQA·MQA/既有早退），
连同它们原来的实验目录，整体搬进新的**参照台账** `axes/7-references/`，并在台账里把每条写成
「必须打败的跑分线」（是什么 + 要打败它什么 + 数字 + 证据落点）；同时新建**工程纪律台账**
`axes/8-engineering/`（精度/初始化/归一化/损失权重/数据构造，含损失权重那条的扫描实验）；
再把两条已判「移出主线」的研究线做成 **Backlog 研究项**：`9.1 调度 / 选择器`（原 G 轴：选择器 /
观测信号 / 校准——按 55-review 改成「同预算最优固定路径」判据、禁工具自洽 KPI）与
`9.2 迭代精化的其余`（原 F 轴：回传通道 / 训练信号——按 55-review 补目的、换掉被永久降级的分布外
主判据腿）。全程**只搬不复制、零删除**（工作树文件数 2504 → 2510，只增了 6 份新文档），
所有证据目录用 `git mv` 整搬，`.scratch/**/evidence` 原地不动只作锚点。搬完跑通了三条自检；
搬 R7 时打断了别轴两处指向它的活指针（轴 2 的 `2.1.1`、轴 6 的 `6.2`）与一处单测 import 路径，
已按新家一并修好（这是本次搬移直接造成的，不是扩范围）。

## 一、交付物（逐项）

### 轴 7 · 参照台账（不写 `project.yaml`）

- `axes/7-references/README.md`：登记表把四条参照写成「必须打败的跑分线」并给出证据落点：
  - **全注意力基线**（旧格 A2）：100% 归一基线；13M B0 = 1.5074 bpc 快照 / 1.3459 渐近，100M = 1.4612。
  - **RoPE（位置语义族，旧格 A8）**：长上下文前提；本仓零经验。
  - **GQA/MQA（KV 头共享，旧格 W3）**：KV 预算轴同字节对手；mha 1.4820 / gqa 1.4963（+0.0143，
    重跑 +0.0139 跨门槛 ⇒ 判读作废）/ mqa 1.4938；KV 12288→3072/1536 B。
  - **CALM/LayerSkip/MoD（既有早退，旧格 R7）**：深度/投机轴对手；CALM 3.43/8 层、LayerSkip
    接受率 0.9109、MoD isoFLOP +0.398 bpc。
  - 另保留他轴登记项（ALiBi→`1.1.2`、量化基线→`5-quant`、`1.3 寻址`、`6.1 自回归`）。
- 证据整搬（`git mv` 整目录，47 个 tracked 文件，全部 100% rename）：
  - `axes/A-mixing/A2-full-attention/experiments` → `axes/7-references/experiments/A2-full-attention`
  - `axes/A-mixing/A8-positional-semantics/experiments` → `.../A8-positional-semantics`
  - `axes/W-width/W3-kv-head-sharing/experiments` → `.../W3-kv-head-sharing`
  - `axes/R-references/R7-existing-early-exit/experiments` → `.../R7-existing-early-exit`

### 轴 8 · 工程纪律台账（不写 `project.yaml`）

- `axes/8-engineering/README.md`：登记 精度 / 初始化 / 归一化 / 损失权重 / 数据构造 五条纪律
  （决定什么 + 已固定取值 + 出处 + 证据落点），按 55-review 口径更正为「工程纪律、非研究」。
- `axes/8-engineering/experiments/README.md`：来源表。
- 证据整搬：`axes/P-modifiers/P5-loss-weighting/experiments/2523b-shallow-weight-sweep`
  → `axes/8-engineering/experiments/2523b-shallow-weight-sweep`。

### 轴 9 · Backlog 研究项

- `axes/9-backlog/README.md`（轴级聚合，与其它轴一致）。
- `axes/9-backlog/9.1-selector-scheduling/project.yaml`（扩展 foundation exemplar）：
  全字段齐全；`purpose` 改为痛点（同预算下质量最高）；`objective_criteria` 写清该涨/该跌/必须不变；
  `falsification` 含「假设在 100M 失效」；`cost_model` 只用等效理论量（layer-passes / FLOPs /
  tokens-per-forward，含 pipelining），无墙钟；`role_map` 四角色齐；单一 `milestone`；
  `work_packages` WP1–WP4（含 **WP3 观测信号 / WP4 校准**，均为手段且 `review` 为独立复核）；
  `status: 未开始`。
- `axes/9-backlog/9.2-iteration-refinement/project.yaml`（新建）：同字段契约；补目的与可判定判据；
  WP1（现底座重建回传干预）/ WP2（换掉 §17.18 永久降级的分布外主腿）/ WP3（目的与归属裁决）/
  WP4（等参数同算力对照）；`status: 未开始`。
- 证据整搬：G1 的 2508/2514/2515/2517 + G3 的 2522 → `9.1/experiments/`；
  F2 的 2511 + F4 的 2619 → `9.2/experiments/`；各自补 `experiments/README.md` 来源表。

## 二、逐项验收自证

| 验收项 | 结果 | 证据 |
|---|---|---|
| 用词 = `docs/structure.md` exact wording（不改名/不造词） | ✅ | 标题逐字：`调度 / 选择器（原 G 轴）`、`迭代精化的其余（原 F 轴剩余）`、轴名 `9 Backlog（不删，移出主线）`；台账用词 `全注意力基线`/`RoPE`/`GQA/MQA`/`CALM/LayerSkip/MoD` 与 §12 一致 |
| ② 证据整搬、拆并一格一主家、`.scratch` 原地不动只更新指针 | ✅ | 全部 98 个 tracked 文件为 `git mv` 100% rename；`.scratch/**` 未出现在提交里；旧格只留 README |
| ④ 字段齐全（role_map / 单 milestone / work_packages 八字段 / cost_model 等） | ✅ | `check_tree.py` 全过（46 项）；`cost_model` 无墙钟/日期；WP `review` 均为独立复核 |
| ⑤ `status` 仅四值 | ✅ | 9.1 / 9.2 均 `未开始`；`check_tree` 校验枚举 |
| 不写 project.yaml（轴 7 / 轴 8） | ✅ | 两目录下无 `project.yaml` |
| 不删证据、旧格目录与旧 README 保留 | ✅ | `find axes -type f` 2504→2510（+6 文档，无丢失）；旧 `A2/A8/W3/R7/G*/F*/P5` README 仍在 |
| 不改 catalog.yaml / INDEX.md，工作树既有改动不回滚 | ✅ | 提交内不含 `catalog.yaml`/`INDEX.md`/`axes/C-vertical`；提交后它们仍为 ` M` |
| 不碰其他轴目录（除本次搬移直接造成的指针修复） | ✅（见未决项） | 仅改 `2-depth/2.1.1`、`6-predict/6.2`、`tests/test_r7_exit_policies.py` 的 R7 **路径字符串**，无内容/语义改动 |

## 三、命令与结果（照抄）

```
python3 tools/check_tree.py
→ OK：46 个 project.yaml 全部符合 docs/research-schema.md。        (exit 0)

python3 tools/summary.py --check
→ OK：46 个 project.yaml 全部符合 docs/research-schema.md。        (exit 0)

python3 -m pytest tests/ -q
→ 827 passed in 93.41s                                             (0 失败)

git diff --check HEAD~1 HEAD → clean                               (无行尾空白)

find axes -type f | wc -l → 2510   (本轮起始 2504；新增 6 份文档)
git ls-files axes | wc -l → 1733   (本轮起始 1727)
```

提交：`bacaaeb`，`110 files changed, 291 insertions(+), 59 deletions(-)`（98 rename + 6 create + 6 modify）。
显式路径 `git add`（未用 `-A/-u/.`），提交后 `git diff --cached --name-only | wc -l` = 0。

## 四、未决项 / 残留风险

1. **他轴证据里指向 W3 旧路径的「数据源」字段未改**（有意不改）：`axes/5-quant/5.1-kv-quant/experiments/{2619,2620,2621,2631,2632}/` 的
   `config.json` / `results.json` / `table.md` 里 `root` / `registry` / `source` 字段仍写
   `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/...`。它们是**生成时记录的证据/溯源字段**，
   改写会篡改证据内容，故留待 integrator 决定（补指针文件或统一改口径）。
2. **旧格 README 仍引用本格已搬走的 experiments 路径**（预期）：如 `C1` 引用旧 R7 路径、
   各旧格 README 的 `evidence` 仍指旧 `experiments/**`。旧格 README 由 integrator 统一退役，不在本 WP 范围。
3. **搬移造成的他轴活指针修复**：本次修改了 `axes/2-depth/2.1.1-early-exit/project.yaml`、
   `axes/2-depth/2.1.1-early-exit/experiments/README.md`、`axes/6-predict/6.2-speculative-decoding/project.yaml`
   的 R7 路径字符串（不改语义），以及 `tests/test_r7_exit_policies.py` 的 import 路径。若主编排认为
   越界，请回退该三处（但不修则 `check_tree.py` 必红）。
4. **`2.1.1` / `6.2` 的 R7 线索仍指旧文本区**：`2.1.1` 的 `experiments` 已指新家；
   `6.2` 的 `sota_baseline` 已指新家，其余文本无旧路径残留（已 grep 复核）。
5. **本 WP 未做独立复核**：按 `SCHEMA.md` §14/§14b，本项 PASS 须由新 fresh 复核者（非本 writer）给出。
