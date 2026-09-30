# SCHEMA — 课题页与实验的格式契约

本仓库是一份**设计空间目录 + 单实验 Playbook**。它成立的前提是格式可机检：
`tools/build_wiki.py` 由 `catalog.yaml` 生成所有索引与课题页，
`tools/check_schema.py` 在 CI 里让格式错误直接失败。

## 0. 「完备」的三级定义

| 级别 | 含义 | 谁来保证 |
|---|---|---|
| **目录完备** | 每个「轴 × 格子」都在 `catalog.yaml` 里被枚举，并且有实体目录与 `README.md` | `catalog.yaml` + `check_schema.py` |
| **条目完备** | 每页九节齐全；未做的格子也必须给出**预测与判决实验设计**（不是留白） | 页模板 |
| **证据完备** | 已判格子必须有：判据、判读、原始数据/日志、复现命令、对照与方差界、混杂声明 | `check_schema.py` + 评审 |

**空课题是合法的，但必须是「有内容的空」**：`status: stub` 的页面里，
选项枚举、判据、判决实验设计、成本估计四件必须齐全。

## 1. 课题页 front-matter（机器可读，必须）

```yaml
---
id: A3-window-anchor           # 全局唯一，等于目录名
axis: A                        # 轴 id（A/C/H/W/F/M/G/R/P）
grid_dir: A-mixing/A3-window-anchor
title: 窗口 / 锚点 / sink
status: judged                 # stub | partial | judged | falsified | reference
judge: "KV 消融保留曲线（真删位置 j 的 KV 测损伤）"
cost: 13M 单臂 ~15–25 min
tax: "+0.121（原 X2 口径）/ +0.058（needle v2 重设计后）"
benefit: "ring-buffer 解码免重 prefill；锚点通道 needle 76.8%"
verdict: "丢前缀在 bpc 上很便宜，但检索当场归零；锚点通道是真产物。"
deps: [A4-soft-distance-penalty, M2-carrier-compression]
evidence:
  - "archive/legacy-docs/findings.md @7b47555"
  - ".scratch/07-x1-x3/evidence @7b47555"
---
```

- `status: judged | falsified` ⇒ `judge`、`verdict`、`evidence` 三者必填。
- `evidence` 的写法是 `"<仓库内相对路径> @<commit>"`；路径必须存在（CI 校验）。

## 2. 课题页九节（顺序与标题固定）

| 节 | 内容 | 空课题的最低要求 |
|---|---|---|
| 1. 命题 | 一句话、可证伪 | 必填 |
| 2. 选项枚举 | 本格全部可能取值（生成式列举，不是灵感） | 必填 |
| 3. 状态 | 本地做了什么 / 行业做了什么 | 必填 |
| 4. 结论与数字 | 税 / 收益 / 判读，**带口径**（快照·渐近·matched-step·墙钟·FLOPs） | 「本地零经验」 |
| 5. 失败路径与原因 | **列表**：试了什么 → 为什么失败 → 排除了哪些替代解释 | 「（无）」 |
| 6. 判据与判决实验设计 | 可证伪的判据线 + 干预定义 | 必填 |
| 7. 成本估计 | 13M/100M 分钟数 + 显存 | 必填 |
| 8. 依赖与耦合声明 | 只**记录**已知依赖与耦合方向，**不构成组合实验授权** | 「（无）」 |
| 9. 复现手段 | 脚本 / 命令 / 数据 / 权重哈希 | 「本地未做」 |

自动块夹在 `<!-- AUTO:begin/end -->`；手写块夹在 `<!-- MANUAL:begin/end -->`，
重跑 `build_wiki.py` 不丢手写内容。

## 3. 实验目录契约

```
<grid>/experiments/YYMM-<干预>-<规模>-<引擎>/
├── README.md        # 契约：命题 / 干预 / 判据线 / 对照 / 预算 / 混杂声明 / 结果
├── run.py           # 复现入口（或指向仓库内脚本 + 参数）
├── config.json      # resolved config（含 data/参数/seed）
├── results.json     # 原始结果
├── analysis.md      # 判读（逐格表格；失败也写）
├── controls/        # 对照臂与方差界
└── MODEL.md         # 权重清单：哈希 / 体积 / 生成命令（权重本身不入库）
```

- **权重与数据集不入库**（`.gitignore` 已覆盖 `*.pt` 与 `data/`）。
- 冒烟纪律：任何新开关先 20 步冒烟 + 全电池，再全跑；每个语义开关有可失败单测。
- 数字入账前必须过 `docs/methodology.md` 的方差界（Δ < 0.014 不可排序）。

## 4. scope 声明（写入本仓库的每个实验）

> 本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——
> 组合留白是显式声明，不是遗漏。`deps` 字段只记录已知耦合方向。
> 组合取舍属具体产品方 scope。

## 5. 校验命令

```bash
python3 tools/build_wiki.py --check     # 生成物是否与 catalog 一致
python3 tools/check_schema.py           # 格式/完备/证据锚点
```

## 6. 实验契约的必需字段（`experiments/<name>/README.md`）

除 `docs/methodology.md` 的一般纪律外，每个实验契约必须包含以下小节（缺一不可）：

1. **命题**：一句话、可证伪。
2. **单因素声明**：干预只改一个因素；基底是主流标准 decoder-only transformer
   （或 RAP 的三条共享底座）。**若单因素无法独立生效**：写明为什么、引入了哪个
   「主流 SOTA 的非标准因素」、它对结论的影响（必要性例外，须显式记账）。
3. **指标表**（多维，逐项分析）：它解决什么问题 / 代价 / 有效性（vs 同尺寸或同标的对照）/
   收益与用法（每种用法一行）/ 可控性 / 副产品。
4. **判据线**：预先写下，不许事后调整。
5. **对照与方差**：对照臂、seed 数、方差界（Δ < 0.014 不可排序）。
6. **成本**：显存预算（4090 共享卡）/ 预计墙钟（按 2–3× 折扣估）/ 可中断性 / 冒烟计划。
7. **数据与脚本清单**：数据集（名称/来源/规模/切分/生成命令）、入口脚本、
   resolved config、落盘文件。
8. **失败决策树**：无效时的处置路径（测量 / 实现 / 规模 / 设计，按此顺序排除）。
9. **产出**：`results.json` / 日志 / `analysis.md` / `MODEL.md`（权重哈希，不入库）。

## 7. 规模纪律（写进每个契约的 §6）

- 硬件 = 单卡 RTX 4090（24GB）且**被多个任务共享**：GPU 任务串行提交，批大小保守。
- 只做小规模实验验证**原理与 scaling 趋势**；外推靠**公式**，必要时多档次小实验拟合，
  **不做大实验**。
