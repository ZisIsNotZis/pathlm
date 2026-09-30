# 01 — Research Wiki：把收口项目改造成「LM 算力路径设计空间」的完备目录 + Playbook

## 背景与目标

PathLM（回环时代 + 深度自适应 AR 时代，25 个 ticket）已收口，结论沉淀在
`docs/findings.md` / `docs/retrospective.md`。用户决策：把仓库改造为一份
**面向本地 NLU/架构玩家的 playbook**——一张 LM 算力路径设计空间的**完备目录**，
每个格子一个课题，课题内部记录**成功路径与失败路径**（含原因、原始数据、复现手段）。

用户原话要点（L1 记录）：

> 这个表格里面每一行对应一课题嘛。一个课题里面的内容也应该是完备的……不只是我们
> 这个成功路径，还有我们这个失败路径，就是为什么会失败……做成一个整个像个 ripple
> 一样的，像一个 wiki 一样的东西……我们的这个目录应该是完备的，至于里面目录里面的
> 内容，可能是大部分都是空的……这个里面有的东西应该是尽可能详细的……哪些我们做了，
> 哪些我们没做。这个东西哪些好，哪些效果不好，不好的原因是什么？数据是什么？这个
> 需要原始数据，或者说需要有这个复现手段，有脚本之类的。就是要尽可能的这个 rigorous，
> 这个严谨一些，尽可能这个科学一些……我是希望这玩意能成为未来所有的这种本地的这种
> NLU 玩家的这种 playbook……就是我们从来不考虑进行组合实验……只做单个实验……
> 如果我们未来有算力的话，也只可能是往更大的模型做，但还是做单个实验……只有目录
> 索引是在根目录的，然后包括整个这个 RAP 的这个设计在什么 docs 里面，然后剩下的每
> 一个的各种实验、各种文档都在这个各个各自的文件夹里面，各自文件夹首先是分这个大类，
> 然后是分小类嘛，这个小类里面就是各自的这个实验目录……至于以前的内容……属于
> deprecated 的，可以把它放到 scratch 里面之类的。

## 范围（本 ticket 做什么）

1. **契约层**：`SCHEMA.md`（课题页 front-matter + 九节模板 + 校验规则）、
   `CONTRIBUTING.md`、`docs/methodology.md`（天平）、`docs/playbook.md`（已测定律）、
   `docs/design.md`（RAP 设计 + 共享底座）。
2. **目录层**：`catalog.yaml`（网格枚举 = 完备性依据）+ `axes/` 七轴 + 参照线 +
   修饰符，每格一个课题目录（README.md）。
3. **工具层**：`tools/build_wiki.py`（由 catalog 生成/更新课题页、轴 README、根
   `INDEX.md`，自动区块用 marker 注入，人写区块保留）、`tools/check_schema.py`
   （CI 校验：缺页 / 缺字段 / judged 无证据）。
4. **回填层**：把 25 个 ticket 的判读（含失败路径）按格子填进课题页；旧文档整体
   移入 `archive/`（只读），并在每页用 `[路径, commit]` 锚定。
5. **骨架层**：`axes/*/…/experiments/` 目录 + `experiments` 命名规范。

## 非目标（明确不做）

- **不做组合实验**（scope 声明，写入 README 与 SCHEMA）：只做单轴单格实验，
  基底恒为 stock 对照；`deps` 字段只**记录**已知耦合方向，不构成组合实验授权。
- **不重写 git 历史**、不删旧内容（只 `git mv` 归档）。
- **不搬迁 Python 代码**（`pathlm/` / `tests/` / `train_*.py` / `probe_*.py` /
  `configs/` 原地不动）：本 ticket 只整理文档/目录树，代码重定位单列为后续项
  （避免在一次改动里同时做内容重构与代码重构）。理由与计划见
  `docs/design.md` §仓库布局。
- **不入库权重/大数据**：`*.pt`、`data/` 保持 gitignore；实验目录只放
  `MODEL.md`（哈希+体积+生成命令）。

## 判据（可失败）

| # | 判据 | 验收方式 |
|---|---|---|
| J1 | 目录完备：`catalog.yaml` 枚举的每一格都有实体目录与 README.md | `check_schema.py` 报 0 缺页 |
| J2 | 模板完备：每页九节齐全；`status=judged` 必须有 `judge` + `evidence` | `check_schema.py` 退出码 0 |
| J3 | 索引自动：`INDEX.md` / 轴 README 表格由脚本生成，重跑幂等（git diff 为空） | `build_wiki.py --check` |
| J4 | 旧内容可回溯：每页 `evidence` 指向 `archive/` 或 `.scratch/` 的真实路径 + commit | 抽查 ≥5 页路径存在 |
| J5 | 不破坏现有工程：129 单测保持全绿 | `pytest tests/ -q` |
| J6 | 事实一致：回填数字与 `archive/legacy-docs/findings.md` 逐条一致 | 抽查 ≥10 个数字 |

## 交付物

- `SCHEMA.md` / `CONTRIBUTING.md` / `INDEX.md` / `README.md`（重写为门面）
- `docs/{design,methodology,playbook}.md`
- `catalog.yaml` / `tools/{build_wiki,check_schema}.py`
- `axes/**/README.md`（轴页 + 课题页）
- `archive/README.md` + `archive/legacy-docs/*`
- 本 ticket 的 `evidence/SUMMARY.md`（判据逐条验收记录 + 数字抽查表）

## 决策记录（用户已确认）

- 2026-09-30 — 用户 — 批准执行：YAML front-matter + 脚本统一更新索引；
  「失败路径」必须是列表（试了什么、为什么失败）；先整理结构 + 回填已有研究，
  然后停下一起分析下一步跑什么；旧跨条目内容 deprecated → `archive/`（原话说
  scratch，落地为 `archive/` + `.scratch/README.md` 双向说明）。
- 2026-09-30 — 用户 — 「完备」的层级化解释（目录完备 / 条目完备 / 证据完备）
  与「不组合」scope 声明，用户确认按 agent 建议执行。

## 时间线

- 2026-09-30 — agent — 建 ticket、目录骨架；基线 `pytest` 129 绿；
  `.git` 6.6M（旧 1.6GB 问题已在历史重写中解决）；`.scratch` 20G 主要为
  未入库权重/日志。
