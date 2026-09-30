# PathLM Research Wiki — agent notes

本仓库是一份**LM 算力路径设计空间的完备目录 + 单实验 Playbook**（起点为已收口的
PathLM 项目，25 个执行 ticket）。人类读者请看 `README.md`，这里只写 agent 约定。

## 真相来源（不要互相矛盾）

- **`catalog.yaml`**：目录枚举 = 完备性的唯一来源（「应有哪些格子」）。
- **`axes/**/README.md`**：课题页（**自动生成**，由 `catalog.yaml` 渲染；手写内容
  只能放在 `<!-- MANUAL:begin/end -->` 之间）。
- **`SCHEMA.md`**：课题页与实验的格式契约。
- **`docs/`**：`design.md`（RAP 设计 + 共享底座）、`methodology.md`（天平）、
  `playbook.md`（已测定律）。改这些要过用户。
- **`archive/`**：旧项目只读；**旧数字有口径陷阱，引用前先读 `archive/README.md`**。
- **`.scratch/`**：历史工单与原始证据，只增不改；新研究不放这里。

## 工作流

```bash
python3 tools/build_wiki.py        # 改完 catalog.yaml 后刷新索引与课题页
python3 tools/check_schema.py      # 必过（缺页/缺字段/已判无证据/证据悬空都 fail）
python3 -m pytest tests/ -q        # 共享底座 129 绿
```

- 新研究 = 在 `axes/<轴>/<格子>/experiments/YYMM-<干预>-<规模>-<引擎>/` 建目录
  （契约 / 代码 / results.json / analysis.md / MODEL.md）。
- **只做单轴单格实验**；不做组合实验；`deps` 只记录耦合方向。
- 每个新开关：20 步冒烟 + 全电池 + 可失败单测 +（关键开关）变异验证。
- 数字必须带口径；Δ < 0.014 不可排序；n=1 负结果必须复现后入账。
- 权重（`*.pt`）与 `data/` **不入库**。

## 语言

- 对话、文档、工单、证据 README、commit message 用**中文**；
  代码标识符、日志字符串、配置键保持**英文**；一个文档内不中英段落交替。
- 用户原始陈述逐字记录（L1），不翻译。

## 工作区策略

`/home/z/vibe/AGENTS.md` 的 workspace policy 适用。
