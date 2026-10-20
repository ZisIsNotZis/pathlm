# 工级（harness）：格式契约、测量纪律、引擎能力面

这一层拥有的不是结论，而是**做结论时必须遵守的机械与纪律**：文档与实验的格式契约、判据与口径的
测量纪律、引擎真实支持的组合面。研究级（`axes/**`、`atlas/**`）与元级（仓库根、`docs/`）只**调用**它，
不在这里写任何一项研究的判词。

## 索引

| 文件 | 它拥有什么事实 |
|---|---|
| `README.md` | 本索引：工级有哪几份契约，各自约束什么 |
| `SCHEMA.md` | 实验与旧课题页的格式契约、分析纪律（编号条款，§6–§24 生效） |
| `research-schema.md` | 每个研究项 `project.yaml` 的字段、取值域与纪律（研究项 SSOT 的格式面） |
| `methodology.md` | 测量纪律（天平）：三列账本、方差界、双货币口径、冒烟与变异、预登记绑定 |
| `engine_capability.md` | 引擎能力面：哪些结构组合真跑得起来、哪些明确拒绝、拒绝在哪一行 |

机检入口（`tools/**`、`tests/**` 也属工级，它们把本层的规矩变成可失败的检查）：

```bash
python3 tools/summary.py --check           # 研究项字段 / 证据 / 口径
python3 tools/check_tree.py                # 同上，纯校验入口
python3 tools/check_docs.py --check        # 引用可解析 + 三层索引完整
python3 tools/build_cost_table.py --check  # atlas/cost-table.md 与 project.yaml 逐字节一致
python3 tools/inventory_runs.py --check     # atlas/repro.md 与磁盘一致
python3 -m pytest tests/ -q                # 共享底座测试
```

## 边界

- 不拥有**任何一项研究的结论**：判词在 `axes/**/project.yaml` 与各实验 `analysis.md`。
- 不拥有**跨格汇总**：定律正文、代价表、复现清单、排序打分在 `atlas/**`。
- 不拥有**里程碑与工作包**：`docs/plan.md`。
