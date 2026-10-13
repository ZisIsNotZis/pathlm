# CONTRIBUTING — 如何给这份 Playbook 添一格

欢迎 PR。这里的目标不是「多写文章」，而是**让下一个人少踩一个坑**，
并且他能核对你说的每一句话。

## 你可以做的五种贡献

| 类型 | 怎么做 | 需要注意 |
|---|---|---|
| **推进一个研究项** | 编辑 `axes/<轴>/<研究项>/project.yaml`（字段见 `docs/research-schema.md`） | 目的/创新/SOTA 基线/端到端验证/单一 milestone 齐全 |
| **加一次实验** | 在该研究项的 `experiments/` 下建目录（见 `SCHEMA.md` §3） | 必须含判据线、对照、方差界、混杂声明、复现命令 |
| **加一个研究项** | 按 `docs/structure.md` §12 的布局新增 `axes/<轴>/<id>-<slug>/project.yaml` | 必须来自树上的节点，不能是孤立灵感；一个研究项一个 milestone |
| **修正一个结论** | 改对应 `project.yaml` 的 `conclusion` 与实验，并给出反证数据 | 只改结论不改证据 = 不合并 |
| **加一条参照线** | `axes/7-references/README.md` 下新增 | 参照线必须附判据线（我们要打败它什么） |

## 硬性规则（CI 会拦）

1. `python3 tools/summary.py --check`（= `tools/check_tree.py`）必须退出 0
   （缺字段 / milestone 非单数 / role_map 缺角色 / 已完成无证据 / 证据路径悬空 / cost_model 含墙钟都 fail）。
2. 新研究项/新状态一律写进 `project.yaml`；旧的 `catalog.yaml` / `build_wiki.py` / `INDEX.md` **已退役**
   （见 `docs/structure.md` §15）——不要再改它们。
4. 新开关必须有可失败单测；实验前先 20 步冒烟。
5. 数字必须**带口径**（1.1-epoch 快照 / 4.4-epoch 渐近 / matched-step / 墙钟 / FLOPs）；
   单货币记账的结论会被要求双报。
6. `Δ < 0.014` 的差异**不可排序**（种子散布 ≤0.0138）。n=1 的负结果必须复现后入账。
7. 不做组合实验（见 `SCHEMA.md` §4）。

## 三种可复用的审计模板（失败路径写法）

1. **撤回型**：宣布一个排序结论无效——前提是「格间跨度 < 种子散布」。
   范例：`.scratch/11-retry-matrix-validity/VERDICTS.md`。
2. **拆混杂型**：把效应拆成「形态 / 引擎 / 配方」三列归属。
   范例：`.scratch/24-depth-ar-100m/`（24b）。
3. **干预-恢复型**：四格判读表（干预是否收回代价），并给出反证方向。
   范例：`.scratch/25-tax-attribution/`。

> 上述 `.scratch/**` 范例是**过程史（日志）指针**，只作写法范本；**证据锚点**要用**原始证据路径**
> （`.scratch/**/evidence/**` 的落盘证据或 `axes/**` 落盘产物）——见 `AGENTS.md` §2.1。

## 流程

```bash
# 1) 编辑 axes/<轴>/<研究项>/project.yaml（新研究项/新状态）或写实验目录
# 2) 自检
python3 tools/summary.py            # 看全表
python3 tools/summary.py --check && python3 -m pytest tests/ -q
# 3) 提 PR：说明「你替别人趟了什么坑」
```

## 语言

中文为主（本仓库面向中文读者），代码标识符/日志/配置键保持英文。
一个文档内不要中英段落交替。用户原始陈述（L1）逐字记录，不翻译。
