# CONTRIBUTING — 如何给这份 Playbook 添一格

欢迎 PR。这里的目标不是「多写文章」，而是**让下一个人少踩一个坑**，
并且他能核对你说的每一句话。

## 你可以做的五种贡献

| 类型 | 怎么做 | 需要注意 |
|---|---|---|
| **补一个空课题** | 在 `catalog.yaml` 对应的轴下把 `status: todo` 的格子填成 `judged` | 九节齐全；失败路径必须是列表 |
| **加一次实验** | 在该格子的 `experiments/` 下建目录（见 `SCHEMA.md` §3） | 必须含判据线、对照、方差界、混杂声明、复现命令 |
| **加一个新格子** | 在 `catalog.yaml` 相应轴下新增条目（含选项枚举与判据） | 新增格子必须来自**枚举规则**，不能是孤立灵感；随后 `build_wiki.py` 会生成页面 |
| **修正一个结论** | 直接改对应课题页/实验，并在 PR 说明里给出反证数据 | 只改结论不改证据 = 不合并 |
| **加一条参照线** | `axes/R-references/` 下新增 | 参照线必须附判据线（我们要打败它什么） |

## 硬性规则（CI 会拦）

1. `python3 tools/check_schema.py` 必须退出 0（缺页 / 缺字段 / judged 无证据 / 证据路径悬空都 fail）。
2. `python3 tools/build_wiki.py --check` 与 `python3 tools/backfill_legacy.py --check` 必须干净
   ——**不要手改 `INDEX.md` 或自动块**。
3. 新增实验目录后要重跑 `build_wiki.py`（格子页的实验清单是自动列出的）。
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
# 1) 改 catalog.yaml（新格子/新状态）或写实验目录
# 2) 生成/刷新索引与页面（顺序：先 backfill，再 build）
python3 tools/backfill_legacy.py    # 仅回填 archive/legacy-experiments.yaml 里的历史条目
python3 tools/build_wiki.py
# 3) 自检
python3 tools/check_schema.py && python3 -m pytest tests/ -q
# 4) 提 PR：说明「你替别人趟了什么坑」
```

## 语言

中文为主（本仓库面向中文读者），代码标识符/日志/配置键保持英文。
一个文档内不要中英段落交替。用户原始陈述（L1）逐字记录，不翻译。
