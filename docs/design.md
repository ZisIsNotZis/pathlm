# 核心设计（RAP）与仓库布局

> 「RAP」是用户对本仓库核心设计的简称（沿用对话原文）。本文只写**设计契约**与
> **共享底座**；每一条轴上的实验研究在 `axes/` 下，方法学在 `docs/methodology.md`，
> 已验证定律在 `docs/playbook.md`。

## 1. RAP：同空间逐层精化 + 变深度提交

目标不是「更强的模型」，而是**同参数、同智商下更灵活的多路模型**——
把「Pro 当 Flash 用」变成结构事实：同一句内逐 token 变深度、不重 prefill、
出口草稿 + 全栈验证。

四条设计不变量：

1. **同空间（one shared geometry）**：残差流永不离开 embedding 空间；
   层是**同一空间内的精化算子**（idempotent 方向），不是特征变换；
   因此 skip = 恒等、无需 remap，深度可跳可退。
2. **提交不变量**：可变路径不改变正确性——**候选序列由全深度重打分定稿**；
   浅层的 staleness 不污染最终输出。
3. **全深度并行监督**：每个深度都有读出与监督 ⇒ 出口资产（浅层截断质量）
   只能由此产生。纯 dense trunk 没有浅层截断资产（实测 tf@4 0.12 vs 0.97）。
4. **主干不动，判断头可动**：挂在主干外的头（MTP / probe 式判断头）实测零税；
   **当前证据支持的倾向**是：「侵入主干本体」的结构付结构性税（见 `docs/playbook.md`
   定律 7）。**【2026-10-02 更正（C8）】**原文作「任何……都会带来结构性税」——那是**禁令式**表述，
   以小样本推向普遍；降级为**倾向**（同仓已有反证方向：读出改动有 −0.0812/−0.0888 快照改善）。

## 2. 共享底座（所有研究线继承，只有三处必须偏离 stock Transformer）

| # | 偏离点 | 为什么 stock 设计不够用 | 实现 |
|---|---|---|---|
| 1 | **tied 读出 + 无 final LayerNorm** | 同空间前提需要 `logits = h·Eᵀ` 的几何；prob0、软提交 Σp·E、latent 传输都建在它上面 | `pathlm/depth_ar.py::readout` |
| 2 | **变深度增量解码的填充语义** | stock KV cache 无法表达「某些位置只跑到第 k 层」；已退出位置的 KV 用冻结表征经本层投影填充 | `DepthARDecoder._layer_kv`（与并行 forward 逐位一致，单测钉死） |
| 3 | **全深度重打分的定稿不变量** | 变路径下必须有一个与路径无关的正确性判据 | `depth_ar.tf_acceptance` / `spec_cost_model` |

**引擎能力（ENG1，ticket 39-X2）**：`KVHeadProjectionMixin`（KV 头投影从 block 类解耦）
+ 窗口掩码作为**正交装饰层** + `ComposableBlock` 承载 `GQA × carry` ⇒
`GQA × carry × window` **可组合**（此前三者互斥、块类二选一，导致该组合在引擎里不存在）。
缺省（全关）与 stock `Block` 逐位相同。

**其余部件一律用 stock 设计**（这是 scope 声明的一部分：基底恒为 stock 对照）。

## 3. 评测套件（eval kit，唯一不解耦的东西）

| 组件 | 作用 | 位置 |
|---|---|---|
| clean bpc / 深度-bpc 曲线 | 税与精化单调性 | `pathlm/depth_ar.py::depth_curve` |
| tf@k 接受率（出口资产） | 草稿质量（TF 代理） | `depth_ar.tf_acceptance` |
| 等效验证吞吐（FLOPs 口径） | spec 收益的主口径 | `depth_ar.spec_cost_model` |
| workspace 探针 | null 能量 / 有效读出秩 / functional_dirs | `depth_ar.workspace_probe`、`readout_workspace_energy` |
| needle 电池 | 保留/检索（v2 构造） | `pathlm/eval.py` + `probe_*.py` |
| 三信号监测栈 | prob0 / 路径分歧 js / distinct-2 | `pathlm/gen_quality.py` |
| Slider 分配器 | 校准 → 双货币成本模型 → 求解器 → 验证解码 | `pathlm/slider.py` |
| 天平方法学 | 三列账本 / 方差界 / 双货币 / 冒烟+变异 | `docs/methodology.md` |

**新研究线必须先接上这套天平**，否则结果不可与历史对照。

## 4. 仓库布局

```
README.md            门面（本仓库是什么、怎么用、怎么提 PR）
INDEX.md             状态灯总表（自动生成）
SCHEMA.md            课题页与实验的格式契约
CONTRIBUTING.md      贡献流程与硬性规则
catalog.yaml         目录枚举 = 完备性依据（唯一的「应有哪些格子」来源）
docs/                设计 / 方法学 / 定律（不随课题变）
  design.md          本文
  methodology.md     天平
  playbook.md        已测定律
axes/                大类 = 轴；小类 = 格子（课题）；格子内 = 实验目录
  A-mixing/A3-window-anchor/README.md
  A-mixing/A3-window-anchor/experiments/2507-x2-eviction-13m/
  ...
tools/build_wiki.py  由 catalog 生成索引与课题页
tools/check_schema.py CI 校验（完备/模板/证据/无孤儿）
archive/             旧项目（只读）：legacy-docs/ 与说明
.scratch/            历史工单与原始证据（只读，见其 README）
pathlm/ tests/ configs/ train_*.py probe_*.py slider.py
                     ↑ 共享底座与历史引擎（原地保留，见 §5）
```

## 5. 代码布局：现状与迁移计划（本 ticket 未做）

本 ticket 只整理**文档/目录树**，不动 Python 代码。理由：代码搬迁会同时触碰
导入路径、测试与历史复现命令，与内容重构混在一次改动里违反「先拆混杂再判读」
的项目纪律。

**迁移计划（后续 ticket，可选）**：

1. 把 `pathlm/` 变成可安装包（`pyproject.toml` + `pip install -e .`），
   测试与探针改为包内导入；
2. 引擎/历史形态进 `archive/engines/`，**评测套件**抽成 `tools/evalkit/`
   （共享底座，见 §3）；
3. 每个格子实验目录自带 `run.py`，从 `tools/evalkit/` 取天平。

在此之前，任何格子实验都应该：在实验目录写 `run.py` → 调用仓库内现有脚本与
探针 → 把原始结果落进实验目录。

## 6. 阅读路径

- 想快速看**哪里是空白** → `INDEX.md`（状态灯）。
- 想读**一条轴的完整故事** → `axes/<轴>/README.md`（含格子表）。
- 想复现/接手**一格** → `axes/<轴>/<格子>/README.md` → 其中 §9 与 `experiments/`。
- 想知道**方法论** → `docs/methodology.md`；想知道**已验证的约束** → `docs/playbook.md`。
- 想知道**旧项目发生了什么** → `archive/README.md` → `archive/legacy-docs/retrospective.md`。
