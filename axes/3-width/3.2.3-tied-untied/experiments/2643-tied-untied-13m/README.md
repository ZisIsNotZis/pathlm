---
name: 2643-tied-untied-13m
ticket: 64-axis3-width
date: "2026-10-14"
grid: 3.2.3-tied-untied
engine: depth-ar（13M 档 · dense 形态）
budget: "gpu_wall_budget: ≤30 min（2 臂 × 12k 步 ≈ 8 min/臂 + 评测）；峰值显存 ≤5GB net；锁被占时排队（单次 ≤25 min）"
status: planned
---

# 2643 · 输入/输出嵌入绑定 vs 解绑（tied / untied / 部分绑定）· 13M · dense

本文件是**预登记契约**：§1–§10 在起跑前写定，起跑后**不改**；口径差异只在 `analysis.md` 里声明。

## 1. 命题（一句话，可证伪）

在**训练/数据/主干逐位相同**、唯一变量是**输入 embedding `E` 与输出 unembed 是否共享同一张表**的前提下，
把输出头解绑（独立 `U`）或部分解绑（`logits = h·(E + A·B)ᵀ`）相对绑定的 clean bpc 差异，
能否超过项目方差带（0.014 bpc）？

- **H1（备择）**：解绑/部分解绑在 13M 尺度上可排序更优（按 0.014 带判）；
- **H0（零假设）**：不可排序（|Δbpc| < 0.014）⇒ 在本尺度，绑定关系不是独立的质量维度，
  绑定的**参数节省**（少一份 `vocab×d`）即占优。

对准的痛点：解绑要多一份 `vocab×d` 参数；本仓 vocab=206 使绑定只省 **52,736 参数（总量的 0.76%）**，
因此本条在本仓**几乎测不到参数账**，测到的是**「读出自由度」这一结构本身**是否有独立质量增益。

## 2. 单因素声明

**唯一变量 = 输出读出头与输入 embedding 表是否共享**。

- 基底 = 主流标准 decoder-only transformer（`train_depth_ar.py --dense --topology sequential
  --exit-mode uniform`：无 carry 门、无退出采样、仅最终深度 CE；三条共享底座偏离见 `docs/design.md` §2）；
- attention / LN / FFN / 位置嵌入 / 残差接线 / 优化器 / 步数 / 批大小 / lr / seed / 数据切分 /
  评测批（`depth_curve`，`n_batches=20, bs=16, generator seed 2`）**逐字段相同**；
- 三个臂的**主干权重来自同一次随机抽样**：同一 `torch.manual_seed(0)` 下先构造 stock `DepthARModel`
  （= `T` 臂），`U`/`PR` 臂在**同一构造路径之后**才初始化各自多出的读出头参数 ⇒ 臂间只差读出头；
- **必要性例外：无**（读出头的改动是纯结构单因素）。

### 2.1 干预实现（为什么它不是「换了个模型」）

- 挂点 = `pathlm/depth_ar.py::DepthARModel.readout(h, depth=None)`（tied、fp32 头，
  实测表达式为 `hf @ self.embed.weight.float().T`）。
- 本实验在**实验目录内**定义子类 `ReadoutModel(DepthARModel)` 覆写 `readout`，**不改引擎**
  （`pathlm/**`、`train_depth_ar.py` 一字未动）；训练时把 `train_depth_ar` 模块里的全局名
  `DepthARModel` 换成工厂（与 `2641/run.py` 同款接线），其余训练/落盘全走共享 `main()`。
- `T` 臂的覆写表达式与 stock 逐字相同（`hf @ self.embed.weight.float().T`）⇒ **逐位等价**，
  由 §8 的逐位门（T1/T2）钉住。
- `U` 臂：`logits = h·Uᵀ`，`U` 形状 `[206, 256]`，`U~N(0, d^-0.5)`（与 embedding 同尺度）独立初始化。
- `PR` 臂：`logits = h·(E + A·B)ᵀ`，`A [256, 32]`、`B [32, 206]`；**init `B ~ N(0, 0.02)`、`A = 0`**
  （单边零，见 `SCHEMA.md` §10）：起步**逐位等于 tied**（`A·B = 0`），但 `A` 在 step 0 就有非零梯度
  （`dL/dA = hᵀ·g_logits·Bᵀ ≠ 0`），非鞍点；`dL/dB` 在 step 0 为零，`A` 一旦离开零点 `B` 即获得梯度。
  （把两侧都置零会落进鞍点、学习永久停住——§8 的 `T5` 单测用「双侧零 ⇒ grad_A == 0」钉住这一点。）

## 3. 臂定义

配置（逐字段沿用共同不变量）：`d=256, layers=8, heads=8, seq=256, mlp_mult=4, steps=12000, bs=32, lr=1e-3, seed=0`，
enwik8 字符级（vocab 206），`data/enwik8_full.npz`。

| 臂 | 输出读出 | 新增参数 | 总参数（解析） | 备注 |
|---|---|---|---|---|
| `T`（锚） | `logits = h·Eᵀ`（绑定） | 0 | 6,962,688 | **复用** `2641/runs/U4-s0-12k`（clean bpc **1.5314**），按 §9 三道门复用 |
| `U`（新增训练） | `logits = h·Uᵀ`，`U [206,256]` | +52,736 | 7,015,424 | 独立随机初始化（同 seed 流） |
| `PR`（新增训练） | `logits = h·(E + A·B)ᵀ`，`A[256,32]`、`B[32,206]` | +14,784 | 6,977,472 | 部分绑定；`B~N(0,0.02)`、`A=0` |

代价口径：attention / FFN 的每 token 等效 MACs 三臂逐位相同（6,814,032 + 读出头项）。
读出头项（在线实现）：`T`/`U` = `d·V` = 52,736；`PR` = `d·V + d·r + r·V` = 52,736 + 14,784 = **67,520**
（低秩项在线算出）。**离线合并**（`E + A·B` 预先算好）后 `PR` 的读出也是 `d·V`。
本条不把墙钟当判据或代价表述（`SCHEMA.md` §20 / `docs/methodology.md` §3）。

## 4. 判据线（预登记，不可事后调整）

基准唯一且显式：**基准 = `T` 臂（2641 `U4-s0`，clean bpc 1.5314）**；
**代价项相对谁算** = 相对同一配置的解析量（参数 / 每 token MACs）；
**主判据** = `P1` 的 clean bpc 配对 Δ；其余并列。

- **P1（质量）**：Δbpc(`U`−`T`)、Δbpc(`PR`−`T`)。带 = **0.014 bpc**（配对 Δ、≥5 窗口的口径，
  `docs/methodology.md` §2）。判定：`|Δ| ≤ 0.014` ⇒ **不可排序**；`Δ ≤ −0.014` ⇒ 解绑可排序更优；
  `Δ ≥ +0.014` ⇒ 解绑可排序更差。**seed 0 单点**，跨 seed 复现不在本轮预算内，结论须标注 n=1。
- **P2（语义代价 + 参数账）**：
  - 参数：总线单列（见 §3 表），强调 `T` 省 `V·d = 52,736`（读出头单列）；
  - **prob0 一致性**（解绑的语义代价）= 同一 `h` 下 `argmax(Eh) == argmax(Uh)` 的位置占比
    （在各自模型的最终深度状态上算，`n_batches=20, bs=16, seed 2`，序列位移位后逐位置）；
    `PR` 的 `U_eff = E + A·B`。**率类指标单独定门**：主判 = 按 batch 聚类的配对 bootstrap
    95% CI 与 Wilson 95% CI；`0.014` **不**跨量纲借用（`docs/methodology.md` §2）。
    `T` 臂该值为 1.0（构造性对照）。
- **P3（必须不变）**：
  - 因果性：单测断言改「未来 token」不改变更早位置的 logits（`U`/`PR` 臂）；
  - 评测确定性：`--no-train` 重评两次，`depth_curve` 逐深度 bpc **逐位一致**；
  - 活性：新增读出头参数幅度非 0（`PR` 的 `A` 必须已离开 0，否则机制是死的）。
- **P4（否证线）**：
  - 若 `|Δbpc| ≤ 0.014` ⇒ 判「本尺度绑定/解绑**质量中性**，`T` 省参数即占优」，不采纳解绑；
  - 若解绑可排序提升 bpc，但 (a) prob0 一致性显著偏离 1（语义代价）**或** (b) 读出参数
    `V×d` 随词表不可 scale ⇒ 判「**不采纳**：收益不抵语义代价/不可 scale」；
  - 若解绑可排序**更差** ⇒ 判「绑定占优」，并记为本尺度负结果。
- **带声明**：clean bpc 差用 0.014 bpc 带；prob0 一致性等率类指标走配对 bootstrap / Wilson 95% CI，
  不跨量纲借用。
- **大词表档不可判（显式）**：本仓 `vocab=206`（`V·d = 52,736` = 总参数 0.76%），
  解绑的真实参数账在大词表（`V ≈ 10⁵`，读出头占 10–20%）下才显著。**本条的一切结论只在
  `vocab=206` 的 13M 尺度内成立，不得外推到大词表参数账**；大词表档在本轮**不可判**。

## 5. 对照与方差

- 对照 = `T` 臂（2641 `U4-s0`），复用按 `SCHEMA.md` §9 三道门：
  1. **实现层 equality**：`T` 臂构造 + 前向与引擎 stock 逐位相同（单测 `T1`），
     且与 `2641/runs/U4-s0-12k/model.pt` 载入后逐张量逐位相同、键无缺无余（单测 `T2`）；
  2. **配置层 equality**：读 `2641/config.json` 逐字段列表对照（见 `analysis.md` §2 表），
     不等字段必须显式列出；
  3. **口径标注**：历史同配置对照 + 等值性证明 + 已知口径，写进 `analysis.md` 首段与 §2。
  任一字段不等 ⇒ 回退到自包含双跑（本轮预算不允许，故若发现不等即降级为「不可判」写入）。
- 方差界 = 0.014 bpc（`docs/methodology.md` §2）；同 config 跨 seed 散布（2641: 1.5257/1.5298）作参照，
  **不替代** 0.014。
- n=1（仅 seed 0）：所有「越带」结论须标 n=1 快照；只报增量方向，不称「已收敛后的最终质量」。

## 6. 成本与预算

- 硬件 = 单卡 RTX 4090（24GB）共享；GPU 工作按 `SCHEMA.md` §20 取/放 `/tmp/pathlm_gpu.lock`，
  **按臂释放**、臂间 ≥61 s，单次等锁 ≤25 min，**不杀他人进程**。
- 每臂 12k 步 ≈ 7–8 min（2641 实测 `wall_minutes ≈ 7.8`、`peak_gpu_mem_mb ≈ 935`）⇒ 2 臂 ≈ 16 min GPU。
- 冒烟：60 步（rc=0 才允许起 12k）；评测 ≈ 20 s/臂。
- 可中断性：臂级可中断（每臂独立 run 目录 + checkpoint）；中断不入账。
- 评测复用 2641 的 `T` 臂 ⇒ GPU 预算 = 2 臂训练 + 3 臂评测。

## 7. 数据与脚本

- 数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval，vocab 206），经
  `pathlm.data.load_enwik8_full(".tmp/enwik8", ...)`；不重生成、不入库。
- 入口：`run.py --arm {T,U,PR} --seed 0 --steps 12000 [--no-train]`；汇总 `run.py --aggregate`。
- 驱动：`train_arms.sh`（按臂取/放锁）。
- resolved config：`config.json`。
- 产出：`results_{T,U,PR}_s0.json`、`runs/{U,PR}-s0-12k/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}`、
  `results.json`、`analysis.md`、`MODEL.md`、日志 `train_*.log` / `run_log.txt`。

## 8. 单测与冒烟（含变异）

- 先 **60 步冒烟 rc=0**，再全跑（`docs/methodology.md` §5）。
- 单测 `test_2643_tied_untied.py`（自包含，不改共享 `tests/`）：
  - `T1` **逐位门**：`T` 臂前向最终深度 logits 与 stock `DepthARModel` **逐位相同**（解绑开关关闭）；
  - `T2` **构造/复用门**：同 seed 下 `T` 臂 `state_dict` 与 stock 构造逐位相同；载入
    `2641/runs/U4-s0-12k/model.pt` 后**键无缺无余**、逐张量逐位相同；
  - `T3` **变异（U 头真的在用）**：`U` 臂 prob0 一致性 < 1；把 `readout` 变体退化为「忽略 `U`、
    走 tied」后一致性**变红为 1.0**（证明 P2 探针能区分「解绑真生效」与「静默绑定」）；
  - `T4` **PR 恒等起步**：`A=0` 时 `PR` 前向最终深度 logits 与 `T` **逐位相同**；
  - `T5` **非鞍点（可失败）**：`A=0,B~N(0,0.02)` 时 `A.grad` 非零；把 `B` 也置零后 `A.grad == 0`
    （鞍点确实存在 ⇒ 门能变红）；
  - `T6` **参数量**：`U` 新增 52,736、`PR` 新增 14,784（逐位）；
  - `T7` **因果性**：改未来 token 不改变 `U`/`PR` 臂更早位置的 logits。

## 9. 失败决策树

- 冒烟 rc≠0 ⇒ 修实现（测量/接线），不跑 12k。
- 单测红 ⇒ 修实现；门恒真（变异不变红）⇒ 门无效，重写门。
- 锁等不到（>25 min × 重试）⇒ 按优先级 `U` → `PR`，未跑臂如实写「未跑」，**不编读数**。
- 训练中途失败 ⇒ 看 `train_*.log` 归类（实现/规模/显存），不静默重跑不同配置。

## 10. 产出与终点状态

- `results.json` 含 `P1` Δ 与判定、`P2` 参数/prob0（CI）、`P3` 确定性/活性、`P4` 否证线结论、
  `variance_bound=0.014`、`reuse` 三道门证据、`big_vocab_undecidable=true`。
- `analysis.md` 首段为**人话摘要**，数据表作附录。
- `MODEL.md` 权重哈希清单（权重不入库）。

## 11. 既有做法（站巨人肩膀）

- `docs/design.md` §2 三条共享底座偏离；`SCHEMA.md` §9 复用三道具、§10 活性、§15 等参数对照。
- 本仓既有的读出重参数化族（W4：gated / linear_resid / perdepth / wide_shared，见 `2641` 的
  `README.md` 与 `pathlm/depth_ar.py::readout` 文档）已经证明「读出重参数化 + 2d² 参数」在
  不可排序带内不产生独立增益（`SCHEMA.md` §15 实例：gated −0.0812 vs 等参数线性 −0.0888，
  差 0.0076 < 0.014）——本条是该族最底线的版本：**连多出的参数都不给**（`T`）。
- 业界（tied vs untied）结论普遍是「大词表下解绑略优、小词表下绑定省参数」；本仓
  vocab=206 ⇒ **待核实的基线**，本轮以本地单跑对齐，不引用未核实数字（`SCHEMA.md` §21）。
