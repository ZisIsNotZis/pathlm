# 8 工程纪律（不是研究）· 台账

> 轴：**8 工程纪律（不是研究）**（目录 `axes/8-engineering/`）。用词、编号、目录以
> `docs/structure.md` §12 为准。
> 轴 8 是**工程纪律台账**（只写本 `README.md`）：这里记录「不研究、但每个实验都必须遵守」
> 的工程纪律——**不是研究项，不写 `project.yaml`**（`docs/structure.md` §12 轴 8 / §13.5）。
> 每条纪律登记「它决定什么 / 已固定的取值与理由 / 出处 / 证据落点」。

## 纪律登记表

| 工程纪律（exact wording） | 它决定什么 | 已固定的取值 / 理由 | 出处 / 证据落点 |
|---|---|---|---|
| 精度 | 哪些部分必须高精度——决定机制能否稳定训练 | reward / 判据类损失必须 **float32**，并配 **pre-step grad-norm gate**；bf16 autocast 下 negative / reward 项两次发散（capped −JS 多样性压力） | `docs/playbook.md` 定律 5（稳定性工程）；`.scratch/09-m3-interactions/evidence @39b1838` |
| 初始化 | 新参数插在哪、初始化成什么——决定「恒等起步」能否成立 | 恒等初始化 adapter 在 13M matched-step 族内最优（proj-fill 1.8356 < fill 1.8582 < nofill 1.9438）；恒等初始化**不消耗随机数** ⇒ 同 seed 下与 fill 臂逐位一致（可严格对照）；**恒等初始化 ≠ 零初始化**（乘性结构上全零是鞍点，两端梯度同时为 0 ⇒ 干预系统性失效而测试全绿，见 `SCHEMA.md` §10 干预活性判据） | `.scratch/23-depth-ar/evidence @6b27b5e`、`.scratch/23-depth-ar/issues @1586745`、`axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/zero-init-saddle/ @cba2f39` |
| 归一化 | LayerNorm 放在哪、是否有 final norm——决定残差流是否保持 embedding 几何 | **pre-LN + 无 final LayerNorm** 是「同空间」几何的前提（`logits = h·Eᵀ` 语义成立，readout scale 由 `U` 学）；已固定使用 | `.scratch/03-engine-m1/issues @e38349f`、`axes/1-time/1.1.4.1-explicit-memory/experiments/2618-carry-channel-13m/analysis.md @cc64112` |
| 损失权重 | 主损失与辅助 / 深度损失的相对权重与调度——决定浅层监督是否稀释主目标 | **浅层降权全败且单调反向**（w∈{1,0.1,0.02}：depth-L 2.45→2.57→2.73）⇒ 浅层全深度监督是**表示正则**而非稀释主因；**退火**（末 25% 线性 1.0→0.3、final 2×→1.0）使 C 配方成为族内最优（1.8429） | [`experiments/2523b-shallow-weight-sweep/`](experiments/2523b-shallow-weight-sweep/)（旧格 P5 整搬）；`.scratch/23-depth-ar/evidence @b8c9f86`、`.scratch/25-tax-attribution/evidence @2e86924` |
| 数据构造 | 任务与数据怎么构造——决定结论在「任务依赖」上是否稳健 | **任务构造是本项目最大的单点偏差来源**（enwik8 局部统计极强）；损坏增广支撑修复侧结论（I 族）；needle **v2** 构造可测（v1 信号饥饿会测到 0.0 假失败，见 `SCHEMA.md` §19） | `.scratch/05-m1-followup/evidence @480d50b`、`.scratch/07-x1-x3/evidence @7b47555` |

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md` §1 / §4：旧修饰符格 `P1-precision`、`P3-param-insertion-init`、
`P4-normalization`、`P5-loss-weighting`、`P6-data-construction` 归本台账（`docs/structure.md` §13.5）。
其中 P5 的自有实验 `2523b-shallow-weight-sweep` 整目录 `git mv` 到本目录
`experiments/2523b-shallow-weight-sweep/`（一格一份家，只搬不复制）；`.scratch/**/evidence` 原地不动，
只作锚点。旧格目录及其 README 仍保留在 `axes/P-modifiers/**`，待 integrator 统一退役。

## 怎么读

- 本轴无 `project.yaml`：工程纪律不参与 `status` 枚举，也不计入里程碑。
- 引用纪律条款时以本表 + `docs/playbook.md` / `docs/methodology.md` 为准（每类事实一个家）。
