# RESEARCH · `5.5` fp8 / 大 head_dim

研究级**计划视图**：目标、目的、唯一里程碑、工作包（owner / 验收 / 预算 / 状态）与证据指针。
**事实与数字的家不在这里**——研究项事实（判据线、结论、`cost_rows`、`gaps`）在 `project.yaml`，
读数在各实验目录的 `results.json` / `analysis.md`。本文件只放指针，不复制读数。

## 0 这是什么

8 位量化有两条路：定点的 **int8**（需要标尺，标尺沿 `head_dim` 摊薄，每值多花 `2/G` 字节）与浮点的
**fp8**（动态范围大，可免标尺，每值 1 字节）。另一根杠杆是**头维度**：`G` 的上限 = `head_dim`，
头维度越大同一把标尺罩住的值越多、省幅越接近 50%。本项判的是：**在同一「每位置部署字节」轴上，
这两种 8 位存法各值不值，以及头维度把省幅往 50% 推的时候质量要不要付钱。**

## 1 目的（对着谁）

- **痛点**：长上下文推理的显存/带宽瓶颈在 KV cache，cache 随序列长度线性增长。
- **对标**：fp8 格式有正式定义（arXiv:2209.05433），KV/权重的 fp8 推理已工程化（vLLM / TensorRT-LLM，
  `记忆·未核`）；块标尺（沿 `head_dim` 分块）与 int8 量化在本仓 `5.1` 已落参照线。
  **本项不重造这些**：增量是「同一 13M 受控设定下，fp8 与 int8+块标尺放在**同一字节轴**上比质量」，
  以及「`head_dim` → G → 每位置字节」这条链的解析 × 实测对账（业界多在 ≥1B 尺度按位宽分别报）。
- **本项不做**：组合实验（量化 × 窗口 / 头共享 / 权重 / 激活，见 `harness/SCHEMA.md` §4 留白）；
  与实际显存峰值（评测实现立即反量化，字节账是解析价签）。

## 2 唯一里程碑

**输出**：一份「fp8 / 大 head_dim」判决包——`{fp16, int8 块标尺, fp8 E4M3, fp8 E5M2}` 在同一
「每位置字节」轴上的 clean bpc + 每位置部署字节 + 严格前缀不变性，以及 `head_dim` → G → 每位置字节的
解析 + 实测对照，含 n≥2。

**验收口径的家**：`project.yaml::milestone.acceptance`（原文在那里，本文件不转述判据数字）。
三条可观测要求：①四臂在同字节轴可比、二选一明确；②严格前缀不变性两条为 True；
③大 `head_dim` 省幅与解析一致；结论停在「该尺度该值不值」，不外推。

## 3 工作包

owner = `axis5-lead`（轴 5 量化轴 lead）；writer = `axis5-w5.5`（单一写入者）；reviewer =
`axis5-reviewer`（fresh 上下文，不得为 writer）；integrator = `tree-integrator`。角色定义见
`project.yaml::role_map`；逐包的 `what/goal/means/acceptance/evidence/review/status` 原文在
`project.yaml::work_packages`，本表只给计划口径的 owner / 验收 / 预算 / 状态。

| WP | 交付物 | 验收（可观测） | 预算 | 状态 |
|---|---|---|---|---|
| **WP1** fp8 vs int8+块标尺同字节判决 | `experiments/2653-fp8-vs-int8-13m/`（4 臂 @ `head_dim`=32） | 四臂同字节轴可比；fp8 越/不越带二选一明确；前缀不变性两条 True；非量化臂逐位不变 | ≈10 min（纯 CPU 评测） | 已完成（独立复核 PASS） |
| **WP2** 大 head_dim 下标尺摊薄判决 | `experiments/2655-fp8-hd64-head-dim/`（`head_dim`=64 实测腿 + 第二 seed 腿） | 实测每位置字节与解析一致（差 0）；质量不越带；对账差异有解释 | ≈18 min GPU 训练 + ≈8 min CPU 评测/腿 | 已完成（解析 + hd32 腿经复核 PASS；hd64 腿待独立复核） |
| **WP3** `head_dim`=128 腿（分诊表单一点动作） | `experiments/2657-fp8-hd128-head-dim/`（3 主臂 + 2 辅助臂 × 两 seed 腿） | Q1（`int8_block_g128` vs `fp16`）与 Q2（`fp8_e4m3 − int8_block_g128`）各自二选一明确；实测标尺密度 = `1/G` 与解析一致；前缀不变性两条 True；n=2 两腿同向 | ≈20 min GPU 训练（2 臂）+ ≈8 min CPU 评测/腿 | 已完成（两腿落盘；**独立复核待派**） |

## 4 证据指针

- 研究项事实与判据线：`project.yaml`（`milestone` / `work_packages` / `conclusion` / `cost_rows` / `gaps`）。
- WP1：`experiments/2653-fp8-vs-int8-13m/analysis.md` + `results.json`（4 臂、`head_dim`=32）。
- WP2：`experiments/2655-fp8-hd64-head-dim/analysis.md` + `results.json` + `hd64-s1/results.json`。
- WP3：`experiments/2657-fp8-hd128-head-dim/analysis.md` + `results.json` + `hd128-s1/results.json`；
  契约与定序证据在 `prereg_binding.json`（绑定早于读数）。
- 跨格汇总：代价事实见 `atlas/cost-table.md`（由 `project.yaml::cost_rows` 生成）；复现三态见
  `atlas/repro.md`（由 `tools/inventory_runs.py` 生成）；结论索引见 `atlas/FINDINGS.md`；
  本格在下一次分诊里的位置见 `atlas/triage.md`。

## 5 当前状态与残留

- 状态字的家是 `project.yaml::status`；本文件不重述。
- 残留缺口（原文在 `project.yaml::gaps`）：`head_dim` 只有 32 / 64 / 128 三个点（不外推 256）；
  `fp8 E5M2` 只在 `head_dim`=32 上测过；字节账是解析价签、非实测峰值；与 `5.3` 的 fp8 激活组合留白。
- WP3 已交付（判词在 `project.yaml::conclusion`）：两条 seed 腿同向、两腿都不越 0.014 带，
  字节账实测 = 解析。**WP3 与 WP2 的 hd64 实测腿的独立复核都仍待派**（fresh 上下文，writer 停止之后）。
- 本轮执行期间 GPU 锁被另一条研究线（`1.1.1` / `1.1.4.1` 的训练臂）长时间占着，seed 1 腿按
  `harness/SCHEMA.md` §20 **排队等锁 525 s** 才起训（不杀他人进程、不降级配方），实耗已记账。
