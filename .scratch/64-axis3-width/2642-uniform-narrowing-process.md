# 2642 · uniform 减宽 / 嵌套宽度 dropout · 13M（process 记录，非 SSOT）

契约 = `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/README.md`
（预登记、不可改）；事实读数 = 同目录 `results.json` / `results_<arm>_s<seed>.json`。
本文件只记过程与决定，**不得**与上述事实文档冲突；冲突以事实文档为准。

## 授权边界（本 WP）

- 只写 `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/**`
  与 `.scratch/64-axis3-width/**`；`pathlm/**`、`train_*.py`、`docs/**`、`tools/**`、
  其它轴与其它格的文件**零改动**。
- 干预实现在实验目录内：`prefix_forward_full`（训练期宽度前缀子网络前向）+
  `convert_state_dict` / `build_sliced_model`（checkpoint → 原生 d=w 切片模型）。

## 决定记录（就地工程决定，不改契约）

- **宽度采样**：三档 `{128,192,256}` **等概率**、每步独立（`torch.Generator(seed+7)`，
  不进 `state_dict`）。契约只写「每步对残差通道前缀取随机宽度 ρ∈{4/8,6/8,8/8}」，
  未指定分布 ⇒ 取最简且对称的等概率（在 README §2.2 写明）。
- **QKV 前缀语义**：`qkv` 是 `Linear(d, 3d)`，输出按 `chunk(3)` 分成 q|k|v 三块
  ⇒ 宽度前缀 = **每块各自取前 w 行 + 全体取前 w 列**（拼成 [3w, w]），
  **不是**输出行连续前缀 `[:3w]`（后者是掩码式错切，单测 T3e 钉住必须抓红）。
- **训练直方图口径**：训练每步恰好一次前向 ⇒ 前 `steps` 条采样就是训练步；
  `train_depth_ar.main()` 训练后的活性探针（如 `kv_head_liveness`）也会各走一次前向，
  这些**不计入**直方图（单列 `n_post_train_probe_forwards_ignored`）。
- **`gates` 参数**：dense 形态下 carry 门照建但**不参与前向**（引擎原生行为）
  ⇒ `params_total` 与 2641 同口径（含死门参数），另报 `params_live` / `gates_dead_in_dense`。
- **冒烟**：120 步、seed 99（run 目录与结果文件与正式 seed 0 分开，不入判据）。

## 过程流水

- 2026-10-14T12:52Z 起；读 2641 的 `run.py`/`README.md`/`config.json`/`train_arms.sh` 各一次，
  读 `pathlm/depth_ar.py` 的 `Block`/`DepthARModel` 前向与 `train_depth_ar.py` 的 `train()`。
- 契约 `README.md` + `run.py` + 单测 `test_2642_uniform_narrowing.py` 落盘；
  `pytest -q` 迭代修到 **23 passed**（含 bf16 autocast 下的 T1 与三档 T2 逐位相等）。
- 期间修正的实现缺陷（均由单测抓出）：① `gates.*` 的 state_dict 键层级错（应为模型级
  `gates.l.*`）；② QKV 前缀行数误用全宽 d（应为 w）；③ 变异用例必须**改变取值**
  （stock 里 ln1/ln2 权重初值都是 ones，拿另一个冒充是等值操作，抓不红）。
