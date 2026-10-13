# 2621 · 权重清单与代码指纹（KV 预算图剩余缺失组合）

> 契约：同目录 `README.md`。本实验是**测量 + 汇总**类，**不训练、不新增权重**；
> 被测量 checkpoint 全部**复用**（只读 + 运行时 sha256 断言）。权重本身不入库
> （`.gitignore` 覆盖 `*.pt`）。

## 1. 被测量 checkpoint（运行时断言 sha256，`sha256_verified = true`）

| 臂 | checkpoint | sha256 | 体积 (B) | 生成来源 |
|---|---|---|---|---|
| `fp16` / `int8` / `int4` / `win16` / `win32` / `win64` / `win128` / `win144` / `win160` / `win176` / `win192` / `win16_anchor16` / `win32_anchor32` / `win32_anchor8` / `win32_int8` / `win64_int8` / `win16_anchor16_int4` / `win32_anchor32_int4` | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | 54,193,675 | 2610 冻结 trunk（13.5M dense，12k 步，clean bpc 1.483569） |
| `mha` | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt` | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` | 54,193,675 | 2616 同配方 12k 步（d256/L12/H8/mlp6/seq256/seed0） |
| `gqa` / `gqa_int8` / `gqa_int4` / `gqa_win64` / `gqa_win128` | `.../runs/gqa-12k-s0/model.pt` | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` | 49,456,651 | 同上，`n_kv_heads=2` |
| `mqa` / `mqa_int8` / `mqa_int4` | `.../runs/mqa-12k-s0/model.pt` | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` | 48,667,147 | 同上，`n_kv_heads=1` |

- `quant_head` 臂（`gqa_int8` / `mqa_int8` / `gqa_int4` / `mqa_int4`）与 `combo_mask` 臂
  （`win32_int8` / `win64_int8` / `win16_anchor16_int4` / `win32_anchor32_int4` /
  `gqa_win64` / `gqa_win128`）**不新增权重**：它们是「（checkpoint，fp16/fp32 存储）
  × 评测期可见性掩码 × KV 量化往返」。`results.json::points[*].checkpoint` 记录了
  `base_arm` / `base_model` / `mask_window` / `mask_anchors` / `quantized_mode`。
- 全部 checkpoint 在 `run.py` 装载时按上表 sha256 **逐臂断言**，不符直接 `SystemExit`。

## 2. 代码锚（sha256，`results.json::code` 前 16 位）

| 文件 | sha256（本次执行时，前 16 位） | 角色 |
|---|---|---|
| `axes/.../2621-kv-budget-rest/run.py` | `7e9373e8f88339f3` | 本实验入口（复用 2619 管线 + 本批新臂） |
| `axes/.../2621-kv-budget-rest/rest_arms.py` | `973b80338e011f37` | 新组合臂 `combo_mask` 唯一语义层 |
| `axes/.../2620-kv-budget-gaps/gaps_arms.py` | `2787e99678bcfd62` | `quant_head` / `window_anchor` 语义（**只读不改**） |
| `axes/.../2619-kv-budget-chart/run.py` | `7920d2002e9eb34c` | 参考臂评估管线（`importlib` 载入，**只读不改**） |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99` | 合成长程任务电池 |
| `tools/evalkit/kv_quant.py` | `1630f4519fa76562` | KV 量化往返（`quantize_kv` 复用） |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e3` | 可见性掩码 |

## 3. 共享文件与并发写（SCHEMA §22）

- **本线不修改任何共享文件**：新逻辑全部放在**新模块** `rest_arms.py`；`gaps_arms.py`、
  `2619/run.py`、`tools/evalkit/*` 只被**读取 / `importlib` 载入**。故本线未持
  `/tmp/pathlm_code.lock`，也不涉及「回读邻线标记 / 禁止回滚他人改动」。
- GPU 面上本线与同批六线共享 `/tmp/pathlm_gpu.lock`，遵守 SCHEMA §20 的串行协议
  （取锁 → 写 `holder_pid` + `heartbeat` → 释放先清内容再 `rmdir`）。本实验整程
  **<10 min**，按 §20.1 第 1 条允许整程持锁。
- **口径重叠如实记录**：本目录之外的同批线（W3/W4 改 `pathlm/depth_ar.py`、K9/K10 知识线）
  在执行期间各有自己的工作树改动；本线只在自己的目录内写文件，与它们不相交。本线读取的
  `pathlm/depth_ar.py`（`window_anchor_keep_mask` / `GQABlock` / `kv_head_liveness`）在本次
  执行时的 sha256 = `16a27e96ac30a8958891d7fdc5c7de252c0becb949c9542897ee7faa416b4d48`
  （该文件已含 38-W3/W4 线**已提交**的改动——顶部注释见 `readout_perdepth` / `readout_wide_shared`，
  与本线无关；本线未改它）。该 sha256 不在 `results.json::code`（非 code_anchors），故单列在此。
  > 注意：本目录的引用读数基于上述 sha；若该文件后续被其他线再次改动，复核时须以
  > `results.json::instrument.combo_mask_reduction` 的三条退化还原判定为准（那三条与文件版本无关）。

## 4. 复现

```bash
python3 run.py --smoke   # 冒烟（缩档，管线自检）
python3 run.py           # 全跑（纯推理，≈2.4 min）
python3 reconcile_numbers.py
python3 -m pytest tests/test_kv_budget_rest.py -q
```
