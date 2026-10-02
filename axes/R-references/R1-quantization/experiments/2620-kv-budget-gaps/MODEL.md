# 2620 · 权重清单与代码指纹（KV 预算图缺失组合）

> 契约：同目录 `README.md`。本实验是**测量 + 汇总**类，**不训练、不新增权重**；
> 被测量 checkpoint 全部**复用**（只读 + 运行时 sha256 断言）。权重本身不入库
> （`.gitignore` 覆盖 `*.pt`）。

## 1. 被测量 checkpoint（运行时断言 sha256）

| 臂 | checkpoint | sha256 | 体积 (B) | 生成来源 |
|---|---|---|---|---|
| `fp16` / `int8` / `win16` / `win32` / `win64` / `win16_anchor16` / `win32_anchor32` / `win32_anchor8` | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | 54,193,675 | 2610 冻结 trunk（13.5M dense，12k 步，clean bpc 1.483569） |
| `mha` | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt` | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` | 54,193,675 | 2616 同配方 12k 步（d256/L12/H8/mlp6/seq256/seed0） |
| `gqa`, `gqa_int8` | `.../runs/gqa-12k-s0/model.pt` | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` | 49,456,651 | 同上，`n_kv_heads=2` |
| `mqa`, `mqa_int8` | `.../runs/mqa-12k-s0/model.pt` | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` | 48,667,147 | 同上，`n_kv_heads=1` |

- `quant_head` 臂（`gqa_int8` / `mqa_int8`）**不新增权重**：它们是「（头共享 checkpoint，fp16 存储）
  × 评测期 KV 量化往返」。`results.json::points[*].checkpoint` 记录了 `base_arm` 与
  `quantized_mode`。
- 全部 checkpoint 在 `run.py` 装载时按上表 sha256 **逐臂断言**，不符直接 `SystemExit`。

## 2. 代码锚（sha256，`results.json::code` 前 16 位）

| 文件 | sha256（本次执行时） | 角色 |
|---|---|---|
| `axes/.../2620-kv-budget-gaps/run.py` | 见 `results.json::code` | 本实验入口（复用 2619 管线 + 新臂） |
| `axes/.../2620-kv-budget-gaps/gaps_arms.py` | 见 `results.json::code` | 新组合臂唯一语义层 |
| `axes/.../2619-kv-budget-chart/run.py` | `7920d2002e9eb34ce352893ab6fcca5f1a7497ed5c2fc049278df40c9e7a2da7` | 参考臂评估管线（`importlib` 载入，**只读不改**） |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` | 合成长程任务电池 |
| `tools/evalkit/kv_quant.py` | `1630f4519fa765627e9e52ceab14483aee34d90581031753d76a388564cb7da5` | KV 量化往返（`quantize_kv` 复用） |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` | 可见性掩码 |
| `pathlm/depth_ar.py` | `248a92754154d65e0857ca5ceccda91ef63ffd9b90b3cfde2199da29b6b59135` | `window_anchor_keep_mask` / `GQABlock` / `kv_head_liveness` |

## 3. 共享文件与并发写（SCHEMA §22）

- **本线不修改任何共享文件**：新逻辑全部放在新模块 `gaps_arms.py`；`2619/run.py` 与
  `tools/evalkit/*` 只被**读取 / `importlib` 载入**。故本线未持 `/tmp/pathlm_code.lock`，
  也不涉及「回读邻线标记 / 禁止回滚他人改动」。
- GPU 面上我线与同批 V2（`2620-capacity-converged`）、V3 共享 `/tmp/pathlm_gpu.lock`，
  遵守 SCHEMA §20 的串行协议（取锁 → 写 holder_pid + heartbeat → 释放先清内容再 rmdir）。
- **口径重叠如实记录**：本目录之外的相邻 V2/V3 线在执行期间有各自未提交的工作树改动；
  本线只在自己的目录内写文件，与它们不相交。

## 4. 复现

```bash
python3 run.py --smoke   # 冒烟（缩档，管线自检）
python3 run.py           # 全跑（纯推理）
python3 reconcile_numbers.py
python3 -m pytest tests/test_kv_budget_gaps.py -q
```
