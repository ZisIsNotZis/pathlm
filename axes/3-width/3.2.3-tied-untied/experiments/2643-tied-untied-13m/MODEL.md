# 2643 · 权重与证据清单（权重本身不入库，`*.pt` 已 gitignore）

## 本轮产出的权重

**无。** `U`（完全解绑）/ `PR`（部分绑定）两臂未起跑（GPU 互斥锁等待超时 `rc=9`），
`runs/` 下没有 `U-s0-*` / `PR-s0-*` 目录，因此没有新权重。详见 `analysis.md` §0/§5
与 `results.json.status`。

## 复用锚（只读，不入库）

| 文件 | sha256 | 来源 |
|---|---|---|
| `runs/U4-s0-12k/model.pt`（2641） | `d4e6e33ef55d08f0cb71aadcc0c2c810d7c670cf6776c477c3fbcd90f966c1c2` | `axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/runs/U4-s0-12k/` |

`T` 臂 = 该 checkpoint 的**纯评测复用**（不重训）。载入审计见 `results_T_s0.json.checkpoint`：
`missing_keys = []`、`unexpected_keys = []`，与 2641 `MODEL.md` 记录的哈希逐位一致。

## 代码哈希（sha256）

| 文件 | 说明 |
|---|---|
| `run.py` | 见 `results_T_s0.json.code_hash_run_py`（本轮唯一被加载过的入口） |
| `test_2643_tied_untied.py` | 8 个自包含单测（`python3 -m pytest -q` ⇒ 8 passed） |
| `compare_anchor_config.py` | 复用门② 逐字段对照（可失败） |

## 数据

| 文件 | sha256 | 备注 |
|---|---|---|
| `data/enwik8_full.npz` | `73f92a31f005960a7dd50aa4b9309dcdc445bfe143ed6f9d513f61cc10f8b9c0`（与 2641 `MODEL.md` 一致） | 数据集不入库 |
