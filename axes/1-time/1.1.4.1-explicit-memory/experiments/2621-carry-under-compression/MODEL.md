# MODEL · 2621 搬运通道 × KV 精度压缩

权重（`*.pt`）**不入库**（`.gitignore` 屏蔽）；本实验**复用** ticket 35 的检查点，
只读 + 运行时 **sha256 断言**。清单如下。

## 被测量检查点

| 臂 | 文件 | 架构 | 参数 | 体积 | SHA-256 |
|---|---|---|---|---|---|
| `U2`（delta 写通道 + 窗口 16 + MHA） | `../2618-carry-channel-13m/train/U2/model.pt` | stock dense + 单共享记忆槽（delta 残差写） | 13,801,064 | 55,255,619 B | `40ec6472fee5be3b8d3650ee74f541eb388f6d86e525942d44479e7279923420` |

- **来源**：`axes/1-time/1.1.4.1-explicit-memory/experiments/2618-carry-channel-13m`（ticket 35 契约 U），
  本实验**不重训**、不改权重、不加参数（`params_added = 0`）。
- `run.py` 每次加载都核对 sha256；不一致直接 `SystemExit`。
- `config.json::checkpoint` 与 `results.json::checkpoint` 都记录了路径、sha256、体积与
  `sha256_verified`。

## 本实验的语义只有一个维度：评测期 KV 存储精度

| mode | 处理 | 每值字节 | 每位置字节（MHA，12 层） |
|---|---|---|---|
| `fp32` | 释放版前向（`quantize_kv` 恒等、不动数值） | 4 | 24576 |
| `fp16` | K/V 存 fp16 再读回（**基线**） | 2 | 12288 |
| `int8` | 对称 per-channel 量化（**压缩档**） | 1 | **6144（−50%）** |

量化算子 = `tools/evalkit/kv_quant.py::quantize_kv`（复用，未改）。

## 生成 / 复现

```bash
# 被复用检查点由 ticket 35 生成（见 2618/MODEL.md）：
#   python3 train_depth_ar.py U2 ... --carry-write delta --train-window 16 \
#     --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14
# 本实验（纯推理，先取 /tmp/pathlm_gpu.lock，见 SCHEMA §20）：
cd axes/1-time/1.1.4.1-explicit-memory/experiments/2621-carry-under-compression
python3 run.py --smoke          # 冒烟（缩档，实测耗时/显存，不外推）
python3 run.py                  # 全电池（≈1 min，峰值 ≈432 MB）
python3 reconcile_numbers.py    # 三道对账（退出码 0）
python3 mutate.py               # 变异验证（4 处抓红 + 逐字节还原）
```

## 代码指纹（`results.json::code`，与本文档同一修订）

| 文件 | SHA-256 |
|---|---|
| `run.py` | `6c3dbfa70114174dbc71715ca341ab2d019e6a2d2ce7dae4050278ed2ba3e5b1` |
| `carry_quant.py` | `beb74dde6f63ee92e57f36430c0b84a0fb27805e2021230528cba26c033d9117` |
| `config.json` | `7430840ab1ec18694cac51d9c60835cf6b4a4c95d7f96c7d1dc0c86600b416fd` |
| `README.md` | `767a5d2970cfdf52ddaaec20edac16730cb90864d2326a5cbda3b5d9a01eb703` |
| `reconcile_numbers.py` | `c272a8787d27a13cc4ab8864e75b5e05657ae7ec92cd1dbf9de1d0613006ba53` |
| `mutate.py` | `0cd01527a754bf98563dc3ad0a31536af299dde43e19e70af5561a3a3a0c2712` |
| `tests/test_carry_quant.py` | `91c97a41cef200bd57dd7974e749f5da440c828199f6a6f8573cebfb0f362a9f` |
| `pathlm/depth_ar.py` | `16a27e96ac30a8958891d7fdc5c7de252c0becb949c9542897ee7faa416b4d48` |
| `tools/evalkit/kv_quant.py` | `1630f4519fa765627e9e52ceab14483aee34d90581031753d76a388564cb7da5` |
| `tools/evalkit/carry_channel.py` | `4dcd91facd60a512dcc983836fb29a7c91aad988bada1c95ada666824d226753` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` |

> `README.md` / `run.py` 的指纹：注意到 `README.md` 的 hash 是**本实验目录内**的文件
> （`run.py` 的 code-hash 解析优先取本目录，避免误取仓库根 README）。`reconcile_numbers.py`
> 的指纹在它落盘后由最后一次 `run.py` 采集，故与当前文件一致。

## 共享文件与并发（SCHEMA §22）

- 本实验**不修改任何共享文件**：`pathlm/`、`train_depth_ar.py`、`tools/evalkit/*` 全部只读。
  新逻辑全在 `carry_quant.py`（本目录），测试在 `tests/test_carry_quant.py`（新增文件）。
- 代码指纹已记录在上表；`pathlm/depth_ar.py` 的指纹是**本波 W4 线并发改动之后**的修订，
  口径重叠如实记录（本实验只读它）。
