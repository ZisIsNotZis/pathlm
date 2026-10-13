# MODEL — 2621 深度草稿端到端（W2）

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录本次用到的
checkpoint 清单、哈希、体积与复现命令。

## 1. 权重清单（本轮**不训练**，复用已入账的 depth-ar checkpoint）

| 用途 | 文件 | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|
| 主 checkpoint（全部臂共用） | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt` | 6,962,688 | 27,882,155 | `2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6` |

**为什么不必先训一个 13M**：契约允许「仓库内没有带 sha256 的 depth-ar checkpoint」
时先训一个。本仓**已有**带 sha256 记录的 depth-ar checkpoint——`23c-H2-w0-12k` 的哈希
写在已提交文件 `axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/MODEL.md`
里（该文件是本仓对它的登记），本轮 `run.py` 启动时**重新计算并校验**该 sha256
（`run_log.txt` 的 `[ckpt]` 行），与记录值逐位相同。故按「纯 eval」口径执行，预算
从 13 min 训练 + 25 min 评测降为 25 min 纯评测。

- 该 checkpoint 的口径：`d=256 / L=8 / h=8 / seq=256 / mlp=4`，12k 步、`exit_mode=uniform`、
  `fill_kv=True`（缺省）、`proj_fill=False`、seed 0。`tf@4`（enwik8 代理）历史记录值
  **0.9694**（`results.json` 复核为 0.9694–0.9707，见 `analysis.md` §2）。
- 选择它而不是 100M run（`.scratch/24-depth-ar-100m/evidence/100m-s0/model.pt`，`tf@4=0.9726`）
  的理由：① 100M checkpoint **没有入库的 sha256 记录**（契约要求「带 sha256」）；
  ② 本实验是**纯推理调度**口径（tok/s、接受率），形态结论在 13M 上已成立，规模轴
  按 SCHEMA §17.11「单点不外推」另行记录为下一步。

## 2. 生成命令（全部不训练）

```bash
cd /home/z/vibe/pathlm
# 冒烟（先测步时再定预算；≤2min/≤1GB，§20.1 的并发例外条件）
python3 axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/run.py \
    --ckpt .scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt \
    --out /tmp/w2_smoke --smoke --device cuda

# 全量（GPU 锁内；见 .scratch/38-auto-parallel 的锁台账）
python3 axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/run.py \
    --ckpt .scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt \
    --out axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e \
    --device cuda --repeats 3 --prompts 4

# 对账（报告里每个数字都从这里复核）
python3 axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/reconcile_numbers.py
```

## 3. 复现资产

- `results.json`：全部臂的逐 prompt 墙钟/接受率记录 + 代理口径 + 等效 FLOPs。
- `run_log.txt`：`[ckpt]` 哈希、`[gpu]`、逐臂一行汇总。
- `cycle_log.jsonl`：**每 cycle 一行**的活性量（`accepted` / `mid_accepted` /
  `emitted` / `gamma_effective` / `committed`）——SCHEMA §10「活性量周期性落盘」在纯
  推理实验里的对应物。
- `spec_decode.py`：spec 解码实现本体（新模块，**不改** `pathlm/`；ticket 38 期间
  `pathlm/depth_ar.py` 由 W4 线独占，见 SCHEMA §22）。
- `tests/test_depth_draft_e2e.py`：语义钉（10 条可失败单测）。

## 4. 无权重入库声明

`data/enwik8_full.npz`（200 MB）与上述 `model.pt` 均在 `.gitignore` 覆盖范围内，
未入库。本目录只登记哈希与命令。

## 5. 本轮产物文件指纹（最终 revision，供复核逐字节对照）

| 文件 | sha256 |
|---|---|
| `spec_decode.py` | `d34cd78eeba03d39883118dcac42d199d6713ee8c9e2bb440837146062d993eb` |
| `run.py` | `9cef74d007138decc84fd73e662f30ce6bc6bba68535b3aa88b2ff727ee653fb` |
| `reconcile_numbers.py` | `e5f82c880396ac6a1027d91825d90ffca528340d2302ee12f70ca0ede9f81a3a` |
| `results.json` | `32ff8db35f24a76f823aac985ffc8f2db9b2df54acdf2b0d110cd35c81d44ab6` |
| `cycle_log.jsonl` | `513ed328e6fb5e19ff9a5a764cdd4f4a68f0868e46acb9fe0f94ca9184850c4a` |

注：`results.json` 由那次锁内运行写出；`run.py` 在此之后**只**
改过日志路径（`log()` 改为 stdout-only，去掉重复写入），功能路径（臂定义/接受/提交/代理）
未动；对账脚本对 `results.json` 的 1027 条断言全部通过。
