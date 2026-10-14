# 2614 · 变异自检重跑说明（E2 后的锚点更新）

> 事由：E2（ticket 67，commit `4638882`）改了 `pathlm/depth_ar.py` 的 `TopKBlock(...)` 接线行，
> 本目录 `mutate.py` 的 `k-ignored` 变异锚点 `old=` 失配（重跑会 `unmatched`）。

## 做了什么

1. **锚点更新**（`mutate.py`，只动锚点文本、不动变异语义）：`old` 由 E2 前的
   `TopKBlock(mcfg, cfg.attn_sparse_k) … for _ in range(cfg.n_layers))` 改为 E2 后的
   `TopKBlock(mcfg, cfg.attn_sparse_k, _mh(i)) … for i in range(cfg.n_layers))`；
   `new` 同步改为 `TopKBlock(mcfg, mcfg.seq_len, _mh(i)) …`（语义仍是「忽略 attn_sparse_k ⇒ k 不生效」）。
2. **重跑自检**：`python3 mutate.py` ⇒ 五处变异**全部抓红**（rc=1 / `test_failure`）、
   还原后与基线**逐字节一致**、还原后基线单测**全绿**（6 passed）。结果 = 本目录
   `mutation_report.json`（`baseline_sha256 == final_sha256 == 45b7d406…`）。
3. **隔离运行（为什么不在仓库里跑）**：变异是**改写共享引擎文件**的破坏性操作，而共享卡上
   还有别的轴在起训练/评测子进程；被变异窗口覆盖到的 `import pathlm` 会静默污染它们的读数。
   故本次在**隔离副本** `/tmp/axis1mut/`（`pathlm/`/`tests/`/`tools/`/`train_depth_ar.py`/`configs/`
   为拷贝、`data/` 为符号链接）里运行；该副本的 `pathlm/depth_ar.py` 与仓库版 **sha256 逐位相同**
   （`45b7d406…`），运行前后仓库版 **未被触碰**（同 sha）。
4. **本目录其余四处变异锚点**（`ignore-topk`/`positional-select`/`noncausal-leak`/`unmasked-softmax`）
   经扫描**仍然匹配**当前引擎，未改。
   轴 1 全量 `mutate*.py`（32 个）按「模块声明的 `TARGET`」逐个扫过：失配者**只有**本目录这一处。

- 时间：2026-10-14T22:14:12+08:00（actor：axis1-lead-3）。过程记录见 `.scratch/62-axis1-time/TICKET.md` §10.17。
