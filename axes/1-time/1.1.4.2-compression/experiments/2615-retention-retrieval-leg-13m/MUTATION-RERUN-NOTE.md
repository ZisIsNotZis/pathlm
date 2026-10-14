# 2615-retention-retrieval-leg · 变异自检重跑说明（锚点更新）

> 事由：`mutate.py` 的 **M2** 锚点 `old=` 与当前 `tools/evalkit/retention.py::needle_fate` 失配
> （该函数早前改过名：`visible = (keep is None …)` → 先取 `ks = keep if (keep is None or
> keep.dim()==2) else keep[row]`，再由 `visible = (ks is None …)` 判可见）。重跑会在
> `assert mut["old"] in baseline` 处失败。**本次失配不是 E2/E3 造成的**（E2/E3 未动
> `tools/evalkit/retention.py`），但按同一纪律一并修。

## 做了什么

1. **锚点更新**（只动锚点文本、不动变异语义）：M2 的 `old`/`new` 由 `keep` 改为 `ks`
   （仍是 `all` → `any`：半删的针被当成「可见」）。
2. **重跑自检**：`python3 mutate.py` ⇒ M1–M4 **全部抓红**（`returncode=1`、
   `verdict=FAILED（抓红）`）、还原后 `restored_sha256_equal=true`、还原后基线绿。
   结果 = 本目录 `mutation_report.json`（`sha256_before = 9ef983b7…`）。
3. **隔离运行**：变异改写的是共享文件 `tools/evalkit/retention.py`，共享卡上还有别的轴在起
   训练/评测子进程 ⇒ 在隔离副本 `/tmp/axis1mut/`（同一棵拷贝树）里跑；该副本的
   `tools/evalkit/retention.py` 与仓库版 **sha256 逐位相同**（`9ef983b7…`），运行前后
   **仓库版未被触碰**（同 sha）。

- 时间：2026-10-14T22:15:43+08:00（actor：axis1-lead-3）。过程记录见 `.scratch/62-axis1-time/TICKET.md` §10.17。
