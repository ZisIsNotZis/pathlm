# 夹具：代码哈希落盘 smoke（不是研究项）

这份目录不是研究项，只用来验证一件事：共享训练器 `train_depth_ar.py` 落盘 `results.json` 时，
`code_hash_run_py` / `code_hashes` 两个字段真的被写出来、且哈希等于盘上的文件。
判据线不适用：本目录没有判词、不进研究目录（`docs/catalog.md`）。

- 入口：`run.py`（本夹具的「实验入口」，训练器会把它的 sha256 记进 `code_hash_run_py`）
- 落盘：`train/hash-smoke/results.json`（训练器产出）
