# 代码哈希落盘 smoke（机制验证，不是研究项）

**人话**：这份目录不是研究项，是一次 4 秒的「落盘格式」验证——让共享训练器按新格式写一份
`results.json`，再核对它记下的哈希是否真的等于盘上那几个文件。核过了：`code_hash_run_py` 等于本目录
`run.py` 的 sha256，`code_hashes.train_depth_ar.py` 等于训练器自身的 sha256，另外两个兄弟文件
（`config.json` / `train_arms.sh`）也逐字相等。目的是回答一个问题：「这份读数由哪一版代码产出」
以后能不能核对——以前 57 份单跑产物答不上来，现在新写出来的产物答得上。

## 怎么跑出来的

```bash
# §20 GPU 互斥锁内，训练器直接把产物写到本目录下的 train/
python3 train_depth_ar.py hash-smoke --d 64 --layers 2 --heads 2 --seq-len 64 \
  --batch-size 2 --steps 2 --dense --out-root .scratch/run-hash-smoke/exp/train
```

训练 2 步、256 个 token（`--dense` ⇒ 不跑 enwik8 评测电池），4 秒结束，退出码 0。
本目录的 `run.py` 是「实验入口」的替身：训练器解析出这一层是实验目录（`README.md` + `run.py`），
于是把它的 sha256 记进 `code_hash_run_py`；训练器自己的哈希只进 `code_hashes`（见
`harness/methodology.md` §8）。

## 读数（`train/hash-smoke/results.json`，产出时间 2026-10-21T13:09:21+08:00）

| 字段 | 记下的值 | 盘上文件 | 相等 |
|---|---|---|---|
| `code_hash_run_py` | `7df228ab574389666890e4754ed374c5f8f01f717cd6b29239dfe6457931805a` | `.scratch/run-hash-smoke/exp/run.py` | ✓ |
| `code_hashes.train_depth_ar.py` | `dab9ebe637428019fcf4f04ba114f0cb8ef9469fa976306672234219e186b15e` | `train_depth_ar.py` | ✓ |
| `code_hashes[.scratch/run-hash-smoke/exp/run.py]` | `7df228ab…805a` | `.scratch/run-hash-smoke/exp/run.py` | ✓ |
| `code_hashes[.scratch/run-hash-smoke/exp/config.json]` | `0ebe82f5dfa520efeb21e0ba885b93085dc3812adcee4bc9a327135b24df26fe` | `config.json` | ✓ |
| `code_hashes[.scratch/run-hash-smoke/exp/train_arms.sh]` | `dcb58a2f9e018c6047e425ad57d1e82a74b107205f7a584cb57bdc9d8a8aede8` | `train_arms.sh` | ✓ |

## 预登记检查会不会把这份读数判假红

拿一份「绑住 `run.py` / `config.json`」的绑定去问 `tools/prereg_bind.py`：

```
适配结果（字段 → 指认的文件）：
  .scratch/run-hash-smoke/exp/config.json => ['code_hashes..scratch/run-hash-smoke/exp/config.json']
  .scratch/run-hash-smoke/exp/run.py => ['code_hash_run_py', 'code_hashes..scratch/run-hash-smoke/exp/run.py']
  .scratch/run-hash-smoke/exp/train_arms.sh => ['code_hashes..scratch/run-hash-smoke/exp/train_arms.sh']
  train_depth_ar.py => ['code_hashes.train_depth_ar.py']
violations = []
stats = {'unhashed': 0, 'mismatch': 0, 'uncovered': 0, 'predate': 0, 'ordered': 1}
```

四份文件里每一份被记的哈希都与绑定值相同 ⇒ 不报 `hash-mismatch`；`code_hash_run_py` 指的正是那份
`run.py`，训练器自己的哈希在 `code_hashes` 里、不参与「= run.py」的判定。这条规则由
`tests/test_run_provenance.py::test_trainer_reading_is_not_flagged_by_prereg_bind` 钉住。

## 入库范围

`results.json`（8 KB）与两份 jsonl 日志入库（它们是这次验证的证据）；`model.pt`（510 KB）是派生物，
按 `.gitignore` 的 `*.pt` 不入库。
