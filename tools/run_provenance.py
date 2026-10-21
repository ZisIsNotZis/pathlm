"""run_provenance.py — 读数落盘时必须写的代码哈希（`code_hash_run_py` / `code_hashes`）。

「这份读数由哪一版代码产出」只能记在读数自己里面：跑完之后再去盘上算哈希，算的是**现在**的
文件，不是产出它的那份。写入点自己就是唯一知道这件事的地方，所以落盘由本模块统一做。

两个字段的分工（`tools/prereg_bind.py::reading_code_hashes()` 按这张表读）：

- `code_hash_run_py`：**真正产出这份读数的那份 `run.py`** 的 sha256。产物目录向上最近的、
  同时含 `run.py` 与 `README.md`（契约）的实验目录里的那一份；解析不出来就**不写**这个字段
  ——宁缺勿假：`.scratch/` 下直跑共享训练器时，声称某个 `run.py` 产出了它是一句谎。
- `code_hashes`：{仓库根相对路径 → sha256}，记**写这份读数的脚本自身**（如共享训练器
  `train_depth_ar.py`）与实验目录里的 `run.py` / `config.json` / `train_arms.sh`（存在即记）。

分成两个字段是因为绑定检查会把 `code_hash_run_py` 当作实验 `run.py` 的哈希去比对：训练器自身
的哈希只能落在 `code_hashes` 里，否则每一份臂级读数都会在「该实验绑了 `run.py`」时报假红。

用法（写读数的地方都过它，纪律见 `harness/methodology.md` §8）：

    write_results(os.path.join(run_dir, "results.json"), results, source=__file__)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: 仓库根：`code_hashes` 的键按它取相对路径（仓外的临时夹具退回文件名）。
ROOT = Path(__file__).resolve().parent.parent
#: 实验目录里要一并记指纹的兄弟脚本 / 配置：存在即记。
CODE_SIBLINGS = ("run.py", "config.json", "train_arms.sh")
#: 认作「实验目录」的那一层必须同时有契约和入口脚本（只认它，`run.py` 放错地方不算）。
CONTRACT = "README.md"
ENTRY = "run.py"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_key(path: str | Path) -> str:
    """`code_hashes` 的键：仓库根相对路径；仓外（临时夹具）退回文件名。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def experiment_dir(out_dir: str | Path) -> Path | None:
    """产物目录所属的实验目录：向上最近的、同时有 `run.py` 与 `README.md` 的那一层。

    `.scratch/` 里随手放的 `run.py` 不是任何实验的入口（不带契约），不认。
    """
    cur = Path(out_dir).resolve()
    for d in (cur, *cur.parents):
        if (d / CONTRACT).is_file() and (d / ENTRY).is_file():
            return d
    return None


def code_provenance(source: str | Path, out_dir: str | Path) -> dict:
    """落盘时要写进读数的两个字段；解析不出 `run.py` 时不写 `code_hash_run_py`。

    `source` 是写这份读数的脚本自身，`out_dir` 是产物目录。
    """
    source = Path(source).resolve()
    hashes = {code_key(source): sha256_file(source)}
    exp = experiment_dir(out_dir)
    if exp is not None:
        for name in CODE_SIBLINGS:
            p = exp / name
            if p.is_file():
                hashes[code_key(p)] = sha256_file(p)
    out: dict = {"code_hashes": hashes}
    run_py = hashes.get(code_key(exp / ENTRY)) if exp is not None else None
    if run_py is not None:
        out["code_hash_run_py"] = run_py
    return out


def write_results(path: str | Path, payload: dict, source: str | Path,
                  out_dir: str | Path | None = None) -> dict:
    """把代码哈希合进 `payload`，按整数缩进 2、非 ASCII 原样地落盘（与既有读数同格式）。

    返回落盘的内容（调用方要打印读数的摘要时不必再读一次文件）。
    哈希取自磁盘上**此刻**的字节：代码在运行中被改动，记下来的就是改动后的版本，
    这正是「这份读数由哪一版代码产出」要能被核对的地方。
    """
    path = Path(path)
    record = {**payload,
              **code_provenance(source, out_dir if out_dir is not None else path.parent)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record
