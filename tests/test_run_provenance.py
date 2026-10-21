"""读数落盘时的代码哈希（`tools/run_provenance.py`）的红线，每条都构造出会失败的反例。

1. **落盘必须自带哈希**：`write_results()` 写出的读数里有 `code_hash_run_py`（实验 `run.py`
   的 sha256）与 `code_hashes`（写读数的脚本自身 + 实验目录的兄弟脚本），且每个值都等于
   **盘上那个文件**的 sha256（不是常量、不是别的文件）。
2. **文件改一个字节，哈希必须变**（负向）：同一份夹具改 `run.py` 一个字节再落盘，
   `code_hash_run_py` 不再相等 ⇒ 字段真的在指认代码版本，而不是写死的串。
3. **解析不出 `run.py` 就不写这个字段**（宁缺勿假）：产物目录上方没有「同时带 `README.md`
   与 `run.py`」的实验目录时，`code_hash_run_py` 缺席，`code_hashes` 仍记写读数的脚本自身。
4. **训练器写出的臂级读数不被预登记检查判假红**：实验绑了 `run.py` 时，
   `prereg_bind.reading_violations()` 对训练器写的读数不报 `hash-mismatch` /
   `unhashed` / `uncovered`（训练器自身的哈希只在 `code_hashes` 里，不参与「= run.py」的判定）。

夹具树造在临时目录里，真仓库一个文件都不动。
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import prereg_bind  # noqa: E402
import run_provenance  # noqa: E402

#: 训练器位置的替身：真正写读数的脚本（夹具里放实验目录之外，与 `train_depth_ar.py` 同形）。
SOURCE_PY = "print('train')\n"
#: 实验入口的替身。
RUN_PY = "print('arm')\n"
CONFIG = '{"arm": "K50"}\n'
ARMS_SH = "#!/usr/bin/env bash\n"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def lab(tmp_path):
    """一个实验目录（README.md + run.py + config.json + train_arms.sh）与它外面的写数脚本。"""
    exp = tmp_path / "axes" / "1-time" / "1.1-x" / "experiments" / "9001-arm"
    exp.mkdir(parents=True)
    (exp / "README.md").write_text("# 契约\n判据线写死在这里。\n", encoding="utf-8")
    (exp / "run.py").write_text(RUN_PY, encoding="utf-8")
    (exp / "config.json").write_text(CONFIG, encoding="utf-8")
    (exp / "train_arms.sh").write_text(ARMS_SH, encoding="utf-8")
    source = tmp_path / "driver.py"
    source.write_text(SOURCE_PY, encoding="utf-8")
    return exp, source


def _write(exp, source, run_name="arm0") -> tuple:
    """按训练器的落盘方式写一份臂级读数：out-root 是实验目录下的 `train/<run>/`。"""
    out_dir = exp / "train" / run_name
    path = out_dir / "results.json"
    run_provenance.write_results(path, {"bpc": 1.5}, source=source)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_written_reading_records_code_hashes_of_its_code(lab):
    exp, source = lab
    path, reading = _write(exp, source)

    assert reading["code_hash_run_py"] == _sha(exp / "run.py")
    # 写读数的脚本自身在这里，实验目录的兄弟脚本与配置也在（存在即记）。
    assert reading["code_hashes"] == {
        "driver.py": _sha(source),
        "run.py": _sha(exp / "run.py"),
        "config.json": _sha(exp / "config.json"),
        "train_arms.sh": _sha(exp / "train_arms.sh"),
    }
    # 落盘的就是返回值（调用方打印摘要不必再读一次文件）。
    assert path.exists()


def test_one_byte_change_breaks_the_hash(lab):
    exp, source = lab
    _, before = _write(exp, source, run_name="arm0")

    (exp / "run.py").write_text(RUN_PY + "# 一行注释就换了代码\n", encoding="utf-8")
    _, after = _write(exp, source, run_name="arm1")

    assert after["code_hash_run_py"] != before["code_hash_run_py"]
    assert after["code_hash_run_py"] == _sha(exp / "run.py")


def test_no_experiment_above_means_no_run_py_claim(lab):
    """没有实验目录（无 `README.md`）时，`run.py` 不是任何实验的入口 ⇒ 不写 `code_hash_run_py`。"""
    _, source = lab
    stray = source.parent / "scratch-out"
    (stray / "arm0").mkdir(parents=True)
    (stray / "run.py").write_text(RUN_PY, encoding="utf-8")   # 只有 run.py，没有契约

    path = stray / "arm0" / "results.json"
    run_provenance.write_results(path, {"bpc": 1.5}, source=source)
    reading = json.loads(path.read_text(encoding="utf-8"))

    assert "code_hash_run_py" not in reading
    assert reading["code_hashes"] == {"driver.py": _sha(source)}


def test_trainer_reading_is_not_flagged_by_prereg_bind(lab, monkeypatch):
    """实验绑了 `run.py` 时，训练器写的读数里 `code_hash_run_py` 只指那份 `run.py` ⇒ 不假红。"""
    exp, source = lab
    root = exp.parents[3]                     # 假仓库根：<tmp>/axes/1-time/1.1-x/experiments/9001-arm
    monkeypatch.setattr(prereg_bind, "ROOT", root)
    monkeypatch.setattr(prereg_bind, "AXES", root / "axes")

    path, _ = _write(exp, source)
    contract = {name: prereg_bind.sha256(exp / name)
                for name in ("README.md", "config.json", "run.py")}
    stats = {"unhashed": 0, "mismatch": 0, "uncovered": 0, "predate": 0,
             "predate_names": [], "ordered": 0}
    bound_at = datetime.now().astimezone() - timedelta(minutes=5)

    found = prereg_bind.reading_violations(
        path, exp, bound_at.isoformat(timespec="seconds"), bound_at,
        prereg_bind.bound_code_hashes(exp, contract), {}, stats)

    assert found == []
    assert stats["mismatch"] == 0 and stats["unhashed"] == 0 and stats["uncovered"] == 0
    assert stats["ordered"] == 1              # 读数晚于 bound_at（反向对照：不是「什么都过」）
