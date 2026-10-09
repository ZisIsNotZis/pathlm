"""契约 2626（ticket 43 V2）的变异验证：每个被检验的落盘产物/语义接缝都必须
「故意破坏 ⇒ 抓红 ⇒ 逐字节还原 ⇒ 回绿」。

对象：
  M1 `configs13m/l1.json` 的 `p_skip` 改 0.1→0.05（阶梯声明键被动）
  M2 `configs13m/repr01.json` 的 `w_repr_decorr` 改 0.1→0.2（干预键被动）
  M3 `configs13m/repr01.json` 的 `train.seed` 改 0→1（seed 不再逐臂一致）
  M4 `configs13m/repr01.json` 的 `train.steps` 改 24000→18000（不再是 4× token）
  M5 `config.json` 的轨迹点改 [6000,12000,18000]→[6000,9000,18000]（评测配置被动）
  M6 运行期猴补丁 `pathlm.model.repr_decorr_penalty` 恒返回 0（语义开关读数变死）——
     跑 X1 的语义单测（`tests/test_decorrelation_v2.py`），证明该开关仍可失败

M1–M5 用**文件写-读-还原**（还原后校验 sha256 逐字节相同）；M6 只在进程内打补丁，
**不改任何文件**（SCHEMA §22：共享文件不得被变异脚本改写）。

用法：python3 mutate.py    # 输出 mutation_report.json；任一步不符合预期 ⇒ 非零退出
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402
os.chdir(ROOT)

TFILE = "tests/test_mechanism_convergence.py"
X1_TFILE = "tests/test_decorrelation_v2.py"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(tfile: str, node: str | None = None) -> int:
    args = ["-q"]
    if node:
        args.append("-x")
        args.append(f"{tfile}::{node}")
    else:
        args.append(tfile)
    return mutation.pytest_run([sys.executable, "-m", "pytest", *args],
                          capture_output=True, text=True).returncode


# ------------------------------------------------------------- 文件型变异
def _mutate_file(rel_path: str, mutation, label: str, report: dict) -> bool:
    path = os.path.join(HERE, rel_path)
    before_sha = sha256_file(path)
    original = open(path, encoding="utf-8").read()
    obj = json.loads(original)
    mutation(obj)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        red = _run(TFILE)
    finally:
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)
    after_sha = sha256_file(path)
    restored = (after_sha == before_sha)
    green = _run(TFILE)
    caught = (red != 0 and restored and green == 0)
    report["mutants"].append({
        "name": label, "target": rel_path, "kind": "file",
        "mutated_exit": int(red), "restored_byte_identical": bool(restored),
        "restored_exit": int(green), "caught_red": bool(red != 0),
        "as_expected": bool(caught)})
    print(f"{label}: mutated={red} restored_byte_identical={restored} green={green} ⇒ {caught}")
    return caught


# ------------------------------------------------------------- 语义接缝变异
def _run_engine_mutant(report: dict) -> bool:
    import pytest
    import torch

    import pathlm.model as PM

    def run(node: str | None = None) -> int:
        args = ["-q"]
        if node:
            args += ["-x", f"{X1_TFILE}::{node}"]
        else:
            args += [X1_TFILE]
        return int(pytest.main(args))

    def test_modules() -> list:
        return [m for m in list(sys.modules.values())
                if getattr(m, "__name__", "").split(".")[-1] == "test_decorrelation_v2"]

    def set_globals(fn) -> None:
        for m in test_modules():
            if hasattr(m, "repr_decorr_penalty"):
                m.repr_decorr_penalty = fn

    orig = PM.repr_decorr_penalty
    node = "test_repr_penalty_identical_is_one_and_independent_is_zero"
    run()                                            # import 测试模块后才能改它的全局名
    base = run(node)

    def zero(h_prev, h, eps=1e-5):
        return torch.zeros((), dtype=torch.float32)

    PM.repr_decorr_penalty = zero
    set_globals(zero)
    try:
        red = run(node)
    finally:
        PM.repr_decorr_penalty = orig
        set_globals(orig)
    green = run(node)
    caught = (base == 0 and red != 0 and green == 0)
    report["mutants"].append({
        "name": "M6_repr_penalty_always_zero", "target": "pathlm.model.repr_decorr_penalty",
        "kind": "runtime_monkeypatch", "baseline_exit": int(base), "mutated_exit": int(red),
        "restored_exit": int(green), "caught_red": bool(red != 0),
        "as_expected": bool(caught), "files_modified": False})
    print(f"M6_repr_penalty_always_zero: base={base} mutated={red} green={green} ⇒ {caught}")
    return caught


def main() -> None:
    report: dict = {"file": TFILE, "mutants": [],
                    "note": ("M1–M5 写-读-还原并校验 sha256 逐字节相同；M6 只做运行期猴补丁，"
                             "不改文件（SCHEMA §22）")}
    ok = True
    # 基线：未变异必须绿
    base = _run(TFILE)
    report["baseline_exit"] = int(base)
    ok = ok and (base == 0)

    def mut_m1(o):
        o["path"]["p_skip"] = 0.05

    def mut_m2(o):
        o["path"]["w_repr_decorr"] = 0.2

    def mut_m3(o):
        o["train"]["seed"] = 1

    def mut_m4(o):
        o["train"]["steps"] = 18000

    def mut_m5(o):
        o["scales"]["13m"]["traj_ckpts"] = [6000, 9000, 18000]

    ok = _mutate_file("configs13m/l1.json", mut_m1, "M1_l1_p_skip_changed", report) and ok
    ok = _mutate_file("configs13m/repr01.json", mut_m2, "M2_repr01_w_decorr_changed", report) and ok
    ok = _mutate_file("configs13m/repr01.json", mut_m3, "M3_repr01_seed_changed", report) and ok
    ok = _mutate_file("configs13m/repr01.json", mut_m4, "M4_repr01_steps_changed", report) and ok
    ok = _mutate_file("config.json", mut_m5, "M5_traj_ckpts_changed", report) and ok
    ok = _run_engine_mutant(report) and ok

    report["all_caught_and_restored"] = bool(ok)
    with open(os.path.join(HERE, "mutation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("all_caught_and_restored =", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
