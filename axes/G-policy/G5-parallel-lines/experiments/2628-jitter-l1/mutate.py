"""契约 2628（ticket 44-X1 后续，l1 臂抖动补全）的变异验证：每个被检验的落盘产物/语义接缝
都必须「故意破坏 ⇒ 抓红 ⇒ 逐字节还原 ⇒ 回绿」。

对象：
  M1 `configs13m/l1.json` 的 `train.steps` 改 24000→18000（不再是「同配置重跑」；
     该配置应与 2626 逐字节相同）
  M2 `config.json` 的 `jitter_rule.downgrade_if_range_ge` 改 0.0144→0.02（预登记阈值被动）
  M3 `config.json` 的 `effect_v2` 改 0.0144→0.010（效应基准被动）
  M4 `config.json` 的 `scales.13m.arms` 去掉 `l1_B`（本轮重跑臂被动）
  M5 运行期猴补丁 `jitter_l1.jitter_verdict` 恒返回 `maintained_robust`（三态判决读数变死）——
     跑 `tests/test_jitter_l1.py::test_jitter_verdict_three_states_and_boundaries` 证明仍可失败

M1–M4 用**文件写-读-还原**（还原后校验 sha256 逐字节相同）；M5 只在进程内打补丁，
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
sys.path.insert(0, HERE)
os.chdir(ROOT)

TFILE = "tests/test_jitter_l1.py"


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
    return subprocess.run([sys.executable, "-m", "pytest", *args],
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
def _run_jitter_mutant(report: dict) -> bool:
    import pytest

    import jitter_l1 as JM

    def run(node: str | None = None) -> int:
        args = ["-q"]
        if node:
            args += ["-x", f"{TFILE}::{node}"]
        else:
            args += [TFILE]
        return int(pytest.main(args))

    def test_modules() -> list:
        return [m for m in list(sys.modules.values())
                if getattr(m, "__name__", "").split(".")[-1] == "test_jitter_l1"]

    def set_globals(fn) -> None:
        for m in test_modules():
            if hasattr(m, "jitter_verdict"):
                m.jitter_verdict = fn

    orig = JM.jitter_verdict
    node = "test_jitter_verdict_three_states_and_boundaries"
    run()                                            # import 测试模块后才能改它的全局名
    base = run(node)

    def always_robust(range_value, **kw):
        return {"state": "maintained_robust", "range": range_value,
                "effect": JM.EFFECT_V2, "downgrade_ge": JM.DOWNGRADE_GE,
                "upgrade_lt": JM.UPGRADE_LT}

    JM.jitter_verdict = always_robust
    set_globals(always_robust)
    try:
        red = run(node)
    finally:
        JM.jitter_verdict = orig
        set_globals(orig)
    green = run(node)
    caught = (base == 0 and red != 0 and green == 0)
    report["mutants"].append({
        "name": "M5_jitter_verdict_always_robust", "target": "jitter_l1.jitter_verdict",
        "kind": "runtime_monkeypatch", "baseline_exit": int(base), "mutated_exit": int(red),
        "restored_exit": int(green), "caught_red": bool(red != 0),
        "as_expected": bool(caught), "files_modified": False})
    print(f"M5_jitter_verdict_always_robust: base={base} mutated={red} green={green} ⇒ {caught}")
    return caught


def main() -> None:
    report: dict = {"file": TFILE, "mutants": [],
                    "note": ("M1–M4 写-读-还原并校验 sha256 逐字节相同；M5 只做运行期猴补丁，"
                             "不改文件（SCHEMA §22）")}
    ok = True
    base = _run(TFILE)
    report["baseline_exit"] = int(base)
    ok = ok and (base == 0)

    def mut_m1(o):
        o["train"]["steps"] = 18000

    def mut_m2(o):
        o["jitter_rule"]["downgrade_if_range_ge"] = 0.02

    def mut_m3(o):
        o["effect_v2"] = 0.010

    def mut_m4(o):
        o["scales"]["13m"]["arms"] = ["l1_A", "repr01_A", "repr01_B"]

    ok = _mutate_file("configs13m/l1.json", mut_m1, "M1_l1_steps_changed", report) and ok
    ok = _mutate_file("config.json", mut_m2, "M2_downgrade_threshold_changed", report) and ok
    ok = _mutate_file("config.json", mut_m3, "M3_effect_v2_changed", report) and ok
    ok = _mutate_file("config.json", mut_m4, "M4_arm_list_changed", report) and ok
    ok = _run_jitter_mutant(report) and ok

    report["all_caught_and_restored"] = bool(ok)
    with open(os.path.join(HERE, "mutation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("all_caught_and_restored =", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
