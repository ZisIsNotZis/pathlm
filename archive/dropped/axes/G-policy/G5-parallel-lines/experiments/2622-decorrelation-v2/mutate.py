"""契约 2622 的变异验证（每个语义开关都必须「故意破坏 ⇒ 抓红」）。

对象 = 本线新增的语义接缝：
  M1 `pathlm.model.repr_decorr_penalty` 恒返回 0（表征去相关读数变死）
  M2 `train_m1.activity_line` 丢掉 `repr_decorr`（活性量从日志里消失）
  M3 `train_m1.liveness_record` 忽略 repr 直方图（v2 臂被误报成死开关）
  M4 `train_m1.gate_skip` 退回旧的 `>1e4` 跳过（gate 混杂修复被撤销）
  M5 `train_m1.validate_pcap` 变成空操作（单因素护栏失效）
  M6 `train_m1.sample_rounds` 忽略 `w_repr_decorr`（不采并行路径 ⇒ 罚则永不触发）

**为什么用运行期猴子补丁而不是改文件**（SCHEMA §22）：`train_m1.py` / `pathlm/model.py`
在本轮是共享面。本脚本**不改任何文件**，只在进程内 import 之后打补丁、跑指定单测
（期望**红**）、再还原并跑一遍（期望**绿**）。

用法：python3 mutate.py    # 输出 mutation_report.json；任一步不符合预期 ⇒ 非零退出
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402
os.chdir(ROOT)

import pytest                                                    # noqa: E402
import torch                                                     # noqa: E402
import train_m1 as TM                                            # noqa: E402
from pathlm import model as PM                                   # noqa: E402
from pathlm.config import sample_path                           # noqa: E402

TFILE = "tests/test_decorrelation_v2.py"

ORIGINALS = {
    "repr_decorr_penalty": PM.repr_decorr_penalty,
    "activity_line": TM.activity_line,
    "liveness_record": TM.liveness_record,
    "gate_skip": TM.gate_skip,
    "validate_pcap": TM.validate_pcap,
    "sample_rounds": TM.sample_rounds,
}


def _run(test_names) -> int:
    if not test_names:
        return mutation.pytest_main(["-q", TFILE])
    return mutation.pytest_main(["-q", "-x", *[f"{TFILE}::{t}" for t in test_names]])


# ------------------------------------------------------------------ 变异体
def mut_penalty_zero(h_prev, h, eps=1e-5):
    return torch.zeros((), dtype=torch.float32)


def mut_activity_drops_repr(step, loss, lr, minutes, gnorm, paths, aux, n_layers):
    line = ORIGINALS["activity_line"](step, loss, lr, minutes, gnorm, paths, aux, n_layers)
    line.pop("repr_decorr", None)
    return line


def mut_liveness_ignores_repr(pcap, div_hist, skip_hist, gnorm_hist, gate_skips, n_paths,
                              repr_hist=None):
    rec = ORIGINALS["liveness_record"](pcap, div_hist, skip_hist, gnorm_hist, gate_skips,
                                       n_paths, repr_hist=repr_hist)
    rec["alive"] = bool(div_hist)          # v2 臂（div_hist 为空）被误报成死开关
    return rec


def mut_gate_old_threshold(gnorm):
    return not bool(torch.isfinite(gnorm)) or float(gnorm) > 1e4


def mut_validate_noop(pcap):
    return None


def mut_sample_rounds_ignores_repr(pcap, rng, n_layers):
    base = sample_path(pcap, rng, n_layers)
    paths = [base] + [sample_path(pcap, rng, n_layers) for _ in range(base.n_retries)]
    if pcap.w_diversity > 0:          # 漏掉 w_repr_decorr
        paths.append(sample_path(pcap, rng, n_layers))
    return paths


MUTANT_TARGETS = {
    "M1_repr_penalty_always_zero": (PM, "repr_decorr_penalty"),
    "M2_activity_line_drops_repr_decorr": (TM, "activity_line"),
    "M3_liveness_ignores_repr_hist": (TM, "liveness_record"),
    "M4_gate_skip_old_threshold": (TM, "gate_skip"),
    "M5_validate_pcap_noop": (TM, "validate_pcap"),
    "M6_sample_rounds_ignores_repr": (TM, "sample_rounds"),
}

MUTANTS = [
    ("M1_repr_penalty_always_zero", mut_penalty_zero,
     ["test_repr_penalty_identical_is_one_and_independent_is_zero"]),
    ("M2_activity_line_drops_repr_decorr", mut_activity_drops_repr,
     ["test_activity_line_and_liveness_carry_repr_decorr"]),
    ("M3_liveness_ignores_repr_hist", mut_liveness_ignores_repr,
     ["test_activity_line_and_liveness_carry_repr_decorr"]),
    ("M4_gate_skip_old_threshold", mut_gate_old_threshold,
     ["test_gate_skip_only_on_non_finite"]),
    ("M5_validate_pcap_noop", mut_validate_noop,
     ["test_validate_pcap_rejects_two_decorrelation_terms"]),
    ("M6_sample_rounds_ignores_repr", mut_sample_rounds_ignores_repr,
     ["test_model_wires_repr_penalty_only_when_enabled"]),
]


def _test_modules() -> list:
    stem = os.path.splitext(os.path.basename(TFILE))[0]
    return [m for m in list(sys.modules.values())
            if getattr(m, "__name__", "").split(".")[-1] == stem]


def _set_test_globals(mapping: dict) -> None:
    for mod in _test_modules():
        for name, fn in mapping.items():
            if hasattr(mod, name):
                setattr(mod, name, fn)


def _apply(name: str, fn) -> None:
    mod_mut, attr = MUTANT_TARGETS[name]
    setattr(mod_mut, attr, fn)
    _set_test_globals({attr: fn})


def _restore_all() -> None:
    for name, (mod, attr) in MUTANT_TARGETS.items():
        setattr(mod, attr, ORIGINALS[attr])
    _set_test_globals(ORIGINALS)


def main() -> None:
    report = {"file": TFILE, "mutants": []}
    ok = True
    _run([])                               # 先 import 测试模块（之后才能改它的全局名）
    for name, fn, tests in MUTANTS:
        base = _run(tests)
        _apply(name, fn)
        try:
            red = _run(tests)
        finally:
            _restore_all()
        green = _run(tests)
        caught = (base == 0 and red != 0 and green == 0)
        ok = ok and caught
        report["mutants"].append({
            "name": name, "tests": tests,
            "target": f"{MUTANT_TARGETS[name][0].__name__}.{MUTANT_TARGETS[name][1]}",
            "baseline_exit": int(base), "mutated_exit": int(red), "restored_exit": int(green),
            "caught_red": bool(red != 0), "restored_green": bool(green == 0),
            "as_expected": bool(caught)})
        print(f"{name}: base={base} mutated={red} restored={green} ⇒ 抓红={red != 0}")
    report["all_caught_and_restored"] = ok
    report["note"] = ("猴子补丁在进程内完成，**未改动任何文件** ⇒ 逐字节还原是构造性的"
                      "（SCHEMA §22：共享文件不得被变异脚本改写）")
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("all_caught_and_restored =", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
