"""契约 V3 的变异验证（每个语义开关都必须「故意破坏 ⇒ 抓红」）。

对象 = 本线新增的语义接缝：
  M1 `train_m1.activity_line` 丢掉 `diversity_js`（活性量从日志里消失）
  M2 `train_m1.liveness_record` 无条件报 `alive=True`（对照臂被误报成活干预）
  M3 `ensemble_probe.reading_B` 弄错预登记形式的第二个括号（代数对账失效）
  M4 `ensemble_probe.pairwise_js` 恒返回 0（路径相关度读数变死）
  M5 `ensemble_probe.path_ensemble_bpc` 不做 K 平均（集成价签作废）
  M6 `ensemble_probe.deterministic_paths` 不真的关掉路径随机（确定性对照作废）

**为什么用运行期猴子补丁而不是改文件**（SCHEMA §22）：`train_m1.py` 在本轮是共享面
（loop-era 引擎，邻线 V1/V2 可能同时有未提交改动）。本脚本**不改任何文件**，只在进程内
import 之后打补丁、跑指定单测（期望**红**）、再还原并跑一遍（期望**绿**）。

用法：python3 mutate.py    # 输出 mutation_report.json；任一步不符合预期 ⇒ 非零退出
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pytest                                                    # noqa: E402
import train_m1 as TM                                            # noqa: E402
from tools.evalkit import ensemble_probe as EP                   # noqa: E402

TFILE = "tests/test_decorrelation_at_scale.py"

_orig_activity_line = TM.activity_line
_orig_liveness = TM.liveness_record
_orig_reading_b = EP.reading_B
_orig_pairwise_js = EP.pairwise_js
_orig_ensemble = EP.path_ensemble_bpc
_orig_det_paths = EP.deterministic_paths


def _run(test_names) -> int:
    return pytest.main(["-q", "-x", *[f"{TFILE}::{t}" for t in test_names]])


# ------------------------------------------------------------------ 变异体
def mut_activity_drops_diversity(step, loss, lr, minutes, gnorm, paths, aux, n_layers):
    """M1：活性量从日志里消失（去相关压力变成不可核对的死开关）。"""
    line = _orig_activity_line(step, loss, lr, minutes, gnorm, paths, aux, n_layers)
    line.pop("diversity_js", None)
    return line


def mut_liveness_always_alive(pcap, div_hist, skip_hist, gnorm_hist, gate_skips, n_paths):
    """M2：对照臂被误报成活干预。"""
    rec = _orig_liveness(pcap, div_hist, skip_hist, gnorm_hist, gate_skips, n_paths)
    rec["alive"] = True
    return rec


def mut_reading_b_wrong_brackets(l1, stock, div):
    """M3：预登记形式被写错（第二个括号里的符号反了）⇒ 代数对账失效。"""
    out = _orig_reading_b(l1, stock, div)
    out["verbatim"] = (l1 - div) - (l1 + stock)
    return out


def mut_pairwise_js_zero(probs, eps=1e-8):
    """M4：路径相关度读数恒为 0。"""
    out = _orig_pairwise_js(probs, eps)
    out["mean_nats"] = 0.0
    out["mean_bits"] = 0.0
    return out


def mut_ensemble_no_averaging(model, eval_arr, k_paths, **kw):
    """M5：不做 K 平均（把 k 条路径的和当集成）。"""
    out = _orig_ensemble(model, eval_arr, k_paths, **kw)
    out["bpc"] = round(out["bpc"] + k_paths * 0.01, 4)
    return out


class _NoOpDeterministic:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, *exc):
        return False


def mut_deterministic_paths_noop(model):
    """M6：确定性对照不真的关掉路径随机。"""
    return _NoOpDeterministic(model)


def _test_modules() -> list:
    """已加载的测试模块（pytest.main 在同进程反复调用时会复用 sys.modules 里的模块，
    因此只补丁源模块**不够**——测试模块的全局名字也要一起换，否则变异白做）。"""
    stem = os.path.splitext(os.path.basename(TFILE))[0]
    return [m for m in list(sys.modules.values())
            if getattr(m, "__name__", "").split(".")[-1] == stem]


def _set_test_globals(mapping: dict) -> None:
    for mod in _test_modules():
        for name, fn in mapping.items():
            if hasattr(mod, name):
                setattr(mod, name, fn)


def _apply(name: str, fn) -> None:
    """把**单个**变异同时打到源模块与已加载的测试模块上。"""
    mod_mut, attr = MUTANT_TARGETS[name]
    setattr(mod_mut, attr, fn)
    _set_test_globals({attr: fn})


def _restore_all() -> None:
    for name, (_, attr) in MUTANT_TARGETS.items():
        setattr(MUTANT_TARGETS[name][0], attr, ORIGINALS[attr])
    _set_test_globals(ORIGINALS)


# 每个变异体要换掉哪个模块的哪个名字（重命名/拼错即 KeyError，不会静默失效）
MUTANT_TARGETS = {
    "M1_activity_line_drops_diversity_js": (TM, "activity_line"),
    "M2_liveness_always_alive": (TM, "liveness_record"),
    "M3_reading_B_wrong_brackets": (EP, "reading_B"),
    "M4_pairwise_js_always_zero": (EP, "pairwise_js"),
    "M5_ensemble_no_averaging": (EP, "path_ensemble_bpc"),
    "M6_deterministic_paths_noop": (EP, "deterministic_paths"),
}

ORIGINALS = {
    "activity_line": _orig_activity_line,
    "liveness_record": _orig_liveness,
    "reading_B": _orig_reading_b,
    "pairwise_js": _orig_pairwise_js,
    "path_ensemble_bpc": _orig_ensemble,
    "deterministic_paths": _orig_det_paths,
}


MUTANTS = [
    ("M1_activity_line_drops_diversity_js", mut_activity_drops_diversity,
     ["test_activity_line_carries_diversity_and_skip_frac"]),
    ("M2_liveness_always_alive", mut_liveness_always_alive,
     ["test_liveness_record_marks_control_arm_as_control_not_dead_switch"]),
    ("M3_reading_B_wrong_brackets", mut_reading_b_wrong_brackets,
     ["test_reading_b_verbatim_form_equals_reduced_form"]),
    ("M4_pairwise_js_always_zero", mut_pairwise_js_zero,
     ["test_pairwise_js_zero_for_identical_positive_for_distinct"]),
    ("M5_ensemble_no_averaging", mut_ensemble_no_averaging,
     ["test_ensemble_bpc_equals_single_path_when_paths_are_deterministic"]),
    ("M6_deterministic_paths_noop", mut_deterministic_paths_noop,
     ["test_deterministic_paths_make_k_forwards_bitwise_identical"]),
]


def main() -> None:
    report = {"file": TFILE, "mutants": []}
    ok = True
    # 先把测试模块 import 进来（之后每次变异都会同时改它的全局名字）
    _run([])
    for name, fn, tests in MUTANTS:
        base = _run(tests)                 # 基线：目标测试必须绿
        _apply(name, fn)
        try:
            red = _run(tests)
        finally:
            _restore_all()
        green = _run(tests)
        caught = (base == 0 and red != 0 and green == 0)
        ok = ok and caught
        report["mutants"].append({
            "name": name, "tests": tests, "target": f"{MUTANT_TARGETS[name][0].__name__}."
            f"{MUTANT_TARGETS[name][1]}",
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
