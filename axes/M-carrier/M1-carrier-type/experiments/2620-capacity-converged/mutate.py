"""2620 · 变异验证（本实验的判据实现必须「故意破坏 ⇒ 抓红 ⇒ 还原 ⇒ 变绿」）。

靶子分两类：

A. **既有共享开关** `carry_state_dim`（本实验唯一依赖的语义开关；用既有单测钉住）
   —— `tests/test_depth_ar.py::test_carry_state_dim_decouples_state_from_attention_heads`
   与 `::test_carry_cli_plumbed_to_config`。
B. **本目录新增的判据实现** `capacity_converged_lib.py`（门禁 / 主判据 / 毁源作废线）
   —— `test_capacity_converged.py` 里的对应单测。

**为什么用运行期猴子补丁而不是改文件**（SCHEMA §22）：`pathlm/depth_ar.py` 与
`train_depth_ar.py` 是跨线共享文件；本脚本**不改任何文件**，只在进程内 import 之后打补丁，
跑目标单测（期望**红**），再还原跑一遍（期望**绿**）。每个变异都记录补丁名与目标单测。

用法：python3 mutate.py    # 输出 mutation_report.json；任一步不符合预期 ⇒ 非零退出
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import dataclasses          # noqa: E402
import pytest               # noqa: E402

import pathlm.depth_ar as DA                 # noqa: E402
import train_depth_ar as TA                   # noqa: E402
import capacity_converged_lib as L            # noqa: E402

TFILE = "tests/test_depth_ar.py"
LFILE = os.path.join("axes", "M-carrier", "M1-carrier-type", "experiments",
                     "2620-capacity-converged", "test_capacity_converged.py")


def _run(f, test_names):
    return pytest.main(["-q", "-x", *[f"{f}::{t}" for t in test_names]])


# ----------------------------------------------------------------- A 类：共享开关
_orig_carry_init = DA.CarryChannel.__init__
_orig_block_init = DA.CarryChannelBlock.__init__
_orig_build_cfg = TA.build_depthar_config


def mut_carry_ignores_state_dim(self, cfg, rule, chunk=DA.DEFAULT_CARRY_CHUNK):
    """M1：`CarryChannel` 忽略 `carry_state_dim`（状态永远 dh=d/n_heads），
    但 `CarryChannelBlock` 仍按它定形读出增益 ⇒ 半接线。"""
    if getattr(cfg, "carry_state_dim", None):
        cfg = dataclasses.replace(cfg, carry_state_dim=None)
    return _orig_carry_init(self, cfg, rule, chunk)


def mut_block_ignores_state_dim(self, cfg, carry_state_dim=None):
    """M2：`CarryChannelBlock` 忽略 `carry_state_dim`（读出增益恒为注意力头数）。"""
    return _orig_block_init(self, cfg, None)


def mut_build_cfg_drops_state_dim(args):
    """M3：CLI 接线被去掉（`--carry-state-dim` 变成死开关）⇒ 两臂退化成同一模型。"""
    cfg = _orig_build_cfg(args)
    cfg.carry_state_dim = None
    return cfg


# ----------------------------------------------------------------- B 类：判据实现
_orig_linear_window = L.linear_window
_orig_convergence_gate = L.convergence_gate
_orig_capacity_verdict = L.capacity_verdict
_orig_corrupt_invalidates = L.corrupt_invalidates


def mut_linear_window_always_flat(lines, key, lo, hi):
    """M4：末窗斜率恒返回 0（伪造「已收敛」）。"""
    out = _orig_linear_window(lines, key, lo, hi)
    out["slope_per_1k"] = 0.0
    out["rel_drop"] = 0.0
    return out


def mut_gate_uses_total_loss(ansce_linear):
    """M5：门禁忽略传入指标、直接判「收敛」（等价于用总 CE 口径 ⇒ 恒真）。"""
    out = _orig_convergence_gate(ansce_linear)
    out["converged"] = True
    out["passes_slope"] = True
    out["passes_rel_drop"] = True
    return out


def mut_verdict_ignores_ci_overlap(acc_small, n_small, acc_big, n_big,
                                   ci_small, ci_big, threshold_pp=L.DELTA_PP_THRESHOLD):
    """M6：主判据忽略 CI 重叠（只看 Δ ≥ 3pp）。"""
    out = _orig_capacity_verdict(acc_small, n_small, acc_big, n_big,
                                 ci_small, ci_big, threshold_pp)
    out["capacity_axis_effective"] = bool(out["meets_delta"])
    return out


def mut_corrupt_line_is_zero(acc, ci):
    """M7：毁源作废线被去掉（永不判作废）。"""
    return False


MUTATIONS = [
    ("M1", "CarryChannel 忽略 carry_state_dim（状态宽度不再随开关变）",
     DA.CarryChannel, "__init__", mut_carry_ignores_state_dim,
     TFILE, ["test_carry_state_dim_decouples_state_from_attention_heads"]),
    ("M2", "CarryChannelBlock 忽略 carry_state_dim（读出增益形状不随开关变）",
     DA.CarryChannelBlock, "__init__", mut_block_ignores_state_dim,
     TFILE, ["test_carry_state_dim_decouples_state_from_attention_heads"]),
    ("M3", "build_depthar_config 丢掉 carry_state_dim（CLI 死开关）",
     TA, "build_depthar_config", mut_build_cfg_drops_state_dim,
     TFILE, ["test_carry_cli_plumbed_to_config"]),
    ("M4", "linear_window 末窗斜率恒为 0（伪造收敛）",
     L, "linear_window", mut_linear_window_always_flat,
     LFILE, ["test_convergence_gate_rejects_still_declining_ansce"]),
    ("M5", "convergence_gate 忽略答案行 CE、恒判收敛（= 总 CE 口径的恒真行为）",
     L, "convergence_gate", mut_gate_uses_total_loss,
     LFILE, ["test_convergence_gate_rejects_still_declining_ansce",
             "test_convergence_gate_insufficient_points_is_not_converged"]),
    ("M6", "capacity_verdict 忽略 CI 重叠（只看 Δ ≥ 3pp）",
     L, "capacity_verdict", mut_verdict_ignores_ci_overlap,
     LFILE, ["test_capacity_verdict_rejects_overlapping_ci_even_with_big_delta"]),
    ("M7", "corrupt_invalidates 恒 False（毁源作废线失效）",
     L, "corrupt_invalidates", mut_corrupt_line_is_zero,
     LFILE, ["test_corrupt_invalidate_line_is_4x_chance"]),
]


def main():
    report = {"note": ("运行期猴子补丁式变异验证（SCHEMA §22：共享文件不得改动）；"
                       "每个变异：打补丁 → 目标单测期望红 → 还原 → 期望绿；"
                       "**不改任何文件**，因此不存在「回滚他人改动」的风险"),
              "targets": {"shared_switch": TFILE, "local_criteria": LFILE},
              "mutations": []}
    ok_all = True
    for mid, desc, target, attr, patch, tfile, tests in MUTATIONS:
        orig = getattr(target, attr)
        setattr(target, attr, patch)
        try:
            rc_red = _run(tfile, tests)
        finally:
            setattr(target, attr, orig)
        rc_green = _run(tfile, tests)
        caught = (rc_red != 0) and (rc_green == 0)
        ok_all &= caught
        report["mutations"].append({
            "id": mid, "desc": desc, "target": getattr(target, "__name__", str(target)),
            "attr": attr, "test_file": tfile, "tests": tests,
            "exit_code_mutated": int(rc_red), "exit_code_restored": int(rc_green),
            "caught_red_then_green": bool(caught)})
        print(f"{mid}: mutated rc={rc_red} restored rc={rc_green} "
              f"=> {'CAUGHT' if caught else 'NOT CAUGHT'}", flush=True)
    report["all_caught"] = bool(ok_all)
    report["no_source_file_modified"] = True
    report["restore_method"] = ("运行期 setattr 还原（进程内），源文件从未被写 ⇒ "
                                "「逐字节还原」由「零文件写入」更强地保证")
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"all_caught={ok_all} -> mutation_report.json")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
