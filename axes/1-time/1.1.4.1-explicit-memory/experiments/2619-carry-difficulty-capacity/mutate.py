"""2619 · 变异验证（曲线容量开关 `carry_state_dim` 必须「故意破坏 ⇒ 抓红」）。

对象：本实验唯一的新语义开关 `carry_state_dim` 的三条实现接缝
（`CarryChannel` 的状态宽度 / `CarryChannelBlock` 的读出增益形状 /
`train_depth_ar.build_depthar_config` 的 CLI 接线 / 配置校验）。

**为什么用运行期猴子补丁而不是改文件**（SCHEMA §22）：`pathlm/depth_ar.py` 与
`train_depth_ar.py` 在本轮是**共享文件**（U2 线 F4 重建损失有未提交改动）。
§22 规定不得回滚/触碰他人改动。因此本脚本**不改任何文件**，而是在进程内 import 之后
对目标函数打补丁、跑指定单测（期望**红**）、再还原并跑一遍（期望**绿**）。
每个变异都记录其补丁函数名与目标单测。

用法：python3 mutate.py    # 输出 mutation_report.json；任一步不符合预期 ⇒ 非零退出
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402
os.chdir(ROOT)

import dataclasses          # noqa: E402
import pytest               # noqa: E402

import pathlm.depth_ar as DA                 # noqa: E402
import train_depth_ar as TA                   # noqa: E402

TFILE = "tests/test_depth_ar.py"


def _run(test_names):
    return mutation.pytest_main(["-q", "-x", *[f"{TFILE}::{t}" for t in test_names]])


# ----------------------------------------------------------------- 变异体
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
    """M3：CLI 接线被去掉（`--carry-state-dim` 变成死开关）。"""
    cfg = _orig_build_cfg(args)
    cfg.carry_state_dim = None
    return cfg


MUTATIONS = [
    ("M1", "CarryChannel 忽略 carry_state_dim（状态宽度不再随开关变）",
     DA.CarryChannel, "__init__", mut_carry_ignores_state_dim,
     ["test_carry_state_dim_decouples_state_from_attention_heads"]),
    ("M2", "CarryChannelBlock 忽略 carry_state_dim（读出增益形状不随开关变）",
     DA.CarryChannelBlock, "__init__", mut_block_ignores_state_dim,
     ["test_carry_state_dim_decouples_state_from_attention_heads"]),
    ("M3", "build_depthar_config 丢掉 carry_state_dim（CLI 死开关）",
     TA, "build_depthar_config", mut_build_cfg_drops_state_dim,
     ["test_carry_cli_plumbed_to_config"]),
]


def main():
    report = {"note": ("运行期猴子补丁式变异验证（SCHEMA §22：共享文件不得改动）；"
                       "每个变异：打补丁 → 目标单测期望红 → 还原 → 期望绿"),
              "suite_file": TFILE, "mutations": []}
    ok_all = True
    for mid, desc, target, attr, patch, tests in MUTATIONS:
        orig = getattr(target, attr)
        setattr(target, attr, patch)
        try:
            rc_red = _run(tests)
        finally:
            setattr(target, attr, orig)
        rc_green = _run(tests)
        caught = (rc_red != 0) and (rc_green == 0)
        ok_all &= caught
        report["mutations"].append({
            "id": mid, "desc": desc, "target": getattr(target, "__name__", str(target)),
            "attr": attr, "tests": tests,
            "exit_code_mutated": int(rc_red), "exit_code_restored": int(rc_green),
            "caught_red_then_green": bool(caught),
        })
        print(f"{mid}: mutated rc={rc_red} restored rc={rc_green} "
              f"=> {'CAUGHT' if caught else 'NOT CAUGHT'}", flush=True)
    report["all_caught"] = bool(ok_all)
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"all_caught={ok_all} -> mutation_report.json")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
