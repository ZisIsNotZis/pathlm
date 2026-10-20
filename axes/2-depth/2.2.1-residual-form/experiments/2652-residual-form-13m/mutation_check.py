"""2652 · 等值门的**可达性自证**（`harness/SCHEMA.md` §24 两条强制动作 ① / `proof.gate_probe`）：
「没被抓红过的门不是门」。本脚本在同一进程里注入三处**故意的**变异，断言等值门
`smoke.init_equality_gate` **必须**相应失败：

- **M1** `gated` 的门初值从 1.0 改成 0.99 ⇒ `gated@init` 与 `base` 不再逐位一致
  （参数量审计仍过 ⇒ 只有等值门能抓）；
- **M3** 把 `postln` 的块偷偷换成 `base` 公式 ⇒ 非无操作检查失败
  （「归一化后置」这个干预成了死干预，`harness/SCHEMA.md` §10）；
- **M4** 让 `apply_residual_form` 对所有形式都原样返回 stock 模型 ⇒ `gated` 的
  参数量增量从 16 变 0 ⇒ 参数量审计失败。

每处变异跑完立即在同一进程里恢复（重设 `residual_block` 的全局名），三处都必须
「红」。任一处不红 ⇒ 非零退出（说明对应的门是恒真的摆设）。

用法：`python3 mutation_check.py`
"""

from __future__ import annotations

import sys

import torch

import common  # noqa: F401  （先把仓库根插进 sys.path，再让 residual_block 能 import pathlm）
import residual_block as RB
import smoke


def _must_fail(tag: str, fn) -> str:
    try:
        fn()
    except smoke.GateFailure as e:
        print(f"[mutation] {tag}: 红 ✓（{str(e)[:70]}...）", flush=True)
        return "red"
    print(f"[mutation] {tag}: **绿（门没抓住变异）**", flush=True)
    return "green_not_caught"


def m1_gate_init_099():
    orig = RB.ResidualFormBlock

    class Bad(RB.ResidualFormBlock):
        def __init__(self, stock, form, index):
            super().__init__(stock, form, index)
            if form == "gated":
                with torch.no_grad():
                    self.attn_gate.fill_(0.99)

    RB.ResidualFormBlock = Bad
    try:
        smoke.init_equality_gate("cpu")
    finally:
        RB.ResidualFormBlock = orig


def m3_postln_becomes_base():
    orig = RB.ResidualFormBlock

    class Bad(RB.ResidualFormBlock):
        def __init__(self, stock, form, index):
            super().__init__(stock, form, index)
            if form == "postln":
                self.form = "base"

    RB.ResidualFormBlock = Bad
    try:
        smoke.init_equality_gate("cpu")
    finally:
        RB.ResidualFormBlock = orig


def m4_forms_not_applied():
    orig = RB.apply_residual_form
    RB.apply_residual_form = lambda model, form: model
    try:
        smoke.init_equality_gate("cpu")
    finally:
        RB.apply_residual_form = orig


def main():
    results = {
        "M1_gate_init_0.99": _must_fail("M1 gate init 0.99", m1_gate_init_099),
        "M3_postln_falls_back_to_base": _must_fail("M3 postln→base",
                                                   m3_postln_becomes_base),
        "M4_forms_not_applied": _must_fail("M4 forms not applied",
                                           m4_forms_not_applied),
    }
    # 恢复后必须重新变绿（证明变异是「注入—恢复」，不是把门永久打红）
    back = smoke.init_equality_gate("cpu")["pass"]
    print(f"[mutation] 恢复后等值门 pass={back}", flush=True)
    all_red = all(v == "red" for v in results.values())
    print(f"[mutation] {'PASS' if all_red and back else 'FAIL'}: {results}",
          flush=True)
    return 0 if (all_red and back) else 1


if __name__ == "__main__":
    sys.exit(main())
