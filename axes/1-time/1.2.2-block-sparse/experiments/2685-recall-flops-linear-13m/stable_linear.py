"""2685 · 线性核的**数值无伤**档：不改核，只改求值方式与分母 guard 的生效阈值。

`pathlm.depth_ar.linear_attn_phi` 是 `F.elu(x) + 1`。对 `x` 很负时 `elu` 走 `exp(x) − 1`，在 bf16 里
`1 − exp(x)` 舍入回 `1` ⇒ φ **下溢为 0**（`phi_min = 0`），这是**求值方式**的产物：同一个函数
φ(x) = elu(x) + 1 = exp(min(x, 0)) + relu(x) 换成无消去形式后，φ(0 附近的负数) 仍是非零小量。

本模块在**进程内**把引擎的 φ 换成无消去求值，并把分母 guard 的 `LINEAR_ATTN_EPS` 从 `1e-6` 降到
`1e-30`（`out = num / max(den, eps)`：eps 越小越接近数学定义 `num/den`，eps 本身不是核的一部分）。
**不改核形式、不改参数量、不改配方、不改共享文件**——`pathlm/**` 全部只读调用，替换只发生在本进程的
模块属性上（与 `2683` 用子类替换模型类是同一手法）。

诚实边界：若训练中 q/k 的量级使 φ 落到 1e-30 以下（`x ≲ −70`），分母仍会下溢到 0 ⇒ 该行的注意力
输出被静默清零；这**不是**本模块能修的，它是「无界输入 + 无归一化」这个核本身的数值前提问题。
`run.py::linear_phi_audit` 把 φ 的零占比逐层量出来，健康门按它判（README §3）。
"""

from __future__ import annotations

import torch

import pathlm.depth_ar as da

#: 分母 guard 的声明值（落盘在训练臂 `results.json::linear_phi_numeric`）：guard 生效 ⇔ den < 该值。
LINEAR_ATTN_EPS_DECLARED = 1e-30

#: 「健康」的 φ 零占比上界：零占比 = 前向里 φ 恰为 0 的元素比例，标定见 README §3。
PHI_ZERO_FRACTION_MAX = 1e-3


def stable_phi(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = elu(x) + 1 的无消去求值（与 `linear_attn_phi` 逐值相同，只避开 `exp(x) − 1` 的消去）。"""
    return torch.where(x < 0, torch.exp(torch.minimum(x, torch.zeros_like(x))), x + 1.0)


def install() -> None:
    """把本进程的 φ 与分母 eps 换成无消去档（幂等；共享文件不动）。"""
    da.linear_attn_phi = stable_phi
    da.LINEAR_ATTN_EPS = LINEAR_ATTN_EPS_DECLARED
