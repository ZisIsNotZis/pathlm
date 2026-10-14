#!/usr/bin/env python3
"""2639 · 训练入口：**完整机制**（逐层独立状态 + 内容相关写入门 + 可选衰减 γ）。

为什么不直接用 `train_depth_ar.py`：E1（`.scratch/67-engine-changes/E1-CARRYCHANNEL.md`）
按 ticket「不新增公开开关（先内部实验）」只在 `DepthARConfig` 上加了
`carry_per_layer` / `carry_decay` 两个字段，**没加 CLI**。本实验**不改共享文件**
（边界纪律），故在本文件里把 `train_depth_ar` 命名空间中的 `DepthARConfig` 换成一个
「带 E1 缺省的子类」——`build_depthar_config()` 里的 `DepthARConfig(...)` 在调用时解析
模块全局，因此换掉即可让共享 runner 原样跑出逐层/衰减臂，而不动它一行代码。

用法（与 `train_depth_ar.py` 的 CLI 完全一致，额外两个环境变量）：
  CARRY_PER_LAYER=1 CARRY_DECAY=0.99 python3 train_pl.py pl16 --d 256 ... --carry-write delta

缺省 CARRY_PER_LAYER=1、CARRY_DECAY=1.0（= 完整机制的最小完备形：逐层 + 内容写门，无衰减）。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_depth_ar as T  # noqa: E402

CARRY_PER_LAYER = os.environ.get("CARRY_PER_LAYER", "1") not in ("0", "false", "False")
CARRY_DECAY = float(os.environ.get("CARRY_DECAY", "1.0"))


class _PLConfig(T.DepthARConfig):
    """E1 开关的**缺省注入**：共享 runner 不传这两个字段时按本实验的臂设置取值。"""

    def __init__(self, *a, **kw):
        kw.setdefault("carry_per_layer", CARRY_PER_LAYER)
        kw.setdefault("carry_decay", CARRY_DECAY)
        super().__init__(*a, **kw)


T.DepthARConfig = _PLConfig

if __name__ == "__main__":
    print(f"[2639] carry_per_layer={CARRY_PER_LAYER} carry_decay={CARRY_DECAY}",
          flush=True)
    T.main()
