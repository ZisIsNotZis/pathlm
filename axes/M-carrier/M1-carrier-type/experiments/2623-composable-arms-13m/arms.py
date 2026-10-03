"""2623-composable-arms-13m · 四臂定义 / 评测可见性 / 阶梯增量（可失败单测的靶子）。

本模块只放**纯逻辑**（不含 GPU、不读写权重），好让 `tests/test_composable_arms_13m.py`
把「四臂真的是四个不同配置」「窗口臂在 T=256 上非退化」「叠加增量是逐臂相邻差」这三件
事钉成可失败单测，并被 `mutate.py` 逐处抓红。

四臂（唯一变量 = 特性叠加，逐字段同配方；base = stock）：

| 臂 | 叠加的特性 | 引擎位点 |
|---|---|---|
| `stock`            | （无，基准） | `Block` |
| `gqa`              | +GQA(2 KV 头) | `GQABlock` |
| `gqa_carry`        | +carry(delta 写) | `ComposableBlock`（ENG1 前不存在） |
| `gqa_carry_window` | +window(16, anchors 0) | `ComposableBlock` + `attn_mask` |

`window` 是与块类**正交**的装饰层：它不改块类，只改可见集；`carry` 在 `gqa` 之上叠加
时块类从 `GQABlock` 变成 `ComposableBlock`（ENG1 解锁）。窗口也是**制造长程需求**的那一步
（key 落在窗外 ⇒ 不搬就答不出，SCHEMA §19）。
"""

from __future__ import annotations

#: 四臂，顺序 = 阶梯顺序（相邻两臂只差一个特性的叠加）。
ARMS = (
    {"name": "stock", "flags": {}, "stacked": []},
    {"name": "gqa", "flags": {"n_kv_heads": 2}, "stacked": ["gqa"]},
    {"name": "gqa_carry",
     "flags": {"n_kv_heads": 2, "carry_rule": "delta"},
     "stacked": ["gqa", "carry"]},
    {"name": "gqa_carry_window",
     "flags": {"n_kv_heads": 2, "carry_rule": "delta",
               "train_window": 16, "train_anchors": 0},
     "stacked": ["gqa", "carry", "window"]},
)

ARM_NAMES = tuple(a["name"] for a in ARMS)
BASELINE_ARM = "stock"
TARGET_ARM = "gqa_carry_window"


def arm_spec(name: str) -> dict:
    for a in ARMS:
        if a["name"] == name:
            return a
    raise KeyError(f"未知臂 {name!r}；期望 {ARM_NAMES}")


def model_config_kwargs(spec: dict) -> dict:
    """臂 → `DepthARConfig` 关键字（唯一变量 = 这些字段的叠加）。"""
    return dict(spec["flags"])


def expected_block_class(spec: dict, n_heads: int = 8) -> str:
    """该臂在引擎里应落的块类（ENG1 的组合选择规则的镜像；ACTIVITY 用）。

    GQA × carry 同时开 ⇒ `ComposableBlock`；单开 GQA ⇒ `GQABlock`；单开 carry ⇒
    `CarryChannelBlock`；都关 ⇒ `Block`。窗口是正交装饰层，不影响块类。
    """
    f = spec["flags"]
    gqa = f.get("n_kv_heads") is not None and f["n_kv_heads"] < n_heads
    carry = f.get("carry_rule") is not None
    if gqa and carry:
        return "ComposableBlock"
    if gqa:
        return "GQABlock"
    if carry:
        return "CarryChannelBlock"
    return "Block"


def native_visibility(spec: dict) -> tuple[int, int]:
    """评测口径的可见性（= 该臂训练时的可见性；训练/评测同一掩码）。

    前两臂与 `gqa_carry` 在**全因果**注意力下训练与评测；只有 `gqa_carry_window`
    用 `window=16 / anchors=0`。返回 `(window, anchors)`，`window=0` 表示全因果。
    """
    f = spec["flags"]
    return int(f.get("train_window", 0)), int(f.get("train_anchors", 0))


def staircase_increments(acc: dict) -> list[dict]:
    """逐叠加步的增量（§17.17 分量分解）：相邻两臂的准确率差。

    `acc = {arm_name: accuracy}`。返回每步的 `delta_pp = 100·(acc[to] − acc[from])`，
    外加净和 `net_pp = 100·(acc[target] − acc[baseline])`。
    """
    out = []
    for prev, cur in zip(ARMS, ARMS[1:]):
        a, b = acc[prev["name"]], acc[cur["name"]]
        out.append({"from": prev["name"], "to": cur["name"],
                    "added": cur["stacked"][-1],
                    "delta_pp": round(100.0 * (b - a), 3)})
    return out


def net_vs_baseline(acc: dict) -> float:
    """净和：目标臂 − 基准臂（pp）。"""
    return round(100.0 * (acc[TARGET_ARM] - acc[BASELINE_ARM]), 3)
