"""2627 · G5 机制定论抖动补点 —— 净效应与「抖动判决」（纯函数，可失败单测对象）。

背景：2626（V2）在 13M、24k 步测得**机制净效应** `l1@K8 − repr01@K8 = +0.0144`，
只比 `docs/methodology.md` 的不可排序门槛 **0.014** 高 **0.0004（≈3%）**。本实验把
`repr01` 臂**同配置同 24k 再跑一遍**，用两次读数的**极差**估计 run-to-run 抖动，
并按 README §4 的预登记规则把结论定为「稳健 / 仍贴门槛 / 降级不可排序」。

本模块只放判据逻辑（不碰 GPU、不碰数据），供 `run.py` 与单测共用。
"""

# 2626/V2 24k 终点的机制净效应（本实验要估计其抖动的那个效应本身）。
EFFECT_V2 = 0.0144
# 预登记阈值（README §4，跑前写下、跑完不得改）：
DOWNGRADE_GE = 0.0144   # 极差 ≥ 效应本身 ⇒ 机制结论降级为「不可排序」
UPGRADE_LT = 0.005      # 极差 ≪ 效应（<0.005）⇒ 机制结论升格为「稳健」


def mechanism_net(l1_k8: float, repr_k8: float) -> float:
    """机制净效应 `l1@K8 − repr01@K8`：正 = 去相关臂的 K8 集成更好 = 机制净成立。"""
    return l1_k8 - repr_k8


def jitter_verdict(range_value: float, *, effect: float = EFFECT_V2,
                   downgrade_ge: float = DOWNGRADE_GE,
                   upgrade_lt: float = UPGRADE_LT) -> dict:
    """按预登记阈值判两次读数的**极差**（`range_value`）对机制结论的影响。

    三态（README §4）：
      - ``upgraded_to_robust``  极差 < ``upgrade_lt``（如 <0.005）⇒ 效应 ≫ 抖动 ⇒ 稳健；
      - ``downgraded_to_unorderable``  极差 ≥ ``downgrade_ge``（= 效应本身）⇒ 抖动吞掉效应；
      - ``on_threshold``  其余（``upgrade_lt`` ≤ 极差 < ``downgrade_ge``）⇒ 仍贴门槛、未决。
    """
    if range_value >= downgrade_ge:
        state = "downgraded_to_unorderable"
    elif range_value < upgrade_lt:
        state = "upgraded_to_robust"
    else:
        state = "on_threshold"
    return {
        "state": state,
        "range": range_value,
        "effect": effect,
        "downgrade_ge": downgrade_ge,
        "upgrade_lt": upgrade_lt,
        "range_over_effect": (range_value / effect) if effect else None,
        "note": ("极差 ≥ 效应本身 ⇒ 降级不可排序；< 0.005 ⇒ 升格稳健；其余 ⇒ 仍贴门槛"
                 "（README §4 预登记）"),
    }
