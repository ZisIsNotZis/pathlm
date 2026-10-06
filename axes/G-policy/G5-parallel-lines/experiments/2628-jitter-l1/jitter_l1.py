"""2628 · G5 机制定论抖动**补全**（l1 臂）—— 两臂合并极差与「抖动判决」（纯函数，可失败单测对象）。

背景：2627（ticket 44-X1）把 `repr01` 臂同配置同 24k 再跑一遍，测出机制净效应
`l1@K8 − repr01@K8` 的 run-to-run 极差 **0.0035**，据此把「机制熬过收敛」从「贴门槛」
升格为「稳健」。但 `l1@K8` 是 **2626 的固定读数（没有重跑）** ⇒ 那个极差只含 `repr01`
臂自身的抖动，是净效应抖动的**下界口径**（2627 `analysis.md` §10 未测项 #1）。

本实验把 **`l1` 臂也同配置同 24k 重跑一遍**，用**两臂合并极差**
（`repr01` 的 0.0035 与 `l1` 臂抖动**取大**）复核「稳健」结论是否站得住：
  - 合并极差 ≥ 效应本身（0.0144）⇒ 机制结论**降级回「不可排序」**；
  - 合并极差 ≪ 效应（<0.005）⇒ **维持「稳健」**，并把口径从下界升为**两臂完整口径**；
  - 其余 ⇒ **仍贴门槛 / 未决**。

本模块只放判据逻辑（不碰 GPU、不碰数据），供 `run.py` 与单测共用。
"""

# 2627（ticket 44-X1）已测：repr01 臂两次 `repr01@K8` 之差（run B − run A = 0.0035）。
REPR01_JITTER_2627 = 0.0035
# 2626（V2）24k 终点的机制净效应（本实验要保护其稳健性的那个效应本身）。
EFFECT_V2 = 0.0144
# 预登记阈值（README §4，跑前写下、跑完不得改）：
DOWNGRADE_GE = 0.0144   # 合并极差 ≥ 效应本身 ⇒ 机制结论降级为「不可排序」
UPGRADE_LT = 0.005      # 合并极差 ≪ 效应（<0.005）⇒ 维持「稳健」并升格口径


def mechanism_net(l1_k8: float, repr_k8: float) -> float:
    """机制净效应 `l1@K8 − repr01@K8`：正 = 去相关臂的 K8 集成更好 = 机制净成立。"""
    return l1_k8 - repr_k8


def arm_jitter(a: float, b: float) -> float:
    """同一臂两次同配置训练的读数极差（n=2 ⇒ 差与极差同值）。"""
    return abs(b - a)


def combined_range(l1_jitter: float, repr_jitter: float) -> float:
    """**两臂合并极差** = 两臂各自极差的**较大者**（README §4 预登记：
    「repr01 的 0.0035 与 `l1` 的抖动取大」）。"""
    return max(l1_jitter, repr_jitter)


def worst_case_additive_range(l1_jitter: float, repr_jitter: float) -> float:
    """**并列口径（更保守，非 primary）**：净效应 = `l1 − repr01` 的两个被减数各自抖动，
    若两个极端同时出现（对抗性配对），净效应极差的上界 = 两臂极差之和。
    本实验**只把 `combined_range`（取大）作为预登记 primary**，本函数仅作并列披露。"""
    return l1_jitter + repr_jitter


def combos_range(net_values: list[float]) -> float:
    """**并列口径（非 primary）**：把两臂各自的两次读数做 2×2 组合，报净效应取值的极差
    （= max − min，恒等于两臂极差之和）。仅作披露，不参与判决。"""
    return max(net_values) - min(net_values) if net_values else 0.0


def jitter_verdict(range_value: float, *, effect: float = EFFECT_V2,
                   downgrade_ge: float = DOWNGRADE_GE,
                   upgrade_lt: float = UPGRADE_LT) -> dict:
    """按预登记阈值判**两臂合并极差**（`range_value`）对机制结论的影响。

    三态（README §4）：
      - ``maintained_robust``  合并极差 < ``upgrade_lt``（如 <0.005）⇒ 效应 ≫ 抖动 ⇒ 维持稳健；
      - ``downgraded_to_unorderable``  合并极差 ≥ ``downgrade_ge``（= 效应本身）⇒ 抖动吞掉效应；
      - ``on_threshold``  其余（``upgrade_lt`` ≤ 极差 < ``downgrade_ge``）⇒ 仍贴门槛、未决。
    """
    if range_value >= downgrade_ge:
        state = "downgraded_to_unorderable"
    elif range_value < upgrade_lt:
        state = "maintained_robust"
    else:
        state = "on_threshold"
    return {
        "state": state,
        "range": range_value,
        "effect": effect,
        "downgrade_ge": downgrade_ge,
        "upgrade_lt": upgrade_lt,
        "range_over_effect": (range_value / effect) if effect else None,
        "note": ("合并极差 ≥ 效应本身 ⇒ 降级不可排序；< 0.005 ⇒ 维持稳健；其余 ⇒ 仍贴门槛"
                 "（README §4 预登记）"),
    }
