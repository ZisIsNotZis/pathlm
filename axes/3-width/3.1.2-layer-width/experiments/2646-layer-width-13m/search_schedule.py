"""2646 · 逐层宽度 schedule 的搜索（结果落盘，供契约 §2/§3 引用）。

契约（`README.md` §3）要求：**所有臂 Σ_l w_l² 逐位相等**，等宽臂取 `w = 192`
（Σ = 8·192² = 294,912），非均匀臂必须在「32 的整数倍、32 ≤ w ≤ 256」的格点上
找到 Σ_l w_l² == 294,912 的 schedule，并把搜索过程与结果落盘。

本脚本枚举三种形态，并把结果写进 `schedule_search.json`：

1. **PYR（非增金字塔）**：全部非增 8 元组；从中按「与契约示例
   `[256,224,224,192,192,160,160,128]` 的 L1 距离最小」挑选（契约示例本身
   Σw² = 307,200 ≠ 294,912，不可行，故取最近的可行 schedule）。
2. **HG（沙漏：中段宽两端窄）**：先枚举**严格左右对称**的单峰 schedule
   （`w1=w8 ≤ w2=w7 ≤ w3=w6 ≤ w4=w5`）。若为空集（本仓实测为空），
   退化为「PYR 的宽度多重集重排成单峰」——多重集不变 ⇒ Σw² 与 Σw 与 PYR
   **逐位相同**，臂间唯一差别就是「哪个宽度放在哪一层」。
3. **APYR**：PYR 的镜像（逐层倒序）。

用法：

    python3 search_schedule.py            # 打印并写 schedule_search.json
"""

import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

N_LAYERS = 8
GRID = list(range(32, 257, 32))          # 32 的整数倍，head_dim 恒 32 ⇒ 整头对齐
TARGET_SUM_W2 = N_LAYERS * 192 ** 2      # 等宽臂 w=192 的 Σw²
CONTRACT_EXAMPLE = [256, 224, 224, 192, 192, 160, 160, 128]   # 契约 §3 的示例（不可行）


def sum_w2(schedule) -> int:
    return sum(w * w for w in schedule)


def sum_w(schedule) -> int:
    return sum(schedule)


def non_increasing_schedules():
    """全部非增 8 元组（32 的整数倍，Σw² == 294,912）。"""
    out = []

    def rec(i, remaining, cur, last):
        if i == N_LAYERS:
            if remaining == 0:
                out.append(tuple(cur))
            return
        for w in GRID:
            if w > last or w * w > remaining:
                continue
            rec(i + 1, remaining - w * w, cur + [w], w)

    rec(0, TARGET_SUM_W2, [], 256)
    return out


def symmetric_unimodal_schedules():
    """严格左右对称的单峰 8 元组（w1=w8 ≤ w2=w7 ≤ w3=w6 ≤ w4=w5）。"""
    out = []
    for a, b, c, d in itertools.combinations_with_replacement(GRID, 4):
        if a > b or b > c or c > d:
            continue
        if 2 * (a * a + b * b + c * c + d * d) == TARGET_SUM_W2:
            out.append((a, b, c, d, d, c, b, a))
    return out


def l1(a, b) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def pick_pyr(candidates):
    """契约示例的 L1 最近可行解；并列时取字典序最小（确定性）。"""
    ranked = sorted(candidates, key=lambda s: (l1(list(s), CONTRACT_EXAMPLE), list(s)))
    best = ranked[0]
    return best, [(list(s), l1(list(s), CONTRACT_EXAMPLE)) for s in ranked]


def deal_ascending_to_ends(widths):
    """把宽度多重集按升序交替发到两端（外→内）⇒ 单峰（中段宽、两端窄）。

    升序多重集 `[w_1 ≤ … ≤ w_n]` 依次放到位置 `[0, n−1, 1, n−2, …]`。
    结果：最窄的两端在外侧、最宽的在正中（n 为偶数时落在中央两格之一）。
    """
    n = len(widths)
    positions = []
    lo, hi = 0, n - 1
    while lo <= hi:
        positions.append(lo)
        lo += 1
        if lo <= hi:
            positions.append(hi)
            hi -= 1
    out = [None] * n
    for w, pos in zip(sorted(widths), positions):
        out[pos] = w
    return out


def main():
    ni = non_increasing_schedules()
    sym = symmetric_unimodal_schedules()
    sym_nonuniform = [s for s in sym if len(set(s)) > 1]
    pyr, ranked = pick_pyr(ni)
    apyr = tuple(reversed(pyr))
    hg = tuple(deal_ascending_to_ends(list(pyr)))
    eq = (192,) * N_LAYERS

    result = {
        "grid": "32 的整数倍，32 ≤ w ≤ 256（head_dim 恒 32 ⇒ 整头对齐）",
        "target_sum_w2": TARGET_SUM_W2,
        "contract_example": CONTRACT_EXAMPLE,
        "contract_example_sum_w2": sum_w2(CONTRACT_EXAMPLE),
        "contract_example_feasible": sum_w2(CONTRACT_EXAMPLE) == TARGET_SUM_W2,
        "n_non_increasing": len(ni),
        "non_increasing_candidates": [list(s) for s in ni],
        "n_symmetric_unimodal": len(sym),
        "symmetric_unimodal_candidates": [list(s) for s in sym],
        "n_symmetric_unimodal_nonuniform": len(sym_nonuniform),
        "symmetric_unimodal_nonuniform_candidates": [list(s) for s in sym_nonuniform],
        "pyr_rule": ("非增 schedule 中与契约示例 [256,224,224,192,192,160,160,128] 的 "
                     "L1 距离最小者（并列取字典序最小）"),
        "pyr": list(pyr),
        "pyr_l1_to_contract_example": l1(list(pyr), CONTRACT_EXAMPLE),
        "pyr_layers_equal_to_contract_example": [
            i + 1 for i in range(N_LAYERS) if pyr[i] == CONTRACT_EXAMPLE[i]],
        "ranked_by_l1": [{"schedule": s, "l1": d} for s, d in ranked],
        "hg_rule": ("非均匀的严格左右对称单峰不可行（唯一的对称单峰解就是等宽本身，"
                    "见 symmetric_unimodal_candidates）⇒ 取 PYR 的宽度"
                    "多重集按升序交替发到两端（外→内）的单峰重排；多重集不变 ⇒ Σw² "
                    "与 Σw 与 PYR 逐位相同"),
        "hg": list(hg),
        "apyr_rule": "PYR 的镜像（逐层倒序）",
        "apyr": list(apyr),
        "arms": {name: {"schedule": list(s), "sum_w2": sum_w2(s), "sum_w": sum_w(s),
                        "sum_w2_eq_target": sum_w2(s) == TARGET_SUM_W2}
                 for name, s in (("EQ", eq), ("PYR", pyr), ("HG", hg), ("APYR", apyr))},
    }
    json.dump(result, open(os.path.join(HERE, "schedule_search.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in (
        "target_sum_w2", "contract_example_sum_w2", "n_non_increasing",
        "n_symmetric_unimodal", "n_symmetric_unimodal_nonuniform",
        "pyr", "hg", "apyr", "pyr_l1_to_contract_example",
        "pyr_layers_equal_to_contract_example")}, ensure_ascii=False, indent=2))
    assert result["arms"]["EQ"]["sum_w2"] == TARGET_SUM_W2
    for name in ("PYR", "HG", "APYR"):
        assert result["arms"][name]["sum_w2"] == TARGET_SUM_W2, name
    assert sorted(pyr) == sorted(hg) == sorted(apyr), "三重集必须逐位相同"
    print("OK: Σw² 四臂逐位相等 =", TARGET_SUM_W2)


if __name__ == "__main__":
    main()
