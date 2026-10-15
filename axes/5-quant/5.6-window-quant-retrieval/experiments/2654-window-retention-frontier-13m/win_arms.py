"""2654 · WP1 语义层：保留预算曲线的窗口臂 + 窗口 × int8/int4 块标尺组合臂的**臂定义**、
同字节水平匹配与配对 Δ/单调性判读工具。

契约：同目录 `README.md` + `config.json::criterion`（**运行前**写入）。
本模块**不实现前向**：量化/掩码前向只读复用 `2632/block_arms.py`（块标尺，scheme="block"、
G=32）与 `2621/rest_arms.py`（可见性掩码），纯窗口/fp32 臂复用 `2619/run.py` 的分派。
本模块只做三件既有实现里没有的事：

1. **臂规格的唯一home**（`SPECS`）——每个臂的 kind/mode/value_bytes/window 与它在
   `2642` 那 7 个同字节水平上的落点；
2. **同字节水平匹配**（`level_of`）——只接受**精确命中**，不插值；
3. **配对 Δ 的统计**（`paired_band`）与**单调性判读**（`monotone_report`）。

横轴（与 `2619`/`2621`/`2642` 逐字同一式，部署口径）：

    部署字节/位置 = 2(K,V) · L · n_kv_heads_eff · head_dim · (值字节 + 标尺字节/值) · (kept / T)

块标尺 `标尺字节/值 = 2/G = 0.0625`（G=32，沿 head_dim，严格前缀确定）；非量化臂为 0。
"""

from __future__ import annotations

#: `2642` 给出的 7 个跨家族同字节水平（部署口径，块标尺 G=32）。
#: 来源：axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json::same_byte_groups
SAME_BYTE_LEVELS_2642 = (432.0, 768.0, 816.0, 864.0, 1536.0, 1632.0, 3072.0)

FP16_FULL_BYTES_PER_POSITION = 12288.0
VALUE_BYTES = {"fp32": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}
SCALE_BYTES_PER_VALUE_BLOCK_G32 = 2.0 / 32.0


#: WP1 的臂规格（唯一 home）。`positions_kept` 只给 kind="window"/"quant" 用；
#: combo_mask 的保留位置数由 `rest_arms.positions_kept`（window + anchors）推出，不在这里重复。
SPECS = {
    "full": {"family": "trunk", "kind": "quant", "mode": "fp32", "value_bytes": 2.0,
             "positions_kept": 256,
             "role": "不压缩基线：严格因果、同拓扑数值恒等（fp32 恒等往返）"},
    "win16": {"family": "trunk", "kind": "window", "d": 16, "value_bytes": 2.0,
              "positions_kept": 16,
              "role": "纯窗口 W=16（768 B/位置）：816 B 水平下最接近的**单用窗口**对照点"},
    "win32": {"family": "trunk", "kind": "window", "d": 32, "value_bytes": 2.0,
              "positions_kept": 32,
              "role": "纯窗口 W=32（1536 B/位置）：M2 ERRATA keep32 对照；1536 水平组"},
    "win64": {"family": "trunk", "kind": "window", "d": 64, "value_bytes": 2.0,
              "positions_kept": 64,
              "role": "纯窗口 W=64（3072 B/位置）：M2 ERRATA keep64 对照；3072 水平组"},
    "win96": {"family": "trunk", "kind": "window", "d": 96, "value_bytes": 2.0,
              "positions_kept": 96,
              "role": "纯窗口 W=96（4608 B/位置）：keep32→192 里**历史缺失**的保留预算点"},
    "win128": {"family": "trunk", "kind": "window", "d": 128, "value_bytes": 2.0,
               "positions_kept": 128, "role": "纯窗口 W=128（6144 B/位置）"},
    "win160": {"family": "trunk", "kind": "window", "d": 160, "value_bytes": 2.0,
               "positions_kept": 160, "role": "纯窗口 W=160（7680 B/位置）"},
    "win192": {"family": "trunk", "kind": "window", "d": 192, "value_bytes": 2.0,
               "positions_kept": 192, "role": "纯窗口 W=192（9216 B/位置）：M2 ERRATA keep192 对照"},
    "win32_int8": {"family": "trunk", "kind": "combo_mask", "base_model": "trunk",
                   "window": 32, "anchors": 0, "mode": "int8", "value_bytes": 1.0,
                   "role": "组合：窗口 W=32 × int8 块标尺（部署 816 B/位置 = 2642 水平之一）"},
    "win64_int8": {"family": "trunk", "kind": "combo_mask", "base_model": "trunk",
                   "window": 64, "anchors": 0, "mode": "int8", "value_bytes": 1.0,
                   "role": "组合：窗口 W=64 × int8 块标尺（部署 1632 B/位置 = 2642 水平之一）"},
    "win32_int4": {"family": "trunk", "kind": "combo_mask", "base_model": "trunk",
                   "window": 32, "anchors": 0, "mode": "int4", "value_bytes": 0.5,
                   "role": "组合：窗口 W=32 × int4 块标尺（部署 432 B/位置 = 2642 水平之一）"},
    "win64_int4": {"family": "trunk", "kind": "combo_mask", "base_model": "trunk",
                   "window": 64, "anchors": 0, "mode": "int4", "value_bytes": 0.5,
                   "role": "组合：窗口 W=64 × int4 块标尺（部署 864 B/位置 = 2642 水平之一）"},
}

#: 臂的顺序 = 执行顺序 = 优先级（被 30 min 硬上限切断时，前面的必须已经落盘）。
#: 先给曲线与两个契约点名的组合臂，再给配对带，最后给补齐的保留预算点与 int4 组合。
ARM_ORDER = ["full", "win32", "win64", "win96", "win32_int8", "win64_int8", "win16",
             "win192", "win128", "win160", "win32_int4", "win64_int4"]
BAND_AFTER = "win16"                      # 跑到这里就跑配对带（5 窗口），再跑后面的补点

CORRUPT_ARMS = ("full", "win64", "win64_int8")
QUANT_MODES = ("int8", "int4")


def is_block_quant(arm: str) -> bool:
    """该臂的量化腿是否走本实验的块标尺（否则走 per_value 的既有分派）。"""
    return SPECS[arm].get("mode") in QUANT_MODES


def deployment_bytes_of(spec: dict, point: dict, cfg: dict) -> float:
    """部署字节（值 + 标尺）：量化臂按块标尺 G=32 摊标尺，非量化臂 = 值字节。

    与 `2632/scale_caliber.py::deployment_bytes` **同一式**（本函数只做非量化臂的短路，
    量化臂的账由 `scale_caliber` 算并逐位对账）。
    """
    values_per_position = (2.0 * cfg["model"]["n_layers"] * point["factor_n_kv_heads_eff"]
                           * (cfg["model"]["d_model"] // cfg["model"]["n_heads"])
                           * point["factor_positions_kept_over_T"])
    mode = spec.get("mode", "fp16")
    if mode not in QUANT_MODES:
        return values_per_position * VALUE_BYTES[mode]
    return values_per_position * (VALUE_BYTES[mode] + SCALE_BYTES_PER_VALUE_BLOCK_G32)


def level_of(deployment_bytes: float) -> float | None:
    """命中的 `2642` 同字节水平；**只接受精确命中**，否则 None（禁插值）。"""
    for lv in SAME_BYTE_LEVELS_2642:
        if abs(deployment_bytes - lv) < 1e-6:
            return lv
    return None


def paired_band(bpc_by_seed: dict[str, dict[str, float]], arm_a: str,
                arm_b: str) -> dict:
    """配对 Δ（`arm_b − arm_a`，正 = b 更差）在两个臂共有的每个评测窗口上的读数。

    标准口径 = **配对 Δ 的区间，≥5 个窗口**；<5 窗口时调用方必须退用更保守的单臂带
    （见 `single_arm_band`），且两带不得混用。
    """
    seeds = sorted(set(bpc_by_seed.get(arm_a, {})) & set(bpc_by_seed.get(arm_b, {})))
    deltas = {s: round(bpc_by_seed[arm_b][s] - bpc_by_seed[arm_a][s], 9) for s in seeds}
    vals = [deltas[s] for s in seeds]
    if not vals:
        return {"pair": f"{arm_b} - {arm_a}", "n_windows": 0, "deltas_by_seed": deltas,
                "interval": None, "mean": None, "standard_met": False}
    return {"pair": f"{arm_b} - {arm_a}", "n_windows": len(seeds), "deltas_by_seed": deltas,
            "interval": [min(vals), max(vals)],
            "mean": round(sum(vals) / len(vals), 9),
            "spread": round(max(vals) - min(vals), 9),
            "standard_met": len(seeds) >= 5,
            "note": "配对 Δ 在**同一评测窗口**上做差 ⇒ 窗口数据方差对消，只留臂间差。"}


def single_arm_band(bpc_by_seed: dict[str, dict[str, float]]) -> dict:
    """单臂跨窗口 bpc 极差（保守退路带；标准口径跑不满 ≥5 窗口时用它，两带不混用）。"""
    out = {}
    for arm, per_seed in bpc_by_seed.items():
        vals = [per_seed[s] for s in sorted(per_seed)]
        if len(vals) < 2:
            continue
        out[arm] = {"bpc_by_seed": {s: per_seed[s] for s in sorted(per_seed)},
                    "range": round(max(vals) - min(vals), 9)}
    return {"per_arm": out,
            "in_force_band_bpc": (max(v["range"] for v in out.values()) if out else None)}


def monotone_report(damages: list[dict]) -> dict:
    """保留预算曲线是否单调：`damages` = [{"kept": int, "damage_bpc": float}, ...]。

    「单调」= 保留位置数升序时 damage_bpc 严格不增（允许相等）；报逐点数值与第一个违例。
    """
    pts = sorted(damages, key=lambda d: d["kept"])
    viol = []
    for a, b in zip(pts, pts[1:]):
        if b["damage_bpc"] > a["damage_bpc"] + 1e-12:
            viol.append({"from_kept": a["kept"], "to_kept": b["kept"],
                         "damage_from": a["damage_bpc"], "damage_to": b["damage_bpc"]})
    return {"points": [{"kept": p["kept"], "damage_bpc": p["damage_bpc"]} for p in pts],
            "monotone_nonincreasing": not viol, "violations": viol,
            "range_bpc": (round(pts[0]["damage_bpc"] - pts[-1]["damage_bpc"], 9)
                          if len(pts) >= 2 else None)}
