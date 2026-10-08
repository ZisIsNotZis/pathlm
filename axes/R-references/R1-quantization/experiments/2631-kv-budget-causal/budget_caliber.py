"""2631 · 部署口径的字节核算（**值 + 标尺**）—— 纯解析，不取 GPU，不跑 torch。

契约：同目录 `config.json::causal_caliber`（判据线在运行前写入）与 `analysis.md`。

**为什么需要这个模块。** `2619/2620/2621` 的横轴只算**值字节**
（`2·n_layers·n_kv_heads_eff·head_dim·value_bytes·kept/T`），等于把 scale 当免费。
L1 的 F1 修复把量化器改成「逐请求、因果、流式」：位置 `t` 的 scale =
`max_{t'≤t}|x|/qmax`、写完不再改（`tools/evalkit/kv_quant.quantize_kv` 的 `cummax` 版）。
「过去位置严格前缀确定」的价钱是 **scale 逐位置变、无法摊薄** ⇒ 每个值配一个 fp16 标尺。
于是同一个 int8 档：值 1 B/值 + 标尺 2 B/值 = **3 B/值**，比 fp16 基线的 2 B/值更差。
值字节列看不出这笔开销，本模块把它算清楚，给出两套**可行**方案的每位置真实字节：

- `a_per_value` · **逐值标尺（严格前缀确定）**：每值 1 个 fp16 标尺 ⇒ `+2 B/值`。
  与本实验质量数字的实现（`quantize_kv` 的 `cummax` 版）**一一对应** ⇒ 这是同源价签。
- `b_grouped` · **分块标尺 + 块内回填**：每 `G` 个**连续可见**位置共享 1 个 fp16 标尺
  （`scale = 块内 amax/qmax`，块关闭时按最终 scale 重编码回填）⇒ 摊薄后 `+2/G B/值`。
  代价三条，全部记账、不许隐去：
  1. **块内不再严格前缀确定**：块首位置的存储表示会被同块后续 `G−1` 个位置改写；出块即冻结。
     最大受影响距离 = `G−1` 个位置（见 `group_violation_distance`）。
  2. **回填要 staging**：块关闭前必须留着原始 fp16 值才能重编码 ⇒ 每请求平均
     `min(G−1, kept)/2` 个位置的 fp16 缓冲（见 `staging_bytes_per_request`），与 `T` 无关。
  3. **摊薄分母 = 该标尺覆盖的**保留**值数**：本模块取 `min(G, 连续段长)`，是**乐观上界**
     （窗口/块不对齐时分母更小、省得略少；最坏 2/8 vs 2/16，量级 ≤0.125 B/值）。
  (b) 是**放弃严格前缀确定性换来的省内存**，两者不可同时拿；本实验**没有**实现 (b)，
  所以 (b) 只有解析字节、质量代价未知。

另有一条只作解析注记、本模块不算进两方案（`mixed_axis_note`）：标尺沿**非 token 轴**共享时
可以不违反因果——V 若按 token 取 `head_dim` 维 amax，位置 `t` 的标尺只由 `t` 决定
⇒ 严格因果且 `+2/32 B/值`；K 侧做不到（沿 token 轴才有足够动态范围），所以业界（KIVI）是
「K 分块 + V 逐 token」。该设计下违反前缀确定的只有 K 一半的值。

本模块**不改**横轴公式：`deployment_bytes` 直接复用 `results.json` 里该臂已记录的
`factor_n_kv_heads_eff` / `factor_positions_kept_over_T`（判据 C1 的唯一公式），并按
`kept_segments` 把保留位置拆成连续段；两者对不上时**直接报错**（防口径漂移）。
"""

from __future__ import annotations

#: 两套方案的标识（`analysis.md` / `table.md` 用同名字段引用）。
SCHEMES = ("a_per_value", "b_grouped")

#: 每个标尺的存储字节数：fp16 标尺 = 2 B（`quantize_kv` 用 fp32 计算、按 fp16 存的标准做法）。
SCALE_BYTES = 2.0

#: (b) 的默认块大小（连续 token 位置数）：KIVI 类设计的常见档。
DEFAULT_GROUP = 32

#: 块大小敏感性表用的档位。
GROUP_LADDER = (8, 16, 32, 64, 128)

#: 量化档（需要标尺的档）；fp16/fp32 是纯精度转换，没有标尺。
_QUANT_MODES = ("int8", "int4")

#: 每值字节（解析口径），与 `tools/evalkit/kv_quant.py` 的存储精度一一对应。
VALUE_BYTES = {"fp32": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}

#: (b) 方案「反解」出的一条重要事实：**(a) 是 (b) 在 G = 1 时的特例**
#: （见单测 `test_b_group_size_one_equals_per_value_scheme`）。


# --------------------------------------------------------------- 每值 / 每段


def mode_of(spec: dict) -> str:
    """臂的存储精度。校验 `mode` 与 `value_bytes` **标签与事实一致**（不一致直接报错，
    沿用 R1 线 `gaps_arms` / `rest_arms` 的同一条规则）。

    没有 `mode` 键的臂（`window` / `head`）按 `value_bytes` 反推：2 B/值 = fp16 存储假设
    （同 2619 的全精度基线列，不是 fp32 的 4 B/值）。
    """
    if "mode" in spec:
        mode = spec["mode"]
        if mode not in VALUE_BYTES:
            raise ValueError(f"未知 mode={mode!r}；已知 {sorted(VALUE_BYTES)}")
        vb = float(spec["value_bytes"])
        if abs(vb - VALUE_BYTES[mode]) > 1e-9:
            raise ValueError(
                f"臂 {spec.get('role', '?')!r}: value_bytes={vb} 与 mode={mode}"
                f"（应 {VALUE_BYTES[mode]}）不一致 ⇒ 标签与事实不符，拒绝出数")
        return mode
    vb = float(spec["value_bytes"])
    for mode in ("fp16", "int8", "int4", "fp32"):
        if abs(vb - VALUE_BYTES[mode]) < 1e-9:
            return mode
    raise ValueError(f"value_bytes={vb} 不对应任何已知精度 {sorted(VALUE_BYTES)}")


def needs_scale(mode: str) -> bool:
    """该精度是否需要标尺（fp16 / fp32 只是精度转换，不需要）。"""
    if mode not in VALUE_BYTES:
        raise ValueError(f"未知 mode={mode!r}")
    return mode in _QUANT_MODES


def scale_bytes_per_value(mode: str, scheme: str, group: int = DEFAULT_GROUP) -> float:
    """每值摊到的标尺字节数。

    - `a_per_value`：每值 1 个 fp16 标尺 ⇒ 2.0（与 `quantize_kv` 的 `cummax` 版一一对应）。
    - `b_grouped`：每 `group` 个连续可见位置 1 个标尺 ⇒ `2 / group`。
    - 非量化档两套都是 0。
    """
    if scheme not in SCHEMES:
        raise ValueError(f"未知方案 {scheme!r}；已知 {SCHEMES}")
    if not needs_scale(mode):
        return 0.0
    if scheme == "a_per_value":
        return SCALE_BYTES
    if group < 1:
        raise ValueError(f"块大小必须是 ≥1 的整数，收到 {group}")
    return SCALE_BYTES / group


def bytes_per_value(mode: str, scheme: str, group: int = DEFAULT_GROUP) -> float:
    """值 + 标尺的实际每值字节（(a)/(b) 的差别全在这一个数字上）。"""
    return VALUE_BYTES[mode] + scale_bytes_per_value(mode, scheme, group)


def kept_segments(spec: dict, T: int) -> list[int]:
    """保留位置的**连续段**长度（判据 C1 的 `positions_kept` 的分解，按各 kind 的真实掩码语义）。

    - `quant` / `head` / `quant_head`：全因果 ⇒ 一段 `[T]`；
    - `window`：最近 `d` 个位置 ⇒ `[d]`；
    - `window_anchor` / `combo_mask`（带锚点）：最前 `anchors` ∪ 最近 `window` ⇒ `[anchors, window]`；
    - `combo_mask`（无锚点）：`[window]`。
    """
    kind = spec["kind"]
    if kind in ("quant", "head", "quant_head"):
        return [T]
    if kind == "window":
        return [int(spec["d"])]
    if kind in ("window_anchor", "combo_mask"):
        w = int(spec["window"])
        a = int(spec.get("anchors", 0))
        return ([a] if a > 0 else []) + [w]
    raise ValueError(f"未知 kind={kind!r}：本模块的段模型需要逐 kind 定义")


def values_per_position(cfg: dict, spec: dict, n_kv_heads_eff: int,
                        positions_kept_over_T: float) -> float:
    """每上下文位置的**标量值数**（K + V）：`2 · n_layers · n_kv_heads_eff · head_dim · kept/T`。"""
    n_layers = cfg["model"]["n_layers"]
    hd = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    return 2.0 * n_layers * n_kv_heads_eff * hd * positions_kept_over_T


# --------------------------------------------------------------- 两套方案的字节


def _factors_of(point: dict) -> tuple[int, float]:
    """从 `results.json` 的点里取判据 C1 记录的两个乘子（`n_kv_heads_eff` / `kept/T`）。"""
    for key in ("factor_n_kv_heads_eff", "factor_positions_kept_over_T"):
        if key not in point:
            raise KeyError(f"点 {point.get('arm')!r} 缺 {key}：无法复用判据 C1 的横轴公式")
    return int(point["factor_n_kv_heads_eff"]), float(point["factor_positions_kept_over_T"])


def deployment_bytes(point: dict, spec: dict, cfg: dict,
                     scheme: str, group: int = DEFAULT_GROUP) -> dict:
    """一个臂在两套方案下的**真实**字节/位置（值 + 标尺），并与值字节口径逐位对账。

    返回：`bytes_per_position`（总值字节）、`scale_bytes_per_position`（标尺那部分）、
    `bytes_per_value`、`ratio_vs_value_caliber`、以及 (b) 的 `group_used` 明细。
    `point["bytes_per_position"]` 是旧的值字节口径（本实验横轴），两者相除即倍率。
    """
    mode = mode_of(spec)
    nkv, kept_over_T = _factors_of(point)
    segments = kept_segments(spec, cfg["model"]["seq_len"])
    seg_sum = sum(segments)
    kept = int(point.get("factor_positions_kept", seg_sum))
    if seg_sum != kept:
        raise ValueError(
            f"臂 {point.get('arm')!r}：段模型和 = {seg_sum} 与判据 C1 记录的 "
            f"positions_kept = {kept} 不一致（口径漂移，拒绝出数）")
    if abs(sum(segments) / cfg["model"]["seq_len"] - kept_over_T) > 1e-9:
        raise ValueError(
            f"臂 {point.get('arm')!r}：Σ段/T = {sum(segments) / cfg['model']['seq_len']:.9f} "
            f"与记录的 kept/T = {kept_over_T:.9f} 不一致（口径漂移，拒绝出数）")

    n_layers = cfg["model"]["n_layers"]
    hd = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    per_seg = []
    for seg in segments:
        seg_over_T = seg / cfg["model"]["seq_len"]
        # (a)：每个值自己一个标尺 ⇒ 分母 1；(b)：该段内每 group 个位置一个标尺，
        # 分母 = min(段长, group)（乐观上界口径，见模块 docstring 第 3 条）。
        group_used = 1 if scheme == "a_per_value" else min(group, seg)
        bpe = bytes_per_value(mode, scheme, group_used)
        vals = 2.0 * n_layers * nkv * hd * seg_over_T
        per_seg.append({"segment_len": seg, "group_used": group_used,
                        "values_per_position": vals,
                        "bytes_per_value": bpe,
                        "bytes_per_position": vals * bpe})
    total = sum(s["bytes_per_position"] for s in per_seg)
    vals_total = sum(s["values_per_position"] for s in per_seg)
    value_part = vals_total * VALUE_BYTES[mode]
    if abs(value_part - float(point["bytes_per_position"])) > 1e-6:
        raise ValueError(
            f"臂 {point.get('arm')!r}：本模块算的值字节 {value_part} 与该点已记录的 "
            f"bytes_per_position = {point['bytes_per_position']} 不等（判据 C1 对账失败）")
    scale_part = total - value_part
    return {"scheme": scheme, "mode": mode,
            "group": group if scheme == "b_grouped" else 1,
            "needs_scale": needs_scale(mode),
            # 名义摊薄比例（(a) 恒 2、(b) 恒 2/group）与**实际加权**比例（多段臂会略高于名义值）
            "scale_bytes_per_value_nominal": (scale_bytes_per_value(mode, scheme, group)
                                              if needs_scale(mode) else 0.0),
            "values_per_position": vals_total,
            "value_bytes_per_position": value_part,
            "scale_bytes_per_position": scale_part,
            "bytes_per_position": total,
            "scale_bytes_per_value": scale_part / vals_total if vals_total else 0.0,
            "bytes_per_value": value_part / vals_total + (scale_part / vals_total
                                                          if vals_total else 0.0),
            "ratio_vs_value_caliber": total / float(point["bytes_per_position"]),
            "per_segment": per_seg}


def group_violation_distance(group: int) -> int:
    """(b) 的**违反程度**：块边界处过去值受未来影响的最大距离（位置数）。

    `group = 1` ⇒ 退回逐值标尺（0：严格前缀确定）；否则块首位置会被同块后续
    `group − 1` 个位置改写。**这是 (b) 相对于 (a) 唯一让出的性质**。
    """
    if group < 1:
        raise ValueError(f"块大小必须是 ≥1 的整数，收到 {group}")
    return group - 1


def staging_bytes_per_request(cfg: dict, spec: dict, point: dict,
                             group: int = DEFAULT_GROUP) -> dict:
    """(b) 回填所需的 staging 缓冲（数组本身不进「每位置字节」列，单列在此）。

    块未关闭前要留住原始 fp16 值才能按最终 scale 重编码 ⇒ 每请求约
    `min(G−1, kept)/2` 个位置的 fp16 KV（平均占用半个块）。与 `T` 无关：`T` 越大越可忽略。
    """
    nkv, _ = _factors_of(point)
    hd = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    vals_per_position = 2.0 * cfg["model"]["n_layers"] * nkv * hd
    positions = 0.5 * min(group - 1, int(point.get("factor_positions_kept",
                                                  cfg["model"]["seq_len"])))
    b = positions * vals_per_position * VALUE_BYTES["fp16"]
    T = cfg["model"]["seq_len"]
    return {"group": group, "mean_positions_held": positions,
            "bytes_per_request": b,
            "bytes_per_position_at_T": b / T,
            "T": T,
            "note": "fp16 原值 staging（重编码用）；与目标精度无关，int4 也要留"}


# --------------------------------------------------------------- 汇总块（落 results.json）


def caliber_block(cfg: dict, specs: dict, points: list[dict],
                  group: int = DEFAULT_GROUP) -> dict:
    """两套方案的定义 + 逐臂数字 + (b) 的违反距离 / staging / 块大小敏感性。

    `specs` = `cfg["arms"]`；`points` = `results.json::points`（带去重后的臂顺序）。
    """
    by_arm = {p["arm"]: p for p in points}
    arms = {}
    for arm, p in by_arm.items():
        spec = specs[arm]
        arms[arm] = {s: deployment_bytes(p, spec, cfg, s, group) for s in SCHEMES}
    quant_arms = [a for a, p in by_arm.items()
                  if needs_scale(mode_of(specs[a]))]
    full_trunk = by_arm.get("fp16")
    staging = (staging_bytes_per_request(cfg, specs["int8"], full_trunk, group)
               if full_trunk is not None else None)
    ladder = []
    for g in GROUP_LADDER:
        ladder.append({
            "group": g,
            "violation_distance_positions": group_violation_distance(g),
            "scale_bytes_per_value": SCALE_BYTES / g,
            "int8_bytes_per_value": bytes_per_value("int8", "b_grouped", g),
            "int4_bytes_per_value": bytes_per_value("int4", "b_grouped", g),
            "int8_reduction_vs_fp16": 1 - bytes_per_value("int8", "b_grouped", g)
            / bytes_per_value("fp16", "b_grouped", g),
            "int4_reduction_vs_fp16": 1 - bytes_per_value("int4", "b_grouped", g)
            / bytes_per_value("fp16", "b_grouped", g)})
    return {
        "group_default": group,
        "schemes": {
            "a_per_value": ("逐值标尺（严格前缀确定）：每值 1 个 fp16 标尺 = +2 B/值；"
                            "与 tools/evalkit/kv_quant.quantize_kv 的 cummax 实现同源"),
            "b_grouped": (f"分块标尺 + 块内回填：每 {group} 个连续可见位置 1 个 fp16 标尺 "
                          f"= +2/{group} B/值；违反前缀确定性距离 = {group - 1} 个位置"),
        },
        "b_violation_distance_positions": group_violation_distance(group),
        "b_staging": staging,
        "b_group_ladder": ladder,
        "quantized_arms": sorted(quant_arms),
        "mixed_axis_note": ("第三种（更省的）口径想同时要因果与便宜：标尺沿非 token 轴共享。"
                            "V 按 token 取 head_dim 维 amax ⇒ 位置 t 的标尺只由 t 决定（严格因果、"
                            "+2/32 B/值）；K 侧做不到，只能分块。该设计未测、未授权，本表不采用。"),
        "per_arm": arms,
    }
