"""2632 · KV 量化标尺口径的**部署字节账**（值 + 标尺 + 非因果方案的 staging）——纯解析。

契约：同目录 `config.json`（判据线，**运行前**写入）与 `README.md`。

**为什么需要这个模块。** `2631` 把 `2621` 的横轴（只算**值**字节）保留成对照，另立了
「部署口径」两列，但**(a) 逐值标尺**是当时唯一**实现了**的方案、**(b) 分块标尺**只有解析。
本模块把标尺方案讲清到**每值字节**这一层，并与 `tools/evalkit/kv_quant.py` 的
`quantize_kv(scheme=...)` **一一对应**：

- `per_value`：位置 t 的标尺 = 同请求 ≤ t 的 running max / qmax ⇒ 每值 +2 B（对照臂）。
- `block`：沿 `head_dim` 每 `G` 个相邻值共享一把标尺 ⇒ 每值 +2/G B（需 `G | head_dim`）；
  **严格前缀确定**（标尺只由当前 token 自己的值决定）。`G = head_dim` ⇒ per-token 标尺。
- `block_tok`：沿 **token** 轴每 `G` 个相邻位置共享一把标尺 ⇒ 每值 +2/G B；
  块内过去值会被同块后续位置改写（违反前缀确定 ≤ `G−1` 个位置）。
- `kivi`：K per-channel（token 轴每 `kivi_group` 个位置一把通道标尺，+2/kivi_group B）+ V
  per-token（每 token 沿 `head_dim` 一把，+2/head_dim B）；K 侧违反前缀确定 ≤ `kivi_group−1`。

**值字节必须与 `results.json`（判据 C1 的旧横轴）逐点对账**：对不上直接报错（防口径漂移）。
非因果方案要额外记账 **staging**（回填必须留住原始 fp16 值：每请求 `G` 个位置的 fp16 KV，
与 T 无关），本模块单列，不进每位置列。
"""

from __future__ import annotations

from tools.evalkit.kv_quant import (SCALE_BYTES, SCALE_SCHEMES,
                                    scheme_scale_bytes_per_value)

SCHEMES = SCALE_SCHEMES
DEFAULT_GROUP = 32
DEFAULT_KIVI_GROUP = 32
#: 沿 head_dim 的块阶梯（必须整除 head_dim = 32；G=64 做不到前缀确定，见 config caveats）。
HEAD_DIM_LADDER = (8, 16, 32)
#: 沿 token 轴的块阶梯（非因果；= 2631 方案 (b) 的 G 阶梯）。
TOKEN_LADDER = (16, 32, 64)

VALUE_BYTES = {"fp32": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}
#: 需要标尺的量化档。
_QUANT_MODES = ("int8", "int4")
#: 非因果方案（需帧内回填 ⇒ 需 staging）。
_NONCAUSAL = ("block_tok", "kivi")


def mode_of(spec: dict) -> str:
    """臂的存储精度，并校验 `mode` 与 `value_bytes` 标签一致（不一致直接报错）。"""
    vb = float(spec["value_bytes"])
    if "mode" in spec:
        mode = spec["mode"]
        if mode not in VALUE_BYTES:
            raise ValueError(f"未知 mode={mode!r}")
        if abs(vb - VALUE_BYTES[mode]) > 1e-9:
            raise ValueError(
                f"臂 {spec.get('role', '?')!r}: value_bytes={vb} 与 mode={mode}"
                f"（应 {VALUE_BYTES[mode]}）不一致 ⇒ 标签与事实不符，拒绝出数")
        return mode
    for mode, want in VALUE_BYTES.items():
        if abs(vb - want) < 1e-9:
            return mode
    raise ValueError(f"value_bytes={vb} 不对应任何已知精度 {sorted(VALUE_BYTES)}")


def needs_scale(mode: str) -> bool:
    return mode in _QUANT_MODES


def _factors(point: dict) -> tuple[int, float]:
    for key in ("factor_n_kv_heads_eff", "factor_positions_kept_over_T"):
        if key not in point:
            raise KeyError(f"点 {point.get('arm')!r} 缺 {key}：无法复用判据 C1 的横轴公式")
    return int(point["factor_n_kv_heads_eff"]), float(point["factor_positions_kept_over_T"])


def deployment_bytes(point: dict, spec: dict, cfg: dict, scheme: str,
                     group: int = DEFAULT_GROUP, kivi_group: int = DEFAULT_KIVI_GROUP) -> dict:
    """一个臂在某标尺方案下的**真实**每位置字节（值 + 标尺），与旧值字节口径对账。

    `point["bytes_per_position"]` 是旧的值字节口径（2631 的横轴），本模块算出的值字节
    必须与之逐位一致（对不上 ⇒ 报错，防口径漂移）。
    """
    if scheme not in SCHEMES:
        raise ValueError(f"未知方案 {scheme!r}；已知 {SCHEMES}")
    mode = mode_of(spec)
    nkv, kept_over_T = _factors(point)
    n_layers = cfg["model"]["n_layers"]
    hd = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    # 每位置标量数（K + V）：2 · L · n_kv_heads_eff · head_dim · kept/T。
    values_per_position = 2.0 * n_layers * nkv * hd * kept_over_T
    bpe = VALUE_BYTES[mode]
    value_part = values_per_position * bpe
    if abs(value_part - float(point["bytes_per_position"])) > 1e-6:
        raise ValueError(
            f"臂 {point.get('arm')!r}：本模块算的值字节 {value_part} 与该点已记录的 "
            f"bytes_per_position = {point['bytes_per_position']} 不等（判据 C1 对账失败）")
    scale_bpe = (scheme_scale_bytes_per_value(mode, scheme, hd, group, kivi_group)
                 if needs_scale(mode) else 0.0)
    scale_part = values_per_position * scale_bpe
    return {"scheme": scheme, "mode": mode,
            "group": group if scheme in ("block", "block_tok") else 1,
            "kivi_group": kivi_group if scheme == "kivi" else None,
            "head_dim": hd,
            "needs_scale": needs_scale(mode),
            "noncausal": scheme in _NONCAUSAL and needs_scale(mode),
            "values_per_position": values_per_position,
            "value_bytes_per_value": bpe,
            "scale_bytes_per_value": scale_bpe,
            "value_bytes_per_position": value_part,
            "scale_bytes_per_position": scale_part,
            "bytes_per_position": value_part + scale_part,
            "ratio_vs_value_caliber": (value_part + scale_part) / value_part}


def staging_bytes_per_request(cfg: dict, point: dict, group: int) -> dict:
    """非因果方案的回填 staging：每请求留住最近 `group` 个位置的 fp16 KV（与 T 无关）。"""
    nkv, _ = _factors(point)
    hd = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    vals_per_position = 2.0 * cfg["model"]["n_layers"] * nkv * hd
    T = cfg["model"]["seq_len"]
    b = group * vals_per_position * VALUE_BYTES["fp16"]
    return {"group": group, "positions_held": group, "bytes_per_request": b,
            "bytes_per_position_at_T": b / T, "T": T,
            "note": "fp16 原值帧内缓冲（回填用）；与目标精度无关，int4 也要留"}


def caliber_block(cfg: dict, specs: dict, points: list[dict]) -> dict:
    """逐臂 × 逐方案的部署字节 + head_dim 块阶梯 + kivi / block_tok 阶梯 + staging。"""
    by_arm = {p["arm"]: p for p in points}
    fp16 = by_arm.get("fp16")
    per_arm: dict = {}
    for arm, p in by_arm.items():
        spec = specs[arm]
        per_arm[arm] = {
            "primary": deployment_bytes(p, spec, cfg, "block", DEFAULT_GROUP),
            "by_scheme": {},
        }
        for scheme in SCHEMES:
            if scheme == "block_tok":
                per_arm[arm]["by_scheme"][scheme] = deployment_bytes(p, spec, cfg, scheme, 32)
            elif scheme == "kivi":
                per_arm[arm]["by_scheme"][scheme] = deployment_bytes(
                    p, spec, cfg, scheme, DEFAULT_GROUP, DEFAULT_KIVI_GROUP)
            else:
                per_arm[arm]["by_scheme"][scheme] = deployment_bytes(p, spec, cfg, scheme,
                                                                     DEFAULT_GROUP)
    head_ladder = []
    for g in HEAD_DIM_LADDER:
        head_ladder.append({
            "group": g, "axis": "head_dim", "prefix_invariant": True,
            "scale_bytes_per_value": SCALE_BYTES / g,
            "int8_bytes_per_value": VALUE_BYTES["int8"] + SCALE_BYTES / g,
            "int4_bytes_per_value": VALUE_BYTES["int4"] + SCALE_BYTES / g,
            "int8_reduction_vs_fp16": 1 - (VALUE_BYTES["int8"] + SCALE_BYTES / g) / 2.0,
            "int4_reduction_vs_fp16": 1 - (VALUE_BYTES["int4"] + SCALE_BYTES / g) / 2.0})
    token_ladder = []
    for g in TOKEN_LADDER:
        token_ladder.append({
            "group": g, "axis": "token", "prefix_invariant": False,
            "violation_distance_positions": g - 1,
            "scale_bytes_per_value": SCALE_BYTES / g,
            "int8_bytes_per_value": VALUE_BYTES["int8"] + SCALE_BYTES / g,
            "int4_bytes_per_value": VALUE_BYTES["int4"] + SCALE_BYTES / g,
            "int8_reduction_vs_fp16": 1 - (VALUE_BYTES["int8"] + SCALE_BYTES / g) / 2.0,
            "int4_reduction_vs_fp16": 1 - (VALUE_BYTES["int4"] + SCALE_BYTES / g) / 2.0})
    staging = (staging_bytes_per_request(cfg, fp16, DEFAULT_GROUP)
               if fp16 is not None else None)
    return {
        "value_bytes": VALUE_BYTES, "scale_bytes": SCALE_BYTES,
        "group_default": DEFAULT_GROUP, "kivi_group_default": DEFAULT_KIVI_GROUP,
        "head_dim_ladder": head_ladder, "token_ladder": token_ladder,
        "token_ladder_staging": staging,
        "head_dim_g64_note": ("head_dim = 32 ⇒ 沿 head_dim 的块只能整除到 G = 32；G = 64 要求"
                              "块跨 token，无法同时保持前缀确定。非因果的 token 轴块见 token_ladder。"),
        "per_arm": per_arm,
    }


def quantization_factor(point: dict, spec: dict, cfg: dict, ref_point: dict,
                        ref_spec: dict, scheme: str, group: int = DEFAULT_GROUP,
                        kivi_group: int = DEFAULT_KIVI_GROUP) -> float:
    """在同一拓扑上再叠量化后的字节倍率（<1 = 省）；`ref` = 同拓扑不量化臂。"""
    a = deployment_bytes(point, spec, cfg, scheme, group, kivi_group)["bytes_per_position"]
    b = deployment_bytes(ref_point, ref_spec, cfg, scheme, group, kivi_group)
    # 不量化臂没有标尺 ⇒ b 的值字节 = 部署字节。
    return a / b["bytes_per_position"]
