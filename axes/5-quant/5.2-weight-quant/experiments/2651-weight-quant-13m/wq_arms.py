"""2651 · 权重量化 WP1 —— 权重假量化的**语义层**（本实验唯一的新逻辑）。

契约：同目录 `README.md` + `config.json`（`criterion` **运行前**写入）。
本模块只做一件事：把「权重与标尺方案」这一个因素做成可施加、可还原、可记账的臂。

## 臂的唯一改动因素

**只改权重矩阵的存储精度与标尺方案**；KV / 激活 / 偏置 / 归一化 / tied 读出**逐位不动**，
不训练、不新增参数（`params_added = 0`）。

| 臂 | `bits` | `scheme` | `group` | 每参数字节（值 + 标尺） |
|---|---|---|---|---|
| `fp16` | 16 | `identity` | — | 2 |
| `int8_per_channel` | 8 | `per_channel` | — | `1 + 2·Σn_out/Σ(n_out·n_in)` |
| `int8_group32` | 8 | `group` | 32 | `1 + 2/32` |
| `int4_group32` | 4 | `group` | 32 | `0.5 + 2/32` |

- **对称、零点固定为 0**（与 `tools/evalkit/kv_quant.py` 同口径）：`qmax = 2^(bits−1) − 1`，
  `scale = amax / qmax`，`q = clamp(round(w/scale), −qmax, qmax)`，`ŵ = q · scale`。
- **标尺是 fp16 存储**（`SCALE_BYTES = 2.0`，与 `tools/evalkit/kv_quant.SCALE_BYTES` 同口径）。
- **假量化往返**：`w → ŵ` 只在**前向**发生（实例级 `forward` 覆盖，见 `arm_context`）；
  `module.weight` 张量本身**逐位不变** ⇒ 臂可逐位还原。

## 为什么 `group` 沿**收缩维**（`in_features`）分组

权重矩阵 `[n_out, n_in]` 的收缩维是 `n_in`（`F.linear(x, W)` 里被求和的轴）。
沿收缩维每 `G` 个相邻值共享一把标尺 ⇒ 该行内部的分辨率提高，且**标尺个数与推理批/序列无关**
（权重量化没有 KV 量化那种「前缀确定」的因果约束——本项不涉及 KV 时序）。

## 字节账与断言

`byte_account()` 逐矩阵给出 `{name, shape, n_params, n_scales, value_bytes, scale_bytes,
total_bytes, bytes_per_param}`，并断言标尺个数与实现一一对应：

- `per_channel`：标尺个数 = `n_out`（每输出通道一把）；
- `group G`：标尺个数 = `n_out · (n_in / G)`，且必须 `G | n_in`（否则块不整齐，报错）。

## 死权重声明（`gates.*`）

`DepthARModel` 在 `carry_gate=True` 时建 `gates`（12 个 `nn.Linear(256,256)`，786 432 个参数），
但 `dense=True` + `exit_depths=None` 的前向**从不读它们**（门只在非 dense 的变深度路径上用）。
`dead_gate_probe()` 用「把门权重与偏置清零后 logits 是否逐位不变」证明这一点。
因此本模块给出两个权重面口径：

- `include_dead_gates=True`（**主口径**，本实验的臂 = 模型里**全部** `nn.Linear` 权重矩阵）；
- `include_dead_gates=False`（旁列，48 个活矩阵）——两者只差字节账里的 ≤0.06%，
  但必须分开报，否则「量化了一份死权重」会被误读成「干预有效」。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: 每把标尺的存储字节数（fp16 标尺）。与 `tools/evalkit/kv_quant.SCALE_BYTES` 同口径。
SCALE_BYTES = 2.0

#: 对称量化上界 `qmax = 2^(bits−1) − 1`（零点固定 0）。
QMAX = {8: 127, 4: 7}

#: 本实验主臂的分组大小。理由：① 与 `5.1` 块标尺主臂 `G=32` 同口径；② 整除本模型每个被量化
#: 矩阵的 `in_features ∈ {256, 1536}`；③ 与块量化（每块 32 个连续值共享一把 scale）的业界块长一致。
PRIMARY_GROUP = 32

#: 解析用的分组阶梯（只进字节账，不新增质量臂；回答「int4 反占内存需要多小的 G」）。
GROUP_LADDER = (8, 16, 32, 64, 128, 256)

#: 在 `dense=True, exit_depths=None` 前向上从不被读的 `nn.Linear` 名字前缀（见模块文档与
#: `dead_gate_probe`）。用前缀而不是硬编码名字，因为每个 block 都有一个 `gates.k`。
DEAD_LINEAR_PREFIXES = ("gates.",)

#: `scale` 的下限，避免全零行的 0/0（全零行的量化结果为 0，误差 0）。
_SCALE_FLOOR = 1e-12


@dataclass(frozen=True)
class ArmSpec:
    """一个权重量化臂：唯一改动因素 = `(bits, scheme, group)`。"""

    name: str
    bits: int
    scheme: str              # "identity" | "per_channel" | "group"
    group: int | None = None
    role: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "bits": self.bits, "scheme": self.scheme,
                "group": self.group, "role": self.role}


#: 四个臂（契约 `config.json::arms` 的唯一权威在这里；`config.json` 的 `arms` 是它的副本）。
ARM_SPECS: dict[str, ArmSpec] = {
    "fp16": ArmSpec("fp16", 16, "identity",
                    role="基线：权重 fp16 存储、前向逐位 = stock 释放版（锚 1.483569）"),
    "int8_per_channel": ArmSpec("int8_per_channel", 8, "per_channel",
                                role="每输出通道一把对称 scale（标尺最省的一档）"),
    "int8_group32": ArmSpec("int8_group32", 8, "group", PRIMARY_GROUP,
                            role="沿收缩维每 32 个相邻值一把 scale（比 per_channel 细、字节多）"),
    "int4_group32": ArmSpec("int4_group32", 4, "group", PRIMARY_GROUP,
                            role="4 位 + 每 32 个值一把 scale（本表字节最小的一档）"),
}


def arm_spec(name: str) -> ArmSpec:
    return ARM_SPECS[name]


# ----------------------------------------------------------------- 量化算子


def qmax_of(bits: int) -> int:
    if bits not in QMAX:
        raise ValueError(f"只定义 int8/int4 的对称上界，收到 bits={bits}（QMAX={QMAX}）")
    return QMAX[bits]


def scale_count(n_out: int, n_in: int, scheme: str, group: int | None = None) -> int:
    """标尺个数（**与实现一一对应**：`per_channel` 每输出通道一把；`group G` 每 G 个值一把）。"""
    if scheme == "identity":
        return 0
    if scheme == "per_channel":
        return n_out
    if scheme == "group":
        if not group or group < 1:
            raise ValueError(f"group 方案要求 group >= 1，收到 {group!r}")
        if n_in % group:
            raise ValueError(f"分组标尺要求 group 整除 in_features={n_in}，收到 group={group}")
        return n_out * (n_in // group)
    raise ValueError(f"未知标尺方案 {scheme!r}（期望 identity / per_channel / group）")


def fake_quantize_weight(w: torch.Tensor, bits: int, scheme: str,
                         group: int | None = None) -> tuple[torch.Tensor, dict]:
    """权重假量化往返 `W → q·s`（对称、零点 0）；返回 `(w_hat, stats)`。

    - `per_channel`：标尺 = 每输出通道（行）的 `amax/qmax`，形状 `[n_out, 1]`；
    - `group G`：`[n_out, n_in] → [n_out, n_in/G, G]`，最后一条轴取 `amax`；
    - 取整误差上界 `|ŵ − w| ≤ 0.5·s`（`clamp` 不触发额外 clip 误差，因为 `q` 由 `w/s` 取整得到，
      `|w/s| ≤ qmax` 恒成立）。
    """
    if scheme == "identity":
        return w.clone(), {"bits": bits, "scheme": scheme, "group": None,
                           "n_values": int(w.numel()), "n_scales": 0,
                           "changed_frac": 0.0, "mean_abs_err": 0.0, "max_abs_err": 0.0,
                           "rel_err": 0.0, "max_rel_err": 0.0, "scale_max": 0.0}
    n_out, n_in = int(w.shape[0]), int(w.shape[1])
    if w.dim() != 2:
        raise ValueError(f"只量化二维权重矩阵，收到 shape={tuple(w.shape)}")
    qmax = qmax_of(bits)
    if scheme == "per_channel":
        s = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(_SCALE_FLOOR)
        w_hat = torch.round(w / s).clamp(-qmax, qmax) * s
    elif scheme == "group":
        scale_count(n_out, n_in, scheme, group)     # 断言 G | n_in
        s = (w.reshape(n_out, n_in // group, group).abs().amax(dim=-1, keepdim=True)
             / qmax).clamp_min(_SCALE_FLOOR)
        ratio = w.reshape(n_out, n_in // group, group) / s
        w_hat = (torch.round(ratio).clamp(-qmax, qmax) * s).reshape(n_out, n_in)
    else:
        raise ValueError(f"未知标尺方案 {scheme!r}")
    diff = (w_hat - w).abs()
    denom = w.abs()
    # 逐元素相对误差会被**接近 0 的权重**放大（|w|→0 时 |Δ|/|w|→1）⇒ 它只作旁证；
    # 活性量的主数是下面的 L1 相对误差 `rel_err = Σ|Δ| / Σ|w|`。
    rel = diff / denom.clamp_min(denom.amax() * 1e-6)
    stats = {
        "bits": bits, "scheme": scheme, "group": group if scheme == "group" else None,
        "n_values": int(w.numel()),
        "n_scales": int(s.numel()),
        "changed_frac": round(float((w_hat != w).float().mean()), 8),
        "mean_abs_err": round(float(diff.mean()), 10),
        "max_abs_err": round(float(diff.max()), 10),
        "rel_err": round(float(diff.sum() / denom.sum().clamp_min(_SCALE_FLOOR)), 10),
        "max_rel_err": round(float(rel.max()), 8),
        "scale_max": round(float(s.max()), 10),
        "rounding_bound_ok": bool(float(diff.max()) <= 0.5 * float(s.max()) * (1 + 1e-6)),
    }
    return w_hat, stats


# ----------------------------------------------------------------- 权重面（哪些矩阵）


def is_dead_linear(name: str) -> bool:
    return any(name == p.rstrip(".") or name.startswith(p) for p in DEAD_LINEAR_PREFIXES)


def linear_weight_matrices(model, include_dead_gates: bool = True) -> list[tuple[str, object]]:
    """遍历 `model.named_modules()` 找 `nn.Linear`，按名字排序返回 `(name, module)`。

    `include_dead_gates=False` 时剔除 `DEAD_LINEAR_PREFIXES` 前缀下的死权重（48 个 vs 60 个）。
    """
    out = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    out.sort(key=lambda kv: kv[0])
    if not include_dead_gates:
        out = [(n, m) for n, m in out if not is_dead_linear(n)]
    if not out:
        raise ValueError("模型里没有 nn.Linear —— 权重面为空，臂无意义")
    return out


@torch.no_grad()
def dead_gate_probe(model, tokens: torch.Tensor) -> dict:
    """证明「把 `DEAD_LINEAR_PREFIXES` 下的权重与偏置清零 ⇒ logits 逐位不变」。

    这是 excluded-from-quality 的证据：这些矩阵只进字节账，不进质量读数。
    """
    from tools.evalkit.retention import masked_logits

    base = masked_logits(model, tokens, None)
    saved = []
    for name, mod in linear_weight_matrices(model, include_dead_gates=True):
        if not is_dead_linear(name):
            continue
        saved.append((mod, mod.weight.detach().clone(),
                      None if mod.bias is None else mod.bias.detach().clone()))
        with torch.no_grad():
            mod.weight.zero_()
            if mod.bias is not None:
                mod.bias.zero_()
    try:
        zeroed = masked_logits(model, tokens, None)
    finally:
        for mod, w, b in saved:
            with torch.no_grad():
                mod.weight.copy_(w)
                if mod.bias is not None:
                    mod.bias.copy_(b)
    return {
        "dead_prefixes": list(DEAD_LINEAR_PREFIXES),
        "n_dead_modules": len(saved),
        "dead_param_count": int(sum(w.numel() + (0 if b is None else b.numel())
                                    for _, w, b in saved)),
        "zeroed_logits_bitwise_equal": bool(torch.equal(base, zeroed)),
        "restored_bitwise": bool(all(torch.equal(mod.weight.detach(), w)
                                     for mod, w, _ in saved)),
    }


# ----------------------------------------------------------------- 施加 / 还原


def apply_arm(model, spec: ArmSpec, include_dead_gates: bool = True):
    """把臂的假量化前向装上（实例级 `forward` 覆盖），返回 `restore()`。

    `scheme="identity"`（fp16 臂）**不装任何东西** ⇒ 前向逐位 = stock 释放版。
    `restore()` 逐块删除实例属性 ⇒ 回到类的 `forward`（可断言逐位还原）。
    """
    if spec.scheme == "identity":
        return lambda: None
    patched: list = []
    for name, mod in linear_weight_matrices(model, include_dead_gates):
        w_hat, _ = fake_quantize_weight(mod.weight.detach(), spec.bits, spec.scheme, spec.group)
        bias = mod.bias

        def fwd(x, _w=w_hat, _b=bias):
            return F.linear(x, _w, _b)

        mod.forward = fwd
        patched.append(mod)

    def restore():
        for mod in patched:
            mod.__dict__.pop("forward", None)

    return restore


@contextlib.contextmanager
def arm_context(model, spec: ArmSpec, include_dead_gates: bool = True):
    """`with arm_context(model, spec): ...` —— 退出时保证逐位还原（异常路径也还原）。"""
    restore = apply_arm(model, spec, include_dead_gates)
    try:
        yield
    finally:
        restore()


# ----------------------------------------------------------------- 字节账


def matrix_bytes(n_out: int, n_in: int, bits: int, scheme: str,
                 group: int | None = None) -> dict:
    """一个权重矩阵的部署字节（值 + 标尺）。标尺个数由 `scale_count` 一一对应地给出。"""
    n_params = n_out * n_in
    n_scales = scale_count(n_out, n_in, scheme, group)
    value_bytes = n_params * bits / 8.0
    scale_bytes = n_scales * SCALE_BYTES
    total = value_bytes + scale_bytes
    return {"shape": [n_out, n_in], "n_params": n_params, "n_scales": n_scales,
            "value_bytes": value_bytes, "scale_bytes": scale_bytes,
            "total_bytes": total, "bytes_per_param": total / n_params,
            "scale_bytes_per_param": scale_bytes / n_params}


def byte_account(model, spec: ArmSpec, include_dead_gates: bool = True) -> dict:
    """逐矩阵的部署字节账 + 三个口径的总账 + 分组阶梯（解析）。

    - `per_matrix`：`{name, shape, n_params, n_scales, value_bytes, scale_bytes, total_bytes,
      bytes_per_param}`（判据要求「逐矩阵列出」）；
    - `scope_linear_all` / `scope_linear_live`：按参数数加权的每参数字节（横轴主口径）；
    - `scope_whole_model`：把**未量化**的偏置 / 归一化 / embedding / pos_embed 按 fp16 计进分母
      （诚实旁列：它回答「整个模型平均下来省了多少」，不是本实验的横轴）。
    """
    rows = []
    for name, mod in linear_weight_matrices(model, include_dead_gates):
        n_out, n_in = int(mod.weight.shape[0]), int(mod.weight.shape[1])
        rows.append({"name": name,
                     **matrix_bytes(n_out, n_in, spec.bits, spec.scheme, spec.group)})
    live_rows = [r for r in rows
                 if not is_dead_linear(r["name"])]

    def _agg(rs):
        n_params = sum(r["n_params"] for r in rs)
        total = sum(r["total_bytes"] for r in rs)
        n_scales = sum(r["n_scales"] for r in rs)
        return {"n_matrices": len(rs), "n_params": n_params, "n_scales": n_scales,
                "value_bytes": sum(r["value_bytes"] for r in rs),
                "scale_bytes": sum(r["scale_bytes"] for r in rs),
                "total_bytes": total,
                "bytes_per_param": (total / n_params) if n_params else None,
                "to_fp16_multiple": (total / n_params / 2.0) if n_params else None,
                "reduction_vs_fp16": (1.0 - total / n_params / 2.0) if n_params else None}

    linear_all = _agg(rows)
    linear_live = _agg(live_rows)

    total_model_params = int(sum(p.numel() for p in model.parameters()))
    other_params = total_model_params - linear_all["n_params"]
    whole_total = linear_all["total_bytes"] + other_params * 2.0
    whole = {"n_params": total_model_params, "n_params_quantized": linear_all["n_params"],
             "n_params_untouched": other_params,
             "bytes_per_param": whole_total / total_model_params,
             "to_fp16_multiple": whole_total / total_model_params / 2.0,
             "reduction_vs_fp16": 1.0 - whole_total / total_model_params / 2.0}
    if spec.scheme == "identity":
        # fp16 臂：整个模型都是 2 B/参数，三个口径必须一致。
        assert abs(linear_all["bytes_per_param"] - 2.0) < 1e-12, "fp16 臂线性面应为 2 B/参数"
        assert abs(whole["bytes_per_param"] - 2.0) < 1e-12, "fp16 臂全面应为 2 B/参数"
    return {"arm": spec.name, "bits": spec.bits, "scheme": spec.scheme, "group": spec.group,
            "scale_bytes": SCALE_BYTES, "include_dead_gates": include_dead_gates,
            "per_matrix": rows,
            "scope_linear_all": linear_all,
            "scope_linear_live": linear_live,
            "scope_whole_model": whole,
            "ladder": group_ladder_rows(linear_weight_matrices(model, include_dead_gates))}


def group_ladder_rows(matrices) -> list[dict]:
    """解析阶梯：只回答「多小的 G 会让 int4/int8 的标尺开销把每参数字节顶到 fp16 之上」。"""
    out = [{"scheme": "fp16", "bits": 16, "group": None, "bytes_per_param": 2.0,
            "to_fp16_multiple": 1.0}]
    for scheme in ("per_channel", "group"):
        groups = [None] if scheme == "per_channel" else list(GROUP_LADDER)
        for g in groups:
            n_params = 0
            total = 0.0
            n_scales = 0
            for _, mod in matrices:
                n_out, n_in = int(mod.weight.shape[0]), int(mod.weight.shape[1])
                if scheme == "group" and n_in % g:
                    n_params = total = None      # 该 G 不能整除这个矩阵 ⇒ 无效行
                    break
                mb = matrix_bytes(n_out, n_in, 8, scheme, g)
                n_params += mb["n_params"]
                total += mb["total_bytes"]
                n_scales += mb["n_scales"]
            if n_params is None:
                out.append({"scheme": scheme, "bits": 8, "group": g,
                            "bytes_per_param": None, "to_fp16_multiple": None,
                            "note": f"G={g} 不能整除所有 in_features ⇒ 该实现无效"})
                continue
            bpp = total / n_params
            out.append({"scheme": scheme, "bits": 8, "group": g,
                        "n_scales": n_scales, "bytes_per_param": bpp,
                        "to_fp16_multiple": bpp / 2.0,
                        "int4_bytes_per_param": (n_params * 0.5 + n_scales * SCALE_BYTES) / n_params,
                        "int4_reduction_vs_fp16": 1.0 - (n_params * 0.5 + n_scales * SCALE_BYTES)
                        / n_params / 2.0})
    return out


# ----------------------------------------------------------------- 活性量


def activity_of(model, spec: ArmSpec, include_dead_gates: bool = True) -> dict:
    """SCHEMA §10 活性量：本实验无训练 ⇒ 报**推理期权重是否真被改**。

    聚合 + 逐矩阵两套（判据要求「每臂最大相对误差」）：`changed_frac`（被改的元素比例）、
    `rel_err`（L1 相对误差）、`max_abs_err`、`max_rel_err`、标尺个数对账。
    """
    per_matrix = []
    agg = {"n_values": 0, "n_scales": 0, "n_changed": 0, "sum_abs_err": 0.0,
           "sum_abs_w": 0.0, "max_abs_err": 0.0, "max_rel_err": 0.0,
           "scale_max": 0.0, "bound_failures": []}
    for name, mod in linear_weight_matrices(model, include_dead_gates):
        w = mod.weight.detach()
        w_hat, st = fake_quantize_weight(w, spec.bits, spec.scheme, spec.group)
        expect = scale_count(int(w.shape[0]), int(w.shape[1]), spec.scheme, spec.group)
        assert st["n_scales"] == expect, (
            f"{name}: 标尺个数 {st['n_scales']} != 解析式 {expect}（字节账与实现不一致）")
        per_matrix.append({"name": name, "shape": list(w.shape), **st,
                           "expected_n_scales": expect})
        agg["n_values"] += st["n_values"]
        agg["n_scales"] += st["n_scales"]
        agg["n_changed"] += int(round(st["changed_frac"] * st["n_values"]))
        agg["sum_abs_err"] += st["mean_abs_err"] * st["n_values"]
        agg["sum_abs_w"] += float(w.abs().sum())
        agg["max_abs_err"] = max(agg["max_abs_err"], st["max_abs_err"])
        agg["max_rel_err"] = max(agg["max_rel_err"], st["max_rel_err"])
        agg["scale_max"] = max(agg["scale_max"], st["scale_max"])
        if spec.scheme != "identity" and not st.get("rounding_bound_ok", True):
            agg["bound_failures"].append(name)
    agg["changed_frac"] = (round(agg["n_changed"] / agg["n_values"], 8)
                          if agg["n_values"] else 0.0)
    agg["rel_err"] = round(agg["sum_abs_err"] / max(agg["sum_abs_w"], _SCALE_FLOOR), 10)
    # 「每臂最大相对误差」的**良定义**版本：逐矩阵的 L1 相对误差 `Σ|Δ|/Σ|w|` 取最大。
    # 上面的 `max_rel_err` 是**逐元素**相对误差，会被接近 0 的权重放大到 ≈1 ⇒ 只作旁证。
    agg["max_rel_err_per_matrix_l1"] = (round(max(pm["rel_err"] for pm in per_matrix), 8)
                                        if per_matrix else 0.0)
    agg["max_abs_err_per_matrix"] = (round(max(pm["max_abs_err"] for pm in per_matrix), 10)
                                     if per_matrix else 0.0)
    agg["mean_abs_err"] = round(agg["sum_abs_err"] / agg["n_values"], 10) if agg["n_values"] else 0.0
    for k in ("sum_abs_err", "sum_abs_w"):
        agg.pop(k)
    return {"arm": spec.name, "bits": spec.bits, "scheme": spec.scheme, "group": spec.group,
            "n_matrices": len(per_matrix), **agg, "per_matrix": per_matrix,
            "scales_match_analytic": True}
