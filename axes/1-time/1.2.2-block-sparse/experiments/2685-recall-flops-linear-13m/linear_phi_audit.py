"""2685 · φ 的数值健康审计（健康门 H3/H4 的落盘证据）：在**真实评测批**上量 φ 与分母。

三个量，逐层 + 汇总：

- `phi_zero_frac_{engine,stable}`：φ 恰为 0 的元素占比。`engine` = 引擎默认的 `F.elu(x)+1`
  （`x ≲ −17` 时 `1 − exp(x)` 在 fp32/bf16 里舍入回 1 ⇒ φ 下溢为 0）；`stable` =
  `stable_linear.stable_phi`（同一个函数的无消去求值）。
- `min_phi_stable`：稳定档的 φ 最小值（机制前提「φ > 0」的直接读数）。
- `den_below_eps_frac`：分母 `den = Σ_{j≤i} φ(q_i)·φ(k_j)` 低于声明 eps（`1e-30`）的 query-头行占比
  —— 分母 guard 生效的行 = 该行注意力输出被静默清零的行。

dtype 两档都量：`fp32`（评测前向实际用的）与 `bf16`（训练前向用的 autocast 精度）。训练期的
`den_min = 0` 是 bf16 里的乘积下溢，不是 fp32 评测前向的病，这个差别必须落盘，否则会把「训练期
telemetry 的最小值」误读成「推理期存在死行」。
"""

from __future__ import annotations

import torch

import stable_linear


def _phi_stats(q: torch.Tensor, k: torch.Tensor) -> dict:
    qe = torch.nn.functional.elu(q) + 1.0
    ke = torch.nn.functional.elu(k) + 1.0
    qs = stable_linear.stable_phi(q)
    ks = stable_linear.stable_phi(k)
    return {
        "phi_zero_frac_engine": round(float((qe == 0).float().mean()
                                            + (ke == 0).float().mean()) / 2, 9),
        "phi_zero_frac_stable": round(float((qs == 0).float().mean()
                                            + (ks == 0).float().mean()) / 2, 9),
        "min_phi_stable": float(min(qs.min(), ks.min())),
        "qk_abs_max": float(max(q.abs().max(), k.abs().max())),
    }


def _den_stats(q: torch.Tensor, k: torch.Tensor, eps: float) -> dict:
    """按线性核的定义重算分母（显式 `[B,H,T,T]` 分数，仅审计用）。"""
    qf = stable_linear.stable_phi(q).float()
    kf = stable_linear.stable_phi(k).float()
    T = qf.shape[-2]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=qf.device))
    scores = (qf @ kf.transpose(-2, -1)).masked_fill(~causal, 0.0)
    den = scores.sum(-1)
    return {"den_min": float(den.min()),
            "den_below_eps_frac": round(float((den < eps).float().mean()), 9)}


@torch.no_grad()
def audit(model, tokens: torch.Tensor, eps: float = stable_linear.LINEAR_ATTN_EPS_DECLARED
          ) -> dict:
    """逐层量 φ 与分母；`model` 必须是线性核臂（`cfg.attn_linear = True`）。"""
    model.eval()
    B, T = tokens.shape
    H = model.cfg.n_heads
    dh = model.cfg.d_model // H
    shape = lambda t: t.view(B, T, H, dh).transpose(1, 2)
    h = model._input_states(tokens)
    per_layer, agg = [], {"phi_zero_frac_engine": 0.0, "phi_zero_frac_stable": 0.0,
                          "den_below_eps_frac": 0.0, "den_min": float("inf"),
                          "min_phi_stable": float("inf"), "qk_abs_max": 0.0}
    for block in model.blocks:
        q, k, _ = block.qkv(block.ln1(h)).chunk(3, dim=-1)
        q32, k32 = shape(q).float(), shape(k).float()
        qb, kb = q32.to(torch.bfloat16), k32.to(torch.bfloat16)
        row = {"layer": len(per_layer),
               "fp32": {**_phi_stats(q32, k32), **_den_stats(q32, k32, eps)},
               "bf16": _phi_stats(qb, kb)}
        per_layer.append(row)
        agg["phi_zero_frac_engine"] += row["fp32"]["phi_zero_frac_engine"] / len(model.blocks)
        agg["phi_zero_frac_stable"] += row["fp32"]["phi_zero_frac_stable"] / len(model.blocks)
        agg["den_below_eps_frac"] += row["fp32"]["den_below_eps_frac"] / len(model.blocks)
        agg["den_min"] = min(agg["den_min"], row["fp32"]["den_min"])
        agg["min_phi_stable"] = min(agg["min_phi_stable"], row["fp32"]["min_phi_stable"])
        agg["qk_abs_max"] = max(agg["qk_abs_max"], row["fp32"]["qk_abs_max"])
        h, _ = block(h)
    agg["phi_zero_frac_max_any_layer"] = max(r["fp32"]["phi_zero_frac_stable"]
                                             for r in per_layer)
    agg["phi_zero_frac_engine_max_any_layer"] = max(r["fp32"]["phi_zero_frac_engine"]
                                                    for r in per_layer)
    agg["thresholds"] = {"phi_zero_fraction_max": stable_linear.PHI_ZERO_FRACTION_MAX,
                         "linear_attn_eps_declared": eps}
    return {"batch": int(B), "seq_len": int(T), "layers": len(per_layer),
            "aggregate": agg, "per_layer": per_layer,
            "note": ("fp32 = 评测前向实际精度；bf16 = 训练前向的 autocast 精度。φ 零占比与"
                     "`den < eps` 的行占比是健康门 H3/H4 的落盘量（README §3）。")}
