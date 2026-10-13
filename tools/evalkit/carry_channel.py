"""契约 U 的评测仪器：三臂（U0/U1/U2）在合成长程任务电池上的对照读数。

契约：`axes/1-time/1.1.4.1-explicit-memory/experiments/2618-carry-channel-13m/README.md`
（判据线逐字）。任务与判分复用 `tools/evalkit/longrange_tasks.py`（只增不改）；
本模块只加**对照与统计**这一层：

- `wilson_ci` / `two_proportion`：二项 95% CI 与两比例检验（判据 2/3 的「CI 不重叠」）；
- `arm_battery`：一臂在给定任务批上的准确率 + 距离曲线 + **毁源对照臂**（判据 4）；
- `param_equality` / `trunk_equality`：SCHEMA §15 的**逐位等参数**硬检查（运行期断言）；
- `analytic_memory`：解析显存（KV / 状态 / 参数）与实测峰值分开报；
- `loss_slope`：三臂**同方法**的训练 loss 斜率（SCHEMA §16 收敛对账）。

**可见性口径**：训练与评测必须用同一掩码（契约表里写明「训练+评测一致」）——
评测统一走 `window_anchor_keep_mask(T, train_window, train_anchors)`，不传掩码会
变成全因果（那就把源信息白白送给模型，判据 1/2 都失效）。
"""

from __future__ import annotations

import math

import torch

from tools.evalkit.longrange_tasks import (
    TaskBatch, accuracy, corrupt_span, local_prior_accuracy)
from pathlm.depth_ar import window_anchor_keep_mask


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score 二项 95% CI（小 p 下比正态近似稳）。"""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def two_proportion(k_a: int, n_a: int, k_b: int, n_b: int) -> dict:
    """两比例检验（pooled z 检验）+ 差值的 95% CI（Newcombe 独立比例）。

    判据线用的是「二项 95% CI 不重叠」，这里把两件事都算出来：
    `ci_overlap` = 两臂各自的 Wilson CI 是否重叠（判据线口径），
    `diff_ci` = 差值本身的 CI（更严格的同口径比较，作为附加证据）。
    """
    if n_a <= 0 or n_b <= 0:
        return {}
    pa, pb = k_a / n_a, k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        z, p_value = 0.0, 1.0
    else:
        z = (pb - pa) / se
        p_value = math.erfc(abs(z) / math.sqrt(2))
    ci_a, ci_b = wilson_ci(k_a, n_a), wilson_ci(k_b, n_b)
    diff = pb - pa
    se_d = math.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    return {"p_a": round(pa, 6), "p_b": round(pb, 6),
            "delta_pp": round(100 * diff, 3),
            "ci_a": ci_a, "ci_b": ci_b,
            "ci_overlap": not (ci_b[0] > ci_a[1] or ci_a[0] > ci_b[1]),
            "diff_ci_95": [round(diff - 1.96 * se_d, 6),
                           round(diff + 1.96 * se_d, 6)],
            "z": round(z, 3), "p_value": round(p_value, 8)}


def arm_battery(model, task: TaskBatch, window: int, anchors: int = 0,
                device: str = "cuda", corrupt_span_bounds: tuple | None = None,
                max_chunk: int = 64) -> dict:
    """一臂在一个任务批上的三条读数：主臂 / 毁源对照臂 / 局部先验 + chance。

    `corrupt_span_bounds = (lo, hi)` 时额外跑**毁源对照臂**（判据 4）：把那一段
    源内容换成新随机符号，其余（查询行/答案/距离）逐位不变。主臂读数若来自那段
    源信息，毁源后必须掉回 chance。
    """
    keep = window_anchor_keep_mask(task.T, window, anchors)
    dev = device
    main = accuracy(model, task, keep_mask=keep, device=dev, max_chunk=max_chunk)
    out = {"task": task.name, "n": main["n"], "n_hits": main["n_hits"],
           "accuracy": main["accuracy"],
           "accuracy_ci": wilson_ci(main["n_hits"], main["n"]),
           "by_distance": main["by_distance"],
           "chance_uniform_answer_set": round(task.chance(), 8),
           "n_answers": task.n_answers,
           "local_prior": local_prior_accuracy(task, k=8)["accuracy"],
           "keep_mask": {"train_window": window, "train_anchors": anchors,
                         "n_visible_pairs": int(keep.sum().item()),
                         "causal_pairs": task.T * (task.T + 1) // 2}}
    if corrupt_span_bounds is not None:
        lo, hi = corrupt_span_bounds
        bad = corrupt_span(task, lo, hi,
                           generator=torch.Generator().manual_seed(0))
        c = accuracy(model, bad, keep_mask=keep, device=dev, max_chunk=max_chunk)
        out["corrupt_source"] = {"span": [lo, hi], "n": c["n"],
                                 "n_hits": c["n_hits"], "accuracy": c["accuracy"],
                                 "accuracy_ci": wilson_ci(c["n_hits"], c["n"]),
                                 "by_distance": c["by_distance"]}
    return out


def param_equality(model_a, model_b, prefix: str = "") -> dict:
    """SCHEMA §15：两臂参数**逐位相等**（名字集合、形状、取值三件都要过）。"""
    pa, pb = model_a.state_dict(), model_b.state_dict()
    missing = sorted(set(pa) ^ set(pb))
    shape_mismatch, value_mismatch = [], []
    for k in sorted(set(pa) & set(pb)):
        if pa[k].shape != pb[k].shape:
            shape_mismatch.append(k)
        elif not torch.equal(pa[k], pb[k]):
            value_mismatch.append(k)
    return {"n_tensors": len(pa), "name_symmetric_difference": missing,
            "shape_mismatch": shape_mismatch, "value_mismatch": value_mismatch,
            "bitwise_equal": not (missing or shape_mismatch or value_mismatch),
            "n_params_a": sum(p.numel() for p in model_a.parameters()),
            "n_params_b": sum(p.numel() for p in model_b.parameters())}


def trunk_equality(carry_model, base_model) -> dict:
    pc, pb = carry_model.state_dict(), base_model.state_dict()
    added = sorted(set(pc) - set(pb))
    missing = sorted(set(pb) - set(pc))
    diff = [k for k in sorted(set(pc) & set(pb)) if not torch.equal(pc[k], pb[k])]
    return {"added_tensors": added, "missing_tensors": missing,
            "trunk_value_mismatch": diff,
            "trunk_bitwise_equal": not (diff or missing),
            "n_params_added": sum(pc[k].numel() for k in added),
            "n_params_carry_arm": sum(p.numel() for p in carry_model.parameters()),
            "n_params_base_arm": sum(p.numel() for p in base_model.parameters())}


def analytic_memory(cfg, batch_size: int, window: int, anchors: int = 0) -> dict:
    """解析显存（与实测峰值分开报，契约判据 6 要求）。

    - 参数：`params × 4 B`（fp32 存储）+ AdamW 两个动量缓冲（×3）；
    - 训练激活：窗口注意力下每层每样本的注意力矩阵 `window×T`（不是 `T×T`）+
      残差流 `[B,T,d]` × 层数；
    - 通道状态：单共享槽 `H×dh×dh` + 每层读出增益（**与序列长度无关**，这是
      「有限状态」的解析证据）；
    - KV 缓存（推理口径）：`2·H·dh·T` 每层每样本。
    """
    d, L, H = cfg["d_model"], cfg["n_layers"], cfg["n_heads"]
    dh, T = d // H, cfg["seq_len"]
    n_params = cfg["n_params"]
    state_floats = H * dh * dh
    per_sample_act = L * (5 * T * d)          # 残差流/中间量粗估（5 个 [T,d] 量级）
    return {
        "params": n_params,
        "params_bytes_fp32": n_params * 4,
        "optimizer_bytes_adamw": n_params * 4 * 2,
        "train_act_floats_per_sample_est": per_sample_act,
        "train_act_bytes_bf16_batch": round(
            per_sample_act * batch_size * 2 / 2 ** 20, 3),
        "window_attn_scores_floats_per_layer": batch_size * H * window * T,
        "window_attn_scores_mb": round(
            batch_size * H * window * T * 4 / 2 ** 20, 3),
        "full_attn_scores_mb_for_contrast": round(
            batch_size * H * T * T * 4 / 2 ** 20, 3),
        "carry_state_floats": state_floats,
        "carry_state_bytes": state_floats * 4,
        "carry_state_note": ("单共享槽 H·dh·dh = %d floats/样本，**与 T 无关**"
                             % state_floats),
        "kv_cache_bytes_per_sample": 2 * L * H * dh * T * 4,
    }


def loss_slope(log_lines: list[dict], lo: int = 200, hi: int = 12000) -> dict:
    """**同方法**的训练 loss 斜率（SCHEMA §16）：末段线性拟合 log-loss 对 log-step。

    三臂都用同一配方（同任务/同窗掩码/同步数）⇒ 斜率可比；只用一个点或不同窗口
    的斜率不可比。
    """
    xs = [math.log(max(r["step"], 1)) for r in log_lines
          if lo <= r["step"] <= hi and r.get("loss") is not None]
    ys = [math.log(max(r["loss"], 1e-6)) for r in log_lines
          if lo <= r["step"] <= hi and r.get("loss") is not None]
    n = len(xs)
    if n < 3:
        return {"n_points": n, "slope": None}
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    slope = cov / var if var else 0.0
    return {"n_points": n, "slope": round(slope, 6),
            "first_loss": log_lines[0]["loss"], "last_loss": log_lines[-1]["loss"],
            "window_steps": [lo, hi]}
