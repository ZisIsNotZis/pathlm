"""2631 · 因果口径 KV 预算图 —— 本批新逻辑的唯一语义层。

契约：同目录 `README.md`（判据线**预登记**，§4）。

本模块只加两件在 `2619` / `2620` / `2621` 里没有的东西（SCHEMA §22 第 7 条：新逻辑进
新模块，不改 `2619/run.py`、`2620/gaps_arms.py`、`2621/rest_arms.py`、`tools/evalkit/*`）：

1. `prefix_invariance_of_logits(logits_fn, tokens)` / `prefix_invariance_check(model, tokens, mode)`
   把复核 A2 的两条反例做成**可失败的运行时仪器**：
   - 改**未来** token，不改**过去位置**的 logits；
   - 改**同批其他请求**，不改**本请求**的 logits。
   这是任务预登记判据 ① 的直接证据。旧的「沿 batch+token 取全序列 amax」实现在这两条上
   都会红（`2614/mutate.py` 的 M2 就是把生产实现退回该缺陷、由单测抓红）。

2. `compare_causal_vs_noncausal(old_points, new_points, quantized_arms)`
   把 `2621` 的旧（非因果）点表与本次因果点表**并列**，逐臂给 Δbpc = bpc_新 − bpc_旧
   与方向；并核对「非量化臂必须逐位不变、量化臂允许变」——这是「本轮唯一改动的因子
   就是量化器口径」的可控性证据（SCHEMA §9/§15）。

单测（**在本目录内**，不写 `tests/`：本波 `tests/**` 归 L1 线）：`test_causal_check.py`。
变异验证（破坏上述语义 → 单测抓红 → 逐字节还原）：`mutate.py` + `mutation_report.json`。
"""

from __future__ import annotations

import torch

from tools.evalkit.kv_quant import kv_quant_logits

#: bpc 的「逐位不变」容差。纯推理、无随机采样 ⇒ 非量化臂应**逐位**相同（1e-9 已是宽松上界）。
BITWISE_TOL = 1e-6


# --------------------------------------------------------- ① 前缀不变性仪器


@torch.no_grad()
def prefix_invariance_of_logits(logits_fn, tokens: torch.Tensor) -> dict:
    """对「tokens → logits」的任意一条前向，检验两条前缀不变性（复核 A2）。

    - **未来 token**：只改第 0 个请求的最后一个 token ⇒ 第 0 个请求**过去位置**的 logits
      必须逐位不变；
    - **同批邻居**：只改同批第 1..B-1 个请求 ⇒ 第 0 个请求的全部 logits 必须逐位不变。

    返回三个 bool（供 `results.json::instrument` 留证）。批 ≥ 2，长度 ≥ 2。
    """
    if tokens.dim() != 2 or tokens.shape[0] < 2 or tokens.shape[1] < 2:
        raise ValueError(f"需要 [B>=2, T>=2] 的 token 批，收到 {tuple(tokens.shape)}")
    vocab = int(tokens.max().item()) + 1
    base = logits_fn(tokens)

    future = tokens.clone()
    future[0, -1] = (future[0, -1] + 1) % max(vocab, 2)   # 改成不同的 id
    got_future = logits_fn(future)

    peer = tokens.clone()
    for b in range(1, tokens.shape[0]):
        peer[b] = (peer[b] + 1 + b) % max(vocab, 2)
    got_peer = logits_fn(peer)

    return {
        "n_requests": int(tokens.shape[0]),
        "future_token_past_logits_equal": bool(
            torch.equal(base[0, :-1], got_future[0, :-1])),
        "peer_request_logits_equal": bool(torch.equal(base[0], got_peer[0])),
    }


@torch.no_grad()
def prefix_invariance_check(model, tokens: torch.Tensor, mode: str) -> dict:
    """stock `Block` 主干在 KV `mode` 精度下的前缀不变性（`quantize_kv` 的生产口径）。

    头共享（`GQABlock`）等非 stock 主干请直接用 `prefix_invariance_of_logits`
    传入该臂自己的 `logits_fn`。
    """
    out = prefix_invariance_of_logits(lambda x: kv_quant_logits(model, x, mode)[0], tokens)
    out["mode"] = mode
    return out


# --------------------------------------------------------- ② 新旧口径并列比较


def quantized_arms_of(cfg: dict) -> list[str]:
    """真正施加 int8/int4 KV 量化的臂（fp16 基线不算；纯掩码/纯头共享不算）。

    只有这些臂的 bpc 允许随量化器口径改变；其余臂必须逐位不变。
    """
    out = []
    for arm, spec in cfg["arms"].items():
        if spec.get("mode") in ("int8", "int4"):
            out.append(arm)
    return out


def direction_label(delta_bpc: float, tol: float = BITWISE_TOL) -> str:
    """Δbpc = bpc_新 − bpc_旧（正 = 因果口径更差）。"""
    if delta_bpc < -tol:
        return "改善（因果口径质量代价更小）"
    if delta_bpc > tol:
        return "变差（因果口径质量代价更大）"
    return "不变（逐位）"


def compare_causal_vs_noncausal(old_points: list[dict], new_points: list[dict],
                                quantized_arms, tol: float = BITWISE_TOL) -> dict:
    """把旧（非因果，`2621`）点表与本次因果点表逐臂并列。

    返回：`rows`（逐臂 Δ + 方向 + 召回 Δ）、`nonquant_bitwise_preserved`（可控性证据）、
    量化臂的 Δbpc 汇总（哪些变了 / 方向）。
    """
    old = {p["arm"]: p for p in old_points}
    new = {p["arm"]: p for p in new_points}
    qset = set(quantized_arms)
    rows: list[dict] = []
    for arm, n in new.items():
        o = old.get(arm)
        if o is None:
            rows.append({"arm": arm, "quantized": arm in qset, "missing_in_old": True})
            continue
        d = float(n["bpc"]) - float(o["bpc"])
        rd = float(n["recall_copy"]["accuracy"]) - float(o["recall_copy"]["accuracy"])
        rows.append({
            "arm": arm,
            "quantized": arm in qset,
            "bpc_old": o["bpc"],
            "bpc_new": n["bpc"],
            "delta_bpc": round(d, 9),
            "direction": direction_label(d, tol),
            "recall_copy_old": o["recall_copy"]["accuracy"],
            "recall_copy_new": n["recall_copy"]["accuracy"],
            "delta_recall_copy": round(rd, 9),
            "bytes_per_position": n["bytes_per_position"],
        })
    nonq = [r for r in rows if not r["quantized"] and not r.get("missing_in_old")]
    q = [r for r in rows if r["quantized"] and not r.get("missing_in_old")]
    return {
        "tol_bpc": tol,
        "rows": rows,
        "nonquant_max_abs_delta_bpc": max((abs(r["delta_bpc"]) for r in nonq), default=None),
        "nonquant_bitwise_preserved": all(abs(r["delta_bpc"]) <= tol for r in nonq),
        "quant_arms_changed": [r["arm"] for r in q if abs(r["delta_bpc"]) > tol],
        "quant_arms_unchanged": [r["arm"] for r in q if abs(r["delta_bpc"]) <= tol],
        "quant_delta_bpc": {r["arm"]: r["delta_bpc"] for r in q},
        "quant_improved": {r["arm"]: r["delta_bpc"] for r in q if r["delta_bpc"] < -tol},
        "quant_worsened": {r["arm"]: r["delta_bpc"] for r in q if r["delta_bpc"] > tol},
    }


def int8_int4_quality_delta(comparison: dict) -> dict:
    """判据 ② 的判读量：int8 / int4 两个**同族**纯量化臂的 Δbpc。

    Δ = bpc_新 − bpc_旧 < 0 ⇒ 旧（非因果）数字**夸大**了量化代价（因果口径更便宜）；
    Δ > 0 ⇒ 旧数字**缩小**了代价。缺任一臂则对应字段为 None（如实留空）。
    """
    by_arm = {r["arm"]: r for r in comparison["rows"] if not r.get("missing_in_old")}
    out: dict = {}
    for name, arm in (("int8", "int8"), ("int4", "int4")):
        r = by_arm.get(arm)
        if r is None:
            out[name] = None
            continue
        d = r["delta_bpc"]
        out[name] = {
            "delta_bpc": d,
            "direction": r["direction"],
            "old_overstated_damage": bool(d < -comparison["tol_bpc"]),
            "old_understated_damage": bool(d > comparison["tol_bpc"]),
        }
    return out
