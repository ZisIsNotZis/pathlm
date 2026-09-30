"""KV 消融保留曲线仪器（ticket 31 契约 I）——评测期删除 KV，不重训。

契约：`axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/README.md`

这件天平回答一个问题：**把某些位置的 KV 从注意力可见集里删掉，质量掉多少。**
它不改模型权重（纯推理），只改「哪些 key 能被哪些 query 看到」——在注意力
掩码上删位置，其余全部逐字段一致（同一 checkpoint / 同一数据流 / 同一位置集）。

三条曲线共用同一实现，只改「删哪些位置」：

1. `distance_keep_mask(T, d)`：删除「与查询距离**恰为** d」的 key ——
   query i 不可见 key `i-d`。d ∈ {1,2,4,8,16,32,64,128,255}。
2. `prefix_keep_mask(T, N)`：删除**最远的 N 个绝对位置** `[0, N)` 的 key
   （左前缀截断）——这是「左前缀能丢多少」的直接答案。
3. `segment_keep_mask(T, start, length)`：删除一段**连续位置**
   `[start, start+length)` 的 key（随机起点）。

掩码语义（单测钉住）：
- 返回 bool `[T, T]`，`True = 允许该 (query, key) 对参与注意力`（与
  `torch.nn.functional.scaled_dot_product_attention` 的 bool 掩码语义一致）；
- **恒含因果**：`keep[i, j] == False` 当 `j > i`（不许看未来）；
- **对角线恒保留**：`keep[i, i] == True`。否则被删的 query 自身那一行会
  全掩码 ⇒ softmax 全 `-inf` ⇒ NaN（前缀/连续段删除会命中；距离曲线
  `d >= 1` 不涉及对角线，因为 `i - j == d` 蕴含 `j < i`）。

数值等价性（单测 + run.py 的仪器自检）：把**精确因果掩码**显式传给
`scaled_dot_product_attention`（`is_causal=False`）与走释放版的
`is_causal=True` 路径，在真实 13M 主干上逐位相等（实测 max|Δlogit| = 0.0），
所以「全掩码基线」不是另一个模型。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LN2 = 0.6931471805599453  # nats → bits


# --------------------------------------------------------------- 位置集合（掩码）


def _query_key_index(T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(1), idx.unsqueeze(0)  # i [T,1], j [1,T]


def distance_keep_mask(T: int, d: int, device="cpu") -> torch.Tensor:
    """删除「与查询距离恰为 d」的 key：query i 不可见 key `i-d`。

    masked 对数 = T - d（query i >= d 各删 1 个 key）。
    """
    if not 0 < d < T:
        raise ValueError(f"distance d must be in (0, {T}), got {d}")
    i, j = _query_key_index(T, device)
    return (j <= i) & ((i - j) != d)


def prefix_keep_mask(T: int, n_dropped: int, device="cpu") -> torch.Tensor:
    """删除最远的 N 个绝对位置（左前缀 `[0, N)`）的 key；对角线恒保留。

    某个 query 的自身位置落在前缀里时它只剩「看自己」——这是「前缀被整个
    砍掉」的诚实语义（不是把该 query 也删掉）。
    """
    if not 0 <= n_dropped <= T:
        raise ValueError(f"n_dropped must be in [0, {T}], got {n_dropped}")
    i, j = _query_key_index(T, device)
    dropped_key = j < n_dropped                       # [1, T] 被删的绝对位置
    keep = (j <= i) & ~dropped_key
    return keep | torch.eye(T, dtype=torch.bool, device=device)


def segment_keep_mask(T: int, start: int, length: int, device="cpu") -> torch.Tensor:
    """删除一段连续位置 `[start, start+length)` 的 key；对角线恒保留。"""
    if not 0 <= start <= T:
        raise ValueError(f"start must be in [0, {T}], got {start}")
    if not 0 <= length <= T:
        raise ValueError(f"length must be in [0, {T}], got {length}")
    if start + length > T:
        raise ValueError(f"segment [{start}, {start + length}) exceeds T={T}")
    i, j = _query_key_index(T, device)
    in_segment = (j >= start) & (j < start + length)   # [1, T]
    keep = (j <= i) & ~in_segment
    return keep | torch.eye(T, dtype=torch.bool, device=device)


def mask_stats(keep: torch.Tensor) -> dict:
    """干预活性量（SCHEMA §10）：被删的 causal 对数 / 占比 + 被删 key 集合大小。

    `n_dropped_pairs` 是「(query, key) 可见性对」的减少量，只在**因果三角内**
    计数（上三角本来就不允许看，不算干预）；不是被删位置的个数——两条曲线
    删同样多的位置也可能删掉不同数量的可见性对。
    """
    T = keep.shape[0]
    causal = torch.ones(T, T, dtype=torch.bool, device=keep.device).tril()
    causal_pairs = int(causal.sum().item())               # T(T+1)/2
    n_dropped = int((causal & ~keep).sum().item())
    # 被删的**位置**数（至少对一个 query 不可见的 key）——距离/前缀/连续段曲线
    # 分别是 T−d / N / L。
    dropped_keys = int((~keep & causal).any(dim=0).sum().item())
    return {"n_dropped_pairs": n_dropped,
            "causal_pairs": causal_pairs,
            "dropped_frac": round(n_dropped / causal_pairs, 8),
            "n_dropped_keys": dropped_keys,
            "diagonal_kept": bool(keep.diag().all().item())}


# --------------------------------------------------------------- 评测期前向


@torch.no_grad()
def masked_logits(model, tokens: torch.Tensor,
                  keep_mask: torch.Tensor | None = None) -> torch.Tensor:
    """dense 主干的 logits `[B, T, V]`，注意力可见集由 `keep_mask [T, T]` 给定。

    与 `DepthARModel.forward` 的 dense 分支逐位一致（无门 / 无退出 / 纯残差
    堆叠 + tied fp32 读出）；`keep_mask=None` 走释放版的 `is_causal=True` 路径。
    """
    B, T = tokens.shape
    attn_mask = None
    if keep_mask is not None:
        if keep_mask.shape != (T, T):
            raise ValueError(f"keep_mask must be [{T},{T}], got {tuple(keep_mask.shape)}")
        # CUDA SDPA 的 bool 掩码不广播 head 维（同 pathlm.depth_ar._ragged_mask），
        # 显式 expand 到 [B, H, T, T]。
        attn_mask = keep_mask.view(1, 1, T, T).expand(B, model.cfg.n_heads, T, T)
    h = model._input_states(tokens)
    for blk in model.blocks:
        h, _ = blk(h, attn_mask=attn_mask)
    return model.readout(h)


@torch.no_grad()
def masked_bpc(model, eval_arr, keep_mask: torch.Tensor | None = None,
               n_batches: int = 24, batch_size: int = 16,
               window_seed: int = 2, device="cuda") -> float:
    """clean bpc（每字符比特数，越低越好）：因果移位 CE 的 token 平均。

    窗口采样与 `pathlm.depth_ar.depth_curve` 同一顺序（同一 `window_seed`、
    同一 `batch()`）⇒ 与释放版基线 **同一数据流、同一位置集**（契约方法行）。
    """
    from pathlm.data import batch

    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats, n_tok = 0.0, 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        logits = masked_logits(model, x, keep_mask)[:, :-1]
        tgt = x[:, 1:]
        nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                tgt.reshape(-1), reduction="sum").item()
        n_tok += tgt.numel()
    return nats / n_tok / LN2


def eval_mask(model, eval_arr, keep_mask: torch.Tensor, **kw) -> dict:
    """跑一条配置：bpc + 掩码活性量。"""
    bpc = masked_bpc(model, eval_arr, keep_mask, **kw)
    return {"bpc": round(bpc, 6), **mask_stats(keep_mask)}


@torch.no_grad()
def row_bpc(model, eval_arr, keep_mask: torch.Tensor | None = None,
            n_batches: int = 24, batch_size: int = 16, window_seed: int = 2,
            device="cuda") -> list[float]:
    """每个 query 行（`0..T-2`，行 i 预测 token `i+1`）的 token 平均 CE（bits）。

    用于把一条曲线的损伤拆成「被删段内的 query」与「被删段外的 query」两部分
    （每行每样本恰好贡献 1 个 token，所以行均值 == token 均值——`masked_bpc`
    必须等于本函数输出的均值，run.py 逐配置断言这一点）。
    """
    from pathlm.data import batch

    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats = torch.zeros(T - 1, dtype=torch.float64)
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        logits = masked_logits(model, x, keep_mask)[:, :-1].float()
        tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tgt.reshape(-1), reduction="none").view(x.shape[0], -1)
        nats += ce.double().sum(0).cpu()
        n += x.shape[0]
    return (nats / n / LN2).tolist()


def safe_radius(curve: list[dict], band: float = 0.014,
                key: str = "d", delta: str = "delta_bpc") -> dict:
    """安全丢弃半径：Δbpc 在噪声带内（|Δ| < band）的最大参数取值。

    Δ = bpc_对照 − bpc_实验（正 = 改善，SCHEMA §17.3）；删除只会抬高 bpc，
    所以安全判据等价于 `delta > -band`。返回最大满足的 key 值与其 Δ。
    """
    ok = [(row[key], row[delta]) for row in curve if abs(row[delta]) < band]
    if not ok:
        return {"value": None, "delta_bpc": None, "band": band, "n_safe": 0,
                "param": key}
    # 取参数最大者；参数同值不可能出现（曲线按参数枚举）
    val, dlt = max(ok, key=lambda kv: kv[0])
    return {"value": val, "delta_bpc": dlt, "band": band, "n_safe": len(ok),
            "param": key}


def monotonicity(rows: list[dict], key: str, field: str = "damage_bpc",
                 direction: str = "increasing", tol: float = 0.0) -> dict:
    """曲线单调性检查。在 **damage** 口径上做（damage = bpc_实验 − bpc_对照 =
    −Δ ≥ 0 表示变差），因为「丢得越多损伤不减」是 damage 的自然方向：

    - `distance` 曲线：参数 d 越大（删得越远）damage **非增**；
    - `prefix`/`segment` 曲线：参数（N / L）越大（删得越多）damage **非减**。

    `tol` 为噪声容差：`diff < -tol` 才算逆序。返回是否单调 + 每个逆序点的幅度。
    """
    rows = sorted(rows, key=lambda r: r[key])
    vals = [r[field] for r in rows]
    params = [r[key] for r in rows]
    viol = []
    for a, b, pa, pb in zip(vals, vals[1:], params, params[1:]):
        diff = (b - a) if direction == "increasing" else (a - b)
        if diff < -tol - 1e-12:
            viol.append({"from": pa, "to": pb, "damage_change": round(b - a, 6)})
    return {"param": key, "field": field, "direction": direction, "tol": tol,
            "n_points": len(rows), "monotone": not viol, "violations": viol,
            "max_violation": (min([v["damage_change"] for v in viol])
                              if direction == "increasing" else
                              max([v["damage_change"] for v in viol])) if viol else 0.0}
