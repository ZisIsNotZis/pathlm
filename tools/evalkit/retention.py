"""KV 消融保留曲线仪器（ticket 31 契约 I）——评测期删除 KV，不重训。

契约：`axes/1-time/1.1.4.2-compression/experiments/2614-kv-ablation-retention-13m/README.md`

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

---

## 检索腿（ticket 32 契约 M）——「依赖」之外的「保留」

上半部分量的是**依赖**：bpc 对被删位置的敏感度。低损伤也可能是**模型本来就不
用**它——那是 X3「假保留」的另一面。因此这里再加一条腿：把「针」（一段被植入
窗口的真实文本短语）放进去，删掉记忆后再问，看还能不能答出来。

- `recent_keep_mask(T, d)`：只保留与查询距离 `< d` 的 key（= 保留最近 d 个位置）；
- `phrase_needle_batch(...)`：**needle v3 构造**（见函数 docstring 的构造修复说明）；
- `needle_fate(keep, meta, ...)`：每根针的归属（anchor / window / beyond，恰好一个桶）；
- `needle_recall(model, tokens, meta, keep, ...)`：一根针一次计数，出召回率与分桶；
- `needle_summary(records)`：把多批的逐针记录聚合成总召回 + 分桶召回。

---

## 优先级规则（ticket 32 契约 N）——「先压谁」

上面两件工具量的都是「删多少」，这一件量「删哪些」：**在删除预算完全相同**（删同样
多的位置）时，按不同的优先级规则挑被删位置，两条腿各掉多少。

- `attention_stats(model, tokens, recent_ks=...)`：在**全保留**前向里逐层逐头统计
  每个位置收到的注意力侧代理量（评测期可观测）：`mass_total`（收到注意力质量之和）、
  `mass_recent[k]`（最近 k 个查询给它的质量）、`cos_query`（与查询的平均余弦相似度）、
  `key_norm`、`soft_recency`（质量加权平均查询下标）。**逐窗口**统计（每行一条
  序列各有自己的历史）⇒ 返回 `[B, T]`。
- `priority_drop_mask(stats, n_drop, rule, k=...)`：按规则给出 `[B, T]` 的被删位置
  布尔掩码（`distance` / `lru` / `content` / `lastused`，平局次序固定 ⇒ 确定）。
- `drop_mask_keep(drop)`：`[B, T]` 被删掩码 → `[B, T, T]` 注意力可见掩码
  （因果 ∧ 未被删；对角线恒保留）。
- `drop_mask_overlap(a, b)`：两组被删集合的重叠度（逐窗口交集 / 并集 / Jaccard）。
- `attention_stats_summary(stats, ...)`：注意力质量的分布（分箱均值、分位数、被读得
  最多的位置、零质量位置占比）——契约要求「报注意力质量的分布」。

`masked_logits` / `masked_bpc` / `row_bpc` / `needle_recall` 支持**逐样本**掩码
`[B, T, T]`（同一批里每个窗口可以有各自的删除集）；`[T, T]` 共享掩码路径逐位不变。
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
    """dense 主干的 logits `[B, T, V]`，注意力可见集由 `keep_mask` 给定。

    与 `DepthARModel.forward` 的 dense 分支逐位一致（无门 / 无退出 / 纯残差
    堆叠 + tied fp32 读出）；`keep_mask=None` 走释放版的 `is_causal=True` 路径。

    `keep_mask` 可以是**共享**的 `[T, T]`（所有窗口同一套可见集），也可以是
    **逐样本**的 `[B, T, T]`（每个窗口有自己的删除集，契约 N 的优先级规则）。
    """
    B, T = tokens.shape
    attn_mask = None
    if keep_mask is not None:
        if keep_mask.shape == (T, T):
            # CUDA SDPA 的 bool 掩码不广播 head 维（同 pathlm.depth_ar._ragged_mask），
            # 显式 expand 到 [B, H, T, T]。
            attn_mask = keep_mask.view(1, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        elif keep_mask.shape == (B, T, T):
            attn_mask = keep_mask.view(B, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        else:
            raise ValueError(f"keep_mask must be [{T},{T}] or [{B},{T},{T}], "
                             f"got {tuple(keep_mask.shape)}")
    h = model._input_states(tokens)
    for blk in model.blocks:
        h, _ = blk(h, attn_mask=attn_mask)
    return model.readout(h)


@torch.no_grad()
def masked_bpc(model, eval_arr, keep_mask: torch.Tensor | None = None,
               n_batches: int = 24, batch_size: int = 16,
               window_seed: int = 2, device="cuda", mask_fn=None) -> float:
    """clean bpc（每字符比特数，越低越好）：因果移位 CE 的 token 平均。

    窗口采样与 `pathlm.depth_ar.depth_curve` 同一顺序（同一 `window_seed`、
    同一 `batch()`）⇒ 与释放版基线 **同一数据流、同一位置集**（契约方法行）。

    `mask_fn(x [B, T]) -> [B, T, T]` 给**逐样本**可见集（契约 N：每个窗口凭自己的
    注意力统计选删除集）；与 `keep_mask` 互斥。
    """
    from pathlm.data import batch

    if keep_mask is not None and mask_fn is not None:
        raise ValueError("keep_mask 与 mask_fn 互斥")
    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats, n_tok = 0.0, 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        km = mask_fn(x) if mask_fn is not None else keep_mask
        logits = masked_logits(model, x, km)[:, :-1]
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
            device="cuda", mask_fn=None) -> list[float]:
    """每个 query 行（`0..T-2`，行 i 预测 token `i+1`）的 token 平均 CE（bits）。

    用于把一条曲线的损伤拆成「被删段内的 query」与「被删段外的 query」两部分
    （每行每样本恰好贡献 1 个 token，所以行均值 == token 均值——`masked_bpc`
    必须等于本函数输出的均值，run.py 逐配置断言这一点）。

    `mask_fn` 同 `masked_bpc`（逐样本可见集；与 `keep_mask` 互斥）。
    """
    from pathlm.data import batch

    if keep_mask is not None and mask_fn is not None:
        raise ValueError("keep_mask 与 mask_fn 互斥")
    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats = torch.zeros(T - 1, dtype=torch.float64)
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        km = mask_fn(x) if mask_fn is not None else keep_mask
        logits = masked_logits(model, x, km)[:, :-1].float()
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


# ------------------------------------------------- 检索腿：保留曲线（ticket 32 契约 M）


def recent_keep_mask(T: int, d: int, device="cpu") -> torch.Tensor:
    """只保留「与查询距离 < d」的 key（= 保留最近 d 个位置）；对角线恒保留。

    这是「只保留最近 d」的保留配置：query i 只可见 `[i-d+1, i]`。
    `d = T` 等价于精确因果（全保留），单测钉住这一点。
    """
    if not 0 < d <= T:
        raise ValueError(f"d must be in (0, {T}], got {d}")
    i, j = _query_key_index(T, device)
    keep = (j <= i) & ((i - j) < d)
    return keep | torch.eye(T, dtype=torch.bool, device=device)


def phrase_needle_batch(batch_size: int, seq_len: int, data, n_real_tokens: int,
                        mask_token: int, generator, anchors: int = 8,
                        anchor_frac: float = 0.25, n_needles: int = 8,
                        phrase_len: int = 6, plant: bool = True):
    """**needle v3**：真实文本短语针（构造修复，见下），返回 `(x, meta)`。

    每行布局：`[0, q0)` 是真实文本窗口（针被植入其中），尾块 `[q0, T)` 每根针
    占 `phrase_len + 1` 行：`[mask, cue_0 … cue_{L-1}]`，其中 `cue = phrase[:L]`
    （L = phrase_len），答案 = 短语的下一个字节 `phrase[L]`，在答案行 `q_row`
    （输入 = `cue[-1]`，目标 = `phrase[L]`）上计命中。

    meta[b] = 每根针的 `{i, q_row, p, ans, cue, dist}`；`plant=False` 时**只写尾块的
    线索、不植入窗口里的短语副本**（同一 generator 序列 ⇒ 与 `plant=True` 逐一配对），
    这就是「记忆不存在」的对照臂（检索腿的**地板**）。

    构造修复（为什么不是 `pathlm.data.needle_batch` 的 v2）：v2 把**随机字节对**
    `(x, y)` 植入窗口，需要「任意内容复制」能力；本项目被测量的 13.5M dense 主干
    是在纯文本上训练的（`p_needle = 0`），实测 v2 基线召回 **1.5%**（见 run.py 的
    `construction_diagnosis`），低于契约的 50% 门槛 ⇒ 按契约「先修构造再测」。
    修复：针的内容改用**真实 enwik8 短语**（模型的原生能力 = 从上下文续写真实文本），
    并让线索足够长（L = 6）。修复后基线召回 ≈ 0.70（≥ 50%）。
    「针位置只被删一次」的语义：一根针的短语占 `[p, p+L]` 共 L+1 个位置，**互不重叠**
    （间距 ≥ L+1）且互不重复；`needle_fate` 依「全部位置可见」判可见（半删也算被删）。
    """
    import numpy as np

    if data is None:
        raise ValueError("phrase needle batches need real-text filler: pass the data array")
    T, K, L = seq_len, n_needles, phrase_len
    if not 0 < L < T:
        raise ValueError(f"phrase_len must be in (0, {T}), got {L}")
    span = L + 1                      # 尾块每根针占的行数：[mask] + cue
    q0 = T - K * span
    if q0 - L <= anchors:
        raise ValueError(f"no room: T={T}, K={K}, L={L} leaves q0={q0}")
    x = np.zeros((batch_size, T), dtype=np.int64)
    hi = len(data) - T - 1
    starts = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
    meta: list[list[dict]] = []
    for b in range(batch_size):
        # 唯一的关联判据：短语不能已经出现在该行的**原始**文本里（否则答案有第二个来源）。
        # 用原始副本做判据（不是逐针改写后的 row），这样 plant 与否不改变抽样序列。
        fill = np.array(data[starts[b]:starts[b] + T], dtype=np.int64)
        fill_bytes = fill[:q0].tobytes()
        row = fill.copy()
        used, placed, rows_meta = [], set(), []
        for i in range(K):
            for _ in range(64):
                off = int(torch.randint(0, len(data) - span, (1,), generator=generator))
                phrase = np.array(data[off:off + span], dtype=np.int64)
                if int(phrase.max()) >= n_real_tokens:
                    continue
                if phrase.tobytes() in fill_bytes or phrase.tobytes() in placed:
                    continue
                if anchors > 0 and torch.rand(1, generator=generator).item() < anchor_frac:
                    p = int(torch.randint(0, anchors, (1,), generator=generator))
                else:
                    p = int(torch.randint(anchors, q0 - L, (1,), generator=generator))
                if any(abs(p - u) < span for u in used):     # 位置互不重叠（间距 ≥ L+1）
                    continue
                base = q0 + i * span
                row[base] = mask_token                       # 线索标记（文本里不会出现）
                row[base + 1:base + 1 + L] = phrase[:L]
                if plant:
                    row[p:p + span] = phrase
                used.append(p)
                placed.add(phrase.tobytes())
                rows_meta.append({"i": i, "q_row": base + L, "p": p,
                                  "ans": int(phrase[L]),
                                  "cue": [int(v) for v in phrase[:L]],
                                  "dist": (base + L) - (p + L)})
                break
        x[b] = row
        meta.append(rows_meta)
    return torch.from_numpy(x), meta


def needle_fate(keep: torch.Tensor | None, meta, anchors: int = 8,
                phrase_len: int = 6, row: int = 0) -> list[dict]:
    """每根针的归属（**恰好一个桶**）；`row` 选出逐样本掩码 `[B, T, T]` 里的第几行
    （共享掩码 `[T, T]` / 全保留时忽略）。

    - `anchor`：针的起点落在最前面的 `anchors` 个位置（历史的「锚区」语义：
      那一段在历史配置里由锚通道保住，所以单列一桶）；
    - `window`（窗口内，可见）：非锚针，且短语全部 `L+1` 个位置对其答案行可见；
    - `beyond`（窗口外，被删）：非锚针，短语**有任意一个位置不可见**。

    可见性用 `all`（不是 `any`）：**半删的针判为「被删」**——关联需要短语的每一个
    位置都在，少一个位置该关联就不完整，不能算「保住了」。桶之间互斥，
    `sum(bucket 计数) == n_needles`（单测钉住；变异验证会故意破坏它）。
    `keep = None` = 全保留（所有针可见），用于基线。
    """
    ks = keep if (keep is None or keep.dim() == 2) else keep[row]
    out = []
    for m in meta:
        q, p = m["q_row"], m["p"]
        visible = (ks is None
                   or all(bool(ks[q, p + j]) for j in range(phrase_len + 1)))
        bucket = "anchor" if p < anchors else ("window" if visible else "beyond")
        out.append({**m, "visible": visible, "bucket": bucket})
    return out


@torch.no_grad()
def needle_score(scores: torch.Tensor, meta, keep_mask: torch.Tensor | None = None,
                 anchors: int = 8, phrase_len: int = 6) -> list[dict]:
    """已算好的 logits `[B, T, V]`（或 `[T, V]`）→ 逐针 top-1 命中 + 归属桶。

    与 `needle_recall` 同一套语义，只是把前向留给调用者——契约 N 要在**同一批**上
    跑「同一批针 × 多个优先级规则的删除集」，一批只算一次前向。
    """
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
    per = []
    for b, rows in enumerate(meta):
        for fate in needle_fate(keep_mask, rows, anchors=anchors, phrase_len=phrase_len,
                                row=b):
            hit = int(scores[b, fate["q_row"]].argmax(-1).item() == fate["ans"])
            per.append({**fate, "hit": hit})
    return per


@torch.no_grad()
def needle_recall(model, tokens: torch.Tensor, meta, keep_mask: torch.Tensor | None = None,
                  anchors: int = 8, phrase_len: int = 6, device="cuda") -> dict:
    """跑一批针：逐针 top-1 命中 + 归属桶。一根针一次计数（不重复、不遗漏）。

    返回 `{"per_needle": [...], "n":, "recall":, "buckets": {桶: {recall, n}}}`。
    掩码语义与依赖腿**完全同一件仪器**（`keep[i, j] = False` = key j 对 query i
    不可见）；`keep_mask=None` = 全保留基线；`keep_mask [B, T, T]` = 逐样本删除集。
    """
    x = tokens.to(device)
    logits = masked_logits(model, x, keep_mask).cpu()
    per = needle_score(logits, meta, keep_mask, anchors=anchors, phrase_len=phrase_len)
    return {"per_needle": per, **needle_summary(per)}


def needle_summary(records: list[dict]) -> dict:
    """把逐针记录聚合成总召回 + 分桶召回（空桶不出现在结果里）。"""
    if not records:
        return {"recall": None, "n": 0, "buckets": {}}
    buckets: dict[str, list[int]] = {}
    for r in records:
        buckets.setdefault(r["bucket"], []).append(r["hit"])
    return {"recall": round(sum(r["hit"] for r in records) / len(records), 6),
            "n": len(records),
            "buckets": {k: {"recall": round(sum(v) / len(v), 6), "n": len(v)}
                        for k, v in buckets.items()}}


# ------------------------------------------- 优先级规则（ticket 32 契约 N：先压谁）

#: 「先压谁」的四条规则。`distance` / `lru` / `content` 是契约 N 的判据臂；
#: `lastused` 是用户 LRU 语义的另一种落地（敏感性臂，不参与判据）。
PRIORITY_RULES = ("distance", "lru", "content", "lastused")


@torch.no_grad()
def attention_stats(model, tokens: torch.Tensor,
                    recent_ks: tuple[int, ...] = (32, 64, 128)) -> dict:
    """**全保留**前向里逐窗口的「注意力质量」统计（`[B, T]`，逐层逐头平均）。

    这是压缩优先级的可观测量（评测期可观测：注意力权重来自这次前向本身）。

    - `mass_total[j]`：位置 j 收到的注意力质量 `Σ_{i>j} A[i,j]`（**排除对角线**：
      自己读自己不算「被读过」）；`mass_total_incl_diag` 为含对角线的版本；
    - `mass_recent[k][j]`：最近 k 个查询（`i ≥ T-k`）给它的质量 `Σ_{i≥T-k, i>j} A[i,j]`
      ——契约 N 的 `lru` 读数（「最近 k 步内被注意质量最少」）；
    - `cos_query[j]`：与查询的平均余弦相似度 `mean_{i>j} cos(q_i, k_j)`（契约 N 的
      `content` 读数之一）；
    - `key_norm[j]`：`mean_h ‖k_j‖`（`content` 的另一个读数）；
    - `soft_recency[j]`：质量加权平均查询下标 `Σ A[i,j]·i / Σ A[i,j]`（**LRU 语义**：
      越小 = 越久没被读过）；零质量位置回退为 `j`（从未被读 ⇒ 按位置排序）。

    每行序列各有自己的统计（删除决策是逐序列的）⇒ 不跨 batch 平均，返回 `[B, T]`。
    注意力用显式 softmax 复算（`is_causal`，与 SDPA 路径同一套 q/k/v），只是为了
    把权重取出来；单测钉住它与 `masked_logits` 的等价路径同分布（见
    `test_attention_stats_mass_is_causal_and_excludes_self`）。
    """
    x = tokens
    B, T = x.shape
    d = model.cfg.d_model
    nh = model.cfg.n_heads
    dh = d // nh
    dev = x.device
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")
    for k in recent_ks:
        if not 0 < k <= T:
            raise ValueError(f"recent k must be in (0, {T}], got {k}")
    shape = lambda t: t.view(B, T, nh, dh).transpose(1, 2)
    nh_inv = 1.0 / nh
    idx = torch.arange(T, device=dev)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)                  # [T, T]
    off_diag = causal & ~torch.eye(T, dtype=torch.bool, device=dev)
    n_off = off_diag.sum(0).clamp_min(1).to(torch.float64)         # [T] 每个 j 的 i 数
    mass = torch.zeros(B, T, dtype=torch.float64, device=dev)
    mass_diag = torch.zeros_like(mass)
    cos_sum = torch.zeros_like(mass)
    knorm = torch.zeros_like(mass)
    rec_num = torch.zeros_like(mass)
    rec_den = torch.zeros_like(mass)
    recent = {k: torch.zeros_like(mass) for k in recent_ks}
    layer_bins: dict[int, list[float]] = {k: [] for k in recent_ks}
    bins = ((0, 8), (8, 64), (64, 192), (192, T))

    h = model._input_states(x)
    for blk in model.blocks:
        q, k, v = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)
        qs, ks = shape(q), shape(k)
        scores = qs @ ks.transpose(-2, -1) / dh ** 0.5
        w = scores.masked_fill(~causal, float("-inf")).softmax(-1)      # [B,H,T,T]
        w = w * nh_inv                                                # 逐头平均
        wo = w * off_diag                                             # 排除自身读出
        mass += wo.sum((1, 2)).double()
        mass_diag += w.sum((1, 2)).double()
        for kk in recent_ks:                                          # 最近 k 个查询
            recent[kk] += wo[:, :, T - kk:, :].sum((1, 2)).double()
            layer_bins[kk].append(
                float(wo[:, :, T - kk:, :].sum((1, 2)).mean().item()))
        kn = ks.norm(dim=-1)                                          # [B,H,T]
        knorm += kn.mean(1).double()
        qn = qs / qs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        knn = ks / kn.clamp_min(1e-6).unsqueeze(-1)
        cosm = qn @ knn.transpose(-2, -1)                             # [B,H,T,T]
        cos_sum += (cosm * off_diag).sum((1, 2)).double() * nh_inv
        wo_h = wo.sum(1)                                              # [B,T,T]：i 是查询、j 是 key
        rec_num += (wo_h * idx.view(1, T, 1)).sum(1).double()         # Σ_i A[i,j]·i
        rec_den += wo_h.sum(1).double()                               # Σ_i A[i,j]
        # 位置依赖：块输出推进残差流（与 masked_logits 同一套 block 调用）
        h, _ = blk(h, attn_mask=None)

    n_layers = len(model.blocks)
    out = {
        "mass_total": (mass / n_layers).float(),
        "mass_total_incl_diag": (mass_diag / n_layers).float(),
        "cos_query": (cos_sum / n_layers / n_off.view(1, T)).float(),
        "key_norm": (knorm / n_layers).float(),
        "soft_recency": torch.where(rec_den > 0, rec_num / rec_den.clamp_min(1e-12),
                                    idx.view(1, T).to(torch.float64)).float(),
        "mass_recent": {k: (recent[k] / n_layers).float() for k in recent_ks},
        "meta": {"n_layers": n_layers, "n_heads": nh, "T": T, "n_windows": B,
                 "recent_ks": list(recent_ks),
                 "layer_mean_mass_recent_bins": {
                     k: {"bins": [list(b) for b in bins],
                         "per_layer_mean": [round(v, 8) for v in layer_bins[k]]}
                     for k in recent_ks}},
    }
    return out


def _rank(primary: torch.Tensor, secondary: torch.Tensor, n_drop: int,
          candidates: torch.Tensor | None = None) -> torch.Tensor:
    """逐窗口取 `(primary, secondary, 位置)` 升序的前 `n_drop` 个位置（布尔 `[B, T]`）。

    平局次序固定 ⇒ 规则完全确定（同一输入 ⇒ 同一删除集，单测钉住）。
    `candidates [T]` 限定**候选位置集**（例如「针的线索块不参与压缩」）：非候选位置
    的排序键置 `+inf`，预算仍是 `n_drop` 个位置。
    """
    B, T = primary.shape
    if candidates is not None:
        if candidates.shape != (T,):
            raise ValueError(f"candidates must be [{T}], got {tuple(candidates.shape)}")
        if n_drop > int(candidates.sum()):
            raise ValueError(f"n_drop={n_drop} 超过候选位置数 {int(candidates.sum())}")
        candidates = candidates.to(primary.device)
        inf = torch.full((), float("inf"), device=primary.device,
                         dtype=primary.dtype)
        primary = torch.where(candidates.view(1, T), primary, inf)
        secondary = torch.where(candidates.view(1, T), secondary, inf)
    drop = torch.zeros(B, T, dtype=torch.bool, device=primary.device)
    p = primary.tolist()
    s = secondary.tolist()
    for b in range(B):
        order = sorted(range(T), key=lambda j: (p[b][j], s[b][j], j))
        for j in order[:n_drop]:
            drop[b, j] = True
    return drop


def priority_drop_mask(stats: dict, n_drop: int, rule: str,
                       k: int = 64, key: str | None = None,
                       candidates: torch.Tensor | None = None) -> torch.Tensor:
    """按优先级规则给出**先压**的 `n_drop` 个位置（`[B, T]` 布尔，True = 被删）。

    **删除预算 = 位置个数**，四条规则完全相同的 `n_drop`（契约 N 的「同等删除预算」；
    实际影响的 (query, key) 对数随位置不同而不同，用 `mask_stats` 的
    `n_dropped_pairs` 如实另报）。

    - `distance`：最远的（绝对位置最小）——现基线语义（= `prefix_keep_mask`）；
    - `lru`：最近 `k` 步内收到的注意力质量最少（`mass_recent[k]`），次级键 =
      全窗口质量 `mass_total`；
    - `content`：与查询平均余弦相似度最低（`cos_query`），次级键 = `key_norm`；
      契约允许的另一种内容读数 = `key="key_norm"`（按 key 范数最小，次级键反过来）。
      `key` 只对 `content` 合法，其他规则传入会显式报错。
    - `lastused`：质量加权平均查询下标最小（`soft_recency`，用户 LRU 语义的另一种
      落地），次级键 = `mass_total`。

    `candidates [T]` 把可删位置限定在一个子集（诊断用：针的线索块不属于 KV 压缩对象），
    预算仍是 `n_drop` 个位置。
    """
    if rule not in PRIORITY_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {PRIORITY_RULES}")
    B, T = stats["mass_total"].shape
    if not 0 <= n_drop <= T:
        raise ValueError(f"n_drop must be in [0, {T}], got {n_drop}")
    if rule == "distance":                      # 最远 = 绝对位置最小的 n_drop 个
        idx = torch.arange(T, device=stats["mass_total"].device).view(1, T)
        return _rank(idx.expand(B, T).to(torch.float64),
                     torch.zeros(B, T, dtype=torch.float64, device=idx.device),
                     n_drop, candidates)
    if key is not None and rule != "content":
        raise ValueError(f"key 只对 content 规则有意义，rule={rule!r}")
    if rule == "lru":
        if k not in stats["mass_recent"]:
            raise ValueError(f"stats 里没有 mass_recent[{k}]；"
                             f"可用 {sorted(stats['mass_recent'])}")
        return _rank(stats["mass_recent"][k], stats["mass_total"], n_drop, candidates)
    if rule == "content":
        if key not in (None, "cos_query", "key_norm"):
            raise ValueError(f"content 的 key 只能是 cos_query / key_norm，得到 {key!r}")
        if key == "key_norm":
            return _rank(stats["key_norm"], stats["cos_query"], n_drop, candidates)
        return _rank(stats["cos_query"], stats["key_norm"], n_drop, candidates)
    return _rank(stats["soft_recency"], stats["mass_total"], n_drop, candidates)


def drop_mask_keep(drop: torch.Tensor) -> torch.Tensor:
    """`[B, T]` 被删位置掩码 → `[B, T, T]` 注意力可见掩码。

    `keep[b, i, j] = (j <= i) ∧ ¬drop[b, j]`，**对角线恒保留**（否则被删 query
    自身那一行全掩码 ⇒ softmax 全 -inf ⇒ NaN）。与 `prefix_keep_mask` 在
    「逐行同一删除集」时逐位一致（单测钉住）。
    """
    if drop.dim() != 2:
        raise ValueError(f"drop must be [B, T], got {tuple(drop.shape)}")
    B, T = drop.shape
    idx = torch.arange(T, device=drop.device)
    keep = (idx.unsqueeze(1) >= idx.unsqueeze(0)).unsqueeze(0) & ~drop.unsqueeze(1)
    return keep | torch.eye(T, dtype=torch.bool, device=drop.device).unsqueeze(0)


def drop_mask_overlap(a: torch.Tensor, b: torch.Tensor) -> dict:
    """两组 `[B, T]` 被删集合的重叠度（逐窗口 → 均值；契约 N 要求报）。

    `mean_jaccard` = 逐窗口 |∩| / |∪| 的均值；两集合都空时该窗口记 1.0（完全相同）。
    """
    if a.shape != b.shape or a.dim() != 2:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    inter = (a & b).sum(-1).double()
    union = (a | b).sum(-1).double()
    jac = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
    return {"mean_intersection": round(float(inter.mean().item()), 6),
            "mean_union": round(float(union.mean().item()), 6),
            "mean_jaccard": round(float(jac.mean().item()), 6),
            "n_identical_windows": int((a == b).all(dim=-1).sum().item()),
            "n_windows": int(a.shape[0])}


def attention_stats_summary(stats: dict, ks: tuple[int, ...] | None = None,
                            bins: tuple[tuple[int, int], ...] | None = None,
                            top_n: int = 10) -> dict:
    """注意力质量的**分布**（契约 N 要求）：分箱均值 / 分位数 / 热点位置 / 零质量占比。

    所有量都先**跨窗口取均值**到「每个位置一个数」（`[T]`），再给分箱与分位数——
    位置轴是本实验的解释轴（哪个位置被读得多），所以不做位置平均。
    """
    mass = stats["mass_total"].double().mean(0)                       # [T]
    T = mass.shape[0]
    ks = tuple(ks if ks is not None else sorted(stats["mass_recent"]))
    bins = bins or ((0, 8), (8, 32), (32, 64), (64, 128), (128, 192), (192, T))

    def _q(v: torch.Tensor) -> dict:
        return {"min": round(float(v.min()), 8), "p25": round(float(v.quantile(0.25)), 8),
                "median": round(float(v.median()), 8),
                "p75": round(float(v.quantile(0.75)), 8),
                "max": round(float(v.max()), 8), "mean": round(float(v.mean()), 8)}

    def _bins(v: torch.Tensor) -> dict:
        return {f"[{lo},{hi})": round(float(v[lo:hi].mean()), 8) for lo, hi in bins}

    top = torch.argsort(mass, descending=True)[:top_n].tolist()
    out = {
        "position_bins": [list(b) for b in bins],
        "mass_total": {"by_bin": _bins(mass), "quantiles": _q(mass),
                       "zero_frac": round(float((mass <= 1e-9).double().mean()), 6),
                       "profile_per_window_mean": [round(float(v), 8) for v in mass],
                       "top_positions": [{"pos": int(j), "mass": round(float(mass[j]), 8)}
                                         for j in top]},
        "mass_recent": {str(k): {
            "by_bin": _bins(stats["mass_recent"][k].double().mean(0)),
            "zero_frac": round(float((stats["mass_recent"][k] <= 1e-9)
                                     .double().mean()), 6)}
            for k in ks},
        "cos_query": {"by_bin": _bins(stats["cos_query"].double().mean(0)),
                      "quantiles": _q(stats["cos_query"].double().mean(0))},
        "key_norm": {"by_bin": _bins(stats["key_norm"].double().mean(0)),
                     "quantiles": _q(stats["key_norm"].double().mean(0))},
        "soft_recency": {"by_bin": _bins(stats["soft_recency"].double().mean(0))},
        "n_windows": stats["meta"]["n_windows"], "n_layers": stats["meta"]["n_layers"],
        "layer_mean_mass_recent_bins": stats["meta"]["layer_mean_mass_recent_bins"],
    }
    return out


def drop_mask_stats(drop: torch.Tensor) -> dict:
    """逐窗口被删掩码 `[B, T]` 的活性量（与 `mask_stats` **同一口径的闭式**）。

    `keep = drop_mask_keep(drop)` 之后，窗口 b 掉掉的因果 (query, key) 对数 =
    `Σ_{j ∈ drop_b} (T−1−j)`（key j 对 query `i > j` 不可见），被删位置数 = `|drop_b|`。
    两个量分别对应 `mask_stats` 的 `n_dropped_pairs` / `n_dropped_keys`，单测钉住
    「逐行同一删除集时两者相等」。删除预算在三条规则间按**位置数**对齐，但实际
    可见性对数是位置的函数 ⇒ 必须两个都报（契约 N「同等删除预算」的诚实落地）。
    """
    if drop.dim() != 2:
        raise ValueError(f"drop must be [B, T], got {tuple(drop.shape)}")
    B, T = drop.shape
    idx = torch.arange(T, device=drop.device).to(torch.float64)
    pairs = (drop.to(torch.float64) * (T - 1 - idx).view(1, T)).sum(-1)   # [B]
    causal_pairs = T * (T + 1) // 2
    return {"n_dropped_keys_mean": round(float(drop.to(torch.float64).sum(-1).mean()
                                               .item()), 6),
            "n_dropped_pairs_mean": round(float(pairs.mean().item()), 6),
            "n_dropped_pairs_total": int(pairs.sum().item()),
            "dropped_pair_frac_mean": round(float((pairs / causal_pairs).mean().item()), 8),
            "n_windows": B}
