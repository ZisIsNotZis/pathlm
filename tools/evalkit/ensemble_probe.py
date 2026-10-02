"""K-path softmax ensemble instruments (ticket 37 · 契约 V3).

**Method identity**: `path_ensemble_bpc` reproduces `probe_tts.bpc_k` step for
step — same `random.Random(11)` / `Generator().manual_seed(11)`, same
`pathlm.data.batch` draws, same per-batch path loop (one `sample_path` per
forward), same mixture (`sum of per-path softmax / K`), same temperature 1.0 and
same `nats / tokens / ln 2` aggregation. That is what makes the 13M published
K-sweep (`L1` 1.6926/1.6657/1.6543/1.6454, `DIVL1` 1.6842/1.6482/1.6339/1.6227)
and the 50M points a **same-method** comparison. `run.py` asserts that identity
against `probe_tts.bpc_k` on a live checkpoint instead of trusting this comment.

On top of that identity this module adds the three things the contract needs and
`probe_tts` does not have:

1. **per-batch nats/tokens** — so a CI can be bootstrapped over eval batches and
   differences can be taken **paired** (same batch index in both arms);
2. **per-path probabilities** — so the pairwise JS between the ensemble members
   (the path-correlation readout) is measured on exactly the same forwards;
3. **deterministic-path control** — `deterministic_paths(model)` forces the path
   knobs off so the K "paths" are the same path K times; the control reports the
   K=8/K=1 gain and whether the per-path logits were bitwise identical (the
   exact statement) alongside the measured float-level delta.

Nothing here trains or mutates a checkpoint; every function is inference-only.
"""

import contextlib
import random

import torch
import torch.nn.functional as F

from pathlm.config import sample_path
from pathlm.data import batch

LN2 = 0.6931471805599453


def mixture_ce(mix: torch.Tensor, targets: torch.Tensor) -> dict:
    """CE of an already-mixtured prediction. mix: [N, T, V] probabilities,
    targets: [N, T]. Returns nats/tokens/correct for the whole batch. The
    mixture is formed per position BEFORE the CE (a mixture of predictions,
    never a mixture of rows) by the caller, so that the arithmetic is exactly
    `probe_tts.bpc_k`'s (sequential `ps = ps + softmax`, then `/ K`)."""
    tgt = targets.reshape(-1)
    lp = mix.clamp_min(1e-12).log()
    nats = float(-lp.reshape(-1, mix.shape[-1]).gather(1, tgt.unsqueeze(1)).sum())
    return {"nats": nats, "tokens": int(tgt.numel()),
            "correct": int((mix.argmax(-1).reshape(-1) == tgt).sum())}


def pairwise_js(probs: torch.Tensor, eps: float = 1e-8) -> dict:
    """Mean pairwise Jensen-Shannon divergence between the ensemble members'
    node-1 distributions — the path-correlation readout (0 = identical paths).
    probs: [K, N, T, V]. Averaged over positions/rows and over the K(K-1)/2
    pairs; reported in nats and bits. This is the same JS quantity the training
    loss caps (`pathlm/model.py` diversity term), measured at eval time."""
    k = probs.shape[0]
    if k < 2:
        return {"mean_nats": 0.0, "mean_bits": 0.0, "n_pairs": 0, "k_paths": k}
    p = probs.clamp_min(eps)
    total, pairs = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            q = probs[j].clamp_min(eps)
            m = 0.5 * (p[i] + q)
            js = 0.5 * (F.kl_div(m.log(), p[i], reduction="none").sum(-1)
                        + F.kl_div(m.log(), q, reduction="none").sum(-1))
            total += float(js.mean())
            pairs += 1
    mean_nats = total / pairs
    return {"mean_nats": mean_nats, "mean_bits": mean_nats / LN2,
            "n_pairs": pairs, "k_paths": k}


@contextlib.contextmanager
def deterministic_paths(model):
    """Force every path knob off for the deterministic-path control: the sampled
    path becomes layer order [0..n-1] with one repeat each, so K forwards are K
    copies of the same computation. Restores the previous knobs on exit."""
    pc = model.pcap
    saved = (pc.p_skip, pc.p_redo, pc.shuffle_locality)
    pc.p_skip, pc.p_redo, pc.shuffle_locality = 0.0, 0.0, 0.0
    try:
        yield model
    finally:
        pc.p_skip, pc.p_redo, pc.shuffle_locality = saved


@torch.no_grad()
def path_ensemble_bpc(model, eval_arr, k_paths: int, *, n_batches: int = 60,
                      batch_size: int = 32, seed: int = 11,
                      temperature: float = 1.0, vocab_size: int | None = None,
                      collect_js: bool = False) -> dict:
    """K-path entropying of node-1 predictions; see module docstring for the
    `probe_tts.bpc_k` identity. Returns point estimate + per-batch records
    (+ pairwise JS and the bitwise-identity flag when asked)."""
    model.eval()
    T = model.mcfg.seq_len
    V = vocab_size or model.vocab_size
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    per_batch, js_rows, identical = [], [], []
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(model.embed.weight.device)
        tgt = x[:, 1:]
        ps = torch.zeros(batch_size, T - 1, V, device=x.device)
        logits = [] if collect_js or k_paths > 1 else None
        for _k in range(k_paths):
            path = sample_path(model.pcap, rng, model.mcfg.n_layers)
            _, aux = model(x, [path], x)
            lg = aux["rounds"][0][1]["logits"][:, :-1] / max(temperature, 1e-6)
            if logits is not None:
                logits.append(lg)
            ps = ps + lg.softmax(-1)      # sequential add: probe_tts.bpc_k's order
        rec = mixture_ce(ps / k_paths, tgt)
        rec["bpc"] = rec["nats"] / rec["tokens"] / LN2
        per_batch.append(rec)
        if collect_js and k_paths > 1:
            js_rows.append(pairwise_js(torch.stack(logits).softmax(-1)))
        if k_paths > 1:
            identical.append(all(torch.equal(logits[0], lg) for lg in logits[1:]))
    nats = sum(r["nats"] for r in per_batch)
    toks = sum(r["tokens"] for r in per_batch)
    out = {"k_paths": k_paths, "n_batches": n_batches, "batch_size": batch_size,
           "seed": seed, "temperature": temperature,
           "bpc": round(nats / toks / LN2, 4),
           "acc": round(sum(r["correct"] for r in per_batch) / toks, 4),
           "per_batch": per_batch}
    if js_rows:
        out["pairwise_js"] = {
            "mean_nats": sum(r["mean_nats"] for r in js_rows) / len(js_rows),
            "mean_bits": sum(r["mean_bits"] for r in js_rows) / len(js_rows),
            "n_batches": len(js_rows), "k_paths": k_paths}
    if identical:
        out["paths_bitwise_identical"] = all(identical)
        out["n_batches_paths_identical"] = sum(identical)
    return out


def bootstrap_ci(values, n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI of the mean of `values` (per-batch bpc deltas or
    per-batch bpc), enough batches that a normal approximation is also fine."""
    t = torch.as_tensor([float(v) for v in values], dtype=torch.float64)
    n = t.numel()
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_resamples, n), generator=g)
    means = t[idx].mean(dim=1)
    lo = float(torch.quantile(means, alpha / 2))
    hi = float(torch.quantile(means, 1 - alpha / 2))
    return {"mean": float(t.mean()), "lo": lo, "hi": hi, "n": int(n),
            "n_resamples": n_resamples, "alpha": alpha}


def paired_delta_ci(a_batches, b_batches, **kw) -> dict:
    """Paired CI of mean(a) - mean(b) using the SAME batch indices (a and b are
    per-batch bpc lists from the same eval stream). `excludes_zero` is the
    significance read; `unanchored_ci_overlap` is the conservative pre-registered
    read from the two arms' own CIs."""
    deltas = [float(x) - float(y) for x, y in zip(a_batches, b_batches)]
    if len(deltas) != len(a_batches):
        raise ValueError("paired deltas need equal-length batch lists")
    ci = bootstrap_ci(deltas, **kw)
    ca, cb = bootstrap_ci(a_batches, **kw), bootstrap_ci(b_batches, **kw)
    return {"delta": ci["mean"], "delta_ci": [ci["lo"], ci["hi"]],
            "paired_ci_excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0),
            "arm_a_ci": [ca["lo"], ca["hi"]], "arm_b_ci": [cb["lo"], cb["hi"]],
            "arm_cis_overlap": bool(not (ca["hi"] < cb["lo"] or cb["hi"] < ca["lo"])),
            "n_batches": ci["n"]}


def per_batch_bpc(run: dict) -> list:
    return [r["bpc"] for r in run["per_batch"]]


def reading_B(bpc_l1_k1: float, bpc_stock_k1: float, bpc_divl1_kmax: float) -> dict:
    """主判据（读法 B，部署问题；契约 README §4.1 的预登记形式）：
    `[bpc_l1@K1 − bpc_divl1@Kmax] − [bpc_l1@K1 − bpc_stock@K1] > 0`
    等价于 `bpc_stock@K1 − bpc_divl1@Kmax > 0`（同一项相消）。两种写法都算出来，
    并断言代数等价（§12 算术对账）——防止只报化简式、丢了预登记形式。
    这里只给点估计的判定；CI 由 `paired_delta_ci` 另算（见 run.py）。"""
    verbatim = (bpc_l1_k1 - bpc_divl1_kmax) - (bpc_l1_k1 - bpc_stock_k1)
    reduced = bpc_stock_k1 - bpc_divl1_kmax
    return {"verbatim": verbatim, "reduced": reduced,
            "algebra_consistent": abs(verbatim - reduced) < 1e-9,
            "passed_point_estimate": reduced > 0}


def reading_A(bpc_l1_k1: float, bpc_divl1_kmax: float) -> dict:
    """并列判据（读法 A，机制问题；契约 README §4.2）：
    在**固定路径机制**下，去相关压力的净效果 `bpc_l1@K1 − bpc_divl1@Kmax > 0`。"""
    v = bpc_l1_k1 - bpc_divl1_kmax
    return {"value": v, "passed_point_estimate": v > 0}
