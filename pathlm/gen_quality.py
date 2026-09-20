"""Free-generation quality signals (ticket 22): teacher-forced multi-path
re-scoring disagreement.

Hypothesis (falsifiable): the mean pairwise JS divergence between node-1
distributions from independently sampled layer paths — computed by
RE-SCORING a fixed token window (never generating) — is a ground-truth-free
output-quality signal on free-generated segments, where mean prob0 fails
(ticket 15: streamed corruption pushes free generation into high-confidence
repetition attractors and prob0 reads REVERSED: 0.976 corrupted vs 0.954
clean; prob0 measures context self-consistency, not output quality).

Protocol constraint that shapes the GPU probe: every path is scored by its
own teacher-forced forward, with the stage-0 corruption seed REPLAYED
(``torch.manual_seed(seed)``) before each forward, so all paths read the
bitwise-identical input and their disagreement measures the computation
path, never input noise. Passing several paths to ONE forward would NOT
do this: round r>0 re-enters from the PREVIOUS round's latent (retry
semantics) and re-runs the stack on it — deepening the computation, not
re-scoring the same input.

Pure pieces (unit-tested in tests/test_gen_quality.py, no GPU):

- ``js_bits``            JS divergence (bits) between probability tensors.
- ``pairwise_js``        mean pairwise JS over the path dimension.
- ``pairwise_agreement`` mean pairwise argmax agreement over the path dim.
- ``discrimination_auc`` Mann–Whitney AUC of a signal vs a miss indicator.
- ``rank_corr``          Spearman rank correlation (average-rank ties).
- ``signal_bins``        equal-count bins of a signal vs realized miss rate.
- ``monotone_frac``      adjacent-bin error-rate monotonicity (detecting dir).

GPU probe:

- ``rescore_disagreement``  teacher-forced multi-path pass over token
  windows; returns per-position js / argmax agreement / prob0 / realized
  next-token hit.
"""

from __future__ import annotations

import random

import torch

from .config import sample_path
from .eval import eval_pc
from .slider import pearson

EPS = 1e-8


# ---------------------------------------------------------------- pure pieces

def js_bits(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Jensen–Shannon divergence in BITS between probability tensors [..., V].
    Symmetric, bounded by 1.0; zero entries are clamped to eps (a zero entry
    then contributes ~eps*log2(eps/m) ~ 0, never -inf)."""
    if p.shape != q.shape:
        raise ValueError(f"js_bits needs equal shapes, got {tuple(p.shape)} vs {tuple(q.shape)}")
    m = 0.5 * (p + q)

    def kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.clamp_min(eps)
        return (a * (a / b.clamp_min(eps)).log2()).sum(-1)

    return 0.5 * (kl(p, m) + kl(q, m))


def pairwise_js(probs: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Mean pairwise JS (bits) across dim 0 — the path dimension.
    probs: [P, ..., V] with P >= 2; returns the trailing shape."""
    if probs.dim() < 2 or probs.shape[0] < 2:
        raise ValueError("pairwise_js needs a path dim >= 2 at dim 0")
    vals = [js_bits(probs[i], probs[j], eps)
            for i in range(probs.shape[0]) for j in range(i + 1, probs.shape[0])]
    return torch.stack(vals).mean(0)


def pairwise_agreement(hard: torch.Tensor) -> torch.Tensor:
    """Mean pairwise equality across dim 0. hard: [P, ...] argmax ids."""
    if hard.dim() < 1 or hard.shape[0] < 2:
        raise ValueError("pairwise_agreement needs a path dim >= 2 at dim 0")
    vals = [(hard[i] == hard[j]).float()
            for i in range(hard.shape[0]) for j in range(i + 1, hard.shape[0])]
    return torch.stack(vals).mean(0)


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks 1..n (float64) with ties sharing their average rank."""
    x = x.to(torch.float64)
    n = x.numel()
    order = torch.argsort(x)
    sx = x[order]
    new_group = torch.ones(n, dtype=torch.bool)
    if n > 1:
        new_group[1:] = sx[1:] != sx[:-1]
    group = torch.cumsum(new_group.to(torch.int64), 0) - 1
    n_groups = int(group[-1]) + 1
    ranks1n = torch.arange(1, n + 1, dtype=torch.float64)
    sums = torch.zeros(n_groups, dtype=torch.float64).index_add_(0, group, ranks1n)
    cnts = torch.zeros(n_groups, dtype=torch.float64).index_add_(
        0, group, torch.ones(n, dtype=torch.float64))
    out = torch.empty(n, dtype=torch.float64)
    out[order] = (sums / cnts)[group]
    return out


def rank_corr(x, y) -> float:
    """Spearman rank correlation of two equal-length samples (ties averaged)."""
    a = torch.as_tensor(x, dtype=torch.float32).flatten()
    b = torch.as_tensor(y, dtype=torch.float32).flatten()
    if a.numel() != b.numel() or a.numel() < 2:
        raise ValueError("rank_corr needs equal-length samples (n >= 2)")
    return pearson(_average_ranks(a), _average_ranks(b))


def discrimination_auc(score, miss) -> float:
    """P(a randomly chosen miss scores higher than a randomly chosen hit),
    ties 0.5 (Mann–Whitney; rank-based, memory-safe at large n).
    `score` should be HIGH where the position is WRONG. 0.5 = no signal."""
    s = torch.as_tensor(score, dtype=torch.float32).flatten()
    m = torch.as_tensor(miss, dtype=torch.float32).flatten()
    if s.numel() != m.numel() or s.numel() == 0:
        raise ValueError("discrimination_auc needs equal-length non-empty inputs")
    pos_mask = m > 0.5
    n_pos, n_neg = int(pos_mask.sum()), int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("discrimination_auc needs both classes present")
    r_pos_sum = float(_average_ranks(s)[pos_mask].sum())
    return round((r_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def signal_bins(signal, miss, n_bins: int = 10) -> list[dict]:
    """Equal-count bins ascending in `signal` with the realized error rate —
    the same equal-count binning convention as slider.reliability."""
    s = torch.as_tensor(signal, dtype=torch.float32).flatten()
    m = torch.as_tensor(miss, dtype=torch.float32).flatten()
    if s.numel() != m.numel() or s.numel() == 0:
        raise ValueError("signal_bins needs equal-length non-empty inputs")
    order = torch.argsort(s)
    n = s.numel()
    bins = []
    for i in range(n_bins):
        sl = order[i * n // n_bins:(i + 1) * n // n_bins]
        if sl.numel() == 0:
            continue
        bins.append({"mean_signal": round(float(s[sl].mean()), 6),
                     "err_rate": round(float(m[sl].mean()), 4),
                     "n": int(sl.numel())})
    return bins


def monotone_frac(bins: list[dict]) -> float:
    """Fraction of adjacent bin pairs whose error rate moves UP with the
    signal (bins ascend in signal, so up = detecting direction)."""
    if len(bins) < 2:
        raise ValueError("monotone_frac needs at least 2 bins")
    errs = [b["err_rate"] for b in bins]
    ups = sum(1 for a, b in zip(errs, errs[1:]) if b > a)
    return round(ups / (len(errs) - 1), 4)


def joint_high_conf_bucketing(js, prob0, miss) -> dict:
    """Complementary criterion A reading: within the most-confident prob0
    half (>= median — where prob0 claims 'all fine'), do js terciles still
    stratify the error rate? Returns tercile error rates + monotonicity."""
    js = torch.as_tensor(js, dtype=torch.float32).flatten()
    prob0 = torch.as_tensor(prob0, dtype=torch.float32).flatten()
    miss = torch.as_tensor(miss, dtype=torch.float32).flatten()
    hi = prob0 >= prob0.median()
    if int(hi.sum()) < 3:
        raise ValueError("joint bucketing needs a non-trivial confident half")
    js_hi, miss_hi = js[hi], miss[hi]
    e1, e2 = torch.quantile(js_hi, torch.tensor([1.0 / 3, 2.0 / 3]))
    terciles = (js_hi <= e1, (js_hi > e1) & (js_hi <= e2), js_hi > e2)
    errs = [round(float(miss_hi[t].mean()), 4) for t in terciles]
    return {"err_by_js_tercile": errs,
            "monotone": bool(errs[0] < errs[1] < errs[2]),
            "spread": round(errs[2] - errs[0], 4),
            "n": int(hi.sum())}


# ----------------------------------------------------------------- GPU probe

@torch.no_grad()
def rescore_disagreement(model, windows: torch.Tensor, n_paths: int = 2,
                         corrupt_wrong: float = 0.0, corrupt_mask: float = 0.0,
                         seed: int = 0) -> dict[str, torch.Tensor]:
    """Teacher-forced multi-path re-score of token windows [B, T'] (T' <= seq_len).

    Each path is ONE teacher-forced forward with its own freshly sampled
    layer path (n_retries=0, no token retry), and the stage-0 corruption
    seed is REPLAYED (``torch.manual_seed(seed)``) before every forward —
    all paths read the bitwise-identical corrupted input, so the returned
    disagreement is pure computation-path signal (module docstring records
    why a single multi-round forward would measure deepening instead).

    ``seed`` fixes BOTH the corruption draw and the path sampling, so a
    call is reproducible; give different windows/batches different seeds.

    Returns per-position CPU tensors shaped [B, T'-1] (row i predicts the
    input token i+1):
      js       mean pairwise node-1 JS (bits) across paths
      agree    mean pairwise node-1 argmax agreement across paths
      prob0    node-0 confidence of the FIRST path (the ticket-15 proxy,
               same rows — path-to-path prob0 spread is negligible vs its
               per-position range; pinned by the identical-path test)
      hit      first path's node-1 argmax == next clean token (realized
               correctness)
    """
    if n_paths < 2:
        raise ValueError("rescore_disagreement needs n_paths >= 2")
    T = windows.shape[1]
    if T > model.mcfg.seq_len:
        raise ValueError(f"window length {T} exceeds seq_len {model.mcfg.seq_len}")
    if T < 2:
        raise ValueError("window must have at least 2 tokens")
    rng = random.Random(seed)
    x = windows.to(model.embed.weight.device)
    probs = []
    aux = None
    with eval_pc(model, transport="none", reentry_mix=False, p_retry=0.0,
                 p_token_retry=0.0, w_dense_exit=0.0, retry_gate=0.0,
                 corrupt_wrong=corrupt_wrong, corrupt_mask=corrupt_mask):
        for _ in range(n_paths):
            p = sample_path(model.pcap, rng, model.mcfg.n_layers)
            p.n_retries = 0
            p.token_retry = False
            torch.manual_seed(seed)  # replay: identical corruption draw per path
            _, aux = model(x, [p], x)
            probs.append(aux["rounds"][0][1]["logits"][:, :T - 1].float().softmax(-1))
    probs = torch.stack(probs)                                # [P, B, T-1, V]
    hard = probs.argmax(-1)                                   # [P, B, T-1]
    return {"js": pairwise_js(probs).cpu(),                   # [B, T-1]
            "agree": pairwise_agreement(hard).cpu(),
            "prob0": torch.sigmoid(aux["rounds"][0][0]["conf"][:, :T - 1]).cpu(),
            "hit": (hard[0] == x[:, 1:]).float().cpu()}
