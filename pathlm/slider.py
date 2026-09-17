"""Slider machinery: the runtime allocator over the quality-compute frontier
(docs/mental_model.md §3). Scale parameters (model size, n_mtp, max depth)
fix the frontier's *shape*; the Slider theta = (k, tau_retry) moves along it.

Pure pieces (unit-tested in tests/test_slider.py, no GPU):

- ``fire_curve``      P(prob0 < tau) from a calibration prob0 sample.
- ``spec_cost``       v1 cost model — equivalent width-1 forwards per emitted
                      token (WORKSPACE convention: one retry round = one
                      equivalent forward; a width-(1+k) batched verify pass
                      costs 1+k units and emits 1+a2 tokens on average).
- ``Frontier``        interpolation over measured (k, tau) points + budget
                      solve / quality solve.
- ``reliability``     equal-count prob0 bins vs realized accuracy — the
                      calibration curve behind the online-quality proxy
                      (mean prob0 ≈ accuracy when ECE-calibrated).
- ``proxy_summary``   aggregate a generated window's prob0 log into the
                      online proxy values.

GPU probe (``gate_probe``): one teacher-forced two-round pass returning flat
per-position arrays (prob0, per-round hit/nll, corruption mask, spec-draft
agreement) so the WHOLE tau axis — fire curve, gated quality, reliability —
is computed from a single forward pair by the pure helpers. Deployment
semantics: round-2 re-entry uses the soft-transport overwrite exactly as
``Decoder.retry`` does, so ``reentry_mix`` is disabled inside the probe
(a deliberate, documented mismatch with training-time mixture re-entry).

Budget currency note (honest bookkeeping): cost_fwd counts equivalent
width-1 forwards per token — hardware-free, matching the WORKSPACE v1
convention. Wall-clock tok/s is measured alongside but NEVER mixed into
cost_fwd (the width-2 batched forward is ~1.2-1.5x a width-1 forward in
wall-clock on the shared 4090, not 2x — batching amortizes kernel/python
overhead; the two currencies disagree and both are reported).
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import sample_path
from .data import batch
from .eval import eval_pc

LN2 = math.log(2)


# ---------------------------------------------------------------- pure pieces

def fire_curve(probs, taus) -> dict[float, float]:
    """P(prob0 < tau) per tau from a 1-D sample of prob0 values."""
    p = torch.as_tensor(probs, dtype=torch.float32).flatten()
    if p.numel() == 0:
        raise ValueError("fire_curve needs a non-empty prob0 sample")
    return {float(t): float((p < t).float().mean()) for t in taus}


def spec_cost(k: int, fire_rate: float, a2: float) -> float:
    """FLOP-normalized cost: equivalent width-1 forwards per emitted token.

    k = 0 (plain): one width-1 pass per token, plus fire_rate retry rounds
    (each a width-1 forward).
    k >= 1 (spec): each loop runs ONE width-(1+k) batched verify pass
    (cost 1+k units) emitting 1+a2 tokens on average; a fired retry round is
    a width-1 forward before the verify pass. a2 = P(spec draft accepted).

    See `forwards_cost` for the v1 budget currency (mental_model §1 defines
    the budget as FORWARD COUNT per token x depth — spec amortizes and can
    go below 1; the two currencies disagree and both are reported)."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not 0.0 <= a2 <= 1.0:
        raise ValueError(f"accept rate a2 out of [0,1]: {a2}")
    if fire_rate < 0:
        raise ValueError(f"fire rate must be >= 0, got {fire_rate}")
    if k == 0:
        return 1.0 + fire_rate
    return (1 + k + fire_rate) / (1.0 + a2)


def forwards_cost(k: int, fire_rate: float, a2: float) -> float:
    """v1 budget currency (mental_model §1: budget = forwards/token x depth;
    all-levers-here run full depth, so cost = forward count per emitted
    token). Plain: 1 + fired retries. Spec: 1 verify forward per loop
    emitting 1+a2 tokens on average, +1 forward when the retry fires."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if not 0.0 <= a2 <= 1.0:
        raise ValueError(f"accept rate a2 out of [0,1]: {a2}")
    if fire_rate < 0:
        raise ValueError(f"fire rate must be >= 0, got {fire_rate}")
    if k == 0:
        return 1.0 + fire_rate
    return (1.0 + fire_rate) / (1.0 + a2)


COST_CURRENCIES = {"forwards": forwards_cost, "flop": spec_cost}


def _better(a: float, b: float, key: str) -> bool:
    """Is quality value a better than b for metric `key`? (*_bpc/*_nll: lower
    is better; *_acc and raw means: higher is better.)"""
    return a < b if key.endswith(("_bpc", "_nll")) else a > b


@dataclass
class FrontierPoint:
    """One slider operating point: predicted cost + measured quality +
    measured decode behaviour. tau=None means retry off (fire rate 0).
    cost_forwards = v1 budget currency (forward count/token);
    cost_flop = equivalent width-1 units/token (FLOP-normalized)."""
    k: int
    tau: float | None
    fire_rate: float | None          # calibration P(prob0<tau); 0.0 for retry-off
    a2: float | None                 # predicted spec accept rate; None for k=0
    cost_forwards: float             # predicted forwards / emitted token
    quality: dict = field(default_factory=dict)    # from the gate probe (quality split)
    measured: dict = field(default_factory=dict)   # from decode bench (held-out prompts)
    cost_flop: float = 0.0           # FLOP-normalized view (secondary)

    @property
    def tau_eff(self) -> float:
        """Retry-off is the tau->0 limit of the gate axis (never fires)."""
        return 0.0 if self.tau is None else float(self.tau)

    def label(self) -> str:
        t = "off" if self.tau is None else f"{self.tau:g}"
        return f"k={self.k} tau={t}"


class Frontier:
    """Measured slider grid with linear-in-tau interpolation per k.

    Quality and fire rate are interpolated linearly between neighbouring
    measured taus; cost is recomputed from the interpolated fire rate through
    the v1 cost model (monotone: cost rises with tau — more firing).
    """

    def __init__(self, points: list[FrontierPoint]):
        self.points = sorted(points, key=lambda p: (p.k, p.tau_eff))
        by_k: dict[int, list[FrontierPoint]] = {}
        for p in self.points:
            by_k.setdefault(p.k, []).append(p)
        if not by_k:
            raise ValueError("empty frontier")
        self._by_k = by_k

    def ks(self) -> list[int]:
        return sorted(self._by_k)

    def _series(self, k: int, key: str | None, currency: str = "forwards") -> list[tuple[float, float]]:
        """[(tau_eff, value)] for cost (key=None) or quality[key], sorted by
        tau, None-quality points dropped."""
        out = []
        for p in self._by_k[k]:
            if key is None:
                v = p.cost_forwards if currency == "forwards" else p.cost_flop
            else:
                v = p.quality.get(key)
                if v is None:
                    continue
            out.append((p.tau_eff, float(v)))
        return sorted(out)

    @staticmethod
    def _interp(series: list[tuple[float, float]], tau: float) -> float:
        """Piecewise-linear value at tau (clamped to the series' range)."""
        if len(series) == 1:
            return series[0][1]
        if tau <= series[0][0]:
            return series[0][1]
        if tau >= series[-1][0]:
            return series[-1][1]
        for (t0, v0), (t1, v1) in zip(series, series[1:]):
            if t0 <= tau <= t1:
                w = 0.0 if t1 == t0 else (tau - t0) / (t1 - t0)
                return v0 + w * (v1 - v0)
        return series[-1][1]

    def cost_at(self, k: int, tau: float, currency: str = "forwards",
                a2: float | None = None) -> float:
        """Predicted cost at arbitrary tau via interpolated fire rate."""
        fire = self._interp(self._fire_series(k), tau)
        if a2 is None:
            a2 = self._by_k[k][0].a2 or 0.0
        return COST_CURRENCIES[currency](k, fire, a2)

    def _fire_series(self, k: int) -> list[tuple[float, float]]:
        return [(p.tau_eff, p.fire_rate or 0.0) for p in self._by_k[k]]

    def quality_at(self, k: int, tau: float, key: str) -> float | None:
        s = self._series(k, key)
        return None if not s else self._interp(s, tau)

    def _candidate(self, k: int, tau: float, key: str,
                   currency: str = "forwards") -> FrontierPoint:
        """Interpolated point (predicted cost + interpolated quality)."""
        a2 = self._by_k[k][0].a2
        fire = self._interp(self._fire_series(k), tau)
        cost_f = forwards_cost(k, fire, a2 or 0.0)
        cost_p = spec_cost(k, fire, a2 or 0.0)
        q = self.quality_at(k, tau, key)
        return FrontierPoint(k=k, tau=tau, fire_rate=fire, a2=a2,
                             cost_forwards=cost_f, cost_flop=cost_p,
                             quality={key: q} if q is not None else {})

    def solve_budget(self, budget: float, key: str = "corrupt_bpc",
                     currency: str = "forwards") -> FrontierPoint:
        """Best-quality point with predicted cost (in `currency`) <= budget.

        Feasible set = measured points +, per k, the interpolated tau where
        cost(tau) = budget exactly (if it lies inside the measured tau span).
        Ties in quality break to lower cost, then lower k, then higher tau.
        """
        if budget <= 0:
            raise ValueError("budget must be positive")
        best: FrontierPoint | None = None
        for k in self.ks():
            cost_series = self._series(k, None, currency)
            taus = {t for t, _ in cost_series}
            lo, hi = cost_series[0][1], cost_series[-1][1]
            if lo <= budget <= hi and lo < hi:
                # invert cost(tau) = budget on the piecewise-linear series
                for (t0, c0), (t1, c1) in zip(cost_series, cost_series[1:]):
                    if min(c0, c1) <= budget <= max(c0, c1) and c0 != c1:
                        w = (budget - c0) / (c1 - c0)
                        taus.add(t0 + w * (t1 - t0))
            for t in taus:
                c = self.cost_at(k, t, currency)
                if c > budget + 1e-9:
                    continue
                cand = self._candidate(k, t, key, currency)
                if cand.quality.get(key) is None:
                    continue
                cand_cost = cand.cost_forwards if currency == "forwards" else cand.cost_flop
                if best is None or _better(cand.quality[key], best.quality[key], key) \
                        or (cand.quality[key] == best.quality[key] and cand_cost < best_cost):
                    best, best_cost = cand, cand_cost
        if best is None:
            floor = min((p.cost_forwards if currency == "forwards" else p.cost_flop)
                        for p in self.points)
            raise ValueError(
                f"budget {budget} below the cheapest frontier point "
                f"({floor:.3f} {currency}/tok in currency {currency!r})")
        return best

    def solve_quality(self, q_target: float, key: str = "corrupt_bpc",
                      currency: str = "forwards") -> FrontierPoint:
        """Cheapest point reaching quality[key] >= q_target (for *_bpc keys:
        bpc <= q_target). Raises when no measured point reaches the target."""
        best: FrontierPoint | None = None
        best_cost = float("inf")
        for p in self.points:
            q = p.quality.get(key)
            if q is None:
                continue
            ok = q <= q_target if key.endswith(("_bpc", "_nll")) else q >= q_target
            if not ok:
                continue
            c = p.cost_forwards if currency == "forwards" else p.cost_flop
            if c < best_cost:
                best, best_cost = p, c
        if best is None:
            raise ValueError(f"no frontier point reaches {key} target {q_target}")
        return best


def reliability(probs, hits, n_bins: int = 10) -> dict:
    """Equal-count prob0 bins vs realized accuracy — the proxy's calibration
    curve. Returns per-bin stats + overall mean_prob/acc/ece (absolute gap)."""
    p = torch.as_tensor(probs, dtype=torch.float32).flatten()
    h = torch.as_tensor(hits, dtype=torch.float32).flatten()
    if p.numel() != h.numel() or p.numel() == 0:
        raise ValueError("reliability needs equal-length non-empty prob0/hits")
    order = torch.argsort(p)
    bins, n = [], p.numel()
    for i in range(n_bins):
        sl = order[i * n // n_bins:(i + 1) * n // n_bins]
        if sl.numel() == 0:
            continue
        bins.append({"mean_prob": round(float(p[sl].mean()), 4),
                     "acc": round(float(h[sl].mean()), 4),
                     "n": int(sl.numel())})
    mean_prob, acc = float(p.mean()), float(h.mean())
    ece = sum(abs(b["mean_prob"] - b["acc"]) * b["n"] for b in bins) / n
    return {"bins": bins, "mean_prob": round(mean_prob, 4), "acc": round(acc, 4),
            "ece": round(ece, 4)}


def proxy_summary(prob0_log: list[dict]) -> dict:
    """Online proxy over a generated window (entries from decode(prob0_log=)).

    proxy = mean PRE-retry prob0 over emitted tokens — the value an online
    observer sees before deciding to retry. Also reports the post-retry
    value on fired positions and the fire fraction."""
    if not prob0_log:
        raise ValueError("empty prob0 log")
    pre = [e["prob0_pre"] for e in prob0_log for _ in range(e["n_emitted"])]
    fired = [e for e in prob0_log if e["fired"]]
    post = [e["prob0_post"] for e in fired for _ in range(e["n_emitted"])]
    return {"proxy_mean_prob0": round(sum(pre) / len(pre), 4),
            "proxy_mean_prob0_post": round(sum(post) / len(post), 4) if post else None,
            "fire_fraction": round(len(fired) / len(prob0_log), 4),
            "n_emitted": int(sum(e["n_emitted"] for e in prob0_log))}


def gated_quality(probe: dict[str, torch.Tensor], taus) -> dict:
    """Per-tau gated quality + effective-proxy stats from a gate_probe.

    Deployment gate semantics: the prediction of t_{i+1} comes from the
    refined (round-2) row iff prob0 < tau. The effective proxy substitutes
    round-2 prob0 on fired positions — the confidence an online observer
    would compute after the gate has acted. Returns per-tau dicts plus a
    retry-off (tau None) row."""
    p0, p0r2 = probe["prob0"], probe["prob0_r2"]
    corrupt = probe["corrupt"] > 0.5
    rows: dict = {}
    taus = [None] + [float(t) for t in taus]
    for tau in taus:
        gate = torch.zeros_like(p0, dtype=torch.bool) if tau is None else (p0 < tau)
        nll = torch.where(gate, probe["nll_r2"], probe["nll_r1"])
        hit = torch.where(gate, probe["hit_r2"], probe["hit_r1"])
        eff_conf = torch.where(gate, p0r2, p0)
        row = {
            "fire_rate": round(float(gate.float().mean()), 4),
            "gated_bpc": round(float(nll.mean()) / LN2, 4),
            "gated_acc": round(float(hit.mean()), 4),
            "proxy_eff_mean": round(float(eff_conf.mean()), 4),
        }
        if corrupt.any():
            row["gated_acc_corr"] = round(float(hit[corrupt].mean()), 4)
            row["gated_acc_cleanpos"] = round(float(hit[~corrupt].mean()), 4)
        rows["off" if tau is None else tau] = row
    return rows


def pearson(x, y) -> float:
    """Pearson correlation of two equal-length samples (pure)."""
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    y = torch.as_tensor(y, dtype=torch.float64).flatten()
    if x.numel() != y.numel() or x.numel() < 2:
        raise ValueError("pearson needs equal-length samples (n >= 2)")
    xs, ys = x - x.mean(), y - y.mean()
    denom = xs.norm() * ys.norm()
    if denom == 0:
        return 0.0
    return float((xs @ ys / denom).item())


# ----------------------------------------------------------------- GPU probe

@torch.no_grad()
def gate_probe(model, eval_arr, vocab_size: int, n_batches: int = 24,
               batch_size: int = 32, corrupted: bool = False, seed: int = 0,
               mixture_round2: bool = False) -> dict[str, torch.Tensor]:
    """Teacher-forced two-round probe (the whole tau axis from one forward pair).

    Round 2 re-enters via the soft-transport overwrite — exactly the state
    ``Decoder.retry`` produces at decode time — by disabling ``reentry_mix``
    for the probe (documented deployment/training mismatch avoidance).
    ``mixture_round2=True`` instead keeps the training-time mixture re-entry:
    an ORACLE reference the decode engine cannot currently produce (recorded
    engine gap), used to quantify what a mixture-aware ``Decoder.retry``
    would buy.

    Returns flat per-position arrays over rows 0..T-2 (the predictors of
    t_{i+1}): prob0 / prob0_r2 (round-1/2 node-0 conf), hit0_r1/hit0_r2
    (node-0 argmax == current token — the realized correctness prob0
    estimates), hit_r1/hit_r2 (node-1 next-token argmax hit), nll_r1/nll_r2
    (per-position nats), corrupt (input corruption mask), and hit_spec
    (node-1's argmax at i+1 agrees with node-2's draft at i — the
    teacher-forced spec accept-rate estimate)."""
    T = model.mcfg.seq_len
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    torch.manual_seed(1234)  # eval.repair corruption convention
    out: dict[str, list] = {k: [] for k in
                            ("prob0", "prob0_r2", "hit0_r1", "hit0_r2",
                             "hit_r1", "hit_r2", "nll_r1", "nll_r2",
                             "corrupt", "hit_spec")}
    with eval_pc(model, w_dense_exit=0.0, p_retry=0.0, p_token_retry=0.0,
                 reentry_mix=mixture_round2,
                 corrupt_wrong=(0.075 if corrupted else 0.0),
                 corrupt_mask=(0.075 if corrupted else 0.0)):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            x = x.to(model.embed.weight.device)
            tgt = x[:, 1:]
            paths = [sample_path(model.pcap, rng, model.mcfg.n_layers) for _ in range(2)]
            for p in paths:
                p.n_retries = 0
                p.token_retry = False
            _, aux = model(x, paths, x)
            mask = aux["corrupt_mask"][:, :T - 1]
            conf0 = torch.sigmoid(aux["rounds"][0][0]["conf"][:, :T - 1])
            conf0_r2 = torch.sigmoid(aux["rounds"][1][0]["conf"][:, :T - 1])
            cur = x[:, :T - 1]
            a0 = aux["rounds"][0][0]["logits"][:, :T - 1].argmax(-1)
            a0_r2 = aux["rounds"][1][0]["logits"][:, :T - 1].argmax(-1)
            n1 = aux["rounds"][0][1]["logits"][:, :T - 1]
            n2 = aux["rounds"][0][2]["logits"][:, :T - 1] if 2 in aux["rounds"][0] else None
            l1 = -F.log_softmax(n1.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            r2 = aux["rounds"][1][1]["logits"][:, :T - 1]
            l2 = -F.log_softmax(r2.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            a1, a2m = n1.argmax(-1), r2.argmax(-1)
            hit_spec = torch.zeros_like(a1)
            if n2 is not None:
                draft = n2.argmax(-1)          # row i drafts t_{i+2}
                hit_spec[:, :-1] = (a1[:, 1:] == draft[:, :-1])
            for key, val in (("prob0", conf0), ("prob0_r2", conf0_r2),
                             ("hit0_r1", (a0 == cur).float()),
                             ("hit0_r2", (a0_r2 == cur).float()),
                             ("hit_r1", (a1 == tgt).float()),
                             ("hit_r2", (a2m == tgt).float()), ("nll_r1", l1),
                             ("nll_r2", l2), ("corrupt", mask.float()),
                             ("hit_spec", hit_spec.float())):
                out[key].append(val.reshape(-1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}
