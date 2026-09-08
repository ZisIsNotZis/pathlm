"""Config-driven path sampling. Knobs map 1:1 to docs/design.md stage tunables.

M0 scope: wrong-token corruption, MTP+confidence heads, consistency loss,
direct latent retry. Unimplemented knobs raise NotImplementedError (deferred
to the main-effect runs of docs/experiments.md).
"""

from dataclasses import dataclass, field
import random


@dataclass
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    seq_len: int = 128
    mlp_mult: int = 4
    dropout: float = 0.0
    # Stage E: input norm cap c (embedding rows init to norm ~1)
    norm_cap: float = 1.0


@dataclass
class PathConfig:
    # Stage 0 — token source corruption (rates are per-position probabilities)
    corrupt_wrong: float = 0.0    # replace token with a uniform random vocab token
    corrupt_mask: float = 0.0     # replace token with the [mask] token
    span_mode: str = "iid"        # "iid" | "span" (span deferred past M0)
    # Stage 1 — latent perturbation (rates per-position)
    perturb_noise: float = 0.0    # add Gaussian noise (sigma relative to norm cap)
    pure_noise: float = 0.0       # replace latent entirely with noise (bypass E)
    noise_sigma: float = 0.1
    # Stage L — layer-stack path knobs (sampled once per batch = global schedule)
    shuffle_locality: float = 0.0  # 0 = normal order, 1 = fully random permutation
    p_skip: float = 0.0
    p_redo: float = 0.0
    p_exit: float = 0.0            # early exit (dense supervision) — deferred past M0
    # Stage 2/3 — MTP block and return transport
    n_mtp: int = 2                 # heads k=1..n_mtp (self k=0 always present)
    transport: str = "none"        # "none" | "direct" (linear/soft/decode deferred)
    p_retry: float = 0.0           # P(one latent-retry round), sampled per batch
    # Stage 4 — token retry: deferred past M0
    p_token_retry: float = 0.0
    # Losses
    w_consistency: float = 0.0     # weight of T2 ~ T1@T1 consistency loss


@dataclass
class PathSample:
    """One globally-sampled compute path (shared by all tokens in the batch)."""
    layer_order: list[int] = field(default_factory=list)  # execution order of layer indices
    layer_repeats: list[int] = field(default_factory=list)  # repeats per executed step
    n_retries: int = 0               # latent-retry rounds after the first pass


def sample_path(cfg: PathConfig, rng: random.Random, n_layers: int) -> PathSample:
    """Sample one global path. Called once per batch during training."""
    for knob, val in (("span_mode", cfg.span_mode), ("transport", cfg.transport)):
        allowed = {"span_mode": ("iid",), "transport": ("none", "direct")}[knob]
        if val not in allowed:
            raise NotImplementedError(f"{knob}={val!r} deferred past M0 (allowed: {allowed})")
    if cfg.p_exit > 0:
        raise NotImplementedError("p_exit (dense early-exit supervision) deferred past M0")
    if cfg.p_token_retry > 0:
        raise NotImplementedError("p_token_retry (stage-4 retry) deferred past M0")

    n = n_layers
    # Shuffle: sort key = index + symmetric noise of width locality*n.
    # locality 0 -> keys are the indices themselves (normal order); locality 1
    # -> noise spans +/-n, i.e. an effectively uniform random permutation.
    keys = [i + (rng.random() * 2 - 1) * cfg.shuffle_locality * n for i in range(n)]
    order = sorted(range(n), key=lambda i: keys[i])
    repeats = []
    for _ in order:
        r = 1
        while rng.random() < cfg.p_redo:
            r += 1
        repeats.append(r)
    # Per-layer skip: drop layer i with prob p_skip
    kept = [(i, r) for i, r in zip(order, repeats) if rng.random() >= cfg.p_skip]
    order = [i for i, _ in kept] or [order[0]]  # never drop the entire stack
    repeats = [r for _, r in kept]
    n_retries = 1 if rng.random() < cfg.p_retry else 0
    return PathSample(layer_order=order, layer_repeats=repeats, n_retries=n_retries)
