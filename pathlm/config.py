"""Config-driven path sampling. Knobs map 1:1 to docs/design.md stage tunables.

M1 scope adds: linear/soft transports, dense early-exit supervision, eviction
training with anchor slots, and the needle (copy-from-context) objective used
by the X2 eval. Still unimplemented knobs raise NotImplementedError.
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
    # Dense early-exit supervision (L2): weight of the per-depth MTP losses at
    # depths 1..n_layers-1 (the final depth is always the normal pass). At
    # inference, exit at the first depth whose prob_0 clears exit_threshold.
    w_dense_exit: float = 0.0
    exit_threshold: float = 0.9
    # Eviction (X2): attention may see the last `window` positions plus the
    # first `anchors` positions. 0 = off. Anchors are exempt from eviction and
    # act as the long-range channel (needle-in-anchor eval relies on this).
    window: int = 0
    anchors: int = 0
    # Distance penalty (X1): additive attention bias `-dist_pen * log(1 + d)`
    # on every head, d = query/key distance. Low weight — correctness
    # dominates; the gate is the per-head distance telemetry, not the loss.
    dist_pen: float = 0.0
    # Needle objective (X2): fraction of training batches replaced by the
    # copy-from-context task (see pathlm/data.py needle_batch).
    p_needle: float = 0.0
    # Stage 2/3 — MTP block and return transport
    n_mtp: int = 2                 # heads k=1..n_mtp (self k=0 always present)
    transport: str = "none"        # "none" | "direct" | "linear" | "soft"
    p_retry: float = 0.0           # P(one latent-retry round), sampled per batch
    # Stage 4 — token retry: discrete re-entry (re-embed the self node's
    # predicted correction, re-run the stack). One coin per batch.
    p_token_retry: float = 0.0
    # Mixture re-entry (design §3 amendment): instead of overwriting, the
    # re-entry state is a confidence-weighted accumulator over all rounds
    # (init anchor w=1, detached weights). Applies to latent transports
    # (weighted mean of latents / expected embedding of the accumulated
    # distribution) and to the token round (re-embed argmax of the mixture).
    reentry_mix: bool = False
    # Losses
    w_consistency: float = 0.0     # weight of T2 ~ T1@T1 consistency loss


@dataclass
class PathSample:
    """One globally-sampled compute path (shared by all tokens in the batch)."""
    layer_order: list[int] = field(default_factory=list)  # execution order of layer indices
    layer_repeats: list[int] = field(default_factory=list)  # repeats per executed step
    n_retries: int = 0               # latent-retry rounds after the first pass
    token_retry: bool = False        # stage-4 discrete round after latent rounds


def sample_path(cfg: PathConfig, rng: random.Random, n_layers: int) -> PathSample:
    """Sample one global path. Called once per batch during training."""
    for knob, val, allowed in (("span_mode", cfg.span_mode, ("iid",)),
                               ("transport", cfg.transport, ("none", "direct", "linear", "soft"))):
        if val not in allowed:
            raise NotImplementedError(f"{knob}={val!r} not implemented (allowed: {allowed})")
    if cfg.p_needle > 0 and cfg.window == 0:
        raise ValueError("needle batches are an eviction-training element: set window > 0")
    if cfg.p_retry > 0 and cfg.transport == "none":
        raise ValueError("latent retry needs a transport (retry without a return channel is a no-op)")

    n = n_layers
    # Shuffle: sort key blends the index with i.i.d. uniform keys.
    # locality 0 -> keys are the indices (normal order); locality 1 -> keys are
    # i.i.d. U(0, n), i.e. an exactly uniform random permutation.
    keys = [(1 - cfg.shuffle_locality) * i + cfg.shuffle_locality * rng.random() * n
            for i in range(n)]
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
    token_retry = rng.random() < cfg.p_token_retry
    return PathSample(layer_order=order, layer_repeats=repeats, n_retries=n_retries,
                      token_retry=token_retry)
