"""M0 probe metrics: ECE, accuracy, ensemble-vs-best-single, retry effect."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(-1) == targets).float().mean().item()


@torch.no_grad()
def ece(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error: |avg confidence - accuracy| per bin, weighted."""
    conf, corr = confidence.flatten().float().cpu(), correct.flatten().float().cpu()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(corr[m].mean().item() - conf[m].mean().item())
    return ece


@torch.no_grad()
def brier(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    return (confidence.flatten().float().cpu() - correct.flatten().float().cpu()).pow(2).mean().item()


@torch.no_grad()
def ensemble(logits_conf_pairs):
    """Confidence-weighted token-probability mixture (inference-time math only).
    pairs: iterable of (logits, conf_probability). Weights = softmax over raw
    confidence logits (log-space combination per docs/design.md §5)."""
    logits_list = [l for l, _ in logits_conf_pairs]
    raw = torch.stack([c for _, c in logits_conf_pairs], 0)
    w = raw.softmax(0).unsqueeze(-1)                       # (E, B, T)
    probs = torch.stack([F.softmax(l, -1) for l in logits_list], 0)
    mix = (w.unsqueeze(-1) * probs).sum(0)                 # (B, T, V)
    conf_mix = (w * torch.stack([c.sigmoid() for _, c in logits_conf_pairs], 0)).sum(0)
    return mix, conf_mix
