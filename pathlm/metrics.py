"""M0 probe metrics: ECE, accuracy, ensemble."""

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
def ensemble(logits_conf_pairs):
    """Confidence-weighted token-probability mixture (inference-time math only).
    pairs: iterable of (logits, raw_conf_logits). Weights are the normalized
    calibrated confidences sigmoid(conf) — detached, matching design §5's
    'confidence-weighted average'; mixture preserves multimodality."""
    probs = torch.stack([F.softmax(l, -1) for l, _ in logits_conf_pairs], 0)  # (E,N,T,V)
    confs = torch.stack([c.sigmoid() for _, c in logits_conf_pairs], 0)       # (E,N,T)
    w = confs / confs.sum(0, keepdim=True).clamp_min(1e-6)
    mix = (w.unsqueeze(-1) * probs).sum(0)
    conf_mix = (w * confs).sum(0)
    return mix, conf_mix
