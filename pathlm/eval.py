"""M1 evaluation battery: bpc (external anchor), repair/rounds curves,
quality-vs-depth curve, needle accuracy vs distance, decode speed, ensemble.
Each metric auto-skips when the run's config makes it meaningless."""

import contextlib, time

import torch
import torch.nn.functional as F

import random

from .config import PathConfig, sample_path
from .data import needle_batch, batch
from .metrics import ece, ensemble
from .model import PathLM


@contextlib.contextmanager
def eval_pc(model: PathLM, **overrides):
    """Temporarily swap the model's PathConfig (corruption and knobs are read
    from it inside forward). Restores the training config after."""
    saved = model.pcap
    model.pcap = PathConfig(**{**saved.__dict__, **overrides})
    try:
        yield model.pcap
    finally:
        model.pcap = saved


@torch.no_grad()
def bpc(model: PathLM, eval_arr, vocab_size: int, n_batches: int = 200,
        batch_size: int = 32, generator: torch.Generator | None = None) -> dict:
    """Next-token (node 1) CE on the clean eval stream — the published-bits-
    per-char anchor. Random-byte guess = log2(V); published small-model range
    ~1.4–1.6 for this size class."""
    T = model.mcfg.seq_len
    generator = generator or torch.Generator().manual_seed(7)
    nats, tokens, correct = 0.0, 0, 0
    with eval_pc(model, corrupt_wrong=0.0, corrupt_mask=0.0, p_retry=0.0,
                 p_token_retry=0.0, w_dense_exit=0.0, perturb_noise=0.0, pure_noise=0.0):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [model.pcap and sample_path(model.pcap, random, model.mcfg.n_layers)], x)
            logits = aux["rounds"][0][1]["logits"][:, :-1]  # node 1, rows 0..T-2
            tgt = x[:, 1:]
            nats += F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1),
                                    reduction="sum").item()
            correct += (logits.argmax(-1) == tgt).sum().item()
            tokens += tgt.numel()
    return {"bpc": round(nats / tokens / 0.6931471805599453, 4),
            "next_token_acc": round(correct / tokens, 4)}


@torch.no_grad()
def repair(model: PathLM, eval_arr, corrupt_rate: float, n_batches: int = 40,
           batch_size: int = 32, generator: torch.Generator | None = None) -> dict:
    """Self-node repair accuracy on corrupted positions, per round."""
    T = model.mcfg.seq_len
    rng = random.Random(1)
    generator = generator or torch.Generator().manual_seed(1)
    masks, hits = [], {r: [] for r in range(2)}
    with eval_pc(model, corrupt_wrong=corrupt_rate, corrupt_mask=0.0, p_retry=0.0,
                 p_token_retry=0.0, w_dense_exit=0.0):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            paths = [sample_path(model.pcap, rng, model.mcfg.n_layers) for _ in range(2)]
            _, aux = model(x, paths, x)
            masks.append(aux["corrupt_mask"].cpu())
            for r in range(2):
                pred = aux["rounds"][r][0]["logits"].argmax(-1).cpu()
                hits[r].append((pred == x.cpu()).float())
    cm = torch.cat(masks)
    out = {}
    for r in range(2):
        h = torch.cat(hits[r])
        out[f"round{r}"] = {"self_acc_all": round(h.mean().item(), 4),
                            "repair_acc": round(h[cm].mean().item(), 4),
                            "n_corrupted": int(cm.sum().item())}
    return out


@torch.no_grad()
def depth_curve(model: PathLM, eval_arr, n_batches: int = 40, batch_size: int = 32,
                generator: torch.Generator | None = None) -> list[float]:
    """Per-depth node-0+1 CE (mean over components) for dense-exit models.
    aux['depth_ce'] holds depths 1..n of the base pass."""
    T = model.mcfg.seq_len
    rng = random.Random(2)
    generator = generator or torch.Generator().manual_seed(2)
    curves = []
    with eval_pc(model, corrupt_wrong=0.0, p_retry=0.0, p_token_retry=0.0,
                 w_dense_exit=1.0):  # hook must fire to record the curve
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, rng, model.mcfg.n_layers)], x)
            curves.append(torch.stack(aux["depth_ce"]))
    return [round(v, 4) for v in torch.stack(curves).mean(0).tolist()]


@torch.no_grad()
def needle_acc(model: PathLM, n_real_tokens: int, dists=(16, 64, 128, 256, 480),
               batch_size: int = 64, generator: torch.Generator | None = None) -> dict:
    """Node-1 accuracy at the query row of needle batches, per distance.
    Meaningful only for runs trained with p_needle > 0 (the convention is
    learned, not known)."""
    T = model.mcfg.seq_len
    generator = generator or torch.Generator().manual_seed(3)
    out = {}
    with eval_pc(model, corrupt_wrong=0.0, p_retry=0.0, p_token_retry=0.0, w_dense_exit=0.0):
        for d in dists:
            if d > T - 5:
                continue
            x, _ = needle_batch(batch_size, T, n_real_tokens, model.mask_token,
                                generator, dist=d)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, random.Random(0), model.mcfg.n_layers)], x)
            pred = aux["rounds"][0][1]["logits"][:, T - 2].argmax(-1).cpu()
            out[str(d)] = round((pred == x[:, T - 1].cpu()).float().mean().item(), 4)
    return out


@torch.no_grad()
def decode_speed(model: PathLM, pcap: PathConfig, prompt_len: int = 256, n_new: int = 128) -> dict:
    """Tokens/s per elastic configuration (greedy), on the current device."""
    from .decode import decode
    T = model.mcfg.seq_len
    prompt = torch.randint(0, model.n_real_tokens, (1, prompt_len),
                           generator=torch.Generator().manual_seed(5)).to(model.embed.weight.device)
    variants = {
        "as_configured": dict(),                      # the run's own knobs
        "exit": dict(exit_threshold=0.9),
        "retry": dict(retry_threshold=0.99),
        "full_attention": dict(),                     # window off, for comparison
    }
    out = {}
    for name, kw in variants.items():
        if name == "full_attention":
            with eval_pc(model, window=0, anchors=0) as pc_full:
                torch.cuda.synchronize(); t0 = time.time()
                decode(model, prompt[0], n_new, pc_full, **kw)
                torch.cuda.synchronize()
                out[name] = round(n_new / (time.time() - t0), 1)
            continue
        torch.cuda.synchronize()
        t0 = time.time()
        decode(model, prompt[0], n_new, pcap, **kw)
        torch.cuda.synchronize()
        out[name] = round(n_new / (time.time() - t0), 1)
    out["window_config"] = {"window": pcap.window, "anchors": pcap.anchors}
    return out
