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
    confs, hits = [], []
    with eval_pc(model, corrupt_wrong=0.0, corrupt_mask=0.0, p_retry=0.0,
                 p_token_retry=0.0, w_dense_exit=0.0, perturb_noise=0.0, pure_noise=0.0):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, random, model.mcfg.n_layers)], x)
            node1 = aux["rounds"][0][1]
            logits = node1["logits"][:, :-1]  # node 1, rows 0..T-2
            tgt = x[:, 1:]
            nats += F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1),
                                    reduction="sum").item()
            correct += (logits.argmax(-1) == tgt).sum().item()
            tokens += tgt.numel()
            confs.append(node1["conf"][:, :-1].cpu())
            hits.append((logits.argmax(-1) == tgt).float().cpu())
    return {"bpc": round(nats / tokens / 0.6931471805599453, 4),
            "next_token_acc": round(correct / tokens, 4),
            "ece_node1": round(ece(torch.cat(confs).sigmoid(), torch.cat(hits)), 4)}


@torch.no_grad()
def repair(model: PathLM, eval_arr, corrupt_rate: float, n_batches: int = 40,
           batch_size: int = 32, generator: torch.Generator | None = None) -> dict:
    """Self-node repair accuracy on corrupted positions, per round; ECE of the
    confidence heads; confidence-weighted ensemble over the per-round node-1
    estimates vs the best single round (design §8 battery columns).
    Rounds are driven by the run's own config: round 0 always; one latent
    round iff p_retry > 0; one token-retry round iff p_token_retry > 0 (the
    element under test in R1)."""
    T = model.mcfg.seq_len
    rng = random.Random(1)
    generator = generator or torch.Generator().manual_seed(1)
    torch.manual_seed(1234)  # seed the global RNG stage-0 corruption draws from
    pc = model.pcap
    n_paths = 1 + (pc.p_retry > 0) + (pc.p_token_retry > 0)
    # forward emits one aux round per path, plus one more when the last path
    # carries the token-retry flag (its own pass + the discrete round). The
    # flagged path's own pass is a redundant base pass — the measured rounds
    # are: base, latent retry (if any), token retry (the LAST aux round).
    masks = []                       # [B, T] corruption flags per batch
    streams = [[] for _ in range(n_paths + (pc.p_token_retry > 0))]  # per aux round
    tgt0s = []                       # per-batch clean targets (node 0 = tokens)
    with eval_pc(model, corrupt_wrong=corrupt_rate, corrupt_mask=0.0, p_retry=0.0,
                 p_token_retry=0.0, w_dense_exit=0.0):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            paths = []
            for _ in range(n_paths):
                p = sample_path(model.pcap, rng, model.mcfg.n_layers)
                p.n_retries = 0           # deterministic round count
                p.token_retry = False
                paths.append(p)
            paths[-1].token_retry = pc.p_token_retry > 0
            _, aux = model(x, paths, x)
            masks.append(aux["corrupt_mask"].cpu())
            tgt0s.append(x.cpu())
            for r in range(len(aux["rounds"])):
                streams[r].append((aux["rounds"][r][0]["logits"].cpu(),
                                   aux["rounds"][r][0]["conf"].cpu(),
                                   aux["rounds"][r][1]["logits"][:, :T - 1].cpu(),
                                   aux["rounds"][r][1]["conf"][:, :T - 1].cpu()))
    cm = torch.cat(masks)
    tgt0 = torch.cat(tgt0s)   # node 0 targets: the tokens themselves
    tgt1 = tgt0[:, 1:]        # node 1 targets: shifted
    # measured aux indices: base(0), latent(1) if present, token(last) if present
    measured = [0]
    if pc.p_retry > 0:
        measured.append(1)
    if pc.p_token_retry > 0:
        measured.append(len(streams) - 1)
    kinds = (["base"] + (["latent"] if pc.p_retry > 0 else [])
             + (["token"] if pc.p_token_retry > 0 else []))
    out = {"n_corrupted": int(cm.sum().item()), "rounds": {}, "round_kinds": kinds}
    pairs = []
    for r, aux_idx in enumerate(measured):
        n0_logits, n0_conf, n1_logits, n1_conf = (torch.cat(comp) for comp in
                                                  zip(*streams[aux_idx]))
        hit0 = n0_logits.argmax(-1) == tgt0
        acc = hit0[cm].float().mean().item() if cm.any() else float("nan")
        out["rounds"][f"r{r}"] = {
            "self_acc_all": round(hit0.float().mean().item(), 4),
            "repair_acc": round(acc, 4),
            "ece_node0": round(ece(n0_conf.sigmoid(), hit0.float()), 4),
        }
        pairs.append((n1_logits, n1_conf))
    out["ece_node1"] = round(ece(
        pairs[0][1].sigmoid(), (pairs[0][0].argmax(-1) == tgt1).float()), 4)
    if len(pairs) > 1:
        mix, conf_mix = ensemble(pairs)
        hit_mix = (mix.argmax(-1) == tgt1).float()
        best = max(round((p[0].argmax(-1) == tgt1).float().mean().item(), 4) for p in pairs)
        out["ensemble_vs_best"] = {
            "ensemble_acc": round(hit_mix.mean().item(), 4),
            "ensemble_ece": round(ece(conf_mix, hit_mix), 4),
            "best_single_acc": best,
        }
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
    learned, not known). With anchors configured, anchor-regime distances
    (needle inside the first `anchors` positions, d far beyond the window) are
    added and prefixed 'anchor:' — the anchor channel's own gate."""
    T = model.mcfg.seq_len
    generator = generator or torch.Generator().manual_seed(3)
    eval_dists = [("", d) for d in dists]
    if model.pcap.anchors > 0:
        for p in (0, model.pcap.anchors // 2, model.pcap.anchors - 1):
            eval_dists.append(("anchor:", T - 3 - p))
    out = {}
    with eval_pc(model, corrupt_wrong=0.0, p_retry=0.0, p_token_retry=0.0, w_dense_exit=0.0):
        for tag, d in eval_dists:
            if d > T - 5:
                continue
            x, _ = needle_batch(batch_size, T, n_real_tokens, model.mask_token,
                                generator, dist=d)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, random.Random(0), model.mcfg.n_layers)], x)
            pred = aux["rounds"][0][1]["logits"][:, T - 2].argmax(-1).cpu()
            out[f"{tag}{d}"] = round((pred == x[:, T - 1].cpu()).float().mean().item(), 4)
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
        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if name == "full_attention":
            with eval_pc(model, window=0, anchors=0) as pc_full:
                sync(); t0 = time.time()
                decode(model, prompt[0], n_new, pc_full, **kw)
                sync()
                out[name] = round(n_new / (time.time() - t0), 1)
            continue
        sync(); t0 = time.time()
        decode(model, prompt[0], n_new, pcap, **kw)
        sync()
        out[name] = round(n_new / (time.time() - t0), 1)
    out["window_config"] = {"window": pcap.window, "anchors": pcap.anchors}
    return out
