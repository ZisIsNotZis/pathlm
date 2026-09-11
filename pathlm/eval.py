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
    rng = random.Random(7)
    nats, tokens, correct = 0.0, 0, 0
    confs, hits = [], []
    with eval_pc(model, corrupt_wrong=0.0, corrupt_mask=0.0, p_retry=0.0,
                 p_token_retry=0.0, w_dense_exit=0.0, perturb_noise=0.0, pure_noise=0.0):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, rng, model.mcfg.n_layers)], x)
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
def repair(model: PathLM, eval_arr, n_batches: int = 40,
           batch_size: int = 32, generator: torch.Generator | None = None,
           rounds: int | None = None) -> dict:
    """Self-node repair accuracy on corrupted positions, per round; ECE of the
    confidence heads; confidence-weighted ensemble over the per-round node-1
    estimates vs the best single round (design §8 battery columns).
    Corruption is the RUN'S OWN element (whatever pcap has — wrong-token,
    mask, noise) — for noise-only runs no positions are flagged, so only
    self_acc_all is meaningful there.
    rounds=None: driven by the run's own config (base pass always, one latent
    round iff p_retry > 0, one token-retry round iff p_token_retry > 0).
    rounds=k: force a k-round loop — the retry-curve mode (inference-time
    only; tests quality past the trained loop length)."""
    T = model.mcfg.seq_len
    rng = random.Random(1)
    generator = generator or torch.Generator().manual_seed(1)
    torch.manual_seed(1234)  # seed the global RNG stage-0 corruption draws from
    pc = model.pcap
    if rounds is not None:
        n_paths = rounds  # forced k-round loop (token flag rides the last path)
    else:
        n_paths = 1 + (pc.p_retry > 0) + (pc.p_token_retry > 0)
    # forward emits one aux round per path, plus one more when the last path
    # carries the token-retry flag (its own pass + the discrete round). The
    # flagged path's own pass is a redundant base pass — the measured rounds
    # are: base, latent retry (if any), token retry (the LAST aux round).
    masks = []                       # [B, T] corruption flags per batch
    streams = [[] for _ in range(n_paths + (pc.p_token_retry > 0))]  # per aux round
    tgt0s = []                       # per-batch clean targets (node 0 = tokens)
    with eval_pc(model, p_retry=0.0, p_token_retry=0.0, w_dense_exit=0.0):
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
    # measure every aux round; label the last one "token" when the discrete
    # round ran, "base" for the first, "latent"/"pass" for loop passes
    n_aux = len(streams)
    measured = list(range(n_aux))
    kinds = []
    for i in range(n_aux):
        if pc.p_token_retry > 0 and i == n_aux - 1:
            kinds.append("token")
        elif i == 0:
            kinds.append("base")
        else:
            kinds.append("latent" if pc.p_retry > 0 else f"pass{i}")
    # pi-lens-ignore: unchecked-throwing-call-python
    out = {"n_corrupted": int(cm.sum().item()), "rounds": {}, "round_kinds": kinds}
    pairs = []
    for r, aux_idx in enumerate(measured):
        n0_logits, n0_conf, n1_logits, n1_conf = (torch.cat(comp) for comp in
                                                  zip(*streams[aux_idx]))
        hit0 = n0_logits.argmax(-1) == tgt0
        # pi-lens-ignore: unchecked-throwing-call-python
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
    # With skips/redos the executed depth count VARIES per path — bucket by
    # depth index and average each bucket (ragged-safe).
    buckets: dict[int, list] = {}
    with eval_pc(model, corrupt_wrong=0.0, p_retry=0.0, p_token_retry=0.0,
                 w_dense_exit=1.0):  # hook must fire to record the curve
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, generator)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, rng, model.mcfg.n_layers)], x)
            for d, v in enumerate(aux["depth_ce"], start=1):
                buckets.setdefault(d, []).append(v)
    return [round(torch.stack(buckets[d]).mean().item(), 4)
            for d in sorted(buckets)]


@torch.no_grad()
def needle_acc(model: PathLM, eval_arr, dists=None,
               batch_size: int = 32, generator: torch.Generator | None = None) -> dict:
    """Needle accuracy by REGIME bucket on multi-needle real-text batches
    (X2v2): in-window (needle reachable through the eviction window), beyond
    (evicted, non-anchor — allowed to fail by design), anchor (needle inside
    the anchor positions — the anchor channel's own gate)."""
    T = model.mcfg.seq_len
    pc = model.pcap
    generator = generator or torch.Generator().manual_seed(3)
    buckets = {"in_window": [], "beyond": [], "anchor": []}
    with eval_pc(model, corrupt_wrong=0.0, p_retry=0.0, p_token_retry=0.0, w_dense_exit=0.0):
        for _ in range(40):
            x, _, meta = needle_batch(batch_size, T, model.n_real_tokens,
                                      model.mask_token, generator, data=eval_arr,
                                      anchors=pc.anchors, anchor_frac=0.2)
            x = x.to(model.embed.weight.device)
            _, aux = model(x, [sample_path(model.pcap, random.Random(0), model.mcfg.n_layers)], x)
            logits = aux["rounds"][0][1]["logits"].cpu()
            for b, row_meta in enumerate(meta):
                for q_row, p, d in row_meta:
                    hit = (logits[b, q_row].argmax(-1) == x[b, q_row + 1].cpu()).item()
                    if pc.window > 0 and p < pc.anchors:
                        buckets["anchor"].append(hit)
                    elif pc.window > 0 and d >= pc.window - 1:
                        # the needle's x byte is attendable only when q_row - p <
                        # window, i.e. d <= window - 2 — boundary needles are
                        # guaranteed misses and belong to the beyond bucket
                        buckets["beyond"].append(hit)
                    else:
                        buckets["in_window"].append(hit)
    return {k: (round(sum(v) / len(v), 4) if v else None) for k, v in buckets.items()}


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


@torch.no_grad()
def locality_sweep(model: PathLM, eval_arr, n_batches: int = 60, batch_size: int = 32,
                   levels=(0.0, 0.25, 0.5, 1.0)) -> dict:
    """Next-token bpc under evaluation-time shuffle locality — the order-
    tolerance curve for shuffle-trained models (trained level vs OOD levels)."""
    out = {}
    for loc in levels:
        with eval_pc(model, shuffle_locality=loc, p_retry=0.0, p_token_retry=0.0,
                     w_dense_exit=0.0, corrupt_wrong=0.0):
            res = bpc(model, eval_arr, model.vocab_size, n_batches=n_batches,
                      batch_size=batch_size)
        out[str(loc)] = res["bpc"]
    return out
