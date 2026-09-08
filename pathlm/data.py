"""enwik8 byte-level data pipeline (the enwik8 convention is byte-level, 205 vocab)."""

import numpy as np
import torch


def _distinct_bytes(raw: bytes) -> list[int]:
    return sorted(set(raw))


def load_enwik8_subset(raw_path: str, n_bytes: int, cache: str, val_bytes: int = 1_000_000):
    """Return (train_u8, val_u8, vocab_size). Vocab = distinct bytes + [mask] slot."""
    import os
    if os.path.exists(cache):
        blob = np.load(cache)
        return blob["train"], blob["val"], int(blob["vocab_size"])
    with open(raw_path, "rb") as f:
        data = f.read(n_bytes)
    vocab = _distinct_bytes(data)
    stoi = {b: i for i, b in enumerate(vocab)}
    arr = np.array([stoi[b] for b in data], dtype=np.uint16)
    train, val = arr[:-val_bytes], arr[-val_bytes:]
    np.savez(cache, train=train, val=val, vocab_size=len(vocab) + 1)  # +1 = [mask]
    return train, val, len(vocab) + 1


def load_enwik8_full(raw_path: str, cache: str, eval_bytes: int = 10_000_000):
    """Standard enwik8 split: first 90M bytes train, last 10M eval. Vocab is
    computed over the FULL file (enwik8 has 205 distinct bytes), so the eval
    tail never contains an out-of-vocab byte."""
    import os
    if os.path.exists(cache):
        blob = np.load(cache)
        return blob["train"], blob["val"], int(blob["vocab_size"])
    with open(raw_path, "rb") as f:
        data = f.read()
    vocab = _distinct_bytes(data)
    stoi = {b: i for i, b in enumerate(vocab)}
    arr = np.array([stoi[b] for b in data], dtype=np.uint16)
    train, val = arr[:-eval_bytes], arr[-eval_bytes:]
    np.savez(cache, train=train, val=val, vocab_size=len(vocab) + 1)  # +1 = [mask]
    return train, val, len(vocab) + 1


def batch(np_arr, batch_size: int, seq_len: int, generator: torch.Generator):
    """Random (x, y) windows; y is x shifted by one (standard AR targets)."""
    hi = len(np_arr) - seq_len - 1
    idx = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
    x = np.stack([np_arr[i:i + seq_len] for i in idx])
    y = np.stack([np_arr[i + 1:i + seq_len + 1] for i in idx])
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def needle_batch(batch_size: int, seq_len: int, n_real_tokens: int, mask_token: int,
                 generator: torch.Generator, max_dist: int | None = None,
                 dist: int | None = None):
    """Copy-from-context needle task (the needle-in-haystack eval, train form).

    Row layout: needle pair (x_n, y_n) at positions p, p+1; random filler
    elsewhere; query tail at positions T-3, T-2, T-1 = [mask, x_n, y_n]. The
    model at row T-2 (input x_n, cue [mask] behind it) can only predict y_n by
    attending back to the needle, distance d = (T-2) - (p+1); filler tokens are
    resampled away from x_n/y_n so the association is unambiguous. Node-1
    accuracy at row T-2 is the needle metric. With eviction, needles with d <=
    window survive naturally; needles placed in the first `anchors` positions
    survive via the anchor channel — distances are sampled to cover both.

    Filler is random, so the AR loss on non-query positions carries no signal
    (predicting uniform noise); it is left unmasked to keep forward simple."""
    T = seq_len
    if max_dist is None:
        max_dist = T - 6
    x = torch.randint(0, n_real_tokens, (batch_size, T), generator=generator)
    for b in range(batch_size):
        xn = int(torch.randint(0, n_real_tokens, (1,), generator=generator))
        yn = int(torch.randint(0, n_real_tokens, (1,), generator=generator))
        d = dist if dist is not None else int(torch.randint(2, max_dist + 1, (1,), generator=generator))
        p = T - 3 - d
        x[b, p], x[b, p + 1] = xn, yn
        x[b, T - 3], x[b, T - 2], x[b, T - 1] = mask_token, xn, yn
        stray = (x[b] == xn) | (x[b] == yn)
        stray[[p, p + 1, T - 2, T - 1]] = False
        while stray.any():
            n = int(stray.sum())
            x[b, stray] = torch.randint(0, n_real_tokens, (n,), generator=generator)
            stray = ((x[b] == xn) | (x[b] == yn))
            stray[[p, p + 1, T - 2, T - 1]] = False
    return x, x.clone()  # targets = the sequence itself (self-repair convention)
