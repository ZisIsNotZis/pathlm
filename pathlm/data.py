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
                 generator: torch.Generator, data=None, anchors: int = 0,
                 anchor_frac: float = 0.2, n_needles: int = 8):
    """Multi-needle copy-from-context on REAL-TEXT filler (X2v2 redesign —
    the v1 single-needle form starved the convention of signal: 1 supervised
    token per batch against 98M tokens of natural text, measured 0.0 accuracy).

    Row layout: positions [0, T-3K) are a real data window with K needle pairs
    (x_i, y_i) carved in at random positions p_i; the tail [T-3K, T) is the
    query block [mask, x_1, y_1, ..., mask, x_K, y_K]. Each query's leading
    [mask] is the cue (never occurs in text). Node-1 accuracy at the x_i query
    rows is the needle metric; (x_i, y_i) pairs are chosen absent from the
    row's text so the association is unambiguous.

    Needle placement: with probability anchor_frac the needle lands in the
    first `anchors` positions (the anchor channel regime), else uniformly in
    the text body. Returns (x, y, meta) with meta[b] = list of
    (query_row, needle_pos, dist) for regime-resolved accuracy."""
    import numpy as np
    T, K = seq_len, n_needles
    if data is None:
        raise ValueError("needle batches need real-text filler: pass the data array")
    q0 = T - 3 * K  # query block start
    x = np.zeros((batch_size, T), dtype=np.int64)
    meta = []
    hi = len(data) - T - 1
    idx = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
    for b in range(batch_size):
        row = np.array(data[idx[b]:idx[b] + T - 3 * K], dtype=np.int64)
        row_meta = []
        used: list[int] = []
        for i in range(K):
            # pair absent from the row's text (unambiguous association).
            # 16 tries is effectively always enough (205^2 pairs vs ~500
            # occupied bigrams), but exhaustion must SKIP the needle, never
            # place a colliding pair (the silent-ambiguous bug).
            ok = False
            xn, yn = -1, -1
            for _ in range(16):
                xn = int(torch.randint(0, n_real_tokens, (1,), generator=generator))
                yn = int(torch.randint(0, n_real_tokens, (1,), generator=generator))
                hit = bool(np.any((row[:-1] == xn) & (row[1:] == yn)))
                if not hit:
                    ok = True
                    break
            if not ok:
                continue
            p = -1  # set by the placement branches below
            if anchors > 0 and torch.rand(1, generator=generator).item() < anchor_frac:
                for _ in range(32):  # same spacing discipline as the body branch
                    p = int(torch.randint(0, anchors, (1,), generator=generator))
                    if all(abs(p - u) > 2 for u in used):
                        break
                if not all(abs(p - u) > 2 for u in used):
                    continue  # no free anchor slot: skip this needle
            else:
                for _ in range(32):
                    p = int(torch.randint(anchors, q0 - 2, (1,), generator=generator))
                    if all(abs(p - u) > 2 for u in used):
                        break
                if not all(abs(p - u) > 2 for u in used):
                    continue
            row[p], row[p + 1] = xn, yn
            used.append(p)
            q_row = q0 + 3 * i + 1
            row_meta.append((q_row, p, q_row - (p + 1)))
        x[b, :q0] = row
        meta.append(row_meta)
    # place query blocks properly (vector-safe second pass)
    for b in range(batch_size):
        for i, (q_row, p, d) in enumerate(meta[b]):
            x[b, q0 + 3 * i] = mask_token
            x[b, q_row] = x[b, p]           # x_i = the needle's first byte
            x[b, q_row + 1] = x[b, p + 1]   # y_i = the needle's second byte
    x_t = torch.from_numpy(x)
    return x_t, x_t.clone(), meta
