"""enwik8 byte-level data pipeline (the enwik8 convention is byte-level, 205 vocab)."""

import numpy as np
import torch


def load_enwik8_subset(raw_path: str, n_bytes: int, cache: str, val_bytes: int = 1_000_000):
    """Return (train_u8, val_u8, vocab_size). Vocab = distinct bytes + [mask] slot."""
    import os
    if os.path.exists(cache):
        blob = np.load(cache)
        return blob["train"], blob["val"], int(blob["vocab_size"])
    with open(raw_path, "rb") as f:
        data = f.read(n_bytes)
    vocab = sorted(set(data))
    stoi = {b: i for i, b in enumerate(vocab)}
    arr = np.array([stoi[b] for b in data], dtype=np.uint16)
    train, val = arr[:-val_bytes], arr[-val_bytes:]
    np.savez(cache, train=train, val=val, vocab_size=len(vocab) + 1)  # +1 = [mask]
    return train, val, len(vocab) + 1


def batch(np_arr, batch_size: int, seq_len: int, generator: torch.Generator):
    """Random (x, y) windows; y is x shifted by one (standard AR targets)."""
    hi = len(np_arr) - seq_len - 1
    idx = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
    x = np.stack([np_arr[i:i + seq_len] for i in idx])
    y = np.stack([np_arr[i + 1:i + seq_len + 1] for i in idx])
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
