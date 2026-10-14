"""latent-regression vs discrete head (WP 6.3.3 · WP1) — 库。

契约 = 本目录 `README.md`（预登记，不得改）。核心对象：
- 冻结主干（2631 的 13m_s0/s1）提供特征 `h_i`（最终层残差流）。
- 每个输出位一个 `pathlm.model.TransformHead(d)`（`z_j = head_j(h_i)`）。
- 共享 tied 读出 `logits_j = z_j @ model.embed.weight.T`（E = 主干 embed 权重）。
- 损失 `Σ_j CE(logits_j, t_{i+1+j}) + λ·Σ_j MSE(z_j, h_true_{i+1+j})`，**λ 是唯一变量**。

j ∈ 1..k，j=1 ⇒ 预测 `t_{i+2}`（自查项 2）。k=2 ⇒ 预测 t+2、t+3。
`disc_k*` 与 `lat_k*_lam0` 走**同一条代码路径**（λ=0 时完全跳过 MSE）⇒ 同种子逐位相同（L0）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pathlm.config import ModelConfig, PathConfig, PathSample  # noqa: E402
from pathlm.model import PathLM, TransformHead, cap_norm  # noqa: E402

VOCAB_SIZE = 206
SEQ_LEN = 256
N_TRAIN_BLOCKS = 1024
N_VAL_BLOCKS = 512
TRAIN_STEPS = 3000
BATCH_POSITIONS = 4096
LR = 1e-3
DATA_SEED = 2660
BOOT_B = 2000
BOOT_SEED = 2660

DATA_NPZ = os.path.join(REPO, "data", "enwik8_full.npz")


# ---------------------------------------------------------------- checkpoint

def load_ckpt(root: str) -> PathLM:
    """加载 2631 冻结主干（只读）。结构同 2631/run.py::load_ckpt。"""
    with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = PathLM(ModelConfig(**cfg["model"]), PathConfig(**cfg["path"]), vocab_size=VOCAB_SIZE)
    sd = torch.load(os.path.join(root, "model.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return freeze(m.eval())


def freeze(model: PathLM) -> PathLM:
    """冻结主干全部参数（L0 硬门的 `requires_grad=False` 那半边）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def embed_sha256(model: PathLM) -> str:
    return hashlib.sha256(model.embed.weight.detach().cpu().numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ features

def trunk_h(model: PathLM, tokens: torch.Tensor) -> torch.Tensor:
    """全序列前向取最终层残差流 h [B,T,d]。

    与 `Decoder.step`（单位置）逐位一致：`cap_norm(embed+pos_embed)` → 12 层 → h；
    因果注意力下位置 i 的 h 只依赖 ≤ i，故全序列前向的 h_i == 逐位置解码 ingest 完 t_0..t_i 时的 h。
    """
    T = tokens.shape[1]
    h = model.embed(tokens) + model.pos_embed.weight[:T]
    h = cap_norm(h, model.mcfg.norm_cap)
    n = model.mcfg.n_layers
    path = PathSample(layer_order=list(range(n)), layer_repeats=[1] * n)
    h, _ = model._run_layers(h, path, attn_mask=None)
    return h


def _wrap_mtp_nodes(model: PathLM):
    """非侵入式包装 `_mtp_nodes`：调用时把入参 h 记进 buffer，其余行为不变。

    返回 (restore, buffer)。buffer["h"] = 最近一次调用的 h 参数。
    """
    orig = model._mtp_nodes
    buf: dict[str, torch.Tensor] = {}

    def wrapper(h, condition=None):
        buf["h"] = h
        return orig(h, condition=condition)

    model._mtp_nodes = wrapper

    def restore():
        model._mtp_nodes = orig

    return restore, buf


@torch.no_grad()
def extract_blocks(model: PathLM, tokens: torch.Tensor, n_blocks: int,
                   batch_blocks: int = 16) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """把 enwik8 分片切成 n_blocks 个 SEQ_LEN 块，返回 (H [nb,T,d], tok [nb,T], n1_pred [nb,T-1])。

    `n1_pred[b, i]` = 冻结主干 **node 1**（原生 t+1 头，`decode.py:399`）对位置 i 的
    下一 token 预测 argmax，用于 `ar_t1` 参照。
    """
    T = SEQ_LEN
    tokens = tokens[: n_blocks * T].reshape(n_blocks, T).astype(np.int64)
    dev = model.embed.weight.device
    H, AR = [], []
    restore, buf = _wrap_mtp_nodes(model)
    try:
        for b0 in range(0, n_blocks, batch_blocks):
            chunk = torch.tensor(tokens[b0:b0 + batch_blocks], device=dev)
            h = trunk_h(model, chunk)
            nodes = model._mtp_nodes(h)  # 包装层在此记下 h（非侵入式抓取）
            assert buf["h"] is h
            H.append(buf["h"].cpu())
            AR.append(nodes[1]["logits"][:, :-1].argmax(-1).cpu().numpy())
    finally:
        restore()
    return torch.cat(H, 0), torch.tensor(tokens), np.concatenate(AR, 0)


def valid_positions(T: int, k: int) -> np.ndarray:
    """有效位置 i：所有目标 `t_{i+1+j}`（j=1..k）都在块内 ⇒ i+1+k <= T-1。"""
    return np.arange(0, T - 1 - k, dtype=np.int64)


def ar_t1_block_hits(tok: torch.Tensor, n1_pred: np.ndarray, k: int) -> np.ndarray:
    """`ar_t1` 参照：在**同一评测位置集**上 node 1 对 t_{i+1} 的逐块命中数。"""
    T = tok.shape[1]
    pos = valid_positions(T, k)
    tgt = tok.numpy()[:, pos + 1]                    # [nb, n_pos] = t_{i+1}
    pred = n1_pred[:, pos]                           # [nb, n_pos]
    return (pred == tgt).sum(1).astype(np.int64)


# --------------------------------------------------------------------- heads

def make_heads(k: int, seed: int, d: int, device) -> nn.ModuleList:
    torch.manual_seed(seed)
    heads = nn.ModuleList(TransformHead(d) for _ in range(k))
    return heads.to(device)


def head_forward(heads: nn.ModuleList, h_i: torch.Tensor, E: torch.Tensor, k: int):
    """逐头产出 (z_j, logits_j)；logits 与 E 共享（E = 主干 embed 权重，tied 读出）。"""
    zs, logits = [], []
    for j in range(k):
        z = heads[j](h_i)
        zs.append(z)
        logits.append(z @ E.T)
    return zs, logits


def param_stats(k: int, d: int, vocab: int) -> dict:
    n_head = sum(p.numel() for p in TransformHead(d).parameters())
    return {
        "k": k,
        "d": d,
        "vocab": vocab,
        "params_per_head": n_head,
        "head_params_total": n_head * k,
        "head_macs_per_position": k * (d * 2 * d + 2 * d * d),
        "tied_readout_macs_per_position": k * d * vocab,
        "backbone_params_frozen": None,  # 由 run.py 填
    }


# ------------------------------------------------------------------ training

def train_arm(H: torch.Tensor, tok: torch.Tensor, E: torch.Tensor, k: int, lam: float,
              seed: int, steps: int = TRAIN_STEPS, batch: int = BATCH_POSITIONS,
              log_every: int = 100, device: str = "cuda", log: list | None = None):
    """训一个臂的头。数据顺序固定 seed=DATA_SEED；头初始化 seed=seed。

    λ=0 时**完全跳过** MSE（disc_* 与 lat_*_lam0 走同一路径 ⇒ 逐位相同）。
    """
    nb, T, d = H.shape
    pos = valid_positions(T, k)
    ii, bb = np.meshgrid(pos, np.arange(nb), indexing="ij")
    flat_b = bb.reshape(-1).copy()
    flat_i = ii.reshape(-1).copy()
    n_pos = flat_b.shape[0]

    heads = make_heads(k, seed, d, device)
    opt = torch.optim.Adam(heads.parameters(), lr=LR, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(DATA_SEED)

    H = H.to(device, dtype=torch.float32)
    tok = tok.to(device)
    E = E.to(device, dtype=torch.float32)
    fb = torch.from_numpy(flat_b)
    fi = torch.from_numpy(flat_i)
    traces = []

    for step in range(steps):
        sel = torch.randint(0, n_pos, (batch,), generator=gen)
        b = fb[sel].to(device)
        i = fi[sel].to(device)
        h_i = H[b, i]                       # [batch, d] frozen feature, no grad
        zs, logits = head_forward(heads, h_i, E, k)
        ce = torch.zeros((), device=device)
        mse = torch.zeros((), device=device)
        per_head_ce = []
        for j in range(k):
            tgt = tok[b, i + 1 + (j + 1)]   # j=0 -> t_{i+2}; j=1 -> t_{i+3}
            c = F.cross_entropy(logits[j], tgt)
            ce = ce + c
            per_head_ce.append(float(c.detach()))
            if lam > 0.0:
                mse = mse + F.mse_loss(zs[j], H[b, i + 1 + (j + 1)])
        loss = ce + (lam * mse if lam > 0.0 else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "ce": float(ce.detach()), "mse": float(mse.detach()),
                   "per_head_ce": per_head_ce}
            traces.append(rec)
            if log is not None:
                log(rec)
    return heads, traces


# ------------------------------------------------------------------ eval (path i)

@torch.no_grad()
def eval_hits(heads: nn.ModuleList, H: torch.Tensor, tok: torch.Tensor, E: torch.Tensor,
              k: int, device: str = "cuda", batch_blocks: int = 64) -> dict:
    """teacher-forced 命中率 + CE + MSE，按评测块聚合（供聚类 bootstrap）。"""
    nb, T, d = H.shape
    pos = valid_positions(T, k)
    npos = len(pos)
    pos_t = torch.from_numpy(pos)
    E = E.to(device, dtype=torch.float32)
    hits = np.zeros((k, nb), dtype=np.int64)
    ce_blk = np.zeros((k, nb), dtype=np.float64)
    mse_blk = np.zeros((k, nb), dtype=np.float64)
    n_pos_blk = np.full(nb, npos, dtype=np.int64)
    for b0 in range(0, nb, batch_blocks):
        b1 = min(b0 + batch_blocks, nb)
        h = H[b0:b1].to(device, dtype=torch.float32)
        tok_b = tok[b0:b1].to(device)
        for j in range(k):
            z = heads[j](h)                                 # [B,T,d]
            logits = z @ E.T                                # shared tied readout
            tgt = tok_b[:, pos_t + 1 + (j + 1)]             # t_{i+1+j}, i in eval set
            lg = logits[:, pos_t]                           # head j reads h_i, row i
            hits[j, b0:b1] = (lg.argmax(-1) == tgt).sum(-1).cpu().numpy()
            ce = F.cross_entropy(lg.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
                                 reduction="none").view(tgt.shape)
            ce_blk[j, b0:b1] = ce.sum(-1).cpu().numpy()
            ht = H[b0:b1, pos_t + 1 + (j + 1)].to(device, dtype=torch.float32)
            mse_blk[j, b0:b1] = ((z[:, pos_t + (j + 1)] - ht) ** 2).mean(-1).sum(-1).cpu().numpy()
    total = int(n_pos_blk.sum())
    out = {"n_pos": total, "n_blocks": int(nb), "n_pos_blk": n_pos_blk.tolist()}
    for j in range(k):
        out[f"hit_t{j+2}"] = float(hits[j].sum() / total)
        out[f"ce_t{j+2}"] = float(ce_blk[j].sum() / total)
        out[f"mse_t{j+2}"] = float(mse_blk[j].sum() / total)
    out["_hits"] = hits
    out["_ce_blk"] = ce_blk
    out["_mse_blk"] = mse_blk
    out["_n_pos_blk"] = n_pos_blk
    return out


# ---------------------------------------------------------------- statistics

def wilson_ci(hits: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [None, None]
    p = hits / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - hw), 5), round(float(c + hw), 5)]


def cluster_bootstrap_paired(hits_a: np.ndarray, hits_b: np.ndarray, n_pos_blk: np.ndarray,
                             B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """按评测块聚类的配对 bootstrap：重采样块，Δ = Σ(a-b)/Σ n。percentile 95% CI。"""
    nb = hits_a.shape[0]
    a = hits_a.astype(np.float64)
    b = hits_b.astype(np.float64)
    n = n_pos_blk.astype(np.float64)
    delta = float((a.sum() - b.sum()) / n.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(B, nb))
    num = a[idx].sum(1) - b[idx].sum(1)
    den = n[idx].sum(1)
    deltas = num / den
    return {
        "delta": round(delta, 5),
        "ci95": [round(float(np.percentile(deltas, 2.5)), 5),
                 round(float(np.percentile(deltas, 97.5)), 5)],
        "excludes_zero": bool(float(np.percentile(deltas, 2.5)) > 0
                              or float(np.percentile(deltas, 97.5)) < 0),
        "B": B, "seed": seed, "n_blocks": int(nb), "cluster": "eval_block",
        "ci_kind": "percentile95",
    }


def save_heads(heads: nn.ModuleList, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {f"head{j}": heads[j].state_dict() for j in range(len(heads))}
    torch.save(sd, path)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def head_files_all_tensors_equal(path_a: str, path_b: str) -> bool:
    """两个头权重文件的**内容**同一性：逐头逐键 `torch.equal`（覆盖全部张量）。

    `.pt` 的 sha256 只标识**文件字节**（zip entry 含保存时刻），不构成内容同一性证明；
    L0 的权重硬门以本函数为准。"""
    a = torch.load(path_a, map_location="cpu", weights_only=True)
    b = torch.load(path_b, map_location="cpu", weights_only=True)
    if sorted(a) != sorted(b):
        return False
    for head in a:
        if sorted(a[head]) != sorted(b[head]):
            return False
        for name in a[head]:
            if not torch.equal(a[head][name], b[head][name]):
                return False
    return True
