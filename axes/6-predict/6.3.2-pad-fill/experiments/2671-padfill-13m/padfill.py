#!/usr/bin/env python3
"""2671-padfill-13m · 契约实现库（S1；契约见本目录 README.md）。

本实验在**自己的目录内**自带 harness：共享引擎 `pathlm/model.py` 的注意力是硬编码因果，
本实验需要 `full`（双向）可见性的填空臂，按边界纪律不得改 `pathlm/**`。因此这里实现一个
pre-LN char LM，**唯一结构变量 = 注意力可见性**（`attn_mask ∈ {causal, full}`）；AR 参照臂
用同一 harness 的因果配置，保证单因素可比。

结构（写死，契约 §2）：pre-LN decoder block，`d_model=256 / n_layers=12 / n_heads=8 /
mlp_mult=6 / seq_len=512`、共享 tied 读出（`logits = h @ E^T`）、无 dropout。读出前一个
final LayerNorm（稳定 12 层深残差流；两臂完全相同，不进入唯一变量）。

提供：
- `PadFillLM`：模型（含 `attn_mask ∈ {causal, full}` 开关、`count_params`）。
- `sample_fill_mask` / `make_fill_input`：mask 调度（按 r 把位置替换为 `[mask]`；`leak` 模式
  只用于 gate probe —— 刻意把真 token 喂进被填位置，L0 必须因此变红）。
- `train_arm` / `sample_window`：训练循环。
- `eval_block_set` / `bpc_from_logits` / `paired_bootstrap_ci`：评测（bpc / block 聚类配对
  bootstrap CI）。
- `has_wall_clock`：契约 L5 的禁墙钟自检（S2 的 run.py 复用）。

    python3 padfill.py --params     # 打印实测参数量（一键）
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ---- 结构常量（契约 §2）--------------------------------------------------------
VOCAB_SIZE = 206          # data/enwik8_full.npz::vocab_size（205 字节 + 1 个 [mask] 槽，实测）
MASK_TOKEN = VOCAB_SIZE - 1
SEQ_LEN = 512
D_MODEL = 256
N_LAYERS = 12
N_HEADS = 8
MLP_MULT = 6
BATCH_SIZE = 32
LR = 1e-3

# ---- 预算 / 口径（契约 §3、§4）--------------------------------------------------
DATA_SEED = 2671          # 训练数据顺序 seed（全部 run 共用 ⇒ 同 token 预算）
MASK_SEED = 2672          # 训练侧 mask 调度 seed（全部填空 run 共用）
BOOT_B = 2000             # block 聚类配对 bootstrap 重采样数
BOOT_SEED = 2673
EVAL_BLOCKS = 64          # enwik8 val 的 64 个固定 offset
EVAL_MASK_SEED = 2674     # 评测侧固定 mask seed
MIN_MASK_POS = 8          # 评测位置集 M_r：i >= 8（契约 §4）

# 契约 §2 的五臂：唯一结构变量 = 注意力可见性 + 损失位置集。
ARMS = {
    "ar":             {"attn_mask": "causal", "r": 0.0,  "loss": "ar"},
    "fill_bidir_r10": {"attn_mask": "full",   "r": 0.10, "loss": "fill"},
    "fill_bidir_r25": {"attn_mask": "full",   "r": 0.25, "loss": "fill"},
    "fill_bidir_r50": {"attn_mask": "full",   "r": 0.50, "loss": "fill"},
    "fill_causal_r25": {"attn_mask": "causal", "r": 0.25, "loss": "fill"},
}
INF_TOTAL_SEC = 1500.0    # 5 臂 × 2 seed = 10 run 的训练总时上限（25 min）
INF_RUNS = 10
STEP_QUANTUM = 500
STEP_MIN = 1500


# ---- 模型 ---------------------------------------------------------------------
class Attention(nn.Module):
    """多头注意力，`causal=True` 时下三角可见，`False` 时全可见（唯一结构变量）。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        # SDPA（仓库既有引擎同款）：causal 路径按行只读 j<=i 的 key，左侧位置逐位不受
        # 右侧扰动影响（L0 的 max|Δ|==0 依赖这一点）；full 路径无掩码 = 全可见。
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class Block(nn.Module):
    """Pre-LN block：`x += attn(ln1(x)); x += mlp(ln2(x))`。"""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model), nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal)
        x = x + self.mlp(self.ln2(x))
        return x


class PadFillLM(nn.Module):
    """pre-LN char LM，唯一结构变量 = 注意力可见性；共享 tied 读出（`logits = h @ E^T`）。"""

    def __init__(self, vocab_size: int = VOCAB_SIZE, attn_mask: str = "causal",
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 mlp_mult: int = MLP_MULT, seq_len: int = SEQ_LEN):
        super().__init__()
        if attn_mask not in ("causal", "full"):
            raise ValueError(f"attn_mask must be causal|full, got {attn_mask!r}")
        self.attn_mask = attn_mask
        self.causal = attn_mask == "causal"
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_token = vocab_size - 1
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_embed.weight, std=d_model ** -0.5)  # tied 表行范数 ~1
        self.pos_embed = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, mlp_mult) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.seq_len:
            raise ValueError(f"seq len {T} > {self.seq_len}")
        pos = torch.arange(T, device=idx.device)
        h = self.tok_embed(idx) + self.pos_embed(pos)[None]
        for blk in self.blocks:
            h = blk(h, self.causal)
        h = self.ln_f(h)
        return h @ self.tok_embed.weight.t()  # tied 读出：E 是唯一读出基


def count_params(model: nn.Module) -> int:
    """实测参数量（tied 权重是同一 Parameter，只计一次）。"""
    return sum(p.numel() for p in model.parameters())


# ---- mask 调度 -----------------------------------------------------------------
def sample_fill_mask(shape, r: float, generator: torch.Generator,
                     min_pos: int = 0) -> torch.Tensor:
    """按概率 r 独立取「被填」位置（True = 替换为 [mask]）；`min_pos` 之前的位置永不被填。"""
    m = torch.rand(shape, generator=generator) < r
    if min_pos > 0:
        m[:, :min_pos] = False
    return m


def make_fill_input(true_tokens: torch.Tensor, M: torch.Tensor,
                    mode: str = "mask", mask_token: int = MASK_TOKEN) -> torch.Tensor:
    """把 M 位置替换为 [mask] 得到模型输入。

    mode="mask"：正常填空输入（M 位置含 [mask]，被填位置不携带真值）。
    mode="leak"：gate probe 专用 —— **不替换**，把真 token 喂进 M 位置（泄漏），
                 使 L0 的逐位不变检查**必然变红**。
    """
    if mode == "mask":
        x = true_tokens.clone()
        x[M] = mask_token
        return x
    if mode == "leak":
        return true_tokens.clone()
    raise ValueError(f"unknown mode {mode!r}")


# ---- bpc / CI ------------------------------------------------------------------
def nll_of(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """`-ln p(target)`（nats）；logits (..., V)，targets (...)。"""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def bpc_from_nll(nll_sum: float, n: int) -> float:
    """bpc（每字符比特数，越低越好）= (Σ −ln p) / (n · ln2)。"""
    if n <= 0:
        raise ValueError("bpc needs n > 0")
    return float(nll_sum) / (n * math.log(2))


def paired_bootstrap_ci(deltas, B: int = BOOT_B, seed: int = BOOT_SEED,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """block 聚类配对 bootstrap 95% CI：对 block 级配对 Δ 做有放回重采样。

    返回 (point_mean, lo, hi)。CI 不含 0 ⇔ lo>0 或 hi<0。
    """
    d = np.asarray(deltas, dtype=np.float64)
    if d.size == 0:
        raise ValueError("no paired deltas")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(B, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(d.mean()), float(lo), float(hi)


def has_wall_clock(obj, path: str = "") -> list[str]:
    """契约 L5 禁墙钟自检：key 命中墙钟语义即报（短词精确匹配，避免误伤 arms/params）。"""
    subs = ("wall", "elapsed", "clock", "duration", "seconds", "milliseconds",
            "latency", "throughput")
    exact = {"time", "ms", "sec", "took", "runtime", "speed", "tps", "timing"}
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(w in kl for w in subs) or kl in exact:
                bad.append(f"{path}.{k}")
            bad += has_wall_clock(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += has_wall_clock(v, f"{path}[{i}]")
    return bad


# ---- 数据 ----------------------------------------------------------------------
def load_data():
    """(train, val, vocab_size)；缓存 data/enwik8_full.npz 已存在（立即返回）。"""
    from pathlm.data import load_enwik8_full
    npz = os.path.join(REPO, "data", "enwik8_full.npz")
    return load_enwik8_full("", npz)


def eval_offsets(val_len: int, T: int = SEQ_LEN, n: int = EVAL_BLOCKS) -> list[int]:
    """enwik8 val 上 n 个固定 offset（逐条落进 results.json）。"""
    hi = val_len - T - 1
    return [int(v) for v in np.linspace(0, hi, n, dtype=np.int64)]


def sample_window(train_arr, batch_size: int, T: int, generator: torch.Generator):
    """随机窗口 (B, T)，int64；T 含 AR 需要的 +1 目标位。"""
    hi = len(train_arr) - T - 1
    idx = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
    x = np.stack([train_arr[i:i + T] for i in idx])
    return torch.from_numpy(x.astype(np.int64))


# ---- 训练 ----------------------------------------------------------------------
def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)


def train_steps(model: PadFillLM, opt: torch.optim.Optimizer, arm: str, n_steps: int,
                train_arr, device: str, dg: torch.Generator, mg: torch.Generator,
                step_offset: int = 0, log_fn=None) -> None:
    """跑 `n_steps` 个训练 step（契约 §3：batch 32、T=512、Adam 1e-3、无 weight decay）。"""
    cfg = ARMS[arm]
    V = model.vocab_size
    model.train()
    for i in range(n_steps):
        win = sample_window(train_arr, BATCH_SIZE, SEQ_LEN + 1, dg).to(device)
        inp = win[:, :SEQ_LEN]
        if cfg["loss"] == "ar":
            logits = model(inp)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), win[:, 1:].reshape(-1))
        else:
            M = sample_fill_mask(inp.shape, cfg["r"], mg).to(device)
            logits = model(make_fill_input(inp, M, "mask"))
            loss = F.cross_entropy(logits[M], inp[M])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_fn is not None and (step_offset + i + 1) % 100 == 0:
            log_fn(step_offset + i + 1, float(loss.detach()))


def train_arm(arm: str, seed: int, steps: int, train_arr, device: str = "cuda",
              log_fn=None) -> PadFillLM:
    """按契约 §3 训一个臂。数据顺序 seed 固定（同 token 预算）；模型 init 用 `seed`。"""
    torch.manual_seed(seed)
    model = PadFillLM(attn_mask=ARMS[arm]["attn_mask"]).to(device)
    opt = build_optimizer(model)
    dg = torch.Generator().manual_seed(DATA_SEED)
    mg = torch.Generator().manual_seed(MASK_SEED)
    train_steps(model, opt, arm, steps, train_arr, device, dg, mg, log_fn=log_fn)
    model.eval()
    return model


# ---- 评测 ----------------------------------------------------------------------
@torch.no_grad()
def eval_block_set(ar_model: PadFillLM, fill_model: PadFillLM | None, val,
                   offsets: list[int], r: float, mask_seed: int = EVAL_MASK_SEED,
                   device: str = "cuda", T: int = SEQ_LEN) -> dict:
    """按契约 §4 在固定 64 block 上算 bpc。

    - `ar_model`（因果）：用于 `ar_full`（全部位置 1..T−1）与 `ar_at_M_r`（M_r 位置）。
      AR 对 token i 的预测 = `logits[i-1]`（teacher-forced）。
    - `fill_model`（None = 只评 AR）：一次前向，`logits[i]` 在 M_r 位置上预测 token i。
    - `ar_at_M_r` 与 `fill_*_r` 用**同一批 M_r**（同一个采样器 → 同一序列）。

    返回 dict：每 block 的 (n_m, s_ar_M, s_fill_M, n_full, s_ar_full) 与派生 bpc。
    """
    gen = torch.Generator().manual_seed(mask_seed)
    rows = []
    for off in offsets:
        block = torch.from_numpy(np.ascontiguousarray(val[off:off + T]).astype(np.int64)).to(device)
        M = sample_fill_mask((1, T), r, gen, min_pos=MIN_MASK_POS)[0].to(device)
        lg_ar = ar_model(block[None])[0]                       # (T, V)
        # ar_full：i = 1..T-1，用 logits[i-1] 预测 token i
        n_full = T - 1
        s_ar_full = float(nll_of(lg_ar[:-1], block[1:]).sum())
        idx_m = torch.nonzero(M, as_tuple=False).squeeze(-1)   # i in M_r
        s_ar_M = float(nll_of(lg_ar[idx_m - 1], block[idx_m]).sum())
        n_m = int(idx_m.numel())
        if fill_model is None:
            s_fill_M = float("nan")
        else:
            x = make_fill_input(block[None], M[None], "mask", mask_token=fill_model.mask_token)
            lg_f = fill_model(x)[0]
            s_fill_M = float(nll_of(lg_f[idx_m], block[idx_m]).sum())
        rows.append({"offset": int(off), "n_m": n_m, "s_ar_M": s_ar_M,
                     "s_fill_M": s_fill_M, "n_full": n_full, "s_ar_full": s_ar_full})
    ln2 = math.log(2)
    s_ar_full_tot = sum(x["s_ar_full"] for x in rows)
    n_full_tot = sum(x["n_full"] for x in rows)
    s_ar_M_tot = sum(x["s_ar_M"] for x in rows)
    n_m_tot = sum(x["n_m"] for x in rows)
    out = {
        "blocks": rows,
        "ar_full_bpc": bpc_from_nll(s_ar_full_tot, n_full_tot),
        "ar_at_M_bpc": bpc_from_nll(s_ar_M_tot, n_m_tot),
        "n_m_total": n_m_tot,
    }
    if fill_model is not None:
        deltas, s_fill_tot = [], 0.0
        for x in rows:
            deltas.append((x["s_fill_M"] - x["s_ar_M"]) / (x["n_m"] * ln2))
            s_fill_tot += x["s_fill_M"]
        out["fill_M_bpc"] = bpc_from_nll(s_fill_tot, n_m_tot)
        out["delta_blocks"] = deltas
        out["delta_mean"], out["delta_lo"], out["delta_hi"] = paired_bootstrap_ci(deltas)
    return out


# ---- 步时 → S（契约 §3 唯一允许的调整）------------------------------------------
def steps_from_steptime(steptime_s: float, total_sec: float = INF_TOTAL_SEC,
                        runs: int = INF_RUNS, quantum: int = STEP_QUANTUM,
                        min_steps: int = STEP_MIN) -> int:
    """由实测步时反解 matched-step S：10 run 训练总时 ≤ 25 min，取整到 500 的倍数、≥1500。"""
    if steptime_s <= 0:
        raise ValueError("step time must be > 0")
    raw = total_sec / (runs * steptime_s)
    s = int(raw // quantum) * quantum
    return max(s, min_steps)


def main() -> int:
    if "--params" in sys.argv:
        for tag in ("causal", "full"):
            m = PadFillLM(attn_mask=tag)
            print(f"attn_mask={tag:6s}  params={count_params(m):,} "
                  f"({count_params(m) / 1e6:.3f}M)")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
