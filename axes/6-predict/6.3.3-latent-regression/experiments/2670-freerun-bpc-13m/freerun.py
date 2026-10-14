"""2670-freerun-bpc-13m — free-run 序列级 bpc 的库（路径 (i) vs AR）。

契约 = 本目录 `README.md`（预登记，不得改）。唯一变量 = **未来位置的产出方式**：

- `ar`（参照）：同一冻结主干 **node-1**（= `model.transforms[0]`，tied 读出 `@ E^T`）逐 token 贪心。
- `path_i`（候选，路径 (i)，估计得贪心、**不验证**）：一次冻结主干前向 → 取最后位置 `i` 的最终层隐
  状态 `h_i` → 对 j=1..k 用第 j 个变换头 `z_j = head_j(h_i)`、`logits_j = z_j @ E^T`（E = 主干
  `embed.weight`，tied 读出）→ 各自 argmax、**全部**追加（k=2 ⇒ 每轮 2 个 token）。

两臂都按「**自己发出的 token 在自家预测分布下的 NLL**」计 bpc（prompt 不计分，无真值比对）。

头权重与主干权重复用 2660（只读，不修改其任何文件）：`latent_reg.load_ckpt` / `freeze` /
`trunk_h` 是本库复用的 harness 来源（见 `MODEL.md`）。
"""

from __future__ import annotations

import hashlib
import json
import math
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

EXP_2660 = os.path.join(REPO, "axes", "6-predict", "6.3.3-latent-regression",
                        "experiments", "2660-latentreg-vs-disc-13m")
if EXP_2660 not in sys.path:
    sys.path.append(EXP_2660)   # append（不前置）：避免 2660 目录的 run.py/gpu_lock.py 遮蔽本目录同名文件

# 只读复用 2660 的 harness（load_ckpt / freeze / trunk_h / head_files_all_tensors_equal）
import latent_reg as L  # noqa: E402
from pathlm.model import TransformHead  # noqa: E402

VOCAB_SIZE = 206
K = 2                     # 一次前向发出的 token 数（契约 §2）
N_NEW = 200               # 每个 prompt 生成的新 token 数（契约 §3）
N_PROMPTS = 12
PROMPT_LEN = 64           # 契约 §3：12 个固定 prompt × 长度 64
STRIDE = 5000             # 同 2640/2650 的 prompt offset 取法（val[i*stride : i*stride+len]）
BOOT_B = 2000             # 契约 §3：prompt 聚类配对 bootstrap B=2000
BOOT_SEED = 2670
LOG2 = math.log(2.0)

DATA_NPZ = os.path.join(REPO, "data", "enwik8_full.npz")

BACKBONES = {
    "13m_s0": os.path.join(REPO, "axes", "6-predict", "6.2-speculative-decoding",
                           "experiments", "2631-spec-three-arms", "train", "13m_s0"),
    "13m_s1": os.path.join(REPO, "axes", "6-predict", "6.2-speculative-decoding",
                           "experiments", "2631-spec-three-arms", "train", "13m_s1"),
}
# 每个候选臂（头集合）在两条主干上的头文件目录：s0 用 2660 主目录，s1 用 2660 的 second-backbone 副本
HEAD_DIRS = {
    "13m_s0": EXP_2660,
    "13m_s1": os.path.join(EXP_2660, "second-backbone-13m_s1"),
}
HEAD_ARMS = ("disc_k2", "lat_k2_lam1")     # 契约 §2：两主干点 × 两头种子
HEAD_SEEDS = (0, 1)


# ------------------------------------------------------------------ 基本件

def logits_of(z: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """共享 tied 读出：`logits = z @ E^T`，E = 主干 `embed.weight`（不另取权重）。"""
    return z @ E.T


def stream_sha256(stream: torch.Tensor) -> str:
    """一条（prompt 的）生成流的可复现凭证：token id 序列的 sha256。"""
    return hashlib.sha256(stream.detach().cpu().numpy().astype(np.int64).tobytes()).hexdigest()


def make_prompts(val: np.ndarray, n_prompts: int = N_PROMPTS, prompt_len: int = PROMPT_LEN,
                 stride: int = STRIDE) -> tuple[torch.Tensor, list[int]]:
    """与 2640/2650 同口径：`val[i*stride : i*stride+prompt_len]`，返回 [B, prompt_len] 与 offset 列表。"""
    offsets = [i * stride for i in range(n_prompts)]
    rows = [val[o:o + prompt_len].astype(np.int64) for o in offsets]
    return torch.tensor(np.stack(rows)), offsets


def load_heads(path: str, d_model: int, device) -> nn.ModuleList:
    """加载 2660 落盘头（`{head0: sd, head1: sd}`），键 `head{j}` ⇒ 第 j+1 个头。"""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    heads = nn.ModuleList(TransformHead(d_model) for _ in range(K))
    for j in range(K):
        heads[j].load_state_dict(sd[f"head{j}"])
    heads = heads.to(device).eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    return heads


def heads_equal_vs_file(heads: nn.ModuleList, path: str) -> bool:
    """L0：内存中的头逐张量等于 2660 落盘文件（覆盖 k 个头的**全部**张量）。"""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for j in range(len(heads)):
        cur = heads[j].state_dict()
        if sorted(cur) != sorted(sd[f"head{j}"]):
            return False
        for name, t in cur.items():
            if not torch.equal(t.detach().cpu(), sd[f"head{j}"][name]):
                return False
    return True


def frozen_check(model) -> dict:
    """L0：主干 `requires_grad=False`；一次 backward 后主干 `.grad is None`。"""
    frozen = all(not p.requires_grad for p in model.parameters())
    probe = TransformHead(model.mcfg.d_model).to(model.embed.weight.device)
    z = probe(torch.zeros(2, model.mcfg.d_model, device=model.embed.weight.device))
    (z @ model.embed.weight.T).sum().backward()
    no_grad = all(p.grad is None for p in model.parameters())
    return {"requires_grad_all_false": bool(frozen), "grad_is_none_after_backward": bool(no_grad),
            "pass": bool(frozen and no_grad)}


# -------------------------------------------------------------- 生成（free-run）

@torch.no_grad()
def freerun_ar(model, prompts: torch.Tensor, n_new: int = N_NEW):
    """参照臂：node-1 逐 token 贪心。返回 (nll_nats [B], stream [B, n_new])。"""
    E = model.embed.weight
    dev = E.device
    toks = prompts.to(dev)
    B = toks.shape[0]
    nll = torch.zeros(B, dtype=torch.float64, device=dev)
    emitted = []
    for _ in range(n_new):
        h = L.trunk_h(model, toks)                 # [B, L, d]，全因果
        z1 = model.transforms[0](h[:, -1:, :])     # node-1 latent（主干原生 t+1 头）
        lp = F.log_softmax(logits_of(z1, E), dim=-1)[:, 0, :]   # [B, V]
        nxt = lp.argmax(-1)                        # [B]
        nll += -lp.gather(1, nxt[:, None])[:, 0].double()
        emitted.append(nxt)
        toks = torch.cat([toks, nxt[:, None]], dim=1)
    return nll, torch.stack(emitted, dim=1)


@torch.no_grad()
def freerun_path_i(model, heads: nn.ModuleList, prompts: torch.Tensor, n_new: int = N_NEW,
                   k: int = K):
    """路径 (i)：一次前向发 k 个位置（估计得贪心、不验证）。返回 (nll_nats [B], stream [B, n_new])。"""
    E = model.embed.weight
    dev = E.device
    toks = prompts.to(dev)
    B = toks.shape[0]
    nll = torch.zeros(B, dtype=torch.float64, device=dev)
    rounds = []
    n_rounds = n_new // k
    for _ in range(n_rounds):
        h = L.trunk_h(model, toks)                 # 1 次冻结主干前向
        h_i = h[:, -1, :]                          # 最后位置 i 的最终层隐状态 [B, d]
        step_toks = []
        for j in range(k):
            z = heads[j](h_i)                      # 第 j+1 个头
            lp = F.log_softmax(logits_of(z, E), dim=-1)         # [B, V]
            nxt = lp.argmax(-1)                    # [B]
            nll += -lp.gather(1, nxt[:, None])[:, 0].double()
            step_toks.append(nxt)
        # 不验证 / 不拒绝：k 个 token 全部追加
        toks = torch.cat([toks] + [t[:, None] for t in step_toks], dim=1)
        rounds.append(torch.stack(step_toks, dim=1))            # [B, k]
    return nll, torch.cat(rounds, dim=1)                       # [B, n_new]


# ------------------------------------------------------------------ 口径与统计

def bpc_from_nll(nll_nats, n_tokens: int) -> float:
    """序列级 bpc = Σ−ln p / (n_tokens · ln2)，越低越好。

    `n_tokens` = 这一批被计分的 token 总数（= n_prompts · n_new），与 `nll_nats` 的元素数无关。"""
    return float(np.asarray(nll_nats, dtype=np.float64).sum() / (n_tokens * LOG2))


def paired_bpc_ci(nll_cand, nll_ar, n_tok, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """prompt 聚类配对 bootstrap：重采样 prompt（cluster），Δbpc = (Σ_cand − Σ_ar)/(Σ n·ln2)。
    percentile 95% CI。契约 §3 主判口径。"""
    a = np.asarray(nll_cand, dtype=np.float64)
    b = np.asarray(nll_ar, dtype=np.float64)
    n = np.asarray(n_tok, dtype=np.float64)
    delta = float((a.sum() - b.sum()) / (n.sum() * LOG2))
    rng = np.random.default_rng(seed)
    P = a.shape[0]
    idx = rng.integers(0, P, size=(B, P))
    num = a[idx].sum(1) - b[idx].sum(1)
    den = n[idx].sum(1) * LOG2
    d = num / den
    lo = float(np.percentile(d, 2.5))
    hi = float(np.percentile(d, 97.5))
    return {
        "delta_bpc": round(delta, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "B": B, "seed": seed, "n_prompts": int(P), "cluster": "prompt",
        "ci_kind": "percentile95",
    }


# ------------------------------------------------- 真值锚（复核 N1 的补充读数）

def degeneracy_stats(tokens) -> dict:
    """一条生成流的**客观退化统计**（不需要模型）：

    - `max_identical_run`：最长连续相同 token 的 run 长度；
    - `distinct_2`：不同 bigram 数 / bigram 总数（越低越退化）；
    - `repeated_4gram_rate`：重复 4-gram 占比 = 1 − 不同 4-gram / 4-gram 总数。
    """
    s = [int(t) for t in np.asarray(tokens).reshape(-1)]
    n = len(s)
    max_run, run = (1 if n else 0), 1
    for i in range(1, n):
        run = run + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, run)

    def _ngram_distinct(ng: int):
        if n < ng:
            return 1.0, 0
        grams = [tuple(s[i:i + ng]) for i in range(n - ng + 1)]
        return len(set(grams)) / len(grams), len(grams)

    d2, n2 = _ngram_distinct(2)
    d4, n4 = _ngram_distinct(4)
    return {"n_tokens": n, "max_identical_run": int(max_run),
            "distinct_2": round(d2, 6), "n_bigrams": n2,
            "repeated_4gram_rate": round(1.0 - d4, 6), "n_4grams": n4}


@torch.no_grad()
def node1_logits(model, seq: torch.Tensor) -> torch.Tensor:
    """全序列 teacher-forced 的 node-1 logits [B, T, V]。

    `node-1 = model.transforms[0]` 作用在主干最终层残差流 `h` 上，再走 tied 读出
    (`@ embed.weight.T`)——与 `pathlm/eval.py::bpc` 取 `aux["rounds"][0][1]["logits"]`
    是同一条计算（同一口径，未改 `pathlm/**`）。
    """
    return logits_of(model.transforms[0](L.trunk_h(model, seq)), model.embed.weight)


@torch.no_grad()
def teacher_forced_stream_nll(model, prompt: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
    """`stream`（长度 n）在 `model` **teacher-forced** 下的 node-1 NLL（nats / 序列）。

    与 `pathlm/eval.py::bpc` 同口径（node-1 logits → 下一 token 交叉熵 → /ln2），但**确定性地**
    覆盖给定流（不随机开窗），故可逐 prompt 配对。构造 `[prompt, stream[:n-1]]` 为输入，读位置
    `prompt_len-1 + i` 的 logits 预测 `stream[i]`（i = 0..n-1）；返回 [B] 的 nats 和，
    分母 = n·ln2（见 `bpc_from_nll`）。
    """
    pl, n = prompt.shape[1], stream.shape[1]
    seq = torch.cat([prompt, stream[:, :n - 1]], dim=1)      # [B, pl+n-1]
    logits = node1_logits(model, seq)                        # [B, pl+n-1, V]
    sel = logits[:, pl - 1:pl - 1 + n, :]                    # 位置 pl-1.. 预测 s0..
    lp = F.log_softmax(sel, dim=-1)
    return -lp.gather(-1, stream[..., None])[..., 0].double().sum(-1)


# ---------------------------------------------------------------- 禁墙钟自检

WALL_SUBSTR = ("wall", "elapsed", "clock", "duration", "seconds", "milliseconds",
               "latency", "throughput")
WALL_EXACT = {"time", "ms", "sec", "took", "runtime", "speed", "tps", "timing"}


def has_wall_clock(obj, path: str = "") -> list[str]:
    """results.json 内不得出现墙钟语义字段（沿用 2660 的 has_wall_clock 口径）。"""
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(w in kl for w in WALL_SUBSTR) or kl in WALL_EXACT:
                bad.append(f"{path}.{k}")
            bad += has_wall_clock(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += has_wall_clock(v, f"{path}[{i}]")
    return bad
