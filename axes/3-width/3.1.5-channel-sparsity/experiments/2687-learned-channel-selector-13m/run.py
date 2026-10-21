"""2687 · 可学习通道选择器（打分器 + 负载均衡项）· 13M · dense.

契约 = 同目录 `README.md`（判据线 G1/G2/G3 **预登记、不可改**）。本文件是唯一入口。

要回答的问题（`axes/3-width/3.1.5-channel-sparsity/project.yaml::conclusion` 留下的 `gaps` 第一条）：
上一轮把「按内容选通道」判成**条件性有害**，但那一轮的选择器是**无参数**的 `|pre-activation|`
top-k。本条给它**参数**与**训练信号**，问的是：有害的是那个实现，还是「逐 token 条件选通道」这个机制。

设计要点（为什么这样接线，见契约 §2）：

* **训练整段走共享代码**：把 `train_depth_ar` 模块里的全局名 `DepthARModel` 换成
  `SelectorModel` 工厂（构造 stock 模型后把每块 `blk.mlp` 包成 `ScoredFFN`），再调
  `train_depth_ar.main()`。参数解析、配置装配、`train()` 循环、checkpoint、`results.json`
  全部是共享实现。
* **四臂同一个模块、同一份参数账**：`D`（不选）/`K50`（无参数 `|pre|` 选）/`F50`（固定前缀）
  /`L50`（可学习打分器）都带同一个低秩打分器模块（每层 `A[d,r] + B[r,H]`，r=8），只有 `L50`
  的打分器 `requires_grad=True`。⇒ 总参数量逐臂逐位相同（7 044 608），每 token 每层同样给
  `d·r + r·H = 10 240` MACs 的选择器预算（`D`/`K50`/`F50` 拿到预算但不花）。
* **`L50` 从「`|pre|` 规则」起步**：`s = |pre| · exp(g(x))`，打分器输出 `g = GELU(xA)B`，
  `B` 零初始化 ⇒ 训练第一步 `g ≡ 0` ⇒ `s = |pre|` **逐位**相同 ⇒ `L50` 的初始选择集等于
  `K50` 的选择集（单测 `T5` 钉住）。臂间唯一差异 = 「这份预算是否被训练」。
* **直通估计（steepest-through）**：前向用硬 0/1 掩码（与 `K50` 同形，不做缩放），反向经
  `softmax(s)` 把梯度送到打分器；`mask` 的前向值与硬掩码**逐位相同**（`t - t.detach()` 恒为
  精确 0），单测 `T6` 钉住。
* **负载均衡项**（Switch 式，`ALPHA_LB=0.01`）：`aux = Σ_j f̂_j·P_j`，`f̂_j` = 通道 j 的
  被选频率（按 `H/k` 归一，均值 1）、`P_j` = `softmax(s)` 的均值。加在 CE 上（`SelectorModel`
  覆写 `forward` 加残项，不改共享训练器）。

用法：

    python3 run.py --arm L50 --seed 0 --steps 12000                # 训练 + 评测
    python3 run.py --arm K50 --seed 0 --no-train                   # 只评测（复用 2644 的 checkpoint）
    python3 run.py --arm L50 --seed 0 --steps 60 --out-root /tmp/x --skip-evaluate   # 冒烟
    python3 run.py --aggregate                                     # 汇总 G1–G3 → results.json
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                   # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full, batch                # noqa: E402

DEVICE = "cuda"

# ---------------------------------------------------------------- 臂定义（契约 §2/§3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
VOCAB = 206
HIDDEN = MLP_MULT * D_MODEL                                # 1024
ARMS = ("D", "K50", "F50", "L50")
KEEP_FRAC = {"D": 1.0, "K50": 0.5, "F50": 0.5, "L50": 0.5}
SELECTING = ("K50", "F50", "L50")        # 三支同 k=512 的臂（可同框比）
SCORER_RANK = 8                          # 低秩打分器的秩 r
ALPHA_LB = 0.01                          # 负载均衡项权重
SCORER_EPS = 1e-6                        # |pre| 的下限（防 0 值）
SCORER_CLAMP = 4.0                       # exp(g) 的指数护栏（数值稳定，见 §2.1）

# 判据常数（契约 §4；`config.json::criterion` 逐字相同）
BAND = 0.014                             # clean bpc 配对带（harness/methodology.md §2）
SCORER_MOVE_MIN = 0.01                   # G1 活性门：打分器参数的相对 L2 位移下界
FLIP_MIN = 0.05                          # G1 活性门：与初始打分器相比的选择翻转率下界

# 打分器参数/算力（逐层、每 token）
SCORER_PARAMS_PER_LAYER = D_MODEL * SCORER_RANK + SCORER_RANK * HIDDEN   # 2048+8192 = 10240
SCORER_MACS_PER_LAYER = SCORER_PARAMS_PER_LAYER                          # 两个矩阵各一遍

# 复用 checkpoint（= 上一轮与 `3.1.4` 已落盘的臂；`published_bpc` 是它们各自落盘时的读数，
# 本条在自己的代码路径上重评一遍，用来钉「等值/复现门」）。
REUSE = {
    ("D", 0): {
        "ckpt": "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/runs/U4-s0-12k/model.pt",
        "source": "2641 U4-s0（dense 锚）", "published_bpc": 1.5314},
    ("K50", 0): {
        "ckpt": "axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/runs/K50-s0-12k/model.pt",
        "source": "2644 K50-s0（无参数 |pre| 选 512）", "published_bpc": 1.6295},
    ("K50", 1): {
        "ckpt": "axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/runs/K50-s1-12k/model.pt",
        "source": "2644 K50-s1", "published_bpc": 1.6263},
    ("F50", 0): {
        "ckpt": "axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/runs/F50-s0-12k/model.pt",
        "source": "2644 F50-s0（固定前缀 512）", "published_bpc": 1.5678},
    ("F50", 1): {
        "ckpt": "axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/runs/F50-s1-12k/model.pt",
        "source": "2644 F50-s1", "published_bpc": 1.5625},
}


# ---------------------------------------------------------------- 干预：通道打分器 + 掩码

class ScoredFFN(nn.Module):
    """把 stock `blk.mlp`（[Linear, GELU, Linear, Dropout]）包成
    `out = drop(lin2(gelu(lin1(x)) * mask))`，`mask` 由本模块的**低秩打分器**（`L50`）
    或**无参数规则**（`K50`/`F50`）或**不筛**（`D`）给出。

    四臂都带同一个打分器模块（`A[d,r]`、`B[r,H]`，`B` 零初始化），只有 `L50` 的
    `requires_grad=True`：参数账逐臂逐位相同，臂间唯一差异 = 这份预算是否被训练。

    `L50` 的打分 = `s = |pre| · exp(g(x))`，`g(x) = GELU(xA)B`；`B ≡ 0` 时 `s = |pre|`
    逐位相同 ⇒ 初始选择集等于 `K50`（单测 `T5`）。前向掩码是硬 0/1（与 `K50` 同形），
    反向经 `softmax(s)` 直通（`t − t.detach()` 恒为精确 0，单测 `T6` 钉住前向逐位不变）。
    """

    def __init__(self, mlp: nn.Sequential, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.mlp = mlp
        self.arm = arm
        self.k = int(round(KEEP_FRAC[arm] * HIDDEN))
        self.score_a = nn.Parameter(torch.randn(D_MODEL, SCORER_RANK) / math.sqrt(D_MODEL),
                                    requires_grad=(arm == "L50"))
        self.score_b = nn.Parameter(torch.zeros(SCORER_RANK, HIDDEN),
                                    requires_grad=(arm == "L50"))
        # 构造时刻的打分器快照（`L50` 的「初始选择器」参照；不随训练变、不入 state_dict）
        self._init_ref = {"score_a": self.score_a.detach().clone(),
                          "score_b": self.score_b.detach().clone()}
        self._lb = None            # 本步负载均衡项（`L50` 训练时）
        self.rec = None            # 活性列累积器（评测期）
        self.counts = None
        self.n_tokens = 0
        self.mode_masks = None
        self.mode_batch = []

    # ---------- 打分器

    def _score(self, x: torch.Tensor, pre: torch.Tensor,
               a: torch.Tensor | None = None, b: torch.Tensor | None = None) -> torch.Tensor:
        a = self.score_a if a is None else a
        b = self.score_b if b is None else b
        if a.device != x.device:            # 初始打分器快照是构造时刻的 CPU 张量（不入 state_dict）
            a = a.to(x.device)
        if b.device != x.device:
            b = b.to(x.device)
        g = torch.nn.functional.gelu(x @ a) @ b                      # [B,T,H]
        # 数值护栏：`exp(g)` 限在 [e^-4, e^4]，防 12k 步里打分器漂到溢出（不影响 `B ≡ 0` 的起步）
        scale = torch.exp(g.float().clamp(-SCORER_CLAMP, SCORER_CLAMP))
        return pre.detach().float().abs().clamp_min(SCORER_EPS) * scale

    # ---------- 掩码

    def mask_for(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """返回 float32 的 0/1 硬掩码 `[B,T,H]`（`D` 全 1）。`L50` 走打分器。"""
        if self.arm == "D":
            return torch.ones_like(pre, dtype=torch.float32)
        if self.arm == "F50":
            m = torch.zeros_like(pre, dtype=torch.float32)
            m[..., : self.k] = 1
            return m
        if self.arm == "K50":
            s = pre.detach().float().abs()
        else:
            s = self._score(x, pre)
        idx = s.topk(self.k, dim=-1).indices
        m = torch.zeros_like(s)
        m.scatter_(-1, idx, 1)
        return m

    # ---------- 前向

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin1, act, lin2, drop = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3]
        pre = lin1(x)
        g = act(pre)
        self._lb = None
        if self.rec is not None:
            # 活性列要在**同一批激活**上比对规则：留下本层的输入与 pre-activation
            self.rec["pre"] = pre.detach().float()
            self.rec["x"] = x.detach()
        if self.arm in ("D", "F50", "K50"):
            mask = self.mask_for(x, pre).to(g.dtype)
            self._record(mask)
            return drop(lin2(g * mask))
        # ---- L50：硬掩码前向 + 直通梯度 + 负载均衡项
        s = self._score(x, pre)
        idx = s.topk(self.k, dim=-1).indices
        m = torch.zeros_like(s)
        m.scatter_(-1, idx, 1)
        p = torch.softmax(s, dim=-1)
        ps = p.gather(-1, idx)
        ps = ps / ps.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        ms = torch.zeros_like(p).scatter_(-1, idx, ps).to(g.dtype)
        t = g * ms
        gated = g * m.to(g.dtype) + (t - t.detach())      # 前向 = g*m（t 残项恒精确 0）
        if self.training:
            f = m.mean(dim=(0, 1)) * (HIDDEN / self.k)     # f̂：均值 1（硬、不回传）
            pbar = p.mean(dim=(0, 1))
            self._lb = (f * pbar).sum()
        self._record(m, s=s)
        return drop(lin2(gated))

    # ---------- 活性列（评测期）

    def _record(self, m: torch.Tensor, s: torch.Tensor | None = None) -> None:
        rec = self.rec
        with torch.no_grad():
            if self.counts is not None:
                self.counts += m.double().sum(dim=(0, 1)).cpu()
                self.n_tokens += int(m.shape[0] * m.shape[1])
            if self.mode_masks is not None:
                eq = (m.bool() == self.mode_masks.to(m.device)).all(dim=-1)
                self.mode_batch.append((float(eq.float().mean()), int(eq.numel())))
            if rec is not None and self.arm == "L50":
                k = self.k
                pre = rec["pre"]
                idx = s.topk(k, dim=-1).indices.sort(dim=-1).values
                pre_idx = pre.abs().topk(k, dim=-1).indices.sort(dim=-1).values
                rec["flip_vs_pre"][0] += int((idx != pre_idx).any(dim=-1).sum())
                rec["flip_vs_pre"][1] += int(idx.shape[0] * idx.shape[1])
                s0 = self._score(rec["x"], pre, self._init_ref["score_a"],
                                 self._init_ref["score_b"])
                i0 = s0.topk(k, dim=-1).indices.sort(dim=-1).values
                rec["flip_vs_init"][0] += int((idx != i0).any(dim=-1).sum())
                rec["flip_vs_init"][1] += int(idx.shape[0] * idx.shape[1])


def patch_ffn(model: DepthARModel, arm: str) -> DepthARModel:
    """把每块 `blk.mlp` 换成 `ScoredFFN`（只改前向的通道子集，不加**活**参数）。"""
    for blk in model.blocks:
        blk.mlp = ScoredFFN(blk.mlp, arm)
    return model


def scored_ffns(model: DepthARModel) -> list[ScoredFFN]:
    return [blk.mlp for blk in model.blocks]


class SelectorModel(DepthARModel):
    """在共享 `DepthARModel` 上只加一件事：训练时把负载均衡项加到 CE 上。

    `L50` 的 `aux_weight = ALPHA_LB`，其余臂 `0`（此时前向与共享实现逐位相同）。
    评测（`eval()`）不加任何项。
    """

    aux_weight = 0.0

    def forward(self, tokens, **kw):
        loss, aux = super().forward(tokens, **kw)
        if self.aux_weight and self.training:
            lbs = [f._lb for f in scored_ffns(self) if f._lb is not None]
            if lbs:
                loss = loss + self.aux_weight * torch.stack(lbs).mean()
        return loss, aux


def build_model(arm: str, vocab_size: int = VOCAB, seed: int = 0) -> SelectorModel:
    """所有臂共用同一构造路径（同 seed ⇒ 同一份主干抽样 + 同一份打分器初值）。"""
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True)
    model = SelectorModel(cfg, vocab_size)
    patch_ffn(model, arm)
    model.aux_weight = ALPHA_LB if arm == "L50" else 0.0
    return model


# ---------------------------------------------------------------- 代价（等效理论量）

def analytic_cost(arm: str, d: int = D_MODEL, h: int = HIDDEN, T: int = SEQ_LEN,
                  L: int = N_LAYERS, vocab: int = VOCAB) -> dict:
    """每 token 等价 MACs（解析，**不含任何时间量**）——契约 §3 的三列口径。

    * `ffn_macs_impl`（primary）：按内容选通道必须先算全部 pre-activation ⇒ `K*`/`L50` =
      `d·h`（lin1 全）+ `k·d`（lin2 按 k）；`F50`（固定前缀）可同时切两个矩阵 ⇒
      `(h/2)·d + d·(h/2)`；`D` = `2·h·d`。
    * `selector_macs`：**四臂同额**（`d·r + r·H` 每层）= 选择器预算；`D`/`K50`/`F50` 拿到
      这笔预算但不花（记在账上以保持等算力位点）。
    * `ffn_macs_contract`（并列）：按激活比例缩放 FFN 项 `2·h·d·(k/h)`。
    """
    if arm == "D":
        k = h
        ffn_impl = L * (2 * h * d)
    elif arm == "F50":
        k = h // 2
        ffn_impl = L * (d * k + k * d)
    else:
        k = int(round(KEEP_FRAC[arm] * h))
        ffn_impl = L * (d * h + k * d)
    ffn_contract = L * (2 * h * d) * (k / h)
    selector = L * SCORER_MACS_PER_LAYER
    attn = L * (4 * d * d + 2 * T * d)
    readout = d * vocab
    total = attn + ffn_impl + readout + selector
    dense_total = attn + L * 2 * h * d + readout + selector
    params = 6962688 + L * SCORER_PARAMS_PER_LAYER
    return {
        "d_model": d, "hidden": h, "seq_len": T, "n_layers": L, "vocab": vocab,
        "k_active_channels": k,
        "ffn_macs_impl": ffn_impl, "ffn_macs_contract": ffn_contract,
        "selector_macs_per_token": selector,
        "params_total": params,
        "macs_per_token": {"attention": attn, "ffn": ffn_impl, "selector": selector,
                           "readout_tied": readout, "total": total},
        "macs_total_rel_to_D": round(total / dense_total, 6),
        "ffn_macs_rel_to_D": round(ffn_impl / (L * 2 * h * d), 6),
        "note": ("等效理论量：MACs/token（乘加），不含时间量。实现口径（primary）= "
                 "lin1 全算 + lin2 按 k（K50/L50）/ 前缀切片（F50）；四臂同额选择器预算"
                 "（D/K50/F50 不花但记账）；契约口径 = 按激活比例缩放 FFN 项。"),
    }


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory_for(arm: str):
    def factory(cfg, vocab_size):
        model = SelectorModel(cfg, vocab_size)
        patch_ffn(model, arm)
        model.aux_weight = ALPHA_LB if arm == "L50" else 0.0
        return model
    return factory


def shared_argv(run_name: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    return [os.path.join(ROOT, "train_depth_ar.py"), run_name,
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory_for(arm)
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(f"{arm}-s{seed}-{steps // 1000}k", seed, steps,
                               batch_size, lr, out_root)
        t0 = time.time()
        TDA.main()
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


# ---------------------------------------------------------------- 评测（本实验自持）

def adapt_state(state: dict) -> dict:
    """把 stock 键 `blocks.i.mlp.<j>.weight` 改名为包装后的 `blocks.i.mlp.mlp.<j>.weight`。

    包装层只是多一层命名，权重张量不变（`2641` 的 `U4-s0` 是 stock 结构存的；`2644` 的
    `K50`/`F50` 已经是包装结构，键名不变）。不改名就会把 FFN 权重当成 `unexpected` 丢掉，
    而 `strict=False` 只报不熟——那个门就是 `T1` 与复现门要防的事。
    """
    out = {}
    for k, v in state.items():
        if k.startswith("blocks.") and ".mlp." in k and ".mlp.mlp." not in k:
            head, _, tail = k.partition(".mlp.")
            if tail[:1].isdigit():          # 只改 `blocks.i.mlp.<j>.*`（Sequential 下标），
                k = f"{head}.mlp.mlp.{tail}"  # 打分器键 `blocks.i.mlp.score_*` 原地不动
        out[k] = v
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_audit(model: SelectorModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(p.numel() for blk in model.blocks for p in blk.mlp.mlp.parameters())
    scorer = sum(p.numel() for blk in model.blocks
                 for p in (blk.mlp.score_a, blk.mlp.score_b))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_ffn": ffn, "params_selector": scorer,
            "params_attn_and_rest": total - ffn - scorer,
            "params_trainable": trainable}


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 6), round(centre + half, 6)]


def cluster_bootstrap_ci(rates: list, n_iter: int = 2000, seed: int = 1234) -> list:
    if not rates:
        return [None, None]
    g = torch.Generator().manual_seed(seed)
    n = len(rates)
    ms = sorted(sum(rates[int(i)] for i in torch.randint(0, n, (n,), generator=g)) / n
                for _ in range(n_iter))
    return [round(ms[int(0.025 * n_iter)], 6),
            round(ms[min(int(0.975 * n_iter), n_iter - 1)], 6)]


def scorer_move(model: SelectorModel) -> dict:
    """G1 的第一个量：打分器参数相对初始值的位移（相对 L2）。"""
    num, den, per_layer = 0.0, 0.0, []
    for f in scored_ffns(model):
        d = 0.0
        for name in ("score_a", "score_b"):
            cur = getattr(f, name).detach().float()
            init = f._init_ref[name].float().to(cur.device)   # 快照是构造时刻的 CPU 张量
            d += float(((cur - init) ** 2).sum())
            den += float((init ** 2).sum())
        num += d
        per_layer.append(round(math.sqrt(d), 8))
    return {"scorer_move_l2": round(math.sqrt(num), 8),
            "init_l2": round(math.sqrt(den), 8),
            "scorer_move_relative": round(math.sqrt(num) / max(math.sqrt(den), 1e-12), 8),
            "per_layer_l2": per_layer}


@torch.no_grad()
def selector_profile(model: SelectorModel, eval_arr, n_batches: int = 20,
                     batch_size: int = 16, gen_seed: int = 2) -> dict:
    """活性列：逐层通道使用率/熵、逐 token 选择一致率（坍缩上界）、
    `L50` 的选择翻转率（对 `|pre|` 规则、对初始打分器）。"""
    T = model.cfg.seq_len
    ffns = scored_ffns(model)
    k = ffns[0].k
    model.eval()
    for f in ffns:
        f.counts = torch.zeros(HIDDEN, dtype=torch.float64)
        f.n_tokens = 0
        f.mode_masks = None
        f.mode_batch = []
        f.rec = {"flip_vs_pre": [0, 0], "flip_vs_init": [0, 0], "pre": None, "x": None}
    # --- pass 1：逐通道计数（同时采 L50 的翻转率）
    gen = torch.Generator().manual_seed(gen_seed)
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        model(x.to(DEVICE), exit_depths=None)
    counts = [f.counts for f in ffns]
    n_tok = [f.n_tokens for f in ffns]
    flips = {name: [f.rec[name] for f in ffns] for name in ("flip_vs_pre", "flip_vs_init")}
    for f in ffns:
        f.counts = None
        f.rec = None
    mode_sets = [torch.topk(c, k).indices.sort().values for c in counts]
    layer_stats = []
    for l in range(len(ffns)):
        q = (counts[l] / (n_tok[l] * k)).clamp_min(1e-12)
        ent = float(-(q * q.log()).sum())
        layer_stats.append({
            "entropy_nats": round(ent, 6),
            "entropy_over_logH": round(ent / math.log(HIDDEN), 6),
            "uniform_value": 1.0,
            "collapse_value_k": round(math.log(k) / math.log(HIDDEN), 6),
            "channels_ever_used": int((counts[l] > 0).sum()),
        })
    # --- pass 2：选择一致率（与逐层众数集比对，`F50` = 1.0 是构造性上界）
    mode_masks = []
    for l in range(len(ffns)):
        m = torch.zeros(HIDDEN, dtype=torch.bool)
        m[mode_sets[l]] = True
        mode_masks.append(m)
    for f, m in zip(ffns, mode_masks):
        f.mode_masks = m
        f.mode_batch = []
    gen = torch.Generator().manual_seed(gen_seed)
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        model(x.to(DEVICE), exit_depths=None)
    per_batch_layer = [[float(r) for r, _ in f.mode_batch] for f in ffns]
    n_per_layer = [[int(n) for _, n in f.mode_batch] for f in ffns]
    for f in ffns:
        f.mode_masks = None
    per_batch = [sum(r * n for r, n in zip(per_batch_layer[l], n_per_layer[l]))
                 / max(sum(n_per_layer[l]), 1) for l in range(len(ffns))]
    matches = sum(r * n for l in range(len(ffns))
                  for r, n in zip(per_batch_layer[l], n_per_layer[l]))
    tot = sum(sum(nl) for nl in n_per_layer)
    out = {
        "k_active_channels": k, "hidden": HIDDEN,
        "n_batches": n_batches, "batch_size": batch_size, "n_tokens_per_layer": n_tok,
        "by_layer": layer_stats,
        "mode_set_share": round(matches / max(tot, 1), 6),
        "mode_set_share_wilson_95ci": wilson_ci(int(matches), int(tot)),
        "mode_set_share_cluster_bootstrap_95ci": cluster_bootstrap_ci(per_batch),
    }
    if any(sum(v) for v in flips["flip_vs_init"]):
        for name, label in (("flip_vs_pre", "对 |pre| 规则"),
                            ("flip_vs_init", "对初始打分器")):
            num = sum(v[0] for v in flips[name])
            den = sum(v[1] for v in flips[name])
            out[name] = {
                "rate": round(num / max(den, 1), 6),
                "n_positions_x_layers": int(den),
                "wilson_95ci": wilson_ci(int(num), int(den)),
                "note": f"每个 (层,位置) 的选择集与{label}的选择集不完全相同的比例",
            }
    out["note"] = ("entropy_over_logH 两端：完全坍缩 = log(k)/log(H)（k=512 时 0.9）、"
                   "均匀随机 = 1.0；F50 构造性地落在坍缩端且 mode_set_share = 1.0。")
    return out


@torch.no_grad()
def causal_check(model: SelectorModel, gen_seed: int = 7) -> dict:
    g = torch.Generator().manual_seed(gen_seed)
    T = model.cfg.seq_len
    x = torch.randint(0, VOCAB, (2, T), generator=g).to(DEVICE)
    x2 = x.clone()
    j = T // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, T - j), generator=g).to(DEVICE)
    model.eval()
    _, a1 = model(x, exit_depths=None)
    _, a2 = model(x2, exit_depths=None)
    lg1 = a1["depth_logits"][-1][:, :j].float()
    lg2 = a2["depth_logits"][-1][:, :j].float()
    return {"max_abs_diff_before_j": float((lg1 - lg2).abs().max()),
            "bitwise_equal_before_j": bool(torch.equal(lg1, lg2))}


@torch.no_grad()
def load_balance_probe(model: SelectorModel, eval_arr, n_batches: int = 4,
                       batch_size: int = 8, gen_seed: int = 11) -> dict:
    """负载均衡项在**已训权重**上的取值（`L50` 的 aux；其余臂 None）。

    均衡项 = `Σ_j f̂_j · P_j`（`f̂` = 通道被选频率按 `H/k` 归一、`P` = `softmax(s)` 的均值）：
    完美均匀 ⇒ 1.0。同时在 `train()` 模式跑一次前向取该项（它会随 `self.training` 计算）。
    """
    if model.aux_weight == 0:
        return {"aux_lb": None, "note": "本臂无负载均衡项（aux_weight = 0）"}
    T = model.cfg.seq_len
    gen = torch.Generator().manual_seed(gen_seed)
    vals = []
    model.train()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        with torch.enable_grad():
            model(x.to(DEVICE), exit_depths=None)
        lbs = [float(f._lb.detach()) for f in scored_ffns(model) if f._lb is not None]
        vals.append(sum(lbs) / max(len(lbs), 1))
    model.eval()
    return {"aux_lb": round(statistics.mean(vals), 6), "alpha": ALPHA_LB,
            "weighted_term_on_loss": round(ALPHA_LB * statistics.mean(vals), 6),
            "n_batches": n_batches,
            "note": "均衡项 = Σ_j f̂_j·P_j；完美均匀 = 1.0，越大 = 选择频率与路由概率越不一致"}


@torch.no_grad()
def evaluate(arm: str, seed: int, steps: int, out_root: str, ckpt: str | None = None,
             n_batches: int = 20) -> dict:
    rd = run_dir(arm, seed, steps, out_root)
    ckpt_path = ckpt or os.path.join(rd, "model.pt")
    ckpt_abs = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(ROOT, ckpt_path)
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    model = build_model(arm, vocab, seed).to(DEVICE)
    state = torch.load(ckpt_abs, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(adapt_state(state), strict=False)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    curve2 = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                         device=DEVICE)
    out = {
        "arm": arm, "seed": seed, "steps": steps, "keep_frac": KEEP_FRAC[arm],
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16, "eval_generator_seed": 2,
                 "bpc_by_depth": curve, "clean_bpc": curve[-1]["bpc"]},
        "params": param_audit(model),
        "cost": analytic_cost(arm, vocab=vocab),
        "selector_profile": selector_profile(model, eval_arr, n_batches=n_batches),
        "load_balance": load_balance_probe(model, eval_arr),
        "causal": causal_check(model),
        "reeval_bitwise_identical": bool(curve == curve2),
        "checkpoint": {"path": os.path.relpath(ckpt_abs, ROOT),
                       "sha256": sha256_file(ckpt_abs),
                       "missing_keys": list(missing), "unexpected_keys": list(unexpected)},
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    if arm == "L50":
        out["scorer_move"] = scorer_move(model)
    if os.path.exists(os.path.join(rd, "train_log.jsonl")):
        rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
                if l.strip()]
        out["loss_slope"] = loss_slope(rows, steps)
        out["lb_trace"] = lb_trace(rows)
        shared = json.load(open(os.path.join(rd, "results.json")))
        out["shared_train_results"] = {
            "params": shared.get("params"),
            "train_tokens": shared.get("train_tokens"),
            "wall_minutes": shared.get("wall_minutes"),
            "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb")}
        out["run_dir"] = os.path.relpath(rd, ROOT)
    with open(os.path.join(HERE, f"results_{arm}_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    prof = out["selector_profile"]
    print(f"[2687] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"params={out['params']['params_total']} "
          f"mode_set_share={prof['mode_set_share']} "
          f"entropy/logH={prof['by_layer'][0]['entropy_over_logH']}"
          + (f" flip_pre={prof['flip_vs_pre']['rate']} "
             f"flip_init={prof['flip_vs_init']['rate']} "
             f"move={out['scorer_move']['scorer_move_relative']}" if arm == "L50" else ""),
          flush=True)
    return out


def loss_slope(rows: list, steps: int, frac: float = 0.25) -> dict:
    start = int(steps * (1 - frac))
    win = [r for r in rows if start <= r["step"] <= steps]
    if len(win) < 2:
        return {"n_points": len(win), "slope_per_1k_steps": None}
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - xm) ** 2 for x in xs)
    slope = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
             if den else float("nan"))
    return {"n_points": len(win), "slope_per_1k_steps": round(slope * 1000, 5),
            "window": f"末 {int(frac * 100)}%"}


def lb_trace(rows: list) -> dict:
    """训练日志里 `loss` = CE + α·aux（`L50`），`depth_ce[0]` = CE（dense 臂只有一个深度）
    ⇒ `aux = (loss − depth_ce[0]) / α`（日志按 4 位小数四舍五入，读数有 ~1e-4/α 的量化）。"""
    if not rows or "depth_ce" not in rows[0]:
        return {"available": False}
    aux = [(r["loss"] - r["depth_ce"][0]) / ALPHA_LB for r in rows
           if r.get("depth_ce") and len(r["depth_ce"]) == 1]
    if not aux:
        return {"available": False}
    return {"available": True, "n_points": len(aux),
            "aux_first": round(aux[0], 3), "aux_last": round(aux[-1], 3),
            "aux_mean": round(statistics.mean(aux), 3),
            "note": "aux 由 train_log.jsonl 的 (loss − depth_ce[0]) / ALPHA_LB 反推；"
                    "非 L50 臂该差恒 ~0（无均衡项）"}


# ---------------------------------------------------------------- 汇总（G1–G3）

def _load(name: str) -> dict | None:
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def aggregate(arms: list, seeds: list) -> dict:
    recs = {}
    for arm in arms:
        for s in seeds:
            r = _load(f"results_{arm}_s{s}.json")
            if r is not None:
                recs[(arm, s)] = r
    s0 = seeds[0]
    rows, repro = {}, {}
    for arm in arms:
        r = recs.get((arm, s0))
        if r is None:
            rows[arm] = {"clean_bpc": None, "note": "未跑"}
            continue
        rows[arm] = {"clean_bpc": r["eval"]["clean_bpc"],
                     "ffn_macs_impl": r["cost"]["ffn_macs_impl"],
                     "selector_macs": r["cost"]["selector_macs_per_token"],
                     "macs_total": r["cost"]["macs_per_token"]["total"],
                     "macs_total_rel_to_D": r["cost"]["macs_total_rel_to_D"],
                     "params_total": r["params"]["params_total"],
                     "params_trainable": r["params"]["params_trainable"],
                     "entropy_over_logH": r["selector_profile"]["by_layer"][0]["entropy_over_logH"],
                     "mode_set_share": r["selector_profile"]["mode_set_share"],
                     "reeval_bitwise_identical": r["reeval_bitwise_identical"],
                     "code_hash_run_py": r["code_hash_run_py"]}
    dense = recs.get(("D", s0))
    if dense is not None:
        dv = dense["eval"]["clean_bpc"]
        for arm in arms:
            if rows.get(arm, {}).get("clean_bpc") is not None:
                rows[arm]["delta_vs_D"] = round(rows[arm]["clean_bpc"] - dv, 4)
    # 复现门：复用 checkpoint 的读数必须与本条同一代码路径重评的结果逐位相同
    for (arm, s), meta in REUSE.items():
        if (arm, s) in recs:
            got = recs[(arm, s)]["eval"]["clean_bpc"]
            repro[f"{arm}-s{s}"] = {"published_bpc": meta["published_bpc"],
                                    "reproduced_bpc": got,
                                    "delta": round(got - meta["published_bpc"], 4),
                                    "bitwise_equal": got == meta["published_bpc"],
                                    "source": meta["source"]}
    # 逐 seed 配对差
    def d(a_arm, b_arm, s):
        a, b = recs.get((a_arm, s)), recs.get((b_arm, s))
        if a is None or b is None:
            return None
        return round(a["eval"]["clean_bpc"] - b["eval"]["clean_bpc"], 4)

    pairs = {"L50_minus_K50": ("L50", "K50"), "L50_minus_F50": ("L50", "F50"),
             "L50_minus_D": ("L50", "D"), "K50_minus_D": ("K50", "D")}
    per_seed = {name: {f"s{s}": d(a, b, s) for s in seeds} for name, (a, b) in pairs.items()}

    def band_verdict(v):
        if v is None:
            return "未跑"
        if v <= -BAND:
            return "可排序更优（越带）"
        if v >= BAND:
            return "可排序更差（越带）"
        return "不可排序（带内）"

    pair_verdict = {name: {k: band_verdict(v) for k, v in per_seed[name].items()}
                    for name in pairs}
    # G1：打分器活性（两 seed 都要过）
    g1 = {}
    for s in seeds:
        r = recs.get(("L50", s))
        if r is None:
            continue
        mv = r["scorer_move"]["scorer_move_relative"]
        fp = r["selector_profile"].get("flip_vs_init")
        fi = r["selector_profile"].get("flip_vs_pre")
        g1[f"s{s}"] = {
            "scorer_move_relative": mv, "scorer_move_threshold": SCORER_MOVE_MIN,
            "flip_vs_init": (fp or {}).get("rate"),
            "flip_vs_init_wilson_95ci": (fp or {}).get("wilson_95ci"),
            "flip_vs_pre": (fi or {}).get("rate"),
            "flip_vs_pre_wilson_95ci": (fi or {}).get("wilson_95ci"),
            "move_pass": mv >= SCORER_MOVE_MIN,
            "flip_pass": bool(fp and fp["rate"] >= FLIP_MIN
                              and fp["wilson_95ci"][0] is not None
                              and fp["wilson_95ci"][0] > 0),
        }
        g1[f"s{s}"]["pass"] = g1[f"s{s}"]["move_pass"] and g1[f"s{s}"]["flip_pass"]
    g1_pass = bool(g1) and all(v["pass"] for v in g1.values())
    # 判决（预登记映射，契约 §4）
    dk = [v for v in per_seed["L50_minus_K50"].values() if v is not None]
    dd = [v for v in per_seed["L50_minus_D"].values() if v is not None]
    df = [v for v in per_seed["L50_minus_F50"].values() if v is not None]
    same = (len(dk) >= 2 and len({v > 0 for v in dk}) == 1)
    if not g1_pass:
        verdict = ("本尺度不可判：打分器在 13M / 12k 步下没被训练动"
                   "（G1 活性门未过）⇒ 机制没有被检验到")
        verdict_code = "inconclusive_at_this_scale"
    elif same and all(v >= BAND for v in dk):
        verdict = ("机制不成立：可学习打分器（已证明被训练动）比无参数 |pre| 规则可排序更差"
                   "⇒ 上一轮「条件性有害」不是「无参数」这一个实现的毛病")
        verdict_code = "mechanism_refuted"
    elif same and all(v <= -BAND for v in dk):
        gains = (any(v <= -BAND for v in dd) or any(v <= -BAND for v in df))
        verdict = (("本实现不成立：可学习打分器把选择质量改到了可排序更好，且在同一每位置预算下"
                    "越过 dense/固定前缀 ⇒ 上一轮的负结论是那个无参数实现的毛病")
                   if gains else
                   ("机制部分成立但有条件：可学习打分器比无参数规则可排序更好，"
                    "但仍未越过 dense 锚 ⇒ 前沿上不占优"))
        verdict_code = "implementation_refuted" if gains else "mechanism_partial"
    else:
        verdict = ("本尺度不可判：可学习打分器与无参数规则/稠密锚的差都在 0.014 带内或跨 seed 方向不一致"
                   "⇒ 本 n 下没有可排序的机制差")
        verdict_code = "inconclusive_at_this_scale"
    # 所需样本量（预登记公式）：n = ceil((2σ/band)²)，σ = Δ(L50−K50) 的跨 seed SD
    need = None
    if len(dk) >= 2:
        sd = statistics.pstdev(dk)
        need = {"sigma_cross_seed": round(sd, 6),
                "formula": "n = ceil((2·σ/band)²)（两样本均值差的可排序门；σ = 跨 seed SD）",
                "n_seeds_needed": (None if sd == 0 else
                                   math.ceil((2 * sd / BAND) ** 2))}
    out = {
        "experiment": "2687-learned-channel-selector-13m",
        "criterion": ("G1 打分器活性门（scorer_move_relative ≥ 0.01 且 flip_vs_init ≥ 0.05、"
                      "Wilson 下界 > 0）；G2 主判 Δ(L50−K50) 对 0.014 带（两 seed 同向）；"
                      "G3 前沿门 Δ(L50−D)/Δ(L50−F50) 对 0.014 带。"),
        "band": BAND,
        "delta_directions": {
            "delta_vs_D": "arm_minus_comparator",
            "delta": "arm_minus_comparator",
        },
        "params": {"params_total": dense["params"]["params_total"] if dense else None,
                   "selector_params": dense["params"]["params_selector"] if dense else None,
                   "rule": "四臂都带同一个打分器模块 ⇒ 总参数逐臂逐位相同；L50 的 trainable 多 81 920"},
        "arms": rows,
        "pairs": per_seed,
        "pair_verdict": pair_verdict,
        "reproduction": repro,
        "G1_liveness": {"per_seed": g1, "pass": g1_pass,
                        "thresholds": {"scorer_move_relative": SCORER_MOVE_MIN,
                                       "flip_vs_init": FLIP_MIN}},
        "load_balance": {f"{a}-s{s}": recs[(a, s)]["load_balance"]
                         for (a, s) in sorted(recs) if recs[(a, s)]["load_balance"]["aux_lb"] is not None},
        "n_seeds": len({s for (_, s) in recs}),
        "seeds_measured": sorted({s for (_, s) in recs}),
        "verdict_branch": {"code": verdict_code, "text": verdict, "n_seeds_needed": need},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("arms", "pairs", "pair_verdict",
                                          "G1_liveness", "verdict_branch")},
                     indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--skip-evaluate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default="D,K50,F50,L50")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.arms.split(","),
                  [int(s) for s in args.seeds.split(",") if s.strip() != ""])
        return
    if not args.arm:
        raise SystemExit("--arm 或 --aggregate 必给其一")
    if not args.no_train:
        os.makedirs(args.out_root, exist_ok=True)
        info = train_arm(args.arm, args.seed, args.steps, args.batch_size,
                         args.lr, args.out_root)
        rd = run_dir(args.arm, args.seed, args.steps, args.out_root)
        with open(os.path.join(rd, "train_wall.json"), "w") as f:
            json.dump(info, f)
    if args.skip_evaluate:
        return
    ckpt = args.ckpt or REUSE.get((args.arm, args.seed), {}).get("ckpt")
    evaluate(args.arm, args.seed, args.steps, args.out_root, ckpt=ckpt)


if __name__ == "__main__":
    main()
