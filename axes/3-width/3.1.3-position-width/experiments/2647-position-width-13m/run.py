"""2647 · 同层内位置相关宽度（前 128 / 后 128 位置走不同宽度子网络）· 13M · dense.

契约 = 同目录 `README.md`（判据线 P1–P4 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2.1）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成工厂（构造 stock 模型后把每块 `blk.forward` 换成 `position_width_forward`），
  再调 `train_depth_ar.main()` —— 参数解析 / 配置装配 / `train()` / checkpoint /
  `results.json` 全是共享实现，本文件不改 `pathlm/**`、`train_*.py`。
* **唯一变量 = 同一层内每个位置参与前向的通道前缀长度**（段由绝对位置决定）。
  不新增、不裁剪任何参数 ⇒ 各臂总参数逐位相同（`2641` 锚 `U4-s0` = 6,962,688）。
* **可见集合不变**：注意力仍是 stock 的 `SDPA(qs, ks, vs, attn_mask=None,
  is_causal=True)` 同一个调用（单测 T2 记录调用参数逐层比对）。
* **等值门**：位置宽度恒 256 时前向**走 stock 的 `Block.forward` 原实现** ⇒ 逐位相同（T1）。
* 窄段位置 = 只在残差流前 w 维上运算（LN/前缀子矩阵 + qkv 尾部头置零 + FFN 隐藏 = 4w）；
  `GELU(0)=0` 使尾部隐藏单元对输出的贡献恒 0 ⇒ 逐位等价于「宽 w 的子网络」。

用法：

    python3 run.py --arm PN --seed 0 --steps 12000              # 训练 + 评测
    python3 run.py --arm PN --seed 0 --steps 12000 --no-train    # 仅评测（复用 model.pt）
    python3 run.py --synth-probe --arm PN --seed 0               # 合成召回探针（纯推理）
    python3 run.py --synth-probe --arm FULL --ckpt <model.pt> --tag anchor2641U4-s0
    python3 run.py --aggregate                                   # 汇总 P1–P4 → results.json
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                   # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full, batch                # noqa: E402
from pathlm.model import Block                                 # noqa: E402
from tools.evalkit.longrange_tasks import assoc_recall_multi_batch  # noqa: E402
from tools.evalkit.retention import masked_logits              # noqa: E402
from tools.evalkit.carry_channel import wilson_ci              # noqa: E402

DEVICE = "cuda"

# ---------------------------------------------------------------- 臂定义（契约 §3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
HIDDEN = MLP_MULT * D_MODEL                                    # 1024
VOCAB = 206
NW = 128                        # 窄段宽度
SPLIT = 128                     # 位置分段点（前 128 / 后 128）
VARIANCE_BOUND = 0.014

# 锚（2641 `U4-s0`，契约 §4：三道门复用）
ANCHOR_BPC = 1.5314
ANCHOR_PARAMS = 6962688
ANCHOR_MACS = 7392768
DATA_RAW = os.path.join(ROOT, ".tmp/enwik8")
DATA_CACHE = os.path.join(ROOT, "data/enwik8_full.npz")
ANCHOR_CKPT = os.path.join(
    ROOT, "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m",
    "runs/U4-s0-12k/model.pt")

# 位置 → 宽度（逐位置的激活通道前缀长度）
WIDTH_BY_POS = {
    "FULL": [D_MODEL] * SEQ_LEN,
    "PN": [D_MODEL] * SPLIT + [NW] * (SEQ_LEN - SPLIT),        # 前宽后窄
    "NP": [NW] * SPLIT + [D_MODEL] * (SEQ_LEN - SPLIT),        # 前窄后宽
}
ALL_ARMS = ("FULL", "PN", "NP")
TRAIN_ARMS = ("PN", "NP")

# 合成召回探针（契约 §5-P3）
SYNTH = {"K": 8, "Q": 8, "gap": 14, "sym_vocab": 64, "filler_token": 64,
         "cue_token": 205, "n_batches": 4, "batch_size": 16, "seed": 6001}


# ---------------------------------------------------------------- 位置宽度：前向实现

class PositionWidthSpec:
    """一次前向里「位置 → 激活通道前缀长度」的映射（契约 §2.1）。"""

    def __init__(self, arm: str, d: int = D_MODEL, nw: int = NW,
                 hidden: int = HIDDEN, seq_len: int = SEQ_LEN):
        if arm not in WIDTH_BY_POS:
            raise ValueError(f"unknown arm {arm!r}; expected {sorted(WIDTH_BY_POS)}")
        self.arm = arm
        self.d = int(d)
        self.nw = int(nw)
        self.hidden = int(hidden)
        self.width_by_pos = list(WIDTH_BY_POS[arm])
        if len(self.width_by_pos) != seq_len:
            raise ValueError(f"宽度表长度 {len(self.width_by_pos)} != seq_len "
                             f"{seq_len}")
        for i, w in enumerate(self.width_by_pos):
            if not 0 < w <= self.d:
                raise ValueError(f"位置 {i} 宽度 {w} 不在 (0, {self.d}]")

    @property
    def all_full(self) -> bool:
        return all(w == self.d for w in self.width_by_pos)

    def narrow_mask(self, x: torch.Tensor) -> torch.Tensor:
        """[1, T, 1] bool：该位置是否是窄段（契约里唯一「按位置」的量）。"""
        w = torch.as_tensor(self.width_by_pos, device=x.device)
        return (w < self.d).view(1, -1, 1)

    def keep_channels(self, x: torch.Tensor) -> torch.Tensor:
        """[1, T, d] bool：该位置参与前向的通道前缀（`k(p) = {0..w(p)−1}`）。"""
        w = torch.as_tensor(self.width_by_pos, device=x.device)
        idx = torch.arange(self.d, device=x.device)
        return (idx.view(1, 1, -1) < w.view(1, -1, 1))

    def keep_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """[1, T, hidden] bool：FFN 隐藏维前缀（隐藏宽 = mlp_mult·w(p)）。"""
        w = torch.as_tensor(self.width_by_pos, device=x.device) * MLP_MULT
        idx = torch.arange(self.hidden, device=x.device)
        return (idx.view(1, 1, -1) < w.view(1, -1, 1))

    def summary(self) -> dict:
        n_narrow = sum(1 for w in self.width_by_pos if w < self.d)
        return {"arm": self.arm, "d_model": self.d, "narrow_width": self.nw,
                "split": SPLIT, "n_narrow_positions": n_narrow,
                "width_by_position_segment":
                    {"0..127": self.width_by_pos[0],
                     "128..255": self.width_by_pos[-1]},
                "all_full": self.all_full}


def _prefix_ln(x: torch.Tensor, ln, nw: int) -> torch.Tensor:
    """在残差流前 `nw` 维上做 LayerNorm（用 `ln` 的**前缀**参数），返回 [B,T,nw]。"""
    return F.layer_norm(x[..., :nw], (nw,), ln.weight[:nw], ln.bias[:nw], ln.eps)


def position_width_forward(blk: Block, spec: PositionWidthSpec, x: torch.Tensor,
                           attn_mask: torch.Tensor | None = None,
                           dist_pen: float = 0.0,
                           kv_from: torch.Tensor | None = None,
                           gate_vec: torch.Tensor | None = None):
    """把 `Block.forward` 换成「按位置取通道前缀」的等价前向（契约 §2.1）。

    返回 `(x, None)`（本干预不带遥测）。`dist_pen` / `kv_from` / `gate_vec` /
    非 None 的 `attn_mask` 本实验都不用（dense + 无窗口）⇒ 显式拒绝，避免静默走错路径。
    """
    if attn_mask is not None or dist_pen > 0 or kv_from is not None \
            or gate_vec is not None:
        raise ValueError("position_width_forward 只支持 dense 无窗口路径"
                         "（attn_mask/dist_pen/kv_from/gate_vec 都必须缺省）")
    if spec.all_full:
        # 等值门 T1：不缩时**走 stock 原实现**（同一条代码路径 ⇒ 逐位相同）
        return Block.forward(blk, x)

    B, T, d = x.shape
    keep = spec.keep_channels(x)                     # [1,T,d]
    keepf = spec.keep_hidden(x)                      # [1,T,hidden]
    narrow = spec.narrow_mask(x)                     # [1,T,1]

    # --- ① LN1：宽段整宽 LN；窄段在前 nw 维上 LN（前缀参数）
    h1 = torch.where(narrow, F.pad(_prefix_ln(x, blk.ln1, spec.nw),
                                   (0, d - spec.nw)), blk.ln1(x))
    # --- ② qkv：窄段输入尾部置零 + 输出尾部（头 4..7）置零 = 权重前缀子矩阵 W[:w,:w]
    q, k, v = blk.qkv(torch.where(keep, h1, 0.0)).chunk(3, dim=-1)
    q = torch.where(keep, q, 0.0)
    k = torch.where(keep, k, 0.0)
    v = torch.where(keep, v, 0.0)
    shape = lambda t: t.view(B, T, N_HEADS, d // N_HEADS).transpose(1, 2)
    # --- ③ 注意力：与 stock **同一个调用**（可见集合逐位不变）
    a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                       attn_mask=None, is_causal=True)
    a = a.transpose(1, 2).reshape(B, T, d)
    # --- ④ 输出投影前缀子矩阵 W_proj[:w,:w]，残差只加在前缀通道上
    x = x + torch.where(keep, blk.proj(a), 0.0)
    # --- ⑤ LN2 / FFN：隐藏维 = mlp_mult·w(p)，同样按前缀子矩阵；GELU(0)=0
    h2 = torch.where(narrow, F.pad(_prefix_ln(x, blk.ln2, spec.nw),
                                   (0, d - spec.nw)), blk.ln2(x))
    pre = torch.where(keepf, blk.mlp[0](torch.where(keep, h2, 0.0)), 0.0)
    out = torch.where(keep, blk.mlp[2](blk.mlp[1](pre)), 0.0)
    x = x + blk.mlp[3](out)
    return x, None


def patch_position_width(model: DepthARModel, arm: str,
                         spec: "PositionWidthSpec | None" = None) -> DepthARModel:
    """把每块的 `forward` 换成按位置取通道前缀的实现（不新增/裁剪任何参数）。

    `spec` 缺省由臂名构造；单测 T4（变异）传入朝向错的 spec 以证明门会变红。
    """
    if spec is None:
        spec = PositionWidthSpec(arm, d=model.cfg.d_model,
                                 hidden=MLP_MULT * model.cfg.d_model,
                                 seq_len=model.cfg.seq_len)
    if any(hasattr(b, "_pw_spec") for b in model.blocks):
        raise RuntimeError("模型已被 patch 过一次（避免双重包装）")
    for blk in model.blocks:
        blk._pw_spec = spec
        blk.forward = (lambda b=blk, s=spec:
                       (lambda x, attn_mask=None, dist_pen=0.0, kv_from=None,
                               gate_vec=None:
                        position_width_forward(b, s, x, attn_mask, dist_pen,
                                               kv_from, gate_vec)))()
    model._pw_spec = spec
    return model


def build_model(arm: str, vocab_size: int = VOCAB, seed: int = 0,
                patch: bool = True) -> DepthARModel:
    """所有臂共用同一构造路径（同 seed ⇒ 同一次随机抽样）；patch 不改任何参数。"""
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True)
    model = DepthARModel(cfg, vocab_size)
    return patch_position_width(model, arm) if patch else model


# ---------------------------------------------------------------- 代价（等效理论量）

def analytic_cost(arm: str, d: int = D_MODEL, T: int = SEQ_LEN,
                  n_layers: int = N_LAYERS, vocab: int = VOCAB,
                  nw: int = NW) -> dict:
    """每 token 等效 MACs（解析，**不含任何时间量**）与平均激活通道数（契约 §3）。

    每层每 token（宽 w）：attention `4w² + 2Tw`、FFN `2·(4w)·w = 8w²`；
    按位置段加权（段由 `WIDTH_BY_POS[arm]` 给定），再加 tied 读出 `d·V`。
    """
    spec = PositionWidthSpec(arm, d=d, nw=nw, hidden=MLP_MULT * d, seq_len=T)
    per_pos = [4 * w * w + 2 * T * w + 8 * w * w for w in spec.width_by_pos]
    attn_per_pos = [4 * w * w + 2 * T * w for w in spec.width_by_pos]
    ffn_per_pos = [8 * w * w for w in spec.width_by_pos]
    avg1 = statistics.mean(per_pos)
    readout = d * vocab
    total = n_layers * avg1 + readout
    full = analytic_full_macs(d=d, T=T, n_layers=n_layers, vocab=vocab)
    return {
        "d_model": d, "seq_len": T, "n_layers": n_layers, "vocab": vocab,
        "width_by_position_segment": {"0..127": spec.width_by_pos[0],
                                      "128..255": spec.width_by_pos[-1]},
        "avg_activated_channels_per_token": statistics.mean(spec.width_by_pos),
        "macs_per_token": {
            "attention": round(n_layers * statistics.mean(attn_per_pos), 3),
            "ffn": round(n_layers * statistics.mean(ffn_per_pos), 3),
            "readout_tied": readout,
            "total": round(total, 3),
        },
        "macs_rel_to_full": round(total / full, 6),
        "note": ("等效理论量：MACs/token（乘加）。attention 的 T·w 项随序列长度线性、"
                 "FFN 的 8w² 项与 T 无关；两列分开报，不用任何时间量。"
                 "tokens-per-forward = 1（本格不涉及多 token 前向）。"),
    }


def analytic_full_macs(d: int = D_MODEL, T: int = SEQ_LEN,
                       n_layers: int = N_LAYERS, vocab: int = VOCAB) -> int:
    """stock 全宽 13M 的每 token 等效 MACs（与 `2641` `U4-s0` 实测值对账）。"""
    per_layer = 4 * d * d + 2 * T * d + 8 * d * d
    return n_layers * per_layer + d * vocab


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory_for(arm: str):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 main() 调它建模型。"""
    def factory(cfg, vocab_size):
        model = DepthARModel(cfg, vocab_size)
        return patch_position_width(model, arm)
    return factory


def shared_argv(run_name: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `2641` 的 `shared_argv` 逐字段相同（唯一变量 = 位置→宽度）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"),
            run_name,
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


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

def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_audit(model: DepthARModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"params_total": total,
            "params_matches_anchor": total == ANCHOR_PARAMS,
            "anchor_params": ANCHOR_PARAMS}


def _logits(model: DepthARModel, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(tokens)
    return out[1]["depth_logits"][-1].float()


@torch.no_grad()
def activation_profile(model: DepthARModel, eval_arr, n_batches: int = 2,
                       batch_size: int = 8, seed: int = 1234) -> dict:
    """实测活性（`harness/SCHEMA.md` §10）：① 每位置**实测参与前向的通道/隐藏单元数**
    （进入 `qkv` 的输入、GELU 之后进入 `mlp[2]` 的隐藏激活，按段分报，应与段宽逐位一致）；
    ② 每块增量 `‖out − in‖`（>0 ⇒ 干预真的在前向里生效，不是死开关）。"""
    spec = model._pw_spec
    T = model.cfg.seq_len
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    nz_qkv = []          # 逐位置：进入 qkv 的非零通道数
    nz_hid = []          # 逐位置：GELU 之后（进入 mlp[2]）非零的隐藏单元数
    deltas = [[] for _ in model.blocks]
    handles = []

    def qkv_pre(_mod, inp):
        cnt = (inp[0].detach() != 0).sum(-1).float().mean(0)      # [T]
        nz_qkv.append(cnt)

    def gelu_post(_mod, _inp, out):
        cnt = (out.detach() != 0).sum(-1).float().mean(0)         # [T]
        nz_hid.append(cnt)

    def blk_hook(l):
        def hook(_mod, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            deltas[l].append((y.detach().float()
                              - inp[0].detach().float()).abs().mean(0))  # [T]
        return hook

    for l, blk in enumerate(model.blocks):
        handles.append(blk.qkv.register_forward_pre_hook(qkv_pre))
        handles.append(blk.mlp[1].register_forward_hook(gelu_post))
        handles.append(blk.register_forward_hook(blk_hook(l)))
    was_training = model.training
    model.eval()
    try:
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            model(x.to(dev), exit_depths=None)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    def seg_mean(rows: list[torch.Tensor]) -> dict:
        m = torch.stack(rows).mean(0)                        # [T]
        return {"0..127": round(float(m[:SPLIT].mean()), 6),
                "128..255": round(float(m[SPLIT:].mean()), 6)}

    ch = seg_mean(nz_qkv)
    hid = seg_mean(nz_hid)
    exp_ch = {"0..127": spec.width_by_pos[0], "128..255": spec.width_by_pos[-1]}
    exp_hid = {"0..127": MLP_MULT * spec.width_by_pos[0],
               "128..255": MLP_MULT * spec.width_by_pos[-1]}
    return {
        "spec": spec.summary(),
        "measured_nonzero_channels_into_qkv_by_segment": ch,
        "expected_channels_by_segment": exp_ch,
        "measured_nonzero_hidden_units_by_segment": hid,
        "expected_hidden_units_by_segment": exp_hid,
        "nonzero_matches_spec": bool(
            all(abs(ch[s] - exp_ch[s]) < 1e-6 for s in exp_ch)
            and all(abs(hid[s] - exp_hid[s]) < 1e-6 for s in exp_hid)),
        "block_delta_absmean_first_layer_by_segment": {
            "0..127": round(float(torch.stack(deltas[0]).mean(0)[:SPLIT].mean()), 6),
            "128..255": round(float(torch.stack(deltas[0]).mean(0)[SPLIT:].mean()), 6)},
        "block_delta_absmean_last_layer_by_segment": {
            "0..127": round(float(torch.stack(deltas[-1]).mean(0)[:SPLIT].mean()), 6),
            "128..255": round(float(torch.stack(deltas[-1]).mean(0)[SPLIT:].mean()), 6)},
        "note": ("实测 = eval 下 2 批 × 8 序列的真实前向，用 forward hook 抓"
                 "`qkv` 的输入与 GELU 的输出、逐位置取非零计数后按段平均"
                 "（窄段位置尾部被置零 ⇒ 计数 = 段宽；宽段计数 = 全宽）。"
                 "block_delta = ‖out − in‖ 的逐位置均值（>0 ⇒ 该块真的在算）。"),
    }


@torch.no_grad()
def narrow_isolation_probe(model: DepthARModel, eval_arr, n_batches: int = 1,
                           batch_size: int = 4, seed: int = 99,
                           amp: float = 5.0) -> dict:
    """门 T5 的落盘版：窄段的块内增量**对自身尾部通道扰动逐位不变**（正对照：扰前缀必变）。

    「窄段真的是窄的前向」的实测证据（不是读代码得出的结论）。
    """
    spec = model._pw_spec
    if spec.all_full:
        return {"enabled": False, "reason": "FULL 臂（无窄段）"}
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
    x = x.to(dev)
    d = model.cfg.d_model
    probe = torch.zeros(1, model.cfg.seq_len, 1, device=dev)
    narrow = spec.narrow_mask(probe)                 # [1,T,1]

    def run(perturb_mask: torch.Tensor) -> list[torch.Tensor]:
        """在**窄段位置**的指定通道上加扰动，返回每块在窄段位置上的逐元素增量。"""
        accs: list[list[torch.Tensor]] = [[] for _ in model.blocks]

        def blk_hook(l):
            def hook(_mod, inp, out):
                y = out[0] if isinstance(out, tuple) else out
                delta = y.detach().float() - inp[0].detach().float()   # [B,T,d]
                sel = narrow.expand(delta.shape[0], -1, -1)            # [B,T,1]
                accs[l].append(delta.masked_select(sel.expand_as(delta))
                               .view(-1, d))
            return hook

        handles = [blk.register_forward_hook(blk_hook(l))
                   for l, blk in enumerate(model.blocks)]
        was = model.training
        model.eval()
        try:
            def pre_hook(_mod, inp):
                xx = inp[0]
                pm = (narrow.expand(xx.shape[0], -1, -1)
                      & (perturb_mask > 0).view(1, 1, d)).to(xx.dtype)
                return (xx + pm * amp,)
            h0 = model.blocks[0].register_forward_pre_hook(pre_hook)
            try:
                model(x, exit_depths=None)
            finally:
                h0.remove()
        finally:
            for h in handles:
                h.remove()
            model.train(was)
        return [torch.cat(a, dim=0) for a in accs]

    tail_mask = torch.zeros(d, device=dev)
    tail_mask[spec.nw:] = 1.0                      # 尾部通道（本应不参与）
    head_mask = torch.zeros(d, device=dev)
    head_mask[:spec.nw] = 1.0                      # 前缀通道（参与）
    base = run(torch.zeros(d, device=dev))
    tail = run(tail_mask)
    head = run(head_mask)
    tail_diff = max(float((a - b).abs().max()) for a, b in zip(base, tail))
    head_diff = max(float((a - b).abs().max()) for a, b in zip(base, head))
    return {"enabled": True, "amp": amp,
            "narrow_tail_perturb_max_abs_delta_diff": tail_diff,
            "narrow_prefix_perturb_max_abs_delta_diff": head_diff,
            "tail_invariant": tail_diff == 0.0,
            "positive_control_prefix_changes": head_diff > 0.0,
            "note": ("尾部扰动 = 在首块输入、窄段位置的前 nw 维之外加 5.0；"
                     "比较量 = 各块在窄段位置上的 ‖out − in‖ 逐元素最大差。"
                     "尾部差恒 0 + 前缀差 > 0 ⇒ 窄段真的只在 nw 维上运算，且门可红。")}


@torch.no_grad()
def causal_check(model: DepthARModel, tokens: torch.Tensor) -> dict:
    """因果性必须不变：改最后一个 token 不改变前面位置的 logits。"""
    l1 = _logits(model, tokens)
    t2 = tokens.clone()
    t2[:, -1] = (t2[:, -1] + 1) % VOCAB
    l2 = _logits(model, t2)
    max_abs = float((l1[:, :-1] - l2[:, :-1]).abs().max())
    self_abs = float((l1[:, -1] - l2[:, -1]).abs().max())
    return {"max_abs_logit_diff_earlier_positions": max_abs,
            "max_abs_logit_diff_last_position": self_abs,
            "causal": max_abs == 0.0}


def loss_slope(rows: list[dict], steps: int, frac: float = 0.25) -> dict:
    """同方法 loss 斜率（`harness/SCHEMA.md` §16）：末 frac 窗、最小二乘、×1000。"""
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
    return {"n_points": len(win), "step_first": xs[0], "step_last": xs[-1],
            "slope_per_1k_steps": round(slope * 1000, 5),
            "window": f"末 {int(frac * 100)}%"}


def evaluate(arm: str, seed: int, steps: int, out_root: str,
             n_batches: int = 20, suffix: str = "") -> dict:
    rd = run_dir(arm, seed, steps, out_root)
    _train_arr, eval_arr, vocab = load_enwik8_full(DATA_RAW, DATA_CACHE)
    model = build_model(arm, vocab, seed)
    state = torch.load(os.path.join(rd, "model.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
            if l.strip()]
    shared = json.load(open(os.path.join(rd, "results.json")))
    gen = torch.Generator().manual_seed(4321)
    tok, _ = batch(eval_arr, 4, model.cfg.seq_len, gen)
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "spec": model._pw_spec.summary(),
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": 2,
                 "bpc_by_depth": curve,
                 "clean_bpc": curve[-1]["bpc"],
                 "delta_vs_anchor": round(curve[-1]["bpc"] - ANCHOR_BPC, 6),
                 "anchor": {"name": "2641 U4-s0", "clean_bpc": ANCHOR_BPC},
                 "within_band": abs(curve[-1]["bpc"] - ANCHOR_BPC)
                 < VARIANCE_BOUND,
                 "acc_by_depth": [{"depth": r["depth"], "acc": r["acc"]}
                                  for r in curve]},
        "params": param_audit(model),
        "cost": analytic_cost(arm, vocab=vocab),
        "activation": activation_profile(model, eval_arr),
        "narrow_isolation": narrow_isolation_probe(model, eval_arr),
        "causal": causal_check(model, tok.to(DEVICE)),
        "loss_slope": loss_slope(rows, steps),
        "train_log_rows": len(rows),
        "shared_train_results": {
            "params": shared.get("params"),
            "train_tokens": shared.get("train_tokens"),
            "wall_minutes": shared.get("wall_minutes"),
            "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
            "config": shared.get("config"),
        },
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    out["cost"]["macs_pure_full_analytic"] = analytic_full_macs(vocab=vocab)
    out["cost"]["macs_full_matches_anchor_measured"] = (
        analytic_full_macs(vocab=vocab) == ANCHOR_MACS)
    path = os.path.join(HERE, f"results_{arm}_s{seed}{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2647] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"Δ={out['eval']['delta_vs_anchor']} "
          f"params={out['params']['params_total']} "
          f"avg_ch={out['cost']['avg_activated_channels_per_token']} "
          f"macs={out['cost']['macs_per_token']['total']}", flush=True)
    return out


# ---------------------------------------------------------------- 合成召回探针（P3）

def _paired_bootstrap_ci(hits_a: list[int], hits_b: list[int], n_iter: int = 2000,
                         seed: int = 1234) -> dict:
    """配对 bootstrap：Δ = acc(a) − acc(b) 的 95% 分位 CI（同批任务、逐查询配对）。"""
    if len(hits_a) != len(hits_b):
        raise ValueError("配对 bootstrap 要求两组命中位数相同")
    a = torch.tensor(hits_a, dtype=torch.float64)
    b = torch.tensor(hits_b, dtype=torch.float64)
    n = a.numel()
    g = torch.Generator().manual_seed(seed)
    diffs = torch.empty(n_iter, dtype=torch.float64)
    for i in range(n_iter):
        idx = torch.randint(0, n, (n,), generator=g)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo = float(torch.quantile(diffs, 0.025))
    hi = float(torch.quantile(diffs, 0.975))
    return {"delta": round(float(a.mean() - b.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)], "n_iter": n_iter,
            "n_pairs": n, "excludes_zero": bool(lo > 0 or hi < 0)}


def synth_probe(arm: str, tag: str, ckpt: str, seed: int = 0,
                out_root: str | None = None) -> dict:
    """零样本长程召回探针（`assoc-mqar`）：同一 checkpoint 上跑同一条任务电池。"""
    _tr, _ev, vocab = load_enwik8_full(DATA_RAW, DATA_CACHE)
    out_root = out_root or os.path.join(HERE, "runs")
    model = build_model(arm, vocab, seed)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    hits_all, dists_all, ce_all, per_batch = [], [], [], []
    for i in range(SYNTH["n_batches"]):
        tb = assoc_recall_multi_batch(SYNTH["batch_size"], SEQ_LEN, SYNTH["K"],
                                      SYNTH["Q"], sym_vocab=SYNTH["sym_vocab"],
                                      filler_token=SYNTH["filler_token"],
                                      cue_token=SYNTH["cue_token"],
                                      gap=SYNTH["gap"],
                                      generator=torch.Generator()
                                      .manual_seed(SYNTH["seed"] + 17 * i))
        logits = masked_logits(model, tb.tokens.to(DEVICE))
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, tb.rows.to(DEVICE))
        hits = (sel == tb.answers.to(DEVICE)).to(torch.long).cpu()
        # 辅助列（分析里声明为「后加的辅助诊断」，不是判据）：查询行答案上的 NLL
        lp = torch.log_softmax(logits.float(), dim=-1)
        rows = tb.rows.to(DEVICE).unsqueeze(-1)
        lp_rows = lp.gather(1, rows.expand(-1, -1, lp.shape[-1]))
        ce = (-lp_rows.gather(-1, tb.answers.to(DEVICE).unsqueeze(-1))
              .squeeze(-1).cpu())
        hits_all.extend(int(v) for v in hits.flatten().tolist())
        ce_all.extend(float(v) for v in ce.flatten().tolist())
        dists_all.extend(int(v) for v in tb.dists.flatten().tolist())
        per_batch.append({"batch": i, "n": int(hits.numel()),
                          "n_hits": int(hits.sum().item()),
                          "acc": round(float(hits.float().mean()), 6)})
    n = len(hits_all)
    k = sum(hits_all)
    by_dist = {}
    for d in sorted(set(dists_all)):
        idx = [i for i, v in enumerate(dists_all) if v == d]
        by_dist[str(d)] = {"n": len(idx),
                           "accuracy": round(sum(hits_all[i] for i in idx)
                                             / len(idx), 6)}
    out = {
        "probe": "assoc-mqar（引擎自带合成任务，零样本；同一 checkpoint 纯推理）",
        "task_config": {"synth_task": "assoc-mqar", "n_pairs": SYNTH["K"],
                        "n_queries": SYNTH["Q"], "gap": SYNTH["gap"],
                        "seq_len": SEQ_LEN, "sym_vocab": SYNTH["sym_vocab"],
                        "filler_token": SYNTH["filler_token"],
                        "cue_token": SYNTH["cue_token"],
                        "n_batches": SYNTH["n_batches"],
                        "batch_size": SYNTH["batch_size"],
                        "generator_seed_base": SYNTH["seed"],
                        "cli_equivalent": "--synth-task assoc-mqar --synth-pairs 8 "
                                          "--synth-queries 8 --synth-gap 14"},
        "arm": arm, "tag": tag, "seed": seed, "device": DEVICE,
        "ckpt": os.path.relpath(ckpt, ROOT),
        "ckpt_sha256": sha256_file(ckpt),
        "accuracy": round(k / n, 6), "n": n, "n_hits": k,
        "accuracy_wilson_ci95": wilson_ci(k, n),
        "answer_ce_mean": round(sum(ce_all) / len(ce_all), 6),
        "chance_ce_log_answer_set": round(math.log(SYNTH["sym_vocab"]), 6),
        "chance_ce_log_model_vocab": round(math.log(vocab), 6),
        "ce": [round(v, 6) for v in ce_all],
        "chance_uniform_answer_set": round(1.0 / SYNTH["sym_vocab"], 8),
        "chance_uniform_model_vocab": round(1.0 / vocab, 8),
        "by_distance": by_dist,
        "distance_summary": {"min": min(dists_all), "max": max(dists_all),
                             "median": sorted(dists_all)[len(dists_all) // 2]},
        "per_batch": per_batch,
        "hits": hits_all,
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    path = os.path.join(HERE, f"synth_{tag}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2647] synth {tag}: acc={out['accuracy']} "
          f"CI={out['accuracy_wilson_ci95']} n={n} chance={out['chance_uniform_answer_set']}",
          flush=True)
    return out


# ---------------------------------------------------------------- 汇总（P1–P4）

def _load(name: str) -> dict | None:
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def aggregate(steps: int, out_root: str, arms: list[str], seeds: list[int],
              synth_tag_map: dict[str, str] | None = None) -> dict:
    synth_tag_map = synth_tag_map or {"FULL": "anchor2641U4-s0", "PN": "PN",
                                      "NP": "NP"}
    recs = {}
    for arm in arms:
        for s in seeds:
            r = _load(f"results_{arm}_s{s}.json")
            if r is not None:
                recs[(arm, s)] = r
    synth = {}
    for arm, tag in synth_tag_map.items():
        r = _load(f"synth_{tag}_s0.json")
        if r is not None:
            synth[arm] = r

    def bpc(arm, s):
        return recs[(arm, s)]["eval"]["clean_bpc"] if (arm, s) in recs else None

    seeds_with = [s for s in seeds if all((a, s) in recs for a in arms)]
    verdicts = {}
    for s in seeds_with:
        row = {}
        for arm in arms:
            d = round(bpc(arm, s) - ANCHOR_BPC, 6)
            row[arm] = {"delta_vs_anchor": d,
                        "verdict": ("不可排序（H0）：|Δ| < 0.014 ⇒ 位置分段宽度在本尺度"
                                    "质量中性" if abs(d) < VARIANCE_BOUND else
                                    ("可分：退化越带" if d > 0 else "可分：优于锚"))}
        if "PN" in arms and "NP" in arms:
            dd = round(bpc("PN", s) - bpc("NP", s), 6)
            row["PN_minus_NP"] = {"delta": dd,
                                  "verdict": ("不可排序（同代价、相反位置分配）"
                                              if abs(dd) < VARIANCE_BOUND else
                                              ("PN 更好" if dd < 0 else "NP 更好"))}
        verdicts[str(s)] = row

    # P3：配对 bootstrap（PN / NP 相对锚）
    p3 = {}
    if "FULL" in synth:
        anchor_hits = synth["FULL"]["hits"]
        for arm in ("PN", "NP"):
            if arm not in synth:
                continue
            a, b = synth[arm], synth["FULL"]
            p3[arm] = {
                "accuracy": a["accuracy"],
                "answer_ce_mean": a["answer_ce_mean"],
                "anchor_answer_ce_mean": b["answer_ce_mean"],
                "chance_ce_log_answer_set": a["chance_ce_log_answer_set"],
                "paired_delta_ce_vs_anchor": _paired_bootstrap_ci(a["ce"],
                                                                  b["ce"]),
                "accuracy_wilson_ci95": a["accuracy_wilson_ci95"],
                "anchor_accuracy": b["accuracy"],
                "anchor_accuracy_wilson_ci95": b["accuracy_wilson_ci95"],
                "chance_uniform_answer_set": a["chance_uniform_answer_set"],
                "paired_delta_vs_anchor": _paired_bootstrap_ci(a["hits"],
                                                               anchor_hits),
                "by_distance": a["by_distance"],
            }
    reeval = {}
    for arm in arms:
        for s in seeds:
            r = _load(f"results_{arm}_s{s}rerun.json")
            if r is not None and (arm, s) in recs:
                reeval[f"{arm}-s{s}"] = {
                    "clean_bpc_first": recs[(arm, s)]["eval"]["clean_bpc"],
                    "clean_bpc_rerun": r["eval"]["clean_bpc"],
                    "bitwise_identical": (recs[(arm, s)]["eval"]["clean_bpc"]
                                          == r["eval"]["clean_bpc"])}

    out = {
        "experiment": "2647-position-width-13m",
        "criterion": ("P1 clean bpc vs 锚 2641 U4-s0=1.5314（0.014 bpc 带）；"
                      "P2 平均激活通道数 / 等效 MACs（该跌）；"
                      "P3 长程召回 assoc-mqar（率类，Wilson + 配对 bootstrap，"
                      "不跨量纲借用 0.014）；P4 否证线。墙钟禁入判据。"),
        "variance_bound": VARIANCE_BOUND,
        "anchor": {"name": "2641 U4-s0", "clean_bpc": ANCHOR_BPC,
                   "params": ANCHOR_PARAMS, "macs_per_token": ANCHOR_MACS},
        "steps": steps,
        "arms": {f"{a}-s{s}": {"clean_bpc": bpc(a, s),
                               "delta_vs_anchor": round(bpc(a, s) - ANCHOR_BPC, 6)
                               if bpc(a, s) is not None else None,
                               "avg_activated_channels": recs[(a, s)]["cost"]
                               ["avg_activated_channels_per_token"],
                               "macs_per_token": recs[(a, s)]["cost"]
                               ["macs_per_token"]["total"],
                               "macs_rel_to_full": recs[(a, s)]["cost"]
                               ["macs_rel_to_full"],
                               "params_total": recs[(a, s)]["params"]
                               ["params_total"],
                               "params_matches_anchor": recs[(a, s)]["params"]
                               ["params_matches_anchor"],
                               "slope_per_1k": recs[(a, s)]["loss_slope"]
                               ["slope_per_1k_steps"],
                               "causal": recs[(a, s)]["causal"]["causal"],
                               "nonzero_matches_spec": recs[(a, s)]["activation"]
                               ["nonzero_matches_spec"],
                               "tail_invariant": recs[(a, s)]["narrow_isolation"]
                               .get("tail_invariant"),
                               "prefix_perturb_changes": recs[(a, s)]
                               ["narrow_isolation"]
                               .get("positive_control_prefix_changes")}
                          for (a, s) in sorted(recs)},
        "P1_verdict_by_seed": verdicts,
        "P2_cost": {f"{a}-s{s}": {"avg_activated_channels": recs[(a, s)]["cost"]
                                  ["avg_activated_channels_per_token"],
                                  "macs_per_token": recs[(a, s)]["cost"]
                                  ["macs_per_token"]["total"],
                                  "macs_rel_to_full": recs[(a, s)]["cost"]
                                  ["macs_rel_to_full"],
                                  "measured_nonzero_matches_spec":
                                  recs[(a, s)]["activation"]
                                  ["nonzero_matches_spec"]}
                    for (a, s) in sorted(recs)},
        "P3_long_range_recall": p3,
        "P3_anchor_probe": ({k: synth["FULL"][k] for k in
                            ("arm", "tag", "device", "accuracy",
                             "accuracy_wilson_ci95", "answer_ce_mean",
                             "chance_ce_log_answer_set",
                             "chance_ce_log_model_vocab", "n", "n_hits",
                             "ckpt", "ckpt_sha256")}
                           if "FULL" in synth else None),
        "P3_anchor_note": ("锚 = 2641 U4-s0 的 model.pt（只读复用、纯推理）；"
                           "三臂同探针同任务批 ⇒ 配对可比"),
        "determinism_reeval": reeval,
        "equal_gates": {
            "params_identical_all_arms": (len({recs[k]["params"]["params_total"]
                                              for k in recs}) <= 1
                                         and recs != {}),
            "params": {f"{a}-s{s}": recs[(a, s)]["params"]["params_total"]
                       for (a, s) in sorted(recs)},
            "full_analytic_macs_matches_anchor_measured": (
                recs[list(recs)[0]]["cost"]["macs_full_matches_anchor_measured"]
                if recs else None),
            "narrow_isolation": {f"{a}-s{s}": {
                "tail_invariant": recs[(a, s)]["narrow_isolation"]
                .get("tail_invariant"),
                "prefix_perturb_changes": recs[(a, s)]["narrow_isolation"]
                .get("positive_control_prefix_changes")} for (a, s) in sorted(recs)},
        },
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ALL_ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用 runs/…/model.pt 评测")
    ap.add_argument("--out-suffix", default="",
                    help="评测落盘后缀（重评确定性用 'rerun'）")
    ap.add_argument("--synth-probe", action="store_true",
                    help="合成召回探针（纯推理；--arm 指定前向语义，--ckpt 指定权重）")
    ap.add_argument("--ckpt", default=None, help="探针用权重路径（缺省 = 本臂 model.pt）")
    ap.add_argument("--tag", default=None, help="探针落盘标签（缺省 = 臂名）")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default=",".join(TRAIN_ARMS))
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.steps, args.out_root, args.arms.split(","),
                  [int(s) for s in args.seeds.split(",") if s.strip() != ""])
        return
    if not args.arm:
        raise SystemExit("--arm 或 --aggregate 必给其一")
    if args.synth_probe:
        ckpt = args.ckpt or os.path.join(
            run_dir(args.arm, args.seed, args.steps, args.out_root), "model.pt")
        synth_probe(args.arm, args.tag or args.arm, ckpt, args.seed, args.out_root)
        return
    if not args.no_train:
        os.makedirs(args.out_root, exist_ok=True)
        info = train_arm(args.arm, args.seed, args.steps, args.batch_size,
                         args.lr, args.out_root)
        rd = run_dir(args.arm, args.seed, args.steps, args.out_root)
        with open(os.path.join(rd, "train_wall.json"), "w") as f:
            json.dump(info, f)
    evaluate(args.arm, args.seed, args.steps, args.out_root,
             suffix=args.out_suffix)


if __name__ == "__main__":
    main()
