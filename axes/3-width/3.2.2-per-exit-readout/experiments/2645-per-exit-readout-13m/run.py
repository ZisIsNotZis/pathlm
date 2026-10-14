"""2645 · 逐出口读出（per-exit readout）· 13M · depth-AR —— 复现入口。

契约 = 同目录 `README.md`（臂 / 判据线 P1–P4 / 否证线 **预登记、不可改**）。
本文件是唯一入口。

基底 = depth-AR 形态（`dense=false`、变深度 + 逐层读出 + 随机退出 + `fill_kv`、
`aux_weight=1.0`、`exit_mode=uniform`，d=256/L=8/heads=8/mlp_mult=4/seq 256/
batch 32/lr 1e-3/12000 步/seed 0/enwik8 字符级 vocab=206）。

设计要点（为什么这样接线）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名
  `DepthARModel` 换成一个工厂（构造 stock 模型后挂一个「每出口读出」子模块），
  然后调用 `train_depth_ar.main()` —— 参数解析、配置装配、`train()` 循环、
  checkpoint、`results.json` 全部是**共享实现**，本文件不改 `pathlm/**`。
* **每出口读出子模块建在 stock 构造之后** ⇒ 不消耗主干参数的随机数；
  同 seed 下四臂的 embedding / attention / LN / carry 门权重**逐位相同**
  （单测 `T3` 钉住）⇒ 臂间差别只有「出口 → 词表」的映射，不是「两个不同初始化
  的模型在比」。
* 出口数 = L+1 = 9（深度 0..8）。引擎的 `tf_acceptance` 给 8 行（深度 0..7，
  与全深度出口 8 比 greedy argmax），深度 8 是验证者本身（自参照、构造性 1.0）。
* 锚复用：`.scratch/23-depth-ar/evidence/23c-H2-w0-12k`（clean bpc 1.8582、
  tf@4 0.9694、tf@6 0.9938），按 `SCHEMA.md` §9 三道门——①「T 臂 = 开关关闭」
  与 stock `DepthARModel.readout` 逐位同前向单测 ②逐字段 config 对照
  ③口径标注：用**本文件自己的**评测电池重测该 checkpoint，并与 2610 用当前代码
  重测的 `results.json::A_arm` 逐位对照 ⇒ 口径可比（`--stage anchor`）。

用法（每臂一次训练，`--no-train` 仅评测复用 `model.pt`）：

    python3 run.py --stage anchor                                   # 锚重测（不需训练）
    python3 run.py --arm T --seed 0 --steps 12000                    # 训练 + 评测
    python3 run.py --arm PE --seed 0 --steps 12000 --no-train        # 仅评测
    python3 run.py --aggregate                                       # 汇总 P1–P4 → results.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                    # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel, depth_curve,  # noqa: E402
                             tf_acceptance)
from pathlm.data import load_enwik8_full, batch                 # noqa: E402
from pathlm.metrics import ece                                  # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 臂与常量（契约 §3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
VOCAB = 206
EXITS = N_LAYERS + 1                     # 出口数 = 9（深度 0..L）
LR_RANK = 16                             # LR 臂的低秩 r
ARMS = ("T", "PE", "IND", "LR")
VARIANCE_BOUND = 0.014                   # clean bpc 带（限配对 Δ、≥5 窗口）
ECE_TOLERANCE = 0.003                    # prob0 ECE 容差（承自 2610 判据线，§README P3）

# 锚（复用：`.scratch/23-depth-ar/evidence/23c-H2-w0-12k`，SCHEMA §9 三道门）
ANCHOR_DIR = os.path.join(ROOT, ".scratch", "23-depth-ar", "evidence",
                          "23c-H2-w0-12k")
ANCHOR_BPC = 1.8582
ANCHOR_TF4 = 0.9694
ANCHOR_TF6 = 0.9938
# 2610 用当前评测代码重测同一 checkpoint 的读数（口径对照的第二列）
GATED_2610_RESULTS = os.path.join(
    ROOT, "axes", "3-width", "3.2.1-readout-reparam", "experiments",
    "2610-gated-readout-13m-depthar", "results.json")
DATA = "data/enwik8_full.npz"


def base_cfg(**kw) -> DepthARConfig:
    """契约基底：d=256/L=8/h=8/seq256/mlp4、变深度、随机退出、fill_kv。"""
    return DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                         seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=False,
                         fill_kv=True, **kw)


# ---------------------------------------------------------------- 干预：每出口读出

class PerExitReadout(nn.Module):
    """「出口 → 词表」的映射（唯一变量），四臂各一种形态。

    * `T`：无参数（读出 = stock 共享 tied 线性表，`logits_k = h_k·Eᵀ`）；
    * `PE`：每出口一个 d×d 变换，`logits_k = (M_k h_k)·Eᵀ`；`M_k = I` 起步
      （严格恒等于 `T`，单测钉住）。该点非鞍点：∂L/∂M_k = Ū_kᵀ g_k h_kᵀ ≠ 0。
    * `IND`：每出口一张独立线性表 `[V,d]`，`logits_k = tab_k(h_k)`；
      `tab_k ~ N(0, 0.02)`（与主干后的 embedding 同分布约定）。**无 tied 桥**。
    * `LR`：每出口对 tied 表做低秩校正 `F_k = E + A_k B_k`（`A_k` [V,r]、
      `B_k` [r,d]，r = 16）；`B_k ~ N(0, 0.02)`、**`A_k = 0`**（单边零，与引擎
      `readout_perdepth` 的 P/Q 同款）⇒ 起步 `F_k = E`，逐位恒等于 `T`（单测
      钉住），而 ∂L/∂A_k = (∂L/∂F_k)·B_kᵀ ≠ 0、∂L/∂B_k = A_kᵀ(∂L/∂F_k) ≠ 0 ⇒
      非鞍点（单测钉住）。
    """

    def __init__(self, arm: str, d: int, vocab: int, rank: int = LR_RANK):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.d, self.vocab, self.rank = d, vocab, rank
        if arm == "PE":
            self.m = nn.ModuleList(nn.Linear(d, d, bias=False)
                                   for _ in range(EXITS))
            for mod in self.m:                     # 恒等起步
                with torch.no_grad():
                    mod.weight.copy_(torch.eye(d))
        elif arm == "IND":
            self.tab = nn.ModuleList(nn.Linear(d, vocab, bias=False)
                                     for _ in range(EXITS))
            for mod in self.tab:
                nn.init.normal_(mod.weight, std=0.02)
        elif arm == "LR":
            self.a = nn.ModuleList(nn.Linear(rank, vocab, bias=False)
                                   for _ in range(EXITS))     # 外：A_k [V,r]
            self.b = nn.ModuleList(nn.Linear(d, rank, bias=False)
                                   for _ in range(EXITS))     # 内：B_k [r,d]
            for mod in self.b:
                nn.init.normal_(mod.weight, std=0.02)
            for mod in self.a:                     # 单边零 ⇒ 起步 F_k = E
                nn.init.zeros_(mod.weight)

    def extra_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def head(self, depth: int, e: torch.Tensor) -> torch.Tensor:
        """该出口的**有效读出矩阵** `F_k`（[V,d]，`logits_k = h_k·F_kᵀ`）。

        四臂统一到同一形状 ⇒ 「相对 tied 表偏离多少」可同列比较。
        """
        if self.arm == "PE":
            return e @ self.m[depth].weight                 # E·M_k
        if self.arm == "IND":
            return self.tab[depth].weight                   # 独立表
        if self.arm == "LR":
            return e + self.a[depth].weight @ self.b[depth].weight
        return e

    def logits(self, e: torch.Tensor, hf: torch.Tensor,
               depth: int) -> torch.Tensor:
        """`e` = tied 表 `E`（[V,d]、fp32），`hf` = 该出口隐藏态（fp32）。"""
        if self.arm == "PE":
            return (self.m[depth](hf)) @ e.T
        if self.arm == "IND":
            return self.tab[depth](hf)
        if self.arm == "LR":
            return hf @ (e + self.a[depth].weight @ self.b[depth].weight).T
        raise ValueError("T 臂无自有读出参数")


class PerExitModel(DepthARModel):
    """stock `DepthARModel` + 每出口读出子模块（主干逐位不变）。"""

    def __init__(self, cfg: DepthARConfig, vocab_size: int, arm: str = "T"):
        super().__init__(cfg, vocab_size)          # 主干 + tied 表（同一 RNG 流）
        self.per_exit = PerExitReadout(arm, cfg.d_model, vocab_size)
        self.arm = arm

    def readout(self, h: torch.Tensor, depth: int | None = None) -> torch.Tensor:
        """同 stock 语义：fp32 头。`depth=None` 落全深度出口（= cfg.n_layers）。

        引擎在归一化路径里显式传 depth（forward 逐深度读出、workspace/探针），
        但 `readout_dims` 相关的旧路径可能不传；此处把 None 定义为全深度出口并
        在 results 里标注，避免静默用错出口。
        """
        if self.arm == "T":
            return super().readout(h, depth=depth)
        with torch.autocast("cuda", enabled=False):
            hf = h.float()
            k = self.cfg.n_layers if depth is None else int(depth)
            if not 0 <= k <= self.cfg.n_layers:
                raise ValueError(f"出口深度 {k} 不在 [0, {self.cfg.n_layers}]")
            return self.per_exit.logits(self.embed.weight.float(), hf, k)


def build_model(arm: str, vocab_size: int = VOCAB, seed: int = 0,
                cfg: DepthARConfig | None = None) -> PerExitModel:
    """所有臂共用同一构造路径（同 seed ⇒ 同一份主干抽样）。"""
    torch.manual_seed(seed)
    return PerExitModel(cfg if cfg is not None else base_cfg(), vocab_size, arm)


def _factory_for(arm: str):
    def factory(cfg, vocab_size):
        return PerExitModel(cfg, vocab_size, arm)
    return factory


# ---------------------------------------------------------------- 参数与代价审计

def param_audit(model: PerExitModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    extra = model.per_exit.extra_params()
    per_exit = {
        "T": 0,
        "PE": D_MODEL * D_MODEL,
        "IND": VOCAB * D_MODEL,
        "LR": D_MODEL * LR_RANK + LR_RANK * VOCAB,
    }[model.arm]
    return {"arm": model.arm, "params_total": total,
            "params_trunk_and_tied": total - extra,
            "params_per_exit_readout": extra,
            "params_per_exit_readout_per_exit": per_exit,
            "n_exits": EXITS,
            "params_per_exit_readout_analytic": per_exit * EXITS,
            "readout_share_of_total": round(extra / total, 6) if total else 0.0}


def analytic_cost(arm: str, d: int = D_MODEL, L: int = N_LAYERS,
                  T: int = SEQ_LEN, vocab: int = VOCAB, rank: int = LR_RANK,
                  n_exits: int = EXITS) -> dict:
    """每 token 等效理论量（MACs，**不含任何时间量**）。

    全部臂的主干相同（L 层 attention + FFN + 最终 LN）；差别只在出口读出：
    * `T` 每出口 `d·V`（tied 表）；
    * `PE` 每出口 `d²`（变换）+ `d·V`（tied 表）；
    * `IND` 每出口 `d·V`（独立表，与 tied 同 MACs，但**参数**是 vocab×d）；
    * `LR` 每出口 `d·r + r·V`（校正项）+ `d·V`（tied 表基）。
    训练期每个 token 在全部 (L+1) 个出口算 logits（逐层读出监督）；推理期每次
    前向只用 1 个出口 —— 两列分开报。
    """
    attn = L * (4 * d * d + 2 * T * d)
    ffn = L * 2 * MLP_MULT * d * d
    extra = {"T": 0, "PE": d * d, "IND": 0, "LR": d * rank + rank * vocab}[arm]
    readout_one = d * vocab + extra
    return {
        "d_model": d, "n_layers": L, "seq_len": T, "vocab": vocab,
        "n_exits": n_exits, "per_exit_readout_rank": rank,
        "macs_per_token": {
            "trunk": attn + ffn,
            "readout_one_exit": readout_one,
            "readout_all_exits_train": n_exits * readout_one,
            "total_train_all_exits": attn + ffn + n_exits * readout_one,
            "total_infer_one_exit": attn + ffn + readout_one,
        },
        "readout_params": {
            "T": 0, "PE": n_exits * d * d, "IND": n_exits * vocab * d,
            "LR": n_exits * (d * rank + rank * vocab)}[arm],
        "note": ("等效理论量：MACs/token（乘加），不含时间量；tokens-per-forward = 1"
                 "（出口读出不重跑主干，layer-passes 每 token 不增加）。"),
    }


# ---------------------------------------------------------------- 率类指标与 CI

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 6), round(centre + half, 6)]


def bootstrap_ci(values: list, n_iter: int = 2000, seed: int = 1234) -> list:
    """cluster bootstrap（按批重抽）——率类差异的 95% CI。"""
    if not values:
        return [None, None]
    g = torch.Generator().manual_seed(seed)
    n = len(values)
    ms = sorted(sum(values[int(i)] for i in torch.randint(0, n, (n,), generator=g))
                / n for _ in range(n_iter))
    return [round(ms[int(0.025 * n_iter)], 6),
            round(ms[min(int(0.975 * n_iter), n_iter - 1)], 6)]


@torch.no_grad()
def tf_acceptance_paired(model: PerExitModel, eval_arr, n_batches: int = 20,
                         batch_size: int = 16, gen_seed: int = 3,
                         device: str = DEVICE) -> dict:
    """逐深度接受率（= 引擎 `tf_acceptance` 口径：同 batch_size、同 generator
    seed、同 n_batches），额外返回**逐批计数** ⇒ 可算 Wilson 与配对 bootstrap CI。
    `depth = 0..L-1` 与全深度出口 `L` 的 greedy argmax 比（深度 L 自参照）。
    """
    gen = torch.Generator().manual_seed(gen_seed)
    L = model.cfg.n_layers
    T = model.cfg.seq_len
    agree = [0] * L
    per_batch = []
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        full = aux["depth_logits"][-1][:, :-1].argmax(-1)
        nb = int(full.numel())
        counts = []
        for k in range(L):
            c = int((aux["depth_logits"][k][:, :-1].argmax(-1)
                     == full).sum().item())
            counts.append(c)
            agree[k] += c
        per_batch.append(counts)
        n += nb
    by_depth = [{"depth": k, "accept": round(agree[k] / n, 6), "n": n,
                 "wilson_95ci": wilson_ci(agree[k], n),
                 "cluster_bootstrap_95ci": bootstrap_ci([c / nb for c in
                                                         (pb[k] for pb in per_batch)])}
                for k in range(L)]
    return {"by_depth": by_depth, "per_batch_agree": per_batch,
            "n_per_batch": n // n_batches, "n_total": n,
            "verifier_depth": L, "note": "depth L（全深度）是验证者本身，自参照 1.0"}


def paired_accept_delta(arm: dict, ref: dict, k: int) -> dict:
    """同批配对差 Δ tf@k（arm − ref）的 cluster bootstrap CI。"""
    nb = arm["n_per_batch"]
    diffs = [(a[k] - b[k]) / nb for a, b in zip(arm["per_batch_agree"],
                                                ref["per_batch_agree"])]
    ci = bootstrap_ci(diffs)
    point = arm["by_depth"][k]["accept"] - ref["by_depth"][k]["accept"]
    return {"depth": k, "delta": round(point, 6),
            "paired_bootstrap_95ci": ci,
            "separable_from_zero": bool(ci[0] is not None
                                        and (ci[0] > 0 or ci[1] < 0)),
            "n_batches": len(diffs)}


# ------------------------- prob0 ECE（承 2610 实现，口径一致） -------------------------

@torch.no_grad()
def prob0_ece(model: PerExitModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    gen = torch.Generator().manual_seed(seed)
    Ln = model.cfg.n_layers
    conf = [[] for _ in range(Ln + 1)]
    hit = [[] for _ in range(Ln + 1)]
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        tgt = x[:, 1:]
        for k, lg in enumerate(aux["depth_logits"]):
            lg = lg[:, :-1].float()
            p = lg.softmax(-1)
            conf[k].append(p.max(-1).values.reshape(-1).cpu())
            hit[k].append((lg.argmax(-1) == tgt).reshape(-1))
    per_depth = [round(ece(torch.cat(conf[k]), torch.cat(hit[k]).float(),
                           n_bins=n_bins), 4) for k in range(Ln + 1)]
    d4 = 4 if 4 <= Ln else Ln
    return {"prob0_ece": per_depth[-1], "exit4_ece": per_depth[d4],
            "ece_by_depth": per_depth, "n_bins": n_bins,
            "n_batches": n_batches, "batch_size": batch_size}


@torch.no_grad()
def causal_check(model: PerExitModel, seed: int = 7) -> dict:
    g = torch.Generator().manual_seed(seed)
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
def soft_submission_probe(model: PerExitModel, eval_arr, n_batches: int = 4,
                          batch_size: int = 8, gen_seed: int = 11) -> dict:
    """P3 的「软提交」语义探针（承 2610 `readout_workspace_energy` 的同名概念）：
    软提交 = 用出口概率在**共享 tied 表**上取期望 `Σ p·E`。若出口的 logits 不再由
    tied 表生成（IND），该期望与 logits 的 argmax 不再对应 ⇒ 桥断裂。
    这里用**可直接测的量**：概率与「tied 表基上的坐标」的一致性
    `argmax_k == argmax(h_k·Eᵀ)` 的位置占比（prob0 一致性），IND 应当明显 < 1。
    """
    gen = torch.Generator().manual_seed(gen_seed)
    L = model.cfg.n_layers
    T = model.cfg.seq_len
    agree = [0] * (L + 1)
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        E = model.embed.weight.float()
        for k, st in enumerate(aux["states"]):
            tied = st.float() @ E.T
            agree[k] += int((aux["depth_logits"][k].argmax(-1)
                             == tied.argmax(-1)).sum().item())
        n += aux["depth_logits"][-1][..., 0].numel()
    return {"tied_head_argmax_agreement": round(agree[-1] / n, 6),
            "by_depth": [round(a / n, 6) for a in agree], "n": n,
            "wilson_95ci": wilson_ci(agree[-1], n),
            "note": ("出口 logits 的 argmax 与「状态在 tied 表基上的坐标」argmax 的"
                     "一致率（全深度出口）。T/PE/LR 的出口头都含 tied 表因子 ⇒ "
                     "构造性 1.0；IND 不保证。")}


@torch.no_grad()
def evaluate(arm: str, seed: int, steps: int, out_root: str,
             n_batches: int = 20) -> dict:
    rd = run_dir(arm, seed, steps, out_root)
    ckpt = os.path.join(rd, "model.pt")
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    model = build_model(arm, vocab, seed).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    curve2 = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                         device=DEVICE)
    tf = tf_acceptance_paired(model, eval_arr, n_batches=n_batches)
    tf_ref = tf_acceptance(model, eval_arr, n_batches=n_batches,
                           device=DEVICE)
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "eval": {"metric": "clean bpc（最终深度、无退出、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": 2, "bpc_by_depth": curve,
                 "clean_bpc": curve[-1]["bpc"]},
        "tf_acceptance": tf["by_depth"],
        # 引擎 tf_acceptance 只保留 4 位小数 ⇒ 按 4 位比较（否则是舍入假阴性，
        # 见 smoke 抓到的 defect：2610 同款函数）。
        "tf_acceptance_engine_agrees": bool(
            [round(r["accept"], 4) for r in tf["by_depth"]]
            == [r["accept"] for r in tf_ref]),
        "tf_per_batch_agree": tf["per_batch_agree"],
        "tf_n_per_batch": tf["n_per_batch"],
        "params": param_audit(model),
        "cost": analytic_cost(arm, vocab=vocab),
        "calibration": prob0_ece(model, eval_arr),
        "soft_submission": soft_submission_probe(model, eval_arr),
        "causal": causal_check(model),
        "reeval_bitwise_identical": bool(curve == curve2),
        "head_fro_by_depth": head_fro(model),
        "checkpoint": {"path": os.path.relpath(ckpt, ROOT),
                       "sha256": sha256_file(ckpt),
                       "missing_keys": list(missing),
                       "unexpected_keys": list(unexpected)},
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    with open(os.path.join(HERE, results_name(arm, seed, steps)), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2645] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"tf4={out['tf_acceptance'][4]['accept']} "
          f"tf6={out['tf_acceptance'][6]['accept']} "
          f"params_readout={out['params']['params_per_exit_readout']}",
          flush=True)
    return out


@torch.no_grad()
def head_fro(model: PerExitModel) -> list:
    """干预活性（SCHEMA §10 强制项）：每出口自有读出头相对 tied 表的偏离幅度。

    四臂同列报 `rel_to_tied = ‖F_k − E‖_F / ‖E‖_F`（`T` 构造性 0；`PE`/`LR`
    起步 0；`IND` 起步即非零——它没有 tied 桥）。
    """
    E = model.embed.weight.detach().float()
    den = float(E.norm())
    out = []
    for k in range(EXITS):
        F = model.per_exit.head(k, E)
        out.append({"depth": k, "fro": round(float(F.norm()), 4),
                    "rel_to_tied": round(float((F - E).norm()) / den, 6)})
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


# ---------------------------------------------------------------- 训练（走共享 main）

def shared_argv(run_name: str, seed: int, steps: int, batch_size: int,
                lr: float, out_root: str) -> list:
    return [os.path.join(ROOT, "train_depth_ar.py"), run_name,
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--exit-mode", "uniform",
            "--aux-weight", "1.0", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr),
            "--seed", str(seed), "--eval-batches", "20", "--out-root", out_root]


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory_for(arm)
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(run_name(arm, seed, steps), seed, steps,
                               batch_size, lr, out_root)
        t0 = time.time()
        TDA.main()
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


def run_name(arm: str, seed: int, steps: int) -> str:
    return f"{arm}-s{seed}-{steps // 1000}k"


def results_name(arm: str, seed: int, steps: int) -> str:
    """逐臂读数文件名带步数（同一臂的 60 步冒烟与 12k 不互相覆盖，也不会被误聚合）。"""
    return f"results_{arm}_s{seed}_{steps // 1000}k.json"


# ---------------------------------------------------------------- 锚三道门（SCHEMA §9）

ANCHOR_FIELDS = [
    ("d_model", 256), ("n_layers", 8), ("n_heads", 8), ("seq_len", 256),
    ("mlp_mult", 4), ("dense", False), ("exit_mode", "uniform"),
    ("aux_weight", 1.0), ("aux_anneal_to", None), ("shallow_stopgrad", False),
    ("dense_aux", False), ("steps", 12000), ("batch_size", 32), ("lr", 0.001),
    ("seed", 0), ("vocab_size", 206), ("final_weight", 1.0),
]


def anchor_equivalence_table(hist: dict) -> list:
    """逐字段 config 对照：历史锚的 resolved config 与本契约声明的基底逐字段比。"""
    cfg = hist["config"]
    tr = hist["train"]
    src = {"d_model": cfg, "n_layers": cfg, "n_heads": cfg, "seq_len": cfg,
           "mlp_mult": cfg, "dense": cfg, "exit_mode": hist, "aux_weight": hist,
           "aux_anneal_to": hist, "shallow_stopgrad": hist, "dense_aux": hist,
           "steps": tr, "batch_size": tr, "lr": tr, "seed": tr,
           "vocab_size": hist}
    rows = []
    for field, want in ANCHOR_FIELDS:
        if field == "final_weight":
            # 23c 无该字段（等权 v1）⇒ 由代码谱系标注为 1.0
            got = 1.0
            note = "23c results.json 无 final_weight 字段；等权 v1 = 1.0（代码谱系）"
        else:
            got = src[field].get(field)
            note = "results.json"
        rows.append({"field": field, "anchor": got, "this": want,
                     "equal": got == want, "source": note})
    return rows


def stage_anchor(args) -> dict:
    """口径门：用**本文件**的电池重测历史锚 checkpoint，并与 2610 的 `A_arm`
    读数（当前代码重测）逐位对照 ⇒ 本实验与锚的口径可比。"""
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    hist = json.load(open(os.path.join(ANCHOR_DIR, "results.json")))
    ref = json.load(open(GATED_2610_RESULTS))["A_arm"]
    model = build_model("T", vocab, seed=args.seed)
    sd = torch.load(os.path.join(ANCHOR_DIR, "model.pt"), map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    curve = depth_curve(model, eval_arr, n_batches=args.eval_batches,
                        batch_size=16, device=DEVICE)
    tf = tf_acceptance_paired(model, eval_arr, n_batches=args.eval_batches)
    calib = prob0_ece(model, eval_arr)
    eq = anchor_equivalence_table(hist)
    out = {"stage": "anchor", "source": os.path.relpath(ANCHOR_DIR, ROOT),
           "config_equivalence": eq,
           "config_equivalence_all_equal": all(r["equal"] for r in eq),
           "checkpoint_missing_keys": list(missing),
           "checkpoint_unexpected_keys": list(unexpected),
           "clean_bpc_remeasured": curve[-1]["bpc"],
           "clean_bpc_hist": hist["depth_curve"][-1]["bpc"],
           "clean_bpc_2610_remeasure": ref["depth_curve"][-1]["bpc"],
           "tf_acceptance": tf["by_depth"],
           "tf_acceptance_hist": hist["tf_acceptance"],
           "tf_acceptance_2610_remeasure": ref["tf_acceptance"],
           "tf_matches_2610_bitwise": bool(
               [r["accept"] for r in tf["by_depth"]]
               == [r["accept"] for r in ref["tf_acceptance"]]),
           "calibration": calib,
           "calibration_2610_remeasure": ref["calibration"],
           "params": param_audit(model),
           "bpc_by_depth": curve,
           "note": ("锚 = 23c-H2-w0-12k（复用，SCHEMA §9 三道门：①T 臂 = stock "
                    "readout 逐位同前向单测 ②逐字段 config 对照 ③本表口径标注）")}
    with open(os.path.join(HERE, "anchor_recheck.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2645 anchor] bpc={out['clean_bpc_remeasured']} "
          f"(hist {out['clean_bpc_hist']}, 2610 {out['clean_bpc_2610_remeasure']}) "
          f"tf@4={out['tf_acceptance'][4]['accept']} "
          f"tf_matches_2610={out['tf_matches_2610_bitwise']} "
          f"config_all_equal={out['config_equivalence_all_equal']}", flush=True)
    return out


# ---------------------------------------------------------------- 汇总（P1–P4）

def _ci_text(ci: list) -> str:
    return f"[{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "n/a"


def aggregate(args) -> dict:
    seeds = [args.seed]
    recs = {}
    for arm in ARMS:
        p = os.path.join(HERE, results_name(arm, args.seed, args.steps))
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("steps") != args.steps:
                continue        # 步数不符（如 60 步冒烟）⇒ 视为未跑，不误聚合
            recs[arm] = r
    anchor_recheck = {}
    ap = os.path.join(HERE, "anchor_recheck.json")
    if os.path.exists(ap):
        anchor_recheck = json.load(open(ap))

    # 参考臂：优先同批 T；无 T 则用锚（口径门必须已过）
    ref_name = "T" if "T" in recs else "anchor"
    ref_tf = (recs["T"]["tf_acceptance"] if "T" in recs
              else anchor_recheck.get("tf_acceptance"))
    ref_per_batch = (recs["T"]["tf_per_batch_agree"] if "T" in recs
                     else anchor_recheck.get("tf_per_batch_agree"))
    ref_bpc = (recs["T"]["eval"]["clean_bpc"] if "T" in recs
               else anchor_recheck.get("clean_bpc_remeasured"))

    rows = {}
    for arm in ARMS:
        if arm not in recs:
            rows[arm] = {"clean_bpc": None, "status": "未跑"}
            continue
        r = recs[arm]
        row = {"clean_bpc": r["eval"]["clean_bpc"],
               "params_readout": r["params"]["params_per_exit_readout"],
               "params_total": r["params"]["params_total"],
               "tf_acceptance": r["tf_acceptance"],
               "prob0_ece": r["calibration"]["prob0_ece"],
               "tied_argmax_agreement":
                   r["soft_submission"]["tied_head_argmax_agreement"],
               "reeval_bitwise_identical": r["reeval_bitwise_identical"],
               "status": "已跑"}
        if arm != ref_name and ref_bpc is not None:
            row["delta_clean_bpc_vs_ref"] = round(
                r["eval"]["clean_bpc"] - ref_bpc, 4)
            row["bpc_beyond_band"] = bool(
                abs(row["delta_clean_bpc_vs_ref"]) > VARIANCE_BOUND)
            if ref_tf is not None and ref_per_batch is not None:
                tfref = {"by_depth": ref_tf, "per_batch_agree": ref_per_batch,
                         "n_per_batch": r["tf_n_per_batch"]}
                row["delta_tf_by_depth"] = [
                    paired_accept_delta({"by_depth": r["tf_acceptance"],
                                         "per_batch_agree":
                                             r["tf_per_batch_agree"],
                                         "n_per_batch": r["tf_n_per_batch"]},
                                        tfref, k)
                    for k in range(N_LAYERS)]
        rows[arm] = row

    # 判据线结论（预登记：README §4-P1..P4）
    verdict = {}
    shallow = (1, 2, 3, 4)
    pe = rows.get("PE", {})
    ind = rows.get("IND", {})
    lr = rows.get("LR", {})
    if pe.get("status") == "已跑":
        seps = [d for d in pe.get("delta_tf_by_depth", [])
                if d["depth"] in shallow and d["separable_from_zero"]
                and d["delta"] > 0]
        verdict["P1"] = {
            "question": "PE 相对共享 tied 能否让浅出口 tf@k（深度 1..4）可排序提升",
            "shallow_separable_positive_depths": [d["depth"] for d in seps],
            "separated": bool(seps),
            "detail": [{"depth": d["depth"], "delta": d["delta"],
                        "paired_bootstrap_95ci": d["paired_bootstrap_95ci"]}
                       for d in pe.get("delta_tf_by_depth", [])],
        }
        verdict["P2"] = {
            "question": "PE 末层 clean bpc 是否不劣于参考臂（不越 0.014 带）",
            "delta_clean_bpc": pe.get("delta_clean_bpc_vs_ref"),
            "beyond_band": pe.get("bpc_beyond_band"),
            "params_readout": pe.get("params_readout"),
            "params_readout_per_exit": pe.get("params_readout", 0) and
                pe["params_readout"] // EXITS,
        }
        verdict["P3"] = {
            "question": "prob0 / 软提交语义是否仍可用",
            "prob0_ece": pe.get("prob0_ece"),
            "delta_ece_vs_ref": (round(pe["prob0_ece"]
                                       - (recs["T"]["calibration"]["prob0_ece"]
                                          if "T" in recs else
                                          anchor_recheck.get(
                                              "calibration", {}).get(
                                              "prob0_ece", float("nan"))), 4)
                                 if pe.get("prob0_ece") is not None else None),
            "tolerance": ECE_TOLERANCE,
            "tied_argmax_agreement": pe.get("tied_argmax_agreement"),
        }
        verdict["P4"] = {
            "falsification_lines": {
                "shallow_tf_not_separable": not bool(seps),
                "only_unscalable_IND_helps": None,
                "last_layer_degraded": pe.get("beyond_band"),
            },
            "IND": {"status": ind.get("status", "未跑"),
                    "params_readout": ind.get("params_readout"),
                    "scalable": False},
            "LR": {"status": lr.get("status", "未跑"),
                   "params_readout": lr.get("params_readout")},
        }
    else:
        verdict["P1"] = {"status": "未跑", "note": "PE 未训练 ⇒ 无读数，不做任何推断"}
        verdict["P2"] = {"status": "未跑"}
        verdict["P3"] = {"status": "未跑"}
        verdict["P4"] = {"status": "未跑"}

    out = {
        "experiment": "2645-per-exit-readout-13m",
        "base": {"form": "depth-AR（dense=false + 变深度 + 逐层读出 + 随机退出 + fill_kv）",
                 "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                 "seq_len": SEQ_LEN, "mlp_mult": MLP_MULT, "exits": EXITS,
                 "steps": args.steps, "batch_size": 32, "lr": 1e-3, "seed": args.seed,
                 "data": DATA, "vocab_size": VOCAB},
        "reference_arm": ref_name,
        "anchor": {"source": os.path.relpath(ANCHOR_DIR, ROOT),
                   "hist_clean_bpc": ANCHOR_BPC, "hist_tf4": ANCHOR_TF4,
                   "hist_tf6": ANCHOR_TF6,
                   "recheck_all_equal": anchor_recheck.get(
                       "config_equivalence_all_equal"),
                   "recheck_tf_matches_2610": anchor_recheck.get(
                       "tf_matches_2610_bitwise"),
                   "recheck_clean_bpc": anchor_recheck.get(
                       "clean_bpc_remeasured")},
        "band": {"clean_bpc": VARIANCE_BOUND, "ece_tolerance": ECE_TOLERANCE,
                 "rates": "tf@k 走配对 cluster bootstrap（按批）+ Wilson 95% CI，"
                          "不跨量纲借用 0.014"},
        "arm_rows": rows,
        "verdict": verdict,
        "arms_declared": list(ARMS),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("anchor", "arm", "aggregate"),
                    default="arm")
    ap.add_argument("--arm", choices=ARMS, default="PE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.out_root is None:
        args.out_root = os.path.join(HERE, "runs")
    if args.aggregate or args.stage == "aggregate":
        aggregate(args)
        return
    if args.stage == "anchor":
        stage_anchor(args)
        return
    if not args.no_train:
        tr = train_arm(args.arm, args.seed, args.steps, args.batch_size,
                       args.lr, args.out_root)
        print(f"[2645] {run_name(args.arm, args.seed, args.steps)} wall="
              f"{tr['wall_s']}s", flush=True)
    evaluate(args.arm, args.seed, args.steps, args.out_root,
             n_batches=args.eval_batches)


if __name__ == "__main__":
    main()
