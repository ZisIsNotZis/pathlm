"""2642 · uniform 减宽（D192/D128）+ 嵌套宽度 dropout（WD256）· 13M · dense.

契约 = 同目录 `README.md`（判据线 P1–P5 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2）：

* **WP1（D192/D128）走原生配置**：引擎原生支持任意 `--d/--heads`（head_dim 恒 32），
  因此 uniform 减宽臂不需要任何干预代码 —— 只把 `shared_argv` 的 `--d/--heads`
  换成该臂的值，其余参数逐字段沿用 2641 的 dense 配方。
* **WP2（WD256）走前缀子网络**：训练每步采样宽度 `w∈{128,192,256}`，只在前 w 个
  残差通道组成的前缀子网络上做前向（`prefix_forward_full`：QKV/attn-out/FFN 权重
  按前缀行/列取子矩阵，LN 与残差在活跃前缀上算，读出头取 `E[:, :w]`）。
  推理时把同一 checkpoint 的**前缀子矩阵**装进一个**原生 d=w 的 stock 模型**
  （`build_sliced_model`）—— 那是**真**变小的模型（不是掩码），每 token 等效算力按 w 计。
* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成工厂（WD256 时工厂返回 `WidthDropoutModel`：stock 构造 + 前缀前向），然后调用
  `train_depth_ar.main()` —— 参数解析、配置装配、`train()` 循环、checkpoint、`results.json`
  全部是**共享实现**，本文件不改共享代码（`pathlm/**`、`train_*.py` 零改动）。
* 所有臂的初始化来自**同一次**随机抽样路径（`torch.manual_seed(seed)` 后同一构造顺序），
  与 2641 的 U4-s0 同 seed 同配方 ⇒ 臂间不是「两个不同初始化的模型」在比。

用法：

    python3 run.py --arm D192 --seed 0 --steps 12000        # 训练 + 评测
    python3 run.py --arm WD256 --seed 0 --steps 12000       # 训练 + 三档切片评测
    python3 run.py --arm D192 --seed 0 --steps 12000 --no-train   # 仅重评（确定性核对）
    python3 run.py --aggregate --arms D192,D128,WD256 --seeds 0   # 汇总 P1–P5 → results.json
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                        # noqa: E402
from pathlm.data import batch, load_enwik8_full                     # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.model import cap_norm                                   # noqa: E402

DEVICE = "cuda"

# ---------------------------------------------------------------- 常量（契约 §2/§3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
HEAD_DIM = 32
VOCAB = 206
VARIANCE_BOUND = 0.014                    # bpc 带（docs/research-schema.md §12.1 / C1）
GATE_BATCH, GATE_SEED = 8, 1234           # 等值门的固定批（B=8，generator seed 1234）
LIVENESS_SEED = 1234
REEVAL_EVAL_GENERATOR_SEED = 2           # = depth_curve 缺省（与 2641 同口径）

ANCHOR_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "3.1.4-ffn-ratio", "experiments",
    "2641-ffn-ratio-frontier-13m"))

# 历史同配置 dense 13M/12k run 散布（**citation**：2641 run.py 的 HIST_* 常量与 U4-s0 实测；
# 只作参照，不替代 0.014 预登记带）
HIST_DENSE_SPREAD = {"1.5257": "2641 run.py HIST_2613_BASELINE_S0（2613 全新重训 baseline-s0）",
                     "1.5287": "2641 run.py HIST_DENSE_BPC（.scratch/23-depth-ar/evidence/23d-dense-B-12k）",
                     "1.5298": "2641 run.py HIST_2613_BASELINE_S1（2613 baseline-s1）",
                     "1.5314": "2641 results_U4_s0.json（本实验复用锚）"}

WD_WIDTHS = (128, 192, 256)               # ρ = 4/8, 6/8, 8/8，等概率

ARMS = {
    "D192": {"kind": "uniform", "d": 192, "heads": 6},
    "D128": {"kind": "uniform", "d": 128, "heads": 4},
    "WD256": {"kind": "width_dropout", "d": 256, "heads": 8, "widths": WD_WIDTHS},
}


# ---------------------------------------------------------------- 代价与参数（等效理论量）

def analytic_cost(width: int, mlp_mult: int = MLP_MULT, T: int = SEQ_LEN,
                  L: int = N_LAYERS, vocab: int = VOCAB) -> dict:
    """每 token 等效 MACs（解析，**不含任何时间量**）。width = 活跃残差通道数。"""
    attn_per_layer = 4 * width * width + 2 * T * width   # qkv 3w² + proj w² + QKᵀ T·w + AV T·w
    ffn_per_layer = 2 * mlp_mult * width * width
    readout = width * vocab                              # tied 读出，每 token w×V
    total = L * (attn_per_layer + ffn_per_layer) + readout
    return {
        "width": width, "mlp_mult": mlp_mult, "seq_len": T, "n_layers": L,
        "vocab": vocab,
        "macs_per_token": {"attention": L * attn_per_layer,
                           "ffn": L * ffn_per_layer,
                           "readout_tied": readout,
                           "total": total},
        "note": ("等效理论量：MACs/token（乘加）。attention 的 T·w 项随序列长度线性、"
                 "FFN 的 w² 项与 T 无关；不用任何时间量。tokens-per-forward = 1。"
                 "width = 该档活跃残差通道数（uniform 臂 = d_model）"),
    }


def analytic_params(width: int, mlp_mult: int = MLP_MULT, T: int = SEQ_LEN,
                    L: int = N_LAYERS, vocab: int = VOCAB) -> dict:
    """dense 形态原生模型的解析参数量（分列）。

    `gates_*` 是 dense 形态下**不参与前向**的 carry 门参数（引擎在 `carry_gate=True`
    时照建）——与 2641 的 `params_total` 同一口径，故必须并列报出，
    防把「门参数」误读成「宽度参数」。`params_live` = 总参数 − 死门参数。
    """
    m = mlp_mult * width
    embed = vocab * width
    pos = T * width
    ffn = L * ((m * width + m) + (width * m + width))
    gates = L * (width * width + width)
    attn_ln = L * ((3 * width * width + 3 * width) + (width * width + width)
                   + 2 * (2 * width))
    total = embed + pos + ffn + gates + attn_ln
    return {"embedding": embed, "pos_embedding": pos,
            "embedding_total": embed + pos, "ffn": ffn, "attention_and_ln": attn_ln,
            "gates_dead_in_dense": gates, "params_total": total,
            "params_live": total - gates, "mlp_hidden": m,
            "note": ("解析参数量（dense 形态、carry_gate=True 原生构造）。"
                     "gates_dead_in_dense = 引擎照建但在 dense 前向中不用的门参数，"
                     "与 2641 params_total 同口径")}


# ---------------------------------------------------------------- WP2 干预：前缀子网络

def prefix_forward_full(model: DepthARModel, tokens: torch.Tensor, w: int,
                        mlp_mult: int = MLP_MULT, head_dim: int = HEAD_DIM,
                        capture: list | None = None, mutate: dict | None = None):
    """在**全宽模型的参数**上，只走宽度 w 的残差通道前缀子网络的前向。

    逐算子复制 stock `Block.forward` / `DepthARModel._input_states`+`readout` 的
    dtype 与算子顺序（同形状、同连续布局）⇒ 与「原生 d=w 模型」逐位相同（T2）。
    `mutate` 只用于单测的变异验证（故意切错），缺省 None = 无变异。
    """
    cfg = model.cfg
    d = cfg.d_model
    m = mlp_mult * w
    B, T = tokens.shape
    nh = w // head_dim
    mut = mutate or {}

    def take(t, key, rows, cols=None):
        """按 mutate 指定的替换张量 / 故意取错的切片（缺省 = 正确的前缀切片）。

        mutate[key] 直接给**替换张量**（形状须匹配）——单测的变异验证用。
        """
        if key in mut:
            return mut[key]
        return t[rows] if cols is None else t[rows, cols]

    Ew = model.embed.weight[:, :w].contiguous()
    h = F.embedding(tokens, Ew) + model.pos_embed.weight[:T, :w]
    h = cap_norm(h, cfg.norm_cap)
    states = [h]
    for l, blk in enumerate(model.blocks):
        x = h
        ln1_w = take(blk.ln1.weight, "ln1_w", slice(0, w))
        ln1_b = blk.ln1.bias[:w]
        qkv_w = torch.cat([blk.qkv.weight[0:w, :w],
                           blk.qkv.weight[d:d + w, :w],
                           blk.qkv.weight[2 * d:2 * d + w, :w]], dim=0)
        qkv_b = torch.cat([blk.qkv.bias[0:w], blk.qkv.bias[d:d + w],
                           blk.qkv.bias[2 * d:2 * d + w]], dim=0)
        qkv = F.linear(F.layer_norm(x, (w,), ln1_w, ln1_b, blk.ln1.eps),
                       qkv_w, qkv_b)
        q, k, v = qkv.chunk(3, dim=-1)
        nh_use = mut.get("n_heads", nh)
        shape = lambda t: t.view(B, T, nh_use, head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, w)
        h = x + F.linear(a, take(blk.proj.weight, "proj_w",
                                 slice(0, w), slice(0, w)),
                         blk.proj.bias[:w])
        ln2_w = take(blk.ln2.weight, "ln2_w", slice(0, w))
        ln2_b = blk.ln2.bias[:w]
        y = F.linear(F.layer_norm(h, (w,), ln2_w, ln2_b, blk.ln2.eps),
                     blk.mlp[0].weight[:m, :w], blk.mlp[0].bias[:m])
        y = F.gelu(y)                       # nn.GELU() 缺省 approximate='none'
        y = F.linear(y, blk.mlp[2].weight[:w, :m], blk.mlp[2].bias[:w])
        h = h + y                           # mlp[3] = Dropout(0.0) ⇒ 恒等
        if capture is not None:
            with torch.no_grad():
                xn = x.detach().float().norm(dim=-1).mean().item()
                yn = h.detach().float().norm(dim=-1).mean().item()
            capture.append({"layer": l + 1, "active_width": w, "hidden": m,
                            "in_norm_mean": round(xn, 6), "out_norm_mean": round(yn, 6),
                            "out_over_in": round(yn / max(xn, 1e-12), 6)})
        states.append(h)
    with torch.autocast("cuda", enabled=False):     # 与 stock `readout` 同款 fp32 头
        hf = h.float()
        logits = hf @ Ew.float().T
    return logits, states


def prefix_loss_full(model: DepthARModel, tokens: torch.Tensor, w: int,
                     mlp_mult: int = MLP_MULT, head_dim: int = HEAD_DIM):
    """前缀前向的 dense 损失：与 stock dense 路径**同一公式**（final_weight=1、
    readouts 只有最终深度 ⇒ loss = CE_final / 1）。返回 (loss, logits)。"""
    logits, _ = prefix_forward_full(model, tokens, w, mlp_mult, head_dim)
    mask = torch.ones(tokens.shape[0], tokens.shape[1] - 1, dtype=torch.bool,
                      device=tokens.device)
    ce = DepthARModel._masked_ce(logits[:, :-1], tokens[:, 1:], mask)
    loss = torch.zeros(()).float().to(ce.device) + 1.0 * ce
    return loss / (1.0 * 0 + 1.0), logits


def convert_state_dict(sd: dict, w: int, mlp_mult: int = MLP_MULT,
                       head_dim: int = HEAD_DIM, d_full: int = D_MODEL,
                       mask_instead_of_slice: bool = False) -> dict:
    """全宽 state_dict → 原生 d=w 模型的 state_dict（**前缀子矩阵**）。

    `mask_instead_of_slice=True` 是单测变异：把 QKV 当成「连续输出前缀 [:3w]」
    （经典的掩码式错切）而不是 q|k|v 三个块各自取前缀 —— 必须被 T2 抓红。
    """
    out = {"embed.weight": sd["embed.weight"][:, :w].contiguous(),
           "pos_embed.weight": sd["pos_embed.weight"][:, :w].contiguous()}
    for l in range(N_LAYERS):
        p = f"blocks.{l}."
        out[p + "ln1.weight"] = sd[p + "ln1.weight"][:w].contiguous()
        out[p + "ln1.bias"] = sd[p + "ln1.bias"][:w].contiguous()
        out[p + "ln2.weight"] = sd[p + "ln2.weight"][:w].contiguous()
        out[p + "ln2.bias"] = sd[p + "ln2.bias"][:w].contiguous()
        if mask_instead_of_slice:
            out[p + "qkv.weight"] = sd[p + "qkv.weight"][:3 * w, :w].contiguous()
        else:
            out[p + "qkv.weight"] = torch.cat(
                [sd[p + "qkv.weight"][0:w, :w],
                 sd[p + "qkv.weight"][d_full:d_full + w, :w],
                 sd[p + "qkv.weight"][2 * d_full:2 * d_full + w, :w]], dim=0).contiguous()
        out[p + "qkv.bias"] = torch.cat(
            [sd[p + "qkv.bias"][0:w], sd[p + "qkv.bias"][d_full:d_full + w],
             sd[p + "qkv.bias"][2 * d_full:2 * d_full + w]], dim=0).contiguous()
        out[p + "proj.weight"] = sd[p + "proj.weight"][:w, :w].contiguous()
        out[p + "proj.bias"] = sd[p + "proj.bias"][:w].contiguous()
        out[p + "mlp.0.weight"] = sd[p + "mlp.0.weight"][:mlp_mult * w, :w].contiguous()
        out[p + "mlp.0.bias"] = sd[p + "mlp.0.bias"][:mlp_mult * w].contiguous()
        out[p + "mlp.2.weight"] = sd[p + "mlp.2.weight"][:w, :mlp_mult * w].contiguous()
        out[p + "mlp.2.bias"] = sd[p + "mlp.2.bias"][:w].contiguous()
        out[f"gates.{l}.weight"] = sd[f"gates.{l}.weight"][:w, :w].contiguous()
        out[f"gates.{l}.bias"] = sd[f"gates.{l}.bias"][:w].contiguous()
    return out


def width_cfg(w: int, mlp_mult: int = MLP_MULT, head_dim: int = HEAD_DIM) -> DepthARConfig:
    """原生 d=w 的 dense stock 配置（head_dim 恒 32）。"""
    if w % head_dim:
        raise ValueError(f"宽度 {w} 必须整除 head_dim {head_dim}")
    return DepthARConfig(d_model=w, n_layers=N_LAYERS, n_heads=w // head_dim,
                         seq_len=SEQ_LEN, mlp_mult=mlp_mult, dense=True,
                         topology="sequential", fill_kv=True)


def build_sliced_model(sd: dict, w: int, vocab: int = VOCAB,
                       mask_instead_of_slice: bool = False) -> DepthARModel:
    """**真**变小：原生 d=w 模型 + 前缀子矩阵权重（不是掩码）。"""
    model = DepthARModel(width_cfg(w), vocab)
    model.load_state_dict(convert_state_dict(sd, w,
                                             mask_instead_of_slice=mask_instead_of_slice))
    return model


class WidthDropoutModel(DepthARModel):
    """stock 构造 + 每步采样宽度、只走残差通道前缀子网络（WP2 的干预）。"""

    def __init__(self, cfg: DepthARConfig, vocab_size: int, widths=WD_WIDTHS,
                 seed: int = 0):
        super().__init__(cfg, vocab_size)
        self.wd_widths = tuple(int(x) for x in widths)
        self.wd_gen = torch.Generator().manual_seed(int(seed) + 7)  # 不进 state_dict
        self.wd_samples: list[int] = []
        if not cfg.dense:
            raise ValueError("WidthDropoutModel 只在 dense 形态下定义（单因素）")
        for w in self.wd_widths:
            if w % HEAD_DIM or w > cfg.d_model:
                raise ValueError(f"宽度 {w} 非法（需整除 {HEAD_DIM} 且 ≤ d_model）")

    def sample_width(self) -> int:
        idx = int(torch.randint(len(self.wd_widths), (1,),
                                generator=self.wd_gen).item())
        w = self.wd_widths[idx]
        self.wd_samples.append(w)
        return w

    def forward(self, tokens, exit_depths=None, targets=None, shallow_weight=1.0,
                final_weight=1.0, shallow_stopgrad=False, dense_aux=False):
        """与 stock dense 前向同损失公式（readouts 只有最终深度 ⇒ loss = CE_final）。"""
        w = self.sample_width()
        logits, states = prefix_forward_full(self, tokens, w)
        tg = tokens if targets is None else targets
        mask = torch.ones(tokens.shape[0], tokens.shape[1] - 1, dtype=torch.bool,
                          device=tokens.device)
        ce = self._masked_ce(logits[:, :-1], tg[:, 1:], mask)
        loss = torch.zeros(()).float().to(ce.device) + final_weight * ce
        loss = loss / (shallow_weight * (1 - 1) + final_weight)
        return loss, {"states": states, "depth_logits": [logits], "depth_ce": [ce],
                      "gate_mean": [], "exit_depths": exit_depths,
                      "sampled_width": w}


# ---------------------------------------------------------------- 训练（走共享 main）

_WD_INSTANCES: list = []          # 工厂创建的模型实例（训练后读宽度采样直方图）
_ORIG_KV_HEAD_LIVENESS = TDA.kv_head_liveness


def _kv_head_liveness_wd_safe(model, tokens):
    """共享 `kv_head_liveness` 的 WD256 安全包装。

    共享探针在 `model.blocks[*].qkv` 上挂 hook 并跑 `model(...)`，断言每层都被 hook 到。
    但 `WidthDropoutModel.forward` 走前缀子网络（`prefix_forward_full` 直接 `F.linear`
    取 `blk.qkv.weight` 切片，不调用 `blk.qkv` 模块）⇒ qkv hook 一个也不触发，共享断言
    `len(widths) == n_layers` 失败（`pathlm/depth_ar.py:713`），冒烟 rc=1。

    修法：探测期换成「同 cfg、同 state_dict、原生 d=256 的 stock 模型」再跑原探针。
    WD256 在 w=256 的前向与 stock 逐位相同（T1 门），而该探针只读 qkv **输出宽度**
    （形状类活性量）、不参与训练/评测数值路径，也不入判据。`pathlm/**`、`train_*.py`
    零改动。
    """
    if isinstance(model, WidthDropoutModel):
        stock = DepthARModel(model.cfg, model.vocab_size)
        stock.load_state_dict(model.state_dict())
        stock.to(next(model.parameters()).device)
        stock.eval()
        model = stock
    return _ORIG_KV_HEAD_LIVENESS(model, tokens)


def _factory(arm: str, seed: int):
    def factory(cfg, vocab_size):
        if ARMS[arm]["kind"] == "width_dropout":
            m = WidthDropoutModel(cfg, vocab_size, widths=ARMS[arm]["widths"],
                                  seed=seed)
            _WD_INSTANCES.append(m)
            return m
        return DepthARModel(cfg, vocab_size)
    return factory


def shared_argv(arm: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字段对齐（dense 形态；唯一变量 = 宽度）。"""
    a = ARMS[arm]
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-s{seed}-{steps // 1000}k",
            "--d", str(a["d"]), "--layers", str(N_LAYERS),
            "--heads", str(a["heads"]), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    orig = TDA.DepthARModel
    orig_kv = TDA.kv_head_liveness
    TDA.DepthARModel = _factory(arm, seed)
    TDA.kv_head_liveness = _kv_head_liveness_wd_safe
    argv_backup = list(sys.argv)
    t0 = time.time()
    try:
        sys.argv = shared_argv(arm, seed, steps, batch_size, lr, out_root)
        TDA.main()                       # 共享实现：解析 / 配置 / train() / 落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        TDA.kv_head_liveness = orig_kv
        sys.argv = argv_backup
    info = {"wall_s": round(wall, 1)}
    if _WD_INSTANCES:
        samples = list(_WD_INSTANCES[-1].wd_samples)
        # 训练每步恰好一次模型前向 ⇒ 前 `steps` 条采样就是训练步；训练结束后
        # `train_depth_ar.main()` 里的活性探针（如 kv_head_liveness）也会走一次前向，
        # 它们不计入训练直方图（单列报出，防污染）。
        train_samples = samples[:steps]
        counts = {str(w): train_samples.count(w) for w in ARMS[arm]["widths"]}
        info["width_samples"] = {
            "n_train_steps": len(train_samples),
            "counts": counts,
            "fracs": {k: round(v / max(len(train_samples), 1), 4)
                      for k, v in counts.items()},
            "n_post_train_probe_forwards_ignored": len(samples) - len(train_samples),
            "first_24": train_samples[:24]}
        rd = run_dir(arm, seed, steps, out_root)
        with open(os.path.join(rd, "width_samples.json"), "w") as f:
            json.dump(info["width_samples"], f, indent=2, ensure_ascii=False)
    return info


# ---------------------------------------------------------------- 评测（本实验自持）

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_sd(rd: str) -> dict:
    return torch.load(os.path.join(rd, "model.pt"), map_location="cpu")


def _build_full_model(d: int, heads: int, vocab: int) -> DepthARModel:
    cfg = DepthARConfig(d_model=d, n_layers=N_LAYERS, n_heads=heads,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True,
                        topology="sequential", fill_kv=True)
    return DepthARModel(cfg, vocab).to(DEVICE)


@torch.no_grad()
def block_liveness(model: DepthARModel, eval_arr, w: int | None = None,
                   n_batches: int = 2, batch_size: int = 8,
                   seed: int = LIVENESS_SEED) -> dict:
    """活性（`SCHEMA.md` §10）：**真实前向**中用 forward hook 抓每层块的输入/输出，
    报逐层 ‖out‖/‖in‖ 的逐位置均值 —— 每层真的参与前向、不是死路径。"""
    gen = torch.Generator().manual_seed(seed)
    acc = [[] for _ in model.blocks]
    handles = []

    def hook(l):
        def fn(_mod, inp, out):
            # patch: `Block.forward`（`pathlm/model.py`）返回 `(block_out, telemetry)`
            # 元组，TopK/Carry 等块同构 ⇒ 主输出恒为第 0 项。只比 tensor 时取元组会
            # `AttributeError`（回归门见 test_2642_uniform_narrowing.py 的 liveness 用例）。
            out_t = out[0] if isinstance(out, tuple) else out
            x = inp[0].detach().float()
            y = out_t.detach().float()
            acc[l].append(float((y.norm(dim=-1) /
                                 x.norm(dim=-1).clamp_min(1e-6)).mean()))
        return fn

    for l, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(hook(l)))
    was_training = model.training
    model.eval()
    try:
        with torch.autocast("cuda", enabled=False):
            for _ in range(n_batches):
                x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
                model(x.to(DEVICE), exit_depths=None)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {"active_width": (w if w is not None else model.cfg.d_model),
            "hidden_by_layer": [int(blk.mlp[0].weight.shape[0])
                                for blk in model.blocks],
            "block_out_over_in_by_layer": [round(statistics.mean(v), 6)
                                           for v in acc],
            "min_ratio": round(min(statistics.mean(v) for v in acc), 6),
            "note": ("逐层 ‖block_out‖/‖in‖ 的逐位置均值（>0 ⇒ 该层真的参与前向）；"
                     "hidden_by_layer 必须逐位等于 4×活跃宽度")}


@torch.no_grad()
def causality_check(model: DepthARModel, eval_arr, seed: int = LIVENESS_SEED,
                    batch_size: int = 2) -> dict:
    """因果性：改**末位** token ⇒ 之前位置的 logits 必须逐位不变，末位必须变。"""
    gen = torch.Generator().manual_seed(seed)
    x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
    x = x.to(DEVICE)
    with torch.autocast("cuda", enabled=False):
        lg1 = model(x, exit_depths=None)[1]["depth_logits"][-1]
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % model.vocab_size
        lg2 = model(x2, exit_depths=None)[1]["depth_logits"][-1]
    d_past = float((lg1[:, :-1] - lg2[:, :-1]).abs().max().item())
    d_last = float((lg1[:, -1] - lg2[:, -1]).abs().max().item())
    return {"max_abs_dlogit_past_positions": d_past,
            "max_abs_dlogit_last_position": d_last,
            "pass": bool(d_past == 0.0 and d_last > 0.0),
            "note": "改末位 token：之前位置逐位不变（因果掩码真的在）、末位必须变（活性对照）"}


def _eval_curve(model: DepthARModel, eval_arr, n_batches: int) -> list[dict]:
    return depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                       device=DEVICE)


def evaluate_uniform(arm: str, seed: int, steps: int, out_root: str,
                     n_batches: int = 20) -> dict:
    """WP1 臂（原生 d/heads）：评测 + 参数/活性/因果性审计 → results_<arm>_s<seed>.json"""
    rd = run_dir(arm, seed, steps, out_root)
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    a = ARMS[arm]
    model = _build_full_model(a["d"], a["heads"], vocab)
    model.load_state_dict(_load_sd(rd))
    model.eval()
    curve = _eval_curve(model, eval_arr, n_batches)
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(p.numel() for blk in model.blocks for p in blk.mlp.parameters())
    ap = analytic_params(a["d"], vocab=vocab)
    out = {
        "arm": arm, "seed": seed, "steps": steps, "kind": a["kind"],
        "width": a["d"], "n_heads": a["heads"], "head_dim": a["d"] // a["heads"],
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": REEVAL_EVAL_GENERATOR_SEED,
                 "bpc_by_depth": curve, "clean_bpc": curve[-1]["bpc"],
                 "acc_by_depth": [{"depth": r["depth"], "acc": r["acc"]} for r in curve]},
        "params": {"params_total_measured": total, "params_ffn_measured": ffn,
                   "params_embedding_measured": model.embed.weight.numel()
                   + model.pos_embed.weight.numel(),
                   "params_gates_dead_measured": sum(
                       p.numel() for p in model.gates.parameters()),
                   "analytic": ap,
                   "params_total_matches_analytic": total == ap["params_total"],
                   "params_ffn_matches_analytic": ffn == ap["ffn"],
                   "hidden_by_layer": [int(blk.mlp[0].weight.shape[0])
                                       for blk in model.blocks]},
        "cost": analytic_cost(a["d"], vocab=vocab),
        "liveness": block_liveness(model, eval_arr, w=a["d"]),
        "causality": causality_check(model, eval_arr),
        "shared_train_results": _shared_summary(rd),
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "model_pt_sha256": sha256_file(os.path.join(rd, "model.pt")),
        "no_engine_change": ("pathlm/**、train_*.py 零改动：本臂只改 shared_argv 的 "
                             "--d/--heads（引擎原生配置）"),
    }
    out["params"]["hidden_matches_width"] = (
        out["params"]["hidden_by_layer"] == [MLP_MULT * a["d"]] * N_LAYERS)
    json.dump(out, open(os.path.join(HERE, f"results_{arm}_s{seed}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[2642] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"macs={out['cost']['macs_per_token']['total']} "
          f"params={total} hidden={out['params']['hidden_by_layer']}", flush=True)
    return out


def _shared_summary(rd: str) -> dict:
    try:
        shared = json.load(open(os.path.join(rd, "results.json")))
    except Exception:
        return {}
    return {"params": shared.get("params"), "train_tokens": shared.get("train_tokens"),
            "wall_minutes": shared.get("wall_minutes"),
            "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
            "config": shared.get("config"),
            "gnorm_monitor_max": (shared.get("gnorm_monitor") or {}).get("gnorm_max")}


def _gate_batch(eval_arr, vocab: int, batch_size: int = GATE_BATCH):
    gen = torch.Generator().manual_seed(GATE_SEED)
    x, _ = batch(eval_arr, batch_size, SEQ_LEN, gen)
    return x.to(DEVICE)


def evaluate_width_dropout(seed: int, steps: int, out_root: str,
                           n_batches: int = 20) -> dict:
    """WP2 臂：同一 checkpoint 三档切片评测 + 等值门 T1/T2 + 参数/活性/因果性审计。"""
    rd = run_dir("WD256", seed, steps, out_root)
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    sd = _load_sd(rd)
    full = _build_full_model(D_MODEL, N_HEADS, vocab)
    full.load_state_dict(sd)
    full.eval()
    xg = _gate_batch(eval_arr, vocab)

    # ---- 等值门（训练后复核 T1/T2；单测见 test_2642_uniform_narrowing.py）----
    gates = {}
    for mode in ("fp32", "bf16_autocast"):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode != "fp32")):
                # T1：w = d 的前缀前向 == stock 引擎前向（逐位）
                lp, _ = prefix_forward_full(full, xg, D_MODEL)
                ls, _ = prefix_loss_full(full, xg, D_MODEL)
                lf = full(xg, exit_depths=None)[1]["depth_logits"][-1]
                loss_f = full(xg, exit_depths=None)[0]
                t1 = {"bitwise_equal_logits": bool(torch.equal(lp, lf)),
                      "bitwise_equal_loss": bool(torch.equal(ls, loss_f)),
                      "max_abs_logit_diff": float((lp.float() - lf.float()).abs().max()),
                      "max_abs_loss_diff": float((ls.float() - loss_f.float()).abs().max())}
                # T2：切片模型前向 == 同宽度训练期前缀前向（逐位）
                t2 = {}
                for w in ARMS["WD256"]["widths"]:
                    sl = build_sliced_model(sd, w, vocab).to(DEVICE).eval()
                    lpw, _ = prefix_forward_full(full, xg, w)
                    lsw = sl(xg, exit_depths=None)[1]["depth_logits"][-1]
                    loss_pw, _ = prefix_loss_full(full, xg, w)
                    loss_sw = sl(xg, exit_depths=None)[0]
                    t2[str(w)] = {
                        "bitwise_equal_logits": bool(torch.equal(lpw, lsw)),
                        "bitwise_equal_loss": bool(torch.equal(loss_pw, loss_sw)),
                        "max_abs_logit_diff": float((lpw.float() - lsw.float()).abs().max()),
                        "max_abs_loss_diff": float((loss_pw.float() - loss_sw.float()).abs().max())}
        gates[mode] = {"T1_prefix_w_eq_d_vs_stock": t1, "T2_sliced_vs_prefix_by_width": t2}

    # ---- 三档切片模型的部署读数 ----
    per_width = {}
    for w in ARMS["WD256"]["widths"]:
        sl = build_sliced_model(sd, w, vocab).to(DEVICE).eval()
        curve = _eval_curve(sl, eval_arr, n_batches)
        measured = sum(p.numel() for p in sl.parameters())
        ap = analytic_params(w, vocab=vocab)
        max_dim = max(max(t.shape) for t in sl.state_dict().values() if t.dim() >= 2)
        per_width[str(w)] = {
            "width": w, "n_heads": w // HEAD_DIM, "head_dim": HEAD_DIM,
            "eval": {"metric": "clean bpc（切片后原生 d=w 模型，同评测批）",
                     "n_batches": n_batches, "batch_size": 16,
                     "eval_generator_seed": REEVAL_EVAL_GENERATOR_SEED,
                     "bpc_by_depth": curve, "clean_bpc": curve[-1]["bpc"]},
            "params": {"params_total_measured": measured,
                       "params_ffn_measured": sum(p.numel() for blk in sl.blocks
                                                  for p in blk.mlp.parameters()),
                       "analytic": ap, "params_total_matches_analytic": measured == ap["params_total"],
                       "max_tensor_dim": int(max_dim),
                       "no_full_width_tensor": bool(max_dim <= MLP_MULT * w),
                       "hidden_by_layer": [int(blk.mlp[0].weight.shape[0])
                                           for blk in sl.blocks]},
            "cost": analytic_cost(w, vocab=vocab),
            "liveness": block_liveness(sl, eval_arr, w=w),
            "causality": causality_check(sl, eval_arr),
        }
        per_width[str(w)]["params"]["active_dim_matches_target"] = bool(
            sl.cfg.d_model == w and per_width[str(w)]["params"]["hidden_by_layer"]
            == [MLP_MULT * w] * N_LAYERS)
    monotone = [per_width[str(w)]["eval"]["clean_bpc"] for w in (128, 192, 256)]
    out = {
        "arm": "WD256", "seed": seed, "steps": steps, "kind": "width_dropout",
        "training_widths": list(ARMS["WD256"]["widths"]),
        "training_sample_distribution": "每步等概率采样（torch.Generator，seed+7）",
        "width_samples": _read_width_samples(rd),
        "per_width": per_width,
        "clean_bpc_by_width": {str(w): per_width[str(w)]["eval"]["clean_bpc"]
                               for w in (128, 192, 256)},
        "monotone_wider_is_better": bool(monotone[0] > monotone[1] > monotone[2]),
        "adjacent_gaps": {"128_vs_192": round(monotone[0] - monotone[1], 4),
                          "192_vs_256": round(monotone[1] - monotone[2], 4)},
        "equivalence_gates_after_training": gates,
        "all_gates_bitwise_equal": bool(
            all(g["T1_prefix_w_eq_d_vs_stock"]["bitwise_equal_logits"]
                and g["T1_prefix_w_eq_d_vs_stock"]["bitwise_equal_loss"]
                and all(v["bitwise_equal_logits"] and v["bitwise_equal_loss"]
                        for v in g["T2_sliced_vs_prefix_by_width"].values())
                for g in gates.values())),
        "shared_train_results": _shared_summary(rd),
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "model_pt_sha256": sha256_file(os.path.join(rd, "model.pt")),
        "no_engine_change": ("pathlm/**、train_*.py 零改动：前缀前向与切片转换都实现在"
                             "本实验目录内；等值性由 T1/T2 逐位钉住"),
    }
    json.dump(out, open(os.path.join(HERE, f"results_WD256_s{seed}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[2642] WD256-s{seed} clean_bpc_by_width={out['clean_bpc_by_width']} "
          f"monotone={out['monotone_wider_is_better']} "
          f"gates={out['all_gates_bitwise_equal']}", flush=True)
    return out


def _read_width_samples(rd: str) -> dict:
    p = os.path.join(rd, "width_samples.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def evaluate_arm(arm: str, seed: int, steps: int, out_root: str) -> dict:
    if ARMS[arm]["kind"] == "width_dropout":
        return evaluate_width_dropout(seed, steps, out_root)
    return evaluate_uniform(arm, seed, steps, out_root)


# ---------------------------------------------------------------- 重评（评测确定性，P4-①）

def reeval(arm: str, seed: int, steps: int, out_root: str) -> dict:
    """`--no-train`：只重评并把与首评的**逐位一致性**落进 reeval_<arm>_s<seed>.json。"""
    prev_path = os.path.join(HERE, f"results_{arm}_s{seed}.json")
    prev = json.load(open(prev_path)) if os.path.exists(prev_path) else None
    if ARMS[arm]["kind"] == "width_dropout":
        now = evaluate_width_dropout(seed, steps, out_root)
        bw = {}
        for w in ARMS["WD256"]["widths"]:
            k = str(w)
            pv = prev["per_width"][k]["eval"] if prev else None
            nw = now["per_width"][k]["eval"]
            bw[k] = {"prev_clean_bpc": pv["clean_bpc"] if pv else None,
                     "now_clean_bpc": nw["clean_bpc"],
                     "bitwise_equal": bool(pv and pv["clean_bpc"] == nw["clean_bpc"]),
                     "bpc_by_depth_bitwise_equal": bool(
                         pv and pv["bpc_by_depth"] == nw["bpc_by_depth"])}
        out = {"arm": arm, "seed": seed, "steps": steps,
               "clean_bpc_by_width_bitwise_equal": bool(
                   all(v["bitwise_equal"] and v["bpc_by_depth_bitwise_equal"]
                       for v in bw.values())),
               "per_width": bw, "prev_file": os.path.relpath(prev_path, HERE)}
    else:
        now = evaluate_uniform(arm, seed, steps, out_root)
        pv = prev["eval"] if prev else None
        out = {"arm": arm, "seed": seed, "steps": steps,
               "prev_clean_bpc": pv["clean_bpc"] if pv else None,
               "now_clean_bpc": now["eval"]["clean_bpc"],
               "bitwise_equal": bool(pv and pv["clean_bpc"]
                                     == now["eval"]["clean_bpc"]),
               "bpc_by_depth_bitwise_equal": bool(
                   pv and pv["bpc_by_depth"] == now["eval"]["bpc_by_depth"]),
               "prev_file": os.path.relpath(prev_path, HERE)}
    json.dump(out, open(os.path.join(HERE, f"reeval_{arm}_s{seed}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[2642 reeval] {arm}-s{seed} {json.dumps(out, ensure_ascii=False)}",
          flush=True)
    return out


# ---------------------------------------------------------------- 锚（复用 2641，不重训）

def load_anchor(arm: str, seed: int = 0) -> dict:
    p = os.path.join(ANCHOR_DIR, f"results_{arm}_s{seed}.json")
    d = json.load(open(p))
    return {"arm": arm, "seed": seed, "clean_bpc": d["eval"]["clean_bpc"],
            "clean_bpc_by_depth": d["eval"]["bpc_by_depth"],
            "macs_per_token": d["cost"]["macs_per_token"]["total"],
            "macs_attn": d["cost"]["macs_per_token"]["attention"],
            "macs_ffn": d["cost"]["macs_per_token"]["ffn"],
            "macs_readout": d["cost"]["macs_per_token"]["readout_tied"],
            "params_total": d["params"]["params_total"],
            "params_ffn": d["params"]["params_ffn"],
            "source": os.path.relpath(p, ROOT), "sha256": sha256_file(p),
            "run_dir": d["run_dir"], "code_hash_run_py": d["code_hash_run_py"],
            "reuse": "不重训：本实验逐字段沿用 2641 的 dense 配方（见 anchor_config_parity）"}


def anchor_config_parity(steps: int, batch_size: int, lr: float) -> dict:
    """逐字段配置对照（复用纪律的三道具之二）：本实验 U4 等价轴配置 == 2641 config.json。"""
    cfg = json.load(open(os.path.join(ANCHOR_DIR, "config.json")))
    base, tr = cfg["base"], cfg["training"]
    fields = {"d_model": [base["d_model"], D_MODEL], "n_layers": [base["n_layers"], N_LAYERS],
              "n_heads": [base["n_heads"], N_HEADS], "seq_len": [base["seq_len"], SEQ_LEN],
              "mlp_mult": [base["mlp_mult"], MLP_MULT], "dense": [base["dense"], True],
              "dropout": [base["dropout"], 0.0], "norm_cap": [base["norm_cap"], 1.0],
              "topology": [base["topology"], "sequential"],
              "exit_mode": [base["exit_mode"], "uniform"],
              "fill_kv": [base["fill_kv"], True],
              "steps": [tr["steps"], steps],
              "batch_size": [tr["batch_size"], batch_size], "lr": [tr["lr"], lr]}
    diff = {k: v for k, v in fields.items() if v[0] != v[1]}
    return {"anchor_config": os.path.relpath(os.path.join(ANCHOR_DIR, "config.json"), ROOT),
            "fields_compared": fields, "mismatched_fields": diff,
            "all_equal": not diff,
            "note": ("D192/D128 的**唯一**偏离 = d_model/n_heads（契约 §2：唯一变量 = 宽度）；"
                     "WD256 的训练配置与 U4 逐字段相同，干预在前向（宽度前缀）而非配置")}


# ---------------------------------------------------------------- 汇总（P1–P5）

def p5b_frontier_verdict(slopes: dict, s_ffn, *, in_band: bool,
                         delta_bpc_gap, bound: float = VARIANCE_BOUND) -> str:
    """P5b 判词：两只旋钮的斜率差**回乘算力节省 %** 后才与噪声带同量纲。

    口径来源（复核 R-3.1.1 的 blocking B1）：D192 的 0.001287 与 U2 的 0.001392 都是
    「相对 U4 的 Δbpc ÷ 算力节省 %」；把 beide 相减得到的 0.000105 本身不是 bpc 量纲，
    要乘以相应算力节省 % 得到 `delta_bpc_gap`（本实验 = 0.0132）才能与 0.014 带比。
    带内 ⇒ 只输出「不可排序」，**取消**方向性宣称；越带才允许方向性判词。
    本函数是 P5b 的**唯一**判词入口，供 `aggregate()` 与单测共用（红-绿门）。
    """
    if in_band:
        return "未触发（P5a 已成立）"
    if s_ffn is None:
        return "U2 斜率缺失，无法比较"
    detail = " / ".join(f"{a}={v}" for a, v in slopes.items())
    if delta_bpc_gap is not None and delta_bpc_gap < bound:
        return (f"不可排序（本尺度）：每 1% 算力节省的质量代价 {detail} vs "
                f"U2(FFN 旋钮)={s_ffn} 的差，对应 Δbpc 差 {round(delta_bpc_gap, 4)} "
                f"< {bound} 噪声带 ⇒ d_model 减宽与 FFN 比例不可排序；"
                f"能确定的只是两者互不支配")
    worse = all(v is not None and v >= s_ffn for v in slopes.values())
    return (f"d_model 减宽不占前沿优势：每 1% 算力节省的质量代价 {detail} "
            f"均 ≥ U2(FFN 旋钮)={s_ffn}"
            if worse else
            f"至少一臂优于 FFN 旋钮：{detail} vs U2={s_ffn}（Δbpc 差越带）")

def _verdict(arm_bpc: float | None, anchor_bpc: float | None) -> str | None:
    if arm_bpc is None or anchor_bpc is None:
        return None
    d = round(arm_bpc - anchor_bpc, 4)
    if d >= VARIANCE_BOUND:
        return f"可分：{d:+.4f} bpc（更差，越 {VARIANCE_BOUND} 带）"
    if d <= -VARIANCE_BOUND:
        return f"可分：{d:+.4f} bpc（更好，越 {VARIANCE_BOUND} 带）"
    return f"不可排序：{d:+.4f} bpc 落在 ±{VARIANCE_BOUND} 带内"


def aggregate(steps: int, out_root: str, arms: list[str], seeds: list[int],
              batch_size: int = 32, lr: float = 1e-3) -> dict:
    recs = {}
    for arm in arms:
        for s in seeds:
            p = os.path.join(HERE, f"results_{arm}_s{s}.json")
            if os.path.exists(p):
                recs[(arm, s)] = json.load(open(p))
    anchors = {a: load_anchor(a) for a in ("U4", "U2")
               if os.path.exists(os.path.join(ANCHOR_DIR, f"results_{a}_s0.json"))}

    def bpc(arm, s):
        if (arm, s) not in recs:
            return None
        r = recs[(arm, s)]
        return (r["clean_bpc_by_width"]["256"] if r["kind"] == "width_dropout"
                else r["eval"]["clean_bpc"])

    def bpc_w(arm, w, s):
        if (arm, s) not in recs:
            return None
        return recs[(arm, s)]["per_width"][str(w)]["eval"]["clean_bpc"]

    def macs(arm, s, w=None):
        if (arm, s) not in recs:
            return None
        r = recs[(arm, s)]
        if r["kind"] == "width_dropout":
            return r["per_width"][str(w)]["cost"]["macs_per_token"]["total"]
        return r["cost"]["macs_per_token"]["total"]

    def params(arm, s, w=None):
        if (arm, s) not in recs:
            return None
        r = recs[(arm, s)]
        if r["kind"] == "width_dropout":
            return r["per_width"][str(w)]["params"]["params_total_measured"]
        return r["params"]["params_total_measured"]

    # ---- 前沿表（U4 / U2 锚 + D192 / D128 + WD256 三档）----
    frontier = {}
    for a, d in anchors.items():
        frontier[f"{a}-s0(anchor)"] = {"clean_bpc": d["clean_bpc"],
                                       "macs_per_token": d["macs_per_token"],
                                       "params_total": d["params_total"],
                                       "params_ffn": d["params_ffn"],
                                       "source": d["source"], "anchor": True}
    for a in ("D192", "D128"):
        if (a, 0) in recs:
            r = recs[(a, 0)]
            frontier[f"{a}-s0"] = {"clean_bpc": r["eval"]["clean_bpc"],
                                   "macs_per_token": r["cost"]["macs_per_token"]["total"],
                                   "params_total": r["params"]["params_total_measured"],
                                   "params_ffn": r["params"]["params_ffn_measured"],
                                   "anchor": False}
    if ("WD256", 0) in recs:
        for w in (256, 192, 128):
            k = str(w)
            frontier[f"WD256-s0@{w}"] = {
                "clean_bpc": recs[("WD256", 0)]["per_width"][k]["eval"]["clean_bpc"],
                "macs_per_token": recs[("WD256", 0)]["per_width"][k]["cost"]
                ["macs_per_token"]["total"],
                "params_total": recs[("WD256", 0)]["per_width"][k]["params"]
                ["params_total_measured"],
                "params_ffn": recs[("WD256", 0)]["per_width"][k]["params"]
                ["params_ffn_measured"], "anchor": False}

    # ---- 相对锚的节省与斜率（bpc 代价 / 每 1% 算力节省）----
    u4_bpc, u4_macs = anchors.get("U4", {}).get("clean_bpc"), anchors.get("U4", {}).get("macs_per_token")
    savings = {}

    def slope(arm_bpc_, arm_macs_):
        if None in (arm_bpc_, arm_macs_, u4_bpc, u4_macs) or arm_macs_ == u4_macs:
            return None
        cost_cut_pct = (1 - arm_macs_ / u4_macs) * 100
        return {"macs_cut_pct": round(cost_cut_pct, 3),
                "delta_bpc": round(arm_bpc_ - u4_bpc, 4),
                "bpc_cost_per_1pct_macs_saved": round(
                    (arm_bpc_ - u4_bpc) / cost_cut_pct, 6)}

    for name, key in (("U2", "U2-s0(anchor)"), ("D192", "D192-s0"), ("D128", "D128-s0")):
        if key in frontier:
            savings[name] = slope(frontier[key]["clean_bpc"], frontier[key]["macs_per_token"])
    for w in (256, 192, 128):
        key = f"WD256-s0@{w}"
        if key in frontier:
            savings[f"WD256@{w}"] = slope(frontier[key]["clean_bpc"],
                                          frontier[key]["macs_per_token"])

    # ---- P1 ----
    p1 = {}
    for a in ("D192", "D128"):
        if a in savings and (a, 0) in recs:
            p1[a] = {"delta_bpc_vs_U4": savings[a]["delta_bpc"],
                     "macs_cut_pct": savings[a]["macs_cut_pct"],
                     "bpc_cost_per_1pct_macs_saved":
                         savings[a]["bpc_cost_per_1pct_macs_saved"],
                     "params_total": frontier[f"{a}-s0"]["params_total"],
                     "params_ffn": frontier[f"{a}-s0"]["params_ffn"],
                     "verdict": _verdict(frontier[f"{a}-s0"]["clean_bpc"], u4_bpc)}

    # ---- P2（前沿对照：d_model 减宽 vs FFN 比例，等算力位点）----
    p2 = {"plane": "x = 每 token 等效 MACs，y = clean bpc（越低越好）",
          "table": {k: v for k, v in frontier.items()},
          "slopes_from_U4": savings,
          "U2_macs_diff_vs_D192": (None if not all(
              k in frontier for k in ("U2-s0(anchor)", "D192-s0")) else
              frontier["D192-s0"]["macs_per_token"] - frontier["U2-s0(anchor)"]["macs_per_token"]),
          "note": ("U2 = d_model 固定、FFN 预算减半的等算力近似位点（另一只旋钮）；"
                   "D192 = 整体减宽到 0.75² 算力位点。参数量与 MACs 两列并列，"
                   "防把「减参数」记成「减宽度」")}

    # ---- P3（WP2 弹性宽度）----
    p3 = None
    if ("WD256", 0) in recs:
        r = recs[("WD256", 0)]
        curve = r["clean_bpc_by_width"]
        pairs = {}
        for w, uni in ((192, "D192"), (128, "D128")):
            if f"{w}" in curve and (uni, 0) in recs:
                d = round(curve[str(w)] - recs[(uni, 0)]["eval"]["clean_bpc"], 4)
                pairs[f"{uni}_vs_WD256@{w}"] = {
                    "delta_bpc": d,
                    "verdict": ("不可排序 ⇒ 一训多档不额外付代价（弹性宽度成立）"
                                if abs(d) < VARIANCE_BOUND else
                                ("WD256 更差 ≥ 带 ⇒ 弹性宽度付额外代价" if d > 0
                                 else "WD256 更好 ≥ 带"))}
        if "256" in curve and "U4-s0(anchor)" in frontier:
            pairs["U4_vs_WD256@256"] = {"delta_bpc": round(
                curve["256"] - frontier["U4-s0(anchor)"]["clean_bpc"], 4),
                "note": "描述列（非判据）：宽度 dropout 作为正则对全宽档的影响"}
        mono = [curve[str(w)] for w in (128, 192, 256)] if all(
            str(w) in curve for w in (128, 192, 256)) else None
        p3 = {"clean_bpc_by_width": curve,
              "monotone_wider_is_better": bool(mono and mono[0] > mono[1] > mono[2]),
              "adjacent_gaps": {"128_vs_192": (None if not mono else round(mono[0] - mono[1], 4)),
                                "192_vs_256": (None if not mono else round(mono[1] - mono[2], 4))},
              "gaps_within_band": {k: bool(v is not None and abs(v) < VARIANCE_BOUND)
                                   for k, v in {"128_vs_192": (None if not mono else mono[0] - mono[1]),
                                                "192_vs_256": (None if not mono else mono[1] - mono[2])}.items()},
              "vs_same_width_uniform": pairs,
              "same_checkpoint_evidence": {
                  "n_checkpoints": 1, "model_pt_sha256": recs[("WD256", 0)]["model_pt_sha256"],
                  "slices": [{"width": w, "params_total": params("WD256", 0, w),
                              "macs_per_token": macs("WD256", 0, w)} for w in (256, 192, 128)],
                  "note": "三档来自同一 model.pt 的前缀子矩阵（build_sliced_model），不是分别训练"},
              "gate_verdict": {
                  "monotone": ("成立" if (mono and mono[0] > mono[1] > mono[2]) else "不成立"),
                  "elastic_vs_uniform": (
                      "弹性宽度成立（一训多档与分别训练的同宽 uniform 不可排序）"
                      if all(v["verdict"].startswith("不可排序")
                             for k, v in pairs.items()
                             if k.split("_vs_")[0] in ("D192", "D128"))
                      else "不成立（至少一档越 0.014 带）")}}

    # ---- P4（必须不变/活性）----
    def reeval_rec(a, s):
        p = os.path.join(HERE, f"reeval_{a}_s{s}.json")
        return json.load(open(p)) if os.path.exists(p) else None

    p4 = {"eval_determinism": {}, "causality": {}, "liveness": {}, "param_audit": {},
          "active_width_audit": {}}
    for (a, s), r in sorted(recs.items()):
        rr = reeval_rec(a, s)
        p4["eval_determinism"][f"{a}-s{s}"] = (
            {"reeval_file_missing": True,
             "note": "--no-train 重评尚未跑（P4-① 证据缺）"} if rr is None else
            {k: v for k, v in rr.items() if k not in ("prev_file",)})
        if r["kind"] == "width_dropout":
            p4["causality"][f"{a}-s{s}"] = {k: r["per_width"][k]["causality"]["pass"]
                                            for k in r["per_width"]}
            p4["liveness"][f"{a}-s{s}"] = {k: {"min_block_out_over_in":
                                               r["per_width"][k]["liveness"]["min_ratio"],
                                               "hidden_by_layer":
                                               r["per_width"][k]["liveness"]["hidden_by_layer"]}
                                           for k in r["per_width"]}
            p4["param_audit"][f"{a}-s{s}"] = {k: {
                "measured": r["per_width"][k]["params"]["params_total_measured"],
                "analytic": r["per_width"][k]["params"]["analytic"]["params_total"],
                "matches": r["per_width"][k]["params"]["params_total_matches_analytic"]}
                for k in r["per_width"]}
            p4["active_width_audit"][f"{a}-s{s}"] = {k: {
                "target_width": int(k), "d_model": r["per_width"][k]["width"],
                "hidden_by_layer": r["per_width"][k]["params"]["hidden_by_layer"],
                "no_full_width_tensor": r["per_width"][k]["params"]["no_full_width_tensor"],
                "active_dim_matches_target":
                    r["per_width"][k]["params"]["active_dim_matches_target"]}
                for k in r["per_width"]}
        else:
            p4["causality"][f"{a}-s{s}"] = r["causality"]["pass"]
            p4["liveness"][f"{a}-s{s}"] = {"min_block_out_over_in":
                                           r["liveness"]["min_ratio"],
                                           "hidden_by_layer": r["liveness"]["hidden_by_layer"]}
            p4["param_audit"][f"{a}-s{s}"] = {
                "measured": r["params"]["params_total_measured"],
                "analytic": r["params"]["analytic"]["params_total"],
                "matches": r["params"]["params_total_matches_analytic"],
                "ffn_matches": r["params"]["params_ffn_matches_analytic"]}
            p4["active_width_audit"][f"{a}-s{s}"] = {
                "target_width": r["width"], "d_model": r["width"],
                "hidden_by_layer": r["params"]["hidden_by_layer"],
                "active_dim_matches_target": r["params"]["hidden_matches_width"]}
    p4["equivalence_gates"] = {str(s): bool(recs[("WD256", s)]["all_gates_bitwise_equal"])
                               for s in seeds if ("WD256", s) in recs}
    p4["equivalence_gates_all_pass"] = bool(
        p4["equivalence_gates"] and all(p4["equivalence_gates"].values()))

    # ---- P5（否证线）----
    p5 = {
        "arms_run": sorted({a for (a, _s) in recs}),
        "arms_dropped": {
            "D128": ("GPU 锁排队超预算：按 README §5「预算与显存」里写的砍臂次序"
                      "（`README.md:122`；决策点见 `run_log.txt` 中带 `接手者调度调整` 的几行）"
                      "正式臂缩为 D192:0 → WD256:0 ⇒ 本项无 D128 读数；"
                      "下一步最小实验 = D128:0 一臂 12k（98.3M token 匹配步数）+ 同口径评测，"
                      "读数可直接并入本表 P1/P2（排程代价见 ticket 64 §10）")},
    }
    run_narrow = [a for a in ("D192", "D128") if a in p1]
    if run_narrow:
        in_band = all(abs(p1[a]["delta_bpc_vs_U4"]) < VARIANCE_BOUND for a in run_narrow)
        names = "/".join(run_narrow)
        p5["a_noise_band"] = (
            f"不成立（否证线被触发）：{names} 的 Δbpc 越 ±0.014 带 ⇒ 减宽代价可分，"
            f"见 P5b" if not in_band else
            f"成立：{names} 的 Δbpc 均落在 ±0.014 带内 ⇒ 在本尺度质量代价落在噪声带内，减宽该不跌")
        slopes = {a: p1[a]["bpc_cost_per_1pct_macs_saved"] for a in run_narrow}
        s_ffn = (savings.get("U2") or {}).get("bpc_cost_per_1pct_macs_saved")
        s_ffn_delta = (savings.get("U2") or {}).get("delta_bpc")
        if not in_band and s_ffn is not None:
            delta_bpc_gap = (max(abs(p1[a]["delta_bpc_vs_U4"] - s_ffn_delta)
                                 for a in run_narrow)
                             if s_ffn_delta is not None else None)
            p5["b_frontier_advantage"] = p5b_frontier_verdict(
                slopes, s_ffn, in_band=False,
                delta_bpc_gap=delta_bpc_gap, bound=VARIANCE_BOUND)
        else:
            p5["b_frontier_advantage"] = ("未触发（P5a 已成立 / U2 斜率缺失）"
                                          if in_band else "U2 斜率缺失，无法比较")
        p5["c_wd256_elasticity"] = (
            (p3 or {}).get("gate_verdict", {}).get("elastic_vs_uniform", "WD256 结果缺失")
            if p3 else "WD256 结果缺失")
    else:
        p5["a_noise_band"] = "无减宽臂读数（D192/D128 均未跑）—— 否证线不可判，不得据此下结论"

    out = {
        "experiment": "2642-uniform-narrowing-13m",
        "grid": "3.1.1-uniform-narrowing",
        "criterion": ("P1 减宽 Δbpc vs U4（带 0.014）；P2 前沿对照（d_model 旋钮 vs FFN 旋钮，"
                      "等算力位点，参数量与 MACs 并列）；P3 WD256 同 checkpoint 三档质量曲线 + "
                      "与同宽 uniform 比较；P4 评测确定性/因果性/活性/参数与维数审计；"
                      "P5 否证线"),
        "variance_bound": VARIANCE_BOUND,
        "variance_bound_declaration": ("docs/research-schema.md §12.1 / C1：0.014 bpc 带，"
                                       "限配对 Δ、≥5 窗口；率类指标才走 bootstrap/Wilson"),
        "steps": steps,
        "seeds_available": {a: [s for (aa, s) in sorted(recs) if aa == a] for a in arms},
        "history_same_config_dense_13m_12k_spread": {
            "values": HIST_DENSE_SPREAD, "range": 0.0057,
            "note": "参照（非预登记门槛）；0.014 带来自 methodology §2 的跨 run 漂移界"},
        "anchors_reused": anchors,
        "anchor_config_parity": anchor_config_parity(steps, batch_size, lr),
        "frontier": frontier,
        "P1_uniform_narrowing_vs_U4": p1,
        "P2_frontier_vs_FFN_knob": p2,
        "P3_WD256_elastic_width": p3,
        "P4_invariants_and_liveness": p4,
        "P5_falsification": p5,
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)]
        + [v["source"] for v in anchors.values()],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("P1_uniform_narrowing_vs_U4",
                                          "P3_WD256_elastic_width",
                                          "P5_falsification")}, indent=2,
                     ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用 runs/<arm>-s<seed>-12k/model.pt 重评（写 reeval_*.json）")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default="D192,D128,WD256")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.steps, args.out_root,
                  [a for a in args.arms.split(",") if a],
                  [int(s) for s in args.seeds.split(",") if s.strip() != ""],
                  batch_size=args.batch_size, lr=args.lr)
        return
    if not args.arm:
        raise SystemExit("--arm 或 --aggregate 必给其一")
    if args.no_train:
        reeval(args.arm, args.seed, args.steps, args.out_root)
        return
    os.makedirs(args.out_root, exist_ok=True)
    info = train_arm(args.arm, args.seed, args.steps, args.batch_size, args.lr,
                     args.out_root)
    rd = run_dir(args.arm, args.seed, args.steps, args.out_root)
    with open(os.path.join(rd, "train_wall.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    evaluate_arm(args.arm, args.seed, args.steps, args.out_root)


if __name__ == "__main__":
    main()
