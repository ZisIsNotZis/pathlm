"""2646 · 逐层宽度 schedule（金字塔 / 沙漏 / 倒金字塔）· 13M · dense · 前缀子空间切片.

契约 = 同目录 `README.md`（臂 / 判据 P1–P4 / 否证线 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2）：

* **引擎无逐层 `d_model`**（`Block` 的宽度由 `cfg.d_model` 全局给定）⇒ 干预在
  实验目录内自造，自造方式 = **前缀子空间切片**：残差流恒为 256 宽，第 l 层只在
  前 `w_l` 个通道上运算（该层的 LN / attention qkv-proj / FFN 权重取**前缀子矩阵**，
  FFN 隐藏 = `4·w_l`），结果写回 `x[..., :w_l]`，其余通道原样通过。
* **复用共享 `Block.forward`**：`PrefixSubspaceBlock` 把前缀 `x[..., :w_l]` 交给
  共享实现（块内部 `d = w_l`、头数 `= w_l/32`），只在本层外把未触及的通道拼回。
  ⇒ 前缀路径与 stock 路径是**同一段数值代码**，`w_l = d` 时逐位相同（等值门 T1）。
* **臂间唯一的差别** = 每层宽度的**分配**：所有臂先按同一 seed 建**同一份全宽
  stock 模型**，再按各臂 schedule 取**同一份抽样的前缀子矩阵** ⇒ 不是「两个不同
  初始化的模型」在比（同 2641 的做法）。
* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成工厂（stock 构造 + 按 schedule 换块），然后调用 `train_depth_ar.main()` ——
  参数解析、配置装配、`train()` 循环、checkpoint、`results.json` 全是共享实现，
  `pathlm/**`、`train_*.py` 零改动。

用法：

    python3 run.py --arm PYR --seed 0 --steps 12000               # 训练 + 评测
    python3 run.py --arm PYR --seed 0 --steps 12000 --no-train    # 仅重评（确定性核对）
    python3 run.py --aggregate --arms EQ,PYR,HG,APYR --seeds 0    # 汇总 P1–P4
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

import train_depth_ar as TDA                                          # noqa: E402
from pathlm.config import ModelConfig                                 # noqa: E402
from pathlm.data import batch, load_enwik8_full                       # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve   # noqa: E402
from pathlm.model import Block                                        # noqa: E402

DEVICE = "cuda"

# ---------------------------------------------------------------- 常量（契约 §2/§3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
HEAD_DIM = 32
VOCAB = 206
VARIANCE_BOUND = 0.014                    # bpc 带（harness/research-schema.md §12.1 / C1）
EVAL_GENERATOR_SEED = 2                   # = depth_curve 缺省（与 2641 同口径）
LIVENESS_SEED = 1234

EQ_WIDTH = 192                            # 等宽臂的宽度（Σw² = 8·192² = 294,912）
# PYR = 非增 schedule 中与契约示例 [256,224,224,192,192,160,160,128] 的 L1 距离最小者
# （唯一解，与示例仅第 8 层不同：128 → 64）；搜索与全部候选见 `schedule_search.json`
# / `search_schedule.py`。
PYR_SCHEDULE = [256, 224, 224, 192, 192, 160, 160, 64]
EQ_SCHEDULE = [EQ_WIDTH] * N_LAYERS

ANCHOR_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "3.1.4-ffn-ratio", "experiments",
    "2641-ffn-ratio-frontier-13m"))


def deal_ascending_to_ends(widths: list[int]) -> list[int]:
    """把宽度多重集按升序交替发到两端（外→内）⇒ 单峰（中段宽、两端窄）。

    `HG = deal_ascending_to_ends(PYR_SCHEDULE)`：**多重集不变** ⇒ Σw² 与 Σw 与
    PYR 逐位相同，臂间唯一差别是「哪个宽度放在哪一层」。（非均匀的严格左右对称
    沙漏在 32 倍数的格点上无 Σw² = 294,912 解 —— 唯一的对称单峰解就是等宽本身，
    见 `schedule_search.json` 的 `symmetric_unimodal_candidates`。）
    """
    n = len(widths)
    positions, lo, hi = [], 0, n - 1
    while lo <= hi:
        positions.append(lo)
        lo += 1
        if lo <= hi:
            positions.append(hi)
            hi -= 1
    out: list[int | None] = [None] * n
    for w, pos in zip(sorted(widths), positions):
        out[pos] = w
    return [int(w) for w in out]


SCHEDULES = {
    "EQ":   EQ_SCHEDULE,                                    # 等宽 192（对照）
    "PYR":  PYR_SCHEDULE,                                   # 金字塔（前宽后窄）
    "HG":   deal_ascending_to_ends(PYR_SCHEDULE),           # 沙漏（中段宽两端窄）
    "APYR": list(reversed(PYR_SCHEDULE)),                   # 倒金字塔（= PYR 镜像）
}


def sum_w2(schedule: list[int]) -> int:
    return sum(w * w for w in schedule)


# ---------------------------------------------------------------- 代价与参数（等效理论量）

def analytic_cost(schedule: list[int], mlp_mult: int = MLP_MULT, T: int = SEQ_LEN,
                  vocab: int = VOCAB, readout_width: int = D_MODEL) -> dict:
    """每 token 等效 MACs（解析，**不含任何时间量**）。

    第 l 层：attention `4·w_l² + 2·T·w_l`（qkv `3w²` + proj `w²` + QKᵀ `T·w` + AV
    `T·w`）、FFN `2·(4·w_l)·w_l = 8·w_l²`。读出恒在 **256 宽残差**上做（残差流恒
    256 宽 ⇒ 读出 = `d·V` with d = 256，与等宽臂/锚同口径）。
    """
    L = len(schedule)
    attn = sum(4 * w * w + 2 * T * w for w in schedule)
    ffn = sum(8 * w * w for w in schedule)
    readout = readout_width * vocab
    total = attn + ffn + readout
    return {
        "schedule": list(schedule), "mlp_mult": mlp_mult, "seq_len": T,
        "n_layers": L, "vocab": vocab, "readout_width": readout_width,
        "sum_w2": sum_w2(schedule), "sum_w": sum(schedule),
        "macs_per_token": {"attention": attn, "ffn": ffn,
                           "readout_tied": readout, "total": total},
        "macs_attn_per_layer": [4 * w * w + 2 * T * w for w in schedule],
        "macs_ffn_per_layer": [8 * w * w for w in schedule],
        "note": ("等效理论量：MACs/token（乘加）。attention 的 T·w 项随序列长度线性、"
                 "FFN 的 w² 项与 T 无关；不用任何时间量。tokens-per-forward = 1。"),
    }


def analytic_params(schedule: list[int], vocab: int = VOCAB,
                    T: int = SEQ_LEN, mlp_mult: int = MLP_MULT) -> dict:
    """解析参数量（分列）。残差流恒 256 宽 ⇒ embedding/位置嵌入/读出（tied）与
    carry 门都是 **256 宽**、逐臂相同；只有各层块参数随 `w_l` 变。

    `gates_dead_in_dense` = 引擎在 dense 形态下照建、但 dense 前向**不用**的门参数
    （`cfg.carry_gate=True` 是 dense 的强制项）——与 2641 的 `params_total` 同口径。
    """
    L = len(schedule)
    embed = vocab * D_MODEL
    pos = T * D_MODEL
    blocks = sum(12 * w * w + 13 * w for w in schedule)   # 见下 note
    gates = L * (D_MODEL * D_MODEL + D_MODEL)
    total = embed + pos + blocks + gates
    return {
        "schedule": list(schedule), "sum_w2": sum_w2(schedule),
        "sum_w": sum(schedule),
        "embedding": embed, "pos_embedding": pos,
        "embedding_total": embed + pos, "blocks": blocks,
        "gates_dead_in_dense": gates, "params_total": total,
        "params_live": total - gates,
        "params_live_rel_to_EQ": None,        # 由 aggregate 逐臂填
        "note": ("解析参数量。每层 = qkv(3w²+3w) + proj(w²+w) + FFN(8w²+5w)"
                 " + LN(2·2w) = 12w² + 13w；embedding/位置嵌入/tied 读出/carry 门"
                 "都是 256 宽、逐臂相同。gates_dead_in_dense 引擎照建但 dense 前向"
                 "不用（与 2641 params_total 同口径）。"),
    }


# ---------------------------------------------------------------- 干预：前缀子空间切片

class PrefixSubspaceBlock(Block):
    """第 l 层只在残差流的**前 `width` 个通道**上运算（前缀子空间切片）。

    前向 = 取前缀 `x[..., :width]` → 交给共享 `Block.forward`（块内部 `d = width`、
    头数 `width/head_dim`）→ 把未触及的通道原样拼回 `x[..., width:]`。
    ⇒ 权重是 w 级的前缀子矩阵（真算少、不是掩码），每 token MACs 按 w 计。

    `width == x.shape[-1]`（= 256）时 `x[..., :width]` 与 `x` 同形状同连续布局，
    共享前向逐位还原 stock，拼回的尾段为空 ⇒ **逐位等于 stock `Block`**（等值门 T1；
    由「同一份权重的 state_dict 逐位相同 + 同一段共享前向代码」保证，测试实测）。
    """

    def __init__(self, cfg: ModelConfig, width: int):
        super().__init__(cfg)
        if cfg.d_model != int(width):
            raise ValueError(f"cfg.d_model={cfg.d_model} != width={width}")
        if int(width) % HEAD_DIM:
            raise ValueError(f"宽度 {width} 必须整除 head_dim {HEAD_DIM}")
        self.width = int(width)

    def forward(self, x, attn_mask=None, dist_pen=0.0, kv_from=None,
                gate_vec=None):
        # 本实验是 dense + sequential 单因素形态：另三条路径（掩码/旁路 KV/携带门）
        # 的 KV 与门的形状都按全宽 d 定义，前缀切片后在它们上面没有定义——**显式
        # 报错**而不是静默按错形状跑。
        if attn_mask is not None or dist_pen > 0 or kv_from is not None \
                or gate_vec is not None:
            raise ValueError(
                "PrefixSubspaceBlock 只实装 dense + sequential 单因素路径"
                f"（收到 attn_mask={attn_mask is not None}, dist_pen={dist_pen}, "
                f"kv_from={kv_from is not None}, gate_vec={gate_vec is not None}）")
        y, dist = super().forward(x[..., :self.width])
        return torch.cat([y, x[..., self.width:]], dim=-1), dist


def prefix_block_from_stock(stock_blk: Block, width: int) -> PrefixSubspaceBlock:
    """按**前缀子矩阵**从全宽 stock 块造一个宽度 `width` 的前缀块。

    qkv 的前缀 = `q|k|v` **每块各自取前 width 行**（行序 = `chunk(3)` 的块序）
    + 全体取前 width 列，拼成 `[3·width, width]` —— 不是输出行的连续前缀
    `[:3·width]`（后者是掩码式错切，单测 T3 钉住必须抓红）。
    """
    d = D_MODEL
    m = MLP_MULT * width
    mcfg = ModelConfig(d_model=width, n_layers=N_LAYERS, n_heads=width // HEAD_DIM,
                       seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dropout=0.0,
                       norm_cap=1.0)
    blk = PrefixSubspaceBlock(mcfg, width)
    with torch.no_grad():
        blk.ln1.weight.copy_(stock_blk.ln1.weight[:width])
        blk.ln1.bias.copy_(stock_blk.ln1.bias[:width])
        blk.ln2.weight.copy_(stock_blk.ln2.weight[:width])
        blk.ln2.bias.copy_(stock_blk.ln2.bias[:width])
        blk.qkv.weight.copy_(torch.cat(
            [stock_blk.qkv.weight[0:width, :width],
             stock_blk.qkv.weight[d:d + width, :width],
             stock_blk.qkv.weight[2 * d:2 * d + width, :width]], dim=0))
        blk.qkv.bias.copy_(torch.cat(
            [stock_blk.qkv.bias[0:width], stock_blk.qkv.bias[d:d + width],
             stock_blk.qkv.bias[2 * d:2 * d + width]], dim=0))
        blk.proj.weight.copy_(stock_blk.proj.weight[:width, :width])
        blk.proj.bias.copy_(stock_blk.proj.bias[:width])
        blk.mlp[0].weight.copy_(stock_blk.mlp[0].weight[:m, :width])
        blk.mlp[0].bias.copy_(stock_blk.mlp[0].bias[:m])
        blk.mlp[2].weight.copy_(stock_blk.mlp[2].weight[:width, :m])
        blk.mlp[2].bias.copy_(stock_blk.mlp[2].bias[:width])
    return blk


def apply_width_schedule(model: DepthARModel, schedule: list[int]) -> DepthARModel:
    """把 `model.blocks` 逐层换成宽度 `schedule[l]` 的前缀子空间块。

    建块的 `nn.Linear` 构造会消耗全局随机数 ⇒ 前后存/还原 RNG 状态，保证
    「同一 seed ⇒ 同一份全宽抽样」在所有臂间严格成立（块权重全部由 stock 前缀
    覆盖，建块自身的 init 不进模型）。
    """
    if len(schedule) != len(model.blocks):
        raise ValueError(f"schedule 长度 {len(schedule)} != 层数 {len(model.blocks)}")
    rng = torch.get_rng_state()
    try:
        new_blocks = [prefix_block_from_stock(blk, w)
                      for blk, w in zip(model.blocks, schedule)]
    finally:
        torch.set_rng_state(rng)
    for l, blk in enumerate(new_blocks):
        model.blocks[l] = blk
    return model


def stock_cfg() -> DepthARConfig:
    """与 2641 U4-s0 逐字段相同的 dense 轴配置。"""
    return DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                         seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True,
                         topology="sequential", fill_kv=True)


def build_model(schedule: list[int], vocab_size: int, seed: int) -> DepthARModel:
    """所有臂共用的构造路径：同 seed ⇒ 同一次全宽抽样，再取各自的 schedule 切片。"""
    torch.manual_seed(int(seed))
    return apply_width_schedule(DepthARModel(stock_cfg(), vocab_size), schedule)


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory(schedule: list[int]):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 main() 调它建模型。"""
    def factory(cfg, vocab_size):
        return apply_width_schedule(DepthARModel(cfg, vocab_size), schedule)
    return factory


def _kv_head_liveness_prefix_safe(model: DepthARModel, vocab_size: int,
                                  seq_len: int) -> dict:
    """共享 `_kv_head_liveness` 的逐层前缀宽安全替换。

    共享探针断言「每层 `qkv` 输出宽 == 3·d（d = cfg.d_model = 256）」并按单一全局
    d 推 K/V 头数（`pathlm/depth_ar.py:684` 起）——**逐层宽度**下第 l 层的 qkv 输出
    宽是 `3·w_l`，该断言必然失败（冒烟 rc=1）。本替换在**一次真实前向**上抓每层
    `qkv` 的**实际输出形状**（形状类活性量的硬证据），并把目标宽度一并列出。
    探针只读形状、不参与训练/评测数值路径，也不入判据；`pathlm/**`、`train_*.py`
    零改动。
    """
    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, vocab_size, (2, min(seq_len, 64)), generator=g)
    x = x.cuda() if torch.cuda.is_available() else x
    seen: dict[int, tuple] = {}
    handles = [blk.qkv.register_forward_hook(
        (lambda k: lambda mod, inp, out: seen.__setitem__(k, tuple(out.shape)))(k))
        for k, blk in enumerate(model.blocks)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    active = [int(blk.width) for blk in model.blocks]
    qkv_out = [int(seen[k][-1]) for k in sorted(seen)]
    out = {
        "probe": "per_layer_prefix_width（替换 train_depth_ar._kv_head_liveness）",
        "replaced_probe": "kv_head_liveness（其断言 qkv 输出宽 == 3·cfg.d_model 按单一"
                          "全局宽度定义 ⇒ 逐层宽度下不适用）",
        "n_layers": len(model.blocks),
        "active_width_by_layer": active,
        "qkv_out_width_by_layer": qkv_out,
        "expected_qkv_out_width_by_layer": [3 * w for w in active],
        "all_layers_hooked": len(seen) == len(model.blocks),
        "qkv_out_matches_expected": qkv_out == [3 * w for w in active],
        "probe_tokens_shape": list(x.shape), "probe_seed": 1234,
        "note": "形状类活性量：在一次真实前向里抓每层 qkv 的**实际输出宽度**，"
                "与目标 3·w_l 逐位对照 ⇒ 逐层宽度真的到了模型，不是只改了标签",
    }
    if not (out["all_layers_hooked"] and out["qkv_out_matches_expected"]):
        raise AssertionError(f"逐层宽度活性证据不成立：{out}")
    return out


def shared_argv(arm: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态）。

    **所有臂的 argv 逐位相同**：干预不在配置里（引擎不支持逐层 d_model），而在
    `_factory` 换块那一处；唯一的变量就是 schedule（契约 §2 单因素声明）。
    """
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-s{seed}-{steps // 1000}k",
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
    orig_model = TDA.DepthARModel
    orig_kv = TDA._kv_head_liveness
    TDA.DepthARModel = _factory(SCHEDULES[arm])
    TDA._kv_head_liveness = _kv_head_liveness_prefix_safe
    argv_backup = list(sys.argv)
    t0 = time.time()
    try:
        sys.argv = shared_argv(arm, seed, steps, batch_size, lr, out_root)
        TDA.main()                       # 共享实现：解析 / 配置 / train() / 落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig_model
        TDA._kv_head_liveness = orig_kv
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


# ---------------------------------------------------------------- 评测（本实验自持）

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_sd(rd: str) -> dict:
    return torch.load(os.path.join(rd, "model.pt"), map_location="cpu")


def eval_model(schedule: list[int], seed: int, rd: str,
               vocab: int) -> DepthARModel:
    model = build_model(schedule, vocab, seed).to(DEVICE)
    model.load_state_dict(_load_sd(rd))
    model.eval()
    return model


@torch.no_grad()
def prefix_liveness(model: DepthARModel, eval_arr, n_batches: int = 2,
                    batch_size: int = 8, seed: int = LIVENESS_SEED) -> dict:
    """逐层活性与**实测活跃维**（`harness/SCHEMA.md` §10）。

    在**真实前向**里抓：① 每层块输入/输出的逐位置范数比（>0 ⇒ 该层真的参与前向）；
    ② 每层 `ln1` 的**实际输入宽度** —— `F.layer_norm(x, (w,), …)` 在形状上要求
    最后一维恰为 w ⇒ 这个宽度是「前缀真的只切了 w 列」的硬证据（不是标签）；
    ③ 每层 qkv 实际输出宽（应为 3w）与 FFN 隐藏维（应为 4w）。
    """
    gen = torch.Generator().manual_seed(seed)
    d = model.cfg.d_model
    ratio, ln_in, qkv_out = [[] for _ in model.blocks], {}, {}
    handles = []

    def ratio_hook(l):
        def fn(_mod, inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            x_in = inp[0][..., :int(model.blocks[l].width)].detach().float()
            y = out_t[:, :, :int(model.blocks[l].width)].detach().float()
            ratio[l].append(float((y.norm(dim=-1)
                                   / x_in.norm(dim=-1).clamp_min(1e-6)).mean()))
        return fn

    for l, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(ratio_hook(l)))
        handles.append(blk.ln1.register_forward_hook(
            (lambda k: lambda mod, inp, out: ln_in.__setitem__(k, int(inp[0].shape[-1])))(l)))
        handles.append(blk.qkv.register_forward_hook(
            (lambda k: lambda mod, inp, out: qkv_out.__setitem__(k, int(out.shape[-1])))(l)))
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
    measured = [ln_in[l] for l in sorted(ln_in)]
    target = [int(blk.width) for blk in model.blocks]
    return {
        "residual_width": d,
        "target_width_by_layer": target,
        "measured_ln_input_width_by_layer": measured,
        "measured_active_width_matches_target": measured == target,
        "qkv_out_width_by_layer": [qkv_out[l] for l in sorted(qkv_out)],
        "qkv_out_matches_3x_target": [qkv_out[l] for l in sorted(qkv_out)]
                                     == [3 * w for w in target],
        "hidden_by_layer": [int(blk.mlp[0].weight.shape[0]) for blk in model.blocks],
        "hidden_matches_4x_target": [int(blk.mlp[0].weight.shape[0])
                                     for blk in model.blocks]
                                    == [MLP_MULT * w for w in target],
        "block_out_over_in_by_layer": [round(statistics.mean(v), 6)
                                       for v in ratio],
        "min_ratio": round(min(statistics.mean(v) for v in ratio), 6),
        "note": ("逐层 ‖block_out‖/‖block_in‖（只在前缀通道上取范数）的逐位置均值，"
                 ">0 ⇒ 该层真的参与前向；measured_ln_input_width 是 LayerNorm 的"
                 "实际输入宽度（形状约束 ⇒ 前缀列数的硬证据）"),
    }


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


def _equivalence_gate_zero_change(schedule: list[int], eval_arr, seed: int,
                                  batch_size: int = 2) -> dict:
    """等值门 T1（训练前）：全层 `w_l = 256` 时，前缀模型前向与 stock 引擎前向逐位相同。

    两条证据：① 前缀模型的 state_dict 与同 seed stock 模型**逐位相同**；② 同批输入
    下 logits 与 dense 损失**逐位相同**（fp32 与 bf16 autocast 两种模式）。
    """
    gen = torch.Generator().manual_seed(LIVENESS_SEED)
    x, _ = batch(eval_arr, batch_size, model_seq_len(), gen)
    x = x.to(DEVICE)
    out = {}
    torch.manual_seed(seed)
    stock = DepthARModel(stock_cfg(), VOCAB).to(DEVICE).eval()
    prefix = build_model(schedule, VOCAB, seed).to(DEVICE).eval()
    same_sd = stock.state_dict().keys() == prefix.state_dict().keys() and all(
        torch.equal(prefix.state_dict()[k], stock.state_dict()[k])
        for k in stock.state_dict())
    out["state_dict_bitwise_equal"] = bool(same_sd)
    for mode in ("fp32", "bf16_autocast"):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(mode == "bf16_autocast")):
                lp = prefix(x, exit_depths=None)
                ls = stock(x, exit_depths=None)
        out[mode] = {
            "bitwise_equal_logits": bool(torch.equal(
                lp[1]["depth_logits"][-1], ls[1]["depth_logits"][-1])),
            "bitwise_equal_loss": bool(torch.equal(lp[0], ls[0])),
            "max_abs_logit_diff": float(
                (lp[1]["depth_logits"][-1].float()
                 - ls[1]["depth_logits"][-1].float()).abs().max()),
        }
    return out


def model_seq_len() -> int:
    return SEQ_LEN


def evaluate(arm: str, seed: int, steps: int, out_root: str,
             n_batches: int = 20) -> dict:
    """评测 + 参数/活性/因果性审计 + 等值门 → `results_<arm>_s<seed>.json`。"""
    schedule = SCHEDULES[arm]
    rd = run_dir(arm, seed, steps, out_root)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    model = eval_model(schedule, seed, rd, vocab)
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    measured_total = sum(p.numel() for p in model.parameters())
    measured_blocks = sum(p.numel() for blk in model.blocks
                          for p in blk.parameters())
    measured_gates = sum(p.numel() for p in model.gates.parameters())
    measured_ffn = sum(p.numel() for blk in model.blocks for p in blk.mlp.parameters())
    ap, ac = analytic_params(schedule, vocab=vocab), analytic_cost(schedule, vocab=vocab)
    live = prefix_liveness(model, eval_arr)
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "schedule": list(schedule), "sum_w2": sum_w2(schedule),
        "sum_w": sum(schedule),
        "residual_width": D_MODEL,
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": EVAL_GENERATOR_SEED,
                 "bpc_by_depth": curve, "clean_bpc": curve[-1]["bpc"],
                 "acc_by_depth": [{"depth": r["depth"], "acc": r["acc"]} for r in curve]},
        "params": {"params_total_measured": measured_total,
                   "params_blocks_measured": measured_blocks,
                   "params_ffn_measured": measured_ffn,
                   "params_gates_dead_measured": measured_gates,
                   "analytic": ap,
                   "params_total_matches_analytic": measured_total == ap["params_total"],
                   "params_blocks_matches_analytic": measured_blocks == ap["blocks"],
                   "params_live": measured_total - measured_gates},
        "cost": ac,
        "liveness": live,
        "causality": causality_check(model, eval_arr),
        "equivalence_gate_w_eq_d": _equivalence_gate_zero_change(
            [D_MODEL] * N_LAYERS, eval_arr, seed),
        "shared_train_results": _shared_summary(rd),
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "model_pt_sha256": sha256_file(os.path.join(rd, "model.pt")),
        "no_engine_change": ("pathlm/**、train_*.py 零改动：前缀子空间块实现在本实验"
                             "目录内，复用共享 `Block.forward`；等值性由 T1 逐位钉住"),
    }
    json.dump(out, open(os.path.join(HERE, f"results_{arm}_s{seed}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[2646] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"sum_w2={out['sum_w2']} macs={ac['macs_per_token']['total']} "
          f"params={measured_total} schedule={schedule}", flush=True)
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
            "gnorm_monitor_max": (shared.get("gnorm_monitor") or {}).get("gnorm_max"),
            "kv_head_liveness": shared.get("kv_head_liveness")}


# ---------------------------------------------------------------- 重评（评测确定性）

def reeval(arm: str, seed: int, steps: int, out_root: str) -> dict:
    """`--no-train`：只重评，把与首评的**逐位一致性**落进 `reeval_<arm>_s<seed>.json`。"""
    prev_path = os.path.join(HERE, f"results_{arm}_s{seed}.json")
    prev = json.load(open(prev_path)) if os.path.exists(prev_path) else None
    now = evaluate(arm, seed, steps, out_root)
    pv = prev["eval"] if prev else None
    out = {"arm": arm, "seed": seed, "steps": steps,
           "prev_clean_bpc": pv["clean_bpc"] if pv else None,
           "now_clean_bpc": now["eval"]["clean_bpc"],
           "bitwise_equal": bool(pv and pv["clean_bpc"] == now["eval"]["clean_bpc"]),
           "bpc_by_depth_bitwise_equal": bool(
               pv and pv["bpc_by_depth"] == now["eval"]["bpc_by_depth"]),
           "prev_file": os.path.relpath(prev_path, HERE)}
    json.dump(out, open(os.path.join(HERE, f"reeval_{arm}_s{seed}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[2646 reeval] {arm}-s{seed} {json.dumps(out, ensure_ascii=False)}",
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
    """逐字段配置对照（复用纪律 `harness/SCHEMA.md` §9 三道具之二）。"""
    cfg = json.load(open(os.path.join(ANCHOR_DIR, "config.json")))
    base, tr = cfg["base"], cfg["training"]
    fields = {"d_model": [base["d_model"], D_MODEL],
              "n_layers": [base["n_layers"], N_LAYERS],
              "n_heads": [base["n_heads"], N_HEADS],
              "seq_len": [base["seq_len"], SEQ_LEN],
              "mlp_mult": [base["mlp_mult"], MLP_MULT],
              "dense": [base["dense"], True],
              "dropout": [base["dropout"], 0.0],
              "norm_cap": [base["norm_cap"], 1.0],
              "topology": [base["topology"], "sequential"],
              "exit_mode": [base["exit_mode"], "uniform"],
              "fill_kv": [base["fill_kv"], True],
              "steps": [tr["steps"], steps],
              "batch_size": [tr["batch_size"], batch_size], "lr": [tr["lr"], lr]}
    diff = {k: v for k, v in fields.items() if v[0] != v[1]}
    return {"anchor_config": os.path.relpath(
                os.path.join(ANCHOR_DIR, "config.json"), ROOT),
            "fields_compared": fields, "mismatched_fields": diff,
            "all_equal": not diff,
            "note": ("本实验**所有臂**的轴配置与 2641 U4-s0 逐字段相同（残差流恒 256 宽、"
                     "头数 8、mlp_mult 4）；干预在块的前缀切片，不在配置里。U4 只作"
                     "阈值/前沿位点引用，不替代 EQ 这一同预算对照")}


# ---------------------------------------------------------------- 汇总（P1–P4）

def band_verdict(delta: float | None) -> dict:
    """0.014 bpc 带（`harness/research-schema.md` §12.1 / C1）判「可排序 / 不可排序」。"""
    if delta is None:
        return {"delta": None, "sorted": None, "verdict": "缺读数"}
    d = round(delta, 4)
    if d >= VARIANCE_BOUND:
        return {"delta": d, "sorted": True,
                "verdict": f"可分：{d:+.4f} bpc（更差，越 {VARIANCE_BOUND} 带）"}
    if d <= -VARIANCE_BOUND:
        return {"delta": d, "sorted": True,
                "verdict": f"可分：{d:+.4f} bpc（更好，越 {VARIANCE_BOUND} 带）"}
    return {"delta": d, "sorted": False,
            "verdict": f"不可排序：{d:+.4f} bpc 落在 ±{VARIANCE_BOUND} 带内"}


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
    u4 = anchors.get("U4", {})

    def bpc(arm, s):
        r = recs.get((arm, s))
        return r["eval"]["clean_bpc"] if r else None

    def live_params(arm, s):
        r = recs.get((arm, s))
        return r["params"]["params_live"] if r else None

    # P1：逐臂的 schedule / Σw² / 实测参数 / 每 token MACs / clean bpc（与 EQ、U4 并列）
    table = []
    base_bpc = {"EQ": bpc("EQ", seeds[0])}
    for arm in arms:
        s = seeds[0]
        r = recs.get((arm, s))
        if r is None:
            table.append({"arm": arm, "missing": True})
            continue
        ac = r["cost"]
        table.append({
            "arm": arm, "schedule": r["schedule"], "sum_w2": r["sum_w2"],
            "sum_w": r["sum_w"],
            "params_total_measured": r["params"]["params_total_measured"],
            "params_live": r["params"]["params_live"],
            "params_gates_dead_in_dense": r["params"]["params_gates_dead_measured"],
            "params_ffn_measured": r["params"]["params_ffn_measured"],
            "params_total_matches_analytic": r["params"]["params_total_matches_analytic"],
            "macs_per_token_total": ac["macs_per_token"]["total"],
            "macs_attention": ac["macs_per_token"]["attention"],
            "macs_ffn": ac["macs_per_token"]["ffn"],
            "macs_readout": ac["macs_per_token"]["readout_tied"],
            "clean_bpc": r["eval"]["clean_bpc"],
            "delta_vs_EQ": band_verdict(None if base_bpc["EQ"] is None
                                        else r["eval"]["clean_bpc"] - base_bpc["EQ"]),
            "delta_vs_U4": band_verdict(None if not u4
                                        else r["eval"]["clean_bpc"] - u4["clean_bpc"]),
            "nonuniform": bool(len(set(r["schedule"])) > 1),
        })
    for row in table:
        if row.get("missing"):
            continue
        row["params_live_rel_to_EQ"] = (
            None if live_params("EQ", seeds[0]) in (None, 0)
            else round(row["params_live"] / live_params("EQ", seeds[0]), 6))

    # P2：非均匀 vs 等宽的 Δbpc（带内 / 带外）——判据是**逐臂对 EQ 的配对 Δ**
    p2 = {row["arm"]: row["delta_vs_EQ"] for row in table
          if not row.get("missing") and row["nonuniform"]}

    # P3：必须不变
    p3 = {}
    for arm in arms:
        s = seeds[0]
        r = recs.get((arm, s))
        if r is None:
            continue
        rp = os.path.join(HERE, f"reeval_{arm}_s{s}.json")
        rv = json.load(open(rp)) if os.path.exists(rp) else None
        gate = r["equivalence_gate_w_eq_d"]
        lv = r["liveness"]
        p3[arm] = {
            "reeval_bitwise_equal": (rv["bitwise_equal"] if rv else None),
            "reeval_file": os.path.relpath(rp, HERE) if rv else None,
            "causality_pass": r["causality"]["pass"],
            "liveness_min_ratio": lv["min_ratio"],
            "liveness_positive": bool(lv["min_ratio"] > 0),
            "measured_active_width_matches_target":
                lv["measured_active_width_matches_target"],
            "measured_params_match_analytic":
                r["params"]["params_total_matches_analytic"],
            "equivalence_gate_all_w_eq_d": bool(
                gate["state_dict_bitwise_equal"]
                and gate["fp32"]["bitwise_equal_logits"]
                and gate["fp32"]["bitwise_equal_loss"]
                and gate["bf16_autocast"]["bitwise_equal_logits"]
                and gate["bf16_autocast"]["bitwise_equal_loss"]),
        }

    # P4：否证线
    p4 = {}
    for arm, d in p2.items():
        if d["delta"] is None:
            p4[arm] = "缺读数"
        elif d["sorted"] and d["delta"] > 0:
            p4[arm] = f"{arm}：相对等宽更差且越带 ⇒ 逐层宽度变化在本尺度**反向有害**"
        elif d["sorted"] and d["delta"] < 0:
            p4[arm] = f"{arm}：相对等宽更好且越带 ⇒ 逐层宽度变化有可排序增益"
        else:
            p4[arm] = (f"{arm}：相对等宽的 Δbpc 不越 ±{VARIANCE_BOUND} 带 ⇒ "
                       "逐层宽度变化在本尺度不是独立有增益的维度（并入 2.3.1 深度形状结论）")

    n_from_each_arm = {a: sum(1 for k in recs if k[0] == a) for a in arms}
    out = {
        "experiment": "2646-layer-width-13m",
        "step": steps, "seeds": seeds, "arms_read": list(arms),
        "n_results_per_arm": n_from_each_arm,
        "variance_bound_bpc": VARIANCE_BOUND,
        "variance_bound_source": "harness/research-schema.md §12.1 / C1（限配对 Δ、≥5 窗口）",
        "eval_metric": ("clean bpc = depth_curve 最终深度 bpc（n_batches=20、"
                        "batch_size=16、generator seed 2）"),
        "p1_table": table,
        "p2_nonuniform_vs_EQ": p2,
        "p3_invariants": p3,
        "p4_falsification": p4,
        "anchors": anchors,
        "anchor_reuse": {
            "gates": {
                "switch_off_bitwise": ("本实验 T1：全层 w_l = 256 的前缀模型 == stock "
                                       "引擎逐位相同（见各臂 equivalence_gate_w_eq_d）"),
                "config_fieldwise": anchor_config_parity(steps, batch_size, lr),
                "caliber_annotation": ("clean bpc 口径与 2641 逐字段相同："
                                       "depth_curve(n_batches=20, batch_size=16, "
                                       "generator seed 2) 最终深度；代价用同一解析公式 "
                                       "attention 4w²+2Tw / FFN 8w² / readout d·V"),
            },
        },
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "schedule_search": (json.load(open(os.path.join(HERE, "schedule_search.json")))
                            if os.path.exists(os.path.join(HERE, "schedule_search.json"))
                            else None),
    }
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"p1_table": table, "p2": p2, "p4": p4}, indent=2,
                     ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(SCHEDULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，仅用 runs/<arm>-s<seed>-<k>k/model.pt 重评")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default="EQ,PYR,HG,APYR")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.steps, args.out_root,
                  [a for a in args.arms.split(",") if a],
                  [int(s) for s in args.seeds.split(",") if s != ""],
                  batch_size=args.batch_size, lr=args.lr)
        return
    if not args.arm:
        ap.error("需要 --arm（或 --aggregate）")
    os.makedirs(args.out_root, exist_ok=True)
    if args.no_train:
        reeval(args.arm, args.seed, args.steps, args.out_root)
        return
    info = train_arm(args.arm, args.seed, args.steps, args.batch_size, args.lr,
                     args.out_root)
    print(f"[2646] {args.arm}-s{args.seed} train wall_s={info['wall_s']}", flush=True)
    evaluate(args.arm, args.seed, args.steps, args.out_root)


if __name__ == "__main__":
    main()
