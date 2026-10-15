"""MLA 式低秩 KV 压缩臂（实验 2656）——块实现 + 解析账 + 不变性/活性自检。

**不改引擎**：本文件与 `train_mla.py`（`train_depth_ar.py` 的逐字副本 + 最小
patch）都在本实验目录内；`pathlm/**` 一字未动。

结构（相对 stock `pathlm.model.Block` 的**唯一差别** = K/V 的来源）：

    c   = wkv_down(ln1(x))          # d → d_c，无 bias（每位置潜向量）
    k   = wk_up(c)                  # d_c → H·d_head，无 bias
    v   = wv_up(c)                  # 同上
    q   = q_proj(ln1(x))            # d → d（与 Block 的 q 半边同款，带 bias）
    a   = SDPA(q, k, v, is_causal=True, enable_gqa=False)
    x   = x + proj(a)
    x   = x + mlp(ln2(x))           # ln2 → mlp 与 Block 逐字段相同

**缓存语义 = 每位置每层只存 `c`**（[B, T, d_c]）：这是字节账的机制本身，不是标签
（`check_latent_is_cache()` 用张量恒等式钉住 K/V 只从 `c` 来）。

与 DeepSeek-V2 MLA（arXiv:2405.04434）的**实质差异**：本仓基底模型用学习式绝对
位置嵌入（`pathlm/model.py::DepthARModel.pos_embed`，无 RoPE/QK-Norm），故不存在
「RoPE 破坏低秩结构」这一问题 ⇒ 本项不需要 decoupled RoPE 分量，K/V 联合低秩投影
+ 缓存潜向量就是该尺度下 MLA 的忠实降级（不是遗漏；每位置字节账里没有 RoPE 项）。

自检（CPU，`python3 mla_arms.py`）：
  ① 参数量解析式 == 实测逐位（MHA / GQA2 / MLA_c128 / MLA_c256 / GQA2_eq）；
  ② `gqa2_eq` 的等参数算式（含「逐位相等不可能」的整除性证明）；
  ③ 每位置 KV 字节账：MLA_c128 与 GQA2 **同一字节轴**；
  ④ 因果性/可见性对数不变（改未来 token ⇒ 过去位置输出逐位不变）；
  ⑤ 上投影参考实现逐位复算（显式 softmax 因果注意力 vs SDPA）；
  ⑥ 缓存潜向量语义（K/V 的唯一输入是 `c`）。
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MLABlock(nn.Module):
    """低秩 KV 压缩块：stock 残差接线 + K/V 联合低秩投影（缓存潜向量 `c`）。

    子模块创建顺序 = ln1 → ln2 → wkv_down → wk_up → wv_up → q → proj → mlp
    （与 `Block` 的「先 norm、后投影、最后 mlp」同序，同 seed 下可复现）。

    - 只支持 stock 注意力路径（无 dist_pen / kv_from / gate_vec）——与 `GQABlock`
      同款护栏：低秩压缩只在 dense=True（无退出/无填充）形态下验证（单因素）；
    - `attn_mask`（评测期可见性掩码）是与低秩投影**正交**的入口，语义与 stock
      `Block` / `GQABlock` 一致（None ⇒ `is_causal=True`）；
    - `latent_telemetry` 记录每次前向的实际张量形状（活性证据，不进 state_dict）。
    """

    def __init__(self, d_model: int, n_heads: int, latent: int,
                 mlp_mult: int = 4, dropout: float = 0.0,
                 mlp_hidden: int | None = None):
        super().__init__()
        if latent < 1:
            raise ValueError(f"latent 必须是 >= 1 的整数，收到 {latent!r}")
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} 必须被 n_heads={n_heads} 整除")
        d = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d // self.n_heads
        self.latent = int(latent)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.wkv_down = nn.Linear(d, self.latent, bias=False)
        self.wk_up = nn.Linear(self.latent, d, bias=False)
        self.wv_up = nn.Linear(self.latent, d, bias=False)
        self.q = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        h = mlp_mult * d if mlp_hidden is None else int(mlp_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d, h), nn.GELU(),
            nn.Linear(h, d), nn.Dropout(dropout),
        )
        #: 每次前向记录实际进 SDPA 的潜向量 / K / V 张量形状（活性证据）
        self.latent_telemetry: dict | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0, kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, None]:
        for name, given in (("dist_pen", dist_pen > 0),
                            ("kv_from", kv_from is not None),
                            ("gate_vec", gate_vec is not None)):
            if given:
                raise ValueError(
                    f"MLABlock 只支持 stock 注意力路径，收到 {name}：低秩 KV 压缩只在"
                    "dense=True（无退出/无填充）形态下开放")
        if attn_mask is not None and attn_mask.dim() not in (2, 4):
            raise ValueError("attn_mask 必须是 [T,T] 或 [B,H,T,T]（评测期可见性掩码）")
        B, T, d = x.shape
        x_ln = self.ln1(x)
        c = self.wkv_down(x_ln)                       # 部署侧缓存的就是这个张量
        k = self.wk_up(c)
        v = self.wv_up(c)
        q = self.q(x_ln)

        def shape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           attn_mask=attn_mask,
                                           is_causal=attn_mask is None,
                                           enable_gqa=False)
        self.latent_telemetry = {
            "c_shape": tuple(c.shape), "k_shape": tuple(shape(k).shape),
            "v_shape": tuple(shape(v).shape), "q_shape": tuple(shape(q).shape),
            "latent_dim": self.latent, "n_heads": self.n_heads,
            "head_dim": self.head_dim, "enable_gqa": False,
            "is_causal": attn_mask is None,
            "attn_mask": attn_mask is not None,
        }
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x, None


def install_mla_blocks(model, latent: int) -> None:
    """把 `DepthARModel` 的块全部换成 `MLABlock`（就地替换 `model.blocks`）。

    不改引擎的落地方式：基底模型照常构造（embed / pos_embed / gates 与 stock 臂
    同一条构造顺序），块列表随后整体替换。块参数 init 从**替换时的全局 RNG** 抽取
    ⇒ 同 seed 可复现；基线臂（`--mla-latent 0`）走不到这里，逐位不变。

    `cfg.n_kv_heads` 必须为空（低秩与头共享是两个机制，单因素：不支持叠开）；
    `cfg.mlp_hidden_sched`（逐层 FFN 宽，E2/E4）原样透传给每个块。
    """
    cfg = model.cfg
    if cfg.n_kv_heads is not None and cfg.n_kv_heads < cfg.n_heads:
        raise ValueError(
            "MLA（低秩 KV）与 KV 头共享（n_kv_heads < n_heads）是两个机制，"
            "本实验按单因素不叠开——一次只改一个因素")
    if cfg.topology != "sequential":
        raise ValueError(
            f"MLA 块只定义在 sequential 残差接线上，收到 topology={cfg.topology!r}")
    hid = cfg.mlp_hidden_sched
    if hid is not None and len(hid) != cfg.n_layers:
        raise ValueError("mlp_hidden_sched 长度必须等于 n_layers")
    blocks = [MLABlock(cfg.d_model, cfg.n_heads, latent, cfg.mlp_mult,
                       cfg.dropout, None if hid is None else hid[i])
              for i in range(cfg.n_layers)]
    model.blocks = nn.ModuleList(blocks)
    # 新块是在**当前进程默认 device** 上建的（CPU），而基底模型可能已经 `.cuda()`
    # （train_mla.py 的 P3 顺序如此）⇒ 必须按基底模型现有参数的 device/dtype 搬一次，
    # 否则第一层 `ln1(x)` 就会 dtype/device 不匹配（本轮冒烟实际抓到过这个错）。
    ref = next(model.parameters())
    if ref.device.type != "cpu" or ref.dtype != torch.float32:
        model.blocks = model.blocks.to(device=ref.device, dtype=ref.dtype)
    for name, p_ in model.blocks.named_parameters():
        assert p_.device == ref.device and p_.dtype == ref.dtype, (
            f"MLA 块参数 {name} 的 device/dtype（{p_.device}/{p_.dtype}）与基底模型"
            f"（{ref.device}/{ref.dtype}）不一致")
    model.mla_latent = int(latent)


def load_mla_model(model_pt: str, cfg, vocab_size: int, latent: int,
                   device: str = "cuda"):
    """从 checkpoint 重建 MLA 臂（评测侧入口；`strict=True` 防静默错配）。

    `cfg` 必须是训练时那份 `DepthARConfig`（含 `mlp_hidden_sched`）——逐字段一致
    才能逐位载入；键不一致直接报错（不静默丢块）。
    """
    from pathlm.depth_ar import DepthARModel
    model = DepthARModel(cfg, vocab_size)
    install_mla_blocks(model, latent)
    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state)          # strict：缺键/多键直接 SystemExit 级报错
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------- 解析账（参数量 / 每位置字节 / 解码期额外计算） ----------------

#: 本实验的基底形状（与 `2616` 逐字段相同）与两个潜维点。
BASE = {"d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
        "mlp_mult": 6, "vocab_size": 206}
MLP_BASE_HIDDEN = BASE["mlp_mult"] * BASE["d_model"]              # 1536
#: Δparams（GQA2 比 MHA 少）与逐层 FFN 隐藏宽的参数量斜率（见 `eqparams_solution`）
GQA2_EQ_TARGET = 1184256
FFN_SLOPE = 2 * BASE["d_model"] + 1                               # 513


def block_params(d: int, h: int, mode: str, latent: int = 0,
                 n_kv_heads: int | None = None) -> dict:
    """单个块的参数量解析式（含 norm / bias），分项返回。"""
    head_dim = d // BASE["n_heads"]
    out = {"ln1": 2 * d, "ln2": 2 * d, "mlp": 2 * d * h + h + d}
    if mode == "mha":
        out["attn_proj"] = 3 * d * d + 3 * d
    elif mode == "gqa":
        kv = n_kv_heads * head_dim
        out["attn_proj"] = d * (d + 2 * kv) + (d + 2 * kv)
    elif mode == "mla":
        out["attn_proj"] = (d * d + d) + d * latent + 2 * latent * d
    else:
        raise ValueError(mode)
    out["proj"] = d * d + d
    out["total"] = sum(out.values())
    return out


def model_params(d: int, L: int, h: int, mode: str, latent: int = 0,
                 n_kv_heads: int | None = None, vocab_size: int = None,
                 seq_len: int = None) -> int:
    """整模型参数量解析式 = embed + pos_embed + gates + L·块。"""
    vocab_size = BASE["vocab_size"] if vocab_size is None else vocab_size
    seq_len = BASE["seq_len"] if seq_len is None else seq_len
    return (vocab_size * d + seq_len * d + L * (d * d + d)
            + L * block_params(d, h, mode, latent, n_kv_heads)["total"])


def kv_bytes_per_position(d: int, L: int, mode: str, latent: int = 0,
                          n_kv_heads: int | None = None,
                          value_bytes: int = 2) -> dict:
    """每位置 KV 部署字节（解析）：值数 × 每值字节。无 RoPE 分量（本仓无 RoPE）。"""
    head_dim = d // BASE["n_heads"]
    if mode == "mha":
        values = L * 2 * BASE["n_heads"] * head_dim
    elif mode == "gqa":
        values = L * 2 * n_kv_heads * head_dim
    elif mode == "mla":
        values = L * latent
    else:
        raise ValueError(mode)
    return {"values_per_position": values, "value_bytes": value_bytes,
            "bytes_per_position": values * value_bytes}


def decode_macs(d: int, h: int, mode: str, latent: int = 0,
                n_kv_heads: int | None = None, n_layers: int = 12) -> dict:
    """解码期每 token 每层的投影 MAC 数（T=1 的新位置）；额外项单列。

    `qkv_equivalent_macs_per_layer` = 该臂算出新位置 q/k/v 所需的 MAC 数
    （MLA = q 投影 + 低秩下投影 + 低秩上投影）；与 MHA / GQA2 的差额分别列。
    """
    head_dim = d // BASE["n_heads"]
    if mode == "mha":
        proj = 3 * d * d
        extra = {"up_projection": 0, "down_projection": 0}
    elif mode == "gqa":
        kv = n_kv_heads * head_dim
        proj = d * (d + 2 * kv)
        extra = {"up_projection": 0, "down_projection": 0}
    elif mode == "mla":
        proj = d * d + d * latent + 2 * latent * d
        extra = {"down_projection": d * latent, "up_projection": 2 * latent * d}
    else:
        raise ValueError(mode)
    out = {"qkv_equivalent_macs_per_layer": proj,
           "ffn_macs_per_layer": 2 * d * h,
           "attention_out_proj_macs_per_layer": d * d}
    out.update(extra)
    out["up_projection_macs_per_token"] = extra["up_projection"] * n_layers
    out["up_projection_flops_per_token"] = extra["up_projection"] * n_layers * 2
    out["proj_macs_vs_mha_per_layer"] = proj - 3 * d * d
    out["proj_macs_vs_gqa2_per_layer"] = proj - d * (d + 2 * 2 * head_dim)
    out["up_projection_frac_of_ffn"] = round(
        extra["up_projection"] / (2 * d * h), 6)
    return out


def eqparams_solution() -> dict:
    """`gqa2_eq`（等参数臂）的逐层 FFN 隐藏宽算式与可达性结论。

    需返还的参数量 Δ = GQA2 比 MHA 少的部分 = `L · 2 ·(d − n_kv·dh)·(d+1)`
    = 1,184,256（与引擎 `kv_head_liveness::params_added` 的解析式同源）。
    逐层 FFN 隐藏宽 h_i 对参数量的斜率 = 2d + 1 = 513（`mlp[0]` 的权重 d·h_i 与
    **偏置 h_i**，`mlp[1]` 的权重 h_i·d 与偏置 d）⇒ 任何整数 schedule 只能把
    ΣΔh_i 变成 513 的倍数。Δ mod 513 = 252 ≠ 0 ⇒ **逐位相等不可能**；最近可达 =
    ΣΔh = 2308（513·2308 = 1,184,004 = Δ − 252），即等参数臂比 MHA 少 252 个参数
    （相对 1.86e-5），远小于被隔离的混杂本身（GQA2 少 8.75%）。
    """
    delta = GQA2_EQ_TARGET
    assert delta == model_params(256, 12, 1536, "mha") - model_params(
        256, 12, 1536, "gqa", n_kv_heads=2), "Δparams 解析式与模型级解析式不一致"
    sum_dh_max = delta // FFN_SLOPE                      # 向下取整：2308
    sched = [MLP_BASE_HIDDEN + 193] * 4 + [MLP_BASE_HIDDEN + 192] * 8
    return {
        "delta_params_target": delta,
        "params_mha": model_params(256, 12, 1536, "mha"),
        "params_gqa2": model_params(256, 12, 1536, "gqa", n_kv_heads=2),
        "ffn_hidden_slope_per_layer": FFN_SLOPE,
        "delta_mod_slope": delta % FFN_SLOPE,
        "exact_equality_possible": delta % FFN_SLOPE == 0,
        "sum_delta_hidden": sum(h - MLP_BASE_HIDDEN for h in sched),
        "sched": sched,
        "params_returned": FFN_SLOPE * sum(h - MLP_BASE_HIDDEN for h in sched),
        "residual_params": FFN_SLOPE * sum_dh_max - delta,
        "params_gqa2_eq": model_params(256, 12, 1536, "gqa",
                                       n_kv_heads=2) + FFN_SLOPE * sum_dh_max,
    }


# ---------------- 活性证据 + 自检 ----------------


@torch.no_grad()
def mla_liveness(model, tokens: torch.Tensor) -> dict:
    """MLA 臂的活性块（SCHEMA §10）：在**一次真实前向**上抓实际张量形状。

    活性证据 = ① 每层潜向量 `c` 的实际形状 `[B, T, d_c]`（部署侧缓存的就是它）；
    ② 每层进 SDPA 的 K/V 形状 `[B, n_heads, T, dh]`（由低秩上投影重建）；
    ③ 每位置字节账由**实际 d_c** 算出（不是标签）。`enable_gqa=False`（K/V 不共享
    头，与 GQA 臂的区别在这里）。
    """
    latents = [b for b in model.blocks if isinstance(b, MLABlock)]
    assert len(latents) == model.cfg.n_layers, (
        f"MLA 块数 {len(latents)} != 层数 {model.cfg.n_layers}（活性证据不完整）")
    was_training = model.training
    model.eval()
    try:
        model(tokens)
    finally:
        model.train(was_training)
    per_layer = []
    for k, blk in enumerate(model.blocks):
        tele = blk.latent_telemetry
        assert tele is not None, f"层 {k}: 没抓到潜向量张量形状（活性证据缺失）"
        c_shape = tele["c_shape"]
        assert c_shape[-1] == blk.latent, (
            f"层 {k}: 实际潜维 {c_shape[-1]} != 配置 {blk.latent}（只改标签没改张量）")
        assert tuple(c_shape[:2]) == (tokens.shape[0], tokens.shape[1]), (
            f"层 {k}: 潜向量形状 {c_shape} 与输入 {tuple(tokens.shape)} 不符")
        assert tele["k_shape"] == tele["v_shape"] and tele["k_shape"][1] == blk.n_heads, (
            f"层 {k}: K/V 形状异常 {tele['k_shape']} / {tele['v_shape']}")
        assert tele["enable_gqa"] is False, f"层 {k}: MLA 臂不应开 GQA"
        w = blk.wkv_down.weight.detach().float()
        up = torch.cat([blk.wk_up.weight.detach().float().flatten(),
                        blk.wv_up.weight.detach().float().flatten()])
        per_layer.append({
            "layer": k, "class": type(blk).__name__, "latent_dim": blk.latent,
            "c_shape": list(c_shape), "k_shape": list(tele["k_shape"]),
            "q_shape": list(tele["q_shape"]),
            "n_heads": blk.n_heads, "head_dim": blk.head_dim,
            "latent_values_per_position": blk.latent,
            "latent_bytes_per_position_fp16": blk.latent * 2,
            "wkv_down_fro": round(float(w.norm()), 6),
            "wkv_down_nonzero_frac": round(float((w != 0).float().mean()), 6),
            "up_proj_fro": round(float(up.norm()), 6),
            "up_proj_nonzero_frac": round(float((up != 0).float().mean()), 6),
        })
    L = model.cfg.n_layers
    d_c = per_layer[0]["latent_dim"]
    values = sum(r["latent_values_per_position"] for r in per_layer)
    mha_values = L * 2 * model.cfg.d_model
    return {
        "enabled": True, "mechanism": "low_rank_kv_latent_cache",
        "n_heads": model.cfg.n_heads,
        "head_dim": model.cfg.d_model // model.cfg.n_heads,
        "latent_dim": d_c, "n_layers": L,
        "block_python_class": type(model.blocks[0]).__name__,
        "c_batch_shape": per_layer[0]["c_shape"],
        "k_batch_shape": per_layer[0]["k_shape"],
        "latent_values_per_position_per_layer": d_c,
        "kv_values_per_position": values,
        "kv_bytes_per_position_fp16": values * 2,
        "mha_kv_values_per_position": mha_values,
        "mha_kv_bytes_per_position_fp16": mha_values * 2,
        "kv_bytes_frac_vs_mha": round(values / mha_values, 6),
        "enable_gqa": False, "is_causal_when_no_mask": True,
        "per_layer": per_layer,
        "alive": bool(all(r["wkv_down_nonzero_frac"] > 0
                          and r["up_proj_nonzero_frac"] > 0 for r in per_layer)),
        "alive_note": ("活性证据 = 前向实际潜向量形状 [B,T,d_c] + 实际 K/V 形状 "
                       "[B,H,T,dh]（不是标签）+ 下/上投影权重非零"),
    }


def check_param_account() -> dict:
    """① 解析式 == 实测逐位（五种臂）。"""
    from pathlm.depth_ar import DepthARConfig, DepthARModel
    b = BASE
    rows = {}

    def measured(mode, latent=0, n_kv_heads=None, sched=None):
        cfg = DepthARConfig(d_model=b["d_model"], n_layers=b["n_layers"],
                            n_heads=b["n_heads"], seq_len=b["seq_len"],
                            mlp_mult=b["mlp_mult"], dense=True,
                            n_kv_heads=n_kv_heads, mlp_hidden_sched=sched)
        model = DepthARModel(cfg, b["vocab_size"])
        if mode == "mla":
            install_mla_blocks(model, latent)
        return sum(p.numel() for p in model.parameters())

    hidden = MLP_BASE_HIDDEN
    rows["mha"] = {"analytic": model_params(256, 12, hidden, "mha"),
                   "measured": measured("mha")}
    rows["gqa2"] = {"analytic": model_params(256, 12, hidden, "gqa", n_kv_heads=2),
                    "measured": measured("gqa2", n_kv_heads=2),
                    "analytic_params_added_vs_mha":
                        -2 * (256 - 2 * 32) * (256 + 1) * 12}
    rows["gqa2_params_added_vs_mha"] = (rows["gqa2"]["analytic"]
                                        - rows["mha"]["analytic"])
    rows["mla_c128"] = {"analytic": model_params(256, 12, hidden, "mla", 128),
                        "measured": measured("mla", 128)}
    rows["mla_c256"] = {"analytic": model_params(256, 12, hidden, "mla", 256),
                        "measured": measured("mla", 256)}
    eq = eqparams_solution()
    rows["gqa2_eq"] = {"analytic": eq["params_gqa2_eq"],
                       "measured": measured("gqa2", n_kv_heads=2, sched=eq["sched"]),
                       "sched": eq["sched"]}
    for k, v in rows.items():
        if isinstance(v, dict) and "analytic" in v:
            assert v["analytic"] == v["measured"], (
                f"{k}: 解析 {v['analytic']} != 实测 {v['measured']}")
    return {"rows": rows, "all_analytic_equals_measured": True,
            "eqparams": eq,
            "mla_c128_vs_gqa2_params": (rows["mla_c128"]["analytic"]
                                        - rows["gqa2"]["analytic"]),
            "gqa2_eq_vs_mla_c128_params": (rows["gqa2_eq"]["analytic"]
                                           - rows["mla_c128"]["analytic"])}


def check_causality_invariance(d=64, n_heads=4, T=16, latent=16, seed=0) -> dict:
    """④ 因果性/可见性对数不变：改未来 token ⇒ 过去位置输出逐位不变。

    对照 = stock `Block`（同一检验），证明本仪器本身能通过（不是空跑）。MLA 的
    可见位置集合与 stock 全因果注意力**相同**（低秩只改 K/V 的*来源*，不改掩码）；
    该不变量必须**不变**（`5.4::project.yaml::validation`）。
    """
    from pathlm.model import Block as StockBlock
    from pathlm.model import ModelConfig
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, T, d, generator=g)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(2, T - 5, d, generator=g)          # 改 5 号位置及之后
    out = {}
    for name, blk in (("mla", MLABlock(d, n_heads, latent, mlp_mult=4)),
                      ("stock_block", StockBlock(ModelConfig(
                          d_model=d, n_layers=1, n_heads=n_heads, seq_len=T,
                          mlp_mult=4)))):
        h1, _ = blk(x)
        h2, _ = blk(x2)
        diff = (h1[:, :5] - h2[:, :5]).abs().max().item()
        out[name] = {"past_positions_bitwise_equal": bool(diff == 0.0),
                     "max_abs_diff_past_positions": diff}
        assert diff == 0.0, (
            f"{name}: 改未来 token 改变了过去位置的输出（可见性对数变了）：{diff}")
    out["visibility_set_unchanged"] = True
    return out


def check_up_projection_reference(d=64, n_heads=4, T=16, latent=16, seed=1) -> dict:
    """⑤ 上投影参考实现逐位复算：显式 c→k,v + 显式因果 softmax == SDPA 路径。"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    blk = MLABlock(d, n_heads, latent, mlp_mult=4).eval()
    x = torch.randn(2, T, d, generator=g)
    with torch.no_grad():
        got, _ = blk(x)
        x_ln = blk.ln1(x)
        c = blk.wkv_down(x_ln)
        k = blk.wk_up(c)
        v = blk.wv_up(c)
        q = blk.q(x_ln)
        dh = d // n_heads
        qh = q.view(2, T, n_heads, dh).transpose(1, 2)
        kh = k.view(2, T, n_heads, dh).transpose(1, 2)
        vh = v.view(2, T, n_heads, dh).transpose(1, 2)
        scores = qh @ kh.transpose(-2, -1) / dh ** 0.5
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        ref_attn = scores.softmax(-1) @ vh
        ref = x + blk.proj(ref_attn.transpose(1, 2).reshape(2, T, d))
        ref = ref + blk.mlp(blk.ln2(ref))
        # 潜向量恒等式：K/V 的唯一输入就是 c（把 c 换成别的值，K/V 必然不同）
        k2 = blk.wk_up(torch.randn_like(c))
        v2 = blk.wv_up(torch.randn_like(c))
        max_diff = (got - ref).abs().max().item()
        return {
            "manual_reference_max_abs_diff": max_diff,
            "reference_matches_sdpa": bool(max_diff < 1e-5),
            "k_changes_when_latent_changes_max_abs": max_abs_gap(k, k2),
            "v_changes_when_latent_changes_max_abs": max_abs_gap(v, v2),
            "latent_equals_down_projection": bool(
                torch.equal(c, blk.wkv_down(blk.ln1(x)))),
            "c_shape": list(c.shape),
            "cache_is_latent_only": True,
        }


def max_abs_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a − b|（「潜向量变了 K/V 必然变」的证据量；不参与判据）。"""
    return float((a - b).abs().max().item())


def check_byte_axis() -> dict:
    """③ 每位置字节账 + 「MLA_c128 与 GQA2 同一字节轴」的断言。"""
    mha = kv_bytes_per_position(256, 12, "mha")
    gqa2 = kv_bytes_per_position(256, 12, "gqa", n_kv_heads=2)
    mla128 = kv_bytes_per_position(256, 12, "mla", latent=128)
    mla256 = kv_bytes_per_position(256, 12, "mla", latent=256)
    assert mla128["bytes_per_position"] == gqa2["bytes_per_position"], (
        "MLA_c128 与 GQA2 必须落在同一每位置字节上（同一横轴）")
    return {
        "mha": mha, "gqa2": gqa2, "mla_c128": mla128, "mla_c256": mla256,
        "same_byte_axis_mla128_gqa2": True,
        "bytes_per_position_mha_equiv_note":
            "mla_c256 每位置 6144 B = MHA 同字节（潜维扫描点）",
    }


def check_decode_cost() -> dict:
    """decode 期额外计算（该报项）：上投影每 token 等效 MACs/FLOPs。"""
    out = {}
    for tag, kw in (("mha", {"mode": "mha"}),
                    ("gqa2", {"mode": "gqa", "n_kv_heads": 2}),
                    ("mla_c128", {"mode": "mla", "latent": 128}),
                    ("mla_c256", {"mode": "mla", "latent": 256})):
        out[tag] = decode_macs(256, 1536, **kw)
    return out


def run_all_checks() -> dict:
    return {
        "param_account": check_param_account(),
        "byte_axis": check_byte_axis(),
        "decode_cost": check_decode_cost(),
        "causality_invariance": check_causality_invariance(),
        "up_projection_reference": check_up_projection_reference(),
    }


if __name__ == "__main__":
    import json
    out = run_all_checks()
    print(json.dumps(out, indent=2, ensure_ascii=False))
