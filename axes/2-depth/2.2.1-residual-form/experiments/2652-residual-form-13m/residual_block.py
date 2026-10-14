"""2652 · 残差形式的**实验目录内**实现（工厂替换；不改 `pathlm/**`）。

四个臂共用一个块类 `ResidualFormBlock`：它**接管**一个 stock `Block` 的同一批子模块
（`ln1` / `ln2` / `qkv` / `proj` / `mlp`，同一个 Parameter 对象），于是各臂共享
**同一次随机初始化抽样**，臂与臂之间唯一的差别 = 子层输出如何接回残差流。

形式定义（`A(x) = proj(SDPA(qkv(x)))` = 注意力子层输出，`M(x) = mlp(x)`）：

| 形式 | 公式 | 额外参数 |
|---|---|---|
| `base` | `x' = x + A(ln1 x)`；`x' = x' + M(ln2 x')`（引擎 stock Block，**不包裹**） | 0 |
| `postln` | `x' = ln2( x1 + M(x1) )`，`x1 = ln1( x + A(x) )`（归一化**后置**） | 0 |
| `scaled` | 同 base，但每个子层增量乘固定 `α_l = 1/√(l+1)` | 0 |
| `gated` | 同 base，但每个子层增量乘可学习逐层标量门（`attn_gate` / `mlp_gate`，init=1.0） | 2L = 16 |

`gated` 在 init 处与 `base` **逐位一致**：门值恒 `1.0`，而浮点乘 `1.0` 精确
⇒ `x + 1.0·t` 与 `x + t` 是同一位模式。门在乘之前 `to(x.dtype)`（bf16 autocast 下
把 fp32 标量降到 bf16，`1.0` 在 bf16 里可精确表示）——**不**让残差流被数值提升成 fp32
（否则「只改残差形式」这条单因素声明就不成立）。`smoke.py` 的等值门在 fp32 与
bf16 autocast 两条路径上钉住这一点。

非必要性例外：无。基底 = 主流 stock decoder-only transformer（`--dense`：无 carry 门、
无退出采样、仅最终深度 CE）；块内的 `ln1` / `ln2` / `qkv` / `proj` / `mlp` 与 attention
算子（SDPA causal）逐位取自 stock `Block`。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlm.depth_ar import DepthARModel
from pathlm.model import Block

FORMS = ("base", "postln", "scaled", "gated")


class ResidualFormBlock(nn.Module):
    """把 stock `Block` 的子模块重新接线成四种残差形式之一。

    `forward` 的签名与 stock `Block` 一致（`attn_mask` / `dist_pen` / `kv_from` /
    `gate_vec`），但本实验只在 **dense + sequential 的无掩码路径**上定义残差形式；
    其余入参非缺省时直接报错（不静默降级成一个「看不出区别」的前向）。
    """

    def __init__(self, stock: Block, form: str, layer_index: int):
        super().__init__()
        if form not in FORMS:
            raise ValueError(f"unknown residual form {form!r}; expected {FORMS}")
        if not isinstance(stock, Block):
            raise TypeError(f"expected a stock pathlm.model.Block, got {type(stock)}")
        self.form = form
        self.layer_index = layer_index          # 0-based，用于 α_l = 1/√(l+1)
        self.n_heads = stock.n_heads
        # 接管同一批子模块（同一 Parameter 对象）⇒ 初始化与参数量逐位可比
        self.ln1 = stock.ln1
        self.ln2 = stock.ln2
        self.qkv = stock.qkv
        self.proj = stock.proj
        self.mlp = stock.mlp
        if form == "scaled":
            self.alpha = 1.0 / math.sqrt(layer_index + 1)
        elif form == "gated":
            self.attn_gate = nn.Parameter(torch.ones(()))
            self.mlp_gate = nn.Parameter(torch.ones(()))

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        """stock `Block` 的注意力子层输出 `proj(SDPA(qkv(x)))`（一字不差地复刻
        `pathlm/model.py::Block.forward` 的 SDPA 路径，保证 `gated@init` 与 stock
        逐位一致）。`x` 是**子层输入**：base/scaled/gated 传 `ln1(x)`，postln 传 `x`。"""
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        hd = d // self.n_heads
        qs = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        ks = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        vs = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None,
                                           is_causal=True)
        return self.proj(a.transpose(1, 2).reshape(B, T, d))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0,
                kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None):
        if (attn_mask is not None or dist_pen != 0.0 or kv_from is not None
                or gate_vec is not None):
            raise ValueError(
                "ResidualFormBlock 只在 dense + sequential 的无掩码路径上定义"
                "（单因素）；收到非缺省的 attn_mask/dist_pen/kv_from/gate_vec")
        if self.form == "base":
            x = x + self._attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        elif self.form == "postln":
            x = self.ln1(x + self._attn(x))
            x = self.ln2(x + self.mlp(x))
        elif self.form == "scaled":
            a = self.alpha
            x = x + a * self._attn(self.ln1(x))
            x = x + a * self.mlp(self.ln2(x))
        else:  # gated：init 处 attn_gate=mlp_gate=1.0 ⇒ 逐位等价 base
            ga = self.attn_gate.to(x.dtype)
            gm = self.mlp_gate.to(x.dtype)
            x = x + ga * self._attn(self.ln1(x))
            x = x + gm * self.mlp(self.ln2(x))
        return x, None

    def extra_repr(self) -> str:
        return f"form={self.form}, layer={self.layer_index}"


def apply_residual_form(model: DepthARModel, form: str) -> DepthARModel:
    """把 stock `model.blocks` 换成 `ResidualFormBlock`（`base` = 不包裹，逐字用
    引擎 stock 块）。"""
    if form not in FORMS:
        raise ValueError(f"unknown residual form {form!r}; expected {FORMS}")
    if form == "base":
        return model
    if not model.cfg.dense or model.cfg.topology != "sequential":
        raise ValueError(
            "残差形式干预只在 dense=True + topology='sequential' 下定义（单因素）——"
            f"收到 dense={model.cfg.dense} topology={model.cfg.topology!r}")
    model.blocks = nn.ModuleList(
        ResidualFormBlock(blk, form, i) for i, blk in enumerate(model.blocks))
    return model


def build_model(cfg, vocab_size: int, form: str) -> DepthARModel:
    """所有臂共用同一构造路径：先按引擎 stock 路径建**完整**模型（同 seed ⇒ 同一次
    抽样），再按 form 重新接线。`base` 返回的就是 stock 模型本身。"""
    return apply_residual_form(DepthARModel(cfg, vocab_size), form)


def alpha_by_layer(form: str, n_layers: int) -> list[float] | None:
    """`scaled` 的逐层固定缩放 α_l = 1/√(l+1)（其余形式返回 None）。"""
    if form != "scaled":
        return None
    return [1.0 / math.sqrt(l + 1) for l in range(n_layers)]


def gate_values(model: DepthARModel) -> dict | None:
    """`gated` 的逐层门终值（attn / mlp 各一份）；其余形式返回 None。"""
    gates = [b for b in model.blocks if getattr(b, "form", None) == "gated"]
    if not gates:
        return None
    return {
        "attn_by_layer": [float(b.attn_gate.detach()) for b in gates],
        "mlp_by_layer": [float(b.mlp_gate.detach()) for b in gates],
    }
