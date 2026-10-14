"""2652 · 激活假量化语义层（AQ arms）—— 本实验**唯一**的新增语义代码。

契约：同目录 `README.md`（§1–§10）+ `config.json::quantizer` / `config.json::arms`。
判据线在 `config.json::criterion`（**运行前**写入）。

它回答一个可证伪的问题：**在 13M 受控设定下，把前向的中间激活以 int8 / 4 位存放
（含每 token 动态标尺），激活字节 → 序列级质量（clean bpc + 长程召回）的代价是什么？
离群通道吃掉多少位？**

## 干预的定义（单因素）

**唯一改动 = 每层 4 个 `nn.Linear` 的输入激活的存储精度**（+ 组合臂 `W8A8` 的权重
per-output-channel int8）。张量集（与 `config.json::axis_definition.tensor_set` 逐字一致）：

| tag | Linear | 输入激活 | 每 token 元素数 |
|---|---|---|---|
| `qkv` | `blk.qkv` | `ln1(x)`（attn 前的 q/k/v 输入） | `d` |
| `proj` | `blk.proj` | 注意力输出 `a` | `d` |
| `mlp0` | `blk.mlp[0]` | `ln2(x)`（FFN 第一个 matmul 输入） | `d` |
| `mlp2` | `blk.mlp[2]` | `GELU(blk.mlp[0] 输出)`（FFN 第二个 matmul 输入） | `mlp_mult·d` |

标尺粒度：**per-token 动态**（每个 `(batch, token)` 一行共享一把 fp16 标尺 =
该行 `|x|` 的 amax / qmax）。选它的理由：激活的难度是「同一 token 内不同通道量级
差几个数量级」（离群通道），per-token 标尺把动态范围锚到该 token 自己，是业界
W8A8 动态量化的标准做法；且它**只在当前 token 内归约**，不引入任何跨位置依赖
（因果性不变）。

## 与历史 harness 的实现等值（硬护栏 ②）

`activation_mode == "fp16"` 时前向**不做任何数值改动**（恒等），因此 `fp16` 臂必须
与历史锚 `1.483569`（`bpc_of` / `retention.masked_bpc` 两条独立路径）
**逐位一致**。本模块把 `Block.forward` 的 dense 分支（`attn_mask=None`、
`dist_pen=0`、`kv_from=None`、`gate_vec=None`）逐行复刻，`fp16` 档下与
`retention.masked_logits` 逐位相同（`run.py::instrument` 钉住）。

## 活性量（SCHEMA §10）

每个被量化张量落：`n_values` / `n_scales` / `rel_err` / `mean_abs_err` /
`changed_frac` / `clipped_frac` / 标尺分布（min / median / max）；另有**离群通道**
读数：`chan_amax_outlier_index` = max/median（通道 |激活| 峰值）、
`effective_bits_median_channel` = `bits − log2(outlier_index)`（离群通道吃掉的位）、
`revert_chan_frac_6x` = 峰值 > 6×median 的通道占比（LLM.int8() 式回退阈值下的
回退通道比例）。fp16 档只报通道结构（无标尺、`changed_frac = 0`）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pathlm.model import Block

#: 激活精度档（`fp16` = 恒等，不引入往返）。
ACT_MODES = ("fp16", "int8", "int4")
#: 权重精度档（只对组合臂 `W8A8` 打开）。
W_MODES = ("fp16", "int8")
#: 每个整型档的位数。**唯一的位数定义处**（字节账与 qmax 都从这里派生）。
_BITS = {"int8": 8, "int4": 4}
#: 每把标尺的存储字节数（fp16 标尺 = 2 B；与 5.1 的 `kv_quant.SCALE_BYTES` 同口径）。
SCALE_BYTES = 2.0
#: 对称量化的 qmax 定义式（唯一处）。
def _qmax(mode: str) -> int:
    return 2 ** (_BITS[mode] - 1) - 1


# --------------------------------------------------------------- 量化 / 反量化


def fake_quant_activation(x: torch.Tensor, mode: str):
    """激活的对称假量化往返。返回 `(dq, scale, ratio)`。

    - `mode == "fp16"`：逐位恒等（`dq is x`，`scale = ratio = None`）——这是
      「不做任何数值改动」的定义处，硬护栏 ② 依赖它。
    - `int8` / `int4`：`scale = 该行 |x| 的 amax / qmax`（per-token，沿最后一维
      归约，`clamp_min(1e-12)` 防 0 行），`q = clamp(round(x/scale), −qmax, qmax)`，
      `dq = q · scale`。往返误差上界 `|dq − x| ≤ 0.5·scale`（取整误差，clamp 不越界）。
    """
    if mode == "fp16":
        return x, None, None
    if mode not in _BITS:
        raise ValueError(f"unknown activation mode {mode!r}; expected one of {ACT_MODES}")
    qmax = _qmax(mode)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / qmax).clamp_min(1e-12)
    ratio = x / scale
    q = torch.round(ratio).clamp(-qmax, qmax)
    return q * scale, scale, ratio


def weight_quant_scale(w: torch.Tensor, mode: str) -> torch.Tensor:
    """权重的 per-output-channel 标尺（`[out, 1]`）——权重档的标尺定义处。"""
    if mode == "fp16":
        return w.new_zeros(w.shape[0], 1)
    qmax = _qmax(mode)
    amax = w.abs().amax(dim=1, keepdim=True)
    return (amax / qmax).clamp_min(1e-12)


def fake_quant_weight(w: torch.Tensor, mode: str) -> torch.Tensor:
    """权重的对称假量化往返（**per-output-channel**：每个输出通道一行共享一把标尺）。

    只被组合臂 `W8A8` 使用。`mode == "fp16"` 逐位恒等。
    """
    if mode == "fp16":
        return w
    if mode not in _BITS:
        raise ValueError(f"unknown weight mode {mode!r}; expected one of {W_MODES}")
    qmax = _qmax(mode)
    wscale = weight_quant_scale(w, mode)
    q = torch.round(w / wscale).clamp(-qmax, qmax)
    return q * wscale


# --------------------------------------------------------------- 字节账（解析）


def activation_tensor_shapes(cfg: dict) -> list[tuple[str, int]]:
    """每层被量化的 4 个激活张量的 `(tag, in_features)`——**唯一的张量集定义处**。"""
    d = cfg["model"]["d_model"]
    return [("qkv", d), ("proj", d), ("mlp0", d), ("mlp2", cfg["model"]["mlp_mult"] * d)]


def activation_bytes_per_token(cfg: dict, act_mode: str) -> dict:
    """`bytes_per_token_activation` 的**解析**账（判据的主口径横轴）。

    式：`bytes/token = Σ_tensor in_features · bits/8 · n_layers
    + (act_mode != fp16 ? SCALE_BYTES · n_tensors_per_layer · n_layers : 0)`。

    标尺摊薄写清：per-token 标尺 ⇒ 每个 (张量, token) **恰好一把** fp16 标尺
    （2 B）⇒ 每 token 每张量摊 2 B；对 `in_features` 个元素，等价于
    `+ 2/in_features` B/元素。实现对应：`fake_quant_activation` 的
    `scale.shape[-1] == 1`、`scale.numel() == B·T`（`run.py::instrument.n_scales`
    对账）。
    """
    if act_mode not in ACT_MODES:
        raise ValueError(f"unknown activation mode {act_mode!r}")
    n_layers = cfg["model"]["n_layers"]
    shapes = activation_tensor_shapes(cfg)
    bits = 16 if act_mode == "fp16" else _BITS[act_mode]
    scale_per_tensor_per_token = 0.0 if act_mode == "fp16" else SCALE_BYTES
    per_tensor = {}
    value_sum = 0.0
    scale_sum = 0.0
    for tag, inf in shapes:
        v = inf * bits / 8
        value_sum += v
        scale_sum += scale_per_tensor_per_token
        per_tensor[tag] = {"in_features": inf, "value_bytes_per_token": v,
                           "scale_bytes_per_token": scale_per_tensor_per_token}
    per_layer = value_sum + scale_sum
    return {
        "act_mode": act_mode,
        "bits": bits,
        "n_tensors_per_layer": len(shapes),
        "n_tensors_total": len(shapes) * n_layers,
        "elements_per_token_per_layer": sum(f for _, f in shapes),
        "elements_per_token_total": sum(f for _, f in shapes) * n_layers,
        "value_bytes_per_token": round(value_sum * n_layers, 6),
        "scale_bytes_per_token": round(scale_sum * n_layers, 6),
        "bytes_per_token_per_layer": round(per_layer, 6),
        "bytes_per_token_activation": round(per_layer * n_layers, 6),
        "per_layer": round(per_layer, 6),
        "per_tensor": per_tensor,
    }


def weight_tensor_shapes(cfg: dict) -> list[tuple[str, int, int]]:
    """每层被量化权重的 `(tag, out_features, in_features)`——**唯一的权重集定义处**。"""
    d = cfg["model"]["d_model"]
    m = cfg["model"]["mlp_mult"]
    return [("qkv", 3 * d, d), ("proj", d, d), ("mlp0", m * d, d), ("mlp2", d, m * d)]


def weight_bytes(cfg: dict, w_mode: str) -> dict:
    """被量化 Linear 的权重字节（**整模型口径，不是每 token**）——与激活字节分列。

    `int8`：`out·in·1 B + out·2 B`（per-output-channel fp16 标尺）；
    `fp16`：`out·in·2 B`。embedding / pos_embed / dense 前向未执行的 gates
    **不计入**（README §3 明列）；被计入权重占全模型参数的比例由 `run.py` 从
    活模型报出（`bytes_account.weight_accounted_frac_of_all_params`）。
    """
    if w_mode not in W_MODES:
        raise ValueError(f"unknown weight mode {w_mode!r}")
    n_layers = cfg["model"]["n_layers"]
    bits = 16 if w_mode == "fp16" else _BITS[w_mode]
    total = 0.0
    n_params = 0
    per_tensor = {}
    for tag, out, inf in weight_tensor_shapes(cfg):
        n = out * inf
        b = n * bits / 8 + (0.0 if w_mode == "fp16" else out * SCALE_BYTES)
        per_tensor[tag] = {"out_features": out, "in_features": inf,
                           "n_params": n * n_layers, "bytes": round(b * n_layers, 6)}
        total += b * n_layers
        n_params += n * n_layers
    return {
        "w_mode": w_mode,
        "bits": bits,
        "n_params_quantized_linears": n_params,
        "bytes": round(total, 6),
        "mib": round(total / 2 ** 20, 6),
        "per_tensor": per_tensor,
        "excluded": ["embed.weight（tied unembed，V×d）", "pos_embed.weight",
                     "gates（dense 前向不执行）", "LayerNorm 参数"],
        "note": "整模型口径；不得与 per-token 激活字节相加成一条横轴。",
    }


# --------------------------------------------------------------- 前向（复刻 dense 分支）


class AqForward:
    """一个臂的前向：`Block.forward` 的 dense 分支 + 在每个 Linear 输入上做假量化。

    `activation_mode == "fp16"` 时逐位等于 `retention.masked_logits`（硬护栏 ②）。
    `weight_mode == "int8"` 时把每个 Linear 的权重换成它的假量化往返（缓存一次）。
    """

    def __init__(self, model, spec: dict):
        self.model = model
        self.spec = spec
        for key in ("activation_mode", "weight_mode"):
            if spec[key] not in (ACT_MODES if key == "activation_mode" else W_MODES):
                raise ValueError(f"bad {key}={spec[key]!r}")
        for blk in model.blocks:
            if type(blk) is not Block:
                raise ValueError(f"激活假量化只定义在 stock Block 上，遇到 "
                                 f"{type(blk).__name__}")
        self._wcache: dict[int, torch.Tensor] = {}
        self._wstats: dict | None = None
        if spec["weight_mode"] != "fp16":
            self._build_weight_cache()

    # -- 权重缓存（组合臂） --

    def _linears(self):
        for i, blk in enumerate(self.model.blocks):
            yield i, "qkv", blk.qkv
            yield i, "proj", blk.proj
            yield i, "mlp0", blk.mlp[0]
            yield i, "mlp2", blk.mlp[2]

    def _build_weight_cache(self) -> None:
        """把权重假量化往返**一次性**算好并缓存（前向里不再重复算）。"""
        acc = {"n_values": 0, "sum_abs_err": 0.0, "sum_abs_x": 0.0, "n_changed": 0,
               "max_abs_err": 0.0, "scale_min": None, "scale_max": None,
               "per_tensor": []}
        with torch.no_grad():
            for i, tag, mod in self._linears():
                w = mod.weight
                wq = fake_quant_weight(w, self.spec["weight_mode"])
                self._wcache[id(mod)] = wq
                err = (wq - w).abs()
                acc["n_values"] += w.numel()
                acc["sum_abs_err"] += float(err.sum())
                acc["max_abs_err"] = max(acc["max_abs_err"], float(err.max()))
                acc["sum_abs_x"] += float(w.abs().sum())
                acc["n_changed"] += int((wq != w).sum())
                s = weight_quant_scale(w, self.spec["weight_mode"])
                acc["scale_min"] = (float(s.min()) if acc["scale_min"] is None
                                    else min(acc["scale_min"], float(s.min())))
                acc["scale_max"] = (float(s.max()) if acc["scale_max"] is None
                                    else max(acc["scale_max"], float(s.max())))
                acc["per_tensor"].append({"layer": i, "tag": tag, "rel_err":
                                          float(err.sum() / w.abs().sum().clamp_min(1e-12)),
                                          "changed_frac": float((wq != w).float().mean())})
        n = max(acc["n_values"], 1)
        self._wstats = {
            "w_mode": self.spec["weight_mode"],
            "n_values": acc["n_values"],
            "rel_err": acc["sum_abs_err"] / max(acc["sum_abs_x"], 1e-12),
            "mean_abs_err": acc["sum_abs_err"] / n,
            "max_abs_err": acc["max_abs_err"],
            "changed_frac": acc["n_changed"] / n,
            "scale_min": acc["scale_min"], "scale_max": acc["scale_max"],
        }

    # -- 激活量化 + 活性量 --

    def _act(self, x: torch.Tensor, stats: dict | None, layer: int, tag: str):
        mode = self.spec["activation_mode"]
        # 唯一量化入口：fp16 档在 `fake_quant_activation` 里定义成逐位恒等，
        # 所有档都从这里过（「恒等」只有一个定义处）。
        dq, scale, ratio = fake_quant_activation(x, mode)
        if stats is not None:
            if mode == "fp16":
                _record_outlier_structure(x, stats, layer, tag)
            else:
                _record_activation(x, dq, scale, ratio, mode, stats, layer, tag)
        return dq

    # -- 前向 --

    def block_forward(self, blk, x: torch.Tensor, stats: dict | None, layer: int):
        B, T, d = x.shape
        qkv_in = blk.ln1(x)
        q, k, v = F.linear(self._act(qkv_in, stats, layer, "qkv"), self._w(blk.qkv),
                           blk.qkv.bias).chunk(3, dim=-1)

        def shape(t):
            return t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)

        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + F.linear(self._act(a, stats, layer, "proj"), self._w(blk.proj),
                         blk.proj.bias)
        h1 = F.linear(self._act(blk.ln2(x), stats, layer, "mlp0"), self._w(blk.mlp[0]),
                      blk.mlp[0].bias)
        g = F.gelu(h1)
        x = x + F.linear(self._act(g, stats, layer, "mlp2"), self._w(blk.mlp[2]),
                         blk.mlp[2].bias)
        return x

    def _w(self, module):
        if self.spec["weight_mode"] == "fp16":
            return module.weight
        return self._wcache[id(module)]

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor, stats: dict | None = None):
        h = self.model._input_states(tokens)
        for i, blk in enumerate(self.model.blocks):
            h = self.block_forward(blk, h, stats, i)
        return self.model.readout(h)

    def weight_activity(self):
        return self._wstats


# --------------------------------------------------------------- 活性量记录


def _record_outlier_structure(x: torch.Tensor, stats: dict, layer: int, tag: str) -> None:
    """fp16（恒等）档：只报激活本身的通道/ token 量级结构，无标尺、无误差。"""
    ab = x.abs()
    chan = ab.amax(dim=(0, 1)) if x.dim() == 3 else ab.amax(dim=0)
    tok = ab.amax(dim=-1)
    med = chan.median().clamp_min(1e-12)
    bits = 16.0
    idx = float(chan.amax() / med)
    stats["per_tensor"][f"L{layer:02d}.{tag}"] = {
        "mode": "fp16", "n_values": int(x.numel()), "n_scales": 0,
        "rel_err": 0.0, "mean_abs_err": 0.0, "changed_frac": 0.0, "clipped_frac": 0.0,
        "scale_min": None, "scale_median": None, "scale_max": None,
        "chan_amax_outlier_index": idx,
        "chan_amax_median": float(med),
        "token_amax_outlier_index": float(tok.amax() / tok.median().clamp_min(1e-12)),
        "revert_chan_frac_6x": float((chan > 6.0 * med).float().mean()),
        "effective_bits_median_channel": bits - math.log2(max(idx, 1.0)),
        "n_revert_channels": int((chan > 6.0 * med).sum()),
    }
    acc = stats["aggregate"]
    acc["n_values"] += int(x.numel())
    acc["n_scales"] += 0


def _record_activation(x, dq, scale, ratio, mode, stats, layer, tag) -> None:
    qmax = _qmax(mode)
    ab = x.abs()
    chan = ab.amax(dim=(0, 1)) if x.dim() == 3 else ab.amax(dim=0)
    tok = ab.amax(dim=-1)
    med = chan.median().clamp_min(1e-12)
    idx = float(chan.amax() / med)
    err = (dq - x).abs()
    bits = float(_BITS[mode])
    revert = chan > 6.0 * med
    stats["per_tensor"][f"L{layer:02d}.{tag}"] = {
        "mode": mode, "n_values": int(x.numel()), "n_scales": int(scale.numel()),
        "rel_err": float(err.sum() / ab.sum().clamp_min(1e-12)),
        "mean_abs_err": float(err.mean()),
        "changed_frac": float((dq != x).float().mean()),
        "clipped_frac": float((ratio.abs() > qmax).float().mean()),
        "scale_min": float(scale.min()), "scale_median": float(scale.median()),
        "scale_max": float(scale.max()),
        "chan_amax_outlier_index": idx,
        "chan_amax_median": float(med),
        "token_amax_outlier_index": float(tok.amax() / tok.median().clamp_min(1e-12)),
        "revert_chan_frac_6x": float(revert.float().mean()),
        "effective_bits_median_channel": bits - math.log2(max(idx, 1.0)),
        "n_revert_channels": int(revert.sum()),
    }
    acc = stats["aggregate"]
    acc["n_values"] += int(x.numel())
    acc["n_scales"] += int(scale.numel())
    acc["sum_abs_err"] += float(err.sum())
    acc["sum_abs_x"] += float(ab.sum())
    acc["n_changed"] += int((dq != x).sum())
    acc["max_abs_err"] = max(acc["max_abs_err"], float(err.max()))


def new_stats(spec: dict) -> dict:
    return {"activation_mode": spec["activation_mode"],
            "weight_mode": spec["weight_mode"],
            "prod_scale": ("identity" if spec["activation_mode"] == "fp16"
                           else "per_token"),
            "aggregate": {"n_values": 0, "n_scales": 0, "sum_abs_err": 0.0,
                          "sum_abs_x": 0.0, "n_changed": 0, "max_abs_err": 0.0},
            "per_tensor": {}}


def summarize_stats(stats: dict, n_layers: int, n_tensors_per_layer: int) -> dict:
    """把逐张量记录折成活性量总表（含逐层聚合与离群通道判定）。"""
    agg = stats["aggregate"]
    n = max(agg["n_values"], 1)
    per = stats["per_tensor"]
    agg_out = {
        "n_values": agg["n_values"],
        "n_scales": agg["n_scales"],
        "n_tensors_observed": len(per),
        "n_tensors_expected": n_layers * n_tensors_per_layer,
        "rel_err": agg["sum_abs_err"] / max(agg["sum_abs_x"], 1e-12),
        "mean_abs_err": agg["sum_abs_err"] / n,
        "max_abs_err": agg["max_abs_err"],
        "changed_frac": agg["n_changed"] / n,
        "scale_min": min((v["scale_min"] for v in per.values()
                          if v["scale_min"] is not None), default=None),
        "scale_max": max((v["scale_max"] for v in per.values()
                          if v["scale_max"] is not None), default=None),
    }
    # 逐层聚合
    per_layer = []
    for layer in range(n_layers):
        cell = {k.split(".", 1)[1]: v for k, v in per.items()
                if k.startswith(f"L{layer:02d}.")}
        if not cell:
            per_layer.append({"layer": layer, "tensors_observed": 0})
            continue
        ar = {"layer": layer, "tensors_observed": len(cell)}
        ar["scale_min_by_tensor"] = {t: c["scale_min"] for t, c in cell.items()}
        ar["scale_max_by_tensor"] = {t: c["scale_max"] for t, c in cell.items()}
        ar["changed_frac_by_tensor"] = {t: c["changed_frac"] for t, c in cell.items()}
        ar["rel_err_by_tensor"] = {t: c["rel_err"] for t, c in cell.items()}
        ar["outlier_index_channel_by_tensor"] = {
            t: c["chan_amax_outlier_index"] for t, c in cell.items()}
        ar["revert_chan_frac_6x_by_tensor"] = {
            t: c["revert_chan_frac_6x"] for t, c in cell.items()}
        ar["effective_bits_median_channel_by_tensor"] = {
            t: c["effective_bits_median_channel"] for t, c in cell.items()}
        ar["n_revert_channels_total"] = sum(c["n_revert_channels"] for c in cell.values())
        ar["max_outlier_index_channel"] = max(c["chan_amax_outlier_index"]
                                              for c in cell.values())
        ar["mean_outlier_index_channel"] = (
            sum(c["chan_amax_outlier_index"] for c in cell.values()) / len(cell))
        ar["min_effective_bits_median_channel"] = min(
            c["effective_bits_median_channel"] for c in cell.values())
        per_layer.append(ar)
    worst = max(per, key=lambda k: per[k]["chan_amax_outlier_index"])
    stats_out = {
        "activation_mode": stats["activation_mode"],
        "weight_mode": stats["weight_mode"],
        "prod_scale": stats["prod_scale"],
        "aggregate": agg_out,
        "worst_tensor_by_outlier_index": {
            "key": worst, "outlier_index_channel": per[worst]["chan_amax_outlier_index"],
            "effective_bits_median_channel":
                per[worst]["effective_bits_median_channel"],
            "revert_chan_frac_6x": per[worst]["revert_chan_frac_6x"]},
        "mean_outlier_index_channel": sum(
            v["chan_amax_outlier_index"] for v in per.values()) / max(len(per), 1),
        "mean_effective_bits_median_channel": sum(
            v["effective_bits_median_channel"] for v in per.values()) / max(len(per), 1),
        "revert_chan_frac_6x_mean": sum(
            v["revert_chan_frac_6x"] for v in per.values()) / max(len(per), 1),
        "n_revert_channels_total": sum(v["n_revert_channels"] for v in per.values()),
        "layers_with_any_revert_channel": sorted(
            {int(k.split(".")[0][1:]) for k, v in per.items() if v["n_revert_channels"] > 0}),
        "per_layer": per_layer,
        "per_tensor": per,
    }
    return stats_out


def revert_byte_penalty(stats_out: dict, cfg: dict) -> dict:
    """离群通道「回退到高精度」的字节代价（**反事实**：本实现不做回退，全通道整型）。

    若把 `n_revert_channels` 个通道按 LLM.int8() 式留在 fp16，每 token 每通道多花
    `(2 − bits/8)` B；把它除以标称省幅，得到「省幅被离群通道吃掉的比例」。
    """
    act_mode = stats_out["activation_mode"]
    n_layers = cfg["model"]["n_layers"]
    shapes = {t: f for t, f in activation_tensor_shapes(cfg)}
    # 每个张量的通道数 = in_features；回退通道数按 tag 汇总。
    n_revert = 0
    n_channels = 0
    per_tag_revert: dict[str, int] = {}
    for key, v in stats_out["per_tensor"].items():
        tag = key.split(".", 1)[1]
        n_revert += v["n_revert_channels"]
        (per_tag_revert.__setitem__(tag, per_tag_revert.get(tag, 0)
                                    + v["n_revert_channels"]))
    for tag, f in shapes.items():
        n_channels += f * n_layers
    if act_mode == "fp16":
        return {"applicable": False, "n_revert_channels_total": 0,
                "n_channels_total": n_channels}
    bits = _BITS[act_mode]
    extra_per_token = n_revert * (2.0 - bits / 8)
    nominal_saving = 55296.0 - activation_bytes_per_token(cfg, act_mode)[
        "bytes_per_token_activation"]
    return {
        "applicable": True,
        "n_revert_channels_total": n_revert,
        "n_channels_total": n_channels,
        "revert_chan_frac": n_revert / max(n_channels, 1),
        "revert_extra_bytes_per_token": extra_per_token,
        "nominal_saving_bytes_per_token": nominal_saving,
        "saving_eaten_frac": (extra_per_token / nominal_saving
                              if nominal_saving > 0 else None),
        "note": "反事实：回退**未实现**（本实验用全通道整型）；此列只回答『离群通道会不会"
                "迫使回退、会吃掉多少省幅』。",
    }
