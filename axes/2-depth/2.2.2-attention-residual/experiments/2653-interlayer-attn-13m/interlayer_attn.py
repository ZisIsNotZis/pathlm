"""2653 实验本地的层间 causal attention 聚合（实验内 monkey-patch，不改 `pathlm/**`）。

结构（相对 stock 单残差 `h_{k} = h_{k-1} + sublayer(h_{k-1})`）：

    agg_k = softmax_j( τ_k · ⟨q(h_{k-1}), k(h_j)⟩ / √d ) · v(h_j),  j = 0..k-1
    h_k   = Block_k(agg_k)                       # stock Block（pre-LN + 残差），不动

`states[0] = 输入嵌入`、`states[j] = 层 j 的输出`；层 k（1-index 深度 k）只允许看
`states[0..k−1]` ⇒ **层索引因果**（看不到当前层之后的信息）。`j = k−1` 是「紧邻上一层」，
也就是 stock 会喂给该 Block 的输入。

两条臂（实验唯一的变量 = 聚合的参数化）：

* `scalar`：q/k/v **恒等**（无学习投影），每层一个可学习标量温度 `τ_k`
  ⇒ 额外参数 = L（= 8；`dense` 的线性加权和是 +36，同量级 ⇒ 等参数对照）；
* `qkv`：同聚合，改成**可学习共享** `W_q/W_k/W_v`（`bias=False`，init = I）
  ⇒ 额外参数 = 3d² + L；init = I 时 `qkv` 的逐层权重与 `scalar` 逐位相同
  ⇒ 臂间起点的唯一差别是「多了可学投影」。

**等值门**（契约 §4-A）：`force_last=True` 时把层维权重**强制成 one-hot 到 `j = k−1`**，
并**走同一条聚合循环** ⇒ 前向与 stock 单残差逐位一致（`test_2653_interlayer_attn.py` 钉住）。
one-hot 下的聚合循环与引擎 `DepthARModel._layer_mix_input` 的逐项累加同款（`0·h_j` 相加
不改变数值），因此这里是**同一条代码路径**上的等值性，不是另开一条捷径。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pathlm.depth_ar import DepthARModel

#: 由 `run.py` / `smoke.py` 在建模型前设置（模块全局，工厂读它）。
MODE = "scalar"
#: 单测/等值门用：强制层维权重 one-hot 到紧邻上一层。
FORCE_LAST = False


class InterLayerAttentionModel(DepthARModel):
    """`topology="dense"` 的引擎前向路径 + 本地的层间注意力聚合。

    复用引擎的 dense 分支：`h, _ = blk(self._layer_mix_input(states, k))`
    （`pathlm/depth_ar.py` 的 `DepthARModel.forward`），本类只覆写
    `_layer_mix_input`（聚合函数）与参数表（删掉线性加权和的 α、换成注意力参数）。
    """

    def __init__(self, cfg, vocab_size: int):
        if cfg.topology != "dense":
            raise ValueError("InterLayerAttentionModel 只在 topology='dense' 下可用"
                             f"（收到 {cfg.topology!r}）")
        # 臂 / 等值门模式在建实例时**冻结**（模块全局只是工厂的入参：若在前向时读
        # 全局，同一个已建模型会在建下一个实例后悄悄换行为，等值门就测不到东西）。
        self.mode = MODE
        self.force_last = FORCE_LAST
        super().__init__(cfg, vocab_size)
        d, L = cfg.d_model, cfg.n_layers
        # 删掉线性加权和的层权参数（本臂不用它；留着会把 +36 混进额外参数账）
        del self.layer_mix
        # 每层一个可学习标量温度（init 1.0 = 标准 1/√d 缩放；不消耗随机数流
        # ⇒ trunk / gates 的初始化与 stock 臂逐位相同）
        self.tau = nn.Parameter(torch.ones(L))
        if self.mode == "qkv":
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
            with torch.no_grad():          # init = I ⇒ 起点与 scalar 臂同权重
                for w in (self.wq, self.wk, self.wv):
                    w.weight.copy_(torch.eye(d))
        elif self.mode != "scalar":
            raise ValueError(f"unknown MODE {self.mode!r}")
        #: 逐层注意力权重遥测（活性量；前向时追加，调用方 flush）
        self.attn_tele: list[dict] = []

    # ------------------------------------------------------------ 聚合
    def _layer_mix_input(self, states: list[torch.Tensor], k: int) -> torch.Tensor:
        """层 k（深度 k，1-index）的输入 = 对 `states[0..k−1]` 的 causal attention 聚合。"""
        H = torch.stack(states[:k], dim=2)          # [B, T, k, d]
        cur = H[:, :, k - 1]                        # 紧邻上一层输出（stock 的输入）
        if self.force_last:
            w = torch.zeros(H.shape[:3], dtype=H.dtype, device=H.device)
            w[..., k - 1] = 1.0                     # one-hot ⇒ agg == states[k−1]
            V = H
        else:
            if self.mode == "scalar":
                q, K, V = cur, H, H                 # q/k/v 恒等（无学习投影）
            else:
                q, K, V = self.wq(cur), self.wk(H), self.wv(H)
            scores = torch.einsum("btd,btjd->btj", q, K) * (self.cfg.d_model ** -0.5)
            scores = scores.float() * self.tau[k - 1].float()
            w = torch.softmax(scores, dim=-1).to(H.dtype)
            self.attn_tele.append({
                "layer": k,
                "w": w.detach().float(),
                "w_immediate_prev": float(w[..., k - 1].detach().float().mean()),
                "w_max": float(w.detach().float().amax(dim=-1).mean()),
                "entropy": float(-(w.detach().float()
                                   * w.detach().float().clamp_min(1e-12).log())
                                 .sum(-1).mean()),
            })
        # 逐项累加（与引擎 `_layer_mix_input` 的 dense 臂同款；one-hot 时逐位等于
        # states[k−1] ⇒ 等值门成立）
        out = w[..., 0:1] * V[:, :, 0]
        for j in range(1, k):
            out = out + w[..., j:j + 1] * V[:, :, j]
        return out

    # ------------------------------------------------------------ 代价（等效理论量）
    def aggregation_cost(self) -> dict:
        """层间聚合的每 token 额外 MACs（解析，不含任何时间量）。"""
        d, L = self.cfg.d_model, self.cfg.n_layers
        stock_attn = L * (4 * d * d + 2 * self.cfg.seq_len * d)   # stock 每层注意力
        agg = 0
        for k in range(1, L + 1):
            q_proj = 0 if self.mode == "scalar" else d * d        # W_q 只对 cur
            k_proj = 0 if self.mode == "scalar" else k * d * d    # W_k 对 k 个源
            v_proj = 0 if self.mode == "scalar" else k * d * d    # W_v 对 k 个源
            agg += (q_proj + k_proj + v_proj    # 投影
                    + k * d                     # ⟨q, k_j⟩
                    + k * d)                    # Σ w_j v_j
        return {"mode": self.mode, "extra_macs_per_token_interlayer": agg,
                "base_macs_per_token_attention": stock_attn,
                "extra_over_base_attention": round(agg / stock_attn, 6),
                "extra_params": self.cfg.n_layers + (
                    3 * d * d if self.mode == "qkv" else 0),
                "note": ("等效理论量：MACs/token（乘加）。qkv 的额外项 = d²·L + 2d²·Σk "
                         "+ 2d·Σk（W_q 每层对 cur 一次 = d²·L；W_k/W_v 每层对 k 个源各一次 "
                         "= 2d²·Σk；⟨q, k_j⟩ 与 Σ w_j v_j 各 k·d = 2d·Σk），d=256 / L=8 / "
                         "Σk=36 ⇒ 5 261 312；scalar 无投影，额外项只有 2d·Σk = 18 432。"
                         "两列分开报，不用任何时间量。"
                         "tokens-per-forward = 1（每 token 每层一次聚合）。")}

    # ------------------------------------------------------------ 活性（SCHEMA §10）
    def topology_liveness(self) -> dict:
        """本臂的层间聚合不是线性层权（构造时已删 `layer_mix`）⇒ 由本类自报活性：
        温度 τ 的终值，以及有真实前向样本时逐层聚合权重的摘要。

        共享入口 `pathlm.depth_ar.topology_liveness` 只转调本方法；若不覆写，它会
        对 `layer_mix` 抛错（2653 曾在落盘阶段因此崩 781 次）。

        - `params_added` 相对 sequential：scalar = L（每层一个 τ），qkv = 3d² + L；
        - `alive` = 某层 τ 偏离初值 1.0：τ 不动 ⇔ 层间聚合停在未学习的均匀
          softmax 上，干预没学；
        - `attn_weights` = `attn_tele` 里 k ≥ 2 的层的汇总（`run.py::
          attention_liveness` 先 flush 再真实前向）；无样本时为 None。
        """
        d, L = self.cfg.d_model, self.cfg.n_layers
        tau = self.tau.detach().float()
        dev = (tau - 1.0).abs()
        out = {
            "topology": self.cfg.topology,
            "aggregation": f"interlayer-causal-attention/{self.mode}",
            "params_added": L + (3 * d * d if self.mode == "qkv" else 0),
            "parameterized": True,
            "params_identical_to_sequential": False,
            "start_bitwise_equals_sequential": False,
            "tau_by_layer": [round(float(v), 6) for v in tau],
            "tau_absmax_dev_from_init": round(float(dev.max()), 6),
            "tau_frac_layers_moved": round(float((dev > 0).float().mean()), 6),
        }
        deg = [t for t in self.attn_tele if t["layer"] >= 2]
        if deg:
            prev = [t["w_immediate_prev"] for t in deg]
            ent = [t["entropy"] for t in deg]
            out["attn_weights"] = {
                "n_samples": len(deg),
                "w_immediate_prev_mean": round(sum(prev) / len(prev), 6),
                "w_immediate_prev_min": round(min(prev), 6),
                "entropy_mean": round(sum(ent) / len(ent), 6),
            }
        else:
            out["attn_weights"] = None
        out["alive"] = out["tau_absmax_dev_from_init"] > 0
        return out
