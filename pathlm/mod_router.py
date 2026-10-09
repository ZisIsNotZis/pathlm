"""MoD 式逐 token 深度路由（R7「既有早退实现」对照臂）。

机制（MoD = Mixture-of-Depths，Raposo et al. 2024）：每个 transformer 块前放一个
**逐 token 二值路由器**，每步只让容量内的 top-k 个 token 进块、其余走残差旁路；
路由器用一条二值交叉熵辅助损失训练。**论文的具体系数/超参在本环境无法检索**
（`记忆·未核`），本模块按下面的机制式实装，并把每处与论文可能不同的选择写明。

实装约定（唯一变量 = 「token 级跳过」这一机制本身）：

- 基底 = 本仓 stock 无门 trunk（`DepthARModel(dense=True)` 的 `Block` 栈），
  块就是 `Block`（与无门/有门早退臂同一块类，只改路由）；
- 每块一个路由器 `nn.Linear(d, 1)`（新增参数 L·(d+1)）；logits → `sigmoid` 概率；
- 容量 = **逐序列** top-k：`k = max(1, round(mod_ratio · T))`（MoD 的 per-sequence
  capacity；全局 top-k 会让部分序列整段被丢，不采用）；
- 选中 token 收集成 `[B, k, d]` 进块；块内注意力 = **同序列 + 原位置因果**
  （被旁路 token 的 K/V 不进注意力）——这是与全稠密注意力唯一的语义差，也是 MoD
  的 routed-subset 注意力形式；
- 未选中 token 的状态**逐位不变**（残差旁路）；
- 损失 = 最终深度 CE（主目标，权重 1，不降权）+ `router_weight · router_loss`。

路由器损失（MoD §3.3 的形式，系数按「选中推向 1、未选中推向容量比例」实现；
论文原式的确切系数为 `记忆·未核`）：

    L_R = mean_{i∈S} (1 − p_i) + ratio · mean_{i∉S} p_i

即未选中 token 的概率只被推向容量比例（阈值附近），不压到 0——这正是 MoD
「路由器只学『比阈值更值』」的推导结果。

算力口径（解析，不靠墙钟）：每 token 的块作用次数 = `ratio · L`（每层恰好放行
`k` 个 token）；路由器开销 = L·d 乘加 / token（相对一个块的 ~4·d²·(1+mlp_mult)
乘加可忽略）。因此 `MoDModel(L=2n, ratio=0.5)` 与 `dense L=n` **isoFLOP**。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_ar import DepthARModel

LN2 = 0.6931471805599453


def mod_router_loss(prob: torch.Tensor, sel: torch.Tensor,
                    ratio: float) -> torch.Tensor:
    """MoD 式路由器二值损失（见模块文档）。`prob`/`sel` = [B,T]。"""
    m = sel.to(prob.dtype)
    n_sel = m.sum().clamp_min(1.0)
    n_un = (1.0 - m).sum().clamp_min(1.0)
    return (((1.0 - prob) * m).sum() / n_sel
            + ratio * (prob * (1.0 - m)).sum() / n_un)


class MoDModel(DepthARModel):
    """`DepthARModel(dense=True)` + 逐层容量路由。见模块文档。

    `forward` 的签名/返回与 `DepthARModel.forward` 兼容（`train_depth_ar.train`
    直接复用，训练配方逐字段相同）；`aux["gate_mean"]` 在 MoD 臂里装的是
    **逐层放行比例**（没有 carry 门可报），契约与结果里写明。
    """

    _implements_mod_router = True

    def __init__(self, cfg, vocab_size: int, router_weight: float = 0.01):
        if not cfg.dense:
            raise ValueError("MoDModel 只定义在 dense=True（stock 无门 trunk）上："
                             "路由与 carry 门/退出填充混开时无法单因素归因")
        super().__init__(cfg, vocab_size)
        if not 0.0 < cfg.mod_ratio <= 1.0:
            raise ValueError(f"mod_ratio 必须落在 (0, 1]，收到 {cfg.mod_ratio}")
        if router_weight < 0:
            raise ValueError(f"router_weight 必须 >= 0，收到 {router_weight}")
        self.router_weight = float(router_weight)
        # 路由器初始化：与其余 Linear 同款（std 0.02、bias 0），不消耗训练随机数
        # 之外的额外预算；构造顺序在**父类全部初始化之后** ⇒ 同 seed 下 trunk 权重
        # 与同参数量的 stock 臂逐位一致（单测钉住）。
        self.routers = nn.ModuleList(nn.Linear(cfg.d_model, 1)
                                     for _ in range(cfg.n_layers))
        for r in self.routers:
            nn.init.normal_(r.weight, std=0.02)
            nn.init.zeros_(r.bias)

    # ---- 前向（路由） ----
    def forward(self, tokens: torch.Tensor, exit_depths: torch.Tensor | None = None,
                targets: torch.Tensor | None = None,
                shallow_weight: float = 1.0, final_weight: float = 1.0,
                shallow_stopgrad: bool = False, dense_aux: bool = False):
        B, T = tokens.shape
        L = self.cfg.n_layers
        cap = max(1, int(round(self.cfg.mod_ratio * T)))
        h = self._input_states(tokens)
        states = [h]
        routed_frac: list[torch.Tensor] = []
        router_losses: list[torch.Tensor] = []
        fwd_counts = torch.zeros(B, T, device=tokens.device)   # 逐位置放行次数
        for k0, blk in enumerate(self.blocks):
            logits = self.routers[k0](h).squeeze(-1)           # [B,T]
            prob = torch.sigmoid(logits)
            idx = torch.topk(logits, cap, dim=1).indices        # [B,cap]（并列时取前者）
            sel = torch.zeros_like(prob, dtype=torch.bool)
            sel.scatter_(1, idx, True)
            d = h.shape[-1]
            h_sel = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))  # [B,cap,d]
            pos = idx                                              # [B,cap] 原位置
            # 因果掩码按**原位置**：同一序列内 j ≤ i 才可见（旁路 token 不可见）
            allow = pos.unsqueeze(-1) >= pos.unsqueeze(-2)          # [B,cap,cap]
            allow = allow.unsqueeze(1).expand(-1, self.cfg.n_heads, -1, -1)
            out, _ = blk(h_sel, attn_mask=allow)                    # [B,cap,d]
            h_next = h.clone()
            h_next.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, d), out)
            h = h_next
            states.append(h)
            routed_frac.append(sel.to(torch.float32).mean())
            router_losses.append(mod_router_loss(prob, sel, self.cfg.mod_ratio))
            fwd_counts = fwd_counts + sel.to(torch.float32)

        targets = tokens if targets is None else targets
        logits = self.readout(h, depth=L)
        ce = self._masked_ce(logits[:, :-1], targets[:, 1:],
                             torch.ones(B, T - 1, dtype=torch.bool,
                                        device=tokens.device))
        rl = torch.stack(router_losses).mean() if router_losses \
            else tokens.new_zeros(()).float()
        loss = ce + self.router_weight * rl
        aux = {"states": states, "depth_logits": [logits], "depth_ce": [ce],
               # gate_mean 槽位装「逐层放行比例」（MoD 臂无 carry 门）
               "gate_mean": routed_frac,
               "exit_depths": exit_depths,
               "mod": {"router_loss": rl, "routed_frac_per_layer":
                       [float(x) for x in routed_frac],
                       "router_weight": self.router_weight,
                       "capacity_per_seq": cap,
                       "fwd_counts_mean": fwd_counts.mean()}}
        return loss, aux


@torch.no_grad()
def mod_eval(model: MoDModel, eval_arr, n_batches: int = 20, batch_size: int = 16,
             device: str = "cuda") -> dict:
    """MoD 臂的 enwik8 电池：最终深度 bpc/acc + 路由活性量（解析算力账）。

    路由是**确定性**的（top-k 无采样）⇒ 逐层放行比例恒等于容量比例；报实测值
    与解析账两列，防「路由器是死的」。
    """
    from .data import batch
    T = model.cfg.seq_len
    L = model.cfg.n_layers
    gen = torch.Generator().manual_seed(4)
    nats = 0.0
    hits = 0
    n = 0
    frac_sum = torch.zeros(L, dtype=torch.float64)
    count_sum = 0.0
    p_sel = 0.0
    p_un = 0.0
    n_p = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        _, aux = model(x)
        lg = aux["depth_logits"][-1][:, :-1]
        tgt = x[:, 1:]
        nats += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                reduction="sum").item()
        hits += (lg.argmax(-1) == tgt).sum().item()
        n += tgt.numel()
        frac_sum += torch.tensor(aux["mod"]["routed_frac_per_layer"],
                                 dtype=torch.float64)
        count_sum += float(aux["mod"]["fwd_counts_mean"]) * x.numel()
        n_p += x.numel()
    ratio = model.cfg.mod_ratio
    return {
        "bpc": round(nats / n / LN2, 4),
        "acc": round(hits / n, 4),
        "routed_frac_per_layer": [round(float(v) / max(n_batches, 1), 6)
                                  for v in frac_sum],
        "routed_frac_mean": round(float(frac_sum.sum()) / (n_batches * L), 6),
        "capacity_frac": round(ratio, 6),
        "layer_apps_per_token_measured": round(count_sum / max(n_p, 1), 4),
        "layer_apps_per_token_analytic": round(ratio * L, 4),
        "router_params": sum(p.numel() for p in model.routers.parameters()),
        "router_weight": model.router_weight,
        "eval_batches": n_batches,
        "eval_batch_size": batch_size,
        "n_eval_positions": n,
    }
