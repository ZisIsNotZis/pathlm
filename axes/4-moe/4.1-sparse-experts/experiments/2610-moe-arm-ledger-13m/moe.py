"""最小 top-k 稀疏激活专家（MoE）FFN —— 轴 4 实验用，**不动共享 `pathlm/**`**。

设计约束（`harness/SCHEMA.md` §10 / §22.7 / §15）：

- `TopKMoEFFN` 与 stock `pathlm.model.Block.mlp`（`nn.Sequential(Linear(d,H),GELU,
  Linear(H,d),Dropout(0))`）**接口相同**：`[B,T,d] → [B,T,d]`，因此可以**透明替换**
  `Block.mlp`——`Block.forward` 内部那一行 `x = x + self.mlp(self.ln2(x))` 完全不用改。
- E=1 且 top_k=1 时，MoE 输出 **逐位等于** 该唯一专家的输出；把 stock mlp 权重灌进
  该专家即得与 dense 臂**逐位一致**的前向（pre-flight 断言，见 `run.py preflight`）。
- 路由是**确定性 top-k**（无采样）⇒ 每 token 激活 k 个专家，每 token FFN 等效算力 =
  k ×（单专家），与 dense 的比较口径在 `run.py ledger` 里算死。

单因素边界（`docs/structure.md` §4）：

- `aux_weight > 0` 属于 **4.2（路由与负载均衡）**；4.1（top-k 本体）必须 `aux_weight=0`。
- 细粒度 / 共享专家属于 **4.3**；本模块只提供均匀专家（每专家同宽），4.3 在配置层再改
  `n_experts / expert_hidden`。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlm.depth_ar import DepthARModel


class TopKMoEFFN(nn.Module):
    """E 个同宽专家 + 一个 d×E 路由器；每 token 走 top-k。

    参数量：E·(2·d·h + h + d)（专家）+ E·d（路由器，缺省无 bias）。
    每 token FFN MACs：k·(2·d·h)（专家）+ E·d（路由 logits）+ E（softmax 归一，忽略）。
    """

    def __init__(self, d: int, n_experts: int, top_k: int, expert_hidden: int,
                 dropout: float = 0.0, renorm: bool = False,
                 aux_weight: float = 0.0):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts 必须 >= 1，收到 {n_experts}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k 必须落在 [1, n_experts]，收到 {top_k}/{n_experts}")
        if expert_hidden < 1:
            raise ValueError(f"expert_hidden 必须 >= 1，收到 {expert_hidden}")
        if aux_weight < 0:
            raise ValueError(f"aux_weight 必须 >= 0，收到 {aux_weight}")
        self.d = d
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_hidden = expert_hidden
        self.renorm = bool(renorm)
        self.aux_weight = float(aux_weight)
        # 专家创建顺序 = 顺序创建；与 stock mlp 同款子模块（Linear→GELU→Linear→Dropout）
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, expert_hidden), nn.GELU(),
                          nn.Linear(expert_hidden, d), nn.Dropout(dropout))
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(d, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        # 最近一次前向的活性量（`harness/SCHEMA.md` §10；评测/训练日志用）
        self.last_stats: dict | None = None

    # ---- 干预活性量（§10） ----
    @torch.no_grad()
    def liveness(self) -> dict:
        """专家使用率分布（活性，防坍缩）：熵 / 归一化熵 / 最大占比。"""
        if self.last_stats is None:
            return {"usage": [], "usage_entropy": None, "usage_entropy_norm": None,
                    "usage_max_frac": None, "router_params": 0}
        usage = self.last_stats["usage"]
        p = usage / usage.sum().clamp_min(1.0)
        eps = 1e-12
        ent = float(-(p * (p + eps).log()).sum())
        import math
        ent_max = math.log(self.n_experts)
        return {
            "usage": [int(x) for x in usage.tolist()],
            "usage_entropy": round(ent, 6),
            "usage_entropy_norm": round(ent / ent_max, 6),
            "usage_max_frac": round(float(p.max()), 6),
            "router_params": sum(p_.numel() for p_ in self.router.parameters()),
        }

    def aux_loss(self) -> torch.Tensor | None:
        """Switch 式负载均衡 aux loss（**4.2 用**；4.1 必须 aux_weight=0）。

        `L_aux = E · Σ_e f_e · P_e`，`f_e` = 分配到专家 e 的 token 比例，
        `P_e` = 路由器对专家 e 的平均概率（Switch §2.2 形式，系数按机制式实装）。
        """
        if self.aux_weight <= 0 or self.last_stats is None:
            return None
        st = self.last_stats
        f = st["usage"] / st["usage"].sum().clamp_min(1.0)
        p = st["aux_mean_prob"]          # 保留计算图：aux loss 必须能反传到路由器
        return self.aux_weight * self.n_experts * (f * p).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        logits = self.router(flat)                     # [N, E]
        probs = logits.softmax(-1)                     # [N, E]
        topv, topi = probs.topk(self.top_k, dim=-1)    # [N, k]
        if self.renorm:
            topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            sel = (topi == e)                          # [N, k] bool
            row = sel.any(-1)                          # [N]
            if not bool(row.any()):
                continue
            w = (topv * sel).sum(-1)                   # [N]
            y = expert(flat[row])                      # [n_sel, d]
            out[row] = out[row] + w[row].unsqueeze(-1) * y
        usage = torch.zeros(self.n_experts, device=flat.device)
        usage.index_add_(0, topi.reshape(-1),
                         torch.ones(topi.numel(), device=flat.device))
        self.last_stats = {"usage": usage, "probs": probs.detach(),
                           "logits": logits.detach(),
                           # aux loss 用：保留计算图（否则 aux 收不到路由器梯度）
                           "aux_mean_prob": probs.mean(0)}
        return out.reshape(B, T, d)


class MoEDepthARModel(DepthARModel):
    """stock `DepthARModel(dense=True)` + 把每层 `Block.mlp` 换成 `TopKMoEFFN`。

    只改 FFN 的**专家化**这一个因素：注意力 / LN / embedding / 读出 / 残差接线全不动，
    层数 L、每 token 激活专家数 k、每专家宽 h 由配置给。
    """

    def __init__(self, cfg, vocab_size: int, n_experts: int, top_k: int,
                 expert_hidden: int, renorm: bool = False, aux_weight: float = 0.0):
        if not cfg.dense:
            raise ValueError("MoEDepthARModel 只定义在 dense=True（stock 无门 trunk）上："
                             "路由与退出/填充混开时无法单因素归因")
        super().__init__(cfg, vocab_size)
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_hidden = expert_hidden
        for blk in self.blocks:
            if not hasattr(blk, "mlp"):
                raise TypeError(f"块类 {type(blk).__name__} 没有 .mlp，无法作 MoE 替换")
            blk.mlp = TopKMoEFFN(cfg.d_model, n_experts, top_k, expert_hidden,
                                 dropout=cfg.dropout, renorm=renorm,
                                 aux_weight=aux_weight)

    def moe_modules(self) -> list[TopKMoEFFN]:
        return [blk.mlp for blk in self.blocks]

    def moe_aux_loss(self) -> torch.Tensor:
        """所有层 aux loss 之和（4.2 用；4.1 时全为 None ⇒ 返回 0）。"""
        terms = [m.aux_loss() for m in self.moe_modules()]
        terms = [t for t in terms if t is not None]
        if not terms:
            return torch.zeros((), dtype=torch.float32)
        return torch.stack(terms).sum()

    @torch.no_grad()
    def moe_liveness(self) -> dict:
        """逐层 + 全局的专家使用率活性量（§10）。"""
        per_layer = [m.liveness() for m in self.moe_modules()]
        usages = [m.last_stats["usage"].detach().double()
                  for m in self.moe_modules() if m.last_stats is not None]
        if not usages:
            return {"per_layer": per_layer, "global": {}}
        g = torch.stack(usages).sum(0)
        p = g / g.sum().clamp_min(1.0)
        import math
        ent = float(-(p * (p + 1e-12).log()).sum())
        return {"per_layer": per_layer,
                "global": {"usage": [int(x) for x in g.tolist()],
                           "usage_entropy": round(ent, 6),
                           "usage_entropy_norm": round(ent / math.log(self.n_experts), 6),
                           "usage_max_frac": round(float(p.max()), 6),
                           "top_k": self.top_k, "n_experts": self.n_experts}}


def copy_shared_from(dense: DepthARModel, moe: MoEDepthARModel) -> list[str]:
    """把 dense 臂里**非 FFN** 的参数/缓冲逐位拷进 MoE 臂（注意力/LN/embedding 同源）。

    返回 MoE 侧多出来、dense 侧没有的**非 mlp** 键（应为空 ⇒ 单因素：只有 FFN 专家化）。
    """
    dsd = {k: v for k, v in dense.state_dict().items() if ".mlp." not in k}
    msd = moe.state_dict()
    to_load = {k: v for k, v in dsd.items() if k in msd}
    moe.load_state_dict(to_load, strict=False)
    extra = [k for k in msd if ".mlp." not in k and k not in to_load]
    return sorted(extra)
