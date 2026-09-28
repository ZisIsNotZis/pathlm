"""深度自适应 AR（方向修订后的新形态，mental_model §0 / ticket 23）。

纯 AR（无 corruption / 无 retry / 无 mask 站），渐进精化 + 早退 + 自 spec：

- **全深度并行监督**：depth 0..L 每层都有 tied-unembed fp32 读出
  （depth 0 = embedding 直连读出），各深度 CE 等权平均（v1；ticket 23b 起
  支持浅层 aux 降权 `shallow_weight`，最终深度恒权重 1）。
- **逐通道携带门**：h_{k+1} = h_k + g_k ⊙ Refine_k(h_k)，
  g_k = σ(Gate_k(h_k))，Gate 偏置负初始化（carry：g≈0，恒等起步）。
- **随机退出深度（训练）**：每个 position 采样 d_i ∈ {0..L}。位置 i 在
  层 k ≤ d_i 正常精化；k > d_i 后状态冻结在 h_{d_i}[i]，且它在层 k 的
  KV = 层 k 的 kv 投影作用于冻结表征（ln1 → W_qkv 的 k/v 半边）——
  混合深度 context。fill 语义由 tests/test_depth_ar.py 钉住。
- **推理提交**：soft（Σp·E[t]，默认）/ hard（argmax→embed）/
  latent（ablation：直通退出深度状态，预期漂移）。

对照 arm：dense=True 退化为普通 AR transformer（无门/无退出/仅最终 CE），
即 dense-B 参照。训练入口 train_depth_ar.py。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import Block, cap_norm

LN2 = 0.6931471805599453  # nats → bits


@dataclass
class DepthARConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    seq_len: int = 256
    mlp_mult: int = 4
    dropout: float = 0.0
    # 输入侧 norm cap（与 PathLM 约定一致；只 cap embedding 输入，不 cap 残差流）
    norm_cap: float = 1.0
    # carry 门偏置初始化：σ(-2)≈0.12 起步（恒等近似 + 梯度可流）；
    # 测试用 -30 验证精确恒等极限。
    gate_bias_init: float = -2.0
    # 对照 arm：纯 dense AR（无门 / 无退出采样 / 仅最终深度 CE）
    dense: bool = False
    # 混合深度 context 填充策略（ticket 23e）：True = v1 fill（已退出行 KV 用
    # 冻结表征经本层 kv 投影填充，W0 复现点）；False = no-fill ragged（早退
    # 位置在未跑层的 KV 中缺席，原引擎 per-layer cache 语义，decode.Decoder 同款）
    fill_kv: bool = True
    # proj-fill 臂（ticket 23e 可选）：fill 基础上给已退出行 KV 源加逐层线性
    # adapter（恒等初始化——起步严格等于 fill，remap 从 fill 出发学）。需 fill_kv
    proj_fill: bool = False


def sample_exit_depths(B: int, T: int, n_layers: int, generator: torch.Generator,
                       device: str = "cpu") -> torch.Tensor:
    """训练时均匀采样退出深度 d_i ∈ {0..L}（契约：均匀分布起步）。"""
    return torch.randint(0, n_layers + 1, (B, T), generator=generator,
                         device=device)


class DepthARModel(nn.Module):
    """深度自适应 AR。forward(tokens, exit_depths) → (loss, aux)。

    exit_depths=None 表示全深度（无退出）——评测 depth 曲线与 dense 语义。
    aux: states[k]（depth k 表征）、depth_logits[k]、depth_ce[k]、
    gate_mean[k]、exit_depths。
    """

    def __init__(self, cfg: DepthARConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        d = cfg.d_model
        # Tied E=U：embedding 行 norm ~1（init std = d^-0.5），U = E^T（fp32 头）
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=d ** -0.5)
        self.pos_embed = nn.Embedding(cfg.seq_len, d)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        mcfg = ModelConfig(d_model=cfg.d_model, n_layers=cfg.n_layers,
                           n_heads=cfg.n_heads, seq_len=cfg.seq_len,
                           mlp_mult=cfg.mlp_mult, dropout=cfg.dropout,
                           norm_cap=cfg.norm_cap)
        self.blocks = nn.ModuleList(Block(mcfg) for _ in range(cfg.n_layers))
        # 逐通道携带门：gates[k] 作用于 depth k → k+1 的精化 delta
        self.gates = nn.ModuleList(nn.Linear(d, d) for _ in range(cfg.n_layers))
        for g in self.gates:
            nn.init.normal_(g.weight, std=0.02)
            nn.init.constant_(g.bias, cfg.gate_bias_init)
        gate_ids = {id(g) for g in self.gates}
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in gate_ids:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # proj-fill 逐层 KV 源 adapter：恒等初始化（不消耗随机数，同 seed 下
        # 其余参数初始化与 fill 臂逐位一致），起步 == fill（单测钉住）
        if cfg.proj_fill:
            self.kv_adapters = nn.ModuleList(
                nn.Linear(d, d) for _ in range(cfg.n_layers))
            for a in self.kv_adapters:
                nn.init.zeros_(a.bias)
                with torch.no_grad():
                    a.weight.copy_(torch.eye(d))

    # ---------- 读出 ----------

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """Tied unembed，fp32 头（autocast 之外强制 fp32）。"""
        with torch.autocast("cuda", enabled=False):
            return h.float() @ self.embed.weight.float().T

    def _input_states(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        h = self.embed(tokens) + self.pos_embed.weight[:T]
        return cap_norm(h, self.cfg.norm_cap)

    # ---------- forward / loss ----------

    def forward(self, tokens: torch.Tensor, exit_depths: torch.Tensor | None = None,
                targets: torch.Tensor | None = None,
                shallow_weight: float = 1.0,
                shallow_stopgrad: bool = False,
                dense_aux: bool = False):
        """tokens [B,T] = 干净输入流（无 corruption）；targets 缺省 = tokens
        本身，标准 AR 移位在 loss 内完成（行 i 的 depth-k 读出预测 t_{i+1}，
        仅当该行确实到达 depth k，即 d_i ≥ k）。

        shallow_weight（ticket 23b）：浅层深度 0..L-1 的 CE 权重，最终深度恒
        权重 1；总损失按 (w·L + 1) 归一保持量级。w=1 严格还原 v1 等权平均
        （W0 复现点）；dense 臂只有最终深度 CE，不受此参数影响。

        shallow_stopgrad（ticket 23c，H3）：浅层（depth 0..L-1）读出前对状态
        detach——浅层 CE 只训练 tied U 读出头，梯度不穿 trunk（纯读出）；
        损失值不变，只改梯度通道。

        dense_aux（ticket 23c，H1）：dense 臂开启全深度并行监督（L+1 个 CE
        分量，梯度穿 trunk）；缺省 False 保持 dense 臂仅最终深度 CE。"""
        B, T = tokens.shape
        L = self.cfg.n_layers
        h = self._input_states(tokens)
        states = [h]
        gate_means: list[torch.Tensor] = []
        if not self.cfg.dense:
            if self.cfg.proj_fill and not self.cfg.fill_kv:
                raise ValueError("proj_fill requires fill_kv=True")
            if exit_depths is None:
                exit_depths = torch.full((B, T), L, dtype=torch.long,
                                         device=tokens.device)
            frozen = h
            for k in range(1, L + 1):
                active = exit_depths >= k                       # [B,T]
                g = torch.sigmoid(self.gates[k - 1](h))
                if self.cfg.fill_kv:
                    # 混合深度 KV 源：活跃行用当前表征，已退出行用冻结表征
                    # （proj-fill 臂：冻结表征先经本层线性 adapter remap）
                    kv_src = frozen
                    if self.cfg.proj_fill:
                        kv_src = self.kv_adapters[k - 1](frozen)
                    kv_in = torch.where(active.unsqueeze(-1), h, kv_src)
                    out, _ = self.blocks[k - 1](h, kv_from=kv_in, gate_vec=g)
                else:
                    # no-fill ragged：早退位置在本层 KV 中缺席（原引擎语义，
                    # decode.Decoder per-layer cache 同款）——只有因果键 ∩
                    # 已达层 k 的行参与注意力，无填充。掩码显式 expand 到各
                    # head（CUDA SDPA 内核不广播 bool 掩码的 head 维）
                    mask = self._ragged_mask(exit_depths, k)
                    mask = mask.expand(-1, self.cfg.n_heads, -1, -1)
                    out, _ = self.blocks[k - 1](h, attn_mask=mask,
                                                gate_vec=g)
                h_next = torch.where(active.unsqueeze(-1), out, frozen)
                # 本层退出的位置在此冻结（其 depth-k 表征定格）
                frozen = torch.where((exit_depths == k).unsqueeze(-1),
                                     h_next, frozen)
                h = h_next
                states.append(h)
                gate_means.append(g.detach().float().mean())
        else:  # dense 对照：无门、无退出，普通残差堆叠
            for blk in self.blocks:
                h, _ = blk(h)
                states.append(h)

        targets = tokens if targets is None else targets
        sg = bool(shallow_stopgrad)
        n_states = len(states)
        # H3 stop-grad：浅层读出走 detached 状态（梯度不穿 trunk）；最终深度不变
        depth_logits = [self.readout(s.detach() if sg and k < n_states - 1 else s)
                        for k, s in enumerate(states)]
        # H1 dense+aux：dense 臂也监督全部深度；缺省 dense 仅最终深度
        readouts = depth_logits if (not self.cfg.dense or dense_aux) else depth_logits[-1:]
        loss = tokens.new_zeros(()).float()
        depth_ce: list[torch.Tensor] = []
        w = float(shallow_weight)
        for k, logits in enumerate(readouts):
            if self.cfg.dense:
                mask = torch.ones(tokens.shape[0], tokens.shape[1] - 1,
                                  dtype=torch.bool, device=tokens.device)
            else:
                mask = exit_depths[:, :-1] >= k  # 预测行必须真正到达 depth k
            ce = self._masked_ce(logits[:, :-1], targets[:, 1:], mask)
            depth_ce.append(ce)
            # aux 加权（ticket 23b）：浅层深度乘 w，最终深度恒 1
            wk = 1.0 if k == len(readouts) - 1 else w
            loss = loss + wk * ce
        loss = loss / (w * (len(readouts) - 1) + 1.0)
        aux = {"states": states, "depth_logits": depth_logits, "depth_ce": depth_ce,
               "gate_mean": gate_means, "exit_depths": exit_depths}
        return loss, aux

    @staticmethod
    def _masked_ce(logits: torch.Tensor, targets: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """掩码行平均 CE；掩码全空时贡献 0（不产生 NaN）。"""
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             targets.reshape(-1), reduction="none")
        m = mask.reshape(-1).float()
        return (ce * m).sum() / m.sum().clamp_min(1.0)

    def _ragged_mask(self, exit_depths: torch.Tensor, k: int) -> torch.Tensor:
        """层 k 的 no-fill 注意力掩码 [B,1,T,T]（bool，True=允许）：
        允许 = 因果（j ≤ i）且 key 行已到达层 k（d_j ≥ k）。对角恒开——
        全掩码行（必为已退出行，输出本就被丢弃）softmax 全 -inf 会产生 NaN
        并经共享 K/V 梯度扩散；开对角对其余行语义零影响。"""
        T = exit_depths.shape[1]
        device = exit_depths.device
        causal = torch.ones(1, 1, T, T, dtype=torch.bool,
                            device=device).tril()               # [1,1,T,T]
        key_reached = exit_depths[:, None, None, :] >= k         # [B,1,1,T]
        allow = causal & key_reached                             # [B,1,T,T]
        allow = allow | torch.eye(T, dtype=torch.bool,
                                  device=device)
        return allow

    def _gated_step(self, k0: int, h: torch.Tensor) -> torch.Tensor:
        """层 k0（0-index）的带门精化：h + g ⊙ delta，KV 取 ln1(h)
        （全深度语义，kv_in == h）。探针/解码复用同一精化定义。"""
        out, _ = self.blocks[k0](h)
        g = torch.sigmoid(self.gates[k0](h))
        return h + g * (out - h)


# ---------- 探针 ----------

@torch.no_grad()
def depth_curve(model: DepthARModel, eval_arr, n_batches: int = 20,
                batch_size: int = 16, generator: torch.Generator | None = None,
                device: str = "cpu") -> list[dict]:
    """全深度前向（无退出）的逐深度 bpc + next-token acc：depth-bpc 曲线。
    单调递减 = 精化成立。"""
    from .data import batch
    generator = generator or torch.Generator().manual_seed(2)
    T = model.cfg.seq_len
    nats = [0.0] * (model.cfg.n_layers + 1)
    hits = [0] * (model.cfg.n_layers + 1)
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        for k, lg in enumerate(aux["depth_logits"]):
            lg = lg[:, :-1]
            tgt = x[:, 1:]
            nats[k] += F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                       tgt.reshape(-1), reduction="sum").item()
            hits[k] += (lg.argmax(-1) == tgt).sum().item()
        n += x[:, 1:].numel()
    return [{"depth": k, "bpc": round(nats[k] / n / LN2, 4),
             "acc": round(hits[k] / n, 4)} for k in range(len(nats))]


@torch.no_grad()
def tf_acceptance(model: DepthARModel, eval_arr, n_batches: int = 20,
                  batch_size: int = 16, generator: torch.Generator | None = None,
                  device: str = "cpu") -> list[dict]:
    """出口草稿 + 全栈验证的接受率（teacher-forcing 代理）：全深度前向后，
    depth k 的 argmax 是否等于 depth L 的 argmax（greedy 自 spec 的草稿
    命中率代理；同一 token 被两个深度独立预测）。"""
    from .data import batch
    generator = generator or torch.Generator().manual_seed(3)
    T = model.cfg.seq_len
    L = model.cfg.n_layers
    agree = [0] * L
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        full = aux["depth_logits"][-1][:, :-1].argmax(-1)
        for k in range(L):
            draft = aux["depth_logits"][k][:, :-1].argmax(-1)
            agree[k] += (draft == full).sum().item()
        n += full.numel()
    return [{"depth": k, "accept": round(agree[k] / n, 4)} for k in range(L)]


@torch.no_grad()
def workspace_probe(model: DepthARModel, eval_arr, n_batches: int = 4,
                    batch_size: int = 8, eps_frac: float = 0.05,
                    delta: float = 0.01, device: str = "cpu",
                    generator: torch.Generator | None = None) -> dict:
    """workspace 探针（ticket 23 判据 2 的可操作化）。

    - **有效读出秩**：logit-Jacobian = tied U（[V,d]）；SVD 奇异值 > 1%
      峰值计为有效秩。null 空间 = d − rank 维 = 读出不可见的携带余量
      （d<V 时结构性为 0）。
    - **null 能量占比**：各深度状态能量落在 null(U) 的比例（模型是否
      在用携带余量存信息）。
    - **扰动敏感度（功能性判据）**：把 mid-depth 状态 h_mid 沿每个 null
      方向 ±ε（ε = 5% 行范数），重跑后半栈，测 depth-L logits 的中位
      |Δ|；> δ 的方向计为 functional（携带信息确实被后续计算读走）。
      同方向**当前行** depth-mid logits 的变化 ≈ 0 是数值对照（null
      正交性自检）。

    判据：sanity-B（d>V，null 维 > 0）workspace 维数 > sanity-C（d<V，
    结构性 0）。
    """
    from .data import batch
    U = model.embed.weight.detach().float()          # [V, d]
    V, d = U.shape
    _, S, Vt = torch.linalg.svd(U, full_matrices=True)
    sv = S[: min(V, d)]
    rank = int((sv > sv[0] * 0.01).sum().item())
    null_dim = d - rank
    Nb = Vt[rank:]                                    # [null_dim, d]，正交规范
    L = model.cfg.n_layers
    k_mid = L // 2
    generator = generator or torch.Generator().manual_seed(4)
    model.eval()
    energy = [[] for _ in range(L + 1)]
    dir_medians: list[float] = []
    control = float("inf")
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        if null_dim == 0:
            for k in range(L + 1):
                energy[k].append(0.0)
            continue
        for k in range(L + 1):
            h = aux["states"][k].float()
            null_part = (h @ Nb.T) @ Nb
            frac = (null_part.norm(dim=-1) ** 2) / h.norm(dim=-1).pow(2).clamp_min(1e-12)
            energy[k].append(frac.mean().item())
        h_mid = aux["states"][k_mid].float()
        logits_mid = model.readout(aux["states"][k_mid])
        logits_L = model.readout(aux["states"][-1])
        eps_rows = eps_frac * h_mid.norm(dim=-1, keepdim=True)     # [B,T,1]
        for j in range(null_dim):
            n_j = Nb[j]                                            # [d]
            for sign in (+1.0, -1.0):
                h_p = h_mid + sign * eps_rows * n_j
                hh = h_p
                for gi in range(k_mid, L):
                    hh = model._gated_step(gi, hh)
                dlog = (model.readout(hh) - logits_L).abs().flatten()
                dir_medians.append(dlog.median().item())
                ctrl = (model.readout(h_p) - logits_mid).abs().flatten().max().item()
                control = min(control, ctrl)
    med = sorted(dir_medians)
    return {
        "vocab_size": V, "d_model": d,
        "readout_rank": rank,
        "null_dim": null_dim,
        "null_energy_frac_by_depth": [round(sum(e) / len(e), 4) for e in energy],
        "perturb_eps_frac": eps_frac,
        "delta": delta,
        "functional_dirs": sum(1 for m_ in med if m_ > delta),
        "median_dlogit_over_dirs": round(med[len(med) // 2], 6) if med else 0.0,
        "max_dlogit_over_dirs": round(med[-1], 6) if med else 0.0,
        "control_same_row_max_dlogit": (round(control, 8)
                                        if control != float("inf") else None),
    }


# ---------- 提交与增量解码 ----------

def commit_input(model: DepthARModel, logits: torch.Tensor,
                 prev_state: torch.Tensor, mode: str, pos: int) -> torch.Tensor:
    """把退出深度的预测提交为下一个输入状态 [.., 1, d]（含 pos）。

    soft（默认）：e = Σ p·E[t]，cap_norm(e + pos)——与新鲜 token 同 gauge；
    hard：e = E[argmax]，同上；
    latent（ablation）：直通退出深度状态 + pos，不过 cap（gauge 断裂，
    预期漂移，如实记录）。"""
    pe = model.pos_embed.weight[pos:pos + 1]
    if mode == "latent":
        return prev_state + pe
    if mode == "soft":
        p = logits.float().softmax(-1)
        e = p @ model.embed.weight.float()
    elif mode == "hard":
        e = model.embed.weight.float()[logits.argmax(-1)]
    else:
        raise ValueError(f"unknown commit mode {mode!r}")
    return cap_norm(e + pe, model.cfg.norm_cap)


class DepthARDecoder:
    """增量单序列解码：逐层 KV cache + 冻结态填充。

    缓存每层的 K/V 及其绝对位置；已退出 position 的冻结态与退出深度也
    入缓存。层 k0 的完整 KV = 旧缓存按绝对位置散点 + 冻结态填充
    （depths[p] ≤ k0 的位置用 kv_proj_{k0}(ln1(frozen_p))）——与训练
    forward 的混合深度语义逐位一致
    （tests/test_depth_ar.py::test_decoder_matches_parallel_forward 钉住）。
    """

    def __init__(self, model: DepthARModel):
        self.model = model
        L = model.cfg.n_layers
        self.K: list[torch.Tensor | None] = [None] * L   # [1,H,C,dh] per layer
        self.V: list[torch.Tensor | None] = [None] * L
        self.pos_idx: list[torch.Tensor | None] = [None] * L  # 缓存列的绝对位置
        self.frozen: torch.Tensor | None = None          # [1,S,d]
        self.depths: list[int] = []
        self.n = 0                                       # 已缓存 position 数

    def _layer_kv(self, k0: int) -> tuple[torch.Tensor, torch.Tensor]:
        """层 k0（0-index）的完整 K/V [1,H,S,dh]：旧缓存散点 + 冻结填充。"""
        model = self.model
        d, nh = model.cfg.d_model, model.cfg.n_heads
        block = model.blocks[k0]
        S = self.n
        k_all = torch.zeros(1, nh, S, d // nh, device=model.embed.weight.device)
        v_all = torch.zeros_like(k_all)
        if self.K[k0] is not None:
            idx = self.pos_idx[k0]
            k_all[:, :, idx] = self.K[k0]
            v_all[:, :, idx] = self.V[k0]
        depths = torch.tensor(self.depths, device=k_all.device)
        shallow = (depths <= k0).nonzero().flatten()     # 未跑到层 k0 的位置
        if shallow.numel():
            kv = block.qkv(block.ln1(self.frozen[:, shallow]))
            kk, vv = kv[..., d:2 * d], kv[..., 2 * d:]
            shape = lambda t: t.view(1, -1, nh, d // nh).transpose(1, 2)
            k_all[:, :, shallow] = shape(kk)
            v_all[:, :, shallow] = shape(vv)
        return k_all, v_all

    def step(self, inp: torch.Tensor,
             forced_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        """inp [1,1,d] 已含 pos 的输入状态；forced_depth ∈ 0..L。
        返回（退出深度表征 [1,1,d]，该深度 logits [1,V]）。"""
        model = self.model
        L = model.cfg.n_layers
        d = model.cfg.d_model
        nh = model.cfg.n_heads
        shape = lambda t: t.view(1, 1, nh, d // nh).transpose(1, 2)
        h = inp
        frozen_self = inp if forced_depth == 0 else None  # depth 0 = 输入态本身
        for k0 in range(forced_depth):                    # 跑层 0..forced_depth-1
            block = model.blocks[k0]
            k_all, v_all = self._layer_kv(k0)
            qkv = block.qkv(block.ln1(h))
            q, k_self, v_self = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
            a = F.scaled_dot_product_attention(shape(q),
                                               torch.cat([k_all, shape(k_self)], dim=2),
                                               torch.cat([v_all, shape(v_self)], dim=2))
            a = a.transpose(1, 2).reshape(1, 1, d)
            y = h + block.proj(a)
            y = y + block.mlp(block.ln2(y))
            g = torch.sigmoid(model.gates[k0](h))
            h = h + g * (y - h)                           # carry 门乘整个 delta
            if k0 == forced_depth - 1:
                frozen_self = h
            self.K[k0] = shape(k_self) if self.K[k0] is None else \
                torch.cat([self.K[k0], shape(k_self)], dim=2)
            self.V[k0] = shape(v_self) if self.V[k0] is None else \
                torch.cat([self.V[k0], shape(v_self)], dim=2)
            pos_t = torch.tensor([self.n], device=h.device)
            self.pos_idx[k0] = pos_t if self.pos_idx[k0] is None else \
                torch.cat([self.pos_idx[k0], pos_t])
        self.frozen = frozen_self if self.frozen is None else \
            torch.cat([self.frozen, frozen_self], dim=1)
        self.depths.append(forced_depth)
        self.n += 1
        return h, model.readout(h)


def token_input(model: DepthARModel, token: int, pos: int) -> torch.Tensor:
    """普通 token 输入状态 [1,1,d]（embed + pos，cap），prefill 用。"""
    e = model.embed.weight.float()[token].view(1, 1, -1)
    pe = model.pos_embed.weight[pos:pos + 1]
    return cap_norm(e + pe, model.cfg.norm_cap)


@torch.no_grad()
def generate(model: DepthARModel, prompt, n_new: int, commit: str = "soft",
             forced_depth: int | None = None) -> tuple[list[int], dict]:
    """贪心生成。forced_depth 给定则所有 position 固定退出深度（出口质量
    测量）；None = 满深度。commit 决定下一输入的提交方式。"""
    model.eval()
    dec = DepthARDecoder(model)
    L = model.cfg.n_layers
    fd = L if forced_depth is None else forced_depth
    prompt = prompt.tolist() if torch.is_tensor(prompt) else list(prompt)
    if len(prompt) + n_new + 1 > model.cfg.seq_len:
        raise ValueError("prompt+n_new exceeds seq_len")
    h = logits = None
    for t in prompt:
        h, logits = dec.step(token_input(model, t, dec.n), fd)
    gen: list[int] = []
    for _ in range(n_new):
        gen.append(int(logits.argmax()))
        if len(gen) >= n_new:
            break
        inp = commit_input(model, logits, h, commit, dec.n)
        h, logits = dec.step(inp, fd)
    return gen, {"commit": commit, "forced_depth": fd, "depths": [fd] * n_new}


@torch.no_grad()
def generation_quality(model: DepthARModel, eval_arr, prompt_len: int = 64,
                       n_new: int = 128, depths: list[int] | None = None,
                       commits: tuple = ("soft", "hard", "latent"),
                       seed: int = 5, device: str = "cpu") -> list[dict]:
    """出口提交测量（判据 3）：每深度 × 每提交方式生成 128 token，报
    合法性（有限/在词表内）、distinct-2（塌缩参考，22b 教训）、生成文本
    在全深度下的重评分 bpc（漂移参考）与出口平均 top-1 概率。"""
    from .data import batch
    generator = torch.Generator().manual_seed(seed)
    n_new = min(n_new, model.cfg.seq_len - prompt_len - 1)  # 位置预算内
    prompt, _ = batch(eval_arr, 1, prompt_len, generator)
    prompt = prompt.to(device)
    L = model.cfg.n_layers
    depths = [0, L // 2, L] if depths is None else depths
    out = []
    model.eval()
    for depth in depths:
        for commit in commits:
            gen, stats = generate(model, prompt[0], n_new, commit=commit,
                                  forced_depth=depth)
            legal = all(0 <= t < model.vocab_size for t in gen)
            # 生成段 distinct-2
            bigrams = list(zip(gen, gen[1:]))
            d2 = len(set(bigrams)) / max(len(bigrams), 1)
            # 重评分：prompt + 生成段的全深度 CE（nats/token → bpc）
            seq = torch.cat([prompt, torch.tensor([gen], device=device)], dim=1)
            _, aux = model(seq[:, :model.cfg.seq_len], exit_depths=None)
            lg = aux["depth_logits"][-1][0, prompt_len - 1:-1]
            tgt = seq[0, prompt_len:prompt_len + lg.shape[0]]
            nats = F.cross_entropy(lg, tgt, reduction="mean").item()
            # 出口平均 top-1 概率：TF 重放 prompt 的前 min(prompt_len, rest) 个位置
            _, aux_p = model(seq[:, :prompt_len], exit_depths=None)
            probs = aux_p["depth_logits"][depth][0].float().softmax(-1)
            top1 = probs[:-1].max(-1).values.mean().item() if depth < prompt_len else None
            out.append({"depth": depth, "commit": commit,
                        "legal": bool(legal and all(torch.isfinite(torch.tensor(gen)))),
                        "distinct2": round(d2, 4),
                        "rescore_bpc": round(nats / LN2, 4),
                        "exit_top1_prob_tf": round(top1, 4) if top1 is not None else None})
    return out
