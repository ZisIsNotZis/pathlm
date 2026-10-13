"""R7（WI-16）共用解码机械与三种早退策略。

**为什么共用一套机械**：`C-AUDIT.md` §4① 的教训是「同协议倍率在三 run 间漂移
1.5–5.5×」——那是单臂抽样 + 实现层开销混在一起的产物。本模块让所有臂走**同一套**
KV/填充/提交机械，唯一差别是**策略函数**（全深度 / 固定退出 / 置信度门控 / 深度草稿），
因此臂间比值只反映策略，且主口径用「同一进程内交错轮次」。

与 `pathlm.depth_ar.DepthARDecoder` 的关系：同 fill 语义（已退出位置的 KV 用其冻结
表征经本层 kv 投影填充），但①`gated` 显式（dense / 无门臂走无门路径），②`step` 支持
**逐层停靠**（CALM 的置信度门控在每层结束后决定是否退出），③可挂在「已提交前缀」
（spec 的草稿阶段）。语义一致性由单测钉住（`tests/test_r7_exit_policies.py`）。

策略：
- `greedy_decode`：全深度 / 固定退出深度（部署参照点）；
- `calm_calibrate` + `calm_decode`：CALM 式置信度门控（阈值按「与全深度预测的一致率」
  标定，逐 token 取最浅的 `conf ≥ θ_k` 深度）；
- `spec_decode`：浅出口（depth `dd`）当草稿、全深度验证的自投机解码（LayerSkip 式）。
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from pathlm.depth_ar import commit_input, token_input


LN2 = 0.6931471805599453


class MixDepthCache:
    """增量单序列解码缓存（fill 语义），带显式 `gated` 开关、逐层停靠与可选前缀。

    `base_kv(k0) -> (K, V)` 给定时，它提供**已提交前缀**（每位置跑满 L 层）的逐层
    KV，本缓存只管自己的位置（绝对位置 = `base_n + self.n`）。spec 的草稿阶段就用这个
    挂法（草稿链在已提交前缀之后生长）。
    """

    def __init__(self, model, gated: bool, base_kv=None, base_n: int = 0):
        self.model = model
        self.gated = bool(gated)
        self.base_kv = base_kv
        self.base_n = int(base_n)
        L = model.cfg.n_layers
        self.K: list[torch.Tensor | None] = [None] * L
        self.V: list[torch.Tensor | None] = [None] * L
        self.pos_idx: list[torch.Tensor | None] = [None] * L
        self.frozen: torch.Tensor | None = None
        self.depths: list[int] = []
        self.n = 0

    @property
    def abs_pos(self) -> int:
        return self.base_n + self.n

    def _layer_kv(self, k0: int) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        d, nh = model.cfg.d_model, model.cfg.n_heads
        block = model.blocks[k0]
        S = self.n
        parts_k, parts_v = [], []
        if self.base_kv is not None:
            bk, bv = self.base_kv(k0)
            parts_k.append(bk)
            parts_v.append(bv)
        if S:
            shape = lambda t: t.view(1, -1, nh, d // nh).transpose(1, 2)
            k_own = torch.zeros(1, nh, S, d // nh,
                                device=model.embed.weight.device,
                                dtype=self.frozen.dtype)
            v_own = torch.zeros_like(k_own)
            if self.K[k0] is not None:
                idx = self.pos_idx[k0]
                k_own[:, :, idx] = self.K[k0]
                v_own[:, :, idx] = self.V[k0]
            depths = torch.tensor(self.depths, device=k_own.device)
            shallow = (depths <= k0).nonzero().flatten()
            if shallow.numel():            # 已退出位置：用冻结表征的 KV 填充
                kv = block.qkv(block.ln1(self.frozen[:, shallow]))
                kk, vv = kv[..., d:2 * d], kv[..., 2 * d:]
                k_own[:, :, shallow] = shape(kk)
                v_own[:, :, shallow] = shape(vv)
            parts_k.append(k_own)
            parts_v.append(v_own)
        if not parts_k:            # 空前缀（prefill 首 token）：零长度 KV
            z = torch.zeros(1, nh, 0, d // nh,
                            device=model.embed.weight.device,
                            dtype=model.embed.weight.dtype)
            return z, z
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    @torch.no_grad()
    def step(self, inp: torch.Tensor, max_depth: int,
             depth_fn=None) -> tuple[int, torch.Tensor, torch.Tensor]:
        """喂一个已含 pos 的输入状态 `[1,1,d]`，最多跑 `max_depth` 层。

        `depth_fn(k, logits_k) -> bool` 给定时逐层询问「就此退出吗」（CALM）：
        第一个返回 True 的 k（或 max_depth）即该位置的退出深度。返回
        `(exit_depth, h_exit, logits_exit)`。
        """
        model = self.model
        d, nh = model.cfg.d_model, model.cfg.n_heads
        shape = lambda t: t.view(1, 1, nh, d // nh).transpose(1, 2)
        h = inp
        logits = model.readout(h, depth=0)
        exit_depth = 0
        stop_at_zero = depth_fn is not None and depth_fn(0, logits)
        for k0 in range(0 if stop_at_zero else max_depth):
            block = model.blocks[k0]
            k_all, v_all = self._layer_kv(k0)
            qkv = block.qkv(block.ln1(h))
            q, k_self, v_self = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
            a = F.scaled_dot_product_attention(
                shape(q), torch.cat([k_all, shape(k_self)], dim=2),
                torch.cat([v_all, shape(v_self)], dim=2))
            a = a.transpose(1, 2).reshape(1, 1, d)
            y = h + block.proj(a)
            y = y + block.mlp(block.ln2(y))
            if self.gated:
                g = torch.sigmoid(model.gates[k0](h))
                h = h + g * (y - h)
            else:
                h = y
            self.K[k0] = shape(k_self) if self.K[k0] is None else \
                torch.cat([self.K[k0], shape(k_self)], dim=2)
            self.V[k0] = shape(v_self) if self.V[k0] is None else \
                torch.cat([self.V[k0], shape(v_self)], dim=2)
            pos_t = torch.tensor([self.n], device=h.device)   # **本地**索引
            self.pos_idx[k0] = pos_t if self.pos_idx[k0] is None else \
                torch.cat([self.pos_idx[k0], pos_t])
            exit_depth = k0 + 1
            logits = model.readout(h, depth=exit_depth)
            if depth_fn is not None and depth_fn(exit_depth, logits):
                break
        self.frozen = h if self.frozen is None else torch.cat([self.frozen, h], dim=1)
        self.depths.append(exit_depth)
        self.n += 1
        return exit_depth, h, logits


def is_gated(model) -> bool:
    """该模型在**变深度解码**里是否走 carry 门：非 dense 且 `carry_gate=True`。
    dense（stock/MoD）主干本来就没有门，解码必须走无门路径（否则静默套一个
    训练时从未作用的门，语义错位）。"""
    return bool(model.cfg.carry_gate) and not bool(model.cfg.dense)


def _confidence(logits: torch.Tensor) -> float:
    """readout 的 top-1 softmax 概率（fp32）。"""
    return float(logits.float().softmax(-1).max())


@torch.no_grad()
def greedy_decode(model, prompt, n_new: int, forced_depth: int | None = None,
                  gated: bool | None = None, commit: str = "hard",
                  timeit: bool = False):
    """共享机械上的贪心解码。返回 `(tokens, stats)`：逐位置退出深度、解析算力
    （层×位置）与（可选）墙钟。"""
    model.eval()
    gated = is_gated(model) if gated is None else bool(gated)
    L = model.cfg.n_layers
    fd = L if forced_depth is None else forced_depth
    prompt = prompt.tolist() if torch.is_tensor(prompt) else list(prompt)
    dec = MixDepthCache(model, gated)
    t0 = time.time() if timeit else None
    h = logits = None
    layerpos_pre = 0
    for t in prompt:
        k, h, logits = dec.step(token_input(model, t, dec.abs_pos), fd)
        layerpos_pre += k
    gen: list[int] = []
    depths: list[int] = []
    layerpos_gen = 0
    for _ in range(n_new):
        tok = int(logits.argmax())
        gen.append(tok)
        if len(gen) >= n_new:
            break
        inp = commit_input(model, logits, h, commit, dec.abs_pos)
        k, h, logits = dec.step(inp, fd)
        depths.append(k)
        layerpos_gen += k
    if timeit:
        torch.cuda.synchronize()
    stats = {"forced_depth": fd, "commit": commit, "gated": gated,
             "layerpos_prefill": layerpos_pre,
             "layerpos_generation": layerpos_gen,
             "layerpos_per_emitted_token": layerpos_gen / max(len(gen), 1),
             "depths": depths}
    if t0 is not None:
        stats["wall_s"] = time.time() - t0
        stats["tok_s"] = len(gen) / max(stats["wall_s"], 1e-9)
    return gen, stats


@torch.no_grad()
def calm_calibrate(model, eval_arr, eps_list=(0.02, 0.05, 0.10), n_batches: int = 8,
                   batch_size: int = 16, device: str = "cuda"):
    """CALM 式置信度阈值标定（enwik8 val，teacher forcing）。

    每个深度 k 收一组 `(conf_k, agree_k)`，`agree_k = 1[argmax_k == argmax_L]`。
    阈值 `θ_k` = 使 `P(agree_k | conf_k ≥ θ_k) ≥ 1−ε` 的**最小**阈值（= 退得最浅的
    那个）。标定只用 val 的**前一半**批次，后一半批次**不参与标定**，只用来核验
    门控的实际平均退出深度与一致率（`*_holdout` 字段）——防「在标定集上自证」。
    teacher forcing 口径：喂的是真值流；部署口径（自回归自喂）在 `calm_decode` 另测。
    """
    from pathlm.data import batch
    T = model.cfg.seq_len
    L = model.cfg.n_layers
    gen = torch.Generator().manual_seed(3)
    n_batches = max(2, n_batches)
    n_first = n_batches // 2
    conf = [[[] for _ in range(L + 1)] for _ in range(2)]     # [half][depth]
    agree = [[[] for _ in range(L + 1)] for _ in range(2)]
    model.eval()
    for b in range(n_batches):
        half = 0 if b < n_first else 1
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        full = aux["depth_logits"][-1][:, :-1].argmax(-1)
        for k in range(L + 1):
            lg = aux["depth_logits"][k][:, :-1]
            conf[half][k].append(lg.float().softmax(-1).max(-1).values.flatten())
            agree[half][k].append((lg.argmax(-1) == full).flatten())
    for half in range(2):
        for k in range(L + 1):
            conf[half][k] = torch.cat(conf[half][k])
            agree[half][k] = torch.cat(agree[half][k])
    out = {"by_eps": {}, "n_calib": int(conf[0][0].numel()),
           "n_holdout": int(conf[1][0].numel()),
           "n_batches": n_batches, "n_calib_batches": n_first,
           "note": ("阈值只用 val 前一半批次标定；`*_holdout` 是后一半批次上的实测"
                    "（未参与标定）。teacher forcing 口径。")}
    for eps in eps_list:
        theta = []
        for k in range(L + 1):
            c, a = conf[0][k], agree[0][k]
            order = torch.argsort(c, descending=True)
            cs, as_ = c[order], a[order].float()
            ar = torch.arange(1, as_.numel() + 1, device=as_.device,
                              dtype=as_.dtype)
            cum = torch.cumsum(as_, 0) / ar
            ok = (cum >= 1 - eps).nonzero().flatten()
            theta.append(float(cs[int(ok[0])]) if ok.numel() else 1.1)
        entry = {"epsilon": eps, "theta": [round(t, 4) for t in theta]}
        for half, tag in ((0, ""), (1, "_holdout")):
            n = conf[half][0].numel()
            dev = conf[half][0].device
            chosen = torch.full((n,), L, dtype=torch.long, device=dev)
            for k in range(L, -1, -1):      # 从深到浅覆盖 ⇒ 最浅达标者胜
                chosen = torch.where(conf[half][k] >= theta[k],
                                     torch.full_like(chosen, k), chosen)
            agree_stack = torch.stack([a.float() for a in agree[half]])  # [L+1,n]
            hit = agree_stack[chosen, torch.arange(n, device=dev)]
            d = chosen.float()
            entry[f"mean_exit_depth{tag}"] = round(float(d.mean()), 4)
            entry[f"frac_exit_at_full{tag}"] = round(
                float((d == L).float().mean()), 4)
            entry[f"agree_with_full{tag}"] = round(float(hit.mean()), 4)
        out["by_eps"][f"{eps:.2f}"] = entry
    # ---- CALM 原式的**联合规则**标定（单个全局阈值 θ）----
    # CALM（Schuster et al.）的校准口径是**退出流自己的不一致率**：在标定集上选一个
    # 全局阈值 θ，使「逐 token 与全深度预测不一致」的比例 ≤ ε（越激进的 θ 越浅）。
    # 这与上面「逐深度条件一致率」的保守口径不同——两者都报，后者是更严的界。
    grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]

    def _joint(conf_h, agree_h, th):
        n = conf_h[0].numel()
        dev = conf_h[0].device
        chosen = torch.full((n,), L, dtype=torch.long, device=dev)
        for k in range(L, -1, -1):
            chosen = torch.where(conf_h[k] >= th, torch.full_like(chosen, k),
                                 chosen)
        stack = torch.stack([a.float() for a in agree_h])
        hit = stack[chosen, torch.arange(n, device=dev)]
        return chosen.float(), hit

    out["calm_joint"] = {}
    for eps in eps_list:
        chosen_theta = None
        for th in grid:
            _, hit = _joint(conf[0], agree[0], th)
            if float(1.0 - hit.mean()) <= eps:
                chosen_theta = th
                break
        entry = {"epsilon": eps, "theta": chosen_theta}
        if chosen_theta is not None:
            for half, tag in ((0, "_calib"), (1, "_holdout")):
                d, hit = _joint(conf[half], agree[half], chosen_theta)
                entry[f"mean_exit_depth{tag}"] = round(float(d.mean()), 4)
                entry[f"frac_exit_at_full{tag}"] = round(
                    float((d == L).float().mean()), 4)
                entry[f"agree_with_full{tag}"] = round(float(hit.mean()), 4)
        out["calm_joint"][f"{eps:.2f}"] = entry
    # 固定阈值扫描（CALM 的成本-质量边界形状）：θ 全深度统一，报样本外
    # 「平均退出深度 / 跑满比例 / 与全深度一致率」。它回答「CALM 的边界长什么样」，
    # 与上面「按目标一致率标定」是同一份数据的两种读法。
    out["fixed_theta_sweep"] = {}
    for th in (0.5, 0.7, 0.9, 0.95, 0.99):
        n = conf[1][0].numel()
        dev = conf[1][0].device
        chosen = torch.full((n,), L, dtype=torch.long, device=dev)
        for k in range(L, -1, -1):
            chosen = torch.where(conf[1][k] >= th, torch.full_like(chosen, k),
                                 chosen)
        agree_stack = torch.stack([a.float() for a in agree[1]])
        hit = agree_stack[chosen, torch.arange(n, device=dev)]
        d = chosen.float()
        out["fixed_theta_sweep"][f"{th:.2f}"] = {
            "theta": th, "mean_exit_depth_holdout": round(float(d.mean()), 4),
            "frac_exit_at_full_holdout": round(float((d == L).float().mean()), 4),
            "agree_with_full_holdout": round(float(hit.mean()), 4)}
    return out


@torch.no_grad()
def calm_decode(model, prompt, n_new: int, thetas, gated: bool | None = None,
                timeit: bool = False):
    """真解码路径上的 CALM 门控（自回归、自喂）：每个位置逐层检查置信度，
    取最浅的 `conf ≥ θ_k` 深度退出。返回 `(tokens, stats)`。"""
    model.eval()
    gated = is_gated(model) if gated is None else bool(gated)
    L = model.cfg.n_layers
    prompt = prompt.tolist() if torch.is_tensor(prompt) else list(prompt)
    dec = MixDepthCache(model, gated)
    t0 = time.time() if timeit else None
    h = logits = None
    layerpos_pre = 0
    for t in prompt:
        k, h, logits = dec.step(token_input(model, t, dec.abs_pos), L)
        layerpos_pre += k
    gen: list[int] = []
    depths: list[int] = []
    layerpos_gen = 0
    stopped_at = [0] * (L + 1)
    for _ in range(n_new):
        tok = int(logits.argmax())
        gen.append(tok)
        if len(gen) >= n_new:
            break
        inp = commit_input(model, logits, h, "hard", dec.abs_pos)
        k, h, logits = dec.step(
            inp, L, depth_fn=lambda kk, lg: _confidence(lg) >= thetas[kk])
        depths.append(k)
        stopped_at[k] += 1
        layerpos_gen += k
    if timeit:
        torch.cuda.synchronize()
    stats = {"gated": gated, "thetas": list(thetas),
             "layerpos_prefill": layerpos_pre,
             "layerpos_generation": layerpos_gen,
             "layerpos_per_emitted_token": layerpos_gen / max(len(gen), 1),
             "mean_exit_depth": (sum(depths) / len(depths)) if depths else None,
             "depth_hist": {str(k): v for k, v in enumerate(stopped_at) if v}}
    if t0 is not None:
        stats["wall_s"] = time.time() - t0
        stats["tok_s"] = len(gen) / max(stats["wall_s"], 1e-9)
    return gen, stats


# ---------------- LayerSkip 式自投机解码 ----------------


class SpecCache:
    """spec 解码的已提交前缀：每个位置都已跑满 L 层（无退化位置）。"""

    def __init__(self, model):
        self.model = model
        L = model.cfg.n_layers
        self.K: list[torch.Tensor | None] = [None] * L
        self.V: list[torch.Tensor | None] = [None] * L
        self.n = 0

    def prefix_kv(self, k0: int):
        model = self.model
        d, nh = model.cfg.d_model, model.cfg.n_heads
        if self.n == 0:
            z = torch.zeros(1, nh, 0, d // nh,
                            device=model.embed.weight.device,
                            dtype=model.embed.weight.dtype)
            return z, z
        return self.K[k0], self.V[k0]

    def commit(self, kv: dict, n_commit: int) -> None:
        for k0, (ks, vs) in kv.items():
            ks, vs = ks[:, :, :n_commit], vs[:, :, :n_commit]
            self.K[k0] = ks if self.K[k0] is None else \
                torch.cat([self.K[k0], ks], dim=2)
            self.V[k0] = vs if self.V[k0] is None else \
                torch.cat([self.V[k0], vs], dim=2)
        self.n += n_commit


def _run_layers(model, inputs: torch.Tensor, start: int, end: int,
                gated: bool, prefix_kv):
    """在 `inputs [k,1,d]` 上跑 block `start..end-1`（注意力 = 已提交前缀 + 批内因果）。

    返回 `(states, kv)`：`states[0] = inputs`，`states[j]` = 跑完 block `start+j-1`
    后的状态；`kv[k0]` = (K,V) 形状 `[1,nh,k,dh]`。**不改缓存**。
    """
    d, nh = model.cfg.d_model, model.cfg.n_heads
    k = inputs.shape[0]
    has_prefix = prefix_kv is not None and start < end
    Kp, Vp = prefix_kv(start) if has_prefix else (None, None)
    S = 0 if Kp is None else Kp.shape[2]
    shape = lambda t: t.view(1, k, nh, d // nh).transpose(1, 2)
    h = inputs
    states = [h]
    kv: dict[int, tuple] = {}
    for k0 in range(start, end):
        block = model.blocks[k0]
        qkv = block.qkv(block.ln1(h))
        q, ks, vs = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
        mask = torch.ones(k, S + k, dtype=torch.bool, device=inputs.device)
        mask[:, S:] = torch.tril(torch.ones(k, k, dtype=torch.bool,
                                            device=inputs.device))
        if S:
            kk0, vv0 = prefix_kv(k0)
            K_all, V_all = torch.cat([kk0, shape(ks)], dim=2), \
                torch.cat([vv0, shape(vs)], dim=2)
        else:
            K_all, V_all = shape(ks), shape(vs)
        a = F.scaled_dot_product_attention(shape(q), K_all, V_all,
                                           attn_mask=mask)
        a = a.transpose(1, 2).reshape(k, 1, d)   # 与 inputs 同 gauge（[k,1,d]）
        y = h + block.proj(a)
        y = y + block.mlp(block.ln2(y))
        if gated:
            g = torch.sigmoid(model.gates[k0](h))
            h = h + g * (y - h)
        else:
            h = y
        states.append(h)
        kv[k0] = (shape(ks), shape(vs))
    return states, kv


@torch.no_grad()
def spec_decode(model, prompt, n_new: int, gamma: int = 4, draft_depth: int = 4,
                gated: bool | None = None, commit: str = "hard",
                timeit: bool = False, cycle_log: list | None = None):
    """LayerSkip 式自投机解码：草稿 = depth-`dd` 出口、验证 = 全深度。

    不变量：发出的 token 流 == 同 arm 的全深度贪心流（草稿只省前向次数，不改质量）；
    接受判定一律对「目标（全深度）argmax」，不与真值比（`SCHEMA` §11）。

    解析算力：每轮 `(γ+1)·L` 层×位置（草稿浅层 `(γ+1)·dd` + 验证 `(γ+1)·(L−dd)`），
    发出 `1+τ` 个 token ⇒ 每 token `L(γ+1)/(1+τ)`。
    """
    model.eval()
    gated = is_gated(model) if gated is None else bool(gated)
    L = model.cfg.n_layers
    prompt = prompt.tolist() if torch.is_tensor(prompt) else list(prompt)
    cache = SpecCache(model)
    t0 = time.time() if timeit else None
    # ---- prefill：整段 prompt 一次前向（无前缀）----
    pin = torch.cat([token_input(model, t, i) for i, t in enumerate(prompt)],
                    dim=0)
    states, kv = _run_layers(model, pin, 0, L, gated, None)
    cache.commit(kv, len(prompt))
    h = states[-1][-1:]
    logits = model.readout(h, depth=L)
    layerpos_pre = L * len(prompt)
    layerpos_gen = 0
    # 不变量（与 C1 的 2621 同一构造）：**先发出 prefill 末位的 argmax**（位置 n 的
    # token），它由 prefill 免费给出；此后每轮开始时「位置 n 的 token 已作为 bonus
    # 发出」。漏掉这一 token 会让整条流相对贪心流错位一格（本波实测事故，单测钉住）。
    emitted: list[int] = []
    if n_new > 0:
        emitted.append(int(logits.argmax()))
    accept_by_pos: list[list[int]] = [[] for _ in range(gamma)]
    n_cycles = 0
    n_draft = 0
    n_accept = 0
    while len(emitted) < n_new:
        n_cycles += 1
        pending = commit_input(model, logits, h, commit, cache.n)
        # ---- 草稿：depth `dd` 上串行 γ+1 步（挂已提交前缀，O(γ·dd)）----
        scratch = MixDepthCache(model, gated, base_kv=cache.prefix_kv,
                               base_n=cache.n)
        dd_states: list[torch.Tensor] = []
        draft_toks: list[int] = []
        inp_i = pending
        for _ in range(gamma + 1):
            _, hdd, lg_dd = scratch.step(inp_i, draft_depth)
            dd_states.append(hdd)
            tok = int(lg_dd.argmax(-1))
            draft_toks.append(tok)
            inp_i = commit_input(model, lg_dd, hdd, commit, scratch.abs_pos)
        # ---- 验证：把 γ+1 个 depth-dd 状态成批喂进 dd..L-1 ----
        v_in = torch.cat(dd_states, dim=0)
        v_states, v_kv = _run_layers(model, v_in, draft_depth, L, gated,
                                     cache.prefix_kv)
        y = model.readout(v_states[-1], depth=L)              # [γ+1, V]
        target = y.argmax(-1).flatten().tolist()
        # ---- 接受：最长前缀 draft_toks[j] == target[j]（j = 0..γ-1）----
        k_acc = 0
        for j in range(gamma):
            n_draft += 1
            accept_by_pos[j].append(int(draft_toks[j] == target[j]))
            if draft_toks[j] == target[j]:
                k_acc += 1
            else:
                break
        n_accept += k_acc
        n_emit_now = 0
        for j in range(k_acc + 1):
            if len(emitted) < n_new:
                emitted.append(target[j])
                n_emit_now += 1
        # ---- 提交前 k_acc+1 个位置：浅层 KV = 草稿、深层 KV = 验证 ----
        n_commit = k_acc + 1
        commit_kv = {}
        for k0 in range(L):
            if k0 < draft_depth:
                commit_kv[k0] = (scratch.K[k0], scratch.V[k0])
            else:
                commit_kv[k0] = v_kv[k0]
        cache.commit(commit_kv, n_commit)
        h = v_states[-1][n_commit - 1:n_commit]
        logits = model.readout(h, depth=L)
        layerpos_gen += (gamma + 1) * draft_depth + (gamma + 1) * (L - draft_depth)
        if cycle_log is not None:
            cycle_log.append({"cycle": n_cycles, "accepted": k_acc,
                              "emitted_now": n_emit_now,
                              "gamma": gamma, "draft_depth": draft_depth,
                              "committed": n_commit, "cache_n": cache.n})
    if timeit:
        torch.cuda.synchronize()
    stats = {"gated": gated, "gamma": gamma, "draft_depth": draft_depth,
             "n_cycles": n_cycles, "n_drafts": n_draft, "n_accepts": n_accept,
             "accept_rate": (n_accept / n_draft) if n_draft else None,
             "accept_by_pos": [round(sum(v) / max(len(v), 1), 4)
                               for v in accept_by_pos],
             "emitted": len(emitted), "layerpos_prefill": layerpos_pre,
             "layerpos_generation": layerpos_gen,
             "layerpos_per_emitted_token": layerpos_gen / max(len(emitted), 1),
             "layerpos_per_emitted_token_analytic": round(
                 L * (gamma + 1) / (n_accept / max(n_cycles, 1) + 1), 4),
             "tokens_per_cycle": len(emitted) / max(n_cycles, 1)}
    if t0 is not None:
        torch.cuda.synchronize()
        stats["wall_s"] = time.time() - t0
        stats["tok_s"] = len(emitted) / max(stats["wall_s"], 1e-9)
    return emitted, stats
