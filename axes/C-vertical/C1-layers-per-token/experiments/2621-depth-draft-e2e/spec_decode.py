"""真实端到端「深度草稿 + 全栈验证」解码（ticket 38 W2）。

C1 的出口资产一直是 **TF 代理口径**（`pathlm/depth_ar.py::tf_acceptance`：在 enwik8
真值流上比较 depth-k 与 depth-L 的 argmax）。本模块实现**真正的自 spec 解码循环**，
用来量实测接受率与实测 tok/s，并与代理口径对照。

本模块**只读**模型，不改 `pathlm/`（ticket 38 期间该文件由 W4 线独占，SCHEMA §22）。

## 循环结构（标准 draft-then-verify，γ 草稿 → γ+1 位置验证）

不变量：轮开始时，位置 `0..n-1` 已提交（每个位置跑满 L 层、逐层 KV 已缓存），
位置 `n` 的 token **已在上轮作为 bonus 发出**、其提交态记为 `cache.pending`。
于是本轮：

1. **草稿**（depth `dd`，串行 γ+1 步，KV 存临时缓存、永不提交）：
   位置 `n..n+γ` 的 depth-`dd` 状态 `hdd[0..γ]`；`d_i = argmax(dd 出口 @ 位置 n+i-1)`
   是位置 `n+i` 的草稿 token（i = 1..γ）；第 γ+1 步的 logits 只用它的**状态**。
2. **验证**：把 `hdd[0..γ]` 一次性喂进 block `dd..L-1`（批内因果 + 已提交前缀），
   得到目标输出 `Y_{n..n+γ}`。`V_i = argmax(Y_{n+i-1})` = 位置 `n+i` 的**目标**
   token（i = 1..γ+1，最后一个就是 bonus）。
3. **接受**：`k` = 最长的 i 使 `d_i == V_i`（i ≤ γ）。发出 `V_1..V_{k+1}`
   （k+1 个 token）。
4. **提交**：把位置 `n..n+k`（k+1 个）的逐层 KV 写进缓存（草稿提供 block `0..dd-1`、
   验证提供 `dd..L-1`），新 `pending` = `commit(V_{k+1})`。

由此发出的 token 流 == 同 checkpoint 的**普通全栈贪心流**（自 spec 不改质量，
单测 `test_spec_matches_plain_greedy` 钉住）。

## 两个臂（唯一变量 = 验证分不分级）

- `single`：草稿 d4 → **一次全栈**（L=8）验证（判据 ③ 的「单级」）。
- `graded`：草稿 d4 → 先 d6 验证取接受前缀 → **只用全栈确认该前缀**
  （d4 → d6 → 全栈，逐级放行）。发出的流与 `single` 逐 token 相同，验证算力更省。
"""

import torch
import torch.nn.functional as F

from pathlm.depth_ar import commit_input, token_input


class CommittedCache:
    """已提交前缀：每个位置都跑满 L 层（无混深度/无 frozen 填充）。

    因此 `layer_kv(k)` 就是第 k 层的缓存列，与
    `DepthARDecoder._layer_kv` 在「全部位置 forced_depth=L」时逐位一致
    （单测 `test_committed_cache_matches_depthar_decoder`）。
    """

    def __init__(self, model):
        self.model = model
        L = model.cfg.n_layers
        self.K: list[torch.Tensor | None] = [None] * L
        self.V: list[torch.Tensor | None] = [None] * L
        self.positions: list[int] = []
        self.n = 0
        # 最后一个已提交位置的逐深度状态（key = depth k ↔ states[k]）：
        # 草稿读 depth-dd 出口，graded 读 depth-mid 出口。
        self.depth_h: dict[int, torch.Tensor] = {}
        self.pending: torch.Tensor | None = None   # 位置 n 的提交态（token 已发出）

    def layer_kv(self, k0: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.model.cfg.d_model
        nh = self.model.cfg.n_heads
        dh = d // nh
        if self.n == 0:
            dev = self.model.embed.weight.device
            z = torch.zeros(1, nh, 0, dh, device=dev,
                            dtype=self.model.embed.weight.dtype)
            return z, z
        return self.K[k0], self.V[k0]

    def commit(self, kv: dict, n_accept: int, depth_h_last: dict,
               first_pos: int) -> None:
        """把接受前缀（前 `n_accept` 个位置，从 `first_pos` 起）的逐层 KV 写进缓存。

        `kv[k0]` = (k, v) 形状 [1, n_head, ≥n_accept, d/n_head]；`depth_h_last`
        是**最后一个被接受位置**的逐深度状态（{depth: h [1,1,d]}）。"""
        if n_accept <= 0:
            raise ValueError("commit 需要 n_accept ≥ 1（每轮至少提交一个位置）")
        for k0, (ks, vs) in kv.items():
            ck, cv = ks[:, :, :n_accept], vs[:, :, :n_accept]
            if self.K[k0] is None:
                self.K[k0], self.V[k0] = ck, cv
            else:
                self.K[k0] = torch.cat([self.K[k0], ck], dim=2)
                self.V[k0] = torch.cat([self.V[k0], cv], dim=2)
        self.positions.extend(range(first_pos, first_pos + n_accept))
        self.n += n_accept
        self.depth_h = dict(depth_h_last)
        L = self.model.cfg.n_layers
        self.pending = commit_input(self.model,
                                    self.model.readout(depth_h_last[L]),
                                    depth_h_last[L], "hard", self.n)


@torch.no_grad()
def run_layers(model, cache: CommittedCache, inputs: torch.Tensor,
               start_layer: int, end_layer: int | None = None):
    """在 `inputs`（[k, d]，位置从 `cache.n` 起的连续 k 个）上跑 block
    `start_layer..end_layer-1`：注意力 = 已提交前缀 + 批内因果。

    返回 `(states, kv)`：`states[j]` = 跑完 block `start_layer+j` 后的状态
    （`states[0]` = `inputs`），于是 `states[depth - start_layer]` 就是该位置
    在 depth 处的状态（与 `DepthARModel.forward` 的 `aux["states"]` 同约定）；
    `kv[b]` = block b 在本批上的 (k, v)。**不修改缓存**。
    """
    L = model.cfg.n_layers
    d = model.cfg.d_model
    nh = model.cfg.n_heads
    end_layer = L if end_layer is None else end_layer
    k = inputs.shape[0]
    S = cache.n
    shape = lambda t: t.view(1, k, nh, d // nh).transpose(1, 2)
    h = inputs
    states = [h]
    kv: dict[int, tuple] = {}
    for k0 in range(start_layer, end_layer):
        Kc, Vc = cache.layer_kv(k0)
        block = model.blocks[k0]
        qkv = block.qkv(block.ln1(h))
        q, ks, vs = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
        mask = torch.ones(k, S + k, dtype=torch.bool, device=inputs.device)
        mask[:, S:] = torch.tril(torch.ones(k, k, dtype=torch.bool,
                                            device=inputs.device))
        a = F.scaled_dot_product_attention(shape(q),
                                           torch.cat([Kc, shape(ks)], dim=2),
                                           torch.cat([Vc, shape(vs)], dim=2),
                                           attn_mask=mask)
        a = a.transpose(1, 2).reshape(k, 1, d)   # 保持 [k,1,d]，与 token_input 同 gauge
        y = h + block.proj(a)
        y = y + block.mlp(block.ln2(y))
        g = torch.sigmoid(model.gates[k0](h))
        h = h + g * (y - h)
        states.append(h)
        kv[k0] = (shape(ks), shape(vs))
    return states, kv


class Draft:
    """depth-`depth` 草稿：KV 只在临时缓存里，永不提交。

    第 k0 层的注意力 = 已提交前缀的同层 KV + 本草稿此前位置的同层 KV + 自身。
    草稿逐位置串行（自回归），因此不需要因果掩码。
    """

    def __init__(self, model, cache: CommittedCache, depth: int):
        self.model = model
        self.cache = cache
        self.depth = depth
        self.K: list[torch.Tensor | None] = [None] * model.cfg.n_layers
        self.V: list[torch.Tensor | None] = [None] * model.cfg.n_layers

    @torch.no_grad()
    def step(self, inp: torch.Tensor):
        """`inp` [1,1,d]（已含 pos 的提交态）→ (depth 状态 [1,1,d], logits [1,V])。"""
        model = self.model
        d = model.cfg.d_model
        nh = model.cfg.n_heads
        shape = lambda t: t.view(1, 1, nh, d // nh).transpose(1, 2)
        h = inp
        for k0 in range(self.depth):
            Kc, Vc = self.cache.layer_kv(k0)
            block = model.blocks[k0]
            qkv = block.qkv(block.ln1(h))
            q, ks, vs = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
            ks_, vs_ = shape(ks), shape(vs)
            K = torch.cat([Kc, self.K[k0], ks_], dim=2) if self.K[k0] is not None \
                else torch.cat([Kc, ks_], dim=2)
            V = torch.cat([Vc, self.V[k0], vs_], dim=2) if self.V[k0] is not None \
                else torch.cat([Vc, vs_], dim=2)
            a = F.scaled_dot_product_attention(shape(q), K, V)
            a = a.transpose(1, 2).reshape(1, 1, d)
            y = h + block.proj(a)
            y = y + block.mlp(block.ln2(y))
            g = torch.sigmoid(model.gates[k0](h))
            h = h + g * (y - h)
            self.K[k0] = ks_ if self.K[k0] is None else \
                torch.cat([self.K[k0], ks_], dim=2)
            self.V[k0] = vs_ if self.V[k0] is None else \
                torch.cat([self.V[k0], vs_], dim=2)
        return h, model.readout(h)


def _argmax(t: torch.Tensor) -> int:
    return int(t.reshape(-1).argmax())


def accept_prefix(draft_tokens: list[int], v_tokens: list[int],
                  gamma: int) -> int:
    """最长被接受的草稿前缀长度 k（d_i == V_i，i = 1..gamma，全 0-indexed）。

    这是判据 ③「接受」的唯一判定处：草稿 token 必须与**同一位置的目标 argmax**
    逐 token 相同。别处不得重复实现该比较（单测 `test_accept_prefix_*` 钉住，
    变异验证改这一行即可抓红）。"""
    k = 0
    while k < gamma and draft_tokens[k] == v_tokens[k]:
        k += 1
    return k


@torch.no_grad()
def prefill(model, tokens: list[int]) -> tuple[CommittedCache, int]:
    """把 prompt 一次性提交（batch 前向，纯因果），并发出首个 token。

    返回 (cache, first_token)：`first_token` = 位置 `len(tokens)` 的目标贪心 token，
    已发出；`cache.pending` = 该 token 在位置 `len(tokens)` 的提交态。"""
    if not tokens:
        raise ValueError("prefill 需要非空 prompt")
    cache = CommittedCache(model)
    inp = torch.cat([token_input(model, t, i) for i, t in enumerate(tokens)], dim=0)
    states, kv = run_layers(model, cache, inp, start_layer=0)
    depth_h = {j: states[j][-1:] for j in range(len(states))}
    cache.commit(kv, len(tokens), depth_h, 0)
    first = _argmax(model.readout(states[-1][-1:]))
    return cache, first


@torch.no_grad()
def _draft_phase(model, cache: CommittedCache, gamma: int, dd: int):
    """草稿阶段：串行 γ+1 个 depth-`dd` 前向（最后一个只用状态）。

    返回 (tokens, hdd, draft)：`tokens[i]` = 位置 `n+i+1` 的草稿 token（长度 γ）；
    `hdd[j]` = 位置 `n+j` 的 depth-`dd` 状态（长度 γ+1，[1,1,d]）；`draft` 持有
    临时 KV（供提交时取 block `0..dd-1`）。"""
    draft = Draft(model, cache, dd)
    tokens: list[int] = []
    hdd: list[torch.Tensor] = []
    inp = cache.pending
    for j in range(gamma + 1):
        h, lg = draft.step(inp)
        hdd.append(h)
        if j < gamma:
            tokens.append(_argmax(lg))
            inp = commit_input(model, lg, h, "hard", cache.n + j + 1)
    return tokens, hdd, draft


@torch.no_grad()
def _finish(cache: CommittedCache, draft, k: int, v_tokens: list[int],
            deep_kv: dict, depth_h_at):
    """接受 k 个草稿后：发出 V_1..V_{k+1}、提交位置 n..n+k。"""
    n_acc = k + 1
    kv = {k0: (draft.K[k0], draft.V[k0]) for k0 in range(draft.depth)}
    kv.update(deep_kv)
    cache.commit(kv, n_acc, depth_h_at, cache.n)
    return v_tokens[:n_acc]


@torch.no_grad()
def spec_cycle(model, cache: CommittedCache, gamma: int, dd: int,
               mid: int | None, scheme: str, budget: int):
    """跑一轮（见模块 docstring）。返回 (emitted_tokens, cycle_stats)。"""
    L = model.cfg.n_layers
    draft_tokens, hdd, draft = _draft_phase(model, cache, gamma, dd)
    h_in = torch.cat(hdd, dim=0)                       # [γ+1, d]

    if scheme == "single":
        states, kv = run_layers(model, cache, h_in, start_layer=dd)
        Y = model.readout(states[-1])                  # [γ+1, V]
        v = [_argmax(Y[i]) for i in range(gamma + 1)]
        k = accept_prefix(draft_tokens, v, gamma)

        def depth_h_at(j):
            return {dd: hdd[j], L: states[-1][j:j + 1]}
        emitted = _finish(cache, draft, k, v, kv, depth_h_at(k))
        cost = (gamma + 1) * dd + (gamma + 1) * (L - dd)
        stats = {"accepted": k, "gamma_effective": gamma, "emitted": len(emitted),
                 "mid_accepted": None, "eq_layerpos": cost}
        return emitted[:budget], stats

    if scheme == "graded":
        assert mid is not None
        states6, kv6 = run_layers(model, cache, h_in, start_layer=dd,
                                  end_layer=mid)
        Y6 = model.readout(states6[-1])
        v6 = [_argmax(Y6[i]) for i in range(gamma + 1)]
        k6 = accept_prefix(draft_tokens, v6, gamma)
        h_mid = states6[-1][:k6 + 1]                   # [k6+1, d]
        states8, kv8 = run_layers(model, cache, h_mid, start_layer=mid)
        Y8 = model.readout(states8[-1])
        v8 = [_argmax(Y8[i]) for i in range(k6 + 1)]
        k8 = accept_prefix(draft_tokens, v8, k6)
        deep = {k0: kv6[k0] for k0 in range(dd, mid)}
        deep.update(kv8)

        def depth_h_at(j):
            return {dd: hdd[j], mid: states6[-1][j:j + 1],
                    L: states8[-1][j:j + 1]}
        emitted = _finish(cache, draft, k8, v8, deep, depth_h_at(k8))
        cost = (gamma + 1) * dd + (gamma + 1) * (mid - dd) + (k6 + 1) * (L - mid)
        stats = {"accepted": k8, "gamma_effective": gamma,
                 "emitted": len(emitted), "mid_accepted": k6,
                 "eq_layerpos": cost}
        return emitted[:budget], stats

    raise ValueError(f"unknown scheme {scheme!r}")


@torch.no_grad()
def spec_run(model, prompt: list[int], n_new: int, gamma: int = 4,
             draft_depth: int = 4, mid_depth: int | None = 6,
             scheme: str = "single", max_len: int | None = None):
    """自 spec 贪心生成。返回 (gen, stats, cache)：`cache` 是提交前缀的 KV
    （供一致性单测对照普通批量前向）。"""
    if scheme not in ("single", "graded"):
        raise ValueError(f"unknown scheme {scheme!r}")
    L = model.cfg.n_layers
    if draft_depth in (0, L):
        raise ValueError(f"draft_depth 必须在 (0, {L}) 内，收到 {draft_depth}")
    if scheme == "graded" and (mid_depth is None or not draft_depth < mid_depth < L):
        raise ValueError(f"graded 需要 draft_depth < mid_depth < {L}，收到 "
                         f"draft_depth={draft_depth} mid_depth={mid_depth}")
    max_len = model.cfg.seq_len if max_len is None else max_len
    if len(prompt) + n_new > max_len:
        raise ValueError("prompt + n_new 超出 max_len（位置嵌入硬上界）")

    cache, first = prefill(model, prompt)
    gen = [first]
    cycles: list[dict] = []
    eq = L * len(prompt)                               # prefill = L 层 × prompt 位置
    while len(gen) < n_new:
        # 位置嵌入硬上界：草稿要跑到位置 n+gamma（commit_input 用 pos_embed[n+gamma]）
        if cache.n + gamma + 1 > max_len:
            break
        emitted, cyc = spec_cycle(model, cache, gamma, draft_depth, mid_depth,
                                  scheme, n_new - len(gen))
        gen.extend(emitted)
        eq += cyc["eq_layerpos"]
        cyc["cycle"] = len(cycles)
        cyc["committed"] = cache.n
        cycles.append(cyc)
    gen = gen[:n_new]
    n_drafts = sum(c["gamma_effective"] for c in cycles)
    n_acc = sum(c["accepted"] for c in cycles)
    stats = {
        "scheme": scheme, "gamma": gamma, "draft_depth": draft_depth,
        "mid_depth": mid_depth, "prompt_len": len(prompt), "n_new": len(gen),
        "eq_layerpos": eq, "n_cycles": len(cycles), "cycles": cycles,
        "n_drafts": n_drafts, "n_accepted": n_acc,
        "acceptance": round(n_acc / n_drafts, 4) if n_drafts else None,
        "tokens_per_cycle": round(len(gen) / max(len(cycles), 1), 4),
    }
    return gen, stats, cache


@torch.no_grad()
def spec_generate(model, prompt: list[int], n_new: int, gamma: int = 4,
                  draft_depth: int = 4, mid_depth: int | None = 6,
                  scheme: str = "single", max_len: int | None = None):
    """`spec_run` 的便捷包装，只返回 (gen, stats)。"""
    gen, stats, _ = spec_run(model, prompt, n_new, gamma, draft_depth,
                             mid_depth, scheme, max_len)
    return gen, stats


@torch.no_grad()
def plain_greedy_cache(model, prompt: list[int], n_new: int,
                       exit_depth: int | None = None):
    """**与本实验 spec 路径共用同一套缓存机械**的普通贪心解码：每步一个位置
    （`run_layers` k=1）、无草稿、无验证。用来把「spec 调度」从「缓存实现质量」
    里剥离出来——`plain_greedy`（`DepthARDecoder`）是仓库原有路径，两者差 = 实现层
    开销，`plain_greedy_cache` vs spec = 纯调度差。"""
    L = model.cfg.n_layers
    fd = L if exit_depth is None else exit_depth
    if fd != L:
        raise ValueError("plain_greedy_cache 只定义全栈（fd=L）——固定退出深度请用 "
                         "plain_greedy（DepthARDecoder 支持 forced_depth）")
    cache, first = prefill(model, prompt)
    gen = [first]
    while len(gen) < n_new:
        if cache.n + 1 > model.cfg.seq_len:
            break
        states, kv = run_layers(model, cache, cache.pending, start_layer=0)
        depth_h = {j: states[j][:1] for j in range(len(states))}
        tok = _argmax(model.readout(states[-1][:1]))
        cache.commit(kv, 1, depth_h, cache.n)
        gen.append(tok)
    return gen[:n_new]


@torch.no_grad()
def plain_greedy(model, prompt: list[int], n_new: int, exit_depth: int | None = None):
    """同 checkpoint 的普通贪心解码（全栈或指定固定退出深度），用于口径对照。

    全栈等价于 `pathlm.depth_ar.generate(..., forced_depth=None)`；这里用
    `CommittedCache` 重写以便与 spec 路径共用同一套 KV 语义。"""
    from pathlm.depth_ar import DepthARDecoder, token_input as ti
    d = DepthARDecoder(model)
    L = model.cfg.n_layers
    fd = L if exit_depth is None else exit_depth
    h = logits = None
    for t in prompt:
        h, logits = d.step(ti(model, int(t), d.n), fd)
    gen = []
    for _ in range(n_new):
        gen.append(int(logits.argmax()))
        if len(gen) >= n_new:
            break
        inp = commit_input(model, logits, h, "hard", d.n)
        h, logits = d.step(inp, fd)
    return gen
