"""PathLM model (M0 skeleton). See docs/design.md for the stage map.

Implemented at M0: stages 0/1 (corruption, noise, pure-noise), L (shuffle/skip/
redo), stage 2 (MTP transform heads + confidence heads, self node k=0, chain
estimate for node 2), stage 3 transport=direct (latent retry). Tied E=U with
norm-capped inputs. Retry rounds are separate estimator passes (detached), per
the training recipe in docs/design.md §6.

Node semantics (design §4): node k at row i predicts the clean token t_{i+k}.
Node 0 is the self/repair estimate of the CURRENT token; node 1 is the standard
AR next-token head. Each round gets its own freshly sampled layer path (design
§3: "fresh shuffle + fresh skips" per pass).
"""

import random
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, PathConfig, PathSample, sample_path


def cap_norm(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Rescale rows whose L2 norm exceeds cap (stage-E norm cap)."""
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > cap, x * cap / norm.clamp_min(1e-6), x)


class Block(nn.Module):
    """Pre-LN causal transformer block ('enhancer' in design terms)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_mult * d), nn.GELU(),
            nn.Linear(cfg.mlp_mult * d, d), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (block output, per-head mean attended distance [B, H]).
        The distance tensor is None when dist_pen == 0 (exact no-op path);
        when dist_pen > 0 attention is computed explicitly (softmax over
        causal + -pen*log(1+d) bias) so both the bias and the telemetry come
        from the same probabilities."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        qs, ks, vs = shape(q), shape(k), shape(v)
        dist = None
        if dist_pen > 0:
            i = torch.arange(T, device=x.device).unsqueeze(1)
            j = torch.arange(T, device=x.device).unsqueeze(0)
            dmat = (i - j).clamp_min(0).float()          # query-key distance
            causal = (j <= i).view(1, 1, T, T)
            dist_log = torch.log1p(dmat.clamp_min(0.0))  # non-negative domain
            bias = (-dist_pen * dist_log).masked_fill(~causal, float("-inf"))
            bias = bias.expand(B, 1, T, T)
            if attn_mask is not None:
                bias = bias + attn_mask.to(bias.dtype)   # eviction mask (bool 0/1)
            scores = qs @ ks.transpose(-2, -1) / (d // self.n_heads) ** 0.5 + bias
            w = scores.softmax(-1)                       # [B, H, T, T]
            a = w @ vs
            dist = (w * dmat.view(1, 1, T, T)).sum(-1).mean(2)  # [B, H] (mean over queries)
        else:
            a = F.scaled_dot_product_attention(qs, ks, vs,
                                               attn_mask=attn_mask,
                                               is_causal=attn_mask is None)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x)), dist


class TransformHead(nn.Module):
    """Residual latent-to-latent map: latent_{i+k} ~ h + MLP_k(h)."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(h)


class PathLM(nn.Module):
    def __init__(self, mcfg: ModelConfig, pcap: PathConfig, vocab_size: int):
        super().__init__()
        self.mcfg, self.pcap = mcfg, pcap
        # Seeded RNG for in-forward path sampling (token-retry round): derived
        # from the construction-time torch seed, so training is reproducible.
        self._path_rng = random.Random(torch.initial_seed() ^ 0x5EED)
        d = mcfg.d_model
        self.vocab_size = vocab_size
        self.n_real_tokens = vocab_size - 1  # last slot = [mask]
        self.mask_token = vocab_size - 1
        # Tied E=U: embedding rows have norm ~1 (init std = d^-0.5), U = E^T.
        # No final LayerNorm: the residual stream keeps the embedding geometry
        # end-to-end (design §1); readout scale is learned through U.
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=d ** -0.5)
        self.pos_embed = nn.Embedding(mcfg.seq_len, d)  # standard AR decoder requirement
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(mcfg) for _ in range(mcfg.n_layers))
        self.transforms = nn.ModuleList(TransformHead(d) for _ in range(pcap.n_mtp))
        self.conf = nn.ModuleList(nn.Linear(d, 1) for _ in range(pcap.n_mtp + 1))
        self.conf_chain2 = nn.Linear(d, 1)  # separate head for the chained node-2 estimate
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- stages 0/1 ----------

    def _input_latents(self, tokens: torch.Tensor):
        """Corrupt tokens (stage 0), embed + perturb (stage 1).
        Returns (latents, corrupt_mask). Wrong-token sampling never returns the
        [mask] slot nor (up to a vanishingly rare wrap collision) the original
        token, so corrupt_mask marks exactly the changed positions."""
        B, T = tokens.shape
        pc, dev = self.pcap, tokens.device
        u0 = torch.rand(B, T, device=dev)
        mask_pos = u0 < pc.corrupt_mask
        wrong_pos = (u0 >= pc.corrupt_mask) & (u0 < pc.corrupt_mask + pc.corrupt_wrong)
        x = torch.where(mask_pos, torch.full_like(tokens, self.mask_token), tokens)
        rand_tok = torch.randint_like(tokens, self.n_real_tokens)
        rand_tok = (rand_tok + (rand_tok == tokens).int()) % self.n_real_tokens  # no collision
        x = torch.where(wrong_pos, rand_tok, x)
        h = self.embed(x) + self.pos_embed.weight[:T]
        if pc.perturb_noise > 0 or pc.pure_noise > 0:
            u1 = torch.rand(B, T, device=dev)
            sigma = pc.noise_sigma * self.mcfg.norm_cap
            noise = torch.randn_like(h) * sigma
            h = h + noise * (u1 < pc.perturb_noise).unsqueeze(-1)
            amp = torch.rand(B, T, 1, device=dev) * self.mcfg.norm_cap
            direction = torch.randn_like(h)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            h = torch.where((u1 >= 1 - pc.pure_noise).unsqueeze(-1), direction * amp, h)
        return cap_norm(h, self.mcfg.norm_cap), wrong_pos | mask_pos

    # ---------- stage L ----------

    def _run_layers(self, h: torch.Tensor, path: PathSample,
                    attn_mask: torch.Tensor | None = None,
                    depth_hook: Callable[[int, torch.Tensor], None] | None = None,
                    dist_pen: float = 0.0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Execute the sampled path. depth_hook(depth, h) fires after each
        executed layer (depth = layers applied so far) for dense-exit training.
        dist_pen > 0 adds the X1 attention distance bias; dists collects the
        per-layer per-head mean attended distance. Returns (h, dists) —
        dists is [] when the penalty is off."""
        depth = 0
        dists: list[torch.Tensor] = []
        for i, r in zip(path.layer_order, path.layer_repeats):
            for _ in range(r):
                h, dist = self.blocks[i](h, attn_mask=attn_mask, dist_pen=dist_pen)
                if dist is not None:
                    dists.append(dist)
                depth += 1
                if depth_hook is not None:
                    depth_hook(depth, h)
        return h, dists

    def _eviction_mask(self, T: int, device) -> torch.Tensor | None:
        """Causal AND eviction: position i attends j iff (j < anchors) or
        (i - j < window). Anchors (first positions) are the long-range channel."""
        pc = self.pcap
        if pc.window <= 0:
            return None
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        causal = j <= i
        in_window = (i - j) < pc.window
        is_anchor = j < pc.anchors
        return (causal & (in_window | is_anchor)).view(1, 1, T, T)

    def _transport(self, h: torch.Tensor) -> torch.Tensor:
        """Stage-3 re-entry transform (retry rounds only). Only transforms that
        can inject information are meaningful here; direct re-derives the same
        fixed point (M0 gate d)."""
        U = self.embed.weight  # [V, d], tied E=U
        if self.pcap.transport == "linear":
            return cap_norm((h @ U.T) @ U, self.mcfg.norm_cap)  # project onto vocab span
        if self.pcap.transport == "soft":
            # expected embedding under the current token distribution
            return cap_norm((h @ U.T).softmax(-1) @ U, self.mcfg.norm_cap)
        return h

    # ---------- stage 2 ----------

    def _mtp_nodes(self, h: torch.Tensor) -> dict:
        """One round of MTP node outputs. Node k=0 is the self estimate;
        node k>=1 is the direct transform estimate; node 1 additionally
        carries the chained estimate of node 2 (T1 @ T1) with its own head."""
        nodes = {}
        for k in range(self.pcap.n_mtp + 1):
            latent = h if k == 0 else self.transforms[k - 1](h)
            node = {"latent": latent, "logits": latent @ self.embed.weight.T,
                    "conf": self.conf[k](latent).squeeze(-1)}
            if k == 1 and self.pcap.n_mtp >= 2:
                chain = self.transforms[0](latent)
                node["chain2"] = {"latent": chain, "logits": chain @ self.embed.weight.T,
                                  "conf": self.conf_chain2(chain).squeeze(-1)}
            nodes[k] = node
        return nodes

    @staticmethod
    def _node_loss(loss, node: dict, targets: torch.Tensor, vocab_size: int):
        """Per-component CE + confidence BCE against the token-identity event."""
        logits = node["logits"][:, :targets.shape[1]]
        loss = loss + F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        with torch.no_grad():
            hit = (logits.argmax(-1) == targets).float()
        loss = loss + F.binary_cross_entropy_with_logits(node["conf"][:, :targets.shape[1]], hit)
        node["hit"], node["tgt"] = hit, targets
        return loss

    # ---------- forward / loss ----------

    def forward(self, tokens: torch.Tensor, paths: list[PathSample], targets: torch.Tensor):
        """tokens = the CLEAN input sequence (corruption happens internally).
        targets = the clean token stream the nodes predict into: node k of round
        r is supervised on targets[:, k:] (with targets = tokens, node k
        predicts t_{i+k}; node 0 is the self/repair estimate of the current
        token). paths: one sampled layer path per round (design: fresh
        shuffle/skips each pass)."""
        h, corrupt_mask = self._input_latents(tokens)
        pc = self.pcap
        attn_mask = self._eviction_mask(tokens.shape[1], tokens.device)
        aux = {"corrupt_mask": corrupt_mask, "rounds": [], "depth_ce": []}
        loss = tokens.new_zeros(()).float()
        total_depths = 0
        mix = None  # mixture accumulator (design §3 amendment): W, S (prob sum), L (latent sum)
        for r, path in enumerate(paths):
            if r > 0:
                if pc.reentry_mix and mix is not None and pc.transport != "none":
                    h = self._mixture_reentry(mix)
                elif pc.transport != "direct":
                    h = self._transport(h)  # overwrite re-entry (retry rounds only)
            pending_dense: list = []
            hook_fn: Callable[[int, torch.Tensor], None] | None = None
            if r == 0 and pc.w_dense_exit > 0:
                total_depths = sum(path.layer_repeats)

                def depth_hook(depth: int, h_d: torch.Tensor) -> None:
                    # Dense early-exit supervision (L2): supervise the MTP block
                    # at every depth 1..n-1 of the base pass, so early exits are
                    # calibrated prefixes of the final estimate (design §6). CE AND
                    # confidence BCE — the decode exit gate reads sigmoid(conf)
                    # at depth d < n, so the head must be calibrated there too.
                    # (total_depths/path close over this loop iteration's values,
                    # which are fixed per pass — no late-binding hazard.)
                    nodes = self._mtp_nodes(h_d)
                    dloss = h_d.new_zeros(())
                    for k, node in nodes.items():
                        tgt = targets[:, k:]
                        logits = node["logits"][:, :tgt.shape[1]]
                        dloss = dloss + F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                                        tgt.reshape(-1))
                        with torch.no_grad():
                            hit = (logits.argmax(-1) == tgt).float()
                        dloss = dloss + F.binary_cross_entropy_with_logits(
                            node["conf"][:, :tgt.shape[1]], hit)
                    aux["depth_ce"].append(dloss.detach() / (pc.n_mtp + 1))
                    if pc.collect_depth_logits and 1 in nodes:
                        aux.setdefault("depth_logits", []).append(nodes[1]["logits"].detach())
                        aux.setdefault("depth_conf", []).append(nodes[1]["conf"].detach())
                    if depth < total_depths:  # final depth is the normal pass below
                        pending_dense.append(pc.w_dense_exit * dloss)
                hook_fn = depth_hook
            h, _dists = self._run_layers(h, path, attn_mask=attn_mask,
                                         depth_hook=hook_fn, dist_pen=pc.dist_pen)
            if _dists:
                # per-head mean attended distance, averaged over executed layers:
                # [n_layers, B, H] -> [B, H]
                aux["attn_dist"] = torch.stack(_dists).mean(0)
            for dloss in pending_dense:
                loss = loss + dloss
            nodes = self._mtp_nodes(h)
            for k, node in nodes.items():
                tgt = targets[:, k:]
                loss = self._node_loss(loss, node, tgt, self.vocab_size)
                if "chain2" in node:
                    loss = self._node_loss(loss, node["chain2"], targets[:, 2:], self.vocab_size)
            if r == 0 and pc.w_consistency > 0 and pc.n_mtp >= 2:
                cons = 1 - F.cosine_similarity(nodes[2]["latent"], nodes[1]["chain2"]["latent"],
                                               dim=-1).mean()
                loss = loss + pc.w_consistency * cons
                aux["consistency"] = cons.detach()
            # Diversity pressure (TTS follow-up): with multiple paths per batch,
            # reward useful disagreement between the paths' node-1 predictions —
            # capped so the optimum cannot be uniform outputs. Each path still
            # minimizes its own CE; this term pays the paths to decorrelate.
            if pc.w_diversity > 0 and len(paths) > 1 and "diversity_pairs" in aux:
                prev_logits = aux["diversity_pairs"]  # node-1 logits of path r-1
                p = nodes[1]["logits"][:, :-1].log_softmax(-1).exp()
                q = prev_logits[:, :-1].log_softmax(-1).exp()  # same row span
                js = 0.5 * (F.kl_div(p.log(), q, reduction="none").sum(-1)
                            + F.kl_div(q.log(), p, reduction="none").sum(-1)).mean()
                loss = loss - pc.w_diversity * js.clamp_max(2.0)
                aux["diversity"] = js.detach()
            if 1 in nodes:  # node-1 may be absent (n_mtp=0)
                aux["diversity_pairs"] = nodes[1]["logits"].detach()
            aux["rounds"].append(nodes)
            h = h.detach()  # retry rounds are separate estimator passes
            if pc.reentry_mix:
                # accumulate: W += w_r, S += w_r·P_r, L += w_r·h_r (all detached —
                # aggregation stays inference-time math per design §5)
                w = torch.sigmoid(nodes[0]["conf"]).detach().unsqueeze(-1)  # [B, T, 1]
                P = nodes[0]["logits"].softmax(-1).detach()
                if mix is None:
                    mix = {"W": torch.ones_like(w), "S": P, "L": h.detach().unsqueeze(0)}
                else:
                    mix["W"] = mix["W"] + w
                    mix["S"] = mix["S"] + w * P
                    mix["L"] = torch.cat([mix["L"], (w * h.detach()).unsqueeze(0)])
            # Stage-4 token retry: discrete re-entry. With reentry_mix, the
            # re-embedded token is the argmax of the ACCUMULATED distribution
            # (the init anchors it — a wrong round cannot fully take over, the
            # M1/R1 lesson). Fires once, after the final latent round.
            if r == len(paths) - 1 and path.token_retry:
                if pc.reentry_mix and mix is not None:
                    pred = (mix["S"] / mix["W"]).argmax(-1)  # [B, T] mixture vote
                else:
                    pred = nodes[0]["logits"].argmax(-1)  # single-round gamble
                h = cap_norm(self.embed(pred) + self.pos_embed.weight[:pred.shape[1]],
                             self.mcfg.norm_cap)
                h = h.detach()
                path2 = sample_path(pc, self._path_rng, self.mcfg.n_layers)
                h, _ = self._run_layers(h, path2, attn_mask=attn_mask)
                nodes = self._mtp_nodes(h)
                for k, node in nodes.items():
                    loss = self._node_loss(loss, node, targets[:, k:], self.vocab_size)
                aux["rounds"].append(nodes)
        return loss, aux

    def _mixture_reentry(self, mix: dict) -> torch.Tensor:
        """Re-entry state from the accumulated mixture: S = Σ w·P / Σ w is the
        confidence-weighted distribution over tokens; per transport — soft:
        expected embedding of S; linear: project the weighted latent mean;
        direct: the weighted latent mean itself. All detached inputs."""
        W = mix["W"]                                   # [B, T, 1]
        if self.pcap.transport == "soft":
            S = mix["S"] / W
            return cap_norm(S @ self.embed.weight, self.mcfg.norm_cap)
        L_bar = mix["L"].sum(0) / W                    # weighted latent mean
        if self.pcap.transport == "linear":
            U = self.embed.weight
            return cap_norm((L_bar @ U.T) @ U, self.mcfg.norm_cap)
        return cap_norm(L_bar, self.mcfg.norm_cap)  # direct
