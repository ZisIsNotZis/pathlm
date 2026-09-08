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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           attn_mask=attn_mask, is_causal=attn_mask is None)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


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
                    attn_mask: torch.Tensor | None = None, depth_hook=None):
        """Execute the sampled path. depth_hook(depth, h) fires after each
        executed layer (depth = layers applied so far) for dense-exit training."""
        depth = 0
        for i, r in zip(path.layer_order, path.layer_repeats):
            for _ in range(r):
                h = self.blocks[i](h, attn_mask=attn_mask)
                depth += 1
                if depth_hook is not None:
                    depth_hook(depth, h)
        return h

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
    def _depth_loss(h_d: torch.Tensor, targets: torch.Tensor, n_mtp: int,
                    vocab_size: int) -> torch.Tensor:
        """One depth's MTP losses: CE + confidence BCE per node (the conf heads
        must be calibrated at intermediate depths too — the decode exit gate
        reads sigmoid(conf) at depth d < n)."""
        nodes = PathLM._mtp_nodes_static(h_d, n_mtp)
        dloss = h_d.new_zeros(())
        for k, node in nodes.items():
            tgt = targets[:, k:]
            logits = node["logits"][:, :tgt.shape[1]]
            dloss = dloss + F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                hit = (logits.argmax(-1) == tgt).float()
            dloss = dloss + F.binary_cross_entropy_with_logits(
                node["conf"][:, :tgt.shape[1]], hit)
        return dloss

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
        for r, path in enumerate(paths):
            if r > 0 and pc.transport != "direct":
                h = self._transport(h)  # re-entry transform (retry rounds only)
            pending_dense: list = []
            depth_hook = None
            if r == 0 and pc.w_dense_exit > 0:
                total_depths = sum(path.layer_repeats)

                def depth_hook(depth: int, h_d: torch.Tensor, _p=path, _n=total_depths):
                    # Dense early-exit supervision (L2): supervise the MTP block
                    # at every depth 1..n-1 of the base pass, so early exits are
                    # calibrated prefixes of the final estimate (design §6). CE AND
                    # confidence BCE — the decode exit gate reads sigmoid(conf)
                    # at depth d < n, so the head must be calibrated there too.
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
                    if depth < _n:  # final depth is the normal pass below
                        pending_dense.append(pc.w_dense_exit * dloss)
            h = self._run_layers(h, path, attn_mask=attn_mask, depth_hook=depth_hook)
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
            aux["rounds"].append(nodes)
            h = h.detach()  # retry rounds are separate estimator passes
            # Stage-4 token retry: discrete re-entry — re-embed the self node's
            # predicted correction and re-run the stack. This is the channel
            # that rewrites state hardest (the M0 lesson: information must
            # enter the loop for a retry to help). Fires once, after the
            # final latent round.
            if r == len(paths) - 1 and path.token_retry:
                pred = nodes[0]["logits"].argmax(-1)  # [B, T] predicted corrections
                h = cap_norm(self.embed(pred) + self.pos_embed.weight[:pred.shape[1]],
                             self.mcfg.norm_cap)
                h = h.detach()
                path2 = sample_path(pc, self._path_rng, self.mcfg.n_layers)
                h = self._run_layers(h, path2, attn_mask=attn_mask)
                nodes = self._mtp_nodes(h)
                for k, node in nodes.items():
                    loss = self._node_loss(loss, node, targets[:, k:], self.vocab_size)
                aux["rounds"].append(nodes)
        return loss, aux
