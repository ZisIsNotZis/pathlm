"""PathLM model (M0 skeleton). See docs/design.md for the stage map.

Implemented at M0: stages 0/1 (corruption, noise, pure-noise), L (shuffle/skip/
redo), stage 2 (MTP transform heads + confidence heads, self node k=0, chain
estimate for node 2), stage 3 transport=direct (latent retry). Tied E=U with
norm-capped inputs. Retry rounds are separate estimator passes (detached), per
the training recipe in docs/design.md §6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, PathConfig, PathSample


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
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
        d = mcfg.d_model
        self.vocab_size = vocab_size
        self.mask_token = vocab_size - 1  # last slot = [mask]
        # Tied E=U: embedding rows have norm ~1 (init std = d^-0.5), U = E^T
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=d ** -0.5)
        self.blocks = nn.ModuleList(Block(mcfg) for _ in range(mcfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.transforms = nn.ModuleList(TransformHead(d) for _ in range(pcap.n_mtp))
        self.conf = nn.ModuleList(nn.Linear(d, 1) for _ in range(pcap.n_mtp + 1))
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.embed:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- stages 0/1 ----------

    def _input_latents(self, tokens: torch.Tensor):
        """Corrupt tokens (stage 0), embed + perturb (stage 1).
        Returns (latents, corrupt_mask)."""
        B, T = tokens.shape
        pc, dev = self.pcap, tokens.device
        u0 = torch.rand(B, T, device=dev)
        mask_pos = u0 < pc.corrupt_mask
        wrong_pos = (u0 >= pc.corrupt_mask) & (u0 < pc.corrupt_mask + pc.corrupt_wrong)
        x = torch.where(mask_pos, torch.full_like(tokens, self.mask_token), tokens)
        x = torch.where(wrong_pos, torch.randint_like(tokens, self.vocab_size), x)
        h = self.embed(x)
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

    def _run_layers(self, h: torch.Tensor, path: PathSample) -> torch.Tensor:
        for i, r in zip(path.layer_order, path.layer_repeats):
            for _ in range(r):
                h = self.blocks[i](h)
        return self.ln_f(h)

    # ---------- stage 2 ----------

    def _mtp_nodes(self, h: torch.Tensor) -> dict:
        """One round of MTP node outputs. Node k=0 is the self estimate;
        node k>=1 is the direct transform estimate; node 1 additionally
        carries the chained estimate of node 2 (T1 @ T1)."""
        nodes = {}
        for k in range(self.pcap.n_mtp + 1):
            latent = h if k == 0 else self.transforms[k - 1](h)
            node = {"latent": latent, "logits": latent @ self.embed.weight.T,
                    "conf": self.conf[k](latent).squeeze(-1)}
            if k == 1 and self.pcap.n_mtp >= 2:
                chain = self.transforms[0](latent)
                node["chain2"] = {"latent": chain, "logits": chain @ self.embed.weight.T,
                                  "conf": self.conf[2](chain).squeeze(-1)}
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

    def forward(self, tokens: torch.Tensor, path: PathSample, targets: torch.Tensor):
        """Per-component CE + confidence BCE (+ consistency loss) over all rounds."""
        h, corrupt_mask = self._input_latents(tokens)
        pc = self.pcap
        aux = {"corrupt_mask": corrupt_mask, "rounds": []}
        loss = tokens.new_zeros(()).float()
        for r in range(path.n_retries + 1):
            h = self._run_layers(h, path)
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
        return loss, aux
