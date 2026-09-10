"""Inference-time decode for PathLM: incremental KV-cache generation with the
elastic-compute knobs actually applied — eviction window + anchors (no
re-prefill ring-buffer decoding, the X2 gate), a dense-exit threshold (stop
applying layers once prob_0 is confident; meaningful for L2-trained models),
and a latent-retry round driven by low prob_0 with the configured transport.

B=1 only (deployment semantics). Each layer keeps its own K/V cache AND its
own list of cached absolute positions: early exit skips deeper layers, so
those caches legitimately never see the token (correct causal semantics —
future tokens attending a skipped layer must not find it there). Eviction
drops positions outside (anchors ∪ last `window` positions) per layer, so
caches stay bounded and no prefix recompute is ever needed.
"""

import torch
import torch.nn.functional as F

from .config import PathConfig
from .model import PathLM, cap_norm


class Decoder:
    """Incremental single-sequence decoder."""

    def __init__(self, model: PathLM, window: int = 0, anchors: int = 0):
        self.model = model
        self.window, self.anchors = window, anchors
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # layer -> (k, v)
        self.layer_pos: dict[int, list[int]] = {}  # layer -> cached absolute positions
        self.n = 0                                 # absolute position of the NEXT token
        self.last_nodes: dict | None = None

    def _kept_for(self, i: int) -> list[int]:
        """Indices (into layer i's cache) that survive eviction this step."""
        pos = self.layer_pos.get(i, [])
        if self.window <= 0:
            return list(range(len(pos)))
        return [j for j, p in enumerate(pos)
                if p < self.anchors or self.n - p < self.window]  # matches the training mask

    def _run_stack(self, h: torch.Tensor, exit_threshold: float | None):
        """Run layers over h [1, 1, d], attending each layer's (evicted)
        cache + current position. Returns (h, depth_used)."""
        model = self.model
        B, _, d = h.shape
        depth, depth_used = 0, model.mcfg.n_layers
        new_kv: dict[int, tuple] = {}
        from pathlm.model import Block  # local import: typing only
        blocks = [b for b in model.blocks if isinstance(b, Block)]
        for i, block in enumerate(blocks):
            n_heads, d_head = block.n_heads, d // block.n_heads
            q, k, v = block.qkv(block.ln1(h)).chunk(3, dim=-1)
            shape = lambda t: t.view(B, 1, n_heads, d_head).transpose(1, 2)
            q, k, v = shape(q), shape(k), shape(v)
            kept = self._kept_for(i)
            if i in self.kv and kept:
                k_all = torch.cat([self.kv[i][0][:, :, kept], k], dim=2)
                v_all = torch.cat([self.kv[i][1][:, :, kept], v], dim=2)
            else:
                k_all, v_all = k, v
            new_kv[i] = (k_all, v_all)
            a = F.scaled_dot_product_attention(q, k_all, v_all)
            a = a.transpose(1, 2).reshape(B, 1, d)
            h = h + block.proj(a)
            h = h + block.mlp(block.ln2(h))
            depth += 1
            if exit_threshold is not None and depth < model.mcfg.n_layers:
                if torch.sigmoid(model.conf[0](h)).item() >= exit_threshold:
                    depth_used = depth
                    break
        self.kv.update(new_kv)  # only executed layers get the current position
        for i in new_kv:
            pos = self.layer_pos.setdefault(i, [])
            self.layer_pos[i] = [pos[j] for j in self._kept_for(i)] + [self.n]
        return h, depth_used

    def step(self, token: int, exit_threshold: float | None = None) -> tuple[torch.Tensor, int]:
        """Process one token; returns (final latent [1, 1, d], depth used)."""
        model = self.model
        if self.n >= model.mcfg.seq_len:
            raise ValueError(
                f"decode length {self.n + 1} exceeds seq_len {model.mcfg.seq_len}: "
                "absolute position embeddings bound total length (M1 limitation; "
                "sliding-window position handling is future work)")
        dev = model.embed.weight.device
        h = model.embed(torch.tensor([[token]], device=dev)) \
            + model.pos_embed.weight[self.n:self.n + 1]
        h = cap_norm(h, model.mcfg.norm_cap)
        h, depth_used = self._run_stack(h, exit_threshold)
        self.n += 1
        self.last_nodes = model._mtp_nodes(h)
        return h, depth_used

    def cache_len(self) -> int:
        """Number of prefix positions currently cached (layer 0)."""
        return len(self.layer_pos.get(0, []))

    def retry(self, h: torch.Tensor) -> torch.Tensor:
        """One latent-retry round: pop the current position from every layer
        cache that holds it, transport-transform the latent, re-run the full
        stack (layers skipped by an earlier exit attend their older cache)."""
        cur = self.n - 1  # absolute position of the just-processed token
        for i in self.kv:
            pos = self.layer_pos[i]
            if pos and pos[-1] == cur:
                self.kv[i] = (self.kv[i][0][:, :, :-1], self.kv[i][1][:, :, :-1])
                self.layer_pos[i] = pos[:-1]
        self.n = cur
        h = self.model._transport(h)
        h, _ = self._run_stack(h, exit_threshold=None)
        self.n = cur + 1  # re-run_stack consumed position cur again; restore
        self.last_nodes = self.model._mtp_nodes(h)
        return h


@torch.no_grad()
def decode(model: PathLM, prompt: torch.Tensor, n_new: int, pcap: PathConfig,
           exit_threshold: float | None = None, retry_threshold: float | None = None,
           temperature: float = 0.0, generator: torch.Generator | None = None) -> tuple[list[int], dict]:
    """Greedy (temperature=0) or sampled decode with the configured elastic
    knobs. Retry fires when the final node-0 confidence falls below
    retry_threshold (and a transport exists). Returns (generated ids, stats)."""
    dec = Decoder(model, window=pcap.window, anchors=pcap.anchors)
    h, _ = dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        # pi-lens-ignore: unchecked-throwing-call-python
        h, _ = dec.step(int(tok))  # prefill: always full depth
    gen, depths, retries = [], [], []
    assert dec.last_nodes is not None, "decode() must run dec.step() before sampling"
    for _ in range(n_new):
        nodes = dec.last_nodes
        n_retry = 0
        if (retry_threshold is not None and pcap.transport != "none"
                and torch.sigmoid(nodes[0]["conf"]).item() < retry_threshold):
            h = dec.retry(h)
            nodes = dec.last_nodes
            n_retry = 1
        assert nodes is not None, "step/retry must populate last_nodes"
        logits = nodes[1]["logits"][0, 0] / max(temperature, 1e-6)
        if temperature <= 0:
            # pi-lens-ignore: unchecked-throwing-call-python
            nxt = int(logits.argmax())
        else:
            # pi-lens-ignore: unchecked-throwing-call-python
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator))
        gen.append(nxt)
        if len(gen) < n_new:  # no need to process the final generated token
            h, depth = dec.step(nxt, exit_threshold=exit_threshold)
            depths.append(depth)
        retries.append(n_retry)
    return gen, {"depths": depths, "retries": retries}
