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

    def truncate(self, after_pos: int) -> None:
        """Drop cached positions > after_pos from every layer and re-anchor
        `self.n` at after_pos + 1. Spec decode uses this to discard the
        rejected suffix of a batched verification; because the accepted prefix
        was computed causally, every retained KV entry is exactly what the
        per-position path would have stored."""
        for i in list(self.kv):
            pos = self.layer_pos.get(i, [])
            keep = [j for j, p in enumerate(pos) if p <= after_pos]
            self.kv[i] = (self.kv[i][0][:, :, keep], self.kv[i][1][:, :, keep])
            self.layer_pos[i] = [pos[j] for j in keep]
        self.n = after_pos + 1

    def _run_stack_multi(self, h: torch.Tensor) -> torch.Tensor:
        """Run the full stack over K > 1 new positions in ONE forward, each
        query using its own eviction cutoff so the per-position logits are
        reproduced (early exit is not supported here: skipping layers mid-batch
        would desynchronise the K per-position caches). Returns the final h."""
        model = self.model
        B, K, d = h.shape
        from pathlm.model import Block  # local import: typing only
        blocks = [b for b in model.blocks if isinstance(b, Block)]
        p0 = self.n
        new_kv: dict[int, tuple] = {}
        for i, block in enumerate(blocks):
            n_heads, d_head = block.n_heads, d // block.n_heads
            q, k, v = block.qkv(block.ln1(h)).chunk(3, dim=-1)
            shape = lambda t: t.view(B, K, n_heads, d_head).transpose(1, 2)
            q, k, v = shape(q), shape(k), shape(v)
            old_pos = self.layer_pos.get(i, [])
            # union of old keys any query in the batch may need: eviction as of
            # the FIRST new position (the other queries only mask further).
            kept = [j for j, p in enumerate(old_pos)
                    if self.window <= 0 or p < self.anchors
                    or p0 - p < self.window]
            if i in self.kv and kept:
                ck = self.kv[i][0][:, :, kept]
                cv = self.kv[i][1][:, :, kept]
            elif i in self.kv:
                ck, cv = self.kv[i][0][:, :, :0], self.kv[i][1][:, :, :0]
            else:
                ck, cv = k[:, :, :0], v[:, :, :0]
            k_all = torch.cat([ck, k], dim=2)
            v_all = torch.cat([cv, v], dim=2)
            L = k_all.shape[2]
            mask = torch.zeros(K, L, dtype=torch.bool, device=h.device)
            # new keys are causal (position p0 + j attends new keys 0..j)
            mask[:, L - K:] = torch.ones(K, K, dtype=torch.bool,
                                         device=h.device).tril()
            if self.window > 0:  # per-query eviction of the old keys
                for j in range(K):
                    qpos = p0 + j
                    for t, p in enumerate(old_pos[jj] for jj in kept):
                        if p < self.anchors or qpos - p < self.window:
                            mask[j, t] = True
            else:
                mask[:, :L - K] = True
            a = F.scaled_dot_product_attention(
                q, k_all, v_all, attn_mask=mask.view(1, 1, K, L))
            a = a.transpose(1, 2).reshape(B, K, d)
            h = h + block.proj(a)
            h = h + block.mlp(block.ln2(h))
            if self.window > 0:
                # final cache keeps what survives eviction at the LAST query
                qfin = p0 + K - 1
                all_pos = [old_pos[jj] for jj in kept] + [p0 + j for j in range(K)]
                cols = [t for t, p in enumerate(all_pos)
                        if p < self.anchors or qfin - p < self.window]
                new_kv[i] = (k_all[:, :, cols], v_all[:, :, cols])
            else:
                new_kv[i] = (k_all, v_all)
        self.kv.update(new_kv)
        for i in new_kv:
            old_pos = self.layer_pos.get(i, [])
            kept = [j for j, p in enumerate(old_pos)
                    if self.window <= 0 or p < self.anchors
                    or p0 - p < self.window]
            all_pos = [old_pos[jj] for jj in kept] + [p0 + j for j in range(K)]
            if self.window > 0:
                qfin = p0 + K - 1
                all_pos = [p for p in all_pos
                           if p < self.anchors or qfin - p < self.window]
            self.layer_pos[i] = all_pos
        return h

    def step_multi(self, tokens: list[int]) -> tuple[torch.Tensor, dict]:
        """Process K > 1 tokens in one batched forward. Returns (final h,
        per-position MTP nodes). Updates the KV cache; `self.n` advances by K.
        `last_nodes` is set to the LAST position only (the plain path's form)."""
        model = self.model
        K = len(tokens)
        if K < 1:
            raise ValueError("step_multi needs at least one token")
        if self.n + K > model.mcfg.seq_len:
            raise ValueError(
                f"decode length {self.n + K} exceeds seq_len {model.mcfg.seq_len}: "
                "absolute position embeddings bound total length (M1 limitation)")
        dev = model.embed.weight.device
        idx = torch.tensor([list(tokens)], device=dev)
        h = model.embed(idx) + model.pos_embed.weight[self.n:self.n + K]
        h = cap_norm(h, model.mcfg.norm_cap)
        h = self._run_stack_multi(h)
        self.n += K
        nodes = model._mtp_nodes(h)
        self.last_nodes = _slice_nodes(nodes, K - 1)
        return h, nodes

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


def _slice_nodes(nodes: dict, j: int) -> dict:
    """Take position j out of a per-position MTP node dict, preserving the
    [B, 1, ...] shape the single-position path produces."""
    out: dict = {}
    for k, node in nodes.items():
        if isinstance(node, dict):
            out[k] = _slice_nodes(node, j)
        elif torch.is_tensor(node) and node.dim() >= 2:
            out[k] = node[:, j:j + 1]
        else:
            out[k] = node
    return out


@torch.no_grad()
def decode_spec(model: PathLM, prompt: torch.Tensor, n_new: int, pcap: PathConfig,
                max_drafts: int | None = None,
                draft_fn=None) -> tuple[list[int], dict]:
    """Medusa-style batched verification (greedy, temperature 0).

    At the current position we draft node-2..node-n_mtp and node-1's own next
    token, feed all of them in ONE forward, verify each draft against node-1's
    argmax at the corresponding position, commit the longest accepted prefix,
    and truncate the rejected suffix from the KV cache. Because the verifier is
    the same greedy node-1, the emitted sequence is identical to `decode`'s.

    `draft_fn(nodes, n)` overrides the draft source (tests inject wrong drafts).
    Returns (generated ids, stats with per-round accept counts).
    """
    if model.pcap.n_mtp < 2:
        return decode(model, prompt, n_new, pcap)
    dec = Decoder(model, window=pcap.window, anchors=pcap.anchors)
    # pi-lens-ignore: unchecked-throwing-call-python
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        # pi-lens-ignore: unchecked-throwing-call-python
        dec.step(int(tok))
    if max_drafts is None:
        max_drafts = model.pcap.n_mtp - 1  # node-2..node-n_mtp
    max_drafts = max(min(max_drafts, model.pcap.n_mtp - 1), 0)
    gen: list[int] = []
    accepts: list[int] = []
    n_forward = 0
    while len(gen) < n_new:
        nodes = dec.last_nodes
        assert nodes is not None, "step must populate last_nodes"
        v1 = int(nodes[1]["logits"][0, 0].argmax())
        if max_drafts == 0:
            gen.append(v1)
            if len(gen) < n_new:
                dec.step(v1)
            accepts.append(0)
            continue
        if draft_fn is None:
            drafts = [int(nodes[k]["logits"][0, 0].argmax())
                      for k in range(2, 2 + max_drafts)]
        else:
            drafts = [int(t) for t in draft_fn(nodes, max_drafts)]
        inputs = [v1] + drafts
        if dec.n + len(inputs) > model.mcfg.seq_len:
            break  # position budget exhausted; caller sees a short sequence
        _, fnodes = dec.step_multi(inputs)
        n_forward += 1
        n_acc = 0
        while n_acc < len(drafts):
            if int(fnodes[1]["logits"][0, n_acc].argmax()) != drafts[n_acc]:
                break
            n_acc += 1
        gen.append(v1)
        gen.extend(drafts[:n_acc])
        accepts.append(n_acc)
        if n_acc < len(drafts):
            dec.truncate(dec.n - len(inputs) + n_acc)  # reject suffix, re-anchor
            dec.last_nodes = _slice_nodes(fnodes, n_acc)
        else:
            dec.last_nodes = _slice_nodes(fnodes, len(inputs) - 1)
    return gen[:n_new], {
        "accepts": accepts,
        "n_forward": n_forward,
        "drafts_accepted": int(sum(accepts)),
        "drafts_offered": int(len(accepts) * max_drafts),
        "cache_len": dec.cache_len(),
    }


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
