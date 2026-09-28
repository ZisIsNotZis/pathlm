"""25 税收归因探针：门分布 + 墙钟（ticket 25；副记录口径）。

- eq 臂（深度自适应形态，B/C 臂）：全深度前向 batch 8 的逐层门分布
  （mean/p90/frac>0.5 → gate_distribution.json）+ DepthARDecoder 增量解码
  墙钟（full depth-12 / fixed exit-4 / uniform-exit U{0..12}；greedy hard
  commit；prefill 64 + 96 new × 3 prompts，口径对齐 24b）。
- dense 臂：dense 专用增量解码器 full-depth 墙钟（逐层 KV、无门/无退出，
  语义同 24b；tf 已在 results.json，proxy 口径）。

用法：python 25-probe-gates-wall.py <run_dir> <eq|dense> <out.json>
"""

import json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from pathlm.depth_ar import (DepthARConfig, DepthARModel, DepthARDecoder,
                             token_input, commit_input)
from pathlm.data import load_enwik8_full, batch


@torch.no_grad()
def gate_distribution(model, eval_arr, n_batches=2, batch_size=8):
    """全深度前向的逐层门分布（ticket 24 口径）：mean / p90 / frac>0.5。"""
    g = torch.Generator().manual_seed(11)
    L = model.cfg.n_layers
    model.eval()
    samples = [[] for _ in range(L)]
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, g)
        x = x.cuda()
        _, aux = model(x, exit_depths=None)
        # 逐元素门需重算（forward 只回均值）：沿状态链逐层取 σ(gate(h))
        h = aux["states"][0]
        for k in range(L):
            out, _ = model.blocks[k](h)
            gk_full = torch.sigmoid(model.gates[k](h)).float()
            samples[k].append(gk_full.flatten())
            h = h + gk_full.to(h.dtype) * (out - h)
    out = []
    for k, chunks in enumerate(samples):
        s = torch.cat(chunks)
        out.append({"layer": k, "mean": round(s.mean().item(), 4),
                    "p90": round(s.quantile(0.9).item(), 4),
                    "frac_gt_0.5": round((s > 0.5).float().mean().item(), 6)})
    return out


@torch.no_grad()
def eq_wall(model, prompts, n_new=96):
    """EQ 臂三分支墙钟（24b 口径）：full / fixed-4 / uniform-exit。"""
    L = model.cfg.n_layers
    out = {}
    for mode, fd_fn in (("full_depth_12", lambda i, g: L),
                        ("fixed_exit_4", lambda i, g: 4),
                        ("uniform_exit", None)):
        per_prompt = []
        for i, prompt in enumerate(prompts):
            g = torch.Generator().manual_seed(100 + i)
            dec = DepthARDecoder(model)
            t0 = time.perf_counter()
            h = logits = None
            for t in prompt.tolist():
                fd = (int(torch.randint(0, L + 1, (1,), generator=g))
                      if fd_fn is None else fd_fn(i, g))
                h, logits = dec.step(token_input(model, int(t), dec.n), fd)
            t_prefill = time.perf_counter()
            gen = []
            for _ in range(n_new):
                gen.append(int(logits.argmax()))
                if len(gen) >= n_new:
                    break
                inp = commit_input(model, logits, h, "hard", dec.n)
                fd = (int(torch.randint(0, L + 1, (1,), generator=g))
                      if fd_fn is None else fd_fn(i, g))
                h, logits = dec.step(inp, fd)
            t_end = time.perf_counter()
            legal = all(0 <= t < model.vocab_size for t in gen)
            per_prompt.append({
                "prefill_s": round(t_prefill - t0, 3),
                "gen_s": round(t_end - t_prefill, 3),
                "tok_s_gen": round(n_new / (t_end - t_prefill), 2),
                "tok_s_total": round((len(prompt) + n_new) / (t_end - t0), 2),
                "legal": bool(legal)})
        out[mode] = {
            "tok_s_gen_mean": round(sum(p["tok_s_gen"] for p in per_prompt) / len(per_prompt), 2),
            "tok_s_total_mean": round(sum(p["tok_s_total"] for p in per_prompt) / len(per_prompt), 2),
            "per_prompt": per_prompt}
    return out


@torch.no_grad()
def dense_wall(model, prompts, n_new=96):
    """dense 臂 full-depth 墙钟（24b 语义：逐层 KV，无门/无填充/无退出）。"""
    L = model.cfg.n_layers
    d, nh = model.cfg.d_model, model.cfg.n_heads
    shape = lambda t: t if t.dim() == 4 else t.view(1, 1, nh, d // nh).transpose(1, 2)
    per_prompt = []
    for prompt in prompts:
        K = [[] for _ in range(L)]
        V = [[] for _ in range(L)]
        h = logits = None
        t0 = time.perf_counter()
        for t in prompt.tolist():
            h = token_input(model, int(t), len(K[0]))
            for k0 in range(L):
                blk = model.blocks[k0]
                qkv = blk.qkv(blk.ln1(h))
                q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
                Kc = torch.cat(K[k0] + [shape(k)], dim=2) if K[k0] else shape(k)
                Vc = torch.cat(V[k0] + [shape(v)], dim=2) if V[k0] else shape(v)
                a = torch.nn.functional.scaled_dot_product_attention(shape(q), Kc, Vc)
                a = a.transpose(1, 2).reshape(1, 1, d)
                h = h + blk.proj(a)
                h = h + blk.mlp(blk.ln2(h))
                K[k0].append(shape(k))
                V[k0].append(shape(v))
            logits = model.readout(h)
        t_prefill = time.perf_counter()
        gen = []
        for _ in range(n_new):
            gen.append(int(logits.argmax()))
            if len(gen) >= n_new:
                break
            inp = commit_input(model, logits, h, "hard", len(K[0]))
            h = inp
            for k0 in range(L):
                blk = model.blocks[k0]
                qkv = blk.qkv(blk.ln1(h))
                q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
                Kc = torch.cat(K[k0] + [shape(k)], dim=2)
                Vc = torch.cat(V[k0] + [shape(v)], dim=2)
                a = torch.nn.functional.scaled_dot_product_attention(shape(q), Kc, Vc)
                a = a.transpose(1, 2).reshape(1, 1, d)
                h = h + blk.proj(a)
                h = h + blk.mlp(blk.ln2(h))
                K[k0].append(shape(k))
                V[k0].append(shape(v))
            logits = model.readout(h)
        t_end = time.perf_counter()
        legal = all(0 <= t < model.vocab_size for t in gen)
        per_prompt.append({
            "prefill_s": round(t_prefill - t0, 3),
            "gen_s": round(t_end - t_prefill, 3),
            "tok_s_gen": round(n_new / (t_end - t_prefill), 2),
            "tok_s_total": round((len(prompt) + n_new) / (t_end - t0), 2),
            "legal": bool(legal)})
    return {"full_depth_12": {
        "tok_s_gen_mean": round(sum(p["tok_s_gen"] for p in per_prompt) / len(per_prompt), 2),
        "tok_s_total_mean": round(sum(p["tok_s_total"] for p in per_prompt) / len(per_prompt), 2),
        "per_prompt": per_prompt}}


def main():
    run_dir, arm, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    res = json.load(open(f"{run_dir}/results.json"))
    cfg = DepthARConfig(**res["config"])
    model = DepthARModel(cfg, res["vocab_size"])
    model.load_state_dict(torch.load(f"{run_dir}/model.pt", map_location="cpu",
                                     weights_only=True))
    model.cuda().eval()
    train_arr, eval_arr, _ = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    g = torch.Generator().manual_seed(7)
    prompts, _ = batch(eval_arr, 3, 64, g)      # 3 prompt × 64，各臂同 prompt
    prompts = prompts.cuda()
    out = {"arm": arm, "run_dir": run_dir,
           "protocol": "greedy hard commit；prefill 64 + 96 new × 3 prompts "
                       "（generator seed 7，各臂同 prompt）；增量逐层 KV cache",
           "note": "Python 循环主导的小 batch 解码墙钟 = 副记录口径（主指标 = "
                   "全深度 bpc 与等效验证加速）"}
    if arm == "eq":
        out["gate_distribution"] = gate_distribution(model, eval_arr)
        out["wall"] = eq_wall(model, prompts)
    else:
        out["wall"] = dense_wall(model, prompts)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    summary = {f"wall.{m}": {"tok_s_gen": w["tok_s_gen_mean"],
                             "tok_s_total": w["tok_s_total_mean"]}
               for m, w in out["wall"].items()}
    if "gate_distribution" in out:
        summary["gate_mean_by_layer"] = [r["mean"] for r in out["gate_distribution"]]
        summary["gate_p90_by_layer"] = [r["p90"] for r in out["gate_distribution"]]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
