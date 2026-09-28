"""24b 拆混杂探针：墙钟 + dense-tf（ticket 24b）。

- EQ 臂（深度自适应形态）：DepthARDecoder 增量解码墙钟，full depth-12 /
  fixed exit-4 / uniform-exit U{0..12} 三分支；greedy hard commit；prefill 64
  + 96 new × 3 prompts（口径对齐 100m-s0/decode_wall.json：增量逐层 KV cache，
  Python 循环主导的小 batch 口径，方向性参考）。tok/s 报两口径：
  gen-only（96 token / 生成段墙钟）与含 prefill（160 token / 全循环）。
- dense 臂（同引擎 dense）：dense 专用增量解码器（逐层 KV cache，无门/无
  填充/无退出——所有 position 跑全栈，缓存永不缺位），full-depth 墙钟 +
  tf@4/tf@6（depth-k 残差态 tied 读出 vs depth-L 读出 argmax 一致率；
  dense 中间读出无直接监督，系截断资产 proxy，如实标注）。
- uniform-exit wall 对 dense 臂 N/A（引擎无退出机制，逐 position 混深截断
  在 dense 语义下未定义）。

用法：python 24b-probe-wall-tf.py <run_dir> <eq|dense> <out.json>
"""

import json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from pathlm.depth_ar import (DepthARConfig, DepthARModel, DepthARDecoder,
                             token_input, commit_input, tf_acceptance, LN2)
from pathlm.data import load_enwik8_full, batch


@torch.no_grad()
def eq_wall(model, prompts, n_new=96):
    """EQ 臂三分支墙钟。depth_mode: full / fixed-4 / uniform（逐 position
    U{0..L}，含 prefill 步——训练分布口径）。"""
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
    """dense 臂 full-depth 增量解码墙钟：逐层 KV cache，无门/无填充/无退出
    （所有 position 跑全栈，缓存永不缺位；块语义与 Block.forward SDPA 路
    径逐位一致）。"""
    L = model.cfg.n_layers
    d, nh = model.cfg.d_model, model.cfg.n_heads
    shape = lambda t: t if t.dim() == 4 else t.view(1, 1, nh, d // nh).transpose(1, 2)
    per_prompt = []
    for prompt in prompts:
        K = [[] for _ in range(L)]
        V = [[] for _ in range(L)]
        h = None
        logits = None
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
    prompts, _ = batch(eval_arr, 3, 64, g)      # 3 个 prompt × 64，两臂同 prompt
    prompts = prompts.cuda()
    wall = eq_wall(model, prompts) if arm == "eq" else dense_wall(model, prompts)
    out = {"arm": arm, "run_dir": run_dir, "wall": wall,
           "protocol": "greedy hard commit；prefill 64 + 96 new × 3 prompts "
                       "（generator seed 7，两臂同 prompt）；增量逐层 KV cache；"
                       "tok/s 报 gen-only 与含 prefill 两口径",
           "note": "Python 循环+kernel-launch 主导的小 batch 解码，墙钟为方向性"
                   "参考（非 FLOPs 口径）"}
    if arm == "dense":
        tf = tf_acceptance(model, eval_arr, device="cuda", n_batches=20)
        out["tf_acceptance"] = tf
        out["tf_note"] = ("dense 中间深度读出无直接监督（仅最终 CE），tf 为 "
                          "depth-k 残差态 tied 读出与 depth-L 读出的 argmax "
                          "一致率——截断资产 proxy，非训练出口资产")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "wall"} | {
        f"wall.{m}": {"tok_s_gen": w["tok_s_gen_mean"],
                      "tok_s_total": w["tok_s_total_mean"]}
        for m, w in wall.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
