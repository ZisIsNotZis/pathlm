"""2632 · 追加探针（把「α₁ 排序随前缀 regime 翻转」钉死为可复现证据）。

三个探针（全部纯推理，CPU）：

P1 **incremental == batched**：逐 token `Decoder.step` 的 next-token 准确率必须与
   批量 teacher-forced 前向**逐位相同**（排除「增量解码坏了」这一解释）。
P2 **real-text by context**：真实文本前缀上，按上下文长度分块报
   `a_direct`（=P(argmax node2 == 模型自己的 AR 目标 T2*)）与 `a_chain`
   （=P(argmax chain2(cond=v1) == T2*)）。用来判定翻转是不是「上下文长度」造成的。
P3 **free-gen by context + top-k**：自生成前缀上同样分块，并报对 T2* 的 top-k 召回
   （回答 Q2「多候选/树能否翻转排序」）。

输出 `probes_output.json`（原始）+ `probes_output.txt`（人话）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "pathlm")):
    ROOT = os.path.dirname(ROOT)
SPEC = os.path.join(ROOT, "axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms")
for p in (ROOT, SPEC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import diag  # noqa: E402  (reuse load_ckpt / cluster_bootstrap)
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402
from pathlm.decode import Decoder, _slice_nodes  # noqa: E402

CTX_BLOCKS = [(1, 48), (48, 80), (80, 120), (120, 161)]
TOPKS = (1, 2, 4, 8)


@torch.no_grad()
def inc_vs_batched(m, val, offsets, L=64):
    """P1：逐 token 与批量前向的 next-token 一致率（两侧都算 node1）。"""
    inc_hits = bat_hits = tot = 0
    for off in offsets:
        seg = val[off:off + L + 1].astype(np.int64)
        x = m.embed(torch.tensor(seg[:-1])[None]) + m.pos_embed.weight[:L]
        from pathlm.model import cap_norm
        x = cap_norm(x, m.mcfg.norm_cap)
        for b in m.blocks:
            q, k, v = b.qkv(b.ln1(x)).chunk(3, dim=-1)
            B, T, d = x.shape
            nh, dh = b.n_heads, d // b.n_heads
            q = q.view(B, T, nh, dh).transpose(1, 2)
            k = k.view(B, T, nh, dh).transpose(1, 2)
            v = v.view(B, T, nh, dh).transpose(1, 2)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, d)
            x = x + b.proj(a)
            x = x + b.mlp(b.ln2(x))
        pred = m._mtp_nodes(x)[1]["logits"][0].argmax(-1)
        tgt = torch.tensor(seg[1:])
        bat_hits += int((pred == tgt).sum())
        dec = Decoder(m)
        dec.step(int(seg[0]))
        for i in range(1, L + 1):
            pred_i = int(dec.last_nodes[1]["logits"][0, 0].argmax())
            inc_hits += int(pred_i == int(seg[i]))
            if i < L:
                dec.step(int(seg[i]))
        tot += L
    return {"batched": bat_hits / tot, "incremental": inc_hits / tot, "n": tot}


@torch.no_grad()
def real_text(m, val, offsets, L=160):
    """P2：真实文本前缀上的 a_direct / a_chain / acc_next（逐 token step，mirror 生产）。"""
    recs = []
    for off in offsets:
        toks = val[off:off + L + 3].astype(np.int64)
        dec = Decoder(m)
        h, _ = dec.step(int(toks[0]))
        for j in range(L):
            nodes = dec.last_nodes
            v1 = int(nodes[1]["logits"][0, 0].argmax())
            dd = int(nodes[2]["logits"][0, 0].argmax())
            dc = int(nodes[1]["chain2"]["logits"][0, 0].float().argmax())
            _, fn = dec.step_multi([v1])
            T2 = int(fn[1]["logits"][0, 0].argmax())
            dec.truncate(j)
            recs.append({"ctx": j + 1, "dd": dd, "dc": dc, "T2": T2,
                         "v1": v1, "true_next": int(toks[j + 1])})
            h, _ = dec.step(int(toks[j + 1]))
    return recs


@torch.no_grad()
def free_gen(m, prompt, n_new):
    """P3：自生成前缀（生产 draft 规则：direct）。返回逐轮记录 + 生成文本。"""
    dec = Decoder(m)
    dec.step(int(prompt[0]))
    for t in prompt.tolist()[1:]:
        dec.step(int(t))
    gen, recs = [], []
    while len(gen) < n_new:
        nodes = dec.last_nodes
        v1 = int(nodes[1]["logits"][0, 0].argmax())
        dd = int(nodes[2]["logits"][0, 0].argmax())
        lc = nodes[1]["chain2"]["logits"][0, 0].float()
        dc = int(lc.argmax())
        d3 = int(nodes[3]["logits"][0, 0].argmax())
        td = nodes[2]["logits"][0, 0].float().topk(max(TOPKS)).indices.tolist()
        tc = lc.topk(max(TOPKS)).indices.tolist()
        ctx = dec.n
        _, fn = dec.step_multi([v1, dd, d3])
        T2 = int(fn[1]["logits"][0, 0].argmax())
        n_acc = 0
        drafts = [dd, d3]
        while n_acc < len(drafts) and int(fn[1]["logits"][0, n_acc].argmax()) == drafts[n_acc]:
            n_acc += 1
        recs.append({"ctx": ctx, "dd": dd, "dc": dc, "T2": T2, "td": td, "tc": tc})
        gen.append(v1)
        gen.extend(drafts[:n_acc])
        if n_acc < len(drafts):
            dec.truncate(dec.n - 3 + n_acc)
            dec.last_nodes = _slice_nodes(fn, n_acc)
        else:
            dec.last_nodes = _slice_nodes(fn, 2)
    return gen[:n_new], recs


def _rate(recs, key):
    if not recs:
        return None
    return round(sum(1 for r in recs if r[key] == r["T2"]) / len(recs), 4)


def _by_ctx(recs):
    out = {}
    for lo, hi in CTX_BLOCKS:
        sub = [r for r in recs if lo <= r["ctx"] < hi]
        out[f"[{lo},{hi})"] = {"n": len(sub), "a_direct": _rate(sub, "dd"),
                               "a_chain": _rate(sub, "dc")}
    return out


def main() -> int:
    val = np.load(os.path.join(ROOT, "data/enwik8_full.npz"))["val"]
    ckpts = sys.argv[1:] or [os.path.join(SPEC, "train", c) for c in ("13m_s0", "13m_s1", "53m_s0")]
    out = {}
    for ck in ckpts:
        name = os.path.basename(ck.rstrip("/"))
        m = diag.load_ckpt(ck)
        offsets = [i * 5000 for i in range(4)]
        p1 = inc_vs_batched(m, val, offsets)
        rt = real_text(m, val, offsets, L=160)
        fg_recs, fg_gen = [], []
        for pi in range(4):
            prompt = torch.tensor(val[pi * 5000:pi * 5000 + 32].astype(np.int64))
            gen, recs = free_gen(m, prompt, 200)
            fg_recs += recs
            fg_gen.append("".join(chr(t) if 32 <= t < 127 else "." for t in gen[:80]))
        out[name] = {
            "P1_inc_vs_batched_next_token_acc": p1,
            "P2_real_text_by_context": _by_ctx(rt),
            "P2_real_text_overall": {"n": len(rt), "a_direct": _rate(rt, "dd"),
                                     "a_chain": _rate(rt, "dc"),
                                     "acc_next": round(sum(1 for r in rt if r["v1"] == r["true_next"]) / len(rt), 4)},
            "P3_free_gen_by_context": _by_ctx(fg_recs),
            "P3_free_gen_overall": {"n": len(fg_recs), "a_direct": _rate(fg_recs, "dd"),
                                    "a_chain": _rate(fg_recs, "dc")},
            "P3_free_gen_topk_vs_T2": {
                k: {"direct": round(sum(1 for r in fg_recs if r["T2"] in r["td"][:k]) / len(fg_recs), 4),
                    "chain": round(sum(1 for r in fg_recs if r["T2"] in r["tc"][:k]) / len(fg_recs), 4)}
                for k in TOPKS},
            "P3_free_gen_sample_text": fg_gen,
        }
        print(f"[{name}] P1 inc={p1['incremental']} bat={p1['batched']}")
        print(f"[{name}] P2 real-text overall {out[name]['P2_real_text_overall']}")
        print(f"[{name}] P2 real-text by ctx {json.dumps(out[name]['P2_real_text_by_context'])}")
        print(f"[{name}] P3 free-gen overall {out[name]['P3_free_gen_overall']}")
        print(f"[{name}] P3 free-gen by ctx {json.dumps(out[name]['P3_free_gen_by_context'])}")
        print(f"[{name}] P3 topk {json.dumps(out[name]['P3_free_gen_topk_vs_T2'])}")
        del m
    with open(os.path.join(os.environ.get("PROBE_OUT", HERE), "probes_output.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.exit(main())
