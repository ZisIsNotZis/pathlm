"""2625 · CPU 诊断探针：跨 T 量测「注意力 logits 尺度 / 熵 / 残差流范数」。

假设 H-T2：全因果臂的爆炸与 T 相关的注意力 logits 尺度增长有关。
本探针只读**已落盘的 checkpoint**（2622 的 T=256/512 四点、2623/2624 的 T=1024 四点），
在 CPU 上逐层重算 q·kᵀ / softmax，量测：

- `resid_norm[l]`：每层输出残差流的逐 token L2 范数均值；
- `logit_rms[l]`：因果（或窗口）允许对的注意力 logit（q·kᵀ/√d_h）RMS；
- `logit_max[l]`：每行允许键上的最大 logit 的均值；
- `entropy[l]`：每行注意力分布的香农熵（nats）均值；
- `entropy_frac[l]`：熵 / log(可见键数)（1 = 均匀，0 = 一点集中）；
- `top1[l]`：每行最大注意力概率的均值。

口径：单批（bs8，enwik8 eval 流，seed=1234）；只描述，不判因果。
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel          # noqa: E402
from pathlm.depth_ar import window_anchor_keep_mask              # noqa: E402
from pathlm.data import load_enwik8_full, batch                  # noqa: E402

D, L, H, MLP, BS = 256, 8, 8, 4, 8

#: (label, experiment_dir, arm_dir, seq_len, window)
CKPTS = [
    ("full-256",   "2622-longT-512",          "full-256",              256,  0),
    ("win128-256", "2622-longT-512",          "win128-256",            256,  128),
    ("full-512",   "2622-longT-512",          "full-512",              512,  0),
    ("win256-512", "2622-longT-512",          "win256-512",            512,  256),
    ("full-1024",  "2623-longT-1024",         "full-1024",             1024, 0),
    ("win512-1024", "2623-longT-1024",        "win512-1024",           1024, 512),
    ("full-1024-s1", "2624-longT-1024-seed1", "full-1024-seed1",       1024, 0),
    ("win512-1024-s1", "2624-longT-1024-seed1", "win512-1024-seed1",   1024, 512),
]
EXP_ROOT = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window",
                        "experiments")


@torch.no_grad()
def layer_stats(model: DepthARModel, x: torch.Tensor,
                keep: torch.Tensor | None) -> list[dict]:
    B, T = x.shape
    dh = model.cfg.d_model // model.cfg.n_heads
    h = model._input_states(x)
    init_norm = float(h.norm(dim=-1).mean())
    rows = []
    for li, blk in enumerate(model.blocks):
        h1 = blk.ln1(h)
        q, k, v = blk.qkv(h1).chunk(3, dim=-1)
        shp = lambda t: t.view(B, T, model.cfg.n_heads, dh).transpose(1, 2)
        qs, ks, vs = shp(q), shp(k), shp(v)
        scores = (qs @ ks.transpose(-2, -1)) / (dh ** 0.5)   # [B,H,T,T]
        i = torch.arange(T).unsqueeze(1)
        j = torch.arange(T).unsqueeze(0)
        causal = (j <= i)                                   # [T,T]
        mask = causal if keep is None else (causal & keep)
        mrow = mask.view(1, 1, T, T)
        neg = torch.finfo(scores.dtype).min
        sm = scores.masked_fill(~mrow, neg).softmax(-1)     # [B,H,T,T]
        ent = -(sm.clamp_min(1e-12).log() * sm).sum(-1)     # [B,H,T]
        vis = mrow.sum(-1).clamp_min(1).float()             # [B,H,T]
        logit_max = scores.masked_fill(~mrow, neg).amax(-1)  # [B,H,T]
        logit_rms = torch.sqrt((scores ** 2 * mrow).sum(-1)
                               / vis).clamp_min(0)
        # 真正的前向（SDPA 路径），继续堆残差
        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None,
                                           is_causal=keep is None)
        a = a.transpose(1, 2).reshape(B, T, model.cfg.d_model)
        h = h + blk.proj(a)
        h = h + blk.mlp(blk.ln2(h))
        rows.append({
            "layer": li,
            "resid_norm": round(float(h.norm(dim=-1).mean()), 4),
            "logit_rms": round(float(logit_rms.mean()), 4),
            "logit_max": round(float(logit_max.mean()), 4),
            "entropy": round(float(ent.mean()), 4),
            "entropy_frac": round(float((ent / vis.log()).mean()), 4),
            "top1": round(float(sm.amax(-1).mean()), 4),
        })
    return rows, init_norm


def main():
    torch.set_num_threads(4)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    out = {}
    for label, exp, arm, T, w in CKPTS:
        torch.manual_seed(0)
        cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=T,
                            mlp_mult=MLP, dense=True, train_window=w)
        m = DepthARModel(cfg, vocab)
        path = os.path.join(EXP_ROOT, exp, "runs", arm, "model.pt")
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        gen = torch.Generator().manual_seed(1234)
        x, _ = batch(eval_arr, BS, T, gen)
        keep = (None if w == 0 else
                window_anchor_keep_mask(T, w, 0, torch.device("cpu")))
        rows, init_norm = layer_stats(m, x, keep)
        out[label] = {"seq_len": T, "window": w, "init_resid_norm": round(init_norm, 4),
                      "layers": rows}
        print(f"[probe] {label}: init_norm={init_norm:.3f} "
              f"resid_final={rows[-1]['resid_norm']:.3f} "
              f"logit_rms_final={rows[-1]['logit_rms']:.3f} "
              f"top1_final={rows[-1]['top1']:.3f}", flush=True)
    with open(os.path.join(HERE, "smoke", "attention_scale_probe.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
