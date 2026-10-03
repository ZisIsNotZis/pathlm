"""2625 · CPU 诊断探针 2：固定初始化下，梯度尺度/注意力尺度随 T 的变化。

假设 H-T2（待验）：全因果臂的梯度尺度随 T 增长（⇒ 同 LR 下长 T 更靠近不稳定边界）。
口径：**同一 seed（0）同一批数据**（enwik8 eval 流 seed=1234，bs8），只改 seq_len
（与训练口径一致：全因果 / 窗口 50%）。单步 fwd+bwd（无优化器步进）。
只描述，不判因果。
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel            # noqa: E402
from pathlm.depth_ar import window_anchor_keep_mask                # noqa: E402
from pathlm.data import load_enwik8_full, batch                    # noqa: E402

D, L, H, MLP, BS = 256, 8, 8, 4, 8
T_LIST = (256, 512, 1024)


def group_norms(model: DepthARModel, x: torch.Tensor, keep) -> dict:
    model.zero_grad(set_to_none=True)
    T = x.shape[1]
    mask = (None if keep is None else
            window_anchor_keep_mask(T, model.cfg.train_window, 0, x.device))
    am = (None if mask is None else
          mask.view(1, 1, T, T).expand(x.shape[0], H, T, T))
    depths = torch.full((x.shape[0], T), L, dtype=torch.long)      # noqa: F841
    loss, aux = model(x, exit_depths=None)
    loss.backward()
    g_total = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                             if p.grad is not None))
    groups = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.norm())
        if name in ("embed.weight", "pos_embed.weight"):
            groups[name] = round(g, 4)
        elif name.startswith("blocks.0."):
            groups.setdefault("blocks.0", 0.0)
        elif name.startswith("blocks.7."):
            groups.setdefault("blocks.7", 0.0)
    # 组范数
    gn = {}
    for tag, pred in (("embed", lambda n: n == "embed.weight"),
                      ("pos_embed", lambda n: n == "pos_embed.weight"),
                      ("blocks.0", lambda n: n.startswith("blocks.0.")),
                      ("blocks.7", lambda n: n.startswith("blocks.7."))):
        gn[tag] = round(float(torch.sqrt(sum(
            (p.grad ** 2).sum() for n, p in model.named_parameters()
            if p.grad is not None and pred(n)))), 4)
    return {"loss": round(float(loss), 4), "gnorm_total": round(float(g_total), 4),
            "gnorm_group": gn}


@torch.no_grad()
def weight_norms(model: DepthARModel) -> dict:
    out = {}
    for tag, pred in (("embed", lambda n: n == "embed.weight"),
                      ("pos_embed", lambda n: n == "pos_embed.weight"),
                      ("blocks.0.qkv", lambda n: n == "blocks.0.qkv.weight"),
                      ("blocks.0.proj", lambda n: n == "blocks.0.proj.weight"),
                      ("blocks.0.mlp", lambda n: n.startswith("blocks.0.mlp.")),
                      ("blocks.7", lambda n: n.startswith("blocks.7."))):
        out[tag] = round(float(torch.sqrt(sum(
            (p ** 2).sum() for n, p in model.named_parameters() if pred(n)))), 4)
    return out


def main():
    torch.set_num_threads(4)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    out = {}
    for w in (0, 512):
        for T in T_LIST:
            if w and w >= T // 2 and T == 256:
                pass
            win = (0 if w == 0 else T // 2)
            torch.manual_seed(0)
            cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=T,
                                mlp_mult=MLP, dense=True, train_window=win)
            m = DepthARModel(cfg, vocab)
            keep = (None if win == 0 else
                    window_anchor_keep_mask(T, win, 0, torch.device("cpu")))
            gen = torch.Generator().manual_seed(1234)
            x, _ = batch(eval_arr, BS, T, gen)
            row = group_norms(m, x, keep)
            row["weight_norms"] = weight_norms(m)
            row["seq_len"], row["window"] = T, win
            out[f"T{T}-w{win}"] = row
            print(f"[grad] T={T} w={win}: loss={row['loss']} "
                  f"gnorm={row['gnorm_total']} {row['gnorm_group']}", flush=True)
    with open(os.path.join(HERE, "smoke", "grad_scale_probe.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
