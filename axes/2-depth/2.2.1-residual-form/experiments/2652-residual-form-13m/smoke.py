"""2652 · 冒烟与等值门（`SCHEMA.md` §9 / `docs/methodology.md` §5）。

**等值门（必须能失败，抓红即停）**：

1. `gated` 在 init 处（`attn_gate = mlp_gate = 1.0`）与 `base` **逐位一致**：
   同 seed 构造的两个模型的**共用子模块参数逐位相同**，且同输入下
   `logits` 与 `loss` 逐位相同（`torch.equal`）。fp32（CPU）与 bf16 autocast
   （CUDA，若可用）两条路径都测——后者是训练真正走的路径。
2. 参数量审计：`gated` 相对 `base` 的增量 **恰好 = 2L = 16**（这是唯一被允许的
   额外参数；其余三臂为 0）。
3. **非无操作检查**：`postln` / `scaled` 在 init 处与 `base` 的前向**必须不同**
   （否则干预是死的，`SCHEMA.md` §10）。

任一条不成立 ⇒ 非零退出。

用法：`python3 smoke.py`（CPU）；`python3 smoke.py --cuda`（加测 bf16 autocast 路径）。
"""

from __future__ import annotations

import argparse
import sys

import torch

from common import DEFAULTS, make_cfg, param_count  # noqa: E402
from residual_block import build_model  # noqa: E402


class GateFailure(AssertionError):
    pass


def _params_equal(m_a, m_b) -> bool:
    """共用参数（键交集）逐位相同；再要求交集的键数 = 较少的那个的全集，
    即多出来的键只能是 `gated` 的门（由参数量审计另行钉住）。"""
    sa, sb = m_a.state_dict(), m_b.state_dict()
    common = set(sa) & set(sb)
    if len(common) != min(len(sa), len(sb)):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in common)


def init_equality_gate(device: str = "cpu", seed: int = 0) -> dict:
    """等值门 + 参数量审计 + 非无操作检查。设备上跑一次真实前向。"""
    vocab = DEFAULTS["vocab_size"]
    cfg = make_cfg(vocab)
    torch.manual_seed(seed)
    base = build_model(cfg, vocab, "base").to(device)
    torch.manual_seed(seed)
    gated = build_model(cfg, vocab, "gated").to(device)
    torch.manual_seed(seed)
    postln = build_model(cfg, vocab, "postln").to(device)
    torch.manual_seed(seed)
    scaled = build_model(cfg, vocab, "scaled").to(device)

    x = torch.randint(0, vocab, (2, DEFAULTS["seq_len"]),
                      generator=torch.Generator().manual_seed(7)).to(device)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.startswith("cuda") else torch.autocast("cpu", enabled=False))

    def fwd(m):
        with torch.no_grad(), autocast:
            loss, aux = m(x)
            return aux["depth_logits"][-1], loss

    b_logits, b_loss = fwd(base)
    g_logits, g_loss = fwd(gated)
    eq_gated = bool(torch.equal(b_logits, g_logits)
                    and torch.equal(b_loss, g_loss))
    diff_postln = bool(not torch.equal(b_logits, fwd(postln)[0]))
    diff_scaled = bool(not torch.equal(b_logits, fwd(scaled)[0]))

    n_base = param_count(base)
    n_gated = param_count(gated)
    n_postln = param_count(postln)
    n_scaled = param_count(scaled)
    shared_equal = _params_equal(base, gated)

    out = {
        "device": device,
        "autocast_dtype": ("bfloat16" if device.startswith("cuda") else "none"),
        "shared_params_bitwise_equal": shared_equal,
        "extra_param_keys": sorted(set(gated.state_dict())
                                    - set(base.state_dict())),
        "gated_forward_bitwise_equals_base": eq_gated,
        "params_total": {"base": n_base, "postln": n_postln, "scaled": n_scaled,
                         "gated": n_gated},
        "params_added": {"postln": n_postln - n_base,
                         "scaled": n_scaled - n_base,
                         "gated": n_gated - n_base},
        "params_added_expected": {"postln": 0, "scaled": 0,
                                  "gated": 2 * DEFAULTS["n_layers"]},
        "postln_forward_differs_from_base": diff_postln,
        "scaled_forward_differs_from_base": diff_scaled,
    }
    ok = (shared_equal and eq_gated and n_gated - n_base == 2 * DEFAULTS["n_layers"]
          and n_postln == n_base and n_scaled == n_base
          and diff_postln and diff_scaled)
    out["pass"] = ok
    if not ok:
        raise GateFailure(f"等值门/参数量审计失败：{out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true",
                    help="额外在 CUDA bf16 autocast 路径上跑一遍等值门")
    args = ap.parse_args()
    res = [init_equality_gate("cpu")]
    if args.cuda and torch.cuda.is_available():
        res.append(init_equality_gate("cuda"))
    for r in res:
        print(f"[smoke] device={r['device']} autocast={r['autocast_dtype']} "
              f"params_added={r['params_added']} pass={r['pass']}", flush=True)
        print(f"        gated@init == base (bitwise): "
              f"{r['gated_forward_bitwise_equals_base']}", flush=True)
    print(f"[smoke] PASS ({len(res)} 条路径)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
