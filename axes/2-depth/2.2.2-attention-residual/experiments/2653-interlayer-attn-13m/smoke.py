"""2653 冒烟 + 等值门 / 活性门（CPU，不取 GPU 锁）。

* **A 等值门**（契约 §4-A）：`force_last=True`（层维权重强制 one-hot 到紧邻上一层）
  的本地模型前向与 stock 单残差（`topology="sequential"`）**逐位一致**；且 trunk 参数
  逐位相同。探针：`force_last=False` 时前向必须**不**逐位一致（否则门没有分辨力）。
* **B 活性门**（`SCHEMA.md` §10）：层维注意力权重非退化——逐层 `w_immediate_prev < 1`、
  熵 > 0，且相对「只取上一层」的偏离量落盘；探针：one-hot 输入下 `w_immediate_prev == 1`
  （证明该量确实能测到「退化成单残差」）。
* **C 参数账**：额外参数 = L（scalar）/ 3d²+L（qkv），与解析值逐位一致。

用法：`python3 smoke.py`（打印 JSON 摘要并在 `smoke_report.json` 落盘；任一探针没红 ⇒ 非零退出）。
"""

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import interlayer_attn as IA                                  # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel        # noqa: E402

BASE = dict(d_model=256, n_layers=8, n_heads=8, seq_len=256, mlp_mult=4,
            dense=True)
VOCAB = 206


def build_stock(seed=0):
    torch.manual_seed(seed)
    return DepthARModel(DepthARConfig(**BASE, topology="sequential"), VOCAB)


def build_attn(mode="scalar", force_last=False, seed=0):
    IA.MODE, IA.FORCE_LAST = mode, force_last
    torch.manual_seed(seed)
    return IA.InterLayerAttentionModel(
        DepthARConfig(**BASE, topology="dense"), VOCAB)


def forward(model, seed=7, batch=2, T=256, aux_out=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch, T), generator=g)
    model.eval()
    with torch.no_grad():
        loss, aux = model(x, exit_depths=None)
    return (loss, aux) if aux_out else aux["depth_logits"][-1]


def trunk_equal(a, b, extra_keys):
    sa, sb = a.state_dict(), b.state_dict()
    ka = {k for k in sa if not any(k.startswith(p) for p in extra_keys)}
    kb = {k for k in sb if not any(k.startswith(p) for p in extra_keys)}
    if ka != kb:
        return False
    return all(torch.equal(sa[k], sb[k]) for k in ka)


def gate_equivalence() -> dict:
    stock = build_stock()
    forced = build_attn("scalar", force_last=True)
    free = build_attn("scalar", force_last=False)
    sd_eq = trunk_equal(stock, forced, ("tau", "wq", "wk", "wv"))
    fw_eq = torch.equal(forward(stock), forward(forced))
    loss_eq = torch.equal(forward(stock, aux_out=True)[0],
                          forward(forced, aux_out=True)[0])
    probe_ne = not torch.equal(forward(stock), forward(free))
    return {"trunk_state_dict_bitwise_equal": bool(sd_eq),
            "forward_logits_bitwise_equal": bool(fw_eq),
            "loss_bitwise_equal": bool(loss_eq),
            "probe_free_weights_forward_differs": bool(probe_ne),
            "params_extra": {"forced": extra_params(forced),
                             "stock": sum(p.numel() for p in stock.parameters())},
            "pass": bool(sd_eq and fw_eq and loss_eq and probe_ne)}


def extra_params(model) -> dict:
    return {"numel": sum(p.numel() for p in model.parameters()),
            "tau": model.tau.numel(),
            "projections": (model.wq.weight.numel() + model.wk.weight.numel()
                            + model.wv.weight.numel()) if IA.MODE == "qkv" else 0}


def _stats(w: torch.Tensor, k: int) -> tuple[float, float]:
    """与 `interlayer_attn` 同一套公式：紧邻上一层权重 / 层维熵。"""
    prev = float(w[..., k - 1].mean())
    ent = float(-(w * w.clamp_min(1e-12).log()).sum(-1).mean())
    return prev, ent


def gate_liveness() -> dict:
    m = build_attn("scalar", force_last=False)
    forward(m)
    tele = m.attn_tele
    per_layer = [{"layer": t["layer"], "entropy": round(t["entropy"], 6),
                  "w_immediate_prev": round(t["w_immediate_prev"], 6),
                  "w_max": round(t["w_max"], 6)} for t in tele]
    nondeg = all(t["entropy"] > 0.0 and t["w_immediate_prev"] < 1.0
                 for t in tele if t["layer"] >= 2)
    # 探针（度量分辨力）：人工构造 one-hot / uniform 权重，本度量必须分别给出
    # 「退化到单残差」与「均摊」两个端点值；否则本量说明不了活性。
    k = 4
    w_one = torch.zeros(1, 1, k)
    w_one[..., k - 1] = 1.0
    w_uni = torch.full((1, 1, k), 1.0 / k)
    p_one, e_one = _stats(w_one, k)
    p_uni, e_uni = _stats(w_uni, k)
    probe_ok = (abs(p_one - 1.0) < 1e-6 and abs(e_one) < 1e-6
                and abs(p_uni - 1.0 / k) < 1e-6
                and abs(e_uni - math.log(k)) < 1e-6)
    return {"n_layers_observed": len(tele), "per_layer": per_layer,
            "non_degenerate": bool(nondeg),
            "non_degenerate_scope": ("层 1（k=1）只有唯一源（输入嵌入）⇒ 「只取紧邻"
                                   "上一层」平凡成立，不计入非退化要求；非退化门只看 k ≥ 2 的层"),
            "probe_metric_endpoints": {"one_hot_prev": p_one,
                                       "one_hot_entropy": e_one,
                                       "uniform_prev": p_uni,
                                       "uniform_entropy": e_uni},
            "pass": bool(nondeg and probe_ok and len(tele) == BASE["n_layers"])}


def gate_params() -> dict:
    stock = build_stock()
    n0 = sum(p.numel() for p in stock.parameters())
    out = {"stock_params": n0}
    ok = True
    for mode, expected_extra in (("scalar", 8), ("qkv", 8 + 3 * 256 * 256)):
        m = build_attn(mode, force_last=False)
        n = sum(p.numel() for p in m.parameters())
        got = n - n0
        out[mode] = {"params_total": n, "params_added": got,
                     "params_added_analytic": expected_extra}
        ok = ok and got == expected_extra
    # 探针：qkv 的额外参数必须**不**等于 scalar 的（否则这条账没分辨力）
    out["probe_qkv_differs_from_scalar"] = (out["qkv"]["params_added"]
                                            != out["scalar"]["params_added"])
    out["pass"] = bool(ok and out["probe_qkv_differs_from_scalar"])
    return out


def main() -> int:
    res = {"A_equivalence_gate": gate_equivalence(),
           "B_liveness_gate": gate_liveness(),
           "C_param_accounting": gate_params()}
    res["pass"] = all(v["pass"] for k, v in res.items() if k != "pass")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(os.path.join(HERE, "smoke_report.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
