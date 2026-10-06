"""2629 · 派发前的「能力面分析」（SCHEMA §22.8）——组合是否存在，不靠嘴说。

契约要求的组合 = `--dense --topology sequential --seq-len 2048
--train-window {0, 1024} --steps 6000 --candidate qk-normfrom512`。本脚本逐条核对
`pathlm/depth_ar.py` 的互斥校验，并**正向构造**两臂配置证明可达、**反向构造**被禁
组合证明校验真的在（不是死代码）。`--steps` 不在引擎校验面内（它是训练循环参数），
故只核对结构面。

用法：python3 capability_check.py     # 写 controls/capability_analysis.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel       # noqa: E402

BASE = dict(d_model=256, n_layers=8, n_heads=8, seq_len=2048, mlp_mult=4)


def try_build(**over):
    cfg = DepthARConfig(**{**BASE, **over})
    try:
        m = DepthARModel(cfg, 206)
        n = sum(p.numel() for p in m.parameters())
        return {"ok": True, "params": n}
    except ValueError as e:
        return {"ok": False, "error": str(e)[:200]}


def main():
    out = {
        "contract_combo": "--dense --topology sequential --seq-len 2048 "
                          "--train-window {0,1024} --steps 6000 "
                          "--candidate qk-normfrom512（两臂）",
        "forward": {
            "full-2048-clean": try_build(dense=True, topology="sequential",
                                         train_window=0),
            "win1024-2048-clean": try_build(dense=True, topology="sequential",
                                            train_window=1024),
        },
        "reverse_mutex_present": {
            # 说明：互斥校验在 DepthARModel.__init__（不是 DepthARConfig）
            "train_window>0 and not dense": try_build(dense=False,
                                                      train_window=1024),
            "train_window>0 and topology!=sequential": try_build(
                dense=True, topology="parallel", train_window=1024),
        },
        "qk_norm_note": ("QK-Norm 不是引擎 flag：它是 qkv 的 forward hook（min_t=512），"
                         "seq_len=2048 > 512 ⇒ 两臂各挂 8 hook、零新增参数，"
                         "不触碰 DepthARConfig 校验面"),
        "conclusion": None,
    }
    fwd_ok = all(v["ok"] for v in out["forward"].values())
    rev_ok = all(not v["ok"] for v in out["reverse_mutex_present"].values())
    out["conclusion"] = (
        "契约组合在引擎中**可达**（两臂正向构造成功）；互斥校验在 "
        "`DepthARModel.__init__` 中**确实生效**（反向构造被拒）。"
        if fwd_ok and rev_ok else
        f"能力面分析**不通过**：forward_ok={fwd_ok} reverse_ok={rev_ok}——停下请示")
    os.makedirs(os.path.join(HERE, "controls"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "controls", "capability_analysis.json"),
                        "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if not (fwd_ok and rev_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
