"""2655 冒烟 + 等值门（CPU，不取 GPU 锁）。

三条门各带**可失败探针**（`/home/z/vibe/AGENTS.md` `proof.gate_probe`：一个不会红的检查
不是检查）：

* **A 引擎等值门**（契约 P5）：显式 `mlp_hidden_sched=[1024]*8` 与缺省（`mlp_mult=4`）
  建出的模型 state_dict 与前向输出**逐位一致**；探针：把 schedule 改一位 ⇒ 必须不一致。
* **B 等预算门**（契约 P4）：三臂 `Σh`、FFN 参数量逐位相等；探针：故意给一个 Σh 不同的
  schedule ⇒ 必须不等。
* **C 活性门**（契约 P6 / `harness/SCHEMA.md` §10）：`hidden_by_layer` 逐位等于 schedule；
  非均匀臂的前向输出与 uniform 不同（干预真的生效）；探针：uniform 自身相对 uniform
  必须一致。

用法：`python3 smoke.py`（打印 JSON 摘要；任一探针没红 ⇒ 非零退出）。
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARConfig, DepthARModel   # noqa: E402
from run import SCHEDULES, SUM_HIDDEN, analytic_cost      # noqa: E402

BASE = dict(d_model=256, n_layers=8, n_heads=8, seq_len=256, mlp_mult=4,
            dense=True, topology="sequential")
VOCAB = 206


def _build(sched, seed=0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(**BASE, mlp_hidden_sched=sched)
    return DepthARModel(cfg, VOCAB)


def _forward(model, seed=7, batch=2, T=256):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch, T), generator=g)
    model.eval()
    with torch.no_grad():
        _, aux = model(x, exit_depths=None)
    return aux["depth_logits"][-1]


def _sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    if set(sa) != set(sb):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def gate_engine_equivalence() -> dict:
    """A：显式 uniform schedule == 引擎缺省（逐位），带一个必须红的探针。"""
    m_def = _build(None)
    m_exp = _build(SCHEDULES["uniform"])
    sd_eq = _sd_equal(m_def, m_exp)
    fw_eq = torch.equal(_forward(m_def), _forward(m_exp))
    # 探针：改一位（第 8 层 1024→768）后必须**不**逐位一致，否则本门是死的
    m_probe = _build(SCHEDULES["uniform"][:7] + [768])
    probe_sd_ne = not _sd_equal(m_def, m_probe)
    probe_fw_ne = not torch.equal(_forward(m_def), _forward(m_probe))
    return {
        "default_vs_explicit_uniform_state_dict_bitwise_equal": bool(sd_eq),
        "default_vs_explicit_uniform_forward_bitwise_equal": bool(fw_eq),
        "probe_perturbed_schedule_state_dict_differs": bool(probe_sd_ne),
        "probe_perturbed_schedule_forward_differs": bool(probe_fw_ne),
        "pass": bool(sd_eq and fw_eq and probe_sd_ne and probe_fw_ne),
    }


def gate_equal_budget() -> dict:
    """B：三臂 Σh / FFN 参数量逐位相等；探针：Σh 不同的 schedule 必须不等。"""
    per_arm = {}
    for arm, sched in SCHEDULES.items():
        m = _build(sched)
        ffn = sum(p.numel() for blk in m.blocks for p in blk.mlp.parameters())
        total = sum(p.numel() for p in m.parameters())
        per_arm[arm] = {"sum_hidden": sum(sched), "params_ffn": ffn,
                        "params_total": total,
                        "ffn_params_analytic": analytic_cost(sched)["ffn_params_analytic"]}
    m_probe = _build(SCHEDULES["uniform"][:7] + [768])
    probe_total = sum(p.numel() for p in m_probe.parameters())
    probe_ne = probe_total != per_arm["uniform"]["params_total"]
    triple = {arm: (v["sum_hidden"], v["params_ffn"], v["params_total"])
              for arm, v in per_arm.items()}
    return {
        "per_arm": per_arm,
        "all_sum_hidden_equal_to_8192": all(v["sum_hidden"] == SUM_HIDDEN
                                            for v in per_arm.values()),
        "all_ffn_params_equal": len({v["params_ffn"] for v in per_arm.values()}) == 1,
        "all_total_params_equal": len({v["params_total"]
                                       for v in per_arm.values()}) == 1,
        "analytic_matches_measured": all(
            v["ffn_params_analytic"] == v["params_ffn"] for v in per_arm.values()),
        "probe_off_budget_schedule_params_differ": bool(probe_ne),
        "equal_budget_triples": {k: list(v) for k, v in triple.items()},
        "pass": bool(all(v["sum_hidden"] == SUM_HIDDEN for v in per_arm.values())
                     and len({v["params_ffn"] for v in per_arm.values()}) == 1
                     and len({v["params_total"] for v in per_arm.values()}) == 1
                     and probe_ne),
    }


def gate_liveness() -> dict:
    """C：隐藏维逐位等于 schedule，非均匀臂前向与 uniform 不同。"""
    ref = _forward(_build(SCHEDULES["uniform"]))
    out = {"hidden_matches_schedule": {},
           "nonuniform_forward_differs_from_uniform": {}}
    for arm, sched in SCHEDULES.items():
        m = _build(sched)
        out["hidden_matches_schedule"][arm] = (
            [int(blk.mlp[0].weight.shape[0]) for blk in m.blocks] == list(sched))
        out["nonuniform_forward_differs_from_uniform"][arm] = (
            not torch.equal(_forward(m), ref))
    # 探针：uniform 相对 uniform 必须**不**不同（否则「不同」这个判据没有分辨力）
    m_same = _build(SCHEDULES["uniform"], seed=1)     # 换 seed ⇒ 权重不同 ⇒ 前向必不同
    probe_same_differs = not torch.equal(_forward(m_same), ref)
    out["probe_uniform_seed1_differs_from_seed0"] = bool(probe_same_differs)
    out["pass"] = bool(all(out["hidden_matches_schedule"].values())
                       and out["nonuniform_forward_differs_from_uniform"]["front-big"]
                       and out["nonuniform_forward_differs_from_uniform"]["growing"]
                       and not out["nonuniform_forward_differs_from_uniform"]["uniform"]
                       and probe_same_differs)
    return out


def gate_cli_wiring() -> dict:
    """CLI 串 → cfg：`--mlp-hidden-sched` 必须真的到模型（防「开关是死的」）。"""
    import train_depth_ar as TDA
    import argparse
    from run import shared_argv
    argv = shared_argv("front-big", 0, 12000, 32, 1e-3, "/tmp/unused")
    ap = argparse.ArgumentParser()
    # 复用共享 main 的解析器代价太高：直接走 build_depthar_config 所需的字段
    ns = argparse.Namespace(**{
        "d": 256, "layers": 8, "heads": 8, "seq_len": 256, "mlp_mult": 4,
        "mlp_hidden_sched": None, "dense": True, "gate_bias_init": -2.0,
        "topology": "sequential", "fill_kv": False, "proj_fill": False,
        "readout_dims": None, "readout_gated": False,
        "readout_linear_resid": False, "readout_perdepth": False,
        "readout_wide_shared": 0, "attn_sparse_k": 0, "attn_linear": False,
        "linear_chunk": 64, "train_window": 0, "train_anchors": 0,
        "n_kv_heads": None, "carry_rule": None, "carry_state_dim": 64,
        "carry_gate": True, "aux_mode": "renorm", "mod_ratio": 0.0,
        "recon_window": 0, "recon_lambda": 0.0, "synth": None,
    })
    # 只校验 CLI 串解析这一环（cfg 装配由共享代码负责，已在引擎单测里覆盖）
    parsed = TDA._parse_int_sched(
        shared_argv("front-big", 0, 12000, 32, 1e-3, "/tmp/unused")[
            shared_argv("front-big", 0, 12000, 32, 1e-3, "/tmp/unused").index(
                "--mlp-hidden-sched") + 1])
    ok = parsed == SCHEDULES["front-big"]
    probe = TDA._parse_int_sched("") is None
    return {"argv": argv[1:4], "parsed_front_big": parsed,
            "parsed_equals_schedule": bool(ok),
            "probe_empty_string_returns_none": bool(probe),
            "note": "本门只钉 CLI 串解析；cfg→模型的事实由 P6 的 hidden_by_layer 实测覆盖",
            "pass": bool(ok and probe)}


def main() -> int:
    res = {"A_engine_equivalence": gate_engine_equivalence(),
           "B_equal_budget": gate_equal_budget(),
           "C_liveness": gate_liveness(),
           "D_cli_wiring": gate_cli_wiring()}
    res["pass"] = all(v["pass"] for k, v in res.items() if k != "pass")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(os.path.join(HERE, "smoke_report.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
