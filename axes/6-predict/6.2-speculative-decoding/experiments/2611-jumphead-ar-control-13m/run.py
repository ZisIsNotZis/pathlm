"""契约 29-A（ticket 29）：跳跃头对照臂修正 + 草稿接受率（纯推理）。

契约：axes/6-predict/6.2-speculative-decoding/experiments/2611-jumphead-ar-control-13m/README.md

复用 2610 的冻结主干 trunk.pt 与两个跳跃头 head_s{0,1}.pt（先核对 MODEL.md 哈希），
只做**推理**，不训练。产出的量：

- ar1_deploy_t2_hit  诚实 1-step AR（只替换紧邻位置 i+1）argmax == 真值 t+2（主对手）
- jump_t2_hit        一次前向 + 跳跃头 argmax == 真值 t+2（主角，质量口径）
- accept_jump_t2     草稿接受率 = P(跳跃头 argmax == 主干自身 1-step AR argmax)（无作弊）
- oracle_t2_hit      喂真 t+1 的第二次前向（仅上界参考，**不作判据**）
- equiv_speedup      产 1 token 的等效成本比（跳跃头 1 前向 vs AR 2 前向）

用法：
  python run.py --config <exp>/config.json --out-dir <exp> [--smoke]
"""

import argparse, hashlib, json, os, sys, time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    d = _HERE
    while d != "/" and not os.path.isdir(os.path.join(d, "pathlm")):
        d = os.path.dirname(d)
    if not os.path.isdir(os.path.join(d, "pathlm")):
        raise RuntimeError("repo root (dir containing pathlm/) not found")
    return d


ROOT = _repo_root()
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                   # 相对权重路径 & 数据缓存都按 ROOT 解析

from pathlm.data import batch, load_enwik8_full                 # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel         # noqa: E402
from train_jumphead import (HeadBank, assert_frozen,            # noqa: E402
                            equiv_speedup_from_dims, evaluate_ar1,
                            freeze_trunk)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_activity(bank: HeadBank) -> dict:
    """干预活性量（SCHEMA §10）：跳跃头参数规模、Fro / absmax / 非零占比。"""
    p = torch.cat([q.detach().float().flatten() for q in bank.t2.parameters()])
    return {
        "params": int(p.numel()),
        "w_fro": round(float(p.norm()), 4),
        "w_absmax": round(float(p.abs().max()), 6),
        "nonzero_frac": round(float((p != 0).float().mean()), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="小评测冒烟（2 批 × bs4），打印步时与显存")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["eval"]["n_batches"] = 2
        cfg["eval"]["batch_size"] = 4
    os.makedirs(args.out_dir, exist_ok=True)

    reuse = cfg["reuse"]

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(args.config)), p))

    trunk_path = _resolve(reuse["trunk_ckpt"])
    head_paths = {s: _resolve(p) for s, p in reuse["head_ckpts"].items()}

    # --- 复用权重哈希核对（MODEL.md 口径）---
    trunk_sha = sha256_file(trunk_path)
    assert trunk_sha == reuse["trunk_sha256"], \
        f"trunk sha mismatch: {trunk_sha} != {reuse['trunk_sha256']}"
    for s, p in head_paths.items():
        got = sha256_file(p)
        assert got == reuse["head_sha256"][s], f"head seed {s} sha mismatch: {got}"

    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")

    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg["eval"]["seed"])
    trunk = DepthARModel(DepthARConfig(**cfg["trunk"]), vocab).cuda()
    trunk.load_state_dict(torch.load(trunk_path, map_location="cuda"))
    freeze_trunk(trunk)
    ok, info = assert_frozen(trunk)
    print(f"trunk loaded: {sum(p.numel() for p in trunk.parameters())/1e6:.3f}M params, "
          f"frozen_assertion_ok={ok} {info}", flush=True)

    t0 = time.time()
    speed = equiv_speedup_from_dims(trunk, HeadBank(trunk.cfg.d_model).t2)
    per_seed = {}
    for s in sorted(head_paths):
        bank = HeadBank(trunk.cfg.d_model).cuda()
        bank.load_state_dict(torch.load(head_paths[s], map_location="cuda"))
        ev = evaluate_ar1(trunk, bank, eval_arr, cfg["eval"])
        ev["head_activity"] = head_activity(bank)
        per_seed[s] = ev
        print(f"seed {s}: jump_t2={ev['jump_t2_hit']:.4f} "
              f"ar1_deploy_t2={ev['ar1_deploy_t2_hit']:.4f} "
              f"accept={ev['accept_jump_t2']:.4f} "
              f"ratio={ev['ratio_jump_over_ar1']:.4f}", flush=True)

    wall = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20

    def mean(key):
        return round(sum(per_seed[s][key] for s in per_seed) / len(per_seed), 4)

    m_jump, m_ar1 = mean("jump_t2_hit"), mean("ar1_deploy_t2_hit")
    m_accept = mean("accept_jump_t2")
    ratio = round(m_jump / m_ar1, 4) if m_ar1 else None
    recon = 0.4822                                   # 0.6934 × 0.6953（SCHEMA §12 算术下界）
    main_ok = ratio is not None and ratio >= 0.90
    accept_ok = m_accept >= 0.90
    if ratio is None:
        verdict = "无判据"
    elif ratio >= 0.90:
        verdict = "成功：半算力打到诚实 1-step AR 的 ≥90%"
    elif ratio >= 0.80:
        verdict = "中性：比值 ∈ [0.80, 0.90)"
    else:
        verdict = "失败：跳跃头不如自回归两次"

    results = {
        "contract": ("axes/6-predict/6.2-speculative-decoding/experiments/"
                     "2611-jumphead-ar-control-13m/README.md"),
        "config": cfg,
        "reused_weights": {
            "trunk": {"path": reuse["trunk_ckpt"], "sha256": trunk_sha,
                      "size_bytes": os.path.getsize(trunk_path)},
            "heads": {s: {"path": reuse["head_ckpts"][s], "sha256": reuse["head_sha256"][s],
                          "size_bytes": os.path.getsize(head_paths[s])}
                      for s in head_paths},
        },
        "trunk": {"params": sum(p.numel() for p in trunk.parameters()),
                  "frozen_assertion_ok": bool(ok),
                  "frozen_offenders": info},
        "intervention": {
            "arm": "ar1_deploy (1-step AR: 只替换紧邻位置 i+1)",
            "replacement_frac": round(1 - mean("trunk_t1_hit"), 4),
            "replacement_note": "被替换 token ≠ 真值的比例 = 1 − trunk_t1_hit；>0 ⇒ 反馈非平凡",
            "jump_head": {s: per_seed[s]["head_activity"] for s in per_seed},
            "alive": bool(0 < (1 - mean("trunk_t1_hit")) < 1
                          and all(per_seed[s]["head_activity"]["nonzero_frac"] > 0.9
                                  for s in per_seed)),
        },
        "arms": per_seed,
        "arms_mean": {k: mean(k) for k in
                      ("trunk_t1_hit", "jump_t2_hit", "ar1_deploy_t2_hit",
                       "accept_jump_t2", "oracle_t2_hit", "accept_jump_t2_vs_oracle",
                       "ratio_jump_over_ar1", "equiv_speedup")},
        "equiv_speedup_detail": speed,
        "criteria": {
            "reconciliation_lower_bound": recon,
            "reconciliation_ar1_deploy_t2_hit": m_ar1,
            "reconciliation_ok": bool(m_ar1 >= recon),
            "numerator_jump_t2_hit": m_jump,
            "denominator_ar1_deploy_t2_hit": m_ar1,
            "ratio": ratio,
            "main_threshold_0.90x": round(0.90 * m_ar1, 4),
            "main_criterion_ok": bool(main_ok),
            "neutral_band": [0.80, 0.90],
            "parallel_accept_threshold": 0.90,
            "parallel_accept_criterion_ok": bool(accept_ok),
            "oracle_is_upper_bound_only": True,
            "verdict": verdict,
        },
        "cost": {"wall_minutes": round(wall / 60, 2), "peak_mem_mib": round(peak_mb, 1)},
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "arms")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
