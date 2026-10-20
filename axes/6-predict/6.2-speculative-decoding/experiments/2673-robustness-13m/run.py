#!/usr/bin/env python3
"""WP 6.2.4 · 2673-robustness-13m 驱动：温度>0 的无损投机采样 + 53.9M 对抗臂。

契约：`axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m/README.md`（预登记）。

跑法（仓库根，detached）：

    nohup python3 run.py --out <本目录> --jobs p1_13m_s0 p1_13m_s1 p2_53m_s0 \
        > /tmp/wp624.nohup 2>&1 &

- `--jobs`：要跑的后台任务名。每个 job **取一次 GPU 锁 → 跑完 → 释放 → 落盘**；
  job 之间按 `harness/SCHEMA.md` §20.1 留 **≥60 秒让出窗口**（`run_log.txt` 记 `[lock] §20.1 让出窗口`）。
- `--resume`：读回已落盘的 `results.json`，跳过已完成 job。
- 每 job 结束就 `flush()`（`results.json` / `RESUME.md` 增量落盘，容忍中途被杀）。
- **禁墙钟**：`results.json` 无墙钟字段（`has_wall_clock` 自检）；墙钟只进 `run_log.txt`。

job：
- `p1_13m_s0` / `p1_13m_s1`：主臂 `spec_T1`（T=1 无损投机采样）+ 参照 `target_sample` +
  **L0 gate probe**（`accept_always` / `resample_from_draft` 两个故意改错的规则）。
- `p2_53m_s0`：53.9M 冻结主干上的对抗臂 `adv_random_k4` / `adv_bottomk4`（候选构造 / K=4 /
  验证路径与 2650 **逐字一致**，只换主干）。
"""

from __future__ import annotations

import argparse
import glob
import importlib.util as _ilu
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DIR2640 = os.path.abspath(os.path.join(HERE, "..", "2640-tree-conf-13m"))
DIR2650 = os.path.abspath(os.path.join(HERE, "..", "2650-verify-gate-13m"))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for _p in (HERE, DIR2640, DIR2650, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_lossless as S  # noqa: E402
import spec_tree as T  # noqa: E402
from gpu_lock import GPUUnavailable, gpu_lock  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

# 2640 的只读评测驱动（prompt / bootstrap / load_ckpt / summarize_arm）。按**显式路径**加载为
# 独立模块名 `run2640`（本目录也有 `run.py`，裸 `import run` 会撞名）。
_spec = _ilu.spec_from_file_location("run2640", os.path.join(DIR2640, "run.py"))
H = _ilu.module_from_spec(_spec)
sys.modules["run2640"] = H
_spec.loader.exec_module(H)

# 2650 的常规 K 行验证主循环 + 对抗候选构造（只读复用；本实验只换主干到 53.9M）
import verify_gate as V  # noqa: E402

HERE_CTEK = os.path.join(DIR2640, "..", "2631-spec-three-arms", "train")
TRAIN_DIR = os.path.abspath(HERE_CTEK)

#: 采样评测规模（温度>0 的分布检验需要每 prompt 足够 token）
N_PROMPTS = 12
M_SEEDS = 8
N_NEW_SAMPLE = 128
#: L0 gate probe 规模（只要够判红/绿，省预算）
N_PROMPTS_GATE = 4
M_SEEDS_GATE = 4
N_NEW_GATE = 96
#: P2（对抗臂，与 2650 同口径）
N_PROMPTS_ADV = 12
N_NEW_ADV = 200
BOOT = 2000
BOOT_SEED = 2673

VOCAB = 206
#: 禁墙钟自检（沿用 2660/2650 口径）
WALL_SUBSTR = ("wall", "elapsed", "clock", "duration", "seconds", "milliseconds",
               "latency", "throughput")
WALL_EXACT = {"time", "ms", "sec", "took", "runtime", "speed", "tps", "timing"}


def has_wall_clock(obj, path="") -> list:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(w in kl for w in WALL_SUBSTR) or kl in WALL_EXACT:
                bad.append(f"{path}.{k}")
            bad += has_wall_clock(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += has_wall_clock(v, f"{path}[{i}]")
    return bad


# ------------------------------------------------------------------ 统计（按 prompt 聚类）

def boot_ci(vals: list, stat=None, B: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """按 prompt 聚类的 percentile bootstrap 95% CI（值本身即 per-prompt 量）。"""
    stat = stat or statistics.fmean
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return {"point": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    n = len(v)
    bs = [stat([v[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return {"point": round(float(stat(v)), 6),
            "ci": [round(float(np.percentile(bs, 2.5)), 6),
                   round(float(np.percentile(bs, 97.5)), 6)],
            "n": n, "bootstrap_B": B, "seed": seed, "cluster": "prompt"}


def _nll(hist_logp_sum: float, n_tok: int) -> float:
    return -hist_logp_sum / max(n_tok, 1)


# ------------------------------------------------------------------ P1：无损采样

def _sample_arm(model, prompts, pcap, arm, n_new, seeds, broken=None):
    """一个采样臂在全部 prompt 上跑完；返回 per-prompt 池化的直方图 / logp / 代价。

    `seed_hists` = 每个 seed 一份 hist（供噪声地板「同池按 seed 奇偶拆分」，不必重算）。
    """
    pools = []
    for pi, prompt in enumerate(prompts):
        hist = [0] * VOCAB
        seed_hists = []
        logp_sum = 0.0
        n_tok = 0
        n_forward = 0
        accepts: list[int] = []
        tails = []
        for s in seeds:
            if arm == "spec_T1":
                seq, st = S.spec_sample_T1(model, prompt, n_new, pcap, int(s), **(broken or {}))
            else:
                seq, st = S.target_sample_T1(model, prompt, n_new, pcap, int(s))
            sh = [0] * VOCAB
            for t in seq:
                hist[int(t)] += 1
                sh[int(t)] += 1
            seed_hists.append(sh)
            logp_sum += sum(st["logp"])
            n_tok += len(seq)
            n_forward += st["n_forward"]
            accepts.extend(st["accepts"])
            if pi == 0 and s == seeds[0]:
                tails = seq[:8]
        pools.append({"prompt": pi, "hist": hist, "seed_hists": seed_hists,
                      "logp_sum": logp_sum, "n_tok": n_tok, "n_forward": n_forward,
                      "accepts": accepts, "gen_head": tails, "nll": _nll(logp_sum, n_tok)})
    return pools


def _losslessness(spec_pools, target_pools, vocab):
    """per-prompt TV / 目标 NLL 差 + 聚类 bootstrap CI + 噪声地板（target 半分样本）。"""
    tv, dll = [], []
    for sp, tp in zip(spec_pools, target_pools):
        tv.append(S.tv_distance(sp["hist"], tp["hist"], vocab))
        dll.append(sp["nll"] - tp["nll"])
    # 噪声地板：target 池化按 seed 奇偶拆两半，重算 TV（同规模独立子样本）
    return {"tv": boot_ci(tv), "nll_diff": boot_ci(dll)}


def run_p1(name, ckpt, val, log, smoke=False, device="cuda"):
    n_prompts = 2 if smoke else N_PROMPTS
    n_gate = 2 if smoke else N_PROMPTS_GATE
    m_seeds = M_SEEDS if not smoke else 3
    n_seeds_gate = M_SEEDS_GATE if not smoke else 2
    n_new = 48 if smoke else N_NEW_SAMPLE
    n_new_gate = 32 if smoke else N_NEW_GATE

    model = load_ckpt(ckpt, device)
    pcap = model.pcap
    prompts = H.make_prompts(val, n_prompts)
    log(f"[{name}] spec_T1 采样中 … prompts={n_prompts} seeds={m_seeds} n_new={n_new}")
    spec = _sample_arm(model, prompts, pcap, "spec_T1", n_new, list(range(m_seeds)))
    log(f"[{name}] target_sample 采样中 …")
    tgt = _sample_arm(model, prompts, pcap, "target_sample", n_new, list(range(m_seeds)))
    ll = _losslessness(spec, tgt, VOCAB)

    # 噪声地板：把**同一个** target 池按 seed 奇偶拆两半（同规模的独立子样本），不必重算
    even = [sum((p["seed_hists"][i] for i in range(0, len(p["seed_hists"]), 2)
                if i < len(p["seed_hists"])), [0] * VOCAB) for p in tgt]
    odd = [sum((p["seed_hists"][i] for i in range(1, len(p["seed_hists"]), 2)
               if i < len(p["seed_hists"])), [0] * VOCAB) for p in tgt]
    null_vals = [S.tv_distance(e, o, VOCAB) for e, o in zip(even, odd)]
    null = boot_ci(null_vals)

    n_forward = sum(p["n_forward"] for p in spec)
    n_tok = sum(p["n_tok"] for p in spec)
    accepts = [a for p in spec for a in p["accepts"]]
    n_rounds = len(accepts)
    a1 = sum(1 for a in accepts if a >= 1) / max(n_rounds, 1)
    a2 = (sum(1 for a in accepts if a >= 2) / max(sum(1 for a in accepts if a >= 1), 1))
    tpf = n_tok / max(n_forward, 1)
    cost = {
        "n_forward": n_forward, "n_emitted": n_tok, "n_rounds": n_rounds,
        "tokens_per_forward": round(tpf, 6), "tau": round(tpf - 1.0, 6),
        "alpha1": round(a1, 6), "alpha2_given_alpha1": round(a2, 6),
        "positions_per_forward": 1 + T.GAMMA,
        "R_equivalent_flops": round((1 + T.GAMMA) / tpf, 6),
        "R_note": "R=(1+γ)/(1+τ)（契约 §5 等效理论量）；target_sample R≡1",
    }

    log(f"[{name}] L0 gate probe（故意改错）…")
    gate = {}
    for tag, broken in (("accept_always", {"accept_always": True}),
                        ("resample_from_draft", {"resample_from_draft": True})):
        gp = H.make_prompts(val, n_gate)
        gspec = _sample_arm(model, gp, pcap, "spec_T1", n_new_gate,
                            list(range(n_seeds_gate)), broken=broken)
        gtgt = _sample_arm(model, gp, pcap, "target_sample", n_new_gate,
                           list(range(n_seeds_gate)))
        gate[tag] = _losslessness(gspec, gtgt, VOCAB)
        gate[tag]["tv_vs_correct_null"] = {
            "broken_tv_point": gate[tag]["tv"]["point"],
            "correct_tv_point": ll["tv"]["point"],
            "null_tv_point": null["point"]}

    # 正确规则下的同规模 gate 对照（同 n_gate / n_seeds_gate 规模，公平比较红/绿）
    gp = H.make_prompts(val, n_gate)
    cspec = _sample_arm(model, gp, pcap, "spec_T1", n_new_gate, list(range(n_seeds_gate)))
    ctgt = _sample_arm(model, gp, pcap, "target_sample", n_new_gate, list(range(n_seeds_gate)))
    gate_correct = _losslessness(cspec, ctgt, VOCAB)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "path": os.path.relpath(ckpt, ROOT), "params": _params(ckpt),
        "n_prompts": n_prompts, "seeds_per_prompt": m_seeds, "n_new": n_new,
        "temperature": S.TEMPERATURE, "vocab": VOCAB,
        "losslessness": {"spec_T1_vs_target_sample": ll, "null_floor_tv": null},
        "cost": cost,
        "gate_probe": {
            "correct_rule": gate_correct,
            "broken_accept_always": gate["accept_always"],
            "broken_resample_from_draft": gate["resample_from_draft"],
            "n_prompts": n_gate, "seeds_per_prompt": n_seeds_gate, "n_new": n_new_gate,
        },
        "per_prompt": {"spec_T1": [{"prompt": p["prompt"], "nll": round(p["nll"], 6),
                                    "gen_head": p["gen_head"]} for p in spec],
                       "target_sample": [{"prompt": p["prompt"], "nll": round(p["nll"], 6)}
                                         for p in tgt]},
    }


def load_ckpt(root: str, device: str = "cuda") -> PathLM:
    """与 2631/2640 `load_ckpt` 同口径，但设备可控（CPU 冒烟用）。"""
    with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = PathLM(ModelConfig(**cfg["model"]), PathConfig(**cfg["path"]), vocab_size=VOCAB)
    sd = torch.load(os.path.join(root, "model.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m.eval().to(device)


def _params(ckpt):
    try:
        with open(os.path.join(ckpt, "results.json"), encoding="utf-8") as f:
            return json.load(f).get("params")
    except Exception:
        return None


# ------------------------------------------------------------------ P2：对抗臂（53.9M）

def run_p2(name, ckpt, val, log, smoke=False, device="cuda"):
    n_prompts = 3 if smoke else N_PROMPTS_ADV
    n_new = 48 if smoke else N_NEW_ADV
    model = load_ckpt(ckpt, device)
    pcap = model.pcap
    prompts = H.make_prompts(val, n_prompts)
    native = []
    for pi, prompt in enumerate(prompts):
        gen_full, _ = H.decode(model, prompt, n_new + T.GAMMA, pcap)
        native.append(gen_full)
    arms = {}
    for arm in ("adv_random_k4", "adv_bottomk4"):
        rows = V.arm_pass_gate(model, prompts, n_new, pcap, arm, native, log)
        a1_groups = [r["accepts"] for r in rows]
        n_rounds = sum(len(g) for g in a1_groups)
        hits = sum(1 for g in a1_groups for a in g if a >= 1)
        p = hits / max(n_rounds, 1)
        identical = all(r["identical_to_native"] for r in rows)
        arms[arm] = {
            "alpha1": round(p, 6),
            "n_rounds": n_rounds,
            "hits": hits,
            "wilson95": H.wilson_ci(p, n_rounds),
            "stream_identical_to_native": bool(identical),
            "first_mismatches": [{"prompt": r["prompt"], "pos": r["first_mismatch"]}
                                 for r in rows if not r["identical_to_native"]],
            "pseudo_accept_uniform": round(4.0 / VOCAB, 6),
            "per_prompt_accepts_sum": [int(sum(g)) for g in a1_groups],
        }
        log(f"[{name}] {arm}: alpha1={p:.6f} wilson={H.wilson_ci(p, n_rounds)} "
            f"identical={identical} rounds={n_rounds}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "path": os.path.relpath(ckpt, ROOT), "params": _params(ckpt),
        "n_prompts": n_prompts, "n_new": n_new, "K": 4,
        "candidate_construction": "与 2650-verify-gate-13m 逐字一致（verify_gate.arm_pass_gate / "
                                  "make_cands_adv_random / cands_adv_bottomk4），只换主干到 53.9M",
        "arms": arms,
    }


# ------------------------------------------------------------------ 主流程

JOBS = {
    "p1_13m_s0": ("p1", os.path.join(TRAIN_DIR, "13m_s0")),
    "p1_13m_s1": ("p1", os.path.join(TRAIN_DIR, "13m_s1")),
    "p2_53m_s0": ("p2", os.path.join(TRAIN_DIR, "53m_s0")),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--jobs", nargs="+", default=list(JOBS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-yield-s", type=float, default=60.0,
                    help="§20.1 臂间让出窗口（秒）")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "run_log.txt"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    dev = "cuda" if torch.cuda.is_available() and not args.no_lock else \
        ("cpu" if args.smoke else "cuda")
    results = {
        "experiment": "2673-robustness-13m",
        "wp": "6.2.4 (轴 6 预测轴 · Phase C: 温度>0 的无损投机采样 + 53.9M 对抗臂)",
        "contract": "README.md（预登记：臂定义 / L0–L5 判据 / 否证条件）",
        "timing_measured": False,
        "timing_note": "禁墙钟判据；GPU 实耗只在 run_log.txt，不进本文件",
        "config": {"temperature": S.TEMPERATURE, "gamma": T.GAMMA, "vocab": VOCAB,
                   "n_prompts": N_PROMPTS, "seeds_per_prompt": M_SEEDS,
                   "n_new_sample": N_NEW_SAMPLE, "n_new_adv": N_NEW_ADV,
                   "n_prompts_adv": N_PROMPTS_ADV,
                   "bootstrap": {"B": BOOT, "seed": BOOT_SEED, "cluster": "prompt"},
                   "gate_probe": {"n_prompts": N_PROMPTS_GATE, "seeds": M_SEEDS_GATE,
                                  "n_new": N_NEW_GATE},
                   "lock": "per-job §20.1（释放后 ≥60s 让出窗口）" if not args.no_lock
                           else "§20.1.2 一次性并发例外（纯推理≤≤1GB；未取锁）",
                   "no_lock": bool(args.no_lock), "smoke": bool(args.smoke)},
        "checkpoints": {}, "lock_events": [], "gpu_free_mib": {},
    }
    if args.resume and os.path.exists(os.path.join(args.out, "results.json")):
        with open(os.path.join(args.out, "results.json"), encoding="utf-8") as f:
            old = json.load(f)
        results["checkpoints"] = old.get("checkpoints", {})
        log(f"[resume] 读回已完成 job: {sorted(results['checkpoints'])}")

    def flush():
        with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        _write_resume(args.out, results, args.jobs)

    val = H.enwik8_val()
    todo = [j for j in args.jobs if j not in results["checkpoints"]]
    log(f"[run] jobs={args.jobs} todo={todo} smoke={args.smoke} no_lock={args.no_lock} "
        f"device={args.device}")
    use_lock = (not args.no_lock) and args.device == "cuda"
    for k, job in enumerate(todo):
        stage, ckpt = JOBS[job]
        run_fn = run_p1 if stage == "p1" else run_p2
        free0 = _gpu_free()
        ctx = _null_ctx() if not use_lock else gpu_lock("wp6.2.4")
        try:
            with ctx:
                results["gpu_free_mib"][job] = {"before": free0}
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                log(f"[job] {job} start ({ckpt}) gpu_free={free0} MiB")
                t0 = time.time()
                res = run_fn(job, ckpt, val, log, smoke=args.smoke, device=args.device)
                res["gpu_free_mib_before"] = free0
                res["gpu_free_mib_after"] = _gpu_free()
                res["gpu_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) \
                    if args.device == "cuda" else None
                res["lock"] = "none (§20.1.2 一次性并发例外)" if not use_lock else \
                    "gpu_lock §20.1 逐 job"
                results["checkpoints"][job] = res
                results["gpu_free_mib"][job]["after"] = _gpu_free()
                log(f"[job] {job} done wall={time.time() - t0:.0f}s")
        except GPUUnavailable as e:
            log(f"[job] {job} GPUUnavailable: {e}")
            results.setdefault("lock_events", []).append({"job": job, "event": "unavailable",
                                                          "msg": str(e)})
            flush()
            return 9
        flush()
        if use_lock and k < len(todo) - 1:
            log(f"[lock] §20.1 让出窗口：sleep {args.gpu_yield_s:.0f}s")
            time.sleep(args.gpu_yield_s)

    bad = has_wall_clock(results)
    results["_wall_clock_fields"] = bad
    if bad:
        log(f"[wall] results.json 含墙钟字段（应为空）：{bad}")
    flush()
    log(f"[run] wrote results.json / RESUME.md；wall_clock_fields={bad}")
    logf.close()
    return 0


def _gpu_free():
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _write_resume(out, results, jobs):
    lines = ["# RESUME — 2673-robustness-13m", "",
             "WP 6.2.4：温度>0 的无损投机采样（P1）+ 53.9M 对抗臂（P2）。进度逐 job 落盘。", ""]
    done = list(results["checkpoints"])
    lines.append(f"- 已完成 job：{done}")
    lines.append(f"- 未完成 job：{[j for j in jobs if j not in results['checkpoints']]}")
    for job, res in results["checkpoints"].items():
        if "cost" in res:
            c = res["cost"]
            lines.append(f"  - `{job}`：τ={c['tau']} tokens/fwd={c['tokens_per_forward']} "
                         f"α₁={c['alpha1']}")
        if "arms" in res:
            for a, v in res["arms"].items():
                lines.append(f"  - `{job}`/{a}：α₁={v['alpha1']} Wilson={v['wilson95']} "
                             f"identical={v['stream_identical_to_native']}")
    lines.append("")
    lines.append("重跑：`python3 run.py --out . --resume --jobs "
                 + " ".join(jobs) + "`（detached + ≥60s 让出窗口）")
    with open(os.path.join(out, "RESUME.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
