"""R6 · Spec 三臂的评测入口（三臂 × native 对照；两档规模 × 跨种子）。

口径（契约 §4，逐条可核）：

- **native 对照跑在同一条解码路径上**（`pathlm/decode.py::Decoder.step()` 逐 token），
  **不用**批量 teacher-forcing 前向代替——这正是 ticket 12 旧结论出错的地方。
- 三臂 = `chain`（条件链头）/ `direct`（直连头）/ `combined`（概率混合，λ=0.5 预登记），
  唯一变量 = 第 N+2 草稿位的分布来源；第 N+3 位三臂固定（见 `spec_arms.py`）。
- **门禁**：argmax 路线 ⇒ 声明**非分布保持**；给出更强的**逐位等价**（发出流 == native 贪心流）。
- **墙钟**：同进程、同硬件、同批量（B=1）、native 与 spec **交替**测量；报逐次重复极差 +
  按 prompt 聚类的配对 bootstrap CI。单次 tok/s 不作结论。
- **等效算力净账**：`c₃ / (tok/fwd)`（c₃ = 实测 width-3/width-1 前向成本比），
  另报解析的 MTP 头成本占比。

跑法：`python3 run.py --ckpts <ckpt 目录...> --out . [--smoke]`
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start: str) -> str:
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "pathlm")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"找不到仓库根: {start}")


ROOT = _repo_root(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import spec_arms as S  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

PROMPT_LEN = 32
N_ACC = 200            # 接受率口径的每 prompt 生成 token 数
N_TIME = 128           # 墙钟口径的每 prompt 生成 token 数
P_ACC = 12             # 接受率口径的 prompt 数
P_TIME = 6             # 墙钟口径的 prompt 数
REPEATS = 5
BOOT = 2000
BOOT_SEED = 2631
ALL_MODES = ("native",) + S.ARMS


def load_ckpt(root: str) -> PathLM:
    with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = PathLM(ModelConfig(**cfg["model"]), PathConfig(**cfg["path"]), vocab_size=206)
    sd = torch.load(os.path.join(root, "model.pt"), map_location="cpu",
                    weights_only=True)
    m.load_state_dict(sd)
    return m.eval().cuda()


def enwik8_val():
    blob = np.load("data/enwik8_full.npz")
    return blob["val"]


def make_prompts(val, n_prompts: int, stride: int = 5000) -> list[torch.Tensor]:
    out = []
    for i in range(n_prompts):
        off = i * stride
        seg = val[off:off + PROMPT_LEN].astype(np.int64)
        out.append(torch.tensor(seg).cuda())
    return out


@torch.no_grad()
def accept_pass(model, pcap, prompts, n_acc: int) -> dict:
    """接受率口径：每 prompt 跑 native + 三臂，记录每轮接受数与发出流。"""
    per_mode: dict[str, list] = {m: [] for m in ALL_MODES}
    for pi, prompt in enumerate(prompts):
        native, _ = S.arm_decode(model, prompt, n_acc, pcap, "native")
        per_mode["native"].append({"prompt": pi, "gen_head": native[:8],
                                   "rounds": [], "n_rounds": None,
                                   "decoder_path": "pathlm.decode.decode(Decoder.step)"})
        for arm in S.ARMS:
            gen, st = S.arm_decode(model, prompt, n_acc, pcap, arm)
            per_mode[arm].append({
                "prompt": pi, "gen_head": gen[:8],
                "identical_to_native": gen == native,
                "first_mismatch": next((j for j, (a, b) in enumerate(zip(gen, native))
                                        if a != b), None) if gen != native else None,
                "rounds": list(st["accepts"]),
                "n_rounds": len(st["accepts"]),
                "n_forward": st["n_forward"],
            })
    return per_mode


@torch.no_grad()
def timing_pass(model, pcap, prompts, repeats: int, n_time: int) -> list[dict]:
    """墙钟口径：native 与三臂在**同一循环里交替**跑，逐 (rep, prompt) 记墙钟。"""
    recs = []
    for _ in range(2):                      # warmup
        for m in ALL_MODES:
            S.arm_decode(model, prompts[0], 16, pcap, m)
    for rep in range(repeats):
        for pi, prompt in enumerate(prompts):
            row = {"rep": rep, "prompt": pi}
            for m in ALL_MODES:
                S.sync()
                t0 = time.time()
                S.arm_decode(model, prompt, n_time, pcap, m)
                S.sync()
                row[m] = n_time / max(time.time() - t0, 1e-9)
            recs.append(row)
    return recs


def _pct_ci(samples: list[float]) -> list[float]:
    if not samples:
        return [None, None]
    return [round(float(np.percentile(samples, 2.5)), 6),
            round(float(np.percentile(samples, 97.5)), 6)]


def cluster_bootstrap(groups: list[list], stat, B: int = BOOT,
                      seed: int = BOOT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(groups)
    if n == 0:
        return [None, None]
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(stat([groups[i] for i in idx]))
    return _pct_ci(vals)


def _acc_alpha1(groups):
    tot = sum(len(g) for g in groups)
    return sum(1 for g in groups for a in g if a >= 1) / max(tot, 1)


def _acc_alpha2(groups):
    den = sum(1 for g in groups for a in g if a >= 1)
    num = sum(1 for g in groups for a in g if a >= 2)
    return (num / den) if den else 0.0


def _acc_tau(groups):
    tot = sum(len(g) for g in groups)
    return sum(a for g in groups for a in g) / max(tot, 1)


def summarize_arm(per_prompt: list[dict], n_acc: int) -> dict:
    groups = [row["rounds"] for row in per_prompt]
    n_rounds = sum(len(g) for g in groups)
    accepted = sum(a for g in groups for a in g)
    n_tokens = sum(n_acc for _ in per_prompt)
    # `tokens_per_forward` 用**每轮发出结构恒等式** 1 + τ（契约 §4.3 / spec_arms.round_metrics
    # 的口径）：每轮恒发出 1 + a 个 token，故 tok/fwd = (n_rounds + Σa) / n_rounds = 1 + τ。
    # 注意这与「实际截断后落在 gen 里的 token 数 / n_rounds」（= n_tokens / n_rounds）在**末轮**
    # 相差至多 γ 个/prompt（decode_spec 截断 `gen[:n_new]`）；两者都报出来（见
    # `tokens_per_forward_actual_emitted`），使契约恒等式与真实发出量都可核。
    tf_struct = (n_rounds + accepted) / max(n_rounds, 1)
    out = {
        "n_prompts": len(per_prompt), "n_rounds": n_rounds,
        "accepted_total": accepted,
        "alpha1": _acc_alpha1(groups),
        "alpha2_given_alpha1": _acc_alpha2(groups),
        "tau": _acc_tau(groups),
        "tokens_per_forward": tf_struct,
        "tokens_per_forward_actual_emitted": n_tokens / max(n_rounds, 1),
        "alpha1_ci": cluster_bootstrap(groups, _acc_alpha1),
        "alpha2_given_alpha1_ci": cluster_bootstrap(groups, _acc_alpha2),
        "tau_ci": cluster_bootstrap(groups, _acc_tau),
        "stream_identical_to_native": all(
            row.get("identical_to_native", True) for row in per_prompt),
        "n_prompts_stream_checked": sum(
            1 for row in per_prompt if "identical_to_native" in row),
        "n_forward_total": sum(row.get("n_forward") or 0 for row in per_prompt),
    }
    out["tokens_per_forward_ci"] = cluster_bootstrap(
        groups, lambda gs: (sum(len(g) for g in gs) + sum(a for g in gs for a in g))
        / max(sum(len(g) for g in gs), 1))
    return out


def summarize_timing(recs: list[dict]) -> dict:
    by_prompt: dict[int, list[dict]] = {}
    for r in recs:
        by_prompt.setdefault(r["prompt"], []).append(r)
    plain: dict[str, float] = {}
    for m in ALL_MODES:
        plain[m] = statistics.median([r[m] for r in recs])
    # 每个 prompt 的「重复中位数」加速比 ⇒ 以 prompt 为聚类单位
    per_prompt_speed: dict[str, list[float]] = {a: [] for a in S.ARMS}
    for pi, rows in sorted(by_prompt.items()):
        for a in S.ARMS:
            ratios = [r[a] / r["native"] for r in rows]
            per_prompt_speed[a].append(statistics.median(ratios))
    out = {"tok_s_median": {m: round(v, 2) for m, v in plain.items()},
           "repeats": max(r["rep"] for r in recs) + 1,
           "n_prompts": len(by_prompt)}
    for a in S.ARMS:
        ratios_all = [r[a] / r["native"] for r in recs]
        out[f"{a}_speedup"] = {
            "point": round(statistics.median(per_prompt_speed[a]), 4),
            "ci_by_prompt": cluster_bootstrap(per_prompt_speed[a], statistics.median),
            "range_over_reps": [round(min(ratios_all), 4), round(max(ratios_all), 4)],
            "n_ratio_samples": len(ratios_all),
        }
    # 配对差（同 prompt 同 rep）：最强 − 次强，用 prompt 聚类 bootstrap
    pairs = {}
    for i, a in enumerate(S.ARMS):
        for b in S.ARMS[i + 1:]:
            per_prompt_diff = []
            for pi, rows in sorted(by_prompt.items()):
                d = [r[a] / r["native"] - r[b] / r["native"] for r in rows]
                per_prompt_diff.append(statistics.median(d))
            pairs[f"{a}_minus_{b}"] = {
                "point": round(statistics.median(per_prompt_diff), 4),
                "ci_by_prompt": cluster_bootstrap(per_prompt_diff, statistics.median),
            }
    out["paired_speedup_diffs"] = pairs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    n_acc, p_acc, n_time, p_time = N_ACC, P_ACC, N_TIME, P_TIME
    if args.smoke:
        n_acc, p_acc, n_time, p_time = 32, 2, 24, 2
        args.repeats = 2
    val = enwik8_val()
    prompts = make_prompts(val, max(p_acc, p_time))
    log_path = os.path.join(args.out, "run_log.txt")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"[run] {time.strftime('%Y-%m-%dT%H:%M:%S')} ckpts={args.ckpts} smoke={args.smoke}")

    ckpt_dirs: list[str] = []
    for pat in args.ckpts:
        ckpt_dirs.extend(sorted(glob.glob(pat)))
    results: dict = {"config": {"prompt_len": PROMPT_LEN, "n_acc": n_acc,
                                "p_acc": p_acc, "n_time": n_time, "p_time": p_time,
                                "repeats": args.repeats, "gamma": S.GAMMA,
                                "combined_lam": S.COMBINED_LAM,
                                "arms": list(S.ARMS),
                                "tail_decay": "第一个发出位（N+1）不设接受测试（引擎性质）"},
                "checkpoints": {}}

    for ck in ckpt_dirs:
        name = os.path.basename(ck.rstrip("/"))
        log(f"[ckpt] {name} :: {ck}")
        model = load_ckpt(ck)
        pcap = model.pcap
        with open(os.path.join(ck, "results.json"), encoding="utf-8") as f:
            ckres = json.load(f)
        micro = S.micro_forward_cost(model, pcap)
        heads = S.mtp_head_macs(model.mcfg, model.vocab_size)
        log(f"[{name}] micro: {micro}  heads: {heads['head_over_trunk_frac']} of trunk")

        acc = accept_pass(model, pcap, prompts[:p_acc], n_acc)
        arm_summary = {a: summarize_arm(acc[a], n_acc) for a in S.ARMS}
        native_ok = all(row.get("decoder_path", "").startswith("pathlm.decode.decode")
                    for row in acc["native"])
        for a in S.ARMS:
            s = arm_summary[a]
            s["equivalent_compute_ratio"] = round(
                micro["c3_vs_w1"] / s["tokens_per_forward"], 4)
            log(f"[{name}] {a:8s} a1={s['alpha1']:.4f} a2|1="
                f"{s['alpha2_given_alpha1']:.4f} tau={s['tau']:.4f} "
                f"tok/fwd={s['tokens_per_forward']:.4f} "
                f"stream==native:{s['stream_identical_to_native']}")

        tm = timing_pass(model, pcap, prompts[:p_time], args.repeats, n_time)
        wall = summarize_timing(tm)
        log(f"[{name}] wall: {json.dumps(wall['tok_s_median'])}")
        for a in S.ARMS:
            w = wall[f"{a}_speedup"]
            log(f"[{name}] {a:8s} speedup={w['point']} ci={w['ci_by_prompt']} "
                f"range={w['range_over_reps']}")

        results["checkpoints"][name] = {
            "path": ck, "params": ckres.get("params"),
            "train_config": ckres.get("config"),
            "resolved": ckres.get("resolved"),
            "bpc": ckres.get("bpc"), "peak_gpu_mem_mb": ckres.get("peak_gpu_mem_mb"),
            "wall_minutes_train": ckres.get("wall_minutes"),
            "micro": micro, "mtp_head_macs": heads,
            "native_is_plain_decoder_path": native_ok,
            "acceptance": arm_summary,
            "wall": wall,
            "timing_raw": tm,
            "acceptance_trace": {a: acc[a] for a in S.ARMS},
            "bit_identical_all_arms": all(
                arm_summary[a]["stream_identical_to_native"] for a in S.ARMS),
        }
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # 活性量：逐 (arm, prompt, round) 落盘（SCHEMA §10）
    with open(os.path.join(args.out, "cycle_log.jsonl"), "w", encoding="utf-8") as f:
        for ck, rec in results["checkpoints"].items():
            for arm, rows in rec["acceptance_trace"].items():
                for row in rows:
                    f.write(json.dumps({
                        "ckpt": ck, "arm": arm, "prompt": row["prompt"],
                        "n_rounds": row["n_rounds"], "rounds": row["rounds"],
                        "identical_to_native": row.get("identical_to_native"),
                        "first_mismatch": row.get("first_mismatch"),
                    }) + "\n")
    log(f"[run] wrote results.json / cycle_log.jsonl  {time.strftime('%H:%M:%S')}")
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
