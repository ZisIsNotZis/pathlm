"""2632 · 第 N+2 草稿位接受率差的归因诊断（纯推理，CPU）。

不训练、不加参数、不写共享文件。只把 2631 三个 checkpoint 加载起来，
在同一份权重上换「第 N+2 草稿位读哪个头 / 用什么条件」，并给出：

1. **门禁 G1**：free-gen 下三臂发出流与 native 贪心逐位相同（argmax 路线）。
2. **teacher-anchored 分解**：真实文本前缀上逐位置读 4 种草稿 × 3 个口径：
   - `*_q`  = vs 真 t_{N+2}（质量口径）
   - `*_a`  = vs 部署目标 T2*_deploy（前缀含 v1；与 2631 的 α₁ 同定义）
   - `*_at` = vs 教师目标 T2*_teacher（前缀含真 t_{N+1}；条件与目标同前缀，最干净的头部对照）
   并按 `e = 1[v1 == 真 t_{N+1}]` 分层。
3. **Q2 top-k 召回探针**：`direct` 与 `chain` 对部署目标 T2* 的 top-k（k=1,2,4,8）召回。

指标定义见 README §0/§4；判据线 README §4（预登记，事后不得改）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
SPEC_DIR = os.path.join(ROOT, "axes/H-horizon/H3-composition/experiments/2631-spec-three-arms")
for _p in (ROOT, SPEC_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_arms as S  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402
from pathlm.decode import Decoder, decode, decode_spec  # noqa: E402

BOOT = 2000
BOOT_SEED = 2632
TOPKS = (1, 2, 4, 8)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_ckpt(root: str) -> PathLM:
    """CPU 加载 2631 的 checkpoint（复用，不改）。"""
    with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = PathLM(ModelConfig(**cfg["model"]), PathConfig(**cfg["path"]), vocab_size=206)
    sd = torch.load(os.path.join(root, "model.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m.eval()


def enwik8_val() -> np.ndarray:
    return np.load(os.path.join(ROOT, "data/enwik8_full.npz"))["val"]


@torch.no_grad()
def free_gen_pass(model, pcap, val, n_prompts: int, prompt_len: int, n_new: int,
                  stride: int = 5000) -> dict:
    """门禁 G1 + 接受率口径 A1：native 走 Decoder.step（decode），三臂走 decode_spec。"""
    per_mode: dict[str, list] = {m: [] for m in ("native",) + S.ARMS}
    for i in range(n_prompts):
        seg = val[i * stride:i * stride + prompt_len].astype(np.int64)
        prompt = torch.tensor(seg)
        native, _ = decode(model, prompt, n_new, pcap)
        per_mode["native"].append({"prompt": i, "gen": native, "rounds": None})
        for arm in S.ARMS:
            gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=S.GAMMA,
                                  draft_fn=S.make_draft_fn(arm))
            per_mode[arm].append({
                "prompt": i, "gen": gen, "rounds": list(st["accepts"]),
                "identical_to_native": gen == native,
                "first_mismatch": next((j for j, (a, b) in enumerate(zip(gen, native))
                                        if a != b), None) if gen != native else None,
            })
    return per_mode


def _argmax(x) -> int:
    return int(x.argmax())


@torch.no_grad()
def teacher_anchor_pass(model, pcap, val, offsets: list[int], length: int) -> list[dict]:
    """真实文本前缀上逐位置诊断。

    每个位置 j 记录：v1、真 t_{N+1}、真 t_{N+2}、四种 N+2 草稿、
    部署目标 T2*_deploy（前缀含 v1）、教师目标 T2*_teacher（前缀含真 t_{N+1}）、
    node2 / chain2(deploy) 的 top-8。
    """
    rows: list[dict] = []
    for seg_id, off in enumerate(offsets):
        toks = val[off:off + length + 3].astype(np.int64)
        dec = Decoder(model, window=pcap.window, anchors=pcap.anchors)
        h, _ = dec.step(int(toks[0]))
        for j in range(0, length):
            nodes = dec.last_nodes
            l1 = nodes[1]["logits"][0, 0].float()
            l2 = nodes[2]["logits"][0, 0].float()
            lc = nodes[1]["chain2"]["logits"][0, 0].float()
            v1 = _argmax(l1)
            true_next = int(toks[j + 1])
            true_t2 = int(toks[j + 2])
            d_direct = _argmax(l2)
            d_chain = _argmax(lc)
            pc, pd = lc.softmax(-1), l2.softmax(-1)
            d_comb = _argmax(S.COMBINED_LAM * pc + (1.0 - S.COMBINED_LAM) * pd)
            # oracle 条件 = 真 t_{N+1}（走显式 condition 分支，不是 deploy 分支）
            onodes = model._mtp_nodes(h, condition=torch.tensor([[true_next]]))
            d_oracle = _argmax(onodes[1]["chain2"]["logits"][0, 0].float())
            # 部署目标：把模型自己的 v1 当下一位喂进去再读 node1（因果，只依赖 v1）
            _, fn = dec.step_multi([v1])
            t2dep = _argmax(fn[1]["logits"][0, 0].float())
            dec.truncate(j)
            row = {
                "seg": seg_id, "off": int(off), "pos": j,
                "v1": v1, "true_next": true_next, "true_t2": true_t2,
                "d_direct": d_direct, "d_chain": d_chain,
                "d_oracle": d_oracle, "d_comb": d_comb,
                "t2dep": t2dep,
                "topk_direct": l2.topk(max(TOPKS)).indices.tolist(),
                "topk_chain": lc.topk(max(TOPKS)).indices.tolist(),
            }
            rows.append(row)
            h, _ = dec.step(int(toks[j + 1]))
            # 教师目标：前缀 [.., j+1]，= 位置 j+1 的 node1 argmax
            row["t2teacher"] = _argmax(dec.last_nodes[1]["logits"][0, 0].float())
    return rows


def _pct_ci(samples: list[float]) -> list:
    if not samples:
        return [None, None]
    return [round(float(np.percentile(samples, 2.5)), 6),
            round(float(np.percentile(samples, 97.5)), 6)]


def cluster_bootstrap(rows: list[dict], stat, B: int = BOOT, seed: int = BOOT_SEED) -> list:
    """按 segment 聚类的 percentile bootstrap（不把同段位置当独立样本）。"""
    by_seg: dict[int, list[dict]] = {}
    for r in rows:
        by_seg.setdefault(r["seg"], []).append(r)
    segs = sorted(by_seg)
    if not segs:
        return [None, None]
    vals = []
    rng = np.random.default_rng(seed)
    for _ in range(B):
        idx = rng.integers(0, len(segs), len(segs))
        pool = [r for i in idx for r in by_seg[segs[i]]]
        v = stat(pool)
        if v is not None:
            vals.append(v)
    return _pct_ci(vals)


def _mean_bool(rows, key) -> float:
    return float(np.mean([1.0 if r[key] else 0.0 for r in rows])) if rows else None


def _mean_eq(rows, a, b) -> float:
    return float(np.mean([1.0 if r[a] == r[b] else 0.0 for r in rows])) if rows else None


def _topk_recall(rows, draft_key, k) -> float:
    if not rows:
        return None
    return float(np.mean([1.0 if r["t2dep"] in r[draft_key][:k] else 0.0 for r in rows]))


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n_positions": len(rows), "n_segments": len({r["seg"] for r in rows})}
    out["acc_next_token"] = {
        "value": _mean_eq(rows, "v1", "true_next"),
        "ci": cluster_bootstrap(rows, lambda rs: _mean_eq(rs, "v1", "true_next")),
    }
    # 三个口径 × 四种草稿
    def mk(stat_fn, label):
        return {"value": stat_fn(rows), "ci": cluster_bootstrap(rows, stat_fn)}

    out["quality_vs_true_t2"] = {
        "direct": mk(lambda rs: _mean_eq(rs, "d_direct", "true_t2"), "q_direct"),
        "chain_deploy": mk(lambda rs: _mean_eq(rs, "d_chain", "true_t2"), "q_chain"),
        "chain_oracle": mk(lambda rs: _mean_eq(rs, "d_oracle", "true_t2"), "q_oracle"),
        "combined": mk(lambda rs: _mean_eq(rs, "d_comb", "true_t2"), "q_comb"),
    }
    out["accept_vs_deploy_target"] = {
        "direct": mk(lambda rs: _mean_eq(rs, "d_direct", "t2dep"), "a_direct"),
        "chain_deploy": mk(lambda rs: _mean_eq(rs, "d_chain", "t2dep"), "a_chain"),
        "chain_oracle": mk(lambda rs: _mean_eq(rs, "d_oracle", "t2dep"), "a_oracle"),
        "combined": mk(lambda rs: _mean_eq(rs, "d_comb", "t2dep"), "a_comb"),
    }
    out["accept_vs_teacher_target"] = {
        "direct": mk(lambda rs: _mean_eq(rs, "d_direct", "t2teacher"), "at_direct"),
        "chain_deploy": mk(lambda rs: _mean_eq(rs, "d_chain", "t2teacher"), "at_chain"),
        "chain_oracle": mk(lambda rs: _mean_eq(rs, "d_oracle", "t2teacher"), "at_oracle"),
        "combined": mk(lambda rs: _mean_eq(rs, "d_comb", "t2teacher"), "at_comb"),
    }
    # 分层 S1：e = 1[v1 == 真 t_{N+1}]
    e1 = [r for r in rows if r["v1"] == r["true_next"]]
    e0 = [r for r in rows if r["v1"] != r["true_next"]]
    out["strata"] = {
        "e1_pool_size": len(e1), "e0_pool_size": len(e0),
        "e1": {
            "q_direct": _mean_eq(e1, "d_direct", "true_t2"),
            "q_chain": _mean_eq(e1, "d_chain", "true_t2"),
            "q_oracle": _mean_eq(e1, "d_oracle", "true_t2"),
            "a_direct": _mean_eq(e1, "d_direct", "t2dep"),
            "a_chain": _mean_eq(e1, "d_chain", "t2dep"),
        },
        "e0": {
            "q_direct": _mean_eq(e0, "d_direct", "true_t2"),
            "q_chain": _mean_eq(e0, "d_chain", "true_t2"),
            "q_oracle": _mean_eq(e0, "d_oracle", "true_t2"),
            "a_direct": _mean_eq(e0, "d_direct", "t2dep"),
            "a_chain": _mean_eq(e0, "d_chain", "t2dep"),
        },
    }
    # Q2 top-k 召回探针（对部署目标 T2*）
    out["topk_recall_vs_deploy_target"] = {
        "direct": {k: _topk_recall(rows, "topk_direct", k) for k in TOPKS},
        "chain": {k: _topk_recall(rows, "topk_chain", k) for k in TOPKS},
        "topk": list(TOPKS),
    }
    return out


def summarize_free_gen(per_mode: dict, n_new: int) -> dict:
    groups = [row["rounds"] for row in per_mode["direct"]]
    boot = lambda stat: cluster_bootstrap(
        [{"seg": i, "a": g} for i, g in enumerate(groups)],
        lambda rs: stat([r["a"] for r in rs]))
    out = {"n_prompts": len(per_mode["native"])}
    for arm in S.ARMS:
        accs = [a for row in per_mode[arm] for a in row["rounds"]]
        n_rounds = len(accs)
        out[arm] = {
            "alpha1": sum(1 for a in accs if a >= 1) / max(n_rounds, 1),
            "tau": sum(accs) / max(n_rounds, 1),
            "n_rounds": n_rounds,
            "stream_identical_to_native": all(r["identical_to_native"] for r in per_mode[arm]),
            "first_mismatch": [r["first_mismatch"] for r in per_mode[arm]],
        }
    out["bit_identical_all_arms"] = all(out[a]["stream_identical_to_native"] for a in S.ARMS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        n_seg, length, n_prompts, prompt_len, n_new = 1, 8, 1, 32, 24
    else:
        n_seg, length, n_prompts, prompt_len, n_new = 12, 64, 6, 32, 96
    offsets = [i * 5000 for i in range(n_seg)]

    val = enwik8_val()
    log_path = os.path.join(args.out, "run_log.txt")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"[diag] {time.strftime('%Y-%m-%dT%H:%M:%S')} ckpts={args.ckpts} smoke={args.smoke} "
        f"n_seg={n_seg} length={length}")

    results: dict = {"config": {"n_seg": n_seg, "length": length,
                                "n_prompts": n_prompts, "prompt_len": prompt_len,
                                "n_new": n_new, "gamma": S.GAMMA,
                                "combined_lam": S.COMBINED_LAM, "topks": list(TOPKS),
                                "boot": BOOT, "boot_seed": BOOT_SEED,
                                "scope": "teacher-anchored on enwik8 val real text; "
                                         "no GPU, no training, no shared-file writes"},
                "checkpoints": {}}
    rec_all: list[dict] = []
    for ck in args.ckpts:
        name = os.path.basename(ck.rstrip("/"))
        model = load_ckpt(ck)
        pcap = model.pcap
        g1 = free_gen_pass(model, pcap, val, n_prompts, prompt_len, n_new)
        fg = summarize_free_gen(g1, n_new)
        rows = teacher_anchor_pass(model, pcap, val, offsets, length)
        summ = summarize(rows)
        summ["sha256_model_pt"] = sha256(os.path.join(ck, "model.pt"))
        summ["ckpt_path"] = os.path.abspath(ck)
        results["checkpoints"][name] = {"free_gen": fg, "teacher_anchor": summ}
        for r in rows:
            rec_all.append({"ckpt": name, **r})
        log(f"[{name}] G1 bit_identical={fg['bit_identical_all_arms']} "
            f"sha={summ['sha256_model_pt'][:12]}")
        for arm in S.ARMS:
            log(f"[{name}] free-gen {arm:8s} a1={fg[arm]['alpha1']:.4f} "
                f"tau={fg[arm]['tau']:.4f} ident={fg[arm]['stream_identical_to_native']}")
        q = summ["quality_vs_true_t2"]
        log(f"[{name}] q(true t2) direct={q['direct']['value']:.4f} "
            f"chain_deploy={q['chain_deploy']['value']:.4f} "
            f"chain_oracle={q['chain_oracle']['value']:.4f}")
        a = summ["accept_vs_deploy_target"]
        log(f"[{name}] a(T2*_deploy) direct={a['direct']['value']:.4f} "
            f"chain={a['chain_deploy']['value']:.4f} comb={a['combined']['value']:.4f}")
        tk = summ["topk_recall_vs_deploy_target"]
        log(f"[{name}] topk recall direct={ {k: round(v,4) for k,v in tk['direct'].items()} } "
            f"chain={ {k: round(v,4) for k,v in tk['chain'].items()} }")
        del model

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "diag_records.jsonl"), "w", encoding="utf-8") as f:
        for r in rec_all:
            f.write(json.dumps(r) + "\n")
    log(f"[diag] wrote results.json / diag_records.jsonl {time.strftime('%H:%M:%S')}")
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
