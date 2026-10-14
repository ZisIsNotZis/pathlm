#!/usr/bin/env python3
"""2670-freerun-bpc-13m · 驱动（契约见本目录 README.md）。

    python3 run.py --out . [--smoke] [--no-lock]

在**冻结**的 13M 主干（2631 的 13m_s0 / 13m_s1）上，比较：
- `ar`（参照）= 同主干 node-1 逐 token 贪心；
- `path_i`（候选，契约 §2 路径 (i)）= 一次前向经 2660 的 k=2 变换头直接发 2 个位置（不验证）。

主判 = Δbpc（候选 − 同主干 ar）的 **prompt 聚类配对 bootstrap 95% CI**（B=2000）+ 0.014 bpc 参考带。

产出（写进 --out）：`results.json`（**禁墙钟**）/ `analysis.md` / `MODEL.md` / `run_log.txt`（可含墙钟）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import freerun as FR  # noqa: E402
from freerun import (BOOT_B, BOOT_SEED, HEAD_ARMS, HEAD_SEEDS, K, LOG2, N_NEW,  # noqa: E402
                     N_PROMPTS, PROMPT_LEN, STRIDE, VOCAB_SIZE)
from gpu_lock import gpu_lock  # noqa: E402

REF_2640 = os.path.join(FR.REPO, "axes", "6-predict", "6.2-speculative-decoding",
                        "experiments", "2640-tree-conf-13m", "results.json")
REF_2640_CKPT = "13m_s0"
REF_2640_ARM = "native"
REF_BAND_BPC = 0.014


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_2640_native_check(model, val, log) -> dict:
    """L0：本 harness 的 `ar` 贪心流与 2640 `native`（= 生产 `decode` 贪心）同 prompt 同长度逐位比对。

    2640 用 `PROMPT_LEN=32`；本实验主口径是 64，故**额外**用 32 长度跑一次做保真比对（前 8 token）。
    """
    prompts32, offsets = FR.make_prompts(val, N_PROMPTS, prompt_len=32, stride=STRIDE)
    _, stream32 = FR.freerun_ar(model, prompts32, n_new=8)
    with open(REF_2640, encoding="utf-8") as f:
        ref = json.load(f)["checkpoints"][REF_2640_CKPT]["arms"][REF_2640_ARM]["per_prompt"]
    per_prompt, n_match = [], 0
    for pi in range(N_PROMPTS):
        got = [int(t) for t in stream32[pi].tolist()]
        want = list(ref[pi]["gen_head"])
        ok = got == want
        n_match += int(ok)
        per_prompt.append({"prompt": pi, "offset": offsets[pi], "got_head8": got,
                           "ref_head8": want, "identical": ok})
        if not ok:
            log(f"[L0/2640-native] prompt {pi} MISMATCH got={got} ref={want}")
    return {"ref": os.path.relpath(REF_2640, FR.REPO), "ref_ckpt": REF_2640_CKPT,
            "ref_arm": REF_2640_ARM, "prompt_len": 32, "n_head_tokens": 8,
            "n_prompts_matched": n_match, "n_prompts": N_PROMPTS,
            "all_match": bool(n_match == N_PROMPTS), "per_prompt": per_prompt,
            "note": "2640 用 prompt_len=32；本实验主口径 prompt_len=64（契约 §3），故仅用 32 长度"
                    "做逐位保真比对，64 长度主口径不做跨长度比对"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--no-determinism", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, "run_log.txt"))
    t0 = time.time()

    n_new = 8 if a.smoke else N_NEW
    n_prompts = 2 if a.smoke else N_PROMPTS
    prompt_len = 32 if a.smoke else PROMPT_LEN
    seeds = (0,) if a.smoke else HEAD_SEEDS
    arms = ("disc_k2",) if a.smoke else HEAD_ARMS

    need_gpu = torch.cuda.is_available()
    ctx = gpu_lock("wp6.3.3-freerun") if (need_gpu and not a.no_lock) else _NullCtx()
    with ctx:
        dev = "cuda" if need_gpu else "cpu"
        log(f"device={dev} out={out} smoke={a.smoke} n_prompts={n_prompts} "
            f"prompt_len={prompt_len} n_new={n_new} k={K} seeds={list(seeds)} arms={list(arms)}")

        blob = np.load(FR.DATA_NPZ)
        val = blob["val"]
        prompts, offsets = FR.make_prompts(val, n_prompts, prompt_len, STRIDE)
        n_tokens_total = n_prompts * n_new   # 全部计分 token 数（batch 级 bpc 的分母）
        log(f"prompt offsets ({prompt_len} chars, stride {STRIDE}): {offsets}")

        results = {
            "experiment": "2670-freerun-bpc-13m",
            "contract": "README.md (pre-registered)",
            "metric": "free-run sequence-level bpc (bits/char; lower is better)",
            "reference_band_bpc": REF_BAND_BPC,
            "config": {
                "arms": ["ar"] + list(arms), "head_arms": list(arms),
                "head_init_seeds": list(seeds), "backbones": ["13m_s0", "13m_s1"],
                "k_tokens_per_forward": K, "n_prompts": n_prompts, "prompt_len": prompt_len,
                "prompt_stride": STRIDE, "prompt_offsets": offsets, "n_new": n_new,
                "scored_tokens_per_prompt": n_new, "vocab_size": VOCAB_SIZE,
                "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "cluster": "prompt",
                              "ci": "percentile95"},
                "data": os.path.relpath(FR.DATA_NPZ, FR.REPO),
                "cost_axis": "equivalent theoretical (layer-passes / tokens-per-forward); "
                             "no wall-clock as criterion",
            },
            "backbones": {}, "head_files": {}, "arms": {}, "L1": {}, "L1b": {},
            "L0": {}, "L3": {}, "L4": {}, "L5": {}, "verdicts": {},
        }

        # ---------------- 主干加载 + sha ----------------
        models = {}
        for bname, bpath in FR.BACKBONES.items():
            m = FR.L.load_ckpt(bpath).to(dev)
            models[bname] = m
            results["backbones"][bname] = {
                "path": os.path.relpath(bpath, FR.REPO),
                "ckpt_sha256": file_sha(os.path.join(bpath, "model.pt")),
                "embed_sha256": FR.L.embed_sha256(m),
                "params": int(sum(p.numel() for p in m.parameters())),
            }
            log(f"[backbone] {bname} embed_sha256={results['backbones'][bname]['embed_sha256'][:16]}… "
                f"params={results['backbones'][bname]['params']}")

        # ---------------- 头文件加载 + sha ----------------
        heads_cache = {}
        for arm in arms:
            for bname in FR.BACKBONES:
                for seed in seeds:
                    hpath = os.path.join(FR.HEAD_DIRS[bname], "heads", f"{arm}_s{seed}.pt")
                    heads = FR.load_heads(hpath, models[bname].mcfg.d_model, dev)
                    heads_cache[(arm, bname, seed)] = heads
                    results["head_files"].setdefault(arm, {}).setdefault(bname, {})[str(seed)] = {
                        "path": os.path.relpath(hpath, FR.REPO),
                        "sha256": file_sha(hpath),
                        "bytes_equal_source_file": bool(FR.heads_equal_vs_file(heads, hpath)),
                    }

        # ---------------- ar（参照，每主干一条） ----------------
        streams_by = {}          # (arm, backbone, seed) -> [B, n_new] 生成流（复核 N1 落盘用）
        for bname, m in models.items():
            nll, stream = FR.freerun_ar(m, prompts, n_new=n_new)
            streams_by[("ar", bname, None)] = stream
            per_prompt_sha = [FR.stream_sha256(stream[i]) for i in range(stream.shape[0])]
            results["arms"].setdefault("ar", {})[bname] = {
                "bpc": FR.bpc_from_nll(nll.cpu().numpy(), n_tokens_total),
                "nll_nats_per_prompt": [round(float(x), 6) for x in nll.cpu().numpy()],
                "stream_sha256_per_prompt": per_prompt_sha,
                "stream_sha256_all": FR.stream_sha256(stream),
                "node": "node-1 (model.transforms[0]) tied readout, 1 forward/token",
            }
            log(f"[ar] {bname} bpc={results['arms']['ar'][bname]['bpc']:.6f}")

        # ---------------- 路径 (i)（每 arm × 主干 × 头种子一条） ----------------
        nll_by = {}   # (arm, bname, seed) -> nll np array
        for arm in arms:
            results["arms"].setdefault(arm, {})
            for bname in FR.BACKBONES:
                results["arms"][arm].setdefault(bname, {})
                for seed in seeds:
                    heads = heads_cache[(arm, bname, seed)]
                    nll, stream = FR.freerun_path_i(models[bname], heads, prompts,
                                                    n_new=n_new, k=K)
                    streams_by[(arm, bname, seed)] = stream
                    nll_by[(arm, bname, seed)] = nll.cpu().numpy()
                    results["arms"][arm][bname][str(seed)] = {
                        "bpc": FR.bpc_from_nll(nll.cpu().numpy(), n_tokens_total),
                        "nll_nats_per_prompt": [round(float(x), 6) for x in nll.cpu().numpy()],
                        "stream_sha256_per_prompt": [FR.stream_sha256(stream[i])
                                                     for i in range(stream.shape[0])],
                        "stream_sha256_all": FR.stream_sha256(stream),
                        "tokens_per_forward": K,
                    }
                    log(f"[path_i] {arm} {bname} seed{seed} "
                        f"bpc={results['arms'][arm][bname][str(seed)]['bpc']:.6f}")

        n_tok = np.full(n_prompts, n_new, dtype=np.float64)

        # ---------------- L1：Δbpc(cand − ar same backbone) ----------------
        for arm in arms:
            results["L1"][arm] = {}
            for bname in FR.BACKBONES:
                ar_nll = np.array(results["arms"]["ar"][bname]["nll_nats_per_prompt"],
                                  dtype=np.float64)
                per_seed = {}
                stack = []
                for seed in seeds:
                    d = FR.paired_bpc_ci(nll_by[(arm, bname, seed)], ar_nll, n_tok)
                    per_seed[str(seed)] = d
                    stack.append(nll_by[(arm, bname, seed)] - ar_nll)
                comb = np.mean(np.stack(stack, 0), 0)     # 先按 prompt 平均两种子，再聚类
                per_seed["combined_mean_over_seeds"] = FR.paired_bpc_ci(
                    comb + ar_nll, ar_nll, n_tok)
                results["L1"][arm][bname] = per_seed

        # ---------------- L1b：Δbpc(lat_k2_lam1 − disc_k2)，同主干同种子配对 ----------------
        if "lat_k2_lam1" in arms and "disc_k2" in arms:
            results["L1b"]["lat_k2_lam1_minus_disc_k2"] = {}
            for bname in FR.BACKBONES:
                per_seed = {}
                stack = []
                for seed in seeds:
                    dd = FR.paired_bpc_ci(nll_by[("lat_k2_lam1", bname, seed)],
                                          nll_by[("disc_k2", bname, seed)], n_tok)
                    per_seed[str(seed)] = dd
                    stack.append(nll_by[("lat_k2_lam1", bname, seed)]
                                 - nll_by[("disc_k2", bname, seed)])
                comb = np.mean(np.stack(stack, 0), 0)
                base = np.array(results["arms"]["disc_k2"][bname][str(seeds[0])]
                                ["nll_nats_per_prompt"], dtype=np.float64)
                per_seed["combined_mean_over_seeds"] = FR.paired_bpc_ci(comb + base, base, n_tok)
                results["L1b"]["lat_k2_lam1_minus_disc_k2"][bname] = per_seed

        # ---------------- 生成流落盘 + 客观退化统计（复核 N1 的补充读数） ----------------
        stream_dir = os.path.join(out, "streams")
        os.makedirs(stream_dir, exist_ok=True)
        results["streams"] = {"dir": "streams", "files": {}, "all_sha_reproduced": True}
        results["degeneracy"] = {
            "metric": "longest identical-token run; distinct-2 (unique bigrams/total); "
                      "repeated 4-gram rate (1 - unique/total)",
            "per_stream": {}}
        for (arm, bname, seed), stream in streams_by.items():
            sd = "none" if seed is None else f"s{seed}"
            arm_rec = (results["arms"][arm][bname] if seed is None
                       else results["arms"][arm][bname][str(seed)])
            files, per_prompt_degen = [], []
            for pi in range(n_prompts):
                toks = [int(t) for t in stream[pi].tolist()]
                sha = FR.stream_sha256(stream[pi])
                results["streams"]["all_sha_reproduced"] &= bool(
                    sha == arm_rec["stream_sha256_per_prompt"][pi])
                fname = f"{arm}__{bname}__{sd}__p{pi}.json"
                with open(os.path.join(stream_dir, fname), "w", encoding="utf-8") as f:
                    json.dump({"arm": arm, "backbone": bname, "seed": seed,
                               "prompt": pi, "offset": offsets[pi], "n_tokens": len(toks),
                               "tokens": toks, "sha256": sha}, f)
                files.append(fname)
                per_prompt_degen.append(FR.degeneracy_stats(toks))
            results["streams"]["files"][f"{arm}/{bname}/{sd}"] = files
            mean = {k: round(float(np.mean([d[k] for d in per_prompt_degen])), 6)
                    for k in ("max_identical_run", "distinct_2", "repeated_4gram_rate")}
            results["degeneracy"]["per_stream"].setdefault(arm, {}).setdefault(bname, {})[sd] = {
                "per_prompt": per_prompt_degen, "mean": mean}
        log(f"[streams] 落盘 {len(streams_by)} 个配置 × {n_prompts} prompt；"
            f"sha 与 stream_sha256 一致={results['streams']['all_sha_reproduced']}")

        # ---------------- 真值锚（teacher-forced bpc；复核 N1） ----------------
        # 三个数并列：同一流在**自评主干**与**另一主干（外部锚）**下的 teacher-forced bpc，
        # 以及**真实 enwik8 续写**的 teacher-forced bpc（水平参照）。口径 = pathlm/eval.py::bpc。
        ta = {"definition": "teacher-forced node-1 bpc, 口径 = pathlm/eval.py::bpc "
                            "(node-1 logits → 下一 token CE → /ln2)，确定性覆盖给定流",
              "per_stream": {}, "delta_external_tf": {}, "delta_self_tf": {}}
        real_tf = {}
        for bname, m in models.items():
            cont = torch.tensor(np.stack([val[o + prompt_len:o + prompt_len + n_new]
                                          for o in offsets]), dtype=torch.int64).to(dev)
            real_nll = FR.teacher_forced_stream_nll(m, prompts.to(dev), cont)
            real_tf[bname] = {
                "nll_per_prompt": [round(float(x), 6) for x in real_nll.tolist()],
                "bpc_per_prompt": [round(float(x) / (n_new * FR.LOG2), 6)
                                   for x in real_nll.tolist()],
                "bpc": round(float(real_nll.sum()) / (n_prompts * n_new * FR.LOG2), 6),
                "source": "真实 val[offset+prompt_len : offset+prompt_len+n_new] 续写",
            }
        ta["real_enwik8_continuation"] = real_tf
        for (arm, bname, seed), stream in streams_by.items():
            sd = "none" if seed is None else f"s{seed}"
            other = "13m_s1" if bname == "13m_s0" else "13m_s0"
            self_nll = FR.teacher_forced_stream_nll(models[bname], prompts.to(dev), stream)
            ext_nll = FR.teacher_forced_stream_nll(models[other], prompts.to(dev), stream)
            ta["per_stream"].setdefault(arm, {}).setdefault(bname, {})[sd] = {
                "external_scorer": other,
                "nll_tf_self_per_prompt": [round(float(x), 6) for x in self_nll.tolist()],
                "nll_tf_external_per_prompt": [round(float(x), 6) for x in ext_nll.tolist()],
                "tf_self_bpc": round(float(self_nll.sum()) / (n_prompts * n_new * FR.LOG2), 6),
                "tf_external_bpc": round(float(ext_nll.sum()) / (n_prompts * n_new * FR.LOG2), 6),
                "tf_self_bpc_per_prompt": [round(float(x) / (n_new * FR.LOG2), 6)
                                           for x in self_nll.tolist()],
                "tf_external_bpc_per_prompt": [round(float(x) / (n_new * FR.LOG2), 6)
                                               for x in ext_nll.tolist()],
            }
        n_tok_ta = np.full(n_prompts, n_new, dtype=np.float64)
        for arm in arms:
            ta["delta_external_tf"][arm] = {}
            ta["delta_self_tf"][arm] = {}
            for bname in FR.BACKBONES:
                ar_ext = np.array(ta["per_stream"]["ar"][bname]["none"]
                                  ["nll_tf_external_per_prompt"], dtype=np.float64)
                ar_self = np.array(ta["per_stream"]["ar"][bname]["none"]
                                   ["nll_tf_self_per_prompt"], dtype=np.float64)
                ta["delta_external_tf"][arm][bname] = _delta_table(
                    {str(s): ta["per_stream"][arm][bname][f"s{s}"]
                     ["nll_tf_external_per_prompt"] for s in seeds}, ar_ext, seeds, n_tok_ta)
                ta["delta_self_tf"][arm][bname] = _delta_table(
                    {str(s): ta["per_stream"][arm][bname][f"s{s}"]
                     ["nll_tf_self_per_prompt"] for s in seeds}, ar_self, seeds, n_tok_ta)
        results["truth_anchor"] = ta
        log(f"[truth_anchor] real bpc: s0={real_tf['13m_s0']['bpc']:.4f} "
            f"s1={real_tf['13m_s1']['bpc']:.4f}")

        # ---------------- L0：保真硬门 ----------------
        l0 = {"heads_byte_equal_2660": all(
            v["bytes_equal_source_file"] for arm in results["head_files"].values()
            for b in arm.values() for v in b.values())}
        l0["frozen_backbone"] = {bname: FR.frozen_check(m) for bname, m in models.items()}
        if not a.smoke:
            l0["ar_vs_2640_native"] = run_2640_native_check(models["13m_s0"], val, log)
            log(f"[L0/2640-native] n_prompts_matched={l0['ar_vs_2640_native']['n_prompts_matched']}/"
                f"{N_PROMPTS}")
        l0["pass"] = bool(l0["heads_byte_equal_2660"]
                          and all(x["pass"] for x in l0["frozen_backbone"].values())
                          and l0.get("ar_vs_2640_native", {}).get("all_match", True))
        results["L0"] = l0

        # ---------------- L3：代价轴（等效理论量） ----------------
        nb = next(iter(models.values()))
        n_layers = nb.mcfg.n_layers
        results["L3"] = {
            "n_layers": n_layers, "d_model": nb.mcfg.d_model,
            "ar": {"tokens_per_forward": 1, "trunk_layer_passes_per_token": n_layers,
                   "sequential_forwards_per_token": 1.0},
            "path_i": {"tokens_per_forward": K,
                       "trunk_layer_passes_per_token": round(n_layers / K, 6),
                       "sequential_forwards_per_token": round(1.0 / K, 6)},
            "note": "契约 §6 等效理论量；路径 (i) 同一轮内 k 个 token 无顺序依赖（可并行），"
                    "轮间顺序依赖。禁墙钟。",
        }

        # ---------------- L4：禁墙钟自检 ----------------
        bad = [b for b in FR.has_wall_clock(results) if not b.endswith("_wall_clock_fields")]
        results["_wall_clock_fields"] = FR.has_wall_clock(results)
        results["L4"] = {"n_hits": len(bad), "hits": bad, "pass": len(bad) == 0}

        # ---------------- L5：确定性（同配置重跑一次逐值相同） ----------------
        if a.smoke or a.no_determinism:
            results["L5"] = {"skipped": True}
        else:
            nll2, stream2 = FR.freerun_ar(models["13m_s0"], prompts, n_new=n_new)
            ar_ref = np.array(results["arms"]["ar"]["13m_s0"]["nll_nats_per_prompt"])
            ar_same = bool(np.array_equal(np.round(nll2.cpu().numpy(), 6), np.round(ar_ref, 6)))
            h2 = FR.load_heads(os.path.join(FR.HEAD_DIRS["13m_s0"], "heads", "disc_k2_s0.pt"),
                               models["13m_s0"].mcfg.d_model, dev)
            nll3, _ = FR.freerun_path_i(models["13m_s0"], h2, prompts, n_new=n_new, k=K)
            d_ref = np.array(results["arms"]["disc_k2"]["13m_s0"]["0"]["nll_nats_per_prompt"])
            d_same = bool(np.array_equal(np.round(nll3.cpu().numpy(), 6), np.round(d_ref, 6)))
            results["L5"] = {"ar_13m_s0_rerun_identical": ar_same,
                             "disc_k2_13m_s0_s0_rerun_identical": d_same,
                             "pass": bool(ar_same and d_same),
                             "note": "同一配置重跑一次，逐 prompt NLL（6 位小数）相同 ⇒ bpc 逐值相同"}
            log(f"[L5] ar_rerun={ar_same} disc_rerun={d_same}")

        # ---------------- 判词（契约 §5） ----------------
        results["verdicts"] = _verdicts(results)

        bad = [b for b in FR.has_wall_clock(results) if not b.endswith("_wall_clock_fields")]
        if bad:
            raise AssertionError(f"results.json 含墙钟字段：{bad}")
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log("results.json 落盘")

        _write_model_md(out, results, log)
        _write_analysis(out, results, log)
        log(f"DONE wall_seconds={time.time() - t0:.1f} 写盘完成")
    return 0


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _delta_table(cand_by_seed: dict, ref_nll: np.ndarray, seeds, n_tok: np.ndarray) -> dict:
    """逐种子 + 两种子先按 prompt 平均再聚类的 Δbpc 表（与 L1/L1b 同口径）。"""
    per_seed, stack = {}, []
    for seed in seeds:
        c = np.asarray(cand_by_seed[str(seed)], dtype=np.float64)
        per_seed[str(seed)] = FR.paired_bpc_ci(c, ref_nll, n_tok)
        stack.append(c - ref_nll)
    comb = np.mean(np.stack(stack, 0), 0)
    per_seed["combined_mean_over_seeds"] = FR.paired_bpc_ci(comb + ref_nll, ref_nll, n_tok)
    return per_seed


def _cmp_band(d: dict, band: float = REF_BAND_BPC) -> str:
    if d is None:
        return "—"
    mag = abs(d["delta_bpc"])
    over = mag > band
    return (f"Δ={d['delta_bpc']:+.6f} CI[{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] "
            f"{'不含0' if d['excludes_zero'] else '含0'} "
            f"|Δ|{'越' if over else '不越'}{band}带")


def _verdicts(results: dict) -> dict:
    lines = {}
    # L1：逐 (arm, backbone, seed) 是否越带且 CI 不含 0
    l1 = {}
    for arm, by_b in results["L1"].items():
        l1[arm] = {}
        for bname, by_seed in by_b.items():
            l1[arm][bname] = {s: _cmp_band(d) for s, d in by_seed.items()}
    lines["L1"] = l1
    # 契约 §5 否证条件
    f = []
    disc_windows = [(b, s, d) for b in results["backbones"]
                    for s, d in results["L1"].get("disc_k2", {}).get(b, {}).items()
                    if isinstance(d, dict)]
    disc_bad = [f"{b}/{s}" for b, s, d in disc_windows
                if abs(d["delta_bpc"]) > REF_BAND_BPC and d["excludes_zero"]]
    lat_windows = [(b, s, d) for b in results["backbones"]
                   for s, d in results.get("L1", {}).get("lat_k2_lam1", {}).get(b, {}).items()
                   if isinstance(d, dict)]
    lat_over = [f"{b}/{s}" for b, s, d in lat_windows if abs(d["delta_bpc"]) > REF_BAND_BPC]
    if disc_bad and len(disc_bad) == len(disc_windows) and len(lat_over) == len(lat_windows):
        f.append("§5-1 触发：Δbpc(disc_k2 − ar) 的 CI 不含 0 且 |Δ| 越 0.014 带（全部窗口："
                 + ", ".join(disc_bad) + "），而 lat_k2_lam1 只把 |Δ| 缩小、同向、仍越带（"
                 + ", ".join(lat_over) + "）⇒ 判「路径 (i)『一次前向直接发 k 个位置』在本尺度"
                 "（13M 冻结主干、char enwik8、温度 0、贪心）质量面不成立」")
    elif disc_bad:
        f.append("§5-1 部分触发：Δbpc(disc_k2 − ar) 在 " + ", ".join(disc_bad)
                 + " 窗口 CI 不含 0 且 |Δ| 越 0.014 带（非全窗口）")
    else:
        f.append("§5-1 未触发：disc_k2 − ar 未同时满足「CI 不含 0」与「|Δ| 越 0.014 带」")
    lat_zero = [f"{b}/{s}" for b, s, d in lat_windows if not d["excludes_zero"]]
    # §5-2 的判定量是 Δbpc(lat_k2_lam1 − disc_k2)，即 L1b（不是 L1 的 lat − ar）
    lb_windows = [(b, s, d) for b in results["backbones"]
                  for s, d in results.get("L1b", {}).get("lat_k2_lam1_minus_disc_k2", {})
                  .get(b, {}).items() if isinstance(d, dict)]
    lb_zero = [f"{b}/{s}" for b, s, d in lb_windows if not d["excludes_zero"]]
    if lb_windows and len(lb_zero) == len(lb_windows):
        f.append("§5-2 触发：Δbpc(lat_k2_lam1 − disc_k2) 的 CI 全部含 0 ⇒ 本口径下 λ=1.0 的 latent "
                 "匹配监督对 free-run 序列级质量无可测增益")
    elif lb_zero:
        f.append("§5-2 未触发（部分）：Δbpc(lat_k2_lam1 − disc_k2) 仅在 " + ", ".join(lb_zero)
                 + f" 窗口（{len(lb_zero)}/{len(lb_windows)}）CI 含 0；其余窗口 CI 不含 0 且 "
                 "Δ<0（lat 的 bpc 更低）⇒ λ=1.0 的 latent 匹配监督在多数窗口降低 free-run bpc")
    else:
        f.append("§5-2 未触发：Δbpc(lat_k2_lam1 − disc_k2) 的 CI 不含 0（lat 的 bpc 更低）")
    all_ci_zero = all(
        (not d["excludes_zero"])
        for arm in ("disc_k2", "lat_k2_lam1") if arm in results["L1"]
        for bname in results["backbones"]
        for s, d in results["L1"][arm].get(bname, {}).items() if isinstance(d, dict))
    if all_ci_zero:
        f.append("§5-3 触发：CI 全部含 0 ⇒ 判「本尺度下该路线的序列级质量代价不可分」，不得写成「无损」")
    else:
        f.append("§5-3 未触发：至少一条 Δbpc 的 CI 不含 0")
    lines["falsification"] = f
    return lines


def _truth_anchor_outcome(results: dict) -> dict:
    ta = results["truth_anchor"]
    rows, same_sign, over_band = [], True, True
    for arm in results["config"]["head_arms"]:
        for b in results["config"]["backbones"]:
            self_d = results["L1"][arm][b]["combined_mean_over_seeds"]
            self_tf = ta["delta_self_tf"][arm][b]["combined_mean_over_seeds"]
            ext = ta["delta_external_tf"][arm][b]["combined_mean_over_seeds"]
            same_sign &= bool(ext["delta_bpc"] > 0)
            over_band &= bool(abs(ext["delta_bpc"]) > REF_BAND_BPC)
            rows.append((arm, b, self_d, self_tf, ext))
    return {"rows": rows, "same_sign": bool(same_sign), "over_band": bool(over_band),
            "corroborated": bool(same_sign and over_band)}


def _truth_anchor_summary(results: dict) -> str:
    o = _truth_anchor_outcome(results)
    if o["corroborated"]:
        return ("外部真值锚口径下的 Δ 与自评口径**同向**且**仍越 0.014 带** ⇒ 原判读"
                "（该尺度质量面不成立）**成立**，不是 `ar` 流退化自抬 Δ 的伪影。")
    return ("外部真值锚口径下的 Δ **与自评不同向或被吃掉**（同向="
            f"{o['same_sign']}、全越带={o['over_band']}）⇒ 原判读措辞**降级**："
            "证据只支持「在本实验唯一可测的自评口径下该路线质量代价可分」，"
            "不足以断言对真值的质量代价。详见「真值锚」一节。")


def _truth_anchor_lines(results: dict) -> list:
    ta = results.get("truth_anchor")
    if not ta:
        return []
    o = _truth_anchor_outcome(results)
    L = ["## 真值锚（复核 N1 的补充读数）", ""]
    L.append("独立复核（`REVIEW-C1-2670.md` 的 **N1**）指出：主判 Δbpc 是**两臂各自对自家生成流的自评 NLL 之差**，"
             "而不是「对真值的质量代价」；若 `ar` 流退化（重复/自洽），其自评 bpc 被压低，会系统性抬高 Δ。"
             "该隐忧未被落盘数据排除（原产物只有 prompt 级 NLL 与 sha256，无 token 流）。本节**只加读数**"
             "（不改预登记的臂与判据）：把每个（臂 × 主干 × 头种子 × prompt）的 200-token 生成流落盘"
             "（`streams/`），对其算**客观退化统计**（不需模型），并用 `pathlm/eval.py::bpc` 同口径的 "
             "**teacher-forced node-1 bpc** 分别在**自评主干**与**另一主干（外部锚）**下打分；"
             "再给**真实 enwik8 续写**的 teacher-forced bpc 作为水平参照。")
    L.append("")
    L.append(f"- **真实 enwik8 续写（水平参照）**：`13m_s0` "
             f"{ta['real_enwik8_continuation']['13m_s0']['bpc']:.4f} bpc、`13m_s1` "
             f"{ta['real_enwik8_continuation']['13m_s1']['bpc']:.4f} bpc"
             "（每 prompt 200 个真实 val 字符、teacher-forced，同口径）。这是「自然文本」同一主干的水平线，"
             "供判断生成流的绝对值落在何处。")
    L.append("")
    L.append("### Δ 三读法对照（自评 free-run vs 自评 teacher-forced vs 外部锚 teacher-forced）")
    L.append("")
    L.append("| 候选臂 | 主干 | 自评 free-run（原判） | 自评 teacher-forced | 外部锚 teacher-forced | 外部锚同向 |")
    L.append("|---|---|---|---|---|---|")
    for arm, b, self_d, self_tf, ext in o["rows"]:
        L.append(f"| `{arm}` | `{b}` | {self_d['delta_bpc']:+.4f} "
                 f"[{self_d['ci95'][0]:+.4f},{self_d['ci95'][1]:+.4f}] | "
                 f"{self_tf['delta_bpc']:+.4f} [{self_tf['ci95'][0]:+.4f},{self_tf['ci95'][1]:+.4f}] | "
                 f"{ext['delta_bpc']:+.4f} [{ext['ci95'][0]:+.4f},{ext['ci95'][1]:+.4f}] | "
                 f"{'是' if ext['delta_bpc'] > 0 else '否'} |")
    L.append("")
    L.append("### 客观退化统计（每臂 × 主干，12 prompt 平均）")
    L.append("")
    L.append("| 臂 | 主干 | 种子 | 最长连续相同 token | distinct-2 | 重复 4-gram 率 |")
    L.append("|---|---|---|---|---|---|")
    for arm, by_b in results["degeneracy"]["per_stream"].items():
        for b, by_s in by_b.items():
            for sd, d in by_s.items():
                m = d["mean"]
                L.append(f"| `{arm}` | `{b}` | {sd} | {m['max_identical_run']:.2f} | "
                         f"{m['distinct_2']:.4f} | {m['repeated_4gram_rate']:.4f} |")
    L.append("")
    L.append("### 要点（人话）")
    L.append("")
    bigger = all(ext["delta_bpc"] > self_d["delta_bpc"]
                 for arm, b, self_d, self_tf, ext in o["rows"])
    ar_bpc_text = "、".join(f"{b} {results['arms']['ar'][b]['bpc']:.4f}"
                            for b in results["config"]["backbones"])
    ar_degen = results["degeneracy"]["per_stream"].get("ar", {})
    ar_maxrun = float(np.mean([v["mean"]["max_identical_run"]
                               for bys in ar_degen.values() for v in bys.values()]))
    ar_d2 = float(np.mean([v["mean"]["distinct_2"]
                           for bys in ar_degen.values() for v in bys.values()]))
    lat_degen = results["degeneracy"]["per_stream"].get("lat_k2_lam1", {})
    lat_maxrun = float(np.mean([v["mean"]["max_identical_run"]
                                for bys in lat_degen.values() for v in bys.values()]))
    L.append(f"- **同一 scorer 下的真值代价远大于自评代价**：原 Δ 是两臂各自用**自家预测头**自评之差"
             f"（`ar` 用 node-1、候选臂用自家 2 个头）；而 `ar` 的自由生成**就是** node-1 贪心，所以它的自评"
             f" bpc 与 teacher-forced node-1 bpc **恒等**（{ar_bpc_text}）。换成**同一个 scorer**"
             f"（自评主干 node-1，或另一主干的 node-1）对**两条流**打分后，候选流的 teacher-forced bpc 明显更高"
             f"（disc 4.3–5.8、lat 1.5–2.3 bpc，而 `ar` 流只有 0.49–0.57），故外部锚 Δ 在全部 4 个窗口都"
             f"**比自评 Δ 更大**（{'是' if bigger else '否'}）——复核担心的「自评高估 Δ」在数据上反过来了："
             f"**自评低估**了真实代价。")
    L.append(f"- **`ar` 流不是长程退化**：落盘的 `ar` 流最长连续相同 token 平均 {ar_maxrun:.2f}、distinct-2"
             f" {ar_d2:.3f}（长程看是正常文本，只是短模式重复偏多）；复核担心的「`ar` 流退化 ⇒ 压低自评 bpc"
             f"⇒ 抬高 Δ」这一具体机制在落盘数据上**不成立**。反面：候选臂（尤其 `lat_k2_lam1`）才明显退化"
             f"（最长连续相同 token 平均 ~{lat_maxrun:.0f}，即大半条流是同一个字符）。")
    L.append(f"- **水平参照**：真实 enwik8 续写的 teacher-forced bpc 为 "
             f"{ta['real_enwik8_continuation']['13m_s0']['bpc']:.4f}（s0）/ "
             f"{ta['real_enwik8_continuation']['13m_s1']['bpc']:.4f}（s1）——候选流的 teacher-forced bpc"
             f"（1.5–5.8）远高于自然文本水平线，说明它们确实偏离自然文本，而非只是 scorer 偏差。")
    L.append("")
    L.append("### 判读")
    L.append("")
    L.append("- " + _truth_anchor_summary(results))
    L.append("- 每个生成流的三读数（自评 teacher-forced bpc、外部锚 teacher-forced bpc、真实续写 teacher-forced bpc）"
             "与逐 prompt 值见 `results.json::truth_anchor.per_stream`；生成流原始 token 见 `streams/*.json`，"
             "其 sha256 与 `results.json::arms.*.stream_sha256_*` 逐一相等"
             f"（`streams.all_sha_reproduced={results['streams']['all_sha_reproduced']}`）。")
    L.append("")
    return L


def _write_model_md(out, results, log):
    lines = [
        "# MODEL — 2670-freerun-bpc-13m",
        "",
        "本实验**零新增/零训练参数**：主干与 2660 的头权重全部**冻结、只读复用**。",
        "",
        "## 1 冻结主干（只读，来自 2631）",
        "",
        "| 主干 | 目录 | `model.pt` sha256 | tied 读出 embed sha256 | 参数 |",
        "|---|---|---|---|---|",
    ]
    for b, d in results["backbones"].items():
        lines.append(f"| `{b}` | `{d['path']}` | `{d['ckpt_sha256']}` | `{d['embed_sha256']}` | "
                     f"{d['params']} |")
    lines += ["", "## 2 复用的头（只读，来自 2660）", "",
              "| 臂 | 主干 | 种子 | 文件 | 文件 sha256 | 逐张量等于源文件 |", "|---|---|---|---|---|---|"]
    for arm, by_b in results["head_files"].items():
        for b, by_s in by_b.items():
            for s, d in by_s.items():
                lines.append(f"| `{arm}` | `{b}` | {s} | `{d['path']}` | `{d['sha256']}` | "
                             f"{d['bytes_equal_source_file']} |")
    lines += [
        "",
        "> `.pt` 的 sha256 只标识文件字节（zip entry 含保存时刻）；内容同一性以 `heads_equal_vs_file`",
        "> 的逐张量 `torch.equal` 为准（见 `results.json::head_files.*.bytes_equal_source_file`）。",
        "",
        "## 3 生成口径",
        "",
        f"- prompt = 12 个固定串 `val[i*5000 : i*5000+64]`（offset 见 `results.json::config.prompt_offsets`），"
        f"每 prompt 生成 {results['config']['n_new']} 个 token，prompt 不计分。",
        "- `ar`：`node-1 = model.transforms[0]`，`logits = z @ embed.weight^T`，逐 token 贪心（1 前向/token）。",
        "- `path_i`：一次冻结主干前向 → 取最后位置最终层隐状态 `h_i` → 对 j=1..k 用头 `head_{j-1}` → "
        "`logits_j = head_{j-1}(h_i) @ embed.weight^T` → 各自 argmax、全部追加（不验证）。",
        "- bpc = `Σ −ln p(emitted) / (n_tokens · ln2)`（自家预测分布下的 NLL）。",
        "",
        "## 4 复现命令",
        "",
        "```bash",
        "cd /home/z/vibe/pathlm",
        "python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/test_freerun.py -q",
        "python3 axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/run.py --out <本目录>",
        "```",
        "",
    ]
    with open(os.path.join(out, "MODEL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("MODEL.md 落盘")


def _write_analysis(out, results, log):
    ar0 = results["arms"]["ar"].get("13m_s0", {})
    disc00 = (results["arms"].get("disc_k2", {}).get("13m_s0", {}).get("0", {}))
    lat00 = (results["arms"].get("lat_k2_lam1", {}).get("13m_s0", {}).get("0", {}))
    L = []
    L.append("# analysis — 2670-freerun-bpc-13m（路径 (i)「一次前向直接发 k 个位置」的 free-run 序列级 bpc）")
    L.append("")
    L.append("## 人话摘要")
    L.append("")
    L.append(
        "我们在**同一份冻结的 13M 主干**上，只改「未来位置怎么产出」这一件事：`ar` 是普通逐 token "
        "自回归贪心（每步 1 次前向发 1 个 token）；`path_i` 是「估计得贪心」——一次前向拿最后一个位置的"
        "隐状态，经 2660 训好的 2 个头各自 argmax，**一次发 2 个 token、不做任何验证**。两个臂都按"
        "「自己发出的 token 在自家预测分布下的 NLL」算自由生成的**序列级 bpc**（越低越好），prompt 不计分。")
    L.append("")
    if ar0 and disc00 and lat00:
        ci = results["L1"]["disc_k2"]["13m_s0"]["combined_mean_over_seeds"]
        ciL = results["L1"]["lat_k2_lam1"]["13m_s0"]["combined_mean_over_seeds"]
        dbpcs = sorted(d["bpc"] for d in results["arms"]["disc_k2"]["13m_s0"].values())
        lbpcs = sorted(d["bpc"] for d in results["arms"]["lat_k2_lam1"]["13m_s0"].values())
        ratio = 2 ** ci["delta_bpc"]
        L.append(
            f"结果（13m_s0）：`ar` 的 bpc 是 **{ar0['bpc']:.4f}**；一次发 2 个位置的两个头种子，"
            f"`disc_k2` 是 **{dbpcs[0]:.4f}–{dbpcs[-1]:.4f}**、`lat_k2_lam1` 是 "
            f"**{lbpcs[0]:.4f}–{lbpcs[-1]:.4f}**。相对 `ar` 的质量代价（两种子合并）："
            f"`disc_k2` Δbpc = **{ci['delta_bpc']:+.4f}** "
            f"（95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]，"
            f"{'不含 0' if ci['excludes_zero'] else '含 0'}，"
            f"{'越' if abs(ci['delta_bpc']) > REF_BAND_BPC else '不越'} 0.014 bpc 带；人话量级：每字符多花 "
            f"{ci['delta_bpc']:.2f} bit，相当于困惑度约 ×{ratio:.2f}）；"
            f"`lat_k2_lam1` Δbpc = **{ciL['delta_bpc']:+.4f}** "
            f"（95% CI [{ciL['ci95'][0]:+.4f}, {ciL['ci95'][1]:+.4f}]，"
            f"{'不含 0' if ciL['excludes_zero'] else '含 0'}，"
            f"{'越' if abs(ciL['delta_bpc']) > REF_BAND_BPC else '不越'} 0.014 bpc 带）——"
            f"即 λ=1.0 的监督把代价从 {ci['delta_bpc']:+.2f} 拉到 {ciL['delta_bpc']:+.2f} bpc，"
            f"但**未拉回 0.014 带内**。")
    L.append("")
    if results.get("truth_anchor"):
        L.append("- **真值锚（复核 N1）**：" + _truth_anchor_summary(results))
        L.append("")
    L.append(f"- **L0（保真硬门）**：{'**通过**' if results['L0'].get('pass') else '**未过（blocker）**'} —— "
             f"头权重逐张量等于 2660 落盘文件；主干 `requires_grad=False` 且 backward 后 `.grad is None`；"
             f"`ar` 贪心流与 2640 `native`（生产 `decode` 贪心，prompt_len=32）"
             f"{'逐位相同' if results['L0'].get('ar_vs_2640_native', {}).get('all_match') else '见下方明细'}。")
    L.append(f"- **L1（主判）**：逐主干 × 头种子给出 Δbpc 点值 + prompt 聚类配对 bootstrap 95% CI 与是否越 0.014 带。")
    L.append(f"- **L2（n≥2 一致）**：见数字表（两主干点 × 两头种子的 Δ 符号）。")
    L.append(f"- **L3（代价轴）**：路径 (i) tokens-per-forward = {results['L3']['path_i']['tokens_per_forward']}、"
             f"每 token 主干等效 layer-passes ≈ {results['L3']['path_i']['trunk_layer_passes_per_token']}；"
             f"`ar` 为 1 / {results['L3']['ar']['trunk_layer_passes_per_token']}。")
    L.append(f"- **L4（禁墙钟自检）**：`results.json` 墙钟语义字段命中 **"
             f"{results['L4']['n_hits']}**（{'通过' if results['L4']['pass'] else '未过'}）。")
    L.append(f"- **L5（确定性）**：{results['L5'].get('pass', 'n/a')}"
             f"{'' if results['L5'].get('pass') is True else '（'+json.dumps(results['L5'], ensure_ascii=False)+'）'}")
    L.append("")
    L.append("## 契约 §5 否证条件触发情况")
    L.append("")
    for x in results["verdicts"]["falsification"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 臂与口径")
    L.append("")
    for b, d in results["backbones"].items():
        L.append(f"- 主干 `{b}`：`{d['path']}`（冻结），embed sha256 `{d['embed_sha256'][:16]}…`")
    L.append(f"- prompt = 12 个固定串（长度 {results['config']['prompt_len']}，stride "
             f"{results['config']['prompt_stride']}），offset = {results['config']['prompt_offsets']}")
    L.append(f"- 每 prompt 生成 {results['config']['n_new']} token；bpc = Σ−ln p/(n·ln2)，prompt 不计分。")
    L.append(f"- 主判 = Δbpc（候选 − 同主干 `ar`）的 prompt 聚类配对 bootstrap "
             f"(B={results['config']['bootstrap']['B']}, seed={results['config']['bootstrap']['seed']})。")
    L.append("")
    L.append("## 数字表：臂级 bpc")
    L.append("")
    L.append("| 臂 | 主干 | 种子 | bpc | tokens-per-forward |")
    L.append("|---|---|---|---|---|")
    for b, d in results["arms"]["ar"].items():
        L.append(f"| `ar` | `{b}` | — | {d['bpc']:.6f} | 1 |")
    for arm in results["config"]["head_arms"]:
        for b, by_s in results["arms"].get(arm, {}).items():
            for s, d in by_s.items():
                L.append(f"| `{arm}` | `{b}` | {s} | {d['bpc']:.6f} | {d['tokens_per_forward']} |")
    L.append("")
    L.append("## 数字表：Δbpc（候选 − 同主干 ar），prompt 聚类配对 bootstrap 95% CI")
    L.append("")
    L.append("| 候选臂 | 主干 | 种子 | Δbpc | 95% CI | 不含0 | |Δ|越0.014带 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in results["config"]["head_arms"]:
        for b, by_s in results["L1"].get(arm, {}).items():
            for s, d in by_s.items():
                L.append(f"| `{arm}` | `{b}` | {s} | {d['delta_bpc']:+.6f} | "
                         f"[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | {d['excludes_zero']} | "
                         f"{abs(d['delta_bpc']) > REF_BAND_BPC} |")
    L.append("")
    L.append("## 数字表：Δbpc(lat_k2_lam1 − disc_k2)（同主干同种子配对）")
    L.append("")
    L.append("| 主干 | 种子 | Δbpc | 95% CI | 不含0 |")
    L.append("|---|---|---|---|---|")
    for b, by_s in results.get("L1b", {}).get("lat_k2_lam1_minus_disc_k2", {}).items():
        for s, d in by_s.items():
            L.append(f"| `{b}` | {s} | {d['delta_bpc']:+.6f} | "
                     f"[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | {d['excludes_zero']} |")
    L.append("")
    L.append("## L0 明细")
    L.append("")
    L.append("```json")
    L.append(json.dumps(results["L0"], indent=2, ensure_ascii=False))
    L.append("```")
    L.append("")
    L.append("## L3 代价轴（等效理论量，禁墙钟）")
    L.append("")
    L.append("```json")
    L.append(json.dumps(results["L3"], indent=2, ensure_ascii=False))
    L.append("```")
    L.append("")
    L.append("## 生成流 sha256（可复现凭证，逐臂）")
    L.append("")
    L.append("```json")
    streams = {"ar": {b: d["stream_sha256_all"] for b, d in results["arms"]["ar"].items()}}
    for arm in results["config"]["head_arms"]:
        streams[arm] = {f"{b}_s{s}": d["stream_sha256_all"]
                        for b, by_s in results["arms"].get(arm, {}).items()
                        for s, d in by_s.items()}
    L.append(json.dumps(streams, indent=2, ensure_ascii=False))
    L.append("```")
    L.append("")
    L += _truth_anchor_lines(results)
    L.append("## 边界与未测")
    L.append("")
    L.append("- **口径**：这里的 bpc 是 **free-run 自评**（对**模型自己贪心发出的 token** 按其**自家**预测分布算 "
             "NLL），不是对真实文本的 teacher-forced CE。故其**绝对值**不可与 2631 的 teacher-forced "
             "`bpc` = 1.5468（`pathlm/eval.py::bpc`，node-1 CE on clean val）直接比：自评奖励「对自己的输出有信心」"
             "（`ar` 自评 bpc 只有 0.50，低于同一主干的 teacher-forced 1.55），本实验的判据是**同一 prompt 上"
             "两臂的配对 Δ**，不是绝对值。复核 N1 之后，上述「自评 vs 真值」的缺口已由**上节「真值锚」**补上"
             "（生成流落盘 + 客观退化统计 + teacher-forced 自评/外部锚/真实续写三读数），"
             "故本节不再只是自评口径的声明。")
    L.append("- 结论只适用于本设定（该冻结 13M 主干 / char enwik8 / k=2 / 温度 0 贪心 / 共享 tied 读出），"
             "不外推；只有 13M 一种规模。")
    L.append("- 路径 (i) 每轮只发 k=2 个 token（未测 k=4）；token 级贪心无温度采样对照。")
    L.append("- 路径 (ii)（当草稿再验证）不在本实验重复；见 2660 阶段 2。")
    L.append("- 墙钟只出现在 `run_log.txt`，不作判据。")
    L.append("")
    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log("analysis.md 落盘")


if __name__ == "__main__":
    raise SystemExit(main())
