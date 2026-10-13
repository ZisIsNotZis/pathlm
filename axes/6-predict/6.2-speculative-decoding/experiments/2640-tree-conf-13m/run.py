"""WP 6.2.2 · 评测驱动：概率多候选 / 树 + 路径概率加权组合（10 臂 × 3 checkpoint）。

**禁墙钟判据**：本驱动**不测墙钟**，`results.json` 里没有任何墙钟字段（单测
`test_results_json_has_no_wall_clock_fields` 是它的机器可核标记）。所有指标都是解析量：
α₁ / α₂|₁ / τ / tok-fwd / `R = B_verified·(1+γ)/(1+τ)`。

跑法：

    python3 run.py --out . --ckpts <ckpt1> <ckpt2> ...          # 全臂
    python3 run.py --smoke --out /tmp/wp622-smoke               # 冒烟（2 prompt × 32 token）

GPU 按 `SCHEMA.md` §20 取锁（`/tmp/pathlm_gpu.lock`，mkdir 法 + holder_pid/heartbeat +
`trap` 释放，最多等 10 分钟；等不到就上报，不杀别人的进程）。**锁粒度 = 每 checkpoint**：
每个 checkpoint 取锁 → 跑完 → 释放 → 留 ≥60 秒窗口再取下一把（见 `README.md` §5 记账）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_tree as T  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.decode import decode, decode_spec  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

PROMPT_LEN = 32
STRIDE = 5000
N_PROMPTS = 12
N_NEW = 200
BOOT = 2000
BOOT_SEED = 2640
LOCK_PATH = "/tmp/pathlm_gpu.lock"
LOCK_TIMEOUT_S = 600
LOCK_POLL_S = 15
#: 2632 的独立探针（部署口径 recall@k），用于 C1 交叉印证
PROBE_JSON = os.path.join(HERE, "..", "2632-alpha1-attribution", "probes_output.json")


# ------------------------------------------------------------------ 载入与 prompt

def load_ckpt(root: str) -> PathLM:
    """照抄 2631 `run.py::load_ckpt`：配置从 `results.json::config` 还原，`vocab_size=206`。"""
    with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = PathLM(ModelConfig(**cfg["model"]), PathConfig(**cfg["path"]), vocab_size=206)
    sd = torch.load(os.path.join(root, "model.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m.eval().cuda()


def enwik8_val():
    """enwik8 val 段（与 2631 / 2632 同一份数据与同一 offset 口径）。"""
    return np.load(os.path.join(T.ROOT, "data", "enwik8_full.npz"))["val"]


def make_prompts(val, n_prompts: int, stride: int = STRIDE) -> list[torch.Tensor]:
    return [torch.tensor(val[i * stride:i * stride + PROMPT_LEN].astype(np.int64))
            for i in range(n_prompts)]


# ------------------------------------------------------------------ GPU 锁（SCHEMA §20）

class GpuLock:
    """mkdir 法互斥锁；陈旧锁自愈（holder_pid 已死 / 无 holder_pid 且锁龄 >15min）。"""

    def __init__(self, path: str = LOCK_PATH, timeout_s: int = LOCK_TIMEOUT_S,
                 poll_s: int = LOCK_POLL_S, log=None):
        self.path, self.timeout_s, self.poll_s = path, timeout_s, poll_s
        self.log = log or (lambda m: None)
        self.events: list[dict] = []
        self._held = False

    def _stale_reason(self) -> str | None:
        try:
            with open(os.path.join(self.path, "holder_pid"), encoding="utf-8") as f:
                hp = f.read().strip()
        except OSError:
            hp = ""
        if hp:
            try:
                os.kill(int(hp), 0)
            except (OSError, ValueError):
                return f"holder_pid {hp} 已死"
            return None
        try:
            age = time.time() - os.stat(self.path).st_mtime
        except OSError:
            return None
        return "无 holder_pid 且锁龄 >15min" if age > 900 else None

    def acquire(self) -> dict:
        t0 = time.time()
        n_poll = 0
        while True:
            try:
                os.mkdir(self.path)
                break
            except FileExistsError:
                reason = self._stale_reason()
                if reason:
                    self.log(f"[lock] reclaim: {reason}")
                    for f in ("holder_pid", "heartbeat"):
                        try:
                            os.remove(os.path.join(self.path, f))
                        except OSError:
                            pass
                    try:
                        os.rmdir(self.path)
                    except OSError:
                        pass
                    continue
                if time.time() - t0 > self.timeout_s:
                    raise TimeoutError(f"等锁超过 {self.timeout_s}s（不杀别人的进程）")
                n_poll += 1
                time.sleep(self.poll_s)
        with open(os.path.join(self.path, "holder_pid"), "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        self.heartbeat()
        self._held = True
        ev = {"event": "acquire", "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "waited_s": round(time.time() - t0, 1), "n_poll": n_poll}
        self.events.append(ev)
        self.log(f"[lock] acquired after {ev['waited_s']}s")
        return ev

    def heartbeat(self) -> None:
        if self._held:
            with open(os.path.join(self.path, "heartbeat"), "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def release(self, note: str = "") -> None:
        if not self._held:
            return
        for f in ("holder_pid", "heartbeat"):
            try:
                os.remove(os.path.join(self.path, f))
            except OSError:
                pass
        try:
            os.rmdir(self.path)
        except OSError:
            pass
        self._held = False
        ev = {"event": "release", "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note}
        self.events.append(ev)
        self.log(f"[lock] released {note}")


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


# ------------------------------------------------------------------ 统计（按 prompt 聚类）

def _ci(vals: list[float]) -> list:
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 6),
            round(float(np.percentile(vals, 97.5)), 6)]


def _resample(stat, groups: list, B: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(groups)
    return [stat([groups[i] for i in rng.integers(0, n, n)]) for _ in range(B)]


def cluster_ci(groups: list, stat, B: int = BOOT, seed: int = BOOT_SEED) -> list:
    if not groups:
        return [None, None]
    return _ci(_resample(stat, groups, B, seed))


def paired_diff(groups_a: list, groups_b: list, stat, B: int = BOOT,
                seed: int = BOOT_SEED) -> dict:
    """**配对**聚类 bootstrap：同一 prompt 重采样索引下 `stat(A) - stat(B)` 的 95% CI。"""
    if not groups_a or len(groups_a) != len(groups_b):
        return {"point": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    n = len(groups_a)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(stat([groups_a[i] for i in idx]) - stat([groups_b[i] for i in idx]))
    return {"point": round(stat(groups_a) - stat(groups_b), 6), "ci": _ci(vals),
            "n_prompts": n, "bootstrap_B": B, "seed": seed}


def stat_alpha1(groups: list) -> float:
    tot = sum(len(g) for g in groups)
    return sum(1 for g in groups for a in g if a >= 1) / max(tot, 1)


def stat_alpha2(groups: list) -> float:
    den = sum(1 for g in groups for a in g if a >= 1)
    return (sum(1 for g in groups for a in g if a >= 2) / den) if den else 0.0


def stat_tau(groups: list) -> float:
    tot = sum(len(g) for g in groups)
    return sum(a for g in groups for a in g) / max(tot, 1)


def stat_tpf(groups: list) -> float:
    nr = sum(len(g) for g in groups)
    return (nr + sum(a for g in groups for a in g)) / max(nr, 1)


# ------------------------------------------------------------------ 单臂评测

def arm_pass(model, prompts, n_new, pcap, arm, native_streams, log) -> list[dict]:
    """一个臂在一个 checkpoint 上跑完全部 prompt。"""
    rows = []
    for pi, prompt in enumerate(prompts):
        act: list[dict] = []
        if arm == T.NATIVE:
            gen = native_streams[pi][:n_new]
            st = {"accepts": None, "n_forward": n_new, "n_rounds": None,
                  "candidate_sizes": None, "mean_candidates_per_round": 1.0}
        elif arm in T.SINGLE_ARMS:
            gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=T.GAMMA,
                                  draft_fn=T.make_draft_fn(arm, log=act))
            st["n_rounds"] = len(st["accepts"])
            st["candidate_sizes"] = None
            st["mean_candidates_per_round"] = 1.0
        else:
            gen, st = T.decode_tree(model, prompt, n_new, pcap, arm,
                                    oracle_stream=(native_streams[pi]
                                                   if arm == T.ORACLE_ARM else None),
                                    activity=act)
        ref = native_streams[pi][:n_new]
        first_mismatch = next((j for j, (a, b) in enumerate(zip(gen, ref)) if a != b),
                              None) if gen != ref else None
        rows.append({"prompt": pi, "gen_head": gen[:8], "accepts": st["accepts"],
                     "n_rounds": st["n_rounds"], "n_forward": st["n_forward"],
                     "candidate_sizes": st["candidate_sizes"],
                     "mean_candidates_per_round": st["mean_candidates_per_round"],
                     "n_oracle_v1_mismatch": st.get("n_oracle_v1_mismatch", 0),
                     "identical_to_native": gen == ref, "first_mismatch": first_mismatch,
                     "activity": act})
        log(f"[arm] {arm:15s} prompt {pi:2d} n_rounds={st['n_rounds']} "
            f"identical={gen == ref}")
    return rows


def _activity_stats(rows: list[dict]) -> dict:
    recs = [r for row in rows for r in row["activity"]]
    n = len(recs)
    if not n:
        return {"n_rounds_logged": 0}
    return {
        "n_rounds_logged": n,
        "pw_divergence_rate_vs_fixed": round(
            sum(1 for r in recs if r["div_pw_vs_fixed"]) / n, 6),
        "pw_divergence_rate_vs_chain": round(
            sum(1 for r in recs if r["div_pw_vs_chain"]) / n, 6),
        "pw_divergence_rate_vs_direct": round(
            sum(1 for r in recs if r["div_pw_vs_direct"]) / n, 6),
        "nocancel_divergence_rate_vs_pw": round(
            sum(1 for r in recs if r["draft_pw_nocancel"] != r["draft_pw"]) / n, 6),
        "mean_w_A": round(statistics.fmean(r["w_A"] for r in recs), 6),
        "mean_w_B": round(statistics.fmean(r["w_B"] for r in recs), 6),
        "mean_w_A_share": round(statistics.fmean(
            r["w_A"] / (r["w_A"] + r["w_B"]) if (r["w_A"] + r["w_B"]) > 0 else 0.5
            for r in recs), 6),
        "n_pw_fallback": sum(1 for r in recs if r["pw_fallback"]),
    }


def summarize_arm(rows: list[dict], n_new: int, gamma: int) -> dict:
    out = {
        "n_prompts": len(rows),
        "stream_identical_to_native": all(r["identical_to_native"] for r in rows),
        "n_prompts_stream_checked": len(rows),
        "first_mismatches": [{"prompt": r["prompt"], "pos": r["first_mismatch"]}
                             for r in rows if not r["identical_to_native"]],
        "mean_candidates_per_round": (
            round(statistics.fmean(r["mean_candidates_per_round"] for r in rows), 6)
            if all(r["mean_candidates_per_round"] is not None for r in rows) else None),
        "n_oracle_v1_mismatch": sum(r["n_oracle_v1_mismatch"] for r in rows),
        "n_forward_total": sum(r["n_forward"] for r in rows),
        "activity": _activity_stats(rows),
        "per_prompt": [{"prompt": r["prompt"], "accepts": r["accepts"],
                        "n_rounds": r["n_rounds"], "gen_head": r["gen_head"],
                        "identical_to_native": r["identical_to_native"],
                        "first_mismatch": r["first_mismatch"]} for r in rows],
    }
    if rows[0]["accepts"] is None:                    # native 对照
        out.update({"is_control": True, "alpha1": None, "alpha2_given_alpha1": None,
                    "tau": None, "tokens_per_forward": 1.0, "n_rounds": n_new,
                    "tokens_per_forward_ci": [1.0, 1.0], "alpha1_ci": [None, None],
                    "tau_ci": [None, None], "R_equivalent_flops": 1.0,
                    "R_note": "native 逐 token：每 token 1 次前向 ⇒ R ≡ 1（定义）"})
        return out
    groups = [r["accepts"] for r in rows]
    n_rounds = sum(len(g) for g in groups)
    tau = stat_tau(groups)
    mean_k = out["mean_candidates_per_round"]
    out.update({
        "is_control": False,
        "n_rounds": n_rounds,
        "accepted_total": int(sum(a for g in groups for a in g)),
        "alpha1": round(stat_alpha1(groups), 6),
        "alpha1_ci": cluster_ci(groups, stat_alpha1),
        "alpha2_given_alpha1": round(stat_alpha2(groups), 6),
        "alpha2_given_alpha1_ci": cluster_ci(groups, stat_alpha2),
        "tau": round(tau, 6),
        "tau_ci": cluster_ci(groups, stat_tau),
        "tokens_per_forward": round(stat_tpf(groups), 6),
        "tokens_per_forward_ci": cluster_ci(groups, stat_tpf),
        # 每 token 等效 FLOPs：R = B_verified·(1+γ)/(1+τ)（B_verified = 每轮送进前向的行数）
        "R_equivalent_flops": round((mean_k or 1.0) * (1 + gamma) / (1 + tau), 4),
        "R_B_verified_mean": mean_k,
        "R_note": "R>1 是范式定义性质（用 FLOPs 换顺序前向次数），不作判据",
    })
    return out


# ------------------------------------------------------------------ 判据（机械化）

def paired_dir(paired: dict, a: str, b: str, tag: str) -> dict | None:
    """配对差 `stat(A) − stat(B)`：`paired` 里只存无向键（写入顺序），这里按方向取符号。"""
    key = f"{a}_minus_{b}"
    if key in paired:
        return paired[key].get(tag)
    rkey = f"{b}_minus_{a}"
    if rkey in paired and tag in paired[rkey]:
        d = paired[rkey][tag]
        lo, hi = d["ci"]
        return {"point": None if d["point"] is None else round(-d["point"], 6),
                "ci": [None if hi is None else round(-hi, 6),
                       None if lo is None else round(-lo, 6)],
                "derived_from": f"-({rkey})"}
    return None


def wilson_ci(p: float, n: int, z: float = 1.96) -> list:
    """探针比例的二项 Wilson 95% CI（与探针同口径的 n）。"""
    if not n:
        return [None, None]
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def evaluate_criteria(arm_sum: dict, paired: dict, probe: dict | None,
                      ckpt_name: str) -> dict:
    singles = [a for a in T.SINGLE_ARMS if arm_sum.get(a, {}).get("alpha1") is not None]
    best = max(singles, key=lambda a: arm_sum[a]["alpha1"]) if singles else None
    out = {"strongest_single_arm": best, "probe_recall_at_2": probe,
           "delta_sortable": T.DELTA_SORTABLE}
    if best is None:
        return out
    tree_arms = list(T.TREE_ARMS)
    out["P1_alpha1_lift"] = {
        a: {"delta_vs_best_single": round(arm_sum[a]["alpha1"] - arm_sum[best]["alpha1"], 6),
            "paired": paired_dir(paired, a, best, "_alpha1"),
            "pass": (arm_sum[a]["alpha1"] - arm_sum[best]["alpha1"] > T.DELTA_SORTABLE
                     and ((paired_dir(paired, a, best, "_alpha1") or {}).get("ci")
                          or [None, None])[0] is not None
                     and paired_dir(paired, a, best, "_alpha1")["ci"][0] > 0)}
        for a in tree_arms if arm_sum.get(a, {}).get("alpha1") is not None}
    out["P2_tau_lift"] = {
        a: {"delta_tau_vs_best_single": round(arm_sum[a]["tau"] - arm_sum[best]["tau"], 6),
            "delta_tpf_vs_best_single": round(
                arm_sum[a]["tokens_per_forward"] - arm_sum[best]["tokens_per_forward"], 6),
            "paired_tau": paired_dir(paired, a, best, "_tau"),
            "paired_tpf": paired_dir(paired, a, best, "_tpf"),
            "pass": ((arm_sum[a]["tau"] - arm_sum[best]["tau"] > T.DELTA_SORTABLE)
                     and (((paired_dir(paired, a, best, "_tau") or {}).get("ci")
                           or [None, None])[0] or 0) > 0)}
        for a in tree_arms if arm_sum.get(a, {}).get("alpha1") is not None}
    pw, cf = arm_sum.get("combo_pw", {}), arm_sum.get("combined_fixed", {})
    if pw.get("alpha1") is not None and cf.get("alpha1") is not None:
        d = paired_dir(paired, "combo_pw", "combined_fixed", "_alpha1") or {}
        out["P3_pw_vs_fixed"] = {
            "delta_alpha1": round(pw["alpha1"] - cf["alpha1"], 6),
            "paired_alpha1": d,
            "divergence_rate_pw_arm": pw.get("activity", {}).get("pw_divergence_rate_vs_fixed"),
            "divergence_rate_fixed_arm": cf.get("activity", {}).get("pw_divergence_rate_vs_fixed"),
            "pass": (pw["alpha1"] - cf["alpha1"] > T.DELTA_SORTABLE
                     and ((d.get("ci") or [None, None])[0] or 0) > 0),
            "divergence_rate_note": "分歧率 ~0 ⇒ 两臂不可区分（无信息）；>0 ⇒ 两臂真的不同，"
                                    "差别只是没超过接受率噪声带",
        }
    out["P4_lossless"] = {
        "all_arms_stream_identical": all(
            v.get("stream_identical_to_native", False) for v in arm_sum.values()),
        "mismatching_arms": [a for a, v in arm_sum.items()
                             if not v.get("stream_identical_to_native", False)],
    }
    if probe is not None and arm_sum.get("tree_direct_k2", {}).get("alpha1") is not None:
        a1 = arm_sum["tree_direct_k2"]["alpha1"]
        lo, hi = arm_sum["tree_direct_k2"]["alpha1_ci"]
        p_ci = wilson_ci(probe["recall"], probe.get("n") or 0)
        out["C1_cross_check_tree_direct_k2_vs_probe_recall2"] = {
            "measured_alpha1": a1, "measured_alpha1_ci": [lo, hi],
            "probe_recall_at_2": probe["recall"], "probe_n_rounds": probe.get("n"),
            "probe_wilson_ci95": p_ci,
            "probe_inside_ci": (lo is not None and lo <= probe["recall"] <= hi),
            "cis_overlap": (lo is not None and p_ci[1] is not None
                            and not (hi < p_ci[0] or lo > p_ci[1])),
            "measured_below_probe": (hi is not None and hi < probe["recall"]),
            "note": "measured_below_probe=True ⇒ 树实现（多行验证/接受判据/缓存回写）可疑，"
                    "必须先修再报结论；探针 4 prompt vs 本臂 12 prompt，只要求同量级",
        }
    return out


# ------------------------------------------------------------------ 主流程

def run_ckpt(ckpt: str, name: str, val, n_prompts: int, n_new: int, arms: list[str],
             log, budget_s: float, t_start: float) -> dict:
    log(f"[ckpt] {name} :: {ckpt}  arms={arms}")
    model = load_ckpt(ckpt)
    pcap = model.pcap
    with open(os.path.join(ckpt, "results.json"), encoding="utf-8") as f:
        ckres = json.load(f)
    prompts = make_prompts(val, n_prompts)
    # native 流（多出 γ 位，供 oracle 与逐位门禁用；贪心 ⇒ 前缀一致）
    native_streams = []
    for pi, prompt in enumerate(prompts):
        gen_full, _ = decode(model, prompt, n_new + T.GAMMA, pcap)
        native_streams.append(gen_full)
    ref_one, _ = decode(model, prompts[0], n_new, pcap)
    prefix_ok = native_streams[0][:n_new] == ref_one
    log(f"[{name}] native streams ready (n_new+γ={n_new + T.GAMMA}), "
        f"prefix_consistent={prefix_ok}")

    arm_sum, cycle = {}, []
    arms_not_run = []
    for arm in arms:
        if time.time() - t_start > budget_s:
            arms_not_run = arms[arms.index(arm):]
            log(f"[budget] 时间盒用尽 ⇒ 未跑臂: {arms_not_run}")
            break
        rows = arm_pass(model, prompts, n_new, pcap, arm, native_streams, log)
        arm_sum[arm] = summarize_arm(rows, n_new, T.GAMMA)
        for row in rows:
            for j, rec in enumerate(row["activity"]):
                cycle.append({"ckpt": name, "arm": arm, "prompt": row["prompt"],
                              "round": j, **rec})
        log(f"[{name}] {arm:15s} a1={arm_sum[arm]['alpha1']} "
            f"tau={arm_sum[arm]['tau']} tpf={arm_sum[arm]['tokens_per_forward']} "
            f"R={arm_sum[arm]['R_equivalent_flops']} "
            f"identical={arm_sum[arm]['stream_identical_to_native']}")

    groups = {a: [r["accepts"] for r in arm_sum[a]["per_prompt"]]
              for a in arm_sum if not arm_sum[a].get("is_control")}
    paired = {}
    for i, a in enumerate(groups):
        for b in list(groups)[i + 1:]:
            for tag, stat in (("_alpha1", stat_alpha1), ("_tau", stat_tau), ("_tpf", stat_tpf)):
                key = f"{a}_minus_{b}"
                paired.setdefault(key, {})[tag] = paired_diff(groups[a], groups[b], stat)
    probe = probe_recall(name)
    crit = evaluate_criteria(arm_sum, paired, probe, name)
    res = {"path": ckpt, "params": ckres.get("params"), "bpc": ckres.get("bpc"),
           "config": ckres.get("config"), "resolved": ckres.get("resolved"),
           "n_prompts": n_prompts, "n_new": n_new,
           "engine": {"n_mtp": pcap.n_mtp, "chain_mode": pcap.chain_mode,
                      "window": pcap.window, "anchors": pcap.anchors,
                      "transport": pcap.transport},
           "native_is_plain_decoder_path": True,
           "native_prefix_consistent": prefix_ok,
           "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
           if torch.cuda.is_available() else None,
           "arms": arm_sum, "paired_diffs": paired, "criteria": crit,
           "arms_not_run": arms_not_run}
    del model
    torch.cuda.empty_cache()
    return res, cycle


def probe_recall(ckpt_name: str) -> dict | None:
    """2632 探针的部署口径 recall@2（direct，模型自生成前缀）+ 其轮数 n。"""
    try:
        with open(os.path.abspath(PROBE_JSON), encoding="utf-8") as f:
            blob = json.load(f)[ckpt_name]
        return {"recall": float(blob["P3_free_gen_topk_vs_T2"]["2"]["direct"]),
                "n": int(blob["P3_free_gen_overall"]["n"]),
                "source": "2632-alpha1-attribution/probes_output.json::"
                          "P3_free_gen_topk_vs_T2['2']['direct'] (4 prompt)"}
    except Exception:
        return None


def parse_arm_plan(specs: list[str] | None) -> dict:
    out = {}
    for s in specs or []:
        name, arms = s.split("=", 1)
        out[name] = [a.strip() for a in arms.split(",") if a.strip()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--n-new", type=int, default=N_NEW)
    ap.add_argument("--arms", nargs="+", default=list(T.ALL_ARMS))
    ap.add_argument("--arm-plan", nargs="*", default=[],
                    help="每 checkpoint 的臂集合，如 '53m_s0=native,direct1,tree_direct_k2'")
    ap.add_argument("--budget-min", type=float, default=30.0,
                    help="整程时间盒（分钟），超时就落盘已有读数并记未跑臂")
    ap.add_argument("--lock", choices=("ckpt", "run", "none"), default="ckpt")
    args = ap.parse_args()

    if args.smoke:
        args.prompts, args.n_new = 2, 32
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "run_log.txt")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    budget_s = args.budget_min * 60
    plan = parse_arm_plan(args.arm_plan)

    ckpt_dirs: list[str] = []
    for pat in args.ckpts:
        ckpt_dirs.extend(sorted(glob.glob(pat)))
    if not ckpt_dirs:
        log("[run] 没有匹配到 checkpoint")
        return 2
    log(f"[run] {time.strftime('%Y-%m-%dT%H:%M:%S')} ckpts={ckpt_dirs} "
        f"smoke={args.smoke} lock={args.lock} budget_min={args.budget_min}")

    lock = GpuLock(log=log)
    results = {
        "experiment": "2640-tree-conf-13m",
        "wp": "6.2.2 (多位置预测 = 投机解码)",
        "contract": "README.md（预登记：臂定义 / P1–P5 判据 / C1 交叉印证 / 否证条件）",
        "timing_measured": False,
        "timing_note": "本实验禁墙钟判据；任何计时只出现在 run_log.txt（过程日志），不进本文件",
        "config": {"prompt_len": PROMPT_LEN, "stride": STRIDE,
                   "n_prompts": args.prompts, "n_new": args.n_new, "gamma": T.GAMMA,
                   "arms": args.arms, "topk_arms": T.TOPK_ARMS,
                   "delta_sortable": T.DELTA_SORTABLE,
                   "bootstrap": {"B": BOOT, "seed": BOOT_SEED, "cluster": "prompt"},
                   "lock_granularity": args.lock,
                   "budget_min": args.budget_min,
                   "smoke": bool(args.smoke)},
        "checkpoints": {}, "lock_events": [], "unmeasured": []}

    def flush(cycle_all: list[dict]) -> None:
        results["lock_events"] = list(lock.events)
        with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.out, "cycle_log.jsonl"), "w", encoding="utf-8") as f:
            for rec in cycle_all:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cycle_all: list[dict] = []
    for ci, ck in enumerate(ckpt_dirs):
        name = os.path.basename(ck.rstrip("/"))
        arms = plan.get(name, args.arms)
        if args.lock == "run" and ci == 0:
            lock.acquire()
        if args.lock == "ckpt":
            free0 = gpu_free_mib()
            lock.acquire()
        else:
            free0 = gpu_free_mib()
        try:
            torch.cuda.reset_peak_memory_stats()
            res, cycle = run_ckpt(ck, name, enwik8_val(), args.prompts, args.n_new,
                                  arms, log, budget_s, t_start)
            res["gpu_free_mib_before"] = free0
            res["gpu_free_mib_after"] = gpu_free_mib()
            results["checkpoints"][name] = res
            cycle_all.extend(cycle)
            if res["arms_not_run"]:
                results["unmeasured"].append({"ckpt": name, "arms": res["arms_not_run"],
                                              "reason": "时间盒用尽"})
        finally:
            if args.lock == "ckpt":
                lock.release(note=f"after {name}")
            flush(cycle_all)
        if args.lock == "ckpt" and ci < len(ckpt_dirs) - 1:
            log("[lock] §20.1 让出窗口：sleep 60s")
            time.sleep(60)
    if args.lock == "run":
        lock.release(note="end of run")
    flush(cycle_all)

    # 整程机械判据汇总（结论文字在 analysis.md，这里只做可核的机械汇总）
    overall = {}
    for name, rec in results["checkpoints"].items():
        overall[name] = {"strongest_single_arm": rec["criteria"]["strongest_single_arm"],
                         "P1": {a: v["pass"] for a, v in
                                rec["criteria"].get("P1_alpha1_lift", {}).items()},
                         "P2": {a: v["pass"] for a, v in
                                rec["criteria"].get("P2_tau_lift", {}).items()},
                         "P3": rec["criteria"].get("P3_pw_vs_fixed", {}).get("pass"),
                         "P4": rec["criteria"].get("P4_lossless", {}).get(
                             "all_arms_stream_identical"),
                         "C1_probe_ci_overlap": rec["criteria"].get(
                             "C1_cross_check_tree_direct_k2_vs_probe_recall2", {}).get(
                                 "cis_overlap")}
    results["criteria_overall"] = overall
    flush(cycle_all)
    log(f"[run] wrote results.json / cycle_log.jsonl ({time.time() - t_start:.0f}s)")
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
