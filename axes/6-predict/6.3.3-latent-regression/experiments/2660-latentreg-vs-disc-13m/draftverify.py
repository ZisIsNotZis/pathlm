"""路径 (ii) 当草稿再验证（契约 §4 L2）——加载阶段 1 训好的头，用**生产解码器** `decode_spec`
做单候选投机解码，量 α₁ / τ。

接线（契约 §4 L2，单因素 = **位置 N+2 的草稿来源**）：
- **位置 N+2** 的草稿 = 该头**第一个**变换头（head0，训练时 j=1 ⇒ t_{i+2}）的 argmax：
  `argmax( head0(nodes[0]["latent"]) @ model.embed.weight.T )`，其中 `nodes = dec.last_nodes`、
  `nodes[0]["latent"]` = 主干**最终隐状态**。
- **其余草稿位（N+3..）固定** = 引擎自带 `nodes[k]["logits"].argmax()`（与 2640/2631 背景一致）。

保真硬门 `repro_direct1`：用 2640/2631 的 `direct1` 草稿定义（= 生产 `decode_spec` 的引擎默认：
`drafts = [argmax(nodes[k]["logits"]) for k in range(2, 2+max_drafts)]`），其逐轮 `accepts`
必须与 `2640-tree-conf-13m/results.json::checkpoints[13m_s0].arms.direct1.per_prompt[*].accepts`
**逐位相同**；不一致 ⇒ harness 不等价，先修再报，新读数不得当结论。

协议与 2640/2631 完全一致：enwik8 val、stride 5000、`PROMPT_LEN=32`、12 prompt × `n_new=200`、
B=1、温度 0、γ=2（`max_drafts=2`）。**禁墙钟**（`results.json` 不得有墙钟字段）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import latent_reg as L  # noqa: E402
from pathlm.decode import decode_spec  # noqa: E402
from pathlm.model import TransformHead  # noqa: E402

PROMPT_LEN = 32
STRIDE = 5000
N_PROMPTS = 12
N_NEW = 200
GAMMA = 2                      # 草稿位数（N+2, N+3）
BOOT = 2000
BOOT_SEED = 2660               # 契约 §5：B=2000、seed=2660

REF_2640 = os.path.join(L.REPO, "axes", "6-predict", "6.2-speculative-decoding",
                        "experiments", "2640-tree-conf-13m", "results.json")
REF_CKPT = "13m_s0"
REF_ARM = "direct1"
#: 2640 `direct1` 的公开读数（保真硬门期望；见 2640 results.json）
REF_ALPHA1 = 0.802243
REF_TAU = 1.453619

#: 臂定义：repro 门 + 主判两臂（disc_k2 / lat_k2_lam1，两种子）+ 可选臂
DV_ARMS = [
    {"name": "repro_direct1", "kind": "direct1"},
    {"name": "disc_k2_s0", "kind": "head", "head": "heads/disc_k2_s0.pt"},
    {"name": "disc_k2_s1", "kind": "head", "head": "heads/disc_k2_s1.pt"},
    {"name": "lat_k2_lam1_s0", "kind": "head", "head": "heads/lat_k2_lam1_s0.pt"},
    {"name": "lat_k2_lam1_s1", "kind": "head", "head": "heads/lat_k2_lam1_s1.pt"},
    {"name": "lat_k2_lam01_s0", "kind": "head", "head": "heads/lat_k2_lam01_s0.pt",
     "optional": True},
    {"name": "lat_k4_lam1_s0", "kind": "head",
     "head": "second-k4-13m_s0/heads/lat_k4_lam1_s0.pt", "optional": True},
]
CORE_ARMS = [a["name"] for a in DV_ARMS if not a.get("optional")]


def enwik8_val() -> np.ndarray:
    """enwik8 val 段（与 2640/2631/2632 同一份数据、同一 offset 口径）。"""
    return np.load(L.DATA_NPZ)["val"]


def make_prompts(val: np.ndarray, n_prompts: int = N_PROMPTS,
                 stride: int = STRIDE) -> list[torch.Tensor]:
    """与 2640 `make_prompts` 逐字一致：`val[i*stride : i*stride+32]`。"""
    return [torch.tensor(val[i * stride:i * stride + PROMPT_LEN].astype(np.int64))
            for i in range(n_prompts)]


# ------------------------------------------------------------------ 草稿源

def make_draft_fn_direct1(nodes: dict, max_drafts: int) -> list[int]:
    """2640/2631 `direct1` 的草稿定义 = 生产 `decode_spec` 的引擎默认（node-2..）。

    `drafts = [argmax(nodes[k]["logits"]) for k in range(2, 2+max_drafts)]`
    """
    return [int(nodes[k]["logits"][0, 0].argmax()) for k in range(2, 2 + max_drafts)]


def make_draft_fn_head(head, model, log: list | None = None):
    """路径 (ii)：位置 N+2 的草稿 = `argmax(head0(nodes[0]["latent"]) @ E^T)`，
    其余位 = 引擎自带 `nodes[k]["logits"].argmax()`（N+3.. 固定）。"""
    E = model.embed.weight

    def fn(nodes: dict, max_drafts: int) -> list[int]:
        h0 = nodes[0]["latent"]                    # [1,1,d] 主干最终隐状态
        with torch.no_grad():
            z = head(h0)
            d_head = int((z @ E.T)[0, -1].argmax())
        drafts = [d_head]
        for k in range(3, 2 + max_drafts):         # N+3.. 固定用引擎自带 node logits
            drafts.append(int(nodes[k]["logits"][0, 0].argmax()))
        if log is not None:
            log.append({"draft_head": d_head,
                        "draft_v1": int(nodes[1]["logits"][0, 0].argmax())})
        return drafts

    return fn


def load_head(head_path: str, d_model: int, device) -> TransformHead:
    """加载阶段 1 落盘的头（`{head0: sd, head1: sd, ...}`），取 **head0**（j=1 ⇒ t+2）。"""
    sd = torch.load(head_path, map_location="cpu", weights_only=True)
    head = TransformHead(d_model)
    head.load_state_dict(sd["head0"])
    return head.to(device).eval()


# ------------------------------------------------------------------ 跑臂 + 统计

@torch.no_grad()
def run_arm(model, prompts: list[torch.Tensor], pcap, draft_fn, n_new: int = N_NEW,
            log=None) -> list[dict]:
    """一个臂跑完全部 prompt，返回逐 prompt 的 `accepts`（逐轮接受数）。"""
    rows = []
    for pi, prompt in enumerate(prompts):
        gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=GAMMA, draft_fn=draft_fn)
        rows.append({"prompt": pi, "gen_head": gen[:8], "accepts": st["accepts"],
                     "n_rounds": len(st["accepts"])})
        if log is not None:
            log(f"[dv] prompt {pi:2d} n_rounds={len(st['accepts'])} accepted={st['drafts_accepted']}")
    return rows


def stat_alpha1(groups: list) -> float:
    tot = sum(len(g) for g in groups)
    return sum(1 for g in groups for a in g if a >= 1) / max(tot, 1)


def stat_alpha2(groups: list) -> float:
    den = sum(1 for g in groups for a in g if a >= 1)
    return (sum(1 for g in groups for a in g if a >= 2) / den) if den else 0.0


def stat_tau(groups: list) -> float:
    tot = sum(len(g) for g in groups)
    return sum(a for g in groups for a in g) / max(tot, 1)


def _ci(vals: list[float]) -> list:
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 6),
            round(float(np.percentile(vals, 97.5)), 6)]


def cluster_ci(groups: list, stat, B: int = BOOT, seed: int = BOOT_SEED) -> list:
    if not groups:
        return [None, None]
    rng = np.random.default_rng(seed)
    n = len(groups)
    return _ci([stat([groups[i] for i in rng.integers(0, n, n)]) for _ in range(B)])


def paired_diff(groups_a: list, groups_b: list, stat, B: int = BOOT,
                seed: int = BOOT_SEED) -> dict:
    """**配对**（按 prompt 聚类）bootstrap：同一重采样索引下 `stat(A) - stat(B)` 的 percentile 95% CI。"""
    if not groups_a or len(groups_a) != len(groups_b):
        return {"point": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    n = len(groups_a)
    vals = [stat([groups_a[i] for i in idx]) - stat([groups_b[i] for i in idx])
            for idx in (rng.integers(0, n, n) for _ in range(B))]
    lo, hi = _ci(vals)
    return {"point": round(stat(groups_a) - stat(groups_b), 6), "ci": [lo, hi],
            "excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
            "n_prompts": n, "bootstrap_B": B, "seed": seed, "cluster": "prompt",
            "ci_kind": "percentile95"}


def summarize(rows: list[dict]) -> dict:
    groups = [r["accepts"] for r in rows]
    return {
        "n_prompts": len(rows), "n_rounds": sum(len(g) for g in groups),
        "accepted_total": sum(a for g in groups for a in g),
        "alpha1": round(stat_alpha1(groups), 6),
        "alpha1_ci": cluster_ci(groups, stat_alpha1),
        "alpha2_given_alpha1": round(stat_alpha2(groups), 6),
        "tau": round(stat_tau(groups), 6),
        "tau_ci": cluster_ci(groups, stat_tau),
        "per_prompt": [{"prompt": r["prompt"], "accepts": r["accepts"],
                        "n_rounds": r["n_rounds"]} for r in rows],
    }


# ---------------------------------------------------------------- 保真硬门

def repro_gate(rows: list[dict], ref_path: str = REF_2640) -> dict:
    """逐位比对 `repro_direct1` 与 2640 `direct1` 的逐 prompt accepts。"""
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)["checkpoints"][REF_CKPT]["arms"][REF_ARM]["per_prompt"]
    ref_accepts = [r["accepts"] for r in ref]
    got_accepts = [r["accepts"] for r in rows]
    per_prompt = []
    all_same = len(ref_accepts) == len(got_accepts)
    for i in range(max(len(ref_accepts), len(got_accepts))):
        r = ref_accepts[i] if i < len(ref_accepts) else None
        g = got_accepts[i] if i < len(got_accepts) else None
        same = r == g
        all_same = all_same and same
        per_prompt.append({"prompt": i, "bit_identical": bool(same),
                           "ref_n_rounds": len(r) if r is not None else None,
                           "got_n_rounds": len(g) if g is not None else None,
                           "first_mismatch": (next((j for j, (a, b) in enumerate(zip(r, g))
                                                    if a != b), None)
                                              if (r is not None and g is not None and not same)
                                              else None)})
    groups = [r["accepts"] for r in rows]
    a1, tau = round(stat_alpha1(groups), 6), round(stat_tau(groups), 6)
    return {
        "ref": f"2640-tree-conf-13m/results.json::checkpoints[{REF_CKPT}].arms.{REF_ARM}",
        "ref_alpha1": REF_ALPHA1, "ref_tau": REF_TAU,
        "got_alpha1": a1, "got_tau": tau,
        "alpha1_matches": abs(a1 - REF_ALPHA1) < 1e-6, "tau_matches": abs(tau - REF_TAU) < 1e-6,
        "accepts_bit_identical": bool(all_same),
        "n_prompts_compared": len(ref_accepts),
        "per_prompt": per_prompt,
        "pass": bool(all_same and abs(a1 - REF_ALPHA1) < 1e-6 and abs(tau - REF_TAU) < 1e-6),
    }


# ------------------------------------------------------------------ 驱动

def stage(out: str, ckpt: str, log, n_prompts: int = N_PROMPTS, n_new: int = N_NEW,
          smoke: bool = False, arms: list[str] | None = None) -> dict:
    """跑路径 (ii) 全部臂 + 保真门 + 配对判据，写回 results.json 的 `draftverify` 节。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = L.load_ckpt(ckpt).to(dev)
    pcap = model.pcap
    val = enwik8_val()
    prompts = make_prompts(val, n_prompts)
    d_model = model.mcfg.d_model
    log(f"[dv] device={dev} ckpt={ckpt} embed_sha256={L.embed_sha256(model)[:16]}… "
        f"n_prompts={n_prompts} n_new={n_new} gamma={GAMMA}")

    results_path = os.path.join(out, "results.json")
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            blob = json.load(f)
    else:
        blob = {}      # /tmp 冒烟：无阶段 1 结果可比，只验证接线

    dv = {
        "contract": "README.md §4 L2 / §5（预登记）",
        "protocol": {"corpus": "enwik8 val", "stride": STRIDE, "prompt_len": PROMPT_LEN,
                     "n_prompts": n_prompts, "n_new": n_new, "gamma": GAMMA,
                     "max_drafts": GAMMA, "B": 1, "temperature": 0.0,
                     "bootstrap": {"B": BOOT, "seed": BOOT_SEED, "cluster": "prompt",
                                   "ci": "percentile95"}},
        "ckpt": os.path.relpath(ckpt, out),
        "wiring": {"N+2_draft": "argmax(head0(nodes[0]['latent']) @ embed.weight.T)",
                   "N+3_draft": "argmax(nodes[3]['logits']) (引擎自带，固定)",
                   "note": "单因素 = 位置 N+2 的草稿来源；head0 训练时 j=1 ⇒ t_{i+2}"},
        "arms": {}, "repro_gate": None, "L2": {}, "L3": {},
    }
    dv["L3"] = {
        "path_i_trunk_layer_passes_per_token": {"k=2": model.mcfg.n_layers / 2},
        "path_ii_trunk_layer_passes_per_token": {"k=2": (1 + 2) / 2 * model.mcfg.n_layers},
        "note": "契约 §4 L3：路径 (i) ≈ L/k，路径 (ii) ≈ (1+k)/k·L ≈ L；等效理论量，不作判据；禁墙钟",
    }
    dv["verdicts"] = {}

    def persist():
        blob["draftverify"] = dv
        bad = [b for b in _wall_clock_fields(blob) if not b.endswith("_wall_clock_fields")]
        if bad:
            raise AssertionError(f"draftverify 含墙钟字段：{bad}")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2, ensure_ascii=False)

    arm = {a["name"]: a for a in DV_ARMS}
    run_names = arms or ([n for n in CORE_ARMS] + [a["name"] for a in DV_ARMS
                                                   if a.get("optional")])
    done = []
    for name in run_names:
        spec = arm[name]
        if spec["kind"] == "direct1":
            draft_fn = make_draft_fn_direct1
        else:
            head = load_head(os.path.join(HERE, spec["head"]), d_model, dev)
            draft_fn = make_draft_fn_head(head, model)
        rows = run_arm(model, prompts, pcap, draft_fn, n_new=n_new, log=log)
        summary = summarize(rows)
        dv["arms"][name] = {"kind": spec["kind"], "head": spec.get("head"), **summary}
        done.append(name)
        log(f"[dv] {name:16s} alpha1={summary['alpha1']} tau={summary['tau']} "
            f"n_rounds={summary['n_rounds']}")
        if spec["kind"] == "direct1":
            dv["repro_gate"] = repro_gate(rows)
            g = dv["repro_gate"]
            log(f"[dv] repro_gate pass={g['pass']} bit_identical={g['accepts_bit_identical']} "
                f"alpha1={g['got_alpha1']}(exp {REF_ALPHA1}) tau={g['got_tau']}(exp {REF_TAU})")
            if not g["pass"]:
                dv["verdicts"]["repro_gate"] = "**FAIL**：harness 不等价，新读数不得当结论"
            else:
                dv["verdicts"]["repro_gate"] = "pass（repro_direct1 逐位复现 2640 direct1）"
        persist()   # 边做边落盘：每跑完一臂即写盘

    # 保真门未过 ⇒ 不计算判据
    gate_ok = bool(dv["repro_gate"] and dv["repro_gate"]["pass"])
    if gate_ok:
        groups = {n: [r["accepts"] for r in dv["arms"][n]["per_prompt"]] for n in done}
        for ref, cands in (("disc_k2_s0", ["lat_k2_lam1_s0", "lat_k2_lam01_s0"]),
                           ("disc_k2_s1", ["lat_k2_lam1_s1"])):
            if ref not in groups:
                continue
            for cand in cands:
                if cand not in groups:
                    continue
                dv["L2"][cand] = {
                    "ref": ref,
                    "alpha1": paired_diff(groups[cand], groups[ref], stat_alpha1),
                    "tau": paired_diff(groups[cand], groups[ref], stat_tau),
                }
        # 与 2640 direct1（主干内联合训练头）的差：参考，不作判据
        if "repro_direct1" in groups:
            for cand in ("lat_k2_lam1_s0", "disc_k2_s0"):
                if cand in groups:
                    dv["L2"][f"{cand}_vs_2640_direct1"] = {
                        "alpha1": paired_diff(groups[cand], groups["repro_direct1"], stat_alpha1),
                        "tau": paired_diff(groups[cand], groups["repro_direct1"], stat_tau),
                        "note": "训练 regime 不同（主干内联合训练头 vs 冻结特征训头），**不作判据**，只作参考",
                    }
        # L2 判词（按契约 §4 L2/L4 措辞边界）
        main = dv["L2"].get("lat_k2_lam1_s0", {})
        a1, tt = main.get("alpha1", {}), main.get("tau", {})
        adopt_a1 = bool(a1.get("excludes_zero") and a1.get("point", 0) > 0)
        adopt_tau = bool(tt.get("excludes_zero") and tt.get("point", 0) > 0)
        if adopt_a1:
            dv["verdicts"]["L2"] = (
                f"采纳（α₁ 配对 95% CI 不含 0：Δ={a1['point']:+.6f} "
                f"CI[{a1['ci'][0]:+.6f},{a1['ci'][1]:+.6f}]）")
        else:
            dv["verdicts"]["L2"] = (
                "否证（α₁ 配对 95% CI 含 0 ⇒ 在**本设定**（共享 tied 读出 + 同参数量 + "
                "该训练预算 + 单候选贪心投机）下回归臂无独立增益；不得写成『latent 回归无价值』）")
        dv["verdicts"]["L2_tau"] = (
            ("τ 同向：Δ=%+.6f CI[%+.6f,%+.6f]" % (tt["point"], tt["ci"][0], tt["ci"][1]))
            if (tt.get("point") is not None) else "τ 未算")
        log(f"[dv] L2: {dv['verdicts']['L2']}")
    else:
        dv["verdicts"]["L2"] = "未判（保真门未过）"
    dv["arms_run"] = done
    dv["arms_not_run"] = [n for n in arm if n not in done]
    persist()
    return dv


def _wall_clock_fields(obj, path="") -> list[str]:
    """与 run.py::has_wall_clock 同口径（避免循环 import）。"""
    sub = ("wall", "elapsed", "clock", "duration", "seconds", "milliseconds",
           "latency", "throughput")
    exact = {"time", "ms", "sec", "took", "runtime", "speed", "tps", "timing"}
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(w in kl for w in sub) or kl in exact:
                bad.append(f"{path}.{k}")
            bad += _wall_clock_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += _wall_clock_fields(v, f"{path}[{i}]")
    return bad
