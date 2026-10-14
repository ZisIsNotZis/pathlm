"""WP 6.2.3 · 2650-verify-gate-13m：批量验证的正确性门禁、失败模式与验证开销。

只读 import `2640-tree-conf-13m/spec_tree.py` 与 `run.py`（`T` / `H`），在其上做**薄扩展**：
新增「更激进 / 对抗性」候选集（`aggr_*` / `adv_*` / `dup_k4` / `single_k1_ctl`）与一个
**通用 K 行树验证主循环** `decode_gate`（与 `T.decode_tree` 逐行等价，只把候选集换成
可注入的 `cands_fn`）。**不修改** 2640 的任何文件，**不修改** `pathlm/**`。

契约：`axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/README.md`（预登记）。
判据 V0–V6 见该 README §4；本文件只做机械计算与落盘，结论文字在 `analysis.md`。

**禁墙钟判据**：`results.json` 无任何墙钟字段；GPU 实耗只允许出现在 `run_log.txt`。

跑法：

    python3 run.py --out . --ckpts ../2631-spec-three-arms/train/13m_s0 \
        ../2631-spec-three-arms/train/13m_s1
    python3 run.py --smoke --out /tmp/wp623-smoke          # 冒烟（结果落 /tmp）
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
#: 2640 只读资产目录（本实验的唯一变量 = 候选集，其余逐字段复用 2640 口径）
DIR2640 = os.path.abspath(os.path.join(HERE, "..", "2640-tree-conf-13m"))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for _p in (DIR2640, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_tree as T  # noqa: E402  2640 的只读实现

# 2640 的只读评测驱动（prompt / bootstrap / 锁 / load_ckpt）。按**显式路径**加载为
# 独立模块名 `run2640`：本目录也有 `run.py`（本实验入口），裸 `import run` 会撞名。
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("run2640", os.path.join(DIR2640, "run.py"))
H = _ilu.module_from_spec(_spec)
sys.modules["run2640"] = H
_spec.loader.exec_module(H)

GAMMA = T.GAMMA
N_PROMPTS = H.N_PROMPTS
N_NEW = H.N_NEW
BOOT = H.BOOT
BOOT_SEED = 2650            # 本实验的 bootstrap seed（README §5）
ADV_RANDOM_SEED = 2650      # adv_random_k4 的固定 seed（README §3）

#: 读进 sha256 记录的 2640 被 import 文件（README §5：把 sha256 记进 results.json）
IMPORTED_2640 = ("spec_tree.py", "run.py")

#: 2640 results.json（V0 复现门 / V4 饱和配对 / V5 oracle τ 上界的**指针**，不复制数字）
RESULTS_2640 = os.path.join(DIR2640, "results.json")


# ---------------------------------------------------------------- 候选集构造（本实验唯一变量 C）

def _topk(dist: torch.Tensor, k: int) -> list[int]:
    k = min(int(k), int(dist.numel()))
    return [int(i) for i in torch.topk(dist, k).indices.tolist()]


def _bottomk(dist: torch.Tensor, k: int) -> list[int]:
    k = min(int(k), int(dist.numel()))
    return [int(i) for i in torch.topk(dist, k, largest=False).indices.tolist()]


def cands_repro_topk4(nodes: dict) -> list[int]:
    """= 2640 `tree_topk4`（`pw_mix.q` top-4）——V0 复现门。"""
    return T.candidate_tokens(nodes, "tree_topk4")


def cands_q_topk(nodes: dict, k: int) -> list[int]:
    """`pw_mix.q` 的 top-k（K=1/2/4/8 同族：单候选臂也是 q 的 argmax）。"""
    return _topk(T.pw_mix(nodes)["q"], k)


def cands_combo_k1(nodes: dict) -> list[int]:
    """`combo_pw` 的草稿，1 个候选，走 K 行树机制——V3 退化等价门。"""
    return [int(T.first_slot_distribution(nodes, "combo_pw").argmax())]


def cands_dup_k4(nodes: dict) -> list[int]:
    """`combo_pw` 的草稿复制 4 份（**去重前**）——V6(a) 重复/去重门。"""
    d = int(T.first_slot_distribution(nodes, "combo_pw").argmax())
    return [d, d, d, d]


def cands_aggr_topk8(nodes: dict) -> list[int]:
    """`pw_mix.q` top-8——V4 饱和判据。"""
    return _topk(T.pw_mix(nodes)["q"], 8)


def cands_aggr_union_heads(nodes: dict) -> list[int]:
    """跨头联合：`dedup{argmax pc, argmax p2, top-2 pc, top-2 p2}`（K=3–4）。"""
    probs = T.slot_probs(nodes)
    pc, pd = probs["pc"], probs["p2"]
    cands = [int(pc.argmax()), int(pd.argmax())] + _topk(pc, 2) + _topk(pd, 2)
    return list(dict.fromkeys(cands))


def cands_adv_bottomk4(nodes: dict) -> list[int]:
    """`pw_mix.q` 概率**最低**的 4 个 token——V2 负对照（该跌）。"""
    return _bottomk(T.pw_mix(nodes)["q"], 4)


def make_cands_adv_random(vocab: int, seed: int):
    """4 个**均匀随机** token（固定 seed，与任何分布无关）——V2 负对照（该跌）。"""
    rng = torch.Generator().manual_seed(int(seed))

    def fn(_nodes: dict) -> list[int]:
        return [int(x) for x in torch.randint(0, vocab, (4,), generator=rng).tolist()]

    return fn


# ---------------------------------------------------------------- 通用 K 行树验证主循环

@torch.no_grad()
def decode_gate(model, prompt: torch.Tensor, n_new: int, pcap, cands_fn,
                window: int | None = None):
    """每轮**一次** B=K 行验证前向；候选集由 `cands_fn(nodes)` 给出。

    与 `T.decode_tree` 逐行等价（同一 `T.tree_forward` / 同一接受判据
    「候选 == 目标模型该位置 argmax」/ 同一 `T.write_back_accepted_row`），
    唯一差别 = 候选集来源可注入。`window` 非空时传给 `Decoder`（P3 带驱逐），
    此时 `T.tree_forward` 会抛 `NotImplementedError`（本实验登记为引擎需求）。
    """
    dec = T.Decoder(model, window=(pcap.window if window is None else window),
                    anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    gen: list[int] = []
    accepts: list[int] = []
    cand_sizes: list[int] = []
    n_forward = 0
    K = 1 + GAMMA
    while len(gen) < n_new:
        nodes = dec.last_nodes
        assert nodes is not None, "step must populate last_nodes"
        v1 = int(nodes[1]["logits"][0, 0].argmax())
        d3 = int(nodes[3]["logits"][0, 0].argmax())
        cands = list(cands_fn(nodes))
        assert cands and cands[0] is not None, "候选集不能为空"
        if dec.n + K > model.mcfg.seq_len:
            break
        rows = [[v1, c, d3] for c in cands]
        _h, fnodes, new_kv, p0 = T.tree_forward(model, dec, rows)
        n_forward += 1
        logs = fnodes[1]["logits"]
        # 接受判据：存在候选 == 该位置目标模型的 argmax（贪心 ⇒ 至多 1 行能匹配）
        acc_i = next((i for i, c in enumerate(cands)
                      if int(logs[i, 0].argmax()) == c), None)
        n_acc = 0
        if acc_i is not None:
            n_acc = 1
            if int(logs[acc_i, 1].argmax()) == d3:     # 只认**接受行**（真前缀）上的 d3
                n_acc = 2
        row_used = 0 if acc_i is None else acc_i
        n_pos = 1 + n_acc
        T.write_back_accepted_row(dec, new_kv, p0, row_used, n_pos)
        dec.last_nodes = T._take_row(T._slice_nodes(fnodes, n_pos - 1), row_used)
        gen.append(v1)
        if n_acc >= 1:
            gen.append(cands[acc_i])
        if n_acc >= 2:
            gen.append(d3)
        accepts.append(n_acc)
        cand_sizes.append(len(cands))
    n_rounds = len(accepts)
    return gen[:n_new], {
        "accepts": accepts,
        "n_forward": n_forward,
        "n_rounds": n_rounds,
        "candidate_sizes": cand_sizes,
        "mean_candidates_per_round": (sum(cand_sizes) / n_rounds) if n_rounds else None,
        "n_oracle_v1_mismatch": 0,
        "n_pw_fallback": 0,
        "cache_len": dec.cache_len(),
    }


# ---------------------------------------------------------------- 臂注册表

#: kind=tree → decode_gate；kind=single → 生产 decode_spec（= 2640 单候选臂口径）
ARM_KIND = {
    "repro_topk4": ("tree", cands_repro_topk4),
    "repro_combo_pw": ("single", "combo_pw"),
    "single_k1_ctl": ("tree", cands_combo_k1),
    "dup_k4": ("tree", cands_dup_k4),
    "aggr_topk8": ("tree", cands_aggr_topk8),
    "aggr_union_heads": ("tree", cands_aggr_union_heads),
    "adv_random_k4": ("tree", "adv_random"),
    "adv_bottomk4": ("tree", cands_adv_bottomk4),
    "win64_tree_k2": ("tree", lambda nodes: T.candidate_tokens(nodes, "tree_direct_k2")),
}

#: P3 可选臂的 Decoder window（带驱逐；tree_forward 期望抛 NotImplementedError）
ARM_WINDOW = {"win64_tree_k2": 64}

#: 13m_s1 子集（README §5）
SUBSET_13M_S1 = ["repro_topk4", "repro_combo_pw", "single_k1_ctl", "dup_k4",
                 "aggr_topk8", "adv_random_k4"]

#: V0 复现门：本实验臂 → 2640 results.json 对应臂
V0_MAP = {"repro_topk4": "tree_topk4", "repro_combo_pw": "combo_pw"}

#: V3 退化等价门：本实验臂 → 2640 的对照臂
V3_MAP = {"single_k1_ctl": "combo_pw", "dup_k4": "combo_pw"}

#: K=1→2→4→8 同族（同一 q 分布）；K=2 用 2640 `tree_qpw_k2` 指针
K_FAMILY = [("single_k1_ctl", 1), ("tree_qpw_k2*", 2), ("repro_topk4", 4),
            ("aggr_topk8", 8)]


def build_cands_fn(arm: str, vocab: int, prompt_idx: int):
    """为某臂某 prompt 构造 `cands_fn`（adv_random 每个 prompt 独立固定 seed）。"""
    kind, spec = ARM_KIND[arm]
    if spec == "adv_random":
        return make_cands_adv_random(vocab, ADV_RANDOM_SEED + prompt_idx)
    return spec


# ---------------------------------------------------------------- 单臂评测

def arm_pass_gate(model, prompts, n_new, pcap, arm, native_streams, log) -> list[dict]:
    """一个臂在一个 checkpoint 上跑完全部 prompt；返回逐 prompt 行（供 summarize_arm）。"""
    kind, spec = ARM_KIND[arm]
    vocab = model.embed.weight.shape[0]
    rows = []
    for pi, prompt in enumerate(prompts):
        act: list[dict] = []
        if kind == "single":
            gen, st = T.decode_spec(model, prompt, n_new, pcap, max_drafts=GAMMA,
                                    draft_fn=T.make_draft_fn(spec, log=act))
            st["n_rounds"] = len(st["accepts"])
            st["candidate_sizes"] = None
            st["mean_candidates_per_round"] = 1.0
        else:
            gen, st = decode_gate(model, prompt, n_new, pcap,
                                  build_cands_fn(arm, vocab, pi),
                                  window=ARM_WINDOW.get(arm))
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
        log(f"[arm] {arm:18s} prompt {pi:2d} n_rounds={st['n_rounds']} "
            f"identical={gen == ref} mismatch={first_mismatch}")
    return rows


# ---------------------------------------------------------------- V0 / V3 / V2 / V4 / V5

def accepts_of(res: dict, ckpt: str, arm: str) -> list[list[int]] | None:
    """从某 results.json 的 checkpoints[ckpt].arms[arm].per_prompt 取逐 prompt accepts。"""
    ck = res.get("checkpoints", {}).get(ckpt)
    if not ck:
        return None
    a = ck.get("arms", {}).get(arm)
    if not a or a.get("per_prompt") is None:
        return None
    return [pp["accepts"] for pp in a["per_prompt"]]


def check_bitwise_maps(arm_sum: dict, ref_res: dict, ckpt: str, mapping: dict) -> dict:
    """逐位相同检查：本实验臂 accepts vs 2640 对应臂 accepts（同 ckpt × prompt × 轮序）。"""
    out = {}
    for mine, theirs in mapping.items():
        if mine not in arm_sum:
            out[mine] = {"checked": False, "reason": "本臂未跑"}
            continue
        got = [r["accepts"] for r in arm_sum[mine]["per_prompt"]]
        want = accepts_of(ref_res, ckpt, theirs)
        if want is None:
            out[mine] = {"checked": False, "reason": f"2640 无 {theirs} 读数"}
            continue
        same_len = len(got) == len(want)
        per_prompt_same = same_len and all(g == w for g, w in zip(got, want))
        diffs = [{"prompt": i, "mine_rounds": len(g), "ref_rounds": len(w),
                  "first_divergence": next((j for j, (a, b) in enumerate(zip(g, w))
                                            if a != b), None)}
                 for i, (g, w) in enumerate(zip(got, want)) if g != w] if same_len else []
        out[mine] = {
            "checked": True, "vs_2640_arm": theirs,
            "bit_identical": bool(per_prompt_same),
            "n_prompts_mine": len(got), "n_prompts_ref": len(want),
            "n_prompts_differing": len(diffs),
            "diffs": diffs[:5],
            "alpha1_mine": arm_sum[mine]["alpha1"],
            "alpha1_ref": next((a["alpha1"] for a in ref_res["checkpoints"][ckpt]["arms"]
                                .values() if a.get("alpha1") is not None
                                and a is ref_res["checkpoints"][ckpt]["arms"].get(theirs)), None),
        }
    return out


def v4_saturation(arm_sum: dict, ref_res: dict, ckpt: str, ref_arm: str) -> dict:
    """V4：某臂 α₁ 相对 2640 `tree_topk4` 的配对 bootstrap 95% CI 与 |Δα₁| 判饱和。"""
    out = {}
    want_groups = accepts_of(ref_res, ckpt, ref_arm)
    for arm in ("aggr_topk8", "aggr_union_heads"):
        if arm not in arm_sum:
            continue
        mine = [r["accepts"] for r in arm_sum[arm]["per_prompt"]]
        if not want_groups or len(mine) != len(want_groups):
            out[arm] = {"checked": False, "reason": "缺 2640 对照或 prompt 数不等"}
            continue
        d_a1 = H.paired_diff(mine, want_groups, H.stat_alpha1, B=BOOT, seed=BOOT_SEED)
        d_tau = H.paired_diff(mine, want_groups, H.stat_tau, B=BOOT, seed=BOOT_SEED)
        delta = round(arm_sum[arm]["alpha1"] - ref_res["checkpoints"][ckpt]["arms"][ref_arm]["alpha1"], 6)
        ci = d_a1.get("ci") or [None, None]
        ci_has_zero = (ci[0] is not None and ci[0] <= 0 <= ci[1])
        out[arm] = {
            "checked": True, "vs_2640_arm": ref_arm,
            "alpha1_mine": arm_sum[arm]["alpha1"],
            "alpha1_ref": ref_res["checkpoints"][ckpt]["arms"][ref_arm]["alpha1"],
            "delta_alpha1": delta,
            "paired_alpha1": d_a1, "paired_tau": d_tau,
            "ci_has_zero": bool(ci_has_zero),
            "abs_delta_le_0p014": bool(abs(delta) <= T.DELTA_SORTABLE),
            "saturated": bool(ci_has_zero or abs(delta) <= T.DELTA_SORTABLE),
            "mean_candidates": arm_sum[arm]["mean_candidates_per_round"],
        }
    return out


def v2_negatives(arm_sum: dict, v1_all_identical: bool) -> dict:
    """V2：对抗臂 α₁ 的 Wilson 95% CI（负对照）与门禁灵敏度判词。"""
    out = {}
    for arm, thresh in (("adv_random_k4", 0.06), ("adv_bottomk4", 0.02)):
        if arm not in arm_sum:
            out[arm] = {"checked": False}
            continue
        n_rounds = arm_sum[arm]["n_rounds"]
        a1 = arm_sum[arm]["alpha1"]
        wil = H.wilson_ci(a1, n_rounds)
        a1_ident = arm_sum[arm]["stream_identical_to_native"]
        breach = (a1 >= 0.05) or (not a1_ident)
        out[arm] = {
            "checked": True, "alpha1": a1, "n_rounds": n_rounds,
            "wilson_95": wil,
            "wilson_upper_lt_threshold": bool(wil[1] is not None and wil[1] < thresh),
            "stream_identical_to_native": a1_ident,
            "threshold_alpha1": thresh,
            "pseudo_accept_4_over_V": round(4.0 / arm_sum[arm].get("_vocab", 206), 6),
            "verdict_breach": bool(breach),
        }
    return out


def v5_ledger(arm_sum: dict, ref_res: dict, ckpt: str, gamma: int) -> dict:
    """V5：按 K 列的代价台账（如实报、不作判据）。

    同时落两种 R_K 算法：
    - `R_K_2640consistent = 3K/(1+τ_K)`（= 2640 `B_verified·(1+γ)/(1+τ)`，B_verified=K）；
    - `R_K_readme_literal = 3K·(1+γ)/(1+τ_K)`（README §4 V5 原式；与 2640 相差因子 1+γ）。
    """
    tau_of = {}
    for arm, K in K_FAMILY:
        if arm.endswith("*"):
            ref = arm[:-1]
            a = ref_res["checkpoints"][ckpt]["arms"].get(ref)
            tau_of[K] = {"arm": ref, "source": "2640 指针", "K": K,
                         "tau": (a or {}).get("tau"), "alpha1": (a or {}).get("alpha1")}
        elif arm in arm_sum:
            tau_of[K] = {"arm": arm, "source": "本实验", "K": K,
                         "tau": arm_sum[arm]["tau"], "alpha1": arm_sum[arm]["alpha1"]}
    oracle = ref_res["checkpoints"][ckpt]["arms"].get("oracle_accept", {})
    tau_oracle = oracle.get("tau")
    ledger = {}
    for K, rec in tau_of.items():
        tau = rec["tau"]
        if tau is None:
            ledger[str(K)] = {**rec, "note": "缺 τ"}
            continue
        ledger[str(K)] = {
            **rec,
            "main_positions_per_round_flat": 3 * K,
            "main_positions_per_round_shared_tree_ideal": 2 * K + 1,
            "R_K_2640consistent": round(3 * K / (1 + tau), 4),
            "R_K_readme_literal": round(3 * K * (1 + gamma) / (1 + tau), 4),
            "tau_over_K": round(tau / K, 6),
            "tau_over_oracle": (round(tau / tau_oracle, 6) if tau_oracle else None),
        }
    margins = {}
    ks = sorted(tau_of)
    for k0, k1 in zip(ks, ks[1:]):
        t0, t1 = tau_of[k0]["tau"], tau_of[k1]["tau"]
        if t0 is None or t1 is None:
            continue
        margins[f"{k0}->{k1}"] = {
            "delta_tau": round(t1 - t0, 6), "delta_K": k1 - k0,
            "delta_tau_per_delta_K": round((t1 - t0) / (k1 - k0), 6),
        }
    thr = None
    if 1 in tau_of and 2 in tau_of and tau_of[1]["tau"] is not None \
            and tau_of[2]["tau"] is not None:
        base = (tau_of[2]["tau"] - tau_of[1]["tau"]) / 1.0
        thr = {"k1_to_k2_delta_tau_per_delta_K": round(base, 6),
               "threshold_half_base": round(0.5 * base, 6)}
        if "4->8" in margins:
            m = margins["4->8"]["delta_tau_per_delta_K"]
            thr["k4_to_k8_delta_tau_per_delta_K"] = m
            thr["k4_to_k8_worth_it"] = bool(m >= 0.5 * base)
    return {"ledger": ledger, "margins": margins, "k4_to_k8_threshold": thr,
            "tau_oracle_2640": tau_oracle,
            "R_formula_note": "README §4 V5 原式 3K·(1+γ)/(1+τ) 与 2640 口径 3K/(1+τ) "
                              "相差因子 (1+γ)=3；两种并列，未擅改口径（见 LOG）"}


# ---------------------------------------------------------------- 主流程

def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_ckpt(ckpt: str, name: str, val, n_prompts: int, n_new: int, arms: list[str],
             log, ref_res: dict) -> dict:
    log(f"[ckpt] {name} :: {ckpt}  arms={arms}")
    model = H.load_ckpt(ckpt)
    pcap = model.pcap
    with open(os.path.join(ckpt, "results.json"), encoding="utf-8") as f:
        ckres = json.load(f)
    prompts = H.make_prompts(val, n_prompts)
    native_streams = []
    for pi, prompt in enumerate(prompts):
        gen_full, _ = H.decode(model, prompt, n_new + T.GAMMA, pcap)
        native_streams.append(gen_full)
    log(f"[{name}] native streams ready")

    arm_sum, cycle, failures = {}, [], {}
    nfwd_ok: dict[str, bool] = {}
    for arm in arms:
        try:
            rows = arm_pass_gate(model, prompts, n_new, pcap, arm, native_streams, log)
        except NotImplementedError as e:
            log(f"[{name}] {arm} NotImplementedError: {e}")
            failures[arm] = {"error": "NotImplementedError", "message": str(e),
                             "conclusion": "带驱逐（window>0）的树验证未实装"}
            continue
        nfwd_ok[arm] = all(r["n_forward"] == r["n_rounds"] for r in rows)
        arm_sum[arm] = H.summarize_arm(rows, n_new, T.GAMMA)
        arm_sum[arm]["_vocab"] = int(model.embed.weight.shape[0])
        for row in rows:
            for j, rec in enumerate(row["activity"]):
                cycle.append({"ckpt": name, "arm": arm, "prompt": row["prompt"],
                              "round": j, **rec})
        log(f"[{name}] {arm:18s} a1={arm_sum[arm]['alpha1']} tau={arm_sum[arm]['tau']} "
            f"tpf={arm_sum[arm]['tokens_per_forward']} "
            f"identical={arm_sum[arm]['stream_identical_to_native']}")

    v0 = check_bitwise_maps(arm_sum, ref_res, name, V0_MAP)
    v3 = check_bitwise_maps(arm_sum, ref_res, name, V3_MAP)
    v4 = v4_saturation(arm_sum, ref_res, name, "tree_topk4")
    v1_ok = all(a["stream_identical_to_native"] for a in arm_sum.values())
    v2 = v2_negatives(arm_sum, v1_ok)
    v5 = v5_ledger(arm_sum, ref_res, name, T.GAMMA)
    criteria = {
        "V0_harness_faithful": {
            "checked": v0, "pass": all(v.get("bit_identical") for v in v0.values()
                                       if v.get("checked")),
            "all_checked": all(v.get("checked") for v in v0.values())},
        "V1_lossless_all_arms": {
            "all_arms_stream_identical": bool(v1_ok),
            "mismatching_arms": [a for a, v in arm_sum.items()
                                 if not v["stream_identical_to_native"]],
            "n_forward_eq_n_rounds_all": all(nfwd_ok.values()),
            "n_forward_eq_n_rounds_per_arm": nfwd_ok},
        "V2_negatives": v2,
        "V3_degenerate_equivalence": {
            "checked": v3, "pass": all(v.get("bit_identical") for v in v3.values()
                                       if v.get("checked")),
            "all_checked": all(v.get("checked") for v in v3.values())},
        "V4_saturation": v4,
        "V5_cost_ledger": v5,
    }
    res = {"path": ckpt, "params": ckres.get("params"), "bpc": ckres.get("bpc"),
           "config": ckres.get("config"), "resolved": ckres.get("resolved"),
           "n_prompts": n_prompts, "n_new": n_new,
           "engine": {"n_mtp": pcap.n_mtp, "chain_mode": pcap.chain_mode,
                      "window": pcap.window, "anchors": pcap.anchors,
                      "transport": pcap.transport},
           "arms": arm_sum, "arm_failures": failures, "criteria": criteria}
    del model
    torch.cuda.empty_cache()
    return res, cycle


def parse_arm_plan(specs: list[str]) -> dict:
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
    ap.add_argument("--arms", nargs="+", default=list(ARM_KIND))
    ap.add_argument("--arm-plan", nargs="*", default=[])
    ap.add_argument("--budget-min", type=float, default=13.0)
    ap.add_argument("--lock", choices=("ckpt", "run", "none"), default="ckpt")
    args = ap.parse_args()

    if args.smoke:
        args.prompts, args.n_new = 2, 32
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "run_log.txt"), "a", encoding="utf-8")

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
    with open(RESULTS_2640, encoding="utf-8") as f:
        ref_res = json.load(f)

    log(f"[run] {time.strftime('%Y-%m-%dT%H:%M:%S')} ckpts={ckpt_dirs} smoke={args.smoke} "
        f"lock={args.lock} budget_min={args.budget_min}")
    lock = H.GpuLock(log=log)
    results = {
        "experiment": "2650-verify-gate-13m",
        "wp": "6.2.3 (轴 6 预测轴 · Phase B)",
        "contract": "README.md（预登记：臂定义 / V0–V6 判据 / 否证条件）",
        "timing_measured": False,
        "timing_note": "禁墙钟判据；GPU 实耗只在 run_log.txt，不进本文件",
        "imported_2640_sha256": {f: _sha256(os.path.join(DIR2640, f)) for f in IMPORTED_2640},
        "config": {"prompt_len": H.PROMPT_LEN, "stride": H.STRIDE, "n_prompts": args.prompts,
                   "n_new": args.n_new, "gamma": T.GAMMA, "arms": args.arms,
                   "bootstrap": {"B": BOOT, "seed": BOOT_SEED, "cluster": "prompt"},
                   "adv_random_seed": ADV_RANDOM_SEED, "lock_granularity": args.lock,
                   "budget_min": args.budget_min, "smoke": bool(args.smoke)},
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
        free0 = H.gpu_free_mib()
        if args.lock != "none":
            lock.acquire()
        try:
            res, cycle = run_ckpt(ck, name, H.enwik8_val(), args.prompts, args.n_new,
                                  arms, log, ref_res)
            res["gpu_free_mib_before"] = free0
            res["gpu_free_mib_after"] = H.gpu_free_mib()
            results["checkpoints"][name] = res
            cycle_all.extend(cycle)
        finally:
            if args.lock != "none":
                lock.release(note=f"after {name}")
            flush(cycle_all)
        if args.lock == "ckpt" and ci < len(ckpt_dirs) - 1:
            log("[lock] §20.1 让出窗口：sleep 60s")
            time.sleep(60)
    flush(cycle_all)
    log(f"[run] wrote results.json / cycle_log.jsonl ({time.time() - t_start:.0f}s)")
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
