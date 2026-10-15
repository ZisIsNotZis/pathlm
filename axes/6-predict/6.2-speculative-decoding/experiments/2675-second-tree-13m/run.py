#!/usr/bin/env python3
"""2675-second-tree-13m · 驱动（契约 = 本目录 `README.md`；机制裁决 = `QUESTIONS.md`）。

    python3 run.py --stage train --trunks 13m_s0 --seeds 0 1     # 只训二级头（主干冻结）
    python3 run.py --stage eval  --trunks 13m_s0 --seeds 0 --arms native direct1 lvl1_tree lvl2_tree
    python3 run.py --stage all   --trunks 13m_s0 13m_s1 --seeds 0 1 --k 2 4

产出（写进 `--out`，默认本目录）：`results.json`（**禁墙钟**）/ `analysis.md`（人话在前）/
`MODEL.md` / `run_log.txt`（含墙钟，只此一处）/ `train_log.jsonl` / `heads/*.pt`（不入库）/
`RESUME.md`（已完成 / 下一步确切命令）。

GPU 纪律（`SCHEMA.md` §20 / §20.1）：**逐 run 取放锁**，臂间（以及训练与评测之间）释放后
`sleep 60` 再重取（`run_log.txt` 里可见 `[§20.1] yield window`）；整程 <10 min 的冒烟可
`--lock-mode whole` 整程持锁。长任务请用 detached（`nohup` + `while` 重试）方式起，
本驱动**不在前台阻塞等锁**：取不到锁抛 `GPUUnavailable`，由调用方重试。

`--resume`：跳过 `results.json::runs` 里已有的 (stage|trunk|seed) 单元；每个单元跑完立即写盘。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import second_tree as S  # noqa: E402
from gpu_lock import GPUUnavailable, gpu_lock  # noqa: E402

ARMS_ALL = ("native", "ar_1step", "direct1", "lvl1_tree", "lvl2_tree", "lvl2_flat")
KS = (2, 4)
YIELD_S = 60.0


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


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def head_path(out: str, trunk: str, seed: int) -> str:
    return os.path.join(out, "heads", f"chain3_{trunk}_s{seed}.pt")


def load_results(out: str) -> dict:
    p = os.path.join(out, "results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def dump_results(out: str, results: dict) -> None:
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ 头训练单元

def train_unit(out: str, trunk: str, seed: int, results: dict, log, device: str,
               smoke: bool) -> None:
    key = f"train|{trunk}|{seed}"
    if key in results["runs"]:
        log(f"skip（已完成）{key}")
        return
    steps = 20 if smoke else S.TRAIN_STEPS
    torch.manual_seed(seed)                      # 二级头初始化（构造期随机流）
    model = S.load_trunk(S.TRUNKS[trunk], chain_second=True)
    model.to(device)
    data = S.enwik8()
    entry = S.train_chain_head(model, data["train"], seed, log, steps=steps)
    model.cpu()
    hp = head_path(out, trunk, seed)
    entry["file_sha256"] = S.save_head(model, hp)
    entry["head_path"] = os.path.relpath(hp, out)
    entry["n_params"] = S.head_param_stats(model)
    entry["config"] = {"chain_second": True, "n_mtp": model.pcap.n_mtp,
                       "chain_mode": model.pcap.chain_mode, "transport": model.pcap.transport,
                       "window": model.pcap.window, "anchors": model.pcap.anchors}
    results["runs"][key] = entry
    with open(os.path.join(out, "train_log.jsonl"), "a", encoding="utf-8") as f:
        for row in entry["trace"]:
            f.write(json.dumps({"trunk": trunk, "seed": seed, **row}, ensure_ascii=False) + "\n")
    dump_results(out, results)
    log(f"train DONE {key} steps={steps} head_sha={entry['file_sha256'][:12]} "
        f"frozen_assert={entry['frozen_assert']}")


# ------------------------------------------------------------------ 评测单元

def eval_unit(out: str, trunk: str, seed: int, arms: list[str], ks: list[int],
              results: dict, log, device: str, smoke: bool, n_prompts: int,
              n_new: int, determinism: bool) -> None:
    key = f"eval|{trunk}|{seed}"
    done = results["runs"].get(key, {}).get("arms", {})
    todo: list[tuple[str, int]] = []
    for arm in arms:
        if arm in ("lvl1_tree", "lvl2_tree", "lvl2_flat"):
            todo += [(arm, k) for k in ks]
        else:
            todo.append((arm, ks[0]))
    if not todo:
        return
    hp = head_path(out, trunk, seed)
    need_head = any(a.startswith("lvl2") for a, _ in todo)
    if need_head and not os.path.exists(hp):
        results.setdefault("blockers", []).append(
            f"{key}: 缺二级头 {os.path.relpath(hp, out)} ⇒ lvl2* 臂未跑")
        dump_results(out, results)
        log(f"BLOCKER {key}: 缺 {hp}")
        return
    n_prompts = 2 if smoke else n_prompts
    n_new = 32 if smoke else n_new
    data = S.enwik8()
    ps = S.prompts(data["val"], n_prompts)
    ref = S.load_trunk(S.TRUNKS[trunk], chain_second=False).to(device)
    pcap_ref = ref.pcap
    with torch.no_grad():
        native = [S.run_arm(ref, p, n_new, pcap_ref, "native")[0] for p in ps]
    model = S.load_trunk(S.TRUNKS[trunk], chain_second=True, head=hp).to(device)
    pcap = model.pcap
    out_arms = dict(done)
    for arm, k in todo:
        name = f"{arm}_k{k}" if arm.startswith("lvl") else arm
        if name in out_arms:
            log(f"skip（已完成）{key} {name}")
            continue
        groups, rows = [], []
        t0 = time.time()
        for pi, p in enumerate(ps):
            act: list = []
            gen, st = S.run_arm(model, p, n_new, pcap, arm, K=k, activity=act)
            ref_stream = native[pi]
            ident = gen == ref_stream
            first = next((j for j, (a, b) in enumerate(zip(gen, ref_stream)) if a != b),
                         None) if not ident else None
            groups.append(list(st["accepts"]) if st.get("accepts") else [])
            rows.append({"prompt": pi, "identical_to_native": bool(ident),
                         "first_mismatch": first, "n_rounds": st.get("n_rounds"),
                         "n_forward": st.get("n_forward"), "gen_head": gen[:8],
                         "rounds": [{"v1": r["v1"], "candidates": r["candidates"],
                                     "drafts": r["drafts"], "n_acc": r["n_acc"]}
                                    for r in act] if act else None})
        entry = {"arm": arm, "K": k, "rows": rows,
                 "lossless": all(r["identical_to_native"] for r in rows),
                 "mismatching_prompts": [r["prompt"] for r in rows
                                         if not r["identical_to_native"]],
                 "stats": S.arm_stats([g for g in groups if g], k) if groups and groups[0]
                 else {"note": "native 对照（accepts=None）", "tau": None}}
        if groups and groups[0]:
            entry["per_prompt_accepts"] = groups
        out_arms[name] = entry
        results["runs"][key]["arms"] = out_arms
        dump_results(out, results)
        log(f"eval {key} {name}: lossless={entry['lossless']} "
            f"tau={entry['stats'].get('tau')} alpha1={entry['stats'].get('alpha1')} "
            f"rounds={sum(len(g) for g in groups)} ({time.time() - t0:.0f}s)")
    # 确定性（L5）：同 seed 重跑一条臂逐值相同
    if determinism and not smoke:
        a0 = [a for a, _ in todo][:1]
        arm0 = a0[0] if a0 else "lvl2_tree"
        k0 = ks[0]
        r1 = S.run_arm(model, ps[0], n_new, pcap, arm0, K=k0, activity=None)
        r2 = S.run_arm(model, ps[0], n_new, pcap, arm0, K=k0, activity=None)
        results["runs"][key]["determinism"] = {
            "arm": arm0, "K": k0, "prompt": 0,
            "accepts_bit_identical": bool(r1[1].get("accepts") == r2[1].get("accepts")),
            "stream_bit_identical": bool(r1[0] == r2[0])}
        dump_results(out, results)
        log(f"determinism {key} {arm0}: "
            f"{results['runs'][key]['determinism']['stream_bit_identical']}")
    model.cpu()
    ref.cpu()


# ------------------------------------------------------------------ L0 引擎单测复跑

def engine_unit_tests(out: str, results: dict, log, stage: str) -> None:
    tests = out and os.path.join(S.ROOT, "tests")
    cmd = [sys.executable, "-m", "pytest", os.path.join(tests, "test_m1.py"), "-q", "-k",
           "chain_second or step_tree_multi"]
    t = subprocess.run(cmd, cwd=S.ROOT, capture_output=True, text=True, timeout=900)
    tail = [ln for ln in t.stdout.strip().splitlines() if ln.strip()][-1:]
    results.setdefault("L0_engine_unit_tests", {})[stage] = {
        "cmd": " ".join(os.path.relpath(c, S.ROOT) if os.path.isabs(c) else c for c in cmd),
        "returncode": t.returncode, "summary": tail[0] if tail else ""}
    with open(os.path.join(out, "engine_unit_tests.txt"), "a", encoding="utf-8") as f:
        f.write(f"# {now()} stage={stage} rc={t.returncode}\n{t.stdout}\n")
    dump_results(out, results)
    log(f"engine u-tests ({stage}) rc={t.returncode}: {tail[0] if tail else ''}")


# ------------------------------------------------------------------ 判据

def _paired(groups_a: list, groups_b: list, stat_name: str) -> dict:
    stat = getattr(S.R2640, stat_name)
    return S.paired_ci(groups_a, groups_b, stat)


def build_judgement(results: dict) -> dict:
    """主判：每个 (trunk, seed) 上 lvl2 vs lvl1（同 K）的 Δτ / Δα₁（配对 prompt 聚类 bootstrap）。"""
    out: dict = {"per_checkpoint": {}, "contract_5": {}}
    for key, unit in results["runs"].items():
        if not key.startswith("eval|"):
            continue
        arms = unit.get("arms", {})
        cps = {}
        for k in KS:
            a, b = f"lvl1_tree_k{k}", f"lvl2_tree_k{k}"
            if a in arms and b in arms and "per_prompt_accepts" in arms[a] \
                    and "per_prompt_accepts" in arms[b]:
                ga, gb = arms[a]["per_prompt_accepts"], arms[b]["per_prompt_accepts"]
                cps[f"K={k}"] = {
                    "d_tau": _paired(gb, ga, "stat_tau"),
                    "d_alpha1": _paired(gb, ga, "stat_alpha1"),
                    "d_alpha2_given_alpha1": _paired(gb, ga, "stat_alpha2"),
                    "lvl1": arms[a]["stats"], "lvl2": arms[b]["stats"],
                }
        marge = {}
        for lvl in ("lvl1_tree", "lvl2_tree"):
            a, b = f"{lvl}_k2", f"{lvl}_k4"
            if a in arms and b in arms and "per_prompt_accepts" in arms[a] \
                    and "per_prompt_accepts" in arms[b]:
                marge[lvl] = _paired(arms[b]["per_prompt_accepts"],
                                     arms[a]["per_prompt_accepts"], "stat_tau")
        for name, e in arms.items():
            cps.setdefault("losslessness", {})[name] = e["lossless"]
        for name, e in arms.items():
            if name.startswith("lvl2_flat"):
                cps.setdefault("flat_diagnostic", {})[name] = {
                    "lossless": e["lossless"], "mismatching_prompts": e["mismatching_prompts"]}
        if cps:
            cps["marginal_K4_minus_K2_tau"] = marge
            out["per_checkpoint"][key] = cps
    return out


def contract5_verdicts(judgement: dict, results: dict) -> dict:
    """契约 §5 四条否证条件逐条判。"""
    cps = judgement["per_checkpoint"]
    d_tau = {k: {kk: vv["d_tau"] for kk, vv in v.items() if kk.startswith("K=")}
             for k, v in cps.items()}
    all_ci_include_zero = []
    for k, per_k in d_tau.items():
        for kk, d in per_k.items():
            ci = d.get("ci") or [None, None]
            all_ci_include_zero.append(ci[0] is not None and ci[0] <= 0 <= ci[1])
    lossless_all = all(v for k, v in cps.items()
                       for v in (cps[k].get("losslessness", {}) or {}).values())
    flat = {k: v.get("flat_diagnostic", {}) for k, v in cps.items()}
    return {
        "losslessness_all_arms": lossless_all,
        "flat_diagnostic": flat,
        "d_tau_all_ci_include_zero": all(all_ci_include_zero) if all_ci_include_zero else None,
        "d_tau_per_checkpoint": d_tau,
    }


# ------------------------------------------------------------------ 落盘文档

def write_model_md(out: str, results: dict, log) -> None:
    lines = ["# MODEL.md · 权重清单与复现命令（2675-second-tree-13m）", "",
             "主干 = 2631-spec-three-arms/train/{13m_s0,13m_s1}（**只读、冻结**）；本 WP 只训二级头",
             "（`transforms_chain2`/`chain_cat2`/`conf_chain3`，见 `second_tree.py::train_chain_head`）。",
             "头权重 `heads/*.pt` 是派生物、`*.pt` 已被 `.gitignore` 排除（`docs/methodology.md` §8），",
             "此处只登记内容哈希。", "",
             "| 单元 | 头参数 | state_dict 摘要 | 文件 sha256 |", "|---|---|---|---|"]
    for key, e in sorted(results["runs"].items()):
        if not key.startswith("train|"):
            continue
        lines.append(f"| {key} | {e['n_params']} | {e['head_sha256']} | "
                     f"{e['file_sha256'][:16]}… |")
    lines += ["", "## 复现命令", "```bash",
              "python3 run.py --stage train --trunks 13m_s0 13m_s1 --seeds 0 1",
              "python3 run.py --stage eval --trunks 13m_s0 13m_s1 --seeds 0 1 --k 2 4",
              "python3 -m pytest test_second_tree.py -q",
              "python3 mutate.py",
              "```", ""]
    with open(os.path.join(out, "MODEL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("MODEL.md 落盘")


def write_analysis(out: str, results: dict, judgement: dict, log) -> None:
    c5 = results.get("contract_5", {})
    lines = ["# 2675 · 完整二级树的判决（人话摘要先读这一段）", "", "## 0. 人话摘要", ""]
    lines.append("（由 `run.py --stage eval` 的落盘读数生成；原始数字见 `results.json`。）")
    lines.append("")
    lines.append("**它想干什么**：投机解码里位置 N+3 的草稿有两种来源——(a) 一个与上下文无关的直连头"
                 "（`node3`），(b) 专门以「已被验证接受的 N+2 候选」为条件的二级条件头（`chain_second`）。"
                 "后者是 EAGLE/DeepSeek-MTP 那一系的思路：把上一级的**候选**当成条件，让下一级草稿更准。"
                 "本 WP 只训这个二级头（主干与一级头全冻结），然后看它相对同一 checkpoint 的一级树"
                 "能不能真的把「每轮被接受的草稿数 τ」抬起来。")
    lines.append("")
    lines.append("**怎么测**：同一主干 × 同一头种子下，两臂的候选集、行数、位置数、前向次数"
                 "**完全相同**，唯一变量是 N+3 草稿的来源。主判 = 逐 prompt 配对的 Δτ 与该配对"
                 "bootstrap（按 prompt 聚类、B=2000）95% CI；无损性 = 三臂发出流与 `native` 逐位相同。")
    lines.append("")
    for key in sorted(judgement["per_checkpoint"]):
        blk = judgement["per_checkpoint"][key]
        for kk in sorted(x for x in blk if x.startswith("K=")):
            d = blk[kk]
            lines.append(f"- `{key}` {kk}：Δτ = {d['d_tau']['point']} "
                         f"CI={d['d_tau']['ci']}；Δα₁ = {d['d_alpha1']['point']} "
                         f"CI={d['d_alpha1']['ci']}")
    lines += ["", "## 1. 判据逐条（L0–L5）", ""]
    lines.append(f"- **L0 保真**：引擎既有单测复跑 = "
                 f"{json.dumps(results.get('L0_engine_unit_tests', {}), ensure_ascii=False)}；"
                 f"`chain_second=False` 与 2631 既有载入路径逐位一致（单测 "
                 f"`test_chain_second_off_matches_legacy_load`）；二级头单测 "
                 f"`test_chain_second_on_changes_only_the_two_level_heads`；冻结断言见 "
                 f"`results.json::runs.train.*.frozen_assert`。")
    lines.append(f"- **L1（主判）**：见 §0 与 §2 的 Δ 表（每个 (主干,种子,K) 一行）。")
    lines.append("- **L2 n≥2**：覆盖矩阵见 `results.json::coverage`；未跑到的组合见 `RESUME.md`。")
    loss = [k for k, v in judgement["per_checkpoint"].items()
            if not all((v.get("losslessness") or {}).values())]
    lines.append(f"- **L3 无损性**：{'全部臂逐位 == native' if not loss else f'**未过**：{loss}'}；"
                 f"两条机械检查 + gate probe 见 `test_second_tree.py`。")
    lines.append("- **L4 代价轴**：行数 K / 位置数 3K / 前向 1 次每轮；`R = K·(1+γ)/(1+τ)`（契约 §6 口径）。"
                 "**等 FLOPs 对照臂在本引擎 `n_mtp=3` 上限下不可得**（63-F2），R 只作代价面报、不作判据。")
    lines.append("- **L5 禁墙钟**：`results.json` 的墙钟字段命中 = "
                 f"{results.get('_wall_clock_fields', [])}（0 命中即通过）；确定性见 "
                 f"`results.json::runs.*.determinism`。")
    lines += ["", "## 2. 契约 §5 否证条件", "",
              f"- Δτ 的配对 CI 是否**全部含 0**：{c5.get('d_tau_all_ci_include_zero')}；逐条见 §0 表。",
              f"- 无损性：{c5.get('losslessness_all_arms')}；字面 `step_tree_multi` 诊断臂结果："
              f"{json.dumps(c5.get('flat_diagnostic', {}), ensure_ascii=False)}。",
              f"- K 的边际收益（lvl2 相对 lvl1）见 `results.json::judgement.*.marginal_K4_minus_K2_tau`。",
              "", "## 3. 未测因子（禁止外推）", "",
              "见 `GAPS.md`：G1（引擎缺位置忠实的多兄弟树原语，最小 patch 已写）、"
              "G2（chain3 草稿的上下文严格对齐 h_{N+1} 的版本未测）、G3（(主干×种子×K×臂) 覆盖矩阵）。",
              "", "**本尺度 / 本任务 / 冻结主干**的结论不外推；绝对水平只适用该 checkpoint、"
              "该 prompt 集、温度 0、γ=2、enwik8 char 文本。", ""]
    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("analysis.md 落盘")


def update_resume(out: str, results: dict, todo: list[str], log) -> None:
    done = sorted(results["runs"])
    lines = ["# RESUME.md · 2675 的断点与下一步", "",
             f"最后更新：{now()}", "",
             "## 已完成（`results.json::runs` 的键）", ""] + [f"- `{k}`" for k in done] + [
             "", "## 未完成（下一步确切命令）", ""]
    lines += [f"- {c}" for c in todo] or ["- （无：本次派发的单元都已落盘）"]
    lines += ["", "## 纪律提醒", "",
              "- 逐 run 取放锁 + 臂间 `sleep 60`（`SCHEMA.md` §20.1）；长任务用 `nohup` detached 重试。",
              "- `--resume` 幂等：重跑同一命令会跳过已完成的 `train|*`/`eval|*` 单元。", ""]
    with open(os.path.join(out, "RESUME.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("RESUME.md 落盘")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval", "all", "finalize"), default="all")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--trunks", nargs="+", default=["13m_s0", "13m_s1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=list(ARMS_ALL))
    ap.add_argument("--k", nargs="+", type=int, default=list(KS))
    ap.add_argument("--prompts", type=int, default=S.N_PROMPTS)
    ap.add_argument("--n-new", type=int, default=S.N_NEW)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--lock-mode", choices=("unit", "whole"), default="unit")
    ap.add_argument("--no-yield", action="store_true")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, "run_log.txt"))
    results = load_results(out)
    results.setdefault("experiment", "2675-second-tree-13m")
    results.setdefault("contract", "README.md (pre-registered) + QUESTIONS.md (机制裁决)")
    results.setdefault("runs", {})
    results.setdefault("blockers", [])
    log(f"stage={a.stage} trunks={a.trunks} seeds={a.seeds} arms={a.arms} k={a.k} "
        f"smoke={a.smoke} device={a.device} lock={a.lock_mode} resume={a.resume}")

    todo: list[str] = []
    whole = a.lock_mode == "whole"
    need_gpu = a.device != "cpu"
    ctx = (lambda: gpu_lock("wp6.2.5"))() if (need_gpu and not a.no_lock) else _NullCtx()
    if whole:
        ctx.__enter__()
    try:
        if a.stage in ("train", "all"):
            for trunk in a.trunks:
                for seed in a.seeds:
                    key = f"train|{trunk}|{seed}"
                    if a.resume and key in results["runs"]:
                        log(f"skip（--resume）{key}")
                        continue
                    if not whole:
                        with ctx:
                            train_unit(out, trunk, seed, results, log, a.device, a.smoke)
                    else:
                        train_unit(out, trunk, seed, results, log, a.device, a.smoke)
                    if not whole and a.device != "cpu" and not a.no_yield:
                        log(f"[§20.1] yield window {YIELD_S:.0f}s before next lock acquire")
                        time.sleep(YIELD_S)
        if a.stage in ("eval", "all"):
            for trunk in a.trunks:
                for seed in a.seeds:
                    key = f"eval|{trunk}|{seed}"
                    if a.resume and f"eval|{trunk}|{seed}" in results["runs"]:
                        want = [f"{x}_k{k}" if x.startswith("lvl") else x
                                for x in a.arms
                                for k in (a.k if x.startswith("lvl") else a.k[:1])]
                        have = results["runs"][f"eval|{trunk}|{seed}"].get("arms", {})
                        if all(w in have for w in want):
                            log(f"skip（--resume）eval|{trunk}|{seed}")
                            continue
                    results["runs"].setdefault(key, {}).setdefault("arms", {})
                    engine_unit_tests(out, results, log, f"{trunk}_s{seed}")
                    if not whole:
                        with ctx:
                            eval_unit(out, trunk, seed, a.arms, a.k, results, log, a.device,
                                      a.smoke, a.prompts, a.n_new, not a.no_determinism)
                    else:
                        eval_unit(out, trunk, seed, a.arms, a.k, results, log, a.device,
                                  a.smoke, a.prompts, a.n_new, not a.no_determinism)
                    if not whole and a.device != "cpu" and not a.no_yield:
                        log(f"[§20.1] yield window {YIELD_S:.0f}s before next lock acquire")
                        time.sleep(YIELD_S)
        judgement = build_judgement(results)
        results["judgement"] = judgement
        results["contract_5"] = contract5_verdicts(judgement, results)
        results["coverage"] = {
            "trunks": a.trunks, "seeds": a.seeds, "arms": a.arms, "K": a.k,
            "n_prompts": 2 if a.smoke else a.prompts, "n_new": 32 if a.smoke else a.n_new,
            "prompt_len": S.PROMPT_LEN, "stride": S.STRIDE, "temperature": 0.0, "gamma": S.GAMMA,
            "bootstrap": {"B": S.BOOT_B, "seed": S.BOOT_SEED, "cluster": "prompt",
                          "ci": "percentile95"},
            "head_training": {"steps": S.TRAIN_STEPS, "lr": S.TRAIN_LR,
                              "batch_positions": S.TRAIN_BATCH_POSITIONS,
                              "seq_len": S.TRAIN_SEQ_LEN, "data_seed": S.DATA_SEED,
                              "optimizer": "Adam(no weight decay)"},
            "note": "lvl1_tree 与 lvl2_tree 的候选集/行数/位置数/前向次数完全相同；"
                    "唯一变量 = N+3 草稿是否以 N+2 候选为条件（契约 §2 单一变量声明）"}
        bad = [b for b in S.has_wall_clock(results) if not b.endswith("_wall_clock_fields")]
        results["_wall_clock_fields"] = bad
        dump_results(out, results)
        if bad:
            raise AssertionError(f"results.json 含墙钟字段：{bad}")
        if a.stage in ("all", "finalize"):
            write_model_md(out, results, log)
            write_analysis(out, results, judgement, log)
        for trunk in a.trunks:
            for seed in a.seeds:
                for stage in ("train", "eval"):
                    key = f"{stage}|{trunk}|{seed}"
                    if key not in results["runs"]:
                        todo.append(f"python3 run.py --stage {stage} --trunks {trunk} "
                                    f"--seeds {seed} --resume")
        update_resume(out, results, todo, log)
        log("ALL DONE")
    finally:
        if whole:
            ctx.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GPUUnavailable as e:                      # detached 重试等着它
        print(f"GPU_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(75)
