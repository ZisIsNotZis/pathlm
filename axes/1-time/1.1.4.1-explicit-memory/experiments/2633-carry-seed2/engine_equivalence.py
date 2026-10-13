#!/usr/bin/env python3
"""2633-carry-seed2 · **语义级**实现等价证据（SCHEMA §9 第①项的加强版）。

背景（为什么文件哈希不够）：本仓有多条线并行，`pathlm/**` 可能有**别的线的未提交改动**
（本线执行期间确实出现过）。文件哈希只说明「字节是否一样」，回答不了「换了字节之后，
**本实验用到的那条路径**行为是否一样」。

**比较的基准 = 「2631 运行时版本」，不是 `HEAD`**（复核追加要求）：并行线对引擎做过
**默认关闭的加性改动**并用它们自己的提交入库，`HEAD` 的**字节**因此会随别人的工作前进；
§9 真正要判的是「**2631 当时跑的那一版**与**本实验跑的那一版**在开关关闭时行为是否逐位一致」。
⇒ 基准版本**按 2631 `results.json::code_hashes` 里记录的 sha256 定位**（扫 `git log --all`
的每个 blob，取哈希相符的那个 commit）；字节允许变，**行为必须逐位不变**。

**两层覆盖**（复核 P1-3：只覆盖 `pathlm/**` 不够，训练脚本也会漂移）：

1. **`pathlm/**`（前向）**：把基准版本的 `pathlm/` 整包解到临时目录、以独立包名导入，
   与本实验三档配置（`carry_rule` ∈ {none, plain, delta}，dense + topology=sequential +
   窗口 16 + 锚 0，13M 主干）各建一个模型：同 seed ⇒ **参数量与参数张量逐位相同**；
   同一份 `state_dict` 装进两版 ⇒ 固定输入上 **logits 逐位相同**。
2. **`train_depth_ar.py`（训练循环）**：`trainer_loss_sequence_equivalence()` 用基准版本脚本与
   **工作树**版脚本各跑 N 步（同 seed、同合成数据、同配方），逐字段比 `train_log.jsonl`
   （除墙钟 `min`）⇒ 训练循环/loss 归一/数据顺序/落盘路径逐位一致。

结论落 `engine_equivalence.json`。`equality_vs_2631.compare()` 的 verdict：
- 两层都过 ⇒ `reuse_ok`；
- 只有前向过、训练脚本这一层**未证**（例如工作树版本跑不起来） ⇒ `reuse_ok_pathlm_only`
  （**降级声明**，必须如实写进 analysis/MODEL）；
- 前向不过 ⇒ `reuse_rejected`。

用法：python3 engine_equivalence.py            # CPU（前向层）+ GPU（训练脚本层，~1 min）
      python3 engine_equivalence.py --no-trainer   # 只跑前向层
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)          # 让 `pathlm` 可导入（本实验只读导入仓库包）
CFG = json.load(open(os.path.join(HERE, "config.json")))

#: 本实验用到的主干规格（来自 config.json，不另写一套）
BASE = dict(CFG["model"])
VOCAB = CFG["data"]["vocab_size"]
RULES = ("none", "plain", "delta")
PROBE_SEED = 1234


#: 2631 的运行时记录（`code_hashes`）——基准版本的权威来源
RECORD_2631 = os.path.join(ROOT, "axes/1-time/1.1.4.1-explicit-memory/experiments",
                           "2631-carry-reseed/results.json")
PKG_NAME = "base_pathlm"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _recorded_code_hashes() -> dict:
    """2631 运行时记录的 `{相对路径: sha256}`。"""
    return json.load(open(RECORD_2631))["code_hashes"]


def resolve_recorded_commit(rel: str, recorded_sha: str) -> dict:
    """按 2631 记录的 sha256 反查「当时那一版」所在的 commit（唯一命中才算）。

    做法：对 `git log --all -- <rel>` 的每个 commit 取该文件的 blob、算 sha256；命中即返回。
    命中 0 个 ⇒ 记录版本不在可达历史里（报错）；命中 >1 个 ⇒ 多个 commit 内容相同（取最新，
    并如实记录个数）。
    """
    out = subprocess.run(["git", "-C", ROOT, "log", "--format=%H", "--all", "--", rel],
                         capture_output=True, text=True).stdout.split()
    hits = []
    for c in out:
        blob = subprocess.run(["git", "-C", ROOT, "show", f"{c}:{rel}"],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            continue
        if _sha(blob.stdout) == recorded_sha:
            hits.append(c)
    if not hits:
        raise RuntimeError(
            f"{rel}: 按记录 sha256 {recorded_sha[:12]}… 在 git 历史里找不到对应版本 "
            f"（§9 基准无法定位）")
    return {"rel": rel, "recorded_sha256": recorded_sha, "commit": hits[0],
            "n_matching_commits": len(hits),
            "commit_subject": subprocess.run(
                ["git", "-C", ROOT, "log", "-1", "--format=%h %s", hits[0]],
                capture_output=True, text=True).stdout.strip()}


def _extract_package(dst: str, ref: str) -> str:
    """把 `ref` 版的 `pathlm/*.py` 解到 `dst/<PKG_NAME>/`，返回包目录。"""
    pkg = os.path.join(dst, PKG_NAME)
    os.makedirs(pkg, exist_ok=True)
    files = subprocess.run(["git", "-C", ROOT, "ls-tree", "--name-only", ref, "pathlm/"],
                           capture_output=True, text=True).stdout.split()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        blob = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{rel}"],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            continue
        with open(os.path.join(pkg, os.path.basename(rel)), "w") as f:
            f.write(blob.stdout)
    return pkg


#: 训练脚本层的比较：只比这些字段（`min` 是墙钟 ⇒ 排除）
_TRAINER_LOG_FIELDS = ("step", "loss", "gnorm", "aux_w", "final_w", "window_mask",
                       "synth_answer_ce", "synth_chance_ce")


def trainer_loss_sequence_equivalence(steps: int = 1001, *, seed: int = 0,
                                      timeout_s: int = 600,
                                      base_commit: str | None = None) -> dict:
    """**2631 运行时版本**的 `train_depth_ar.py` vs **工作树**版：同 seed/同数据/同配方跑 N 步。

    `base_commit=None` ⇒ 按 2631 记录的 sha256 现场定位（见 `resolve_recorded_commit`）。

    **N 的选择（复核 P2-3）**：训练器每 200 步写一行日志，所以「N 步」并不等于「N 个比较点」；
    默认取 **1001 步**（≈ 6 个日志点：step 0/200/400/600/800/1000），而不是 20 步
    （那只有 step 0 与 19 两个点，措辞上容易被误读成「20 个点的序列」）。
    实际比较点数一律由产物 `n_lines` 给出，**不要在文字里硬编码**。

    判定 = 两份 `train_log.jsonl` 在这些字段上**逐字段逐值相同**（排除墙钟 `min`）。
    这是训练循环（loss 归一、数据顺序、梯度门、落盘）的**行为**等价证据——文件哈希被
    并行线的未提交改动打断时，这一层才能回答「换了字节之后训练行为是否一样」。

    需要 GPU（`train_depth_ar.py` 无条件 `.cuda()`）：属于 §20.1.2 的短任务（~1 min、<1 GB）。
    """
    base_flags = ["--d", "32", "--layers", "2", "--heads", "2", "--mlp-mult", "2",
                  "--seq-len", "64", "--dense", "--topology", "sequential",
                  "--steps", str(steps), "--batch-size", "8", "--lr", "1e-3",
                  "--seed", str(seed), "--exit-mode", "uniform", "--eval-batches", "1",
                  "--synth-task", "assoc-mqar", "--synth-pairs", "2",
                  "--synth-queries", "2", "--synth-gap", "4"]
    rec = _recorded_code_hashes()["train_depth_ar.py"]
    resolved = ({"commit": base_commit, "rel": "train_depth_ar.py",
                 "recorded_sha256": rec, "n_matching_commits": None,
                 "commit_subject": "(调用方指定)"}
                if base_commit else resolve_recorded_commit("train_depth_ar.py", rec))
    logs, meta = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        head_script = os.path.join(tmp, "train_depth_ar_base.py")
        blob = subprocess.run(["git", "-C", ROOT, "show",
                               f"{resolved['commit']}:train_depth_ar.py"],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            return {"verified": False, "base_resolution": resolved,
                    "reason": f"git show {resolved['commit']}:train_depth_ar.py 失败"}
        with open(head_script, "w") as f:
            f.write(blob.stdout)
        for tag, script in (("base", head_script),
                            ("working_tree", os.path.join(ROOT, "train_depth_ar.py"))):
            out_root = os.path.join(tmp, tag)
            t0 = time.time()
            env = dict(os.environ)
            env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
            r = subprocess.run([sys.executable, script, tag] + base_flags
                               + ["--out-root", out_root],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=timeout_s)
            wall = round(time.time() - t0, 1)
            logp = os.path.join(out_root, tag, "train_log.jsonl")
            rows = []
            if os.path.exists(logp):
                rows = [json.loads(l) for l in open(logp) if l.strip()]
            logs[tag] = rows
            meta[tag] = {"returncode": r.returncode, "wall_s": wall,
                         "n_log_lines": len(rows),
                         "stderr_tail": (r.stderr or "")[-400:] if r.returncode else None}
    seq_eq_keys = {}
    ok = True
    if len(logs["base"]) != len(logs["working_tree"]):
        ok = False
    diffs = []
    for i, (a, b) in enumerate(zip(logs["base"], logs["working_tree"])):
        for k in _TRAINER_LOG_FIELDS:
            if a.get(k) != b.get(k):
                diffs.append({"line": i, "field": k, "base": a.get(k),
                              "working_tree": b.get(k)})
        for k in ("step", "loss"):
            seq_eq_keys.setdefault(k, []).append((a.get(k), b.get(k)))
    if diffs:
        ok = False
    verified = bool(ok and meta["base"]["returncode"] == 0
                    and meta["working_tree"]["returncode"] == 0)
    return {
        "verified": verified,
        "base_resolution": resolved,
        "steps": steps,
        "compared_fields": list(_TRAINER_LOG_FIELDS),
        "n_lines": {"base": len(logs["base"]), "working_tree": len(logs["working_tree"])},
        "n_log_points_compared": min(len(logs["base"]), len(logs["working_tree"])),
        "log_points_note": ("训练器每 200 步写一行 ⇒ 1001 步 = 6 个日志点"
                            "（step 0/200/400/600/800/1000）；实际点数以 n_lines 为准"),
        "loss_sequence_base": [r.get("loss") for r in logs["base"]],
        "loss_sequence_working_tree": [r.get("loss") for r in logs["working_tree"]],
        "n_field_diffs": len(diffs),
        "field_diffs": diffs[:10],
        "runs": meta,
        "note": ("训练脚本层的行为等价证据；需要 GPU（§20.1.2 短任务例外，~1 min、<1 GB）。"
                 "`working_tree.returncode != 0` ⇒ 只能降级声明（reuse_ok_pathlm_only）"),
    }


def check(write: bool = True, with_trainer: bool = True) -> dict:
    """跑等价测试；`write=False` 时不落盘（单测用，避免测试改写已入账产物）。

    `out` 键名为 "configs" 的历史拼写保留（它就是 `RULES` 的三档）。
    """
    rec = _recorded_code_hashes()
    base_res = resolve_recorded_commit("pathlm/depth_ar.py", rec["pathlm/depth_ar.py"])
    out: dict = {"configs": [None if r == "none" else r for r in RULES],
                 "probe_seed": PROBE_SEED, "per_rule": {},
                 "comparison_base": "2631 运行时版本（按 2631/results.json::code_hashes 定位）",
                 "base_resolution": base_res}
    with tempfile.TemporaryDirectory() as tmp:
        _extract_package(tmp, base_res["commit"])
        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        # 基准版整包已解到 tmp/base_pathlm/（含 `__init__.py`）⇒ 直接按包导入
        # （相对导入 `.config` 等就能解析；不要手工注册无 `__path__` 的伪模块）。
        import base_pathlm.depth_ar as old_da

        import pathlm.depth_ar as new_da
        g = torch.Generator().manual_seed(PROBE_SEED)
        x = torch.randint(0, VOCAB, (2, BASE["seq_len"]), generator=g)

        def build(mod, rule):
            rule_val = None if rule == "none" else rule
            torch.manual_seed(0)
            m = mod.DepthARModel(mod.DepthARConfig(**{**BASE, "carry_rule": rule_val}), VOCAB)
            m.eval()
            return m

        for rule in RULES:
            mo, mn = build(old_da, rule), build(new_da, rule)
            so = {k: v.detach().clone() for k, v in mo.state_dict().items()}
            sn = {k: v.detach().clone() for k, v in mn.state_dict().items()}
            keys_same = list(so) == list(sn)
            params_bitwise = keys_same and all(torch.equal(so[k], sn[k]) for k in so)
            npo = sum(p.numel() for p in mo.parameters())
            npn = sum(p.numel() for p in mn.parameters())
            mn.load_state_dict(so, strict=True)          # 同一份权重装进新版
            with torch.no_grad():
                lo = mo(x)[1]["depth_logits"][-1]
                ln = mn(x)[1]["depth_logits"][-1]
            max_dev = float((lo - ln).abs().max().item())
            out["per_rule"][rule] = {
                "n_params_base": npo, "n_params_working_tree": npn,
                "n_params_same": bool(npo == npn),
                "params_bitwise_same_at_same_seed": bool(params_bitwise),
                "state_dict_keys_same": bool(keys_same),
                "max_abs_logit_diff_same_weights": max_dev,
                "bitwise_equal_forward": bool(max_dev == 0.0),
            }
    out["bitwise_equivalent_on_this_experiments_configs"] = all(
        v["bitwise_equal_forward"] and v["n_params_same"] for v in out["per_rule"].values())
    out["forward_layer_ok"] = out["bitwise_equivalent_on_this_experiments_configs"]
    if with_trainer:
        out["trainer"] = trainer_loss_sequence_equivalence()
    out["trainer_layer_ok"] = bool((out.get("trainer") or {}).get("verified"))
    out["coverage"] = ("pathlm/**（逐位 logits）+ train_depth_ar.py（逐字段 loss 序列）"
                       if out["trainer_layer_ok"]
                       else "仅 pathlm/**（训练脚本层未证等价 ⇒ 声明降级）")
    out["all_layers_ok"] = bool(out["forward_layer_ok"] and out["trainer_layer_ok"])
    n_lp = (out.get("trainer") or {}).get("n_log_points_compared")
    out["note"] = ("**2631 运行时版本**（按记录 sha256 定位）与**当前工作树**在**本实验用到的**"
                   "配置上逐位等价（前向三档逐位 logits + 训练脚本同一 run 的 "
                   f"{n_lp} 个日志点逐字段相同，见 trainer.n_lines）；"
                   "字节允许变（并行线的默认关闭加性改动），**开关关闭时行为必须逐位不变**；"
                   "这只覆盖本实验的配置，不代表两版在所有配置上等价")
    return out


def main() -> int:
    r = check(write=True, with_trainer="--no-trainer" not in sys.argv)
    for rule, v in r["per_rule"].items():
        print(f"{rule:6s} params {v['n_params_base']}/{v['n_params_working_tree']} "
              f"bitwise_params={v['params_bitwise_same_at_same_seed']} "
              f"max|Δlogit|={v['max_abs_logit_diff_same_weights']:.3e}")
    print("前向层等价:", r["forward_layer_ok"], "| 训练脚本层等价:", r["trainer_layer_ok"])
    print("覆盖:", r["coverage"])
    out = os.path.join(HERE, "engine_equivalence.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("落盘:", out)
    return 0 if r["all_layers_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
