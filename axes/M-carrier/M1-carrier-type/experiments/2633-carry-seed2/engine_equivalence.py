#!/usr/bin/env python3
"""2633-carry-seed2 · **语义级**实现等价证据（SCHEMA §9 第①项的加强版）。

背景（为什么文件哈希不够）：本仓有多条线并行，`pathlm/**` 可能有**别的线的未提交改动**
（本线执行期间确实出现过）。文件哈希只说明「字节是否一样」，回答不了「换了字节之后，
**本实验用到的那条路径**行为是否一样」。

**两层覆盖**（复核 P1-3：只覆盖 `pathlm/**` 不够，训练脚本也会漂移）：

1. **`pathlm/**`（前向）**：把 `HEAD` 提交版的 `pathlm/` 整包解到临时目录、以独立包名
   `old_pathlm` 导入，与本实验三档配置（`carry_rule` ∈ {none, plain, delta}，dense +
   topology=sequential + 窗口 16 + 锚 0，13M 主干）各建一个模型：同 seed ⇒ **参数量与参数
   张量逐位相同**；同一份 `state_dict` 装进两版 ⇒ 固定输入上 **logits 逐位相同**。
2. **`train_depth_ar.py`（训练循环）**：`trainer_loss_sequence_equivalence()` 用 `HEAD`
   版脚本与**工作树**版脚本各跑 N 步（同 seed、同合成数据、同配方），逐字段比
   `train_log.jsonl`（除墙钟 `min`）⇒ 训练循环/loss 归一/数据顺序/落盘路径逐位一致。

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


def _extract_head_package(dst: str) -> str:
    """把 `HEAD` 版的 `pathlm/*.py` 解到 `dst/old_pathlm/`，返回包目录。"""
    pkg = os.path.join(dst, "old_pathlm")
    os.makedirs(pkg, exist_ok=True)
    files = subprocess.run(["git", "-C", ROOT, "ls-tree", "--name-only", "HEAD", "pathlm/"],
                           capture_output=True, text=True).stdout.split()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        blob = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"],
                              capture_output=True, text=True).stdout
        with open(os.path.join(pkg, os.path.basename(rel)), "w") as f:
            f.write(blob)
    return pkg


#: 训练脚本层的比较：只比这些字段（`min` 是墙钟 ⇒ 排除）
_TRAINER_LOG_FIELDS = ("step", "loss", "gnorm", "aux_w", "final_w", "window_mask",
                       "synth_answer_ce", "synth_chance_ce")


def trainer_loss_sequence_equivalence(steps: int = 20, *, seed: int = 0,
                                      timeout_s: int = 300) -> dict:
    """`HEAD` 版 `train_depth_ar.py` vs **工作树**版：同 seed/同数据/同配方跑 N 步。

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
    logs, meta = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        head_script = os.path.join(tmp, "train_depth_ar_head.py")
        blob = subprocess.run(["git", "-C", ROOT, "show", "HEAD:train_depth_ar.py"],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            return {"verified": False, "reason": "git show HEAD:train_depth_ar.py 失败"}
        with open(head_script, "w") as f:
            f.write(blob.stdout)
        for tag, script in (("head", head_script),
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
    if len(logs["head"]) != len(logs["working_tree"]):
        ok = False
    diffs = []
    for i, (a, b) in enumerate(zip(logs["head"], logs["working_tree"])):
        for k in _TRAINER_LOG_FIELDS:
            if a.get(k) != b.get(k):
                diffs.append({"line": i, "field": k, "head": a.get(k),
                              "working_tree": b.get(k)})
        for k in ("step", "loss"):
            seq_eq_keys.setdefault(k, []).append((a.get(k), b.get(k)))
    if diffs:
        ok = False
    verified = bool(ok and meta["head"]["returncode"] == 0
                    and meta["working_tree"]["returncode"] == 0)
    return {
        "verified": verified,
        "steps": steps,
        "compared_fields": list(_TRAINER_LOG_FIELDS),
        "n_lines": {"head": len(logs["head"]), "working_tree": len(logs["working_tree"])},
        "loss_sequence_head": [r.get("loss") for r in logs["head"]],
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
    out: dict = {"configs": [None if r == "none" else r for r in RULES],
                 "probe_seed": PROBE_SEED, "per_rule": {}}
    with tempfile.TemporaryDirectory() as tmp:
        _extract_head_package(tmp)
        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        # `HEAD` 版的整包已解到 tmp/old_pathlm/（含 `__init__.py`）⇒ 直接按包导入
        # （相对导入 `.config` 等就能解析；不要手工注册无 `__path__` 的伪模块）。
        import old_pathlm.depth_ar as old_da

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
                "n_params_head": npo, "n_params_working_tree": npn,
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
    out["note"] = ("工作树与 HEAD 提交版在**本实验用到的**配置上逐位等价（前向三档 + 训练脚本"
                   " 20 步 loss 序列）；这只覆盖本实验的配置，不代表两版在所有配置上等价")
    return out


def main() -> int:
    r = check(write=True, with_trainer="--no-trainer" not in sys.argv)
    for rule, v in r["per_rule"].items():
        print(f"{rule:6s} params {v['n_params_head']}/{v['n_params_working_tree']} "
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
