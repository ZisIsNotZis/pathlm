#!/usr/bin/env python3
"""2633-carry-seed2 · SCHEMA §9「对照复用」的证据生成器（**只读** 2631 的产物）。

复用 2631 的 seed0/seed1 六臂作对照，必须同时满足 §9 四项；本模块把它们变成可核对的
产物 `equality_report.json`：

1. **实现层 equality**：当前仓库的引擎/仪器文件 sha256 **逐文件等于** 2631 运行时记录在
   `2631/results.json::code_hashes` 的值（不是「我记得没改」，而是逐位比对）；
2. **配置层 equality**：本实验 `config.json` 与 2631 `config.json` 的共享字段逐字段对照
   （`model`/`data`/`tokens`/`task`/`visibility`/`eval`/`aux_ood`/`sensitivity_K4`），
   以及 2631 `train_arms.sh` 的 `BASE` 与本实验 `train_recipe_seed2.cmd` 的逐 flag 对照；
   唯一允许不等的是 `--seed`（本实验新增的档）；
3. **口径标注**：写进报告（seed0/1 = 历史同配置对照；seed2 = 本次新训；同快照 12k）；
4. **不等即回退**：任一项不通过 ⇒ `verdict = "reuse_rejected"`，调用方必须改用自包含双跑。

用法：python3 equality_vs_2631.py        # 打印并落盘 equality_report.json
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RES = os.path.abspath(os.path.join(HERE, "..", "2631-carry-reseed"))

SHARED_CFG_KEYS = ("device", "model", "data", "tokens", "task", "visibility",
                   "eval", "aux_ood", "sensitivity_K4")
CODE_FILES = ("pathlm/depth_ar.py", "train_depth_ar.py",
              "tools/evalkit/longrange_tasks.py", "tools/evalkit/carry_channel.py",
              "tools/evalkit/retention.py")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flags(cmd_text: str) -> dict[str, str]:
    """把命令行文本拆成 `{flag: value}`（无值 flag 记为空串）；忽略可执行名。"""
    toks = shlex.split(cmd_text.replace("\\\n", " "))
    out: dict[str, str] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[t] = toks[i + 1]
                i += 2
            else:
                out[t] = ""
                i += 1
        else:
            i += 1
    return out


def compare(cfg: dict | None = None) -> dict:
    """逐项对照；返回报告（同时落盘 `equality_report.json`）。"""
    if cfg is None:
        cfg = json.load(open(os.path.join(HERE, "config.json")))
    rep2631 = json.load(open(os.path.join(RES, "results.json")))
    cfg2631 = json.load(open(os.path.join(RES, "config.json")))

    # ① 实现层
    impl, impl_ok = {}, True
    for rel in CODE_FILES:
        now = sha256_file(os.path.join(ROOT, rel))
        then = rep2631.get("code_hashes", {}).get(rel)
        same = (now == then)
        impl_ok = impl_ok and same
        impl[rel] = {"now": now, "at_2631_run": then, "identical": bool(same)}
    impl_sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    # ② 配置层
    shared = {k: {"2633": cfg.get(k), "2631": cfg2631.get(k),
                  "equal": cfg.get(k) == cfg2631.get(k)} for k in SHARED_CFG_KEYS}
    differing = [k for k, v in shared.items() if not v["equal"]]

    base_txt = open(os.path.join(RES, "train_arms.sh")).read()
    base_line = base_txt.split('BASE="', 1)[1].split('"', 1)[0]
    base = _flags(base_line)
    mine = _flags(cfg["train_recipe_seed2"]["cmd"])
    flags = {}
    for f in sorted(set(base) | set(mine)):
        if f in ("--carry-write",):            # 臂差异（none/plain/delta），非配方差异
            flags[f] = {"2633": mine.get(f), "2631": base.get(f),
                        "equal": True, "note": "臂变量（占位符），非配方差异"}
            continue
        if f == "--out-root":
            flags[f] = {"2633": mine.get(f), "2631": base.get(f), "equal": False,
                        "note": "输出目录按实验不同 ⇒ 允许不等（不进 equality 判据）"}
            continue
        flags[f] = {"2633": mine.get(f), "2631": base.get(f),
                    "equal": mine.get(f) == base.get(f)}
    flag_diff = [f for f, v in flags.items()
                 if not v["equal"] and f != "--seed" and f != "--out-root"]

    # 复用臂的 checkpoint 哈希（2631 config 里记的就是当时落盘的权重哈希）
    ck_hashes = {a["name"]: a["sha256"] for a in cfg2631["arms"]}

    report = {
        "generated_by": "2633-carry-seed2/equality_vs_2631.py",
        "target": os.path.relpath(RES, ROOT),
        "implementation": impl,
        "implementation_identical": bool(impl_ok),
        "git_head": impl_sha,
        "shared_config": shared,
        "differing_fields": differing,
        "train_recipe_note": ("2631 train_arms.sh 的 BASE 与本实验 train_recipe_seed2.cmd "
                              "逐 flag 对照；--seed 是本实验**新增的档**，--out-root 按实验不同"),
        "train_recipe_flags": flags,
        "train_recipe_differing_flags": flag_diff,
        "reused_arm_checkpoint_sha256_at_2631_run": ck_hashes,
        "caliber_note": ("seed0/1 = 2631 的历史同配置对照（同一实现 sha256 + 同 token 预算 "
                         "98.3M + 同评测仪器）；seed2 = 本次新训；七/九臂同快照口径（12k 步）"),
        "verdict": ("reuse_ok" if (impl_ok and not differing and not flag_diff)
                    else "reuse_rejected"),
    }
    with open(os.path.join(HERE, "equality_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report


def main() -> int:
    r = compare()
    print("实现层逐文件相同:", r["implementation_identical"])
    print("配置层不同字段:", r["differing_fields"] or "无")
    print("训练配方不同的 flag:", r["train_recipe_differing_flags"] or "无")
    print("verdict:", r["verdict"])
    return 0 if r["verdict"] == "reuse_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
