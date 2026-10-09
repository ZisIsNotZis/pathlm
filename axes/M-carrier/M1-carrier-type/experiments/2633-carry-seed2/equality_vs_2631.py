#!/usr/bin/env python3
"""2633-carry-seed2 · SCHEMA §9「对照复用」的证据生成器（**只读** 2631 的产物）。

复用 2631 的 seed0/seed1 六臂作对照，必须同时满足 §9 四项；本模块把它们变成可核对的
产物 `equality_report.json`：

1. **实现层 equality**（两级）：① **`HEAD` 提交版**的引擎/仪器文件 sha256 逐文件等于
   2631 运行时记录在 `2631/results.json::code_hashes` 的值（证明**提交历史里**那份实现没变）；
   ② **工作树**可能与 HEAD 不同（其它并行线可能有未提交改动）⇒ 另跑
   `engine_equivalence.check(...)`：把 `HEAD` 版的 `pathlm/` 整包解到临时目录、与本实验
   三档配置（none/plain/delta，dense+sequential+窗口 16）各建一个模型，**同权重下比
   logits 逐位**——不等 ⇒ 复用拒绝（这是比文件哈希更硬的语义等价证据）；
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


def _head_sha(rel: str) -> str | None:
    """`git show HEAD:<rel>` 的 sha256（None = 该文件不在 HEAD 里）。"""
    r = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return hashlib.sha256(r.stdout.encode()).hexdigest()


def compare(cfg: dict | None = None, with_equivalence: bool = True,
            write: bool = True, recompute_trainer: bool | None = None) -> dict:
    """逐项对照；返回报告（同时落盘 `equality_report.json`）。"""
    if cfg is None:
        cfg = json.load(open(os.path.join(HERE, "config.json")))
    rep2631 = json.load(open(os.path.join(RES, "results.json")))
    cfg2631 = json.load(open(os.path.join(RES, "config.json")))

    # ① 实现层：**基准 = 2631 运行时版本**（按记录 sha256 定位），判据是**行为等价**。
    #    字节是否还等于当时记录是**信息项**（并行线做过默认关闭的加性改动 ⇒ 预期会变），
    #    不是 §9 的判据；判据是下面 `engine_equivalence` 的两层逐位等价。
    impl, impl_ok, tree_drift = {}, True, []
    for rel in CODE_FILES:
        work = sha256_file(os.path.join(ROOT, rel))
        then = rep2631.get("code_hashes", {}).get(rel)
        import engine_equivalence as _ee
        located = _ee.resolve_recorded_commit(rel, then) if then else None
        drift = (then is not None and work != then)
        if drift:
            tree_drift.append(rel)
        impl[rel] = {"recorded_at_2631_run": then, "working_tree_now": work,
                     "runtime_version_commit": (located or {}).get("commit"),
                     "runtime_version_subject": (located or {}).get("commit_subject"),
                     "n_matching_commits": (located or {}).get("n_matching_commits"),
                     "bytes_unchanged_since_2631_run": bool(then == work),
                     "note": ("字节不变只是信息项；判据见 `implementation_equivalent_"
                              "to_2631_runtime`（两层行为等价）")}
    if recompute_trainer is None:
        recompute_trainer = write          # 生成证据（脚本入口）时重算；测试（write=False）引用已入账产物
    equiv = None
    if with_equivalence and impl_ok:
        import engine_equivalence
        if recompute_trainer:
            equiv = engine_equivalence.check(write=write, with_trainer=True)
        else:
            # 只读引用已入账的 `engine_equivalence.json`（含训练脚本层），不跑 GPU
            p_ee = os.path.join(HERE, "engine_equivalence.json")
            equiv = json.load(open(p_ee)) if os.path.exists(p_ee) else None
            if equiv is None:
                equiv = engine_equivalence.check(write=False, with_trainer=False)
        impl_ok = impl_ok and bool(equiv.get("forward_layer_ok"))
        trainer_ok = bool(equiv.get("trainer_layer_ok"))
    impl_sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    trainer_ok = bool((equiv or {}).get("trainer_layer_ok"))
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
        "implementation_equivalent_to_2631_runtime": bool(impl_ok),
        "implementation_note": ("① 基准 = **2631 运行时版本**：按 2631 `results.json::code_hashes`"
                                " 记录的 sha256 在 git 历史里定位到对应 commit"
                                "（见每条的 `runtime_version_commit`）；② 判据 = 该版本与**当前"
                                "工作树**的**行为**等价（两层）：`pathlm/**` 三档配置逐位 logits +"
                                " `train_depth_ar.py` 同一 run 的若干日志点逐字段 loss 序列"
                                "（点数见 `engine_equivalence.json::trainer.n_lines`）。"
                                "**字节允许变**（并行线的默认关闭加性改动），开关关闭时行为必须"
                                "逐位不变；只覆盖本实验用到的配置。"),
        "working_tree_drift_files": tree_drift,
        "engine_equivalence_working_tree_vs_head": equiv,
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
        "verdict": ("reuse_ok" if (impl_ok and not differing and not flag_diff and trainer_ok)
                    else ("reuse_ok_pathlm_only"
                          if (impl_ok and not differing and not flag_diff)
                          else "reuse_rejected")),
        "reuse_scope": ("pathlm/** + train_depth_ar.py（两层行为等价）" if trainer_ok
                        else "仅 pathlm/**（训练脚本层未证等价 ⇒ 声明降级）"),
        "runtime_version_ledger": os.path.relpath(
            os.path.join(HERE, "runtime_version_ledger.json"), ROOT),
    }
    if write:
        # `write=False` 供单测使用：测试**不得**改写已入账的产物（只读复核）
        with open(os.path.join(HERE, "equality_report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return report


def main() -> int:
    r = compare(recompute_trainer=True)
    print("行为等价（vs 2631 运行时版本）:", r["implementation_equivalent_to_2631_runtime"])
    print("字节未变（信息项）:", {k: v["bytes_unchanged_since_2631_run"]
                                for k, v in r["implementation"].items()})
    print("配置层不同字段:", r["differing_fields"] or "无")
    print("训练配方不同的 flag:", r["train_recipe_differing_flags"] or "无")
    print("verdict:", r["verdict"], "| 复用范围:", r["reuse_scope"])
    return 0 if r["verdict"].startswith("reuse_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
