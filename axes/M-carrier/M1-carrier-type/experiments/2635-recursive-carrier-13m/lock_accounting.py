#!/usr/bin/env python3
"""§20 记账的**唯一实现**：派生 `lock_note`、校验、落盘（复核 P1）。

为什么必须有这个模块（而不是把函数留在 `run.py` 里）：评测驱动在 `run.py` **之后**才知道墙钟与
峰值显存，它必须**同时**刷新 `results.json` 的 `cost.lock_accounting` **与** `cost.lock_note`——
否则会出现「note 说『§20.1.2 短任务插队例外』、accounting 说『整程持锁』」这种**同一文件自相矛盾**
（复核 P1 的可复现路径：驱动事后覆写 accounting 却不重刷 note）。

- `derive_note(acct)`：按记账派生评测口径文案（缺记账 ⇒ 如实说「未记账」，不默认某种口径）；
- `assert_consistent(cost)`：note 与 `accounting.mode/lock_taken` 必须同口径，否则 `ValueError`；
- CLI `record`：写 `lock_accounting.json`，并在**同一处**刷新 `results.json` 的两个字段后自检。

用法（驱动调用）：
  python3 lock_accounting.py record --mode section_20_lock \
      --free-before 16611 --free-after 16611 --wall-s 270 --rc 0
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {
    "section_20_lock": ("lock", True,
                        "评测由 `eval.sh` **整程持锁**（该驱动 <10 min）"),
    "section_20_1_2_short_job_exception": (
        "exception", False,
        "评测按 **§20.1.2 短任务插队例外** 执行（纯推理、**不取锁、不动锁目录**）"),
}
HEADROOM_GATE_MIB = 3072


def derive_note(acct: dict | None) -> str:
    """按**实际记账**拼装评测口径说明（不默认某种模式）。"""
    base = "训练按臂取/放 /tmp/pathlm_gpu.lock、臂间 ≥60 s（§20.1）"
    if not acct:
        return base + "；评测口径**未记账**（缺 `lock_accounting.json`，见 `analysis.md` §20）"
    mode = acct.get("mode")
    if mode in MODES:
        how = MODES[mode][2]
    else:
        how = f"评测口径未知（mode={mode!r}）"
    return (f"{base}；{how}；前后空闲读数与自身峰值见 `lock_accounting.json`"
            "（文本口径见 analysis §20）")


def assert_consistent(cost: dict) -> None:
    """**可失败的**自相矛盾检查：`lock_note` 与 `lock_accounting.mode/lock_taken` 必须同口径。"""
    acct = cost.get("lock_accounting")
    note = cost.get("lock_note") or ""
    if acct is None:
        if "整程持锁" in note or "短任务插队例外" in note:
            raise ValueError("lock_note 声称了某种评测口径，但 lock_accounting 为 None（未记账）")
        return
    kind = MODES.get(acct.get("mode"))
    if kind is None:
        raise ValueError(f"未知的 lock_accounting.mode={acct.get('mode')!r}")
    _, _taken, how = kind
    if "短任务插队例外" in note and "整程持锁" in how:
        raise ValueError("lock_note 说「§20.1.2 短任务插队例外」但 lock_accounting 是整程持锁口径")
    if "整程持锁" in note and "短任务插队例外" in how:
        raise ValueError("lock_note 说「整程持锁」但 lock_accounting 是 §20.1.2 插队例外口径")


def build_accounting(mode: str, free_before: int, free_after: int, wall_s: int,
                     returncode: int, peak_mb: float | None, extra_note: str = "") -> dict:
    """按驱动口径造记账块（`lock_taken` 由 mode 决定，不接受调用方随手填）。"""
    kind, taken, _how = MODES[mode]
    return {
        "mode": mode, "lock_kind": kind, "lock_taken": taken,
        "lock_dir_touched": taken,
        "why": extra_note or MODES[mode][2],
        "free_gpu_mem_mib_before": free_before,
        "free_gpu_mem_mib_after": free_after,
        "free_headroom_gate_mib": HEADROOM_GATE_MIB,
        "headroom_ok": bool(min(free_before, free_after) >= HEADROOM_GATE_MIB),
        "wall_s": wall_s,
        "eval_peak_gpu_mem_mb": peak_mb,
        "returncode": returncode,
    }


def assert_cross_file_consistent(acct_path: str, results_path: str) -> None:
    """**跨文件**不变量（读两份独立产物；这是写盘路径真正的护栏，复核 P2-2）：

    ① `results.json::cost.lock_accounting` == `lock_accounting.json` 的全文；
    ② `results.json::cost.lock_note` == `derive_note(lock_accounting.json)`（逐字）。
    """
    acct = json.load(open(acct_path))
    results = json.load(open(results_path))
    cost = results.get("cost") or {}
    if cost.get("lock_accounting") != acct:
        raise ValueError("跨文件不一致：results.json::cost.lock_accounting != lock_accounting.json")
    if (cost.get("lock_note") or "") != derive_note(acct):
        raise ValueError("跨文件不一致：results.json::cost.lock_note != derive_note(lock_accounting.json)")


def _write_json_atomic(path: str, obj) -> None:
    """temp + rename 写盘（避免半写文件）。"""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def record(mode: str, free_before: int, free_after: int, wall_s: int, returncode: int,
           results_name: str = "results.json", base_dir: str | None = None) -> dict:
    """写 `lock_accounting.json`，并**同时**刷新 `results.json` 的 accounting 与 note。

    写入顺序（复核 P2-3）：**先构造两个 payload → 先在内存里断言 → 再 temp+rename 写两个文件 →
    最后从磁盘读回两份产物做跨文件断言**。`results.json` 不存在时只写记账文件（无可跨文件比对
    的对象），并打印提示——**不得**留下「一边新一边旧」的产物。

    `base_dir` 默认 = 本实验目录（驱动用）；单测传临时目录，避免改写已入账产物。
    """
    d = base_dir or HERE
    res_path = os.path.join(d, results_name)
    acct_path = os.path.join(d, "lock_accounting.json")
    peak = None
    if os.path.exists(res_path):
        try:
            peak = json.load(open(res_path))["cost"]["eval_peak_gpu_mem_mb"]
        except Exception:
            peak = None
    out = build_accounting(mode, free_before, free_after, wall_s, returncode, peak)
    # ① 先构造两个 payload（内存里）
    res = None
    if os.path.exists(res_path):
        res = json.load(open(res_path))
        res.setdefault("cost", {})
        res["cost"]["lock_accounting"] = out
        res["cost"]["lock_note"] = derive_note(out)      # **同一处**刷新两个字段
        assert_consistent(res["cost"])                   # 结构自检（同源，仅防手改）
    # ② 再写两个文件（temp+rename；每个文件原子）
    _write_json_atomic(acct_path, out)
    if res is not None:
        _write_json_atomic(res_path, res)
    # ③ 最后**从磁盘读回两份独立产物**做跨文件断言（写盘路径的真护栏）
    if res is not None:
        assert_cross_file_consistent(acct_path, res_path)
        print("results.json::cost.{lock_accounting,lock_note} 已同步（跨文件读回自检通过）")
    else:
        print(f"[警告] {res_path} 不存在 ⇒ 只写记账文件（无可跨文件比对的对象）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--mode", choices=sorted(MODES), required=True)
    r.add_argument("--free-before", type=int, required=True)
    r.add_argument("--free-after", type=int, required=True)
    r.add_argument("--wall-s", type=int, required=True)
    r.add_argument("--rc", type=int, default=0)
    r.add_argument("--results-name", default="results.json")
    args = ap.parse_args()
    out = record(args.mode, args.free_before, args.free_after, args.wall_s, args.rc,
                 args.results_name)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
