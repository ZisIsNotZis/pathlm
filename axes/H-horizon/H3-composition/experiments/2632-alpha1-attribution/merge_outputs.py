"""2632 · 产物聚合（`run_all.sh` 的固定合并步；此前是临时命令，无法复现）。

两个模式：

- `--mode diag`：把每个 checkpoint 单独跑出的 `diag.py --out <dir>` 结果合并成仓库里的
  `results.json` / `diag_records.jsonl` / `run_log.txt`；并从 `diag_records.jsonl` **重算**
  `wilson_ci_95` 注入（准确率走二项/Wilson，不借 bpc 的 0.014 带）。
- `--mode probes`：把每个 checkpoint 的 `probes.py`（`PROBE_OUT=<dir>`）输出合并成
  `probes_output.json`；若 `probes_p1.json` 存在，用它覆盖 P1（P1 曾有一处 off-by-one，
  由 `p1_only.py` 修正后单独落盘）。

用法：
  python3 merge_outputs.py --mode diag   --dirs /tmp/diag_run/13m_s0 ... --out .
  python3 merge_outputs.py --mode probes --dirs /tmp/probe_run/13m_s0 ... --out .
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import diag  # noqa: E402

#: 早先 `probes.py` 增量侧 off-by-one 时记录的一对**不相等**值（4 offsets, n=256）。
#: 冻结为常量供 `P1_note` 引用（复核选项②），使 merge 输出**对输入确定**——`probes.py` 已
#: 修复，重跑不会再产生这对值，若在 note 里嵌本次运行值，输出会随修复与否而变。
P1_HISTORICAL_PREFIX = {
    "13m_s0": {"batched": 0.59765625, "incremental": 0.58984375, "n": 256},
    "13m_s1": {"batched": 0.609375, "incremental": 0.59765625, "n": 256},
    "53m_s0": {"batched": 0.6171875, "incremental": 0.609375, "n": 256},
}


def merge_diag(dirs: list[str], out: str) -> None:
    merged = {"config": None, "checkpoints": {}}
    records: list[str] = []
    logs: list[str] = []
    for d in dirs:
        with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
            r = json.load(f)
        merged["config"] = r["config"]
        merged["checkpoints"].update(r["checkpoints"])
        with open(os.path.join(d, "diag_records.jsonl"), encoding="utf-8") as f:
            records.extend(f.readlines())
        logp = os.path.join(d, "run_log.txt")
        if os.path.exists(logp):
            logs.append(open(logp, encoding="utf-8").read())
    # wilson_ci_95：从逐位置原始记录重算（准确率口径）
    rows_by: dict[str, list] = collections.defaultdict(list)
    for line in records:
        row = json.loads(line)
        rows_by[row["ckpt"]].append(row)
    for ck, rows in rows_by.items():
        summ = diag.summarize(rows)
        ta = merged["checkpoints"][ck]["teacher_anchor"]
        for group in ("quality_vs_true_t2", "accept_vs_deploy_target"):
            for name, d in summ[group].items():
                assert abs(d["value"] - ta[group][name]["value"]) < 1e-9, (ck, group, name)
        ta["wilson_ci_95"] = summ["wilson_ci_95"]
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "diag_records.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(records)
    with open(os.path.join(out, "run_log.txt"), "w", encoding="utf-8") as f:
        f.writelines(logs)
    print(f"[merge diag] {len(merged['checkpoints'])} ckpts, {len(records)} records -> {out}")


def merge_probes(dirs: list[str], out: str) -> None:
    merged: dict = {}
    for d in dirs:
        p = os.path.join(d, "probes_output.json")
        if os.path.exists(p):
            merged.update(json.load(open(p, encoding="utf-8")))
    p1 = os.path.join(HERE, "probes_p1.json")
    if os.path.exists(p1):
        corrected = json.load(open(p1, encoding="utf-8"))
        for ck in merged:
            if ck in corrected:
                merged[ck]["P1_inc_vs_batched_next_token_acc"] = corrected[ck]
                merged[ck]["P1_note"] = (
                    "P1 values come from probes_p1.json (p1_only.py, 12 offsets, n=768). "
                    "The earlier probes.py pass had an off-by-one (incremental scored "
                    "positions 1..L-1 while dividing by L); probes.py is now fixed so both "
                    "sides agree exactly. Historical pre-fix pair (4 offsets, n=256): "
                    f"{P1_HISTORICAL_PREFIX.get(ck)}.")
    with open(os.path.join(out, "probes_output.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[merge probes] {len(merged)} ckpts -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("diag", "probes"), required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    (merge_diag if a.mode == "diag" else merge_probes)(a.dirs, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
