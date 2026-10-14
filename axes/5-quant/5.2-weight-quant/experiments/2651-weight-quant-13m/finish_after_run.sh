#!/usr/bin/env bash
# 等待本目录 `run.py` 落盘后自动生成 `analysis.md`（run 未落盘则如实登记，不伪造数字）。
# 用法：nohup bash finish_after_run.sh >/dev/null 2>&1 &
# 背景：CPU 口径一次全量跑约 25–30 min（每臂 ~6.5 min × 4 臂），收尾 writer 的会话可能先被切断
# ⇒ 这个 watcher 保证「数字一落盘，判读就跟着落盘」，不需要再有人守着。
set -u
cd "$(dirname "$0")"
MAX_POLLS=240          # 240 × 10 s = 40 min 上限
for _ in $(seq 1 "$MAX_POLLS"); do
    [ -f results.json ] && break
    sleep 10
done
if [ -f results.json ]; then
    python3 fill_analysis.py > fill_analysis.log 2>&1
    rc=$?
    echo "=== fill_analysis rc=$rc $(date -Iseconds) ===" >> fill_analysis.log
    if [ "$rc" -ne 0 ]; then
        echo "=== fill_analysis 失败：analysis.md 未更新，见本文件 ===" >> fill_analysis.log
    fi
else
    python3 - <<'PY' > blocked_run.json
import json, os, time
print(json.dumps({
    "kind": "run_did_not_finish",
    "started_at": os.environ.get("RUN_STARTED_AT"),
    "waited_seconds": 2400,
    "attempts": 1,
    "device": "cpu",
    "gpu_lock": "not_taken（CPU 口径，SCHEMA §20 不适用）",
    "reason": "results.json 在 40 min 内未出现（run.py 未完成或已死）；本文件如实登记，未伪造任何数字。",
    "next_action": "看 run_log.txt 末尾的 actual 输出；若是 OOM/KeyboardInterrupt/异常，按日志修最小项后重跑：python3 run.py --device cpu",
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}, ensure_ascii=False, indent=1))
PY
fi
