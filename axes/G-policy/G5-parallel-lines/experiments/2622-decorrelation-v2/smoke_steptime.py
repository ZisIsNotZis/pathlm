"""冒烟步时读数（只服务预算判定，不作墙钟结论）。

SCHEMA §17.5：冒烟窗口被进程启动/数据加载主导，**按短窗口外推的墙钟字段不得留在
证据文件里**。所以本脚本只落**测量值**：训练循环内的每步毫秒数（用训练日志 `min`
字段取窗口差，不含数据加载/建模型）+ 每臂峰值显存/参数/活性量。
并落一行 `budget_note`。**不外推**。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE_ARMS = (("13m", "stock"), ("13m", "l1"), ("13m", "repr"),
              ("50m", "l1"), ("50m", "repr"))


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = {"smoke_steps": steps, "arms": {}}
    for scale, arm in SMOKE_ARMS:
        root = os.path.join(HERE, "smoke", f"smoke{scale}_{arm}")
        if not os.path.exists(os.path.join(root, "results.json")):
            continue
        res = json.load(open(os.path.join(root, "results.json")))
        lines = [json.loads(x) for x in open(os.path.join(root, "train_log.jsonl")) if x.strip()]
        window_steps = max(lines[-1]["step"] - lines[0]["step"], 1)
        inner_min = max(lines[-1]["min"] - lines[0]["min"], 1e-9)
        out["arms"][f"{scale}/{arm}"] = {
            "params": res.get("params"),
            "peak_gpu_mem_mb": res.get("peak_gpu_mem_mb"),
            "smoke_wall_min": res.get("wall_minutes"),
            "inner_ms_per_step": round(inner_min * 60_000 / window_steps, 1),
            "window": {"from_step": lines[0]["step"], "to_step": lines[-1]["step"],
                       "inner_minutes": round(inner_min, 4)},
            "intervention": res.get("intervention"),
            "first_line": lines[0], "last_line": lines[-1],
        }
    out["budget_note"] = ("全部墙钟以全跑 results.json::wall_minutes 为准（本读数只用于"
                          "决定是否降步；SCHEMA §17.5 禁止按短窗口外推墙钟字段）")
    path = os.path.join(HERE, "smoke", "steptime.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
