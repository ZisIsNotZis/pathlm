"""冒烟步时读数（只服务预算判定，不作墙钟结论）。

SCHEMA §17.5：冒烟窗口被进程启动/数据加载主导，**按短窗口外推的墙钟字段不得留在
证据文件里**。所以本脚本只落两件**测量值**：
  ① `inner_ms_per_step`：训练循环内（`train()` 的 t0 之后）每步毫秒数——用训练日志的
     `min` 字段取窗口差，**不含**数据加载/建模型；
  ② `peak_gpu_mem_mb` / `params` / `intervention`：`train_m1.py` 落盘的原值。
并落一行 `budget_note` 说明「全跑墙钟以全跑的 `wall_minutes` 为准；本读数只用于决定
是否降步」。**不外推**、不写「预计 12k 分钟」之类的字段。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("stock", "l1", "divl1")


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = {"smoke_steps": steps, "arms": {}}
    for arm in ARMS:
        root = os.path.join(HERE, "smoke", f"smoke_{arm}")
        res = json.load(open(os.path.join(root, "results.json")))
        lines = [json.loads(x) for x in open(os.path.join(root, "train_log.jsonl")) if x.strip()]
        window_steps = max(lines[-1]["step"] - lines[0]["step"], 1)
        inner_min = max(lines[-1]["min"] - lines[0]["min"], 1e-9)
        out["arms"][arm] = {
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
