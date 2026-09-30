"""2613 · 层间拓扑 —— 用 checkpoint 重新推导 §10 活性块（字段名修正后的一致性重跑）。

背景：`topology_liveness` 的字段在四臂落盘后修正过一次——原来的 `identity_init`
对**无专属参数**的 `parallel-dual-ln` 臂给出 True，读起来像「该臂起步与基线逐位等价」
（实际它起步不等于基线，见 `smoke/smoke.json::identity_start_bitwise_vs_sequential`）。
修正后拆成两个可逐位核对的布尔量：

- `params_identical_to_sequential`：参数量是否与基线逐位相同；
- `start_bitwise_equals_sequential`：起步是否与基线逐位等价。

`topology_liveness` 只是 `cfg.topology` + checkpoint 里 `layer_mix` 的纯函数，
因此从 checkpoint 重新推导是**等价重算**，不是补造数据。本脚本把这四个臂的
`topology_liveness`（per-run `results.json`）与实验级 `results.json::arms[*].intervention`
里的对应字段重算写回（保留 `branch_activity` 等评测得到的量，不动任何数值结果）。

用法：`python3 rederive_liveness.py`（CPU，<10 s；从仓库根目录运行同效）。
"""

import json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel, topology_liveness

D, L, H, SEQ, MLP = 256, 8, 8, 256, 4
ARMS = ["sequential", "parallel", "parallel-dual-ln", "dense"]


def rebuild(topo: str) -> dict:
    res_path = os.path.join(HERE, "runs", f"{topo}-12k-s0", "results.json")
    res = json.load(open(res_path))
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                        mlp_mult=MLP, dense=True, topology=topo)
    model = DepthARModel(cfg, res["vocab_size"])
    model.load_state_dict(torch.load(
        os.path.join(HERE, "runs", f"{topo}-12k-s0", "model.pt"),
        map_location="cpu"))
    return res, topology_liveness(model)


def main():
    exp_path = os.path.join(HERE, "results.json")
    exp = json.load(open(exp_path))
    for topo in ARMS:
        res, live = rebuild(topo)
        assert res["params"] == exp["arms"][topo]["params"] == \
            sum(p.numel() for p in DepthARModel(
                DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                              mlp_mult=MLP, dense=True, topology=topo),
                res["vocab_size"]).parameters()), topo
        # per-run results.json：整块替换为修正后的活性块
        old = res.get("topology_liveness", {})
        res["topology_liveness"] = live
        json.dump(res, open(os.path.join(HERE, "runs", f"{topo}-12k-s0",
                                        "results.json"), "w"),
                  indent=2, ensure_ascii=False)
        # 实验级 results.json：只替换活性字段，保留 branch_activity 等评测量
        arm = exp["arms"][topo]["intervention"]
        branch = arm.get("branch_activity")
        arm.clear()
        arm.update(live)
        if branch is not None:
            arm["branch_activity"] = branch
        print(f"[{topo}] liveness 重算：{json.dumps(live)[:120]}"
              f"{'（branch_activity 保留）' if branch else ''}", flush=True)
        # 幂等：重复跑时 old 已是修正后的键（差异只允许是旧字段名 identity_init）
        assert set(old) - set(live) <= {"identity_init"}, sorted(set(old) - set(live))
        assert set(live) - set(old) <= {"params_identical_to_sequential",
                                        "start_bitwise_equals_sequential"}, \
            sorted(set(live) - set(old))
    json.dump(exp, open(exp_path, "w"), indent=2, ensure_ascii=False)
    print("[done] 已写回 per-run results.json ×4 与实验级 results.json")


if __name__ == "__main__":
    main()
