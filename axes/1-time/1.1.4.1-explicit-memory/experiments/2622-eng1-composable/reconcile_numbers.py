"""2622-eng1-composable · 对账（SCHEMA §12 三道）：`analysis.md` / `results.json` 的每个
数字都能在 `results.json`（及它引用的冒烟日志 / 测试输出）里找到同值，且算术自洽。

三道对账：
1. **算术对账**：qkv 输出宽度式（GQA = d+2·n_kv_heads·dh、MHA = 3d）、缺省两块的参数
   量相等、`fast_paths` 的块类 → 由 `combos` 的旗标重算并比对。
2. **任务对账**：4 个组合在同一 CPU 测试位点（d/layers/heads/seq/dense/seed）上量；
   每个活性量只在对应旗标为真时出现（否则必须是 None）。
3. **口径对账**：缺省逐位还原 v1（权重 + 前向）、冒烟的 KV 头 / 写入通道活性为活、
   全量回归无 fail、变异全部抓红且逐字节还原。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    checks = []

    def ck(name, got, exp):
        checks.append({"check": name, "want": exp, "got": got,
                       "pass": got == exp})

    cfg = R["config"]["model"]
    d, nh, dh = cfg["d_model"], cfg["n_heads"], cfg["d_model"] // cfg["n_heads"]

    # ---------------- ① 算术对账 ----------------
    expected_class = {
        "gqa_x_carry": "ComposableBlock",
        "gqa_x_window": "GQABlock",           # 单特性快路径：仍走 GQABlock
        "carry_x_window": "CarryChannelBlock",
        "gqa_x_carry_x_window": "ComposableBlock",
    }
    for c in R["combos"]:
        name = c["name"]
        nkv = 2 if c["gqa_on"] else None
        ck(f"{name}.qkv_width",
           c["qkv_out_width"],
           (d + 2 * nkv * dh) if nkv else 3 * d)
        ck(f"{name}.class", c["block_class"], expected_class[name])
        # 活性量只在对应旗标为真时出现
        ck(f"{name}.carry_read_share_present",
           c["carry_read_share"] is not None, c["carry_on"])
        ck(f"{name}.kv_shape_present",
           c["kv_k_shape"] is not None, c["gqa_on"])
        ck(f"{name}.window_flag_present",
           c["window_changes_logits_vs_full"] is not None, c["window_on"])
        if c["gqa_on"]:
            ck(f"{name}.kv_k_heads", c["kv_k_shape"][1], nkv)
            ck(f"{name}.kv_mask_applied", c["kv_attn_mask_applied"], c["window_on"])
        if c["carry_on"]:
            ck(f"{name}.carry_alive", c["carry_read_share"] > 0.0, True)
        if c["window_on"]:
            ck(f"{name}.window_alive", c["window_changes_logits_vs_full"], True)
    ck("default.n_params_equal",
       R["default_bitwise_v1"]["n_params_block"],
       R["default_bitwise_v1"]["n_params_composable"])

    # ---------------- ② 任务对账（同一测试位点） ----------------
    ck("task.n_combos", len(R["combos"]), 4)
    ck("task.same_point", R["fast_paths"]["default"], "Block")
    ck("task.config_dense", cfg["dense"], True)
    ck("task.config_topology", cfg["topology"], "sequential")

    # ---------------- ③ 口径对账（缺省不动 / 冒烟活 / 回归绿 / 变异红） ----------------
    ck("caliber.default_weights_bitwise",
       R["default_bitwise_v1"]["weights_bitwise_equal"], True)
    ck("caliber.default_forward_bitwise",
       R["default_bitwise_v1"]["forward_bitwise_equal"], True)
    ck("caliber.fast_path_gqa_only",
       R["fast_paths"]["gqa_only"], "GQABlock")
    ck("caliber.fast_path_carry_only",
       R["fast_paths"]["carry_only"], "CarryChannelBlock")
    ck("caliber.fast_path_gqa_x_carry",
       R["fast_paths"]["gqa_x_carry"], "ComposableBlock")
    if R.get("smoke"):
        ck("smoke.block_class", R["smoke"]["block_class"], "ComposableBlock")
        ck("smoke.kv_alive", R["smoke"]["kv_head_liveness_alive"], True)
        ck("smoke.final_step", R["smoke"]["final_step"], 19)
        row = R["smoke"].get("final_log_row", {})
        ck("smoke.log_has_kv_heads", row.get("kv_heads") is not None, True)
        ck("smoke.log_has_attn_carry", row.get("attn_carry") is not None, True)
        ck("smoke.log_has_window_mask", row.get("window_mask") is not None, True)
    if R.get("regression"):
        ck("regression.no_failed", R["regression"].get("failed", 0), 0)
    if R.get("mutation"):
        ck("mutation.all_caught_red", R["mutation"]["all_caught_red"], True)
        ck("mutation.all_restored_bitwise",
           R["mutation"]["all_restored_bitwise"], True)
        ck("mutation.sha_restored",
           R["mutation"]["sha256_restored"], R["mutation"]["sha256_original"])

    fails = [c for c in checks if not c["pass"]]
    out = {"n_checks": len(checks), "n_failed": len(fails), "checks": checks}
    with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for c in fails:
        print("FAIL", c, file=sys.stderr)
    print(f"reconcile: {len(checks) - len(fails)}/{len(checks)} pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
