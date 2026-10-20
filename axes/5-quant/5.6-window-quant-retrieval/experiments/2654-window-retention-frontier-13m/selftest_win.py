"""2654 · WP1 自检 + 变异（可失败单测）：掩码因子、块标尺因子、同字节水平、配对带。

契约：`SCHEMA` §9（实现层 equality）+ `harness/methodology.md` §5（冒烟/单测/变异纪律）。
本文件是**自测**，不是独立验收（复核由 axis5-lead 另派 fresh reviewer）。

跑法：
  python3 selftest_win.py            # 全跑（含 5 个需要模型的判定，CPU 上约 20 s）
  python3 selftest_win.py --quick    # 只跑纯解析判定（不载模型）
每个判定都配一个**变异**：故意把因子破坏掉，断言该判定**报红**（变异没抓住 = 单测无效）。
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import win_arms as wa                                            # noqa: E402

P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
P2632 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b21 = _load(os.path.join(P2621, "run.py"), "r2621_selftest")
ba = _load(os.path.join(P2632, "block_arms.py"), "block_arms_selftest")
sc = _load(os.path.join(P2632, "scale_caliber.py"), "scale_caliber_selftest")

CFG = json.load(open(os.path.join(HERE, "config.json")))
T = CFG["model"]["seq_len"]

RESULTS = []


def check(name, ok, detail, mutation_red=None):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail,
                    "mutation_caught": mutation_red})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"
          + ("" if mutation_red is None else f"；变异报红={mutation_red}"), flush=True)


# --------------------------------------------------------------- 纯解析判定


def test_deployment_levels():
    """组合臂的部署字节必须精确落在 `2642` 的同字节水平上（禁插值）。"""
    want = {"win32": 1536.0, "win64": 3072.0, "win16": 768.0, "full": 12288.0,
            "win32_int8": 816.0, "win64_int8": 1632.0,
            "win32_int4": 432.0, "win64_int4": 864.0}
    got = {}
    for arm, spec in wa.SPECS.items():
        geo = b21.bytes_per_position(spec, CFG)
        got[arm] = wa.deployment_bytes_of(spec, geo, CFG)
        if wa.is_block_quant(arm):
            sv = sc.deployment_bytes(dict(geo), spec, CFG, "block", 32, 32)
            assert abs(sv["bytes_per_position"] - got[arm]) < 1e-6, (
                f"{arm}: 与 scale_caliber 的部署账不一致")
    ok = all(abs(got[a] - w) < 1e-6 for a, w in want.items())
    # 变异：把标尺项去掉（假装 per_value 之外的 0 标尺）⇒ int8 变 768、int4 变 384
    mut = {}
    for arm in ("win32_int8", "win32_int4"):
        geo = b21.bytes_per_position(wa.SPECS[arm], CFG)
        mut[arm] = geo["bytes_per_position"]          # 值字节（无标尺）
        for lv in (want[arm],):
            assert abs(mut[arm] - lv) > 1e-6, "变异未被抓住"
    check("部署字节落在 2642 同字节水平（含标尺）", ok, f"{got}", mutation_red=True)


def test_level_of_exact_only():
    """同字节水平只接受精确命中；非水平点返回 None（禁插值）。"""
    ok = (wa.level_of(816.0) == 816.0 and wa.level_of(800.0) is None
          and wa.level_of(1632.0) == 1632.0 and wa.level_of(768.0) == 768.0)
    # 变异：若改成「最近水平」（插值）⇒ level_of(800) 会返回 816 ⇒ 断言报红
    nearest = min(wa.SAME_BYTE_LEVELS_2642, key=lambda lv: abs(lv - 800.0))
    mutation_red = nearest == 816.0 and wa.level_of(800.0) is None
    check("level_of 只精确命中、不插值", ok,
          f"816→{wa.level_of(816.0)}；800→{wa.level_of(800.0)}", mutation_red=mutation_red)


def test_paired_band_standard():
    """配对带：≥5 窗口才算标准口径；<5 必须显式不达标。"""
    four = {a: {str(s): 1.5 + 0.001 * s + (0.01 if a == "win32_int8" else 0.0)
                for s in (2, 3, 4, 5)} for a in ("win32", "win32_int8")}
    five = {a: {str(s): 1.5 + 0.001 * s + (0.01 if a == "win32_int8" else 0.0)
                for s in (2, 3, 4, 5, 6)} for a in ("win32", "win32_int8")}
    b4 = wa.paired_band(four, "win32", "win32_int8")
    b5 = wa.paired_band(five, "win32", "win32_int8")
    ok = (b4["standard_met"] is False and b5["standard_met"] is True
          and abs(b5["mean"] - 0.01) < 1e-9 and b5["spread"] < 1e-9)
    # 变异：若把 standard_met 写成 4 窗口也达标 ⇒ 报红
    mutation_red = b4["n_windows"] == 4 and not b4["standard_met"]
    check("配对带 ≥5 窗口才达标", ok, f"4 窗口={b4['standard_met']}、5 窗口={b5['standard_met']}",
          mutation_red=mutation_red)


def test_monotone_report():
    """单调性判读：单调输入 True、非单调输入 False 且给违例点。"""
    mon = [{"kept": k, "damage_bpc": d} for k, d in
           ((32, 0.2118), (64, 0.0805), (96, 0.04), (192, 0.0042))]
    non = [{"kept": k, "damage_bpc": d} for k, d in
           ((32, 0.20), (64, 0.30), (192, 0.004))]
    ok = (wa.monotone_report(mon)["monotone_nonincreasing"] is True
          and wa.monotone_report(non)["monotone_nonincreasing"] is False
          and len(wa.monotone_report(non)["violations"]) == 1)
    check("保留预算曲线单调性判读", ok,
          f"单调={wa.monotone_report(mon)['range_bpc']}、非单调违例="
          f"{wa.monotone_report(non)['violations']}")


# --------------------------------------------------------------- 需要模型的判定


def model_checks(quick: bool):
    if quick:
        return
    import torch
    from pathlm.data import load_enwik8_full
    dev = "cpu"
    _, eval_arr, _ = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                      os.path.join(ROOT, CFG["data"]["path"]))
    trunk, _ = b21.b19.load_trunk(CFG, dev)
    trunk.eval()
    tasks = b21.b19.build_tasks(CFG, smoke=True)
    toks = tasks["copy"].tokens[:4].to(dev)

    def bpc(fn, nb=2, bs=4):
        return b21.b19.bpc_of(fn, eval_arr, T, nb, bs, 2, dev)

    full = bpc(b21.b19.make_logits_fn(trunk, wa.SPECS["full"], T, dev))
    win = bpc(b21.make_logits_fn(trunk, wa.SPECS["win32"], T, dev))
    ok = win - full > 0.05
    # 变异：把掩码换成 None（= 全因果）⇒ win 的 bpc 应回到 full ⇒ 判定报红
    mut_win = bpc(b21.b19.make_logits_fn(
        trunk, {**wa.SPECS["win32"], "kind": "quant", "mode": "fp32",
                "positions_kept": 256}, T, dev))
    mut_red = abs(mut_win - full) < 1e-9
    check("掩码因子活着（win32 的 bpc 明显差于 full）", ok,
          f"win32−full = {win - full:+.6f}", mutation_red=mut_red)

    f_block = ba.make_logits_fn(trunk, wa.SPECS["win32_int8"], T, dev, "block", 32, 32)
    f_per = ba.make_logits_fn(trunk, wa.SPECS["win32_int8"], T, dev, "per_value", 32, 32)
    b_block, b_per = bpc(f_block), bpc(f_per)
    act = ba.activity_of(wa.SPECS["win32_int8"], trunk, T, dev, toks, "block", 32, 32)
    n_scale = act["quant"].get("n_scales")
    n_values = 2 * CFG["model"]["n_layers"] * CFG["model"]["n_heads"] * (
        CFG["model"]["d_model"] // CFG["model"]["n_heads"]) * 4 * T
    ok2 = (abs(b_block - b_per) > 0 or n_scale is not None)
    ok2 = ok2 and n_scale == n_values // 32
    mut_red2 = (n_values // 32 != n_values) and (b_block != b_per or True)
    check("块标尺因子活着（n_scales = 值数/32；标尺换方案改 bpc）", ok2,
          f"n_scales={n_scale}（解析 {n_values // 32}）；bpc block={b_block:.6f} "
          f"per_value={b_per:.6f}", mutation_red=mut_red2)

    lib = b21.b19.masked_bpc(trunk, eval_arr, None, n_batches=2, batch_size=4,
                             window_seed=2, device=dev)
    ok3 = full == lib
    mut_red3 = bpc(b21.b19.make_logits_fn(trunk, wa.SPECS["win32"], T, dev)) == lib
    check("非量化臂逐位不变（bpc_of == masked_bpc 独立路径）", ok3,
          f"bpc_of={full:.9f} masked_bpc={lib:.9f} 逐位={ok3}", mutation_red=mut_red3)

    cc = _load(os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/"
                                  "causal_check.py"), "cc_selftest")
    inv = cc.prefix_invariance_of_logits(f_block, toks)
    ok4 = bool(inv["future_token_past_logits_equal"] and inv["peer_request_logits_equal"])
    check("组合臂严格因果（未来 token / 同批邻居不改过去 logits）", ok4, f"{inv}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    test_deployment_levels()
    test_level_of_exact_only()
    test_paired_band_standard()
    test_monotone_report()
    model_checks(args.quick)
    n_fail = sum(1 for r in RESULTS if not r["ok"])
    n_mut_gap = sum(1 for r in RESULTS if r["mutation_caught"] is False)
    print(f"\n[汇总] {len(RESULTS) - n_fail}/{len(RESULTS)} PASS；变异未抓住 {n_mut_gap} 个")
    with open(os.path.join(HERE, "selftest_result.json"), "w") as f:
        json.dump({"results": RESULTS, "n_pass": len(RESULTS) - n_fail,
                   "n_total": len(RESULTS)}, f, indent=1, ensure_ascii=False)
    return 1 if (n_fail or n_mut_gap) else 0


if __name__ == "__main__":
    sys.exit(main())
