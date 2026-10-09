"""2636 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 还原」）。

对象：本实验**自己负责的语义面** —— 同 step 口径常量、四点表（非退化）、
成本口径 `cost_theory`、以及复用 `2625` 的长度条件档 QK-Norm hook。
用**进程内 monkeypatch** 破坏（**不写共享文件**），跑 `tests/test_longT_unified.py`
的对应断言，必须抓红；并用前后 sha256 证明磁盘文件未被触碰（SCHEMA §22.2/§10）。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                            # noqa: E402

TEST_FILE = os.path.join(ROOT, "tests", "test_longT_unified.py")
RUN_FILE = os.path.join(HERE, "run.py")
W1_FILE = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                       "experiments", "2625-longT-stability", "stability_driver.py")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_testmod():
    spec = importlib.util.spec_from_file_location("test_longT_unified_probe", TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tmod = _load_testmod()
    run = tmod.run
    orig = {"S": run.S, "POINTS": run.POINTS, "TOKENS_AT_S": run.TOKENS_AT_S,
            "apply_qk_norm": run._apply_qk_norm, "cost_theory": run.cost_theory}
    digest_before = {
        "tests/test_longT_unified.py": sha256_file(TEST_FILE),
        "experiments/2636-longT-unified/run.py": sha256_file(RUN_FILE),
        "experiments/2625-longT-stability/stability_driver.py": sha256_file(W1_FILE),
    }

    def qk_ignore_min_t(model, seq_len):
        """破坏：长度条件门被忽略（T ≤ 512 也挂 hook）。"""
        return run._load_w1().apply_attn_scale(model, "qk-norm", 1.0, min_t=0)

    def qk_always_off(model, seq_len):
        """破坏：长度条件门永远关（T > 512 也不挂 hook）。"""
        return {"n_qkv_hooks": 0, "active": False}

    def bad_cost_theory(seq_lens=(256, 512, 1024, 2048)):
        """破坏：公式值错（注意力降幅写成 0，all-in 降幅非单调）。"""
        th = orig["cost_theory"](seq_lens)
        for k in th["window_theory"]:
            th["window_theory"][k]["attn_flops_reduction_relative_to_full"] = 0.0
            th["window_theory"][k]["layer_macs_reduction_all_in"] = 0.0
        return th

    mutations = [
        ("M1", "同 step 口径常量被改（S=3000）",
         lambda: setattr(run, "S", 3000), "test_step_budget_is_6000", ()),
        ("M2", "token 计数公式被改（tokens = steps*T，丢 batch）",
         lambda: setattr(run, "TOKENS_AT_S",
                         {T: run.S * T for T in (256, 512, 1024, 2048)}),
         "test_tokens_at_s_match_formula", ()),
        ("M3", "四点表里塞退化点（窗口 = T，恒等算子）",
         lambda: setattr(run, "POINTS",
                         ((256, 256), (256, 512), (512, 1024), (1024, 2048))),
         "test_window_mask_non_degenerate_at_points", ()),
        ("M4", "长度条件门被忽略（T ≤ 512 也挂 hook）",
         lambda: setattr(run, "_apply_qk_norm", qk_ignore_min_t),
         "test_qk_norm_inert_at_or_below_min_t_bitwise", ()),
        ("M5", "长度条件门永远关（T > 512 也不挂 hook）",
         lambda: setattr(run, "_apply_qk_norm", qk_always_off),
         "test_qk_norm_active_above_min_t_and_zero_params", ()),
        ("M6", "成本公式被改（注意力降幅写 0）",
         lambda: setattr(run, "cost_theory", bad_cost_theory),
         "test_cost_theory_values", ()),
    ]

    report = {"target": "run.py（monkeypatch，不改文件）",
              "sha256_before": digest_before,
              "note": "本实验未改动任何共享文件；变异用进程内 monkeypatch 完成（SCHEMA §22.2）",
              "mutations": []}
    ok_all = True
    for mid, desc, apply_broken, test_name, args in mutations:
        apply_broken()
        try:
            fn = getattr(tmod, test_name)
            caught, detail, _status = mutation.inprocess_outcome(fn, args)
        finally:
            run.S = orig["S"]
            run.POINTS = orig["POINTS"]
            run.TOKENS_AT_S = orig["TOKENS_AT_S"]
            run._apply_qk_norm = orig["apply_qk_norm"]
            run.cost_theory = orig["cost_theory"]
        ok_all = ok_all and caught
        report["mutations"].append({"id": mid, "desc": desc,
                                    "target_test": test_name,
                                    "caught_red": caught, "detail": detail})
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — {detail}",
              flush=True)

    digest_after = {
        "tests/test_longT_unified.py": sha256_file(TEST_FILE),
        "experiments/2636-longT-unified/run.py": sha256_file(RUN_FILE),
        "experiments/2625-longT-stability/stability_driver.py": sha256_file(W1_FILE),
    }
    report["sha256_after"] = digest_after
    report["restored_bitwise"] = digest_after == digest_before
    report["all_caught"] = ok_all
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("restored_bitwise", "all_caught")},
                     indent=2, ensure_ascii=False))
    if not (ok_all and report["restored_bitwise"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
