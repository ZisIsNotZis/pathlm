"""2637 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 还原」）。

对象：2×2 臂表、QK-Norm 稳定化档的接线（`none` 零 hook / `global` 全挂）。
用**进程内 monkeypatch** 破坏（**不写共享文件**），跑 `tests/test_linear_attn_quality.py`
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

TEST_FILE = os.path.join(ROOT, "tests", "test_linear_attn_quality.py")
RUN_FILE = os.path.join(HERE, "run.py")
DRV_FILE = os.path.join(HERE, "linear_driver.py")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_testmod():
    spec = importlib.util.spec_from_file_location("test_linear_probe", TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tmod = _load_testmod()
    run, drv = tmod.run, tmod.drv
    orig = {"ARMS": run.ARMS, "apply_qk_norm": drv.apply_qk_norm}
    digest_before = {
        "tests/test_linear_attn_quality.py": sha256_file(TEST_FILE),
        "experiments/2637-linear-attn-quality/run.py": sha256_file(RUN_FILE),
        "experiments/2637-linear-attn-quality/linear_driver.py": sha256_file(DRV_FILE),
    }

    def global_zero_hook(model, mode):
        """破坏：稳定化档没挂任何 hook。"""
        if mode == "global":
            return {"attn_scale_mode": mode, "n_qkv_hooks": 0, "active": False}
        return orig["apply_qk_norm"](model, mode)

    def none_all_hooks(model, mode):
        """破坏：`none` 档也挂 hook（接线错）。"""
        if mode == "none":
            return orig["apply_qk_norm"](model, "global")
        return orig["apply_qk_norm"](model, mode)

    mutations = [
        ("M1", "2×2 臂表被破坏（去掉 global 两档）",
         lambda: setattr(run, "ARMS",
                         [dict(a, qk="none") for a in orig["ARMS"]]),
         "test_arms_2x2", ()),
        ("M2", "稳定化档（global）没挂 hook",
         lambda: setattr(drv, "apply_qk_norm", global_zero_hook),
         "test_qk_norm_none_zero_hooks_global_all_hooks", ()),
        ("M3", "`none` 档也挂 hook（接线错）",
         lambda: setattr(drv, "apply_qk_norm", none_all_hooks),
         "test_qk_norm_none_zero_hooks_global_all_hooks", ()),
    ]

    report = {"target": "run.py + linear_driver.py（monkeypatch，不改文件）",
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
            run.ARMS = orig["ARMS"]
            drv.apply_qk_norm = orig["apply_qk_norm"]
        ok_all = ok_all and caught
        report["mutations"].append({"id": mid, "desc": desc,
                                    "target_test": test_name,
                                    "caught_red": caught, "detail": detail})
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — {detail}",
              flush=True)

    digest_after = {
        "tests/test_linear_attn_quality.py": sha256_file(TEST_FILE),
        "experiments/2637-linear-attn-quality/run.py": sha256_file(RUN_FILE),
        "experiments/2637-linear-attn-quality/linear_driver.py": sha256_file(DRV_FILE),
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
