"""2624 · 变异验证（每个语义开关/不变量必须「故意破坏 ⇒ 抓红 ⇒ 还原」）。

对象：本实验引入的两个**纯函数判据**（`band_class` 的 0.014 带宽、`explosion_reproduction`
的预登记阈值）、seed 单因素声明、以及所依赖的**既有**实现
（`pathlm.depth_ar.window_anchor_keep_mask`）。

**不写共享文件**：用**进程内 monkeypatch** 破坏函数/常量，跑单测断言必须抓红，然后撤销；
并用前后 sha256 证明磁盘文件未被触碰（SCHEMA §22.2）。

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

import torch                                                     # noqa: E402
import pathlm.depth_ar as da                                     # noqa: E402

TEST_FILE = os.path.join(ROOT, "tests", "test_longT_1024_seed1.py")
RUN_FILE = os.path.join(HERE, "run.py")


def _load_testmod():
    spec = importlib.util.spec_from_file_location("test_longT_1024_seed1_probe",
                                                  TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    tmod = _load_testmod()
    target_file = os.path.join(ROOT, "pathlm", "depth_ar.py")
    digest_before = {"pathlm/depth_ar.py": sha256_file(target_file),
                     "tests/test_longT_1024_seed1.py": sha256_file(TEST_FILE),
                     "run.py": sha256_file(RUN_FILE)}

    orig_keep = tmod.window_anchor_keep_mask
    orig_band = tmod.R.band_class
    orig_repro = tmod.R.explosion_reproduction
    orig_steps = tmod.R.steps_for
    orig_seed = tmod.R.SEED

    def keep_identity(T, window, anchors=0, device="cpu"):
        """破坏 1：窗口永远等价于「窗口 = T」⇒ 恒等算子（退化点不再被识别）。"""
        return orig_keep(T, T, anchors, device)

    def band_loose(x, band=0.5):
        """破坏 2：带宽被放到 0.5 ⇒ 越界点被误判为「带内」。"""
        return orig_band(x, band)

    def repro_inverted(full_mon, win_mon):
        """破坏 3：复现判据结果翻转（不安全：把「复现」报成「未复现」）。"""
        r = orig_repro(full_mon, win_mon)
        r["verdict"] = "not_reproduced" if r["verdict"] != "not_reproduced" \
            else "reproduced"
        return r

    def steps_ignore_override(seq_len):
        """破坏 4：忽略降速旋钮 ⇒ 两臂可能不同刻度（§20.3）。"""
        return 12000

    mutations = [
        ("Z1-M1", "窗口掩码被换成恒等（退化点不再被识别）", "window_anchor_keep_mask",
         keep_identity, "test_window512_at_T1024_is_nondegenerate", ()),
        ("Z1-M2", "带宽被放到 0.5（越界点被误判为带内）", "band_class",
         band_loose, "test_band_class_boundary_and_direction", ()),
        ("Z1-M3", "复现判据结果翻转（把复现报成未复现）", "explosion_reproduction",
         repro_inverted, "test_explosion_reproduction_not_reproduced", ()),
        ("Z1-M4", "忽略降速旋钮（两臂可能不同刻度）", "steps_for",
         steps_ignore_override, "test_steps_override_keeps_both_arms_at_same_scale", ()),
    ]

    report = {"target": "process-local monkeypatch（不改文件）",
              "sha256_before": digest_before,
              "note": ("本实验未改动 pathlm/depth_ar.py；变异用进程内 monkeypatch 完成"
                       "（SCHEMA §22.2）。seed 单因素另由 Z1-M5 以常量替换验证。"),
              "mutations": []}
    ok_all = True
    for mid, desc, attr, broken, test_name, args in mutations:
        if attr == "window_anchor_keep_mask":
            tmod.window_anchor_keep_mask = broken
        else:
            setattr(tmod.R, attr, broken)
        try:
            fn = getattr(tmod, test_name)
            caught, detail = False, "test 未抓红（不变量未被钉住）"
            try:
                fn(*args)
            except AssertionError as e:
                caught, detail = True, str(e)[:140]
            except Exception as e:                       # noqa: BLE001
                caught, detail = True, f"{type(e).__name__}: {e}"[:140]
        finally:
            tmod.window_anchor_keep_mask = orig_keep
            tmod.R.band_class = orig_band
            tmod.R.explosion_reproduction = orig_repro
            tmod.R.steps_for = orig_steps
            tmod.R.SEED = orig_seed
        ok_all = ok_all and caught
        report["mutations"].append({"id": mid, "desc": desc,
                                    "target_test": test_name, "args": list(args),
                                    "caught_red": caught, "detail": detail})
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — {detail}",
              flush=True)

    # Z1-M5：seed 单因素（把 SEED 改成 0 ⇒ 单因素声明被破坏）
    tmod.R.SEED = 0
    caught, detail = False, "test 未抓红"
    try:
        tmod.test_seed1_arms_only_differ_in_train_window()
    except AssertionError as e:
        caught, detail = True, str(e)[:140]
    finally:
        tmod.R.SEED = orig_seed
    ok_all = ok_all and caught
    report["mutations"].append({"id": "Z1-M5", "desc": "把 SEED 改成 0（seed 单因素声明被破坏）",
                                "target_test": "test_seed1_arms_only_differ_in_train_window",
                                "args": [], "caught_red": caught, "detail": detail})
    print(f"Z1-M5: {'抓红 OK' if caught else '未抓红'} — {detail}", flush=True)

    digest_after = {"pathlm/depth_ar.py": sha256_file(target_file),
                    "tests/test_longT_1024_seed1.py": sha256_file(TEST_FILE),
                    "run.py": sha256_file(RUN_FILE)}
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
