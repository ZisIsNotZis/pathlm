"""2625 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 还原」）。

对象：本实验**自己新增**的语义面 —— 实验本地驱动器 `stability_driver.py` 的候选补丁
（`temp` / `qk-norm` / `wd` / `warmup` / `clip` / 探针）。用**进程内 monkeypatch**
破坏（**不写共享文件**），跑 `tests/test_longT_stability.py` 的对应断言，必须抓红；
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
from tools.evalkit import mutation                     # noqa: E402

TEST_FILE = os.path.join(ROOT, "tests", "test_longT_stability.py")
DRIVER_FILE = os.path.join(HERE, "stability_driver.py")


def _load_testmod():
    spec = importlib.util.spec_from_file_location("test_longT_stability_probe",
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
    sd = tmod.sd
    orig = {
        "apply_attn_scale": sd.apply_attn_scale,
        "patch_warmup": sd.patch_warmup,
        "patch_clip_threshold": sd.patch_clip_threshold,
        "patch_weight_decay": sd.patch_weight_decay,
    }
    orig_probe_qkv = sd.ProbeHook.record_qkv
    digest_before = {
        "tests/test_longT_stability.py": sha256_file(TEST_FILE),
        "experiments/2625-longT-stability/stability_driver.py":
            sha256_file(DRIVER_FILE),
    }

    def dead_temp(model, mode, tau, min_t=0):
        """破坏 1：`temp`/`qk-norm` 的 hook 变成恒等（开关是死的）。"""
        for blk in model.blocks:
            blk.qkv.register_forward_hook(lambda m, a, o: o)
        return {"attn_scale_mode": mode, "tau": tau,
                "n_qkv_hooks": len(model.blocks)}

    def warmup_ignored(warmup, steps):
        """破坏 2：预热长度被忽略（永远按引擎的 200 步）。"""
        return orig["patch_warmup"](200, steps)

    def clip_ignored(value):
        """破坏 3：裁剪阈值被忽略（永远 1.0）。"""
        return orig["patch_clip_threshold"](1.0)

    def wd_ignored(value):
        """破坏 4：weight decay 实参被忽略（永远 0.01）。"""
        return orig["patch_weight_decay"](0.01)

    def probe_no_logits(self, i, x_in, out):
        """破坏 5：探针不记录注意力统计（活性量缺失）。"""
        self._slot(i)["resid_out"] = 0.0

    mutations = [
        ("W1-M1", "temp/qk-norm 的 qkv hook 被换成恒等（语义开关是死的）",
         lambda: setattr(sd, "apply_attn_scale", dead_temp),
         "test_temp_candidate_actually_changes_logits", ()),
        ("W1-M2", "qk-norm 被换成恒等（每个头不再是 √d_h 尺度）",
         lambda: setattr(sd, "apply_attn_scale", dead_temp),
         "test_qk_norm_gives_unit_scale_heads", ()),
        ("W1-M3", "warmup 长度被忽略（永远 200 步）",
         lambda: setattr(sd, "patch_warmup", warmup_ignored),
         "test_warmup_patch_only_extends_warmup", ()),
        ("W1-M4", "clip 阈值被忽略（永远 1.0）",
         lambda: setattr(sd, "patch_clip_threshold", clip_ignored),
         "test_clip_threshold_patch_replaces_max_norm", ()),
        ("W1-M5", "weight decay 实参被忽略（永远 0.01）",
         lambda: setattr(sd, "patch_weight_decay", wd_ignored),
         "test_weight_decay_patch_sets_optimizer_arg", ()),
        ("W1-M6", "探针丢掉注意力统计（活性量缺失）",
         lambda: setattr(sd.ProbeHook, "record_qkv", probe_no_logits),
         "test_probe_records_all_layer_stats", ()),
    ]

    report = {"target": "stability_driver.py（monkeypatch，不改文件）",
              "sha256_before": digest_before,
              "note": ("本实验未改动任何共享文件；变异用进程内 monkeypatch 完成"
                       "（SCHEMA §22.2）"),
              "mutations": []}
    ok_all = True
    for mid, desc, apply_broken, test_name, args in mutations:
        apply_broken()
        try:
            fn = getattr(tmod, test_name)
            caught, detail, _status = mutation.inprocess_outcome(fn, args)
        finally:
            for name, fn_orig in orig.items():
                setattr(sd, name, fn_orig)
            sd.ProbeHook.record_qkv = orig_probe_qkv
        ok_all = ok_all and caught
        report["mutations"].append({"id": mid, "desc": desc,
                                    "target_test": test_name,
                                    "caught_red": caught, "detail": detail})
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — {detail}",
              flush=True)

    digest_after = {
        "tests/test_longT_stability.py": sha256_file(TEST_FILE),
        "experiments/2625-longT-stability/stability_driver.py":
            sha256_file(DRIVER_FILE),
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
