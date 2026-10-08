"""2629 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 还原」）。

对象：本实验**自己负责的语义面** —— 长度条件档 `qk-normfrom512`（复用 2625 的实现）、
**干预本身（STEPS=6000 的 matched-step）** 与训练探针 `StepProbe`。用**进程内
monkeypatch** 破坏（**不写共享文件**），跑 `tests/test_longT_2048_matched.py` 的对应断言，
必须抓红；并用前后 sha256 证明磁盘文件未被触碰（SCHEMA §22.2/§10）。

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

TEST_FILE = os.path.join(ROOT, "tests", "test_longT_2048_matched.py")
DRIVER_FILE = os.path.join(HERE, "clean_driver.py")
RUN_FILE = os.path.join(HERE, "run.py")
W1_FILE = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                       "experiments", "2625-longT-stability", "stability_driver.py")


def _load_testmod():
    spec = importlib.util.spec_from_file_location("test_longT_2048_mt_probe",
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
    cd = tmod.cd                          # 测试用的 clean_driver 模块实例
    w1 = cd.W1                            # 被复用的 2625 驱动模块
    run = tmod.run2629                    # 本实验 run.py 模块实例
    orig = {"apply_attn_scale": w1.apply_attn_scale,
            "probe_pre": cd.StepProbe._pre,
            "window_arm": run.ARMS[1]["window"],
            "steps": run.STEPS}
    digest_before = {
        "tests/test_longT_2048_matched.py": sha256_file(TEST_FILE),
        "experiments/2629-longT-2048-matched/clean_driver.py": sha256_file(DRIVER_FILE),
        "experiments/2629-longT-2048-matched/run.py": sha256_file(RUN_FILE),
        "experiments/2625-longT-stability/stability_driver.py": sha256_file(W1_FILE),
    }

    def gate_always_off(model, mode, tau, min_t=0):
        """破坏 1：长度条件门永远关（T=2048 也不挂 hook）。"""
        return orig["apply_attn_scale"](model, mode, tau, min_t=10 ** 9)

    def norm_identity(model, mode, tau, min_t=0):
        """破坏 2：归一被换成恒等（q/k 不是 √d_h 尺度）。"""
        for blk in model.blocks:
            blk.qkv.register_forward_hook(lambda m, a, o: o)
        return {"active": True, "n_qkv_hooks": len(model.blocks)}

    def probe_counts_eval(self, module, args):
        """破坏 4：探针把 eval 前向也计入（步号错位——正是要防的缺陷）。"""
        self.n += 1

    def arm_window_degenerate():
        """破坏 3：窗口臂的窗口被写成 T（= 2048）——退化点冒充真观测点。"""
        run.ARMS[1]["window"] = 2048

    def steps_revert_to_3000():
        """破坏 5：干预回退到 3000 步（= 2628 的未匹配口径）。"""
        run.STEPS = 3000

    mutations = [
        ("MT-M1", "长度条件门永远关（T=2048 也不挂 hook）",
         lambda: setattr(w1, "apply_attn_scale", gate_always_off),
         "test_qk_norm_active_at_2048", ()),
        ("MT-M2", "逐头归一被换成恒等（不是 √d_h 尺度）",
         lambda: setattr(w1, "apply_attn_scale", norm_identity),
         "test_qk_norm_unit_scale_and_zero_params_at_2048", ()),
        ("MT-M3", "窗口臂的窗口被写成 T（退化点冒充真点）",
         arm_window_degenerate,
         "test_arm_table_is_one_point_two_arms", ()),
        ("MT-M4", "探针把 eval 前向也计入（步号错位）",
         lambda: setattr(cd.StepProbe, "_pre", probe_counts_eval),
         "test_step_probe_counts_training_forwards_only", ()),
        ("MT-M5", "干预回退：STEPS 写成 3000（不再是 matched-step）",
         steps_revert_to_3000,
         "test_steps_are_6000_and_match_2626_t1024", ()),
    ]

    report = {"target": "clean_driver.py + run.py + 2625 stability_driver.py"
                        "（monkeypatch，不改文件）",
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
            setattr(w1, "apply_attn_scale", orig["apply_attn_scale"])
            cd.StepProbe._pre = orig["probe_pre"]
            run.ARMS[1]["window"] = orig["window_arm"]
            run.STEPS = orig["steps"]
        ok_all = ok_all and caught
        report["mutations"].append({"id": mid, "desc": desc,
                                    "target_test": test_name,
                                    "caught_red": caught, "detail": detail})
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — {detail}",
              flush=True)

    digest_after = {
        "tests/test_longT_2048_matched.py": sha256_file(TEST_FILE),
        "experiments/2629-longT-2048-matched/clean_driver.py": sha256_file(DRIVER_FILE),
        "experiments/2629-longT-2048-matched/run.py": sha256_file(RUN_FILE),
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
