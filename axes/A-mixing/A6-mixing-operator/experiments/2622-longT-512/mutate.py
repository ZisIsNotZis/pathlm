"""2622 · 变异验证（每个语义开关/不变量必须「故意破坏 ⇒ 抓红 ⇒ 还原」）。

对象：本实验依赖的两件**既有**实现（本实验未改动它们，SCHEMA §22 文件所有权见 README §9）：
① `pathlm.depth_ar.window_anchor_keep_mask`（窗口掩码的语义）；② 由它派生的
「非退化 / 同相对刻度 / 闭式计数」不变量（`tests/test_longT_512.py`）。

**不写共享文件**：X2 是本轮唯一共享代码写入者，故这里用**进程内 monkeypatch**
破坏掩码函数，跑单测断言必须抓红，然后撤销；并用前后 sha256 证明磁盘文件未被触碰。
（若用文件改写式变异，会与 X2 的并发写入冲突——SCHEMA §22.2。）

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


def _load_testmod():
    path = os.path.join(ROOT, "tests", "test_longT_512.py")
    spec = importlib.util.spec_from_file_location("test_longT_512_probe", path)
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
    orig = da.window_anchor_keep_mask
    target_file = os.path.join(ROOT, "pathlm", "depth_ar.py")
    test_file = os.path.join(ROOT, "tests", "test_longT_512.py")
    digest_before = {"pathlm/depth_ar.py": sha256_file(target_file),
                     "tests/test_longT_512.py": sha256_file(test_file)}

    def identity(T, window, anchors=0, device="cpu"):
        """破坏 1：窗口永远等价于「窗口 = T」⇒ 恒等算子（退化点不会被抓）。"""
        return orig(T, T, anchors, device)

    def off_by_one(T, window, anchors=0, device="cpu"):
        """破坏 2：窗口边界 off-by-one（< 写成 +1）⇒ 多放一个位置。"""
        return orig(T, window + 1, anchors, device)

    def no_causal(T, window, anchors=0, device="cpu"):
        """破坏 3：丢掉因果项 ⇒ 放出未来键（对称窗口）。"""
        i, j = torch.meshgrid(torch.arange(T, device=device),
                              torch.arange(T, device=device), indexing="ij")
        keep = ((i - j).abs() < window) | torch.eye(T, dtype=torch.bool, device=device)
        return keep

    def clamp128(T, window, anchors=0, device="cpu"):
        """破坏 4：窗口长度被夹到 128 ⇒ 256 的取值失效（主臂其实没生效）。"""
        return orig(T, min(window, 128), anchors, device)

    mutations = [
        ("X3-M1", "窗口掩码被换成恒等（退化点不再被识别）", identity,
         "test_window256_at_T512_is_nondegenerate", ()),
        ("X3-M2", "窗口边界 off-by-one（< 变成多放一个位置）", off_by_one,
         "test_window_keep_count_matches_closed_form", (256, 512)),
        ("X3-M3", "掩码丢掉因果项（放出未来键）", no_causal,
         "test_window_mask_is_causal_and_keeps_diagonal", (256, 512)),
        ("X3-M4", "窗口被夹到 128（主臂的 256 没生效）", clamp128,
         "test_window256_at_T512_is_nondegenerate", ()),
    ]

    report = {"target": "pathlm/depth_ar.py（monkeypatch，不改文件）",
              "sha256_before": digest_before,
              "note": ("本实验未改动 pathlm/depth_ar.py 与 tests/test_longT_512.py；"
                       "变异用进程内 monkeypatch 完成（避免与 X2 共享代码写入冲突）"),
              "mutations": []}
    ok_all = True
    for mid, desc, broken, test_name, args in mutations:
        da.window_anchor_keep_mask = broken
        tmod.window_anchor_keep_mask = broken
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
            da.window_anchor_keep_mask = orig
            tmod.window_anchor_keep_mask = orig
        ok_all = ok_all and caught
        report["mutations"].append({"id": mid, "desc": desc,
                                    "target_test": test_name, "args": list(args),
                                    "caught_red": caught, "detail": detail})
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — {detail}",
              flush=True)

    digest_after = {"pathlm/depth_ar.py": sha256_file(target_file),
                    "tests/test_longT_512.py": sha256_file(test_file)}
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
