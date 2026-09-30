"""2613 · 契约 F 的变异验证（关键语义开关：故意破坏必须抓红）。

对本实验新增的语义接缝（读出重参数化作用在 dense 单出口路径上）做四处临时破坏，
每处用 `pytest -k <测试>` 定点验证必须失败；破坏后**逐字节还原** `pathlm/depth_ar.py`
（该文件可能同时被其它研究线改动，必须先备份再还原）。

用法：python3 mutate.py   → 写 mutation_report.json，退出码 0 表示四处全抓红且已还原。
"""

import hashlib, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TARGET = os.path.join(ROOT, "pathlm", "depth_ar.py")
TEST = os.path.join(HERE, "test_dense_readout_reparam.py")
BACKUP = TARGET + ".mutbak"

MUTATIONS = [
    ("M1_ignore_switch_in_dense",
     "读出永不应用线性重参数化（开关被忽略）",
     "            elif self.cfg.readout_linear_resid:\n"
     "                hf = hf + self.readout_p(self.readout_q(hf))",
     "            elif False:\n"
     "                hf = hf + self.readout_p(self.readout_q(hf))",
     "test_dense_linres_formula_and_activity"),
    ("M2_saddle_zero_init",
     "退回 P=Q=0 零初始鞍点（恒等起步但永不逃离）",
     "            nn.init.normal_(self.readout_q.weight, std=0.02)",
     "            nn.init.zeros_(self.readout_q.weight)",
     "test_dense_linres_off_state_dict_and_init_identity"),
    ("M3_dense_supervises_all_depths",
     "dense 形态误监督全部深度（不再是「仅最终 CE」）",
     "        readouts = depth_logits if (not self.cfg.dense or dense_aux) else depth_logits[-1:]",
     "        readouts = depth_logits",
     "test_dense_only_final_readout_supervised_and_saddle_escape"),
    ("M4_transform_wrong_depth",
     "读出变换放到 trunk 状态上多做一次（公式不再精确）",
     "                hf = hf + self.readout_p(self.readout_q(hf))",
     "                hf = hf + self.readout_p(self.readout_q(hf)) + self.readout_p(self.readout_q(hf))",
     "test_dense_linres_formula_and_activity"),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    shutil.copy(TARGET, BACKUP)
    orig_sha = sha(TARGET)
    report = {"target": os.path.relpath(TARGET, ROOT), "orig_sha256": orig_sha,
              "mutations": []}
    src0 = open(TARGET).read()
    try:
        for name, desc, old, new, test in MUTATIONS:
            assert src0.count(old) == 1, f"{name}: pattern not unique ({src0.count(old)})"
            open(TARGET, "w").write(src0.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", f"{TEST}::{test}", "-q"],
                               cwd=ROOT, capture_output=True, text=True)
            report["mutations"].append({
                "name": name, "desc": desc, "test": test,
                "caught": r.returncode != 0,
                "pytest_tail": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""})
            print(f"[mutate] {name}: caught={r.returncode != 0} :: {report['mutations'][-1]['pytest_tail']}",
                  flush=True)
    finally:
        shutil.copy(BACKUP, TARGET)
        os.remove(BACKUP)
    restore_sha = sha(TARGET)
    report["restored_sha256"] = restore_sha
    report["restored_bitwise"] = restore_sha == orig_sha
    report["all_caught"] = all(m["caught"] for m in report["mutations"])
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[mutate] restored_bitwise={report['restored_bitwise']} "
          f"all_caught={report['all_caught']}")
    return 0 if (report["all_caught"] and report["restored_bitwise"]) else 1


if __name__ == "__main__":
    sys.exit(main())
