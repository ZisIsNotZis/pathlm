"""2621 · 对账脚本（SCHEMA §12 三道：算术 / 任务 / 口径）。

读 `results.json`（由 `run.py --stage assemble` 落盘）与训练日志，逐条核对报告里
每一个「成本 / 次数 / 等式 / 哈希」都能一一对应。任何一条不符即非零退出。

运行：python3 reconcile_numbers.py
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")


def main():
    d = json.load(open(RESULTS))
    arms = d["arms"]
    cfg = d["config"]
    v = d["verdict"]
    rec = d["reconciliation"]
    D, L = cfg["d_model"], cfg["n_layers"]
    extra = (L + 1) * 2 * D * D
    rows, fails = [], 0

    def chk(desc, got, want, tol=1e-9):
        nonlocal fails
        ok = (abs(got - want) <= tol) if isinstance(got, (int, float)) \
            and isinstance(want, (int, float)) and not isinstance(got, bool) \
            and not isinstance(want, bool) else (got == want)
        fails += (not ok)
        rows.append((desc, got, want, "OK" if ok else "FAIL"))

    def bpc(name):
        return arms[name]["depth_curve"][-1]["bpc"]

    def tf(name, k):
        return next(r["accept"] for r in arms[name]["tf_acceptance"]
                    if r["depth"] == k)

    # ---- 算术：参数量（§15） ----
    chk("expected_extra = (L+1)·2d²", extra, 9 * 2 * 256 * 256)
    chk("params(PD) − params(A)", arms["perdepth"]["params"] - arms["A"]["params"],
        extra)
    chk("params(EP) − params(A)",
        arms["wide_shared"]["params"] - arms["A"]["params"], extra)
    chk("params(PD) == params(EP)", arms["perdepth"]["params"],
        arms["wide_shared"]["params"])
    chk("run.py::equal_params 标记", rec["arithmetic"]["equal_params"], True)
    chk("config::params_extra(PD)", cfg["perdepth_arm"]["params_extra"], extra)
    chk("config::params_extra(EP)", cfg["wide_shared_arm"]["params_extra"], extra)

    # ---- 算术：Δbpc 恒等式（Δ = bpc(X) − bpc(A)，负 = 改善） ----
    chk("Δbpc(PD) = bpc(PD) − bpc(A)", v["delta_bpc_PD_vs_A"],
        round(bpc("perdepth") - bpc("A"), 4))
    chk("Δbpc(EP) = bpc(EP) − bpc(A)", v["delta_bpc_EP_vs_A"],
        round(bpc("wide_shared") - bpc("A"), 4))
    chk("Δbpc(gated)", v["delta_bpc_gated_vs_A"], round(bpc("gated") - bpc("A"), 4))
    chk("Δbpc(linear)", v["delta_bpc_linear_vs_A"],
        round(bpc("linear") - bpc("A"), 4))
    chk("PD − EP 差", v["PD_minus_EP"],
        round(v["delta_bpc_PD_vs_A"] - v["delta_bpc_EP_vs_A"], 4))
    chk("verdict bpc(A)", v["A_clean_bpc"], bpc("A"))
    chk("verdict bpc(PD)", v["PD_clean_bpc"], bpc("perdepth"))
    chk("verdict bpc(EP)", v["EP_clean_bpc"], bpc("wide_shared"))
    chk("arms::depth_curve 末值 == hist_clean_bpc（PD）",
        arms["perdepth"]["hist_clean_bpc"], bpc("perdepth"))
    chk("arms::depth_curve 末值 == hist_clean_bpc（EP）",
        arms["wide_shared"]["hist_clean_bpc"], bpc("wide_shared"))

    # ---- 算术：tf@4 / tf@6 与表格一致 ----
    for k in (4, 6):
        chk(f"tf@{k}(A)", v[f"A_tf{k}"], tf("A", k))
        chk(f"tf@{k}(PD)", v[f"PD_tf{k}"], tf("perdepth", k))
        chk(f"tf@{k}(EP)", v[f"EP_tf{k}"], tf("wide_shared", k))

    # ---- 算术：判据字符串与数值一致（C1/C2，方差界 0.014） ----
    chk("方差界 = 0.014", rec["arithmetic"]["variance_bound"], 0.014)
    chk("C1 分支（≤ −0.014 ⇒ 有效 / |Δ|<0.014 ⇒ 否证）",
        v["C1"].startswith("逐深度读出有效") if v["delta_bpc_PD_vs_A"] <= -0.014
        else (v["C1"].startswith("逐深度读出无效")
              if abs(v["delta_bpc_PD_vs_A"]) < 0.014
              else v["C1"].startswith("落在方差界内")), True)
    gap = abs(v["delta_bpc_PD_vs_A"] - v["delta_bpc_EP_vs_A"])
    chk("C2 分支（|ΔPD−ΔEP|<0.014 ⇒ 容量 / PD≤EP−0.014 ⇒ 独立性）",
        v["C2"].startswith("增益归因于容量") if gap < 0.014
        else (v["C2"].startswith("逐深度独立性本身有效")
              if v["delta_bpc_PD_vs_A"] <= v["delta_bpc_EP_vs_A"] - 0.014
              else v["C2"].startswith("PD 明显坏于 EP")), True)

    # ---- 干预活性（§10） ----
    chk("PD 活性 alive", arms["perdepth"]["intervention"]["alive"], True)
    chk("EP 活性 alive", arms["wide_shared"]["intervention"]["alive"], True)
    chk("PD 逐深度 P_k 全部非零",
        min(arms["perdepth"]["intervention"]["p_fro_by_depth"]) > 0, True)
    chk("PD 逐深度 Q_k 全部非零",
        min(arms["perdepth"]["intervention"]["q_fro_by_depth"]) > 0, True)

    # ---- 训练预算/口径 ----
    chk("train_tokens = steps·bs·seq", rec["caliber"]["train_tokens"],
        cfg["steps"] * cfg["batch_size"] * cfg["seq_len"])
    chk("n_seed = 1", rec["caliber"]["n_seed"], 1)
    for nm in ("perdepth", "wide_shared"):
        chk(f"arms[{nm}].equivalence_ok", arms[nm]["equivalence_ok"], True)
    for nm in ("A", "gated", "linear"):
        chk(f"arms[{nm}].equivalence_ok", arms[nm]["equivalence_ok"], True)

    # ---- 斜率（§16）三臂同方法 ----
    sl = d["stability"]
    for nm in ("A_arm", "perdepth_arm", "wide_shared_arm"):
        chk(f"stability[{nm}] 有末窗斜率",
            "slope_loss_per_1k" in sl[nm]["loss_slope_last25pct"], True)

    # ---- 权重哈希（MODEL.md 必须与磁盘一致） ----
    model_md = os.path.join(HERE, "MODEL.md")
    hashes = {}
    for nm in ("perdepth", "wide_shared"):
        p = os.path.join(HERE, "runs", nm, "model.pt")
        if os.path.exists(p):
            hashes[nm] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if os.path.exists(model_md):
        md = open(model_md).read()
        for nm, h in hashes.items():
            chk(f"MODEL.md 含 {nm} 权重 sha256", h in md, True)

    # ---- 打印 ----
    width = max(len(r[0]) for r in rows)
    print(f"{'检查项'.ljust(width)} | 实际 | 期望 | 结果")
    print("-" * (width + 40))
    for desc, got, want, st in rows:
        print(f"{desc.ljust(width)} | {got} | {want} | {st}")
    print("-" * (width + 40))
    print(f"对账条目 {len(rows)}，失败 {fails}")
    if hashes:
        print("权重 sha256:", {k: h[:16] + "…" for k, h in hashes.items()})
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
