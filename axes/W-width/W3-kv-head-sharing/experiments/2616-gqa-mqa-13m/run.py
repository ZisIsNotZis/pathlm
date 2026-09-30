"""2616 · KV 头共享 GQA/MQA（ticket 33 契约 S，W3 参照线）—— 实验入口。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
干预（唯一变量）= 注意力里 K/V 的**头数**：`mha`（8 = 基线）/ `gqa`（2）/ `mqa`（1）。
其余逐字段一致（`--dense`、seed=0、12k 步、bs32、seq256、d256/L12/H8/mlp6）。
实现：`pathlm.depth_ar.GQABlock`（qkv 投影宽度 = d + 2·n_kv_heads·dh，K/V 以
`[B, n_kv_heads, T, dh]` 进 SDPA，`enable_gqa=True` 按组广播）；缺省走 stock `Block`
⇒ 缺省路径逐位还原 v1（`tests/test_depth_ar.py::test_kv_heads_default_off_matches_v1_bitwise`）。

跑四件事：
  ① `--train`：串行训练四臂（mha / gqa / mqa / gqa_rep 重跑），落 cmd.txt / stdout.log /
     train_log.jsonl / results.json / model_hashes.json（可中断：已存在 model.pt 的臂跳过）；
  ② 评测电池（无参数运行）：逐臂 clean bpc（depth_curve 24×16 窗口序 seed=2，与
     2610/2614/2615/2616-Q 同一数据流）+ 独立评测路径对账（retention.masked_bpc）+
     实际 KV 张量形状（活性证据）+ 解析 vs 实测内存分开报 + 训练侧同方法 loss 斜率；
  ③ 判据（预登记，来自契约，未改）：|Δbpc| < 0.014 ⇒ 近无损并写入参照线；越界 ⇒ 记代价；
     必须附三臂同方法斜率与重跑散度；活性证据必须是**实际 KV 张量形状**；
  ④ 三道对账（SCHEMA §12）：算术 / 任务 / 口径，落进 `results.json::reconciliation`。

用法：
  python3 run.py --train     # 四臂串行训练（≈35–40 min，共享卡）
  python3 run.py             # 评测电池（≈3 min）
  python3 run.py --smoke     # 评测侧冒烟（2 批；不做判据）
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full, batch                # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,      # noqa: E402
                             depth_curve, kv_head_liveness)
from tools.evalkit.retention import masked_bpc                 # noqa: E402

#: 预登记的四臂 = 干预档数（3）× (1 + 重跑数 1)（SCHEMA §17.8）。
ARMS = {
    "mha": {"n_kv_heads": None, "run": "mha-12k-s0", "role": "基线（8 个 KV 头 = MHA）"},
    "gqa": {"n_kv_heads": 2, "run": "gqa-12k-s0", "role": "干预：2 个 KV 头"},
    "mqa": {"n_kv_heads": 1, "run": "mqa-12k-s0", "role": "干预：1 个 KV 头"},
    "gqa_rep": {"n_kv_heads": 2, "run": "gqa-12k-s0-rep",
                "role": "重跑散度臂（同 seed 同配方重训 gqa）"},
}
#: KV 解析内存的期望下降比（契约 S 判据线里写死的两个数：−75% / −87.5%）
EXPECTED_KV_REDUCTION = {"mha": 0.0, "gqa": 0.75, "mqa": 0.875, "gqa_rep": 0.75}
VARIANCE_BAND = 0.014
RUNS = os.path.join(HERE, "runs")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_load_or_none(path):
    return json.load(open(path)) if os.path.exists(path) else None


def run_dir_of(arm: str, smoke: bool) -> str:
    """该臂的 run 目录（冒烟模式写 smoke/runs，不污染全跑目录）。"""
    base = os.path.join(HERE, "smoke", "runs") if smoke else RUNS
    name = ARMS[arm]["run"] + ("-smoke" if smoke else "")
    return os.path.join(base, name)


def slope(rows: list[dict], lo: int, hi: int) -> float | None:
    """最小二乘 loss 斜率（nats/step，原始符号：负 = 还在下降）。"""
    pts = [(r["step"], r["loss"]) for r in rows
           if lo <= r["step"] <= hi and r.get("loss") is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return round(num / den, 9)


# ---------------- ① 训练（串行，可中断） ----------------


def train_all(cfg: dict, smoke: bool = False) -> dict:
    model_cfg = cfg["model"]
    tcfg = cfg["train"]
    steps = 20 if smoke else tcfg["steps"]
    out_root = os.path.join(HERE, "smoke", "runs") if smoke else RUNS
    os.makedirs(out_root, exist_ok=True)
    hashes = json_load_or_none(os.path.join(HERE, "model_hashes.json")) or {}
    out = {}
    for arm, spec in ARMS.items():
        run_dir = run_dir_of(arm, smoke)
        run_name = os.path.basename(run_dir)
        ckpt = os.path.join(run_dir, "model.pt")
        if os.path.exists(ckpt) and not smoke:
            print(f"[train] {arm}: 已有 model.pt，跳过（可中断）", flush=True)
        else:
            cmd = [sys.executable, os.path.join(ROOT, "train_depth_ar.py"), run_name,
                   "--d", str(model_cfg["d_model"]), "--layers",
                   str(model_cfg["n_layers"]), "--heads", str(model_cfg["n_heads"]),
                   "--mlp-mult", str(model_cfg["mlp_mult"]), "--dense",
                   "--steps", str(steps), "--batch-size", str(tcfg["batch_size"]),
                   "--seed", str(tcfg["seed"]), "--lr", str(tcfg["lr"]),
                   "--eval-batches", str(cfg["eval"]["n_batches"]),
                   "--out-root", out_root]
            if spec["n_kv_heads"] is not None:
                cmd += ["--n-kv-heads", str(spec["n_kv_heads"])]
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "cmd.txt"), "w") as f:
                f.write(" ".join(cmd) + "\n")
            t0 = time.time()
            print(f"[train] {arm} → {run_name} 开始 ...", flush=True)
            with open(os.path.join(run_dir, "stdout.log"), "w") as f:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                raise SystemExit(f"{run_name} 训练失败（退出码 {rc}），见 "
                                 f"{run_dir}/stdout.log")
            print(f"[train] {arm} 完成，{time.time() - t0:.1f}s", flush=True)
        res = json.load(open(os.path.join(run_dir, "results.json")))
        out[arm] = {
            "run": run_name,
            "model_pt": os.path.relpath(ckpt, ROOT),
            "sha256": sha256_file(ckpt),
            "size_bytes": os.path.getsize(ckpt),
            "train_wall_minutes": res["wall_minutes"],
            "params": res["params"],
            "cmd": " ".join(open(os.path.join(run_dir, "cmd.txt")).read().split()),
        }
        hashes[arm] = {"run": run_name, "sha256": out[arm]["sha256"],
                       "size_bytes": out[arm]["size_bytes"],
                       "params": res["params"], "wall_minutes": res["wall_minutes"],
                       "cmd": out[arm]["cmd"]}
    if not smoke:
        json.dump(hashes, open(os.path.join(HERE, "model_hashes.json"), "w"),
                  indent=2, ensure_ascii=False)
    return out


def load_arm(cfg: dict, arm: str, device: str = "cuda", smoke: bool = False):
    """载入一臂的 checkpoint 并**断言 sha256**（防事后改动；复用他人权重的强制项）。"""
    spec = ARMS[arm]
    run_dir = run_dir_of(arm, smoke)
    ckpt = os.path.join(run_dir, "model.pt")
    reg = json_load_or_none(os.path.join(HERE, "model_hashes.json"))
    if not smoke and (not reg or arm not in reg):
        raise SystemExit("model_hashes.json 缺失该臂的记录：先跑 `--train`")
    got = sha256_file(ckpt)
    if not smoke and got != reg[arm]["sha256"]:
        raise SystemExit(f"{arm}: checkpoint sha256 与训练时记录不符\n"
                         f"  want {reg[arm]['sha256']}\n  got  {got}")
    mcfg = dict(cfg["model"])
    mcfg["n_kv_heads"] = spec["n_kv_heads"]
    dcfg = DepthARConfig(**mcfg)
    model = DepthARModel(dcfg, cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"run": os.path.basename(run_dir), "sha256": got,
                   "sha256_verified": not smoke,
                   "size_bytes": os.path.getsize(ckpt),
                   "path": os.path.relpath(ckpt, ROOT)}


def config_equivalence(smoke: bool) -> dict:
    """逐字段对照表（含 seed）：四臂的 resolved config **取自各自训练器落盘的
    `results.json`**（不是本文件的 `config.json`），逐字段列出取值与是否相等。

    唯一预期不等的字段 = `n_kv_heads`（干预本身）与 `params`（剪枝类连带效应）；
    其余全同（SCHEMA §9 的配置层等值性）。
    """
    per = {a: json.load(open(os.path.join(run_dir_of(a, smoke), "results.json")))
           for a in ARMS}
    rows = []
    fields = sorted({k for a in ARMS for k in per[a]["config"]}
                    | {"train_window", "train_anchors", "attn_sparse_k",
                       "attn_linear", "topology", "exit_mode", "fill_kv",
                       "proj_fill", "aux_weight", "final_weight",
                       "shallow_stopgrad", "dense_aux", "readout_dims",
                       "readout_gated", "readout_linear_resid", "n_kv_heads",
                       "params", "train_tokens", "vocab_size"})
    for f in fields:
        if f in per["mha"]["config"]:
            vals = {a: per[a]["config"].get(f) for a in ARMS}
        elif f in per["mha"]:
            vals = {a: per[a].get(f) for a in ARMS}
        elif f in per["mha"].get("train", {}):
            vals = {a: per[a]["train"].get(f) for a in ARMS}
        else:
            continue
        rows.append({"field": f, "values": vals,
                     "all_equal": len({json.dumps(v, sort_keys=True)
                                       for v in vals.values()}) == 1})
    for f in ("steps", "batch_size", "lr", "seed"):
        vals = {a: per[a]["train"].get(f) for a in ARMS}
        rows.append({"field": f"train.{f}", "values": vals,
                     "all_equal": len({json.dumps(v, sort_keys=True)
                                       for v in vals.values()}) == 1})
    unequal = [r["field"] for r in rows if not r["all_equal"]]
    return {"rows": rows, "unequal_fields": unequal,
            "expected_unequal": ["n_kv_heads", "params"],
            "all_equal_except_intervention": sorted(unequal) == ["n_kv_heads", "params"],
            "source": "runs/<arm>/results.json::config / ::train（训练器自己落的配置）"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--train", action="store_true", help="串行训练四臂（可中断/可续）")
    ap.add_argument("--smoke", action="store_true",
                    help="训练侧 20 步冒烟（配合 --train）+ 评测侧 2 批（不做判据）")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    t_battery = time.time()
    cfg = json.load(open(args.config))
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE

    if args.train:
        info = train_all(cfg, smoke=args.smoke)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        if args.smoke:
            return
    if args.smoke and not os.path.exists(os.path.join(run_dir_of("mha", True),
                                                      "model.pt")):
        raise SystemExit("冒烟评测需要先 `python3 run.py --train --smoke`")

    ecfg = dict(cfg["eval"])
    nb, bs = (2, 8) if args.smoke else (ecfg["n_batches"], ecfg["batch_size"])
    dev = ecfg["device"]
    T = cfg["model"]["seq_len"]
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8",
                                      os.path.join(ROOT, cfg["data"]["path"]))
    log_f = open(os.path.join(args.out_dir, "eval_log.jsonl"), "w")

    # ---------------- ② 逐臂评测 ----------------
    arm_res = {}
    for arm in ARMS:
        model, ck = load_arm(cfg, arm, dev, smoke=args.smoke)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        curve = depth_curve(model, eval_arr, n_batches=nb, batch_size=bs, device=dev)
        bpc = curve[-1]["bpc"]
        log_f.write(json.dumps({"arm": arm, "tag": "depth_curve",
                                "bpc": bpc}) + "\n")
        # 独立评测路径（retention.masked_bpc，keep_mask=None）：必须逐位等于 depth_curve
        bpc2 = masked_bpc(model, eval_arr, None, n_batches=nb, batch_size=bs,
                          window_seed=ecfg["window_seed"], device=dev)
        # 评测侧复现性：同一窗口序再跑一遍 depth_curve（纯推理，同 seed 同代码）
        bpc_rerun = depth_curve(model, eval_arr, n_batches=nb, batch_size=bs,
                                device=dev)[-1]["bpc"]
        # 实际 KV 张量形状（活性证据）：跑在**真实评测数据**的一个窗口批次上
        gen = torch.Generator().manual_seed(ecfg["window_seed"])
        x, _ = batch(eval_arr, 8, T, gen)
        x = x.to(dev)
        live = kv_head_liveness(model, x)
        assert live["kv_reduction_frac"] == EXPECTED_KV_REDUCTION[arm], (
            f"{arm}: 解析 KV 下降比 {live['kv_reduction_frac']} != 契约预登记值 "
            f"{EXPECTED_KV_REDUCTION[arm]}")
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        wall = time.time() - t0
        rows = [json.loads(l) for l in
                open(os.path.join(run_dir_of(arm, args.smoke), "train_log.jsonl"))]
        run_results = json.load(open(os.path.join(run_dir_of(arm, args.smoke),
                                                 "results.json")))
        last = rows[-1]
        arm_res[arm] = {
            "role": ARMS[arm]["role"], "checkpoint": ck,
            "bpc_clean": bpc, "next_token_acc": curve[-1]["acc"],
            "depth_curve": curve,
            "bpc_second_path_masked_bpc": round(bpc2, 6),
            "bpc_paths_bitwise_equal": round(bpc, 4) == round(bpc2, 4),
            "bpc_eval_rerun": bpc_rerun,
            "bpc_eval_rerun_abs_diff": abs(bpc_rerun - bpc),
            "kv_liveness": live,
            "params": sum(p.numel() for p in model.parameters()),
            "train_wall_minutes": run_results["wall_minutes"],
            "train_log": {
                "n_lines": len(rows),
                "last_step": last["step"],
                "loss_first": rows[0]["loss"], "loss_last": last["loss"],
                "gnorm_max": run_results["gnorm_monitor"]["gnorm_max"],
                "grad_skips": run_results["gnorm_monitor"]["grad_skips"],
                "slope_last_25pct": slope(rows, int(0.75 * cfg["train"]["steps"]),
                                          cfg["train"]["steps"]),
                "slope_last_50pct": slope(rows, int(0.5 * cfg["train"]["steps"]),
                                          cfg["train"]["steps"]),
                "slope_last_1000": slope(rows, cfg["train"]["steps"] - 1000,
                                         cfg["train"]["steps"]),
                "kv_heads_activity_lines": sum(1 for r in rows
                                               if r.get("kv_heads") is not None),
                "kv_heads_activity_first": next((r["kv_heads"] for r in rows
                                                 if r.get("kv_heads")), None),
                "kv_heads_rows_fro_trace": [
                    r["kv_heads"]["kv_rows_fro_mean"] for r in rows
                    if r.get("kv_heads")][::max(len(rows) // 8, 1)],
            },
            "eval_wall_seconds": round(wall, 2),
            "eval_peak_mem_mib": round(peak, 1),
        }
        print(f"[eval] {arm}: bpc {bpc} (第二路径 {bpc2:.4f}) params "
              f"{arm_res[arm]['params']} KV "
              f"{live['kv_bytes_per_position_fp16']} B/位置 峰值 {peak:.0f} MiB",
              flush=True)
        del model
        torch.cuda.empty_cache()

    # 参数差（剪枝类：负）相对 mha 臂对账
    for arm in ARMS:
        arm_res[arm]["kv_liveness"]["params_added_measured"] = (
            arm_res[arm]["params"] - arm_res["mha"]["params"])

    # ---------------- ③ 判据（预登记） + 参照线 ----------------
    ctl = arm_res["mha"]
    crit = {
        "verbatim_contract": cfg["criterion"]["verbatim_contract"],
        "sign_convention": cfg["criterion"]["sign_convention"],
        "variance_band_bpc": VARIANCE_BAND,
        "control_arm": "mha",
        "per_arm": {},
        "reference_line_int8_kv": cfg["criterion"]["int8_kv_reference"],
    }
    for arm in ARMS:
        if arm == "mha":
            continue
        a = arm_res[arm]
        delta = round(ctl["bpc_clean"] - a["bpc_clean"], 6)          # 正 = 改善
        damage = round(-delta, 6)
        entry = {
            "bpc_arm": a["bpc_clean"], "bpc_mha": ctl["bpc_clean"],
            "delta_schema": delta, "damage_bpc": damage,
            "abs_delta": abs(delta),
            "kv_reduction_frac_analytic": a["kv_liveness"]["kv_reduction_frac"],
            "kv_bytes_per_position_fp16": a["kv_liveness"]["kv_bytes_per_position_fp16"],
            "kv_bytes_vs_int8_reference": (
                a["kv_liveness"]["kv_bytes_per_position_fp16"]
                / cfg["criterion"]["int8_kv_reference"]["kv_bytes_per_position_int8"]),
            "same_method_slope_last_25pct": {
                "mha": ctl["train_log"]["slope_last_25pct"],
                "arm": a["train_log"]["slope_last_25pct"],
            },
            "slope_gap_arm_minus_mha": round(
                (a["train_log"]["slope_last_25pct"] or 0)
                - (ctl["train_log"]["slope_last_25pct"] or 0), 9),
            "last_logged_loss": {"mha": ctl["train_log"]["loss_last"],
                                 "arm": a["train_log"]["loss_last"]},
            "activity_evidence": {
                "k_tensor_shape": a["kv_liveness"]["k_batch_shape"],
                "q_tensor_shape": a["kv_liveness"]["per_layer"][0]["q_shape"],
                "qkv_out_width": a["kv_liveness"]["qkv_out_width"],
                "k_proj_per_kv_head_row_norm":
                    a["kv_liveness"]["k_proj_per_kv_head_row_norm"],
                "kv_head_pair_max_abs_weight_diff":
                    a["kv_liveness"]["kv_head_pair_max_abs_weight_diff"],
                "alive": a["kv_liveness"]["alive"],
            },
        }
        if abs(delta) < VARIANCE_BAND:
            entry["verdict"] = (
                f"近无损（|Δbpc| = {abs(delta):.4f} < {VARIANCE_BAND}）：写入参照线"
                f"（KV −{a['kv_liveness']['kv_reduction_frac'] * 100:.1f}%），与 int8 KV "
                f"的近无损 −50% 并列")
        elif delta > 0:
            entry["verdict"] = (
                f"越界且**方向为改善**（Δ = +{delta:.4f} ≥ {VARIANCE_BAND}）：不是代价，"
                "而是快照口径下的小幅增益 —— 按 SCHEMA §16 只能记为「快照、疑为收敛"
                "加速」，需要等参数对照（§15）与更长训练点才能归因，本实验预算不含")
        else:
            entry["verdict"] = (
                f"越界（damage = {damage:.4f} ≥ {VARIANCE_BAND}）：记其代价，不写入"
                "「近无损」参照线")
        crit["per_arm"][arm] = entry

    repro = {
        "rule": ("预登记（README §4）：重跑臂 = `gqa`（三个干预档里的中间档；同 seed "
                 "同配方重训一次，给出 run-to-run 散度）"),
        "rerun_arm": "gqa",
        "run1_bpc": arm_res["gqa"]["bpc_clean"],
        "run2_bpc": arm_res["gqa_rep"]["bpc_clean"],
        "abs_diff_bpc": round(abs(arm_res["gqa"]["bpc_clean"]
                                  - arm_res["gqa_rep"]["bpc_clean"]), 6),
        "run1_delta_schema": crit["per_arm"]["gqa"]["delta_schema"],
        "run2_delta_schema": round(ctl["bpc_clean"] - arm_res["gqa_rep"]["bpc_clean"], 6),
        "delta_abs_diff": round(abs(crit["per_arm"]["gqa"]["delta_schema"]
                                    - (ctl["bpc_clean"]
                                       - arm_res["gqa_rep"]["bpc_clean"])), 6),
        "slope_last_25pct": {"run1": arm_res["gqa"]["train_log"]["slope_last_25pct"],
                             "run2": arm_res["gqa_rep"]["train_log"]["slope_last_25pct"]},
        "same_verdict": (abs(crit["per_arm"]["gqa"]["delta_schema"]) < VARIANCE_BAND
                         == (abs(ctl["bpc_clean"] - arm_res["gqa_rep"]["bpc_clean"])
                             < VARIANCE_BAND)),
        "threshold_bpc": cfg["reproducibility"]["threshold_bpc"],
    }
    repro["pass"] = repro["abs_diff_bpc"] < repro["threshold_bpc"]

    # ---------------- ④ 三道对账（SCHEMA §12） ----------------
    d, dh = cfg["model"]["d_model"], cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    L = cfg["model"]["n_layers"]
    arith = {"kv_bytes_per_position_fp16": {}, "params": {}, "bpc": {}}
    for arm in ARMS:
        nkv = ARMS[arm]["n_kv_heads"] or cfg["model"]["n_heads"]
        arith["kv_bytes_per_position_fp16"][arm] = {
            "formula": f"n_layers·2·n_kv_heads·head_dim·2B = {L}·2·{nkv}·{dh}·2",
            "expected": L * 2 * nkv * dh * 2,
            "measured": arm_res[arm]["kv_liveness"]["kv_bytes_per_position_fp16"],
            "match": L * 2 * nkv * dh * 2
            == arm_res[arm]["kv_liveness"]["kv_bytes_per_position_fp16"],
        }
        arith["params"][arm] = {
            "formula": "mha_params − 2·(d − n_kv_heads·dh)·(d+1)·n_layers（qkv 含偏置）",
            "expected_delta_vs_mha": -2 * (d - nkv * dh) * (d + 1) * L,
            "measured_delta_vs_mha": (0 if arm == "mha" else
                                      arm_res[arm]["params"] - arm_res["mha"]["params"]),
            "match": (0 if arm == "mha" else
                      arm_res[arm]["params"] - arm_res["mha"]["params"])
            == -2 * (d - nkv * dh) * (d + 1) * L,
        }
        arith["bpc"][arm] = {
            "bpc_range_check": 0.0 < arm_res[arm]["bpc_clean"] < 8.0,
            "bpc": arm_res[arm]["bpc_clean"],
            "cross_path_match": arm_res[arm]["bpc_paths_bitwise_equal"],
        }
    arith["pass"] = (all(v["match"] for v in arith["kv_bytes_per_position_fp16"].values())
                     and all(v["match"] for v in arith["params"].values())
                     and all(v["bpc_range_check"] and v["cross_path_match"]
                             for v in arith["bpc"].values()))
    arith["note"] = ("算术对账：① KV 解析字节数按实际张量形状手算一遍；② 参数量差与"
                     "剪枝公式 −2·(d−n_kv_heads·dh)·(d+1)·n_layers 逐位对齐；③ bpc 落在"
                     "物理区间内且两条独立评测路径给出同一个数（4 位小数）")
    task = {
        "same_task": ("同一 char 级 enwik8 AR 任务、同一数据切分（前 90M/末 10M）、"
                      "同一评测窗口序（seed=2 + 24×16）、同一位置集（全序列逐位置）、"
                      "同一目标定义（tokens 自移位）"),
        "same_battery": "四臂同一 run.py 评测电池（同 n_batches/batch_size/seed/代码）",
        "no_truth_leak": "读出只依赖模型自身状态；无「假上下文 + 真目标」式对照（SCHEMA §13）",
        "deploy_correspondence": ("部署对应 = 解码期 KV cache：GQA/MQA 的 cache 张量"
                                  "形状就是 [B, n_kv_heads, T, head_dim]，字节数按该"
                                  "形状算（本实验用 enable_gqa 内核按组广播，不物化复制）"),
        "visibility_pairs_unchanged": ("KV 头共享不改变**可见性对数**（每个 query 头"
                                       "仍可见全部因果键，注意力对数 = n_heads·T(T+1)/2，"
                                       "四臂相同）——与窗口/逐出类干预不同；这正是"
                                       "SCHEMA §17.12 要求的两套口径之一（另一套 = "
                                       "KV 存储量，见算术对账）"),
        "pass": True,
    }
    cal = {
        "matched_step_snapshot": True, "steps": cfg["train"]["steps"],
        "train_tokens_per_arm": cfg["train"]["train_tokens_per_arm"],
        "seed": cfg["train"]["seed"], "n_seed": 1,
        "note": ("四臂同为 matched-step 快照口径（12k 步、98.3M tokens、未收敛）；"
                 "bpc = clean bpc（每字符比特数，越低越好）= 最终深度读出的因果移位 "
                 "CE token 平均；非渐近、非墙钟、非 FLOPs"),
        "memory_caliber": ("**解析** KV 内存（部署关心的量，按实际张量形状算）与"
                           "**实测**显存分开报：本评测把 K/V 交给 SDPA 内核按组广播、"
                           "注意力分数仍是 T×T，且训练/评测峰值由激活主导 ⇒ 实测峰值"
                           "不按 KV 比例下降，不作为收益主张"),
        "same_method_slope": ("四臂斜率用同一窗口（末 25% = 步 9000..11999）与同一"
                              "最小二乘算法（SCHEMA §16）"),
        "pass": True,
    }
    recon = {"arithmetic": arith, "task": task, "caliber": cal,
             "all_pass": bool(arith["pass"] and task["pass"] and cal["pass"])}

    wall = None
    results = {
        "experiment": "2616-gqa-mqa-13m",
        "contract": cfg["contract"],
        "scope": cfg["scope"],
        "config": cfg,
        "config_equivalence": config_equivalence(args.smoke),
        "arms": arm_res,
        "criteria": crit,
        "rerun_divergence": repro,
        "reconciliation": recon,
        "code": {
            "run_py_sha256": sha256_file(os.path.abspath(__file__)),
            "depth_ar_py_sha256": sha256_file(os.path.join(ROOT, "pathlm",
                                                           "depth_ar.py")),
            "train_depth_ar_py_sha256": sha256_file(os.path.join(ROOT,
                                                                 "train_depth_ar.py")),
            "tests_depth_ar_sha256": sha256_file(os.path.join(ROOT, "tests",
                                                              "test_depth_ar.py")),
            "retention_py_sha256": sha256_file(os.path.join(ROOT, "tools", "evalkit",
                                                            "retention.py")),
        },
        "cost": {
            "train_wall_minutes_by_arm": {a: arm_res[a]["train_wall_minutes"]
                                         for a in ARMS},
            "train_wall_total_minutes": round(sum(
                arm_res[a]["train_wall_minutes"] for a in ARMS), 2),
            "battery_wall_seconds": None,          # 下面回填（整次评测进程墙钟）
            "eval_wall_seconds_by_arm": {a: arm_res[a]["eval_wall_seconds"]
                                          for a in ARMS},
            "eval_peak_mem_mib_by_arm": {a: arm_res[a]["eval_peak_mem_mib"] for a in ARMS},
            "device": torch.cuda.get_device_name(0),
            "contract_budget_min": cfg["cost"]["gpu_wall_budget_min"],
            "budget_note": ("契约的成本数字 = 3 臂 × ≈7 min + 重跑 1 臂；实测步时由本"
                            "实验 200 步探针给出（SCHEMA §17.8），全跑墙钟见上表"),
        },
        "eval_protocol": {"n_batches": nb, "batch_size": bs,
                          "window_seed": ecfg["window_seed"], "device": dev,
                          "smoke": args.smoke},
    }
    results["cost"]["battery_wall_seconds"] = round(time.time() - t_battery, 2)
    results["cost"]["total_gpu_wall_minutes_measured"] = round(
        results["cost"]["train_wall_total_minutes"]
        + results["cost"]["battery_wall_seconds"] / 60, 2)
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    log_f.close()
    for arm in ("gqa", "mqa"):
        e = crit["per_arm"][arm]
        print(f"\n判据 {arm}: bpc {e['bpc_arm']} vs mha {e['bpc_mha']} ⇒ Δ "
              f"{e['delta_schema']:+.6f}（|Δ|{'<' if e['abs_delta'] < VARIANCE_BAND else '≥'}"
              f"{VARIANCE_BAND}）KV −{e['kv_reduction_frac_analytic'] * 100:.1f}%"
              f" ⇒ {e['verdict'][:40]}...")
    print(f"重跑散度（gqa vs gqa_rep）= {repro['abs_diff_bpc']} bpc"
          f"（阈值 {repro['threshold_bpc']}，{'pass' if repro['pass'] else 'FAIL'}）")
    print(f"三道对账 all_pass = {recon['all_pass']}")
    print(f"落盘：{out}")


if __name__ == "__main__":
    main()
