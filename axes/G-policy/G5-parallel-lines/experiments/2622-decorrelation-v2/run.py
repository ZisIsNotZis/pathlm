"""2622 · 去相关 v2 的决定性实验 —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**预登记，不许改）。
训练：`python3 train_m1.py <arm> --config <scale>/<arm>.json`（见 `train_all.sh` / `MODEL.md`）。
本入口只做**评测** + 汇总（一个 scale 一次）：

  ① 仪器自检：本入口的 K 路径集成函数与 `probe_tts.bpc_k`（13M 锚点的同一方法）
     在同一检查点上**逐位同值**；
  ② K 曲线（K=1/2/4/8，60-batch probe 协议）三臂 + 每档 60 批记录；
  ③ 成对 JS（路径相关度）；
  ④ 确定性路径对照（stock 臂 + repr 臂强制 `p_skip=0`）——增益应恰好 0；
  ⑤ 主判据（= `bpc_repr@K8 − bpc_stock@K1 < 0`）+ 并列判据（`bpc_repr@K8 − bpc_l1@K8 < 0`）：
     配对 bootstrap CI，并断言主判据两种写法的代数等价；
  ⑥ 200-batch battery clean bpc（与仓库公开锚 `B0 1.5074` 同协议）；
  ⑦ 活性量（含 gate skip 数，须为 0）与 loss 斜率读训练产物；
  ⑧ 收敛门禁：判据量在中点 / 终点两点上的斜率（SCHEMA §17.15）；
  ⑨ 口径/复现信息：code sha256、墙钟、显存、算力账。

用法：
  python3 run.py --scale 13m --smoke   # 冒烟：2 批 × K∈{1,2}
  python3 run.py --scale 13m           # 全电池
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # 数据路径（.tmp/enwik8, data/enwik8_full.npz）是相对仓库根的

from pathlm.config import ModelConfig, PathConfig                    # noqa: E402
from pathlm.data import load_enwik8_full                             # noqa: E402
from pathlm.eval import bpc as battery_bpc                           # noqa: E402
from pathlm.model import PathLM                                      # noqa: E402
from tools.evalkit.carry_channel import loss_slope                   # noqa: E402
from tools.evalkit.ensemble_probe import (                           # noqa: E402
    bootstrap_ci, deterministic_paths, paired_delta_ci, pairwise_js,
    path_ensemble_bpc, per_batch_bpc, reading_A, reading_B)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_probe_tts():
    """`probe_tts.py` 只用于**仪器自检**（13M 锚点的同一方法），不参与主电池。"""
    spec = importlib.util.spec_from_file_location("probe_tts", os.path.join(ROOT, "probe_tts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ckpt_path(scale_cfg: dict, arm: str) -> str:
    meta = scale_cfg["arms_meta"].get(arm, {})
    if "reuse_ckpt" in meta:
        p = meta["reuse_ckpt"]
        return p if os.path.isabs(p) else os.path.join(ROOT, p)
    return os.path.join(HERE, scale_cfg["train_dir"], arm, "model.pt")


def load_arm(scale_cfg: dict, arm: str, vocab: int, device: str, want_sha: str | None):
    mcfg = json.load(open(os.path.join(HERE, scale_cfg["config_dir"], f"{arm}.json")))
    path = ckpt_path(scale_cfg, arm)
    if not os.path.exists(path):
        raise SystemExit(f"缺权重 {path}（先跑 train_all.sh）")
    got = sha256_file(path)
    if want_sha and not want_sha.startswith("TO_FILL") and got != want_sha:
        raise SystemExit(f"checkpoint sha256 mismatch ({arm}):\n  want {want_sha}\n  got  {got}")
    model = PathLM(ModelConfig(**mcfg["model"]), PathConfig(**mcfg["path"]), vocab_size=vocab)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "resolved": mcfg,
                   "config_file": os.path.relpath(
                       os.path.join(HERE, scale_cfg["config_dir"], f"{arm}.json"), ROOT),
                   "params": sum(p.numel() for p in model.parameters()),
                   "reused": "reuse_ckpt" in scale_cfg["arms_meta"].get(arm, {})}


def ensemble(arm_model, eval_arr, k: int, cfg: dict, *, collect_js=False,
             deterministic: bool = False, n_batches: int | None = None) -> dict:
    ev = cfg["eval"]
    nb = n_batches if n_batches is not None else ev["n_batches"]
    if deterministic:
        with deterministic_paths(arm_model) as m:
            return path_ensemble_bpc(m, eval_arr, k, n_batches=nb,
                                     batch_size=ev["batch_size"], seed=cfg["probe_seed"],
                                     vocab_size=cfg["vocab_size"], collect_js=collect_js)
    return path_ensemble_bpc(arm_model, eval_arr, k, n_batches=nb,
                             batch_size=ev["batch_size"], seed=cfg["probe_seed"],
                             vocab_size=cfg["vocab_size"], collect_js=collect_js)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=("13m", "50m"), required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-battery", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    scfg = cfg["scales"][args.scale]
    ARMS = tuple(scfg["arms"])
    if args.smoke:
        cfg["eval"] = {**cfg["eval"], "n_batches": 2}
        cfg["probe_seed"] = 11
    out = os.path.join(HERE, f"results_{args.scale}.json")
    log: list[dict] = []
    t0 = time.time()
    device = cfg["device"]
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    vocab, ps = cfg["vocab_size"], cfg["probe_seed"]
    note(f"scale={args.scale} 载入 enwik8 val（后 10M 字符，训练从不碰）")
    _, eval_arr, vocab_data = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    if vocab_data != vocab:
        raise SystemExit(f"vocab mismatch: config {vocab} vs data {vocab_data}")

    note("载入三臂（model.pt sha256 断言）")
    models, ck = {}, {}
    for arm in ARMS:
        entry = cfg.get("arms", {}).get(f"{args.scale}/{arm}")
        want = entry.get("sha256") if isinstance(entry, dict) else entry
        models[arm], ck[arm] = load_arm(scfg, arm, vocab, device, want)
        note(f"  {arm}: sha256 {ck[arm]['sha256'][:16]}… params {ck[arm]['params']} "
             f"reused={ck[arm]['reused']}")

    results: dict = {"scale": args.scale, "config": cfg, "checkpoints": ck}
    results["scope"] = {
        "口径": (f"matched-step 快照（{scfg['steps_full']} 步；K 曲线 = 60-batch probe 协议，"
                 f"与 13M 锚点同方法）"),
        "单点不外推": True,
        "算力价签": "K8 集成 = 8× 前向/token；stock 单线 = 1× 前向/token",
    }

    # ---------------- ① 仪器自检：与 probe_tts.bpc_k 同方法 ----------------
    note("自检 ①：本入口 K 集成 vs probe_tts.bpc_k（13M 锚点同一方法）")
    pt = load_probe_tts()
    instrument = {}
    for k in cfg["selfcheck_k"]:
        mine = ensemble(models["l1"], eval_arr, k, cfg)
        theirs = pt.bpc_k(models["l1"], eval_arr, k,
                          n_batches=cfg["eval"]["n_batches"],
                          batch_size=cfg["eval"]["batch_size"])
        instrument[f"K{k}"] = {
            "mine_bpc": mine["bpc"], "probe_tts_bpc": theirs["bpc"],
            "bpc_equal": mine["bpc"] == theirs["bpc"],
            "mine_acc": mine["acc"], "probe_tts_acc": theirs["acc"],
            "n_batches": cfg["eval"]["n_batches"], "seed": ps}
        note(f"  K={k}: mine {mine['bpc']} vs probe_tts {theirs['bpc']} "
             f"⇒ equal={instrument[f'K{k}']['bpc_equal']}")
    instrument["all_equal"] = all(v["bpc_equal"] for v in instrument.values()
                                  if isinstance(v, dict))
    results["selfchecks"] = {"probe_tts_method_identity": instrument}

    # ---------------- ② K 曲线 ----------------
    note("K 曲线：三臂 × K ∈ %s（60-batch probe 协议）" % cfg["k_sweep"])
    curves: dict = {}
    for arm in ARMS:
        curves[arm] = {}
        for k in cfg["k_sweep"]:
            run = ensemble(models[arm], eval_arr, k, cfg, collect_js=(k in cfg["js_k"]))
            curves[arm][str(k)] = {kk: vv for kk, vv in run.items() if kk != "per_batch"}
            curves[arm][str(k)]["per_batch"] = run["per_batch"]
            note(f"  {arm} K={k}: bpc {run['bpc']} acc {run['acc']}"
                 + (f" | JS {run['pairwise_js']['mean_bits']:.4f} bits"
                    if "pairwise_js" in run else "")
                 + (f" | paths identical={run['paths_bitwise_identical']}"
                    if "paths_bitwise_identical" in run else ""))
    results["k_curve"] = curves

    # ---------------- ④ 确定性路径对照 ----------------
    note("确定性路径对照（增益应恰好 0）：stock 臂 + repr 臂强制 p_skip=0")
    det = {}
    for tag, arm in (("stock", "stock"), ("repr_forced", "repr")):
        r = {}
        for k in cfg["determinism_k"]:
            run = ensemble(models[arm], eval_arr, k, cfg, deterministic=(tag != "stock"))
            r[f"K{k}"] = {kk: vv for kk, vv in run.items() if kk != "per_batch"}
        kmax = max(cfg["determinism_k"])
        r["gain_K1_minus_Kmax"] = round(r["K1"]["bpc"] - r[f"K{kmax}"]["bpc"], 6)
        r["paths_bitwise_identical_all"] = all(
            r[f"K{k}"].get("paths_bitwise_identical", False) for k in cfg["determinism_k"] if k > 1)
        det[tag] = r
        note(f"  {tag}: K1 {r['K1']['bpc']} → K{kmax} {r[f'K{kmax}']['bpc']} | "
             f"增益 {r['gain_K1_minus_Kmax']} | 路径逐位相同={r['paths_bitwise_identical_all']}")
    results["deterministic_path_control"] = det

    # ---------------- ③ 成对 JS（K=8） ----------------
    note("成对 JS（路径相关度，K=%d）" % max(cfg["js_k"]))
    kmax_s = str(max(cfg["js_k"]))
    js = {arm: curves[arm][kmax_s].get("pairwise_js") for arm in ARMS}
    for arm in ARMS:
        note(f"  {arm}: {js[arm]['mean_bits'] if js[arm] else None} bits "
             f"({js[arm]['mean_nats'] if js[arm] else None} nats)")
    results["pairwise_js_at_kmax"] = js

    # ---------------- ⑤ 判据 ----------------
    note("判据：主（task 形式 = repr@K8 − stock@K1 < 0）/ 并列（repr@K8 − l1@K8 < 0）")
    kk = str(cfg["main_k"])
    k1 = "1"
    b1 = {a: curves[a][k1]["bpc"] for a in ARMS}
    bk = {a: curves[a][kk]["bpc"] for a in ARMS}
    # 主判据：任务预登记形式 `(K 路径集成) − (stock 单线) < 0`；等价于 reading_B 的 reduced 取负。
    rb = reading_B(b1["l1"], b1["stock"], bk["repr"])
    main_value = bk["repr"] - b1["stock"]                 # 任务形式：< 0 = 多线首次成立
    main_verbatim = (bk["repr"] - b1["stock"]) - 0.0      # 与 reading_B 的 reduced 对照
    b1_pb = {a: per_batch_bpc(curves[a][k1]) for a in ARMS}
    bk_pb = {a: per_batch_bpc(curves[a][kk]) for a in ARMS}

    ci_stock_vs_repr = paired_delta_ci(bk_pb["repr"], b1_pb["stock"], seed=1)   # repr@K8 − stock@K1
    ci_l1_vs_repr = paired_delta_ci(bk_pb["repr"], bk_pb["l1"], seed=2)         # repr@K8 − l1@K8
    ci_repr_tax = paired_delta_ci(b1_pb["repr"], b1_pb["l1"], seed=3)           # repr@K1 − l1@K1
    ci_mech_tax = paired_delta_ci(b1_pb["l1"], b1_pb["stock"], seed=4)          # l1@K1 − stock@K1

    crit = {
        "main_reading_B": {
            "task_form": "(bpc_repr@K8) - (bpc_stock@K1) < 0",
            "value": round(main_value, 6),
            "baseline": "stock (p_skip=0, w_repr_decorr=0, same tokens/seed/scale)",
            "cost_term": "bpc_l1@K1 - bpc_stock@K1 (same protocol as the baseline read)",
            "primary": True,
            "reading_B_reduced": round(rb["reduced"], 6),
            "task_form_equals_neg_reading_B_reduced": abs(main_value + rb["reduced"]) < 1e-9,
            "algebra_consistent": rb["algebra_consistent"],
            "ci": ci_stock_vs_repr,
            "passed_point_estimate": main_value < 0,
            "passed": bool(main_value < 0 and ci_stock_vs_repr["paired_ci_excludes_zero"]),
            "compute_price_tag": "repr@K8 = 8x forwards/token; stock@K1 = 1x",
        },
        "parallel_mechanism": {
            "task_form": "(bpc_repr@K8) - (bpc_l1@K8) < 0",
            "baseline": "l1 (p_skip=0.1, w_repr_decorr=0, same tokens/seed/scale)",
            "value": round(bk["repr"] - bk["l1"], 6),
            "primary": False,
            "ci": ci_l1_vs_repr,
            "passed_point_estimate": (bk["repr"] - bk["l1"]) < 0,
            "passed": bool((bk["repr"] - bk["l1"]) < 0
                           and ci_l1_vs_repr["paired_ci_excludes_zero"]),
            "js_less_correlated": (
                None if not js.get("repr") or not js.get("l1")
                else bool(js["repr"]["mean_bits"] > js["l1"]["mean_bits"])),
        },
        "taxes": {
            "repr_clean_tax_vs_l1": round(b1["repr"] - b1["l1"], 6),
            "repr_clean_tax_vs_l1_ci": ci_repr_tax,
            "mechanism_clean_tax_vs_stock": round(b1["l1"] - b1["stock"], 6),
            "mechanism_clean_tax_vs_stock_ci": ci_mech_tax,
        },
        "variance_floor": {"delta_unorderable_below": 0.014,
                           "note": "docs/methodology.md：Δ < 0.014 不可排序"},
    }
    note(f"  主判据：repr@K8 − stock@K1 = {crit['main_reading_B']['value']} "
         f"CI {ci_stock_vs_repr['delta_ci']} ⇒ 过={crit['main_reading_B']['passed']}")
    note(f"  并列判据：repr@K8 − l1@K8 = {crit['parallel_mechanism']['value']} "
         f"CI {ci_l1_vs_repr['delta_ci']} ⇒ 过={crit['parallel_mechanism']['passed']}")
    results["criteria"] = crit

    # ---------------- ⑥ 200-batch battery ----------------
    if cfg.get("battery", {}).get("enabled", True) and not args.no_battery:
        note("200-batch battery clean bpc（与公开锚 B0 1.5074 同协议）")
        results["battery_clean_bpc"] = {}
        for arm in ARMS:
            r = battery_bpc(models[arm], eval_arr, vocab, n_batches=cfg["battery"]["n_batches"])
            results["battery_clean_bpc"][arm] = r
            note(f"  {arm}: battery bpc {r['bpc']} acc {r['next_token_acc']}")

    # ---------------- ⑦ 活性量 / 斜率 ----------------
    note("活性量与 loss 斜率（同方法）读训练产物")
    results["train_products"], results["loss_slope"] = {}, {}
    for arm in ARMS:
        rp = os.path.join(HERE, scfg["train_dir"], arm, "results.json")
        lp = os.path.join(HERE, scfg["train_dir"], arm, "train_log.jsonl")
        if "reuse_ckpt" in scfg["arms_meta"].get(arm, {}):
            results["train_products"][arm] = {"note": "复用 2620 检查点，训练产物见 2620 目录"}
            continue
        if not os.path.exists(rp):
            raise SystemExit(f"缺训练产物 {rp}")
        res = json.load(open(rp))
        lines = [json.loads(x) for x in open(lp) if x.strip()]
        results["train_products"][arm] = {
            "params": res.get("params"), "wall_minutes": res.get("wall_minutes"),
            "peak_gpu_mem_mb": res.get("peak_gpu_mem_mb"),
            "train_tokens": res.get("train_tokens"),
            "intervention": res.get("intervention"),
            "unspecified_path_keys": res.get("unspecified_path_keys"),
            "first_log": lines[0], "last_log": lines[-1]}
        results["loss_slope"][arm] = loss_slope(lines, lo=200, hi=int(scfg["steps_full"]))
        iv = res.get("intervention", {})
        note(f"  {arm}: wall {res.get('wall_minutes')} min, peak {res.get('peak_gpu_mem_mb')} MB, "
             f"slope {results['loss_slope'][arm]['slope']}, "
             f"repr_last {iv.get('repr_decorr_last')}, skip_frac_mean {iv.get('skip_frac_mean')}, "
             f"gate_skips {iv.get('gate_skips')}")

    # ---------------- ⑧ 收敛门禁（判据量在中点/终点） ----------------
    note("收敛门禁（§17.15）：判据量在中点 / 终点两点上直接测")
    steps_full = int(scfg["steps_full"])
    mid = int(scfg["midpoint_ckpt"])
    traj = {}
    for st in [mid]:
        row = {}
        for arm in ARMS:
            if "reuse_ckpt" in scfg["arms_meta"].get(arm, {}):
                continue
            p = os.path.join(ROOT, ".tmp", "v2-ckpts", args.scale, arm, f"model_step{st:06d}.pt")
            if not os.path.exists(p):
                continue
            mcfg = json.load(open(os.path.join(HERE, scfg["config_dir"], f"{arm}.json")))
            m = PathLM(ModelConfig(**mcfg["model"]), PathConfig(**mcfg["path"]), vocab_size=vocab)
            m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            m.to(device).eval()
            for q in m.parameters():
                q.requires_grad_(False)
            r1 = ensemble(m, eval_arr, 1, cfg)
            row[arm] = {"K1_bpc": r1["bpc"], "K1_ci": bootstrap_ci(per_batch_bpc(r1), seed=7),
                        "K1_per_batch": per_batch_bpc(r1)}
            if arm == "repr":
                rk = ensemble(m, eval_arr, cfg["main_k"], cfg)
                row[arm][f"K{cfg['main_k']}_bpc"] = rk["bpc"]
                row[arm][f"K{cfg['main_k']}_ci"] = bootstrap_ci(per_batch_bpc(rk), seed=8)
                row[arm][f"K{cfg['main_k']}_per_batch"] = per_batch_bpc(rk)
            del m
        if row and "stock" in row and "repr" in row and f"K{cfg['main_k']}_bpc" in row["repr"]:
            row["criterion_value"] = round(row["repr"][f"K{cfg['main_k']}_bpc"] - row["stock"]["K1_bpc"], 6)
            row["criterion_delta_ci"] = paired_delta_ci(
                row["repr"][f"K{cfg['main_k']}_per_batch"], row["stock"]["K1_per_batch"], seed=9)
        traj[str(st)] = row
        note(f"  轨迹 @{st} 步：" + " ".join(f"{a}@K1 {row[a]['K1_bpc']}" for a in ARMS
                                             if isinstance(row.get(a), dict))
             + (f" | 判据量 {row.get('criterion_value')}" if "criterion_value" in row else ""))
    full_criterion = round(bk["repr"] - b1["stock"], 6)
    results["trajectory"] = {"checkpoints": traj, "full_run_steps": steps_full,
                             "full_run": {"repr@K8": bk["repr"], "stock@K1": b1["stock"],
                                          "criterion_value": full_criterion}}
    if str(mid) in traj and "criterion_value" in traj[str(mid)]:
        d = steps_full - mid
        results["convergence_gate"] = {
            "quantity": "criterion quantity = bpc_repr@K8 − bpc_stock@K1 (eval stream)",
            "why": ("SCHEMA §17.15：收敛门禁必须落在决定判据的量上；本例判据 = 评测 bpc 的差，"
                    "不是训练总 CE"),
            "criterion_value_at_ckpt": traj[str(mid)]["criterion_value"],
            "criterion_value_at_full": full_criterion,
            "criterion_slope_per_1k": round(
                (full_criterion - traj[str(mid)]["criterion_value"]) / (d / 1000.0), 6) if d else None,
            "train_loss_slopes": {a: results["loss_slope"][a]["slope"]
                                  for a in results["loss_slope"]},
            "train_loss_note": "训练总 CE 斜率仅作并报，不得代替上面的判据量斜率",
        }
    else:
        # 50M 的 stock 是复用检查点（无中点），判据量中点不可得 ⇒ 标快照口径
        slope_rows = {}
        for arm in ARMS:
            if "reuse_ckpt" in scfg["arms_meta"].get(arm, {}) or arm not in traj.get(str(mid), {}):
                continue
            slope_rows[arm] = {
                "K1_at_ckpt": traj[str(mid)][arm]["K1_bpc"],
                "K1_at_full": b1[arm],
                "K1_slope_per_1k": round((b1[arm] - traj[str(mid)][arm]["K1_bpc"])
                                         / ((steps_full - mid) / 1000.0), 6)}
        results["convergence_gate"] = {
            "available": False,
            "quantity": "criterion quantity = bpc_repr@K8 − bpc_stock@K1 (eval stream)",
            "why": ("stock 是复用检查点（无中点）⇒ 判据量中点不可得；只能标快照口径并给各臂 K1 "
                    "绝对 bpc 斜率"),
            "absolute_K1_slopes": slope_rows,
            "train_loss_slopes": {a: results["loss_slope"][a]["slope"]
                                  for a in results["loss_slope"]},
            "train_loss_note": "训练总 CE 仅作并报，不得代替判据量",
        }
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "note": "本进程净分配（不含共享卡上其他任务的占用），与训练侧 peak_gpu_mem_mb 同口径"}

    # ---------------- ⑨ 复现信息 ----------------
    nb = cfg["eval"]["n_batches"]
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "gpu_wall_budget_min": cfg["gpu_wall_budget_min"],
        "train_wall_minutes": {a: results["train_products"].get(a, {}).get("wall_minutes")
                               if isinstance(results["train_products"].get(a), dict) else None
                               for a in ARMS},
        "n_forward_batches": (
            sum(cfg["k_sweep"]) * len(ARMS) * nb
            + sum(cfg["selfcheck_k"]) * nb * 2
            + sum(cfg["determinism_k"]) * nb * 2
            + (cfg["battery"]["n_batches"] * len(ARMS)
               if cfg.get("battery", {}).get("enabled", True) and not args.no_battery else 0)
            + len(traj) * nb * 2),
        "note": "每批 = bs×T 个 token 的一次前向（K 路径就是 K 次前向，价签的来源）",
    }
    code_files = ["train_m1.py", "pathlm/model.py", "pathlm/config.py",
                  "pathlm/eval.py", "tools/evalkit/ensemble_probe.py", "probe_tts.py",
                  "tests/test_decorrelation_v2.py", "reconcile_numbers.py"]
    code = {"run.py": sha256_file(os.path.join(HERE, "run.py"))}
    for p in code_files:
        cand = os.path.join(HERE, p)
        path = cand if os.path.exists(cand) else os.path.join(ROOT, p)
        code[p] = sha256_file(path) if os.path.exists(path) else None
    code["README.md (本契约)"] = sha256_file(os.path.join(HERE, "README.md"))
    results["code"] = code
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(HERE, f"run_log_{args.scale}.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{out}（评测墙钟 {results['cost']['wall_minutes_total_eval']} min）")


if __name__ == "__main__":
    main()
