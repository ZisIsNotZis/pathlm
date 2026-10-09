"""2626 · G5 机制收敛收尾（ticket 43 V2）—— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**预登记，不许改）。
训练：`python3 train_m1.py <arm> --config configs13m/<arm>.json --steps 24000`（见 `train_all.sh`）。
本入口只做**评测** + 汇总（一个 scale 一次）：

  ① 仪器自检：本入口的 K 路径集成函数与 `probe_tts.bpc_k`（13M 锚点的同一方法）
     在同一检查点上**逐位同值**；
  ② K 曲线（K=1/2/4/8，60-batch probe 协议）三臂 + 每档 60 批记录；
  ③ 成对 JS（路径相关度）；
  ④ 确定性路径对照（stock 臂 + repr01 臂强制 p_skip=0）——增益应恰好 0；
  ⑤ 主判据（= `bpc_repr01@K8 − bpc_stock@K1`）+ 并列机制判据（`bpc_repr01@K8 − bpc_l1@K8`）：
     配对 bootstrap CI，并断言主判据两种写法的代数等价；
  ⑥ 200-batch battery clean bpc（与仓库公开锚 `B0 1.5074` 同协议）；
  ⑦ 活性量（含 gate skip 数，须为 0）与 loss 斜率读训练产物；
  ⑧ **收敛轨迹**：判据量（主判据缺口、机制净效应 `l1@K8 − repr01@K8`）在
     ckpt@6000/12000/18000/24000 四点直接测（SCHEMA §17.15 门禁落在决定判据的量上）；
  ⑨ @6k 与 X1（2622）既有读数的跨运行对照（**调度不同**，见 analysis.md §8）；
  ⑩ 口径/复现信息：code sha256、墙钟、显存、算力账。

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

CKPT_ROOT = os.path.join(ROOT, ".tmp", "2626-ckpts")
X1_DIR = os.path.join(ROOT, "axes", "G-policy", "G5-parallel-lines", "experiments",
                      "2622-decorrelation-v2")


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


def build_model(mcfg_json: dict, pcfg_json: dict, vocab: int, device: str) -> PathLM:
    model = PathLM(ModelConfig(**mcfg_json), PathConfig(**pcfg_json), vocab_size=vocab)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def ckpt_path(scale_cfg: dict, arm: str) -> str:
    return os.path.join(HERE, scale_cfg["train_dir"], arm, "model.pt")


def traj_ckpt_path(scale_cfg: dict, arm: str, step: int) -> str:
    return os.path.join(CKPT_ROOT, "13m", arm, f"model_step{step:06d}.pt")


def load_arm(scale_cfg: dict, arm: str, vocab: int, device: str, want_sha: str | None):
    mcfg = json.load(open(os.path.join(HERE, scale_cfg["config_dir"], f"{arm}.json")))
    path = ckpt_path(scale_cfg, arm)
    if not os.path.exists(path):
        raise SystemExit(f"缺权重 {path}（先跑 train_all.sh）")
    got = sha256_file(path)
    if want_sha and not want_sha.startswith("TO_FILL") and got != want_sha:
        raise SystemExit(f"checkpoint sha256 mismatch ({arm}):\n  want {want_sha}\n  got  {got}")
    model = build_model(mcfg["model"], mcfg["path"], vocab, device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "resolved": mcfg,
                   "config_file": os.path.relpath(
                       os.path.join(HERE, scale_cfg["config_dir"], f"{arm}.json"), ROOT),
                   "params": sum(p.numel() for p in model.parameters()),
                   "reused": False}


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


def _mech_ordering(v: float) -> str:
    """机制净效应 `l1@K8 − repr01@K8` 的预登记三态：
    keep（≥0.014，可排序保持）/ narrowed（0 ≤ v < 0.014，收窄到不可排序带）/ reversed（<0）。"""
    if v >= 0.014:
        return "keep"
    if v >= 0.0:
        return "narrowed"
    return "reversed"


def measure_traj_point(scale_cfg: dict, step: int, cfg: dict, eval_arr, vocab: int,
                       device: str) -> dict:
    """在一个中间检查点上直接测**决定判据的量**（SCHEMA §17.15）。

    需要的四个读数：`stock@K1`（主判据的基准）、`l1@K1` 与 `l1@K8`、`repr01@K1` 与
    `repr01@K8`（机制判据与两套税）。只测这些，不重跑整条 K 曲线（省算力）。
    """
    row: dict = {"step": step, "arms": {}}
    kmax = cfg["main_k"]
    for arm in scale_cfg["arms"]:
        p = traj_ckpt_path(scale_cfg, arm, step)
        if not os.path.exists(p):
            return {"step": step, "available": False, "arms": {},
                    "note": f"缺中间检查点 {os.path.relpath(p, ROOT)}"}
        mcfg = json.load(open(os.path.join(HERE, scale_cfg["config_dir"], f"{arm}.json")))
        model = build_model(mcfg["model"], mcfg["path"], vocab, device)
        model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        want_k = {1} | ({kmax} if arm in ("l1", scale_cfg["primary_repr"]) else set())
        a = {}
        for k in sorted(want_k):
            run = ensemble(model, eval_arr, k, cfg)
            a[f"K{k}_bpc"] = run["bpc"]
            a[f"K{k}_per_batch"] = per_batch_bpc(run)
        row["arms"][arm] = a
        del model
    # 决定判据的两个量
    primary = scale_cfg["primary_repr"]
    st, l1, rp = row["arms"]["stock"], row["arms"]["l1"], row["arms"][primary]
    row["main_criterion"] = round(rp[f"K{kmax}_bpc"] - st["K1_bpc"], 6)
    row["main_criterion_ci"] = paired_delta_ci(rp[f"K{kmax}_per_batch"], st["K1_per_batch"], seed=9)
    row["mechanism_net"] = round(l1[f"K{kmax}_bpc"] - rp[f"K{kmax}_bpc"], 6)
    row["mechanism_net_ci"] = paired_delta_ci(l1[f"K{kmax}_per_batch"], rp[f"K{kmax}_per_batch"],
                                              seed=10)
    row["repr_tax_vs_l1"] = round(rp["K1_bpc"] - l1["K1_bpc"], 6)
    row["available"] = True
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=("13m",), required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-battery", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    scfg = cfg["scales"][args.scale]
    ARMS = tuple(scfg["arms"])
    PRIMARY = scfg.get("primary_repr") or scfg.get("repr_arms", ["repr01"])[0]
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
        note(f"  {arm}: sha256 {ck[arm]['sha256'][:16]}… params {ck[arm]['params']}")

    results: dict = {"scale": args.scale, "config": cfg, "checkpoints": ck}
    results["scope"] = {
        "口径": (f"matched-step 快照（{scfg['steps_full']} 步、24000 步 cosine 调度；K 曲线 = "
                 f"60-batch probe 协议，与 13M 锚点同方法）"),
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
    det_arms = scfg.get("det_arms", ["stock", PRIMARY])
    note(f"确定性路径对照（增益应恰好 0）：{det_arms} 强制 p_skip=0")
    det = {}
    for arm in det_arms:
        tag = "stock" if arm == "stock" else f"{arm}_forced"
        r = {}
        for k in cfg["determinism_k"]:
            run = ensemble(models[arm], eval_arr, k, cfg, deterministic=(arm != "stock"))
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
    note("判据：主（task 形式 = repr01@K8 − stock@K1 < 0）/ 机制（repr01@K8 − l1@K8 < 0）")
    kk = str(cfg["main_k"])
    k1 = "1"
    b1 = {a: curves[a][k1]["bpc"] for a in ARMS}
    bk = {a: curves[a][kk]["bpc"] for a in ARMS}
    rb = reading_B(b1["l1"], b1["stock"], bk[PRIMARY])
    main_value = bk[PRIMARY] - b1["stock"]                 # 任务形式：< 0 = 多线首次成立
    b1_pb = {a: per_batch_bpc(curves[a][k1]) for a in ARMS}
    bk_pb = {a: per_batch_bpc(curves[a][kk]) for a in ARMS}

    ci_stock_vs_repr = paired_delta_ci(bk_pb[PRIMARY], b1_pb["stock"], seed=1)
    ci_l1_vs_repr = paired_delta_ci(bk_pb[PRIMARY], bk_pb["l1"], seed=2)
    ci_repr_tax = paired_delta_ci(b1_pb[PRIMARY], b1_pb["l1"], seed=3)
    ci_mech_tax = paired_delta_ci(b1_pb["l1"], b1_pb["stock"], seed=4)

    crit = {
        "main_reading_B": {
            "task_form": "(bpc_repr01@K8) - (bpc_stock@K1) < 0",
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
            "compute_price_tag": "repr01@K8 = 8x forwards/token; stock@K1 = 1x",
        },
        "mechanism_net": {
            "task_form": "(bpc_l1@K8) - (bpc_repr01@K8)   # 正 = repr01 集成更好 = 机制净成立",
            "baseline": "l1 (p_skip=0.1, w_repr_decorr=0, same tokens/seed/scale)",
            "value": round(bk["l1"] - bk[PRIMARY], 6),
            "primary": False,
            "variance_floor": 0.014,
            "passed": bool(bk["l1"] - bk[PRIMARY] >= 0.014),
            "ordering": _mech_ordering(bk["l1"] - bk[PRIMARY]),
            "ci": ci_l1_vs_repr,
            "js_less_correlated": (
                None if not js.get(PRIMARY) or not js.get("l1")
                else bool(js[PRIMARY]["mean_bits"] > js["l1"]["mean_bits"])),
        },
        "taxes": {
            "repr_clean_tax_vs_l1": round(b1[PRIMARY] - b1["l1"], 6),
            "repr_clean_tax_vs_l1_ci": ci_repr_tax,
            "mechanism_clean_tax_vs_stock": round(b1["l1"] - b1["stock"], 6),
            "mechanism_clean_tax_vs_stock_ci": ci_mech_tax,
        },
        "variance_floor": {"delta_unorderable_below": 0.014,
                           "note": "docs/methodology.md：Δ < 0.014 不可排序"},
    }
    note(f"  主判据：repr01@K8 − stock@K1 = {crit['main_reading_B']['value']} "
         f"CI {ci_stock_vs_repr['delta_ci']} ⇒ 过={crit['main_reading_B']['passed']}")
    note(f"  机制净效应：l1@K8 − repr01@K8 = {crit['mechanism_net']['value']} "
         f"（{crit['mechanism_net']['ordering']}）CI {ci_l1_vs_repr['delta_ci']}")
    results["criteria"] = crit

    # ---------------- ⑤b 单臂 λ 记录（本实验只有 λ=0.1 一条干预臂）----------------
    sweep = {}
    for a in scfg.get("repr_arms", [PRIMARY]):
        l1_minus = bk["l1"] - bk[a]
        gain = b1[a] - bk[a]
        gain_l1 = b1["l1"] - bk["l1"]
        sweep[a] = {
            "K1": b1[a], "K8": bk[a],
            "ensemble_gain_K1_minus_K8": round(gain, 6),
            "gain_increment_vs_l1": round(gain - gain_l1, 6),
            "clean_tax_vs_l1": round(b1[a] - b1["l1"], 6),
            "l1_K8_minus_arm_K8": round(l1_minus, 6),
            "js_bits_K8": (js[a]["mean_bits"] if js.get(a) else None),
            "ci_arm_vs_l1_K8": paired_delta_ci(bk_pb[a], bk_pb["l1"], seed=11),
        }
    results["lambda_arm"] = {"repr_arms": scfg.get("repr_arms", [PRIMARY]),
                             "variance_floor": 0.014, "arms": sweep}

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

    # ---------------- ⑧ 收敛轨迹（判据量在 6k/12k/18k/24k 四点） ----------------
    note("收敛轨迹（§17.15）：判据量在 ckpt@6k/12k/18k + 终点直接测")
    steps_full = int(scfg["steps_full"])
    traj = {}
    for st in scfg["traj_ckpts"]:
        row = measure_traj_point(scfg, int(st), cfg, eval_arr, vocab, device)
        traj[str(st)] = row
        if row.get("available"):
            note(f"  @{st}: stock@K1 {row['arms']['stock']['K1_bpc']} "
                 f"l1@K8 {row['arms']['l1']['K8_bpc']} repr01@K8 {row['arms'][PRIMARY]['K8_bpc']} "
                 f"| 主判据 {row['main_criterion']} | 机制净 {row['mechanism_net']}")
        else:
            note(f"  @{st}: 不可得（{row.get('note')}）")

    def _slope(vals):
        """(终点 − 该点) / (Δ步/1k)，对每个可用轨迹点给出判据量的**变化率**
        （负 = 判据量随训练下降；符号约定与 X1 `2622` run.py 一致）。"""
        out = {}
        for st, row in vals.items():
            if not row.get("available"):
                continue
            d = (steps_full - int(st))
            if d <= 0:
                continue
            out[st] = {
                "main_criterion_at_ckpt": row["main_criterion"],
                "mechanism_net_at_ckpt": row["mechanism_net"],
                "main_slope_per_1k": round((full_main - row["main_criterion"]) / (d / 1000.0), 6),
                "mechanism_slope_per_1k": round((full_mech - row["mechanism_net"]) / (d / 1000.0), 6),
            }
        return out

    full_main = round(bk[PRIMARY] - b1["stock"], 6)
    full_mech = round(bk["l1"] - bk[PRIMARY], 6)
    results["trajectory"] = {
        "checkpoints": traj, "full_run_steps": steps_full,
        "full_run": {"repr01@K8": bk[PRIMARY], "stock@K1": b1["stock"], "l1@K8": bk["l1"],
                     "main_criterion_value": full_main, "mechanism_net_value": full_mech},
    }
    results["convergence_gate"] = {
        "quantity": ("决定判据的量 = 主判据缺口 `repr01@K8 − stock@K1` 与机制净效应 "
                     "`l1@K8 − repr01@K8`（评测流直接测），不是训练总 CE"),
        "why": ("SCHEMA §17.15：门禁必须落在决定判据的量上；两个量都在 ckpt@6k/12k/18k 与终点"
                "各测一次"),
        "per_checkpoint": _slope(traj),
        "main_criterion_at_full": full_main,
        "mechanism_net_at_full": full_mech,
        "train_loss_slopes": {a: results["loss_slope"][a]["slope"]
                              for a in results["loss_slope"]},
        "train_loss_note": "训练总 CE 斜率仅作并报，不得代替上面的判据量斜率",
    }

    # ---------------- ⑨ @6k 跨运行对照（X1 2622，调度不同） ----------------
    x1_path = os.path.join(X1_DIR, "results_13m.json")
    if os.path.exists(x1_path):
        x1 = json.load(open(x1_path))
        cmp6 = {"schedule_note": ("X1 是 6000 步 cosine 的**末点**（lr→0.1×base）；本实验是 24000 步 "
                                  "cosine，其 @6k 处 lr≈0.868×base ⇒ 两者**调度不同、不可作逐位对照**，"
                                  "此处只登记同方法读数的方向"),
                "x1_source": os.path.relpath(x1_path, ROOT),
                "x1_6000_arm_readings": {
                    "stock_K1": x1["k_curve"]["stock"]["1"]["bpc"],
                    "l1_K1": x1["k_curve"]["l1"]["1"]["bpc"],
                    "l1_K8": x1["k_curve"]["l1"]["8"]["bpc"],
                    "repr01_K8": x1["k_curve"]["repr01"]["8"]["bpc"],
                    "l1_K8_minus_repr01_K8": round(x1["k_curve"]["l1"]["8"]["bpc"]
                                                   - x1["k_curve"]["repr01"]["8"]["bpc"], 6)},
                "this_run_6000_arm_readings": (
                    {k: v for k, v in traj.get("6000", {}).get("arms", {}).items()
                     if isinstance(v, dict)} if traj.get("6000") else {})}
        if traj.get("6000", {}).get("available"):
            cmp6["this_run_6000_criterion"] = traj["6000"]["main_criterion"]
            cmp6["this_run_6000_mechanism_net"] = traj["6000"]["mechanism_net"]
        results["x1_6k_cross_check"] = cmp6

    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "note": "本进程净分配（不含共享卡上其他任务的占用），与训练侧 peak_gpu_mem_mb 同口径"}

    # ---------------- ⑩ 复现信息 ----------------
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
            + (len(scfg["traj_ckpts"]) + 1) * (3 + 2 * cfg["main_k"]) * nb),
        "note": "每批 = bs×T 个 token 的一次前向（K 路径就是 K 次前向，价签的来源）",
    }
    code_files = ["train_m1.py", "pathlm/model.py", "pathlm/config.py",
                  "pathlm/eval.py", "tools/evalkit/ensemble_probe.py", "probe_tts.py",
                  "reconcile_numbers.py"]
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
