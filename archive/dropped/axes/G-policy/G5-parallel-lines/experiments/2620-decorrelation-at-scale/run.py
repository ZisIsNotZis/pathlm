"""2620 · 去相关的规模验证 —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**预登记，不许改）。
训练：`python3 train_m1.py <arm> --config configs/<arm>.json`（见 `MODEL.md`）。
本入口只做**评测** + 汇总，跑八件事：

  ① 仪器自检：本入口的 K 路径集成函数与 `probe_tts.bpc_k`（13M 锚点的同一方法）
     在同一检查点上**逐位同值**；
  ② K 曲线（K=1/2/4/8，60-batch probe 协议）三臂 + 每档 60 批记录；
  ③ 成对 JS（路径相关度）；
  ④ 确定性路径对照（stock 臂 + divl1 强制 `p_skip=0`）——增益应恰好 0；
  ⑤ 主判据（读法 B）+ 并列判据（读法 A）：配对 bootstrap CI，并断言两种写法的代数等价；
  ⑥ 200-batch battery clean bpc（与仓库公开锚 `B0 1.5074` 同协议）；
  ⑦ 活性量与 loss 斜率（同方法）读训练产物；
  ⑧ 口径/复现信息：code sha256、墙钟、显存、锁状态。

用法：
  python3 run.py --smoke     # 冒烟：2 批 × K∈{1,2}
  python3 run.py             # 全电池
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

ARMS = ("stock", "l1", "divl1")


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


def load_arm(arm: str, cfg: dict, vocab: int, device: str) -> tuple:
    mcfg = json.load(open(os.path.join(HERE, "configs", f"{arm}.json")))
    path = os.path.join(HERE, "train", arm, "model.pt")
    if not os.path.exists(path):
        raise SystemExit(f"缺权重 {path}（先跑 train_all.sh）")
    want = cfg["arms"][arm].get("sha256")
    got = sha256_file(path)
    if not want or want.startswith("TO_FILL"):
        raise SystemExit(f"{arm}: config.json 里没有可用 sha256（训练后跑 fill_config_hashes.py）")
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch ({arm}):\n  want {want}\n  got  {got}")
    model = PathLM(ModelConfig(**mcfg["model"]), PathConfig(**mcfg["path"]), vocab_size=vocab)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "resolved": mcfg,
                   "params": sum(p.numel() for p in model.parameters())}


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
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-battery", action="store_true",
                    help="砍 200-batch battery 腿（README §6 的预算砍法 ①）")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["eval"] = {**cfg["eval"], "n_batches": 2}
        cfg["probe_seed"] = 11
    log: list[dict] = []
    t0 = time.time()
    device = cfg["device"]
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    vocab, ps = cfg["vocab_size"], cfg["probe_seed"]
    note("载入 enwik8 val（后 10M 字符，训练从不碰）")
    _, eval_arr, vocab_data = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    if vocab_data != vocab:
        raise SystemExit(f"vocab mismatch: config {vocab} vs data {vocab_data}")

    note("载入三臂（model.pt sha256 断言）")
    models, ck = {}, {}
    for arm in ARMS:
        models[arm], ck[arm] = load_arm(arm, cfg, vocab, device)
        note(f"  {arm}: sha256 {ck[arm]['sha256'][:16]}… params {ck[arm]['params']}")

    results: dict = {"config": cfg, "checkpoints": ck}
    results["scope"] = {
        "口径": (f"matched-step 快照（{cfg['steps_full']} 步 × bs 16 × T 512 = "
                 f"{cfg['steps_full'] * 16 * 512 / 1e6:.1f}M tokens）；K 曲线 = 60-batch probe "
                 f"协议（与 13M 锚点同方法）"),
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

    # ---------------- ② K 曲线（三臂 × K=1/2/4/8） ----------------
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
    note("确定性路径对照（增益应恰好 0）：stock 臂 + divl1 强制 p_skip=0")
    det = {}
    for tag, arm in (("stock", "stock"), ("divl1_forced", "divl1")):
        r = {}
        for k in cfg["determinism_k"]:
            run = ensemble(models[arm], eval_arr, k, cfg, deterministic=(tag != "stock"))
            r[f"K{k}"] = {kk: vv for kk, vv in run.items() if kk != "per_batch"}
        r["gain_K1_minus_Kmax"] = round(r[f"K1"]["bpc"] - r[f"K{max(cfg['determinism_k'])}"]["bpc"], 6)
        r["paths_bitwise_identical_all"] = all(
            r[f"K{k}"].get("paths_bitwise_identical", False) for k in cfg["determinism_k"] if k > 1)
        det[tag] = r
        note(f"  {tag}: K1 {r['K1']['bpc']} → K{max(cfg['determinism_k'])} "
             f"{r[f'K{max(cfg['determinism_k'])}']['bpc']} | 增益 {r['gain_K1_minus_Kmax']} "
             f"| 路径逐位相同={r['paths_bitwise_identical_all']}")
    results["deterministic_path_control"] = det

    # ---------------- ③ 成对 JS（K=8） ----------------
    note("成对 JS（路径相关度，K=%d）" % max(cfg["js_k"]))
    js = {}
    for arm in ARMS:
        kmax = str(max(cfg["js_k"]))
        js[arm] = curves[arm][kmax].get("pairwise_js")
        note(f"  {arm}: {js[arm]['mean_bits'] if js[arm] else None} bits "
             f"({js[arm]['mean_nats'] if js[arm] else None} nats)")
    results["pairwise_js_at_kmax"] = js

    # ---------------- ⑤ 判据（两种读法 + 代数等价断言） ----------------
    note("判据：主（读法 B）/ 并列（读法 A）")
    kk = str(cfg["main_k"])
    k1 = "1"
    b1 = {a: curves[a][k1]["bpc"] for a in ARMS}
    bk = {a: curves[a][kk]["bpc"] for a in ARMS}
    # 主判据按预登记形式写一遍，再断言与化简式一致（§12 算术对账）
    rb = reading_B(b1["l1"], b1["stock"], bk["divl1"])
    ra = reading_A(b1["l1"], bk["divl1"])
    b1_pb = {a: per_batch_bpc(curves[a][k1]) for a in ARMS}
    bk_pb = {a: per_batch_bpc(curves[a][kk]) for a in ARMS}

    ci_stock_vs_div = paired_delta_ci(b1_pb["stock"], bk_pb["divl1"], seed=1)
    ci_l1_vs_div = paired_delta_ci(b1_pb["l1"], bk_pb["divl1"], seed=2)
    ci_diversity_tax = paired_delta_ci(b1_pb["l1"], b1_pb["divl1"], seed=3)
    ci_mechanism_tax = paired_delta_ci(b1_pb["stock"], b1_pb["l1"], seed=4)

    crit = {
        "main_reading_B": {
            "formula": "[bpc_l1@K1 - bpc_divl1@K8] - [bpc_l1@K1 - bpc_stock@K1]",
            "baseline": "stock (p_skip=0, w_diversity=0, same tokens/seed)",
            "cost_term": "bpc_l1@K1 - bpc_stock@K1 (same protocol as the baseline read)",
            "primary": True,
            "value_verbatim_form": round(rb["verbatim"], 6),
            "value_reduced_form": round(rb["reduced"], 6),
            "algebra_consistent": rb["algebra_consistent"],
            "ci": ci_stock_vs_div,
            "passed_point_estimate": rb["passed_point_estimate"],
            "passed": bool(rb["passed_point_estimate"]
                           and ci_stock_vs_div["paired_ci_excludes_zero"]),
            "compute_price_tag": "divl1@K8 = 8x forwards/token; stock@K1 = 1x",
        },
        "parallel_reading_A": {
            "formula": "bpc_l1@K1 - bpc_divl1@K8",
            "baseline": "l1 (p_skip=0.1, w_diversity=0)",
            "cost_term": "bpc_divl1@K1 - bpc_l1@K1",
            "primary": False,
            "value": round(ra["value"], 6),
            "ci": ci_l1_vs_div,
            "passed_point_estimate": ra["passed_point_estimate"],
            "passed": bool(ra["passed_point_estimate"]
                           and ci_l1_vs_div["paired_ci_excludes_zero"]),
        },
        "taxes": {
            "diversity_pressure_clean_tax_vs_l1": round(b1["divl1"] - b1["l1"], 6),
            "diversity_pressure_clean_tax_vs_l1_ci": ci_diversity_tax,
            "mechanism_clean_tax_vs_stock": round(b1["l1"] - b1["stock"], 6),
            "mechanism_clean_tax_vs_stock_ci": ci_mechanism_tax,
        },
        "variance_floor": {"delta_unorderable_below": 0.014,
                           "note": "docs/methodology.md：Δ < 0.014 不可排序"},
    }
    note(f"  主判据（读法 B）：{crit['main_reading_B']['value_reduced_form']} "
         f"CI {ci_stock_vs_div['delta_ci']} ⇒ 过={crit['main_reading_B']['passed']}")
    note(f"  并列判据（读法 A）：{crit['parallel_reading_A']['value']} "
         f"CI {ci_l1_vs_div['delta_ci']} ⇒ 过={crit['parallel_reading_A']['passed']}")
    results["criteria"] = crit

    # ---------------- ⑥ 200-batch battery（公开协议参照） ----------------
    if cfg.get("battery", {}).get("enabled", True) and not args.no_battery:
        note("200-batch battery clean bpc（与公开锚 B0 1.5074 同协议）")
        results["battery_clean_bpc"] = {}
        for arm in ARMS:
            r = battery_bpc(models[arm], eval_arr, vocab,
                            n_batches=cfg["battery"]["n_batches"])
            results["battery_clean_bpc"][arm] = r
            note(f"  {arm}: battery bpc {r['bpc']} acc {r['next_token_acc']}")

    # ---------------- ⑦ 活性量 / 斜率 / 显存 ----------------
    note("活性量与 loss 斜率（同方法）读训练产物")
    results["train_products"], results["loss_slope"] = {}, {}
    for arm in ARMS:
        rp = os.path.join(HERE, "train", arm, "results.json")
        lp = os.path.join(HERE, "train", arm, "train_log.jsonl")
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
        results["loss_slope"][arm] = loss_slope(lines, lo=200, hi=int(cfg["steps_full"]))
        note(f"  {arm}: wall {res.get('wall_minutes')} min, peak {res.get('peak_gpu_mem_mb')} MB, "
             f"slope {results['loss_slope'][arm]['slope']}, "
             f"div_js_last {res.get('intervention', {}).get('diversity_js_last')}, "
             f"skip_frac_mean {res.get('intervention', {}).get('skip_frac_mean')}, "
             f"gate_skips {res.get('intervention', {}).get('gate_skips')}")

    # ---------------- ⑦b 收敛门禁：落在**决定判据的那个量**上（SCHEMA §17.15） ----------------
    # 判据是评测流上的 clean bpc（不是训练 loss）：总 CE 会被已学会的部分掩盖，所以门禁
    # 与趋势都在**评测 bpc**上取，同一方法（60-batch probe）在 3000 / 6000 两个点各测一次；
    # 判据量本身（`stock@K1 − divl1@K8`）也在这两个点上直接测。
    traj = {}
    ckpt_root = cfg.get("trajectory", {}).get("ckpt_root", ".tmp/v3-ckpts")
    ck_steps = cfg.get("trajectory", {}).get("ckpt_steps", [3000])
    if cfg.get("trajectory", {}).get("enabled", True):
        for st in ck_steps:
            row = {}
            for arm in ARMS:
                p = os.path.join(ROOT, ckpt_root, arm, f"model_step{st:06d}.pt")
                if not os.path.exists(p):
                    continue
                mcfg = json.load(open(os.path.join(HERE, "configs", f"{arm}.json")))
                m = PathLM(ModelConfig(**mcfg["model"]), PathConfig(**mcfg["path"]),
                           vocab_size=vocab)
                m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                m.to(device).eval()
                for q in m.parameters():
                    q.requires_grad_(False)
                r1 = ensemble(m, eval_arr, 1, cfg)
                row[arm] = {"K1_bpc": r1["bpc"], "K1_ci": bootstrap_ci(per_batch_bpc(r1), seed=7)}
                if arm == "divl1":
                    rk = ensemble(m, eval_arr, cfg["main_k"], cfg)
                    row[arm][f"K{cfg['main_k']}_bpc"] = rk["bpc"]
                    row[arm][f"K{cfg['main_k']}_ci"] = bootstrap_ci(per_batch_bpc(rk), seed=8)
                    row[arm][f"K{cfg['main_k']}_per_batch"] = per_batch_bpc(rk)
                row[arm]["K1_per_batch"] = per_batch_bpc(r1)
                del m
            if row and all(a in row for a in ARMS) and f"K{cfg['main_k']}_bpc" in row["divl1"]:
                row["criterion_value"] = round(row["stock"]["K1_bpc"]
                                               - row["divl1"][f"K{cfg['main_k']}_bpc"], 6)
                row["criterion_delta_ci"] = paired_delta_ci(
                    row["stock"]["K1_per_batch"], row["divl1"][f"K{cfg['main_k']}_per_batch"],
                    seed=9)
                row["trajectory_label"] = "criterion quantity measured at this checkpoint"
            traj[str(st)] = row
            note(f"  轨迹 @{st} 步：" + " ".join(
                f"{a}@K1 {row[a]['K1_bpc']}" for a in ARMS if a in row)
                + (f" | 判据量 {row.get('criterion_value')}" if "criterion_value" in row else ""))
        full = {"stock@K1": b1["stock"],
                f"divl1@K{cfg['main_k']}": bk["divl1"],
                "criterion_value": round(b1["stock"] - bk["divl1"], 6)}
        results["trajectory"] = {"checkpoints": traj, "full_run_steps": cfg["steps_full"],
                                 "full_run": full}
        # §17.15：门禁与趋势都落在判据量上；两套斜率并报（训练 loss vs 评测 bpc）
        if ck_steps:
            st = max(ck_steps)
            if str(st) in traj and "criterion_value" in traj[str(st)]:
                d = cfg["steps_full"] - st
                results["convergence_gate"] = {
                    "quantity": "eval-stream clean bpc / the criterion quantity itself",
                    "why": ("SCHEMA §17.15：收敛门禁必须落在决定判据的量上；本例判据 = 评测 bpc "
                            "的差，不是训练总 CE"),
                    "criterion_value_at_ckpt": traj[str(st)]["criterion_value"],
                    "criterion_value_at_full": full["criterion_value"],
                    "criterion_slope_per_1k": round(
                        (full["criterion_value"] - traj[str(st)]["criterion_value"])
                        / (d / 1000.0), 6) if d else None,
                    "stock_K1_slope_per_1k": round(
                        (b1["stock"] - traj[str(st)]["stock"]["K1_bpc"]) / (d / 1000.0), 6),
                    "divl1_Kmax_slope_per_1k": round(
                        (bk["divl1"] - traj[str(st)]["divl1"][f"K{cfg['main_k']}_bpc"])
                        / (d / 1000.0), 6),
                    "train_loss_slopes": {a: results["loss_slope"][a]["slope"] for a in ARMS},
                    "train_loss_note": ("训练总 CE 斜率仅作并报（§17.15：它会被已学会的部分掩盖，"
                                        "不得用它代替上面的判据量斜率）"),
                    "note": ("50M/6000 步是匹配 13M 锚点的同步数快照；上面两个斜率说明判据量在这个"
                             "预算下是仍在移动（未收敛 ⇒ 快照口径）还是已平"),
                }
            else:
                results["convergence_gate"] = {
                    "available": False,
                    "quantity": "eval-stream clean bpc / the criterion quantity itself",
                    "why": ("SCHEMA §17.15：门禁必须落在决定判据的量上；本次未拿到中间检查点"
                            "（预算降步后 ckpt 未落盘）⇒ 只能标快照口径"),
                    "train_loss_slopes": {a: results["loss_slope"][a]["slope"] for a in ARMS},
                    "train_loss_note": "训练总 CE 仅作并报，不得代替判据量",
                }
        else:
            results["convergence_gate"] = {
                "available": False, "quantity": "n/a（未安排中间检查点）",
                "train_loss_slopes": {a: results["loss_slope"][a]["slope"] for a in ARMS}}
    else:
        results["convergence_gate"] = {
            "available": False, "quantity": "n/a（trajectory 关闭）",
            "train_loss_slopes": {a: results["loss_slope"][a]["slope"] for a in ARMS}}
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "note": "本进程净分配（不含共享卡上其他任务的占用），与训练侧 peak_gpu_mem_mb 同口径"}

    # ---------------- ⑧ 复现信息 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "gpu_wall_budget_min": cfg["gpu_wall_budget_min"],
        "train_wall_minutes": {a: results["train_products"][a]["wall_minutes"] for a in ARMS},
        "n_forward_batches": (
            sum(cfg["k_sweep"]) * len(ARMS) * cfg["eval"]["n_batches"]      # 主 K 曲线
            + sum(cfg["selfcheck_k"]) * cfg["eval"]["n_batches"] * 2          # 仪器自检（两边各跑）
            + sum(cfg["determinism_k"]) * cfg["eval"]["n_batches"] * 2        # 确定性路径对照
            + (cfg["battery"]["n_batches"] * len(ARMS)
               if cfg.get("battery", {}).get("enabled", True) and not args.no_battery else 0)),
        "note": "每批 = bs×T 个 token 的一次前向（K 路径就是 K 次前向，价签的来源）",
    }
    code_files = ["train_m1.py", "pathlm/model.py", "pathlm/config.py",
                  "pathlm/eval.py", "tools/evalkit/ensemble_probe.py", "probe_tts.py",
                  "tests/test_decorrelation_at_scale.py", "reconcile_numbers.py"]
    code = {"run.py": sha256_file(os.path.join(HERE, "run.py"))}
    for p in code_files:
        cand = os.path.join(HERE, p)
        path = cand if os.path.exists(cand) else os.path.join(ROOT, p)
        code[p] = sha256_file(path) if os.path.exists(path) else None
    code["README.md (本契约)"] = sha256_file(os.path.join(HERE, "README.md"))
    results["code"] = code
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(HERE, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{args.out}（评测墙钟 {results['cost']['wall_minutes_total_eval']} min）")


if __name__ == "__main__":
    main()
