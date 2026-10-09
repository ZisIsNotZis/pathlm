"""2627 · G5 机制定论抖动补点（ticket 44 X1）—— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**预登记，不许改）。
训练：`bash train_all.sh train 13m`（单臂 repr01，同 2626 配置、同 24k 步、seed=0）。

本入口只做**评测** + 汇总（一个 scale 一次），对象是**三个检查点**：
  - `l1_A`      = 2626 的 `l1` 臂（复用既有读数；机制净效应的被减数）
  - `repr01_A`  = 2626 的 `repr01` 臂（既有 run A）
  - `repr01_B`  = 2627 本轮同配置重跑的 `repr01`（run B）

  ① 仪器自检：本入口的 K 路径集成函数与 `probe_tts.bpc_k`（13M 锚点同一方法）在 `l1_A` 上逐位同值；
  ② **跨运行复现检查**：本轮对 `l1_A`/`repr01_A` 的新读数应与 2626 `results_13m.json` 记录**逐位相同**
     （评测是确定性的，seed=11 同一位置集）；
  ③ K 曲线（K=1/8，60-batch probe 协议）三臂 + 每档 60 批；
  ④ 成对 JS（K=8，路径相关度）；
  ⑤ **抖动判决（本实验一号问题）**：两次 repr01 的机制净效应 `l1@K8 − repr01@K8` 的**差值/极差**，
     按 README §4 预登记阈值定为 稳健/仍贴门槛/不可排序；
  ⑥ §17.17 两列分解（clean 税增量 vs 集成增益增量）；
  ⑦ 确定性路径对照（`repr01_B` 强制 p_skip=0，增益应恰好 0）；
  ⑧ 活性量（含 gate 数，须为 0）与训练 CE 斜率（`§17.15` 并报）；
  ⑨ 200-batch battery clean bpc（`repr01_A`/`repr01_B` 的独立 run-to-run 读数）；
  ⑩ 口径/复现信息：code sha256、墙钟、显存、算力账。

用法：
  python3 run.py --scale 13m --smoke   # 冒烟：2 批 × K∈{1}
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
sys.path.insert(0, HERE)
os.chdir(ROOT)   # 数据路径（.tmp/enwik8, data/enwik8_full.npz）是相对仓库根的

from pathlm.config import ModelConfig, PathConfig                    # noqa: E402
from pathlm.data import load_enwik8_full                             # noqa: E402
from pathlm.eval import bpc as battery_bpc                           # noqa: E402
from pathlm.model import PathLM                                      # noqa: E402
from tools.evalkit.carry_channel import loss_slope                   # noqa: E402
from tools.evalkit.ensemble_probe import (                           # noqa: E402
    deterministic_paths, paired_delta_ci, path_ensemble_bpc,
    per_batch_bpc)
from jitter import EFFECT_V2, jitter_verdict, mechanism_net          # noqa: E402


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


def load_arm(cfg: dict, scale: str, arm: str, vocab: int, device: str) -> tuple[PathLM, dict]:
    entry = cfg["arms"][f"{scale}/{arm}"]
    cfg_path = os.path.join(ROOT, entry["config_file"])
    ckpt_path = os.path.join(ROOT, entry["ckpt_file"])
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"缺权重 {entry['ckpt_file']}（先跑 train_all.sh train）")
    mcfg = json.load(open(cfg_path))
    got = sha256_file(ckpt_path)
    want = entry.get("sha256")
    if want and not str(want).startswith("TO_FILL") and got != want:
        raise SystemExit(f"checkpoint sha256 mismatch ({arm}):\n  want {want}\n  got  {got}")
    model = build_model(mcfg["model"], mcfg["path"], vocab, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, {
        "config_file": entry["config_file"], "ckpt_file": entry["ckpt_file"],
        "sha256": got, "size_bytes": os.path.getsize(ckpt_path), "resolved": mcfg,
        "params": sum(p.numel() for p in model.parameters()),
        "source": entry.get("source"), "role": cfg["arms_meta"][arm]["role"]}


def ensemble(arm_model, eval_arr, k: int, cfg: dict, *, collect_js=False,
             deterministic=False, n_batches=None) -> dict:
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
    ap.add_argument("--scale", choices=("13m",), required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-battery", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    scfg = cfg["scales"][args.scale]
    ARMS = tuple(scfg["arms"])
    BASE = scfg["mechanism_baseline"]
    A, B = scfg["repr_pair"]
    if args.smoke:
        cfg["eval"] = {**cfg["eval"], "n_batches": 2}
        cfg["k_sweep"] = [1]
        cfg["js_k"] = []
    out = os.path.join(HERE, f"results_{args.scale}.json")
    log: list[dict] = []
    t0 = time.time()
    device = cfg["device"]
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    vocab = cfg["vocab_size"]
    note(f"scale={args.scale} 载入 enwik8 val（后 10M 字符，训练从不碰）")
    _, eval_arr, vocab_data = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    if vocab_data != vocab:
        raise SystemExit(f"vocab mismatch: config {vocab} vs data {vocab_data}")

    note("载入三臂（model.pt sha256 断言）")
    models, ck = {}, {}
    for arm in ARMS:
        models[arm], ck[arm] = load_arm(cfg, args.scale, arm, vocab, device)
        note(f"  {arm}: sha256 {ck[arm]['sha256'][:16]}… params {ck[arm]['params']} "
             f"({ck[arm]['source']})")

    results: dict = {"scale": args.scale, "config": cfg, "checkpoints": ck}
    results["scope"] = {
        "口径": (f"matched-step 快照（{scfg['steps_full']} 步、24000 步 cosine 调度；K 曲线 = "
                 f"60-batch probe 协议，与 13M 锚点同方法）"),
        "单点不外推": True,
        "本实验只做单轴单格": ("只重跑 repr01 一臂；不改引擎、不改共享文件；"
                              "50M 的 0.0322 缺口不被本实验覆盖"),
        "算力价签": "K8 集成 = 8× 前向/token；l1 单线 = 1× 前向/token",
    }

    # ---------------- ① 仪器自检：与 probe_tts.bpc_k 同方法 ----------------
    note("自检 ①：本入口 K 集成 vs probe_tts.bpc_k（13M 锚点同一方法）")
    pt = load_probe_tts()
    instrument = {}
    for k in cfg["selfcheck_k"]:
        mine = ensemble(models[BASE], eval_arr, k, cfg)
        theirs = pt.bpc_k(models[BASE], eval_arr, k,
                          n_batches=cfg["eval"]["n_batches"],
                          batch_size=cfg["eval"]["batch_size"])
        instrument[f"K{k}"] = {
            "mine_bpc": mine["bpc"], "probe_tts_bpc": theirs["bpc"],
            "bpc_equal": mine["bpc"] == theirs["bpc"],
            "mine_acc": mine["acc"], "probe_tts_acc": theirs["acc"],
            "n_batches": cfg["eval"]["n_batches"], "seed": cfg["probe_seed"]}
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
                    if "pairwise_js" in run else ""))
    results["k_curve"] = curves

    # ---------------- ③ 跨运行复现检查（本轮对 2626 权重的新读数应逐位相同）----------------
    v2 = cfg["source_v2"]
    kmax = str(cfg["main_k"])
    repro = {
        "note": ("评测确定性检查（seed=11 同一位置集）：本轮对 2626 权重的重评应与 2626 "
                 "`results_13m.json` 记录的读数**逐位相同**；若是，则两次 repr01 的差值只反映"
                 "**训练**抖动，不含评测抖动"),
        "source": f"{v2['dir']}/{v2['results']}",
        "checks": {
            "l1_K8": {"recorded_2626": v2["k8_l1"], "this_run": curves[BASE][kmax]["bpc"],
                      "equal": curves[BASE][kmax]["bpc"] == v2["k8_l1"]},
            "repr01_A_K8": {"recorded_2626": v2["k8_repr01"],
                            "this_run": curves[A][kmax]["bpc"],
                            "equal": curves[A][kmax]["bpc"] == v2["k8_repr01"]},
            "l1_K1": {"recorded_2626": v2["k1_l1"], "this_run": curves[BASE]["1"]["bpc"],
                      "equal": curves[BASE]["1"]["bpc"] == v2["k1_l1"]},
            "repr01_A_K1": {"recorded_2626": v2["k1_repr01"], "this_run": curves[A]["1"]["bpc"],
                            "equal": curves[A]["1"]["bpc"] == v2["k1_repr01"]},
        },
    }
    repro["all_equal"] = all(v["equal"] for v in repro["checks"].values())
    note(f"自检 ②：本轮重评 2626 权重 vs 记录 逐位相同 = {repro['all_equal']}")
    results["cross_run_reproduction"] = repro

    # ---------------- ④ 抖动判决（本实验一号问题）----------------
    note("抖动判决 ④：两次 repr01 的机制净效应 `l1@K8 − repr01@K8`")
    l1_k8 = curves[BASE][kmax]["bpc"]
    k1_base = curves[BASE]["1"]["bpc"]
    net_A = round(mechanism_net(l1_k8, curves[A][kmax]["bpc"]), 6)
    net_B = round(mechanism_net(l1_k8, curves[B][kmax]["bpc"]), 6)
    repr_a_k8, repr_b_k8 = curves[A][kmax]["bpc"], curves[B][kmax]["bpc"]
    diff = round(net_B - net_A, 6)          # = repr01_A@K8 − repr01_B@K8（被减数 l1@K8 相消）
    rng = abs(diff)
    verdict = jitter_verdict(rng)
    ci_pair = paired_delta_ci(per_batch_bpc(curves[B][kmax]), per_batch_bpc(curves[A][kmax]),
                              seed=21)

    def tax(arm: str) -> float:
        return round(curves[arm]["1"]["bpc"] - k1_base, 6)

    def gain(arm: str) -> float:
        return round(curves[arm]["1"]["bpc"] - curves[arm][kmax]["bpc"], 6)

    jitter = {
        "quantity": "机制净效应 `bpc_l1@K8 − bpc_repr01@K8`（两次 repr01 读数）",
        "baseline_l1_K8": l1_k8,
        "repr01_A_K8": repr_a_k8, "repr01_B_K8": repr_b_k8,
        "net_A": net_A, "net_B": net_B,
        "diff": diff, "range": rng,
        "range_over_effect": round(rng / EFFECT_V2, 6) if EFFECT_V2 else None,
        "verdict": verdict,
        "repr01_pair_ci": ci_pair,
        "component_decomposition": {   # §17.17：净和不可排序时给两列
            "clean_tax_A_vs_l1": tax(A), "clean_tax_B_vs_l1": tax(B),
            "clean_tax_diff": round(tax(B) - tax(A), 6),
            "ensemble_gain_A": gain(A), "ensemble_gain_B": gain(B),
            "ensemble_gain_diff": round(gain(B) - gain(A), 6),
            "note": ("净效应 = 集成增益 − clean 税（相对 l1）；两列分开报，用以区分"
                     "『机制无用』与『机制有效但代价抵消』"),
        },
        "effect_v2": EFFECT_V2,
        "variance_floor": cfg["variance_floor"],
    }
    note(f"  net_A {net_A} / net_B {net_B} / 差 {diff} / 极差 {rng:.6f} "
         f"⇒ {verdict['state']}")
    results["jitter"] = jitter

    # ---------------- ⑤ 成对 JS（K=8）----------------
    if cfg["js_k"]:
        note("成对 JS（路径相关度，K=%d）" % max(cfg["js_k"]))
        kmax_s = str(max(cfg["js_k"]))
        js = {arm: curves[arm][kmax_s].get("pairwise_js") for arm in ARMS}
        for arm in ARMS:
            note(f"  {arm}: {js[arm]['mean_bits'] if js[arm] else None} bits")
        results["pairwise_js_at_kmax"] = js
    else:
        results["pairwise_js_at_kmax"] = {arm: None for arm in ARMS}

    # ---------------- ⑥ 确定性路径对照 ----------------
    det_arm = scfg["det_arm"]
    note(f"确定性路径对照（增益应恰好 0）：{det_arm} 强制 p_skip=0")
    det = {}
    for k in (1, int(cfg["main_k"])):
        run = ensemble(models[det_arm], eval_arr, k, cfg, deterministic=True)
        det[f"K{k}"] = {kk: vv for kk, vv in run.items() if kk != "per_batch"}
    det["gain_K1_minus_Kmax"] = round(det["K1"]["bpc"] - det[f"K{cfg['main_k']}"]["bpc"], 6)
    det["paths_bitwise_identical_all"] = bool(
        det[f"K{cfg['main_k']}"].get("paths_bitwise_identical", False))
    det["arm"] = det_arm
    note(f"  {det_arm} forced: K1 {det['K1']['bpc']} → K{cfg['main_k']} "
         f"{det['K' + str(cfg['main_k'])]['bpc']} | 增益 {det['gain_K1_minus_Kmax']} "
         f"| 路径逐位相同={det['paths_bitwise_identical_all']}")
    results["deterministic_path_control"] = det

    # ---------------- ⑦ 200-batch battery（repr01 两 run 的独立读数）----------------
    if cfg.get("battery", {}).get("enabled", True) and not args.no_battery:
        note("200-batch battery clean bpc（与公开锚 B0 1.5074 同协议）")
        results["battery_clean_bpc"] = {}
        for arm in (A, B):
            r = battery_bpc(models[arm], eval_arr, vocab, n_batches=cfg["battery"]["n_batches"])
            results["battery_clean_bpc"][arm] = r
            note(f"  {arm}: battery bpc {r['bpc']} acc {r['next_token_acc']}")
        results["battery_clean_bpc"]["repr01_pair_diff"] = round(
            results["battery_clean_bpc"][B]["bpc"] - results["battery_clean_bpc"][A]["bpc"], 6)

    # ---------------- ⑧ 活性量 / 斜率（读训练产物）----------------
    note("活性量与 loss 斜率（同方法）读训练产物")
    results["train_products"], results["loss_slope"] = {}, {}
    arm_dirs = {A: ("2626", "repr01"), B: ("2627", "repr01"), BASE: ("2626", "l1")}
    for arm in ARMS:
        base_dir, sub = arm_dirs[arm]
        if base_dir == "2627":
            root = HERE
        else:
            root = os.path.join(ROOT, v2["dir"])
        rp = os.path.join(root, scfg["train_dir"], sub, "results.json")
        lp = os.path.join(root, scfg["train_dir"], sub, "train_log.jsonl")
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

    # ---------------- ⑨ 收敛门禁（§17.15，落在决定判据的量上）----------------
    results["convergence_gate"] = {
        "quantity": "机制净效应 `bpc_l1@K8 − bpc_repr01@K8`（决定本实验结论的那个量）",
        "why": ("SCHEMA §17.15：门禁必须落在决定判据的量上。本实验的两个 repr01 都在共享的 24k "
                "末点直接测（matched-step 快照）；2626 已证 24k 处主判据缺口/机制净效应末窗斜率分别 "
                "−0.001 / −0.0001 per 1k（基本收敛）——本实验**不改调度、不加步数**，只在同一点"
                "重测，故不再重复轨迹"),
        "mechanism_net_A": net_A, "mechanism_net_B": net_B,
        "repr01_A_train_loss_slope": results["loss_slope"][A]["slope"],
        "repr01_B_train_loss_slope": results["loss_slope"][B]["slope"],
        "train_loss_note": ("训练总 CE 斜率仅作**并报**，不得代替上面的判据量口径（§17.15）；"
                            "两 run 的 CE 斜率应接近（同配方、同预算、同 seed，只差 run-to-run 抖动）"),
        "checkpoint": "两 run 均在 24000 步 cosine 末点（lr→0.1×base）取权重",
    }

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
            + (1 + int(cfg["main_k"])) * nb
            + (cfg["battery"]["n_batches"] * 2
               if cfg.get("battery", {}).get("enabled", True) and not args.no_battery else 0)),
        "note": "每批 = bs×T 个 token 的一次前向（K 路径就是 K 次前向，价签的来源）",
    }
    code_files = ["train_m1.py", "pathlm/model.py", "pathlm/config.py",
                  "pathlm/eval.py", "tools/evalkit/ensemble_probe.py", "probe_tts.py"]
    code = {"run.py": sha256_file(os.path.join(HERE, "run.py")),
            "jitter.py": sha256_file(os.path.join(HERE, "jitter.py"))}
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
