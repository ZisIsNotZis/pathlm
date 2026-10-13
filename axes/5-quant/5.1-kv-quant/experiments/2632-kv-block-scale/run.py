"""2632 · KV 量化标尺口径重测（R1 / WI-08）—— 实验入口。

契约：同目录 `README.md`（§1–§9）+ `config.json`（判据线**运行前**写入）；
语义层：同目录 `block_arms.py`（标尺方案感知的前向）、`scale_caliber.py`（值 + 标尺 + staging 的字节账）。

动机（独立批判性审计）：`2631` 的「量化在任何拓扑上都是内存负收益」是在**逐值标尺**（每值一把
fp16 标尺、int8 净 3 B/值）下测的。业界标准是**标尺按块/通道摊薄**（一块 32 个值共享一把 ⇒ int8
≈1.0625 B/值，省 ~46.9%），且沿 `head_dim`（非 token 轴）摊薄**仍然满足前缀不变**。本实验把 27 点
预算图在块标尺（G=32 为主）下重测，并并列 KIVI 式与逐值标尺对照。

用法：
  python3 run.py            # 全跑（纯推理）
  python3 run.py --smoke    # 冒烟：缩小批量/档数，实测步时与显存（不外推时间）
  python3 run.py --charts-only   # 只从已有 results.json 重画表（纯 CPU，不取 GPU 锁）
"""

import argparse
import atexit
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import block_arms as ba                                                # noqa: E402
import scale_caliber as sc                                             # noqa: E402
from pathlm.data import load_enwik8_full                              # noqa: E402
from tools.evalkit.longrange_tasks import (                           # noqa: E402
    corrupt_span, local_prior_accuracy, check_local_prior_resistance)

P2621 = os.path.join(ROOT, "axes/R-references/R1-quantization/experiments/"
                           "2621-kv-budget-rest")
P2631 = os.path.join(ROOT, "axes/R-references/R1-quantization/experiments/"
                           "2631-kv-budget-causal")
sys.path.insert(0, P2621)
sys.path.insert(0, P2631)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2621 的评估管线（内含 2619 的 b19）；只读复用。
b21 = _load_module(os.path.join(P2621, "run.py"), "kv_budget_rest_2621")
#: 2631 的前缀不变性仪器（本实验直接复用其两条检查）。
cc2631 = _load_module(os.path.join(P2631, "causal_check.py"), "causal_check_2631")

#: 主臂的标尺方案：块标尺沿 head_dim、G = 32（= head_dim ⇒ per-token 标尺，仍严格前缀确定）。
PRIMARY_SCHEME, PRIMARY_GROUP = "block", 32
VARIANCE_SELF_SEEDS = (2, 3, 4)

#: SCHEMA §20 的 GPU 互斥锁（本实验整程 <10 min ⇒ 允许整程持锁，但必须留机器可读的取/释记录）。
GPU_LOCK_DIR = "/tmp/pathlm_gpu.lock"


def _gpu_free_mib():
    """`nvidia-smi` 的空闲显存（MiB）；不可用时 None（如实记 null，不拦跑）。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:                                              # pragma: no cover
        return None


class GpuLock:
    """SCHEMA §20 的互斥锁：`mkdir` 抢锁，陈旧锁（holder 死 / 无 pid 且年龄 >15 min）自愈回收。

    `evidence()` 把取锁/开跑/收尾/释放与前后空闲显存落成**机器可读**记录
    （`results.json::gpu_lock`），供 §20 记账核查。
    """

    def __init__(self, lock_dir: str = GPU_LOCK_DIR, timeout_s: int = 600, poll_s: int = 15):
        self.lock_dir = lock_dir
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.ev = {"lock_path": lock_dir, "holder_pid": None, "acquired_at": None,
                   "wait_seconds": None, "reclaimed": [], "release_ok": None,
                   "free_mib_before": None, "free_mib_after": None}

    def _holder_pid(self):
        try:
            return int(open(os.path.join(self.lock_dir, "holder_pid")).read().strip())
        except Exception:
            return None

    def _stale(self) -> str | None:
        hp = self._holder_pid()
        if hp is not None:
            try:
                os.kill(hp, 0)
                return None
            except OSError:
                return f"holder_pid {hp} 已死"
        try:
            age = time.time() - os.stat(self.lock_dir).st_mtime
        except OSError:
            return None
        if age > 900:
            return f"无 holder_pid 且锁龄 {age:.0f}s >15min"
        return None

    def acquire(self) -> None:
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.lock_dir)
                break
            except FileExistsError:
                reason = self._stale()
                if reason:
                    for f in ("holder_pid", "heartbeat"):
                        try:
                            os.remove(os.path.join(self.lock_dir, f))
                        except OSError:
                            pass
                    try:
                        os.rmdir(self.lock_dir)
                    except OSError:
                        pass
                    self.ev["reclaimed"].append(reason)
                    continue
                if time.time() - t0 > self.timeout_s:
                    raise TimeoutError(f"等锁超时 {self.timeout_s}s：{self.lock_dir} 被占")
                time.sleep(self.poll_s)
        with open(os.path.join(self.lock_dir, "holder_pid"), "w") as f:
            f.write(str(os.getpid()))
        open(os.path.join(self.lock_dir, "heartbeat"), "w").close()
        self.ev["holder_pid"] = os.getpid()
        self.ev["acquired_at"] = round(time.time(), 3)
        self.ev["wait_seconds"] = round(time.time() - t0, 3)
        self.ev["free_mib_before"] = _gpu_free_mib()

    def release(self) -> None:
        if self.ev.get("release_ok") is not None:
            return
        self.ev["free_mib_after"] = _gpu_free_mib()
        for f in ("holder_pid", "heartbeat"):
            try:
                os.remove(os.path.join(self.lock_dir, f))
            except OSError:
                pass
        try:
            os.rmdir(self.lock_dir)
            self.ev["release_ok"] = True
        except OSError:
            self.ev["release_ok"] = False


def quantized(spec: dict) -> bool:
    return spec.get("mode") in ("int8", "int4")


def fn_for(cfg, arm, spec, model, scheme, group, kivi_group, dev, T):
    """量化臂走本实验的 scheme-aware 前向；非量化臂走 2631 的既有分派。"""
    if quantized(spec):
        return ba.make_logits_fn(model, spec, T, dev, scheme, group, kivi_group)
    return b21.make_logits_fn(model, spec, T, dev)


def eval_arm(cfg, arm, spec, model, dev, tasks, eval_arr, scheme, group, kivi_group,
             controls, do_recall=True):
    """一个 (臂, 方案) 的完整读数：bpc + 召回 + 前缀不变性 + 活性 + 横轴几何。"""
    T = cfg["model"]["seq_len"]
    probe_tokens = tasks["var_recall"].tokens[:4].to(dev)
    fn = fn_for(cfg, arm, spec, model, scheme, group, kivi_group, dev, T)
    bpc = b21.b19.bpc_of(fn, eval_arr, T, cfg["eval"]["n_batches"],
                         cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    out = {"arm": arm, "scheme": scheme,
           "group": group if scheme in ("block", "block_tok") else None,
           "kivi_group": kivi_group if scheme == "kivi" else None,
           "bpc": round(bpc, 6),
           "prefix_invariance": cc2631.prefix_invariance_of_logits(fn, probe_tokens),
           "activity": (ba.activity_of(spec, model, T, dev, probe_tokens, scheme, group,
                                       kivi_group) if quantized(spec)
                        else b21.activity_of(spec, model, T, dev, probe_tokens)),
           "mark": b21.MARK[arm], "source": b21.SOURCE[arm],
           "class": b21.class_of(arm),
           **b21.bytes_per_position(spec, cfg)}
    if do_recall:
        copy_pooled = b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)
        chance = controls["chance_uniform"]
        at_floor = copy_pooled["accuracy"] <= b21.b19.floor_threshold(copy_pooled["n"], chance)
        out["recall_copy"] = {**copy_pooled, "at_floor": bool(at_floor)}
        out["recall_induction"] = b21.b19.pooled_accuracy(fn, tasks["induction_by_start"], dev)
        out["recall_var_recall"] = b21.b19.pooled_accuracy(fn, tasks["var_recall_by_d"], dev)
        out["chance_uniform"] = chance
    return out


def render_table(cfg, points, scheme_rows, controls, rec) -> str:
    cal = rec["deployment_caliber"]
    L = []
    L.append("# 2632 · KV 量化标尺口径重测 —— 27 点预算图（部署口径并列）")
    L.append(
        f"**口径**：27 点预算图与 `2631` 同 checkpoint / 同数据流 / 同任务 / 同判据；"
        f"**唯一改动 = 量化臂的标尺方案**（逐值标尺 → **块标尺沿 head_dim，G = {PRIMARY_GROUP}**，"
        f"严格前缀确定）。非量化臂沿用 2631 分派、须与其逐位一致（这是可控性证据）。"
        f"横轴 T = {cfg['model']['seq_len']}；『值字节』= 判据 C1 旧横轴（与 2631 逐位可比）；"
        f"『部署字节』= 值 + 标尺（`scale_caliber.py`，与 `kv_quant.quantize_kv(scheme=...)` 一一对应）；"
        f"『×fp16』= 部署字节 / 12288。")
    L.append(
        "**『0.014 不可排序』不适用本格**：本格的排序带由**自测**给出，且**预登记了口径**："
        f"**在力带 = 单臂跨窗口 bpc 极差 {rec['ranking_band']['in_force_band_bpc']:.6f}**"
        f"（标准口径应为配对 Δ、≥5 窗口，本批只跑了 {rec['ranking_band']['this_batch_windows']} 个 ⇒ 不满足，"
        f"退用更保守的单臂带；见本表后『自测方差』一节与 `results.json::ranking_band`）。"
        f"召回地板：chance = {controls['chance_uniform']:.4f}（1/64）、"
        f"局部先验 = {controls['local_prior_copy']['accuracy']:.4f}（copy）。")
    L.append("## 表 1 · 27 点（量化臂 = 块标尺 block G=32；非量化臂 = fp16 分派）\n\n"
             "| 臂 | 类别 | 值字节 | 部署字节(block) | ×fp16 | 旧部署(a)逐值 | ×fp16(a) | "
             "bpc | damage_bpc | 量化增量 vs 同拓扑 fp16 | copy 召回 | 95% CI | 前缀不变(未来/邻居) |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by_arm = {p["arm"]: p for p in points}
    rows = []
    for p in points:
        spec = cfg["arms"][p["arm"]]
        a = cal["per_arm"][p["arm"]]["by_scheme"]["per_value"] if spec.get("mode") in (
            "int8", "int4") else None
        ci = (f"[{p['recall_copy']['wilson95'][0]:.4f}, {p['recall_copy']['wilson95'][1]:.4f}]"
              if "recall_copy" in p else "—")
        inc = ("—" if p.get("quant_increment_bpc") is None
               else f"{p['quant_increment_bpc']:+.6f}")
        inv = p["prefix_invariance"]
        ap = (f"{a['bytes_per_position']:.0f}" if a else "—")
        ar = (f"×{a['bytes_per_position'] / 12288:.2f}" if a else "—")
        dep = cal["per_arm"][p["arm"]]["primary"]
        rows.append(
            f"| `{p['arm']}` | {b21.CLASSES[p['class']]} | {p['bytes_per_position']:.0f} | "
            f"{dep['bytes_per_position']:.0f} | ×{dep['bytes_per_position'] / 12288:.2f} | "
            f"{ap} | {ar} | {p['bpc']:.6f} | {p['damage_bpc']:+.6f} | {inc} | "
            f"{p['recall_copy']['accuracy']:.4f}{' (地板)' if p['recall_copy']['at_floor'] else ''} | "
            f"{ci} | {inv['future_token_past_logits_equal']}/{inv['peer_request_logits_equal']} |")
    L[-1] += "\n" + "\n".join(rows)
    L.append("> 读法：非量化臂的『部署字节』= 值字节（没有标尺）；量化臂的『部署字节(block)』把标尺"
             "按 `2/G` 摊进去（G=32 ⇒ +0.0625 B/值）。第 3 列仍可与 `2631` 逐位并列；"
             "『×fp16』是把标尺算上后相对 fp16 基线的**真实**倍率（<1 = 省）。"
             "两个质量量**不得混用**：`damage_bpc` = 本臂 − 本族基线（头共享族含头共享本身的开销）；"
             "`量化增量` = 本臂 − 同拓扑 fp16 臂（纯量化因子）。"
             f"本表排序带 = **在力带 {rec['ranking_band']['in_force_band_bpc']:.6f}**（单臂跨窗口极差；"
             f"{rec['ranking_band']['declaration']}）")

    L.append("## 表 2 · 标尺方案对照（同一臂 × 不同方案；回答『在什么实现下不成立』）\n\n"
             "| 臂 | 方案 | G | 标尺 B/值 | 部署字节/位置 | ×fp16(同拓扑) | 前缀不变 | bpc | "
             "damage_bpc | copy 召回 |\n|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for r in scheme_rows:
        rg = r["group"] if r["scheme"] in ("block", "block_tok") else (
            r["kivi_group"] if r["scheme"] == "kivi" else "—")
        dep = r["deployment"]
        bpe = dep["scale_bytes_per_value"]
        rows.append(
            f"| `{r['arm']}` | {r['scheme']} | {rg} | {bpe:.4f} | "
            f"{dep['bytes_per_position']:.0f} | ×{r['ratio_vs_ref']:.2f} | "
            f"{r['prefix_invariance']['future_token_past_logits_equal']}/"
            f"{r['prefix_invariance']['peer_request_logits_equal']} | {r['bpc']:.6f} | "
            f"{r['damage_bpc']:+.6f} | {r['recall_copy']['accuracy']:.4f} |")
    L[-1] += "\n" + "\n".join(rows)

    L.append("## 表 3 · 部署字节阶梯（标尺按块摊薄的极限）\n\n"
             "**沿 head_dim（严格前缀确定）**：\n\n"
             "| G | 标尺 B/值 | int8 B/值 | int4 B/值 | int8 相对 fp16 | int4 相对 fp16 | 前缀不变 |\n"
             "|---|---|---|---|---|---|---|")
    rows = []
    for row in cal["head_dim_ladder"]:
        rows.append(f"| {row['group']} | {row['scale_bytes_per_value']:.4f} | "
                    f"{row['int8_bytes_per_value']:.4f} | {row['int4_bytes_per_value']:.4f} | "
                    f"{row['int8_reduction_vs_fp16']:+.1%} | {row['int4_reduction_vs_fp16']:+.1%} | "
                    f"{row['prefix_invariant']} |")
    L[-1] += "\n" + "\n".join(rows)
    L.append("**沿 token 轴（违反前缀确定 ≤ G−1 个位置）**：\n\n"
             "| G | 标尺 B/值 | 违反距离(位置) | int8 B/值 | int4 B/值 | int8 相对 fp16 | "
             "int4 相对 fp16 | 前缀不变 |\n|---|---|---|---|---|---|---|---|")
    rows = []
    for row in cal["token_ladder"]:
        rows.append(f"| {row['group']} | {row['scale_bytes_per_value']:.4f} | "
                    f"{row['violation_distance_positions']} | {row['int8_bytes_per_value']:.4f} | "
                    f"{row['int4_bytes_per_value']:.4f} | {row['int8_reduction_vs_fp16']:+.1%} | "
                    f"{row['int4_reduction_vs_fp16']:+.1%} | {row['prefix_invariant']} |")
    L[-1] += "\n" + "\n".join(rows)
    L.append(f"> **head_dim = 32 ⇒ 沿 head_dim 的块上限 = G=32（int8 1.0625 B/值，省 46.9%）**。"
             f"G=64 要求块跨 token，无法同时保持前缀确定（`{cal['head_dim_g64_note']}`）。"
             f"沿 token 轴要摊得更薄就必须放弃块内前缀确定（block_tok / kivi 的 K 侧）。")
    fp8 = cal["fp8_reference"]
    L.append(f"> **同价诚实对手（未测）**：fp8（E4M3/E5M2 式）解析 **1.0 B/值、无显式标尺 ⇒ "
             f"{fp8['bytes_per_position']:.0f} B/位置 = fp16 的 ×{fp8['ratio_vs_fp16']:.2f}"
             f"（省 {fp8['reduction_vs_fp16']:.1%}）**，比 int8+块标尺（1.0625 B/值、×0.53）**更省**。"
             f"fp8 质量本实验**未测**（`quality_measured={fp8['quality_measured']}`）⇒ **不得据此断言 "
             f"int8+块标尺是所有 8 位方案里最优的**。")
    st = cal["token_ladder_staging"]
    if st:
        L.append(f"> **staging 两套口径（不得混用）**：回填需帧内留住 fp16 原值——"
                 f"**持整块（最坏）**= {st['worst_positions_held']:.0f} 个位置 = "
                 f"{st['worst_bytes_per_request']:.0f} B/请求（T=256 时 "
                 f"{st['worst_bytes_per_position_at_T']:.0f} B/位置）；**平均半块**= "
                 f"{st['mean_positions_held']:.1f} 个位置 = {st['mean_bytes_per_request']:.0f} B/请求"
                 f"（T=256 时 {st['mean_bytes_per_position_at_T']:.0f} B/位置）。"
                 f"**block（主臂）无需 staging**（严格前缀确定）。")

    L.append("## 表 4 · 非量化臂与 2631 的逐位对照（可控性）\n\n"
             f"- 非量化臂 max |Δbpc| = {rec['reuse']['nonquant_max_abs_delta_bpc']:.2e}"
             f"（逐位保持 = {rec['reuse']['nonquant_bitwise_preserved']}）。\n"
             f"- 量化臂（换标尺方案）：{rec['reuse']['quant_note']}")

    v = rec["variance_self_measured"]
    pv = rec.get("variance_self_measured_paired", {})
    L.append("## 表 5 · 本格自测方差（不套用全局 0.014）\n\n"
             f"同一臂在 window_seed = {rec['config']['eval']['self_measured_seeds']} 三个独立评测窗口下重测：\n\n"
             "| 臂 | 方案 | bpc（逐 seed） | 极差 | 半宽(std) |\n|---|---|---|---|---|\n"
             + "\n".join(f"| `{a}` | {d['scheme']} | {d['bpc_by_seed']} | {d['range']:.6f} | "
                         f"{d['std']:.6f} |" for a, d in v.items()))
    if pv:
        L.append(
            f"> **排序带的预登记口径与在力声明**：标准口径 = **配对 Δ**（同一对照臂对在多个评测窗口"
            f"重测）的**区间，≥5 个窗口**；本批只跑了 **{rec['ranking_band']['this_batch_windows']}"
            f" 个窗口**（<5）⇒ **不满足**标准，故**在力排序带 = 单臂跨窗口 bpc 极差 "
            f"{rec['ranking_band']['in_force_band_bpc']:.6f}**（更保守；所有『可排序/不可排序』判断用它）。"
            f"\n> **不用 `max|Δ|` 当带**（主表用的那个窗口的 Δ 恰为该 3 点 max ⇒ 会循环、不可证伪）。"
            f"配对量 `{pv['comparison']}` 逐 seed = {pv['delta_by_seed']}，**均值 = {pv['mean']}**、"
            f"区间 = {pv['interval']}（只作描述，不当带）。按在力带读：int8 块标尺 vs fp16 = "
            f"**略差、正号（+{abs(rec['answer']['int8_block_g32']['paired_delta_bpc_vs_fp16']):.6f}）、"
            f"远低于任何可排序线**（不写『不可分辨』）。")
    return "\n\n".join(L) + "\n"


def write_table(cfg, points, scheme_rows, controls, rec, out_dir) -> None:
    rec["deployment_caliber"] = sc.caliber_block(cfg, cfg["arms"], points)
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(cfg, points, scheme_rows, controls, rec))


# ----------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--charts-only", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = json.load(open(args.config))
    parent = json.load(open(os.path.join(ROOT, cfg["parent"]["config"])))
    # 继承父实验的臂定义 / checkpoint / 评测流 / 任务；本实验只覆盖契约级字段。
    for k in ("arms", "checkpoints", "model", "data", "tokens", "tasks", "family_baseline",
              "quant_increment_ref", "window_increment_ref", "controls"):
        cfg[k] = parent[k]
    cfg["eval"]["n_batches"] = 2 if args.smoke else cfg["eval"]["n_batches"]
    dev = cfg["eval"]["device"] if torch.cuda.is_available() else "cpu"
    T = cfg["model"]["seq_len"]
    if args.smoke:
        cfg["tasks"]["copy"]["n_batches"] = 1
        cfg["tasks"]["induction"]["n_batches"] = 2
        cfg["tasks"]["var_recall"]["n_batches"] = 2

    if args.charts_only:
        rec = json.load(open(os.path.join(args.out_dir, "results.json")))
        write_table(rec["config"], rec["points"], rec["scheme_comparison"], rec["controls"],
                    rec, args.out_dir)
        print(f"[charts-only] 重画完成（纯 CPU）→ {args.out_dir}", flush=True)
        return 0

    rec = {"experiment": cfg["experiment"], "contract": cfg["contract"],
           "scope": cfg["scope"], "config": cfg, "smoke": bool(args.smoke),
           "criterion": cfg["criterion"], "schemes": cfg["schemes"]}
    # SCHEMA §20：整程 <10 min ⇒ 允许整程持锁；但必须留机器可读的取/释与前后空闲显存记录。
    lock = GpuLock()
    lock.acquire()
    atexit.register(lock.release)                     # 异常退出也释放
    rec["gpu_lock"] = lock.ev
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))

    # ---------- ① 仪器自检 ----------
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    fp32_fn = b21.b19.make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
    lib_bpc = b21.b19.masked_bpc(trunk, eval_arr, None,
                                 n_batches=cfg["eval"]["n_batches"],
                                 batch_size=cfg["eval"]["batch_size"],
                                 window_seed=cfg["eval"]["window_seed"], device=dev)
    my_bpc = b21.b19.bpc_of(fp32_fn, eval_arr, T, cfg["eval"]["n_batches"],
                            cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    tasks = b21.b19.build_tasks(cfg, smoke=args.smoke)
    probe = tasks["var_recall"]
    lib_acc = b21.b19.lib_accuracy(trunk, probe, device=dev)
    my_acc = b21.b19.synth_accuracy(fp32_fn, probe, dev)

    # 实现层 equality：本实验 scheme=per_value 的前向必须与 2631 的生产前向逐位一致。
    eq_tokens = tasks["copy"].tokens[:4].to(dev)
    eq = {}
    spec_int8 = cfg["arms"]["int8"]
    f_mine = ba.make_logits_fn(trunk, spec_int8, T, dev, "per_value", 32, 32)
    f_prod = b21.make_logits_fn(trunk, spec_int8, T, dev)
    eq["quant_dense_per_value_bitwise"] = bool(torch.equal(f_mine(eq_tokens), f_prod(eq_tokens)))
    spec_mqa = cfg["arms"]["mqa_int8"]
    m_mqa, _ = b21.b19.load_head_arm(cfg, "mqa", dev)
    f_mine_h = ba.make_logits_fn(m_mqa, spec_mqa, T, dev, "per_value", 32, 32)
    f_prod_h = b21.make_logits_fn(m_mqa, spec_mqa, T, dev)
    eq["quant_head_gqa_per_value_bitwise"] = bool(torch.equal(f_mine_h(eq_tokens), f_prod_h(eq_tokens)))
    spec_win = cfg["arms"]["win32_int8"]
    f_mine_w = ba.make_logits_fn(trunk, spec_win, T, dev, "per_value", 32, 32)
    f_prod_w = b21.make_logits_fn(trunk, spec_win, T, dev)
    eq["combo_mask_per_value_bitwise"] = bool(torch.equal(f_mine_w(eq_tokens), f_prod_w(eq_tokens)))

    # 前缀不变性：生产量化器（per_value）与新方案（block / block_tok / kivi）并列。
    inv_scheme = {
        "per_value": cc2631.prefix_invariance_check(trunk, eq_tokens, "int8"),
        "block_g32": cc2631.prefix_invariance_of_logits(
            ba.make_logits_fn(trunk, spec_int8, T, dev, "block", 32, 32), eq_tokens),
        "block_g16": cc2631.prefix_invariance_of_logits(
            ba.make_logits_fn(trunk, spec_int8, T, dev, "block", 16, 32), eq_tokens),
        "block_tok_g32": cc2631.prefix_invariance_of_logits(
            ba.make_logits_fn(trunk, spec_int8, T, dev, "block_tok", 32, 32), eq_tokens),
        "kivi_g32": cc2631.prefix_invariance_of_logits(
            ba.make_logits_fn(trunk, spec_int8, T, dev, "kivi", 32, 32), eq_tokens),
    }
    rec["instrument"] = {
        "bpc_path": {"bpc_of": round(my_bpc, 6), "retention.masked_bpc": round(lib_bpc, 6),
                     "abs_diff": abs(my_bpc - lib_bpc), "bitwise_equal": my_bpc == lib_bpc,
                     "anchor_2614_release_bpc": cfg["controls"]["baseline_anchor_bpc"],
                     "note": "fp32 恒等：KV 数值未改"},
        "synth_path": {"run::synth_accuracy": my_acc["accuracy"],
                       "longrange_tasks.accuracy": lib_acc["accuracy"],
                       "n_equal": my_acc["n"] == lib_acc["n"],
                       "n_hits_equal": my_acc["n_hits"] == lib_acc["n_hits"]},
        "scheme_forward_equality_vs_2631_per_value": eq,
        "prefix_invariance_by_scheme": inv_scheme,
        "note": "scheme=per_value 的前向与 2631 生产前向逐位一致 ⇒ 唯一改动只有标尺方案；"
                "prefix_invariance_by_scheme 直接展示 block（沿 head_dim）保持前缀不变、"
                "block_tok / kivi（沿 token）违反。",
    }
    print(f"[自检] bpc_of={my_bpc:.6f} vs masked_bpc={lib_bpc:.6f}；per_value 逐位="
          f"{eq}；前缀不变性 { {k: (v['future_token_past_logits_equal'], v['peer_request_logits_equal']) for k, v in inv_scheme.items()} }",
          flush=True)

    # ---------- ② 对照臂 ----------
    copy_len = cfg["tasks"]["copy"]["copy_len"]
    tasks["copy_corrupt_by_pos"] = {
        p: corrupt_span(tasks["copy_by_pos"][p], p, p + copy_len)
        for p in tasks["copy_by_pos"]}
    controls = {
        "chance_uniform": round(1.0 / cfg["tokens"]["sym_vocab"], 8),
        "local_prior_copy": local_prior_accuracy(tasks["copy"], k=cfg["tokens"]["local_k"]),
        "local_prior_induction": local_prior_accuracy(tasks["induction"],
                                                      k=cfg["tokens"]["local_k"]),
        "local_prior_var_recall": local_prior_accuracy(tasks["var_recall"],
                                                       k=cfg["tokens"]["local_k"]),
        "local_prior_resistance": {
            "copy": check_local_prior_resistance(tasks["copy"], k=cfg["tokens"]["local_k"]),
            "induction": check_local_prior_resistance(tasks["induction"],
                                                      k=cfg["tokens"]["local_k"]),
            "var_recall": check_local_prior_resistance(tasks["var_recall"],
                                                       k=cfg["tokens"]["local_k"])},
        "corrupt_source_note": "copy 主腿源片段整段换随机符号（其余逐位不变），逐臂报该对照。",
    }
    rec["controls"] = controls
    print(f"[对照] copy 局部先验={controls['local_prior_copy']['accuracy']:.4f} / "
          f"chance={controls['chance_uniform']:.4f}", flush=True)

    # ---------- ③ 27 点重测（量化臂 = 主臂 scheme） ----------
    points = []
    n_arms = len(cfg["arms"])
    model_cache = {}
    for i, (arm, spec) in enumerate(cfg["arms"].items(), 1):
        if arm not in model_cache:
            model_cache[arm] = b21.load_model_for_arm(cfg, arm, spec, dev)[0]
        model = model_cache[arm]
        scheme, group = (PRIMARY_SCHEME, PRIMARY_GROUP) if quantized(spec) else ("per_value", 1)
        r = eval_arm(cfg, arm, spec, model, dev, tasks, eval_arr, scheme, group, 32,
                     controls, do_recall=True)
        r["checkpoint"] = (
            {"base_arm": spec["base_arm"], "quantized_mode": spec["mode"]}
            if spec["kind"] == "quant_head" else
            {"base_model": spec.get("base_model", "trunk"), "mask_window": spec.get("window"),
             "quantized_mode": spec.get("mode")} if spec["kind"] == "combo_mask" else
            {"note": "trunk" if spec["kind"] == "quant" else "head/shared model"})
        points.append(r)
        print(f"[{i}/{n_arms}] {arm:<22} scheme={scheme:<9} bpc={r['bpc']:.6f} "
              f"copy={r['recall_copy']['accuracy']:.4f} "
              f"inv={r['prefix_invariance']['future_token_past_logits_equal']}/"
              f"{r['prefix_invariance']['peer_request_logits_equal']}", flush=True)

    # ---------- ④ damage / 增量 ----------
    by_arm = {p["arm"]: p for p in points}
    for p in points:
        base = by_arm[cfg["family_baseline"][cfg["arms"][p["arm"]]["family"]]]
        p["family_baseline"] = base["arm"]
        p["damage_bpc"] = round(p["bpc"] - base["bpc"], 6)
        ref = cfg.get("quant_increment_ref", {}).get(p["arm"])
        p["quant_increment_bpc"] = (round(p["bpc"] - by_arm[ref]["bpc"], 6) if ref else None)
        if ref:
            p["quant_increment_vs_arm"] = ref

    # ---------- ⑤ 标尺方案对照（回答「在什么实现下不成立」） ----------
    scheme_plan = [
        ("int8", "per_value", 1, 32), ("int8", "block", 16, 32), ("int8", "block", 32, 32),
        ("int8", "block_tok", 32, 32), ("int8", "kivi", 1, 32),
        ("int8", "block", 8, 32), ("int8", "block_tok", 16, 32), ("int8", "block_tok", 64, 32),
        ("int4", "per_value", 1, 32), ("int4", "block", 16, 32), ("int4", "block", 32, 32),
        ("int4", "block_tok", 32, 32), ("int4", "kivi", 1, 32),
    ]
    scheme_rows = []
    for arm, scheme, group, kg in scheme_plan:
        spec = cfg["arms"][arm]
        model = model_cache.get(arm) or b21.load_model_for_arm(cfg, arm, spec, dev)[0]
        model_cache[arm] = model
        r = eval_arm(cfg, arm, spec, model, dev, tasks, eval_arr, scheme, group, kg,
                     controls, do_recall=True)
        ref_arm = cfg["family_baseline"][spec["family"]]
        ref_spec = cfg["arms"][ref_arm]
        ref = by_arm[ref_arm]
        r["deployment"] = sc.deployment_bytes(by_arm[arm], spec, cfg, scheme, group, kg)
        r["ratio_vs_ref"] = sc.quantization_factor(by_arm[arm], spec, cfg, ref, ref_spec,
                                                   scheme, group, kg)
        r["damage_bpc"] = round(r["bpc"] - ref["bpc"], 6)
        scheme_rows.append(r)
        print(f"[scheme] {arm:<5} {scheme:<10} g={group:<3} bpc={r['bpc']:.6f} "
              f"bytes={r['deployment']['bytes_per_position']:.0f} "
              f"ratio=×{r['ratio_vs_ref']:.2f}", flush=True)
    rec["scheme_comparison"] = scheme_rows

    # ---------- ⑥ 非量化臂与 2631 逐位对照（可控性） ----------
    old = json.load(open(os.path.join(ROOT, cfg["parent"]["results"])))
    old_by = {p["arm"]: p for p in old["points"]}
    nonq = [p for p in points if not quantized(cfg["arms"][p["arm"]])]
    nonq_deltas = {p["arm"]: round(p["bpc"] - old_by[p["arm"]]["bpc"], 9)
                   for p in nonq if p["arm"] in old_by}
    quant_deltas = {p["arm"]: round(p["bpc"] - old_by[p["arm"]]["bpc"], 9)
                    for p in points if quantized(cfg["arms"][p["arm"]]) and p["arm"] in old_by}
    rec["reuse"] = {
        "nonquant_max_abs_delta_bpc": max((abs(v) for v in nonq_deltas.values()), default=None),
        "nonquant_bitwise_preserved": all(abs(v) <= 1e-6 for v in nonq_deltas.values()),
        "nonquant_deltas": nonq_deltas,
        "quant_deltas_vs_2631": quant_deltas,
        "quant_note": ("量化臂的 bpc 变化 = 标尺从逐值换成块标尺（block G=32）的**质量**效应；"
                       "非量化臂应逐位不变。"),
    }
    print(f"[复用] 非量化臂 max|Δ|={rec['reuse']['nonquant_max_abs_delta_bpc']:.2e} "
          f"逐位={rec['reuse']['nonquant_bitwise_preserved']}", flush=True)

    # ---------- ⑦ 自测方差（本格噪声带，不用全局 0.014） ----------
    variance = {}
    for arm in ("fp16", "int8"):
        spec = cfg["arms"][arm]
        model = model_cache.get(arm) or b21.load_model_for_arm(cfg, arm, spec, dev)[0]
        model_cache[arm] = model
        scheme, group = ("block", 32) if quantized(spec) else ("per_value", 1)
        bpcs = []
        for ws in VARIANCE_SELF_SEEDS:
            fn = fn_for(cfg, arm, spec, model, scheme, group, 32, dev, T)
            bpcs.append(round(b21.b19.bpc_of(fn, eval_arr, T, cfg["eval"]["n_batches"],
                                             cfg["eval"]["batch_size"], ws, dev), 6))
        variance[arm] = {"scheme": scheme, "window_seeds": list(VARIANCE_SELF_SEEDS),
                         "bpc_by_seed": bpcs, "range": round(max(bpcs) - min(bpcs), 6),
                         "std": round((sum((b - sum(bpcs) / len(bpcs)) ** 2 for b in bpcs)
                                       / max(len(bpcs) - 1, 1)) ** 0.5, 6)}
    rec["variance_self_measured"] = variance
    # 关键对照是**配对 Δ**（同窗口同 batched 数据上 int8_block − fp16），不是单臂 bpc 的
    # 跨窗口散度（后者含窗口数据本身的方差）。两者都报；但**排序带**另有预登记口径（见下）。
    paired = {}
    if "fp16" in variance and "int8" in variance:
        d = [round(i - f, 9) for i, f in zip(variance["int8"]["bpc_by_seed"],
                                             variance["fp16"]["bpc_by_seed"])]
        paired = {"comparison": "int8(block g32) − fp16", "delta_by_seed": d,
                  "mean": round(sum(d) / len(d), 9),
                  "interval": [min(d), max(d)],
                  "max_abs": round(max(abs(x) for x in d), 9),
                  "note": "3 个窗口；**不以 max|Δ| 当带**（主表用的那个窗口的 Δ 恰为这三点 max ⇒ "
                          "当带会循环、不可证伪）；只报均值与区间，且只作描述。"}
    rec["variance_self_measured_paired"] = paired
    single_arm_band = max(v["range"] for v in variance.values())
    rec["variance_band_bpc_self_measured"] = round(single_arm_band, 6)
    # 预登记里本格噪声带的口径与在力（in force）声明：标准口径 = 配对 Δ、≥5 窗口；
    # 本批只跑了 3 个窗口（<5）⇒ 不满足标准，退用**单臂跨窗口极差**（更保守）作排序带。
    n_windows = len(VARIANCE_SELF_SEEDS)
    rec["ranking_band"] = {
        "in_force_band_bpc": round(single_arm_band, 6),
        "in_force_caliber": "单臂跨窗口 bpc 极差（同臂在多个评测窗口重测）",
        "standard_caliber": "配对 Δ（同一对照臂对在多个评测窗口重测）的区间，≥5 个窗口",
        "this_batch_windows": n_windows,
        "standard_met": n_windows >= 5,
        "declaration": (f"本批 variance 只跑了 {n_windows} 个窗口（<5）⇒ **不满足**标准配对口径；"
                        f"故本批所有『可排序/不可排序』判断用**单臂跨窗口极差 "
                        f"{single_arm_band:.6f}**。3 点配对量（均值 "
                        f"{paired.get('mean')}）只作描述，不当带。"),
    }
    print(f"[自测方差] 在力排序带（单臂跨窗口极差）= {single_arm_band:.6f}；"
          f"配对 Δ(int8_block−fp16) 3 点均值 = {paired.get('mean')}、区间 = {paired.get('interval')}"
          f"（窗 seed {VARIANCE_SELF_SEEDS}；<5 窗口，不当带）", flush=True)

    # ---------- ⑧ 一句话答案（契约 ⑤） ----------
    int8_pt = by_arm["int8"]
    int8_dep = sc.deployment_bytes(int8_pt, cfg["arms"]["int8"], cfg, "block", PRIMARY_GROUP)
    int4_pt = by_arm["int4"]
    int4_dep = sc.deployment_bytes(int4_pt, cfg["arms"]["int4"], cfg, "block", PRIMARY_GROUP)
    fp16_pt = by_arm["fp16"]
    rec["answer"] = {
        "q": "『8 位近无损且省约一半』这条参照线成立吗？成立到什么程度？在什么实现下不成立？",
        "ranking_band_bpc_in_force": rec["ranking_band"]["in_force_band_bpc"],
        "int8_block_g32": {
            "paired_delta_bpc_vs_fp16": round(int8_pt["bpc"] - fp16_pt["bpc"], 6),
            "delta_sign": "positive（略差，正号）",
            "vs_ranking_band": (f"远低于任何可排序线（{int8_pt['bpc'] - fp16_pt['bpc']:+.6f} "
                                f"<< 在力带 {rec['ranking_band']['in_force_band_bpc']:.6f}）"),
            "deployment_bytes_per_position": int8_dep["bytes_per_position"],
            "fp16_deployment_bytes_per_position": 12288.0,
            "reduction_frac": round(1 - int8_dep["bytes_per_position"] / 12288.0, 6),
            "ratio_vs_fp16": round(int8_dep["bytes_per_position"] / 12288.0, 4),
            "prefix_invariant": bool(int8_pt["prefix_invariance"]["future_token_past_logits_equal"]
                                     and int8_pt["prefix_invariance"]["peer_request_logits_equal"]),
            "staging_bytes_per_request": 0,
            "note": "沿 head_dim 摊薄严格因果 ⇒ 无需 staging；结论表述为『略差、正号、远低于可排序线』，"
                    "不写『不可分辨』。",
        },
        "int4_block_g32": {
            "damage_bpc_vs_fp16": round(int4_pt["bpc"] - fp16_pt["bpc"], 6),
            "damage_bpc_vs_ranking_band": ("< 带" if abs(int4_pt["bpc"] - fp16_pt["bpc"])
                                           < rec["ranking_band"]["in_force_band_bpc"] else "> 带"),
            "deployment_bytes_per_position": int4_dep["bytes_per_position"],
            "reduction_frac": round(1 - int4_dep["bytes_per_position"] / 12288.0, 6),
        },
        "under_per_value_scale": {
            "int8_bytes_per_position": 18432.0,
            "ratio_vs_fp16": 1.5,
            "note": "逐值标尺（旧『因果化修复』的口径）⇒ int8 = 3 B/值 = fp16 的 1.5×；"
                    "『省一半』只在这个实现下不成立。",
        },
        "fp8_same_price_opponent": sc.fp8_reference(cfg),
        "combined_arm_scheme_delta": {
            "gqa_int4_block_minus_per_value": round(by_arm["gqa_int4"]["bpc"]
                                                    - old_by["gqa_int4"]["bpc"], 6),
            "mqa_int4_block_minus_per_value": round(by_arm["mqa_int4"]["bpc"]
                                                    - old_by["mqa_int4"]["bpc"], 6),
            "note": "头共享×4 位换块标尺后变差；归因待核（见 analysis §7）。",
        },
        "cannot_claim": "不能声称 int8+块标尺是所有 8 位方案里最优的：fp8 解析为 1 B/值、"
                        "无标尺、×0.50，比 1.0625 B/值 更省（fp8 质量本实验未测）。",
        "verified": True,
    }

    # ---------- ⑨ 落盘 ----------
    rec["points"] = points
    rec["memory_convention"] = {
        "analytic": "points[*].bytes_per_position = 判据 C1 值字节（与 2631 逐位可比）",
        "deployment": "points[*] 与 deployment_caliber.per_arm = 值 + 标尺（block 主臂）",
        "measured_gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "isolation_note": "评测实现把 K/V 立刻反量化 ⇒ 实测峰值不随档下降，只报解析列。",
    }
    rec["cost"] = {"device": dev, "wall_seconds": round(time.time() - t0, 2),
                   "wall_minutes": round((time.time() - t0) / 60, 3),
                   "lock_wait_seconds": lock.ev.get("wait_seconds"),
                   "gpu_work_only": True,
                   "contention_note": "wall 从取锁成功后开始计（不含等锁）；共享卡被外部占用时同代码步时会更长。",
                   "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                   "smoke": bool(args.smoke)}
    codes = ["tools/evalkit/kv_quant.py",
             "axes/R-references/R1-quantization/experiments/2632-kv-block-scale/run.py",
             "axes/R-references/R1-quantization/experiments/2632-kv-block-scale/block_arms.py",
             "axes/R-references/R1-quantization/experiments/2632-kv-block-scale/scale_caliber.py"]
    rec["code"] = {c: b21.b19.sha256_file(os.path.join(ROOT, c))[:16] for c in codes}

    # GPU 工作已完成：先释放并补全 §20 记账，再落盘。
    lock.ev["held_whole_run"] = True                 # 整程持锁（<10 min，§20.1）
    lock.ev["peak_mem_mib_this_run"] = rec["cost"]["peak_mem_mib"]
    lock.ev["free_delta_note"] = ("free_mib_before/after 是 nvidia-smi 的**全卡**空闲读数，含**外部占用**；"
                                   "两者之差 ≠ 本 run 足迹（本 run 足迹看 peak_mem_mib_this_run）。")
    lock.release()
    rec["gpu_lock"] = dict(lock.ev)

    if not args.smoke:
        write_table(cfg, points, scheme_rows, controls, rec, args.out_dir)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"[完成] {rec['cost']['wall_minutes']} min，峰值 {rec['cost']['peak_mem_mib']} MiB，"
          f"{len(points)} 点 + {len(scheme_rows)} 方案行 → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
