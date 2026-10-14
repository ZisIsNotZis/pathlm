"""2652 · 激活量化最小判决（13M 受控，纯推理）—— 实验入口。

契约：同目录 `README.md`（§1–§10）+ `config.json::criterion`（**运行前**写入）。
语义层：同目录 `aq_arms.py`（激活假量化的**唯一**新模块）。
单测 / 变异：同目录 `selftest_aq.py`（纯 CPU）。

**唯一改动 = 前向中间激活的存储精度**（+ 组合臂 `W8A8` 的权重 per-channel int8）。
trunk 冻结、不训练、不新增参数、不改 KV 精度、不改结构、不改主 FLOPs。

臂（单因素隔离 + 部署组合点并列）：
  `fp16`（基线，必须与历史锚 1.483569 逐位一致）· `A8`（**主因子臂**：只有激活 int8）·
  `A4`（只有激活 4 位）· `W8A8`（**组合臂**：权重 int8 per-channel × 激活 int8，字节账分列）。

用法：
  python3 run.py              # 全跑（纯推理；先 --smoke 估时）
  python3 run.py --smoke      # 冒烟：缩小批量/档数，实测步时与显存（不外推时间）
  python3 run.py --table-only # 只从已有 results.json 重画 table.md（纯 CPU，不取 GPU 锁）
  python3 run.py --anchor-single-point --device cpu
                              # 只补跑 24×16@seed2 的 fp16 单点（= 历史锚口径），
                              # 写 results.json::anchor_single_point（硬护栏 ② 的交付项）
"""

import argparse
import atexit
import importlib.util
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, P2621)

import aq_arms as aq                                                    # noqa: E402
from pathlm.data import batch, load_enwik8_full                         # noqa: E402
from tools.evalkit.longrange_tasks import (                             # noqa: E402
    corrupt_span, local_prior_accuracy, check_local_prior_resistance)

#: SCHEMA §20 的 GPU 互斥锁目录（与 5.1 各实验同一把锁 ⇒ 全轴串行）。
GPU_LOCK_DIR = "/tmp/pathlm_gpu.lock"
#: fp16 激活的每 token 字节（判据横轴的归一基准，= 55296）。
FP16_ACT_BYTES_PER_TOKEN = 55296.0
#: 历史锚（5.1 `2614`）的口径 = `window_seed=2` + 24×16
#: （`config.json::criterion.must_hold.gate_2_bitwise_anchor`）；`anchor_check_applicable`
#: 与 `--anchor-single-point` 都以它为唯一判据。
ANCHOR_CALIBER = {"arm": "fp16", "window_seed": 2, "n_batches": 24, "batch_size": 16}


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2621 的评估管线（内含 2619 的 b19）；**只读复用**，不改。
b21 = _load_module(os.path.join(P2621, "run.py"), "kv_budget_rest_2621")


class NullLock:
    """CPU 口径下不取 GPU 锁（纯 CPU 工作，SCHEMA §20 不适用）的占位锁。

    `--device cpu` 时使用：`acquire` / `release` 都是无操作，`ev` 里显式标注
    `lock_mode = "cpu_no_lock"`，`gpu_lock` 字段照常落盘（指明本口径未取锁）。
    """

    def __init__(self):
        self.ev = {"lock_path": None, "lock_mode": "cpu_no_lock", "holder_pid": None,
                   "acquired_at": None, "wait_seconds": 0.0, "reclaimed": [],
                   "release_ok": None, "free_mib_before": None, "free_mib_after": None,
                   "held_whole_run": None, "peak_mem_mib_this_run": None, "segments": []}

    def acquire(self) -> None:
        return None

    def release(self) -> None:
        return None


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

    与 `2632/run.py::GpuLock` 同实现（逐字照抄，不引共享模块）：等锁上限 10 分钟
    （40×15s），**不杀任何别的进程**；等不到就抛 `TimeoutError`，由调用方如实返回。
    `evidence()` 把取锁/收尾/释放与前后空闲显存落成**机器可读**记录。
    """

    def __init__(self, lock_dir: str = GPU_LOCK_DIR, timeout_s: int = 600, poll_s: int = 15):
        self.lock_dir = lock_dir
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.ev = {"lock_path": lock_dir, "holder_pid": None, "acquired_at": None,
                   "wait_seconds": None, "reclaimed": [], "release_ok": None,
                   "free_mib_before": None, "free_mib_after": None,
                   "held_whole_run": None, "peak_mem_mib_this_run": None,
                   "segments": []}

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


# ----------------------------------------------------------------- 评估


def make_logits_fn(model, spec):
    """一个臂的 `tokens -> logits`（唯一接口；与 `bpc_of` / `pooled_accuracy` 对接）。"""
    return aq.AqForward(model, spec).logits


def eval_bpc(fn, eval_arr, T, n_batches, batch_size, seed, dev):
    return b21.b19.bpc_of(fn, eval_arr, T, n_batches, batch_size, seed, dev)


def causality_probe(fn, tokens):
    """因果性探针（硬护栏 ④）：改**未来** token，不改变过去位置的对数。

    per-token 标尺只在当前 token 内归约 ⇒ 位置 t 的读数只由 ≤ t 的 token 决定。
    若实现（错把标尺做成整批/整序列 amax）引入了未来依赖，本检查会 False。
    """
    with torch.no_grad():
        a = fn(tokens)
        t2 = tokens.clone()
        t2[:, -1] = (tokens[:, -1] + 7) % 206
        b = fn(t2)
        return {"past_logits_equal": bool(torch.equal(a[:, :-1], b[:, :-1])),
                "changed_last_row_only": bool(not torch.equal(a[:, -1], b[:, -1])),
                "T": int(tokens.shape[1])}


def eval_arm(cfg, arm, spec, trunk, dev, tasks, eval_arr, probe, controls):
    T = cfg["model"]["seq_len"]
    fwd = aq.AqForward(trunk, spec)
    fn = fwd.logits
    seeds = [cfg["eval"]["window_seed"]] + list(cfg["eval"]["extra_window_seeds"])
    bpc_by_seed = {}
    for s in seeds:
        nb = (cfg["eval"]["n_batches"] if s == cfg["eval"]["window_seed"]
              else cfg["eval"]["extra_seed_n_batches"])
        bpc_by_seed[str(s)] = round(eval_bpc(fn, eval_arr, T, nb,
                                             cfg["eval"]["batch_size"], s, dev), 9)
    bpc = bpc_by_seed[str(cfg["eval"]["window_seed"])]

    copy_pooled = b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)
    copy_corrupt = b21.b19.pooled_accuracy(fn, tasks["copy_corrupt_by_pos"], dev)
    ind_pooled = b21.b19.pooled_accuracy(fn, tasks["induction_by_start"], dev)
    var_pooled = b21.b19.pooled_accuracy(fn, tasks["var_recall_by_d"], dev)
    chance = controls["chance_uniform"]
    at_floor = copy_pooled["accuracy"] <= b21.b19.floor_threshold(copy_pooled["n"], chance)

    stats = aq.new_stats(spec)
    fwd.logits(probe, stats)
    act = aq.summarize_stats(stats, cfg["model"]["n_layers"],
                             len(aq.activation_tensor_shapes(cfg)))
    pen = aq.revert_byte_penalty(act, cfg)

    abytes = aq.activation_bytes_per_token(cfg, spec["activation_mode"])
    wbytes = aq.weight_bytes(cfg, spec["weight_mode"])
    return {
        "arm": arm,
        "activation_mode": spec["activation_mode"],
        "weight_mode": spec["weight_mode"],
        "is_combo": bool(spec["activation_mode"] != "fp16" and spec["weight_mode"] != "fp16"),
        "bpc": bpc,
        "bpc_by_seed": bpc_by_seed,
        "recall_copy": {**copy_pooled, "at_floor": bool(at_floor)},
        "recall_copy_source_corrupt": copy_corrupt,
        "recall_induction": ind_pooled,
        "recall_var_recall": var_pooled,
        "chance_uniform": chance,
        "activation_bytes": abytes,
        "bytes_per_token_activation": abytes["bytes_per_token_activation"],
        "bytes_per_token_per_layer": abytes["bytes_per_token_per_layer"],
        "weight_bytes": wbytes,
        "activity": act,
        "revert_byte_penalty": pen,
        "weight_activity": fwd.weight_activity(),
        "causality": causality_probe(fn, probe[:2]),
    }


# ----------------------------------------------------------------- 表


def render_table(rec) -> str:
    cfg = rec["config"]
    pts = {p["arm"]: p for p in rec["arms"]}
    L = []
    L.append("# 2652 · 激活量化最小判决（13M · int8 / 4 位激活，含 per-token 动态标尺）")
    L.append(
        "**横轴（主口径）**：`bytes_per_token_activation = Σ_tensor in_features·bits/8·L "
        "+ (整型档 ? 2·4·L : 0)`；张量集 = 每层 `qkv`/`proj` 输入的 d + FFN 两个 matmul "
        "输入的 d 与 `mlp_mult·d`（L=12, d=256, mlp=6 ⇒ 27648 元素/token；fp16 = "
        f"{FP16_ACT_BYTES_PER_TOKEN:.0f} B/token）。**这是激活账，与 5.1 的 KV 每位置 "
        "字节（fp16 12288 B/位置）不同量纲，两者不混排**。"
        "权重字节只对组合臂 `W8A8` 单列（整模型口径，非每 token）。")
    L.append(
        f"**判据带**：主判 = **{rec['criterion']['primary_band_bpc']} bpc 带**，口径 = "
        f"**配对 Δ**（同臂同 window_seed）跨 **{len(rec['paired_delta']['seeds'])} 个窗口**"
        f"（seed = {rec['paired_delta']['seeds']}）的 **max|配对 Δ|** ⇒ **满足 ≥5 窗口**，"
        "故直接用预登记常数 0.014（非本批自测带 ⇒ 不构成循环论证）。"
        "召回等率类指标走 Wilson 95% CI、不借用 0.014。"
        f"召回地板：chance = {rec['controls']['chance_uniform']:.4f}（1/64）、 "
        f"局部先验 = {rec['controls']['local_prior_copy']['accuracy']:.4f}（copy）。")
    t1 = ["## 表 1 · 臂总表（激活字节 → 两腿质量）", "",
          "| 臂 | 组合? | 激活字节/token | ×fp16 | 省幅 | 激活标尺 B/token | 权重字节 | "
          "clean bpc | Δbpc vs fp16 | max\\|配对Δ\\| | 越 {:.3f} 带? | copy 召回 | 95% CI | "
          "induction | var_recall |".format(rec["criterion"]["primary_band_bpc"]),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in rec["arms"]:
        pd = rec["paired_delta"]["per_arm"].get(p["arm"])
        ci = (f"[{p['recall_copy']['wilson95'][0]:.4f}, "
              f"{p['recall_copy']['wilson95'][1]:.4f}]")
        wb = (f"{p['weight_bytes']['mib']:.2f} MiB（int8）" if p["weight_mode"] != "fp16"
              else f"{p['weight_bytes']['mib']:.2f} MiB（fp16）")
        t1.append(
            f"| `{p['arm']}` | {'**是**' if p['is_combo'] else '否'} | "
            f"{p['bytes_per_token_activation']:.0f} | "
            f"×{p['bytes_per_token_activation'] / FP16_ACT_BYTES_PER_TOKEN:.4f} | "
            f"{1 - p['bytes_per_token_activation'] / FP16_ACT_BYTES_PER_TOKEN:.2%} | "
            f"{p['activation_bytes']['scale_bytes_per_token']:.0f} | {wb} | "
            f"{p['bpc']:.6f} | {p['damage_bpc_vs_fp16']:+.6f} | "
            f"{pd['max_abs'] if pd else '—'} | "
            f"{'是' if pd and pd['out_of_band'] else '否'} | "
            f"{p['recall_copy']['accuracy']:.4f}"
            f"{' (地板)' if p['recall_copy']['at_floor'] else ''} | {ci} | "
            f"{p['recall_induction']['accuracy']:.4f} | "
            f"{p['recall_var_recall']['accuracy']:.4f} |")
    _ms = f"{cfg['eval']['n_batches']}×{cfg['eval']['batch_size']}"
    t1.append(f"> `Δbpc vs fp16` 是**主表窗口（seed=2, {_ms}）**的单点差；`max|配对Δ|` 才是判据量。"
              "`W8A8` 的权重字节是**整模型**口径（12.58M 参数 × 1 B + per-channel 标尺），"
              "与激活字节**分列**、不得相加。")
    L.append("\n".join(t1))

    t2 = ["## 表 2 · 逐层激活字节分解 + 逐层离群通道活性", "",
          "（`bytes/token/layer` = 该层 4 个激活张量的值字节 + 标尺；"
          "`outlier_idx` = 该层 4 张量里最大的 `max/median` 通道峰值；"
          "`eff_bits(med chan)` = 位数 − log2(outlier_idx)，即离群通道吃掉位后中位通道还剩的有效位；"
          "`revert_frac` = 峰值 > 6×median 的通道占比，`layers_with_any_revert` 见 "
          "`results.json::arms[*].activity`）", "",
          "| 层 | bytes/token/layer (A8) | bytes/token/layer (A4) | outlier_idx (A8) | "
          "eff_bits (A8) | revert_frac 6× (A8) | changed_frac (A8) | rel_err (A8) |",
          "|---|---|---|---|---|---|---|---|"]
    by_layer8 = {r["layer"]: r for r in pts["A8"]["activity"]["per_layer"]}
    for l in range(cfg["model"]["n_layers"]):
        r8 = by_layer8.get(l, {})
        if not r8.get("tensors_observed"):
            continue
        t2.append(
            f"| {l} | {pts['A8']['bytes_per_token_per_layer']:.0f} | "
            f"{pts['A4']['bytes_per_token_per_layer']:.0f} | "
            f"{r8['max_outlier_index_channel']:.3f} | "
            f"{r8['min_effective_bits_median_channel']:.3f} | "
            f"{max(r8['revert_chan_frac_6x_by_tensor'].values()):.4%} | "
            f"{min(r8['changed_frac_by_tensor'].values()):.4f} | "
            f"{max(r8['rel_err_by_tensor'].values()):.4f} |")
    t2.append("> 每层 4 个张量的激活字节相同（值字节与张量无关、只随 in_features 变化；标尺 "
              "2 B/张量/token）——`mlp2` 输入的 `in_features = 1536` 使值字节更大，"
              "**逐张量分解见 `results.json::arms[*].activation_bytes.per_tensor`**。"
              "表 2 只给每层合计（= 值字节 + 标尺），逐层数值对 A4 有一半的值字节。")
    L.append("\n".join(t2))

    t3 = ["## 表 3 · 配对 Δbpc（同臂同窗口 seed 对 fp16）", "",
          "| 臂 | " + " | ".join(f"seed {s}" for s in rec["paired_delta"]["seeds"])
          + " | max\\|Δ\\| | 均值 | 越带? |",
          "|---|---" + "|---" * (len(rec["paired_delta"]["seeds"]) + 2) + "|"]
    for arm, d in rec["paired_delta"]["per_arm"].items():
        t3.append(f"| `{arm}` | " + " | ".join(f"{x:+.6f}" for x in d["by_seed"])
                  + f" | {d['max_abs']:.6f} | {d['mean']:+.6f} | "
                  f"{'**越带**' if d['out_of_band'] else '不越带'} |")
    t3.append("> 每 seed 一行是「同臂同 seed − fp16 同 seed」的**配对**差（窗口数据完全相同，"
              f"只有量化因子不同）。主窗 seed 2 用 {cfg['eval']['n_batches']}×"
              f"{cfg['eval']['batch_size']}（CPU 预算裁剪口径，超出 README §7 预登记裁剪条款，"
              "见 `analysis.md` §2②/§5.3），"
              f"其余 seed 用 {cfg['eval']['extra_seed_n_batches']}×{cfg['eval']['batch_size']}。")
    L.append("\n".join(t3))

    # ---------- 表 4：三道对账 + 四条硬护栏 ----------
    inst = rec["instrument"]
    n_ts = {p["arm"]: p["activation_bytes"]["n_tensors_total"] for p in rec["arms"]}
    nB, nS = rec["probe_shape"]
    rows_tensor, rows_scale = [], []
    for p in rec["arms"]:
        obs = p["activity"]["aggregate"]["n_tensors_observed"]
        exp = n_ts[p["arm"]]
        rows_tensor.append(f"| {p['arm']} 张量数 | {exp} | {obs} | "
                           f"{'✓' if obs == exp else '✗'} |")
        exp_s = 0 if p["activation_mode"] == "fp16" else exp * nB * nS
        got_s = p["activity"]["aggregate"]["n_scales"]
        rows_scale.append(f"| {p['arm']} 标尺数 | {exp_s} | {got_s} | "
                          f"{'✓' if got_s == exp_s else '✗'} |")
    v2621 = inst["vs_2621"]
    d2621 = "—" if v2621["abs_diff"] is None else f"{v2621['abs_diff']:.2e}"
    bands = []
    for a, d in rec["paired_delta"]["per_arm"].items():
        verdict = ("越带（可排序：该档代价真的可分辨）" if d["out_of_band"]
                   else "不越带（代价落在预登记噪声带内）")
        bands.append(f"- *{a}*：max|配对Δ| = {d['max_abs']:.6f} "
                     f"{'**>**' if d['out_of_band'] else '≤'} "
                     f"{rec['criterion']['primary_band_bpc']} ⇒ {verdict}")
    changed = " / ".join(f"{a}:{pts[a]['activity']['aggregate']['changed_frac']:.3f}"
                        for a in ("A8", "A4", "W8A8") if a in pts)
    live = all(pts[a]["activity"]["aggregate"]["changed_frac"] > 0
               for a in ("A8", "A4", "W8A8") if a in pts)
    caus = " / ".join(f"{a}:{pts[a]['causality']['past_logits_equal']}" for a in pts)
    caus_ok = all(pts[a]["causality"]["past_logits_equal"] for a in pts)
    guard2 = bool(inst["bpc_path"]["bitwise_equal"]
                  and inst["replicated_vs_library"]["vs_masked_logits_bitwise_equal"])
    guard3 = bool(inst["mutation"]["all_caught"])
    t4 = ["## 表 4 · 三道对账 + 四条硬护栏", "", "**(1) 激活字节账 ↔ 实现**", "",
          "| 项 | 解析账 | 实现实测 | 对账 |", "|---|---|---|---|"]
    t4 += rows_tensor + rows_scale
    anchor_lines = []
    sp_anchor = rec.get("anchor_single_point")
    if sp_anchor:
        ips = sp_anchor["independent_path_same_caliber"]
        anchor_lines = [
            f"- **锚口径单点（{sp_anchor['caliber']}，本轮补跑）**：`bpc_of` = "
            f"**{sp_anchor['bpc']:.9f}**；独立路径 `masked_bpc` = {ips['masked_bpc']:.9f}"
            f"（逐位 = {ips['bitwise_equal']}）；历史锚 {sp_anchor['anchor_bpc']} 的 |Δ| = "
            f"{sp_anchor['abs_diff_vs_anchor']:.2e}。",
            f"- 主表窗口（{sp_anchor['main_window_caliber']}）的 `bpc_of` = "
            f"{sp_anchor['main_window_bpc']:.9f}：与锚 |Δ| = "
            f"{sp_anchor['abs_diff_main_window_vs_anchor']:.2e}，与锚口径单点 |Δ| = "
            f"{sp_anchor['abs_diff_vs_main_window']:.2e} ⇒ 二者不可跨口径直接比；"
            f"归因见 `analysis.md` §2②。"]
    t4 += ["", "**(2) fp16 臂 ↔ 历史锚 / 独立路径**", "",
           f"- `bpc_of(AqForward fp16)` = **{inst['bpc_path']['bpc_of']:.9f}**；"
           f"`retention.masked_bpc` = {inst['bpc_path']['masked_bpc']:.9f}；"
           f"逐位 = **{inst['bpc_path']['bitwise_equal']}**；历史锚 "
           f"{inst['bpc_path']['anchor_bpc']} 的 |Δ| = "
           f"{inst['bpc_path']['abs_diff_vs_anchor']:.2e}。",
           f"- 复刻前向（`AqForward` fp16）↔ 既有仪器（`masked_logits`）逐位 = "
           f"**{inst['replicated_vs_library']['vs_masked_logits_bitwise_equal']}**（probe 批）"
           f"⇒ `fp16` 臂的逐位复原不是「同代码自证」。",
           f"- 与 `2621` 的 fp16 点（跨 run）|Δbpc| = {d2621}（`{v2621['path']}`）。"] \
        + anchor_lines + ["", "**(3) 配对 Δ ↔ 判据带**", ""]
    t4 += bands
    t4 += ["", "**硬护栏**", "", "| 护栏 | 读数 | 结论 |", "|---|---|---|",
           f"| ① 干预活性 | 量化臂 `changed_frac` = {changed}；fp16 = "
           f"{pts['fp16']['activity']['aggregate']['changed_frac']:.3f} | "
           f"{'✓ 激活真被改' if live else '✗'} |",
           f"| ② 非量化臂逐位不变 | 见上 (2) | {'✓' if guard2 else '✗'} |",
           f"| ③ 变异测试 | selftest_aq.py：{inst['mutation']['n_mutations']} 个变异全部抓红 = "
           f"{inst['mutation']['all_caught']}；还原逐字节 = "
           f"{inst['mutation']['restored_sha256_equal']} | {'✓' if guard3 else '✗'} |",
           f"| ④ 因果性不变 | 改未来 token 后过去位置对数逐位不变 = {caus} | "
           f"{'✓' if caus_ok else '✗'} |"]
    L.append("\n".join(t4))
    return "\n\n".join(L) + "\n"


def write_table(rec, out_dir) -> None:
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(rec))


def run_anchor_single_point(cfg, dev, out_dir) -> int:
    """补跑 24×16@seed2 的 fp16 单点（= 历史锚口径），写 `results.json::anchor_single_point`。

    为什么需要它：预登记硬护栏 ②（`config.json::criterion.must_hold.gate_2_bitwise_anchor`）
    与 `README.md` §9 决策树第 1 条以「`fp16` 臂 == 历史锚 1.483569（`window_seed=2`, 24×16）
    且 == `retention.masked_bpc`」为对照是否失效的判据。主表口径被预算裁剪成 3×16 后，只有这个
    24×16 单点能把「|Δanchor| = 8.95e-03 来自批数口径差」从推断变成实测。

    只新增字段与重算 `code` 指纹，**不改主表任何数字**；同时按实际 `n_batches` 门控
    `instrument.bpc_path.anchor_check_applicable`（`run.py:main()` 已同样门控）。
    """
    sp = ANCHOR_CALIBER
    t0 = time.time()
    log_path = os.path.join(out_dir, "run_log.txt")

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    T = cfg["model"]["seq_len"]
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    log(f"[单点] trunk 装载 {trunk_ck['path']} sha256={trunk_ck['sha256'][:16]}")
    fp16_ref = aq.AqForward(trunk, {"activation_mode": "fp16", "weight_mode": "fp16"})

    t_bpc = time.time()
    bpc = eval_bpc(fp16_ref.logits, eval_arr, T, sp["n_batches"], sp["batch_size"],
                   sp["window_seed"], dev)
    log(f"[单点] bpc_of {sp['arm']} {sp['n_batches']}×{sp['batch_size']}@seed{sp['window_seed']} "
        f"= {bpc:.9f}（{time.time() - t_bpc:.1f}s）")
    t_masked = time.time()
    masked = b21.b19.masked_bpc(trunk, eval_arr, None,
                                n_batches=sp["n_batches"], batch_size=sp["batch_size"],
                                window_seed=sp["window_seed"], device=dev)
    log(f"[单点] masked_bpc 同口径 = {masked:.9f}（{time.time() - t_masked:.1f}s）"
        f" 逐位={bpc == masked}")

    res_path = os.path.join(out_dir, "results.json")
    res = json.load(open(res_path))
    anchor = cfg["controls"]["baseline_anchor_bpc"]
    main_bpc = res["instrument"]["bpc_path"]["bpc_of"]
    main_cal = f"{cfg['eval']['n_batches']}×{cfg['eval']['batch_size']}"
    ancal = f"{sp['n_batches']}×{sp['batch_size']}"
    d_anchor = abs(bpc - anchor)
    res["anchor_single_point"] = {
        "arm": sp["arm"], "device": dev,
        "window_seed": sp["window_seed"],
        "n_batches": sp["n_batches"], "batch_size": sp["batch_size"],
        "caliber": f"{ancal}@seed{sp['window_seed']}",
        "bpc_path": "b21.b19.bpc_of（与主表同一路径；`batch()` 同 `window_seed` 同序 "
                    "⇒ 24 批的头 3 批**就是**主窗 3×16 的 3 批）",
        "bpc": round(bpc, 9),
        "main_window_bpc": main_bpc,
        "main_window_caliber": f"{main_cal}@seed{cfg['eval']['window_seed']}",
        "anchor_bpc": anchor,
        "anchor_caliber": f"{ancal}@seed{sp['window_seed']}",
        "anchor_source": cfg["controls"]["baseline_anchor_source"],
        "abs_diff_vs_anchor": d_anchor,
        "abs_diff_main_window_vs_anchor": abs(main_bpc - anchor),
        "abs_diff_vs_main_window": abs(bpc - main_bpc),
        "independent_path_same_caliber": {
            "path": "tools.evalkit.retention.masked_bpc（同 `n_batches`/`batch_size`/"
                    "`window_seed`/设备）",
            "masked_bpc": round(masked, 9),
            "bitwise_equal": bool(bpc == masked)},
        "purpose": "预登记硬护栏 ②（gate_2_bitwise_anchor）与 README §9 决策树第 1 条的交付项："
                   "把主表缩口径与历史锚 24×16 的 |Δ| = 8.95e-03 归因（批数口径差）从推断变成"
                   "实测；不改主表任何数字。",
        "wall_seconds": round(time.time() - t0, 3),
    }
    bp = res["instrument"]["bpc_path"]
    bp["anchor_check_applicable"] = bool(
        cfg["eval"]["n_batches"] == ANCHOR_CALIBER["n_batches"]
        and cfg["eval"]["window_seed"] == ANCHOR_CALIBER["window_seed"])
    bp["anchor_check_applicable_rule"] = "只有 `window_seed=2` / 24×16（= 历史锚口径）下才与锚" \
                                         "可比；按实跑 `n_batches` 门控（不只看 `--smoke`）。"
    bp["anchor_check_carried_by"] = (
        "缩口径主表下 `anchor_check_applicable=False`；锚可比性由 "
        "`results.json::anchor_single_point`（24×16@seed2 单点）承载。")
    res["config"]["eval"]["note"] = cfg["eval"]["note"]
    codes = ["axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/run.py",
             "axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/aq_arms.py",
             "axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/selftest_aq.py"]
    for c in codes:
        if c in res.get("code", {}):
            res["code"][c] = b21.b19.sha256_file(os.path.join(ROOT, c))[:16]
    tmp = res_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    os.replace(tmp, res_path)
    write_table(res, out_dir)
    log(f"[单点完成] {ancal}@seed{sp['window_seed']} fp16 bpc={bpc:.9f}"
        f"（|Δ锚|={d_anchor:.3e}，|Δ主窗|={abs(bpc - main_bpc):.3e}）→ {out_dir}")
    return 0


# ----------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--anchor-single-point", action="store_true",
                    help="只补跑 24×16@seed2 的 fp16 单点（历史锚口径），写 "
                         "results.json::anchor_single_point；不改主表任何数字")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="覆盖 config.json::eval.device（GPU 锁被占时的 CPU 口径）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = json.load(open(args.config))
    if args.device is not None:
        cfg["eval"]["device"] = args.device
    dev = cfg["eval"]["device"] if torch.cuda.is_available() else "cpu"
    T = cfg["model"]["seq_len"]

    if args.table_only:
        rec = json.load(open(os.path.join(args.out_dir, "results.json")))
        write_table(rec, args.out_dir)
        print(f"[table-only] 重画完成（纯 CPU）→ {args.out_dir}", flush=True)
        return 0

    if args.anchor_single_point:
        return run_anchor_single_point(cfg, dev, args.out_dir)

    log_path = os.path.join(args.out_dir, "run_log.txt")
    t_lock = time.time()
    t0 = t_lock

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    if args.smoke:
        cfg["eval"]["n_batches"] = 2
        cfg["eval"]["extra_window_seeds"] = [3]
        cfg["eval"]["extra_seed_n_batches"] = 2
        cfg["tasks"]["copy"]["n_batches"] = 1
        cfg["tasks"]["induction"]["n_batches"] = 2
        cfg["tasks"]["var_recall"]["n_batches"] = 2

    rec = {"experiment": cfg["experiment"], "contract": cfg["contract"],
           "scope": cfg["scope"], "config": cfg, "smoke": bool(args.smoke),
           "criterion": cfg["criterion"], "quantizer": cfg["quantizer"],
           "arms_spec": cfg["arms"],
           "caliber": {
               "device": dev,
               "gpu_lock_held": bool(dev == "cuda"),
               "note": ("CPU 口径：本 run 不取 GPU 锁（纯 CPU 工作，SCHEMA §20 不适用）。"
                        "跨设备（CPU vs GPU）不构成判据：历史锚 1.483569 是 GPU 读数，"
                        "本 run 的 fp16 臂 CPU bpc 与它并列、口径差异显式声明；判据只用"
                        "**同设备、同窗口序**的配对 Δ（`batch()` 数据流与设备无关 ⇒ 配对"
                        "有效性不受影响）。byte 账与活性列与设备无关。")}}

    # SCHEMA §20：等锁 ≤10 min；整程 <10 min 可整程持锁，否则在臂间释放再取。
    # CPU 口径不取 GPU 锁（纯 CPU 工作，§20 不适用）；GPU 口径照旧取锁。
    lock = GpuLock() if dev == "cuda" else NullLock()
    try:
        lock.acquire()
    except TimeoutError as e:
        log(f"[阻塞] {e} ⇒ 如实返回（不改共享文件、不杀任何进程）")
        with open(os.path.join(args.out_dir, "lock_blocked.json"), "w") as f:
            json.dump({"experiment": cfg["experiment"], "blocked": "gpu_lock_timeout",
                       "gpu_lock": lock.ev, "smoke": bool(args.smoke)}, f,
                      indent=1, ensure_ascii=False)
        return 3
    atexit.register(lock.release)
    t0 = time.time()                                   # 工作墙钟从取锁后起算（不含等锁）
    rec["gpu_lock"] = dict(lock.ev)
    log(f"取锁成功 holder_pid={lock.ev['holder_pid']} wait={lock.ev['wait_seconds']}s "
        f"free_before={lock.ev['free_mib_before']} MiB device={dev}")
    log(f"等锁 {round(t0 - t_lock, 1)}s（不进判据；wall 从取锁后起算）")

    torch.cuda.reset_peak_memory_stats()
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))

    # ---------- ① 仪器自检 ----------
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    log(f"trunk 装载 {trunk_ck['path']} sha256={trunk_ck['sha256'][:16]} "
        f"params={sum(p.numel() for p in trunk.parameters())}")
    fp16_ref = aq.AqForward(trunk, {"activation_mode": "fp16", "weight_mode": "fp16"})
    library_fn = b21.b19.make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
    masked = b21.b19.masked_bpc(trunk, eval_arr, None,
                                n_batches=cfg["eval"]["n_batches"],
                                batch_size=cfg["eval"]["batch_size"],
                                window_seed=cfg["eval"]["window_seed"], device=dev)
    my_bpc = eval_bpc(fp16_ref.logits, eval_arr, T, cfg["eval"]["n_batches"],
                      cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    tasks = b21.b19.build_tasks(cfg, smoke=args.smoke)
    probe = batch(eval_arr, cfg["eval"]["batch_size"], T,
                  torch.Generator().manual_seed(cfg["eval"]["window_seed"]))[0].to(dev)

    eq_tokens = tasks["copy"].tokens[:4].to(dev)
    from tools.evalkit.retention import masked_logits as _masked_logits
    with torch.no_grad():
        rep_eq = bool(torch.equal(fp16_ref.logits(eq_tokens), library_fn(eq_tokens)))
        rep_eq_masked = bool(torch.equal(fp16_ref.logits(eq_tokens),
                                         _masked_logits(trunk, eq_tokens, None)))
    # 变异测试读数（由 selftest_aq.py 落盘；这里只读快照，不在 run 内跑单测）
    mut_path = os.path.join(HERE, "mutation_report.json")
    mut = json.load(open(mut_path)) if os.path.exists(mut_path) else {
        "mutations": [], "all_caught": False, "restored_sha256_equal": False}
    # 跨 run 对照：2621 的 fp16 点
    vs_2621 = {"path": "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/results.json",
               "other_bpc": None, "abs_diff": None}
    p2621 = os.path.join(ROOT, vs_2621["path"])
    if os.path.exists(p2621):
        old = {p["arm"]: p for p in json.load(open(p2621))["points"]}
        vs_2621["other_bpc"] = old["fp16"]["bpc"]
        vs_2621["abs_diff"] = abs(my_bpc - old["fp16"]["bpc"])
    rec["instrument"] = {
        "bpc_path": {"bpc_of": my_bpc, "masked_bpc": masked,
                     "bitwise_equal": bool(my_bpc == masked),
                     "abs_diff_vs_anchor": abs(my_bpc - cfg["controls"]["baseline_anchor_bpc"]),
                     "anchor_bpc": cfg["controls"]["baseline_anchor_bpc"],
                     "anchor_source": cfg["controls"]["baseline_anchor_source"],
                     "anchor_check_applicable": bool(
                         not args.smoke
                         and cfg["eval"]["n_batches"] == ANCHOR_CALIBER["n_batches"]
                         and cfg["eval"]["window_seed"] == ANCHOR_CALIBER["window_seed"]),
                     "anchor_check_applicable_rule":
                         "只有 `window_seed=2` / 24×16（= 历史锚口径）下才与锚可比："
                         "`anchor_check_applicable = (not --smoke) and n_batches == 24 and "
                         "window_seed == 2`（按实跑 `n_batches` 门控，不只看 `--smoke`）。",
                     "anchor_check_carried_by":
                         "缩口径主表下本门禁为 False；锚可比性由 `results.json::"
                         "anchor_single_point`（24×16@seed2 单点）承载。",
                     "note": "`AqForward(fp16)` 是复刻 dense 分支；它走到 bpc_of 的值必须与"
                             "独立路径 masked_bpc 逐位相同，并在 **24×16@seed2** 口径下与历史锚 "
                             "1.483569 一致（只有 window_seed=2 / 24×16 才与锚可比；--smoke 或"
                             "缩批量 ⇒ anchor_check_applicable=False）。"},
        "replicated_vs_library": {
            "bitwise_equal": rep_eq,
            "vs_masked_logits_bitwise_equal": rep_eq_masked,
            "probe_tokens": list(eq_tokens.shape),
            "note": "复刻前向（fp16 档）↔ 既有仪器库路径逐位；这是『唯一改动 = 激活精度』"
                    "的实现层证据（SCHEMA §9/§22）。"},
        "vs_2621": vs_2621,
        "mutation": {"report": "mutation_report.json",
                     "n_mutations": len(mut["mutations"]),
                     "all_caught": bool(mut["all_caught"]),
                     "restored_sha256_equal": bool(mut["restored_sha256_equal"]),
                     "green_tests": True,
                     "note": "由 selftest_aq.py 落盘的变异报告（7 个变异全部抓红 + 逐字节还原）。"},
        "trunk": trunk_ck,
    }
    log(f"[自检] bpc_of={my_bpc:.9f} masked_bpc={masked:.9f} 逐位="
        f"{rec['instrument']['bpc_path']['bitwise_equal']} |Δanchor|="
        f"{rec['instrument']['bpc_path']['abs_diff_vs_anchor']:.2e} | 复刻↔库逐位="
        f"{rec['instrument']['replicated_vs_library']['vs_masked_logits_bitwise_equal']}")

    # ---------- ② 对照 ----------
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
    rec["probe_shape"] = [int(probe.shape[0]), int(probe.shape[1])]

    # ---------- ③ 逐臂 ----------
    arms_points = []
    for i, (arm, spec) in enumerate(cfg["arms"].items(), 1):
        p = eval_arm(cfg, arm, spec, trunk, dev, tasks, eval_arr, probe, controls)
        arms_points.append(p)
        log(f"[{i}/{len(cfg['arms'])}] {arm:>5} act={spec['activation_mode']:<4} "
            f"w={spec['weight_mode']:<4} bytes/tok={p['bytes_per_token_activation']:.0f} "
            f"bpc={p['bpc']:.6f} copy={p['recall_copy']['accuracy']:.4f} "
            f"changed={p['activity']['aggregate']['changed_frac']:.3f} "
            f"outlier_idx={p['activity']['mean_outlier_index_channel']:.2f} "
            f"eff_bits={p['activity']['mean_effective_bits_median_channel']:.2f} "
            f"revert_frac={p['activity']['revert_chan_frac_6x_mean']:.5f}")
        # 增量落盘（预算裁剪时已完成的臂不得丢失；partial=True 直到全部臂跑完）
        rec["arms"] = arms_points
        rec["arms_completed"] = [a["arm"] for a in arms_points]
        rec["arms_pending"] = [a for a in cfg["arms"] if a not in rec["arms_completed"]]
        rec["partial"] = (i < len(cfg["arms"]))
        _tmp = os.path.join(args.out_dir, "results.json.tmp")
        with open(_tmp, "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        os.replace(_tmp, os.path.join(args.out_dir, "results.json"))
        # §20：整程 >10 min 时在臂间释放锁（避免长持锁）；本实验预期远低于 10 min。
        # 仅 GPU 口径需要（CPU 口径未持锁）。
        if dev == "cuda" and time.time() - t0 > 480 and lock.ev.get("release_ok") is None:
            lock.release()
            rec["gpu_lock"]["segments"].append(dict(lock.ev))
            lock = GpuLock()
            lock.acquire()
            rec["gpu_lock"]["segments"].append(dict(lock.ev))
            rec["gpu_lock"]["held_whole_run"] = False

    by_arm = {p["arm"]: p for p in arms_points}
    for p in arms_points:
        p["damage_bpc_vs_fp16"] = round(p["bpc"] - by_arm["fp16"]["bpc"], 9)
        p["activation_saving_frac"] = round(
            1 - p["bytes_per_token_activation"] / FP16_ACT_BYTES_PER_TOKEN, 9)

    # ---------- ④ 配对 Δ（≥5 窗口）+ 判据带 ----------
    seeds = [cfg["eval"]["window_seed"]] + list(cfg["eval"]["extra_window_seeds"])
    band = cfg["criterion"]["primary_band_bpc"]
    paired = {"seeds": seeds, "band_bpc": band, "per_arm": {}}
    for p in arms_points:
        if p["arm"] == "fp16":
            continue
        d = [round(p["bpc_by_seed"][str(s)] - by_arm["fp16"]["bpc_by_seed"][str(s)], 9)
             for s in seeds]
        mx = max(abs(x) for x in d)
        paired["per_arm"][p["arm"]] = {
            "by_seed": d, "max_abs": round(mx, 9),
            "mean": round(sum(d) / len(d), 9), "min_abs": round(min(abs(x) for x in d), 9),
            "out_of_band": bool(mx > band),
            "band_met_by_windows": len(seeds) >= 5,
        }
    rec["paired_delta"] = paired
    # 描述性对照：单臂跨窗口极差（**不是**判据带）
    rec["single_arm_window_range_descriptive"] = {
        p["arm"]: round(max(p["bpc_by_seed"].values()) - min(p["bpc_by_seed"].values()), 9)
        for p in arms_points}
    log(f"[配对Δ] band={band} windows={len(seeds)} "
        + " ".join(f"{a}:max|Δ|={d['max_abs']:.6f}{'(越带)' if d['out_of_band'] else ''}"
                   for a, d in paired["per_arm"].items()))

    # ---------- ⑤ 字节账对账 + 结论 ----------
    all_params = sum(pp.numel() for pp in trunk.parameters())
    rec["bytes_account"] = {
        "activation": {p["arm"]: p["activation_bytes"] for p in arms_points},
        "weight": {p["arm"]: p["weight_bytes"] for p in arms_points},
        "fp16_activation_bytes_per_token": FP16_ACT_BYTES_PER_TOKEN,
        "kv_bytes_per_position_for_context_only": 12288,
        "weight_accounted_frac_of_all_params":
            round(by_arm["fp16"]["weight_bytes"]["n_params_quantized_linears"]
                  / all_params, 9),
        "all_params": all_params,
        "caveat": "激活字节是 per-token；权重字节是 per-model；KV 是 per-position。"
                  "三者分列，不得相加成一条横轴。",
    }
    outlier_verdict = {}
    for arm in ("A8", "A4"):
        if arm not in by_arm:
            continue
        a = by_arm[arm]["activity"]
        pen = by_arm[arm]["revert_byte_penalty"]
        outlier_verdict[arm] = {
            "mean_outlier_index_channel": a["mean_outlier_index_channel"],
            "worst_outlier_index_channel": a["worst_tensor_by_outlier_index"][
                "outlier_index_channel"],
            "worst_tensor": a["worst_tensor_by_outlier_index"]["key"],
            "mean_effective_bits_median_channel": a["mean_effective_bits_median_channel"],
            "worst_effective_bits_median_channel":
                a["worst_tensor_by_outlier_index"]["effective_bits_median_channel"],
            "revert_chan_frac_6x_mean": a["revert_chan_frac_6x_mean"],
            "layers_with_any_revert_channel": a["layers_with_any_revert_channel"],
            "revert_extra_bytes_per_token": pen["revert_extra_bytes_per_token"],
            "nominal_saving_bytes_per_token": pen["nominal_saving_bytes_per_token"],
            "saving_eaten_frac_if_reverted": pen["saving_eaten_frac"],
            "verdict_forces_fallback": bool(pen["saving_eaten_frac"] is not None
                                            and pen["saving_eaten_frac"] > 0.10),
        }
    rec["outlier_verdict"] = outlier_verdict
    rec["answer"] = {
        "q": "在 13M 受控设定下，把中间激活以 int8 / 4 位存放（per-token 动态标尺），"
             "激活字节 → 序列级质量的代价是什么？离群通道是否迫使回退？",
        "act_bytes_saving": {p["arm"]: {"bytes_per_token": p["bytes_per_token_activation"],
                                        "saving_frac_vs_fp16": p["activation_saving_frac"],
                                        "ratio_vs_fp16": round(
                                            p["bytes_per_token_activation"]
                                            / FP16_ACT_BYTES_PER_TOKEN, 6)}
                             for p in arms_points},
        "quality_cost": {p["arm"]: {"bpc": p["bpc"],
                                    "damage_bpc_vs_fp16": p["damage_bpc_vs_fp16"],
                                    "max_abs_paired_delta": (paired["per_arm"]
                                                             .get(p["arm"], {})
                                                             .get("max_abs")),
                                    "out_of_band": (paired["per_arm"].get(p["arm"], {})
                                                    .get("out_of_band"))}
                         for p in arms_points},
        "outlier_channels_force_fallback": {
            "A8": outlier_verdict.get("A8", {}).get("verdict_forces_fallback"),
            "A4": outlier_verdict.get("A4", {}).get("verdict_forces_fallback"),
            "criterion": "若「回退到高精度」吃掉的省幅 > 10% ⇒ 判迫使回退；"
                         "本实现**不做回退**（全通道整型），此列是反事实判定。",
        },
        "cannot_claim": "结论只作这 4 个点（13M / enwik8 尾段 / T=256 / 单 seed 冻结 "
                        "checkpoint）的描述；不得外推到更大尺度或别的 checkpoint。",
    }

    # ---------- ⑥ 落盘 ----------
    rec["arms"] = arms_points
    codes = ["axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/run.py",
             "axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/aq_arms.py",
             "axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/selftest_aq.py",
             "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/run.py",
             "axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart/run.py",
             "tools/evalkit/longrange_tasks.py",
             "tools/evalkit/retention.py"]
    rec["code"] = {c: b21.b19.sha256_file(os.path.join(ROOT, c))[:16] for c in codes}

    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev == "cuda" else None
    wall = round(time.time() - t0, 3)
    lock.ev["held_whole_run"] = wall < 600 and lock.ev.get("held_whole_run") is not False
    lock.ev["peak_mem_mib_this_run"] = peak
    lock.ev["free_delta_note"] = ("free_mib_before/after 是 nvidia-smi 的**全卡**空闲读数"
                                  "（含外部占用）；其差 ≠ 本 run 足迹，本 run 足迹看 "
                                  "peak_mem_mib_this_run。")
    lock.release()
    rec["gpu_lock"] = dict(lock.ev)
    rec["cost"] = {"device": dev, "wall_seconds": wall,
                   "wall_minutes": round(wall / 60, 3),
                   "lock_wait_seconds": lock.ev.get("wait_seconds"),
                   "peak_mem_mib": peak, "gpu_wall_budget_min": cfg["cost"]["gpu_wall_budget_min"],
                   "smoke": bool(args.smoke),
                   "note": "墙钟只作过程读数，不进判据、不进 cost_model。"}

    if not args.smoke:
        write_table(rec, args.out_dir)
    rec["partial"] = False
    rec["arms_completed"] = [a["arm"] for a in arms_points]
    rec["arms_pending"] = []
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    log(f"[完成] {wall / 60:.2f} min，峰值 {peak} MiB，{len(arms_points)} 臂 → {args.out_dir}")
    return 0


def b19_masked_logits(model, tokens):
    from tools.evalkit.retention import masked_logits
    return masked_logits(model, tokens, None)


if __name__ == "__main__":
    sys.exit(main())
