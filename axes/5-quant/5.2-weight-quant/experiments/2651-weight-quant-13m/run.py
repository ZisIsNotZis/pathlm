"""2651 · 权重量化最小判决（含标尺字节账）—— 实验入口。

契约：同目录 `README.md`（§1–§11）+ `config.json::criterion`（判据线**运行前**写入）。
语义层：同目录 `wq_arms.py`（权重假量化的唯一新逻辑：假量化往返 + 字节账 + 活性量 + 死权重探针）。

命题：在 13.5M 冻结 trunk、纯推理、单因素的设定下，把权重以 int8 / int4（含分组标尺）存放，
「每参数部署字节（含标尺）→ 序列级两腿质量（clean bpc + 合成长程召回）」的前沿是什么？
int8 / int4 是否落在占优侧？

唯一改动因素 = 权重矩阵的 `(bits, scheme, group)`；KV / 激活 / 偏置 / Norm / tied 读出逐位不动。
不训练、不新增参数（`params_added = 0`）。

用法：
  python3 run.py              # 全跑（纯推理）
  python3 run.py --smoke      # 冒烟：缩小批量/档数，实测步时与显存（不外推时间）
  python3 run.py --charts-only  # 只从已有 results.json 重画表（纯 CPU，不取 GPU 锁）
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
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import wq_arms as wq                                                   # noqa: E402
from pathlm.data import load_enwik8_full                              # noqa: E402
from tools.evalkit.kv_quant import release_logits                      # noqa: E402
from tools.evalkit.longrange_tasks import (                           # noqa: E402
    corrupt_span, local_prior_accuracy, check_local_prior_resistance)

P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
P2631 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal")

#: 本实验的臂 = 模型里**全部** `nn.Linear`（60 个，含 12 个死权重 gates）——主口径；
#: `scope_linear_live`（48 个）是旁列，见 `wq_arms` 模块文档。
INCLUDE_DEAD_GATES = True

#: 在力（in force）排序带：预登记 = 0.014 bpc，口径 = 配对 Δ、≥5 窗口（本批 5 个）。
IN_FORCE_BAND_BPC = 0.014

#: SCHEMA §20 的 GPU 互斥锁。
GPU_LOCK_DIR = "/tmp/pathlm_gpu.lock"


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2621 的评估管线（内含 2619 的 b19：load_trunk / bpc_of / masked_bpc / build_tasks /
#: pooled_accuracy / floor_threshold / sha256_file）；只读复用，不改它。
b21 = _load_module(os.path.join(P2621, "run.py"), "kv_budget_rest_2621")
#: 2631 的前缀不变性仪器（本实验直接复用其两条检查）。
cc2631 = _load_module(os.path.join(P2631, "causal_check.py"), "causal_check_2631")


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
    （`results.json::gpu_lock`），供 §20 记账核查。等锁上限 10 min（40×15 s）。
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

    def _stale(self):
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

    def acquire(self):
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

    def release(self):
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


# ----------------------------------------------------------------- 一个臂的完整读数


def b21_fp32_fn(model, T: int, dev: str):
    """本实验的统一 `logits_fn`：KV 走 `fp32`（恒等）⇒ 与 stock 释放版逐位一致。

    权重由 `wq_arms.arm_context` 在模型上装/拆 ⇒ 同一条 `logits_fn` 覆盖四个臂。
    """
    return b21.b19.make_logits_fn(model, {"kind": "quant", "mode": "fp32"}, T, dev)


def eval_arm(cfg, model, spec, dev, tasks, eval_arr, controls, n_batches, seeds,
             paired_n_batches, do_recall=True):
    """一个臂的完整读数：clean bpc（两条独立路径）+ 召回 + 前缀不变 + 活性 + 字节账 + 配对窗口。"""
    T = cfg["model"]["seq_len"]
    bs = cfg["eval"]["batch_size"]
    ws = cfg["eval"]["window_seed"]
    probe_tokens = tasks["var_recall"].tokens[:4].to(dev)
    restore = wq.apply_arm(model, spec, INCLUDE_DEAD_GATES)
    try:
        fn = b21_fp32_fn(model, T, dev)
        bpc = b21.b19.bpc_of(fn, eval_arr, T, n_batches, bs, ws, dev)
        bpc_lib = b21.b19.masked_bpc(model, eval_arr, None, n_batches=n_batches,
                                     batch_size=bs, window_seed=ws, device=dev)
        out = {"arm": spec.name, "spec": spec.as_dict(),
               "bpc": round(bpc, 6), "bpc_retention_path": round(bpc_lib, 6),
               "bpc_paths_bitwise_equal": bool(bpc == bpc_lib),
               "prefix_invariance": cc2631.prefix_invariance_of_logits(fn, probe_tokens),
               "activity": wq.activity_of(model, spec, INCLUDE_DEAD_GATES),
               "byte_account": wq.byte_account(model, spec, INCLUDE_DEAD_GATES)}
        if do_recall:
            copy_pooled = b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)
            chance = controls["chance_uniform"]
            out["recall_copy"] = {**copy_pooled,
                                  "at_floor": bool(copy_pooled["accuracy"]
                                                   <= b21.b19.floor_threshold(
                                                       copy_pooled["n"], chance))}
            out["recall_copy_corrupt"] = b21.b19.pooled_accuracy(
                fn, tasks["copy_corrupt_by_pos"], dev)
            out["recall_induction"] = b21.b19.pooled_accuracy(
                fn, tasks["induction_by_start"], dev)
            out["recall_var_recall"] = b21.b19.pooled_accuracy(
                fn, tasks["var_recall_by_d"], dev)
            out["chance_uniform"] = chance
        # 配对窗口（同一批评测窗口上臂与 fp16 的 bpc 逐 seed 配对）
        out["paired_bpc_by_seed"] = {str(s): round(
            b21.b19.bpc_of(fn, eval_arr, T, paired_n_batches, bs, s, dev), 6)
            for s in seeds}
    finally:
        restore()
    return out


# ----------------------------------------------------------------- 表渲染


def render_table(rec) -> str:
    cfg = rec["config"]
    pts = rec["points"]
    by = {p["arm"]: p for p in pts}
    base = by["fp16"]
    L = []
    L.append("# 2651 · 权重量化最小判决（含标尺字节账）—— 前沿总表")
    anchor_sentence = (
        f"本批 device=cpu ⇒ 其 bpc 与 **GPU** 口径的历史锚 "
        f"{rec['instrument']['anchor_bpc']:.6f} **并列报告**，跨设备不构成判据（浮点归约顺序不同 ⇒ "
        f"末位可能不同）；判据 = 同设备、同窗口序的配对 Δbpc。"
        if rec.get("caliber", {}).get("is_cpu") else
        f"= 历史锚 {rec['instrument']['anchor_bpc']:.6f}（同设备 GPU 口径）。")
    L.append(
        f"**口径**：13.5M 冻结 trunk（sha256 断言）、纯推理、不训练、**唯一改动因素 = 权重矩阵的 "
        f"`(bits, scheme, group)`**；KV / 激活 / 偏置 / Norm / tied 读出逐位不动。"
        f"横轴（主口径）= **每参数部署字节（值 + 标尺）**，按参数数加权，"
        f"落在 `scope_linear_all`（模型里全部 60 个 `nn.Linear` 的权重矩阵）；标尺 fp16 存储 = "
        f"2 B/把。`fp16` 臂 = 权重 fp16 = stock 释放前向；"
        + anchor_sentence
        + f"bpc 窗口：主表 = `window_seed={cfg['eval']['window_seed']}` × "
        f"{cfg['eval']['n_batches']}×{cfg['eval']['batch_size']}（与历史锚同窗口序）；"
        f"配对窗口 = {cfg['eval']['paired_windows_seeds']}（每窗 "
        f"{cfg['eval']['paired_windows_n_batches']}×{cfg['eval']['batch_size']}）。"
        f"**在力排序带 = {rec['band']['in_force_band_bpc']:.6f} bpc**（"
        f"{rec['band']['declaration']}）")
    L.append("## 表 1 · 前沿主表（每参数字节 → 两腿质量）\n\n"
             "| 臂 | 每参数字节（值+标尺） | ×fp16 | 省幅 | bpc | Δbpc vs fp16 | "
             "配对 max&#124;Δ&#124;（5 窗） | 与 0.014 带 | copy 召回 | Wilson 95% | "
             "induction | var_recall | 前缀不变（未来/同批） | changed_frac | rel_err(L1) | "
             "max 逐矩阵 rel_err |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pts:
        dep = p["byte_account"]["scope_linear_all"]
        ci = p["recall_copy"]["wilson95"] if "recall_copy" in p else None
        paired = rec["paired"][p["arm"]]
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
        resc = (f"{p['recall_copy']['accuracy']:.4f}"
                f"{' (地板)' if p['recall_copy']['at_floor'] else ''}") if ci else "—"
        L.append(
            f"| `{p['arm']}` | {dep['bytes_per_param']:.6f} | ×{dep['to_fp16_multiple']:.4f} | "
            f"{dep['reduction_vs_fp16']:.1%} | {p['bpc']:.6f} | "
            f"{p['bpc'] - base['bpc']:+.6f} | {paired['max_abs_delta_bpc']:.6f} | "
            f"**{paired['verdict']}** | {resc} | {ci_s} | "
            f"{(p['recall_induction']['accuracy'] if 'recall_induction' in p else float('nan')):.4f} | "
            f"{(p['recall_var_recall']['accuracy'] if 'recall_var_recall' in p else float('nan')):.4f} | "
            f"{p['prefix_invariance']['future_token_past_logits_equal']}/"
            f"{p['prefix_invariance']['peer_request_logits_equal']} | "
            f"{p['activity']['changed_frac']:.4f} | {p['activity']['rel_err']:.6f} | "
            f"{p['activity']['max_rel_err_per_matrix_l1']:.6f} |")
    L.append("> 读法：**Δbpc = bpc_臂 − bpc_fp16**（同窗口同批数据，正 = 量化更差）；"
             "「配对 max|Δ|」= 5 个窗口上该臂与 fp16 的**配对 Δbpc** 的最大绝对值；它 ≤ 0.014 ⇒ "
             "**不越带**（该带内质量代价不可排序），> 0.014 ⇒ **越带**（本尺度确有可排序的质量代价）；"
             "「每参数字节」是**部署口径**（值 + 标尺，标尺不得当免费）；实测显存峰值不随档下降"
             "（实现把假量化权重现算成 fp32），故只报解析列。"
             f"召回地板：chance = {rec['controls']['chance_uniform']:.4f}（1/64）、"
             f"copy 局部先验 = {rec['controls']['local_prior_copy']['accuracy']:.4f}。")

    L.append("## 表 2 · 字节账逐矩阵（`scope_linear_all`，主口径；fp16 臂的标尺列为 0）\n\n"
             "| 矩阵 | shape | 参数量 | 标尺个数(int8 pc / int8 g32 / int4 g32) | "
             "每参数字节(int8 pc / int8 g32 / int4 g32) |\n|---|---|---|---|---|")
    rows = {r["name"]: r for r in pts[0]["byte_account"]["per_matrix"]}
    for arm in ("int8_per_channel", "int8_group32", "int4_group32"):
        for r in by[arm]["byte_account"]["per_matrix"]:
            rows[r["name"]]["ns_" + arm] = r["n_scales"]
            rows[r["name"]]["bpp_" + arm] = r["bytes_per_param"]
    for name, r in rows.items():
        L.append(f"| `{name}` | {r['shape'][0]}×{r['shape'][1]} | {r['n_params']} | "
                 f"{r.get('ns_int8_per_channel')} / {r.get('ns_int8_group32')} / "
                 f"{r.get('ns_int4_group32')} | {r.get('bpp_int8_per_channel'):.6f} / "
                 f"{r.get('bpp_int8_group32'):.6f} / {r.get('bpp_int4_group32'):.6f} |")
    tot = {a: by[a]["byte_account"]["scope_linear_all"] for a in by}
    L.append("> 按参数数加权总账（**主口径**）：" + "；".join(
        f"`{a}` = {tot[a]['bytes_per_param']:.6f} B/参数（×{tot[a]['to_fp16_multiple']:.4f}，"
        f"省 {tot[a]['reduction_vs_fp16']:.1%}，标尺个数 {tot[a]['n_scales']}）"
        for a in ("fp16", "int8_per_channel", "int8_group32", "int4_group32")))
    L.append("> 旁列口径：" + "；".join(
        f"`{a}` live={by[a]['byte_account']['scope_linear_live']['bytes_per_param']:.6f} / "
        f"whole={by[a]['byte_account']['scope_whole_model']['bytes_per_param']:.6f}"
        for a in ("fp16", "int8_per_channel", "int8_group32", "int4_group32"))
        + "（live = 剔除 `gates.*` 的 48 个活矩阵；whole = 把未量化的偏置/Norm/embedding 按 fp16 计进分母）")

    L.append("## 表 3 · 配对 Δbpc 逐窗口（判据带 0.014、口径 = 配对 Δ、5 窗口）\n\n"
             f"窗口 seed = {rec['band']['seeds']}（每窗 "
             f"{rec['config']['eval']['paired_windows_n_batches']}×"
             f"{rec['config']['eval']['batch_size']}）；`Δ = bpc_臂 − bpc_fp16`（同一批窗口）。\n\n"
             "| 臂 | bpc 逐 seed | Δ 逐 seed | Δ 均值 | Δ 区间 | max&#124;Δ&#124; | 判定 |\n"
             "|---|---|---|---|---|---|---|")
    for a in ("int8_per_channel", "int8_group32", "int4_group32"):
        pr = rec["paired"][a]
        L.append(f"| `{a}` | {[round(v, 6) for v in pr['bpc_by_seed']]} | "
                 f"{[round(v, 6) for v in pr['delta_by_seed']]} | {pr['mean_delta_bpc']:+.6f} | "
                 f"[{pr['min_delta_bpc']:+.6f}, {pr['max_delta_bpc']:+.6f}] | "
                 f"{pr['max_abs_delta_bpc']:.6f} | **{pr['verdict']}** |")
    L.append(f"> fp16 基线逐 seed bpc = {[round(v, 6) for v in rec['paired']['fp16']['bpc_by_seed']]}。"
             f"**描述性读数（不是判据带，不得与 0.014 混用）**：单臂跨窗口 bpc 极差 = "
             + "；".join(f"`{a}` {rec['single_arm_range_bpc'][a]:.6f}" for a in rec["single_arm_range_bpc"]))

    L.append("## 表 4 · 解析阶梯（标尺摊薄的极限；只作解析，不新增质量臂）\n\n"
             "| 方案 | 位数 | G | 标尺个数 | 每参数字节 | ×fp16 | 省幅 | int4 每参数字节 | "
             "int4 是否反占内存(≥2) |\n|---|---|---|---|---|---|---|---|---|")
    for r in pts[0]["byte_account"]["ladder"]:
        if r["bytes_per_param"] is None:
            L.append(f"| {r['scheme']} | {r['bits']} | {r['group']} | — | — | — | — | — | "
                     f"{r.get('note', '无效')} |")
        else:
            i4 = r.get("int4_bytes_per_param")
            L.append(f"| {r['scheme']} | {r['bits']} | {r['group']} | {r.get('n_scales', 0)} | "
                     f"{r['bytes_per_param']:.6f} | ×{r['to_fp16_multiple']:.4f} | "
                     f"{1 - r['to_fp16_multiple']:.1%} | "
                     f"{(f'{i4:.6f}' if i4 else '—')} | "
                     f"{('是' if i4 and i4 >= 2.0 else '否') if i4 else '—'} |")
    L.append("> 读法：本模型被量化矩阵的 `in_features ∈ {256, 1536}`，每行远长于 32 ⇒ `per_channel`"
             "（每行一把标尺）比 `group-32`（每 32 个值一把）**字节更省**但标尺更粗。"
             "int4 只有在极小的 G（标尺开销 ~2/G 顶到 0.5 B 以上）下才会反占内存（≥2 B/参数）——"
             "本阶梯给出具体阈值。")

    L.append("## 表 5 · 合成长程召回（第二腿；同一仪器、同一 seed）\n\n"
             "| 臂 | copy 汇总 | copy@src0 | copy@src96 | copy@src191 | copy 毁源对照 | "
             "induction | var_recall | 地板上的腿 |\n|---|---|---|---|---|---|---|---|---|")
    for p in pts:
        if "recall_copy" not in p:
            continue
        ps = p["recall_copy"]["per_setting"]
        floors = [n for n, k in (("copy", "recall_copy"), ("induction", "recall_induction"),
                                 ("var_recall", "recall_var_recall"))
                  if p[k]["accuracy"] <= b21.b19.floor_threshold(
                      p[k]["n"], rec["controls"]["chance_uniform"])]
        L.append(f"| `{p['arm']}` | {p['recall_copy']['accuracy']:.4f} | "
                 f"{ps['0']['accuracy']:.4f} | {ps['96']['accuracy']:.4f} | "
                 f"{ps['191']['accuracy']:.4f} | {p['recall_copy_corrupt']['accuracy']:.4f} | "
                 f"{p['recall_induction']['accuracy']:.4f} | "
                 f"{p['recall_var_recall']['accuracy']:.4f} | "
                 f"{', '.join(floors) if floors else '（无）'} |")
    L.append("> 判据：某腿 ≤ `chance + 2·sigma_wilson` ⇒ 在该任务上**不可分辨**（`at_floor`，"
             "不是「机制无效」，SCHEMA §19）。率类指标**不借用** 0.014 bpc 带，只报 Wilson 95%。")

    inst = rec["instrument"]
    L.append("## 表 6 · 仪器自检与可控性\n\n"
             f"- `fp16` 臂：`bpc_of` = {inst['fp16_identity']['bpc_of']:.6f}、"
             f"`retention.masked_bpc` = {inst['fp16_identity']['retention_masked_bpc']:.6f}、"
             f"逐位相等 = {inst['fp16_identity']['bitwise_equal']}、"
             f"（device={inst['fp16_identity'].get('device', rec.get('caliber', {}).get('device'))}）"
             f"与历史锚 {inst['anchor_bpc']:.6f} 的差 = "
             f"{inst['fp16_identity']['abs_diff_vs_anchor']:.2e}。\n"
             f"- `fp16` 臂 logits 与 stock 释放前向逐位相等 = "
             f"{inst['fp16_identity']['release_logits_bitwise_equal']}。\n"
             f"- 合成任务路径等值 = {inst['synth_path']['n_hits_equal']}（n / n_hits 全等）。\n"
             f"- 死权重探针：`gates.*`（{inst['dead_gate_probe']['n_dead_modules']} 个矩阵、"
             f"{inst['dead_gate_probe']['dead_param_count']} 个参数）清零后 logits 逐位不变 = "
             f"{inst['dead_gate_probe']['zeroed_logits_bitwise_equal']} ⇒ "
             f"只进字节账、不进质量读数。\n"
             f"- 逐臂还原：{inst['patch_reversibility']}\n"
             f"- 变异电池：{rec['mutation']['n_caught_red']}/{rec['mutation']['n_mutations']} 抓红"
             f"（`selftest_wq.py`，见 `mutation_report.json`）。\n"
             f"- §20 取锁：`wait_seconds` = {rec['gpu_lock'].get('wait_seconds')}、"
             f"`release_ok` = {rec['gpu_lock'].get('release_ok')}、"
             f"`held_whole_run` = {rec['gpu_lock'].get('held_whole_run')}、"
             f"`peak_mem_mib_this_run` = {rec['cost']['peak_mem_mib']} MiB。"
             + (f"（{rec['gpu_lock']['skipped']}）" if "skipped" in rec["gpu_lock"] else ""))
    # 表行之间不得留空行：否则 Markdown 不把连续的 `|` 行渲染成表格（只删两侧都是表行的空行；
    # 表后的空行——进入 `> 读法`/正文——保留）。
    text = "\n\n".join(L)
    lines = text.split("\n")
    kept = []
    for i, ln in enumerate(lines):
        if (ln == "" and kept and kept[-1].startswith("|")
                and i + 1 < len(lines) and lines[i + 1].startswith("|")):
            continue
        kept.append(ln)
    return "\n".join(kept) + "\n"


def write_table(rec, out_dir) -> None:
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(rec))


# ----------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None,
                    help="覆盖 config.eval.device（如 cpu）：纯 CPU 口径不取 GPU 锁（SCHEMA §9）")
    ap.add_argument("--charts-only", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_log.txt")
    log = open(log_path, "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def dump_results(rec_, path=None):
        """把当前 rec 原样落盘（每跑完一臂调一次 ⇒ 会话被切断也不丢已完成臂）。"""
        with open(path or os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(rec_, f, indent=1, ensure_ascii=False)

    cfg = json.load(open(args.config))
    say(f"[cfg] {cfg['experiment']}；臂 = {list(cfg['arms'])}；"
        f"窗口 = {cfg['eval']['n_batches']}×{cfg['eval']['batch_size']} @ seed "
        f"{cfg['eval']['window_seed']}；配对窗口 = {cfg['eval']['paired_windows_seeds']}")

    if args.charts_only:
        rec = json.load(open(os.path.join(args.out_dir, "results.json")))
        write_table(rec, args.out_dir)
        say(f"[charts-only] 重画完成（纯 CPU）→ {args.out_dir}")
        return 0

    dev = args.device or (cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    uses_gpu = str(dev).startswith("cuda")
    # `eval.device` 只是预登记默认值，可被 CLI `--device` 覆盖；机器可读地记录**实跑**设备（N2 处置）。
    cfg["eval"]["device_resolved"] = str(dev)
    cfg["eval"]["device_note"] = ("本字段 = 实跑解析设备；`eval.device` 是预登记默认值、可被 CLI "
                                   "`--device` 覆盖 ⇒ 口径以本字段与 `caliber` 为准。")
    T = cfg["model"]["seq_len"]
    paired_seeds = list(cfg["eval"]["paired_windows_seeds"])
    paired_n = cfg["eval"]["paired_windows_n_batches"]
    if args.smoke:
        # 冒烟：批量降到 2（同一份 cfg 副本内），配对窗口取前 2 个 —— 只为量步时与显存。
        cfg["eval"]["n_batches"] = 2
        paired_n = 2
        paired_seeds = paired_seeds[:2]
    n_batches = cfg["eval"]["n_batches"]

    # SCHEMA §20：GPU 口径下整程 <10 min ⇒ 允许整程持锁；但必须留机器可读的取/释与前后空闲显存记录。
    # 纯 CPU 口径（`--device cpu`）不取 GPU 锁：CPU 工作量不与 GPU 作业互斥（SCHEMA §9）。
    lock = GpuLock()
    if uses_gpu:
        lock.acquire()
        atexit.register(lock.release)
    t0 = time.time()
    if uses_gpu:
        torch.cuda.reset_peak_memory_stats()
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    if uses_gpu:
        say(f"[锁] holder_pid={lock.ev['holder_pid']} wait={lock.ev['wait_seconds']}s "
            f"free_before={lock.ev['free_mib_before']} MiB")
    else:
        say(f"[口径] device={dev}，纯 CPU 运行 ⇒ 不取 GPU 锁（SCHEMA §20 不适用）；"
            f"全部臂同一设备；判据 = 同设备同窗口序的配对 Δ")

    # ---------- ① 仪器自检（先证明对照没坏） ----------
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    say(f"[ckpt] {trunk_ck['path']} sha256={trunk_ck['sha256'][:16]}… size={trunk_ck['size_bytes']}")
    tasks = b21.b19.build_tasks(cfg, smoke=args.smoke)
    probe = tasks["var_recall"]
    eq_tokens = tasks["copy"].tokens[:4].to(dev)

    fp32_fn = b21_fp32_fn(trunk, T, dev)
    lib_bpc = b21.b19.masked_bpc(trunk, eval_arr, None, n_batches=cfg["eval"]["n_batches"],
                                 batch_size=cfg["eval"]["batch_size"],
                                 window_seed=cfg["eval"]["window_seed"], device=dev)
    my_bpc = b21.b19.bpc_of(fp32_fn, eval_arr, T, cfg["eval"]["n_batches"],
                            cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    lib_acc = b21.b19.lib_accuracy(trunk, probe, device=dev)
    my_acc = b21.b19.synth_accuracy(fp32_fn, probe, dev)
    release_logits_equal = bool(torch.equal(fp32_fn(eq_tokens),
                                            release_logits(trunk, eq_tokens)))
    anchor = float(cfg["controls"]["baseline_anchor_bpc"])
    dead = wq.dead_gate_probe(trunk, eq_tokens)

    # 逐臂还原自检：量化臂装上 ⇒ logits 改变；拆掉 ⇒ 逐位回到 stock。
    base_logits = fp32_fn(eq_tokens)
    rev = {}
    for arm in cfg["arms"]:
        spec = wq.arm_spec(arm)
        with wq.arm_context(trunk, spec, INCLUDE_DEAD_GATES):
            patched = fp32_fn(eq_tokens)
        restored = fp32_fn(eq_tokens)
        rev[arm] = {"logits_changed_when_patched": bool(not torch.equal(patched, base_logits)),
                    "logits_restored_bitwise": bool(torch.equal(restored, base_logits)),
                    "no_forward_attr_left": bool(all(
                        "forward" not in m.__dict__
                        for _, m in wq.linear_weight_matrices(trunk, INCLUDE_DEAD_GATES)))}

    rec = {"experiment": cfg["experiment"], "contract": cfg["contract"],
           "scope": cfg["scope"], "config": cfg, "smoke": bool(args.smoke),
           "criterion": cfg["criterion"], "axis_definition": cfg["axis_definition"],
           "provenance": cfg["provenance"], "checkpoint": trunk_ck,
           "instrument": {
               "anchor_bpc": anchor,
               "fp16_identity": {"device": dev,
                                 "bpc_of": round(my_bpc, 6),
                                 "retention_masked_bpc": round(lib_bpc, 6),
                                 "abs_diff": abs(my_bpc - lib_bpc),
                                 "bitwise_equal": bool(my_bpc == lib_bpc),
                                 "abs_diff_vs_anchor": abs(my_bpc - anchor),
                                 "release_logits_bitwise_equal": release_logits_equal,
                                 "note": "本项的 `fp16` 臂 = **权重** fp16 = stock 释放路径。"
                                         "注意 5.1 表里的 `fp16` 臂是 **KV** fp16 往返"
                                         "（1.483571），两者不是同一个东西、不得混用。"},
               "synth_path": {"run::synth_accuracy": my_acc["accuracy"],
                              "longrange_tasks.accuracy": lib_acc["accuracy"],
                              "n_equal": my_acc["n"] == lib_acc["n"],
                              "n_hits_equal": my_acc["n_hits"] == lib_acc["n_hits"]},
               "dead_gate_probe": dead,
               "patch_reversibility": rev,
               "note": "对照没坏的证据：① 两条 bpc 路径逐位相等；② fp16 臂 = stock 释放前向；"
                       "③ 量化臂装上真的改 logits、拆掉逐位还原；④ gates.* 被证明是死权重。",
           }}
    say(f"[自检] bpc_of={my_bpc:.6f} vs masked_bpc={lib_bpc:.6f}（逐位={my_bpc == lib_bpc}）；"
        f"锚={anchor:.6f} 差={abs(my_bpc - anchor):.2e}；release_logits 逐位={release_logits_equal}；"
        f"gates 死权重={dead['zeroed_logits_bitwise_equal']}")
    say(f"[还原] {rev}")

    # ---------- ② 对照（地板 / 毁源） ----------
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
    say(f"[对照] chance={controls['chance_uniform']:.4f} 局部先验(copy)="
        f"{controls['local_prior_copy']['accuracy']:.4f}")

    # ---------- ③ 四个臂 ----------
    points = []
    for i, arm in enumerate(cfg["arms"], 1):
        spec = wq.arm_spec(arm)
        t_arm = time.time()
        p = eval_arm(cfg, trunk, spec, dev, tasks, eval_arr, controls, n_batches,
                     paired_seeds, paired_n, do_recall=True)
        points.append(p)
        dep = p["byte_account"]["scope_linear_all"]
        say(f"[{i}/4] {arm:<18} bytes/param={dep['bytes_per_param']:.6f} "
            f"(×{dep['to_fp16_multiple']:.4f}) bpc={p['bpc']:.6f} "
            f"copy={p['recall_copy']['accuracy']:.4f} "
            f"inv={p['prefix_invariance']['future_token_past_logits_equal']}/"
            f"{p['prefix_invariance']['peer_request_logits_equal']} "
            f"({time.time() - t_arm:.1f}s)")
        rec["points"] = points
        # ---------- 逐臂持久化（防会话被切断而丢已完成臂） ----------
        # 写**另一个文件名** `results.partial.json`：`results.json` 只在全跑完后才出现，
        # 这样排在后头的 watcher / 判读不会把半成品当成品。
        rec["partial"] = {
            "partial": len(points) < len(cfg["arms"]),
            "arms_completed": [q["arm"] for q in points],
            "arms_pending": [a for a in cfg["arms"]
                             if a not in {q["arm"] for q in points}],
            "paths_note": "`results.json` 仅在四个臂全跑完后出现；本文件是逐臂落盘的半成品。",
            "missing_until_complete": ["gates", "paired", "band", "single_arm_range_bpc",
                                       "frontier", "frontier_analysis", "毁源对照", "caliber",
                                       "cost", "gpu_lock", "memory_convention", "code"],
        }
        dump_results(rec, os.path.join(args.out_dir, "results.partial.json"))
        say(f"[逐臂落盘] 已完成 {rec['partial']['arms_completed']} ⇒ results.partial.json")
    rec["points"] = points

    # ---------- ③b 门禁（G1 活性 / G2 非量化臂逐位复原 / G3 字节账 / G4 误差界） ----------
    by = {p["arm"]: p for p in points}
    g1 = {a: {"changed_frac": by[a]["activity"]["changed_frac"],
              "max_abs_err": by[a]["activity"]["max_abs_err"],
              "rel_err": by[a]["activity"]["rel_err"],
              "pass": bool(by[a]["activity"]["changed_frac"] > 0.0
                            and by[a]["activity"]["max_abs_err"] > 0.0)}
          for a in cfg["arms"] if a != "fp16"}
    g3 = {a: {"scales_match_analytic": by[a]["activity"]["scales_match_analytic"],
              "bound_failures": by[a]["activity"]["bound_failures"],
              "pass": bool(by[a]["activity"]["scales_match_analytic"]
                            and not by[a]["activity"]["bound_failures"])}
          for a in cfg["arms"] if a != "fp16"}
    anchor_in_force = bool(uses_gpu and not args.smoke)
    g2 = {"fp16_bpc": by["fp16"]["bpc"],
          "fp16_bpc_paths_bitwise_equal": by["fp16"]["bpc_paths_bitwise_equal"],
          "fp16_bpc_retention_path": by["fp16"]["bpc_retention_path"],
          "anchor_bpc": anchor,
          "abs_diff_vs_anchor": abs(by["fp16"]["bpc"] - anchor),
          "anchor_caliber_in_force": anchor_in_force,
          "apply_anchor_check": anchor_in_force,
          "log_label_note": ("run_log 的 `Δ锚(6位)` = |fp16.bpc − 锚|，其中 fp16.bpc 落盘为 6 位小数"
                             "⇒ 显示 0.00e+00；**全精度**锚差见 `instrument.fp16_identity.abs_diff_vs_anchor`。"
                             "另有独立读数 `bpc_paths_bitwise_equal`（两条 bpc 路径逐位相等），与本行不同。"),
          "note": (f"本 run 的 device={dev}。历史锚 {anchor:.6f} 是 **GPU** 口径的释放版基线"
                   f"（2619/2614）⇒ device=cpu 时两者为**跨设备**比较、不构成判据，只并列报告；"
                   f"此时在力的门禁只有『两条 bpc 路径逐位相等』（同设备、同窗口序）。"),
          "pass": bool(by["fp16"]["bpc_paths_bitwise_equal"]
                       and (not anchor_in_force
                            or abs(by["fp16"]["bpc"] - anchor) < 1e-6))}
    gates = {"G1_activity": g1, "G2_nonquant_identity": g2, "G3_G4_byte_account_and_bound": g3,
             "G5_mutation": None, "G6_dead_weights": dead,
             "all_pass_smoke_aware": bool(all(v["pass"] for v in g1.values())
                                          and all(v["pass"] for v in g3.values())
                                          and g2["pass"])}
    rec["gates"] = gates
    say("[门禁] G1=" + repr({a: v["pass"] for a, v in g1.items()})
        + " G2=" + repr(g2["pass"]) + "(Δ锚(6位)=%.2e)" % g2["abs_diff_vs_anchor"]
        + " G3+G4=" + repr({a: v["pass"] for a, v in g3.items()}))
    if not gates["all_pass_smoke_aware"]:                        # pragma: no cover
        say("[门禁] 不通过 ⇒ 按契约作废本批数字（失败决策树第 1/2 岗）")

    # ---------- ④ 配对窗口与判据带 ----------
    fp16_seeds = by["fp16"]["paired_bpc_by_seed"]
    paired = {"fp16": {"bpc_by_seed": [fp16_seeds[str(s)] for s in paired_seeds],
                       "delta_by_seed": [0.0] * len(paired_seeds),
                       "mean_delta_bpc": 0.0, "min_delta_bpc": 0.0, "max_delta_bpc": 0.0,
                       "max_abs_delta_bpc": 0.0, "verdict": "基线"}}
    for arm in cfg["arms"]:
        if arm == "fp16":
            continue
        bps = [by[arm]["paired_bpc_by_seed"][str(s)] for s in paired_seeds]
        fps = [fp16_seeds[str(s)] for s in paired_seeds]
        d = [round(a - b, 9) for a, b in zip(bps, fps)]
        m = max(abs(x) for x in d)
        paired[arm] = {"bpc_by_seed": bps, "delta_by_seed": d,
                       "mean_delta_bpc": round(sum(d) / len(d), 9),
                       "min_delta_bpc": min(d), "max_delta_bpc": max(d),
                       "max_abs_delta_bpc": round(m, 9),
                       "verdict": "越带" if m > IN_FORCE_BAND_BPC else "不越带"}
    rec["paired"] = paired
    rec["single_arm_range_bpc"] = {
        a: round(max(by[a]["paired_bpc_by_seed"][str(s)] for s in paired_seeds)
                 - min(by[a]["paired_bpc_by_seed"][str(s)] for s in paired_seeds), 9)
        for a in cfg["arms"]}
    rec["band"] = {"in_force_band_bpc": IN_FORCE_BAND_BPC,
                   "seeds": paired_seeds,
                   "caliber": "配对 Δ（同一批窗口上 bpc_臂 − bpc_fp16），逐 seed，取 max|Δ|",
                   "windows": len(paired_seeds),
                   "standard_met": len(paired_seeds) >= 5,
                   "declaration": (f"本批 {len(paired_seeds)} 个窗口"
                                   f"{'满足' if len(paired_seeds) >= 5 else '不满足'}标准口径"
                                   f"（≥5）⇒ 在力带 = {IN_FORCE_BAND_BPC} bpc，判决 = max|配对 Δ| vs 该带"),
                   "not_in_force_descriptive": "single_arm_range_bpc（单臂跨窗口极差）是另一口径的"
                                              "描述性读数，不参与判决、不与 0.014 混用。"}
    say(f"[配对] 带={IN_FORCE_BAND_BPC} 窗口={paired_seeds} → "
        + "；".join(f"{a}: max|Δ|={paired[a]['max_abs_delta_bpc']:.6f} "
                    f"({paired[a]['verdict']})" for a in paired if a != "fp16"))

    # ---------- ⑤ 一句话答案与前沿 ----------
    frontier = []
    for a, p in by.items():
        dep = p["byte_account"]["scope_linear_all"]
        frontier.append({"arm": a, "bytes_per_param": round(dep["bytes_per_param"], 6),
                         "to_fp16_multiple": round(dep["to_fp16_multiple"], 6),
                         "reduction_vs_fp16": round(dep["reduction_vs_fp16"], 6),
                         "bpc": p["bpc"], "delta_bpc_vs_fp16": round(p["bpc"] - by["fp16"]["bpc"], 6),
                         "paired_max_abs_delta_bpc": paired[a]["max_abs_delta_bpc"],
                         "verdict_vs_band": paired[a]["verdict"],
                         "recall_copy": (p["recall_copy"]["accuracy"]
                                         if "recall_copy" in p else None),
                         "recall_copy_at_floor": (p["recall_copy"]["at_floor"]
                                                  if "recall_copy" in p else None)})
    frontier.sort(key=lambda r: r["bytes_per_param"])
    rec["frontier"] = frontier

    def dominates(a, b):
        """a 是否在「字节 + 质量」上支配 b：字节 ≤ 且质量不差（带内）。"""
        ra, rb = by[a], by[b]
        ba = ra["byte_account"]["scope_linear_all"]["bytes_per_param"]
        bb = rb["byte_account"]["scope_linear_all"]["bytes_per_param"]
        quality_not_worse = paired[a]["max_abs_delta_bpc"] <= IN_FORCE_BAND_BPC \
            and abs(paired[b]["max_abs_delta_bpc"]) <= IN_FORCE_BAND_BPC
        return ba <= bb and quality_not_worse

    rec["frontier_analysis"] = {
        "int8_per_channel_vs_fp16": {
            "bytes_ratio": round(by["int8_per_channel"]["byte_account"]["scope_linear_all"]
                                 ["to_fp16_multiple"], 6),
            "delta_bpc": round(by["int8_per_channel"]["bpc"] - by["fp16"]["bpc"], 6),
            "paired_max_abs_delta_bpc": paired["int8_per_channel"]["max_abs_delta_bpc"],
            "verdict_vs_band": paired["int8_per_channel"]["verdict"],
            "dominates_fp16": bool(dominates("int8_per_channel", "fp16")),
            "recall_delta_copy": round(by["int8_per_channel"]["recall_copy"]["accuracy"]
                                       - by["fp16"]["recall_copy"]["accuracy"], 6),
        },
        "int8_group32_vs_fp16": {
            "bytes_ratio": round(by["int8_group32"]["byte_account"]["scope_linear_all"]
                                 ["to_fp16_multiple"], 6),
            "delta_bpc": round(by["int8_group32"]["bpc"] - by["fp16"]["bpc"], 6),
            "paired_max_abs_delta_bpc": paired["int8_group32"]["max_abs_delta_bpc"],
            "verdict_vs_band": paired["int8_group32"]["verdict"],
            "dominates_fp16": bool(dominates("int8_group32", "fp16")),
            "recall_delta_copy": round(by["int8_group32"]["recall_copy"]["accuracy"]
                                       - by["fp16"]["recall_copy"]["accuracy"], 6),
        },
        "int8_per_channel_vs_int8_group32": {
            "bytes_ratio_group_over_pc": round(
                by["int8_group32"]["byte_account"]["scope_linear_all"]["bytes_per_param"]
                / by["int8_per_channel"]["byte_account"]["scope_linear_all"]["bytes_per_param"], 6),
            "note": "本模型 in_features ∈ {256,1536} 远大于 32 ⇒ per_channel（每行一把标尺）字节更省；"
                    "谁占优看质量是否可分。",
            "dominates_group32": bool(dominates("int8_per_channel", "int8_group32")),
        },
        "int4_group32_vs_fp16": {
            "bytes_ratio": round(by["int4_group32"]["byte_account"]["scope_linear_all"]
                                 ["to_fp16_multiple"], 6),
            "delta_bpc": round(by["int4_group32"]["bpc"] - by["fp16"]["bpc"], 6),
            "paired_max_abs_delta_bpc": paired["int4_group32"]["max_abs_delta_bpc"],
            "verdict_vs_band": paired["int4_group32"]["verdict"],
            "dominates_fp16": bool(dominates("int4_group32", "fp16")),
            "recall_delta_copy": round(by["int4_group32"]["recall_copy"]["accuracy"]
                                       - by["fp16"]["recall_copy"]["accuracy"], 6),
        },
        "int4_pc_not_measured": "int4 per_channel（解析 0.5 + 2/n_in ≈ 0.508）本批**未测** ⇒ "
                                "只报解析，不进结论（下一步最小实验）。",
    }
    say(f"[前沿] " + "；".join(f"{r['arm']}: {r['bytes_per_param']:.6f} B/param "
                               f"Δbpc={r['delta_bpc_vs_fp16']:+.6f} {r['verdict_vs_band']}"
                               for r in frontier))

    # ---------- ⑥ 变异电池读数（由 selftest_wq.py 落 mutation_report.json） ----------
    mut_path = os.path.join(HERE, "mutation_report.json")
    if os.path.exists(mut_path):
        mut = json.load(open(mut_path))
        ms = mut.get("mutations", [])
        rec["mutation"] = {"n_mutations": len(ms),
                           "n_caught_red": sum(1 for m in ms if m.get("caught_red")),
                           "all_caught": bool(mut.get("all_caught")),
                           "report": os.path.relpath(mut_path, ROOT),
                           "note": "变异电池是 CPU 单测（不进 GPU 判据）；此处只引用其落盘读数。"}
        rec["gates"]["G5_mutation"] = {"pass": bool(mut.get("all_caught")),
                                        "n_mutations": len(ms),
                                        "n_caught_red": rec["mutation"]["n_caught_red"]}
    else:                                                       # pragma: no cover
        rec["mutation"] = {"n_mutations": 0, "n_caught_red": 0, "all_caught": None,
                           "note": "未找到 mutation_report.json（先跑 python3 selftest_wq.py）"}
        rec["gates"]["G5_mutation"] = {"pass": False, "n_mutations": 0, "n_caught_red": 0}

    # ---------- ⑦ 落盘 ----------
    rec["memory_convention"] = {
        "analytic": "points[*].byte_account.scope_linear_all.bytes_per_param = 判据横轴（值 + 标尺）",
        "measured_gpu_peak_mib": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                  if uses_gpu else None),
        "isolation_note": "评测实现把假量化权重现算成 fp32 再前向 ⇒ 实测峰值不随档下降，只报解析列。",
    }
    wall = round(time.time() - t0, 2)
    rec["cost"] = {"device": dev, "wall_seconds": wall, "wall_minutes": round(wall / 60, 3),
                   "lock_wait_seconds": (None if not uses_gpu else lock.ev.get("wait_seconds")),
                   "gpu_work_only": True,
                   "contention_note": "wall 从取锁成功后开始计（不含等锁）；共享卡被外部占用时同代码"
                                      "步时会更长。墙钟**只作过程读数，不进判据**。",
                   "peak_mem_mib": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                    if uses_gpu else None),
                   "smoke": bool(args.smoke)}
    codes = ["axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/run.py",
             "axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/wq_arms.py",
             "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/run.py",
             "axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart/run.py",
             "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/causal_check.py",
             "tools/evalkit/retention.py"]
    rec["code"] = {c: b21.b19.sha256_file(os.path.join(ROOT, c))[:16] for c in codes}

    rec["caliber"] = {
        "device": dev,
        "is_cpu": bool(not uses_gpu),
        "gpu_lock_applicable": bool(uses_gpu),
        "judgement_caliber": rec["band"]["caliber"],
        "cross_device_anchor": {
            "anchor_bpc": anchor,
            "anchor_device": "cuda",
            "this_run_fp16_bpc": by["fp16"]["bpc"],
            "same_device": bool(uses_gpu),
            "in_force": anchor_in_force,
            "note": "跨设备（CPU vs GPU）不构成判据：浮点归约顺序不同，末位可能不同。判据 = **同设备、"
                    "同窗口序**的配对 Δbpc。"},
        "note": (f"本批同一进程、同一数据流、同一窗口序下把四个臂全跑在 **{dev}** 上 ⇒ 臂间比较"
                 f"同设备同口径。『干净 bpc』的跨臂比较与配对 Δ 都在该设备内完成，"
                 f"batch() 的取数只依赖 seed/窗口 ⇒ 设备不影响配对的公平性。"
                 + ("" if not uses_gpu else ""))}

    # GPU 工作已完成：先补全 §20 记账，再释放并落盘。
    if uses_gpu:
        lock.ev["held_whole_run"] = bool(wall < 600)
        lock.ev["peak_mem_mib_this_run"] = rec["cost"]["peak_mem_mib"]
        lock.ev["free_delta_note"] = ("free_mib_before/after 是 nvidia-smi 的**全卡**空闲读数，"
                                      "含**外部占用**；两者之差 ≠ 本 run 足迹"
                                      "（本 run 足迹看 peak_mem_mib_this_run）。")
        if wall >= 600:                                         # pragma: no cover
            lock.ev["note"] = (f"整程持锁 {wall:.0f}s ≥ 600s，超出 SCHEMA §20.1 的 <10 min 例外窗口——"
                               f"如实记录（下次须按臂释放）。")
        lock.release()
        rec["gpu_lock"] = dict(lock.ev)
    else:                                                       # pragma: no cover
        rec["gpu_lock"] = {"skipped": "纯 CPU 运行，未取 GPU 锁（SCHEMA §9）",
                           "lock_path": GPU_LOCK_DIR,
                           "caliber_note": f"device={dev} ⇒ 纯 CPU 工作，不与 GPU 作业互斥，"
                                           f"SCHEMA §20 的取锁/释放记账不适用（不是漏记）；"
                                           f"§20 的五字段（holder_pid / wait_seconds / "
                                           f"free_mib_before / free_mib_after / release_ok）"
                                           f"在 CPU 口径下无定义。",
                           "note": "本 run 的 device=cpu ⇒ 不与 GPU 作业互斥；holder_pid / free_mib_* "
                                   "不适用。"}

    # 先落 results.json（判据与数字的唯一家），再画表：表渲染失败不得丢掉结果。
    rec["partial"] = {"partial": False, "arms_completed": [q["arm"] for q in points],
                      "arms_pending": [],
                      "note": "四个臂全跑完后的完整结果；逐臂半成品见 results.partial.json。"}
    dump_results(rec)
    say(f"[落盘] {os.path.join(args.out_dir, 'results.json')}")
    if not args.smoke:
        write_table(rec, args.out_dir)
    say(f"[完成] {rec['cost']['wall_minutes']} min，峰值 {rec['cost']['peak_mem_mib']} MiB，"
        f"{len(points)} 臂 → {args.out_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
