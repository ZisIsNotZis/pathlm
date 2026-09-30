"""2616 · 线性注意力 vs 全注意力（A6 混合算子续）· 13M · depth-ar/dense。

单因素（ticket 33 契约 R，判据线预先写下未改）：**唯一变量 = 注意力算子**。
两个臂都在同一 stock decoder-only transformer 上（`--dense`：无 carry 门、无随机
退出、仅最终深度 CE、tied 读出），同数据、同 token 预算、同步数、同 seed、同 lr、
同 batch，唯一差异 = 注意力算子：

- `full`  （基线）全因果 softmax 注意力（stock `Block` + SDPA）；
- `linear` 归一化**线性核**注意力 `out_i = Σ_{j≤i}(φ(q_i)·φ(k_j))v_j / Σ φ(q_i)·φ(k_j)`，
  φ = elu+1；用累积状态实现 ⇒ 每层时间/显存对 T **线性**（无 [T,T] 分数矩阵、
  无全量 KV 缓存）。训练实装 = 分块并行形式（块间累积状态 + 块内显式因果打分）。

Δ 符号：契约 R 的判据线原文「|Δbpc| < 0.014 ⇒ 同质量；Δ ≥ +0.014 ⇒ 线性化有害」
含两种可能的符号方向（SCHEMA §17.3 要求全篇统一）。本实验**两个符号都报**、
不做任何取舍：`delta_bpc_schema = bpc_full − bpc_linear`（SCHEMA §17.3 规范：
正 = 线性更好）与 `delta_bpc_contract = bpc_linear − bpc_full`（契约 R 句面读法：
正 = 线性更差）。判据线逐字见 README.md §0/§4。

用量（后台运行 + ≤110s 轮询日志）：
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py --steps 12000 --seed 0 > run_log.txt 2>&1 &
    tail -f run_log.txt
"""

import argparse, hashlib, json, os, statistics, subprocess, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, Block, LinearAttnBlock,
                             depth_curve, tf_acceptance, attn_linear_probe,
                             attn_linear_liveness, attention_mass_stats,
                             linear_attn_phi, linear_attn_chunked,
                             linear_attn_reference, DEFAULT_LINEAR_CHUNK)
from pathlm.data import load_enwik8_full, batch
from tools.evalkit.retention import (distance_keep_mask, recent_keep_mask,
                                     mask_stats, masked_bpc, masked_logits,
                                     phrase_needle_batch, needle_recall)

DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP, BS, LR = 256, 8, 8, 256, 4, 32, 1e-3
CHUNK = DEFAULT_LINEAR_CHUNK        # 线性注意力的训练分块（实现细节）
# 对照复用（SCHEMA §9）：ticket 31-J 的 dense 基线臂（同配置、同 seed、同步数），
# 只作**量级参考**，不替代本实验的自包含双跑。
HIST_FULL_BPC = 1.5304
HIST_FULL_R2_BPC = 1.5279
HIST_RERUN_DIVERGENCE = 0.0025

ARMS = [
    {"arm": "full", "attn_linear": False,
     "role": "基线：全因果 softmax 注意力（stock Block + SDPA）"},
    {"arm": "linear", "attn_linear": True,
     "role": "线性注意力：归一化线性核 φ(q)·φ(k)，φ=elu+1，累积状态 O(T)"},
]
RERUN_ARMS = ("linear",)     # 重跑散度：新机制（linear）的可复现性未验证过


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_base(attn_linear: bool) -> DepthARConfig:
    """契约 R 骨架：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态。"""
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True, attn_linear=attn_linear,
                         linear_chunk=CHUNK)


def run_name(arm: str, rep: int = 1, suffix: str = "") -> str:
    base = f"{arm}-12k-s0" if rep == 1 else f"{arm}-12k-s0-r{rep}"
    return base + suffix


# ------------------------------ 训练子进程 ------------------------------

def train_arm(arm: str, attn_linear: bool, steps: int, seed: int, caps_min: float,
              rep: int = 1, poll_s: int = 20, eval_batches: int = 20,
              suffix: str = "") -> dict:
    """后台启动 train_depth_ar.py（--dense [--attn-linear]），轮询日志 +
    nvidia-smi 峰值显存；超时/失败即中止并抛出。训练前清空本 run 的日志
    （train_depth_ar 是 append 语义）。"""
    name = run_name(arm, rep, suffix)
    run_dir = os.path.join(HERE, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", name,
           "--d", str(D), "--layers", str(L), "--heads", str(H),
           "--seq-len", str(SEQ), "--mlp-mult", str(MLP), "--dense",
           "--steps", str(steps), "--batch-size", str(BS), "--seed", str(seed),
           "--exit-mode", "uniform", "--eval-batches", str(eval_batches),
           "--out-root", os.path.join(HERE, "runs")]
    if attn_linear:
        cmd += ["--attn-linear", "--linear-chunk", str(CHUNK)]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {name}: training started pid={proc.pid} steps={steps} "
          f"attn_linear={attn_linear}", flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{name} training exceeded {caps_min} min cap")
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for line in q.stdout.strip().splitlines():
                pid, mem = [s.strip() for s in line.split(",")]
                if int(pid) == proc.pid:
                    peak = max(peak, float(mem) / 1024.0)
        except Exception:
            pass
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            if lines and int(json.loads(lines[-1])["step"]) % 4000 == 0:
                print(f"[run] {name} {lines[-1].strip()[:160]}", flush=True)
        time.sleep(poll_s)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} training failed rc={proc.returncode}")
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, f)
        if os.path.exists(src):
            with open(os.path.join(HERE, f"{name}-{f}"), "w") as out:
                out.write(open(src).read())
    wall = (time.time() - t0) / 60
    res = json.load(open(os.path.join(run_dir, "results.json")))
    print(f"[run] {name}: done wall={wall:.2f} min peak_child_mem={peak:.2f}GB "
          f"bpc={res['bpc']}", flush=True)
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2),
            "seconds_per_step": round(wall * 60 / steps, 5)}


# ------------------------------ 评测电池 ------------------------------

def load_arm(name: str, attn_linear: bool, vocab: int) -> DepthARModel:
    torch.manual_seed(0)
    m = DepthARModel(cfg_base(attn_linear), vocab).cuda()
    m.load_state_dict(torch.load(os.path.join(HERE, "runs", name, "model.pt"),
                                 map_location="cuda"))
    m.eval()
    return m


def run_battery(model: DepthARModel, eval_arr, eval_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=eval_batches)
    out["clean_bpc"] = out["depth_curve"][-1]["bpc"]
    out["next_token_acc"] = out["depth_curve"][-1]["acc"]
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=eval_batches)
    out["tf@4"] = next(r["accept"] for r in out["tf_acceptance"]
                       if r["depth"] == 4)
    out["attn_linear_liveness"] = attn_linear_probe(model, eval_arr,
                                                    device=DEVICE)
    return out


@torch.no_grad()
def instrument_selfcheck(model: DepthARModel, eval_arr, n_batches: int = 8,
                         batch_size: int = 16) -> dict:
    """仪器自检（口径对账的一部分）：把**精确因果掩码**显式传给评测端
    `masked_logits`（`is_causal=False` 路径）与释放版前向（训练路径）比。

    - 全注意力臂：两条路径逐位相同（2614 已实测 max|Δlogit| = 0.0）；
    - 线性臂：训练路径是 O(T) 分块实装、仪器路径是显式打分参考实现 ⇒ 只有浮点
      累加顺序差，这里量出它换算到 logits 上的真实幅度（bpc 口径的偏移）。
    """
    gen = torch.Generator().manual_seed(2)
    T = model.cfg.seq_len
    ident = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEVICE))
    max_diff, nats_a, nats_b, n_tok = 0.0, 0.0, 0.0, 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(DEVICE)
        rel = model(x, exit_depths=None)[1]["depth_logits"][-1].float()
        inst = masked_logits(model, x, ident).float()
        max_diff = max(max_diff, float((rel - inst).abs().max()))
        tgt = x[:, 1:]
        for logits, acc in ((rel, "a"), (inst, "b")):
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1), reduction="sum").item()
            if acc == "a":
                nats_a += ce
            else:
                nats_b += ce
        n_tok += tgt.numel()
    return {"n_batches": n_batches, "batch_size": batch_size,
            "max_abs_logit_diff_released_vs_instrument": round(max_diff, 8),
            "bpc_released_path": round(nats_a / n_tok / 0.6931471805599453, 6),
            "bpc_instrument_identity_mask": round(nats_b / n_tok / 0.6931471805599453, 6),
            "bpc_abs_offset": round(abs(nats_a - nats_b) / n_tok / 0.6931471805599453, 6),
            "note": ("全注意力臂两条路径逐位相同（max|Δlogit| 应为 0）；线性臂的差异"
                     "只来自「O(T) 分块 vs 显式打分」的浮点累加顺序")}


def dependency_curves(model: DepthARModel, eval_arr, n_batches: int = 24,
                      batch_size: int = 16) -> dict:
    """评测期依赖曲线（`tools/evalkit/retention.py` 同一件仪器，权重不动）：

    - **按距离的单点曲线**（`distance_keep_mask`）：删掉「与查询距离恰为 d」的键；
    - **按累计预算的曲线**（`recent_keep_mask`）：只保留最近 d 个位置。

    SCHEMA §17.11：单点曲线只能描述形状、**不得求和/外推**；「保留最近 d」这条
    是直接测的累计配置，不是单点读数的累加。两条曲线都报，Δ 用臂自身释放版 bpc
    作参照（Δ = bpc_released − bpc_masked，正 = 删得少损伤小）。
    """
    T = model.cfg.seq_len
    rel = masked_bpc(model, eval_arr, None, n_batches=n_batches,
                     batch_size=batch_size, device=DEVICE)
    rows_d, rows_r = [], []
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 192, 255):
        keep = distance_keep_mask(T, d, DEVICE)
        bpc = masked_bpc(model, eval_arr, keep, n_batches=n_batches,
                         batch_size=batch_size, device=DEVICE)
        rows_d.append({"d": d, "bpc": round(bpc, 6),
                       "delta_bpc": round(rel - bpc, 6),
                       **mask_stats(keep)})
    for d in (32, 64, 128, 192, 256):
        keep = recent_keep_mask(T, d, DEVICE)
        bpc = masked_bpc(model, eval_arr, keep, n_batches=n_batches,
                         batch_size=batch_size, device=DEVICE)
        rows_r.append({"d": d, "bpc": round(bpc, 6),
                       "delta_bpc": round(rel - bpc, 6),
                       **mask_stats(keep)})
    return {"released_bpc_same_windows": round(rel, 6),
            "n_batches": n_batches, "batch_size": batch_size,
            "distance_curve": rows_d, "recent_curve": rows_r,
            "delta_convention": "Δ = bpc_released − bpc_masked（正 = 掩码造成的损伤小）",
            "note": "单点距离曲线只描述形状，不得累加/外推；累计预算曲线是直接测的"}


def needle_leg(model: DepthARModel, eval_arr, n_real: int, mask_token: int,
               n_batches: int = 16, batch_size: int = 16, seed: int = 3,
               anchors: int = 8, phrase_len: int = 6) -> dict:
    """检索腿（SCHEMA §17.10 的「保留」一侧）：把真实文本短语针植入窗口，问
    答案行。**全可见**（`keep=None`）下两臂的召回率可直比「长程检索能力」；
    再给一条 `recent:d=64` 的抑制配置（窗口外的针还能不能被检索到）。

    对照臂：`plant=False`（记忆不存在的地板，同一 generator 序列）。
    """
    T = model.cfg.seq_len
    out = {"n_batches": n_batches, "batch_size": batch_size, "seed": seed,
           "anchors": anchors, "phrase_len": phrase_len, "configs": {}}
    for cfg_id, keep_kind in (("full_visibility", None),
                              ("recent_d64", "recent")):
        gen = torch.Generator().manual_seed(seed)
        recs, recs_floor = [], []
        for _ in range(n_batches):
            x, meta = phrase_needle_batch(batch_size, T, eval_arr, n_real,
                                          mask_token, gen, anchors=anchors,
                                          n_needles=8, phrase_len=phrase_len)
            keep = (recent_keep_mask(T, 64, DEVICE)
                    if keep_kind == "recent" else None)
            recs.extend(needle_recall(model, x, meta, keep, anchors=anchors,
                                      phrase_len=phrase_len, device=DEVICE)
                        ["per_needle"])
        # 地板（同一 generator 序列 ⇒ 与 plant=True 逐一配对；只在全可见配置跑）
        if keep_kind is None:
            gen0 = torch.Generator().manual_seed(seed)
            for _ in range(n_batches):
                x0, meta0 = phrase_needle_batch(batch_size, T, eval_arr, n_real,
                                                mask_token, gen0, anchors=anchors,
                                                n_needles=8, plant=False,
                                                phrase_len=phrase_len)
                recs_floor.extend(needle_recall(model, x0, meta0, None,
                                                anchors=anchors,
                                                phrase_len=phrase_len,
                                                device=DEVICE)["per_needle"])
        n = len(recs)
        buckets = {}
        for r in recs:
            buckets.setdefault(r["bucket"], []).append(r["hit"])
        out["configs"][cfg_id] = {
            "recall": round(sum(r["hit"] for r in recs) / n, 6),
            "n_needles": n,
            "buckets": {k: {"recall": round(sum(v) / len(v), 6), "n": len(v)}
                        for k, v in buckets.items()},
            "mean_dist": round(sum(r["dist"] for r in recs) / n, 1)}
        if recs_floor:
            out["configs"][cfg_id]["floor_no_plant"] = {
                "recall": round(sum(r["hit"] for r in recs_floor)
                                / len(recs_floor), 6),
                "n_needles": len(recs_floor),
                "note": "记忆不存在的地板（同一 generator 序列，plant=False）"}
    return out


# ---------------------- 成本口径（理论 + 实测，契约 R 并列判据） ----------------------

def flops_accounting(kind: str, T: int = SEQ, d: int = D, n_heads: int = H,
                     chunk: int = CHUNK, mlp_mult: int = MLP) -> dict:
    """每 token 每层的**乘加对**（MAC；1 MAC = 2 FLOPs）口径：

    - 全注意力：QKᵀ `T·d` + PV `T·d` = `2·T·d`（QKᵀ 的 T² 项按每 token 计就是 T·d）；
    - 线性注意力（分块 C）：状态累积 `dh²·H` + 查询×状态 `dh²·H` = `2·d·dh`，
      加块内显式打分与混合 `2·C·dh·H` = `2·C·d`，加特征映射 `2·d`；
    - 其余部件与算子无关：qkv `3d²`、out 投影 `d²`、MLP `2·mlp_mult·d²`。
    """
    dh = d // n_heads
    attn_full = 2 * T * d
    attn_linear = 2 * d * dh + 2 * chunk * d + 2 * d
    other = 3 * d * d + d * d + 2 * mlp_mult * d * d
    attn = attn_full if kind == "full" else attn_linear
    total = attn + other
    # 流式/循环形式（部署口径，无块内显式打分项）：每 token 常数
    attn_linear_streaming = 2 * d * dh + 2 * d
    return {
        "kind": kind, "T": T,
        "per_token_per_layer_macs": {
            "attn": attn, "other": other, "total": total,
            "attn_linear_streaming": (attn_linear_streaming
                                      if kind == "linear" else None),
        },
        "attn_share_of_layer_macs": round(attn / total, 4),
    }


def crossover_T(d: int = D, n_heads: int = H, chunk: int = CHUNK,
                streaming: bool = False) -> dict:
    """理论成本交叉点：全注意力 `2·T·d` MACs/层/token 与线性注意力
    `2·d·dh + 2·C·d`（或流式形式的 `2·d·dh`）相等处的 T。

    - 本训练实装（chunk=C）：`T* = dh + C`；
    - 流式/循环形式（部署，无块内项）：`T* = dh`。
    """
    dh = d // n_heads
    t_star = dh if streaming else dh + chunk
    return {"streaming": streaming, "dh": dh, "chunk": chunk, "T_star": t_star}


def kv_memory_accounting(T: int = SEQ, d: int = D, n_heads: int = H,
                         bytes_per_element: int = 2,
                         state_bytes: int = 4) -> dict:
    """**解析** KV/状态内存（单层、单序列长度 T）：

    - 全注意力流式推理：KV 缓存 `2·d` 元素/token ⇒ `2·d·T` 元素（bf16 训练/推理
      常用 2 字节）；
    - 线性注意力流式推理：每头一个 `dh×dh` 状态 + `dh` 归一化向量 ⇒
      `H·dh² + d = d·dh + d` 元素，**与 T 无关**（常数）；
    - 本**训练实装**额外物化块级状态 `(T/C)·H·dh²` 与块内分数 `T·C·H` 元素。

    注意这是解析量，**与实测显存分开报**（SCHEMA 的省内存纪律）：实测含框架开销、
    优化器状态与 kernel 选择，不会等于解析值。
    """
    dh = d // n_heads
    full_el = 2 * d * T
    lin_state_el = d * dh + d
    train_states = (T // CHUNK) * d * dh + T * CHUNK * n_heads
    return {
        "T": T, "d": d, "n_heads": n_heads, "dh": dh,
        "full_kv_cache_elements": full_el,
        "full_kv_cache_bytes_bf16": full_el * bytes_per_element,
        "linear_state_elements_constant": lin_state_el,
        "linear_state_bytes_fp32": lin_state_el * state_bytes,
        "linear_kv_reduction_frac": round(1 - lin_state_el / full_el, 4),
        "linear_kv_reduction_formula": "1 − (dh + 1) / (2T)：T 越大越接近 100%",
        "training_impl_intermediate_elements": train_states,
        "training_impl_note": ("本训练实装物化块级状态与块内分数（O(T) 但常数更大）；"
                               "真实显存以实测为准")}


def _timed(fn, warmup: int = 3, iters: int = 8) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": round(statistics.median(ts), 3),
            "min_ms": round(min(ts), 3), "n_iters": iters}


def model_step_cost(attn_linear: bool, T: int = SEQ, steps: int = 8,
                    batch_size: int = BS) -> dict:
    """实测（同一进程、同一 shape、同一 optimizer）：整模型一步 fwd+bwd 的中位
    墙钟与峰值显存（torch.cuda.max_memory_allocated）。T 可变 ⇒ 用于交叉点。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=T,
                        mlp_mult=MLP, dense=True, attn_linear=attn_linear,
                        linear_chunk=CHUNK)
    model = DepthARModel(cfg, 206).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x = torch.randint(0, 206, (batch_size, T)).cuda()
    timing = {}

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, exit_depths=None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    timing = _timed(step, warmup=3, iters=steps)
    timing["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    timing["T"] = T
    timing["batch_size"] = batch_size
    timing["attn_linear"] = attn_linear
    del model, opt
    torch.cuda.empty_cache()
    return timing


def attn_island_cost(kind: str, T: int, B: int = BS, d: int = D,
                     n_heads: int = H, iters: int = 8) -> dict:
    """注意力子层孤岛（qkv → 注意力 → out 投影）fwd+bwd 实测，同一输入、同一
    buffer、同一权重形状。`kind` ∈ {full, linear_chunked, linear_reference}：
    后两者把「O(T) 训练实装」与「显式打分参考实现」分开量。"""
    dh = d // n_heads
    torch.manual_seed(0)
    x = torch.randn(B, T, d, device=DEVICE, requires_grad=True)
    qkv = torch.nn.Linear(d, 3 * d).to(DEVICE)
    proj = torch.nn.Linear(d, d).to(DEVICE)
    shape = lambda t: t.view(B, T, n_heads, dh).transpose(1, 2)

    def step():
        q, k, v = qkv(x).chunk(3, dim=-1)
        qs, ks, vs = shape(q), shape(k), shape(v)
        if kind == "full":
            a = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        elif kind == "linear_chunked":
            a, _ = linear_attn_chunked(linear_attn_phi(qs), linear_attn_phi(ks),
                                       vs, CHUNK)
        elif kind == "linear_reference":
            a = linear_attn_reference(linear_attn_phi(qs), linear_attn_phi(ks), vs)
        else:
            raise ValueError(kind)
        y = proj(a.transpose(1, 2).reshape(B, T, d))
        loss = y.float().pow(2).mean()
        qkv.zero_grad(); proj.zero_grad()
        loss.backward()

    out = _timed(step, warmup=2, iters=iters)
    out.update({"kind": kind, "T": T, "batch_size": B,
                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
    del x, qkv, proj
    torch.cuda.empty_cache()
    return out


def cost_probe(cost_iters: int = 8, Ts=(256, 512, 1024, 2048),
               B: int = BS) -> dict:
    """契约 R 并列判据的完整口径：**理论**（注意力 MACs、KV/状态内存、交叉点 T）
    + **实测**（整模型步时/峰值显存随 T、注意力孤岛两实装）。两组数都报，不做
    「理论 vs 实测取优」的选择性汇报。"""
    theory = {"flops_full_T256": flops_accounting("full"),
              "flops_linear_T256": flops_accounting("linear"),
              "crossover_T_training_impl": crossover_T(streaming=False),
              "crossover_T_streaming": crossover_T(streaming=True),
              "kv_memory": {str(T): kv_memory_accounting(T) for T in Ts},
              "kv_memory_note": ("解析量（单层、序列长 T）：全注意力 KV 随 T 线性增长，"
                                 "线性注意力只需常数状态；实测显存另报")}
    measured = {"model_step_by_T": [], "attn_island_by_T": []}
    for T in Ts:
        for kind, flag in (("full", False), ("linear", True)):
            row = model_step_cost(flag, T=T, steps=cost_iters, batch_size=B)
            row["arm"] = kind
            measured["model_step_by_T"].append(row)
        for kind in ("full", "linear_chunked"):
            measured["attn_island_by_T"].append(
                attn_island_cost(kind, T=T, B=B, iters=cost_iters))
    # 实测交叉点：每个 T 上报 linear/full 的步时比
    by = {}
    for row in measured["model_step_by_T"]:
        by.setdefault(row["T"], {})[row["arm"]] = row
    ratios = []
    for T in Ts:
        f, l = by[T]["full"], by[T]["linear"]
        ratios.append({"T": T, "step_ms_full": f["median_ms"],
                       "step_ms_linear": l["median_ms"],
                       "linear_over_full": round(l["median_ms"] / f["median_ms"], 4),
                       "peak_gib_full": f["peak_mem_gib"],
                       "peak_gib_linear": l["peak_mem_gib"],
                       "linear_over_full_peak": round(l["peak_mem_gib"]
                                                      / f["peak_mem_gib"], 4)})
    measured["ratio_by_T"] = ratios
    measured["measured_crossover_note"] = (
        "实测交叉点 = linear_over_full < 1 的最小 T（本批实测见 results.json）；"
        "若在所测 T 范围内始终 > 1，则如实记「本实装在本卡本批内未回本」")
    measured["note"] = ("整模型步时/显存 = 训练实装（分块线性注意力）；孤岛另报显式"
                        "打分参考实现，用于把「算法本征」与「本实装开销」分开")
    return {"theory": theory, "measured": measured}


# ---------------------- 收敛对账（SCHEMA §16）+ 稳定性 ----------------------

def _slope(win: list[dict]) -> float:
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / \
        sum((x - xm) ** 2 for x in xs)


def tail_window_slope(rows: list[dict], frac: float = 0.25) -> dict:
    """末 frac 窗内 loss 对 step 的最小二乘斜率（**两臂同一方法、同一窗口比例**）。
    样本不足（日志点 < 3 或 step 跨度 = 0）时如实返回 None，不外推。"""
    i0 = int(len(rows) * (1 - frac))
    win = rows[i0:]
    if len(win) < 3 or win[-1]["step"] == win[0]["step"]:
        win = rows[-min(10, len(rows)):]
    span_ok = len(win) >= 3 and win[-1]["step"] != win[0]["step"]
    return {"window_frac": frac, "n_points": len(win),
            "step_from": win[0]["step"], "step_to": win[-1]["step"],
            "loss_from": win[0]["loss"], "loss_to": win[-1]["loss"],
            "slope_nats_per_1k_steps": (round(_slope(win) * 1000, 5)
                                        if span_ok else None),
            "slope_nats_per_1k_steps_late_half": (round(
                _slope(win[len(win) // 2:]) * 1000, 5)
                if span_ok and len(win) >= 6 else None),
            "note": None if span_ok else "日志点不足，斜率不可估（未外推）"}


def load_arm_rows(name: str) -> list[dict]:
    path = os.path.join(HERE, f"{name}-train_log.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def stability_stats(rows: list[dict], gnorm_monitor: dict | None) -> dict:
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    return {"n_logged_samples": len(rows),
            "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
            "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
            "gnorm_spikes_gt3x_median": sum(1 for v in g if v > 3 * med),
            "all_finite": all(torch.isfinite(torch.tensor(v)).item()
                              for v in g + loss),
            "loss_first": loss[0], "loss_last": loss[-1],
            "loss_max_step_increase": round(max(inc), 4),
            "loss_jumps_gt_0.25": sum(1 for v in inc if v > 0.25),
            "gnorm_monitor_all_steps": gnorm_monitor}


# ------------------------------ 主流程 ------------------------------

def main():
    os.chdir(ROOT)          # 数据/入口脚本都是仓库根相对路径（单一来源）
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caps-min", type=float, default=26.0,
                    help="单臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--poll-s", type=int, default=20)
    ap.add_argument("--skip-train", action="store_true",
                    help="复用 runs/ 下已有 checkpoint，只重算电池与判读")
    ap.add_argument("--budget-min", type=float, default=60.0,
                    help="GPU 墙钟预算（契约 R 的 2 臂 × 7–15 min + 评测 20–40 min）："
                         "超出则在下一项开跑前中止（不静默跑残臂）；砍项顺序 = 重跑臂 → "
                         "成本随 T 的点 → 检索腿批数，**不砍判据线、不砍评测批次**")
    ap.add_argument("--rerun", dest="rerun", action="store_true", default=True)
    ap.add_argument("--no-rerun", dest="rerun", action="store_false")
    ap.add_argument("--rerun-arms", default=",".join(RERUN_ARMS))
    ap.add_argument("--arms", default=None, help="只跑指定臂（逗号分隔，必须含 full）")
    ap.add_argument("--cost-iters", type=int, default=6)
    ap.add_argument("--cost-T", default="256,512,1024,2048")
    ap.add_argument("--needle-batches", type=int, default=16)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--results-file", default="results.json",
                    help="结果落盘文件名（调试用的 dry 跑写别的名字，不覆盖正式证据）")
    args = ap.parse_args()
    Ts = tuple(int(v) for v in args.cost_T.split(",") if v.strip())

    arms = list(ARMS)
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in want if a not in [x["arm"] for x in ARMS]]
        if unknown:
            raise SystemExit(f"--arms 无效（{unknown}）：可选 {[x['arm'] for x in ARMS]}")
        # 允许只跑子集（例：只补跑 linear 臂），但随后组装时必须能从 runs/ 拿到
        # 缺失臂的 checkpoint + results.json（否则下次不带 --arms 的 --skip-train
        # 组装跑会直接失败，而不是静默交一份缺臂的结果）。
        missing = [a for a in ("full", "linear") if a not in want]
        if missing and not args.skip_train:
            for name in missing:
                arm = next(x for x in ARMS if x["arm"] == name)
                rp = os.path.join(HERE, "runs", run_name(name, 1, args.suffix),
                                  "results.json")
                if not os.path.exists(rp):
                    raise SystemExit(f"--arms 只跑了 {want}，但 {name} 臂的产物不存在"
                                     f"（{rp}）；不能交缺臂结果")
        arms = [a for a in arms if a["arm"] in want]
    rerun_arms = [a.strip() for a in args.rerun_arms.split(",")
                  if a.strip()] if args.rerun else []

    t_start = time.time()
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)
    cfg_dump = {
        "experiment": "2616-linear-attention-13m",
        "ticket": "33-batch6 契约 R",
        "seed": args.seed, "steps": args.steps, "batch_size": BS, "lr": LR,
        "optimizer": "AdamW(wd=0.01) + warmup 200 步 + cosine 到 0.1×",
        "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size, "train_tokens": args.steps * BS * SEQ,
        "base": {"d_model": D, "n_layers": L, "n_heads": H, "seq_len": SEQ,
                 "mlp_mult": MLP, "dropout": 0.0, "norm_cap": 1.0,
                 "dense": True, "topology": "sequential",
                 "exit_mode": "uniform", "loss": "仅最终深度 CE（stock AR）",
                 "readout": "tied h·Eᵀ，无 final LayerNorm（共享底座偏离 1）",
                 "linear_chunk": CHUNK,
                 "linear_chunk_note": "分块大小 = 实现细节（单测钉住换 chunk 不改语义），"
                                      "不是第二个干预档"},
        "arms": [{"arm": a["arm"], "attn_linear": a["attn_linear"],
                  "role": a["role"],
                  "run": f"runs/{run_name(a['arm'], 1, args.suffix)}"}
                 for a in arms],
        "rerun_arms": rerun_arms,
        "rerun_note": "重跑散度：新机制（linear）的可复现性未验证过，重跑它",
        "variable": "attn_linear（唯一不等字段；其余逐字段一致，见 field_by_field）",
        "delta_convention": ("两个符号都报：delta_bpc_schema = bpc_full − bpc_linear"
                             "（SCHEMA §17.3：正 = 线性更好）；delta_bpc_contract = "
                             "bpc_linear − bpc_full（契约 R 句面读法：正 = 线性更差）"),
        "history_reference": {"full_arm_bpc_2614": HIST_FULL_BPC,
                              "full_arm_rerun_bpc_2614": HIST_FULL_R2_BPC,
                              "rerun_divergence_2614": HIST_RERUN_DIVERGENCE,
                              "note": "ticket 31-J 同配置 dense 基线（复用只作量级参考）"},
    }
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump, "arms": {}}

    # ---- 成本口径：理论 + 实测（与训练解耦，先测，避免 GPU 抖动污染训练日志） ----
    out["cost"] = cost_probe(cost_iters=args.cost_iters, Ts=Ts)
    print("[cost] step ms by T: "
          f"{[(r['T'], r['arm'], r['median_ms']) for r in out['cost']['measured']['model_step_by_T']]}",
          flush=True)

    def budget_ok(what: str) -> bool:
        el = (time.time() - t_start) / 60
        if not args.skip_train and el > args.budget_min:
            print(f"[budget] {el:.1f} min > {args.budget_min} min：中止在 {what} 之前",
                  flush=True)
            return False
        return True

    # ---- 两臂训练 + 电池 ----
    for i, arm in enumerate(arms):
        name = arm["arm"]
        if not budget_ok(f"arm {name}"):
            out["stopped_by_budget_before"] = [a["arm"] for a in arms[i:]]
            break
        if args.skip_train:
            rp = os.path.join(HERE, "runs", run_name(name, 1, args.suffix),
                              "results.json")
            res = json.load(open(rp))
            tr = {"results": res, "wall_min": res["wall_minutes"],
                  "peak_child_mem_gb": None, "seconds_per_step": None}
        else:
            tr = train_arm(name, arm["attn_linear"], args.steps, args.seed,
                           args.caps_min, poll_s=args.poll_s,
                           eval_batches=args.eval_batches, suffix=args.suffix)
        res = tr["results"]
        model = load_arm(run_name(name, 1, args.suffix), arm["attn_linear"],
                         vocab_size)
        t0 = time.time()
        batt = run_battery(model, eval_arr, args.eval_batches)
        out["arms"][name] = {
            "role": arm["role"], "attn_linear": arm["attn_linear"],
            "source": f"runs/{run_name(name)}",
            "params": res["params"], "train_tokens": res["train_tokens"],
            "bpc_from_train_process": res["bpc"],
            "wall_min": tr["wall_min"],
            "peak_child_mem_gb": tr["peak_child_mem_gb"],
            "seconds_per_step": tr["seconds_per_step"],
            "gnorm_monitor": res.get("gnorm_monitor"),
            "intervention": res.get("attn_linear_liveness"),
            **batt, "eval_wall_min": round((time.time() - t0) / 60, 2)}
        print(f"[arm] {name}: clean_bpc={batt['clean_bpc']} "
              f"acc={batt['next_token_acc']} wall={tr['wall_min']} min "
              f"eval={out['arms'][name]['eval_wall_min']} min", flush=True)

    # ---- 逐字段对照表（单因素） ----
    # 顺序（§8 的砍项优先级）：成本口径 → 训练两臂 + 电池 → 逐字段表（不花 GPU）
    # → 仪器自检 + 依赖曲线 + 检索腿（判据 ③）→ 重跑（预算内才跑）→ 判读。
    # 预算不够时先砍重跑，再砍成本随 T 的点/检索腿批数，**不砍判据线**。
    fields = [("d_model", D), ("n_layers", L), ("n_heads", H), ("seq_len", SEQ),
              ("mlp_mult", MLP), ("dropout", 0.0), ("norm_cap", 1.0),
              ("dense", True), ("topology", "sequential"), ("fill_kv", True),
              ("proj_fill", False), ("readout_dims", None),
              ("readout_gated", False), ("readout_linear_resid", False),
              ("exit_mode", "uniform"), ("aux_weight", 1.0),
              ("final_weight", 1.0), ("shallow_stopgrad", False),
              ("dense_aux", False), ("steps", args.steps),
              ("batch_size", BS), ("lr", LR), ("seed", args.seed),
              ("train_tokens", args.steps * BS * SEQ),
              ("vocab_size", vocab_size),
              ("linear_chunk", CHUNK)]
    rows = []
    arm_results = {}
    for arm in arms:
        arm_results[arm["arm"]] = json.load(open(os.path.join(
            HERE, "runs", run_name(arm["arm"], 1, args.suffix), "results.json")))
    for f, want in fields:
        row = {"field": f, "expected": want, "arms": {}}
        for name, res in arm_results.items():
            # 字段来源：模型 config（d_model 等）→ train（steps/seed/lr 等）→ 顶层
            got = res["config"].get(f, res.get("train", {}).get(f, res.get(f)))
            row["arms"][name] = got
        row["all_equal"] = len(set(json.dumps(v) for v in row["arms"].values())) == 1
        row["matches_expected"] = all(
            json.dumps(v) == json.dumps(want) for v in row["arms"].values())
        rows.append(row)
    var_row = {"field": "attn_linear", "expected": "唯一不等字段（干预本身）",
               "arms": {name: res["config"]["attn_linear"]
                        for name, res in arm_results.items()},
               "is_variable": True}
    var_row["all_equal"] = len(set(var_row["arms"].values())) == 1
    rows.append(var_row)
    shared = {"data_split": "first 90M train / last 10M eval, vocab over full file",
              "data": DATA, "optimizer": cfg_dump["optimizer"],
              "note": "共享字段（两臂与所有 run 相同，不构成变量；见 config.json）"}
    fixed_rows = [r for r in rows if not r.get("is_variable")]
    out["field_by_field"] = {
        "rows": rows,
        "shared_fields": shared,
        "variable_fields": ["attn_linear"],
        "field_by_field_all_equal_except_attn_linear": bool(
            all(r["all_equal"] for r in fixed_rows)
            and all(r["matches_expected"] for r in fixed_rows)
            and not var_row["all_equal"]),
        "note": ("唯一不等字段 = attn_linear（干预本身）；linear_chunk 在 config 里逐"
                 "字段相等（两臂都写 64，只有 linear 臂用它），实现层由单测证明它只是"
                 "实现细节"),
    }
    if len(arms) == len(ARMS):
        assert out["field_by_field"]["field_by_field_all_equal_except_attn_linear"], (
            "逐字段对照失败："
            + str([r["field"] for r in fixed_rows
                   if not (r["all_equal"] and r["matches_expected"])]))
    else:
        out["field_by_field"]["note_arms_subset"] = (
            f"本次只跑了 {[a['arm'] for a in arms]}：逐字段表的 all_equal 只在"
            "两臂齐全时才有意义，这里只报各字段的实测值")

    # ---- 仪器自检 + 依赖曲线 + 检索腿 + 收敛对账 + 稳定性 ----
    out["instrument_selfcheck"] = {}
    out["dependency_curves"] = {}
    out["needle_leg"] = {}
    out["convergence"] = {}
    out["stability"] = {}
    mask_token = vocab_size - 1          # 词表末位保留给针的标记（同 2615 口径）
    n_real_tokens = vocab_size - 1
    for arm in arms:
        name = arm["arm"]
        model = load_arm(run_name(name, 1, args.suffix), arm["attn_linear"],
                         vocab_size)
        if not budget_ok(f"instrument/needle for {name}"):
            out["instrument_selfcheck"][name] = {"status": "skipped_budget"}
            continue
        t0 = time.time()
        out["instrument_selfcheck"][name] = instrument_selfcheck(model, eval_arr)
        out["dependency_curves"][name] = dependency_curves(model, eval_arr)
        out["needle_leg"][name] = needle_leg(
            model, eval_arr, n_real=n_real_tokens, mask_token=mask_token,
            n_batches=args.needle_batches)
        out.setdefault("instrument_wall_min", {})[name] = round(
            (time.time() - t0) / 60, 2)
        rows_log = load_arm_rows(run_name(name, 1, args.suffix))
        res = out["arms"][name]
        out["convergence"][name] = tail_window_slope(rows_log, frac=0.25)
        out["stability"][name] = stability_stats(rows_log,
                                                 res.get("gnorm_monitor"))
        print(f"[instrument] {name}: selfcheck max|Δlogit|="
              f"{out['instrument_selfcheck'][name]['max_abs_logit_diff_released_vs_instrument']} "
              f"needle={out['needle_leg'][name]['configs']['full_visibility']['recall']} "
              f"({out['instrument_wall_min'][name]} min)", flush=True)

    # ---- 重跑散度（预算内才跑：它排在判据 ①②③ 的电池之后） ----
    if rerun_arms:
        out["reruns"] = {}
        for name in rerun_arms:
            arm = next(a for a in ARMS if a["arm"] == name)
            if not budget_ok(f"rerun {name}"):
                out["reruns"][name] = {"status": "skipped_budget"}
                continue
            r2_res = os.path.join(HERE, "runs", run_name(name, 2, args.suffix),
                                  "results.json")
            if args.skip_train and os.path.exists(r2_res):
                res = json.load(open(r2_res))
                tr = {"results": res, "wall_min": res["wall_minutes"],
                      "peak_child_mem_gb": None, "seconds_per_step": None,
                      "reused": True}
            else:
                tr = train_arm(name, arm["attn_linear"], args.steps, args.seed,
                               args.caps_min, rep=2, poll_s=args.poll_s,
                               eval_batches=args.eval_batches,
                               suffix=args.suffix)
            model = load_arm(run_name(name, 2, args.suffix), arm["attn_linear"],
                             vocab_size)
            batt = run_battery(model, eval_arr, args.eval_batches)
            base = out["arms"].get(name, {})
            out["reruns"][name] = {
                "status": "done", "source": f"runs/{run_name(name, 2)}",
                "clean_bpc": batt["clean_bpc"],
                "next_token_acc": batt["next_token_acc"],
                "wall_min": tr["wall_min"],
                "peak_child_mem_gb": tr["peak_child_mem_gb"],
                "seconds_per_step": tr["seconds_per_step"],
                "bpc_abs_diff_vs_primary":
                    (round(abs(batt["clean_bpc"] - base.get("clean_bpc",
                                                            float("nan"))), 4)
                     if base else None),
                "attn_linear_liveness": batt["attn_linear_liveness"],
            }
            print(f"[rerun] {name}: clean_bpc={batt['clean_bpc']} "
                  f"(primary {base.get('clean_bpc')})", flush=True)


    # ---- 主判据：Δ 的两种符号都报（判据线逐字见 README §4） ----
    if "full" in out["arms"] and "linear" in out["arms"]:
        f_bpc = out["arms"]["full"]["clean_bpc"]
        l_bpc = out["arms"]["linear"]["clean_bpc"]
        band = 0.014
        d_schema = round(f_bpc - l_bpc, 6)
        d_contract = round(l_bpc - f_bpc, 6)
        out["verdict"] = {
            "bpc_full": f_bpc, "bpc_linear": l_bpc,
            "delta_bpc_schema_(full−linear)": d_schema,
            "delta_bpc_contract_(linear−full)": d_contract,
            "abs_delta": round(abs(d_schema), 6), "band": band,
            "same_quality_within_band": bool(abs(d_schema) < band),
            "linear_worse_by_at_least_band": bool(d_schema <= -band),
            "linear_better_by_at_least_band": bool(d_schema >= band),
            "history_full_bpc_2614": HIST_FULL_BPC,
            "diff_vs_history": round(f_bpc - HIST_FULL_BPC, 6),
            "criterion_1_text": ("契约 R：|Δbpc| < 0.014 ⇒ 与全注意力同质量；"
                                 "Δ ≥ +0.014 ⇒ 线性化在该规模有害（记录幅度）"),
            "criterion_1_reading": ("两种符号读法都已列出；结论只依赖 |Δ| 与方向，"
                                    "不依赖选哪一种符号约定"),
        }

    # ---- 权重清单（sha256 / 体积） ----
    out["checkpoints"] = {}
    for arm in ARMS:
        p = os.path.join(HERE, "runs", run_name(arm["arm"], 1, args.suffix),
                         "model.pt")
        if os.path.exists(p):
            out["checkpoints"][run_name(arm["arm"], 1, args.suffix)] = {
                "path": os.path.relpath(p, ROOT), "sha256": sha256_file(p),
                "size_bytes": os.path.getsize(p),
                "attn_linear": arm["attn_linear"]}
    for name in rerun_arms:
        p = os.path.join(HERE, "runs", run_name(name, 2, args.suffix), "model.pt")
        if os.path.exists(p):
            out["checkpoints"][run_name(name, 2, args.suffix)] = {
                "path": os.path.relpath(p, ROOT), "sha256": sha256_file(p),
                "size_bytes": os.path.getsize(p), "attn_linear": True}
    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    with open(os.path.join(HERE, args.results_file), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("config", "cost", "arms", "field_by_field",
                                   "dependency_curves", "needle_leg",
                                   "instrument_selfcheck", "convergence",
                                   "stability")}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"[done] wall_total_min={out['wall_total_min']}", flush=True)


if __name__ == "__main__":
    main()