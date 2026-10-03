"""2624-composable-followups-13m · 三个补点的**纯逻辑层**（无 GPU、无权重 I/O）。

契约：同目录 `README.md`（§4 逐字抄录 `.scratch/41-followups/issues/01-followups.md` 的 Z2）。
本模块是**可失败单测**（`tests/test_composable_followups_13m.py`）与**变异验证**
（`mutate.py`）的靶子：把三件只有本线才有的判据逻辑钉成可机检的纯函数——

1. **① 的「容量括号 + 插值」**：GQA 少 8.748% 参数 ⇒ 用两个 **MHA** 点
   （`stock` 13,536,768 / `mha_mlp5` 11,960,832）夹住 GQA（12,352,512），
   报 MHA 的容量斜率，并把 GQA 与**插值到 GQA 参数量处的 MHA** 比。
   **这不是逐位等参数臂**（见 README §4b 的显式声明）。
2. **② 的「无泄漏 Q=7」**：多查询关联回忆在 Q = K 时，末块问的那一对是**唯一
   还没被前面问过的**（排除法伪影，2623 实测末块四臂全 100%）。Q = K−1 时
   末块只剩 2 个候选 ⇒ 伪影条件被消除。`unique_unqueried_last_block` 把它做成
   可从 batch 直接核验的谓词（**Q=8 为 True、Q=7 为 False** ⇒ 单测与变异都抓得住）。
3. **③ 的「6k vs 10k 同相对刻度」**：`gqa`/`gqa_carry` 两点各有一个 10k 版本，
   两对**逐字段同配方、只差步数**（SCHEMA §17.16）。

坐标换算（对账用）：`pick = (rows − dists) / 2`（生成器 `dists = rows − 2·pick` 的逆），
块号 `j = (rows − base − 1) / (3 + gap)`，`base = T − Q·(3 + gap)`。
"""

from __future__ import annotations

#: 方差带（`docs/methodology.md`）：Δ < 该值的准确率差不可排序（pp）。
PP_BAND = 1.4
#: bpc 方差带。
BPC_BAND = 0.014

#: 训练用的多查询数（= 2623 训练配方；也是 K）。
Q_TRAIN = 8
#: 无泄漏评测用的多查询数（Q = K − 1；见模块 docstring 第 2 条）。
Q_LEAKFREE = 7

#: 训练配方的公共字段（逐字段与 2623 一致；唯一变量 = arms 表里列出的 flags）。
COMMON_TRAIN = {"d": 256, "layers": 12, "heads": 8, "seq_len": 256, "dense": True,
                "topology": "sequential", "batch_size": 32, "lr": 1e-3, "seed": 0,
                "synth_task": "assoc-mqar", "synth_pairs": 8, "synth_queries": 8,
                "synth_gap": 14}

#: 本线的臂表。
#:   group  = 该臂属于哪一组（`y1_6k` = 2623 的既有四臂；`p1_6k` = ① 的新臂；
#:            `p3_10k` = ③ 的 10k 臂）
#:   source = 权重在哪个实验目录（`2623-composable-arms-13m` / 本目录）
#:   params = 实测参数量（构建后断言；写死在这里是为了让单测能检查「括号非退化」）
ARMS: dict[str, dict] = {
    # ---- 2623 的既有四臂（6k，seed 0，权重复用 + sha256 断言）----
    "stock": {"group": "y1_6k", "source": "2623", "steps": 6000, "mlp_mult": 6,
              "flags": {}, "params": 13_536_768},
    "gqa": {"group": "y1_6k", "source": "2623", "steps": 6000, "mlp_mult": 6,
            "flags": {"n_kv_heads": 2}, "params": 12_352_512},
    "gqa_carry": {"group": "y1_6k", "source": "2623", "steps": 6000, "mlp_mult": 6,
                  "flags": {"n_kv_heads": 2, "carry_rule": "delta"},
                  "params": 12_616_808},
    "gqa_carry_window": {"group": "y1_6k", "source": "2623", "steps": 6000,
                         "mlp_mult": 6,
                         "flags": {"n_kv_heads": 2, "carry_rule": "delta",
                                   "train_window": 16, "train_anchors": 0},
                         "params": 12_616_808},
    # ---- ① 的容量括号：MHA + 收窄 MLP（唯一变量 = mlp_mult）----
    "mha_mlp5": {"group": "p1_6k", "source": "2624", "steps": 6000, "mlp_mult": 5,
                 "flags": {}, "params": 11_960_832},
    # ---- ③ 的两个 10k 点（唯一变量 = carry_rule；配方其余逐字段同 6k 对）----
    "gqa_10k": {"group": "p3_10k", "source": "2624", "steps": 10000, "mlp_mult": 6,
                "flags": {"n_kv_heads": 2}, "params": 12_352_512},
    "gqa_carry_10k": {"group": "p3_10k", "source": "2624", "steps": 10000,
                      "mlp_mult": 6,
                      "flags": {"n_kv_heads": 2, "carry_rule": "delta"},
                      "params": 12_616_808},
}

ARM_NAMES = tuple(ARMS)
#: 基准恒为本仓的 stock 对照（SCHEMA §15/§17.14）。
BASELINE_ARM = "stock"

#: ① 的 MHA 容量括号（两个都是 MHA，只差 MLP 宽）+ 被检验的 GQA 点。
P1_BRACKET = ("mha_mlp5", "stock")
P1_TARGET = "gqa"

#: ② 的四臂（2623 的既有四臂；本点只做**评测**，不重训）。
P2_ARMS = ("stock", "gqa", "gqa_carry", "gqa_carry_window")

#: ③ 的两对（6k vs 10k，逐字段同配方只差步数）。
P3_PAIRS = (("gqa", "gqa_10k"), ("gqa_carry", "gqa_carry_10k"))


def arm_spec(name: str) -> dict:
    if name not in ARMS:
        raise KeyError(f"未知臂 {name!r}；期望 {ARM_NAMES}")
    return ARMS[name]


def model_config_kwargs(name: str) -> dict:
    """臂 → `DepthARConfig` 关键字（mlp_mult + flags；逐字段与训练时一致）。"""
    s = arm_spec(name)
    return {"mlp_mult": s["mlp_mult"], **s["flags"]}


# --------------------------------------------------------------- 坐标换算


def pick_index(rows, dists):
    """`pick = (rows − dists) / 2`（生成器 `dists = rows − 2·pick` 的逆）。"""
    diff = rows - dists
    if bool((diff % 2).any()):
        raise ValueError("rows − dists 不是偶数：该 batch 不是多查询 assoc-mqar 布局")
    return diff // 2


def block_index(rows, seq_len: int, n_queries: int, gap: int):
    """查询行 → 查询**块号** j（`base + j·(3+gap) + 1`；见 `assoc_recall_multi_batch`）。"""
    block = 3 + int(gap)
    base = int(seq_len) - int(n_queries) * block
    off = rows - base - 1
    if bool((off % block).any()):
        raise ValueError("查询行不在块起点上：布局与 (T, Q, gap) 不符")
    return off // block


# --------------------------------------------------------------- ② 无泄漏 Q=7


def unique_unqueried_last_block(rows, dists, seq_len: int, n_queries: int,
                                n_pairs: int, gap: int) -> list[bool]:
    """逐样本：末块的 pair 是否**是唯一还没被前面块问过的那一对**（= 排除法伪影）。

    条件（等价、可从 batch 直接核验）：把末块的 pair 与前面所有块的 pair 并起来
    恰好覆盖全部 `n_pairs` 对 ⇒ 末块问的是「剩下那一对」⇒ 不看前面那张表也能
    由排除法定位。`pick` 逐行互不相同（生成器保证），因此该条件等价于
    `n_queries == n_pairs`（Q=K）**并且**末块存在。

    返回逐样本 bool 列表；`True` = 该样本带伪影。
    """
    pick = pick_index(rows, dists)
    Q = int(n_queries)
    if Q < 1:
        raise ValueError(f"n_queries 必须 >= 1，收到 {Q}")
    if pick.shape[1] != Q:
        raise ValueError(f"pick 的列数 {pick.shape[1]} 与 n_queries={Q} 不符")
    blk = block_index(rows, seq_len, Q, gap)
    if int(blk.max()) != Q - 1:
        raise ValueError(f"块号最大值 {int(blk.max())} != Q−1={Q - 1}：布局异常")
    out = []
    for b in range(pick.shape[0]):
        last_col = int(blk[b].argmax())
        last = int(pick[b, last_col])
        asked = {int(v) for i, v in enumerate(pick[b].tolist()) if i != last_col}
        out.append(len(asked | {last}) == int(n_pairs))
    return out


def leak_free_leg_gap(seq_len: int, n_pairs: int, gap: int) -> int:
    """无泄漏腿（Q = K − 1）在给定 (T, K, gap) 下是否装得下（不装则报错）。"""
    Q = int(n_pairs) - 1
    need = 2 * int(n_pairs) + Q * (3 + int(gap))
    if need > int(seq_len):
        raise ValueError(f"2K + (K−1)(3+gap) = {need} > T = {seq_len}：无泄漏腿放不下")
    return Q


# --------------------------------------------------------------- ① 容量括号 + 插值


def capacity_slope(x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> float:
    """容量斜率 `dy/dx`（x = 参数量）。两点必须**非退化**（x_lo ≠ x_hi，SCHEMA §17.16）。"""
    if float(x_lo) == float(x_hi):
        raise ValueError("容量括号退化：两个 MHA 点的参数量相同 ⇒ 斜率无定义")
    return (float(y_hi) - float(y_lo)) / (float(x_hi) - float(x_lo))


def interp_at_target(x: float, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> float:
    """把 MHA 的读数线性插值到 GQA 的参数量处（**不外推**：x 必须落在括号内）。"""
    if not float(x_lo) <= float(x) <= float(x_hi):
        raise ValueError(f"目标参数量 {x} 不在括号 [{x_lo}, {x_hi}] 内 ⇒ 禁外推（§17.11）")
    return float(y_lo) + capacity_slope(x_lo, x_hi, y_lo, y_hi) * (float(x) - float(x_lo))


def residual_vs_interp(y_obs: float, y_interp: float) -> float:
    """GQA 读数 − 同参数量 MHA 插值读数（**残差**；>0 = GQA 更好）。"""
    return float(y_obs) - float(y_interp)


def bracket_span(params_lo: int, params_hi: int, params_target: int) -> dict:
    """括号的几何对账：两个 MHA 点**跨过** GQA、且 GQA 落在内部（非退化括号）。"""
    lo, hi = sorted((params_lo, params_hi))
    return {"lo": lo, "hi": hi, "target": params_target,
            "target_inside": bool(lo < params_target < hi),
            "span_params": hi - lo,
            "target_offset_frac": round((params_target - lo) / max(hi - lo, 1), 6)}


def bracket_points() -> tuple[str, str]:
    """① 的两个 MHA 点（低参数量在前），并断言括号非退化且夹住 GQA。"""
    lo, hi = P1_BRACKET
    p_lo, p_hi, p_t = (ARMS[lo]["params"], ARMS[hi]["params"], ARMS[P1_TARGET]["params"])
    info = bracket_span(p_lo, p_hi, p_t)
    if not info["target_inside"]:
        raise ValueError(f"容量括号不成立：{info}")
    return lo, hi


# --------------------------------------------------------------- ③ 同相对刻度


def scale_pair_check() -> list[dict]:
    """③ 的两对：逐字段同配方、**只差步数**，且两对用同一相对刻度（§17.16）。"""
    out = []
    for a, b in P3_PAIRS:
        sa, sb = arm_spec(a), arm_spec(b)
        if sa["flags"] != sb["flags"] or sa["mlp_mult"] != sb["mlp_mult"]:
            raise ValueError(f"{a} 与 {b} 的配方不等：{sa} vs {sb}")
        if sa["steps"] == sb["steps"]:
            raise ValueError(f"{a} 与 {b} 步数相同 ⇒ 退化点，不是两点趋势（§17.16）")
        out.append({"base_arm": a, "long_arm": b, "steps": [sa["steps"], sb["steps"]],
                    "ratio": round(sb["steps"] / sa["steps"], 4),
                    "flags": {**sa["flags"], "mlp_mult": sa["mlp_mult"]}})
    ratios = {r["ratio"] for r in out}
    if len(ratios) != 1:
        raise ValueError(f"两对的相对刻度不一致：{ratios}（§17.16 要求同一刻度）")
    return out


def scale_ratio() -> float:
    return scale_pair_check()[0]["ratio"]


# --------------------------------------------------------------- 增量 / 分量分解


def increment_pp(acc_lo: float, acc_hi: float) -> float:
    """两个读数的差（pp）：`Δ = 100·(acc_hi − acc_lo)`。正 = 后者更高。"""
    return round(100.0 * (float(acc_hi) - float(acc_lo)), 3)


def carry_increment(acc: dict, steps: int) -> dict:
    """某一 token 预算下的 carry 增量（§17.17 的两列在这里拆开）：
    `target_pp` = 判决腿（合成召回，目标量）增量；`baseline_pp` = 训练同分布
    （无泄漏 Q=7）增量（对基准质量的增量）。"""
    if steps == 6000:
        lo, hi = acc["gqa"], acc["gqa_carry"]
    elif steps == 10000:
        lo, hi = acc["gqa_10k"], acc["gqa_carry_10k"]
    else:
        raise ValueError(f"未登记的 token 预算 {steps}")
    return {"steps": steps, "no_carry": lo, "carry": hi,
            "target_pp": increment_pp(lo, hi)}


def orderable(delta_pp: float, band_pp: float = PP_BAND) -> bool:
    """可排序性（SCHEMA §17.17 / methodology）：|Δ| ≥ 方差带才算可排序。"""
    return abs(float(delta_pp)) >= float(band_pp)
