"""2631-carry-reseed · 语义层：六臂定义 + 聚类统计 + 跨种子合并判据。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。本模块只做三件事：

1. **臂定义**：`carry_rule` × `seed` 的六个臂（唯一变量 = 写入规则与种子）；
2. **聚类统计**（复核 B2 的修法）：同一序列里的 Q 个查询共享同一份 KV / 记忆状态，
   **不是**独立二项样本 ⇒ 主 CI 用**按序列聚类的 bootstrap**（重采样单位 = 一条序列的
   全部查询），臂间差用**配对聚类 bootstrap**（两臂重采样同一组序列索引）；
   每条查询的 Wilson CI 只作**对照**并显式标注「按独立样本算 ⇒ 偏窄」；
3. **跨种子合并判据**（判据①②③④）：逐种子效应 + 极差 + 方向一致性 + 预登记阈值。

**不改任何共享文件**（`tools/**`、`tests/**`、`pathlm/**`、`train_depth_ar.py` 均只读导入）。
"""

from __future__ import annotations

import random

from tools.evalkit.carry_channel import two_proportion, wilson_ci      # noqa: F401

#: 六臂：唯一变量 = carry_rule（none/plain/delta）× seed（0/1）
ARMS: list[dict] = [
    {"name": "none_s0", "rule": "none", "seed": 0},
    {"name": "plain_s0", "rule": "plain", "seed": 0},
    {"name": "delta_s0", "rule": "delta", "seed": 0},
    {"name": "none_s1", "rule": "none", "seed": 1},
    {"name": "plain_s1", "rule": "plain", "seed": 1},
    {"name": "delta_s1", "rule": "delta", "seed": 1},
]
SEEDS: tuple[int, ...] = (0, 1)
RULES: tuple[str, ...] = ("none", "plain", "delta")

#: 预登记阈值（判据①③，沿用 2618 判据②③的 5pp / 3pp）
DELTA_VS_NONE_MIN_PP = 5.0
DELTA_VS_PLAIN_MIN_PP = 3.0
#: 毁源对照的操作化阈值（判据④）：主臂聚类 CI 下界 > 4×chance 才算「高」
CORRUPT_LEAK_FACTOR = 4.0
#: 仓库预登记的准确率实用阈值（C13；**是预设阈值，不是已测方差**）
BAND_PP = 1.4


def arm_name(rule: str, seed: int) -> str:
    """规则 + 种子 → 臂名（`delta_s1`）。"""
    if rule not in RULES:
        raise ValueError(f"未知写入规则 {rule!r}，可选 {RULES}")
    if seed not in SEEDS:
        raise ValueError(f"未知种子 {seed!r}，可选 {SEEDS}")
    return f"{rule}_s{seed}"


def model_config_kwargs(rule: str) -> dict:
    """`DepthARConfig` 上承载该臂的唯一差异：`carry_rule`（`none` → `None`）。"""
    if rule not in RULES:
        raise ValueError(f"未知写入规则 {rule!r}，可选 {RULES}")
    return {"carry_rule": None if rule == "none" else rule}


def train_flags(rule: str) -> list[str]:
    """`train_depth_ar.py` 的臂差异 flag（其余逐字段一致，见 README §7）。"""
    if rule not in RULES:
        raise ValueError(f"未知写入规则 {rule!r}，可选 {RULES}")
    return [] if rule == "none" else ["--carry-write", rule]


# ----------------------------------------------------------------- 聚类统计（B2）


def _percentile(sorted_vals: list[float], q: float) -> float:
    """稳健的 percentile（线性插值，索引夹紧；`q` ∈ [0,1]）。"""
    if not sorted_vals:
        raise ValueError("percentile 收到空列表")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def pooled_accuracy(cluster_hits: list[int], cluster_n: list[int]) -> float:
    """把逐簇命中/总数汇总成一个准确率（= 全部查询的命中率）。"""
    if len(cluster_hits) != len(cluster_n):
        raise ValueError("cluster_hits 与 cluster_n 长度不一致")
    n = sum(cluster_n)
    if n <= 0:
        raise ValueError("总查询数为 0，无法汇总")
    return sum(cluster_hits) / n


def clustered_bootstrap_ci(cluster_hits: list[int], cluster_n: list[int],
                           n_boot: int = 2000, seed: int = 2631,
                           alpha: float = 0.05) -> list[float]:
    """**按序列聚类的 bootstrap** percentile 95% CI（复核 B2 的主 CI）。

    重采样单位 = 一条序列（连同它的全部查询）；同一序列内的查询共享 KV/记忆状态，
    因此把它们当独立样本会**低估**方差。单簇时 bootstrap 退化 ⇒ 返回点估计。
    """
    if len(cluster_hits) != len(cluster_n):
        raise ValueError("cluster_hits 与 cluster_n 长度不一致")
    point = pooled_accuracy(cluster_hits, cluster_n)
    m = len(cluster_hits)
    if m < 2:
        return [point, point]
    rng = random.Random(seed)
    accs: list[float] = []
    for _ in range(n_boot):
        idx = rng.choices(range(m), k=m)                 # C 层抽样（比逐次 randrange 快）
        kh = sum(map(cluster_hits.__getitem__, idx))
        nn = sum(map(cluster_n.__getitem__, idx))
        accs.append(kh / nn)
    accs.sort()
    return [round(_percentile(accs, alpha / 2), 6),
            round(_percentile(accs, 1 - alpha / 2), 6)]


def paired_clustered_bootstrap_diff_ci(hits_a: list[int], n_a: list[int],
                                       hits_b: list[int], n_b: list[int],
                                       n_boot: int = 2000, seed: int = 2631,
                                       alpha: float = 0.05) -> list[float]:
    """**配对聚类 bootstrap**：Δ = acc_b − acc_a 的 95% CI（重采样同一组序列索引）。

    配对成立的前提：两臂在**同一批序列 / 同一批提问位置**上评测（本实验的生成器
    seed 与模型 seed 无关 ⇒ 成立，见 README §7）。
    """
    if not (len(hits_a) == len(n_a) == len(hits_b) == len(n_b)):
        raise ValueError("配对 bootstrap 要求四列等长")
    m = len(hits_a)
    if m < 2:
        raise ValueError("配对 bootstrap 至少需要 2 条序列")
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.choices(range(m), k=m)                 # 两臂共用同一组索引 ⇒ 配对
        ka = sum(map(hits_a.__getitem__, idx))
        na = sum(map(n_a.__getitem__, idx))
        kb = sum(map(hits_b.__getitem__, idx))
        nb = sum(map(n_b.__getitem__, idx))
        diffs.append(kb / nb - ka / na)
    diffs.sort()
    return [round(_percentile(diffs, alpha / 2), 6),
            round(_percentile(diffs, 1 - alpha / 2), 6)]


def arm_reading(cluster_hits: list[int], cluster_n: list[int], *,
                n_boot: int = 2000, seed: int = 2631) -> dict:
    """一臂一条腿的读数：点估计 + 聚类 bootstrap CI + 对照用的 Wilson CI。"""
    k, n = sum(cluster_hits), sum(cluster_n)
    return {"n_hits": k, "n": n,
            # 逐簇原始计数：`effect()`/配对 bootstrap 的输入（报告落盘时可剥掉）
            "cluster_hits": list(cluster_hits), "cluster_n": list(cluster_n),
            "accuracy": round(pooled_accuracy(cluster_hits, cluster_n), 6),
            "clustered_ci": clustered_bootstrap_ci(cluster_hits, cluster_n,
                                                   n_boot=n_boot, seed=seed),
            "wilson_ci_naive": wilson_ci(k, n),
            "n_clusters": len(cluster_hits),
            "wilson_naive_note": ("按每条查询独立二项样本算 ⇒ 偏窄（复核 B2）；"
                                  "主 CI 用 clustered_ci")}


def effect(base: dict, target: dict, *, n_boot: int = 2000,
           seed: int = 2631) -> dict:
    """目标 − 基准的效应（pp）：配对聚类 bootstrap CI + 两比例检验（对照）。"""
    tp = two_proportion(base["n_hits"], base["n"], target["n_hits"], target["n"])
    paired = paired_clustered_bootstrap_diff_ci(
        base["cluster_hits"], base["cluster_n"],
        target["cluster_hits"], target["cluster_n"], n_boot=n_boot, seed=seed)
    return {"effect_pp": round(100 * (target["accuracy"] - base["accuracy"]), 3),
            "paired_boot_ci_pp": [round(100 * paired[0], 3), round(100 * paired[1], 3)],
            "two_proportion_delta_pp": tp["delta_pp"],
            "two_proportion_diff_ci_pp": [round(100 * v, 3)
                                          for v in tp["diff_ci_95"]],
            "two_proportion_ci_overlap": tp["ci_overlap"],
            "base_accuracy": base["accuracy"], "target_accuracy": target["accuracy"],
            "base_n_hits": base["n_hits"], "base_n": base["n"],
            "target_n_hits": target["n_hits"], "target_n": target["n"]}


# ----------------------------------------------------------------- 跨种子合并判据


def pp_range(values: list[float]) -> float:
    """极差（max − min，pp）。**不是**方差上界，故只作描述量报出。"""
    if not values:
        raise ValueError("pp_range 收到空列表")
    return round(max(values) - min(values), 3)


def pp_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("pp_mean 收到空列表")
    return round(sum(values) / len(values), 3)


def pp_sd(values: list[float]) -> float:
    """**样本** sd（除 n−1）——这是训练侧种子散布的估计量，不是总体 sd。"""
    n = len(values)
    if n < 2:
        raise ValueError(f"样本 sd 至少需要 2 个观测，收到 {n}")
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return round(var ** 0.5, 3)


def sign_consistent(values: list[float]) -> bool:
    """方向一致性：全部严格同号（存在 0 视为不一致）。"""
    if not values:
        raise ValueError("sign_consistent 收到空列表")
    return all(v > 0 for v in values) or all(v < 0 for v in values)


def effect_stable(per_seed: list[dict], min_pp: float) -> dict:
    """跨种子「稳健」= 每个种子都过 `点估计 ≥ min_pp` **且**配对聚类 CI 下界 > 0。"""
    if not per_seed:
        raise ValueError("effect_stable 收到空列表")
    rows = [{"seed": int(r["seed"]), "effect_pp": float(r["effect_pp"]),
             "ci_lo_pp": float(r["paired_boot_ci_pp"][0]),
             "passes": bool(float(r["effect_pp"]) >= min_pp
                            and float(r["paired_boot_ci_pp"][0]) > 0)}
            for r in per_seed]
    effects = [r["effect_pp"] for r in rows]
    return {"min_pp_threshold": min_pp, "per_seed": rows,
            "all_seeds_pass": all(r["passes"] for r in rows),
            "n_seeds": len(rows),
            "range_pp": pp_range(effects),
            "mean_pp": pp_mean(effects),
            "sd_pp": pp_sd(effects) if len(effects) >= 2 else None,
            "sign_consistent": sign_consistent(effects),
            "stable": bool(len(rows) >= 2 and all(r["passes"] for r in rows))}


def corrupt_control_reading(arm_rows: dict, chance: float,
                            factor: float = CORRUPT_LEAK_FACTOR) -> dict:
    """判据④：主臂「高」（聚类 CI 下界 > factor×chance）时，毁源臂必须掉回 ≤ factor×chance。

    返回逐臂判定；`ok=False` 的臂读数**作废**。
    """
    bound = factor * chance
    rows, void = [], []
    for name, r in arm_rows.items():
        main_lo = float(r["main"]["clustered_ci"][0])
        cor_lo = float(r["corrupt"]["clustered_ci"][0])
        high = main_lo > bound
        ok = (not high) or (cor_lo <= bound)
        rows.append({"arm": name, "main_ci_lo": main_lo, "corrupt_ci_lo": cor_lo,
                     "main_is_high": high, "ok": bool(ok)})
        if not ok:
            void.append(name)
    return {"chance": chance, "bound": bound, "factor": factor,
            "per_arm": rows, "void_arms": void,
            "ok": not void}
