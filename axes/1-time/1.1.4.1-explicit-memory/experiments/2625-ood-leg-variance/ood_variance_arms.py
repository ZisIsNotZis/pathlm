"""2625-ood-leg-variance · 臂定义、评测腿口径与方差统计（纯逻辑，可失败单测的靶子）。

契约：同目录 `README.md`（§4 判据线逐字抄录 W2）。本模块只放**纯逻辑**
（无 GPU、无 IO），好让 `tests/test_ood_leg_variance.py` 把下面几件事钉成可失败单测，
并被 `mutate.py` 逐处抓红：

1. **同一个配置**（`GQA2 + carry(delta) + window16`）只换 `seed`：seed∈{0(reuse),1,2}
   共 n=3（W2 契约）；臂之间**除 seed 外逐字段相同**；
2. 该配置在引擎里落的块类 = `ComposableBlock`（GQA × carry，ENG1 后可组合）；
3. **评测同一批位置**：所有臂在同一条腿上用**同一组数据生成器 seed**（与臂无关）；
4. 方差统计（极差 / 均值 ± sd）与预登记判据（极差 ≥ 3pp ⇒ 降级为辅助腿）。

臂（唯一变量 = `seed`；`config_flags` 是**常量**）：

| 臂 | seed | 权重来源 | 角色 |
|---|---|---|---|
| `gqa_carry_win_s0`    | 0 | 复用 Z3（2624）`carry_gqa` | 已有 seed=0 |
| `gqa_carry_win_s1`    | 1 | **本实验训练**（6k） | 新点 |
| `gqa_carry_win_s2`    | 2 | **本实验训练**（6k） | 新点 |
| `y1_gqa_carry_win_s0` | 0 | 复用 Y1（2623）`gqa_carry_window` | **参照**（第二个同配置 seed=0 run，不并入 n=3） |

判据 ② 的阈值 = 结论效应 3 pp（W2 明文「如 ≥3pp」），
`downgrade_to_auxiliary(range_pp)` 只在**极差**上判定（不是 sd、不是 CI 宽度）。
"""

from __future__ import annotations

#: 本实验的唯一配置（**常量**，不是变量）——与 Z3 的 `carry_gqa` / Y1 的
#: `gqa_carry_window` 逐字段相同：GQA(2 KV 头) + carry(delta 写) + window(16, anchors 0)。
CONFIG_FLAGS = {"n_kv_heads": 2, "carry_rule": "delta",
                "train_window": 16, "train_anchors": 0}

#: 训练超参（三臂逐字段相同，只差 seed）。
TRAIN_HYPER = {"d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
               "mlp_mult": 6, "steps": 6000, "batch_size": 32, "lr": 1e-3,
               "synth_task": "assoc-mqar", "synth_pairs": 8, "synth_queries": 8,
               "synth_gap": 14}

#: 两条**本实验训练**的臂（seed=1/2）；权重运行时才在 `train/` 下生成。
TRAINED_ARMS = (
    {"name": "gqa_carry_win_s1", "seed": 1, "flags": CONFIG_FLAGS},
    {"name": "gqa_carry_win_s2", "seed": 2, "flags": CONFIG_FLAGS},
)
TRAINED_ARM_NAMES = tuple(a["name"] for a in TRAINED_ARMS)

#: 两条**复用**的臂（各自的既有 checkpoint，runtime sha256 断言）。
REUSED_ARMS = (
    {"name": "gqa_carry_win_s0", "seed": 0, "flags": CONFIG_FLAGS,
     "source_experiment": "2624-carry-under-compression-quality",
     "source_arm": "carry_gqa",
     "path": ("axes/1-time/1.1.4.1-explicit-memory/experiments/"
              "2624-carry-under-compression-quality/train/carry_gqa/model.pt"),
     "sha256": "02c390d8b1cb41a3f8c64923b45dcadeadcddc60a18b78477e533ad4020b0b09"},
    {"name": "y1_gqa_carry_win_s0", "seed": 0, "flags": CONFIG_FLAGS,
     "source_experiment": "2623-composable-arms-13m",
     "source_arm": "gqa_carry_window",
     "path": ("axes/1-time/1.1.4.1-explicit-memory/experiments/"
              "2623-composable-arms-13m/train/gqa_carry_window/model.pt"),
     "sha256": "48c4e39ec058b8c509de88744e32f4169f993eb4f2680e6b4080b1b410f9c8a6"},
)
REUSED_ARM_NAMES = tuple(a["name"] for a in REUSED_ARMS)

#: **n=3 主集**（W2 契约：seed=0 已有 + seed=1/2 新跑）。
PRIMARY_ARMS = ("gqa_carry_win_s0", "gqa_carry_win_s1", "gqa_carry_win_s2")
#: **参照集**（第二个同配置 seed=0 run；不并入 n=3，只作跨会话参照）。
REFERENCE_ARMS = ("y1_gqa_carry_win_s0",)
#: 全部被评测的臂（主集 + 参照）。
ALL_ARMS = PRIMARY_ARMS + REFERENCE_ARMS

#: 预登记判据 ② 的结论效应阈值（pp）。
EFFECT_THRESHOLD_PP = 3.0
#: 本仓方差带（`harness/methodology.md`）；不足 1.4pp 的差不可排序。
VARIANCE_BAND_PP = 1.4

ALL_ARMS_SPECS = TRAINED_ARMS + REUSED_ARMS


def arm(name: str) -> dict:
    for a in ALL_ARMS_SPECS:
        if a["name"] == name:
            return a
    raise KeyError(f"未知臂 {name!r}；期望 {ALL_ARMS}")


def model_config_kwargs(spec: dict) -> dict:
    """臂 → `DepthARConfig` 关键字（本实验 = 常量 `CONFIG_FLAGS`；seed 不进模型构造）。"""
    return dict(spec["flags"])


def train_seed(spec: dict) -> int:
    """该臂的训练 seed（`--seed`）；模型初始化与训练数据流都由它决定。"""
    return int(spec["seed"])


def expected_block_class(spec: dict, n_heads: int = 8) -> str:
    """该配置在引擎里应落的块类（`DepthARModel.__init__` 的选择规则镜像）。

    GQA(`n_kv_heads<n_heads`) × carry(`carry_rule`) 同时开 ⇒ `ComposableBlock`。
    """
    f = spec["flags"]
    n_kv = f.get("n_kv_heads")
    gqa = n_kv is not None and n_kv < n_heads
    carry = f.get("carry_rule") is not None
    if gqa and carry:
        return "ComposableBlock"
    if gqa:
        return "GQABlock"
    if carry:
        return "CarryChannelBlock"
    return "Block"


def primary_names() -> tuple[str, ...]:
    return PRIMARY_ARMS


# --------------------------------------------------------------- 评测批位置（同批）

#: 与 2624/2623 同一件仪器的数据生成器 seed 基准（`quant_carry` 的默认值）。
SINGLE_SEED_BASE = 1000     # assoc_recall_batch（判决腿，单查询）
MULTI_SEED_BASE = 2000      # assoc_recall_multi_batch（训练型任务，多查询）
VAR_SEED_BASE = 5000        # var_recall_batch（距离曲线）


def single_batch_seeds(n_batches: int, K: int = 8) -> list[int]:
    """判决腿（单查询）每批的生成器 seed —— 与臂无关（同批位置）。"""
    return [SINGLE_SEED_BASE + 7 * K + i for i in range(n_batches)]


def multi_batch_seeds(n_batches: int, K: int = 8, Q: int = 8) -> list[int]:
    """训练型任务（多查询）每批的生成器 seed —— 与臂无关（同批位置）。"""
    return [MULTI_SEED_BASE + 13 * K + Q + i for i in range(n_batches)]


def var_batch_seeds(n_batches: int) -> list[int]:
    """距离曲线每批的生成器 seed —— 与臂无关。"""
    return [VAR_SEED_BASE + i for i in range(n_batches)]


def eval_batch_seeds(leg: str, n_batches: int, **kw) -> list[int]:
    """按腿取「同批位置」的生成器 seed 列表（所有臂共用）。"""
    if leg == "single":
        return single_batch_seeds(n_batches, **kw)
    if leg == "multi":
        return multi_batch_seeds(n_batches, **kw)
    if leg == "var":
        return var_batch_seeds(n_batches)
    raise KeyError(f"未知腿 {leg!r}；期望 'single' / 'multi' / 'var'")


# --------------------------------------------------------------- 方差统计与判据

def range_pp(accs) -> float:
    """极差（percentage points）：`100×(max−min)`。不足两点返回 0。"""
    vals = [float(a) for a in accs]
    if len(vals) < 2:
        return 0.0
    return round(100.0 * (max(vals) - min(vals)), 3)


def mean_pp(accs) -> float:
    vals = [float(a) for a in accs]
    return round(100.0 * sum(vals) / len(vals), 3)


def sd_pp(accs) -> float:
    """**样本**标准差（n−1；n<2 时 0），单位 pp。"""
    vals = [float(a) for a in accs]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return round(100.0 * var ** 0.5, 3)


def downgrade_to_auxiliary(range_pp_value: float,
                           threshold_pp: float = EFFECT_THRESHOLD_PP) -> bool:
    """判据 ②：该腿同配置重跑极差 ≥ 结论效应 ⇒ 降级为辅助腿（§17.18）。"""
    return float(range_pp_value) >= float(threshold_pp)
