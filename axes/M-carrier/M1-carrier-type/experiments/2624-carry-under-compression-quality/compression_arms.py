"""2624-carry-under-compression-quality · 三臂定义 / 评测可见性（可失败单测的靶子）。

本模块只放**纯逻辑**（不含 GPU、不读写权重），好让
`tests/test_carry_under_compression_quality.py` 把下面三件事钉成可失败单测，
并被 `mutate.py` 逐处抓红：

1. 两条**被训练**的臂只差一个因素（`n_kv_heads` 8→2，其余逐字段相同）；
2. 每个训练臂落的块类正确（MHA+carry → `CarryChannelBlock`；
   GQA+carry → `ComposableBlock`，ENG1 前不存在）；
3. 评测三行 = 两条训练臂的「权重 × KV 精度」组合，`int8` 行与 `carry_gqa` 同权重。

三臂（唯一变量 = 头投影 + 评测期 KV 精度；carry 与 window 是**常量**）：

| 评测行 | 训练臂（权重） | 训练 flags | 评测 KV 精度 | 相对谁 |
|---|---|---|---|---|
| `carry_mha_fp16`  | `carry_mha`（MHA 8 KV 头） | carry+window16 | fp16 | 基准（W3 的载体形态） |
| `carry_gqa_fp16`  | `carry_gqa`（GQA 2 KV 头） | carry+window16 | fp16 | 比 MHA（分离「头共享」） |
| `carry_gqa_int8`  | `carry_gqa`（**同**权重） | （同 `carry_gqa`） | int8 | 比 `carry_gqa_fp16`（分离「int8」） |

`carry`（delta 写）+ `window16` 是**常量**（不是变量）：它让「被问的 pair 的 key 落在
窗外」⇒ 不搬就答不出（SCHEMA §19 的局部先验抗性；2618/W3/Y1 的同一件仪器）。
评测行 `carry_gqa_int8` **不训练**、不加参数，只把 `carry_gqa` 的权重在 int8 KV 下评一遍。
"""

from __future__ import annotations

#: 两条被**训练**的臂（顺序 = MHA → GQA，相邻只差 `n_kv_heads`）。
TRAINED_ARMS = (
    {"name": "carry_mha",
     "n_kv_heads": None,
     "flags": {"carry_rule": "delta", "train_window": 16, "train_anchors": 0}},
    {"name": "carry_gqa",
     "n_kv_heads": 2,
     "flags": {"n_kv_heads": 2, "carry_rule": "delta",
               "train_window": 16, "train_anchors": 0}},
)
TRAINED_ARM_NAMES = tuple(a["name"] for a in TRAINED_ARMS)

#: 三条**评测行**：(名称, 权重来源臂, KV 精度)。
EVAL_ARMS = (
    {"name": "carry_mha_fp16", "weights": "carry_mha", "kv_mode": "fp16"},
    {"name": "carry_gqa_fp16", "weights": "carry_gqa", "kv_mode": "fp16"},
    {"name": "carry_gqa_int8", "weights": "carry_gqa", "kv_mode": "int8"},
)
EVAL_ARM_NAMES = tuple(a["name"] for a in EVAL_ARMS)

#: 基准行 = W3 的载体形态（carry + MHA + window，评测 fp16）。
BASELINE_EVAL_ARM = "carry_mha_fp16"
#: 判据锚：压缩档（GQA + 评测期 int8）。
COMPRESSED_EVAL_ARM = "carry_gqa_int8"
#: 「只换头共享」的中间行。
GQA_EVAL_ARM = "carry_gqa_fp16"


def trained_arm(name: str) -> dict:
    for a in TRAINED_ARMS:
        if a["name"] == name:
            return a
    raise KeyError(f"未知训练臂 {name!r}；期望 {TRAINED_ARM_NAMES}")


def eval_arm(name: str) -> dict:
    for a in EVAL_ARMS:
        if a["name"] == name:
            return a
    raise KeyError(f"未知评测行 {name!r}；期望 {EVAL_ARM_NAMES}")


def model_config_kwargs(spec: dict) -> dict:
    """训练臂 → `DepthARConfig` 关键字（唯一变量 = 这些字段）。"""
    return dict(spec["flags"])


def expected_block_class(spec: dict, n_heads: int = 8) -> str:
    """该训练臂在引擎里应落的块类（ENG1 的组合选择规则的镜像；ACTIVITY 用）。

    GQA × carry 同时开 ⇒ `ComposableBlock`；只开 carry ⇒ `CarryChannelBlock`。
    """
    f = spec["flags"]
    gqa = spec.get("n_kv_heads") is not None and spec["n_kv_heads"] < n_heads
    carry = f.get("carry_rule") is not None
    if gqa and carry:
        return "ComposableBlock"
    if gqa:
        return "GQABlock"
    if carry:
        return "CarryChannelBlock"
    return "Block"


def native_visibility(spec: dict) -> tuple[int, int]:
    """评测口径的可见性（= 该训练臂训练时的可见性；训练/评测同一掩码）。

    两条训练臂都用 `window=16 / anchors=0`（常量）；返回 `(window, anchors)`。
    """
    f = spec["flags"]
    return int(f.get("train_window", 0)), int(f.get("train_anchors", 0))


def compress_increment(acc: dict) -> dict:
    """三条评测行之间的分量分解（§17.17）：只差一个因素的两两差（pp）。

    `acc = {eval_arm_name: accuracy}`。返回：
    - `gqa_vs_mha_pp`   = 头共享的增量（`carry_gqa_fp16 − carry_mha_fp16`）；
    - `int8_vs_fp16_pp` = 评测期 int8 的增量（`carry_gqa_int8 − carry_gqa_fp16`）；
    - `net_pp`          = 净和（`carry_gqa_int8 − carry_mha_fp16`）。
    """
    mha = acc[BASELINE_EVAL_ARM]
    gqa = acc[GQA_EVAL_ARM]
    int8 = acc[COMPRESSED_EVAL_ARM]
    return {"gqa_vs_mha_pp": round(100.0 * (gqa - mha), 3),
            "int8_vs_fp16_pp": round(100.0 * (int8 - gqa), 3),
            "net_pp": round(100.0 * (int8 - mha), 3)}
