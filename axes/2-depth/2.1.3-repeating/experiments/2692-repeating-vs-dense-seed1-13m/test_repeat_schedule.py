"""`RepeatModel` 的可失败单测（CPU、秒级）：循环深度这个开关必须真的接上线。

钉住四件事（`harness/methodology.md` §5：每个语义开关都有可失败单测）：

1. **恒等调度与 stock 逐位等值**：`schedule = (0..n−1)` 时 `RepeatModel` 的 logits
   与 `DepthARModel` 逐位相同 —— 证明它不是一个「另写的模型」；
2. **重复是活的**：`repeat2x` / `repeat2x-tail` 的完整调度与「只跑前 `blocks` 次」
   的最终 logits 不同（偏差 > 0）；恒等调度臂的偏差恒 0（无重复可测）；
3. **等参数**：`repeat2x` / `repeat2x-tail` 与 `dense6` 参数量逐位相同，
   且同 seed 下初始化逐位相同（两臂唯一差别 = 每层被调用几次）；
4. **调度覆盖**：`run.py::ARMS` 与契约 `config.json` 的 blocks/schedule 一致。

变异验证（故意破坏时必须抓红）：把 `repeat2x` 的 schedule 改成恒等 ⇒ 第 2 条变红；
把 `dense6` 的 blocks 改成 7 ⇒ 第 3 条变红。
"""

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

import sys  # noqa: E402

sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run as R  # noqa: E402
from pathlm.depth_ar import DepthARModel  # noqa: E402

VOCAB = 206


def _build(spec, seed=0):
    torch.manual_seed(seed)
    return R.build_model(spec, VOCAB)


def _final_logits(model, x):
    model.eval()
    with torch.no_grad():
        return model(x, exit_depths=None)[1]["depth_logits"][-1]


def test_identity_schedule_equals_stock():
    """同一套权重（把 stock 的 state_dict 装进 RepeatModel）⇒ logits 必须逐位相同。"""
    spec = R.ARMS["dense6"]
    torch.manual_seed(1234)
    stock = DepthARModel(R.spec_cfg(spec), VOCAB)
    model = _build(spec)
    model.load_state_dict(stock.state_dict())
    x = torch.randint(0, VOCAB, (2, 16),
                      generator=torch.Generator().manual_seed(3))
    a = _final_logits(stock, x)
    b = _final_logits(model, x)
    assert a.shape == b.shape
    assert torch.equal(a, b), f"恒等调度与 stock 不等值：max|Δ|={(a - b).abs().max()}"


def test_repeat_is_alive_and_prefix_differs():
    x = torch.randint(0, VOCAB, (2, 16),
                      generator=torch.Generator().manual_seed(3))
    for arm in ("repeat2x", "repeat2x-tail"):
        spec = R.ARMS[arm]
        model = _build(spec)
        full = _final_logits(model, x)
        model.schedule = tuple(range(spec["blocks"]))
        pref = _final_logits(model, x)
        rel = float((full - pref).norm() / full.norm())
        assert rel > 0.0, f"{arm}: 重复是死的（与只跑前缀逐位相同）"


def test_identity_arm_has_no_prefix_deviation():
    x = torch.randint(0, VOCAB, (2, 16),
                      generator=torch.Generator().manual_seed(3))
    for arm in ("dense6", "dense12"):
        spec = R.ARMS[arm]
        model = _build(spec)
        full = _final_logits(model, x)
        pref = _final_logits(model, x)     # 调度本来就等于前缀
        assert torch.equal(full, pref)


def test_equal_params_and_equal_init():
    m6 = _build(R.ARMS["dense6"])
    mrep = _build(R.ARMS["repeat2x"])
    mtail = _build(R.ARMS["repeat2x-tail"])
    p6 = sum(p.numel() for p in m6.parameters())
    assert p6 == sum(p.numel() for p in mrep.parameters())
    assert p6 == sum(p.numel() for p in mtail.parameters())

    def sd(model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    a, b = sd(m6), sd(mrep)
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"同 seed 初始化不一致：{k}"


def test_schedule_matches_contract_config():
    import json
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    arms = {a["arm"]: a for a in cfg["arms"]}
    assert set(arms) == set(R.ARM_ORDER)
    for name in R.ARM_ORDER:
        assert arms[name]["blocks"] == R.ARMS[name]["blocks"], name
        assert list(arms[name]["schedule"]) == list(R.ARMS[name]["schedule"]), name
        assert arms[name]["passes"] == R.ARMS[name]["passes"] == len(
            R.ARMS[name]["schedule"]), name


def test_macs_identical_for_same_pass_count():
    assert R.macs_per_token(12, VOCAB) == R.macs_per_token(12, VOCAB)
    assert R.macs_per_token(6, VOCAB) < R.macs_per_token(12, VOCAB)
