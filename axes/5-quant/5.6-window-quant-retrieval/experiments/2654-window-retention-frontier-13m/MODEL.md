# MODEL.md · 2654 的被测模型（权重清单）

本实验**不训练**，只评测**一个**只读 checkpoint（stock decoder-only trunk）：
`full` 与全部窗口/组合臂都是**同一个模型**在不同评测期推理配置下的前向（`README.md` §2）。

| 项 | 值 |
|---|---|
| 路径 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| sha256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| size | 54193675 B |
| 来源 / 训练说明 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/MODEL.md`（12k 步冻结 trunk，clean bpc 1.483569） |
| 配置 | `d_model=256, n_layers=12, n_heads=8, head_dim=32, mlp_mult=6, seq_len=256` |
| 载入断言 | `2619-kv-budget-chart/run.py::load_trunk` 运行时重算 sha256 并比对（不符即 `SystemExit`）；`results.json::checkpoint.sha256_verified` 留证 |

权重、数据、checkpoint **不入库**（`harness/methodology.md` §8）。跨设备（GPU）历史数字只作对照，不作判据。
