# 进度（W4 · 2621-per-depth-readout-13m）

- [x] §22 锚点声明 + 代码锁；改 `pathlm/depth_ar.py` / `train_depth_ar.py` / `tests/test_depth_ar.py`
      （本波唯一写这三文件的线）。
- [x] 单测（+9）通过；`smoke.py`（A 实现层冒烟 + B 20 步训练活性 + C 三条变异验证抓红且逐字节还原）通过。
- [x] 步时实测：A 105 / PD 130 / EP 120 ms/step（GPU util ≈10% ⇒ CPU 开销 + 共享机负载主导）。
- [x] `run.py --stage arms`：A/gated/linear 复用臂同口径重测，等值性门 pass。
- [x] `run.py --stage perdepth`（训练 29.08 min + 电池）
- [x] `run.py --stage wide_shared`（训练 24.06 min + 电池）
- [x] `run.py --stage assemble`（合并 + 判读 + 写 `results.json`）
- [x] `reconcile_numbers.py` 42 条全绿；`analysis.md` / `MODEL.md` / `progress.md` 落盘
- [x] `python3 -m pytest tests/ -q` **361 passed**（含本线 +9；其余为邻线并行新增）

## 结论一句话

逐深度独立读出相对基线**有效**（Δbpc −0.0759），但**同名义参数的共享宽读出严格更好**
（Δbpc −0.1313，差 0.0554 ≫ 0.014）⇒ 「一个共享读出服务所有深度 = 本形态缺陷」这一
具体机制**不被支持**；获益是读出容量效应（共享宽变换最有效）。逐深度读出唯一抬高
浅层草稿资产（tf@4 0.9694→0.9727，而共享宽臂掉到 0.9576）。

## 锁台账

| 阶段 | 锁 | 备注 |
|---|---|---|
| 步时/冒烟 | `/tmp/pathlm_gpu.lock` | 每阶段取、退出即释 |
| arms | 同上 | 纯 eval ≈1.5 min |
| perdepth | 同上 | 训练 29.08 min；结束即释 |
| wide_shared | 同上 | 与上一臂间留 ≥60s 窗口；训练 24.06 min |

无等锁超时、无锁泄漏、无显存超预算（峰值自身 2.69GB，总 < 5GB net）。
