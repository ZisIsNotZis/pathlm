# 2647 · 同层内位置相关宽度（3.1.3 / W-C）过程记录

本文件是**过程记录**（只增不改、日期流水），不是事实文档。研究与判决的事实见
`axes/3-width/3.1.3-position-width/experiments/2647-position-width-13m/{README.md,analysis.md,results.json}`。

## 契约与落点

- 派单 = `.scratch/64-axis3-width/DISPATCH-B.md` §W-C；目录 = `axes/3-width/3.1.3-position-width/experiments/2647-position-width-13m/`。
- 结论口径：clean bpc（0.014 带）+ 每 token 平均激活通道数/等效 MACs + 长程召回（率类 CI）。

## 时间线

- `2026-10-14T23:0x` 起工。读 `2641/run.py`、`2644/run.py`（同轴兄弟）与 `pathlm/model.py::Block`、
  `pathlm/depth_ar.py::DepthARModel.forward`（dense+sequential 分支逐块调 `blk(h, attn_mask=None)`）。
- 干预落点定为 `Block.forward` 的实验目录内替换（不新增/不裁剪参数 ⇒ 各臂参数逐位相同）。
  窄段语义 = LN/qkv/proj/FFN 取前缀子矩阵 + qkv 尾部头置零 + FFN 隐藏维 = 4w
  （`GELU(0)=0` ⇒ 尾部隐藏单元贡献恒 0）；`w ≡ 256` 时**直接委托 stock 的 `Block.forward`**。
- 预登记 `README.md` 写完（§1–§10）后再落 `run.py` / 单测，单测 9 条全绿
  （T1 逐位等值、T2 SDPA 调用逐位相同、T3 解析代价独立重推 + 实测非零通道数、
  T4 尾部掩码变异能变红、T5 窄段对自身尾部扰动逐位不变 + 前缀正对照）。
- 落盘 `train_arms.sh`（2144 同款 §20 按臂取/放锁协议，锁超时 20 min）。
- `2026-10-14T23:29` 起 60 步冒烟（`STEPS=60 HOLD_ALL=1 ARMS="PN:0 NP:0"`，detached，
  `smoke_driver.log`），先确认「窄段真的是窄的前向、mask 未变」再起 12k supervisor。

## 待记（读完数后回填）

- 12k 读数（`results_PN_s0.json` / `results_NP_s0.json`）与到达时刻；
- 合成召回探针（PN / NP / 锚 2641 U4）读数；
- `--aggregate` 汇总与 `analysis.md`；
- 提交哈希。
