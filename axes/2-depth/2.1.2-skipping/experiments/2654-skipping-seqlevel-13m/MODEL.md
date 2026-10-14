# 2654 · 权重清单（MODEL.md）

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` / `data/`）。本文件登记权重哈希与生成命令。

## 状态

**尚无权重**：四臂（`stock` / `skip-mid` / `layers6` / `layerdrop-pt`）本轮**未训练**
（第 4 轮取到锁后跑 20 步冒烟，`layerdrop-pt` 活性门抓红（实测 lp 8.0 ≠ 设计 7.2）、launcher 退出；见 `run_log.txt:2194-2195`）。`runs/` 目录为空。

## 生成命令（续跑时）

```bash
# 训练 + 评测（走共享 train_depth_ar.main()，工厂替换 = 实验本地 SkipBlock）
python3 run.py --arms stock,skip-mid,layers6,layerdrop-pt
# 或（有界 launcher，逐臂取 GPU 锁）
bash run_remaining.sh
```

权重落点：`runs/<arm>-12k-s0/model.pt`（`arm ∈ {stock, skip-mid, layers6, layerdrop-pt}`）。
臂齐后在此登记每个 `model.pt` 的 `sha256` / 体积（与 `results.json` 的 `model_pt_sha256` 逐位一致）。

## 共享文件指纹

本实验**不改**共享文件：`pathlm/**`、`train_depth_ar.py` 零改动（工厂替换只在实验本地进行）。
训练整段走共享 `train_depth_ar.main()`；`run.py` 的 sha256 记在 `results.json` 的
`code_hash_run_py`。
