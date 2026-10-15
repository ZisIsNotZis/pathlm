# 2654 · 权重清单（MODEL.md）

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` / `data/`）。本文件登记权重哈希与生成命令。

## 状态

**尚无权重**：四臂（`stock` / `skip-mid` / `layers6` / `layerdrop-pt`）**未训练**。
P0 根因已修（`layerdrop-pt` 臂 spec 补 `"p": LAYERDROP_P`，`run.py:77`；缺键曾让
`SkipBlock` 的 p 退化成 0.0、`layerdrop-pt` 实测 lp 8.0 ≠ 设计 7.2，见
`run_log.txt:2194-2195`），CPU 回归测试 `test_skipblock_layerdrop.py` 已把红钉住并经变异验证。
后台重试驱动 `run_supervise.sh` 已起（冒烟闸门 + 四臂逐臂取锁），但本轮没抢到 GPU 锁
（持有者 = `2656-lowrank` pid `2620551`），故 `runs/` 目录为空、无 12k 权重。

## 生成命令（续跑时）

```bash
# 训练 + 评测（走共享 train_depth_ar.main()，工厂替换 = 实验本地 SkipBlock）
python3 run.py --arms stock,skip-mid,layers6,layerdrop-pt
# 或（有界 launcher，逐臂取 GPU 锁）
bash run_remaining.sh
# 或（后台有界重试驱动：冒烟闸门 + 逐臂取锁，判活看末行 SUPERVISE DONE）
setsid nohup bash run_supervise.sh >supervise.out 2>&1 </dev/null &
```

权重落点：`runs/<arm>-12k-s0/model.pt`（`arm ∈ {stock, skip-mid, layers6, layerdrop-pt}`）。
臂齐后在此登记每个 `model.pt` 的 `sha256` / 体积（与 `results.json` 的 `model_pt_sha256` 逐位一致）。

## 共享文件指纹

本实验**不改**共享文件：`pathlm/**`、`train_depth_ar.py` 零改动（工厂替换只在实验本地进行）。
训练整段走共享 `train_depth_ar.main()`；`run.py` 的 sha256 记在 `results.json` 的
`code_hash_run_py`。
