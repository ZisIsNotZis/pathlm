# RESUME · 2674-mdiffusion-13m（WP 6.3.1-WP1 掩码扩散最小判决）

- 目标 / 契约：`README.md`（预登记，不得改）；matched-step = `2671/results.json::config.steps` = 1500；
  seed {0,1}；K∈{1,4,8,16} 置信度解掩码采样；AR 只读复用 2671 读数。
- owner / 状态：WP worker（单一写入者）；执行中，**GPU 锁被他人长占**（见下）。

## 已落盘（本轮）

| 文件 | 状态 |
|---|---|
| `mdiff.py`（库：`train_mdiff` + `sample_denoise_k` + `mdiff_metrics`） | 完成 |
| `run.py`（驱动：mdiff 训练 + K 扫描 + L0–L5 汇总 + 落盘） | 完成 |
| `test_mdiff.py`（9 项：K=1 退化等价 / 步数守恒 / 无泄漏 gate probe / 确定性 / bpc / run 纯函数） | **9 passed** |
| `mutate.py` + `mutation_report.json`（4 条变异全抓红 + 逐字节还原 + 还原后全绿） | 完成 |
| `gpu_lock.py`（从 2671 复制，name=wp6.3.1） | 完成 |
| `run_log.txt` | 已建；含 `L0 M-alignment pass=True diffs=0`（M 与 2671 逐 block 对齐） |

## L0 已实证（无需 GPU）

`python3 -c` 复现 2671 `eval_block_set` 的 M 生成顺序（r=0.25、min_pos=8、eval_mask_seed=2674）：
64 block 的 `n_m` 与 `2671/results.json::ar[*].blocks` **逐条一致（差异清单为空）** ⇒ 同一位置集。

## 阻塞：GPU 锁

- `run.py --stage all --ar reuse` 于 22:35:19 启动，排队等 `/tmp/pathlm_gpu.lock`（`max_wait=900s`）。
- 锁被轴 2 的 `2655-layer-capacity-13m` 长作业持有（`--steps 12000`，单臂训练 ~495s，
  `--arm-gap-s 65`）；本 worker **不杀他人进程**，只按协议重试。

## 未完成 / 下一步（最小成本）

1. **mdiff 训练 + K 扫描**（未完成）：等锁释放后
   ```bash
   cd /home/z/vibe/pathlm/axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m
   nohup python3 run.py --stage all --ar reuse --out . --seeds 0 1 > nohup.run.out 2>&1 &
   ```
   预计 GPU 时间 ≈ 2 seed × 1500 step × ~0.088s ≈ 265s + 评测（秒级）。
2. **L3 外部锚**：2671 AR checkpoint 未落盘 ⇒ 本轮记为 gap；补做需与本进程同框重训 AR（≈265s）后 rescore。
3. **gaps**（契约 §4 明列）：均匀核臂、噪声调度扫描、与 6.3.3 同框。

## 可复现命令

```bash
python3 -m pytest test_mdiff.py -q     # 9 passed
python3 mutate.py                      # 4 变异全抓红 + 逐字节还原
```
