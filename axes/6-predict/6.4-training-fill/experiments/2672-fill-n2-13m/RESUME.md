# RESUME.md — 2672-fill-n2-13m 当前状态（接任段收尾）

## 完成到哪

- ✅ **6/6 快照 run 跑完**（3 臂 × seed{0,1}，统一 2000 步）：`snap2k-{fill,nofill,projfill}-s{0,1}`，各含 `model.pt` / `train_log.jsonl` / `gate_log.jsonl` / `results.json`。

- ✅ **WP2 等参数剥离评测扩展到 6 个新 checkpoint**（`rescore-snap2k.json`，逐 checkpoint `stored_match=true`），连同前任的 3 个 12k 档读数（`rescore-seed0.json`）一并进 `results.json::rescore`。

- ✅ **L0 config 对平全通过**：同刻度种子对差异清单为空；跨刻度对 12k 档仅 `train.steps` 差异（allowed）+ 缺席字段取引擎缺省（explained），未解释差异 0。

- ✅ 落盘：`results.json`（无墙钟字段，`l5_pass=true`）、`analysis.md`（人话在前 + L0–L5 + §4）、`MODEL.md`（sha256 + 复现命令）、本文件、`run_log.txt`。

- ⚠️ **本口径是 2k 快照**（锁饥饿 ⇒ 按预登记授权缩步数），**matched-step 12k 的 n≥2 仍未做**。

- ❌ 未做：本目录 `mutate.py`（变异抓红）、逐 prompt rescore 原始数据。


## 为什么是 2000 步

本段时间盒开头 GPU 锁被其它 worker（`2671-padfill-13m`，逐臂取放锁、每臂 ~2:30）连续占用；按 23e 实测（12000 步 ≈ 9.9 min）折算，可用 GPU 预算 ≈10 min ⇒ 取 6 run 总量 ≤10 min 的最大 500 倍数 = **2000**（≥1500 下限、臂间同刻度、契约 §2 明确允许缩步数）。


## 下一步确切命令（补 matched-step n≥2 时才需要）

```bash
D=axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m

# 12k 档 6 个 run（3 臂 × seed{0,1}；每 run ≈10 min GPU，分两段跑）

STEPS=12000 TAG=12k DEADLINE=$(date -d '+55 min' +%s) bash "$D"/run6.sh

TAG=12k SNAP_STEPS=12000 bash "$D"/finish.sh   # rescore + 对平 + 汇聚

python3 "$D"/write_docs.py
```


## 边界

注意：`aggregate.py` 每次只汇聚一个刻度（`TAG`）；若日后跑 12k 档，它会覆盖当前 2k 的 `results.json`，
先把本文件（2k 口径）复制留档，或让 12k 档写进独立 TAG 的 results（当前未实现，属下一段的范围）。

只写 `axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/**`；`README.md`（预登记契约）、`pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、`.scratch/23-depth-ar/evidence/**`（只读复用）、其它轴/实验、任何 `project.yaml` 均未改。


> **§20.1 让出窗口（Phase D 补记，axis6-lead）**：本实验的 driver 在臂间只间隔约 1s 重取 GPU 锁，**未**留 §20.1 要求的 ≥60s 让出窗口（Main 由其它轴的饥饿观测到）。该偏离只影响其它轴的等待时间，**不影响本目录任何判据**（本轴所有判据都是等效理论量、禁墙钟）。**若重跑本 driver：须在臂间插入 `sleep 60` 再取锁**；本目录已落盘的数字与 `MODEL.md` 的 sha256 绑定不回改。
