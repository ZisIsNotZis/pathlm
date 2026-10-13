# MODEL · 2635-recursive-carrier-13m 权重清单

> 递推/线性状态载体（`--attn-linear`，参数量与 stock 臂逐位相同）× 2 臂，各 3,000 步 / bs32 / T256 = 24.6M token（**为什么是 3k**：2616 的停摆二分显示 1.2k–3k 步的独立进程全部跑通，≥6k 的 12k 运行为 ~1/2 停摆概率）。

> 权重（`model.pt`）**不入库**（`.gitignore` 的 `*.pt`）；本文件记录 sha256 / 体积 / 训练墙钟与峰值显存，
> `run.py` 加载时**逐次断言** sha256（`results.json::checkpoints`）。

| 臂 | 参数量 | 体积 (B) | 训练墙钟 (min) | 训练峰值显存 (MB) | sha256 |
|---|---:|---:|---:|---:|---|
| `stock` | 13536768 | 54193675 | 2.56 | 1630.2 | `6c036fb3773e80c2f6d73ab9ee002671429b5742f13747bcf46c8cd4ba15033a` |
| `linear` | 13536768 | 54193675 | 4.43 | 2139.7 | `7b27cfcefa517e0c3b09c089c20c6cfbae797ca0ee1de1890423caff4ba0e730` |

## 结构

- 停摆门 `stall_gate.py`：`PATHLM_DUMP_STACKS=150` 打全线程栈 + 差分两级信号（`--stall-secs` 记事件、`--kill-secs` 才终止）；`stall_report_<arm>.json` 是原始诊断。
- **重要教训（已钉进单测）**：训练器只在**未被梯度门跳过的步**写 `train_log.jsonl`，所以「日志不推进」≠「进程停摆」；本例 `grad_skips + 有效步数 = 3000` 对账通过，进程实际跑完。

## 复现路径

```bash
bash axes/M-carrier/M1-carrier-type/experiments/2635-recursive-carrier-13m/train_*.sh      # 训练（按臂释放 GPU 锁，臂间 ≥60s）
bash axes/M-carrier/M1-carrier-type/experiments/2635-recursive-carrier-13m/eval.sh          # 评测（纯推理，整程 <10 min 持锁）
```

## 说明

- 主干：stock dense decoder-only，d256 × 12 层 × 8 头 × mlp6 × T256（13.54M，两臂**参数逐位相同**）。
- `linear` 臂：注意力换成归一化线性核 `out_i = Σ_{j≤i} φ(q_i)·φ(k_j)·v_j / Σ φ(q_i)·φ(k_j)`（φ = elu+1），累积状态实装（每层每头 `dh×dh`，与 T 无关）。
- **能力面限制**：引擎硬互斥 `attn_linear ⊕ train_window` ⇒ 两臂都全可见（无窗口），「窗口 + 线性载体」不可测。
