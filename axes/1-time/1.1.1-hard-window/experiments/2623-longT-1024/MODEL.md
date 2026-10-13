# MODEL — 2623 · 长 T 第三点 T=1024（权重清单，权重本身不入库）

## 权重（`*.pt` 已由 `.gitignore` 覆盖；`runs/` 整目录亦被忽略）

| 臂 | 路径 | sha256 | 体积（字节） |
|---|---|---|---|
| `full-1024` | `runs/full-1024/model.pt` | `e2eeb2b36782058d6c67b39ecef96811918e7bd5811d1d4155870dd0f444624f` | 28,668,587 |
| `win512-1024` | `runs/win512-1024/model.pt` | `b25a5041c11046f5b7fc6af3f1c3a880876164ab742f12c7b2d6c63261025ff6` | 28,668,587 |

- 两臂参数量相同（7,159,296）⇒ 体积相同；位置表相对 T=512 增大 512 行 × 256 维，多
  131,072 参数 × 4 字节 = **524,288 字节**。
- 生成命令（在实验目录内，逐臂）：

  ```bash
  python3 run.py --stage train --arm full-1024      # 或 win512-1024
  # 等价底层命令：
  python3 train_depth_ar.py <arm> --d 256 --layers 8 --heads 8 \
      --seq-len 1024 --mlp-mult 4 --dense --steps 12000 \
      --batch-size 8 --lr 0.001 --seed 0 --exit-mode uniform \
      [--train-window 512] --eval-batches 20 \
      --out-root axes/1-time/1.1.1-hard-window/experiments/2623-longT-1024/runs
  ```

## 代码指纹（共享文件的并发写声明，SCHEMA §22.6）

| 文件 | sha256 | 本实验是否修改 |
|---|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` | **否**（只读） |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` | **否**（只读） |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a355b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` | **否**（只读） |

- 上述三文件 sha256 与 ticket 39-X3（`2622-longT-512/MODEL.md`）**逐位相同** ⇒ 本实验与 X3 用
  **同一引擎版本**；`git status` 对这三文件为 clean ⇒ 内容未变。
- 本实验**只新增** `tests/test_longT_1024.py` 与实验目录文件，未改任何共享文件。

## 复现（单卡 4090 共享，取锁按 §20/§20.1）

```bash
cd axes/1-time/1.1.1-hard-window/experiments/2623-longT-1024
python3 smoke.py                      # CPU 不变量（不取锁）
python3 run.py --stage preflight      # 20 步冒烟 + 步时/显存（自己取锁）
python3 run.py --stage train-all      # 两臂，按臂释放锁，臂间 ≥60s
python3 run.py --stage eval           # native/release bpc + 合成腿 + 成本
python3 run.py --stage report         # 汇总 results.json / config.json
python3 mutate.py                     # 变异验证（monkeypatch，不写共享文件）
python3 reconcile_numbers.py          # 数字对账
```

## 口径与偏离记账（§20.2）

- **batch = bs8（两臂同配方）**：T=1024 的窗口臂在 **bs32 / bs16 下训练 OOM**（本机背景常驻
  ~18.3 GiB），故降到 bs8；token 预算仍 98.3M。**与 X3 的 bs32 不同** ⇒ 三点不可直接并列（见
  `analysis.md §2`）。
- 峰值显存：full 0.914 GiB / win512 **1.915 GiB**（均 ≤5GB net）；跑前后 free = 4624 / 4099 MB
  （§20.2 的「余量 ≥8GB」在本机背景占卡下不可满足）。
