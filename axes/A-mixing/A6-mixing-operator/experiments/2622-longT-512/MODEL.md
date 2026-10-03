# MODEL — 2622 · 长 T 首探 T=512（权重清单，权重本身不入库）

## 权重（`*.pt` 已由 `.gitignore` 覆盖；`runs/` 整目录亦被忽略）

| 臂 | 路径 | sha256 | 体积（字节） |
|---|---|---|---|
| `full-256` | `runs/full-256/model.pt` | `702f8ae5e55268f6d4594f3ac3313e9e3c30a472b9b7621bd656d2a2297c5334` | 27,882,155 |
| `win128-256` | `runs/win128-256/model.pt` | `6e634858e18a529008fea81828e41ebab9d214d49ebe8da49117a047c61ae138` | 27,882,155 |
| `full-512` | `runs/full-512/model.pt` | `6edb52cdc9962b9d215433a1bccba0ab7f286919d069f33316f36ce93f654147` | 28,144,299 |
| `win256-512` | `runs/win256-512/model.pt` | `4c3658593f972052c61d7a17650849d41664389c71d1fee83efdadbef5c9288a` | 28,144,299 |

- T=512 的两臂比 T=256 的两臂大 **262,144 字节**（= 65,536 参数 × 4 字节 fp32，位置表增量）。
- 生成命令（在实验目录内，逐臂）：

  ```bash
  python3 run.py --stage train --arm full-256      # 或 win128-256 / full-512 / win256-512
  # 等价底层命令：
  python3 train_depth_ar.py <arm> --d 256 --layers 8 --heads 8 \
      --seq-len <256|512> --mlp-mult 4 --dense --steps <12000|6000> \
      --batch-size 32 --lr 0.001 --seed 0 --exit-mode uniform \
      [--train-window <128|256>] --eval-batches 20 \
      --out-root axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512/runs
  ```

## 代码指纹（共享文件的并发写声明，SCHEMA §22.6）

| 文件 | sha256 | 本实验是否修改 |
|---|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` | **否**（只读） |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` | **否**（只读） |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a355b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` | **否**（只读） |

**口径重叠如实记录（§22.6）**：本轮 **X2 是唯一写共享代码者**。`pathlm/depth_ar.py` 含 X2 的
ENG1 改动（`ComposableBlock` / `KVHeadProjectionMixin` / 窗口掩码装饰层），该内容自提交
`b96ac47`（2026-10-03 10:49）起即在库中；本实验的**四臂训练（11:35–12:13）与评测均在
该版本之后**用同一 sha256 的引擎，故四臂**引擎一致**，无跨臂代码混杂。
（`depth_ar.py` 的 mtime 在 `11:41` 变过，是 X2 的变异验证**逐字节还原**造成的 mtime 更新，
内容与 `b96ac47` 逐位相同；`git status` 对该文件为 clean ⇒ 内容未变。）

本实验**只新增** `tests/test_longT_512.py` 与实验目录文件，未改任何共享文件。

## 复现（单卡 4090 共享，取锁按 §20/§20.1）

```bash
cd axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512
python3 smoke.py                      # CPU 不变量（不取锁）
python3 run.py --stage preflight      # 20 步冒烟 + 步时/显存（自己取锁）
python3 run.py --stage train-all      # 四臂，按臂释放锁，臂间 ≥60s
python3 run.py --stage eval           # native/release bpc + 合成腿 + 成本
python3 run.py --stage report         # 汇总 results.json / config.json
python3 mutate.py                     # 变异验证（monkeypatch，不写共享文件）
python3 reconcile_numbers.py          # 数字对账（79 条）
```
