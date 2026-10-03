# MODEL — 2624 · 长 T 第三点 T=1024 的 seed=1 双臂重跑（权重清单，权重本身不入库）

## 权重（`*.pt` 已由 `.gitignore` 覆盖；`runs/` 整目录亦被忽略）

| 臂 | 路径 | sha256 | 体积（字节） |
|---|---|---|---|
| `full-1024-seed1` | `runs/full-1024-seed1/model.pt` | `3c34a1ffcc081fad818892cd4a224fa9d25fddc4d8dc5f151122ed318d0fe5b5` | 28,668,587 |
| `win512-1024-seed1` | `runs/win512-1024-seed1/model.pt` | `b34bd8fdd8379219b927d01feecb5571c1de5f60aaa99b59ade619a3542cf81c` | 28,668,587 |

- 两臂参数量相同（7,159,296）⇒ 体积相同；与 2623（seed0）两臂**同为 7,159,296**
  （本实验只改 seed，不改结构）。
- 生成命令（在实验目录内，逐臂）：

  ```bash
  python3 run.py --stage train --arm full-1024-seed1     # 或 win512-1024-seed1
  # 等价底层命令：
  python3 train_depth_ar.py <arm> --d 256 --layers 8 --heads 8 \
      --seq-len 1024 --mlp-mult 4 --dense --steps 12000 \
      --batch-size 8 --lr 0.001 --seed 1 --exit-mode uniform \
      [--train-window 512] --eval-batches 20 \
      --out-root axes/A-mixing/A6-mixing-operator/experiments/2624-longT-1024-seed1/runs
  ```

## 代码指纹（共享文件的并发写声明，SCHEMA §22.6）

| 文件 | sha256 | 本实验是否修改 |
|---|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` | **否**（只读） |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` | **否**（只读） |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` | **否**（只读） |

- 上述三文件 sha256 与 2623（`2623-longT-1024/MODEL.md`）**逐位相同** ⇒ 本实验与 seed0 的
  T=1024 点用**同一引擎版本**；`git status` 对该三文件为 clean ⇒ 内容未变。
- 本实验**只新增** `tests/test_longT_1024_seed1.py` 与实验目录文件，未改任何共享文件。

## 复现（单卡 4090 共享，取锁按 §20/§20.1）

```bash
cd axes/A-mixing/A6-mixing-operator/experiments/2624-longT-1024-seed1
python3 capability_check.py           # §22.8 能力面（CPU，不取锁）
python3 smoke.py                      # CPU 不变量 + seed 活性（不取锁）
python3 run.py --stage preflight      # 20 步冒烟 + 步时/显存（自己取锁）
python3 run.py --stage train-all      # 两臂，按臂释放锁，臂间 ≥60s
python3 run.py --stage eval           # native/release bpc + 合成腿 + 成本
python3 run.py --stage report         # 汇总 results.json / config.json
python3 mutate.py                     # 变异验证（monkeypatch，不写共享文件）
python3 reconcile_numbers.py          # 数字对账
```

## 口径与偏离记账（§20.2）

- **batch = bs8（两臂同配方，与 2623 一致）**：T=1024 的窗口臂在 bs32 / bs16 下训练 OOM
  （2623 实测），故沿用 bs8；token 预算仍 **98.3M**。**与 X3 的 bs32 不同** ⇒ 三点不可直接
  并列（见 `analysis.md §2`）。
- 峰值显存（本实验实测）：full **0.914 GiB** / win512 **1.915 GiB**（`nvidia-smi` 子进程
  1.48 GB / 2.53 GB），均 ≤5GB net 礼貌线。
- 跑前/跑后 `nvidia-smi memory.free`（**本实验两次读数**）：
  preflight 轮 **17958 → 17435 MB**；eval 轮前后 **15192 → 18546 MB**（期间其他 agent 的
  训练在卡上，读数波动是**外部占用**造成的）。本机背景常驻 `llama-server`（~3.5 GiB）+
  其他线占用 ⇒ **§20.2 的「跑前后余量 ≥8GB」在本机不可持续满足**（如实记账；未碰、不杀别人进程）。
- **并发争用记账（§20.1/§20.3）**：训练期间另有两条 GPU 线（Z2 `composable-followups`、
  Z3 `carry-under-compression-quality`）争锁，且**按臂释放**（60s 窗口被其他线插队）；
  本实验实测步时（preflight 轮）full 41.2 ms / win 26.0 ms（bs8，与 2623 的 ~45 ms 同量级）
  ⇒ **未触发 §20.3 的缩步数**（两臂均跑满 12000 步）。
- **锁粒度观察（§20.1）**：本线 `train-all` 按臂释放、臂间留 60s；但并行的另两条线
  （`train_followups.sh` / `train_arms.sh`）**整程持锁**（观察到 holder_pid 是脚本、持续
  20+ min）⇒ 本线 eval/preflight 多次等待。**仅记录，不影响本线结论**。
