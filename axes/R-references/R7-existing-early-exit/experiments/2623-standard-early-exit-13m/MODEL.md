# MODEL — 2623 标准早退实现（R7 · WI-16）权重清单

**权重本身不入库**（`*.pt` 已 gitignore）：本文件只记哈希、体积与生成命令，
checkpoint 落在 `.scratch/52-r7-early-exit/evidence/<run>/model.pt`（日志文档的家）。
复现命令逐条见 `config.json::arms`；落盘目录：`.scratch/52-r7-early-exit/evidence/`。

## 臂的权重（sha256 全值）

| run | sha256 | 体积 MB | 参数 | 形态（d/L/h/seq/mlp + dense/gate/aux/mod） | 训练 | 墙钟 min | 峰值 MB |
|---|---|---|---|---|---|---|---|
| `r7-stock-s0` | `06aba87d88c451a5ab1afea995917a1377c00e5c364a8f02fb15440f65497177` | 26.59 | 6,962,688 | 256/8/8/256/4 dense=True gate=True aux=renorm mod=0 | 12000×bs32×seq256, seed 0 | 6.79 | 935.3 |
| `r7-c1-legacy-s0` | `77e149f5f74fb0126221c1b88f9988d7715b232531c82d4aff4c078986ad15d0` | 26.59 | 6,962,688 | 256/8/8/256/4 dense=False gate=True aux=renorm mod=0 | 同上 | 10.02 | 1339.9 |
| `r7-c1-fair-s0` | `f772e5836628538548f0c3daa581da299432d8b255c04c1c0c0bb948995e1589` | 26.59 | 6,962,688 | 256/8/8/256/4 dense=False gate=True aux=additive(λ=0.1) | 同上 | 10.36 | 1339.9 |
| `r7-c1-fair-l1-s0` | `aced9d3d23c871ff896c6c93c832ade567b9f757da403dea6afc09f8de8f0dcf` | 26.59 | 6,962,688 | 256/8/8/256/4 dense=False gate=True aux=additive(λ=1.0) | 同上 | 10.12 | 1339.9 |
| `r7-ls-fair-s0` | `818a1186ae9d00ae57156d95cd7dfdb865d237c01b2ba263f31ed3ab4ac230ec` | 24.58 | 6,436,352 | 256/8/8/256/4 dense=False **gate=False** aux=additive(λ=0.1) | 同上 | 9.73 | 1203.5 |
| `r7-ls-legacy-s0` | `2f3fefd4b37899afa67c585215176553870621b2777746c040c6aca6fa7e7388` | 24.58 | 6,436,352 | 256/8/8/256/4 dense=False **gate=False** aux=renorm | 同上 | 8.58 | 1203.5 |
| `r7-mod-s0` | `3547296506874ac96bfbb6f1f6151328c007483858a4755fd9d271d0f82f4274` | 52.75 | 13,811,216 | 256/**16**/8/256/4 dense=True **mod_ratio=0.5**（路由器 4112 参数） | 同上 | 19.16 | 1330.4 |

生成命令（逐字）：

```bash
python3 train_depth_ar.py <run> --out-root .scratch/52-r7-early-exit/evidence \
  --d 256 --layers 8 --heads 8 --mlp-mult 4 --seq-len 256 --batch-size 32 --lr 1e-3 \
  --steps 12000 --seed 0 <臂开关>
# 臂开关：stock = --dense --exit-mode full
#          c1-legacy = --exit-mode uniform --aux-weight 1.0 --final-weight 1.0
#          c1-fair = --exit-mode uniform --aux-mode additive --aux-weight 0.1 --final-weight 1.0
#          c1-fair-l1 = 同上但 --aux-weight 1.0
#          ls-fair = c1-fair + --no-carry-gate
#          ls-legacy = c1-legacy + --no-carry-gate
#          mod = --dense --layers 16 --mod-ratio 0.5 --mod-router-weight 0.01 --exit-mode full
```

## 复现与校验

```bash
# 全部训练臂（按臂取/放 GPU 锁，臂间 ≥60 s）：
bash axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/run_arms.sh all
# 推断电池 + 交错墙钟（锁内、无训练）：
bash axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/run_eval.sh all
# 对账（analysis.md 的数字 ⇔ results.json）：
python3 axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m/reconcile_numbers.py
```

## 共享文件指纹（`SCHEMA` §22.6；本波改过的引擎文件在该时点的哈希）

| 文件 | sha256（本波结束时刻） |
|---|---|
| `pathlm/depth_ar.py` | 见 `git log -1 -- pathlm/depth_ar.py` 对应提交（新增 `aux_mode` / `carry_gate` / `mod_ratio` / `FrozenOpenGate`） |
| `train_depth_ar.py` | 同上（新增 4 个 CLI 开关 + `_r7_intervention` + 规格自检） |
| `pathlm/mod_router.py` | 新文件（MoD 路由，`mod_ratio>0` 才激活） |
| `docs/engine_capability.md` | 行号漂移同步 + 新增 C13–C16 四条能力行 |

> 声明：本波与 `axes/M-carrier/**`（2633/2634/2635）、`axes/A-mixing/**`（2636/2637）、
> `axes/H-horizon/H3-composition/**` 并行运行；`pathlm/depth_ar.py` 的改动使
> `tests/test_carry_seed2_13m.py::test_engine_files_unchanged_since_2631_run`（按文件哈希
> 断言引擎未变的 §9 复用前提）变红——该线应把复用证据换成「开关关闭时逐位等价」的
> 单测口径（`SCHEMA` §9 第 1 条），本波未改其目录。
