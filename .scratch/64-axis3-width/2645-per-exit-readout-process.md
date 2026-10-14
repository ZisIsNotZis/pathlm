# 2645（轴 3 · 3.2.2 逐出口读出）执行过程记录 · W-B2

日志文档（`.scratch/**`，允许日期流水）。事实以
`axes/3-width/3.2.2-per-exit-readout/experiments/2645-per-exit-readout-13m/` 下的
`README.md` / `analysis.md` / `results.json` 为准。

## 2026-10-14T22:47–23:0x+08:00 · axis3-w-3.2.2（fresh worker）

* 契约：`.scratch/64-axis3-width/DISPATCH-B.md` §W-B2（基底 depth-AR、锚
  `.scratch/23-depth-ar/evidence/23c-H2-w0-12k`、四臂 `T`/`PE`/`IND`/`LR`、判据 P1–P4）。
* 已交付（全部落在自己的实验目录）：
  * `run.py`：四臂实现（`PerExitModel` 只覆写 `readout(h, depth)`；`T` 回落 stock 路径）、
    `--stage anchor` 锚三道门、`--aggregate` 聚合（含按批配对 bootstrap / Wilson CI）。
  * `test_2645_per_exit_readout.py`：17 项，全绿（`pytest -q`，6.8 s）。
  * `README.md`：预登记契约（臂 / P1–P4 / 否证线 / 带声明 / 边界）。
  * `train_arms.sh` / `drive.sh`：前台取锁驱动（逐臂取放、臂间 61 s、不回收他线锁）。
  * `analysis.md`：人话摘要 + 未跑清单 + 受阻原因 + 与 3.2.1 的边界。
  * `results.json`：`--aggregate` 纯 CPU 聚合，四臂一律 `status: "未跑"`。
* **实测臂：0 条**（未跑），原因两条独立存在：
  1. 本单硬约束「单条 bash 命令 ≤180 s」vs 同规模 12k 实测墙钟 12.69 min
     （`2610-gated-readout-13m-depthar/run_log.txt`），且引擎无 `--resume` ⇒ 前台不可承载、
     不可分块续训；本单又禁 `setsid`/`nohup`/后台驱动。
  2. GPU 锁饥饿：`/tmp/pathlm_gpu.lock` holder 在本次会话内由 `wp2655-layer-capacity`
     换成 `wp6.3.1-mdiff-s0`（均 LIVE），两次前台等锁（110 s、140 s）均未取到；
     按本单规定不杀他人进程、不回收他锁。
  冒烟（60 步）与 `--stage anchor`（需 GPU 电池）同样卡在锁上，未执行。
* 跑臂前凭单测钉住的关键事实（见 `analysis.md` §1/§2）：四臂主干逐位相同；`PE`/`LR` 恒等起步
  逐位等于 `T`；每出口参数量 65,536 / 52,736 / 7,392；且 `V=206 < d=256` ⇒ **`PE` 的有效出口
  头族 ⊇ `IND` 的表族**（`E·M = Y` 可解，`lstsq` 残差 < 1e-5）⇒ 本尺度「`IND` 表达力更强」
  不成立，`IND` 独赢只能归优化/初始化差异（P4 第二行的读法）。
* 与 `3.2.1` 的关系：`2621-per-depth-readout-13m` 的 `PD` 臂有效出口头
  `F_k = E(I + P_kQ_k)` 落在 `PE` 同一函数族内，且已被等参数「共享宽变换」对照否证
  ⇒ 本实验若跑出 `PE` 读数不得当独立证据，须同框解读（`analysis.md` §5）。
* 下一步（未做）：取锁 → 60 步冒烟 rc=0 → `PE:0`（需 >12 min 前台预算或后台授权）→ `T:0`
  → `run.py --stage anchor` → `--aggregate`。
