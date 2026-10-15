# JOURNAL · 2654（过程史；不是事实文档）

- 2026-10-14：轴 5 Phase C 派发 WP1（`.scratch/66-axis5-quant/DISPATCH-5.6-PHASEC.md`）。
  口径裁决：CPU（GPU 锁被其它轴训练占用）、不取锁；量化腿一律 `5.1` 块标尺（G=32，沿 head_dim）；
  组合臂对齐 `2642` 的 7 个同字节水平（禁插值）；逐臂落盘、30 min 硬上限下先保 `results.json`+`table.md`。
  预登记（`README.md` + `config.json::criterion`）先写，后跑。
- 2026-10-14：先实测本机 CPU 成本（load 0.9 s；`16×256` bpc 批 ≈1.8 s；copy 批 ≈2.5 s；
  `synth_accuracy` 的 induction/var_recall 因拼接后 `B=256` 需 4 个 chunk ⇒ 单腿 >100 s）⇒
  **不跑 induction/var_recall 两条副腿**（登记为未测项 + 下一步成本），只跑 copy 主腿 + 毁源对照（3 臂）。
  本决定在跑之前落进 `config.json::legs_per_arm`。
