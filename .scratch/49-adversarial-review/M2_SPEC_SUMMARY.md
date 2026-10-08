All checks complete; the line is committed and the workspace is clean. Final summary below.

## 人话摘要

我接续了因网关 401 中断的 **Spec 三臂线**（`axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/`）。之前只有契约、代码、冒烟脚本，**训练和评测都没跑**。这一波把它跑完了：先测步时定预算，训练 3 个 checkpoint（13.4M×2 seed + 53.9M×1 seed），再用生产解码器 `Decoder.step()` 做 native 对照、同进程交替测墙钟，交出六项指标。

**结论**：同一个 token 条件链模型上，**直连多 token 头（direct）最好；概率混合（combined，λ=0.5）与它打平、不可排序；条件链（chain，"先出 N+1 再以其为条件出 N+2"）显著更差**——α₁ 低 0.16–0.24，53.9M 上墙钟 1.39× vs 1.58×。三臂发出流与原生贪心**逐位相同**（门禁 9/9），墙钟加速比 95% CI 下界**全部 > 1.0**。这复现并强化了 ticket 12 的 deploy 结论，并取代 2613 回答 H3 命题（复核 A6 的 `T∘T` 反例已在实现层用单测+变异钉死）。

**六项指标**（以 53.9M 为例）：① α₁/α₂|₁ 逐位置 = 0.844/0.872（direct）、0.829/0.853（combined）、0.657/0.826（chain）；② τ = 1.58 / 1.54 / 1.20；③ tok/fwd = 2.58 / 2.54 / 2.20；④ 墙钟加速比 = 1.58 / 1.57 / 1.39（含 6 次重复极差与 prompt 聚类 CI）；⑤ 等效算力净账 = 0.48 / 0.49 / 0.56（含草稿头成本，头占主干 3.7%）；⑥ **argmax 路线 ⇒ 声明非分布保持**，给更强的逐位等价。

**Implemented** the R6 Spec three-arm experiment end-to-end (train 3 checkpoints + locked evaluation + six metrics + full accounting), fixed latent never-exercised bugs in `run.py`/`reconcile_numbers.py`, and committed only my experiment directory.

**Changed files**: 30 files under `.../2631-spec-three-arms/` (code: `spec_arms.py`, `run.py`, `reconcile_numbers.py`, `eval.sh`; artifacts: `results.json`, `cycle_log.jsonl`, `run_log.txt`, `eval.runlog`, `reconcile_report.json`, `mutation_report.json`, `evidence/micro_c3.json`, `smoke/steptime.md`; docs: `analysis.md`, `MODEL.md`, `README.md`, `controls/README.md`; evidence: `train/*/results.json`, `train/*/train_log.jsonl`, smoke outputs). `catalog.yaml`/`INDEX.md` untouched (main agent's `TODO`).

**Validation**: `pytest tests/ -q` **656 passed** (baseline advanced 641→656 by another line's new test file; my dir unchanged by it); `check_schema.py` OK; `mutate.py` 7/7 caught + byte-identical; `reconcile_numbers.py` **141/141 PASS** (139 derived / 2 golden); `test_spec_arms.py` 15 passed; native path confirmed `Decoder.step()` (`native_is_plain_decoder_path=True`).

**Open risks**: (1)共享卡全程有外部 `llama-server`（util 77–100%），墙钟加速比**被外部占用抬高**（spec 每前向吐更多 token ⇒ 更少排队），绝对水平只作该水位快照、**禁外推**；相对排序在两次独立评测中一致。(2) `c₃` 测量噪声 ±15%（13m_s1 那次被占用压低到 1.05，已在 analysis §5 披露）。(3) 53.9M 只 1 seed；γ=2、温度 0；采样/树状/多级条件未测（下一步+成本见 analysis §7）。按 §20.2，53.9M 峰值 6237MB 超 5GB 礼貌线 24%，优先配方一致并三件套记账。

**Recommended next step**: 主 Agent 更新 `H3-composition` 的 catalog `verdict`（用本实验取代 2613）；若要空卡墙钟，等外部占用消退后重跑 `eval.sh`（~27 min）。