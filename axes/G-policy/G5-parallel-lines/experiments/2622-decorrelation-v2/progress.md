# progress · 2622 去相关 v2（ticket 39 X1）

> 状态：**在跑**（每步更新；「在跑 / 判过了 / 被否证了」）
> 契约与预登记判据：同目录 `README.md` §4（**跑完不得改**）。

## 时间线

- [x] 读契约 `.scratch/39-close-gaps/issues/01-close-gaps.md`（X1）+ `SCHEMA` §14b/§17.14/§17.15/§18/§19/§20+20.1/§21/§22+22.8/§23
- [x] 能力面分析（§22.8）：所需组合在 loop-era 引擎存在（depth-AR 引擎的互斥校验不适用）；新增 `validate_pcap` 护栏
- [x] 实现：`pathlm/config.py`（`w_repr_decorr`）、`pathlm/model.py`（`repr_decorr_penalty` + 接线）、`train_m1.py`（`sample_rounds`/`gate_skip`/`validate_pcap`/活性量）
- [x] 单测 `tests/test_decorrelation_v2.py`（12 条，CPU，绿）
- [x] 变异验证 `mutate.py`（6 个变异体全部抓红且逐字节还原）
- [x] 基线 `pytest -q`：**367 passed**（ticket 写的 361 是旧值；本批开工时实测 367）
- [ ] 冒烟（400 步 × 5 臂）+ 步时读数
- [ ] 13M 训练（stock/l1/repr）
- [ ] 13M 评测
- [ ] 50M 训练（l1/repr；stock 复用 2620）
- [ ] 50M 评测
- [ ] 对账 + `analysis.md` + `MODEL.md`
- [ ] 终局 `pytest -q` 全绿
- [ ] 提交

## 关键决策记录

- **λ（w_repr_decorr）= 1.0**：罚则是 [0,1] 归一化的跨路 Pearson 相关（同路径=1.0）；w=1 使其
  initial loss 贡献 ≈ 0.8 nats（≈初值 CE 的 6%）。**理由**：归一到 [0,1] 后 w=1 是「单位权重」，
  不是一个调出来的数；smoke 检查活性与稳定性。校准读数见 `analysis.md`。
- **50M 只训两臂（l1/repr），stock 复用 2620 检查点**（同配置/seed/token，gate_skips=0 ⇒ 等价），
  按契约「两臂」执行；reused 已显式记账。
- **gate 修复**：只在非有限梯度时跳（`clip_grad_norm_` 已处理有限大梯度）。
