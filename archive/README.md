# archive — 旧项目（只读）

本目录保存 **PathLM**（2026-09-14 ~ 09-29，25 个执行 ticket）的原始文档。
它的结论**已经被消化**进 `axes/` 的各课题页；这里保留原始叙述与逐 run 细节，
供回溯与引用。

## ⚠️ 使用前必读：旧数字的口径陷阱

1. **所有早期公布的税是 1.1-epoch 上界**。渐近口径（4.4 epoch）下回环族税
   从 +0.126 收到 +0.082。不要拿旧税与新实验直接比较，除非口径相同。
2. **跨引擎对照会让归属不可分**。ticket 24 的 +0.4563 直到 24b 用两臂
   （等权臂 + 同引擎 dense）拆完才变成干净归属（形态 96% / 引擎 4% / 配方 ≈0）。
3. **Δ < 0.014 不可排序**（种子散布 ≤0.0138）。retry 矩阵的全部排序结论
   已被撤回。
4. **`docs/` 在本仓库已重新定义为新契约**（`design.md` / `methodology.md` /
   `playbook.md`）；旧 `docs/` 的内容整体在此处，文件名不变。

## 文件导航

| 文件 | 内容 |
|---|---|
| `legacy-docs/retrospective.md` | 全轨迹叙事（起点 → 两次转向 → 规模判决 → 归因收口） |
| `legacy-docs/findings.md` | 逐机制结论账本（SSOT，每条带数字与证据路径） |
| `legacy-docs/report.md` / `report_details.md` | 结论层 / 逐实验审计层 |
| `legacy-docs/artifacts.md` | 25 个 ticket 的 run 级证据索引 |
| `legacy-docs/mental_model.md` | 设计哲学演化（回环时代 → 深度自适应 AR） |
| `legacy-docs/design.md` | 回环时代的冻结架构（历史文档） |
| `legacy-docs/experiments.md` | 回环时代的测量程序 |
| `legacy-docs/WORKSPACE-2026-09-29.md` | 收口时的项目状态快照（含 backlog） |
| `legacy-docs/project-status.md` / `people.md` | 项目分类与协作备注 |

原始证据（`results.json` / `SUMMARY.md` / 探针输出）仍在 `.scratch/NN-slug/evidence/`，
见 `.scratch/README.md`。**权重（`*.pt`）不入库**——证据 json/log 足以复现结论，
需要权重时按 ticket 契约里的命令重新训练。

## 代码

历史引擎与探针**原地保留**（未搬迁）：`pathlm/`、`tests/`、`train_*.py`、
`probe_*.py`、`slider.py`、`configs/`。搬迁计划见 `docs/design.md` §5。
