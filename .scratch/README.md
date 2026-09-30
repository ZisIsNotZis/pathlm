# .scratch — 历史工单与原始证据（只读）

本目录是 PathLM 项目（2026-09-14 ~ 09-29）的**历史工单区**，采用
Matt Pocock 的 ticket 布局：

```
NN-slug/
├── issues/     执行契约（命题 / 干预 / 判据 / 对照 / 预算 / 时间线）
└── evidence/   原始证据（results.json / SUMMARY.md / 探针输出 / 逐 run 目录）
```

## 现在怎么用

- **回溯与引用**：`axes/` 里每个课题页的 `evidence` 字段直接指向这里的
  `.scratch/NN-slug/evidence` 路径 + commit。
- **不要在这里继续做研究**。新研究在 `axes/<轴>/<格子>/experiments/` 下开目录
  （见 `SCHEMA.md` §3）。本目录只增不改。
- **目录很大**（约 20G）：绝大部分是未入库的权重与日志（`*.pt` 被 gitignore）；
  入库的是证据 json/log/md。

## 索引

- `docs/artifacts.md`（已归档到 `archive/legacy-docs/artifacts.md`）——
  25 个 ticket 的 run 级证据索引表。
- 26 号 ticket（`26-closing/`）是收口记录。
- 27 号起是本仓库改造为 Research Wiki 的工单。

## 术语

- `ticket 23–25` = 深度自适应 AR 形态的机制验证 / 规模判决 / 归因收口。
- `ticket 01–22` = 回环时代（retry / corruption / 层路径 / spec / Slider）。
- 所有旧口径警告见 `archive/README.md`。
