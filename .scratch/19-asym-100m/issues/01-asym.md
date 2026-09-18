# 19 — 4.4-epoch 渐近口径 @100M：税上界定案

- **Status:** done（渐近定案：税 +0.082，随规模下降；溢价随训练扩大）
- **Type:** scale study（长训）
- **Related:** 16-rung4-scale（100M 1.1-epoch 税 +0.126）、11（D5：13M 税随步数
  收缩 +0.364@1k → +0.122@24k，~3k 平台；"所有已发布绝对税是 1.1-epoch 上界"）
- **目标：** Rung-4 头条结论"税首次不随规模增长（+0.126）"目前是 1.1-epoch
  快照。13M 的先例表明税在训练中收缩——100M 渐近口径下税是多少？
  若收缩 → "税不增"更强化；若不缩（或反向）→ 缩放定律叙事修正。

## 协议

1. **训练**：B0_100M_asym / C1_100M_asim × 2 seeds，96600 步 × batch 8
   （= 4.4 epoch，396M tokens），cosine 全程；--ckpt-every 16000 供税-步数曲线。
   预计 ~2.3h（B0）/ ~3.5h（C1）每 run，4 runs 串行 ~12h。
2. **测量**：终点 clean bpc（税 = C1−B0，2 seeds）；逐 ckpt 税-步数曲线
   （probe_e2e 符号链接法，6 ckpt × 4 run）；对照 13M D5 轨迹。
3. **判据**：渐近税 vs +0.126（1.1-epoch）；D5 收缩模式是否在 100M 重现。

## 产出

`.scratch/19-asym-100m/evidence/`；findings Scale 节补渐近行；report §4.6；
WORKSPACE；commit。
