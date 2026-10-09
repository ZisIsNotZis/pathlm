# controls — 2520-spec-at-scale 的对照臂

本实验是**纯推理双路径对照**，对照臂与主臂在**同进程内逐次交替**测量。

| 对照 | 是什么 | 输入分布 | 预测什么 | 部署对应 |
|---|---|---|---|---|
| **native AR（对照臂）** | `decode()`：逐 token 调 `Decoder.step()` | 同 prompt（random 32 token / enwik8 val 前 32） | 贪心下一个 token | 单流、无草稿的自回归解码 |
| spec（主臂） | `decode_spec()`：`step_multi([v1]+drafts)` 批量验证 | 同上 | 同上 | 带 MTP 草稿的多 token 批量验证 |
| **解析算力参考（非运行臂）** | `R = (1+γ)/tokens_per_forward` | γ、tokens_per_forward | 每 token 算力 | 任何部署的算力下限 |

**可比性三要素**（SCHEMA §13）：

1. **输入分布**：两臂逐位相同（同 prompt 张量、同 `Decoder(window, anchors)`）。
2. **要预测什么**：两臂都产出 400 个贪心 token，且发出流逐位相同（除 top-2 gap<1e-3 的 fp 平局，
   本次两 prompt × 两 k 均**无**翻转，`greedy_divergence: null`）。
3. **部署对应**：native = 无草稿单流解码；spec = MTP 草稿 + 批量验证。

**为什么 native 不用批量 teacher-forcing**：teacher-forcing 会把真值喂进模型（oracle 式），
与部署路径不同；本实验的对照必须是**自循环**对手（SCHEMA §11），故 native 走 `Decoder.step()`。

**方差**：贪心 ⇒ 确定性量（tokens/forward、接受率）run-to-run 方差 = 0；
墙钟为同进程 5 次重复，报极差，不设跨进程阈值（外部占用下不可比）。
