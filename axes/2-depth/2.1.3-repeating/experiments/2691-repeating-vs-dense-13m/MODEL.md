# 2691 · 权重哈希（`model.pt` 不入库：`.gitignore` 覆盖 `*.pt`；本文件只记哈希与大小）

| 臂 | 参数量 | `model.pt` 字节 | sha256 | 与 `results.json` 记的一致 |
|---|---|---|---|---|
| `dense6` | 5,251,584 | 21,030,185 | `9d7e5cb41feffd4d4c13977ebad3df03e991d6efc3d3b3555bb40810a3e42983` | 是 |
| `dense12` | 10,384,896 | 41,586,187 | `3757907949851828cc2e97a7ac1e71c91098ec1e41b13708e1efdaa8d67aeeac` | 是 |
| `repeat2x` | 5,251,584 | 21,030,185 | `6f70d8fec94ebdf6de90e6a12e63e5d0f7fa7ebb9e526d9555663391a799d933` | 是 |
| `repeat2x-tail` | 5,251,584 | 21,030,185 | `d1d769600e822ea6406460dcd66c0b629f19e6303de89893a87d96a41d9c6c37` | 是 |

权重是派生物：重训应得到同类读数（单 seed，不保证逐位复现）；本表只用于核对「盘上这份 checkpoint 就是产生 `results.json` 的那份」。
