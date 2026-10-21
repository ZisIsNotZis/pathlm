# 2692 · 权重哈希（`model.pt` 不入库：`.gitignore` 覆盖 `*.pt`；本文件只记哈希与大小）

| 臂 | 参数量 | `model.pt` 字节 | sha256 | 与 `results.json` 记的一致 |
|---|---|---|---|---|
| `dense6` | 5,251,584 | 21,030,185 | `81de143402d61e2cf94509956ea17ab47dd510c9c61d5d9bc8bc8bb67d835f75` | 是 |
| `dense12` | 10,384,896 | 41,586,187 | `a9e07322d5502efe9b94dbaf1d945f847f9730a115c4006cb33a559610e1d2cd` | 是 |
| `repeat2x` | 5,251,584 | 21,030,185 | `1ae97fff8710f8134c188c549e3268658f9f02df1475dcaaa9e8ed33855ffad2` | 是 |
| `repeat2x-tail` | 5,251,584 | 21,030,185 | `0e32a16b8b935c8f2f0532d4e90bdcadf1d56ee1f080880180f185312fffc220` | 是 |

权重是派生物：seed 1 重训应得到同向读数（不保证逐位复现）；本表只用于核对「盘上这份 checkpoint 就是产生 `results.json` 的那份」。
