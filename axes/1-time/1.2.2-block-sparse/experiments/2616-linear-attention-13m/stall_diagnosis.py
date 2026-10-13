"""诊断：线性注意力训练在 ~500 步卡死（backward 内）是否由 clamp 阈值引起。

用法：python3 /tmp/bisec.py <chunk> <eps> <steps>
"""
import sys

sys.path.insert(0, "/home/z/vibe/pathlm")
import pathlm.depth_ar as da  # noqa: E402

chunk, eps, steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
da.LINEAR_ATTN_EPS = eps
import train_depth_ar as T  # noqa: E402

sys.argv = ["train_depth_ar.py", f"bisec-c{chunk}-e{eps:g}",
            "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
            "--mlp-mult", "4", "--dense", "--attn-linear",
            "--linear-chunk", chunk, "--steps", str(steps),
            "--batch-size", "32", "--seed", "0", "--eval-batches", "1",
            "--out-root", "/tmp/bisec"]
T.main()
