"""CPU-only 评测成本探针（ticket 66 Phase B）：GPU 锁被占时的 fallback 可行性测量。只读。"""
import importlib.util, json, os, sys, time
ROOT = "/home/z/vibe/pathlm"
P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
sys.path.insert(0, ROOT); sys.path.insert(0, P2621)
spec = importlib.util.spec_from_file_location("b21", os.path.join(P2621, "run.py"))
b21 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b21)
from pathlm.data import load_enwik8_full
cfg = json.load(open(os.path.join(P2621, "config.json")))
t0 = time.time()
model, ck = b21.b19.load_trunk(cfg, "cpu")
t1 = time.time()
_, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, "data/enwik8_full.npz"))
t2 = time.time()
fn = b21.b19.make_logits_fn(model, {"kind": "quant", "mode": "fp16"}, 256, "cpu")
b = b21.b19.bpc_of(fn, eval_arr, 256, 2, 16, 2, "cpu")
t3 = time.time()
print(f"load_model={t1-t0:.1f}s load_data={t2-t1:.1f}s bpc(2batches)={t3-t2:.1f}s -> {b:.6f}")
print(f"per-batch={((t3-t2)/2):.1f}s ; est 24x16 main eval = {((t3-t2)/2)*24:.1f}s ; est 8x16 = {((t3-t2)/2)*8:.1f}s")
