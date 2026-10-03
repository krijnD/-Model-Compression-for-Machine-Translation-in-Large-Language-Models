"""Split the generate() wall into its two per-forward terms, under the real inference path.

The batch-size probe showed 2.31x per-sentence speedup from BATCH 4->16, which rules out my
earlier "dequant is a fixed 12.8%" reading. To decide whether to raise BATCH for the running
eval jobs we need the absolute in-generate dequant cost and the matmul cost, measured the way
generate() actually calls them (no_grad/inference_mode, bf16 activations, warm Triton).

Measures, per quantized linear, on a real batch:
  dequant_ms : m.dequantize_weight() once
  matmul_ms  : torch.matmul(x, W) once for the same activation batch
"""
import argparse, statistics, sys, time
from pathlib import Path
import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear

HERE = Path(__file__).resolve().parent          # francesco/analysis
FRANCESCO = HERE.parent                         # francesco/
JSON = FRANCESCO / "results" / "json"           # where these scripts write their results

p = argparse.ArgumentParser()
p.add_argument("--model", default="/gpfs/home6/scur0517/models/ALMA-13B-R-gptq-w2g128")
p.add_argument("--batch", type=int, default=80, help="rows of activation = batch*num_beams")
p.add_argument("--reps", type=int, default=20)
p.add_argument("--out", default=str(JSON / "units_w2.json"))
a = p.parse_args()

model = GPTQModel.load(a.model, device="cuda:0", backend=BACKEND("auto"), torch_dtype=torch.bfloat16)
model.eval()
mods = [m for m in model.model.modules() if isinstance(m, BaseQuantLinear)]
print(f"quant_linears={len(mods)} kernels={sorted({type(m).__name__ for m in mods})}", flush=True)

big = max(mods, key=lambda m: m.in_features * m.out_features)
x = torch.randn(a.batch, big.in_features, device="cuda:0", dtype=torch.bfloat16)
print(f"widest linear: in={big.in_features} out={big.out_features} rows={a.batch} "
      f"({big.in_features*big.out_features/1e6:.1f}M weights)", flush=True)

def bench(fn, reps):
    with torch.inference_mode():
        for _ in range(3): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

dq = bench(lambda: big.dequantize_weight(num_itr=1), a.reps)
W = big.dequantize_weight(num_itr=1).to(torch.bfloat16)
mm = bench(lambda: torch.matmul(x, W), a.reps)
print(f"dequant one {big.in_features}x{big.out_features} matrix: {dq:.2f} ms (median of {a.reps})", flush=True)
print(f"matmul  [{a.batch}x{big.in_features}] @ [{big.in_features}x{big.out_features}]: {mm:.2f} ms", flush=True)
print(f"ratio dequant:matmul = {dq/mm:.2f}", flush=True)
import json
Path(a.out).write_text(json.dumps({"model": a.model, "batch_rows": a.batch,
    "in_features": big.in_features, "out_features": big.out_features,
    "dequant_ms": dq, "matmul_ms": mm, "ratio": dq/mm}, indent=1))
print(f"wrote {a.out}", flush=True)
