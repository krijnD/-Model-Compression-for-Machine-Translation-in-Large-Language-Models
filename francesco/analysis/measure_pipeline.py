"""Attribute the real generate() wall by direct perturbation - no assumptions.

measure_units.py timed dequant (0.15 ms) and matmul (0.08 ms) on the widest linear in isolation.
Scaled to the model's 179 such units over ~30 forward passes that predicts ~0.23 s per input
sentence at BATCH=4, but the pipeline actually takes 4.756 s. So the isolated timings are not the
in-pipeline costs (Triton warm/compiled state, dtype casts, the .to(x.dtype) copy, and the fact
that generate() is not a sequence of independent linears - it interleaves attention and the
KV-cache). This script stops modelling and measures the split by perturbation instead:

  A) full                        - baseline wall
  B) dequantized weights cached  - replaces the kernel's per-forward dequant with the same matmul
                                   on pre-dequantized bf16 weights -> isolates ALL dequant cost
  C) dequant-only model          - computes dequant and discards it (no matmul) -> the floor cost

By construction A = B + dequant_cost, so (A - B) is the dequant share of the wall with no
extrapolation. If B alone reproduces the 4->16 speedup, the whole win is matmul efficiency and
larger batches have nothing left to give; if A-B is large, batch amortization explains it.
"""
import argparse, statistics, sys, time
from pathlib import Path
import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
HERE = Path(__file__).resolve().parent          # francesco/analysis
FRANCESCO = HERE.parent                         # francesco/
REPO = FRANCESCO.parent                         # the repository root
JSON = FRANCESCO / "results" / "json"           # where these scripts write their results
sys.path.insert(0, str(REPO / "scripts"))
from alma_prompt import get_prompt, load_test_sources
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="/gpfs/home6/scur0517/models/ALMA-13B-R-gptq-w2g128")
p.add_argument("--pair", default="de-en")
p.add_argument("--batches", default="4,16")
p.add_argument("--max-source-length", type=int, default=256)
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--num-beams", type=int, default=5)
p.add_argument("--out", default=str(JSON / "pipeline_w2.json"))
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left", add_eos_token=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = GPTQModel.load(a.model, device="cuda:0", backend=BACKEND("auto"), torch_dtype=torch.bfloat16)
model.eval()
mods = [m for m in model.model.modules() if isinstance(m, BaseQuantLinear)]
print(f"quant_linears={len(mods)}", flush=True)
model.model.generation_config.max_length = a.max_source_length + a.max_new_tokens
kw = dict(max_new_tokens=a.max_new_tokens, num_beams=a.num_beams, use_cache=True,
          do_sample=False, temperature=None, top_p=None)
src, tgt = a.pair.split("-")
prompts = [get_prompt(src, tgt, s) for s in load_test_sources(src, tgt)]

def enc_for(b):
    return tok(prompts[:b], max_length=a.max_source_length, padding="max_length",
               truncation=True, return_tensors="pt").to("cuda:0")

def wall(b, reps=2):
    e = enc_for(b); ts = []
    with torch.inference_mode():
        _ = model.generate(**e, **kw)                    # warm
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = model.generate(**e, **kw)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    del e; torch.cuda.empty_cache()
    return statistics.median(ts)

res = {}
for b in [int(x) for x in a.batches.split(",")]:
    A = wall(b)
    res[f"full_b{b}"] = round(A, 3)
    print(f"BATCH={b:3d} A) full wall             = {A:7.3f} s   ({A/b:.3f} s/sentence)", flush=True)

# ---- B/C: dequantized-weight variants, built once and reused so memory is bounded ----
# 2-bit: 12.69G weights in quantized Linears -> bf16 = 25.4 GiB, too much next to the packed
# copy plus 80-sequence activations. Convert the widest half (by weight count) and measure the
# ratio on those, then report the achieved share.
saved = []
for m in sorted(mods, key=lambda m: -(m.in_features * m.out_features)):
    m._probe_W = m.dequantize_weight(num_itr=1).to(torch.bfloat16)
    saved.append(m)
    if sum(s.in_features * s.out_features for s in saved) > 0.55 * sum(m.in_features * m.out_features for m in mods):
        break
cov = sum(s.in_features * s.out_features for s in saved) / sum(m.in_features * m.out_features for m in mods)
print(f"cached dequantized weights for {len(saved)}/{len(mods)} linears = {cov*100:.0f}% of weights", flush=True)

orig = {}
def cached_forward(self, x):
    out_shape = x.shape[:-1] + (self.out_features,)
    xr = x.reshape(-1, x.shape[-1])
    W = getattr(self, "_probe_W", None)
    if W is None:
        return orig[id(self)](x)
    return torch.matmul(xr, W).reshape(out_shape).to(x.dtype)

for m in mods:
    orig[id(m)] = type(m).forward
for m in saved:
    m.forward = cached_forward.__get__(m, type(m))

for b in [int(x) for x in a.batches.split(",")]:
    B = wall(b)
    res[f"cached_b{b}"] = round(B, 3)
    print(f"BATCH={b:3d} B) dequant cached       = {B:7.3f} s   ({B/b:.3f} s/sentence)", flush=True)

# C) dequant-only cost for the cached subset, as a share, scaled by coverage
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.inference_mode():
    for m in saved:
        _ = m.dequantize_weight(num_itr=1)
torch.cuda.synchronize()
one_pass = time.perf_counter() - t0
res["dequant_one_pass_ms"] = round(one_pass * 1000, 2)
res["coverage"] = round(cov, 3)
print(f"C) one full dequant pass over the cached subset = {one_pass*1000:.1f} ms "
      f"(= {one_pass/cov*1000:.1f} ms scaled to the whole model)", flush=True)

res["model"] = a.model
Path(a.out).write_text(__import__("json").dumps(res, indent=1))
print(f"wrote {a.out}", flush=True)
