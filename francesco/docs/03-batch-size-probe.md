# Batch-size probe for slow GPTQ bit-widths (2-bit / 3-bit generation)

> *Moved to `docs/` on 2026-09-30, content unchanged except updated doc names. Paths in this file are relative to `francesco/` (`../` = the repo root), as when it was written. Start at `../README.md`.*

## Layout

```
francesco/
  README.md            this report
  02-cost-and-memory.md  measurement 5 (cost of each precision vs fp16) as a stand-alone
                         note: numbers, mechanism, speed levers, open decisions, pending jobs
  04-kernel-repack.md  lossless w3/w2 -> 4-bit repack for the fused kernel; §8 = measured speed + drift
  06-distillation.md      w3 + LoRA distilled from fp16 (lora.py, distill_lora.py, generate_lora.py,
                         sbatch/distill.job, eval_distill.job, eval_w3as4.job)
  analysis/            the code that produces the numbers
    run_probe.py         measurement 1 (batch sweep) and 4 (zh-en); driven by sbatch/
    measure_units.py     measurement 2: per-linear dequant vs matmul
    measure_pipeline.py  measurement 3: perturbation attribution of the real pipeline
    make_report.py       builds results/tex/*.tex and results/figures/* from <project>/outputs
    measure_quant_cost.py  measurement 5: peak VRAM, load, latency per checkpoint, next to fp16
    measure_channels.py  not part of this study: RQ5 in ../../reasearch.md, per-channel activation
                         energy per language on the fp16 model (no quantization, one forward pass)
    sbatch/              probe.sbatch, units.sbatch, pipe.sbatch, probe_zh.sbatch, cost.sbatch,
                         probe16.sbatch, backends.sbatch, measure_channels.sbatch
  results/             everything produced
    results.pdf          the compiled report
    tex/                 results.tex (entry point) + generated tables.tex, cost.tex, figures.tex,
                         detail.tex
    figures/             fig*.pdf (used by figures.tex) and fig*.png copies
    json/                probe_*.json, probe16_*.json, quant_cost_*.json, units_w2.json,
                         pipeline_w2.json, channels_fp16.npz (RQ5)
    quant_cost_summary.md  generated markdown copy of the cost tables (pasted into this file)
```

`python francesco/analysis/make_report.py` regenerates the `.tex` chunks and the figures, then
`pdflatex -output-directory=.. results.tex` twice inside `francesco/results/tex/` rebuilds
`results/results.pdf`. The `-output-directory=..` is what keeps the PDF in `results/` (and drops the
throwaway `.aux`/`.log`/`.out` there too, delete them after); without it the PDF lands in `tex/`.
Every path is derived from the script's own location, so the `\input{}` chunks and the
`../figures/` graphics resolve wherever the checkout lives.

**Status (2026-09-25).** The grid this report analyses has since finished: all four quantized runs
are scored (`../outputs/baseline/gptq-w{2,3,4,8}g128.tsv`), so the two tables below are the picture
*during* the probe and are kept as written. w2 did overrun its 20 h wall — job `27059677` was
terminated inside `zh-en` at 1551/1875 — and was carried to completion by job `27109859`, i.e. via
`Requeue=1` (fallback **a**), not the `BATCH_LONG=4` rescue (**b**). The analysis and the batch-size
recommendation below are otherwise unrevised; nothing was re-run.

**Added after that (measurement 5, below).** The grid is scored, so the same five checkpoints have now
also been measured on the *cost* side — peak VRAM, load time, latency and GPU-hours, with fp16 as the
reference and `BATCH=4` vs `16` for all five. It changes the recommendation's framing: on this stack
weight-only GPTQ buys memory and not time (fp16 is the fastest of the five per line at equal batch),
the low-bit rows only look cheaper in the grid's GPU-hour ledger because the fp16 baseline job ran
four ranks per node, and the batch lever is what turns the saved memory back into throughput. See "Measurement 5" below, the
"What the quantization costs and buys" section of `results/results.pdf`, and
`results/quant_cost_summary.md`.

**Question:** is `BATCH=4` in `scripts/snellius/eval_quantized.job` leaving an easily-collected
speedup on the table for the 2- and 3-bit runs, or is their cost structural?

**Answer: batch size is a real, immediately collectable lever — 2.31× per input sentence going
`BATCH=4` → `BATCH=16` on the 2-bit model, and 3.49× going `BATCH_LONG=1` → `4` on the zh-en path
(source length 512), which is the most expensive direction and the worst-amortized one. Roughly a
third of the 4→16 win comes from amortizing a fixed per-forward dequantization cost and two thirds
from the rest of the forward pass growing sublinearly with batch (the H100 is nowhere near saturated
at 20–80 rows).**

Where it stands for the grid in flight, measured from each job's own per-direction log rather than
projected from a single rate:

| run | generation | vs 20 h wall (incl. ~70 min scoring) |
|---|---|---|
| `gptq-w4g128` | 4.5 h | ✅ **completed** 21:01, all 5 stages, all 8 metrics × 11 rows |
| `gptq-w8g128` | 6.6 h | ✅ **completed** 23:09, all 5 stages |
| `gptq-w3g128` | ~13.4 h | ✅ fits, ~14.6 h total |
| `gptq-w2g128` | ~19.8 h | ❌ **overruns: ~21 h total** — zh-en alone measures 10.14 s/line at `BATCH_LONG=1`, so its remaining ~5.3 h of zh-en pushes generation to ~19.8 h and the scoring tail past the wall |

So for w2 the lever is now **the rescue, not insurance**: the same work at `BATCH_LONG=4` is ~1.5 h
instead of ~5.3 h, which brings w2 to ~17 h total and inside one window. `Requeue=1` plus the
scripts' `todo()`/`done_metric()` guards would also carry it across two windows unaided.

All numbers below are measured on the cluster, not modelled. Jobs `27066494`, `27066642`, `27066741`,
`27074471` on `gpu_h100` (one empty H100 each, alongside the running eval jobs).

## The problem

Four jobs are evaluating ALMA-13B-R quantized by GPTQ at 2, 3, 4 and 8 bits on WMT'22, same protocol
(beam 5, bf16, seed 42, `BATCH=4` short / `BATCH_LONG=1` zh-en) via `scripts/generate_quantized.py`.
Generation times, now measured per direction from each job's own log (10 directions, 17,471 lines):

| run | job | bits | kernel (GPTQModel auto-select) | s/line | generation | state |
|---|---|---|---|---|---|---|
| `gptq-w4g128` | 27059681 | 4 | `ExllamaV2QuantLinear` | 0.94 | 4.5 h | ✅ completed 21:01, all 5 stages |
| `gptq-w8g128` | 27059688 | 8 | `TritonV2QuantLinear` | 1.37 | 6.6 h | ✅ completed 23:09, all 5 stages |
| `gptq-w3g128` | 27059680 | 3 | `TorchQuantLinear` | 2.39 | 13.4 h (projected) | running, 8/10 directions |
| `gptq-w2g128` | 27059677 | 2 | `TritonV2QuantLinear` | 3.31 | 19.8 h (projected) | running, 5/10 directions |

Surveying every kernel in `GPTQModel@main` showed **no 3-bit accelerated kernel in any maintained
loader**: Marlin/Machete/Swordfish/ExLLamaV2/BitBLAS/`torch_int8` cover only 4/8 or 2/4/8, or are
CPU-only/Blackwell-only, and `tritonv2 SUPPORTS_BITS = [2,4,8]` at 4.2.5, 5.6.0, 5.8.0, 6.0.3 and
7.5.0 — even though the corrected `10-1-10-1-10` 3-bit decoder has shipped in `dequant.py` since
5.8.0, present but gated off. So 3-bit necessarily runs `TorchQuantLinear` and 2-bit gets
`TritonV2QuantLinear`; no released wheel or sdist changes that.

**Correction (added after a reader question).** This is *not* the same as "no fused 3-bit GPTQ kernel
exists", which is what an earlier version of this file implied. The original GPTQ repository
(`IST-DASLab/gptq`) is in fact *3-bit specialised*: its only kernel source is
`quant_cuda_kernel.cu`, containing `VecQuant3MatMulKernel` and `VecQuant3MatMulKernelFaster`, and its
README advertises "a 3-bit quantized matrix full-precision vector product CUDA kernel" plus "optimized
3bit kernels ... e.g. 1.9x -> 3.25x generation speedup for OPT-175B" on A100 via `--faster-kernel`.
Those kernels are **fused** — they read packed int words, shift/mask to 3 bits, multiply by the vector
element and accumulate in registers (`atomicAdd` to the output), with no intermediate dequantized
weight buffer, which is exactly the property `TritonV2QuantLinear` and `TorchQuantLinear` lack.

Verified while checking: the original kernel's 3-bit packing is bit-identical to GPTQModel's, so it
would read our checkpoints correctly element-for-element. Both implement GPTQ's `10-1-10-1-10` scheme
(32 values across 3 int32 words), including the two cross-word fixups
(`(w0 >> 30) | ((w1 << 2) & 0x4)` and `(w1 >> 30) | ((w2 << 1) & 0x6)`); transcribing both decoders
and comparing 2000 random 3-word groups gives identical output every time (0 mismatches). This also
re-confirms that GPTQModel's Torch path (which implements this layout) is the correct one and the
gated-off Triton 3-bit path (linear bitstream) was the outlier — consistent with upstream issue #2251.

Using it anyway is a porting project, not a switch, and it would not help the grid in flight:

| blocker | detail |
|---|---|
| matrix-**vector**, not batched | `VecQuant3MatMulKernel` takes one vector × the matrix; our forwards are 20 rows (batch 4 × 5 beams), so it needs 20 launches per linear per forward. The current path is not bandwidth-bound (the whole 3-bit model's packed weights are 4.47 GiB, vs a ~1.7 s forward), so this is not automatically fatal — but it is 20× the weight traffic and no longer a drop-in. |
| act-order | our checkpoints use `desc_act=True` with `static_groups=false`; the original repo's own README notes `--act-order` "does not require any inference changes" only together with `--static-groups`, so the kernel's per-column `scales[col]`/`zeros[col]` indexing does not match our grouped/act-ordered tensors without more work. |
| toolchain | written for torch ~1.10 / CUDA 11.4 (`vec.type()`, `mat.data<int>()` — both removed in torch 2.x); our venv is torch 2.8.0+cu128 with CUDA 12.6, so it needs porting and a build before it can even load. |
| headline does not transfer | the 3.25× figure is against **fp16** inference for OPT-175B at bs=1 on A100 — not against our `TorchQuantLinear` fallback for ALMA-13B on H100 at batch 4, which is what it would have to beat. |

The perturbation measurement bounds what any fused kernel could recover: it removes exactly the cost
that disappears when the per-forward dequant is replaced by a cached bf16 matmul, which is 26.6 % of
the wall at `BATCH=4` (34 % of the total speedup available from batching). A fused kernel would
additionally avoid the dequantized-weight buffer write+read, so that 26.6 % is a lower bound, not a
ceiling — but there is no measured path to the 3.25× headline here, and w3 is 8/10 directions done,
so this is future work rather than a fix for the current run.

That left one lever that does not require re-quantizing or changing stack: **the batch size.**

## Competing hypotheses, and why the question was genuinely open

Reading the installed kernels (`venv-quant/…/gptqmodel/nn_modules/qlinear/`, GPTQModel 4.2.5):

```python
# triton_utils/dequant.py
def quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=False):
    W = dequant(input.dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
    return input @ W                    # dequantize, THEN matmul - not fused

# qlinear/torch.py
def _forward(self, x, out_shape):
    weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
    out = torch.matmul(x, weights).reshape(out_shape)      # same structure
```

Despite the name, `TritonV2QuantLinear` is **not** fused: it dequantizes the whole weight matrix
with a Triton kernel, then calls `input @ W`. Nothing caches the dequantized weights
(`tritonv2.py:145` → `QuantLinearFunction.apply` → `quant_matmul`; no `_cached_W` anywhere). So every
forward pass, in every quantized linear, re-dequantizes the full matrix — a cost **independent of
batch size**, while the matmul scales with `batch × num_beams`.

That makes batch size a plausible amortization lever. A first static estimate said it was worth
~12.8 % of the wall (2.43 s measured dequant / 19.0 s measured forward), which would have made it
marginal. Both the estimate and the conclusion turned out wrong, and the measurements below explain
why — the static reading mistook *which* term dominates and mis-measured the dequant.

## Measurement 1 — the thing being decided (`run_probe.py`, job 27066494)

Model `gptq-w2g128`, pair `de-en`, `max_source_length=256`, `max_new_tokens=256`, beam 5, bf16,
warmup excluded, 3 timed `generate()` calls per batch size:

```
BATCH=  4 calls=[19.1, 19.0, 19.0] median=19.02s per-input-sentence=4.756s
BATCH= 16 calls=[33.0, 32.8, 33.0] median=33.01s per-input-sentence=2.063s
```

**2.31× per input sentence**, rep spread <0.2 %, reproduced independently by job 27066741
(`18.815 s` → `32.945 s`; 4.704 → 2.059 s/sentence). No OOM at 94 GB with 80 sequences.

## Measurement 2 — the unit cost (`measure_units.py`, job 27066642)

Widest quantized linear (5120×13824, 70.8M weights), 80 activation rows, median of 20 reps after
warmup: `dequantize_weight()` **0.15 ms**, `torch.matmul(x, W)` **0.08 ms** (ratio 1.83).

Scaled by weight-count to the model's 179 such units this implies ~27 ms of dequant per forward
pass. **That figure is wrong** — see "Reconciling three dequant numbers". The useful output of this
job is the *ratio*: per linear, at realistic batch, dequantization costs more than the matmul.

## Measurement 3 — attribution on the real path (`measure_pipeline.py`, job 27066741)

Rather than model the pipeline, perturb it: replace the per-forward dequant with the same matmul on
pre-dequantized bf16 weights, for the largest linears covering 55 % of the model's weights. Then
`d = (A − B)/coverage` gives the whole-model dequant cost per `generate()` with no extrapolation.

| | `BATCH=4` | `BATCH=16` |
|---|---|---|
| A) full wall | 18.815 s | 32.945 s |
| B) 55 % of weights dequantized+cached | 16.064 s | 29.910 s |
| implied whole-model dequant `d` | **5.00 s** | **5.52 s** |
| dequant share of wall | 26.6 % | 16.7 % |
| per sentence: dequant + rest | 1.250 + 3.453 = 4.704 s | 0.345 + 1.714 = 2.059 s |

`d` is the same at both batch sizes (10 % apart, within method error) — the fixed-cost premise
holds. Splitting the 2.645 s/sentence saved:

| source | saving | share |
|---|---|---|
| amortizing the fixed per-forward dequant (÷4 from 4× batch) | 0.906 s | **34 %** |
| the remaining (matmul/attention/overhead) growing only 1.99× for 4× the rows | 1.739 s | **66 %** |

So the honest mechanism is *both*, weighted toward the second: **low-bit GPTQ generation at
`BATCH=4` is dominated by a fixed per-forward dequantization that batch amortizes, and by a
non-dequant term so far from saturating the H100 that quadrupling rows only doubles it.**

## Measurement 4 — the `BATCH_LONG` path, zh-en (`analysis/sbatch/probe_zh.sbatch`, job 27074471)

`zh-en` is the one direction the eval jobs run at `BATCH_LONG=1` (source length 512, batch 1), which
is precisely where a fixed per-forward cost is amortized over the fewest sequences — 5 rows per
forward instead of 20. Model `gptq-w2g128`, `max_source_length=512`, 2 reps per batch size:

```
BATCH=  1 calls=[10.2, 10.1] median=10.14s per-input-sentence=10.139s
BATCH=  4 calls=[11.7, 11.6] median=11.63s per-input-sentence=2.908s
```

**3.49× per input sentence** — a larger win than the 2.31× from the 4→16 probe, exactly as the
amortization model predicts (the worse the starting amortization, the more there is to recover).
No OOM at source length 512 on the 2-bit model. Measurement 5 corrects what the fp16 baseline's
batch-4 OOM at that length actually was: the probe runs the fp16 model at *exactly* that shape
(`BATCH=4` × 5 beams, source 512) in a measured 33.3 GiB peak, while the crashed job's traceback
shows 84 GiB allocated by PyTorch inside one process — so weights plus KV cache (24.2 + 12 GiB) do
not explain it either, and `generate_alma_r.job:77`'s "a 13B model out of memory" is a description
of the symptom, not of the cause.

This matters more than measurement 1, because zh-en is the single most expensive direction and it
runs at the worst batch setting. Applied to a re-run it would cut w2's zh-en from ~5.3 h to ~1.5 h.

## Measurement 5 — what the compression costs, next to fp16 (`cost.sbatch`, `probe16.sbatch`)

**Question:** the grid is scored, so the *quality* side is known (`outputs/baseline/*.tsv`). This is
the other half — what each checkpoint costs to *run*: peak VRAM, load time, latency, throughput and
GPU-hours, measured on one H100 with the eval protocol's own shapes, one process per checkpoint
(`analysis/measure_quant_cost.py`), plus `BATCH=4` vs `16` for all five (`analysis/run_probe.py`).

**Answer: on this stack the compression buys memory, not time.** Peak VRAM at `BATCH=4` on de-en
falls 28.8 GiB (fp16) → 17.2 (w8) → 12.3 (w4) → 10.1 (w3), and per line *nothing* gets faster:
0.359 s (fp16) → 0.603 (w8) → 0.461 (w4) → 1.188 (w3) → 4.709 (w2) on a byte-identical workload.
The reason is the kernel, not the bits: GPTQModel's Triton and Torch linears dequantize the whole
weight matrix to bf16 and *then* matmul (`quant_matmul`), so every forward reads the packed weights
and writes plus re-reads a full bf16 copy of them — more HBM traffic than fp16, which reads its
weights once. w4 is the exception that proves the rule: it gets `ExllamaV2QuantLinear`, and it is the
only quantized checkpoint within 1.3× of fp16. The spread across bits (0.46 / 0.60 / 1.19 / 4.71) is
the kernel lottery, not a smooth function of precision.

<!-- table 1 begin - generated by make_report.py, do not edit -->
| checkpoint | bits | kernel | GiB on disk | peak GiB de-en 256 @4 | fixed s de-en | real s de-en (n) | peak GiB zh-en 512 @1 | fixed s zh-en | real s zh-en (n) | grid GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16 | fp16 | 24.24 | 28.8 | 0.359 | 0.436 (1984) | 26.5 | 0.517 | 1.471 (256) | 10.6 |
| W8G128 | 8 | TritonV2 | 12.72 | 17.2 | 0.607 | 0.755 (512) | 15.1 | 0.877 | 2.683 (256) | 6.6 |
| W4G128 | 4 | ExLLamaV2 | 6.76 | 12.3 | 0.460 | 0.550 (512) | 10.0 | 0.543 | 1.534 (256) | 4.5 |
| W3G128 | 3 | Torch | 5.27 | 10.1 | 1.181 | 1.467 (64) | 7.8 | 1.377 | 4.570 (64) | 13.6 |
| W2G128 | 2 | TritonV2 | 3.78 | 11.7 | 4.705 | 3.513 (64) | 6.4 | 10.138 | 10.577 (64) | 19.7 |

Peak VRAM is the allocator peak of one timed `generate()` at the eval protocol's own shape; *fixed s* is the median of three such calls on the first four sentences of the pair (a byte-identical workload for every row, ~30 generated tokens); *real s (n)* is the same model over `n` full lines of that pair at the same batch. *grid GPU-h* is what the cluster billed for that run's generation (the fp16 baseline is 4x its wall: four ranks over four GPUs). `--` = not measured.
<!-- table 1 end -->



**Memory: peak, not weights.** The de-en `BATCH=4` peak is 28.8 GiB at fp16 of which 24.2 GiB are
weights (84 %), and 12.3 GiB at w4 of which 6.8 GiB are weights (55 %) — the rest is the KV cache,
the activations and the dequantization buffers, none of which shrink with the weights. That is why
each step below 4 bits buys ~1–2 GiB of peak rather than the ~3 GiB the checkpoint loses, and why
the disk ratios (1.9× / 3.6× / 4.6× / 6.4×) are not peak ratios at all: measured peak is 1.67× /
2.34× at 8 and 4 bits, collapsing towards 1× as the KV cache takes over.

**Kernel choice is worth one more look, and then it is exhausted.** Sweeping every kernel GPTQModel
4.2.5 will load for these checkpoints (`analysis/sbatch/backends.sbatch`, same fixed workload, same
checkpoint per row): on w4, `ExllamaV2QuantLinear` (what `auto` picks) 0.462 s/line against
`TorchQuantLinear` 0.561 and `TritonV2QuantLinear` 0.618 — auto already picks the best available. On
w8, `TorchQuantLinear` is 9 % faster than the auto-selected `TritonV2QuantLinear` (0.549 vs 0.601),
so `BACKEND=torch` is free speed for any future 8-bit run. `MarlinQuantLinear` — the only fused kernel
that declares 8-bit support — refuses these checkpoints outright (`only supports [True] bits: actual
sym = False`), so a fused 8-bit path would require re-quantizing with `sym=true`.
`exllama_v1`, `exllama_eora` and Machete are not installed, BitBLAS needs `desc_act=False`, and no
installed kernel gives 2- or 3-bit a fused path.

**Quality per unit of memory.** w8 is free: +0.03 XCOMET-XXL and 0.00 BLEU against fp16 while the
checkpoint halves (and no direction moves more than noise). w4 costs 0.60 XCOMET-XXL and 0.61 BLEU
for another 1.9× of checkpoint. w3 costs 10.91 XCOMET-XXL and 6.61 BLEU for 2.2 GiB less peak than
w4 — a 15 % smaller weight footprint for the collapse — and w2 costs 53.8 XCOMET-XXL and 24.6 BLEU.
The frontier is therefore w8 → w4 and then nothing: below 4 bits the marginal memory is small and the
quality cliff is not.



Seconds per input line for the whole grid (generation only, from each job's own log):

<!-- table 2 begin - generated by make_report.py, do not edit -->
| checkpoint | bits | kernel | GiB on disk | peak GiB de-en 256 @4 | fixed s de-en | real s de-en (n) | peak GiB zh-en 512 @1 | fixed s zh-en | real s zh-en (n) | grid GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16 | fp16 | 24.24 | 28.8 | 0.359 | 0.436 (1984) | 26.5 | 0.517 | 1.471 (256) | 10.6 |
| W8G128 | 8 | TritonV2 | 12.72 | 17.2 | 0.607 | 0.755 (512) | 15.1 | 0.877 | 2.683 (256) | 6.6 |
| W4G128 | 4 | ExLLamaV2 | 6.76 | 12.3 | 0.460 | 0.550 (512) | 10.0 | 0.543 | 1.534 (256) | 4.5 |
| W3G128 | 3 | Torch | 5.27 | 10.1 | 1.181 | 1.467 (64) | 7.8 | 1.377 | 4.570 (64) | 13.6 |
| W2G128 | 2 | TritonV2 | 3.78 | 11.7 | 4.705 | 3.513 (64) | 6.4 | 10.138 | 10.577 (64) | 19.7 |

Peak VRAM is the allocator peak of one timed `generate()` at the eval protocol's own shape; *fixed s* is the median of three such calls on the first four sentences of the pair (a byte-identical workload for every row, ~30 generated tokens); *real s (n)* is the same model over `n` full lines of that pair at the same batch. *grid GPU-h* is what the cluster billed for that run's generation (the fp16 baseline is 4x its wall: four ranks over four GPUs). `--` = not measured.

Seconds per input line for the whole grid (generation only, from each job's own log):

| direction | FP16 | W8G128 | W4G128 | W3G128 | W2G128 |
|---|---|---|---|---|---|
| de-en | 0.284 | 0.750 | 0.541 | 1.382 | 3.584 |
| cs-en | 0.318 | 0.870 | 0.622 | 1.558 | 2.764 |
| is-en | 0.303 | 0.876 | 0.630 | 1.914 | 2.766 |
| zh-en | 1.471 | 3.069 | 1.757 | 5.901 | 9.770 |
| ru-en | 0.309 | 0.812 | 0.583 | 1.512 | 3.330 |
| en-de | 0.412 | 1.102 | 0.784 | 1.953 | 3.688 |
| en-cs | 0.506 | 1.343 | 0.969 | 2.739 | 3.461 |
| en-is | 0.797 | 2.124 | 1.524 | 5.598 | 3.714 |
| en-zh | 0.594 | 1.591 | 1.161 | 3.582 | 3.558 |
| en-ru | 0.470 | 1.258 | 0.901 | 2.683 | 3.243 |
| **all 10** | 0.546 | 1.369 | 0.937 | 2.801 | 4.066 |
| **GPU-h charged** | 10.6 | 6.6 | 4.5 | 13.6 | 19.7 |
| **GPU-s/line** | 2.18 | 1.37 | 0.94 | 2.80 | 4.07 |
| **vs FP16** | 1.00x | 0.63x | 0.43x | 1.28x | 1.86x |
<!-- table 2 end -->


**The grid's own ledger, and one caveat on the fp16 row.** Those per-direction numbers are each run's
wall divided by its lines, which is a *latency* only for the single-GPU runs. The fp16 baseline ran
one job per decoding mode on 4 GPUs with the test set sharded across ranks (its log shows 124 batches
per rank for de-en's 1984 lines at batch 4, i.e. 496 lines per rank), so its wall is ~1/4 of a
single-process latency and what the cluster charged it is 4× that wall: 10.6 GPU-h for the grid
against w8's 6.6 and w4's 4.5 for the same 17,471 lines. Those charged hours flatter the quantized
runs for a job-shape reason, not a numerical one: the baseline's per-GPU throughput is **2.6× lower
than the same model in a one-process job** — de-en 1.13 s/line per rank against a measured 0.435
s/line for a single process over the whole 1984-line pair, and zh-en ~4.5 s/line per rank against
1.471 over 256 lines. Whether that is host contention from four concurrent beam searches on one node
or the node's power/clock envelope with all four H100s saturated, the probe cannot tell from one
measurement; what it does tell is that it is a property of the 4-GPU job, not of fp16. At one-process
efficiency the fp16 baseline would have cost ~2.7 GPU-h — less than w4's 4.5 and w8's 6.6 — so no
reading of this ledger is a speed argument for the compression. Read the fp16 column as *the cost of
the run that produced the baseline*, the `GPU-h charged` row as what the cluster actually billed, and
the single-process numbers in the checkpoint and batch tables as the like-for-like latency.

**Batching is where the freed memory pays.** Same fixed workload, `BATCH=4` → `16`:

| precision | s/line @B=4 | s/line @B=16 | gain | peak GiB @B=4 | peak GiB @B=16 | B=16 vs FP16 |
|---|---|---|---|---|---|---|
| FP16 | 0.359 | 0.226 | 1.59× | 28.8 | 42.3 | 1.00× |
| W8G128 | 0.603 | 0.266 | 2.27× | 17.2 | 30.7 | 1.18× |
| W4G128 | 0.461 | 0.281 | 1.64× | 12.3 | 26.0 | 1.24× |
| W3G128 | 1.188 | 0.427 | 2.78× | 10.1 | 23.3 | 1.89× |
| W2G128 | 4.709 | 2.070 | 2.27× | 11.7 | 35.4 | 9.16× |

The low-bit rows gain most from the batch, because their per-forward dequantization is a fixed cost
the batch amortizes (`measure_pipeline.py`'s attribution: 34 % of the 4→16 win on w2 is dequant, 66 %
is the sublinear growth of everything else) — but the gain never closes the gap: at `BATCH=16` fp16
is still the fastest per line, and w8/w4 stay 1.18× / 1.24× behind it. (`run_probe.py` and
`measure_quant_cost.py` measure the same fixed workload independently and agree to within ~1.5 %; the
table above is `run_probe.py`'s, the `fixed s` column of the checkpoint table is
`measure_quant_cost.py`'s.) What quantization does buy is the *room* for that batch: w8 at
`BATCH=16` needs 30.7 GiB and 0.266 s/line where fp16 at `BATCH=4`
needs 28.8 GiB and 0.359 s/line, so at an equal footprint the quantized-and-batched configuration is
1.35× faster. On a 94 GiB H100 nothing in this grid needed that trade (fp16 at `BATCH=16` fits in
42.3 GiB); on a 40 GiB card it would be the only way to reach batch 16 at all.

**What this means for the question "was quantizing a good idea".** As an *efficiency* measure on this
stack, no: on a byte-identical workload fp16 is the fastest of the five (0.36 s/line against w8's
0.61, w4's 0.46, w3's 1.18 and w2's 4.71), the quantized kernels dequantize on every forward and the
grid never needed the memory that buys — peak VRAM at the eval protocol only falls 2.3× for a 3.6×
smaller checkpoint, and fp16 at `BATCH=16` still fits in 42.3 GiB. The grid's cheaper-looking GPU-hour
ledger for w4/w8 is the baseline job's four-ranks-per-node cost, not a property of the compression.
What the compression *does* buy is the room to spend on batch size, which is the only lever that
returns any of the speed (`BATCH=16`), and even then fp16 at the same batch stays ahead. As an
*experiment*, yes — and that is what the grid is: w3 is where the low-resource-language damage lives
(`is-en` XCOMET-XXL 81.15 → 43.75), w8 is a pure null result and the useful negative control, and w2
is broken globally, which is itself the reason not to narrate w3's collapse as "rare languages die
first". Nothing here is a throughput argument for weight-only GPTQ on H100s; on a smaller card, where
fp16 at `BATCH=4` would not fit beside the KV cache, the memory would be the argument.



## Reconciling three dequant numbers

The dequant cost per forward pass was measured three ways, spanning 90×:

| method | value per forward pass | why |
|---|---|---|
| `run_probe.py` `measure_dequant()` — 280 calls, no warmup, outside `inference_mode` | 2430 ms | includes cold Triton JIT/autotune and allocator growth; measures a cold path |
| `measure_pipeline.py` C — one warmed pass, whole model, on the real modules | **442 ms** | the number to use |
| `measure_units.py` — one 5120×13824 matrix, 20× tight loop, warmed | 27 ms | L2-resident: the 2-bit packed weight is 17.7 MB, and the H100 L2 is 50 MB, so the loop re-reads from L2 instead of HBM |

The L2 explanation is testable and consistent: 17.7 MB of packed weights fit alongside a hot loop,
which is exactly the condition that does *not* hold in a real forward pass streaming 26 GB of model
weights. Using the mid value, 442 ms per pass, gives `5.00 s / 0.442 s ≈ 11.3` forward passes per
`generate()` on this pair — an independent sanity check that the WMT de-en sentences are short,
rather than the ~30 an earlier fit assumed.

This reconciliation is why the first draft of this file concluded "no lever, ~1 %": it divided the
*cold* 2.43 s by a *guessed* longer pass count, arrived at a small share, and then assumed the
matmul dominated. Measuring the thing itself (measurement 1) is what settled it.

## Where the time goes, and how far batch can be pushed

From the measured decomposition above (`d` fixed, rest sublinear), extrapolation is only safe for
small steps; the two measured points are:

| BATCH | per input sentence | dequant share |
|---|---|---|
| 4 | 4.704 s *(measured)* | 26.6 % |
| 16 | 2.059 s *(measured)* | 16.7 % |

Past 16 the fixed term keeps shrinking (0.345 s → 0.17 s at 32) while the sublinear term dominates
the remaining time, so improvements flatten. `BATCH=16` is the recommendation: 2.31× for 4× the
sequence slots, with large memory headroom (≈3.8 GiB of weights at 2-bit plus 80 × 512-token bf16
activations against 94 GB).

## Consequences and recommended action

- **w2 is projected to overrun its wall and has not been touched.** Its generation reaches ~19.8 h
  and the ~70 min scoring tail then crosses the 20 h limit, because zh-en at `BATCH_LONG=1` costs a
  measured 10.14 s/line (~5.3 h for 1875 lines). Nothing was resubmitted or reconfigured — this is a
  finding for you to act on, not an action taken.
- **Two ways to handle it, both requiring your decision:** (a) let it hit the wall — `Requeue=1` plus
  `todo()` means the second window reruns only zh-en plus scoring, unaided; or (b) resubmit with
  `BATCH_LONG=4`, which the measurement says turns zh-en into ~1.5 h and brings w2 to ~17 h total,
  inside one window. Both keep the outputs valid; (b) is cheaper on queue time, (a) needs nothing.
- **The lever's largest value is the `zh-en` path generally** — it is the most expensive direction in
  every run and the worst-amortized (5 rows/forward). w3 hits it last, and it is 167 min of its
  4.1 h remaining; if w3 ever needs a window of its own, raise `BATCH_LONG` there too.
- **If any run is redone, apply `BATCH`/`BATCH_LONG` to all of them, not a subset.** A throughput
  difference between runs must never be confusable with a protocol difference. Beam search with
  identical prompt/beam/length settings is batch-invariant up to bf16 reduction order, but the
  *journal* of a mixed-batch grid is harder to defend than a uniform one; a uniform `BATCH=16` +
  `BATCH_LONG=4` grid would cut generation from ~43 to well under 20 GPU-h.
- **Do not chase kernels to fix 2-/3-bit.** No maintained loader has a 3-bit accelerated kernel, and
  the one that exists (the original GPTQ repo's fused 3-bit matvec) is a porting project with an
  act-order mismatch, not a drop-in — see the correction above. Batch size is the only free lever for
  these checkpoints.

## Files

| file | role |
|---|---|
| `analysis/run_probe.py` / `analysis/sbatch/probe.sbatch` | measurement 1: `generate()` at two batch sizes, warmup excluded (job 27066494) |
| `analysis/measure_units.py` / `analysis/sbatch/units.sbatch` | measurement 2: per-linear dequant vs matmul (job 27066642) |
| `analysis/measure_pipeline.py` / `analysis/sbatch/pipe.sbatch` | measurement 3: perturbation attribution of the real pipeline (job 27066741) |
| `analysis/sbatch/probe_zh.sbatch` (reuses `analysis/run_probe.py`) | measurement 4: the `BATCH_LONG=1` zh-en path at source length 512 (job 27074471) |
| `analysis/measure_quant_cost.py` / `analysis/sbatch/cost.sbatch` | measurement 5: load, peak VRAM, latency and throughput per checkpoint at the two eval protocols, next to fp16, plus N real lines of each pair (`--samples`) |
| `analysis/sbatch/probe16.sbatch` (`run_probe.py`) | measurement 5: the same workload at `BATCH=4` and `BATCH=16` with peak VRAM, all five checkpoints |
| `analysis/sbatch/backends.sbatch` | measurement 5: every GPTQModel kernel each checkpoint will load, same fixed workload (Marlin refuses asymmetric weights, w1/eora are not installed, so only ExLLamaV2/Torch/Triton actually run) |
| `analysis/make_report.py` | the LaTeX report: `results/tex/*.tex` (incl. the generated `cost.tex`) and `results/figures/*`; also writes `results/quant_cost_summary.md` |
| `02-cost-and-memory.md` | measurement 5 as a stand-alone note: every number, the mechanism behind "smaller but not faster", the speed levers with their measured multipliers, open decisions |
| `analysis/measure_channels.py` / `analysis/sbatch/measure_channels.sbatch` | **not part of this study** — RQ5 in `../../reasearch.md`: per-channel activation energy per language on the fp16 model, written to `results/json/channels_fp16.npz` |
| `results/json/probe_ALMA-13B-R-gptq-w2g128.json`, `results/json/units_w2.json`, `results/json/pipeline_w2.json`, `results/json/probe_zh_w2.json` | machine-readable results |
| `results/json/quant_cost_{fp16,w8,w4,w3,w2}.json`, `results/json/probe16_{...}.json` | measurement 5: per-checkpoint cost records and batch sweep (the tables above are generated from these) |
| `results/results.pdf` | the compiled report |
| `../logs/slurm-probe-{batch,units,pipe,zh}-*.out` | raw job logs |
| `../logs/slurm-probe-{cost,b16}-*.out` | measurement 5 raw job logs |

Nothing here writes to `outputs/` or modifies the eval pipeline; the probe only times and annotates.

```bash
sbatch francesco/analysis/sbatch/probe.sbatch                                        # 2-bit, 4 vs 16
sbatch --export=ALL,BATCHES=4,8,16,32 francesco/analysis/sbatch/probe.sbatch         # finer sweep
sbatch --export=ALL,MODEL=$PWD/../models/ALMA-13B-R-gptq-w3g128 francesco/analysis/sbatch/probe.sbatch   # 3-bit
sbatch francesco/analysis/sbatch/pipe.sbatch                                         # perturbation attribution
sbatch francesco/analysis/sbatch/cost.sbatch                                         # all five: load, peak VRAM, latency
sbatch --export=ALL,SAMPLES=512,SAMPLES_LONG=256 francesco/analysis/sbatch/cost.sbatch  # + real lines
sbatch francesco/analysis/sbatch/probe16.sbatch                                      # BATCH=4 vs 16, all five
sbatch francesco/analysis/sbatch/backends.sbatch                                     # kernel sweep
python francesco/analysis/make_report.py && (cd francesco/results/tex && pdflatex -output-directory=.. results.tex && pdflatex -output-directory=.. results.tex)
```