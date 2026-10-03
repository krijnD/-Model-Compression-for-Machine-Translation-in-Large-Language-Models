# Quantization cost study — handoff note (2026-09-25)

> *Moved to `docs/` on 2026-09-30, content unchanged except updated doc names. Paths in this file are relative to `francesco/` (`../` = the repo root), as when it was written. Start at `../README.md`.*

Everything below is measured on Snellius `gpu_h100` (one H100 94 GB per process unless stated),
GPTQModel 4.2.5, bf16, beam 5, seed 42, the same prompts/padding as `scripts/generate_quantized.py`.
Model: ALMA-13B-R and its `gptq-w{2,3,4,8}g128` checkpoints.

**The question this answers.** The grid already says what each precision *loses* in quality
(`../outputs/baseline/*.tsv`, summarised in `README.md` §1 and `results.pdf`). This study measures
what each precision *costs to run* — peak VRAM, load, latency, throughput, GPU-hours, batch
behaviour — i.e. whether quantizing was worth it at all, and what to do if we want inference to be
faster.

**Status: complete — every measurement in this note has landed.** The prose report is
`README.md` ("Measurement 5"); the generated LaTeX tables are `results/tex/{cost,kernels}.tex`, the
markdown copies `results/quant_cost_summary.md`, and `make_report.py` now syncs the two markdown
tables into the README automatically (it replaces the `<!-- table N ... -->` marker lines). Rebuild
the PDF with the two `pdflatex` calls in §4.

---

## 1. Bottom line

| | fp16 | w8 | w4 | w3 | w2 |
|---|---|---|---|---|---|
| checkpoint on disk (GiB) | 24.24 | 12.72 | 6.76 | 5.27 | 3.78 |
| peak VRAM, de-en 256 @ `BATCH=4` (GiB) | 28.8 | 17.2 | 12.3 | 10.1 | 12.0 |
| peak VRAM, zh-en 512 @ `BATCH_LONG=1` (GiB) | 26.5 | 15.1 | 10.0 | 7.8 | 6.6 |
| fixed-workload s/line @ `BATCH=4` | **0.359** | 0.603 | 0.460 | 1.181 | 4.705 |
| fixed-workload s/line @ `BATCH=16` | **0.226** | 0.266 | 0.281 | 0.427 | 2.070 |
| real-line s/line (de-en, same 512-line subset) | **0.443** | 0.755 | 0.550 | 1.467 (64) | 3.513 (64) |
| kernel GPTQModel auto-picked | cuBLAS | TritonV2 (dequant-then-matmul) | **ExLLamaV2 (fused)** | Torch (dequant-then-matmul) | TritonV2 (dequant-then-matmul) |
| grid GPU-h charged (17,471 lines) | 10.6 | 6.6 | 4.5 | 13.6 | 19.7 |
| quality Δ vs fp16 (XCOMET-XXL / BLEU, mean of both halves) | — | +0.03 / 0.00 | −0.60 / −0.61 | −10.91 / −6.61 | −53.84 / −24.63 |

- **Quantization buys memory, not time.** Peak VRAM falls 2.3× at w4 (28.8 → 12.3 GiB on the de-en
  shape) while a byte-identical workload gets 1.28× *slower* (0.359 → 0.460 s/line), and 1.68× at w8.
- **Even the memory is less than the disk ratio suggests**, because the KV cache and activations do
  not shrink with the weights: the weights are 84 % of the fp16 peak but only 55 % of w4's, and w2's
  12.0 GiB peak (with 3.78 GiB of weights) is *larger* than w4's because w2 generates 180-token
  outputs instead of ~40 and the extra KV dominates. Peak ≠ weights.
- **The grid's cheap-looking GPU-hour ledger for w4/w8 is a job-shape artefact**, not a speed win:
  the fp16 baseline job ran 4 ranks per node and each rank is 2.6× slower per line than the same
  model in a one-process job (de-en 1.13 s/line per rank vs 0.435 over the whole 1984-line pair;
  zh-en ~4.5 vs 1.471 over 256 lines). At one-process efficiency the fp16 baseline would have spent
  ~2.7 GPU-h against w4's 4.5 and w8's 6.6 — less than every quantized run.
- **Where the compression does pay:** 8 bits is free in quality (+0.03 XCOMET-XXL) for a 1.9× smaller
  checkpoint; 4 bits costs 0.6 XCOMET-XXL for 3.6×. Below 4 bits the marginal memory is 1–2 GiB of
  peak per step while quality collapses, so w3/w2 are only defensible as the *experiment* they are
  (`is-en` XCOMET-XXL 81.15 → 43.75 at 3 bits), never for cost.
- **The one lever that converts memory into speed is batch size** (see §3).

## 2. Why smaller is not faster

The four quantized checkpoints run three different GPTQModel kernels, and only one of them is fused:

```python
# gptqmodel/nn_modules/triton_utils/dequant.py:170   (w8, w2 — TritonV2QuantLinear)
def quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=False):
    W = dequant(input.dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
    return input @ W                        # dequantize the WHOLE matrix, then matmul

# gptqmodel/nn_modules/qlinear/torch.py:161           (w3 — TorchQuantLinear)
weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
out = torch.matmul(x, weights)

# gptqmodel/nn_modules/qlinear/exllamav2.py:254       (w4 — ExllamaV2QuantLinear)
out = ext_gemm_half_q_half(x, self.q_handle, self.out_features, force_cuda)
```

Nothing caches the dequantized matrix (no `_cached_W` on that path), so the first two move *more*
bytes per forward pass than fp16, not fewer. ALMA's quantized Linears hold 12.59 B weights
(23.45 GiB in bf16):

| precision | read packed | write bf16 copy | read bf16 copy | moved per forward |
|---|---|---|---|---|
| fp16 | — | — | 23.45 GiB | **23.45 GiB** |
| w8 | 11.9 | 23.45 | 23.45 | **58.8 GiB** |
| w3 | 4.47 | 23.45 | 23.45 | **51.4 GiB** |
| w2 | 2.98 | 23.45 | 23.45 | **49.9 GiB** |
| w4 (fused) | 5.96 | — | — | **5.96 GiB** |

And even the fused 4-bit path is not 4× faster, because at 20 sequences per forward (batch 4 × 5
beams) the decode GEMMs are tiny and nothing is near its roofline: fp16 needs ~37 steps of ~7.5 ms
of weight streaming per line (~0.28 s) but measures 0.435 s, i.e. ~3.6× off; w4 is ~7× off its own
(lower) bound. The bottleneck at `BATCH=4` is kernel efficiency and per-step overhead, which scale
with **rows**, not with weight precision — hence §3.1.

Independent evidence from the earlier study (`README.md`, `measure_pipeline.py`): replacing the
per-forward dequantization with a cached bf16 matmul removes 26.6 % of w2's wall at `BATCH=4` — that
26.6 % is what a fused kernel would recover, and it would additionally skip the bf16 write+read.

## 3. How to make inference faster

**3.1 Batch size — measured, free, the only lever that worked on all five precisions.** 20 → 80
sequences per forward, same fixed workload:

| precision | s/line @B=4 | s/line @B=16 | gain | peak GiB @B=4 | peak GiB @B=16 |
|---|---|---|---|---|---|
| FP16 | 0.359 | **0.226** | 1.59× | 28.8 | 42.3 |
| W4 | 0.460 | 0.281 | 1.64× | 12.3 | 26.0 |
| W8 | 0.603 | 0.266 | 2.27× | 17.2 | 30.7 |
| W3 | 1.188 | 0.427 | 2.78× | 10.1 | 23.3 |
| W2 | 4.705 | 2.070 | 2.27× | 12.0 | 35.4 |

The `zh-en` path (source 512, `BATCH_LONG=1`) gains more: **3.49×** going 1 → 4 (10.14 → 2.91 s/line,
measured on w2). Applying `BATCH=16` + `BATCH_LONG=4` to the whole grid takes generation from ~43
GPU-h to well under 20. Note the low-bit rows gain *most* (their dequant is the fixed cost the batch
amortizes), but fp16 at the same batch stays ahead — and fp16 at `BATCH=16` needs 42.3 GiB, which
fits the H100.

**3.2 Node packing — 2.6×, free.** Any multi-GPU job on this cluster pays it (measured on the fp16
baseline: 4 sharded ranks/node, per-rank throughput 2.6× below a single-process job). Whether it is
host contention from four concurrent beam searches or the node's power/clock envelope is not
distinguished by these measurements; it is a property of the job shape, not of the model. Prefer
fewer processes per node.

**3.3 Kernel choice — swept, and the answer is mostly "no".** The eval jobs all use
`BACKEND=auto`. `sbatch/backends.sbatch` loaded every kernel GPTQModel 4.2.5 will accept for these
checkpoints and re-timed the fixed workload (`results/json/backend_<tag>_<backend>.json`):

| kernel | measured, same checkpoint (de-en s/line; zh-en 512 @4 s/line) | consequence |
|---|---|---|
| `ExllamaV2QuantLinear` | w4: **0.462 / 0.728** (auto today) | fused; 1.34× faster than Triton on w4. Best available kernel in the stack. |
| `TorchQuantLinear` | w4: 0.561 / 0.777 · w8: **0.549 / 0.790** · w2: **4.270 / 2.739** · w3: 1.190 / 1.620 (auto for w3) | 9 % faster than Triton on *both* checkpoints that get Triton (`w8`, `w2`) → a free `BACKEND=torch` switch for either; its `dequantize_weight` is a 22 ms pass against Triton's 2.4 s cold. |
| `TritonV2QuantLinear` | w4: 0.618 / 0.831 · w8: 0.601 / 0.834 (auto for w8) · w2: 4.699 / 2.888 (auto for w2) | dequant-then-matmul. |
| `MarlinQuantLinear` | refused: `only supports [True] bits: actual sym = False` | Marlin is the only fused kernel that declares 8-bit support, and our grid is asymmetric — unlocking it means re-quantizing with `sym=true` (16.5 min on one GPU), see §6.5. |
| `ExllamaQuantLinear` (v1) | refused: `No module named 'gptqmodel_exllama_kernels'` | not installed. |
| `ExllamaEoraQuantLinear` | refused: `No module named 'gptqmodel_exllama_eora'` | not installed. |
| `BitBLASQuantLinear` | not attempted: needs `desc_act=False` | our checkpoints are `desc_act=True`. |
| `MacheteQuantLinear` | — | not shipped in GPTQModel 4.2.5. |

**No installed kernel gives w2 or w3 a fused path.** The only 3-bit fused kernel in existence is the
original GPTQ repo's `VecQuant3MatMulKernel` (matvec, act-order/static-groups mismatch, torch 1.10
toolchain) — a porting project, scoped in `README.md`.

**3.4 A different engine — the 2–4× step, at the cost of the baseline.** vLLM / TensorRT-LLM with
Marlin GPTQ kernels, paged attention and continuous batching is the standard answer [INFERENCE, not
measured here]. The catch for this project: different kernels rotate bf16 reduction order, so the
translations change and `ours-beam` (the reference every quantized row was scored against) would have
to be regenerated, and ALMA-R comparability becomes a decoder comparison. That is a protocol
decision, not a flag.

**3.5 What does not help.** Lower bits (w3/w2 are slower *and* worse); caching the dequantized
weights (23.45 GiB resident — the opposite of the point, though useful as the diagnostic that
produced the 26.6 % number); batching past 16 (the fixed term is already down to 0.345 → 0.17 s).

## 4. Files, jobs, commands

| what | where |
|---|---|
| per-checkpoint cost probe (VRAM, load, latency, throughput, N real lines) | `analysis/measure_quant_cost.py`, `analysis/sbatch/cost.sbatch` |
| `BATCH=4` vs `16` sweep with peak VRAM | `analysis/run_probe.py`, `analysis/sbatch/probe16.sbatch` |
| kernel sweep | `analysis/sbatch/backends.sbatch` → `results/json/backend_<tag>_<backend>.json` |
| records | `results/json/quant_cost_{fp16,w8,w4,w3,w2}.json`, `results/json/probe16_*.json` |
| generated tables/figures | `results/tex/cost.tex` (cost, per-direction latency, benefit, batch), `results/figures/fig{5,6}_*.pdf`, `results/quant_cost_summary.md` |
| report prose | `README.md` "Measurement 5" (+ the corrected fp16-OOM sentence in "Measurement 4"); `results/results.pdf` section "What the quantization costs and buys" |
| raw logs | `../logs/slurm-probe-cost-*.out`, `../logs/slurm-probe-b16-*.out`, `../logs/slurm-probe-kernels-*.out` |

```bash
sbatch francesco/analysis/sbatch/cost.sbatch                                            # all five checkpoints
sbatch --export=ALL,SAMPLES=512,SAMPLES_LONG=256 francesco/analysis/sbatch/cost.sbatch   # + real lines
sbatch francesco/analysis/sbatch/probe16.sbatch                                         # BATCH=4 vs 16
sbatch francesco/analysis/sbatch/backends.sbatch                                        # kernel sweep
python francesco/analysis/make_report.py
(cd francesco/results/tex && pdflatex -output-directory=.. results.tex && pdflatex -output-directory=.. results.tex)
```

**Gotcha:** `sbatch --export=ALL,TAGS=w4,w8` does *not* work — Slurm splits `--export` on commas, so
`TAGS` gets only the first value. Use one tag per submission, or `TAGS=w4` / `TAGS=w8`.

## 5. State of the artefacts

Nothing is pending. All jobs are done (`27177735`, `27178575`, `27178577`, `27179003`, `27179014`,
`27179098`); every number in §1-§3 is in `results/json/*.json`, and the generated tables in
`README.md`/`cost.tex`/`kernels.tex` are derived from them.

Two rebuild notes, in case something looks stale:

- `make_report.py` syncs the two markdown tables into `README.md` between
  `<!-- table {1,2} begin ... -->` and `<!-- table {1,2} end -->` markers. The markers *survive* the
  sync, so re-running the generator updates the tables in place; deleting a table's marker pair
  silently freezes that table at its last value (this bit us once).
- The fp16 row of the per-direction grid table is assembled from two sources: the nine short pairs
  come from the fp16 job's own log, `zh-en` comes from the probe's 256-line sample scaled to the
  pair's 1,875 lines. If that sample is missing from the record, the row loses `zh-en` and the fp16
  GPU-h total drops by ~3 — a silent-looking artefact of a partial record, so keep
  `--samples`/`--samples-long` on any fp16 re-run.

## 6. Open decisions for you

1. **Re-run the grid at `BATCH=16` + `BATCH_LONG=4`?** It is the only measured way to cut generation
   (~43 → <20 GPU-h) and it keeps the protocol uniform across runs; it requires regenerating all
   outputs and re-scoring (~1.5 h per run of metric jobs).
2. **Which precision for any future run?** w8 is free in quality and 1.9× smaller; w4 costs 0.60
   XCOMET-XXL and is 3.6× smaller and the only quantized row that is fused (1.24–1.28× fp16). w3 is
   the scientific finding, not a deployment candidate; w2 is broken.
3. **Is a decoder/engine change acceptable?** If yes, vLLM/TensorRT-LLM (Marlin) is where the real
   throughput is; if no, the levers are §3.1–3.3 only.
4. **Should the fp16 baseline be re-run one process per node?** It would cost ~2.7 GPU-h instead of
   10.6 for the same outputs (2.6× measured), and it would remove the paradox in the grid's ledger.
5. **Flip the 8- and 2-bit runs to `BACKEND=torch` for free ~9 %?** Measured on the same
   checkpoints (`w8` 0.549 vs 0.601, `w2` 4.270 vs 4.699 s/line). It is a one-word change in
   `scripts/snellius/generate_quantized.py`, and it changes kernel rounding — for a *finished* grid
   that only matters if the outputs are regenerated anyway.
6. **Re-quantize `w8g128` with `sym=true` to unlock Marlin?** It is the last cheap speed experiment
   on this axis: Marlin is the only fused kernel that declares 8-bit support, it refuses our
   asymmetric checkpoints, and 8 bits is the one precision that is free in quality. Cost: 16.5 min of
   quantization plus a re-score, and `sym` does change the quantizer, so the w8 null result would
   have to be re-established.
7. **Worth porting a fused 3-bit kernel?** Only if 3-bit is going to be used for more than the
   `is`-collapse result; it is scoped (act-order/static-groups mismatch, torch 1.10 → 2.8 toolchain)
   in `README.md`.

## 7. Caveats

- Fixed-workload phases time the *first four sentences* of a pair (~30–40 generated tokens); the
  `real s (n)` columns and the per-direction grid table are the numbers comparable to a real run.
  w2's 64-line de-en sample averaged 180 tokens/line, so its 3.51 s/line is not comparable to w4's
  0.550 (38 tokens) on tokens; the fixed-workload column is the like-for-like one.
- The fp16 row of the per-direction grid table is the *run that produced the baseline* (4 ranks/node,
  see §3.2); its single-process latency is in the checkpoint table (0.435 s/line over the whole
  de-en pair).
- The fp16 baseline's `zh-en` OOM (`scripts/snellius/generate_alma_r.job:77`) is **not** explained by
  weights + KV: the probe runs that exact shape (`BATCH=4` × 5 beams, source 512) in a 33.3 GiB peak,
  while the crashed job's traceback shows 84 GiB allocated by PyTorch inside one process.
- Scoring (COMET/MetricX/lexical) is excluded from all GPU-hours; quantizing a checkpoint costs a
  one-off 16.5 min on one GPU (`../quantization_sizes.txt`).
- Peak VRAM is the PyTorch allocator's peak (allocated; reserved is in the JSONs). The nodes are
  shared, so `nvidia-smi` totals are not used except as a cross-check.
