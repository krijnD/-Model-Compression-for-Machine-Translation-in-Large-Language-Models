# Current state and wrap-up (2026-09-29)

> *Moved to `docs/` on 2026-09-30, content unchanged except updated doc names. Paths in this file are relative to `francesco/` (`../` = the repo root), as when it was written. Start at `../README.md`.*

Project: weight-only GPTQ compression of ALMA-13B-R for machine translation, WMT'22, 10 directions,
8 metrics. Checkpoints: `models/ALMA-13B-R-gptq-w{8,4,3,2}g128` (asymmetric, group 128, act-order).
Every number below comes from a file in this repo; status is marked as measured, preliminary or pending.

## 1. One-line framing

Weight-only GPTQ cuts VRAM 2.3x at essentially no cost at 4 bits, but below that the loss is neither
small nor evenly spread: at 3 bits Icelandic loses about half its translation quality (is-en BLEU
40.21 -> 20.25, -49.6 %) while German loses under a tenth (-8.0 %).

## 2. Memory saved for quality lost (measured)

| precision | checkpoint (GiB) | peak VRAM, de-en @ batch 4 (GiB) | quality vs fp16 |
|---|---|---|---|
| fp16 | 24.24 | 28.8 | baseline |
| 8-bit | 12.72 | 17.2 | none (BLEU +/-0.03) |
| 4-bit | 6.76 | 12.3 | small (-0.61 BLEU, -0.60 XCOMET-XXL) |
| 3-bit | 5.27 | 10.1 | large, language-dependent |
| 2-bit | 3.78 | 11.7 | total collapse, every language, BLEU ~0 |

- Peak VRAM falls less than the checkpoint (2.3x vs 3.6x at 4 bits) because the KV cache and
  activations do not shrink with the weights.
- The trade-off is a cliff, not a slope: 8 and 4 bits are nearly free, 3 bits is where the language
  structure appears, 2 bits is broken globally (not a "rare languages die first" story).
- Compression buys memory, not speed, on this stack (see 5).

## 3. Language-dependent damage at 3 bits (measured)

Icelandic is worst on all 8 metrics in both directions (e.g. XCOMET-XXL is-en 81.15 -> 43.75). BLEU drop
at 3 bits: is-en -49.6 %, en-is -45.9 %, zh-en -33.6 %, cs-en -28.3 %, ru-en -20.4 %, de-en -8.0 %.
Source: `research.md` sections 1.1-1.2, `outputs/baseline/*.tsv`.

## 4. What was ruled out, and what is left

Ruled out (each tested):
- **Calibration under-representation.** Icelandic has the largest calibration share (22.2 % of tokens).
- **Special activation outliers.** RQ5, four sites: Icelandic is not more concentrated than German and
  the channel sets barely differ; the most a per-language re-ordering could buy is 0.0-1.1 pp.
- **3-bit rounding erasing the fine-tuning.** ALMA-13B-R minus ALMA-13B-Pretrain is a rank-16 change on
  `mlp.down_proj` (>99.99 % in the top 16 directions, cosine 0.990-0.996 with the published adapter).
  After GPTQ the change survives in aggregate (rho 1.00-1.02 at every bit width, layers 0/20/39). Rounding
  noise is ~9x larger than the change at w3 but random with respect to it. Keeping the adapter in bf16 over
  a 3-bit base would therefore do almost nothing. (`05-why-icelandic.md`, `results/json/delta_erasure.json`)
- **Icelandic just having harder sentences.** Matching sentences across directions by fp16 confidence:
  w3 XCOMET-XXL gap -0.227 raw vs -0.229 matched; at w4 the gap is 0.000. (`fragility_by_confidence*.json`)

Still the best explanation (correlational): Icelandic is the least-trained language at both stages
(2,009 fine-tuning rows, 8 % OSCAR share); r(%-drop, log10 rows) = 0.82 over 10 points, no error bars.

Preliminary lead: does w3 lose Icelandic itself or only translating it? `lang_nll.py` smoke run
(**32 sentences per item, no error bars**): at 3 bits translation NLL rises ~1.9x (is-en) and ~2.1x (en-is)
against 1.1-1.5x for the other languages, while plain-text NLL rises 1.4x for Icelandic, the same as Czech.
The ordering does not match BLEU exactly (ru-en vs cs-en), so this is a hint, not a result.

Caveats: 50.4 % of the Icelandic fine-tuning data is FLORES devtest; FLORES is fully contaminated for the
five reported languages; 5 languages only; single seed; not tested: the stage-1 (monolingual) weights,
because LLaMA-2-13B is gated for our HF token (403).

## 5. Cost and speed (measured)

- fp16 is the fastest per line at equal batch (0.359 s/line at batch 4, 0.226 at batch 16 on the fixed
  workload). Only w4 gets a fused kernel (ExLlamaV2, 0.46 s/line); w8/w3/w2 dequantize the whole matrix on
  every forward pass (Triton or Torch).
- Batch size is the free lever: batch 4 -> 16 gives 1.6-2.8x on de-en; zh-en at batch 1 -> 4 gave 3.49x.
- **New: repacking w3 into the 4-bit layout is lossless and makes w3 run on the fused kernel.** CPU check:
  0 mismatches on 11 sample modules (every Linear type, layers 0-40, 511M weights) for w3 and for w2, with
  planted-bug controls that fail as they should. The full 280-module check was not run.
  Timings (`francesco/results/json/backend_w*as4_*.json`, `probe16_w*as4.json`):

  | fixed workload | before | after repack | speedup |
  |---|---|---|---|
  | w3 de-en, batch 4 | 1.19 s/line (Torch) | 0.47 | 2.5x |
  | w3 zh-en 512 @4 | 1.62 | 0.78 | 2.1x |
  | w3 de-en, batch 16 | 0.427 | 0.288 | 1.5x |
  | w2 de-en, batch 4 | 4.70 (Triton) | 1.46 auto / 2.01 forced | 2.3-3.2x, noisy |
  | w2 zh-en 512 @4 / batch 16 | 2.89 / 2.07 | 2.53 / 2.05 | ~1.0x |

  w3 is now ~1.3x slower than fp16, not faster. w2's gain is small because its outputs are long (the
  collapse itself). Costs 1.5 GiB (w3) / 3 GiB (w2) more weight memory.
- Ways past fp16 (not measured): vLLM with CUDA graphs and Marlin (needs re-quantizing with sym=true,
  desc_act=False, and regenerating the baseline), `torch.compile` static cache, Machete. Per-step launch
  overhead, not weight traffic, is what limits the current stack.

## 6. Pending and open

- **Drift check (job 27351455): done, see `04-kernel-repack.md` §8.** 2.3–2.6× faster; BLEU within 0.4, but
  only 80 % (de-en) / 38.5 % (en-is) of lines identical, so repacked w3 is a separate run, not a replacement.
- **Distillation (added 2026-09-30, not yet run):** w3 + LoRA distilled from fp16, see `06-distillation.md`.
- **Full NLL run: done** (job 27399375, 500 sentences). Icelandic keeps the language (`mono` mid-table) but loses
  the mapping (largest `trans` dNLL in both directions). See `05-why-icelandic.md` §1, "Result".
- `make_report.py` does not yet list the `*as4` rows in the batch table, and the kernel-table caption still
  says "no kernel gives 2- or 3-bit a fused path", which the repack contradicts. Update both if the report
  is rebuilt.
- Not yet run: capability screen on unseen FLORES languages (research.md section 5), per-layer language
  sensitivity map, rotation-based methods, re-decoding en-is.
- `/scratch-shared/scur0517/alma_delta/` holds 22 GB of ALMA-13B-Pretrain shards; `/scratch-shared/scur0517/
  models/` holds the repacked w3 and w2 checkpoints. Delete when done.

## 7. Where things are

| what | where |
|---|---|
| research plan, RQ1-RQ6, provenance | `research.md` |
| batch/cost/kernel study | `francesco/README.md`, `02-cost-and-memory.md`, `francesco/results/results.pdf` |
| repack and kernel options | `04-kernel-repack.md`, `analysis/repack_to_w4.py`, `analysis/sbatch/repack_bench.job` |
| new research questions and results | `05-why-icelandic.md`, `analysis/measure_delta.py`, `analysis/lang_nll.py` |
| figures | `francesco/results/figures/` |
