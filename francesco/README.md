# Compressing ALMA-13B-R for translation: where GPTQ breaks, why, and a distilled fix

*Francesco's part of Project 4. Updated 2026-09-30. Status marks: **measured**, **pending**
(job running), **[inference]** (reasoned, not measured).*

## The story in six steps

**1. Setup.** We reproduced ALMA-13B-R (WMT'22; WMT'21 for Icelandic; 10 directions; 8 metrics) and
compressed it with one fixed recipe: weight-only GPTQ at 8/4/3/2 bits (group 128, asymmetric,
act-order, 1,024 calibration sentences from ALMA's train data). **The quantizer is never changed after
that.** Every later improvement comes from something added on top of the frozen checkpoint.

**2. Compression is a cliff, and at 3 bits it's uneven** (measured, `../outputs/baseline/*.tsv`)

| | size | BLEU xx→en / en→xx | XCOMET-XXL xx→en / en→xx |
|---|---|---|---|
| fp16 | 24.2 GiB | 35.82 / 26.98 | 89.57 / 94.55 |
| w8 | 12.7 GiB | 35.83 / 26.97 | 89.59 / 94.64 |
| w4 | 6.8 GiB | 35.24 / 25.71 | 88.76 / 93.79 |
| w3 | 5.3 GiB | 25.63 / 20.91 | 73.34 / 83.37 |
| w2 | 3.8 GiB | 0.04 / 0.03 | 13.05 / 32.27 |

8 and 4 bits are nearly free and 2 bits is total collapse. **3 bits is where it gets interesting:**
Icelandic loses about half its BLEU (is-en −49.6 %, en-is −45.9 %) while German loses 8 %.

**3. Compression buys memory, not speed** (measured, `docs/03-batch-size-probe.md`, `docs/04-kernel-repack.md`).
fp16 is the fastest per sentence at equal batch. Only 4-bit gets a fused kernel. Storing the w3 values
losslessly in the 4-bit layout gives w3 that kernel: **2.3–2.6× faster generation, same weights**. The
catch is a small output drift from fp16-vs-bf16 arithmetic, so repacked runs are always compared with
repacked runs. On a big GPU the KV cache, not the weights, limits batch size, so low-bit's value is
fitting on small GPUs, not throughput [inference, from measured memory figures].

**4. Why Icelandic?** (measured, `docs/05-why-icelandic.md`, `docs/01-grid-status.md` §4) Ruled out one
by one: too little Icelandic in calibration (it has the largest share); special activation outliers; 3-bit
rounding erasing the fine-tuning update (it survives, ρ ≈ 1.0); and Icelandic simply having harder
sentences (the gap persists at matched difficulty). Then the deciding test, teacher-forced NLL on the test
sets: **at 3 bits the model still knows Icelandic (plain-text NLL mid-table, below Czech and Russian) but
loses the Icelandic↔English mapping (largest translation NLL increase in both directions).**

**5. The fix follows from the diagnosis: distil the translation behaviour back** (measured,
`docs/06-docs/06-distillation.md`). Freeze w3. Add a rank-16 LoRA (2.3 % extra, 0.12 GiB in bf16). Train it for 43 min to
match the fp16 model's next-token distributions on ALMA's translation data (KL divergence, translation
tokens only).

| test set | fp16 | w4 | w3 | **w3 + adapter** | recovered |
|---|---|---|---|---|---|
| translation NLL increase over fp16, all 10 directions | 0 | −0.001–+0.038 | +0.074–+0.642 | **−0.003–+0.025** | 96–99 % |
| is-en BLEU | 40.21 | 39.09 | 21.62 | **39.75** | 98 % |
| en-is BLEU | 22.26 | 21.82 | 11.66 | **22.07** | 98 % |
| en-is outputs ≥ 2× reference length | 0.2 % | 0.1 % | 10.6 % | **0.2 %** | |
| de-en / en-de BLEU | 31.59 / 28.05 | 31.18 / 27.03 | 28.91 / 25.20 | **31.41 / 28.01** | 93 / 98 % |
| is-en / en-is XCOMET-XXL | 81.15 / 92.82 | 80.70 / 91.73 | 46.65 / 66.71 | **81.23 / 92.83** | 100 / 100 % |
| de-en / en-de XCOMET-XXL | 94.66 / 97.71 | 94.24 / 97.53 | 89.09 / 95.16 | **94.49 / 97.55** | 97 / 94 % |

**w3 + adapter (5.4 GiB) is statistically indistinguishable from fp16 in 7 of 8 direction × metric comparisons,
and matches or exceeds w4 (6.8 GiB): significantly better in 2 of 8 (en-is XCOMET, en-de BLEU), tied in the rest**
(paired bootstrap, 1,000 resamples, 95 % CIs in `results/json/bootstrap_ci.json`, `docs/06-docs/06-distillation.md` §6). (w3 here is
the repacked run on the same kernel as the adapter run. Against the originally scored w3 grid, recovery
is also 98 %.)

**6. Stress test: it rescues 2 bits too** (measured for is-en, `docs/06-docs/06-distillation.md` §5;
`results/figures/fig9_w2_story.png`). Same recipe, rank 64 (0.47 GiB), because w2 is broken everywhere. Plain w2
is BLEU 0.00 with 43 % looping outputs. **w2 + adapter reaches 35.40 BLEU on is-en (88 % [86, 90] of fp16; plain w3 21.62, significantly above) and 29.63
on de-en (94 % [91, 95]; plain w3 28.91, a statistical tie), at 4.25 GiB, 1 GiB smaller than plain w3.** Translation NLL beats plain w3 in 8 of 10 directions. The
twist: plain-text NLL stays worse than plain w3's, so the adapter restored translation more than general language
knowledge. XCOMET is still owed. A continuation run (more steps) is in progress and its held-out KL is already below the
first run's.

**Headline:** *At 3 bits, GPTQ's loss on ALMA-13B-R is concentrated in the translation mapping of the
least-trained language. A 125 MB adapter distilled from the fp16 model recovers all of it (XCOMET-XXL equal
to fp16), without touching the quantizer. The same recipe takes 2-bit GPTQ from BLEU 0 to 88 % of fp16.*

## Limits to state in the report
- One seed per run. Sentence-level uncertainty is covered by paired bootstrap CIs (`docs/06-docs/06-distillation.md` §6); seed variance is not.
- Adapter generation covers 4 of 10 directions (is-en, en-is, de-en, en-de). NLL covers all 10.
- No comparison runs yet: SFT on references instead of distillation, or the same adapter on fp16. So
  "distillation beats plain fine-tuning" is not yet a claim.
- Held-out KL uses rows from ALMA's train files. Only the test-set NLL and generation scores are clean.
- FLORES is in ALMA's training data. The test sets we score (WMT'22/'21) are not.
- Memory numbers assume the packed checkpoint on the slower kernels. The fast repacked path stores
  w3/w2 at 4-bit size (6.76 GiB).

## How to read this folder

1. **This README**: the story and the headline numbers (10 minutes).
2. **`results/RESULTS.md`**: every number in copy-ready tables, regenerated from the files on disk
   (`../venv/bin/python francesco/analysis/make_results_md.py`).
3. **`results/figures/`**: fig 7 training curves, **fig 8** the w3 story, **fig 9** the w2 story, **fig 10** heatmap with the
   adapter (figs 1–6: grid and cost, from `analysis/make_report.py`).
4. **`docs/`**: the full write-ups, in the order the work happened:

| doc | what it covers |
|---|---|
| `docs/01-grid-status.md` | one-page status of the plain GPTQ grid (2026-09-29): trade-off, language damage, what was ruled out |
| `docs/02-cost-and-memory.md` | memory, latency and GPU-hours per precision vs fp16 |
| `docs/03-batch-size-probe.md` | the batch-size probe for slow 2/3-bit generation (2026-09-25), kept as written |
| `docs/04-kernel-repack.md` | lossless w3/w2 → 4-bit repack for the fused kernel; §8 measured speed and drift |
| `docs/05-why-icelandic.md` | why Icelandic: weight-delta analysis, difficulty matching, the NLL result |
| `docs/06-docs/06-distillation.md` | the method, how to run it, all w3 and w2 results, confidence intervals, log of runs and incidents |
| `docs/archive/distillation-lab-notebook.md` | the original chronological notebook of the distillation work, verbatim |
| `../research.md` | the original research plan (RQ1–RQ6) |

| code (`analysis/`) | role |
|---|---|
| `lora.py`, `distill_lora.py`, `generate_lora.py` | adapter, distillation training (`--init-adapter` to continue), generation with adapter |
| `lang_nll.py` | teacher-forced NLL per language (mono vs translation), `--adapter` |
| `bootstrap_ci.py`, `make_results_md.py` | confidence intervals; the all-results file |
| `plot_distill_curves.py`, `plot_w3_story.py`, `plot_w2_story.py`, `plot_heatmap_adapter.py` | figures 7–10 |
| `repack_to_w4.py`, `compare_outputs.py` | lossless repack; line-level drift between two runs |
| `measure_delta.py`, `fragility_by_confidence.py`, `measure_channels.py` | fine-tuning-erasure, difficulty and activation analyses |
| `make_report.py`, `run_probe.py`, `measure_*.py` | the grid/cost report (`results/results.pdf`) and its measurements |
| `sbatch/` | GPU jobs, submitted from the repo root; usage in `docs/06-docs/06-distillation.md` §2 |

Adapters and repacked checkpoints live in `/scratch-shared/$USER/models/` (not in the home quota, not backed up); the
training logs are copied to `results/distill_runs/`. `outputs/ours/wmt22/` contains stray duplicates from a
since-fixed `gen_lora.job` bug; the real files are in `outputs/gptq-w3g128-as4-kd/wmt22/`.
