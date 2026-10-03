# 06 — Recovering quantized quality by distillation (w3 and w2 + LoRA, teacher fp16)

*Reorganised 2026-09-30 from the chronological notebook, which is kept verbatim in
`archive/distillation-lab-notebook.md`. Every number is in both. All results tables in one place:
`../results/RESULTS.md`.*

**Contents:** [0 Status](#0-status-and-headline) · [1 Method](#1-method) · [2 Running it](#2-running-it) ·
[3 Reading the results](#3-how-to-read-the-results) · [4 w3 results](#4-results-w3--kd-rank-16) ·
[5 w2 results](#5-results-w2--kd-rank-64) (incl. [5.8 audit](#58-is-the-w2-recovery-real-audit-of-the-continuation-adapter-2026-10-02-cpu-only)) · [6 Confidence intervals](#6-paired-bootstrap-confidence-intervals) ·
[7 Log](#7-log-runs-incidents-corrections) · [8 Files](#8-files)

---

## 0. Status and headline

| part | status |
|---|---|
| w3 + KD r16: training, test-set NLL (10 directions), BLEU/chrF++/XCOMET (4 directions), CIs | **done** |
| w2 + KD r64: training, test-set NLL, BLEU/chrF++ (is-en, de-en), XCOMET is-en | **done** |
| w2 + KD r64: XCOMET de-en | **done** (92.13, job 27414643); pre-registered verdict in §5.6 |
| w2 + KD r64 continuation: 10 directions × 8 metrics | **done** (§5.7) |
| w2 + KD r64 continuation: "is the recovery real?" audit (overlap, failure modes, error spans, length) | **done** (§5.8), 2026-10-02 |

**Headline.** At 3 bits, a 125 MB adapter distilled from the fp16 model makes GPTQ w3 statistically
indistinguishable from fp16 in 7 of 8 direction × metric comparisons, matching or exceeding w4 at 1.4 GiB less.
The same recipe (rank 64) takes w2 from BLEU 0 to 88–94 % of fp16. The quantizer is never changed.

---

## 1. Method

**Why.** At 3 bits GPTQ loses about half of Icelandic (is-en BLEU −49.6 %, en-is −45.9 %) and far less of the other
languages (`01-grid-status.md` §3). `05-why-icelandic.md` §2 shows that the fine-tuning update survives rounding
(ρ ≈ 1). What hurts is rounding noise about 9× larger than that update, spread isotropically over each matrix.
A low-rank adapter can't undo full-rank weight noise, but it can correct the *outputs* where they matter. The
test-set NLL (`05-why-icelandic.md` §1) located the damage in the translation mapping, so the loss targets
translation tokens.

**Recipe.**
- **Student:** the GPTQ checkpoint with every weight frozen (packed, TorchQuantLinear), plus LoRA on all 7 Linears of
  all 40 decoder layers. w3: r = 16, α = 32 (62.5 M params, ~125 MB in bf16, +2.3 % on 5.27 GiB). w2: r = 64, α = 128
  (250 M, ~0.47 GiB), see §5.1.
- **Teacher:** fp16 ALMA-13B-R, the model the quantized one should behave like.
- **Loss:** token-level forward KL(p_teacher ‖ p_student) over the full vocabulary, on the target tokens only.
- **Data:** ALMA's human-written parallel **train** files, in ALMA's fine-tuning format (the calibration format),
  with no overlap with the test sets. Default is 30 k examples: every Icelandic row twice (2 × 4,018), the rest split
  evenly over the other 8 directions (~2,750 each). 32 rows per language pair are held out of training in both directions.
- **Optimiser:** AdamW, lr 2e-4, 3 % warm-up then cosine to 0, micro-batch 16 × grad-accum 2 = 32 sentences per
  step, one epoch = 936 steps, gradient checkpointing, adapter in fp32.

**Why distil rather than fine-tune on the references.** ALMA-R was trained with CPO (a preference-optimisation
method) precisely because the references are sometimes worse than the model's own outputs, so SFT on them can pull
the model back toward the weaker SFT model. The teacher target avoids that. The same script can run the SFT variant
as a comparison (`EXTRA="--kd-weight 0 --ce-weight 1"`), and the fp16 + LoRA control by pointing `--student` at the
fp16 folder. **Neither comparison has been run.** This covers two of the brief's techniques on top of quantization:
PEFT (LoRA) and knowledge distillation.

**Kernels: train on one, generate on the other.**

| step | checkpoint | kernel | why |
|---|---|---|---|
| training, NLL | `models/ALMA-13B-R-gptq-w{3,2}g128` | TorchQuantLinear (bf16) | the only GPTQ kernel here with a backward pass (`SUPPORTS_TRAINING = True`; ExllamaV2 is `False`) |
| generation | `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w{3,2}g128-as-w4` | ExllamaV2QuantLinear (fp16) | same weights (lossless repack), 2.3–2.6× faster (`04-kernel-repack.md` §8) |

`analysis/lora.py` wraps any Linear-like module (`y = base(x) + α/r · x Aᵀ Bᵀ`, adapter in fp32), so one adapter
works on both kernels. peft isn't used, because it decides per kernel class whether it can wrap a layer.

**Comparison rule.** Repacked generations differ slightly from the scored w3 grid (kernel drift,
`04-kernel-repack.md` §8). So w3 + KD is judged against **w3-as4 without the adapter** (`outputs/gptq-w3g128-as4`),
generated with the same kernel and script, never against `outputs/gptq-w3g128`. For w2 the baseline is the scored
grid `gptq-w2g128`: it is at BLEU ~0, so kernel drift can't matter there.

---

## 2. Running it

All jobs are submitted from the repo root. Adapters and repacked checkpoints live in `/scratch-shared/$USER/models/`.

| step | command | time | output |
|---|---|---|---|
| smoke test | `sbatch --export=ALL,SMOKE=1 francesco/analysis/sbatch/distill.job` | ~4 min | `…-kd-lora-smoke/`, 8 generated lines |
| train w3 | `sbatch francesco/analysis/sbatch/distill.job` | 43 min (H100) | `…-w3g128-kd-lora/` |
| train w2 r64 | `sbatch --export=ALL,BITS=2,ADAPTER=/scratch-shared/$USER/models/ALMA-13B-R-gptq-w2g128-kd-lora-r64,EXTRA="--r 64 --alpha 128" francesco/analysis/sbatch/distill.job` | 40 min | `…-w2g128-kd-lora-r64/` |
| continue a run | `EXTRA="--r 64 --alpha 128 --n-examples 60000 --init-adapter <adapter dir>"` | ~80 min | new folder, fresh lr schedule, same seed (same held-out rows) |
| eval w3 (4 directions) | `sbatch francesco/analysis/sbatch/eval_distill.job` | ~3 h | NLL, plain w3-as4 + w3+KD generation, XCOMET-XXL, lexical |
| gated eval (w2) | `sbatch --export=ALL,BITS=2,TAG=…,KD_RUN=…,ADAPTER=… francesco/analysis/sbatch/eval_distill_gated.job` (add `--partition=gpu_a100 … SKIP_XCOMET=1` on an A100) | ~1 h | NLL, gate (translation dNLL < 1.0), is-en/de-en generation, scores |
| generate any directions | `PAIRS=a,b BITS=… ADAPTER=… KD_RUN=… sbatch --export=ALL francesco/analysis/sbatch/gen_lora.job` (optional `NLL=1 TAG=…`) | ~1.2 s/line (A100) | `outputs/$KD_RUN/wmt22` |
| all 8 metrics | `RUN=<run> sbatch --export=ALL francesco/analysis/sbatch/score_full.job` | ~1.5 h (H100) | `outputs/baseline/<run>.tsv` |
| XCOMET subset | `KD_RUN=<run> PAIRS=a,b sbatch --export=ALL francesco/analysis/sbatch/score_xcomet.job` | ~2 min/direction (H100) | `outputs/xcomet-xxl/<run>/` |
| full w3-as4 grid (optional) | `sbatch francesco/analysis/sbatch/eval_w3as4.job` | ~7 h | 10-direction speed-up and drift |

**Pitfalls.** Never put a comma list inside `--export=…`: Slurm splits it and only the first item arrives (§7). Pass
lists through the environment as above, or as `a:b:c`. `scontrol update … Dependency=` is not permitted here, so
cancel and resubmit a pending job instead. XCOMET-XXL / KIWI-XXL (~43 GB) need an H100. Generation and training fit a
40 GB A100 (training peaks at 37 GiB, tight). Training on an A100 would be ~3× slower.

**Smoke-test checklist** (`logs/slurm-distill-<id>.out`):
- `grad_check`: 280/560 LoRA tensors get a nonzero gradient on step 1. That's the lora_B half; lora_A gets none on the
  first step because B starts at 0. **0/560 means gradients do not flow through the kernel: stop.**
- The `kd` loss is finite and falls over the 30 steps, `peak_mem_gib` is below ~90, and `full_epoch_eta_min` gives the
  full run's time.
- Step-0 `heldout.kl` per direction is the plain-model damage measured as KL. Icelandic should be the largest.
- The 8 en-is lines at the end are real Icelandic sentences, which shows the adapter loads on ExllamaV2.

---

## 3. How to read the results

- **Main numbers:** XCOMET-XXL and BLEU of the adapter run vs the plain run on the same kernel, on is-en/en-is (worst
  hit) and de-en/en-de (barely damaged, the control: does the adapter hurt a language that was fine?). fp16 `ours-beam`
  is the ceiling.
- **Recovered fraction** = (KD − plain) / (fp16 − plain), per direction. It is imprecise when the gap is small
  (German), so report it with its CI or as the absolute difference to fp16 (§6).
- **NLL:** `trans:` (translation) vs `mono:` (plain language) shows *what* the adapter restored. That ties back to
  `05-why-icelandic.md` §1.
- **Pre-registered criterion for w2** (fixed at 17:15 before any w2 XCOMET existed): ≥ 90 % of the XCOMET-XXL gap
  recovered on **both** is-en and de-en = "recovered most of the translation ability"; 70–90 % = "bulk recovered,
  clear residual"; < 70 % = "partial". Also compare the share of segments with XCOMET < 0.5 against fp16, plain w3
  and plain w2, and place the result against plain w3 and w4.
- A clean negative result is also reportable, e.g. "recovers German but not Icelandic".

---

## 4. Results: w3 + KD (rank 16)

### 4.1 Training

| run | job | steps | time | peak mem | train KD loss | held-out KL (is-en) |
|---|---|---|---|---|---|---|
| smoke | 27400056 | 30 | 3 min 40 s total | 35.9 GiB | 0.42 → 0.047 | 0.696 → 0.048 |
| full | 27400299 | 936 | 43 min | 35.9 GiB | → ~0.03 | 0.696 → 0.032 |

Smoke test: gradients flow through TorchQuantLinear (`grad_check` 280/560, as expected); 2.8 s/step, ~1,200 tok/s;
the adapter loads on the repacked w3 (ExllamaV2) and produces fluent Icelandic for the first 8 en-is test lines.
Held-out KL per direction (32 rows per pair, train distribution, not the test set):

| pair | step 0 (plain w3) | smoke, step 30 |
|---|---|---|
| is-en | 0.696 | 0.048 |
| zh-en | 0.652 | 0.065 |
| en-zh | 0.564 | 0.049 |
| en-is | 0.482 | 0.048 |
| cs-en | 0.424 | 0.038 |
| en-cs | 0.357 | 0.041 |
| ru-en | 0.316 | 0.040 |
| en-ru | 0.282 | 0.028 |
| de-en | 0.262 | 0.036 |
| en-de | 0.248 | 0.037 |

At step 0 the ranking matches the damage seen elsewhere: is-en worst, then zh-en, and German least. The full run ends
at 0.023–0.047 in every direction, flat after ~step 200 (`../results/figures/fig7_distill_curves.png`).

### 4.2 Test-set NLL (job 27400300 stage 1; `results/json/lang_nll_*.json`)

dNLL vs fp16, nats/token, WMT'22 (WMT'21 for is), 500 sentences per item, not in the training data:

| item | w4 | w3 | **w3 + KD r16** | w2 | **w2 + KD r64** |
|---|---|---|---|---|---|
| trans:is-en | +0.027 | +0.642 | **+0.006** | +4.654 | **+0.179** |
| trans:en-is | +0.028 | +0.575 | **+0.025** | +6.777 | **+0.320** |
| trans:zh-en | +0.032 | +0.415 | **+0.012** | +4.229 | **+0.208** |
| trans:en-zh | +0.038 | +0.270 | **+0.018** | +6.495 | **+0.217** |
| trans:cs-en | +0.013 | +0.372 | **+0.005** | +4.555 | **+0.138** |
| trans:en-cs | +0.019 | +0.182 | **+0.010** | +6.335 | **+0.168** |
| trans:ru-en | −0.001 | +0.256 | **+0.003** | +4.565 | **+0.122** |
| trans:en-ru | +0.017 | +0.213 | **+0.007** | +8.518 | **+0.172** |
| trans:de-en | −0.001 | +0.074 | **−0.003** | +3.468 | **+0.038** |
| trans:en-de | +0.027 | +0.163 | **+0.012** | +6.974 | **+0.190** |
| mono:is | +0.069 | +0.486 | **+0.086** | +4.687 | **+1.198** |
| mono:de | +0.125 | +0.501 | **+0.107** | +4.476 | **+1.017** |
| mono:cs | +0.097 | +0.833 | **+0.084** | +4.771 | **+1.066** |
| mono:zh | +0.068 | +0.218 | **+0.110** | +2.907 | **+0.852** |
| mono:ru | +0.133 | +0.732 | **+0.117** | +4.849 | **+1.085** |
| mono:en | +0.104 | +0.414 | **+0.059** | +2.925 | **+0.711** |

- w3 + KD: translation NLL recovers 96–99 % of the w3 damage in every direction, including Icelandic. It is at or below
  w4 in 9 of 10 directions (ru-en: +0.003 vs w4 −0.001, negligible). The notebook said "every direction", corrected here.
- Plain-text (`mono`) NLL also recovers 50–90 % for w3, although only translation targets were trained on.
- Teacher-forced likelihood, not generation; §4.3 is the generation check.

### 4.3 Generation (job 27400300, done 16:39; is-en/en-is generated by gen-lora 27403647)

Full test sets, beam 5, repacked w3 (ExllamaV2) with and without the adapter. fp16 = `ours-beam`; w4 = scored grid.

| pair | BLEU fp16 / w4 / w3 / **w3+KD** | rec. | chrF++ fp16 / w4 / w3 / **w3+KD** | rec. | XCOMET-XXL fp16 / w4 / w3 / **w3+KD** | rec. |
|---|---|---|---|---|---|---|
| is-en | 40.21 / 39.09 / 21.62 / **39.75** | 98 % | 61.72 / 61.13 / 44.49 / **61.12** | 97 % | 81.15 / 80.70 / 46.65 / **81.23** | 100 % |
| en-is | 22.26 / 21.82 / 11.66 / **22.07** | 98 % | 50.58 / 50.32 / 37.74 / **50.09** | 96 % | 92.82 / 91.73 / 66.71 / **92.83** | 100 % |
| de-en | 31.59 / 31.18 / 28.91 / **31.41** | 93 % | 55.27 / 55.03 / 52.80 / **55.10** | 93 % | 94.66 / 94.24 / 89.09 / **94.49** | 97 % |
| en-de | 28.05 / 27.03 / 25.20 / **28.01** | 98 % | 55.66 / 54.95 / 52.60 / **55.41** | 92 % | 97.71 / 97.53 / 95.16 / **97.55** | 94 % |

- **w3 + KD (5.4 GiB) matches fp16 on XCOMET-XXL for Icelandic** (81.23 vs 81.15, 92.83 vs 92.82).
  Against w4 (6.8 GiB) it is numerically higher everywhere, but significantly so only in 2 of 8 cases (§6), so the
  claim is "matches or exceeds w4".
- Over-long outputs (≥ 2× reference length) on en-is fall from 10.6 % to 0.2 % (fp16 0.2 %).
- XCOMET-XXL is scored without references (as in the paper). Differences of ±0.1 from fp16 are within noise.
- Only 12–13 % of de-en/en-de lines are identical with vs without the adapter, so the adapter changes most outputs,
  not only the broken ones.
- Against the originally scored w3 grid (Torch kernel) instead of w3-as4, recovery is also 98 % on is-en and en-is.
- Scores: `outputs/xcomet-xxl/gptq-w3g128-as4{,-kd}/`, `results/json/distill_lexical.json`. Figures:
  `../results/figures/fig8_w3_story.png`, `fig10_heatmap_adapter.png`.
- Caveats: 4 of 10 directions, one seed, one training run. is-en and en-is of `gptq-w3g128-as4-kd` were written to
  `outputs/ours/wmt22` by a since-fixed bug and copied (identical settings, §7).

---

## 5. Results: w2 + KD (rank 64)

### 5.1 Design

**Question:** does the same recipe rescue 2 bits, where the model is broken everywhere (plain-text dNLL +2.9 to +4.8,
translation +3.5 to +8.5, BLEU ~0) rather than only on the translation mapping as at w3?
- **Changes vs w3:** student `ALMA-13B-R-gptq-w2g128`, **r = 64, α = 128** (same α/r = 2; 250 M params, ~0.47 GiB in
  bf16, ~1 GB as saved in fp32), because w2's rounding noise is about 2× w3's and a rank-16 adapter may hit a capacity
  ceiling. Data, steps, lr and loss are identical.
- **Confound:** bits and rank both change vs the w3 run. A w2 r = 16 run would isolate the rank (not run).
- **Scope rule:** only the adapter may change; the quantizer (group size, rotations, mixed precision, scales) stays
  fixed. Rank 256 was considered and rejected for deployment: at 1.87 GiB, w2 + r256 = 5.65 GiB > w3 + r16 (5.39 GiB).

### 5.2 Training

| run | job | steps | time | train KD loss | held-out KL is-en / en-is / de-en / zh-en |
|---|---|---|---|---|---|
| r64 | 27405387 | 936 | 40 min, peak 37.2 GiB | 6.8 → ~0.20 | 5.06 → 0.241 / 6.32 → 0.307 / 4.18 → 0.194 / 4.79 → 0.316 |
| r64 continued | 27409780 | +1,873 (60 k examples) | 80 min | ~0.17–0.20 at the end | 0.241 → 0.292 (step 200, lr re-warm) → **0.215** (step 1,400) … see `results/distill_runs/` |

- r64: held-out KL fell from 4.2–9.2 to 0.17–0.32 and was still falling when the cosine schedule reached lr = 0, so
  a continuation was run: `distill_lora.py --init-adapter` starts from the finished adapter with a fresh schedule, same
  seed (same held-out rows). The re-warm to lr 2e-4 first made it worse (step 200: is-en 0.292, en-is 0.363), then it
  dropped below the first run from ~step 1,000 (step 1,200: is-en 0.217, de-en 0.160, en-is 0.315; step 1,400: 0.215 /
  0.156 / 0.299). Curves: `../results/figures/fig7_distill_curves.png`.

### 5.3 Test-set NLL (eval-gated 27409151, A100) and gate

Full table in §4.2. Translation dNLL goes from +3.5 to +8.5 (plain w2) to **+0.04 to +0.32**, recovering 95–98 % of the
w2 damage. It is better than plain w3 in 8 of 10 directions (is-en +0.18 vs +0.64; en-de +0.19 vs +0.16 is the one
clearly worse; en-cs is about tied). Plain-text dNLL goes from +2.9 to +4.8 to +0.71 to +1.20, recovering only ~75 %,
and stays *worse* than plain w3 (+0.22 to +0.83): the adapter restored the task much more than the languages themselves
(it was trained on translation only). Figure: `../results/figures/fig9_w2_story.png` panel D ("mirror images").
**Gate** (translation dNLL < 1.0): is-en +0.179 and de-en +0.038 both pass.

### 5.4 Generation (eval-gated 27409151, A100, done 17:10; `results/json/distill_w2kd-r64.json`)

Full test sets, beam 5, repacked w2 (ExllamaV2) + adapter. Baseline is the scored grid `gptq-w2g128`.

| pair | metric | fp16 | w3 + KD (§4.3) | plain w3 (as4) | w2 (grid) | **w2 + KD r64** | recovered |
|---|---|---|---|---|---|---|---|
| is-en | BLEU | 40.21 | 39.75 | 21.62 | 0.00 | **35.40** | 88 % [86, 90] |
| is-en | chrF++ | 61.72 | 61.12 | 44.49 | 4.37 | **56.83** | 91 % |
| is-en | XCOMET-XXL | 81.15 | 81.23 | 46.65 | 12.60 | **74.45** | 90 % |
| de-en | BLEU | 31.59 | 31.41 | 28.91 | 0.10 | **29.63** | 94 % [91, 95] |
| de-en | chrF++ | 55.27 | 55.10 | 52.80 | 4.71 | **53.02** | 96 % |
| de-en | XCOMET-XXL | 94.66 | 94.49 | 89.09 | 14.72 | **92.13** | 97 % |
| halluc. is-en / de-en | | 0.0 / 0.05 % | 0.0 / 0.0 % | | 42.7 / 62.1 % | **0.0 / 0.25 %** | |

- From total collapse to 88–94 % of fp16 BLEU with a ~0.47 GiB adapter on a 3.78 GiB checkpoint (4.25 GiB, 5.7×
  smaller than fp16). Sample output (is-en line 2): *"The marketing campaign 'Looks Like You Need Iceland' seems to have
  gone well and the cries and calls in the wild are starting to draw people from all over the world."*
- vs plain w3 (1 GiB larger): **significantly above on is-en** (+13.78 BLEU [+12.66, +14.87]); **a tie on de-en**
  (+0.71 [−0.15, +1.45]). The notebook said "above plain w3 here too" for de-en, corrected by §6.
- Clearly below w3 + KD (−4.4 BLEU on is-en, −1.8 on de-en). The recipe recovers most of 2 bits but not all.
- Generated on an A100, not an H100 (small arithmetic drift, irrelevant against plain w2 at BLEU ~0).
- Caveats: 2 directions (xx→en only), rank 64 vs w3's 16 (confound, §5.1), one run.

### 5.6 Pre-registered XCOMET verdict (criterion fixed at 17:15, results 17:45)

| XCOMET-XXL | fp16 | plain w3 (as4) | plain w2 | **w2 + KD r64** | recovered |
|---|---|---|---|---|---|
| is-en mean | 81.15 | 46.65 | 12.60 | **74.45** | **90.2 %** |
| de-en mean | 94.66 | 89.09 | 14.72 | **92.13** | **96.8 %** |
| is-en segments < 0.5 | 13.5 % | 58.9 % | 99.9 % | **18.7 %** | |
| de-en segments < 0.5 | 1.1 % | 3.3 % | 94.6 % | **1.5 %** | |

- Criterion 1 (≥ 90 % on both is-en and de-en): **met** (is-en narrowly).
- Criterion 2 (failure tail): German at fp16 level; Icelandic keeps a modestly larger tail (18.7 % vs 13.5 %).
- **Verdict: w2 + KD r64 recovered most of the translation ability, with a modestly larger share of failed
  Icelandic sentences than fp16.**
- Bootstrap (§6 method): is-en XCOMET recovered **90 % [88, 92]**, so the point estimate meets the threshold but the CI
  straddles it; de-en 97 % [96, 97]. Against plain w3 (1 GiB larger), w2 + KD is significantly better on XCOMET in both
  directions: is-en +27.80 [+25.82, +29.76], de-en **+3.04 [+2.43, +3.68]** (BLEU on de-en was a tie).

**Continuation adapter, test-set NLL** (job 27414048, `results/json/lang_nll_w2kd-r64-cont.json`): translation dNLL
improves over the first r64 adapter in all 10 directions (is-en +0.179 → +0.144, en-is +0.320 → +0.298, de-en +0.038 →
+0.025, en-de +0.190 → +0.159, cs-en +0.138 → +0.106, en-cs +0.168 → +0.131, zh-en +0.208 → +0.173, en-zh +0.217 → +0.180,
ru-en +0.122 → +0.094, en-ru +0.172 → +0.153). Plain-text dNLL gets slightly worse in every language (is +1.198 → +1.220,
de +1.017 → +1.035, cs +1.066 → +1.103, zh +0.852 → +0.896, ru +1.085 → +1.102, en +0.711 → +0.798): longer training
specialises the adapter further on translation.

### 5.6b Peak GPU memory with the adapters (probe-adapter 27418948 + 27419167, A100, de-en, beam 5, source 256)

`analysis/sbatch/probe_adapter.job` (`run_probe.py --adapter`), adapter held in fp32 as in generation.
Results: `results/json/probe16_{w3kd,w2kd,w3as4kd,w2as4kd}.json`, `probe16_w2kd_b1.json`.

| model | kernel | batch 1 | batch 2 | batch 4 | batch 16 |
|---|---|---|---|---|---|
| fp16 | – | | | 28.78 | |
| w4 | ExllamaV2 | | | 12.32 | |
| plain w3 (packed) | Torch | | | 10.15 | |
| **w3 + KD r16 (packed)** | Torch | | | **10.41** | 25.10 |
| plain w2 (packed; runs to 256 tokens) | Triton | | | 11.75 | |
| **w2 + KD r64 (packed)** | Triton | **5.94** (reserved 6.03) | 7.17 (reserved 7.21) | **9.62** | 24.31 |
| w3 + KD r16 (4-bit container) | ExllamaV2 | | | 12.88 | 27.57 |
| w2 + KD r64 (4-bit container) | ExllamaV2 | | | 13.58 | 28.27 |

**Convention: all claims count the adapter in bf16** (0.117 GiB for r16, 0.466 GiB for r64). The runs keep it in fp32, so
the bf16-counted peak = measured − the fp32→bf16 difference of the adapter parameters (−0.117 / −0.466 GiB; exact for the
parameters, conservative for temporaries). **bf16-counted:** w3 + KD packed batch 4 **10.29**; w2 + KD packed batch 1
**5.48**, batch 2 6.70, batch 4 **9.15**. Quality numbers were produced with the fp32 adapter; the effect of bf16 rounding
of the adapter on translation quality is expected to be negligible but is **not measured**.

GiB, `peak_alloc`, as measured (fp32 adapter). **At batch 1 (one sentence at a time, beam 5), packed w2 + adapter peaks at 5.9 GiB
measured, 5.5 GiB with the adapter counted in bf16**, leaving ~1.5 GiB
on an 8 GB card after the CUDA context. So the claim is: "fits an 8 GB consumer GPU at batch 1, beam 5". Batch 2 (7.2 GiB)
is too tight (6.7 GiB bf16-counted, still tight). Caveats: measured on an A100, not a
consumer card; needs CUDA + bf16 + Triton, so NVIDIA RTX 30/40-class, not e.g. an AMD RX 5700 (RDNA1 isn't supported by
ROCm). On the 4-bit container the weights take w4's size, so memory is *higher* than w4. Speed on the A100 isn't
comparable with the H100 probes.

### 5.5 Full 10-direction, 8-metric evaluation of the continuation adapter (running)

Decision (2026-09-30): **w2 gets the full treatment** because it is the more surprising result. w3 + KD keeps
10-direction NLL plus 4-direction BLEU/chrF++/XCOMET. Adapter `…-w2g128-kd-lora-r64-cont`; output run
`gptq-w2g128-as4-kd-r64-cont`.

| job | what | GPU | after |
|---|---|---|---|
| 27414048 | NLL → `lang_nll_w2kd-r64-cont.json`, then zh-en (source 512, batch 1) | A100 | continuation 27409780 OK |
| 27414049 | de-en | A100 | ″ |
| 27414050 | cs-en | A100 | ″ |
| 27414630 | en-is, is-en, en-de | A100 | — |
| 27414640 | en-ru, ru-en | A100 | — |
| 27414642 | en-cs, en-zh | A100 | — |
| 27414648 | all 8 metrics + `summarize.py --vs ours-beam` → `outputs/baseline/gptq-w2g128-as4-kd-r64-cont.tsv` | H100 | all six OK |

The first three were meant to carry 3–4 directions each (§7, `--export` bug). Directions are disjoint, so there's no
duplicated work. Estimated cost: ~6.5 A100-h (~830 SBU) + ~1.5 H100-h (~290 SBU).

---

### 5.7 Result: all 10 directions × 8 metrics (score-full 27414648, done 21:37)

Run `gptq-w2g128-as4-kd-r64-cont` (continuation adapter, packed-equivalent w2 on the 4-bit container, A100 generation),
scored exactly like the grid runs. Averages over the 5 directions each way (`outputs/baseline/*.tsv`); MetricX-24 and
halluc: lower is better. Adapter size counted in bf16.

| model | GiB | dir | BLEU | chrF++ | COMET-22 | KIWI-22 | KIWI-XXL | XCOMET-XXL | MetricX-24 ↓ | halluc % ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 24.24 | xx→en | 35.82 | 60.14 | 85.29 | 81.50 | 82.74 | 89.57 | 2.88 | 0.16 |
| w4 | 6.76 | xx→en | 35.24 | 59.73 | 84.93 | 81.24 | 82.16 | 88.76 | 3.02 | 0.07 |
| w3 | 5.27 | xx→en | 25.63 | 50.70 | 78.39 | 75.27 | 68.12 | 73.34 | 5.34 | 0.74 |
| w2 | 3.78 | xx→en | 0.04 | 3.95 | 24.35 | 29.16 | −0.14 | 13.05 | 9.42 | 49.83 |
| **w2 + KD r64** | **4.25** | xx→en | **32.63** | **57.06** | **83.70** | **80.40** | **79.28** | **86.22** | **3.49** | **0.28** |
| fp16 | 24.24 | en→xx | 26.98 | 47.39 | 88.07 | 83.70 | 86.79 | 94.55 | 1.87 | 0.57 |
| w4 | 6.76 | en→xx | 25.71 | 46.81 | 87.71 | 83.45 | 85.87 | 93.79 | 1.95 | 0.81 |
| w3 | 5.27 | en→xx | 20.91 | 41.22 | 82.02 | 77.48 | 71.15 | 83.37 | 4.09 | 4.01 |
| w2 | 3.78 | en→xx | 0.03 | 3.41 | 29.74 | 23.56 | −2.12 | 32.27 | 10.40 | 47.43 |
| **w2 + KD r64** | **4.25** | en→xx | **23.03** | **43.58** | **85.73** | **81.62** | **79.25** | **89.91** | **2.84** | **1.20** |

- **w2 + KD beats plain w3 (1 GiB larger) on every metric in both directions**, and sits between w3 and w4 overall:
  close to w4 on COMET-22 / KIWI-22 / XCOMET / MetricX, further behind on BLEU and KIWI-XXL.
- Share of the fp16 − w2 gap recovered (averages): xx→en BLEU 91 %, XCOMET 96 %, COMET-22 97 %, MetricX 91 %;
  en→xx BLEU 85 %, XCOMET 93 %, COMET-22 96 %, MetricX 89 %.
- Per direction, XCOMET recovers ≥ 90 % in all 10 (en→zh 90 % lowest, de→en 98 % highest); BLEU 78–95 %
  (en→is lowest). Figures: `../results/figures/fig11_*` to `fig13_*`.
- Caveats: one seed; generated on an A100 with the 4-bit container (bit-identical weights to packed w2); adapter in fp32 at
  run time (sizes and memory counted in bf16).

### 5.8 Is the w2 recovery real? Audit of the continuation adapter (2026-10-02, CPU only)

**Question:** is w2 + KD really translating, or does it fool the metrics (fluent hallucinations, loops, copying,
memorised test data)? Two scripts on the existing generations, no new GPU runs:
`analysis/audit_recovery.py` (overlap, length, XCOMET error spans, length buckets, similarity to the teacher, worst-segment
dumps) and `analysis/score_failures.py` (failure taxonomy below). Systems: fp16 (`ours-beam`), w4, plain w3 (grid),
plain w2, w2 + KD r64-cont; all 10 directions.

**Clean:**
- **Train/test overlap: 0 %** in all 10 directions: exact normalised source, exact reference, and any shared 8-gram
  between a test source and the ALMA train file.
- **Lexical and neural metrics agree:** BLEU, chrF++ and XCOMET recover together (§5.7), so no "XCOMET high, chrF low"
  fluent-hallucination pattern.
- **Not a teacher clone:** BLEU of w2 + KD against fp16's own outputs is 30–61 (w4: 59–80), so it writes its own
  translations of similar quality.
- **Number preservation** (share of segments missing a source number): at or below fp16 everywhere (e.g. is-en 1.1 %
  vs 1.3 %, en-ru 2.1 % vs 3.1 %).
- Data budget argument: the adapter saw ~8 k Icelandic pairs (4,018 unique rows, twice). A 250 M-parameter adapter can't
  learn is→en to BLEU 36 from that, so the knowledge must come from the frozen 2-bit weights (argument, not a test; see
  "Open" below).

**Failure taxonomy** (`score_failures.py`, % of segments):
- *osc* = TNG (Raunak et al. 2021, as used in Guerreiro et al. 2023 EACL §3): top repeated 4-gram of the output occurs
  ≥ 2 more times than the top repeated 4-gram of the source. Plain flag, without the paper's "1 % lowest quality" condition.
- *off-target* (Zhang et al. 2020): fastText NLLB LID (`/scratch-shared/$USER/models/lid/lid218e.bin`, HF
  `facebook/fasttext-language-identification`) label ≠ target. **For a Chinese target the LID is unusable** (it calls
  "他否认了这项指控。" Irish and flags 25 % of the references), so there: fewer than half of the letters are Han.
- *copy*: output = source after lowercasing and dropping non-word characters, and the reference isn't.
- *empty*: 0 tokens. *trunc*: 0 < output tokens / reference tokens < 0.5 (sacrebleu 13a / zh tokens).

Any failure:

| | de-en | cs-en | is-en | zh-en | ru-en | en-de | en-cs | en-is | en-zh | en-ru |
|---|---|---|---|---|---|---|---|---|---|---|
| reference (LID noise floor, off-target only) | 0.25 | 0.69 | 0.20 | 1.17 | 1.74 | 0.29 | 0.88 | 0.00 | 1.82 | 1.67 |
| fp16 | 0.45 | 0.55 | 0.20 | 3.09 | 1.69 | 0.54 | 1.28 | 0.10 | 1.28 | 1.72 |
| w4 | 0.50 | 0.55 | 0.20 | 3.47 | 1.54 | 0.59 | 1.13 | 0.00 | 1.37 | 1.08 |
| plain w3 | 0.71 | 0.76 | 3.50 | 6.83 | 2.08 | 0.49 | 3.14 | 14.70 | 7.02 | 3.29 |
| plain w2 | 97.38 | 99.59 | 100.00 | 97.39 | 99.95 | 99.95 | 99.80 | 100.00 | 100.00 | 100.00 |
| **w2 + KD r64-cont** | 0.45 | 0.55 | 0.20 | 3.31 | 1.69 | **2.41** | 1.47 | **1.20** | **4.17** | 1.91 |

- Empty: 0 everywhere for w2 + KD. Trunc: at fp16 level (≤ 0.43 %, zh-en, same as fp16).
- **w2 + KD has fewer failures than plain w3 (1 GiB larger) in 9 of 10 directions**; the exception is en-de.
- **Residual vs fp16** (osc): en-zh 2.75 % vs 0.29 %, en-is 1.10 % vs 0.10 %, en-ru 0.54 % vs 0.00 %. Extreme cases:
  "轻轻轻…" (42× the reference length), "Řečeno v angličtině: ×20", "no, no, no, …" (de-en, 1 segment). These loops also
  inflate the length-ratio std (en-zh 1.60 vs 0.27, en-cs 1.01 vs 0.21, de-en 0.72 vs 0.17).
- **Residual vs fp16** (copy): en-de 1.77 % vs 0.00 %, almost all short UI strings ("Tap Settings." → "Tap Settings."),
  which also raises en-de off-target to 1.62 % (fp16 0.54 %).
- The earlier `halluc %` (output ≥ 2× reference length, §5.7) mixes loops with merely long outputs; this taxonomy separates them.

**Below fp16, though not catastrophically** (`audit_recovery.py`, XCOMET-XXL error spans):

| | is-en | en-is | cs-en | en-cs | zh-en | en-zh |
|---|---|---|---|---|---|---|
| critical spans fp16 / w2 + KD / plain w3 | 320 / 415 / 809 | 46 / 77 / 425 | 332 / 476 / 773 | 112 / 194 / 289 | 12 / 93 / 158 | 123 / 211 / 340 |
| segments XCOMET < 0.5, fp16 / w2 + KD | 13.5 / 17.4 | 1.2 / 4.6 | 5.4 / 9.3 | 2.3 / 5.2 | 1.1 / 2.4 | 1.8 / 5.7 |

- **The gap grows with source length.** w2 + KD − fp16 XCOMET by source-length quartile (short → long): en-is −3.6 /
  −5.9 / −7.6 / −12.4, en-zh −2.6 / −4.5 / −5.8 / −9.2, is-en −1.9 / −3.8 / −5.9 / −6.6, en-cs −2.9 / −4.0 / −5.5 / −7.9.
  ru-en is flat (−2.5 throughout).
- **New failures in en-is:** only 12 % of w2 + KD's worst 5 % (XCOMET) are also among fp16's worst 5 % (other
  directions 31–62 %). The 40 segments with the largest loss vs fp16 per direction are in
  `../results/audit/<pair>.worst_vs_fp16.tsv` with source, reference, both outputs and XCOMET error spans. **Not yet read
  manually.**

**Verdict.** The recovery is real: plain w2 fails on 97–100 % of segments, w2 + KD on 0.2–4.2 %, at fp16 level in 7 of
10 directions, with no test-set leakage and lexical and neural metrics in agreement. What remains: rare repetition loops
in en→xx (en-zh, en-is, en-ru), English left untranslated in en-de UI strings, more critical errors than fp16, and a
gap that widens on long sentences. The honest claim stays "recovered most of the **translation** ability" (plain-text
NLL recovers only ~75 %, §5.3).

**Trap found: FLORES is in the training data.** All 997 dev and 1,012 devtest FLORES-200 Icelandic sentences are verbatim
in ALMA's `train.is-en.json` (ALMA's human-written data = earlier WMT test sets + FLORES dev/devtest), so FLORES for
cs/de/is/ru/zh can't be an out-of-distribution test. FLORES fra/spa (in `data/flores200_dataset`) are unseen by ALMA.

**Open (not run, adapter or evaluation only, so within scope):**
1. **Leave Icelandic out of training** (r64 on de/cs/ru/zh only), test is-en / en-is: does the adapter repair the model
   generally or only the trained directions? The decisive test.
2. **Unseen languages:** FLORES fr-en, es-en and a non-English pair (de→cs); plain w2 vs w2 + KD vs fp16.
3. Fresh test set: WMT24++ (is, de, cs, ru, zh) or NTREX-128.
4. Prompt / decoding robustness: reworded template, greedy instead of beam 5.
5. Second training seed.
6. `no_repeat_ngram_size` or a repetition penalty on en-zh / en-is / en-ru: is the loop tail a decoding problem?
7. Manual / blind reading of the worst-40 dumps and 50 random is-en segments.

## 6. Paired bootstrap confidence intervals

`analysis/bootstrap_ci.py`, B = 1,000, seed 12345. The test sentences are resampled with the same indices for every
system (paired). BLEU is recomputed from summed n-gram statistics; XCOMET is the resampled mean of segment scores.
Full output: `results/json/bootstrap_ci.json`. Bold = CI excludes 0.

| | is-en | en-is | de-en | en-de |
|---|---|---|---|---|
| w3+KD recovered, BLEU | 98 % [95, 100] | 98 % [93, 104] | 93 % [80, 108] | 98 % [83, 118] |
| w3+KD recovered, XCOMET | 100 % [98, 102] | 100 % [98, 103] | 97 % [92, 101] | 94 % [89, 99] |
| w3+KD − fp16, BLEU | −0.46 [−1.01, +0.07] | −0.19 [−0.79, +0.38] | −0.18 [−0.56, +0.17] | −0.04 [−0.51, +0.42] |
| w3+KD − fp16, XCOMET | +0.08 [−0.59, +0.72] | +0.02 [−0.61, +0.68] | −0.17 [−0.43, +0.08] | **−0.16 [−0.30, −0.03]** |
| w3+KD − w4, BLEU | +0.66 [−0.02, +1.33] | +0.25 [−0.42, +0.89] | +0.23 [−0.17, +0.68] | **+0.97 [+0.45, +1.50]** |
| w3+KD − w4, XCOMET | +0.53 [−0.28, +1.34] | **+1.11 [+0.42, +1.77]** | +0.25 [−0.03, +0.55] | +0.02 [−0.15, +0.19] |
| w3+KD − plain w3, BLEU | **+18.13 [+16.87, +19.31]** | **+10.41 [+9.47, +11.42]** | **+2.49 [+1.81, +3.27]** | **+2.81 [+2.22, +3.39]** |
| w2+KD recovered, BLEU | 88 % [86, 90] | | 94 % [91, 95] | |
| w2+KD − plain w3, BLEU | **+13.78 [+12.66, +14.87]** | | +0.71 [−0.15, +1.45] | |
| w2+KD − fp16, BLEU | **−4.81 [−5.72, −4.01]** | | **−1.97 [−2.77, −1.42]** | |

- w3 + KD vs fp16: indistinguishable in 7 of 8 cases. w3 + KD vs w4: **matches or exceeds** (significant in 2 of 8).
- The German recovered fractions are imprecise because the w3 gap there is small.
- Rerun the script once the w2 XCOMET scores exist to add their CIs.

---

## 7. Log: runs, incidents, corrections

| time (2026-09-30) | event |
|---|---|
| ~12:40 | Code written (`lora.py`, `distill_lora.py`, `generate_lora.py`, jobs). A CPU test with a tiny model was skipped, so the smoke job was the first execution; only syntax was checked. |
| 12:48 | Smoke test 27400056 passed (§4.1). |
| 12:57–13:40 | Full w3 run 27400299 (43 min). |
| 13:41 | eval 27400300 starts: NLL (4.5 min), then plain w3-as4 generation. |
| 14:10 | `gen_lora` 27403647 started in parallel for the adapter generations. **Bug:** it read `RUN`, which `env.sh` presets to `ours`, so its output went to `outputs/ours/wmt22` (nothing overwritten, folder was new). Fixed by using `KD_RUN`; is-en/en-is copied to `outputs/gptq-w3g128-as4-kd/wmt22`. `outputs/ours/` holds stray duplicates. |
| 15:19–15:22 | Duplicate work: eval 27400300 reached its adapter stage before `gen_lora` finished de-en, so both generated de-en. 27403647 was cancelled (~115 SBU wasted). Results unaffected. |
| 14:45 | w2 r64 distill 27405387 + gated eval 27405389 submitted. |
| 16:05 | H100 queue full: 27405389 cancelled while pending and resubmitted on an A100 as 27409151 with `SKIP_XCOMET=1`. Verified first: the ExllamaV2 `.so` contains sm_80 code. |
| 16:15 | Continuation 27409780 submitted. |
| 17:10 | w2 gated eval done (is-en, de-en). |
| 17:12 | XCOMET job 27412752 submitted for w2 is-en,de-en. |
| 17:32 | Full w2 evaluation submitted (27414048/49/50 + scoring 27414051). |
| 17:45 | **`--export` comma bug:** `sbatch --export=ALL,PAIRS=a,b,c` splits the list, so only `a` reached the jobs. 27414048/49/50 therefore generate only zh-en / de-en / cs-en (the NLL in 27414048 is fine), and 27412752 scored only is-en. Fix: three more A100 jobs with `PAIRS` passed through the environment (27414630, 27414640, 27414642). Scoring 27414051 was cancelled while pending (dependency update not permitted) and resubmitted as 27414648. XCOMET de-en resubmitted as 27414643. Both job files now also accept `a:b:c`. |
| 2026-10-02 (evening) | "Is the w2 recovery real?" audit (§5.8), CPU only on the login node: `audit_recovery.py` + `score_failures.py`. Downloaded fastText NLLB LID (lid218e, 1.1 GB) to `/scratch-shared/$USER/models/lid/`. First run of `score_failures.py` used the LID for Chinese too and gave a 25 % off-target rate on the *references*; switched Chinese to a Han-script rule before recording numbers. Found that FLORES dev/devtest is in ALMA's train data. |

**Corrections to earlier wording** (the notebook keeps the original text):
- "w3 + KD is above w4 on every direction and both metrics" → **matches or exceeds w4**: significant in 2 of 8 (§6).
- "w2 + KD above plain w3 on de-en too" → **tie on de-en**, significantly above on is-en (§6).
- "w3 + KD at or below w4 in every direction on translation NLL" → 9 of 10 (ru-en +0.003 vs −0.001, §4.2).

---

## 8. Files

| file | role |
|---|---|
| `analysis/lora.py` | LoRA wrapper for nn.Linear and GPTQ QuantLinears; inject / save / load |
| `analysis/distill_lora.py` | training (data, KD loss, held-out KL, adapter output, `--init-adapter`) |
| `analysis/generate_lora.py` | copy of `scripts/generate_quantized.py`'s prediction path plus `--adapter` |
| `analysis/lang_nll.py` | teacher-forced NLL, `--adapter` |
| `analysis/bootstrap_ci.py` | paired bootstrap CIs (§6) |
| `analysis/audit_recovery.py` | §5.8 audit: overlap, length, loops, numbers, XCOMET error spans, length buckets, worst-segment dumps → `results/json/audit_<run>.json`, `results/audit/` |
| `analysis/score_failures.py` | §5.8 failure taxonomy (osc/TNG, off-target, copy, empty, trunc) → `outputs/failures/<run>/`, `results/json/failures.json` |
| `analysis/plot_distill_curves.py`, `plot_w3_story.py`, `plot_w2_story.py`, `plot_heatmap_adapter.py` | figures 7, 8, 9, 10 |
| `analysis/make_results_md.py` | regenerates `../results/RESULTS.md` |
| `analysis/sbatch/distill.job`, `eval_distill.job`, `eval_distill_gated.job`, `gen_lora.job`, `score_full.job`, `score_xcomet.job`, `eval_w3as4.job` | jobs (§2) |
| `results/distill_runs/<adapter>/` | `train_log.jsonl` + `adapter_config.json` of every run (copied from scratch) |
| `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w{3,2}g128-kd-lora*` | the adapters themselves (fp32 safetensors; scratch is not backed up) |
