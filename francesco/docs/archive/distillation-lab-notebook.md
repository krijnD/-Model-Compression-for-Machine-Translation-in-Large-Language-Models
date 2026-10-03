# Recovering 3-bit quality by distillation: w3 + LoRA, teacher fp16 (2026-09-30)

**Status: smoke test passed (job 27400056); full run not yet done.** Apart from §6, no number in this file is a result yet.

## 1. What and why

At 3 bits GPTQ loses about half of Icelandic (is-en BLEU −49.6 %, en-is −45.9 %) and far less of the other
languages (`current state and wrap up.md` §3). `research-questions.md` §2 shows that the fine-tuning update
survives rounding (ρ ≈ 1). What hurts is rounding noise about 9× larger than that update, spread
isotropically over each matrix.

A low-rank adapter can't undo full-rank weight noise, but it can correct the *outputs* where they matter.
So the method:

- **Student:** `ALMA-13B-R-gptq-w3g128`, every weight frozen (packed 3-bit, TorchQuantLinear), plus LoRA
  r=16, α=32 on all 7 Linears of all 40 decoder layers (62.5 M params, about 125 MB in bf16, +2.3 % on 5.27 GiB).
- **Teacher:** fp16 ALMA-13B-R, the model the quantized one should behave like.
- **Loss:** token-level forward KL(p_teacher ‖ p_student) over the full vocabulary, on the target tokens only.
- **Data:** ALMA's human-written parallel **train** files, in ALMA's fine-tuning format (the calibration
  format). There's no overlap with the test sets. Default is 30 k examples: every Icelandic row twice
  (2 × 4,018), and the rest split evenly over the other 8 directions (~2,750 each).
  32 rows per language pair are held out of training in both directions.

**Why distil rather than fine-tune on the references.** ALMA-R was trained with CPO (a preference-optimisation
method) precisely because the references are sometimes worse than the model's own outputs, so SFT on them
can pull the model back toward the weaker SFT model. The teacher target avoids that. The same script can
run the SFT variant as a comparison (`EXTRA="--kd-weight 0 --ce-weight 1"`), and the fp16 + LoRA control
by pointing `--student` at the fp16 folder. Neither is planned unless the main run works.

This covers two of the brief's techniques on top of quantization: PEFT (LoRA) and knowledge distillation.

## 2. Kernels: train on one, generate on the other

| step | checkpoint | kernel | why |
|---|---|---|---|
| training, NLL | `models/ALMA-13B-R-gptq-w3g128` | TorchQuantLinear (bf16) | the only GPTQ kernel here with a backward pass (`SUPPORTS_TRAINING = True`; ExllamaV2 is `False`) |
| generation | `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-as-w4` | ExllamaV2QuantLinear (fp16) | same weights (lossless repack), 2.3–2.6× faster (`kernel-optimisation.md` §8) |

`analysis/lora.py` wraps any Linear-like module (`y = base(x) + α/r · x Aᵀ Bᵀ`, adapter in fp32), so one
adapter works on both kernels. peft isn't used, because it decides per kernel class whether it can wrap a layer.

**Comparison rule.** Repacked generations differ slightly from the scored w3 grid (kernel drift, §8 there).
So the adapter is judged against **w3-as4 without the adapter** (`outputs/gptq-w3g128-as4`), generated
with the same kernel and script, never against `outputs/gptq-w3g128`.

## 3. Jobs (submit from the repo root, in this order)

| # | command | GPU time | output |
|---|---|---|---|
| 1 | `sbatch --export=ALL,SMOKE=1 francesco/analysis/sbatch/distill.job` | ~20 min | `…-kd-lora-smoke/train_log.jsonl`, 8 generated lines |
| 2 | `sbatch francesco/analysis/sbatch/distill.job` | 43 min (measured, §6) | `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-kd-lora/` |
| 3 | `sbatch francesco/analysis/sbatch/eval_distill.job` | ~3.5 h | NLL, generations, XCOMET-XXL + BLEU/chrF++ on is-en, en-is, de-en, en-de |
| opt. | `sbatch francesco/analysis/sbatch/eval_w3as4.job` | ~7 h | full 10-direction, 8-metric grid of w3-as4, i.e. the full-grid speed-up and drift |

Run `lang_nll.job` (the full NLL run) before job 3, so its `--compare` shows fp16 / w3 / w3+KD side by side.

**What to check in the smoke log** (`logs/slurm-distill-<id>.out`):
- `grad_check`: 280/560 LoRA tensors get a nonzero gradient on step 1. That's the lora_B half; lora_A gets
  none on the first step because B starts at 0. **0/560 means gradients do not flow through the kernel:
  stop.**
- The `kd` loss is finite and falls over the 30 steps, `peak_mem_gib` is below ~90, and `full_epoch_eta_min`
  gives the full run's time.
- Step 0 `heldout.kl` per direction is the plain-w3 damage measured as KL. Icelandic should be the largest.
  That's also a cheap check of the premise.
- The 8 en-is lines at the end are real Icelandic sentences, which shows the adapter loads on ExllamaV2.

## 4. How to read the result

- **Main number:** XCOMET-XXL and BLEU of `gptq-w3g128-as4-kd` vs `gptq-w3g128-as4` on is-en/en-is, and
  on de-en/en-de as the language that was barely damaged. The fp16 `ours-beam` is the ceiling.
- **Recovered fraction** = (KD − w3) / (fp16 − w3), per direction.
- The NLL table (`lang_nll_w3kd.json`) shows whether the adapter recovers `trans:` (translation) NLL,
  `mono:` (plain-language) NLL, or both. That ties it back to the headline question in `research-questions.md` §1.
- A clean negative result is also reportable, e.g. "recovers German but not Icelandic".

## 5. Files

| file | role |
|---|---|
| `analysis/lora.py` | LoRA wrapper for nn.Linear and GPTQ QuantLinears; inject / save / load |
| `analysis/distill_lora.py` | training (data, KD loss, held-out KL, adapter output) |
| `analysis/generate_lora.py` | copy of `scripts/generate_quantized.py`'s prediction path plus `--adapter` |
| `analysis/lang_nll.py` | now takes `--adapter` |
| `analysis/sbatch/distill.job`, `eval_distill.job`, `eval_w3as4.job` | the jobs above |

**Not tested before submission.** A CPU test with a tiny model was skipped, so smoke job 1 is the first
execution of this code. Only syntax was checked.

## 6. Smoke test (job 27400056, 2026-09-30): passed

30 steps, 960 examples, 3 min 40 s in total. Log: `logs/slurm-distill-27400056.out`. Adapter and
`train_log.jsonl` are in `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-kd-lora-smoke/`.

- Gradients flow through TorchQuantLinear: `grad_check` 280/560, as expected.
- Speed: 2.8 s/step, ~1,200 tok/s. The **full epoch is 43 min (measured)**. Peak memory is 35.9 GiB
  (teacher + student on one H100), so there's plenty of headroom.
- The KD loss falls from 0.42 to 0.047 nats/token. Held-out KL per direction (32 rows per pair, train
  distribution, not the test set):

| pair | KL step 0 (plain w3) | KL step 30 |
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

- At step 0 the ranking matches the damage seen elsewhere: is-en worst, then zh-en, and German least.
  After 30 steps every direction is at 0.03–0.07. That's a smoke-test number on 32 rows per pair, not a
  quality result; generation on the test set (job 3) decides.
- The adapter loads on the repacked w3 (ExllamaV2) and produces fluent Icelandic for the first 8 en-is
  test lines.

## 7. Full run (job 27400299) and test-set NLL (job 27400300, stage 1), 2026-09-30

Training: 936 steps, 43 min, peak 35.9 GiB. The adapter is in
`/scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-kd-lora/`. Final held-out KL is 0.023–0.047 in every
direction (is-en 0.032, from 0.696).

Test-set NLL (WMT'22, WMT'21 for is; 500 sentences per item; `results/json/lang_nll_w3kd.json`), dNLL vs fp16:

| item | w3 | w3 + KD adapter |
|---|---|---|
| trans is-en | +0.642 | +0.006 |
| trans en-is | +0.575 | +0.025 |
| trans cs-en / en-cs | +0.372 / +0.182 | +0.005 / +0.010 |
| trans zh-en / en-zh | +0.415 / +0.270 | +0.012 / +0.018 |
| trans ru-en / en-ru | +0.256 / +0.213 | +0.003 / +0.007 |
| trans de-en / en-de | +0.074 / +0.163 | −0.003 / +0.012 |
| mono is / de / cs / zh / ru / en | +0.49 / +0.50 / +0.83 / +0.22 / +0.73 / +0.41 | +0.09 / +0.11 / +0.08 / +0.11 / +0.12 / +0.06 |

- Translation NLL recovers 96–99 % of the w3 damage in every direction, including Icelandic, on the test set.
- Plain-text NLL recovers 50–90 % without being trained on it. It keeps the largest residual.
- This is teacher-forced likelihood. Generation quality (BLEU / XCOMET-XXL, stages 2–3) is pending.

## 7. Full run (job 27400299) and NLL result (job 27400300, stage 1), 2026-09-30

- Training: 936 steps, 43 min, peak 35.9 GiB. Final held-out KL 0.02–0.05 in every direction (is-en 0.032).
  Adapter: `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-kd-lora/`.
- NLL on the **WMT test sets** (500 sentences per item, not in the training data; `results/json/lang_nll_w3kd.json`).
  dNLL vs fp16, nats/token. The w4 column is from `lang_nll_w4.json`, for scale:

| item | w3 | **w3 + KD** | w4 |
|---|---|---|---|
| trans:is-en | +0.642 | **+0.006** | +0.027 |
| trans:en-is | +0.575 | **+0.025** | +0.028 |
| trans:cs-en | +0.372 | +0.005 | +0.013 |
| trans:zh-en | +0.415 | +0.012 | +0.032 |
| trans:de-en | +0.074 | −0.003 | −0.001 |
| trans:en-de | +0.163 | +0.012 | +0.027 |
| mono:is | +0.486 | +0.086 | +0.069 |
| mono:cs | +0.833 | +0.084 | +0.097 |
| mono:ru | +0.732 | +0.117 | +0.133 |

(All 16 items are in the job log.)

- On translation NLL, w3 + adapter (5.27 + 0.12 GiB) recovers 96–99 % of the w3 gap for Icelandic, and is at or
  below **w4** (6.76 GiB) in every direction.
- Plain-text (`mono`) NLL also recovers 50–90 %, although only translation targets were trained on.
- This is teacher-forced NLL, not generation quality. Stages 2–3 (generation, XCOMET-XXL, BLEU) decide.

## 8. w2 + KD, rank 64 (submitted 2026-09-30 14:45, distill 27405387, then eval-gated 27405389)

**Question:** does the same recipe rescue 2 bits, where the model is broken everywhere (plain-text dNLL
+2.9 to +5.0, translation +3.5 to +8.5, BLEU ~0) rather than only on the translation mapping as at w3?

- **Changes vs w3:** student `ALMA-13B-R-gptq-w2g128`, **r = 64, α = 128** (same α/r = 2; 250 M params, ~0.5 GB),
  because w2's rounding noise is about 2× w3's and a rank-16 adapter may hit a capacity ceiling. Everything
  else is identical (data, steps, lr, loss).
- **Confound:** bits and rank both change vs the w3 run. If w2 works, a w2 r=16 run isolates the rank.
- **Gated evaluation** (`analysis/sbatch/eval_distill_gated.job`): test-set NLL first. is-en and de-en are generated
  only if their translation dNLL with the adapter is below 1.0 nats/token. Generation uses the repacked w2
  (`…-w2g128-as-w4`, ExllamaV2). The baseline is the scored grid `gptq-w2g128` (BLEU ~0, so kernel drift
  can't matter there).
- **Adapter:** `/scratch-shared/$USER/models/ALMA-13B-R-gptq-w2g128-kd-lora-r64/`. Results:
  `results/json/lang_nll_w2kd-r64.json`, `results/json/distill_w2kd-r64.json`, `outputs/gptq-w2g128-as4-kd-r64/`.

`distill.job` now takes `BITS` (default 3, so the w3 commands above are unchanged).

**Update 16:05.** The w2 distillation finished (job 27405387, 40 min; final held-out KL 0.17–0.32, still falling at
the end). The gated evaluation was moved from a full H100 queue to an **A100** (job **27409151**; H100 job 27405389
cancelled while pending) with `SKIP_XCOMET=1`, since XCOMET-XXL needs ~43 GB. Its generations therefore ran on
an A100, not an H100 (small arithmetic drift, irrelevant against plain w2 at BLEU ~0).

**TODO: XCOMET-XXL for w2 + KD is still owed.** Once `outputs/gptq-w2g128-as4-kd-r64/wmt22/test-{is-en,de-en}`
exist, run on an H100 (~15 min):
`comet-score -s <src> -t outputs/gptq-w2g128-as4-kd-r64/wmt22/test-<pair> --model Unbabel/XCOMET-XXL --batch_size 8 --gpus 1 --to_json outputs/xcomet-xxl/gptq-w2g128-as4-kd-r64/<pair>.json > .../<pair>.txt`

**w2 test-set NLL (job 27409151, A100): w2 + r64 adapter is in the plain-w3 zone** (`results/json/lang_nll_w2kd-r64.json`,
`results/figures/fig9_w2_story.png`). Translation dNLL vs fp16 went from +3.5 to +8.5 (plain w2) to **+0.04 to +0.32**,
better than plain w3 in 8 of 10 directions (is-en +0.18 vs +0.64; en-de +0.19 vs +0.16 is the one clearly worse).
Plain-text dNLL went from +2.9 to +4.8 to +0.85 to +1.2, which is *worse* than plain w3 (+0.2 to +0.8): the adapter restored translation more
than general language knowledge (it was trained on translation only). Both is-en and de-en passed the gate, and
generation is running (A100).

**Continuation run (job 27409780, submitted 16:15, H100): more training for w2 r64.** The held-out KL was still falling when
the cosine schedule reached lr = 0. `distill_lora.py --init-adapter` now starts from a finished adapter with a
fresh schedule. This run: from `…-w2g128-kd-lora-r64` (936 steps), 60k examples (~1,870 steps, ~80 min), same seed
(same held-out rows), output `…-w2g128-kd-lora-r64-cont`. Evaluate it with `eval_distill_gated.job`
(`TAG=w2kd-r64-cont`, `KD_RUN=gptq-w2g128-as4-kd-r64-cont`).

### 8.1 w2 + KD r64: training and test-set NLL (distill 27405387, eval-gated 27409151 on A100)

Training: 936 steps, 40 min. Held-out KL fell from 4.2–9.2 to 0.17–0.32 (is-en 5.06 → 0.24).
The original eval job 27405389 was cancelled and resubmitted on an A100 as 27409151.

Test-set dNLL vs fp16 (500 sentences per item; `results/json/lang_nll_w2kd-r64.json`):

| item | w2 | w2 + KD r64 | (plain w3, for scale) |
|---|---|---|---|
| trans is-en / en-is | +4.654 / +6.777 | **+0.179 / +0.320** | +0.642 / +0.575 |
| trans de-en / en-de | +3.468 / +6.974 | **+0.038 / +0.190** | +0.074 / +0.163 |
| trans cs, zh, ru (xx-en / en-xx) | +4.2 to +8.5 | +0.12 to +0.22 | +0.18 to +0.42 |
| mono (all languages) | +2.9 to +4.8 | +0.71 to +1.20 | +0.22 to +0.83 |

- Translation NLL recovers 95–98 % of the w2 damage and ends **below plain w3** on the Icelandic directions.
- Plain-text NLL recovers only ~75 %, and stays worse than plain w3. The adapter restores the task much
  more than the languages themselves.
- Gate (< 1.0): is-en and de-en both pass, so generation runs.
- A continuation run (60 k examples, 1873 steps, `…-w2g128-kd-lora-r64-cont`, job 27409780) was started
  separately. It's not part of this evaluation.

## 9. w3 + KD: final generation results (job 27400300, done 16:39), BLEU and XCOMET-XXL

Plain w3 is the repacked run on the same ExllamaV2 kernel (`gptq-w3g128-as4`); fp16 = `ours-beam`; w4 = scored grid.
Recovered = (w3+KD − w3) / (fp16 − w3).

| pair | BLEU fp16 / w4 / w3 / **w3+KD** | recovered | XCOMET-XXL fp16 / w4 / w3 / **w3+KD** | recovered |
|---|---|---|---|---|
| is-en | 40.21 / 39.09 / 21.62 / **39.75** | 98 % | 81.15 / 80.70 / 46.65 / **81.23** | 100 % |
| en-is | 22.26 / 21.82 / 11.66 / **22.07** | 98 % | 92.82 / 91.73 / 66.71 / **92.83** | 100 % |
| de-en | 31.59 / 31.18 / 28.91 / **31.41** | 93 % | 94.66 / 94.24 / 89.09 / **94.49** | 97 % |
| en-de | 28.05 / 27.03 / 25.20 / **28.01** | 98 % | 97.71 / 97.53 / 95.16 / **97.55** | 94 % |

- **w3 + KD (5.4 GiB) matches fp16 on XCOMET-XXL for Icelandic** (81.23 vs 81.15, 92.83 vs 92.82), and is above w4
  (6.8 GiB) on every direction and both metrics.
- Over-long outputs on en-is drop from 10.6 % to 0.2 % (fp16 0.2 %).
- Scores: `outputs/xcomet-xxl/gptq-w3g128-as4{,-kd}/`, `results/json/distill_lexical.json`. Figure: `results/figures/fig8_w3_story.png`.

## 10. w2 + KD (r64): first generation result (job 27409151, A100)

| is-en | size | BLEU | chrF++ | ≥2× length |
|---|---|---|---|---|
| fp16 | 24.2 GiB | 40.21 | 61.72 | 0.0 % |
| plain w3 | 5.3 GiB | 21.62 | 44.49 | 0.6 % |
| plain w2 | 3.8 GiB | 0.00 | 4.37 | 42.7 % |
| **w2 + KD r64** | **4.25 GiB** | **35.40** | **56.83** | **0.0 %** |

That's 88 % of the fp16 BLEU recovered from zero, well above plain w3 at 1 GiB less.

| de-en | BLEU | chrF++ | ≥2× length |
|---|---|---|---|
| fp16 | 31.59 | 55.27 | 0.05 % |
| plain w3 (repacked) | 28.91 | 52.80 | |
| plain w2 | 0.10 | 4.71 | 62.05 % |
| **w2 + KD r64** | **29.63** | **53.02** | **0.25 %** |

94 % recovered; above plain w3 here too. Job 27409151 completed at 17:10 (`results/json/distill_w2kd-r64.json`).
XCOMET-XXL is still owed (run on an H100). Figure: `results/figures/fig9_w2_story.png`.

### 7.1 w3 + KD: generation results (job 27400300, done 16:39; is-en/en-is from gen-lora 27403647)

Full test sets, beam 5, repacked w3 (ExllamaV2) with and without the adapter. Recovered = (KD − w3) / (fp16 − w3).

| pair | metric | fp16 | w4 (grid) | w3-as4 | **w3-as4 + KD** | recovered |
|---|---|---|---|---|---|---|
| is-en | BLEU | 40.21 | 39.09 | 21.62 | **39.75** | 98 % |
| is-en | XCOMET-XXL | 81.15 | | 46.65 | **81.23** | 100 % |
| en-is | BLEU | 22.26 | 21.82 | 11.66 | **22.07** | 98 % |
| en-is | XCOMET-XXL | 92.82 | | 66.71 | **92.83** | 100 % |
| de-en | BLEU | 31.59 | 31.18 | 28.91 | **31.41** | 93 % |
| de-en | XCOMET-XXL | 94.66 | | 89.09 | **94.49** | 97 % |
| en-de | BLEU | 28.05 | 27.03 | 25.20 | **28.01** | 98 % |
| en-de | XCOMET-XXL | 97.71 | | 95.16 | **97.55** | 94 % |

- chrF++ recovery is 91–97 %. The hallucination rate on en-is falls from 10.6 % to 0.2 % (fp16 0.2 %).
- w3 + KD is above the w4 grid on every direction and metric measured, at 5.27 + 0.12 GiB vs 6.76 GiB.
- XCOMET-XXL is scored without references (as in the paper). Differences of ±0.1 from fp16 are within noise.
- Only 12–13 % of de-en/en-de lines are identical with vs without the adapter, so the adapter changes
  most outputs, not only the broken ones.
- Caveats: 4 of 10 directions, one seed, one training run. The outputs `gptq-w3g128-as4-kd` for is-en and en-is
  come from `gen_lora.job` (written to `outputs/ours/wmt22` by a since-fixed bug and copied; identical settings).

### 8.2 w2 + KD r64: generation results (eval-gated 27409151, A100, done 17:10)

Full test sets, beam 5, repacked w2 (ExllamaV2) + adapter. Baseline is the scored grid `gptq-w2g128`.
Results are in `results/json/distill_w2kd-r64.json`. XCOMET-XXL was skipped (`SKIP_XCOMET=1`, it doesn't fit a 40 GB A100).

| pair | metric | fp16 | w3 + KD (§7.1) | w2 (grid) | **w2 + KD r64** | recovered |
|---|---|---|---|---|---|---|
| is-en | BLEU | 40.21 | 39.75 | 0.00 | **35.40** | 88 % |
| is-en | chrF++ | 61.72 | 61.12 | 4.37 | **56.83** | 91 % |
| de-en | BLEU | 31.59 | 31.41 | 0.10 | **29.63** | 94 % |
| de-en | chrF++ | 55.27 | 55.10 | 4.71 | **53.02** | 96 % |
| halluc. (is-en / de-en) | | 0.0 / 0.05 % | 0.0 / 0.0 % | 42.7 / 62.1 % | **0.0 / 0.25 %** | |

- From total collapse (BLEU ~0, XCOMET-XXL 12.6 / 14.7) to 88–94 % of fp16 BLEU, with a ~0.5 GB adapter on a 3.78 GiB
  checkpoint (≈4.3 GiB, ~5.6× smaller than fp16).
- Clearly below w3 + KD (−4.4 BLEU on is-en, −1.8 on de-en). The recipe recovers most of 2 bits but not all.
- Caveats: 2 directions (xx→en only), BLEU/chrF++ only so far, rank 64 vs w3's 16 (confound, §8), one run.

## 11. Full 10-direction, 8-metric evaluation of w2 + KD (continuation adapter), submitted 17:32

Decision (2026-09-30): **w2 gets the full treatment** because it is the more surprising result. w3 + KD keeps
10-direction NLL plus 4-direction BLEU/chrF++/XCOMET. The adapter is `…-w2g128-kd-lora-r64-cont` (r64, 936 + 1,873 steps;
held-out KL below the first run from step ~1,000 on). The output run is `gptq-w2g128-as4-kd-r64-cont`.

| job | what | where | after |
|---|---|---|---|
| 27414048 | NLL (`lang_nll_w2kd-r64-cont.json`) + generate zh-en (512 / batch 1), en-is, is-en | A100 | continuation 27409780 OK |
| 27414049 | generate de-en, en-de, en-ru | A100 | 27409780 OK |
| 27414050 | generate cs-en, en-cs, en-zh, ru-en | A100 | 27409780 OK |
| 27414051 | all 8 metrics + `summarize.py --vs ours-beam` → `outputs/baseline/gptq-w2g128-as4-kd-r64-cont.tsv` | H100 | all three OK |

The three generation jobs have disjoint directions (no duplicate work). `gen_lora.job` is now generic (`BITS`, `NLL=1`,
zh-en at source 512 / batch 1 as in `generate_quantized.job`). The new `score_full.job` = steps 2–5 of `eval_quantized.job`.
Generations run on A100s (small arithmetic drift vs H100; the baseline, plain w2, is at BLEU ~0).

## 12. Paired bootstrap 95 % confidence intervals (`analysis/bootstrap_ci.py`, B = 1,000, seed 12345)

Resampling test sentences with the same indices for every system. BLEU is recomputed from summed n-gram statistics;
XCOMET is the resampled mean of segment scores. Full output: `results/json/bootstrap_ci.json`.

| | is-en | en-is | de-en | en-de |
|---|---|---|---|---|
| w3+KD recovered, BLEU | 98 % [95, 100] | 98 % [93, 104] | 93 % [80, 108] | 98 % [83, 118] |
| w3+KD recovered, XCOMET | 100 % [98, 102] | 100 % [98, 103] | 97 % [92, 101] | 94 % [89, 99] |
| w3+KD − fp16, BLEU | −0.46 [−1.01, +0.07] | −0.19 [−0.79, +0.38] | −0.18 [−0.56, +0.17] | −0.04 [−0.51, +0.42] |
| w3+KD − fp16, XCOMET | +0.08 [−0.59, +0.72] | +0.02 [−0.61, +0.68] | −0.17 [−0.43, +0.08] | **−0.16 [−0.30, −0.03]** |
| w3+KD − w4, BLEU | +0.66 [−0.02, +1.33] | +0.25 [−0.42, +0.89] | +0.23 [−0.17, +0.68] | **+0.97 [+0.45, +1.50]** |
| w3+KD − w4, XCOMET | +0.53 [−0.28, +1.34] | **+1.11 [+0.42, +1.77]** | +0.25 [−0.03, +0.55] | +0.02 [−0.15, +0.19] |
| w2+KD recovered, BLEU | 88 % [86, 90] | | 94 % [91, 95] | |
| w2+KD − plain w3, BLEU | **+13.78 [+12.66, +14.87]** | | +0.71 [−0.15, +1.45] | |

Bold = CI excludes 0. Corrections to earlier wording: w3+KD **matches or exceeds** w4 (significant in 2 of 8, not "every
direction"). w2+KD is **on par with** plain w3 on de-en (not above). The German recovered fractions are imprecise because the
w3 gap there is small, so report them with the CI or as the absolute difference to fp16. w3+KD vs fp16 is indistinguishable
in 7 of 8 cases. w2 XCOMET CIs follow when job 27412752 is done (rerun the script).

**Correction 17:45: `--export` comma bug.** `sbatch --export=ALL,PAIRS=a,b,c` splits the list, so only `a` reaches the job.
Jobs 27414048/49/50 therefore generate only zh-en / de-en / cs-en (the NLL in 27414048 is fine). This was also why XCOMET job
27412752 scored only is-en. Fix: the 7 missing directions went to three more A100 jobs with `PAIRS` passed through the environment:
27414630 (en-is, is-en, en-de), 27414640 (en-ru, ru-en), 27414642 (en-cs, en-zh). Scoring 27414051 was cancelled while pending
(`scontrol update` of dependencies is not permitted here) and resubmitted as **27414648**, after all six. XCOMET de-en for the
first r64 adapter: **27414643**. Both job files now also accept `a:b:c` and document the pitfall. No duplicated work.
