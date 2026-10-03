# Research questions after the grid: does 3-bit GPTQ erase the fine-tuning, the language, or neither?

> *Moved to `docs/` on 2026-09-30, content unchanged except updated doc names. Paths in this file are relative to `francesco/` (`../` = the repo root), as when it was written. Start at `../README.md`.*

Companion to `../research.md` (RQ1–RQ6). That document is not modified here. Every number below
was measured on the Snellius login node, CPU only, on 2026-09-29. The provenance of each number is
given inline. Anything marked [INFERENCE] was not measured.

---

## 0. Summary

- **Hypothesis tested: "3-bit quantization undoes fine-tuning."** It fails for the translation
  stage, and the failure is clean. ALMA-13B-R minus ALMA-13B-Pretrain is exactly a rank-16 update
  on `mlp.down_proj` and is zero on every other Linear. Across 3 layers and 4 bit widths, GPTQ keeps
  that update almost exactly (ρ = 1.00–1.02). The rounding noise is ~9× larger than the update in
  Frobenius norm at w3, but only 0.19 % of it falls inside the update's own rank-16 subspace. That is
  what isotropic noise would give, and there is no sign of targeted erasure. The stage-2/CPO
  "translation skill" survives quantization at the weight level, for every language alike. (§2)
- **The Icelandic gap is not a sentence-difficulty artefact.** Match `is` sentences to the other 8
  directions by how confident fp16 was in its own output, and the w3 XCOMET-XXL gap stays the same:
  −0.227 raw, −0.229 matched. `is` is worst in every difficulty quintile. At w4 the gap is 0.000. (§3)
- **What remains is stage 1**, the full-weight OSCAR fine-tune in which Icelandic got 8 % of 12 B
  tokens. That delta is full-rank, and measuring it needs LLaMA-2-13B, which our HF token cannot
  download (HTTP 403, gated). The new headline question (§1) tests it behaviourally instead. It needs
  no LLaMA-2 and no generation, only forward passes.

---

## 1. Headline question: does 3-bit GPTQ erase the *language* or the *translation skill*?

**Question.** When `is` collapses at w3 (−49.6 % BLEU is-en, −45.9 % en-is; `../research.md` §1), what
did the model lose? It could have lost Icelandic itself: the stage-1 language model, learned from
≈1 B OSCAR tokens and never reinforced. Or it could have lost the mapping between Icelandic and
English: the stage-2/CPO skill.

**Why this is the right next question.**
- §2 shows that the stage-2/CPO weights survive GPTQ intact and identically for all languages. The
  translation adapter is therefore not where a language-specific collapse can come from.
- `is` is the only language whose *both* directions collapse by similar amounts (is-en −49.6 %,
  en-is −45.9 %). Reading and writing Icelandic are both broken, which is what a loss of the language
  model would look like [INFERENCE].
- ALMA's own premise is that stage 1 teaches the language and stage 2 teaches the task. This is the
  question that tests the premise under compression.

**Measurement** (`analysis/lang_nll.py`, `analysis/sbatch/lang_nll.job`). This is teacher-forced
NLL, with no decoding. Each checkpoint (fp16, w8, w4, w3, w2) produces two numbers per language:
- `mono:L` is the per-token NLL of 500 WMT'22 test sentences in language L on their own. This
  measures the stage-1 skill. WMT'22 is clean: unlike FLORES, it is not in the fine-tuning data
  (`../research.md` §4).
- `trans:src-tgt` is the NLL of the reference translation given ALMA's prompt, with the loss on the
  target tokens only. This measures the stage-2 skill.

The quantity reported is dNLL = NLL(w_b) − NLL(fp16), per language.

**Predictions and how the result would be read.**

| outcome at w3 | reading |
|---|---|
| `mono:is` dNLL ≫ `mono:de`, and `trans:*-is`/`is-*` follow it | the model lost Icelandic. Fragility is a property of stage-1 exposure, and §5.4 of `research.md` becomes a dose–response law over *monolingual* tokens. |
| `mono` flat across languages, and only `trans:is-en`/`en-is` diverge | the language survived and the mapping broke. This conflicts with §2, so look at interaction effects (errors in attention and MLP that are each small but combine). RQ3 in `research.md` then leads. |
| both flat while BLEU still collapses | the loss is in decoding, not in the model's likelihoods. Runner-up C (§4) leads. |

w8 is the null run: it should give dNLL ≈ 0 everywhere. w2 should give large dNLL everywhere, which
is `research.md` §1.1's "global break".

**Cost.** One H100, 5 checkpoints × 25 items × ≤ 500 sentences, forward only. Expected under ~1 h
[INFERENCE, from RQ5's 1 m 55 s for 1 024 sequences on fp16].

```bash
sbatch francesco/analysis/sbatch/lang_nll.job                       # all five checkpoints
sbatch --export=ALL,LIMIT=32 francesco/analysis/sbatch/lang_nll.job  # smoke test first
```

**Follow-ups, cheapest first.**
1. Get LLaMA-2-13B access for the HF token (accept Meta's licence on the model page). Then
   `measure_delta.py` can be run on the stage-1 delta Δ1 = W_pretrain − W_llama2 in the same way.
   That tests whether a *full-rank* delta survives GPTQ, which §2 cannot answer.
2. Add the base model as a third reference point in `lang_nll.py`. If `mono:is` NLL at w3 moves
   toward LLaMA-2's `mono:is` NLL, the phrase "quantization reverts stage 1" becomes a measured claim.
3. If `mono:is` does diverge, the practical fix is *stage-1-aware* calibration or mixed precision
   decided on monolingual Icelandic text, not on the translation prompts GPTQ was calibrated on.

**Result (job 27399375, 2026-09-30, 500 sentences per item; `results/json/lang_nll_*.json`, smoke run in
`results/json/smoke32/`).** dNLL vs fp16, nats/token, at w3:

| language | mono (the language itself) | trans xx→en | trans en→xx |
|---|---|---|---|
| is | +0.486 | **+0.642** | **+0.575** |
| de | +0.501 | +0.074 | +0.163 |
| cs | +0.833 | +0.372 | +0.182 |
| zh | +0.218 | +0.415 | +0.270 |
| ru | +0.732 | +0.256 | +0.213 |
| en | +0.414 | | |

- **The language survives, the mapping breaks.** Icelandic's `mono` dNLL is mid-table (below cs, ru and de),
  but both of its translation directions have the largest `trans` dNLL. This is the second row of the table
  above.
- It conflicts with §2 (the rank-16 translation update survives in the weights), so the damage probably comes from
  interaction effects [INFERENCE]: small errors in many modules that combine on the is↔en mapping. RQ3 in `../research.md`
  (per-module attribution) and runner-up A (per-layer map) are the follow-ups.
- The null holds: w8 is ≤0.006 everywhere. w2 breaks everything (+2.9 to +8.5).
- Caveats: single run, no error bars. Plain-text NLL isn't comparable across languages in absolute terms,
  only as a change from fp16 (the fp16 column is in the log). cs-en BLEU drops more than zh-en although
  zh-en's `trans` dNLL is higher, so NLL doesn't rank the BLEU damage exactly.

---

## 2. Measured: the translation fine-tune survives GPTQ (the negative result)

**Setup** (`analysis/measure_delta.py`, CPU only; results in `results/json/delta_erasure.json`).
- ALMA-13B-R is ALMA-13B-Pretrain plus a merged LoRA: stage-2 SFT, then CPO continued on the same
  adapter (`third_party/ALMA/runs/cpo_ft.sh`: `--peft_model_id haoranxu/ALMA-13B-Pretrain-LoRA`).
  The published adapter config gives `r=16`, `lora_alpha=32`, `target_modules=["down_proj"]`.
- Pretrain shards 1, 3 and 6 were downloaded from HF to `/scratch-shared/scur0517/alma_delta/`
  (22 GB, not in the home quota).
- Ŵ_b is dequantized from our real `gptq-w{8,4,3,2}g128` checkpoints using GPTQModel 4.2.5's own
  `TorchQuantLinear.dequantize_weight()`, after its v1→v2 zero-offset conversion.

**Structure of Δ = W_R − W_pretrain.**
- At layer 39, Δ is exactly zero on `q/k/v/o_proj` and `gate/up_proj` (`delta_frac_nonzero = 0.0`).
- On `down_proj`, 99.99 % of Δ's energy is in its top 16 singular values, at layers 0, 20 and 39.
- Its cosine with the published SFT LoRA (2·B·A) is 0.990 / 0.993 / 0.996. CPO therefore moved the
  adapter only slightly.

So the whole translation stage is 40 rank-16 matrices, and ALMA's fine-tuning splits cleanly into
"base" and "adapter".

**Does GPTQ keep it?** Values are `down_proj` at layers 0 / 20 / 39.

| | w8 | w4 | w3 | w2 |
|---|---|---|---|---|
| ‖Δ‖ / ‖Ŵ−W_R‖ (SNR) | 7.9 / 4.9 / 4.0 | 0.47 / 0.29 / 0.24 | **0.22 / 0.13 / 0.11** | 0.09 / 0.06 / 0.05 |
| median \|Δᵢⱼ\| / grid step | 1.15 / 0.96 / 0.78 | 0.067 / 0.057 / 0.046 | 0.031 / 0.026 / 0.021 | 0.013 / 0.011 / 0.009 |
| ρ = ⟨Ŵ−W_pre, Δ⟩ / ‖Δ‖² | 1.017 / 1.014 / 1.015 | 1.018 / 1.014 / 1.015 | **1.018 / 1.014 / 1.014** | 1.008 / 1.003 / 1.001 |
| noise inside Δ's rank-16 subspace / ‖Δ‖ | 0.0003 / 0.0004 / 0.0005 | 0.006 / 0.006 / 0.008 | **0.013 / 0.014 / 0.017** | 0.042 / 0.036 / 0.047 |
| share of noise inside that subspace | 0.25 / 0.19 / 0.20 % | 0.26 / 0.17 / 0.19 % | 0.28 / 0.19 / 0.19 % | 0.39 / 0.20 / 0.23 % |

**What the table shows.**
1. **The per-weight picture predicted erasure, and it is misleading.** At w3 the median delta is
   2–3 % of one grid step, and at w4 it is 5–7 %. Rounding "should" wipe it out. It does not,
   because GPTQ rounds W_R, not W_pretrain. ρ ≈ 1.0 at every bit width, including w2, so on average
   the quantized weights still carry the full delta.
2. **The noise is huge but points elsewhere.** A 16-dimensional subspace of a 5120×13824 matrix
   holds √(16·16/(5120·13824)) ≈ 0.19 % of the norm of isotropic noise. The measured 0.17–0.28 % is
   that value. Inside the adapter's subspace the noise is 1.3–1.7 % of the adapter at w3.
3. **The SNR crossing is at the wrong place.** SNR drops below 1 between w8 and w4, not between w4
   and w3. w4 is nearly lossless, so Frobenius SNR does not predict damage. The planned "phase
   transition between 4 and 3 bits" explanation is therefore **falsified**.
4. **No language signal.** The same quantities weighted by each language's per-channel activation
   energy (RQ5's `channels_fp16.npz`, diagonal approximation [INFERENCE-grade]) are equal across
   `is`/`de`/`cs`/`zh`/`ru`. At layer 20, w3: ρ is 1.001 / 1.002 / 0.999 / 1.000 / 1.002 and SNR is
   0.147 / 0.149 / 0.151 / 0.147 / 0.149. (Weighted ρ at layer 0 drops to ≈ 0.26 at w3 for *every*
   language. Its activation energy is concentrated in a few channels, so the diagonal approximation
   is noisy there. It is still language-neutral.)

**Consequence.** Keeping the adapter in bf16 over a 3-bit base, which was the planned cheap fix,
would change almost nothing [INFERENCE from ρ ≈ 1 and sub-space noise ≤ 1.7 %]. Do not spend a
GPU run on it before §1 comes back.

```bash
# reproduce (CPU, ~2 min/module, 3.5 GB peak RSS at THREADS=2)
THREADS=2 nice ../venv-quant/bin/python francesco/analysis/measure_delta.py \
  --pretrain-dir /scratch-shared/scur0517/alma_delta --layers 0 20 39 --modules mlp.down_proj \
  --out francesco/results/json/delta_erasure.json
```

Scope: 3 of 40 layers are measured. The pretrain shards cover layers 0–7, 15–22 and 38–39, so more
layers are available offline. Run them one command at a time with `nice` and `THREADS=2`. An
18-layer run in one process was killed on the login node.

---

## 3. Measured: `is` is fragile at every level of difficulty

`analysis/fragility_by_confidence.py` uses only existing per-sentence scores (`../../outputs/`), so
it needs no GPU. Damage is XCOMET-XXL(w_b) − XCOMET-XXL(fp16 beam). Difficulty is the fp16 output's
KIWI-XXL score, a different metric from the damage metric so that a shared metric bias cannot
create the result on its own. Results are in `results/json/fragility_by_confidence{,_w4}.json`.

| fp16 KIWI-XXL quintile (pooled, 10 directions) | w3 damage, `is` pairs | w3 damage, other 8 |
|---|---|---|
| 0.011–0.779 | −0.241 | −0.112 |
| 0.779–0.842 | −0.386 | −0.107 |
| 0.842–0.903 | −0.374 | −0.099 |
| 0.903–0.950 | −0.353 | −0.077 |
| 0.950–0.997 | −0.207 | −0.049 |

- The gap is −0.227 raw and **−0.229 after matching difficulty**. At w4 it is +0.000 both ways.
- Within each direction, fp16 confidence correlates slightly *positively* with the damage, Spearman
  +0.07 to +0.20 at w3. That is partly a ceiling effect: w8 gives −0.10 to +0.08, which is the
  null. So w3 does not damage the "hard" sentences more.

This rules out "Icelandic just has harder sentences". The effect belongs to the language, which is
consistent with §1's stage-1 reading [INFERENCE].

---

## 4. Runner-up questions, ranked

**A. A per-layer language-sensitivity map, and a ~3.1-bit model that recovers `is`.**
- Method: swap one decoder layer at a time from fp16 to its w3 weights. The checkpoint already
  contains them, so there is no re-quantization. Measure the per-language `mono`/`trans` dNLL from
  §1 on 100 sentences.
- That gives 40 layers × 5 languages from forward passes only, a few GPU-hours [INFERENCE].
- If the layers that matter for `is` differ from those that matter for `de`, the result is
  "language-specific depth". Keeping the top-k `is`-sensitive layers at w4 (a few % of the size
  budget) gives a mixed-precision checkpoint to score on is-en/en-is.
- Rank 1 because it is the direct actionable follow-up to §1 whatever §1 finds, and it reuses the
  same tool. It also subsumes RQ3 in `research.md` (per-module attribution) at layer resolution.

**B. Is the gap a property of GPTQ, or of 3-bit PTQ in general?**
- Run the same 3-bit budget through an incoherence method (QuIP#/QuaRot-style Hadamard rotation)
  and compare the *gap* between `is` and the other languages, not the average.
- Marchisio et al. (2024) report non-Latin scripts suffering most. Here script is not the axis: `zh`
  is mid-table and `is` (Latin script) is worst.
- If rotation closes the average loss but not the `is` gap, the gap is about exposure, not the
  method. That makes it a claim about language fairness of PTQ in general.
- Rank 2: it is publishable, but it costs a new quantizer plus a full 3-bit generation run.

**C. How much of the en-is loss is decoding pathology?**
- w3 en-is has a 9.90 % hallucination rate (length ≥ 2× the reference).
- Re-decode only en-is/is-en at w3 in three ways: greedy; beam 5 with `no_repeat_ngram_size=4`;
  beam 5 with a length cap of 1.5× the source. Score chrF/XCOMET on the same lines.
- Cost: 2 directions on one GPU, a few hours at w3's 2.4 s/line (faster with the kernel fork's
  repack, if it lands).
- Rank 3: it is cheap and separates model damage from search damage, but §1's "all flat" branch is
  the only case where it becomes the main question. Scoring it only needs the existing
  `score_lexical.py`.

---

## 5. Files and provenance

| item | where |
|---|---|
| LoRA rank/targets | HF `haoranxu/ALMA-13B-Pretrain-LoRA/adapter_config.json` (r=16, α=32, `down_proj`); `third_party/ALMA/utils/utils.py:540` (new adapters would be `all-linear`) |
| CPO continues the stage-2 adapter | `third_party/ALMA/runs/cpo_ft.sh` |
| ALMA-13B-R is released merged | `third_party/ALMA/README.md:81` |
| LLaMA-2-13B gated for our token | HTTP 403 on `meta-llama/Llama-2-13b-hf/resolve/main/config.json`, 2026-09-29 |
| §2 numbers | `analysis/measure_delta.py` → `results/json/delta_erasure.json`, and the log `/scratch-shared/scur0517/alma_delta/log_down.txt` |
| §3 numbers | `analysis/fragility_by_confidence.py` → `results/json/fragility_by_confidence{,_w4}.json` |
| §1 tool (not yet run) | `analysis/lang_nll.py`, `analysis/sbatch/lang_nll.job` (item construction checked on CPU) |
| paper numbers for LLaMA-2 / Pretrain per direction | **not in the repo**. `third_party/ALMA/outputs/` has ALMA-13B-LoRA and ALMA-13B-R outputs but no base-model outputs. §1 follow-up 2 measures the base directly instead. |
