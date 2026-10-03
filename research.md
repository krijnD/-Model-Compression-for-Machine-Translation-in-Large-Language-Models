# Quantization, low-resource languages and calibration: what we know and what to test

Working note for the ALMA-13B-R GPTQ grid in this repo. Every number below was read off this
project's own artifacts on 2026-09-25; the provenance of each is named inline. Nothing here is
projected unless it says so.

---

## 0. Direct answer to the question that started this

**"Quantization killed rare languages like Icelandic (`is`). Can calibration balancing retrieve it?"**

- **The observation is real and it is the strongest signal in the grid.** At 3 bits, `is` degrades
  more than any other direction on *every one* of the 8 metrics. `is-en` BLEU −19.96 (−49.6 %),
  `en-is` BLEU −10.21 (−45.9 %), `en-is` hallucination rate 0.20 % → 9.90 %. Nothing else is close.
- **The *diagnosis* "rare language" is not the right one, and that changes the fix.** `is` is not
  under-represented in the calibration set — it is the **best**-represented language there by tokens
  (22.2 % of calibration tokens, see §2). So "re-balance calibration to give `is` more examples" is
  aimed at a variable that is not starved.
- **`is` is the least-trained language in the *model*, at both training stages** (§2): 8 % OSCAR
  interleave share at stage 1 (lowest of the six, ≈1 B of 12 B tokens) and 2 009 fine-tuning rows at
  stage 2 (lowest, 6–7× below the others). Weight-level fragility from under-training is a better
  explanation than calibration composition, and it makes a *different* experiment the interesting one.
- **What is still worth doing, and it is a good question:** GPTQ minimises *activation-weighted*
  output error on a fixed calibration budget. Reallocating that fixed budget between languages is a
  real knob, and the honest form of your question is: *at a fixed 1 024-example / 104 925-token
  budget, is there a reallocation that recovers `is` without paying for it in `de`/`cs`/`zh`/`ru`?*
  §3 gives that experiment as an arm design with a negative control and a cost estimate.
- **That reallocation knob has now been tested and it is not the answer.** Measured (§6, RQ5,
  job 27178053): `is` activations are *not* more outlier-heavy than `de`'s — they are slightly less
  (τ₃₂ 0.080 vs 0.091), and the loud-channel *sets* are only ~40–50 % shared with the calibration
  mixture for **every** language, not especially for `is`. So there is no Icelandic-specific channel
  set for calibration to protect, and the expected gain from a rebalance is small and
  language-unspecific: RQ1 is demoted in §7. The cause moves to the weights — `is` has 2 009
  fine-tuning rows against `de`'s 14 211 and 8 % of the OSCAR stage-1 share — and the next decisive
  measurement is the direct per-language output error on the real 3-bit checkpoint (§7 step 2).
- **On the "morphologically similar languages" idea: yes, and FLORES is the right corpus — but not
  for the five languages we already report.** FLORES is *fully contaminated* for de/cs/is/zh/ru
  (§4): all 1 012 devtest sentences are inside ALMA's fine-tuning data for every one of the five,
  on both sides. For the ~199 other FLORES languages it is clean, which is exactly why the
  related-language list in §5 is a genuine contribution rather than a re-run. One prerequisite
  though: **we do not know that ALMA-13B-R can translate any of them**, because it was only ever
  trained on 10 directions, and ALMA's own FAQ shrugs at zero-shot ("it may surprise us in other
  directions"). So the list comes with a cheap capability screen that has to pass first (§5.2).

---

## 1. What the existing grid already shows

Weight-only GPTQ (asymmetric, group 128, act-order, `desc_act`, damp 0.05) over all 40 decoder
layers' Linears; embeddings, `lm_head`, norms, scales/zeros stay fp16. Sizes from
`quantization_sizes.txt`, scores from `outputs/baseline/*.tsv`. Baseline = `ours-beam`
(fp16, beam 5, deterministic), same protocol as the quantized runs.

| checkpoint | bits/weight | GiB | vs fp16 | status |
|---|---|---|---|---|
| fp16 | 16 | 24.24 | 1.00× | baseline |
| `gptq-w8g128` | 8.13 | 12.72 | 1.91× | indistinguishable from fp16 |
| `gptq-w4g128` | 4.07 | 6.76 | 3.59× | small, structured loss |
| `gptq-w3g128` | 3.05 | 5.27 | 4.60× | **the interesting regime** |
| `gptq-w2g128` | 2.04 | 3.78 | 6.41× | total break, see below |

### 1.1 Only 3 bits has language structure in it

BLEU against the fp16 beam baseline, `outputs/baseline/`::

| direction | fp16 | w8 | w4 | w3 | w2 |
|---|---|---|---|---|---|
| de-en | 31.59 | 31.60 | 31.18 | 29.06 (−8.0 %) | 0.10 |
| cs-en | 44.83 | 44.86 | 43.89 | 32.13 (−28.3 %) | 0.01 |
| **is-en** | 40.21 | 40.04 | 39.09 | **20.25 (−49.6 %)** | 0.00 |
| zh-en | 22.87 | 22.99 | 22.49 | 15.19 (−33.6 %) | 0.08 |
| ru-en | 39.61 | 39.64 | 39.54 | 31.52 (−20.4 %) | 0.02 |
| en-de | 28.05 | 27.99 | 27.03 | 25.24 (−10.0 %) | 0.04 |
| en-cs | 26.25 | 26.50 | 24.76 | 21.62 (−17.6 %) | 0.06 |
| **en-is** | 22.26 | 22.30 | 21.82 | **12.05 (−45.9 %)** | 0.04 |
| en-zh | 33.92 | 33.81 | 31.60 | 25.27 (−25.5 %) | 0.01 |
| en-ru | 24.41 | 24.24 | 23.35 | 20.39 (−16.5 %) | 0.02 |

Two things to take from this table:

1. **w8 is a null result and w4 is a mild one.** At 8 bits nothing moves. At 4 bits the largest BLEU
   loss is 6.8 % (`en-zh`) and `is` is not special (2.0–2.8 %). If the question is "does compression
   hurt low-resource languages", 4 bits says "not much", 8 bits says "no". **3 bits is where the
   phenomenon lives**, and that is a defensible thing to build the study on.
2. **w2 is not a language story.** Every one of the 10 directions collapses to BLEU 0.00–0.10 and
   hallucination rate 31.8–93.1 %. The 2-bit model is broken *globally*; "rare languages died at
   2 bits" would be a misreading. Do not build the low-resource narrative on w2.

### 1.2 `is` is worst on all 8 metrics, not just BLEU

3-bit, absolute score with the delta against fp16 in brackets, `outputs/baseline/gptq-w3g128.tsv`:

| direction | BLEU | chrF++ | COMET-22 | KIWI-22 | KIWI-XXL | XCOMET-XXL | MetricX-24 ↓ | halluc % ↓ |
|---|---|---|---|---|---|---|---|---|
| de-en | 29.06 (−2.53) | 52.80 (−2.47) | 82.32 (−2.70) | 79.01 (−2.68) | 77.68 (−6.77) | 89.25 (−5.41) | 4.06 (+1.11) | 0.10 |
| cs-en | 32.13 (−12.70) | 56.68 (−10.65) | 80.76 (−6.24) | 76.66 (−6.12) | 68.33 (−15.89) | 70.26 (−18.47) | 5.92 (+2.22) | 0.21 |
| **is-en** | **20.25 (−19.96)** | **43.15 (−18.57)** | **73.08 (−14.18)** | **70.49 (−11.19)** | **58.13 (−27.84)** | **43.75 (−37.40)** | **8.39 (+5.03)** | **0.80** |
| zh-en | 15.19 (−7.68) | 42.78 (−9.42) | 75.17 (−6.41) | 72.75 (−6.71) | 64.31 (−13.06) | 82.18 (−9.60) | 3.97 (+2.31) | 2.29 |
| ru-en | 31.52 (−8.09) | 58.11 (−6.06) | 80.63 (−4.96) | 77.46 (−4.43) | 72.13 (−9.56) | 81.28 (−10.27) | 4.36 (+1.61) | 0.30 |
| en-de | 25.24 (−2.81) | 52.50 (−3.16) | 84.07 (−2.86) | 80.98 (−2.71) | 77.76 (−7.54) | 94.96 (−2.75) | 1.31 (+0.62) | 0.25 |
| en-cs | 21.62 (−4.63) | 47.59 (−4.66) | 86.32 (−4.28) | 80.41 (−4.96) | 75.17 (−13.57) | 84.94 (−9.47) | 4.58 (+1.85) | 2.31 |
| **en-is** | **12.05 (−10.21)** | **38.50 (−12.08)** | **74.00 (−13.05)** | **69.83 (−12.61)** | **56.31 (−30.10)** | **67.05 (−25.77)** | **9.06 (+5.80)** | **9.90** |
| en-zh | 25.27 (−8.65) | 21.98 (−6.39) | 81.24 (−5.79) | 76.35 (−6.23) | 70.56 (−14.67) | 81.63 (−10.84) | 2.68 (+1.39) | 5.06 |
| en-ru | 20.39 (−4.02) | 45.54 (−4.54) | 84.46 (−4.29) | 79.83 (−4.57) | 75.97 (−12.31) | 88.27 (−7.09) | 2.84 (+1.45) | 2.55 |

`is` holds the worst delta in 7 of 8 metrics outright (both directions for every reference-free and
reference-based metric); MetricX is worst on `en-is` and second-worst on `is-en`. The XCOMET-XXL
collapse is the headline: **81.15 → 43.75** on `is-en`. Beware one confound when comparing:
`en-is`'s 9.90 % hallucination rate means part of its BLEU loss is repetition/length blow-up rather
than purely "wrong words" — `scripts/score_lexical.py` flags candidate ≥ 2× reference length, so
this is measurable per sentence and worth splitting out (§6, RQ4).

---

## 2. Two candidates for "why", and the evidence for each

### 2.1 Calibration: `is` is *not* starved there

`scripts/alma_prompt.py::calibration_examples` draws `1024 // 10 = 102` per direction (103 for the
first four), from ALMA's human-written **train** files only, seed 42. Recomputed here with the real
ALMA tokenizer for the shipped settings:

| direction | n | avg tok/ex | tokens | token share | pool rows | % of pool drawn |
|---|---|---|---|---|---|---|
| de-en | 103 | 87.7 | 9 033 | 8.6 % | 14 211 | 0.7 % |
| cs-en | 103 | 97.8 | 10 070 | 9.6 % | 12 076 | 0.9 % |
| **is-en** | 103 | 115.2 | 11 862 | **11.3 %** | **2 009** | **5.1 %** |
| zh-en | 103 | 127.9 | 13 170 | 12.6 % | 15 406 | 0.7 % |
| ru-en | 102 | 95.9 | 9 784 | 9.3 % | 15 000 | 0.7 % |
| en-de | 102 | 85.3 | 8 696 | 8.3 % | 14 211 | 0.7 % |
| en-cs | 102 | 96.0 | 9 797 | 9.3 % | 12 076 | 0.8 % |
| **en-is** | 102 | 112.1 | 11 433 | **10.9 %** | **2 009** | **5.1 %** |
| en-zh | 102 | 115.1 | 11 738 | 11.2 % | 15 406 | 0.7 % |
| en-ru | 102 | 91.6 | 9 342 | 8.9 % | 15 000 | 0.7 % |

Total 1 024 examples / 104 925 tokens; matches `quantization_sizes.txt` exactly.

And the *content* of the Icelandic slot is FLORES: replaying the seed-42 draw, **101 of the 205
`is` calibration examples (49.3 %) are FLORES devtest sentences**, against 5.8–5.9 % for `zh`/`ru`
and 8.7–11.8 % for the others (163/1 024 = 15.9 % overall). Half of what the quantizer is shown for
Icelandic is the very corpus we would want to test it on — see §4.

**`is` already has the largest calibration share by tokens (22.2 % of 104 925) and the largest
fraction of its available corpus sampled (10.2 % of rows).** The calibration set is balanced by
construction. An example-count rebalance has almost nothing to add — which is *not* the same as
saying the recipe is optimal: the pool is only 2 009 sentences, so 205 draws means each `is`
sentence contributes ~10 % of the language's total calibration signal, and what actually varies is
*which* sentences, not how many.

### 2.2 Training data: `is` is starved at *both* stages

| language | stage-1 OSCAR interleave share (`third_party/ALMA/runs/mono_ft.sh`) | impl. tokens of 12 B | stage-2 parallel rows (train `*.json`) |
|---|---|---|---|
| is | **0.08** | ≈0.96 B | **2 009** |
| cs | 0.14 | ≈1.68 B | 12 076 |
| en | 0.17 | ≈2.04 B | — |
| zh | 0.19 | ≈2.28 B | 15 406 |
| de | 0.20 | ≈2.40 B | 14 211 |
| ru | 0.22 | ≈2.64 B | 15 000 |

Icelandic is the least-represented language at stage 1 *and* stage 2. Against the 3-bit BLEU drop:
Pearson `r(%-drop, log10 stage-2 rows) = 0.822` over the 5 languages × 2 directions (n=10).
Tokenizer fertility (tokens/word on FLORES devtest, ALMA tokenizer) gives `r = −0.209` — i.e.
**morphological richness alone does *not* order the damage**, because `zh` is the extreme fertility
outlier (13.99) but only mid-table on damage. [INFERENCE, n=5 languages, no error bars: suggestive,
not established. A proper fit needs more languages — which is what §5 buys.]

| language | FLORES code | tok/word | char/word |
|---|---|---|---|
| de (in-domain) | `deu_Latn` | 1.99 | 6.98 |
| ru (in-domain) | `rus_Cyrl` | 2.55 | 7.18 |
| cs (in-domain) | `ces_Latn` | 2.73 | 6.61 |
| **is (in-domain)** | `isl_Latn` | **2.98** | 6.30 |
| fi | `fin_Latn` | 3.65 | 8.63 |
| zh (in-domain) | `zho_Hans` | 13.99 | 9.83 |

Working hypothesis to test, not a conclusion: **quantization fragility scales with how little of the
model's weight budget was spent on the language**, and `is` is the extreme case at both stages. The
corollary that makes it falsifiable: a language that is *unrelated* to the five but *well* supported
by the base model should also survive 3 bits, and a language that is *related* to `is` but
under-trained should also collapse.

---

## 3. RQ1 — can we recover `is` by spending the *same* calibration budget differently?

The question is worth asking in this form: GPTQ minimises activation-weighted output error on the
calibration set, so the *composition* is a free parameter at fixed cost. Four arms, each a single
re-quantize at 3 bits + a full `eval_quantized.job` run:

| arm | calibration mix (1 024 examples total) | prediction | what it rules out |
|---|---|---|---|
| **A — control** | balanced, seed 42 (= shipped `gptq-w3g128`) | is-en 20.25 BLEU | already exists, no GPU cost |
| **B — is-heavy** | 50 % `is-en`/`en-is` (512), 50 % split over the other 8 | recovers `is` *iff* calibration representation is the binding constraint | — |
| **C — is-heavy, deterministic pool** | as B but the whole 2 009-row `is` pool, sentence-level, no FLORES rows | separates "more Icelandic activations" from "more FLORES activations" | the shuffle's accident |
| **D — token-equal (negative control)** | equal *tokens* per direction instead of equal examples ⇒ **cuts** is share (its examples are the longest) | *worsens* `is` if representation is causal; unchanged if not | directionality of the effect |

D is the important one: it moves the is-share the *opposite* way to B at the same budget, so a
monotone response across D→A→B is real evidence, and a flat response kills the hypothesis cleanly.

**Dose–response variant (cheap, do it first):** run the quantization only (16.5 min/arm on 1 H100,
`quantization_sizes.txt`) at 4 bit, where damage is mild, for is-share ∈ {10 %, 25 %, 50 %, 75 %},
and score a *subset* of directions. If 4-bit `en-is` BLEU moves monotonically with is-share, the
mechanism is established at a tenth of the cost and the 3-bit confirmation becomes a formality.

**Even cheaper inner loop, if `is`-share alone is the question (this is the "just make a new
calibration and try it" loop done cheaply).** Quantize the arm (16.5 min), then measure the
per-language output error of the quantized Linears on held-out per-language activations —
`‖Wx − Ŵx‖ / ‖Wx‖` accumulated at each hook, which is *exactly* the quantity GPTQ minimises — instead
of generating translations. That turns an arm from ~15 GPU-h into ~20 min, using the same hooks as
RQ5 (`francesco/analysis/measure_channels.py`). It is a proxy, not a substitute: it cannot see
hallucination or length blow-up (§6, RQ4), so use it to *rank* arms, and give the winner a real
generation run before claiming anything.

Caveats to write into any result: (i) `is` is the only direction whose pool is small enough that
"more examples" means "more of the same 2 009 sentences", so B and C are mostly *temperature*
changes on one corpus; (ii) 50.4 % of the `is` stage-2 pool is FLORES devtest (§4), so arm C is
also a contamination-removal test and its `is` score is the one to trust for a clean number;
(iii) `desc_act=True` makes GPTQ's column order calibration-dependent, so arms differ in more than
their data — keep everything else byte-identical and log `quant_meta.json` per arm.

**Secondary output of RQ1 that is arguably more publishable than the recovery itself:** the
*damage-vs-calibration-share curve* for the other four languages. If `de` (best-covered) is flat and
`is` is steep, we have quantified "calibration substitution rate" per language — a per-language
exchange rate between calibration and precision, which is a more general statement about
weight-only PTQ than "we fixed Icelandic".

---

## 4. FLORES: what it is good for here, and the contamination fact

`data/flores200_dataset/` is the full FLORES-200 (204 languages, `dev` 997 + `devtest` 1 012
sentences each). The obvious question "can we test on FLORES?" has a hard constraint:

> **All 1 012 FLORES devtest sentences are present, verbatim, in ALMA's fine-tuning data for each of
> the five in-domain languages, on both sides.** Verified by set intersection against
> `third_party/ALMA/human_written_data/{deen,csen,isen,ruen,zhen}/train.*-en.json`: 1012/1012 for
> `is`↔`en`, `de`↔`en`, `cs`↔`en`, `zh`↔`en`, `ru`↔`en`. The `dev` split is likewise fully present
> (997/997 for `is`).

Two consequences, both useful:

1. **FLORES cannot be a held-out test set for our five languages.** It is in the fine-tuning data
   *and* in the calibration sample (15.9 % of the 1 024 drawn examples; 49.3 % of the Icelandic
   ones, §2.1). Any number computed there is in-distribution. Do not mix it into the WMT'22 grid we
   already have.
2. **For every other FLORES language it is clean**, and that is the free lunch: ~199 languages,
   aligned to the same 1 012 English sentences, none of them ever seen by ALMA-13B-R. This is what
   makes the related-language list in §5 an actual experiment.

Also worth stating because it changes what the current calibration *is*: 1 012 of the 2 009 rows in
`isen/train.is-en.json` are FLORES — **50.4 % of Icelandic fine-tuning data is FLORES devtest**, and
`is` is the only one of the five where the fine-tuning corpus nearly *is* FLORES. Any "Icelandic is
fragile" claim should be re-checked against a non-FLORES Icelandic set before it is published.

---

## 5. RQ2 — related languages, unseen by the model

### 5.1 The focused 5-language cohort

FLORES-200 is 100 % contaminated for the 5 reported in-domain languages (§4), but 100 % clean
for the remaining ~199 languages. Rather than an unfocused sweep, we test the **morphological
similarity hypothesis** with a tightly budgeted cohort of **5 unseen languages** (0 training rows,
0 calibration rows). Each language isolates a specific structural variable against our reported
baseline, with Faroese serving as the mandatory Icelandic-similar anchor:

| role | language | FLORES code | family / branch | morphological typology | cases | genders | tok/word (FLORES) | char/word | reported benchmark sister |
|---|---|---|---|---|---|---|---|---|---|
| **Icelandic twin** | Faroese | `fao_Latn` | North Germanic (Insular) | fusional, rich inflection | 4 | 3 | 2.85 | 6.38 | `is` (3.01 tok/word, 2 009 rows) |
| **Czech sister** | Slovak | `slk_Latn` | West Slavic | fusional, rich inflection | 6 | 3 | 2.89 | 6.74 | `cs` (2.74 tok/word, 12 076 rows) |
| **Russian sister** | Ukrainian | `ukr_Cyrl` | East Slavic | fusional, rich inflection | 7 | 3 | 2.78 | 7.01 | `ru` (2.56 tok/word, 15 000 rows) |
| **Germanic control** | Dutch | `nld_Latn` | West Germanic | weakly inflected / analytic trend | 0 | 2 | 1.98 | 6.41 | `de` (2.02 tok/word, 14 211 rows) |
| **Agglutinative control** | Finnish | `fin_Latn` | Uralic (Finno-Ugric) | highly agglutinative | 15 | 0 | 3.71 | 8.77 | *none* (orthogonal family) |

*(Tokenizer statistics measured on `data/flores200_dataset/devtest/*.devtest` using the ALMA-13B-R LLaMA tokenizer).*

### 5.2 What this 5-language design disentangles

The 5 languages form a 2×2+1 factorial design separating the competing explanations of Icelandic's 3-bit collapse:

1. **H_morph: Is damage driven by fusional morphological complexity?**
   - Icelandic (`is`) preserves a 4-case fusional nominal system and collapsed −49.6 % BLEU at 3 bits.
   - **Faroese (`fao`)** is Icelandic's closest living relative, preserving an almost identical 4-case, 3-gender fusional grammar and identical character density (6.38 char/word, 2.85 tok/word vs 3.01 for `is`).
   - If fusional morphology causes PTQ representations to break under weight rounding, Faroese will replicate Icelandic's catastrophic collapse.
2. **H_transfer: Does weight-subspace volume protect unseen sister languages?**
   - Slovak (`slk`) and Ukrainian (`ukr`) possess fusional case systems with equal or greater complexity than Icelandic (6 and 7 cases).
   - However, Slovak has a well-trained sister in Czech (12 076 fine-tuning rows, −28.3 % BLEU), and Ukrainian has an intensely trained sister in Russian (15 000 rows, 22 % OSCAR share, −20.4 % BLEU).
   - If cross-lingual representation in the weight subspace transfers to unseen sisters, damage will rank: `fao` (sister has 2k rows) >> `slk` (sister has 12k rows) > `ukr` (sister has 15k rows). If morphology alone dictates fragility, all three fusional sisters will collapse equally.
3. **H_fertility: Is damage an artifact of subword fragmentation and case richness?**
   - **Finnish (`fin`)** has 15 grammatical cases and the highest tokenizer fertility in the set (3.71 tok/word, vs `is` 3.01 and `en` 1.42), but its morphology is *agglutinative* (concatenative affixes with regular rules) rather than *fusional* (portmanteau endings, vowel ablaut, stem suppletion).
   - If high fertility and large case inventories cause quantization failure, Finnish will fail worse than Icelandic. If fusional irregular stem/affix conflation is the failure point, Finnish will decouple from Icelandic/Faroese.
4. **H_germanic: Is German's robustness lexical transfer or morphological simplification?**
   - German was the most robust 3-bit language (−8.0 % BLEU).
   - **Dutch (`nld`)** shares West Germanic vocabulary roots with German, but has discarded nominal case inflections (0 cases, fertility 1.98).
   - Dutch isolates Germanic lexical survival in the base model from inflectional mechanics.

### 5.3 The prerequisite: a capability screen, because ALMA never trained these

ALMA-13B-R is a 10-direction model: stage 1 is OSCAR monolingual on `en,ru,cs,zh,is,de` only
(`runs/mono_ft.sh`), stage 2 is the parallel fine-tuning on those same 10 directions
(`runs/parallel_ft.sh`). The repo FAQ is explicit that other directions are untested: *"However, it
may surprise us in other directions :)"*. So a zero-shot FLORES score for Faroese is not obviously
above floor, and a quantization study that never checked would be measuring **the floor, not the
damage**.

Screen, before any quantized run. Both existing drivers are WMT-hard-wired, so this needs a small
amount of new code, not just a flag:

1. **Prompt plumbing.** `scripts/alma_prompt.py::LANG_TABLE` (and ALMA's own prompt builder) knows
   only the 6 in-domain languages and interpolates the English name into the prompt
   (`"Translate this from Faroese to English:"`). New languages need entries in both. Flag the
   residual risk: an unseen language is also prompted with a direction *phrase* the model never saw,
   so a low zero-shot score partly measures the instruction, not the capability. If the screen
   passes anyway, that is a strong result; if it fails, try the `en→xx` direction with the
   `--use_target_lang_prompt_eval` style variant before concluding the model cannot do it.
2. **Source loading.** `scripts/generate_quantized.py` reads ALMA's WMT files through
   `load_test_sources()`; ALMA's `run_llmmt.py` reads `test.{src}-{tgt}.json` from the pair folder,
   or an HF dataset via `--override_test_data_path` whose config name is `{src}-{tgt}`
   (`utils/utils.py:298`). FLORES needs either a local JSON shim per pair (fields `translation:
   {src:…, tgt:…}`) placed where that loader looks, or a `--source-dir` loader added to
   `generate_quantized.py` (the file already takes `--max-samples`, handy for a first pass).
3. **Run fp16** with the `venv` path (`transformers` 4.45.2, the same env that produced
   `ours-beam`), beam 5, seed 42, so the screen scores are directly comparable to the existing
   baseline rather than to a differently-configured run.
4. Keep a language iff fp16 chrF++ > ~20 / BLEU > ~5 in **both** directions. Below that there is no
   headroom for a quantizer to destroy, and any "collapse" is a floor artefact.
5. For the survivors, and only then, add 2-bit / 3-bit / 4-bit generations and ask whether the
   *unseen* language degrades like the *least-trained* in-domain one (`is`) or like the
   *best-trained* one (`ru`).

Cost: 1 012 sentences × 5 languages × 2 directions = 10 120 lines total (down from ~26 k lines in
the 13-language draft). At w3 generation rate (~2.39 s/line on `TorchQuantLinear`), a full 10k-line
scoring run takes ~6.7 H100 hours per checkpoint, and the fp16 capability screen takes ~3.5 hours.
Run the fp16 screen on 1 H100 first to ensure zero-shot baseline headroom before launching quantized runs.

### 5.4 The comparison the screen enables

With the screen passed, one figure answers the motivating question directly:

> x-axis: stage-2 training rows for the language (log). y-axis: 3-bit ΔBLEU vs fp16.
> Points: the 5 in-domain languages (2 009 … 15 406 rows) plus the screened unseen languages
> (0 rows).
> If the trend is monotone, "quantization kills rare languages" becomes a *dose–response law*
> instead of an anecdote, and Faroese extends the curve to its extreme.

This is the cheapest way to make the finding general, and it needs no new theory.

---

## 6. Secondary questions (cheap, and each one is a real paper-sized claim)

**RQ3 — Where in the network does `is` break?** Per-module error attribution: quantize one Linear
type at a time (`q/k/v/o_proj` vs `gate/up/down_proj`) at 3 bits and score `is-en` / `de-en` only.
If the `is` loss is concentrated in the MLP `down_proj`, the story is a key-value memory capacity
story; if it is in attention `v/o_proj`, it is a value-mixing story. `scripts/quantize_gptq.py`
already builds `QuantizeConfig` per run — this is a config change, not new code. Cost ≈ 4 arms ×
(16.5 min quantize + short-scoring-only eval on 2 directions).

**RQ4 — Is the `is` loss *words* or *length*?** `scripts/score_lexical.py` already computes
hallucination as candidate ≥ 2× reference length. For `en-is` at 3 bits that rate is 9.90 %, so an
unknown share of the BLEU drop is repetition. Correlate **per-sentence** ΔchrF with **per-sentence**
token fertility of the reference (we have both, no new generation): if high-fertility sentences lose
more, the mechanism is tokenizer-level rather than language-level, and that is directly actionable
(a vocabulary or sub-word-splitting intervention).

**RQ5 — Are the damaged channels the same channels for every language? (activation outliers)**

The mechanism, stated once. A Linear computes `y = Wx` and **only `W` is quantized**, so the output
error is `ΔW·x` — the weight error *weighted by the activation magnitude in that channel*
(`(ΔW·x)_j = Σᵢ ΔW_ji xᵢ`). GPTQ's objective is literally `min ‖Wx − Ŵx‖²` over the calibration set,
so the quantizer is nearly blind to weight error and very sensitive to *where the activation energy
sits*. LLM activations are heavy-tailed — a few channels run 10–100× the median ("outlier features",
"massive activations") — which is why `LLM.int8()` keeps those channels in fp16 and why GPTQ has
group scales and `desc_act`. So damage concentrates in a few channels, and the question is whether
they are the *same* channels for every language.

Two claims, and the second is the interesting one:

1. **Concentration.** `is` activations are more concentrated (larger share of `Σx²` in the top-k
   channels) than `de`/`ru`, because it is the least-trained language and its representations are
   less "smoothed". [INFERENCE — the measurement below does not need this causal story to be true.]
2. **Mismatch with the channel *order* `desc_act` chose.** Our checkpoints use `desc_act=True`: GPTQ
   sorts columns by decreasing Hessian diagonal — i.e. by *calibration* activation energy — and
   quantizes the largest first, so the highest-energy channels come out most accurate. That ordering
   is derived from the **10-direction mixture**. If Icelandic's high-energy channels are a *different
   subset* than the mixture's, `is` gets no protection from that ordering even though its own
   activations are concentrated. This explains point-blank why `is` degrades *despite* a balanced
   calibration set, and it predicts **exactly when RQ1's rebalancing can work**: only if the top-k
   sets genuinely differ. RQ5 is the mechanism behind RQ1, not a separate story.

Measurement — no quantization, no generation, fp16 forward only (minutes on one H100):

- register forward hooks on the quantized `Linear`s, capturing the input `x`; start with the 40
  `mlp.down_proj` (the classic massive-activation site);
- fixed prompt set per language in ALMA's prompt format (`is-en`/`en-is` vs `de-en`/`en-de` vs
  `zh-en`/`en-zh`), ~512 sentences;
- per layer, per language: concentration `τ_k` = share of `Σx²` in the top-k channels
  (k = 1, 8, 32), the max/median `|x|` ratio, and the channel *ranking*; then compare rankings
  across languages by Spearman ρ and top-k Jaccard overlap, including against the mixture ranking.
- Predicted: `τ(is) > τ(de)`, and low overlap between `top-k(is)` and `top-k(mixture)`.

**Result — measured 2026-09-25, job 27178053, 1 m 55 s of one H100.** Tool:
`francesco/analysis/measure_channels.py`; raw vectors in `francesco/results/json/channels_fp16.npz`.
40 `mlp.down_proj`, fp16 ALMA-13B-R, ALMA's fine-tuning prompt format. Internal check: the
per-language token counts are 23 295 (`is`), 17 729 (`de`), 19 867 (`cs`), 24 908 (`zh`), 19 126
(`ru`) and sum to exactly 104 925, the mixture total — so the partition is the quantizer's own.

| profile | examples | τ₁ | τ₈ | τ₃₂ | max/med | Spearman vs mix | Jaccard@32 vs mix (median / worst) |
|---|---|---|---|---|---|---|---|
| mixture | 1 024 | 0.012 | 0.037 | 0.072 | 15.4 | — | — |
| calib:is | 205 | 0.013 | 0.045 | **0.080** | **16.7** | 0.831 | 0.489 / **0.185** |
| calib:de | 205 | 0.014 | 0.048 | **0.091** | **18.4** | 0.875 | 0.471 / 0.208 |
| calib:cs | 205 | 0.012 | 0.046 | 0.089 | 17.6 | 0.892 | 0.524 / 0.231 |
| calib:zh | 205 | 0.014 | 0.045 | 0.082 | 17.1 | 0.848 | 0.455 / 0.143 |
| calib:ru | 204 | 0.015 | 0.047 | 0.089 | 18.0 | 0.893 | 0.542 / 0.280 |
| pool:is:1024 | 2 048 | 0.013 | 0.044 | 0.080 | 16.6 | 0.842 | 0.488 / 0.231 |
| pool:de:1024 | 2 048 | 0.014 | 0.046 | 0.087 | 18.1 | 0.892 | 0.488 / 0.231 |

**Verdict: the hypothesis fails, and the falsification clause fires.**

1. **Claim 1 is refuted, in the wrong direction.** `is` is not the most concentrated language — it
   is the *least*: τ₃₂ 0.080 vs `de` 0.091, max/median 16.7 vs 18.4. The heavy tail is real and
   similar everywhere (one channel in 13 824 holds 1.3 % of the energy, ~170× uniform), but Icelandic
   has no special outlier structure. The `pool:` rows (10× the examples) reproduce the `calib:` rows
   to within 0.004–0.008, so this is a stable property, not a small-sample artifact.
2. **Claim 2 is technically true but not discriminating.** The loud-channel *sets* are largely
   language-specific at the head — every language shares only ~0.46–0.54 of its top 32 with the
   mixture, and every language has some module down at 0.14–0.28. `is` is the *least* aligned on
   Spearman (0.831) and has the worst single-module Jaccard (0.185), i.e. the sign is as predicted
   but the effect is tiny (Spearman 0.831 vs `de` 0.875) next to the damage gap (−49.6 % BLEU vs
   −8.0 %). **A 5 % ranking difference cannot produce a 6× damage difference**, and if head-mismatch
   drove damage then `zh` (Jaccard 0.455) and `ru` (0.542) would not sit at opposite ends of the
   damage table. Mismatch at the head is a property of *every* language here, not of `is`.

**What this does to the earlier questions:**

- **RQ1 loses its rationale.** GPTQ's `desc_act` ordering protects a partly language-specific set for
  every language alike, so re-allocating the calibration budget changes what is protected roughly
  equally for `is` and for `de`/`ru`. The prediction is now a *small, language-unspecific* effect —
  not the targeted `is` recovery the question hoped for. RQ1 should be demoted, and if run, run with
  the cheap output-error loop (§3) rather than the full 3-bit grid.
- **The cause moves to the weights**, i.e. §2.2's under-training story: `is` has 2 009 fine-tuning
  rows against `de`'s 14 211 and 8 % of the OSCAR stage-1 interleave share, so its columns of `W` are
  less well fitted and quantization error lands on a badly-conditioned weight subspace rather than on
  unusually loud activations. The fix on that reading is data/training, not quantizer protection.
- **The decisive next measurement** is therefore not another activation statistic but the direct one:
  `‖Wx − Ŵx‖ / ‖Wx‖` per language on the real 3-bit checkpoint (`models/ALMA-13B-R-gptq-w3g128`),
  using these same hooks. It needs `venv-quant` and a quantized load — minutes, no generation. If
  `is` shows much higher *output* error than `de` under the actual quantized weights, the weight-space
  story is quantified; if the output errors are comparable while BLEU still collapses 6×, the loss is
  not layer error at all and RQ4 (length/repetition, `en-is` hallucination 9.90 %) becomes the leading
  explanation.

```bash
sbatch francesco/analysis/sbatch/measure_channels.sbatch
python francesco/analysis/measure_channels.py --compare francesco/results/json/channels_fp16.npz
```

Falsification (as pre-registered): if `is` and `de` are indistinguishable on both statistics, RQ5 is
dead and the cause is on the **weight** side — the `is` columns of `W` are simply less well fitted,
which is §2.2's under-training story — and the fix moves to training/data rather than quantizer
design. That is what happened, for claim 1 outright and for claim 2 in effect size.

Why it was the deepest of the five — and what survived. The appeal was that a *positive* answer buys a
fix needing neither re-quantizing nor re-training: keep the identified channels in fp16
(`LLM.int8()`-style mixed precision) or rescale them (`SmoothQuant`), recovering `is` for a few
percent of the size budget. The answer came back negative, so that specific fix is off the table:
there is no Icelandic-specific channel set to protect, and a mixed-precision scheme built on
`top-k(mixture)` would protect `de`/`zh`/`ru` no better or worse than `is`. What the measurement did
buy is the elimination of an entire branch — mixed-precision channel protection, per-language
calibration rebalancing, and the `desc_act`-mismatch story are now all ruled out as explanations of
the `is` collapse, for 2 minutes of one H100 rather than ~60 GPU-h of re-quantization. It also
leaves one concrete, measurable successor (§7 step 2) instead of a guess.

**Addendum — the site matters, and it reverses the aggregate (jobs 27178223, 27178479).** The block
above measures one site: the `mlp.down_proj` input. That is the *post-activation MLP hidden*, not the
residual stream, and it is among the **least** concentrated of the four distinct inputs that feed a
quantized Linear. Same tool, same profiles, other sites:

| site (what the Linear reads) | τ₃₂ (`is`) | coverage(mix)@32 | self | gain | Jaccard@32 vs mix | channels shared |
|---|---|---|---|---|---|---|
| `q/k/v_proj` — `input_layernorm(hidden)` | **0.326** | **32.6 %** | 32.6 % | **0.0** | 0.939 | **31/32** |
| `gate/up_proj` — `post_attention_layernorm(hidden)` | 0.050 | 5.0 % | 5.0 % | 0.0 | 0.882 | 30/32 |
| `o_proj` — attention output | 0.084 | 8.0 % | 8.4 % | 0.5 | 0.641 | 25/32 |
| `mlp.down_proj` — MLP activation | 0.080 | 6.9 % | 8.0 % | 1.1 | 0.489 | 21/32 |

Each row is the median over its 40 layers, measured on the representative module (`q_proj`,
`gate_proj`). `coverage` = share of `is`'s energy falling inside the **mixture's** top-32 channels —
i.e. the channels a `desc_act` ordering derived from the mixture is told to protect — and `gain` =
self-coverage minus coverage, the most a per-language *re-ordering* could buy. Raw vectors:
`channels_fp16_residual.npz` (q/gate/o), `channels_fp16_check.npz` (k/up).

Three consequences:

1. **The site that actually has outliers is `q/k/v_proj`, and there the languages are almost
   indistinguishable.** 32.1 % of the mixture's energy sits in 32 of 5 120 channels (uniform 0.62 %,
   so ≈50×), which is ≈4.5× the concentration `down_proj` shows — and `is` shares **31 of its top 32**
   channels with the mixture, for a re-ordering gain of **0.0 pp**. So the "loud-channel sets are
   largely language-specific" reading above is a property of the least-concentrated sites and it
   **reverses where the energy is**. The verdict does not change; its strongest support does, and
   this is the stronger form of the negative result.
2. **The maximum the ordering mechanism can buy is 0.0 pp at `q/k/v` and 1.1 pp (`is`) / 1.6 pp
   (`de`) at `down_proj`**, against a top-32 coverage of 5–33 % — while the damage gap it has to
   explain is −49.6 % vs −8.0 % BLEU. `is` is not the language with most to gain at any site. This
   bounds the *ordering* part of the hypothesis (it does not measure GPTQ's Hessian compensation, so
   it is not a bound on rebalancing as a whole), and the ordering part is what was proposed.
3. **Site taxonomy, verified rather than assumed.** `q/k/v_proj` share one RMSNorm output and
   `gate/up_proj` share another, so their profiles must be identical: measured rank correlation is
   **1.0000** for `q_proj` vs `k_proj` and for `gate_proj` vs `up_proj`, and 0.801 between the two
   groups — and `k_proj`/`up_proj` reproduce their partners' reported coverage and shared-channel
   counts *exactly* (0.326423 in both), so measuring one of each pair is enough. Two module types
   with a provably shared input giving identical profiles is a clean end-to-end check of the
   instrument, and it is why the four rows above are the four distinct input distributions of a
   decoder block rather than seven linears.

**Consequence for §7 step 2:** hook `q/k/v_proj` alongside `down_proj`. That site holds 32 % of the
energy in 32 channels, so if weight-space damage concentrates anywhere it is there, and a `down_proj`
only run would miss it.

**RQ6 — Does quantization damage correlate with how anisotropic the language's hidden states are?**
Same hooks as RQ5, one different global statistic (participation ratio / effective rank of the
hidden states). The RQ5 result lowers its value: an RQ6 correlation would have to beat the *negative*
RQ5 finding, i.e. explain why a global anisotropy statistic singles out `is` when the per-channel
concentration does not. Run it only if §7 step 2 comes back ambiguous.

---

## 7. What to run first, in order

1. ~~**RQ5 — the channel profiles (§6).**~~ **DONE 2026-09-25, negative** (job 27178053, 1 m 55 s of
   one H100; extended to the other three sites by the §6 addendum, jobs 27178223 / 27178479 — same
   verdict, stronger). Do not repeat it. The raw per-channel vectors stay in
   `francesco/results/json/channels_fp16*.npz`; `measure_channels.py --compare` re-prints every table
   offline.
2. **Quantized output error per language — the successor to RQ5 (§6).** Load
   `models/ALMA-13B-R-gptq-w3g128` (this one needs `venv-quant`), attach the same hooks — on
   `q/k/v_proj` **as well as** `mlp.down_proj`, since that is the site holding 32 % of the energy in
   32 of 5 120 channels (§6 addendum) and a `down_proj`-only run would miss it — and measure
   `‖Wx − Ŵx‖ / ‖Wx‖` per language on held-out per-language activations. Minutes, no generation, no
   scoring. This is the decisive test of the weight-space reading: if `is`' output error comes out
   comparable to `de`'s while `en-is` BLEU still collapses 6×, then layer error is not the mechanism
   at all and RQ4 leads instead.
3. **Capability screen (§5.3)** — 1 H100, ~3.5 h for the focused 5-language suite (10 120 lines total).
   Everything in §5 is blocked on it, and §5 is the branch RQ5 did *not* touch: it needs no activation
   instrumentation at all.
4. **RQ2 as the §5.4 dose–response figure** on the surviving screen languages.
5. **RQ3 / RQ4 as cheap follow-ups** — RQ4 (per-sentence fertility vs per-sentence loss) first if
   step 2 says the loss is not layer error; RQ3 (per-module attribution) if it says it is.
6. **RQ1 — demoted.** RQ5 removed its rationale (§6); the expected effect is now small and
   language-unspecific. Run it only if step 2 shows per-language output errors diverging strongly,
   and then through the §3 output-error loop rather than the full 3-bit grid.
7. **RQ6 only if step 2 is ambiguous.**

Operational notes carried over from `francesco/`'s probe study, since they apply to every new run:
generation dominates, 3-bit runs on `TorchQuantLinear` (no accelerated 3-bit kernel exists in any
maintained loader), `BATCH=16` / `BATCH_LONG=4` amortises the fixed per-forward dequantization
(2.31× and 3.49× per sentence respectively), and a uniform batch setting across arms matters more
than the best batch setting per arm.

---

## 8. Provenance of every fact above

| fact | source read this session |
|---|---|
| per-direction scores, all 5 runs | `outputs/baseline/{ours-beam,gptq-w8g128,gptq-w4g128,gptq-w3g128,gptq-w2g128}.tsv` |
| checkpoint sizes, effective bits, calibration recipe | `quantization_sizes.txt` |
| calibration construction, per-direction counts, prompt format | `scripts/alma_prompt.py` (recomputed with the real tokenizer) |
| GPTQ settings, what is quantized / kept fp16 | `scripts/quantize_gptq.py`, `quantization_sizes.txt` §4 |
| FLORES overlap 1012/1012, both sides, all 5 languages | set intersection, `data/flores200_dataset/devtest/*` vs `third_party/ALMA/human_written_data/*/train.*-en.json` |
| stage-1 OSCAR languages and interleave probabilities | `third_party/ALMA/runs/mono_ft.sh` |
| stage-2 directions | `third_party/ALMA/runs/parallel_ft.sh`, `evals/alma_13b_r.sh` |
| 10-direction support, zero-shot untested | `third_party/ALMA/README.md` FAQ |
| tokenizer fertility | ALMA-13B-R tokenizer, `/models/ALMA-13B-R/tokenizer.model`, over FLORES devtest |
| RQ5 concentration and channel-ranking numbers, §6 | measured on fp16 ALMA-13B-R, job 27178053; raw vectors `francesco/results/json/channels_fp16.npz`, re-printable with `measure_channels.py --compare` |
| RQ5 site table and taxonomy check, §6 addendum | jobs 27178223 (`channels_fp16_residual.npz`, q/gate/o) and 27178479 (`channels_fp16_check.npz`, k/up); taxonomy is the measured rank correlation, and follows from the decoder block's two RMSNorms |
| quantized generation rates, batch levers | `francesco/` batch-probe study (`analysis/` after the reorg) |
