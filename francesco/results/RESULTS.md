# Results — ALMA-13B-R compression (francesco/)

*Generated 2026-09-30 17:38 by `analysis/make_results_md.py` from the files on disk. Rerun it after any job finishes. – = not available yet. Story and interpretation: `../README.md`; method and job history: `docs/06-distillation.md`.*

## 1. Plain GPTQ grid (fp16, w8, w4, w3, w2): averages over the 5 directions each way

| model | GiB | dir | bleu | chrF++ | comet-22 | xcomet-xxl | kiwi-22 | kiwi-xxl | metricx-24 | halluc |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 24.24 | xx-en | 35.82 | 60.14 | 85.29 | 89.57 | 81.50 | 82.74 | 2.88 | 0.16 |
| fp16 | 24.24 | en-xx | 26.98 | 47.39 | 88.07 | 94.55 | 83.70 | 86.79 | 1.87 | 0.57 |
| w8 | 12.72 | xx-en | 35.83 | 60.16 | 85.29 | 89.59 | 81.51 | 82.78 | 2.88 | 0.16 |
| w8 | 12.72 | en-xx | 26.97 | 47.36 | 88.04 | 94.64 | 83.69 | 86.77 | 1.88 | 0.58 |
| w4 | 6.76 | xx-en | 35.24 | 59.73 | 84.93 | 88.76 | 81.24 | 82.16 | 3.02 | 0.07 |
| w4 | 6.76 | en-xx | 25.71 | 46.81 | 87.71 | 93.79 | 83.45 | 85.87 | 1.95 | 0.81 |
| w3 | 5.27 | xx-en | 25.63 | 50.70 | 78.39 | 73.34 | 75.27 | 68.12 | 5.34 | 0.74 |
| w3 | 5.27 | en-xx | 20.91 | 41.22 | 82.02 | 83.37 | 77.48 | 71.15 | 4.09 | 4.01 |
| w2 | 3.78 | xx-en | 0.04 | 3.95 | 24.35 | 13.05 | 29.16 | -0.14 | 9.42 | 49.83 |
| w2 | 3.78 | en-xx | 0.03 | 3.41 | 29.74 | 32.27 | 23.56 | -2.12 | 10.40 | 47.43 |

MetricX-24: lower is better. halluc = % of outputs at least 2× the reference length. Per-direction numbers: `outputs/baseline/<run>.tsv`.

## 2. Teacher-forced NLL increase over fp16 (nats/token, 500 test sentences per item)

| item | w8 | w4 | w3 | w2 | w3+KD r16 | w2+KD r64 | w2+KD r64 cont. | fp16 NLL |
|---|---|---|---|---|---|---|---|---|
| trans:de-en | -0.001 | -0.001 | +0.074 | +3.468 | -0.003 | +0.038 | – | 1.200 |
| trans:cs-en | +0.000 | +0.013 | +0.372 | +4.555 | +0.005 | +0.138 | – | 0.641 |
| trans:is-en | -0.000 | +0.027 | +0.642 | +4.654 | +0.006 | +0.179 | – | 0.750 |
| trans:zh-en | -0.001 | +0.032 | +0.415 | +4.229 | +0.012 | +0.208 | – | 1.270 |
| trans:ru-en | -0.000 | -0.001 | +0.256 | +4.565 | +0.003 | +0.122 | – | 0.790 |
| trans:en-de | +0.000 | +0.027 | +0.163 | +6.974 | +0.012 | +0.190 | – | 0.710 |
| trans:en-cs | +0.000 | +0.019 | +0.182 | +6.335 | +0.010 | +0.168 | – | 0.732 |
| trans:en-is | -0.000 | +0.028 | +0.575 | +6.777 | +0.025 | +0.320 | – | 0.593 |
| trans:en-zh | -0.000 | +0.038 | +0.270 | +6.495 | +0.018 | +0.217 | – | 0.674 |
| trans:en-ru | -0.000 | +0.017 | +0.213 | +8.518 | +0.007 | +0.172 | – | 0.695 |
| mono:is | -0.000 | +0.069 | +0.486 | +4.687 | +0.086 | +1.198 | – | 1.726 |
| mono:de | -0.002 | +0.125 | +0.501 | +4.476 | +0.107 | +1.017 | – | 2.654 |
| mono:cs | -0.002 | +0.097 | +0.833 | +4.771 | +0.084 | +1.066 | – | 2.195 |
| mono:zh | +0.000 | +0.068 | +0.218 | +2.907 | +0.110 | +0.852 | – | 1.750 |
| mono:ru | +0.006 | +0.133 | +0.732 | +4.849 | +0.117 | +1.085 | – | 1.997 |
| mono:en | -0.005 | +0.104 | +0.414 | +2.925 | +0.059 | +0.711 | – | 2.801 |

`trans:` = reference translation given the prompt (translation skill); `mono:` = plain text in the language (knows the language). 0 = as good as fp16.

## 3. Adapter generations: BLEU / chrF++ / XCOMET-XXL per direction

w3 and w3+KD are on the same repacked ExllamaV2 kernel; fp16, w4 and w2 are the grid runs. Recovered = (KD − plain) / (fp16 − plain).

**BLEU**

| pair | fp16 | w4 | w3 | w3+KD | w2 | w2+KD r64 | w2+KD cont. | rec. w3+KD | rec. w2+KD | rec. w2+KD cont. |
|---|---|---|---|---|---|---|---|---|---|---|
| de-en | 31.59 | 31.18 | 28.91 | 31.41 | 0.10 | 29.63 | – | 93 % | 94 % | – |
| is-en | 40.21 | 39.09 | 21.62 | 39.75 | 0.00 | 35.40 | – | 98 % | 88 % | – |
| en-de | 28.05 | 27.03 | 25.20 | 28.01 | 0.04 | – | – | 98 % | – | – |
| en-is | 22.26 | 21.82 | 11.66 | 22.07 | 0.04 | – | – | 98 % | – | – |

**chrF++**

| pair | fp16 | w4 | w3 | w3+KD | w2 | w2+KD r64 | w2+KD cont. | rec. w3+KD | rec. w2+KD | rec. w2+KD cont. |
|---|---|---|---|---|---|---|---|---|---|---|
| de-en | 55.27 | 55.03 | 52.80 | 55.10 | 4.71 | 53.02 | – | 93 % | 96 % | – |
| is-en | 61.72 | 61.13 | 44.49 | 61.12 | 4.37 | 56.83 | – | 97 % | 91 % | – |
| en-de | 55.66 | 54.95 | 52.60 | 55.41 | 6.45 | – | – | 92 % | – | – |
| en-is | 50.58 | 50.32 | 37.74 | 50.09 | 5.09 | – | – | 96 % | – | – |

**XCOMET-XXL**

| pair | fp16 | w4 | w3 | w3+KD | w2 | w2+KD r64 | w2+KD cont. | rec. w3+KD | rec. w2+KD | rec. w2+KD cont. |
|---|---|---|---|---|---|---|---|---|---|---|
| de-en | 94.66 | 94.24 | 89.09 | 94.49 | 14.72 | – | – | 97 % | – | – |
| is-en | 81.15 | 80.70 | 46.65 | 81.23 | 12.60 | 74.45 | – | 100 % | 90 % | – |
| en-de | 97.71 | 97.53 | 95.16 | 97.55 | 39.32 | – | – | 94 % | – | – |
| en-is | 92.82 | 91.73 | 66.71 | 92.83 | 13.47 | – | – | 100 % | – | – |

## 4. Paired bootstrap 95 % confidence intervals (1,000 resamples)

| study | pair | metric | KD score [95 % CI] | recovered [95 % CI] | differences (* = CI excludes 0) |
|---|---|---|---|---|---|
| w3+KD | is-en | bleu | 39.75 [38.49, 40.99] | 98 % [95, 100] | w3kd-w3: +18.13 [+16.87, +19.31] *; w3kd-w4: +0.66 [-0.02, +1.33]; w3kd-fp16: -0.46 [-1.01, +0.07] |
| w3+KD | is-en | xcomet | 81.23 [79.69, 82.71] | 100 % [98, 102] | w3kd-w3: +34.58 [+32.62, +36.38] *; w3kd-w4: +0.53 [-0.28, +1.34]; w3kd-fp16: +0.08 [-0.59, +0.72] |
| w3+KD | en-is | bleu | 22.07 [21.07, 23.15] | 98 % [93, 104] | w3kd-w3: +10.41 [+9.47, +11.42] *; w3kd-w4: +0.25 [-0.42, +0.89]; w3kd-fp16: -0.19 [-0.79, +0.38] |
| w3+KD | en-is | xcomet | 92.83 [92.15, 93.49] | 100 % [98, 103] | w3kd-w3: +26.13 [+24.25, +28.05] *; w3kd-w4: +1.11 [+0.42, +1.77] *; w3kd-fp16: +0.02 [-0.61, +0.68] |
| w3+KD | de-en | bleu | 31.41 [30.56, 32.27] | 93 % [80, 108] | w3kd-w3: +2.49 [+1.81, +3.27] *; w3kd-w4: +0.23 [-0.17, +0.68]; w3kd-fp16: -0.18 [-0.56, +0.17] |
| w3+KD | de-en | xcomet | 94.49 [94.09, 94.88] | 97 % [92, 101] | w3kd-w3: +5.40 [+4.78, +5.99] *; w3kd-w4: +0.25 [-0.03, +0.55]; w3kd-fp16: -0.17 [-0.43, +0.08] |
| w3+KD | en-de | bleu | 28.01 [27.20, 28.79] | 98 % [83, 118] | w3kd-w3: +2.81 [+2.22, +3.39] *; w3kd-w4: +0.97 [+0.45, +1.50] *; w3kd-fp16: -0.04 [-0.51, +0.42] |
| w3+KD | en-de | xcomet | 97.55 [97.35, 97.76] | 94 % [89, 99] | w3kd-w3: +2.40 [+2.11, +2.72] *; w3kd-w4: +0.02 [-0.15, +0.19]; w3kd-fp16: -0.16 [-0.30, -0.03] * |
| w2+KD r64 | is-en | bleu | 35.40 [34.13, 36.65] | 88 % [86, 90] | w2kd-w2: +35.40 [+34.13, +36.65] *; w2kd-w3: +13.78 [+12.66, +14.87] *; w2kd-fp16: -4.81 [-5.72, -4.01] * |
| w2+KD r64 | de-en | bleu | 29.63 [28.58, 30.55] | 94 % [91, 95] | w2kd-w2: +29.53 [+28.49, +30.46] *; w2kd-w3: +0.71 [-0.15, +1.45]; w2kd-fp16: -1.97 [-2.77, -1.42] * |

## 5. Distillation runs: training loss and held-out KL vs fp16 (32 held-out train rows per direction)

| run | rank | steps | final train KD | is-en KL | en-is KL | de-en KL | zh-en KL |
|---|---|---|---|---|---|---|---|
| w2g128-kd-lora-r64 | 64 | 936 | 0.223 | 5.061 → 0.241 | 6.321 → 0.307 | 4.180 → 0.194 | 4.792 → 0.316 |
| w2g128-kd-lora-r64-cont | 64 | 1873 | 0.180 | 0.241 → 0.209 | 0.307 → 0.292 | 0.194 → 0.150 | 0.316 → 0.252 |
| w3g128-kd-lora | 16 | 936 | 0.031 | 0.696 → 0.032 | 0.482 → 0.037 | 0.262 → 0.024 | 0.652 → 0.047 |
| w3g128-kd-lora-smoke | 16 | 30 | 0.047 | 0.696 → 0.048 | 0.482 → 0.048 | 0.262 → 0.036 | 0.652 → 0.065 |

KL start → end of each run (a continuation starts where its parent ended). Full logs: `results/distill_runs/<run>/train_log.jsonl`.

## 6. Size and speed

| model | packed checkpoint GiB | + adapter (bf16) GiB | total GiB | fast-kernel container GiB |
|---|---|---|---|---|
| fp16 | 24.24 | – | 24.24 | – |
| w4 | 6.76 | – | 6.76 | 6.76 |
| w3 + KD r16 | 5.27 | 0.12 | 5.39 | 6.76 |
| w2 + KD r64 | 3.78 | 0.47 | 4.25 | 6.76 |

Generation speed (de-en, batch 4, beam 5, H100): fp16 0.359 s/line; w4 0.461; w3 on Torch 1.188; w3 repacked on ExllamaV2 0.470 (2.5×). Details: `docs/04-kernel-repack.md` §8.
