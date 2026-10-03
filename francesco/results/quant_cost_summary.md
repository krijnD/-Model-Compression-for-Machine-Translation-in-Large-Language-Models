| checkpoint | bits | kernel | GiB on disk | peak GiB de-en 256 @4 | fixed s de-en | real s de-en (n) | peak GiB zh-en 512 @1 | fixed s zh-en | real s zh-en (n) | grid GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16 | fp16 | 24.24 | 28.8 | 0.359 | 0.436 (1984) | 26.5 | 0.517 | 1.471 (256) | 10.6 |
| W8G128 | 8 | TritonV2 | 12.72 | 17.2 | 0.607 | 0.755 (512) | 15.1 | 0.877 | 2.683 (256) | 6.6 |
| W4G128 | 4 | ExLLamaV2 | 6.76 | 12.3 | 0.460 | 0.550 (512) | 10.0 | 0.543 | 1.534 (256) | 4.5 |
| W3G128 | 3 | Torch | 5.27 | 10.1 | 1.181 | 1.467 (64) | 7.8 | 1.377 | 4.570 (64) | 13.6 |
| W2G128 | 2 | TritonV2 | 3.78 | 11.7 | 4.705 | 3.513 (64) | 6.4 | 10.138 | 10.577 (64) | 19.7 |

Peak VRAM is the allocator peak of one timed `generate()` at the eval protocol's own shape; *fixed s* is the median of three such calls on the first four sentences of the pair (a byte-identical workload for every row, ~30 generated tokens); *real s (n)* is the same model over `n` full lines of that pair at the same batch. *grid GPU-h* is what the cluster billed for that run's generation (the fp16 baseline is 4x its wall: four ranks over four GPUs). `--` = not measured.

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
