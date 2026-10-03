# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

- **Authors:** Frantar, Ashkboos, Hoefler, Alistarh
- **Venue:** ICLR 2023
- **ID:** arXiv:2210.17323
- **BibTeX key:** `frantar2023gptq`

## Summary

One-shot, weight-only post-training quantization. Each layer is quantized column by column; second-order (Hessian) information from a small calibration set is used to correct the error of the remaining weights. Works to 3-4 bits on 175B models.

## Why we cite it

Our quantization method (w8/w4/w3/w2, group 128, asymmetric, act-order).
