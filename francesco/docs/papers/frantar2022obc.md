# Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning

- **Authors:** Frantar, Alistarh
- **Venue:** NeurIPS 2022
- **ID:** arXiv:2208.11580
- **BibTeX key:** `frantar2022obc`

## Summary

Extends Optimal Brain Surgeon to post-training quantization: weights are quantized one at a time, and the remaining weights are updated with the inverse layer Hessian (from calibration inputs) to compensate the error. Accurate but too slow for billion-parameter models; GPTQ makes it scale.

## Why we cite it

The method GPTQ builds on (layer-wise, Hessian-based error compensation).
