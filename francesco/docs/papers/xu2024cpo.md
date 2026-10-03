# Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation

- **Authors:** Xu, Sharaf, Chen, Tan, Shen, Van Durme, Murray, Kim
- **Venue:** ICML 2024
- **ID:** arXiv:2401.08417
- **BibTeX key:** `xu2024cpo`

## Summary

Introduces CPO and ALMA-13B-R. CPO trains the model to prefer better translations over adequate-but-flawed ones (preference data built from model outputs and references, ranked by reference-free metrics), instead of imitating references, which can be of lower quality than the model output. Applied as LoRA on ALMA-13B-LoRA with only 22K parallel sentences. ALMA-13B-R matches or exceeds WMT competition winners and GPT-4 on WMT'21/'22/'23 test sets. Evaluated mainly with reference-free KIWI-22, KIWI-XXL and XCOMET.

## Why we cite it

Our fp16 baseline; why we distil from the model instead of fine-tuning on references; evaluation protocol.
