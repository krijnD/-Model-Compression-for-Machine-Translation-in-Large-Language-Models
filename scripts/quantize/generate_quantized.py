"""Translate the WMT'22 test sets with a GPTQ-quantized ALMA-13B-R (venv-quant, one GPU per process).

ALMA's run_llmmt.py can't load packed GPTQ weights (it needs transformers<=4.45, GPTQModel needs >=4.56),
so this script reproduces its prediction path for the settings in generate_alma_r.job:
prompt from get_prompt, left padding to exactly --max-source-length (ALMA pads with "max_length"),
truncation, bf16, beam 5, max 256 new tokens, seed 42, decode prompt + generation and cut out the
translation with clean_outputstring. Output files are test-<src>-<tgt>, like run_llmmt.py writes.

Usage: python scripts/quantize/generate_quantized.py --model $MODELS_DIR/ALMA-13B-R-gptq-w4g128 \
           --pairs de-en,en-de --out $OUTPUTS_DIR/gptq-w4g128/wmt22 [--decoding beam]
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from gptqmodel import BACKEND, GPTQModel
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alma_prompt import clean_outputstring, get_key_suffix, get_prompt, load_test_sources  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="quantized model folder written by quantize_gptq.py")
    p.add_argument("--pairs", required=True, help="comma-separated, e.g. de-en,en-de")
    p.add_argument("--out", required=True)
    p.add_argument("--decoding", choices=["beam", "paper"], default="beam",
                   help="beam: do_sample=False (deterministic); paper: the model's generation_config (beam sampling)")
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--backend", default="auto", choices=[b.value for b in BACKEND])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0", help="cpu only for smoke tests with a tiny model")
    p.add_argument("--max-samples", type=int, help="only the first N sentences (for smoke tests)")
    args = p.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", add_eos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPTQModel.load(args.model, device=args.device, backend=BACKEND(args.backend),
                           torch_dtype=getattr(torch, args.dtype))
    model.eval()
    kernels = sorted({type(m).__name__ for m in model.model.modules() if "QuantLinear" in type(m).__name__})
    print(f"Loaded {args.model} | kernels: {kernels} | dtype {args.dtype}")

    gen_config = model.model.generation_config
    gen_config.max_length = args.max_source_length + args.max_new_tokens  # as ALMA's load_model
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, use_cache=True)
    if args.decoding == "beam":
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs.split(","):
        src, tgt = pair.split("-")
        sources = load_test_sources(src, tgt)[: args.max_samples]
        prompts = [get_prompt(src, tgt, s) for s in sources]
        suffix = get_key_suffix(tgt)
        hyps, t0 = [], time.time()
        for i in range(0, len(prompts), args.batch_size):
            enc = tokenizer(prompts[i:i + args.batch_size], max_length=args.max_source_length,
                            padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                gen = model.generate(**enc, **gen_kwargs)
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            hyps += [clean_outputstring(d.strip(), suffix) for d in decoded]
            if (i // args.batch_size) % 25 == 0:
                print(f"{pair}: {len(hyps)}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
        with open(out_dir / f"test-{src}-{tgt}", "w", encoding="utf-8") as f:
            f.writelines(h + "\n" for h in hyps)
        n_empty = sum(h == "" for h in hyps)
        print(f"{pair}: done, {len(hyps)} lines, {n_empty} empty, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
