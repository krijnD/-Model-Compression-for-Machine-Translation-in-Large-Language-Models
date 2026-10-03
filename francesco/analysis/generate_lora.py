"""scripts/generate_quantized.py with an optional LoRA adapter from distill_lora.py (see docs/06-distillation.md).

The prediction path is a copy of generate_quantized.py's (same prompt, left padding to exactly
--max-source-length, beam 5, bf16, seed 42, 256 new tokens, clean_outputstring), so outputs with and
without --adapter differ only by the adapter. Kept as a copy, not an edit, so scripts/ is unchanged.
The adapter wraps whatever kernel GPTQModel picks: on the repacked w3 (…-gptq-w3g128-as-w4) that is
ExllamaV2QuantLinear, which dequantizes to the same weights the adapter was trained on (TorchQuantLinear).

  python francesco/analysis/generate_lora.py --model /scratch-shared/$USER/models/ALMA-13B-R-gptq-w3g128-as-w4 \
      --adapter /scratch-shared/$USER/models/w3-kd-lora --pairs is-en,en-is --out ../outputs/gptq-w3g128-as4-kd/wmt22
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from gptqmodel import BACKEND, GPTQModel
from transformers import AutoTokenizer, set_seed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import lora  # noqa: E402
from alma_prompt import clean_outputstring, get_key_suffix, get_prompt, load_test_sources  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", help="folder written by distill_lora.py; omit for the plain model")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--backend", default="auto", choices=[b.value for b in BACKEND])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int)
    args = p.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", add_eos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gm = GPTQModel.load(args.model, device=args.device, backend=BACKEND(args.backend), torch_dtype=torch.bfloat16)
    model = gm.model  # the adapter goes into the HF model; generate() still goes through gm, as generate_quantized.py
    kernels = sorted({type(m).__name__ for m in model.modules() if "QuantLinear" in type(m).__name__})
    if args.adapter:
        cfg = lora.load(model, args.adapter)
        print(f"Adapter {args.adapter}: r={cfg['r']} alpha={cfg['alpha']} steps={cfg['steps']}")
    model.eval()
    print(f"Loaded {args.model} | kernels: {kernels} | adapter: {args.adapter}")

    gen_config = model.generation_config
    gen_config.max_length = args.max_source_length + args.max_new_tokens  # as ALMA's load_model
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, use_cache=True,
                      do_sample=False, temperature=None, top_p=None)

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
                gen = gm.generate(**enc, **gen_kwargs)
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
