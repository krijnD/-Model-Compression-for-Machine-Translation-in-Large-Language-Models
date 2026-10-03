"""Is batch size a real lever for GPTQ 2-bit / 3-bit generation?

Background: slurm/eval_quantized.job runs at BATCH=4. The 2- and 3-bit runs are
3-10x slower per line than 4-bit, which threatens the 20 h wall on the w2 run. If that time is
per-forward dequantization of the weight matrix, a larger batch amortizes it (dequant runs once
per forward, regardless of batch); if it is the bf16 matmul, batch size buys nothing beyond kernel
efficiency. Everything in gptqmodel's TritonV2 is quant_matmul():
"W = dequant(...); input @ W" - i.e. dequant-then-matmul, NOT a fused kernel.

This probe measures the same generation at several batch sizes and, independently, the cost of
the dequant alone, which is what the decision needs.

Run via slurm/probe.sbatch (which activates venv-quant and holds the GPU). Reuses the repo's
own module loader (alma_prompt.py) so prompts/padding/extraction match make the number comparable
to the job logs: same prompt, max_length padding, max_source_length, bf16, beam 5, seed 42.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from transformers import AutoTokenizer, set_seed

HERE = Path(__file__).resolve().parent          # francesco/analysis
FRANCESCO = HERE.parent                         # francesco/
REPO = FRANCESCO.parent                         # the repository root
JSON = FRANCESCO / "results" / "json"           # where these scripts write their results
sys.path.insert(0, str(REPO / "scripts"))
from alma_prompt import clean_outputstring, get_key_suffix, get_prompt, load_test_sources  # noqa: E402

MIB = 1024 ** 2


def timed_generate(model, tokenizer, enc, gen_kwargs, n):
    """n sequential generate() calls on the same batch; returns per-call seconds."""
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**enc, **gen_kwargs)
        times.append(time.perf_counter() - t0)
        del gen
    return times


def measure_dequant(model):
    """Seconds to dequantize every quantized weight, once. Batch-independent: this is the
    per-forward cost that a bigger batch amortizes (called once per forward either way).
    Not every kernel exposes dequantize_weight() (ExLlamaV2 does not); then this returns None."""
    mods = [m for m in model.model.modules() if isinstance(m, BaseQuantLinear)]
    if not mods or not all(hasattr(m, "dequantize_weight") for m in mods):
        return None, len(mods)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for m in mods:
        w = m.dequantize_weight(num_itr=1)
        del w
    torch.cuda.synchronize()
    return time.perf_counter() - t0, len(mods)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(REPO.parent / "models/ALMA-13B-R-gptq-w2g128"))
    p.add_argument("--pair", default="de-en")
    p.add_argument("--batches", default="4,16", help="comma-separated batch sizes to compare")
    p.add_argument("--reps", type=int, default=3, help="generate() calls timed per batch size")
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--decode", type=int, default=4, help="batches to decode between different batch sizes")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(JSON / "probe_w2.json"))
    p.add_argument("--adapter", help="LoRA folder from distill_lora.py, applied after loading (kept in fp32, as in "
                   "generate_lora.py), so its memory is part of the measured peak")
    args = p.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", add_eos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (Path(args.model) / "quantize_config.json").exists() or \
            (Path(args.model) / "quant_meta.json").exists():
        model = GPTQModel.load(args.model, device=args.device, backend=BACKEND("auto"),
                              torch_dtype=getattr(torch, args.dtype))
    else:                                   # the fp16 baseline: same generate(), plain transformers
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                     device_map={"": args.device})
    if args.adapter:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import lora
        cfg = lora.load(model.model, args.adapter)
        print(f"adapter={args.adapter} r={cfg['r']} steps={cfg['steps']}", flush=True)
    model.eval()
    kernels = sorted({type(m).__name__ for m in model.model.modules() if "QuantLinear" in type(m).__name__})
    n_quant = sum(1 for m in model.model.modules() if isinstance(m, BaseQuantLinear))
    print(f"model={args.model}", flush=True)
    print(f"kernels={kernels} quant_linears={n_quant} dtype={args.dtype}", flush=True)

    gen_config = getattr(model, "generation_config", None) or model.model.generation_config
    gen_config.max_length = args.max_source_length + args.max_new_tokens
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, use_cache=True,
                      do_sample=False, temperature=None, top_p=None)

    src, tgt = args.pair.split("-")
    sources = load_test_sources(src, tgt)
    prompts = [get_prompt(src, tgt, s) for s in sources]
    suffix = get_key_suffix(tgt)
    print(f"pair={args.pair} prompts={len(prompts)}", flush=True)

    results = {"model": args.model, "adapter": args.adapter, "kernels": kernels, "quant_linears": n_quant,
               "pair": args.pair, "reps": args.reps, "per_batch": {}}

    # warmup on the largest batch so lazy init (Triton JIT/autotune) is not charged to any row
    warm = max(batches)
    enc_w = tokenizer(prompts[:warm], max_length=args.max_source_length, padding="max_length",
                      truncation=True, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        _ = model.generate(**enc_w, **gen_kwargs)
    del _, enc_w
    torch.cuda.empty_cache()
    print("warmup done", flush=True)

    for b in batches:
        enc = tokenizer(prompts[:b], max_length=args.max_source_length, padding="max_length",
                        truncation=True, return_tensors="pt").to(args.device)
        torch.cuda.reset_peak_memory_stats()            # so each batch size owns its own peak
        times = timed_generate(model, tokenizer, enc, gen_kwargs, args.reps)
        med = statistics.median(times)
        rec = {"batch": b, "times_s": [round(t, 3) for t in times], "median_s": round(med, 3),
               "s_per_seq_sent": round(med / b, 3),
               "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
               "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1)}
        results["per_batch"][b] = rec
        print(f"BATCH={b:3d} calls={[round(t,1) for t in times]} median={med:.2f}s "
              f"per-input-sentence={med/b:.3f}s peak_alloc={rec['peak_alloc_mib']:.0f} MiB "
              f"peak_reserved={rec['peak_reserved_mib']:.0f} MiB", flush=True)
        del enc

    deq, n_mod = measure_dequant(model)
    results["dequant_once_s"] = None if deq is None else round(deq, 4)
    results["dequant_modules"] = n_mod
    print(f"dequant of all {n_mod} weight matrices, once: "
          + ("not exposed by this kernel" if deq is None else f"{deq*1000:.1f} ms"), flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
