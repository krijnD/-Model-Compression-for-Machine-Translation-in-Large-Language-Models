"""What weight-only GPTQ costs and buys next to fp16, measured on one H100.

The batch probe (run_probe.py) answers "can batch amortize the dequant?". This one answers the
other half of the same decision, for every checkpoint in the grid: does quantizing this model
have a memory payoff, and what does it cost in inference time and throughput?

Per checkpoint, in one process (so the peaks are that model's own):
  * load: wall time, resident weight bytes, peak during load
  * short protocol (de-en, source 256, BATCH=4) and long protocol (zh-en, source 512,
    BATCH_LONG=1) - exactly the two settings slurm/eval_quantized.job runs - one
    warmup pass excluded, then `--reps` timed generate() calls: median wall, per-line time,
    output tokens/s, and the peak *allocated* / *reserved* VRAM during the timed phase
  * the long protocol at BATCH=4 as well, which is the batch the fp16 baseline cannot afford
    there (generate_alma_r.job:77); OOM is a result, not an error
  * for quantized checkpoints: the kernel GPTQModel picked and the cost of one full
    dequantization pass (the per-forward cost that batch 4 pays 4x more often than batch 16)

Loading the fp16 folder goes through transformers (a packed GPTQ checkpoint needs GPTQModel);
everything else - prompts, padding, dtype, beam count, seed, generation kwargs - matches
scripts/quantize/generate_quantized.py, so the fp16 row is comparable to the quantized rows.

  python scripts/analysis/measure_quant_cost.py --model $MODELS_DIR/ALMA-13B-R --tag fp16 \
      --out results/json/quant_cost_fp16.json

Run through slurm/quant_cost.job, which loops over all five checkpoints.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent          # francesco/analysis
FRANCESCO = HERE.parent                         # francesco/
REPO = FRANCESCO.parent                         # the repository root
JSON_DIR = FRANCESCO / "results" / "json"
sys.path.insert(0, str(REPO / "scripts"))
from alma_prompt import get_key_suffix, get_prompt, load_test_sources  # noqa: E402

MIB = 1024 ** 2


def nvml_used_mib():
    """Node-wide used memory from nvidia-smi. The H100 nodes are shared (partition OverSubscribe=NO
    but 4 GPUs/node), so this is only a cross-check against the process-scoped allocator numbers."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return [ln.strip() for ln in out.splitlines()]
    except Exception as exc:                                # noqa: BLE001
        return [f"nvidia-smi unavailable: {exc}"]


def load_model(model_dir, dtype, backend="auto"):
    """(model, loader, kernels, n_quant_linears). Packed GPTQ checkpoints load through GPTQModel,
    the fp16 baseline through transformers. `backend` is the GPTQModel kernel to force ("auto"
    lets GPTQModel pick, which is what the eval jobs do)."""
    if (model_dir / "quantize_config.json").exists() or (model_dir / "quant_meta.json").exists():
        from gptqmodel import BACKEND, GPTQModel
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear
        model = GPTQModel.load(str(model_dir), device="cuda:0", backend=BACKEND(backend),
                               torch_dtype=dtype)
        kernels = sorted({type(m).__name__ for m in model.model.modules()
                          if "QuantLinear" in type(m).__name__})
        return model, f"gptqmodel({backend})", kernels, sum(1 for m in model.model.modules()
                                                            if isinstance(m, BaseQuantLinear))
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype,
                                                 device_map={"": "cuda:0"})
    return model, "transformers", [], 0


def dequant_pass_ms(model):
    """One full dequantization pass over every quantized weight: the cost a forward pass pays
    once regardless of batch (francesco/README.md, "Reconciling three dequant numbers"). Not every
    kernel exposes it (ExLlamaV2 does not), so a missing method is recorded as None, not an error."""
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    mods = [m for m in model.model.modules() if isinstance(m, BaseQuantLinear)]
    if not mods:
        return None, None
    if not all(hasattr(m, "dequantize_weight") for m in mods):
        return None, f"{type(mods[0]).__name__} has no dequantize_weight()"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for m in mods:
            w = m.dequantize_weight(num_itr=1)
            del w
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) * 1000, 1), None


def kv_bytes_per_token(model):
    """Analytic KV cache per token, to separate weights from cache in the measured peak."""
    cfg = getattr(model, "config", None) or model.model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return 2 * n_layers * n_kv * head_dim * 2          # K and V, bf16


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="fp16 / w8 / w4 / w3 / w2 - the row name")
    p.add_argument("--out", default=None)
    p.add_argument("--short-pair", default="de-en")
    p.add_argument("--short-batch", type=int, default=4)
    p.add_argument("--short-source-length", type=int, default=256)
    p.add_argument("--long-pair", default="zh-en")
    p.add_argument("--long-batch", type=int, default=1)
    p.add_argument("--long-batch-probe", type=int, default=4,
                   help="run the long pair at this batch too (0 disables); OOM is recorded, not raised")
    p.add_argument("--long-source-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--samples", type=int, default=0,
                   help="also time this many real lines of each pair (0 = the fixed-workload "
                        "phases only); the first few sentences of a WMT pair generate ~30 tokens, "
                        "so this is what makes a number comparable to the grid's own logs")
    p.add_argument("--samples-long", type=int, default=None,
                   help="lines for the long pair (default: same as --samples; 0 skips it). zh-en at "
                        "source 512 on one GPU costs ~3x per line what de-en does at source 256")
    p.add_argument("--backend", default="auto",
                   help="GPTQModel kernel to force (auto = what the eval jobs do); recorded in the "
                        "JSON as kernels, so a sweep can say which kernel each number came from")
    p.add_argument("--phases", default="short,long,long_batch_probe",
                   help="comma-separated subset of short,long,long_batch_probe to run")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    model_dir = Path(args.model)
    dtype = getattr(torch, args.dtype)
    from transformers import AutoTokenizer, set_seed
    set_seed(args.seed)

    result = {"tag": args.tag, "model": str(model_dir), "dtype": args.dtype,
              "backend_arg": args.backend,
              "beams": args.num_beams, "reps": args.reps,
              "max_new_tokens": args.max_new_tokens,
              "nvml_before": nvml_used_mib()}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), padding_side="left",
                                              add_eos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, loader, kernels, n_quant = load_model(model_dir, dtype, args.backend)
    model.eval()
    torch.cuda.synchronize()
    result.update(
        loader=loader, kernels=kernels, quant_linears=n_quant,
        load_s=round(time.perf_counter() - t0, 1),
        load_peak_alloc_mib=round(torch.cuda.max_memory_allocated() / MIB, 1),
        load_peak_reserved_mib=round(torch.cuda.max_memory_reserved() / MIB, 1),
    )
    resident = (sum(p.numel() * p.element_size() for p in model.parameters())
                + sum(b.numel() * b.element_size() for b in model.buffers()))
    result.update(weights_mib=round(resident / MIB, 1),
                  weights_gib=round(resident / 1024 ** 3, 2),
                  kv_mib_per_token=round(kv_bytes_per_token(model) / MIB, 4),
                  n_params=sum(p.numel() for p in model.parameters()))
    print(f"[{args.tag}] loader={loader} kernels={kernels} linears={n_quant} "
          f"weights={result['weights_gib']} GiB load={result['load_s']}s "
          f"load_peak={result['load_peak_alloc_mib']} MiB", flush=True)

    gen_config = getattr(model, "generation_config", None) or model.model.generation_config
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                      use_cache=True, do_sample=False, temperature=None, top_p=None)
    suffix_cache = {}

    def phase(name, pair, batch, source_length):
        src, tgt = pair.split("-")
        gen_config.max_length = source_length + args.max_new_tokens
        if src not in suffix_cache:
            prompts = [get_prompt(src, tgt, s) for s in load_test_sources(src, tgt)]
            suffix_cache[src] = (prompts, get_key_suffix(tgt))
        prompts, _ = suffix_cache[src]
        enc = tokenizer(prompts[:batch], max_length=source_length, padding="max_length",
                        truncation=True, return_tensors="pt").to(args.device)
        input_len = enc["input_ids"].shape[1]
        rec = {"phase": name, "pair": pair, "batch": batch,
               "source_length": source_length, "input_len": input_len}
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():                     # warmup: Triton JIT / autotune / cudnn
                _ = model.generate(**enc, **gen_kwargs)
            torch.cuda.synchronize()
            del _
            torch.cuda.reset_peak_memory_stats()             # timed phase owns the peak
            walls, new_tokens = [], []
            for _ in range(args.reps):
                t = time.perf_counter()
                with torch.inference_mode():
                    gen = model.generate(**enc, **gen_kwargs)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t)
                new_tokens.append(int((gen.shape[1] - input_len)))
                del gen
            median = statistics.median(walls)
            mean_new = statistics.mean(new_tokens)
            rec.update(ok=True, walls_s=[round(w, 3) for w in walls], median_s=round(median, 3),
                       s_per_line=round(median / batch, 3),
                       new_tokens_per_line=mean_new,
                       out_tokens_per_s=round(batch * mean_new / median, 2),
                       peak_alloc_mib=round(torch.cuda.max_memory_allocated() / MIB, 1),
                       peak_reserved_mib=round(torch.cuda.max_memory_reserved() / MIB, 1))
            rec["kv_mib_in_flight"] = round(
                result["kv_mib_per_token"] * batch * args.num_beams * gen_config.max_length, 1)
            print(f"[{args.tag}] {name}: median={median:.2f}s {median/batch:.3f}s/line "
                  f"peak_alloc={rec['peak_alloc_mib']:.0f} MiB peak_reserved="
                  f"{rec['peak_reserved_mib']:.0f} MiB out_tok/s={rec['out_tokens_per_s']}",
                  flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            rec.update(ok=False, oom=True, error=str(exc).splitlines()[0][:200])
            print(f"[{args.tag}] {name}: OOM (recorded, not raised)", flush=True)
        except RuntimeError as exc:                          # non-OOM CUDA errors
            rec.update(ok=False, oom=False, error=str(exc).splitlines()[0][:200])
            print(f"[{args.tag}] {name}: failed: {rec['error']}", flush=True)
        finally:
            del enc
            torch.cuda.empty_cache()
        return rec

    def sample_phase(name, pair, batch, source_length, samples):
        """Real lines, not the first few: the throughput the grid actually pays, on this node."""
        src, _ = pair.split("-")
        prompts, _ = suffix_cache[src]
        gen_config.max_length = source_length + args.max_new_tokens
        torch.cuda.reset_peak_memory_stats()
        done, gen_tok = 0, 0
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                for i in range(0, samples, batch):
                    enc = tokenizer(prompts[i:i + batch], max_length=source_length,
                                    padding="max_length", truncation=True,
                                    return_tensors="pt").to(args.device)
                    gen = model.generate(**enc, **gen_kwargs)
                    gen_tok += int((gen.shape[1] - enc["input_ids"].shape[1])) * gen.shape[0]
                    done += gen.shape[0]
                    del gen, enc
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            rec = {"phase": name, "pair": pair, "batch": batch, "lines": done,
                   "source_length": source_length, "ok": True, "wall_s": round(wall, 1),
                   "s_per_line": round(wall / done, 3),
                   "mean_new_tokens": round(gen_tok / done, 1),
                   "out_tokens_per_s": round(gen_tok / wall, 2),
                   "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
                   "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1)}
            print(f"[{args.tag}] {name}: {done} real lines, {rec['s_per_line']} s/line "
                  f"({rec['mean_new_tokens']} tokens), peak_alloc={rec['peak_alloc_mib']:.0f} MiB",
                  flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            rec = {"phase": name, "pair": pair, "batch": batch, "lines": done,
                   "source_length": source_length, "ok": False,
                   "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                   "error": str(exc).splitlines()[0][:200]}
            print(f"[{args.tag}] {name}: failed after {done} lines: {rec['error']}", flush=True)
        torch.cuda.empty_cache()
        return rec

    wanted = {x.strip() for x in args.phases.split(",") if x.strip()}
    phases = [p for p in [
        ("short", args.short_pair, args.short_batch, args.short_source_length),
        ("long", args.long_pair, args.long_batch, args.long_source_length),
        ("long_batch_probe", args.long_pair, args.long_batch_probe, args.long_source_length),
    ] if p[0] in wanted]
    if "long" not in wanted:
        args.samples_long = 0                       # nothing to sample if the pair is not run
    if args.samples:                       # prime the prompt cache before the sampled phases
        for _, pair, _, _ in phases:
            src, tgt = pair.split("-")
            if src not in suffix_cache:
                suffix_cache[src] = ([get_prompt(src, tgt, s) for s in load_test_sources(src, tgt)],
                                     get_key_suffix(tgt))
    result["phases"] = {name: phase(name, pair, batch, slen) for name, pair, batch, slen in phases}
    if args.samples:
        long_n = args.samples if args.samples_long is None else args.samples_long
        result["samples"] = {"short": sample_phase("short_samples", args.short_pair,
                                                   args.short_batch, args.short_source_length,
                                                   args.samples)}
        if long_n:                     # --samples-long 0 = skip the expensive long pair
            result["samples"]["long"] = sample_phase("long_samples", args.long_pair,
                                                     args.long_batch, args.long_source_length,
                                                     long_n)

    deq, deq_err = dequant_pass_ms(model)
    result["dequant_pass_ms"], result["dequant_error"] = deq, deq_err
    result["nvml_after"] = nvml_used_mib()
    if deq is not None:
        print(f"[{args.tag}] one dequantization pass over all quantized weights: {deq} ms",
              flush=True)

    out = Path(args.out) if args.out else JSON_DIR / f"quant_cost_{args.tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
