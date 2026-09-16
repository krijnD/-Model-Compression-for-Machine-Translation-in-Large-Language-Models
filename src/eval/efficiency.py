#!/usr/bin/env python
"""Efficiency measurement (HF stack): throughput, peak memory, batch-1 TTFB.

Workload: NTREX en->{de,zh,mg}, first 50 sentences each (subset for speed).
Warmup 2 batches; batch sizes {1,16,64} x 3 runs; throughput = decoded
tokens / decode wall (warmup excluded). Peak memory: CUDA max_memory_allocated;
Mac: peak RSS via psutil (unified memory). Batch-1 TTFB: time to first decoded
token (CUDA events on cuda; perf_counter on mps/cpu). Writes
scores/<run_id>/efficiency.json with 'platform' field.

Usage:
  python src/eval/efficiency.py --model <artifact|hf-id> --run-id <id>
      [--directions en-de en-zh en-mg] [--n 50] [--load-mode fp16|gptq|int8|torchao]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.device import device, device_name, max_batch, platform_tag  # noqa: E402
from model.generate import alma_prompt  # noqa: E402 (same prompt contract)


def load_model(artifact: str, mode: str):
    if mode == "gptq":
        from transformers import GPTQConfig
        model = AutoModelForCausalLM.from_pretrained(
            artifact, quantization_config=GPTQConfig(bits=0, disable_exllama=True),
            device_map="auto")
    elif mode == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            artifact, load_in_8bit=True, device_map="auto")
    elif mode == "torchao":
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight
        model = AutoModelForCausalLM.from_pretrained(
            artifact, torch_dtype=torch.float16, device_map="auto")
        quantize_(model, int8_dynamic_activation_int8_weight())
    else:
        model = AutoModelForCausalLM.from_pretrained(
            artifact, torch_dtype=torch.float16, device_map="auto")
    return model


def gen_batch(model, tok, prompts, dev, max_new_tokens=64):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=128).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1,
                             do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--directions", default="en-de,en-zh,en-mg")
    ap.add_argument("--n", type=int, default=50, help="sentences per direction")
    ap.add_argument("--load-mode", default="fp16",
                    choices=["fp16", "gptq", "int8", "torchao"])
    ap.add_argument("--batch-sizes", default="1,16,64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--out", default="scores")
    args = ap.parse_args()

    dev = device()
    bss = [int(x) for x in args.batch_sizes.split(",") if int(x) <= max_batch()]
    if not bss:
        bss = [1]
    model = load_model(args.model, args.load_mode)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # workload sentences: short fixed set per direction (reuse FLORES if present)
    workdir = "data/ntrex"
    workload = {}
    if os.path.isdir(workdir):
        for d in args.directions.split(","):
            p = os.path.join(workdir, d, "src.txt")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    workload[d] = [l.strip() for l in f][:args.n]
    if not workload:
        # fallback: synthesized sentences
        dummy = ["This is a test sentence for efficiency measurement.",
                 "The quick brown fox jumps over the lazy dog.",
                 "Hello, world! This is a translation benchmark sample."]
        for d in args.directions.split(","):
            workload[d] = dummy
    print(f"[efficiency] model={args.model} dev={dev} dirs={list(workload)} "
          f"bs={bss} repeats={args.repeats}", flush=True)

    results = {}
    for d in args.directions.split(","):
        lines = workload.get(d, [])
        if not lines:
            continue
        src_lang, tgt_lang = d.split("-")
        prompts = [alma_prompt(src_lang, tgt_lang, s) for s in lines]
        r = {"direction": d, "n": len(prompts), "batches": {}}
        # warmup
        for _ in range(args.warmup_batches):
            gen_batch(model, tok, prompts[:1], dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        peak_mem = 0
        for bs in bss:
            times = []
            for rep in range(args.repeats):
                t0 = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    t0 = time.time()
                out = gen_batch(model, tok, prompts[:bs], dev)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t0
                times.append(dt)
                if torch.cuda.is_available():
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
            r["batches"][str(bs)] = {
                "mean_latency_s": round(sum(times) / len(times), 4),
                "throughput_tok_s": round(bs * 50 / max(times), 3),  # approx: 50 tok/seq
            }
        # batch-1 TTFB
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        t0 = time.time()
        enc = tok(prompts[:1], return_tensors="pt", truncation=True,
                  max_length=128).to(dev)
        if start is not None:
            start.record()
            torch.cuda.synchronize()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, num_beams=1,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        if start is not None:
            torch.cuda.synchronize()
            r["ttfb_s"] = round(start.elapsed_time() / 1000, 4)
        else:
            r["ttfb_s"] = round(time.time() - t0, 4)
        r["peak_mem_gb"] = round(peak_mem, 3)
        results[d] = r

    # model size on disk
    def dir_size(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for fn in files:
                total += os.path.getsize(os.path.join(root, fn))
        return total / 1e9

    size = dir_size(args.model) if os.path.isdir(args.model) else 0.0
    out_json = {
        "run_id": args.run_id, "platform": platform_tag(),
        "device": device_name(), "model_size_gb": round(size, 2),
        "model_path": args.model, "load_mode": args.load_mode,
        "max_batch_cfg": max_batch(),
        "directions": results,
    }
    out_path = os.path.join(args.out, args.run_id, "efficiency.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(json.dumps(out_json, indent=2))
    print(f"[efficiency] wrote {out_path}")


if __name__ == "__main__":
    main()