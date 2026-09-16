#!/usr/bin/env python
"""Generation harness (HF stack): fp16 / GPTQ / LLM.int8 / torchao / W8A8.

Loads a model artifact uniformly via transformers AutoModelForCausalLM
(GPTQ via gptqmodel GPTQConfig, int8 via load_in_8bit, torchao/W8A8 via
torchao quant api), builds the ALMA prompt, beam-decodes, writes
outputs/<run_id>/<direction>/{mt.txt,tokens.json} plus timing.

Decode (AGENTS.md): ALMA prompt, num_beams=5, max_new_tokens=256,
max_source_length=256, seed 42.

Usage:
  python src/model/generate.py --model <artifact|hf-id> --run-id <id>
      --direction en-de --src-file data/flores/en-de/src.txt
      [--ref-file ...] [--load-mode fp16|gptq|int8|torchao] [--out outputs]
      [--batch-size N] [--num-beams 5] [--max-new-tokens 256]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.device import device, device_name, max_batch, platform_tag  # noqa: E402

# ALMA language names (full names per fe1ixxu/ALMA run_llmmt.py mapping).
LANG_NAMES = {
    "en": "English", "de": "German", "is": "Icelandic", "fr": "French",
    "mg": "Malagasy", "ru": "Russian", "cs": "Czech", "zh": "Chinese",
    "ja": "Japanese", "ar": "Arabic",
}


def alma_prompt(src_lang: str, tgt_lang: str, sentence: str) -> str:
    s, t = LANG_NAMES[src_lang], LANG_NAMES[tgt_lang]
    return (f"Translate this from {s} into {t}:\n"
            f"{s}: {sentence}\n{t}:")


def postprocess(text: str) -> str:
    """Trim to first newline / trailing whitespace (ALMA-style)."""
    text = text.split("</s>")[0].split("<s>")[-1]
    text = text.split("\n")[0] if "\n" in text else text
    return text.strip()


def load_model(artifact: str, mode: str, device_name: str):
    if mode == "gptq":
        qcfg = GPTQConfig(bits=0, disable_exllama=True)  # bits read from artifact
        model = AutoModelForCausalLM.from_pretrained(
            artifact, quantization_config=qcfg, device_map="auto")
    elif mode == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            artifact, load_in_8bit=True, device_map="auto")
    elif mode == "torchao":
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight
        model = AutoModelForCausalLM.from_pretrained(
            artifact, torch_dtype=torch.float16, device_map="auto")
        quantize_(model, int8_dynamic_activation_int8_weight())
    else:  # fp16 (dense merged)
        model = AutoModelForCausalLM.from_pretrained(
            artifact, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--direction", required=True, help="e.g. en-de")
    ap.add_argument("--src-file", required=True)
    ap.add_argument("--ref-file", default="")
    ap.add_argument("--load-mode", default="fp16",
                    choices=["fp16", "gptq", "int8", "torchao"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-source-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    bs = args.batch_size or max_batch()
    src_lang, tgt_lang = args.direction.split("-")

    src_lines = [l.strip() for l in open(args.src_file) if l.strip()]

    print(f"[generate] model={args.model} mode={args.load_mode} dir={args.direction} "
          f"n={len(src_lines)} device={dev} batch={bs}", flush=True)
    t0 = time.time()
    model = load_model(args.model, args.load_mode, device_name())
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(dev)
    t_load = time.time() - t0
    print(f"[generate] loaded in {t_load:.1f}s", flush=True)

    mt = []
    tokens_out = []
    gen_start = time.time()
    for i in range(0, len(src_lines), bs):
        chunk = src_lines[i:i + bs]
        prompts = [alma_prompt(src_lang, tgt_lang, s) for s in chunk]
        enc = tok(prompts, return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=args.max_source_length).to(dev)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        for j, o in enumerate(out):
            n_prompt = enc["input_ids"][j].shape[0]
            new_toks = o[n_prompt:]
            tokens_out.append(int(new_toks.shape[0]))
            text = tok.decode(new_toks, skip_special_tokens=True)
            mt.append(postprocess(text))
        if (i // bs) % 5 == 0:
            print(f"  {i + len(chunk)}/{len(src_lines)}", flush=True)
    gen_time = time.time() - gen_start

    out_dir = os.path.join(args.out, args.run_id, args.direction)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mt.txt"), "w") as f:
        f.write("\n".join(mt) + "\n")
    tok_json = {"num_sentences": len(mt), "tokens_per_sentence": tokens_out,
                "total_output_tokens": sum(tokens_out),
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "load_time_s": round(t_load, 3),
                "gen_time_s": round(gen_time, 3),
                "device": device_name(),
                "platform": platform_tag(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump(tok_json, f, indent=2)
    print(f"[generate] wrote {len(mt)} lines -> {out_dir}/mt.txt "
          f"({gen_time:.1f}s, {sum(tokens_out)} tok)", flush=True)


if __name__ == "__main__":
    main()