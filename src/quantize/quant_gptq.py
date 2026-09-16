#!/usr/bin/env python
"""GPTQ quantization (cluster-only): GPTQModel, language-matched calibration.

QuantizeConfig(bits in {8,4,3,2}, group_size=128, desc_act=True, sym=True).
Calibration = 128 sequences x up to 2048 tokens sampled from
haoranxu/X-ALMA-Parallel-Data, only the languages of the target group
(language-matched, see arXiv 2508.20893).

Output: models/gptq/g{g}/b{bits}/ (safetensors + quantize_config.json).

Usage:
  python src/quantize/quant_gptq.py --model models/merged/g1 --bits 4 \
      --out models/gptq/g1/b4 --calib-langs de,is
"""
from __future__ import annotations

import ast
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

PARALLEL_DATA = "haoranxu/X-ALMA-Parallel-Data"
PROMPT = ("Translate this from {src} into {tgt}:\n{src}: {sentence}\n{tgt}:")
LANG_NAME = {
    "en": "English", "de": "German", "is": "Icelandic", "fr": "French",
    "mg": "Malagasy", "ru": "Russian", "cs": "Czech", "zh": "Chinese",
    "ja": "Japanese", "ar": "Arabic",
}


def build_calibration(model_dir: str, calib_langs: list[str], n_seq: int = 128,
                      max_len: int = 2048, seed: int = 42) -> list[str]:
    """Build n_seq prompt strings from Parallel-Data for the group's languages."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    torch.manual_seed(seed)
    texts = []
    for lang in calib_langs:
        cfg = f"{lang}-en"
        try:
            ds = load_dataset(PARALLEL_DATA, cfg, split="train", streaming=True)
        except Exception as e:
            print(f"  WARN: calib {cfg} unavailable: {e}")
            continue
        src_lang, tgt_lang = lang, "en"
        count = 0
        for ex in ds:
            # streaming rows: dict with 'translation' (sometimes JSON-string)
            text = ex.get("translation", ex)
            if isinstance(text, str):
                try:
                    text = ast.literal_eval(text)
                except Exception:
                    continue
            src_sent = text.get(src_lang) or (text.get("src") or "")
            if not src_sent:
                continue
            texts.append(PROMPT.format(src=LANG_NAME[src_lang],
                                       tgt=LANG_NAME[tgt_lang],
                                       sentence=src_sent))
            count += 1
            if count >= n_seq // len(calib_langs):
                break
        if count == 0:
            print(f"  WARN: no calib sentences for {cfg}")
    if not texts:
        raise RuntimeError("Empty calibration set — check X-ALMA-Parallel-Data configs")
    return texts[:n_seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged fp16 model dir")
    ap.add_argument("--bits", type=int, required=True, choices=[8, 4, 3, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-langs", required=True, help="comma-separated group langs")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from gptqmodel import GPTQ, QuantizeConfig
    except ImportError:
        print("gptqmodel not installed (cluster-only env).", file=os.sys.stderr)
        raise

    langs = [l.strip() for l in args.calib_langs.split(",") if l.strip()]
    print(f"[gptq] model={args.model} bits={args.bits} calib_langs={langs} "
          f"gs={args.group_size} n_seq={args.n_seq}")

    calib = build_calibration(args.model, langs, args.n_seq, args.max_len, args.seed)
    print(f"[gptq] calibration: {len(calib)} sequences")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(calib, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_len)
    calib_data = [{"input_ids": enc["input_ids"][i],
                   "attention_mask": enc["attention_mask"][i]}
                  for i in range(len(calib))]

    quant_config = QuantizeConfig(bits=args.bits, group_size=args.group_size,
                                  desc_act=True, sym=True, damp_percent=0.1)
    t0 = time.time()
    model = GPTQ.from_pretrained(args.model, quant_config, torch_dtype=torch.float16)
    model.quantize(calib_data, cache_examples_on_gpu=False)
    os.makedirs(args.out, exist_ok=True)
    model.save_quantized(args.out)
    print(f"[gptq] saved -> {args.out} in {time.time()-t0:.0f}s")
    meta = {"model": args.model, "bits": args.bits, "group_size": args.group_size,
            "desc_act": True, "sym": True, "calib_langs": langs,
            "n_seq_actual": len(calib), "max_len": args.max_len, "seed": args.seed,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(args.out, "quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()