#!/usr/bin/env python
"""GGUF generation harness: wraps llama-cli (llama.cpp) per direction.

Uses the ALMA prompt (no chat template), beam size 5 to mirror HF decode,
max_new_tokens 256, seed 42, -ngl 999 on CUDA / -ngl 99 on Metal.

Writes outputs/<run_id>/<direction>/{mt.txt,tokens.json} + timing.

Usage:
  python src/generate_gguf.py --model models/gguf/g1/q4_k_m.gguf --run-id gguf_q4_g1 \
      --direction en-de --src-file data/flores/en-de/src.txt [--threads N] [--jobs P]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device import platform_tag  # noqa: E402

LANG_NAMES = {
    "en": "English", "de": "German", "is": "Icelandic", "fr": "French",
    "mg": "Malagasy", "ru": "Russian", "cs": "Czech", "zh": "Chinese",
    "ja": "Japanese", "ar": "Arabic",
}


def alma_prompt(src_lang: str, tgt_lang: str, sentence: str) -> str:
    s, t = LANG_NAMES[src_lang], LANG_NAMES[tgt_lang]
    return (f"Translate this from {s} into {t}:\n"
            f"{s}: {sentence}\n{t}:")


def llama_cli_cmd(model: str, prompt: str, ngl: int, beams: int, threads: int,
                  max_new_tokens: int, seed: int) -> list[str]:
    return [
        "llama-cli", "-m", model, "-p", prompt,
        "--no-display-prompt", "--no-cnv", "-n", str(max_new_tokens),
        "--beam-size", str(beams), "-ngl", str(ngl),
        "-t", str(threads), "--seed", str(seed), "--temp", "0",
        "--ctx-size", "1024", "-nj", "1",
    ]


def run_one(model: str, prompt: str, ngl: int, beams: int, threads: int,
            max_new_tokens: int, seed: int) -> tuple[str, int, float]:
    t0 = time.time()
    proc = subprocess.run(
        llama_cli_cmd(model, prompt, ngl, beams, threads, max_new_tokens, seed),
        capture_output=True, text=True, timeout=300)
    out = proc.stdout
    # generation is the non-prompt tail; llama-cli echoes prefix in some modes
    gen = out.strip()
    if len(gen) > len(prompt) and gen.startswith(prompt):
        gen = gen[len(prompt):]
    gen = gen.replace("</s>", "").replace("<s>", "").strip()
    # trim to first newline (ALMA-style single sentence)
    gen = gen.split("\n")[0].strip()
    ntok = len(gen.split())
    return gen, ntok, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--src-file", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--ngl", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.ngl is None:
        args.ngl = 999 if platform_tag() == "slurm" else 99
    src_lang, tgt_lang = args.direction.split("-")
    src_lines = [l.strip() for l in open(args.src_file) if l.strip()]
    print(f"[gguf] {args.model} dir={args.direction} n={len(src_lines)} "
          f"ngl={args.ngl} beams={args.beams} jobs={args.jobs}", flush=True)

    def worker(line: str, idx: int):
        prompt = alma_prompt(src_lang, tgt_lang, line)
        gen, ntok, dt = run_one(args.model, prompt, args.ngl, args.beams,
                                args.threads, args.max_new_tokens, args.seed)
        return idx, gen, ntok, dt

    results = [None] * len(src_lines)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for idx, gen, ntok, dt in ex.map(worker, src_lines, range(len(src_lines))):
            results[idx] = (gen, ntok, dt)
            print(f"  [{idx+1}/{len(src_lines)}] {gen[:60]!r} ({dt:.1f}s)", flush=True)
    gen_time = time.time() - t0

    mt = [r[0] for r in results]
    toks = [r[1] for r in results]
    per_s = [r[2] for r in results]

    out_dir = os.path.join(args.out, args.run_id, args.direction)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mt.txt"), "w") as f:
        f.write("\n".join(mt) + "\n")
    tok_json = {"num_sentences": len(mt), "tokens_per_sentence": toks,
                "total_output_tokens": sum(toks),
                "num_beams": args.beams, "max_new_tokens": args.max_new_tokens,
                "seed": args.seed, "ngl": args.ngl, "threads": args.threads,
                "jobs": args.jobs,
                "load_time_s": None, "gen_time_s": round(gen_time, 3),
                "per_sentence_time_s": [round(x, 3) for x in per_s],
                "device": "metal" if args.ngl < 100 else "cuda",
                "platform": platform_tag(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump(tok_json, f, indent=2)
    print(f"[gguf] wrote {len(mt)} lines -> {out_dir}/mt.txt "
          f"({gen_time:.1f}s, {sum(toks)} tok)", flush=True)


if __name__ == "__main__":
    main()