"""GPTQ weight-only quantization of ALMA-13B-R with GPTQModel, one packed-int checkpoint per bit width.

Quantized: every Linear inside the decoder layers (self_attn q/k/v/o_proj, mlp gate/up/down_proj),
which is GPTQModel's Llama definition. Kept in fp16: embed_tokens, the RMSNorms and lm_head.
Calibration: ALMA's human-written parallel train data in its fine-tuning format (mtcompress/alma_prompt.py).

Output: <out-root>/<model folder name>-gptq-w<bits>g<group>[-sym]/ (e.g. models/ALMA-13B-R-gptq-w4g128/)
with the packed weights, quantize_config.json, the original tokenizer and generation_config files,
and quant_meta.json (all settings of the run).

Usage (venv-quant, one GPU): python scripts/quantize/quantize_gptq.py --bits 2 3 4 8
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alma_prompt import WMT22_PAIRS, calibration_examples  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
# Copied verbatim so tokenization and generation defaults are identical to the fp16 model.
COPY_FILES = ["tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]


def run_name(bits, group_size, sym):
    return f"gptq-w{bits}g{group_size if group_size > 0 else 'ch'}{'-sym' if sym else ''}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(PROJECT_DIR / "models/ALMA-13B-R"))
    p.add_argument("--out-root", default=str(PROJECT_DIR / "models"))
    p.add_argument("--bits", type=int, nargs="+", default=[4], choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128, help="-1 = one scale per output channel")
    p.add_argument("--sym", action="store_true", help="symmetric (default: asymmetric, with zero-points)")
    p.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=True,
                   help="act-order: quantize columns by decreasing Hessian diagonal (GPTQModel default)")
    p.add_argument("--damp-percent", type=float, default=0.05)
    p.add_argument("--n-samples", type=int, default=1024, help="calibration examples, balanced over the 10 directions")
    p.add_argument("--calib-pairs", default=",".join(WMT22_PAIRS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1, help="calibration forward batch size")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if Path(args.model, "adapter_config.json").exists():
        sys.exit(f"ERROR: {args.model} contains adapter_config.json; remove adapter_* files (see README)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_eos_token=False)
    calib = calibration_examples(tokenizer, args.n_samples, args.calib_pairs.split(","), seed=args.seed)
    n_tok = sum(len(ex["input_ids"]) for ex in calib)
    print(f"Calibration: {len(calib)} examples, {n_tok} tokens ({n_tok / len(calib):.0f} per example)")

    for bits in args.bits:
        name = run_name(bits, args.group_size, args.sym)
        out = Path(args.out_root) / f"{Path(args.model).name}-{name}"
        if out.exists() and not args.overwrite:
            print(f"Skipping {bits}-bit: {out} exists (pass --overwrite to redo)")
            continue
        print(f"=== {bits}-bit -> {out}")
        qcfg = QuantizeConfig(bits=bits, group_size=args.group_size, sym=args.sym,
                              desc_act=args.desc_act, damp_percent=args.damp_percent)
        # Load fresh per bit width: quantize() replaces the Linears in place.
        model = GPTQModel.load(args.model, qcfg, torch_dtype=torch.float16)
        t0 = time.time()
        model.quantize(calib, batch_size=args.batch_size, tokenizer=tokenizer)
        minutes = (time.time() - t0) / 60
        model.save(str(out))
        for f in COPY_FILES:
            shutil.copy2(Path(args.model, f), out / f)
        meta = {"run": name, "source_model": str(Path(args.model).resolve()), "bits": bits,
                "group_size": args.group_size, "sym": args.sym, "desc_act": args.desc_act,
                "damp_percent": args.damp_percent, "n_samples": len(calib), "calib_tokens": n_tok,
                "calib_pairs": args.calib_pairs, "seed": args.seed, "quantize_minutes": round(minutes, 1),
                "quantized_modules": "all decoder-layer Linears (q/k/v/o_proj, gate/up/down_proj)",
                "fp16_modules": "embed_tokens, input/post_attention_layernorm, model.norm, lm_head"}
        (out / "quant_meta.json").write_text(json.dumps(meta, indent=2))
        size_gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
        print(f"Saved {out} ({size_gb:.1f} GB weights, {minutes:.0f} min)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
