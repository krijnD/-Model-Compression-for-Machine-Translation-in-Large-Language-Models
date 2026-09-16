#!/usr/bin/env python
"""Merge X-ALMA group LoRA adapters into dense fp16 models.

For each group g in 1..8: load X-ALMA-13B-Pretrain (fp16) + the group LoRA,
PeftModel.merge_and_unload(), save to models/merged/g{g} with tokenizer.

Paper-supported loading strategy 2 (arXiv 2410.03115). Dense fp16 per group,
same parameter count as the base model.

Usage:
  python src/merge_xalma.py [--groups 1 2 3 ...] [--out-dir models/merged]
                            [--base haoranxu/X-ALMA-13B-Pretrain]
                            [--temp-dir <path>]   # RAM-starved CPUs: temp offload

Run on the Slurm cluster (13B fp16 ~= 26 GB). On a Mac with < 32 GB RAM this
is infeasible — models must be fetched via src/sync.sh instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "haoranxu/X-ALMA-13B-Pretrain"


def merged_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def merge_group(g: int, out_dir: str, base: str, device_map: dict | str) -> str:
    print(f"[merge] group {g}: loading base {base}", flush=True)
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    print(f"[merge] group {g}: base loaded ({time.time()-t0:.0f}s) base_params={merged_param_count(base_model):,}",
          flush=True)
    adapter = f"haoranxu/X-ALMA-13B-Group{g}"
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, adapter)
    print(f"[merge] group {g}: adapter loaded ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"[merge] group {g}: merged ({time.time()-t0:.0f}s) merged_params={merged_param_count(model):,}",
          flush=True)

    out_g = os.path.join(out_dir, f"g{g}")
    os.makedirs(out_g, exist_ok=True)
    model.save_pretrained(out_g, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(adapter)
    tok.save_pretrained(out_g)
    param_json = {"group": g, "num_params": merged_param_count(model),
                  "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(out_g, "merge_info.json"), "w") as f:
        json.dump(param_json, f, indent=2)
    print(f"[merge] group {g}: saved to {out_g}", flush=True)
    return out_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--out-dir", default="models/merged")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--device-map", default="auto",
                    help="'auto', 'cpu', or a device_map dict; 'auto' uses all visible GPUs")
    ap.add_argument("--temp-dir", default=None,
                    help="HF transformer temp dir for offloaded weights (low RAM)")
    args = ap.parse_args()

    # torch.run dispatch? --device-map auto mirrors ALMA's --multi_gpu_one_model
    torch.manual_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for g in args.groups:
        out_g = merge_group(g, args.out_dir, args.base, args.device_map)
        results[f"g{g}"] = out_g
    print(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())