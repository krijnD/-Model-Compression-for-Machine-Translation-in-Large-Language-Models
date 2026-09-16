#!/usr/bin/env python
"""Matrix definition: 15 quantized points per group + FP16 baseline.

Single source of truth for run_id naming, artifact paths and load mode, used by
the Slurm job generators (P6) and the analysis (P7).

Kinds and variants (AGENTS.md §2):
  fp16      : fp16                 (baseline, dense merged fp16)
  weights   : gguf q8_0 q6_k q5_k_m q4_k_m q3_k_m q2_k
  activation: sq_w8a8, llmint8
  calibration: gptq b8 b4 b3 b2
  layerwise : het_q8q4, het_q8q2q3

Usage:
  python src/matrix.py --groups 1                  # list run_ids for group 1
  python src/matrix.py --groups 1 --format tsv     # run_id<TAB>kind<TAB>artifact<TAB>load_mode
  python src/matrix.py --groups 1 --emit-sbatch src/slurm/generate.sbatch > commands.sh
"""
from __future__ import annotations

import argparse
import json
import os

GGUF_VARIANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
HET_VARIANTS = ["het_q8q4", "het_q8q2q3"]
GPTQ_BITS = [8, 4, 3, 2]
GROUP_LANGS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_langs.json")


def rows(group: int) -> list[dict]:
    g = f"g{group}"
    out = [{
        "run_id": f"fp16_{g}", "kind": "fp16", "variant": "fp16", "bits": 16,
        "group": group, "model_artifact": f"models/x-alma-13b-group{group}",
        "load_mode": "fp16", "stack": "hf",
    }]
    for v in GGUF_VARIANTS:
        out.append({
            "run_id": f"gguf_{v}_{g}", "kind": "weights", "variant": v,
            "bits": {"q8_0": 8, "q6_k": 6, "q5_k_m": 5, "q4_k_m": 4,
                     "q3_k_m": 3, "q2_k": 2}[v],
            "group": group, "model_artifact": f"models/gguf/{g}/{v}.gguf",
            "load_mode": "gguf", "stack": "llamacpp",
        })
    for v in HET_VARIANTS:
        out.append({
            "run_id": f"{v}_{g}", "kind": "layerwise", "variant": v, "bits": None,
            "group": group, "model_artifact": f"models/gguf/{g}/{v}.gguf",
            "load_mode": "gguf", "stack": "llamacpp",
        })
    for b in GPTQ_BITS:
        out.append({
            "run_id": f"gptq_b{b}_{g}", "kind": "calibration", "variant": f"b{b}",
            "bits": b, "group": group, "model_artifact": f"models/gptq/{g}/b{b}",
            "load_mode": "gptq", "stack": "hf",
        })
    out.append({
        "run_id": f"sq_w8a8_{g}", "kind": "activation", "variant": "sq_w8a8",
        "bits": 8, "group": group, "model_artifact": f"models/w8a8/{g}",
        "load_mode": "fake_quant", "stack": "hf",
    })
    out.append({
        "run_id": f"llmint8_{g}", "kind": "activation", "variant": "llmint8",
        "bits": 8, "group": group,
        "model_artifact": f"models/x-alma-13b-group{group}",
        "load_mode": "int8", "stack": "hf", "platform": "slurm",
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--format", default="json", choices=["json", "tsv"])
    args = ap.parse_args()
    langs = json.load(open(GROUP_LANGS_JSON))
    all_rows = []
    for g in args.groups:
        for r in rows(g):
            r["languages"] = langs.get(f"g{g}", [])
            all_rows.append(r)
    if args.format == "tsv":
        for r in all_rows:
            print("\t".join([r["run_id"], r["kind"], r["variant"],
                             str(r["bits"]), r["model_artifact"], r["load_mode"]]))
    else:
        print(json.dumps(all_rows, indent=2))
    print(f"# {len(all_rows)} run points ({len(args.groups)} groups x 15)",
          flush=True) if False else None


if __name__ == "__main__":
    main()