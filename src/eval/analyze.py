#!/usr/bin/env python
"""Analysis: degradation-vs-bit-width curves + tables from ledger + scores.

- Delta = baseline - quantized for higher-is-better (XCOMET, spBLEU, chrF++),
  Delta = quantized - baseline for lower-is-better (MetricX). Degradation = Delta
  (positive = worse).
- Curves: per kind x metric x resource-tier (high/mid/low), Delta vs effective
  bits. GGUF: nominal bits; GPTQ: bits; het: file_size*8/params; W8A8: 8.
- Monotonicity: flag inversions (Delta increasing with bits) per kind x metric x
  tier; write reports/tables/monotonicity.csv.
- Tables: per metric kind x variant x tier mean Delta; efficiency per run_id.

Usage:
  python src/eval/analyze.py [--ledger experiments/ledger.jsonl]
                        [--scores-dir scores] [--out reports]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# resource tiers for our eval langs (high / mid / low + script diversity)
TIER = {"de": "high", "is": "mid", "fr": "high", "mg": "low",
        "ru": "high", "cs": "mid", "zh": "mid", "ja": "mid", "ar": "low"}

# nominal effective bits per variant
BITS = {
    "q8_0": 8, "q6_k": 6, "q5_k_m": 5, "q4_k_m": 4, "q3_k_m": 3, "q2_k": 2,
    "b8": 8, "b4": 4, "b3": 3, "b2": 2,
    "sq_w8a8": 8, "llmint8": 8,
    "het_q8q4": None, "het_q8q2q3": None,  # computed from file size
    "fp16": 16,
}

# higher-is-better metrics; MetricX lower-is-better
HIB = ("xcomet_xl", "spbleu", "chrf++")


def load_scores(scores_dir: str) -> dict[str, dict]:
    """{run_id: {direction: {metric: value}}}"""
    out = {}
    for run_id in os.listdir(scores_dir):
        rd = os.path.join(scores_dir, run_id)
        if not os.path.isdir(rd):
            continue
        out[run_id] = {}
        for fn in os.listdir(rd):
            if fn.endswith(".json") and fn != "efficiency.json":
                try:
                    with open(os.path.join(rd, fn)) as f:
                        out[run_id][fn[:-5]] = json.load(f)
                except Exception:
                    pass
    return out


def effective_bits(row: dict, run_id: str, sizes: dict[str, float]) -> float:
    v = row.get("variant", "")
    b = BITS.get(v)
    if b is not None:
        return b
    if v in ("het_q8q4", "het_q8q2q3"):
        # bits = file_size_bytes * 8 / num_params
        gb = sizes.get(run_id)
        params = 13_000_000_000.0
        if gb:
            return round(gb * 8e9 / params, 2)
    return float(row.get("bits") or 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="experiments/ledger.jsonl")
    ap.add_argument("--scores-dir", default="scores")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--sizes-file", default=None,
                    help="json {run_id: size_gb} (from efficiency.json model_size_gb)")
    args = ap.parse_args()

    if not os.path.exists(args.ledger):
        print("no ledger yet; nothing to analyze", file=sys.stderr)
        return 0

    rows = [json.loads(l) for l in open(args.ledger) if l.strip()]
    scores = load_scores(args.scores_dir)
    sizes = {}
    if args.sizes_file:
        sizes = json.load(open(args.sizes_file))
    else:
        for run_id, d in scores.items():
            eff = os.path.join(args.scores_dir, run_id, "efficiency.json")
            if os.path.exists(eff):
                try:
                    sizes[run_id] = json.load(open(eff)).get("model_size_gb")
                except Exception:
                    pass

    # baseline run_ids: kind=fp16 (variant fp16)
    baselines = [r["run_id"] for r in rows if r.get("kind") == "fp16"]
    if not baselines:
        print("no fp16 baseline rows; analysis needs baseline", file=sys.stderr)
        return 1
    base_id = baselines[0]
    base_scores = scores.get(base_id, {})
    if not base_scores:
        print(f"baseline {base_id} has no scores; nothing to analyze", file=sys.stderr)
        return 1

    # degradation per run
    deg = []  # {run_id, kind, variant, bits, metric, tier, direction, delta}
    for r in rows:
        run_id = r["run_id"]
        if run_id == base_id:
            continue
        rd = scores.get(run_id)
        if not rd:
            continue
        kind = r.get("kind")
        variant = r.get("variant")
        bits = effective_bits(r, run_id, sizes)
        for direction, m in rd.items():
            src_lang = direction.split("-")[0]
            tier = TIER.get(src_lang, "mid")
            for metric in HIB + ("metricx24",):
                if metric not in m or m[metric] is None:
                    continue
                base_m = base_scores.get(direction, {}).get(metric)
                if base_m is None:
                    continue
                if metric in HIB:
                    delta = round(base_m - m[metric], 4)
                else:
                    delta = round(m[metric] - base_m, 4)
                deg.append({"run_id": run_id, "kind": kind, "variant": variant,
                            "bits": bits, "metric": metric, "tier": tier,
                            "direction": direction, "delta": delta})

    os.makedirs(os.path.join(args.out, "curves"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    # aggregate mean delta per (kind, metric, tier, bits, variant)
    agg = defaultdict(list)
    for d in deg:
        agg[(d["kind"], d["metric"], d["tier"], d["variant"], d["bits"])].append(d["delta"])
    rows_out = []
    for k, vals in sorted(agg.items()):
        kind, metric, tier, variant, bits = k
        rows_out.append({"kind": kind, "metric": metric, "tier": tier,
                         "variant": variant, "bits": bits,
                         "mean_delta": round(sum(vals) / len(vals), 4),
                         "n": len(vals)})
    with open(os.path.join(args.out, "tables", "degradation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys() if rows_out else [])
        w.writeheader()
        w.writerows(rows_out)

    # monotonicity: within kind x metric x tier, sort by bits ascending;
    # mean delta should decrease as bits increase (flag inversions)
    mono = []
    groups = defaultdict(list)
    for r in rows_out:
        groups[(r["kind"], r["metric"], r["tier"])].append(r)
    for (kind, metric, tier), lst in groups.items():
        lst.sort(key=lambda x: x["bits"])
        prev = None
        for r in lst:
            inv = False
            if prev is not None and r["mean_delta"] > prev["mean_delta"] + 1e-9:
                inv = True  # higher bits -> worse delta: inversion
            mono.append({"kind": kind, "metric": metric, "tier": tier,
                         "bits": r["bits"], "variant": r["variant"],
                         "mean_delta": r["mean_delta"], "inversion": inv})
            prev = r
    with open(os.path.join(args.out, "tables", "monotonicity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "metric", "tier", "bits",
                                          "variant", "mean_delta", "inversion"])
        w.writeheader()
        w.writerows(mono)

    print(f"analyzed {len(deg)} (run,direction,metric) degradation points "
          f"-> {args.out}/tables/")
    return 0


if __name__ == "__main__":
    sys.exit(main())