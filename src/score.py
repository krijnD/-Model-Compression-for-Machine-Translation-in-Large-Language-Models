#!/usr/bin/env python
"""Scoring: sacrebleu spBLEU + chrF++, XCOMET-XL, MetricX-24-Hybrid-XL.

Writes scores/<run_id>/<direction>.json with per-direction metrics + system
averages. Runs on both platforms (smaller batch on Mac).

spBLEU: sacrebleu with tokenize='spm' + 'flores200' model (mirrors ALMA evals
which use TOK=13a for most pairs; FLORES uses spm). chrF++ chrF.word_order=2.

XCOMET-XL: unbabel-comet >= 2.2 (ref-based), device from src/device.py.
MetricX: google-research/metricx repo predict.py; inputs jsonl
source/hypothesis/reference; prediction in [0,25] lower=better.

Usage:
  python src/score.py --run-id <id> --direction en-de \
      --src data/flores/en-de/src.txt --mt outputs/<id>/en-de/mt.txt \
      --ref data/flores/en-de/ref.txt --out scores/<id>/en-de.json \
      [--metrics spbleu,chrf,xcomet,metricx] [--metricx-py vendor/metricx/metricx24/predict.py]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device import max_batch, device_name, platform_tag  # noqa: E402


def read_lines(p: str) -> list[str]:
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def score_spbleu_chrf(mt: list[str], ref: list[str], direction: str) -> dict:
    import sacrebleu
    # tokenizer per direction: zh -> 'zh', ja -> 'ja-mecab' (ALMA eval_generation.sh)
    src, tgt = direction.split("-")
    spm_tok = "zh" if tgt == "zh" else ("ja-mecab" if tgt == "ja" else "13a")
    out = {}
    try:
        bleu = sacrebleu.corpus_bleu(mt, [ref], tokenize="spm")
        out["spbleu"] = round(bleu.score, 3)
        out["spbleu_flores_code"] = "spm"
    except Exception as e:  # spm model download may fail offline
        out["spbleu"] = None
        out["spbleu_error"] = str(e)
    chrf = sacrebleu.corpus_chrf(mt, [ref], word_order=2)
    out["chrf++"] = round(chrf.score, 3)
    return out


def score_xcomet(mt: list[str], ref: list[str], src: list[str], max_bs: int) -> dict:
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, mt, ref)]
    out = model.predict(data, batch_size=max(min(max_bs, 64), 8), gpus=0 if device_name() != "cuda" else 1)
    return {"xcomet_xl": round(float(out["system_score"]), 4),
            "xcomet_xl_per_sentence": [round(float(x), 4) for x in out["scores"]]}


def score_metricx(mt: list[str], ref: list[str], src: list[str],
                  predict_py: str, max_bs: int) -> dict:
    """Call google-research/metricx predict.py over a temp jsonl."""
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            for s, h, r in zip(src, mt, ref):
                f.write(json.dumps({"source": s, "hypothesis": h, "reference": r}) + "\n")
        out_json = os.path.join(td, "out.json")
        cmd = [sys.executable, predict_py, "--input", inp,
               "--output", out_json, "--batch_size", str(min(max_bs, 64)),
               f"--gpu={1 if device_name()=='cuda' else 0}",
               f"--device={device_name()}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if res.returncode != 0:
            return {"metricx24": None, "metricx_error": res.stderr[-500:]}
        preds = [json.loads(l) for l in open(out_json) if l.strip()]
        scores = [float(p["prediction"]) for p in preds]
        return {"metricx24": round(sum(scores) / len(scores), 4),
                "metricx24_per_sentence": [round(x, 4) for x in scores]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--mt", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="spbleu,chrf,xcomet,metricx",
                    help="comma-separated metric groups")
    ap.add_argument("--metricx-py", default="vendor/metricx/metricx24/predict.py")
    args = ap.parse_args()

    src_lines, mt_lines, ref_lines = (read_lines(args.src), read_lines(args.mt),
                                      read_lines(args.ref))
    n = min(len(src_lines), len(mt_lines), len(ref_lines))
    if len(mt_lines) != len(ref_lines):
        print(f"WARN line mismatch: mt={len(mt_lines)} ref={len(ref_lines)}; clipping to {n}",
              file=sys.stderr)
    src_lines, mt_lines, ref_lines = (src_lines[:n], mt_lines[:n], ref_lines[:n])
    bs = max(8, max_batch())
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    result = {"run_id": args.run_id, "direction": args.direction,
              "num_sentences": n, "device": device_name(), "platform": platform_tag()}
    if "spbleu" in metrics or "chrf" in metrics:
        result.update(score_spbleu_chrf(mt_lines, ref_lines, args.direction))
    if "xcomet" in metrics:
        try:
            result.update(score_xcomet(mt_lines, ref_lines, src_lines, bs))
        except Exception as e:
            result["xcomet_xl"] = None
            result["xcomet_error"] = str(e)[-500:]
    if "metricx" in metrics:
        result.update(score_metricx(mt_lines, ref_lines, src_lines,
                                    args.metricx_py, bs))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if "per_sentence" not in k},
                     indent=2))


if __name__ == "__main__":
    main()