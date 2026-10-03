"""Does a kernel swap change the translations? Compare two output folders line by line.

Written for the lossless 3->4-bit repack (repack_to_w4.py): the repacked checkpoint dequantizes to
exactly the same weights, but ExLlamaV2 computes in fp16 with its own reduction order while the
scored w3 grid ran TorchQuantLinear in bf16, so beam search may still pick different outputs. This
reports, over the first N lines both folders have: exact-match rate, corpus BLEU and chrF++ of each
against the reference (the ALMA-R paper settings, same as scripts/reproduce/score_lexical.py), the
hallucination rate (candidate >= 2x reference length), and BLEU of one output against the other.

  python scripts/analysis/compare_outputs.py --a $OUTPUTS_DIR/gptq-w3g128/wmt22 \\
      --b $OUTPUTS_DIR/repack-check/w3as4 --pairs de-en,en-is --lines 200 --out results/json/x.json
Needs sacrebleu (the scoring venv, not venv-quant).
"""
import argparse
import json
import sys
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

REPO = Path(__file__).resolve().parents[2]
TESTSET = REPO / "third_party/ALMA/outputs/wmt22_outputs/wmt-testset"


def read(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def score(hyps, refs, tgt):
    bleu = BLEU(tokenize="zh" if tgt == "zh" else "13a").corpus_score(hyps, [refs]).score
    chrf = CHRF(word_order=2).corpus_score(hyps, [refs]).score
    halluc = 100 * sum(len(h) >= 2 * len(r) for h, r in zip(hyps, refs)) / len(hyps)
    return {"bleu": round(bleu, 2), "chrf++": round(chrf, 2), "halluc_pct": round(halluc, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="reference run folder (files test-<src>-<tgt>)")
    p.add_argument("--b", required=True, help="run folder to compare")
    p.add_argument("--labels", default="a,b")
    p.add_argument("--pairs", default="de-en,en-is")
    p.add_argument("--lines", type=int, default=0, help="first N lines (0 = all both have)")
    p.add_argument("--out")
    args = p.parse_args()
    la, lb = args.labels.split(",")

    report = {"a": args.a, "b": args.b, "labels": [la, lb], "pairs": {}}
    for pair in args.pairs.split(","):
        src, tgt = pair.split("-")
        fa, fb = Path(args.a) / f"test-{pair}", Path(args.b) / f"test-{pair}"
        if not (fa.exists() and fb.exists()):
            print(f"{pair}: missing {fa if not fa.exists() else fb}", file=sys.stderr)
            continue
        a, b = read(fa), read(fb)
        refs = read(TESTSET / f"{src}{tgt}/test.{src}-{tgt}.{tgt}")
        n = min(len(a), len(b)) if not args.lines else min(args.lines, len(a), len(b))
        a, b, refs = a[:n], b[:n], refs[:n]
        rec = {"lines": n, "exact_match_pct": round(100 * sum(x == y for x, y in zip(a, b)) / n, 2),
               la: score(a, refs, tgt), lb: score(b, refs, tgt),
               f"bleu_{lb}_vs_{la}": round(BLEU(tokenize="zh" if tgt == "zh" else "13a")
                                           .corpus_score(b, [a]).score, 2)}
        report["pairs"][pair] = rec
        print(f"{pair}: n={n} exact={rec['exact_match_pct']}%  BLEU {la}={rec[la]['bleu']} "
              f"{lb}={rec[lb]['bleu']}  chrF++ {la}={rec[la]['chrf++']} {lb}={rec[lb]['chrf++']}  "
              f"halluc {la}={rec[la]['halluc_pct']}% {lb}={rec[lb]['halluc_pct']}%  "
              f"BLEU({lb}|{la})={rec[f'bleu_{lb}_vs_{la}']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
